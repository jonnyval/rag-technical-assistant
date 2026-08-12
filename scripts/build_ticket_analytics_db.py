from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "analytics" / "tp_analyze" / "data" / "все обращения.csv"
DEFAULT_OUTPUT = ROOT / "analytics" / "tp_analyze" / "server_data" / "ticket_analytics.sqlite3"
SCHEMA_VERSION = 1
TICKET_ID_RE = re.compile(r"\bRL\s*[-_]\s*(\d+)\b", re.IGNORECASE)


@dataclass
class TicketAggregate:
    ticket_id: str
    fields: dict[str, str] = field(default_factory=dict)
    symptoms: list[str] = field(default_factory=list)
    solutions: list[str] = field(default_factory=list)
    quality_tags: list[str] = field(default_factory=list)
    product_groups: dict[str, str] = field(default_factory=dict)
    modules: dict[str, tuple[str, str, str, str]] = field(default_factory=dict)
    qdrant_points: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a read-only analytics snapshot by joining ticket CSV data with Qdrant metadata."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Ticket CSV export.")
    parser.add_argument(
        "--db",
        default="",
        help="Qdrant backend from config.yaml; defaults to storage.vector_db.second_db.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="SQLite snapshot path.")
    parser.add_argument("--batch-size", type=int, default=512, help="Qdrant scroll batch size.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and join sources, print statistics, but do not create a database.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly allow atomically replacing an existing analytics snapshot.",
    )
    return parser.parse_args()


def normalize_ticket_id(value: Any) -> str:
    text = str(value or "").strip()
    match = TICKET_ID_RE.search(text)
    if match:
        return f"RL-{int(match.group(1))}"
    return ""


def compact_spaces(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def normalized_topic_text(value: Any) -> str:
    text = compact_spaces(value).lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я_+.#/-]+", " ", text, flags=re.IGNORECASE).strip()


def values_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(values_list(item))
        return unique_texts(result)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped[:1] in "[{(" and stripped[-1:] in "]})":
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(stripped)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    continue
                if parsed != value:
                    return values_list(parsed)
        return [compact_spaces(stripped)]
    return [compact_spaces(value)]


def unique_texts(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = compact_spaces(value)
        key = text.casefold()
        if text and key not in seen:
            result.append(text)
            seen.add(key)
    return result


def merge_values(target: list[str], value: Any) -> None:
    combined = unique_texts([*target, *values_list(value)])
    target[:] = combined


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = compact_spaces(value)
        if text:
            return text
    return ""


def set_first(fields: dict[str, str], key: str, *values: Any) -> None:
    if fields.get(key):
        return
    value = first_nonempty(*values)
    if value:
        fields[key] = value


def csv_field(row: dict[str, str], *names: str) -> str:
    return first_nonempty(*(row.get(name, "") for name in names))


def canonical_ticket_type(raw_category: str, raw_ticket_type: str) -> tuple[str, str]:
    text = normalized_topic_text(f"{raw_category} {raw_ticket_type}")
    if "справоч" in text or "консультац" in text:
        return "reference", "Справочный вопрос"
    if "ошибканастрой" in text or ("ошиб" in text and "настрой" in text):
        return "configuration_error", "Ошибка конфигурации"
    if "отказпрог" in text or ("отказ" in text and "прог" in text):
        return "software_failure", "Программный отказ"
    if "отказобор" in text or ("отказ" in text and "обор" in text):
        return "hardware_failure", "Аппаратный отказ"
    if any(marker in text for marker in ("доработ", "предлож", "пожелан", "feature request")):
        return "change_request", "Запрос на изменение"
    if raw_category or raw_ticket_type:
        return "other", "Прочее"
    return "unclassified", "Не классифицировано"


def load_csv_tickets(path: Path) -> dict[str, TicketAggregate]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    tickets: dict[str, TicketAggregate] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            ticket_id = normalize_ticket_id(csv_field(row, "Код", "ticket_id", "code"))
            if not ticket_id:
                continue
            ticket = tickets.setdefault(ticket_id, TicketAggregate(ticket_id=ticket_id))
            fields = ticket.fields
            set_first(fields, "title", csv_field(row, "Наименование", "title"))
            set_first(fields, "description", csv_field(row, "Текст без html", "Текст", "description"))
            set_first(fields, "result", csv_field(row, "Результат без html", "Результат", "Решение", "result"))
            set_first(fields, "raw_category", csv_field(row, "Категория обращения РегЛаб", "Категория обращения ДАЭС"))
            set_first(fields, "raw_ticket_type", csv_field(row, "Тип обращения РегЛаб", "Тип обращения ПРОСОФТ", "Тип обращения.Имя объекта"))
            set_first(fields, "status", csv_field(row, "Статус.Имя статуса"))
            set_first(fields, "status_type", csv_field(row, "Кеш: Тип статуса"))
            set_first(fields, "project", csv_field(row, "Проект.Имя объекта"))
            set_first(fields, "component", csv_field(row, "Компоненты.Название"))
            set_first(fields, "client", csv_field(row, "Контрагент.Наименование"))
            set_first(fields, "equipment", csv_field(row, "Наименование оборудования", "Тип оборудования РегЛаб"))
            set_first(fields, "priority", csv_field(row, "Приоритет"))
            set_first(fields, "support_level", csv_field(row, "Уровень поддержки"))
            set_first(fields, "close_reason", csv_field(row, "Причина закрытия РегЛаб", "Причина закрытия ПРОСОФТ", "Резолюция.Название"))
            set_first(fields, "created_at", csv_field(row, "Дата создания"))
            set_first(fields, "closed_at", csv_field(row, "Дата закрытия"))
            fields["csv_found"] = "1"
    return tickets


def make_qdrant_client(backend: dict[str, Any]) -> QdrantClient:
    if backend.get("url"):
        return QdrantClient(url=backend["url"])
    if backend.get("path"):
        return QdrantClient(path=str((ROOT / backend["path"]).resolve()))
    raise ValueError("Qdrant backend must define either url or path")


def payload_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else payload


def merge_qdrant_payload(ticket: TicketAggregate, payload: dict[str, Any]) -> None:
    metadata = payload_metadata(payload)
    fields = ticket.fields
    ticket.qdrant_points += 1
    fields["qdrant_found"] = "1"
    set_first(fields, "title", metadata.get("page_title"), payload.get("page_title"))
    set_first(fields, "ticket_url", metadata.get("ticket_url"))
    set_first(fields, "source_file", metadata.get("source_file"), metadata.get("source_cache_file"))
    set_first(fields, "qdrant_category", metadata.get("category"))
    set_first(fields, "qdrant_status", metadata.get("status"))
    set_first(fields, "equipment", metadata.get("equipment_type"))

    merge_values(ticket.symptoms, metadata.get("llm_symptoms"))
    merge_values(ticket.solutions, metadata.get("llm_solution"))
    merge_values(ticket.quality_tags, metadata.get("quality_tags"))

    group_ids = values_list(metadata.get("ticket_product_groups"))
    group_titles = values_list(metadata.get("ticket_product_group_titles"))
    for group_id, title in aligned_values(group_ids, group_titles):
        if group_id:
            ticket.product_groups.setdefault(group_id, title)

    module_codes = values_list(metadata.get("mentioned_module_codes"))
    module_names = values_list(metadata.get("mentioned_modules"))
    module_families = values_list(metadata.get("mentioned_module_families"))
    module_functions = values_list(metadata.get("mentioned_module_functions"))
    for code, name, family, function in aligned_values(
        module_codes,
        module_names,
        module_families,
        module_functions,
    ):
        identity_key = normalized_topic_text(code or name or f"{family}:{function}")
        if not identity_key:
            continue
        existing = ticket.modules.get(identity_key, ("", "", "", ""))
        ticket.modules[identity_key] = tuple(
            previous or current for current, previous in zip((code, name, family, function), existing)
        )


def load_qdrant_tickets(
    client: QdrantClient,
    collection: str,
    tickets: dict[str, TicketAggregate],
    batch_size: int,
) -> tuple[int, int]:
    offset = None
    total_points = 0
    points_without_ticket_id = 0
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            total_points += 1
            payload = point.payload or {}
            metadata = payload_metadata(payload)
            ticket_id = normalize_ticket_id(
                first_nonempty(metadata.get("ticket_id"), metadata.get("source_file"), metadata.get("source_cache_file"))
            )
            if not ticket_id:
                points_without_ticket_id += 1
                continue
            ticket = tickets.setdefault(ticket_id, TicketAggregate(ticket_id=ticket_id))
            merge_qdrant_payload(ticket, payload)
        if offset is None:
            break
    return total_points, points_without_ticket_id


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE tickets (
    ticket_id TEXT PRIMARY KEY,
    csv_found INTEGER NOT NULL CHECK (csv_found IN (0, 1)),
    qdrant_found INTEGER NOT NULL CHECK (qdrant_found IN (0, 1)),
    qdrant_points INTEGER NOT NULL DEFAULT 0,
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    result TEXT NOT NULL DEFAULT '',
    raw_category TEXT NOT NULL DEFAULT '',
    qdrant_category TEXT NOT NULL DEFAULT '',
    raw_ticket_type TEXT NOT NULL DEFAULT '',
    canonical_type_code TEXT NOT NULL,
    canonical_type_label TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT '',
    status_type TEXT NOT NULL DEFAULT '',
    qdrant_status TEXT NOT NULL DEFAULT '',
    project TEXT NOT NULL DEFAULT '',
    component TEXT NOT NULL DEFAULT '',
    client TEXT NOT NULL DEFAULT '',
    equipment TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT '',
    support_level TEXT NOT NULL DEFAULT '',
    close_reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    closed_at TEXT NOT NULL DEFAULT '',
    ticket_url TEXT NOT NULL DEFAULT '',
    source_file TEXT NOT NULL DEFAULT ''
);

CREATE TABLE ticket_symptoms (
    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    PRIMARY KEY (ticket_id, ordinal),
    UNIQUE (ticket_id, normalized_text)
);

CREATE TABLE ticket_solutions (
    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    PRIMARY KEY (ticket_id, ordinal),
    UNIQUE (ticket_id, normalized_text)
);

CREATE TABLE ticket_quality_tags (
    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id) ON DELETE CASCADE,
    value TEXT NOT NULL,
    PRIMARY KEY (ticket_id, value)
);

CREATE TABLE ticket_product_groups (
    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id) ON DELETE CASCADE,
    group_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (ticket_id, group_id)
);

CREATE TABLE ticket_modules (
    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id) ON DELETE CASCADE,
    module_code TEXT NOT NULL DEFAULT '',
    module_name TEXT NOT NULL DEFAULT '',
    module_family TEXT NOT NULL DEFAULT '',
    module_function TEXT NOT NULL DEFAULT '',
    identity_key TEXT NOT NULL,
    PRIMARY KEY (ticket_id, identity_key)
);

CREATE INDEX idx_tickets_type ON tickets(canonical_type_code);
CREATE INDEX idx_tickets_category ON tickets(raw_category);
CREATE INDEX idx_tickets_closed_at ON tickets(closed_at);
CREATE INDEX idx_tickets_equipment ON tickets(equipment);
CREATE INDEX idx_symptoms_normalized ON ticket_symptoms(normalized_text);
CREATE INDEX idx_solutions_normalized ON ticket_solutions(normalized_text);
CREATE INDEX idx_product_groups_group ON ticket_product_groups(group_id);
CREATE INDEX idx_modules_code ON ticket_modules(module_code);
"""


TICKET_COLUMNS = (
    "ticket_id",
    "csv_found",
    "qdrant_found",
    "qdrant_points",
    "title",
    "description",
    "result",
    "raw_category",
    "qdrant_category",
    "raw_ticket_type",
    "canonical_type_code",
    "canonical_type_label",
    "status",
    "status_type",
    "qdrant_status",
    "project",
    "component",
    "client",
    "equipment",
    "priority",
    "support_level",
    "close_reason",
    "created_at",
    "closed_at",
    "ticket_url",
    "source_file",
)


def aligned_values(*groups: list[str]) -> Iterable[tuple[str, ...]]:
    size = max((len(group) for group in groups), default=0)
    for index in range(size):
        yield tuple(group[index] if index < len(group) else "" for group in groups)


def unique_normalized_rows(values: Iterable[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for text in values:
        normalized = normalized_topic_text(text)
        if normalized and normalized not in seen:
            result.append((text, normalized))
            seen.add(normalized)
    return result


def insert_snapshot(
    connection: sqlite3.Connection,
    tickets: dict[str, TicketAggregate],
    source_info: dict[str, Any],
) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [(key, str(value)) for key, value in sorted(source_info.items())],
    )

    placeholders = ",".join("?" for _ in TICKET_COLUMNS)
    ticket_sql = f"INSERT INTO tickets ({','.join(TICKET_COLUMNS)}) VALUES ({placeholders})"

    for ticket_id in sorted(tickets):
        ticket = tickets[ticket_id]
        fields = ticket.fields
        raw_category = first_nonempty(fields.get("raw_category"), fields.get("qdrant_category"))
        raw_ticket_type = fields.get("raw_ticket_type", "")
        type_code, type_label = canonical_ticket_type(raw_category, raw_ticket_type)
        row = {
            **{column: fields.get(column, "") for column in TICKET_COLUMNS},
            "ticket_id": ticket_id,
            "csv_found": int(fields.get("csv_found") == "1"),
            "qdrant_found": int(fields.get("qdrant_found") == "1"),
            "qdrant_points": ticket.qdrant_points,
            "raw_category": raw_category,
            "canonical_type_code": type_code,
            "canonical_type_label": type_label,
        }
        connection.execute(ticket_sql, tuple(row[column] for column in TICKET_COLUMNS))

        symptom_rows = unique_normalized_rows(ticket.symptoms)
        connection.executemany(
            "INSERT INTO ticket_symptoms(ticket_id, ordinal, text, normalized_text) VALUES (?, ?, ?, ?)",
            [(ticket_id, index, text, normalized) for index, (text, normalized) in enumerate(symptom_rows, start=1)],
        )
        solution_rows = unique_normalized_rows(ticket.solutions)
        connection.executemany(
            "INSERT INTO ticket_solutions(ticket_id, ordinal, text, normalized_text) VALUES (?, ?, ?, ?)",
            [(ticket_id, index, text, normalized) for index, (text, normalized) in enumerate(solution_rows, start=1)],
        )
        connection.executemany(
            "INSERT INTO ticket_quality_tags(ticket_id, value) VALUES (?, ?)",
            [(ticket_id, value) for value in ticket.quality_tags],
        )
        connection.executemany(
            "INSERT INTO ticket_product_groups(ticket_id, group_id, title) VALUES (?, ?, ?)",
            [(ticket_id, group_id, title) for group_id, title in sorted(ticket.product_groups.items())],
        )
        module_rows = [
            (ticket_id, code, name, family, function, identity_key)
            for identity_key, (code, name, family, function) in sorted(ticket.modules.items())
        ]
        connection.executemany(
            """INSERT OR IGNORE INTO ticket_modules(
                   ticket_id, module_code, module_name, module_family, module_function, identity_key
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            module_rows,
        )


def validate_snapshot(connection: sqlite3.Connection) -> dict[str, int]:
    checks = {
        "tickets": "SELECT COUNT(*) FROM tickets",
        "csv_tickets": "SELECT COUNT(*) FROM tickets WHERE csv_found = 1",
        "qdrant_tickets": "SELECT COUNT(*) FROM tickets WHERE qdrant_found = 1",
        "joined_tickets": "SELECT COUNT(*) FROM tickets WHERE csv_found = 1 AND qdrant_found = 1",
        "symptoms": "SELECT COUNT(*) FROM ticket_symptoms",
        "solutions": "SELECT COUNT(*) FROM ticket_solutions",
        "quality_tags": "SELECT COUNT(*) FROM ticket_quality_tags",
        "product_groups": "SELECT COUNT(*) FROM ticket_product_groups",
        "modules": "SELECT COUNT(*) FROM ticket_modules",
        "foreign_key_errors": "SELECT COUNT(*) FROM pragma_foreign_key_check",
    }
    result = {name: int(connection.execute(sql).fetchone()[0]) for name, sql in checks.items()}
    if result["foreign_key_errors"]:
        raise RuntimeError(f"Foreign key validation failed: {result['foreign_key_errors']} errors")
    return result


def write_snapshot_atomic(
    output: Path,
    tickets: dict[str, TicketAggregate],
    source_info: dict[str, Any],
    replace: bool,
) -> dict[str, int]:
    output = output.resolve()
    if output.exists() and not replace:
        raise FileExistsError(f"Output already exists: {output}. Use --replace to replace it atomically.")
    output.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        connection = sqlite3.connect(temp_path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                insert_snapshot(connection, tickets, source_info)
            stats = validate_snapshot(connection)
        finally:
            connection.close()
        os.replace(temp_path, output)
        return stats
    finally:
        if temp_path.exists():
            temp_path.unlink()


def print_stats(stats: dict[str, Any]) -> None:
    for key, value in stats.items():
        print(f"{key}: {value}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.output.exists() and not args.replace and not args.dry_run:
        raise SystemExit(f"Output already exists: {args.output}. Use --replace to replace it atomically.")

    sys.path.insert(0, str(ROOT))
    from src.config import settings

    backend_name = args.db or settings.second_db_name
    backend = settings.db_backends.get(backend_name)
    if not backend:
        raise SystemExit(f"Backend not found in config.yaml: {backend_name}")
    if backend.get("type") != "qdrant":
        raise SystemExit(f"Backend is not Qdrant: {backend_name}")
    collection = str(backend.get("collection") or "").strip()
    if not collection:
        raise SystemExit(f"Backend has no collection: {backend_name}")

    print(f"Reading CSV: {args.csv}")
    tickets = load_csv_tickets(args.csv)
    csv_ticket_count = len(tickets)

    print(f"Reading Qdrant (payload only, no vectors): {backend_name}/{collection}")
    client = make_qdrant_client(backend)
    if not client.collection_exists(collection):
        raise SystemExit(f"Qdrant collection not found: {collection}")
    qdrant_points, points_without_ticket_id = load_qdrant_tickets(client, collection, tickets, args.batch_size)

    joined = sum(
        1
        for ticket in tickets.values()
        if ticket.fields.get("csv_found") == "1" and ticket.fields.get("qdrant_found") == "1"
    )
    source_info = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "csv_path": str(args.csv.resolve()),
        "qdrant_backend": backend_name,
        "qdrant_collection": collection,
        "qdrant_points": qdrant_points,
        "qdrant_points_without_ticket_id": points_without_ticket_id,
    }
    preview_stats = {
        "csv_tickets": csv_ticket_count,
        "qdrant_points": qdrant_points,
        "qdrant_points_without_ticket_id": points_without_ticket_id,
        "unique_tickets": len(tickets),
        "joined_tickets": joined,
        "symptoms": sum(len(ticket.symptoms) for ticket in tickets.values()),
        "solutions": sum(len(ticket.solutions) for ticket in tickets.values()),
    }
    print_stats(preview_stats)

    if args.dry_run:
        print("Dry run complete: no files were created or changed.")
        return

    stats = write_snapshot_atomic(args.output, tickets, source_info, replace=args.replace)
    print(f"Analytics snapshot written atomically: {args.output.resolve()}")
    print_stats(stats)


if __name__ == "__main__":
    main()
