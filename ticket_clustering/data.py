from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_+.#/-]+")
TECHNICAL_RE = re.compile(
    r"(?:\d|[A-ZА-Я]{2,}|[A-Za-z]+\d|\d+[A-Za-z]|[/\\]|\b(?:PLC|ПЛК|TCP|UDP|OPC|HMI|IDE|API)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SymptomOccurrence:
    ticket_id: str
    ordinal: int
    text: str
    normalized_text: str
    equipment: str
    canonical_type: str


@dataclass
class SymptomNode:
    node_id: str
    normalized_text: str
    display_text: str
    occurrences: list[SymptomOccurrence] = field(default_factory=list)
    informativeness: float = 0.0
    excluded_reason: str = ""

    @property
    def ticket_count(self) -> int:
        return len({item.ticket_id for item in self.occurrences})


def stable_node_id(normalized_text: str) -> str:
    digest = hashlib.sha256(normalized_text.encode("utf-8", errors="ignore")).hexdigest()[:20]
    return f"symptom-{digest}"


def tokens(text: str) -> list[str]:
    return [token.casefold().replace("ё", "е") for token in TOKEN_RE.findall(text)]


def open_source_readonly(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Analytics database not found: {resolved}")
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        connection.close()
        raise RuntimeError(f"Analytics database integrity check failed: {integrity}")
    return connection


def load_category_occurrences(connection: sqlite3.Connection, category: str) -> list[SymptomOccurrence]:
    rows = connection.execute(
        """
        SELECT
            s.ticket_id,
            s.ordinal,
            s.text,
            s.normalized_text,
            t.equipment,
            t.canonical_type_label
        FROM ticket_symptoms AS s
        JOIN tickets AS t ON t.ticket_id = s.ticket_id
        WHERE COALESCE(NULLIF(t.raw_category, ''), NULLIF(t.qdrant_category, ''), t.canonical_type_label) = ?
        ORDER BY s.ticket_id, s.ordinal
        """,
        (category,),
    ).fetchall()
    return [
        SymptomOccurrence(
            ticket_id=str(row["ticket_id"]),
            ordinal=int(row["ordinal"]),
            text=str(row["text"]),
            normalized_text=str(row["normalized_text"]),
            equipment=str(row["equipment"] or ""),
            canonical_type=str(row["canonical_type_label"] or ""),
        )
        for row in rows
    ]


def build_nodes(occurrences: Iterable[SymptomOccurrence]) -> list[SymptomNode]:
    grouped: dict[str, list[SymptomOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        if occurrence.normalized_text:
            grouped[occurrence.normalized_text].append(occurrence)

    nodes: list[SymptomNode] = []
    for normalized_text, items in sorted(grouped.items()):
        display_counts = Counter(item.text for item in items)
        display_text = sorted(display_counts, key=lambda value: (-display_counts[value], value.casefold()))[0]
        nodes.append(
            SymptomNode(
                node_id=stable_node_id(normalized_text),
                normalized_text=normalized_text,
                display_text=display_text,
                occurrences=items,
            )
        )
    apply_informativeness(nodes)
    return nodes


def apply_informativeness(nodes: list[SymptomNode]) -> None:
    if not nodes:
        return
    document_frequency: Counter[str] = Counter()
    node_tokens: list[list[str]] = []
    for node in nodes:
        values = tokens(node.normalized_text)
        node_tokens.append(values)
        document_frequency.update(set(values))

    denominator = math.log(len(nodes) + 1.0)
    for node, values in zip(nodes, node_tokens):
        if values:
            idf_values = [math.log((len(nodes) + 1.0) / (document_frequency[value] + 1.0)) for value in values]
            specificity = min(1.0, sum(idf_values) / len(idf_values) / max(denominator, 1e-9))
        else:
            specificity = 0.0
        technical = 1.0 if TECHNICAL_RE.search(node.display_text) else 0.0
        length_score = min(1.0, len(values) / 6.0)
        node.informativeness = round(0.55 * specificity + 0.30 * technical + 0.15 * length_score, 6)


def select_eligible_nodes(
    nodes: list[SymptomNode],
    min_chars: int,
    min_tokens: int,
    min_informativeness: float,
) -> tuple[list[SymptomNode], list[SymptomNode]]:
    eligible: list[SymptomNode] = []
    excluded: list[SymptomNode] = []
    for node in nodes:
        reason = ""
        if len(node.normalized_text) < min_chars:
            reason = "too_short_chars"
        elif len(tokens(node.normalized_text)) < min_tokens:
            reason = "too_few_tokens"
        elif node.informativeness < min_informativeness:
            reason = "low_informativeness"
        node.excluded_reason = reason
        (excluded if reason else eligible).append(node)
    return eligible, excluded
