import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
import src.document_processing.parsers_tickets_with_llm as ticket_parser


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_JSON_DIR = PROJECT_ROOT / "data" / "source_docs" / "docs_json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data"
PORTAL_URL = "https://support.prosyst.ru"
RL_ID_RE = re.compile(r"\[([A-Z]+-\d+)\]", re.IGNORECASE)

CSV_CODE_COL = "Код"
CSV_DATE_COL = "Дата создания"
CSV_STATUS_COL = "Статус.Имя статуса"
CSV_EQUIPMENT_COL = "Тип оборудования РегЛаб"
CSV_ENGINEER_COL = "Исполнитель.ФИО"
CSV_REQUESTER_COL = "Постановщик.ФИО"

FILTER_STATUSES = [
    "Закрыто",
    "Решено",
    "Отложено",
    "В разработке",
]

FILTER_EXCLUDE_EQUIPMENT = [
    "Вопросы по порталу тех. поддержки",
]

FILTER_EXCLUDE_ENGINEERS = [
    "Говорухин Владимир Александрович",
    "Сыпченков Юрий Юрьевич",
    "Задоркина Марина Николаевна",
]

FILTER_EXCLUDE_REQUESTERS = [
    "Шихова Анастасия Валерьевна",
    "Задоркина Марина Николаевна",
    "Шутемов Егор Сергеевич",
    "Шутемовв Егорр Сергеевичч",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сформировать LLM-cache JSON по скачанным support-ticket JSON."
    )
    parser.add_argument(
        "--tickets-dir",
        type=Path,
        default=None,
        help="Папка со скачанными JSON. По умолчанию берётся самая свежая data/source_docs/docs_json/tickets_*.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Папка для LLM-cache. По умолчанию data/llm_cache_<имя папки тикетов>.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Опциональный CSV для дополнительной фильтрации. Обычно не нужен, если скачивание уже фильтровало тикеты.",
    )
    parser.add_argument(
        "--date-from",
        default="2022-01-01",
        help="Минимальная дата создания при фильтрации через --csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество файлов для тестового запуска.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересоздать cache-файл, даже если он уже существует.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=("config", "local", "api"),
        default="config",
        help="Режим LLM-обогащения: config из config.yaml, local=Ollama, api=Gemini->Groq.",
    )
    parser.add_argument(
        "--enable-smart-metadata",
        action="store_true",
        help="Включить LLM-обогащение symptoms/solution для этого запуска.",
    )
    return parser.parse_args()


class TicketSettingsOverride:
    def __init__(self, base_settings, *, llm_mode: str, enable_smart_metadata: bool):
        self._base_settings = base_settings
        self._llm_mode = llm_mode
        self._enable_smart_metadata = enable_smart_metadata

    def __getattr__(self, name):
        return getattr(self._base_settings, name)

    @property
    def ticket_active_llm(self) -> str:
        if self._llm_mode == "local":
            return "ollama"
        if self._llm_mode == "api":
            return "api"
        return self._base_settings.ticket_active_llm

    @property
    def enable_smart_metadata(self) -> bool:
        return self._enable_smart_metadata or self._base_settings.enable_smart_metadata


def configure_ticket_parser(args: argparse.Namespace) -> None:
    if args.llm_mode == "config" and not args.enable_smart_metadata:
        return
    ticket_parser.settings = TicketSettingsOverride(
        settings,
        llm_mode=args.llm_mode,
        enable_smart_metadata=args.enable_smart_metadata,
    )


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def latest_tickets_dir() -> Path:
    candidates = [path for path in DOCS_JSON_DIR.glob("tickets_*") if path.is_dir()]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    relative_path_str = settings.source_dirs.get("support_tickets", "data/source_docs/docs_json/tickets")
    return PROJECT_ROOT / relative_path_str


def default_output_dir(tickets_dir: Path) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"llm_cache_{tickets_dir.name}"


def get_ticket_id(path: Path) -> str:
    match = RL_ID_RE.search(path.name)
    return match.group(1).upper() if match else path.stem.upper()


def get_valid_ids_from_csv(csv_path: Path, date_from: str) -> set[str]:
    print(f"Чтение и фильтрация CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", on_bad_lines="warn")

    required_columns = {
        CSV_CODE_COL,
        CSV_DATE_COL,
        CSV_STATUS_COL,
        CSV_EQUIPMENT_COL,
        CSV_ENGINEER_COL,
        CSV_REQUESTER_COL,
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"В CSV не найдены обязательные колонки: {', '.join(missing_columns)}")

    total_before = len(df)
    date_from_ts = pd.Timestamp(date_from, tz="UTC")
    df[CSV_DATE_COL] = pd.to_datetime(df[CSV_DATE_COL], utc=True, errors="coerce")

    df = df[df[CSV_DATE_COL] >= date_from_ts]
    df = df[df[CSV_STATUS_COL].isin(FILTER_STATUSES)]
    df = df[~df[CSV_EQUIPMENT_COL].isin(FILTER_EXCLUDE_EQUIPMENT)]
    df = df[~df[CSV_ENGINEER_COL].isin(FILTER_EXCLUDE_ENGINEERS)]
    df = df[~df[CSV_REQUESTER_COL].isin(FILTER_EXCLUDE_REQUESTERS)]

    valid_ids: set[str] = set()
    for code in df[CSV_CODE_COL].dropna():
        match = re.search(r"\bRL-\d+\b", str(code), re.IGNORECASE)
        if match:
            valid_ids.add(match.group(0).upper())

    print(f"После фильтрации CSV: {len(df)} строк из {total_before}; тикетов: {len(valid_ids)}")
    return valid_ids


def main() -> int:
    args = parse_args()
    tickets_dir = resolve_project_path(args.tickets_dir) if args.tickets_dir else latest_tickets_dir()
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else default_output_dir(tickets_dir)
    csv_path = resolve_project_path(args.csv) if args.csv else None
    configure_ticket_parser(args)

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit должен быть больше 0")
    if not tickets_dir.exists():
        raise SystemExit(f"Папка с тикетами не найдена: {tickets_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Генерация LLM-cache по support tickets ===")
    print(f"Tickets dir: {tickets_dir}")
    print(f"Output dir:  {output_dir}")

    valid_ids = get_valid_ids_from_csv(csv_path, args.date_from) if csv_path else None

    ticket_files = sorted(tickets_dir.rglob("*.json"))
    if valid_ids is not None:
        ticket_files = [path for path in ticket_files if get_ticket_id(path) in valid_ids]
    if args.limit is not None:
        ticket_files = ticket_files[:args.limit]

    print(f"Файлов к обработке: {len(ticket_files)}")
    if not ticket_files:
        return 0

    errors = []
    llm_ok = 0
    llm_fail = 0
    skipped_existing = 0
    written = 0

    for path in tqdm(ticket_files, desc="Обработка LLM"):
        ticket_id = get_ticket_id(path)
        out_file = output_dir / f"{ticket_id}_llm.json"

        if out_file.exists() and not args.force:
            skipped_existing += 1
            continue

        try:
            docs = ticket_parser.process_ticket_file(path, portal_base_url=PORTAL_URL)
            fact_docs = [doc for doc in docs if doc.metadata.get("doc_level") == "fact"]
            if fact_docs and fact_docs[0].metadata.get("llm_symptoms"):
                llm_ok += 1
            elif fact_docs:
                llm_fail += 1

            ticket_data = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ]

            with out_file.open("w", encoding="utf-8") as out:
                json.dump(ticket_data, out, ensure_ascii=False, indent=2)
            written += 1

        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
            tqdm.write(f"\nОшибка файла {path.name}: {exc}")

    if errors:
        error_log = output_dir / "_errors.json"
        with error_log.open("w", encoding="utf-8") as file:
            json.dump(errors, file, ensure_ascii=False, indent=2)
        print(f"\nОшибок при обработке: {len(errors)}. Подробности: {error_log}")

    total_fact = llm_ok + llm_fail
    if total_fact > 0:
        pct = llm_ok / total_fact * 100
        print(f"LLM-обогащение fact-документов: {llm_ok}/{total_fact} успешно ({pct:.1f}%)")

    print(
        "Готово. "
        f"Записано: {written}; пропущено существующих: {skipped_existing}; ошибок: {len(errors)}."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
