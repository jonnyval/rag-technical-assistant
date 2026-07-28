import argparse
import json
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.logger import log

load_dotenv()


DEFAULT_CSV_FILE_PATH = "Выгрузка обращений (5).csv"
DEFAULT_OUTPUT_BASE_DIR = Path("data/source_docs/docs_json")
DEFAULT_OUTPUT_DIR_PREFIX = "tickets"
SUPPORT_API_URL = "https://support.prosyst.ru/api/"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

CSV_CODE_COL = "Код"
CSV_OBJECT_ID_COL = "Идентификатор объекта"
CSV_DATE_COL = "Дата создания"
CSV_STATUS_COL = "Статус.Имя статуса"
CSV_EQUIPMENT_COL = "Тип оборудования РегЛаб"
CSV_ENGINEER_COL = "Исполнитель.ФИО"
CSV_REQUESTER_COL = "Постановщик.ФИО"
REQUIRED_CSV_COLUMNS = {
    CSV_CODE_COL,
    CSV_DATE_COL,
    CSV_STATUS_COL,
    CSV_EQUIPMENT_COL,
    CSV_ENGINEER_COL,
    CSV_REQUESTER_COL,
}

EQUIPMENT_MAPPING = {
    "r01": "Regul R050, R100, R200, R400, R500, R600",
    "r02": "Regul R500S",
    "r03": "ASTRAREGUL",
    "r04": "ИВД-2, ИВД-3, ИВД-5",
    "r05": "МЭД-1",
    "r06": "ALFAREGUL",
    "r07": "ТВПС-1, ПКМ-ТВПС, КШ, КШ-М, КШ-Б-01",
    "r09": "Шлюз-конвертор REGUL",
    "r08": "Вопросы по порталу тех. поддержки",
}

CATEGORY_MAPPING = {
    "otkaz-obor-prosoft": "ОтказОборПрософт",
    "novyj-funkczional": "Новый функционал",
    "nepolnaya-dokumentacziya": "Неполная документация",
    "spravochnoe": "Справочное",
    "tehobsluzhivanie": "Техобслуживание",
    "otkaz-obor-chuzhogo": "ОтказОборЧужого",
    "oshibka-nastrojki-chuzhaya": "ОшибкаНастройкиЧужая",
    "oshibka-nastrojki-prosoft": "ОшибкаНастройкиПрософт",
    "otkaz-prog-chuzhoj": "ОтказПрогЧужой",
    "otkaz-prog-prosoft": "ОтказПрогПрософт",
}

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
    "Шутемовв Егорр Сергеевичч",  # дубль с опечаткой в базе
]

MAX_RETRIES = 4
BASE_BACKOFF = 5
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
RL_ID_RE = re.compile(r"\bRL-\d+\b", re.IGNORECASE)
WINDOWS_FORBIDDEN_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class DownloadConfig:
    csv_path: Path
    output_dir: Path
    date_from: str
    force: bool
    limit: int | None
    api_page_size: int
    sleep_min: float
    sleep_max: float


@dataclass
class DownloadStats:
    requested: int = 0
    skipped_existing: int = 0
    missing_in_portal: int = 0
    downloaded: int = 0
    failed: int = 0


@dataclass(frozen=True)
class TicketRef:
    rl_id: str
    internal_id: str | None = None


def parse_args() -> DownloadConfig:
    parser = argparse.ArgumentParser(
        description="Скачать JSON карточек обращений из support.prosyst.ru по RL-ID из CSV-выгрузки."
    )
    parser.add_argument(
        "--csv",
        default=os.getenv("TICKETS_CSV", DEFAULT_CSV_FILE_PATH),
        help="Путь к CSV-выгрузке. По умолчанию берётся TICKETS_CSV или текущий файл экспорта.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Каталог для JSON-файлов. По умолчанию создаётся data/source_docs/docs_json/tickets_YYYY-MM-DD.",
    )
    parser.add_argument(
        "--date-from",
        default="2022-01-01",
        help="Минимальная дата создания тикета в формате YYYY-MM-DD.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перекачивать тикеты, даже если JSON-файл уже есть.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить число тикетов для скачивания. Удобно для тестового запуска.",
    )
    parser.add_argument(
        "--api-page-size",
        type=int,
        default=1000,
        help="Размер страницы при построении справочника ID через API.",
    )
    parser.add_argument(
        "--sleep-min",
        type=float,
        default=0.5,
        help="Минимальная пауза между тикетами, секунды.",
    )
    parser.add_argument(
        "--sleep-max",
        type=float,
        default=1.2,
        help="Максимальная пауза между тикетами, секунды.",
    )
    args = parser.parse_args()

    if args.api_page_size <= 0:
        parser.error("--api-page-size должен быть больше 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit должен быть больше 0")
    if args.sleep_min < 0 or args.sleep_max < 0 or args.sleep_min > args.sleep_max:
        parser.error("Паузы должны быть неотрицательными, и --sleep-min не может быть больше --sleep-max")

    return DownloadConfig(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir) if args.output_dir else default_output_dir(),
        date_from=args.date_from,
        force=args.force,
        limit=args.limit,
        api_page_size=args.api_page_size,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
    )


def default_output_dir() -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    return DEFAULT_OUTPUT_BASE_DIR / f"{DEFAULT_OUTPUT_DIR_PREFIX}_{today}"


def get_cookie_string() -> str:
    cookie_string = os.getenv("SUPPORT_COOKIES")
    if not cookie_string:
        raise ValueError("Переменная SUPPORT_COOKIES не найдена в .env")
    return cookie_string


def get_cookies_dict(cookie_string: str) -> dict[str, str]:
    return {
        key: value
        for item in cookie_string.split(";")
        if "=" in item
        for key, value in [item.strip().split("=", 1)]
    }


def make_session(cookie_string: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.cookies.update(get_cookies_dict(cookie_string))
    return session


def retry_wait_seconds(response: requests.Response | None, attempt: int) -> float:
    if response is not None and response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
    return BASE_BACKOFF * (2**attempt) + random.uniform(0, 2)


def post_json(session: requests.Session, method: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    url = f"{SUPPORT_API_URL}?m={method}"
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        response: requests.Response | None = None
        try:
            response = session.post(url, json=payload, timeout=timeout)

            if response.status_code in RETRY_STATUS_CODES:
                wait = retry_wait_seconds(response, attempt)
                log.warning(
                    f"HTTP {response.status_code} от {method}. Жду {wait:.1f}с "
                    f"(попытка {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"{method} вернул JSON не в формате объекта")
            if data.get("error"):
                raise RuntimeError(f"{method} вернул ошибку JSON-RPC: {data['error']}")
            return data

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
            wait = retry_wait_seconds(response, attempt)
            log.warning(
                f"Сетевая ошибка при вызове {method}: {exc}. Жду {wait:.1f}с "
                f"(попытка {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            wait = retry_wait_seconds(response, attempt)
            log.warning(
                f"Ошибка HTTP при вызове {method}: {exc}. Жду {wait:.1f}с "
                f"(попытка {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)
        except ValueError as exc:
            raise RuntimeError(f"{method} вернул невалидный JSON") from exc

    raise RuntimeError(f"Не удалось получить ответ от {method} после {MAX_RETRIES} попыток") from last_error


def read_tickets_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV-файл не найден: {csv_path}")

    log.info(f"Чтение CSV-файла: {csv_path}")
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", on_bad_lines="warn")
    missing_columns = sorted(REQUIRED_CSV_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"В CSV не найдены обязательные колонки: {', '.join(missing_columns)}")

    return df


def clean_internal_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    internal_id = str(value).strip()
    return internal_id or None


def get_ticket_refs_from_csv(csv_path: Path, date_from: str, limit: int | None = None) -> list[TicketRef]:
    df = read_tickets_csv(csv_path)
    total_before = len(df)

    date_from_ts = pd.Timestamp(date_from, tz="UTC")
    df[CSV_DATE_COL] = pd.to_datetime(df[CSV_DATE_COL], utc=True, errors="coerce")
    invalid_dates = int(df[CSV_DATE_COL].isna().sum())
    if invalid_dates:
        log.warning(f"В CSV найдено строк с невалидной датой создания: {invalid_dates}")

    df = df[df[CSV_DATE_COL] >= date_from_ts]
    df = df[df[CSV_STATUS_COL].isin(FILTER_STATUSES)]
    df = df[~df[CSV_EQUIPMENT_COL].isin(FILTER_EXCLUDE_EQUIPMENT)]
    df = df[~df[CSV_ENGINEER_COL].isin(FILTER_EXCLUDE_ENGINEERS)]
    df = df[~df[CSV_REQUESTER_COL].isin(FILTER_EXCLUDE_REQUESTERS)]

    log.info(
        f"После фильтрации: {len(df)} строк из {total_before} "
        f"(убрано {total_before - len(df)})"
    )

    ticket_refs: list[TicketRef] = []
    seen: set[str] = set()
    has_object_id = CSV_OBJECT_ID_COL in df.columns
    if has_object_id:
        log.info(f"Будет использована колонка '{CSV_OBJECT_ID_COL}' для internal_id.")
    else:
        log.warning(f"Колонка '{CSV_OBJECT_ID_COL}' не найдена. Будет использован поиск ID через API.")

    for _, row in df.iterrows():
        match = RL_ID_RE.search(str(row[CSV_CODE_COL]))
        if not match:
            continue

        rl_id = match.group(0).upper()
        if rl_id in seen:
            continue

        seen.add(rl_id)
        internal_id = clean_internal_id(row[CSV_OBJECT_ID_COL]) if has_object_id else None
        ticket_refs.append(TicketRef(rl_id=rl_id, internal_id=internal_id))
        if limit is not None and len(ticket_refs) >= limit:
            break

    refs_with_internal_id = sum(1 for ref in ticket_refs if ref.internal_id)
    log.info(
        f"Уникальных тикетов для скачивания: {len(ticket_refs)} "
        f"(с internal_id из CSV: {refs_with_internal_id})"
    )
    return ticket_refs


def extract_rl_id(ticket: dict[str, Any]) -> str | None:
    ticket_str = json.dumps(ticket, ensure_ascii=False)
    match = RL_ID_RE.search(ticket_str)
    return match.group(0).upper() if match else None


def build_id_mapping(
    session: requests.Session,
    needed_rl_ids: set[str],
    page_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    log.info("Сбор справочника RL-ID -> internal_id из API портала...")
    mapping: dict[str, Any] = {}
    owner_mapping: dict[str, Any] = {}
    offset = 0
    previous_found = -1

    while needed_rl_ids - set(mapping):
        payload = {
            "jsonrpc": "2.0",
            "method": "CmfTask.list",
            "no_meta": True,
            "kwargs": {
                "limit": page_size,
                "offset": offset,
            },
        }

        data = post_json(session, "CmfTask.list", payload, timeout=60)
        batch = data.get("result", [])
        if not isinstance(batch, list):
            raise RuntimeError("CmfTask.list вернул result не в формате списка")
        if not batch:
            break

        for ticket in batch:
            if not isinstance(ticket, dict):
                continue

            rl_id = extract_rl_id(ticket)
            internal_id = ticket.get("id")
            if rl_id in needed_rl_ids and internal_id:
                mapping[rl_id] = internal_id
                owner_mapping[rl_id] = ticket.get("cmf_owner_id")

        log.info(
            f"Справочник: найдено {len(mapping)}/{len(needed_rl_ids)}, "
            f"offset={offset}, batch={len(batch)}"
        )
        if len(mapping) == previous_found:
            missing_ids = sorted(needed_rl_ids - set(mapping))
            preview = ", ".join(missing_ids[:10])
            if len(missing_ids) > 10:
                preview += f", ... (+{len(missing_ids) - 10})"
            log.warning(
                "Поиск internal_id остановлен: новых совпадений от API нет. "
                f"Не найдены RL-ID: {preview}"
            )
            break

        previous_found = len(mapping)
        offset += page_size

    log.info(f"Справочник готов. Найдено {len(mapping)} тикетов из {len(needed_rl_ids)}.")
    return mapping, owner_mapping


def downloaded_rl_ids(output_dir: Path) -> set[str]:
    return {
        file.name.split("]")[0][1:].upper()
        for file in output_dir.glob("*.json")
        if file.name.startswith("[") and "]" in file.name
    }


def normalize_result_block(ticket_data: dict[str, Any]) -> dict[str, Any]:
    result_block = ticket_data.get("result", {})
    if isinstance(result_block, list):
        if not result_block:
            return {}
        result_block = result_block[0]
    if not isinstance(result_block, dict):
        raise RuntimeError("CmfTask.ui_get вернул result не в формате объекта")
    return result_block


def enrich_ticket_data(
    ticket_data: dict[str, Any],
    comments_data: dict[str, Any],
    task_owner_id: Any,
) -> dict[str, Any]:
    result_block = normalize_result_block(ticket_data)

    if not task_owner_id:
        task_owner_id = result_block.get("cmf_owner_id")
        if not task_owner_id and isinstance(result_block.get("cmf_owner"), dict):
            task_owner_id = result_block["cmf_owner"].get("id")

    eq_code = result_block.get("cf_tip_oborud_reg")
    if eq_code in EQUIPMENT_MAPPING:
        result_block["cf_tip_oborud_reg_name"] = EQUIPMENT_MAPPING[eq_code]

    cat_code = result_block.get("cf_kategoriya_or")
    if cat_code in CATEGORY_MAPPING:
        result_block["cf_kategoriya_or_name"] = CATEGORY_MAPPING[cat_code]

    if isinstance(result_block.get("cmf_owner"), dict):
        result_block["cmf_owner"]["name"] = "[КЛИЕНТ]"
    if isinstance(result_block.get("responsible"), dict):
        result_block["responsible"]["name"] = "[ИНЖЕНЕР]"

    raw_comments = comments_data.get("result", [])
    if not isinstance(raw_comments, list):
        raise RuntimeError("CmfComment.list вернул result не в формате списка")

    for comment in raw_comments:
        if not isinstance(comment, dict):
            continue

        c_owner = comment.get("cmf_owner_id")
        if isinstance(c_owner, dict):
            c_owner = c_owner.get("id")

        if task_owner_id and c_owner and str(c_owner) == str(task_owner_id):
            comment["author_role"] = "Client"
        elif not task_owner_id:
            comment["author_role"] = "Unknown"
        else:
            comment["author_role"] = "Engineer"

    result_block["comments_list"] = raw_comments
    ticket_data["result"] = result_block
    return ticket_data


def safe_filename_part(value: str, max_length: int = 80) -> str:
    value = WINDOWS_FORBIDDEN_CHARS_RE.sub("_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return (value or "ticket")[:max_length]


def ticket_file_path(output_dir: Path, rl_id: str, ticket_data: dict[str, Any]) -> Path:
    result_block = normalize_result_block(ticket_data)
    title_raw = str(result_block.get("name", rl_id))
    return output_dir / f"[{rl_id}] {safe_filename_part(title_raw)}.json"


def download_one_ticket(
    session: requests.Session,
    output_dir: Path,
    rl_id: str,
    internal_id: Any,
    task_owner_id: Any,
) -> None:
    payload_task = {
        "jsonrpc": "2.2",
        "method": "CmfTask.ui_get",
        "callid": str(uuid.uuid4()),
        "kwargs": {"filter": ["id", "==", internal_id]},
    }
    ticket_data = post_json(session, "CmfTask.ui_get", payload_task, timeout=30)

    payload_comments = {
        "jsonrpc": "2.0",
        "method": "CmfComment.list",
        "no_meta": True,
        "kwargs": {
            "filter": ["parent_id", "==", internal_id],
            "order_by": ["cmf_created_at"],
            "fields": ["id", "cmf_owner_id", "cmf_created_at", "text", "html"],
        },
    }
    comments_data = post_json(session, "CmfComment.list", payload_comments, timeout=30)

    enriched_ticket_data = enrich_ticket_data(ticket_data, comments_data, task_owner_id)
    file_path = ticket_file_path(output_dir, rl_id, enriched_ticket_data)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(enriched_ticket_data, file, ensure_ascii=False, indent=2)


def download_missing_tickets(
    ticket_refs: list[TicketRef],
    session: requests.Session,
    config: DownloadConfig,
) -> DownloadStats:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stats = DownloadStats(requested=len(ticket_refs))

    existing_rl_ids = downloaded_rl_ids(config.output_dir)
    log.info(f"На диске уже есть {len(existing_rl_ids)} JSON-файлов.")

    if config.force:
        target_refs = ticket_refs
    else:
        target_refs = [ref for ref in ticket_refs if ref.rl_id not in existing_rl_ids]
        stats.skipped_existing = len(ticket_refs) - len(target_refs)

    log.info(f"Осталось скачать: {len(target_refs)} файлов.")
    if not target_refs:
        return stats

    refs_without_internal_id = {ref.rl_id for ref in target_refs if not ref.internal_id}
    if refs_without_internal_id:
        id_map, owner_map = build_id_mapping(session, refs_without_internal_id, config.api_page_size)
    else:
        id_map, owner_map = {}, {}

    for index, ref in enumerate(target_refs, 1):
        rl_id = ref.rl_id
        internal_id = ref.internal_id or id_map.get(rl_id)
        task_owner_id = owner_map.get(rl_id)

        if not internal_id:
            stats.missing_in_portal += 1
            log.warning(f"[{index}/{len(target_refs)}] Тикет {rl_id} не найден в API портала.")
            continue

        log.info(f"[{index}/{len(target_refs)}] Загрузка {rl_id}...")
        try:
            download_one_ticket(session, config.output_dir, rl_id, internal_id, task_owner_id)
            stats.downloaded += 1
            log.info(f"[{index}/{len(target_refs)}] {rl_id} сохранён.")
            time.sleep(random.uniform(config.sleep_min, config.sleep_max))
        except RuntimeError as exc:
            stats.failed += 1
            log.error(f"Пропуск {rl_id}: {exc}")
        except Exception as exc:
            stats.failed += 1
            log.error(f"Ошибка при обработке {rl_id}: {exc}", exc_info=True)

    return stats


def main() -> int:
    config = parse_args()
    try:
        session = make_session(get_cookie_string())
        ticket_refs = get_ticket_refs_from_csv(config.csv_path, config.date_from, config.limit)
        if not ticket_refs:
            log.warning("Список тикетов пуст.")
            return 0

        stats = download_missing_tickets(ticket_refs, session, config)
        log.info(
            "Сессия завершена. "
            f"Всего из CSV: {stats.requested}; "
            f"пропущено существующих: {stats.skipped_existing}; "
            f"не найдено в портале: {stats.missing_in_portal}; "
            f"скачано: {stats.downloaded}; "
            f"ошибок: {stats.failed}."
        )
        return 1 if stats.failed else 0
    except KeyboardInterrupt:
        log.warning("Процесс прерван пользователем.")
        return 130
    except Exception as exc:
        log.critical(f"Непредвиденная ошибка в основном цикле: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

