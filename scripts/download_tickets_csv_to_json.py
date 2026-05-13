import os
import time
import random
import requests
import json
import re
import uuid
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.logger import log

load_dotenv()

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
COOKIE_STRING = os.getenv("SUPPORT_COOKIES")
if not COOKIE_STRING:
    raise ValueError("Ошибка: Переменная SUPPORT_COOKIES не найдена в .env!")

OUTPUT_DIR = Path("data/source_docs/docs_json/tickets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ⚠️ УКАЖИТЕ ПУТЬ К ВАШЕМУ CSV ФАЙЛУ ⚠️
CSV_FILE_PATH = "export.CmfPerson_9dbc49fa-d163-11ee-be6f-02420a000855.20260508225641.csv"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
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
    "r08": "Вопросы по порталу тех. поддержки"
}

CATEGORY_MAPPING = {
    "otkaz-obor-prosoft": "ОтказОборПрософт", "novyj-funkczional": "Новый функционал",
    "nepolnaya-dokumentacziya": "Неполная документация", "spravochnoe": "Справочное",
    "tehobsluzhivanie": "Техобслуживание", "otkaz-obor-chuzhogo": "ОтказОборЧужого",
    "oshibka-nastrojki-chuzhaya": "ОшибкаНастройкиЧужая", "oshibka-nastrojki-prosoft": "ОшибкаНастройкиПрософт",
    "otkaz-prog-chuzhoj": "ОтказПрогЧужой", "otkaz-prog-prosoft": "ОтказПрогПрософт"
}

# ==========================================
# 2. ФИЛЬТРЫ (редактируй здесь)
# ==========================================
FILTER_DATE_FROM = "2022-01-01"

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

# --- ПАРАМЕТРЫ RATE LIMITING ---
MAX_RETRIES = 3
BASE_BACKOFF = 5  # секунд


def get_cookies_dict():
    return {k: v for item in COOKIE_STRING.split(';') if '=' in item for k, v in [item.strip().split('=', 1)]}


# ==========================================
# 3. HTTP-ОБЁРТКА С EXPONENTIAL BACKOFF
# ==========================================
def request_with_retry(url, **kwargs):
    """POST с автоматическим exponential backoff при 429/502/503."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, **kwargs)

            if resp.status_code == 429:
                wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 2)
                log.warning(f"Rate limit (429). Жду {wait:.1f}с... (попытка {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            if resp.status_code in (502, 503):
                wait = BASE_BACKOFF * (2 ** attempt)
                log.warning(f"Ошибка сервера ({resp.status_code}). Жду {wait}с... (попытка {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.ConnectionError:
            wait = BASE_BACKOFF * (2 ** attempt)
            log.warning(f"Сетевая ошибка, попытка {attempt + 1}/{MAX_RETRIES}. Жду {wait}с...")
            time.sleep(wait)

    raise RuntimeError(f"Не удалось получить ответ от {url} после {MAX_RETRIES} попыток")


# ==========================================
# 4. ЧТЕНИЕ И ФИЛЬТРАЦИЯ CSV
# ==========================================
def get_rl_ids_from_csv(csv_path):
    log.info(f"Чтение CSV файла: {csv_path}")
    try:
        df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')

        if 'Код' not in df.columns:
            log.error("В CSV файле не найдена колонка 'Код'!")
            return []

        total_before = len(df)
        df['Дата создания'] = pd.to_datetime(df['Дата создания'], utc=True, errors='coerce')

        # --- Применяем фильтры ---
        df = df[df['Дата создания'] >= FILTER_DATE_FROM]
        df = df[df['Статус.Имя статуса'].isin(FILTER_STATUSES)]
        df = df[~df['Тип оборудования РегЛаб'].isin(FILTER_EXCLUDE_EQUIPMENT)]
        df = df[~df['Исполнитель.ФИО'].isin(FILTER_EXCLUDE_ENGINEERS)]
        df = df[~df['Постановщик.ФИО'].isin(FILTER_EXCLUDE_REQUESTERS)]

        log.info(f"После фильтрации: {len(df)} строк из {total_before} (убрано {total_before - len(df)})")

        # --- Извлекаем RL-номера ---
        raw_codes = df['Код'].dropna().unique().tolist()
        rl_ids = [str(code).strip() for code in raw_codes if 'RL-' in str(code).upper()]

        log.info(f"Уникальных тикетов для скачивания: {len(rl_ids)}")
        return rl_ids

    except Exception as e:
        log.error(f"Ошибка при чтении/фильтрации CSV: {e}")
        return []


# ==========================================
# 5. ПОЛУЧЕНИЕ ВСЕХ ID ОДНИМ ЗАПРОСОМ
# ==========================================
def build_id_mapping():
    """Скачивает легкий список всех тикетов и создает словарь RL-ID -> Internal ID"""
    log.info("Сбор справочника внутренних ID со всех тикетов портала (это займет пару секунд)...")
    mapping = {}
    owner_mapping = {}

    payload = {
        "jsonrpc": "2.0",
        "method": "CmfTask.list",
        "no_meta": True,
        "kwargs": {
            "limit": 50000,
            "offset": 0
        }
    }

    try:
        resp = request_with_retry(
            "https://support.prosyst.ru/api/?m=CmfTask.list",
            json=payload, cookies=get_cookies_dict(), headers=HEADERS, timeout=60
        )
        batch = resp.json().get('result', [])

        for ticket in batch:
            ticket_str = json.dumps(ticket, ensure_ascii=False)
            match = re.search(r'(RL-\d+)', ticket_str, re.IGNORECASE)
            internal_id = ticket.get("id")

            if match and internal_id:
                t_id = match.group(1).upper()
                mapping[t_id] = internal_id
                owner_mapping[t_id] = ticket.get('cmf_owner_id')

        log.info(f"Справочник готов! Найдено {len(mapping)} тикетов в базе портала.")
        return mapping, owner_mapping
    except Exception as e:
        log.error(f"Ошибка при создании справочника ID: {e}")
        return {}, {}


# ==========================================
# 6. СКАЧИВАНИЕ И ОБОГАЩЕНИЕ ДАННЫХ
# ==========================================
def download_missing_tickets(all_rl_ids):
    downloaded_files = {f.name.split(']')[0][1:] for f in OUTPUT_DIR.glob("*.json") if ']' in f.name}
    log.info(f"На диске уже есть {len(downloaded_files)} файлов. Они будут пропущены.")

    missing_rl_ids = [rl for rl in all_rl_ids if rl not in downloaded_files]
    log.info(f"Осталось скачать: {len(missing_rl_ids)} файлов.")

    if not missing_rl_ids:
        log.info("Все тикеты из CSV уже скачаны!")
        return

    id_map, owner_map = build_id_mapping()

    downloaded_in_session = 0

    for index, rl_id in enumerate(missing_rl_ids, 1):
        log.info(f"[{index}/{len(missing_rl_ids)}] Загрузка {rl_id}...")

        try:
            internal_id = id_map.get(rl_id)
            task_owner_id = owner_map.get(rl_id)

            if not internal_id:
                log.warning(f"Тикет {rl_id} не найден в базе портала. Возможно, он был удален.")
                continue

            # Карточка
            payload_task = {
                "jsonrpc": "2.2", "method": "CmfTask.ui_get", "callid": str(uuid.uuid4()),
                "kwargs": {"filter": ["id", "==", internal_id]}
            }
            resp_task = request_with_retry(
                "https://support.prosyst.ru/api/?m=CmfTask.ui_get",
                json=payload_task, cookies=get_cookies_dict(), headers=HEADERS, timeout=30
            )
            ticket_data = resp_task.json()

            # Комментарии
            payload_comments = {
                "jsonrpc": "2.0", "method": "CmfComment.list", "no_meta": True,
                "kwargs": {
                    "filter": ["parent_id", "==", internal_id],
                    "order_by": ["cmf_created_at"],
                    "fields": ["id", "cmf_owner_id", "cmf_created_at", "text", "html"]
                }
            }
            resp_comments = request_with_retry(
                "https://support.prosyst.ru/api/?m=CmfComment.list",
                json=payload_comments, cookies=get_cookies_dict(), headers=HEADERS, timeout=30
            )
            comments_data = resp_comments.json()

            # --- ОБРАБОТКА ДАННЫХ ---
            result_block = ticket_data.get('result', {})
            if isinstance(result_block, list) and len(result_block) > 0:
                result_block = result_block[0]

            if not task_owner_id:
                task_owner_id = result_block.get('cmf_owner_id')
                if not task_owner_id and isinstance(result_block.get('cmf_owner'), dict):
                    task_owner_id = result_block['cmf_owner'].get('id')

            eq_code = result_block.get('cf_tip_oborud_reg')
            if eq_code in EQUIPMENT_MAPPING:
                result_block['cf_tip_oborud_reg_name'] = EQUIPMENT_MAPPING[eq_code]

            cat_code = result_block.get('cf_kategoriya_or')
            if cat_code in CATEGORY_MAPPING:
                result_block['cf_kategoriya_or_name'] = CATEGORY_MAPPING[cat_code]

            if isinstance(result_block.get('cmf_owner'), dict):
                result_block['cmf_owner']['name'] = "[КЛИЕНТ]"
            if isinstance(result_block.get('responsible'), dict):
                result_block['responsible']['name'] = "[ИНЖЕНЕР]"

            raw_comments = comments_data.get('result', [])
            for comment in raw_comments:
                c_owner = comment.get('cmf_owner_id')
                if isinstance(c_owner, dict):
                    c_owner = c_owner.get('id')

                if task_owner_id and c_owner and str(c_owner) == str(task_owner_id):
                    comment['author_role'] = 'Client'
                elif not task_owner_id:
                    comment['author_role'] = 'Unknown'
                else:
                    comment['author_role'] = 'Engineer'

            result_block['comments_list'] = raw_comments
            ticket_data['result'] = result_block

            # --- СОХРАНЕНИЕ ---
            title_raw = str(result_block.get('name', rl_id))
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', title_raw)[:60]
            file_path = OUTPUT_DIR / f"[{rl_id}] {safe_title}.json"

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(ticket_data, f, ensure_ascii=False, indent=2)

            downloaded_in_session += 1
            log.info(f"✅ [{index}/{len(missing_rl_ids)}] {rl_id} сохранён.")

            time.sleep(random.uniform(0.3, 0.8))

        except RuntimeError as e:
            log.error(f"Пропуск {rl_id}: {e}")
            continue
        except Exception as e:
            log.error(f"Ошибка при обработке {rl_id}: {e}", exc_info=True)

    log.info(f"Сессия завершена. Новых файлов загружено: {downloaded_in_session}")


if __name__ == "__main__":
    try:
        all_ids = get_rl_ids_from_csv(CSV_FILE_PATH)
        if all_ids:
            download_missing_tickets(all_ids)
        else:
            log.warning("Список тикетов пуст.")
    except KeyboardInterrupt:
        log.warning("Процесс прерван пользователем.")
    except Exception as e:
        log.critical(f"Непредвиденная ошибка в основном цикле: {e}", exc_info=True)