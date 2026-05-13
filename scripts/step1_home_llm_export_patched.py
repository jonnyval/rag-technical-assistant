import sys
import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.document_processing.parsers_tickets_with_llm import process_ticket_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
relative_path_str = settings.source_dirs.get("support_tickets", "data/source_docs/docs_json/tickets")
TICKETS_DIR = PROJECT_ROOT / relative_path_str
OUTPUT_DIR  = PROJECT_ROOT / "data" / "llm_cache_tickets"
PORTAL_URL  = "https://support.prosyst.ru"

# ==========================================
# 1. ФИЛЬТРЫ ИЗ CSV (Настраиваем здесь)
# ==========================================
# ⚠️ УКАЖИТЕ ПУТЬ К ВАШЕМУ CSV ФАЙЛУ (можно указать абсолютный, либо относительно корня проекта)
CSV_FILE_PATH = "export_CmfPerson_9dbc49fa_d163_11ee_be6f_02420a000855_20260508225641.csv"

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
    "Шутемовв Егорр Сергеевичч", 
]

# ==========================================
# 2. ПОЛУЧЕНИЕ "РАЗРЕШЕННЫХ" ID
# ==========================================
def get_valid_ids_from_csv(csv_path: str) -> set:
    print(f"📊 Чтение и фильтрация справочника: {csv_path}")
    try:
        df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
        
        if 'Код' not in df.columns:
            print("❌ В CSV файле не найдена колонка 'Код'!")
            return set()

        total_before = len(df)
        df['Дата создания'] = pd.to_datetime(df['Дата создания'], utc=True, errors='coerce')

        df = df[df['Дата создания'] >= FILTER_DATE_FROM]
        df = df[df['Статус.Имя статуса'].isin(FILTER_STATUSES)]
        df = df[~df['Тип оборудования РегЛаб'].isin(FILTER_EXCLUDE_EQUIPMENT)]
        df = df[~df['Исполнитель.ФИО'].isin(FILTER_EXCLUDE_ENGINEERS)]
        df = df[~df['Постановщик.ФИО'].isin(FILTER_EXCLUDE_REQUESTERS)]

        print(f"✅ После фильтрации CSV: осталось {len(df)} строк из {total_before}.")
        
        raw_codes = df['Код'].dropna().unique().tolist()
        # Вытаскиваем ID тикетов (приводим к верхнему регистру для надежности)
        valid_ids = {str(code).strip().upper() for code in raw_codes if 'RL-' in str(code).upper()}
        
        print(f"🎯 Итоговый список для обработки: {len(valid_ids)} уникальных тикетов.\n")
        return valid_ids
        
    except Exception as e:
        print(f"❌ Ошибка при чтении CSV: {e}")
        return set()

# ==========================================
# 3. ОСНОВНАЯ ЛОГИКА
# ==========================================
def main():
    print("=== ДОМАШНИЙ ПК: ГЕНЕРАЦИЯ LLM-КЭША (ПО ФАЙЛАМ) ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Шаг 1: Получаем разрешенные ID из CSV
    valid_ids = get_valid_ids_from_csv(CSV_FILE_PATH)
    if not valid_ids:
        print("⚠️ Внимание: Список разрешенных тикетов пуст. Проверьте путь к CSV и заданные фильтры.")
        print("Остановка работы.")
        return

    # Шаг 2: Получаем все JSON файлы
    ticket_files = list(TICKETS_DIR.rglob("*.json"))
    print(f"Всего исходных файлов в папке: {len(ticket_files)}")

    # Шаг 3: Предварительная фильтрация файлов (чтобы tqdm показывал точный прогресс)
    filtered_files = []
    for f in ticket_files:
        match = re.search(r'\[([A-Z]+-\d+)\]', f.name)
        ticket_id = match.group(1).upper() if match else f.stem.upper()
        
        # Если тикет есть в валидном списке CSV — добавляем в очередь
        if ticket_id in valid_ids:
            filtered_files.append((f, ticket_id))

    print(f"Файлов, подходящих под фильтры CSV: {len(filtered_files)}\n")

    errors = []
    llm_ok = 0
    llm_fail = 0

    # Шаг 4: Обработка (идем только по отфильтрованным файлам)
    for f, ticket_id in tqdm(filtered_files, desc="Обработка LLM"):
        out_file = OUTPUT_DIR / f"{ticket_id}_llm.json"

        # Если кэш уже есть — пропускаем
        if out_file.exists():
            continue

        try:
            docs = process_ticket_file(f, portal_base_url=PORTAL_URL)

            # Считаем, был ли fact-документ обогащён LLM
            fact_docs = [d for d in docs if d.metadata.get("doc_level") == "fact"]
            if fact_docs and fact_docs[0].metadata.get("llm_symptoms"):
                llm_ok += 1
            elif fact_docs:
                llm_fail += 1

            ticket_data = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ]

            with open(out_file, "w", encoding="utf-8") as out:
                json.dump(ticket_data, out, ensure_ascii=False, indent=2)

        except Exception as e:
            errors.append({"file": f.name, "error": str(e)})
            tqdm.write(f"\nОшибка файла {f.name}: {e}")

    # --- Сохраняем лог ошибок ---
    if errors:
        error_log = OUTPUT_DIR / "_errors.json"
        with open(error_log, "w", encoding="utf-8") as ef:
            json.dump(errors, ef, ensure_ascii=False, indent=2)
        print(f"\n⚠️  Ошибок при обработке: {len(errors)} — подробности в {error_log}")
    else:
        print("\n✅ Все подходящие файлы обработаны без ошибок.")

    total_fact = llm_ok + llm_fail
    if total_fact > 0:
        pct = llm_ok / total_fact * 100
        print(f"📊 LLM-обогащение fact-документов: {llm_ok}/{total_fact} успешно ({pct:.1f}%)")
        if pct < 80:
            print("   ⚠️  Процент успеха ниже 80% — проверьте загрузку Ollama или переключитесь на Groq.")

    print(f"✅ Готово! Запакуйте папку {OUTPUT_DIR} в архив и перенесите на рабочий ПК.")


if __name__ == "__main__":
    main()