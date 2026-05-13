import sys
from pathlib import Path
import json

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
# Импортируем наш обновленный парсер
from src.document_processing.parsers_tickets_with_llm_v3 import process_ticket_file

TICKETS_DIR_STR = settings.source_dirs.get("support_tickets", "data/source_docs/docs_json/tickets")
TICKETS_DIR = Path(TICKETS_DIR_STR)

OUTPUT_FILE = Path("data/tickets_preview.json")
PORTAL_URL  = "https://support.prosyst.ru"

def main():
    print("=== ТЕСТОВЫЙ ПАРСИНГ ТИКЕТОВ (JSON API + РОЛИ) ===")
    print(f"Директория: {TICKETS_DIR}")
    
    use_llm = getattr(settings, "enable_smart_metadata", False)
    if use_llm:
        provider = getattr(settings, "ticket_active_llm", "ollama")
        model = getattr(settings, "ticket_llm_model_name", "qwen2.5")
        print(f"[+] Умное извлечение (LLM) ВКЛЮЧЕНО. Провайдер: {provider}, Модель: {model}")
    else:
        print("[-] Умное извлечение (LLM) ВЫКЛЮЧЕНО.")

    if not TICKETS_DIR.exists():
        print(f"\n❌ Ошибка: Папка {TICKETS_DIR} не найдена!")
        return

    results = []
    ticket_files = [f for f in TICKETS_DIR.rglob("*.json")]
    
    print(f"\nНайдено JSON-файлов: {len(ticket_files)}")

    for f in ticket_files:
        docs = process_ticket_file(f, portal_base_url=PORTAL_URL)
        for doc in docs:
            # Собираем расширенную информацию для проверки
            results.append({
                "ticket_id":   doc.metadata.get("ticket_id"),
                "doc_level":   doc.metadata.get("doc_level"),
                "equipment":   doc.metadata.get("equipment_type"),
                "category":    doc.metadata.get("category"), # НОВОЕ ПОЛЕ
                "status":      doc.metadata.get("status"),
                "text_length": len(doc.page_content),
                "text_preview": doc.page_content[:500] + "...", # Превью текста
                "metadata":    doc.metadata,
            })
            
            # Выводим в консоль короткую сводку по каждому документу
            level = doc.metadata.get("doc_level")
            eq = doc.metadata.get("equipment_type")
            cat = doc.metadata.get("category")
            print(f"  ✅ [{ticket_id_from_file(f)}] Level: {level:5} | Eq: {eq[:15]:15} | Cat: {cat}")

    # Сохраняем подробный отчет в JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)
    
    print(f"\n🚀 Готово! Результаты проверки сохранены в: {OUTPUT_FILE}")
    print("Совет: Откройте этот файл и проверьте, появились ли в тексте теги 👤 [КЛИЕНТ] и 🛠️ [ИНЖЕНЕР].")

def ticket_id_from_file(path):
    import re
    match = re.search(r'\[([A-Z]+-\d+)\]', path.name)
    return match.group(1) if match else "???"

if __name__ == "__main__":
    main()