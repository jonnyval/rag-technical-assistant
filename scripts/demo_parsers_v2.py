import sys
import time  # <-- ДОБАВИЛИ МОДУЛЬ ВРЕМЕНИ
from pathlib import Path

# Подключаем корень проекта
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ИМПОРТИРУЕМ НОВЫЙ ПАРСЕР
from src.document_processing.parsers_qdrant_v2 import process_docx_file

# ==========================================
# ⚙️ НАСТРОЙКИ ДЕМОНСТРАЦИИ
# ==========================================
# Укажите путь к вашему файлу (желательно небольшому для теста)
FILE_TO_TEST = r"C:\Users\jonny\clean_code\data\source_docs\docs_word\R500\astra_ide_oc_linux.docx" 

MAX_CHUNKS_TO_SHOW = 5 

def run_demo():
    file_path = Path(FILE_TO_TEST)
    if not file_path.exists():
        print(f"❌ Файл не найден: {file_path}")
        print("Пожалуйста, укажите правильный путь к тестовому файлу в переменной FILE_TO_TEST.")
        return

    print("🔍 ЗАПУСК РЕНТГЕНА ДОКУМЕНТА С OLLAMA (QWEN)...")
    print(f"📄 Файл: {file_path.name}\n")

    # === ⏱ НАЧИНАЕМ ОТСЧЕТ ВРЕМЕНИ ===
    start_time = time.time()

    # Вызываем наш боевой парсер
    chunks = process_docx_file(
        file_path=file_path,
        source_type="r500_docx",      # Имитируем папку
        images_out_dir="data/images"
    )

    # === ⏱ ОСТАНАВЛИВАЕМ ОТСЧЕТ ===
    end_time = time.time()
    
    total_time = end_time - start_time
    chunks_count = len(chunks)
    avg_time = total_time / chunks_count if chunks_count > 0 else 0

    print("="*80)
    print(f"🎉 Парсинг завершен!")
    print(f"📊 СТАТИСТИКА СКОРОСТИ:")
    print(f"   • Всего чанков: {chunks_count} шт.")
    print(f"   • Общее время : {total_time:.2f} сек.")
    print(f"   • Среднее время на 1 чанк (Ollama): {avg_time:.2f} сек/чанк")
    print("="*80 + "\n")

    # Красиво выводим каждый чанк
    for i, doc in enumerate(chunks[:MAX_CHUNKS_TO_SHOW]):
        meta = doc.metadata
        print(f"📦 ЧАНК №{i+1}")
        print("-" * 80)
        print(f"🔹 Оборудование : {meta.get('equipment_type')}")
        print(f"🔹 Раздел       : {meta.get('breadcrumb_raw')}")
        
        # ВЫВОДИМ УМНЫЕ МЕТАДАННЫЕ ОТ QWEN
        print(f"🧠 Ключевые слова: {meta.get('keywords', 'Не сгенерировано')}")
        print(f"❓ Вопросы: {meta.get('generated_questions', 'Не сгенерировано')}")
        
        print("-" * 80)
        print("📝 ТЕКСТ (с внедренными метаданными для векторного поиска):")
        print(doc.page_content[:1500] + "\n... [текст обрезан для вывода] ...") 
        print("=" * 80 + "\n")

    if len(chunks) > MAX_CHUNKS_TO_SHOW:
        print(f"👀 ... и еще {len(chunks) - MAX_CHUNKS_TO_SHOW} чанков скрыто для компактности вывода.")

if __name__ == "__main__":
    run_demo()