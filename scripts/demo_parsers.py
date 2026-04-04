import sys
from pathlib import Path
import json

# Подключаем корень проекта
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.document_processing.parsers import process_docx_file

# ==========================================
# ⚙️ НАСТРОЙКИ ДЕМОНСТРАЦИИ
# ==========================================
# Укажите путь к вашему маленькому файлу:
FILE_TO_TEST = r"C:\Users\e.valov\testt\clean_code\data\source_docs\docs_word\test\R500\astra_ide_oc_linux.docx" 

# Делаем искусственно МАЛЕНЬКИЙ размер чанка (100 символов), 
# чтобы спровоцировать разрезание абзацев наживо!
DEMO_CHUNK_SIZE = 100
DEMO_CHUNK_OVERLAP = 20
MAX_CHUNKS_TO_SHOW = 20 # <--- ДОБАВИЛИ СРЕЗ [:MAX_CHUNKS_TO_SHOW]
def run_demo():
    file_path = Path(FILE_TO_TEST)
    if not file_path.exists():
        print(f"❌ Файл не найден: {file_path}")
        return

    print("🔍 ЗАПУСК РЕНТГЕНА ДОКУМЕНТА...")
    print(f"📄 Файл: {file_path.name}")
    print(f"✂️ Лимит чанка: {DEMO_CHUNK_SIZE} символов\n")

    # Вызываем наш реальный боевой парсер!
    chunks = process_docx_file(
        file_path=file_path,
        source_type="r500_docx",      # Имитируем, что это папка R500
        images_out_dir="data/images", # Папка для картинок (если бы они там были)
        chunk_size=DEMO_CHUNK_SIZE,
        chunk_overlap=DEMO_CHUNK_OVERLAP
    )

    print(f"🎉 Парсинг завершен! Документ разбит на {len(chunks)} чанков.\n")
    print("="*60)

    # Красиво выводим каждый чанк
    for i, doc in enumerate(chunks[:MAX_CHUNKS_TO_SHOW]): # <--- ДОБАВИЛИ СРЕЗ [:MAX_CHUNKS_TO_SHOW]
        meta = doc.metadata
        print(f"📦 ЧАНК №{i+1} " + (f"(Часть {meta.get('chunk_part')})" if meta.get('chunk_part') else ""))
        print("-" * 60)
        print(f"🔹 Оборудование : {meta.get('equipment_type')}")
        print(f"🔹 Раздел       : {meta.get('breadcrumb_raw')}")
        print(f"🔹 Длина текста : {meta.get('chunk_length')} символов")
        print("-" * 60)
        print("📝 ТЕКСТ, КОТОРЫЙ УЙДЕТ В НЕЙРОСЕТЬ:")
        print(doc.page_content)
        print("=" * 60 + "\n")

    # Если чанков было больше, чем мы вывели, сообщаем об этом
    if len(chunks) > MAX_CHUNKS_TO_SHOW:
        print(f"👀 ... и еще {len(chunks) - MAX_CHUNKS_TO_SHOW} чанков скрыто для компактности вывода.")

if __name__ == "__main__":
    run_demo()