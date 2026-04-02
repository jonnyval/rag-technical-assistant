import sys
from pathlib import Path

# Добавляем корень проекта для корректных импортов
sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb
from src.config import settings

def inspect_database(db_name: str, db_path: str):
    print(f"\n{'='*80}")
    print(f"🔍 ИНСПЕКЦИЯ БАЗЫ: {db_name.upper()}")
    print(f"📁 Путь: {db_path}")
    print(f"{'='*80}")

    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=settings.collection_name)
        
        total_docs = collection.count()
        print(f"📊 Всего чанков: {total_docs}")
        if total_docs == 0:
            print("База пуста.")
            return

        # Попробуем достать 1 документ формата docx и 1 формата html (если есть)
        samples_to_show = []
        
        # Ищем HTML
        try:
            html_docs = collection.get(where={"format": "html"}, limit=1, include=["documents", "metadatas"])
            if html_docs['ids']: samples_to_show.append((html_docs['ids'][0], html_docs['metadatas'][0], html_docs['documents'][0]))
        except: pass

        # Ищем DOCX
        try:
            docx_docs = collection.get(where={"format": "docx"}, limit=1, include=["documents", "metadatas"])
            if docx_docs['ids']: samples_to_show.append((docx_docs['ids'][0], docx_docs['metadatas'][0], docx_docs['documents'][0]))
        except: pass

        # Если поиск по фильтрам не дал результата, берем просто 2 первых попавшихся
        if not samples_to_show:
            any_docs = collection.get(limit=2, include=["documents", "metadatas"])
            for i in range(len(any_docs['ids'])):
                samples_to_show.append((any_docs['ids'][i], any_docs['metadatas'][i], any_docs['documents'][i]))

        # Вывод результатов
        for doc_id, meta, text in samples_to_show:
            print(f"\n📄 [ID: {doc_id}]")
            print("📌 МЕТАДАННЫЕ:")
            for k, v in meta.items():
                print(f"   - {k}: {v}")
            print("📝 ТЕКСТ ЧАНКА (первые 300 символов):")
            print("-" * 40)
            print(text[:300].replace('\n', ' ↵ ') + "...")
            print("-" * 40)

    except Exception as e:
        print(f"❌ Ошибка доступа к базе (возможно, путь не существует): {e}")

def main():
    # 1. Проверяем активную базу
    inspect_database(settings.active_db_name, settings.db_path)

    # 2. Проверяем остальные базы из конфига
    for db_name, db_info in settings.all_databases.items():
        inspect_database(db_name, db_info['path'])

if __name__ == "__main__":
    main()