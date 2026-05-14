import sys
from pathlib import Path

# Добавляем корень проекта для корректных импортов
sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import settings

def inspect_chroma_database(db_name: str, db_path: str, collection_name: str):
    """Показывает несколько документов из локальной Chroma-базы для ручной диагностики."""

    """Инспекция для баз данных ChromaDB"""
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        
        total_docs = collection.count()
        print(f"📊 Всего чанков (ChromaDB): {total_docs}")
        if total_docs == 0:
            print("База пуста.")
            return

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

        print_samples(samples_to_show)

    except Exception as e:
        print(f"❌ Ошибка доступа к базе Chroma '{db_name}': {e}")


def inspect_qdrant_database(db_name: str, db_path: str, collection_name: str):
    """Показывает несколько документов из Qdrant-коллекции для ручной диагностики."""

    """Инспекция для баз данных Qdrant"""
    try:
        client = QdrantClient(path=db_path)
        if not client.collection_exists(collection_name):
            print(f"❌ Коллекция '{collection_name}' не найдена в Qdrant.")
            return

        collection_info = client.get_collection(collection_name=collection_name)
        total_docs = collection_info.points_count
        print(f"📊 Всего чанков (Qdrant): {total_docs}")
        
        if total_docs == 0:
            print("База пуста.")
            return

        samples_to_show = []

        # Ищем HTML через Scroll API
        html_res, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="metadata.format", match=models.MatchValue(value="html"))]
            ),
            limit=1,
            with_payload=True
        )
        if html_res:
            payload = html_res[0].payload
            samples_to_show.append((html_res[0].id, payload.get("metadata", {}), payload.get("page_content", "")))

        # Ищем DOCX через Scroll API
        docx_res, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="metadata.format", match=models.MatchValue(value="docx"))]
            ),
            limit=1,
            with_payload=True
        )
        if docx_res:
            payload = docx_res[0].payload
            samples_to_show.append((docx_res[0].id, payload.get("metadata", {}), payload.get("page_content", "")))

        # Если пусто, берем любые 2
        if not samples_to_show:
            any_res, _ = client.scroll(
                collection_name=collection_name,
                limit=2,
                with_payload=True
            )
            for res in any_res:
                payload = res.payload
                samples_to_show.append((res.id, payload.get("metadata", {}), payload.get("page_content", "")))

        print_samples(samples_to_show)

    except Exception as e:
        print(f"❌ Ошибка доступа к базе Qdrant '{db_name}': {e}")


def print_samples(samples):
    """Печатает найденные образцы документов с metadata и коротким preview текста."""

    """Вспомогательная функция для красивого вывода в консоль"""
    for doc_id, meta, text in samples:
        print(f"\n📄 [ID: {doc_id}]")
        print("📌 МЕТАДАННЫЕ:")
        for k, v in meta.items():
            print(f"   - {k}: {v}")
        print("📝 ТЕКСТ ЧАНКА (первые 300 символов):")
        print("-" * 40)
        # Обработка NoneText
        text_preview = str(text)[:300].replace('\n', ' ↵ ') + "..." if text else "Нет текста"
        print(text_preview)
        print("-" * 40)


def inspect_database(db_name: str, db_path: str, collection_name: str):
    """Выбирает тип инспекции по имени/пути базы и запускает соответствующую проверку."""

    print(f"\n{'='*80}")
    print(f"🔍 ИНСПЕКЦИЯ БАЗЫ: {db_name.upper()}")
    print(f"📁 Путь: {db_path} | Коллекция: {collection_name}")
    print(f"{'='*80}")

    # Умная маршрутизация: определяем тип БД по имени в config.yaml
    if "qdrant" in db_name.lower():
        inspect_qdrant_database(db_name, db_path, collection_name)
    else:
        inspect_chroma_database(db_name, db_path, collection_name)


def main():
    """Запускает интерактивную инспекцию всех баз, описанных в конфиге."""

    # 1. Проверяем активную базу
    inspect_database(settings.active_db_name, settings.db_path, settings.collection_name)

    # 2. Проверяем остальные базы из конфига
    for db_name, db_info in settings.all_databases.items():
        if isinstance(db_info, dict) and 'path' in db_info:
            # Динамически достаем имя коллекции для конкретной базы
            col_name = db_info.get("collection", "reglab_tech_docs") 
            inspect_database(db_name, db_info['path'], col_name)

if __name__ == "__main__":
    main()
