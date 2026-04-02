import sys
from pathlib import Path

# Добавляем корень проекта в пути Python, чтобы импорты из src работали корректно
sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb
from src.config import settings
from src.retrieval.bm25_utils import build_and_save_bm25_index

def main():
    print("\n" + "="*50)
    print("🚀 СБОРКА ИНДЕКСА BM25")
    print("="*50)
    
    print(f"🔌 Подключение к векторной базе: {settings.db_path}")
    
    # 1. Инициализируем клиента ChromaDB
    chroma_client = chromadb.PersistentClient(path=settings.db_path)
    
    try:
        collection = chroma_client.get_collection(name=settings.collection_name)
        print(f"📂 Коллекция '{settings.collection_name}' успешно найдена.")
    except Exception as e:
        print(f"❌ Ошибка: Коллекция '{settings.collection_name}' не найдена в базе! Проверьте пути в config.yaml.")
        return

    # 2. Берем путь для сохранения кэша прямо из нашего единого конфига
    cache_path = settings.bm25_cache
    
    # 3. Запускаем магию сборки
    build_and_save_bm25_index(chroma_collection=collection, save_path=cache_path)
    
    print("\n🎉 Готово! Теперь ваш RAG полностью укомплектован гибридным поиском.")

if __name__ == "__main__":
    main()