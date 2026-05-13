import sys
import json
from pathlib import Path
from langchain_core.documents import Document
from tqdm import tqdm

# Определение корня проекта (на 2 уровня выше папки scripts)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings
from src.engine import RAGEngine

# Указываем абсолютный путь к папке с обработанными тикетами
CACHE_DIR = PROJECT_ROOT / "data" / "llm_cache_tickets"

def main():
    print("=== РАБОЧИЙ ПК: ВЕКТОРИЗАЦИЯ И ЗАГРУЗКА В БАЗУ ===")
    
    # КРИТИЧЕСКИ ВАЖНО: Выключаем LLM-обработку! 
    # Данные уже обработаны на домашнем ПК, нам не нужна Ollama на работе.
    # settings.enable_smart_metadata = False 
    
    print(f"Ищу кэш в папке: {CACHE_DIR.absolute()}")
    
    if not CACHE_DIR.exists():
        print(f"❌ ОШИБКА: Папка не найдена!")
        print("Убедитесь, что вы перенесли разархивированную папку с домашнего ПК.")
        return

    documents = []
    cache_files = list(CACHE_DIR.rglob("*.json"))
    print(f"Найдено файлов кэша: {len(cache_files)}")
    
    if len(cache_files) == 0:
        print("❌ ОШИБКА: Папка существует, но JSON-файлов внутри нет.")
        return

    print("\nЧтение файлов и применение предохранителей...")
    for f in tqdm(cache_files, desc="Подготовка документов"):
        try:
            with open(f, "r", encoding="utf-8") as file:
                data_list = json.load(file)
                for item in data_list:
                    content = item["page_content"]
                    
                    # ПРЕДОХРАНИТЕЛЬ: Обрезаем гигантские логи (защита от "монстра" на 47к символов)
                    if len(content) > 6000:
                        content = content[:3000] + "\n\n...[ДЛИННЫЕ ТЕКСТЫ ОБРЕЗАНЫ СИСТЕМОЙ ДЛЯ ОПТИМИЗАЦИИ ПОИСКА]...\n\n" + content[-3000:]
                    
                    doc = Document(
                        page_content=content,
                        metadata=item["metadata"]
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"\nОшибка при чтении файла {f.name}: {e}")
            
    print(f"\nУспешно собрано {len(documents)} готовых чанков.")
    
    print("\nИнициализация RAGEngine (подключение к базам и загрузка модели Qwen3)...")
    engine = RAGEngine()
    
    print("\n🚀 Начинаем векторизацию! Это может занять несколько минут...")
    try:
        # ПРАВИЛЬНЫЙ ВЫЗОВ ДЛЯ ВАШЕЙ АРХИТЕКТУРЫ
        engine.retriever.parent_retriever.add_documents(documents)
        print("\n✅ УСПЕХ! База обновлена. Векторы сохранены в Qdrant, полные тексты - в SQLite.")
    except Exception as e:
        print(f"\n❌ Ошибка при сохранении в базу: {e}")

if __name__ == "__main__":
    main()