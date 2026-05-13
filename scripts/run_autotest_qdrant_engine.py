import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

# Добавляем корень проекта для корректных импортов
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.config import settings
from src.logger import log
from src.engine import RAGEngine

# ==========================================
# 📝 АВТО-ДОКУМЕНТИРОВАНИЕ ТЕСТА
# ==========================================
def generate_test_readme(output_dir: Path):
    """Создает README.md с настройками текущего запуска, скрывая API ключи."""
    readme_path = output_dir / "README.md"
    
    # Собираем красивый Markdown
    content = f"""# Отчет о тестировании RAG-системы

*Автоматически сгенерированный файл конфигурации.*
**Дата прогона:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 🛠 Модели и Провайдеры
* **LLM Провайдер:** `{settings.active_llm.upper()}`
* **Модель генерации:** `{settings.llm_model_name}`
* **Эмбеддинги (Dense):** `{settings.embedding_model_name}`
* **Реранкер (Cross-Encoder):** `{settings.reranker_model_name}`

## 🗄 База данных и Чанкинг
* **Тип БД (Префикс):** `{settings.active_db_name}`
* **Коллекция Qdrant:** `{settings.collection_name}`
* **Размер дочернего чанка:** `{settings.child_chunk_size}` токенов/символов
* **Перекрытие чанков (Overlap):** `{settings.child_chunk_overlap}`

## 🔍 Настройки Поиска (Retrieval)
* **Первичное извлечение (top_k_retrieval):** `{settings.top_k_retrieval}` (количество чанков, отдаваемых реранкеру)
* **Финальный топ (top_k_final):** `{settings.top_k_final}` (идут в контекст LLM)
* **Порог отсечения реранкера (rerank_threshold):** `{settings.rerank_threshold}`
* **Lost-in-the-Middle сортировка:** `{"Включена" if settings.use_litm else "Отключена"}`
* **Использование HyDE:** `{"Да" if settings.use_hyde else "Нет"}`

## ⚙️ Дополнительно
* **Лимит вопросов на файл:** `{settings.max_questions_per_file if settings.max_questions_per_file else "Все (Без лимита)"}`
"""
    
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info(f"📝 Файл конфигурации сохранен: {readme_path}")

# ==========================================
# ⚙️ ЛОГИКА АВТОТЕСТИРОВАНИЯ
# ==========================================
def main():
    INPUT_DIR = Path("data/50_questions")
    OUTPUT_DIR = Path("result_test_auto") / settings.active_db_name
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_DIR.exists():
        log.error(f"Папка с исходными вопросами не найдена: {INPUT_DIR}")
        return

    # 1. Генерируем конфигурационный README
    generate_test_readme(OUTPUT_DIR)

    # 2. Инициализируем наше ядро
    log.info("Запуск RAGEngine для автотестов...")
    engine = RAGEngine()
    db_prefix = settings.active_db_name

    json_files = list(INPUT_DIR.glob("*.json"))
    log.info(f"📂 Найдено файлов для тестов: {len(json_files)}")

    for file_path in json_files:
        output_file = OUTPUT_DIR / f"{db_prefix}_{file_path.name}"
        
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log.info(f"Продолжаем работу с существующим файлом: {output_file.name}")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log.info(f"Начинаем новый файл: {file_path.name}")

        # Ограничиваем выборку если задан лимит
        if settings.max_questions_per_file:
            data = data[:settings.max_questions_per_file]
            log.info(f"🔢 Лимит из конфига: обрабатываем первые {len(data)} вопросов")

        for i, item in enumerate(data):
            if item.get("Ответ RAG") and str(item.get("Ответ RAG")).strip() != "":
                log.info(f"⏩ Вопрос {i+1}/{len(data)} уже отвечен. Пропускаем.")
                continue

            question = item.get("Вопрос", "")
            options = item.get("Варианты ответа", "")
            full_prompt = f"{question}\n\nВарианты ответа:\n{options}"
            
            log.info(f"▶️ Обработка вопроса {i+1}/{len(data)}...")
            
            try:
                # ✅ Вызываем генерацию
                response = engine.process_query(query=full_prompt)
                
                final_text = response.final_answer if response.final_answer else "Модель вернула пустой ответ."
                
                item["Ответ RAG"] = final_text
                item["Правильность"] = ""
                
                item["SGR_Audit"] = {
                    "Понятый интент": response.user_intent,
                    "Извлеченные факты": [f"{f.source_file}: {f.fact}" for f in response.extracted_facts] if response.extracted_facts else [],
                    "Чего не хватило": response.missing_context
                }
                
                log.info("✅ Успешный ответ получен.")
                
            except Exception as e:
                log.error(f"❌ Ошибка LLM на вопросе {i+1}: {e}")
                item["Ответ RAG"] = f"ОШИБКА: {e}"
                item["Правильность"] = ""

            # Сохранение промежуточного результата
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            # Ожидание
            if i < len(data) - 1:
                sleep_time = 30 if settings.active_llm != "gemini" else 5
                log.info(f"⏱️ Ожидание {sleep_time} секунд перед следующим вопросом...")
                time.sleep(sleep_time)

    log.info("\n🎉 ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ!")

if __name__ == "__main__":
    main()