# Обновление базы тикетов

Инструкция описывает ручной pipeline: скачать новые обращения из портала, создать LLM-cache, отфильтровать cache и загрузить тикеты в Qdrant.

## Что нужно заранее

1. Актуальная CSV-выгрузка обращений из портала поддержки.
2. Актуальная переменная `SUPPORT_COOKIES` в `.env`.
3. API-ключи для LLM, если используется режим `--llm-mode api`:
   - `GOOGLE_API_KEYS` для Gemini;
   - `GROQ_API_KEYS` для Groq.
4. Для векторизации должен быть доступен Qdrant на `http://localhost:6333`.

Python в этом проекте запускается так:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe <script.py>
```

## 1. Скачать тикеты из CSV

Скрипт:

```text
scripts\download_tickets_csv_to_json.py
```

Пример: скачать обращения с 1 июня 2026 по дату, которая есть в CSV-выгрузке:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\download_tickets_csv_to_json.py `
  --output-dir data\source_docs\docs_json\tickets_2026-06-01_2026-07-07 `
  --date-from 2026-06-01
```

Если не указан `--csv`, скрипт берет путь из переменной окружения `TICKETS_CSV`, а если ее нет - из `DEFAULT_CSV_FILE_PATH` внутри скрипта.

Лучше явно указывать CSV:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\download_tickets_csv_to_json.py `
  --csv "Выгрузка обращений (5).csv" `
  --output-dir data\source_docs\docs_json\tickets_2026-06-01_2026-07-07 `
  --date-from 2026-06-01
```

Скрипт скачивает только тикеты, которые есть в CSV и проходят встроенные фильтры по дате, статусу, оборудованию, инженерам и постановщикам.

Защита от дублей работает внутри указанной папки `--output-dir`: если файл вида `[RL-...] ... .json` уже есть, он будет пропущен. Флаг `--force` отключает эту защиту и перекачивает файлы.

Для тестового запуска:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\download_tickets_csv_to_json.py `
  --csv "Выгрузка обращений (5).csv" `
  --output-dir data\source_docs\docs_json\tickets_test `
  --date-from 2026-06-01 `
  --limit 3
```

## 2. Создать LLM-cache

Скрипт:

```text
scripts\step1_home_llm_export_patched.py
```

Пример для API-режима:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\step1_home_llm_export_patched.py `
  --tickets-dir data\source_docs\docs_json\tickets_2026-06-01_2026-07-07 `
  --output-dir data\llm_cache_tickets_2026-06-01_2026-07-07 `
  --llm-mode api `
  --enable-smart-metadata
```

Для тестового запуска:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\step1_home_llm_export_patched.py `
  --tickets-dir data\source_docs\docs_json\tickets_2026-06-01_2026-07-07 `
  --output-dir data\llm_cache_tickets_2026-06-01_2026-07-07 `
  --llm-mode api `
  --enable-smart-metadata `
  --limit 3
```

Без `--force` уже созданные cache-файлы пропускаются. Для пересоздания cache добавь `--force`.

В режиме `--llm-mode api` сейчас используется ротация провайдеров:

1. Gemini: `gemini-2.5-flash`.
2. Groq: `openai/gpt-oss-120b`.

Порядок задается в `config.yaml`:

```yaml
applications:
  support_tickets:
    processing:
      api_provider_order: ["gemini", "groq"]
```

Если API-ключ или лимит исчерпан, скрипт переключается на следующий ключ, затем на следующий провайдер. Если все API-провайдеры исчерпаны, cache-файл может быть создан без полноценного LLM-обогащения `llm_symptoms` / `llm_solution`; это будет видно в итоговой статистике как `llm_fail`.

Для локального режима:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\step1_home_llm_export_patched.py `
  --tickets-dir data\source_docs\docs_json\tickets_2026-06-01_2026-07-07 `
  --output-dir data\llm_cache_tickets_2026-06-01_2026-07-07 `
  --llm-mode local `
  --enable-smart-metadata
```

Локальный режим использует Ollama и модель из `config.yaml`, сейчас это `qwen3:8b`.

## 3. Отфильтровать LLM-cache

Скрипт:

```text
scripts\filter_llm_ticket_cache.py
```

Пример:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\filter_llm_ticket_cache.py `
  --input-dir data\llm_cache_tickets_2026-06-01_2026-07-07 `
  --output-dir data\llm_cache_tickets_filtered_2026-06-01_2026-07-07
```

Фильтр убирает шумные, служебные и нерелевантные обращения перед векторизацией. В выходной папке также создается CSV-отчет фильтрации.

## 4. Векторизовать cache в Qdrant

Перед запуском проверь, что Qdrant доступен:

```powershell
docker compose up -d
```

Скрипт:

```text
scripts\step2_work_vectorize_cache.py
```

Пример:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\step2_work_vectorize_cache.py `
  --cache-dir data\llm_cache_tickets_filtered_2026-06-01_2026-07-07
```

Для проверки cache без записи в Qdrant:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\step2_work_vectorize_cache.py `
  --cache-dir data\llm_cache_tickets_filtered_2026-06-01_2026-07-07 `
  --dry-run
```

## 5. Проверить поиск по тикетам

После векторизации можно выполнить smoke test:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\test_tickets_search.py
```

## Рекомендуемый порядок запуска

1. Скачать JSON тикетов из CSV.
2. Создать LLM-cache.
3. Отфильтровать LLM-cache.
4. Проверить доступность Qdrant.
5. Векторизовать отфильтрованный cache.
6. Проверить поиск по тикетам.

## Можно ли объединить pipeline

Да. Это лучше называть не CI/CD, а data ingestion pipeline или ETL-пайплайн.

CI/CD - это обычно автоматическая проверка, сборка и деплой кода после изменений в Git. Здесь задача другая: регулярно обновлять данные тикетов, прогонять LLM-обогащение и векторизацию.

Объединять можно двумя способами:

1. Простой PowerShell-скрипт, который последовательно вызывает четыре команды.
2. Python-оркестратор, например `scripts/update_tickets_pipeline.py`, который принимает `--csv`, `--date-from`, `--run-id`, `--llm-mode`, умеет запускать этапы по одному и продолжать с нужного шага.

Для production-процесса лучше Python-оркестратор, потому что в нем проще:

- проверять входные папки и CSV;
- создавать согласованные имена папок;
- останавливать pipeline при ошибках;
- логировать итог каждого этапа;
- добавлять режимы `--download-only`, `--cache-only`, `--vectorize-only`;
- делать `--dry-run`.

Минимальный набор аргументов для такого объединенного скрипта:

```text
--csv
--date-from
--run-id
--llm-mode
--limit
--force-download
--force-cache
--skip-download
--skip-cache
--skip-filter
--skip-vectorize
```

Пример будущего запуска:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\update_tickets_pipeline.py `
  --csv "Выгрузка обращений (5).csv" `
  --date-from 2026-06-01 `
  --run-id 2026-06-01_2026-07-07 `
  --llm-mode api
```
