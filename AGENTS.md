# AGENTS.md

Инструкции для AI-агентов, работающих с этим репозиторием. Действуют на весь проект, если в подпапке нет более конкретного `AGENTS.md`.

## Контекст проекта

Это RAG-система для технической поддержки RegLab/Prosyst: поиск по документации, поиск по историческим тикетам, генерация приватного ответа специалисту ТП, аналитика обращений и вспомогательные пайплайны подготовки данных.

Основной production-сценарий: обработка нового обращения через документацию и похожие тикеты. Важные свойства текущего контура:

- основной движок находится в `src/engine.py`;
- конфигурация читается из `config.yaml` и `.env` через `src/config.py`;
- Qdrant используется как основное векторное хранилище, настройки backend-ов лежат в `config.yaml`;
- документация и тикеты индексируются раздельно, retrieval объединяется в `src/retrieval/dual_retriever.py`;
- стабильные source-id вида `[D1]`/`[T1]` формируются в `src/context_formatting.py`;
- детерминированные правила по модулям ПЛК находятся в `src/module_detection.py`;
- post-guards против неподтвержденных выводов находятся в `src/evidence_guard.py`;
- HyDE и `equipment_filter` не считать обязательными для основного production-пути без проверки текущего кода и конфига.

## Команды и окружение

На этой машине Python для проекта запускается так:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe <script.py>
```

Используй этот интерпретатор вместо голого `python`, если запускаешь проектные скрипты, проверки или одноразовые команды.

Типовые команды:

```powershell
docker compose up -d
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\run_ingest_qdrant.py
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\api_server.py
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\test_tickets_search.py
streamlit run scripts\app_streamlit.py
```

Для аналитического приложения:

```powershell
Set-Location analytics\tp_analyze
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe run_server.py
```

или:

```powershell
Set-Location analytics\tp_analyze
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8080
```

Перед ingestion/API/тестами, которым нужен retrieval, проверь, что Qdrant доступен на `http://localhost:6333` и что активные коллекции/parent-store в `config.yaml` соответствуют ожидаемым данным.

## Структура

- `src/` - ядро RAG: конфиг, движок, логирование, retrieval, форматирование контекста, guards.
- `src/document_processing/` - парсинг документов/тикетов, очистка и подготовка parent/child-документов.
- `src/retrieval/` - Qdrant retriever-ы, dual search, rerank и вспомогательная retrieval-логика.
- `scripts/` - CLI, API-сервер, ingestion, autotest/evaluation и утилиты экспорта/подготовки.
- `analytics/tp_analyze/` - FastAPI-приложение аналитики обращений, dashboard data и admin rebuild.
- `faq_pipeline/`, `course_pipeline/`, `log_analysis_pipeline/` - отдельные пайплайны генерации FAQ/курсов/анализа логов.
- `data/`, `vector_dbs/`, `reports/`, `benchmark_results/`, `result_test_auto/` - рабочие данные, базы, кэши и результаты запусков.
- Папки `папка для копирования в нейронку*` и zip-архивы считать экспортными/снимками; не править их без прямой просьбы.

## Правила изменений

- Сначала читай существующий код и README рядом с изменяемой областью. В проекте много скриптов с близкими именами, не угадывай назначение по имени файла.
- Держи изменения узкими. Не рефактори одновременно ingestion, retrieval, аналитику и формат ответа, если задача не требует этого явно.
- Не меняй `config.yaml`, `.env`, пути к коллекциям Qdrant, parent-store или модели без явной причины и объяснения.
- Не удаляй и не пересоздавай `data/`, `vector_dbs/`, `reports/`, `benchmark_results/`, `result_test_auto/` без прямого разрешения.
- При изменении формата метаданных или индексации проверяй совместимость с `DualRetriever`, `context_formatting`, API-ответами и аналитическими скриптами.
- Новые правила распознавания модулей добавляй в `src/module_detection.py` или связанные данные таксономии, а не размазывай по retrieval/LLM-промптам.
- Изменения вида LLM-контекста и ссылок на источники делай в `src/context_formatting.py`.
- Защиту от неподтвержденных расшифровок, домыслов и постобработку ответа добавляй в `src/evidence_guard.py`.
- Сохраняй русскоязычную доменную терминологию и не переводь названия оборудования/модулей произвольно.
- Файлы проекта держи в UTF-8. Если консоль PowerShell показывает mojibake, не делай вывод о повреждении файла без проверки кодировки.

## Безопасность данных и секретов

- Не печатай и не пересказывай содержимое `.env`, `analytics/tp_analyze/server_data/private_token.txt`, API-ключи, токены, cookie, приватные URL и учетные данные.
- Не коммить секреты и не добавляй новые секреты в репозиторий. Для примеров используй плейсхолдеры вроде `your_key`.
- Тикеты, логи и выгрузки могут содержать персональные/клиентские данные. В ответах пользователю давай минимально необходимую выдержку и предпочитай агрегированные выводы.
- Не отправляй данные тикетов/логов во внешние сервисы и не запускай сетевые операции без явной необходимости.

## Проверки

В проекте нет единого тестового harness-а, поэтому выбирай минимальную проверку по области изменений:

- синтаксис/импорты измененных Python-файлов:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe -m py_compile path\to\file.py
```

- проверка конфига:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe -c "from src.config import settings; print(settings.active_llm)"
```

- smoke test RAGEngine, если затронуты engine/retrieval/config:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe -c "from src.engine import RAGEngine; RAGEngine(); print('RAGEngine OK')"
```

- поиск по тикетам:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\test_tickets_search.py
```

- аналитика:

```powershell
Set-Location analytics\tp_analyze
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\build_dashboard_data.py
```

Если проверку нельзя запустить из-за отсутствия Qdrant, ключей LLM, модели или данных, явно укажи это в финальном ответе.

## Git и рабочее дерево

В рабочем дереве могут быть чужие незакоммиченные изменения. Не откатывай и не перезаписывай их. Перед правками проверяй `git status --short`; если файл уже изменен, читай его внимательно и редактируй поверх текущего состояния.

Не используй destructive-команды вроде `git reset --hard`, удаления баз/кэшей или массовой очистки результатов без прямой просьбы пользователя.
