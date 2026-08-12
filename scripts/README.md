# `scripts`

Папка содержит исполняемые сценарии проекта: API, ingestion, экспорт тикетов, автотесты, benchmark-и и ручные проверки.

## API и интерфейсы

- `api_server.py` - FastAPI-сервер для приватного ответа ИИ по тикету и OpenAI-compatible endpoint.
- `app_streamlit.py` - Streamlit-интерфейс для ручной работы с RAG.

`api_server.py` публикует `reglab-ai-adaptive`, `reglab-ai-deep`,
`reglab-ai-hermes-search` и, если разрешён, `reglab-ai-agentic`. `/health` — только liveness;
готовность Qdrant/RAG нужно проверять через `/ready`.

## Ingestion и подготовка данных

- `run_ingest_qdrant.py` - основной пайплайн индексации документации и тикетов в Qdrant.
- `download_tickets_json.py`, `download_tickets_csv_to_json.py` - загрузка тикетов из портала поддержки.
- `step1_home_llm_export_patched.py` - подготовка тикетов с LLM-саммари на отдельной машине.
- `step2_work_vectorize_cache.py` - последующая векторизация подготовленного кеша.
- `preview_tickets_with_llm.py` - просмотр результата парсинга тикетов перед индексацией.
- `filter_llm_ticket_cache.py` - проверка и фильтрация LLM-кэша перед ingestion.

## Аналитика и кластеризация

- `build_ticket_analytics_db.py` - read-only объединение CSV и Qdrant по `RL-...` в SQLite-снимок;
- `cluster_ticket_analytics.py` - сравнимые category- и symptom-кластеры;
- `compare_ticket_clusters.py` - аудит и сравнение вариантов;
- `test_ticket_analytics_db.py`, `test_ticket_analytics_clustering.py` - регрессионные проверки.

## Проверка и оценка

- `run_autotest_qdrant.py` - автотест RAG по JSON-вопросам.
- `run_autotest_qdrant_engine.py` - автотест через `RAGEngine`.
- `run_autotest_qdrant_hyde.py` - экспериментальный автотест с HyDE.
- `run_evaluator.py` - оценка ответов через similarity и LLM-as-a-judge.
- `benchmark_yandex_models.py` - сравнение моделей Yandex AI Studio на RAG-контексте.
- `compare_adaptive_agentic.py` - возобновляемое слепое A/B-сравнение Adaptive и Agentic RAG с токенами, источниками и trace.
- `diagnose_retrieval_recall.py` - поэтапный аудит semantic recall: Qdrant candidates, reranker selection, финальный контекст и ответы; поддерживает Gemini-судью и локальный embedding+CrossEncoder fallback.
- `diagnose_oracle_corpus_coverage.py` - oracle@k-аудит: ищет каждый эталонный факт как идеальный запрос и разделяет потери формулировки запроса от вероятных пробелов корпуса/индекса.
- `generate_rag_debug_report.py` - локальный HTML и JSON с кандидатами, RRF, rerank, metadata и точным LLM-контекстом.
- `test_tickets_search.py` - ручная проверка поиска похожих тикетов.

## Служебные сценарии

- `analyze_new_doc.py` - сравнение нового DOCX с текущей базой знаний и генерация HTML-отчета.
- `hyde.py` - отдельная экспериментальная генерация HyDE-запросов.

Перед запуском скриптов проверь `config.yaml`, `.env`, доступность Qdrant и наличие нужных данных в `data/`.
