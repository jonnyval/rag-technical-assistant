# `src/document_processing`

Папка содержит подготовку данных перед индексацией: парсинг документации, очистку тикетов, формирование метаданных и переиндексацию.

## Основные файлы

- `parsers_qdrant_with_llm.py` - парсит DOCX и HTML, извлекает таблицы, заголовки, breadcrumbs, изображения и технические метаданные.
- `parsers_tickets.py` - базовая обработка тикетов без LLM-обогащения.
- `parsers_tickets_with_llm.py` - обработка тикетов с очисткой PII и LLM-саммари проблемы/решения.
- `parsers_tickets_with_llm_old.py` - старая версия LLM-парсера тикетов, оставлена для совместимости и сравнения.
- `metadata_extractor.py` - вспомогательное извлечение ключевых слов и вопросов из чанков.
- `reindex_with_prefix.py`, `reindex_with_prefix_v2.py` - переиндексация существующих parent-документов в новые Qdrant-коллекции с подготовленными child-чанками.

## Результат работы

Парсеры возвращают `Document`-объекты LangChain. Важные метаданные: `source_file`, `page_title`, `breadcrumb_raw`, `equipment_type`, `doc_level`, `ticket_id`, `ticket_url`, `summary_problem`, `summary_solution`.

Эти поля используются поиском, API-ответами и форматированием похожих обращений, поэтому при изменении схемы нужно синхронно обновлять `src/retrieval` и `src/engine.py`.
