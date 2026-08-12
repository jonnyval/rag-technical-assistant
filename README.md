# RegLab RAG для документации и тикетов

RAG-система для специалистов технической поддержки RegLab/Prosyst. Она ищет сведения в
официальной документации и исторических обращениях, формирует ответы со ссылками на источники,
проводит аудит retrieval и поддерживает отдельные контуры аналитики, кластеризации и подготовки
статей базы знаний.

Все чат-профили работают без памяти: API использует только последний вопрос пользователя.
История предыдущих сообщений OpenWebUI в поиск и генерацию не передаётся.

## Основные компоненты

- `src/engine.py` — основной RAG-движок, planner, adaptive retrieval и генерация ответа;
- `src/retrieval/dual_retriever.py` — раздельный поиск по документации и тикетам, RRF и rerank;
- `src/context_formatting.py` — LLM-контекст и стабильные ссылки `[D1]`/`[T1]`;
- `src/module_detection.py` — детерминированное распознавание модулей ПЛК;
- `src/evidence_guard.py` — фильтрация несовместимых серий и неподтверждённых выводов;
- `scripts/api_server.py` — API портала и OpenAI-совместимый endpoint для OpenWebUI;
- `ticket_clustering/` — аналитическая база обращений и детерминированная кластеризация симптомов;
- `ticket_clustering/hermes_agent/` — агентный поиск Hermes и генерация черновиков статей;
- `data/`, `vector_dbs/`, `reports/`, `benchmark_results/` — локальные данные и результаты,
  которые не должны публиковаться в Git.

## Production retrieval

1. Документация и тикеты индексируются в разные коллекции Qdrant.
2. Child-чанки используются для поиска, полные parent-документы берутся из SQLite parent-store.
3. Planner формирует отдельные запросы для документации и исторических тикетов.
4. Точная серия оборудования фильтруется до RRF и финального отбора. `R500` и `R500S` не
   считаются взаимозаменяемыми.
5. При нескольких запросах кандидаты объединяются через RRF. Adaptive включает CrossEncoder
   условно, если пул слабый или схлопнулся; deep использует полный rerank.
6. Финальный контекст уникализируется и получает устойчивые обозначения источников.
7. LLM формирует ответ, после чего `evidence_guard` удаляет неподтверждённые ссылки, расшифровки и
   выводы за пределами найденных данных.

HyDE не входит в обязательный production-путь. Структурированный planner включён в текущем
`config.yaml`; экспериментальный `query_aware_rerank` по умолчанию выключен.

## Профили OpenWebUI

API публикует модели через `/v1/models`:

| Профиль | Назначение |
|---|---|
| `reglab-ai-adaptive` | Основной умный поиск: факты документации и отдельно опыт похожих тикетов, без роли советчика |
| `reglab-ai-deep` | Более тяжёлый retrieval с полным reranker для максимальной полноты |
| `reglab-ai-hermes-search` | Свободный итеративный поиск Hermes: агент сам уточняет запросы и выбирает источники |
| `reglab-ai-agentic` | Ограниченный агентный цикл встроенного RAG; показывается, если `agentic_rag.enabled: true` |

`reglab-ai-hermes-search` запускает локальный Hermes в режиме one-shot. Управляющая LLM может
работать на удалённом Ollama/Qwen, в Yandex AI Studio или через другой настроенный provider.
Поисковые инструменты Hermes обращаются к `/hermes/mcp` того же API и используют уже прогретый
`RAGEngine`: embedding и reranker второй раз не загружаются. MCP доступен только с loopback.

Hermes сначала получает короткие snippets и `result_ref`, а полный текст читает через
`read_search_results` только для выбранных кандидатов. Это снижает расход контекста без потери
доступа к исходным фрагментам.

## Первый запуск

Команды ниже выполняются из корня проекта. Проектный Python на текущей машине:

```powershell
$ProjectPython = "C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe"
```

1. Проверьте локальный путь хранения Qdrant в `.env` или оставьте значение по умолчанию
   `./qdrant_storage`, затем запустите контейнер:

```powershell
docker compose up -d
Invoke-RestMethod http://localhost:6333/healthz
```

2. Проверьте в `config.yaml`:

- `storage.vector_db.active` — коллекция документации;
- `storage.vector_db.second_db` — коллекция тикетов;
- имена коллекций и пути `parent_store`;
- выбранные embedding, reranker и LLM;
- наличие соответствующих ключей только в `.env`.

3. Для API с Hermes установите MCP-зависимость:

```powershell
& $ProjectPython -m pip install -r ticket_clustering\hermes_agent\requirements-hermes.txt
```

4. Если используется Hermes, создайте локальную конфигурацию из безопасного примера и укажите
   собственный endpoint модели:

```powershell
Copy-Item ticket_clustering\hermes_agent\config.example.yaml `
  ticket_clustering\hermes_agent\config.yaml

& $ProjectPython ticket_clustering\hermes_agent\configure_provider.py --list
& $ProjectPython ticket_clustering\hermes_agent\configure_provider.py --check
```

Локальный `ticket_clustering/hermes_agent/config.yaml`, `.env`, адреса приватной сети, базы,
сессии и результаты Hermes исключены из Git.

5. Запустите API:

```powershell
& $ProjectPython scripts\api_server.py
```

Проверка состояния:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
Invoke-RestMethod http://localhost:8000/v1/models
```

`/health` подтверждает только работу HTTP-процесса. Для запросов RAG обязательно проверяйте
`/ready`: при недоступном Qdrant API может запуститься, но вернёт `503 not_ready`.

## Индексация

Основной запуск ingestion:

```powershell
& $ProjectPython scripts\run_ingest_qdrant.py
```

Перед ingestion проверьте исходные директории в `config.yaml`. Не пересоздавайте production-
коллекции и parent-store без резервной копии и явного понимания выбранного режима скрипта.
Подробности по подготовке тикетов находятся в [scripts/README.md](scripts/README.md).

## Hermes

Обычный поиск из PowerShell без OpenWebUI:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\ticket_clustering\hermes_agent\run_search.ps1 `
  -Query "Что означает FSC session reset и как эту проблему решали?"
```

Исследование свободной темы для статьи EVA:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\ticket_clustering\hermes_agent\run_article.ps1 `
  -Topic "FSC reset"
```

Статейный режим использует отдельный MCP на `127.0.0.1:8765`, сохраняет черновик и
исследовательское досье локально и не изменяет Qdrant. Полная инструкция:
[HERMES_RAG_GUIDE.md](ticket_clustering/hermes_agent/HERMES_RAG_GUIDE.md).

## Аналитическая база и кластеризация

`scripts/build_ticket_analytics_db.py` объединяет CSV-выгрузку с метаданными и симптомами Qdrant
по номеру обращения. Источники читаются без изменения; SQLite-снимок заменяется только с явным
`--replace`.

```powershell
& $ProjectPython scripts\build_ticket_analytics_db.py --dry-run
& $ProjectPython scripts\build_ticket_analytics_db.py --replace
```

Для сравнения доступны детерминированная категоризация и семантическая кластеризация симптомов:

```powershell
& $ProjectPython scripts\cluster_ticket_analytics.py --method category --replace
& $ProjectPython scripts\cluster_ticket_analytics.py --method symptom --replace
```

Изолированный графовый baseline и правила оценки пригодности тем для статей описаны в
[ticket_clustering/README.md](ticket_clustering/README.md).

## Аудит retrieval

`scripts/generate_rag_debug_report.py` сохраняет локальный HTML и соседний сырой JSON: planner,
Qdrant-кандидаты, RRF, reranker, parent-документы, исключённые источники, metadata и точный
контекст перед LLM. `--generate` дополнительно запускает synthesis и сохраняет provider usage.

```powershell
& $ProjectPython scripts\generate_rag_debug_report.py `
  --query "ошибка download denied при загрузке проекта" `
  --profiles adaptive,deep,agentic `
  --compare reranker,multi_query `
  --generate
```

## Минимальные проверки

```powershell
& $ProjectPython scripts\test_engine_guards.py
& $ProjectPython scripts\test_ticket_analytics_db.py
& $ProjectPython scripts\test_ticket_analytics_clustering.py
& $ProjectPython ticket_clustering\tests\test_baseline.py
& $ProjectPython ticket_clustering\tests\test_hermes_service.py
```

Не публикуйте `.env`, пользовательский конфиг Hermes, тикеты, логи, CSV-выгрузки, Qdrant
storage, SQLite, HTML/JSON-отчёты и сгенерированные статьи.
