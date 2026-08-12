# Агентный RAG через Hermes: руководство оператора

Это руководство описывает два контура Hermes Agent:

1. `reglab-ai-hermes-search` — stateless-поиск в OpenWebUI через общий `api_server.py`;
2. отдельный исследовательский контур, который создаёт черновики статей EVA и
   запускается PowerShell-командой.

Оба режима игнорируют историю предыдущих вопросов. Поисковый профиль не сохраняет файлы;
статейный режим может записывать только локальные seed, черновики и досье.

## 1. Что делает режим

Hermes получает тему или готовый кластер, самостоятельно формулирует несколько поисковых
запросов, обращается к документации и историческим тикетам через существующий `DualRetriever`,
проверяет важные тикеты по точным ID и сохраняет:

1. атомарную статью для EVA;
2. полное исследовательское досье;
3. JSON-манифест источников и результаты аудита ссылок.

Основная статья должна содержать одну проблему, одну подтверждённую причину или однородное
условие и одно главное решение. Полезные обходы, противоречия и смежные случаи сохраняются в
досье, поэтому найденная информация не теряется.

Контур работает только с локальными MCP-инструментами `reglab_articles`. В статье-сессии Hermes
запускается с `--ignore-rules`: память, web, terminal и история других чатов в исследование не
подмешиваются.

```text
Тема или cluster_id
        ↓
Hermes planner
        ↓
MCP reglab_articles (127.0.0.1:8765)
        ↓
DualRetriever: документация + тикеты в Qdrant
        ↓
Точная проверка тикетов в аналитической SQLite
        ↓
article.md + article_research.md + article.json
```

MCP не изменяет Qdrant, исходные документы или аналитическую SQLite. Единственная разрешённая
запись — локальные seed и черновики в `ticket_clustering/hermes_agent/output/`.

## 2. Основные файлы

| Файл | Назначение |
|---|---|
| `config.example.yaml` | Безопасный пример локального конфига |
| `config.yaml` | Локальная активная LLM, endpoints, retrieval-лимиты и пути; исключён из Git |
| `ARTICLE_AGENT_PROMPT.md` | Правила исследования, доказательности и формат EVA |
| `SEARCH_AGENT_PROMPT.md` | Постоянные правила обычного агентного поиска |
| `run_article.ps1` | Основной запуск темы или кластера |
| `run_search.ps1` | Один поисковый запрос без OpenWebUI |
| `configure_provider.py` | Просмотр, проверка и переключение LLM-профилей |
| `server.py` | MCP-инструменты, доступные Hermes |
| `api_mcp.py` | Search-only MCP, смонтированный в production API |
| `search_runner.py` | Stateless one-shot запуск Hermes для OpenWebUI |
| `service.py` | Поиск, сериализация результатов, seed и сохранение артефактов |
| `hermes_config.example.yaml` | Пример регистрации MCP в Hermes |
| `output/` | Статьи, досье, манифесты и ручные seed |

Пользовательская конфигурация установленного Hermes находится в
`%LOCALAPPDATA%\hermes\config.yaml`. Вручную переключать модель там не нужно:
`run_article.ps1` синхронизирует её из проектного `config.yaml` перед каждым запуском.

Никогда не публикуйте `%LOCALAPPDATA%\hermes\.env`, `auth.json`, проектный `.env` или полный
пользовательский конфиг Hermes: они могут содержать ключи и служебные данные.

## 3. Предварительные условия

Запускайте команды из корня проекта:

```powershell
Set-Location C:\Users\e.valov\Desktop\qdrant_rag_prod_v2_tickets
```

Проектный Python:

```text
C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe
```

Необходимо:

- доступный Qdrant на `http://localhost:6333`;
- корректные активные коллекции и parent-store в основном `config.yaml` проекта;
- установленный Hermes в `%LOCALAPPDATA%\hermes\hermes-agent`;
- MCP-зависимость в Python-окружении проекта;
- доступ к выбранной LLM;
- Tailscale, если выбран `tailscale_qwen`.

Установка MCP-зависимости:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  -m pip install -r ticket_clustering\hermes_agent\requirements-hermes.txt
```

На новой машине скопируйте `config.example.yaml` в локальный `config.yaml`, заполните endpoint
модели и запустите `configure_provider.py`. Скрипт сам зарегистрирует в Hermes наборы
`reglab_articles`, `reglab_search` и `reglab_search_local`. `hermes_config.example.yaml` нужен как
справочный пример схемы, а не как обязательный ручной шаг.

Быстрые проверки:

```powershell
Test-NetConnection localhost -Port 6333

Test-Path "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\hermes.exe"
```

Для удалённой Qwen подставьте свой Tailscale IP или hostname:

```powershell
tailscale ping <tailscale-host>
Test-NetConnection <tailscale-host> -Port 11434
```

## 4. Выбор модели

Локальный каталог находится в `ticket_clustering/hermes_agent/config.yaml`. На новой
машине скопируйте `config.example.yaml` в `config.yaml` и замените плейсхолдер адреса:

```yaml
llm:
  active: "tailscale_qwen"
  profiles:
    tailscale_qwen:
      type: "ollama"
      base_url: "http://your-tailscale-host:11434/v1"
      model: "qwen3.6:27b-64k"
      context_length: 65536
      max_tokens: 4096
    local_ollama:
      type: "ollama"
      base_url: "http://127.0.0.1:11434/v1"
      model: "qwen3.6:27b-64k"
      context_length: 65536
      max_tokens: 4096
    yandex_deepseek:
      type: "yandex"
      base_url: "https://ai.api.cloud.yandex.net/v1"
      model: "deepseek-v4-flash"
```

Доступные профили:

- `tailscale_qwen` — Qwen на сервере в приватной сети Tailscale;
- `local_ollama` — Ollama на текущем компьютере;
- `yandex_deepseek` — DeepSeek V4 Flash через Yandex AI Studio.

Профиль `local_ollama` предполагает, что указанный model tag уже установлен локально. Если на
текущем компьютере используется другая модель, измените поле `model` этого профиля и проверьте
его через `--profile local_ollama --check`.

Yandex использует `YANDEX_API_KEY` и `YANDEX_FOLDER_ID` из проектного `.env`. Ключи в YAML не
записываются.

### Посмотреть профили

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  ticket_clustering\hermes_agent\configure_provider.py --list
```

Звёздочкой отмечен активный профиль.

### Переключить модель через YAML

Измените только поле:

```yaml
llm:
  active: "yandex_deepseek"
```

Следующий `run_article.ps1` автоматически применит профиль.

### Переключить модель командой

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  ticket_clustering\hermes_agent\configure_provider.py `
  --profile tailscale_qwen --set-active
```

Другие значения: `local_ollama`, `yandex_deepseek`.

Без `--set-active` профиль применяется к Hermes только сейчас, но поле `llm.active` не меняется.
Следующий `run_article.ps1` снова применит профиль из YAML.

### Проверить Ollama

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  ticket_clustering\hermes_agent\configure_provider.py --check
```

Для Ollama проверяется доступность `/v1/models` и наличие указанного model tag. Для
аутентифицированного Yandex-профиля полноценная проверка выполняется первым inference-вызовом.

Прямой просмотр удалённых Ollama-моделей:

```powershell
Invoke-RestMethod http://<tailscale-host>:11434/v1/models
```

## 5. Генерация статьи по свободной теме

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\ticket_clustering\hermes_agent\run_article.ps1 `
  -Topic "FSC reset"
```

Другой пример:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\ticket_clustering\hermes_agent\run_article.ps1 `
  -Topic "Port FSC not specified после обновления СПО"
```

Для темы создаётся детерминированный seed вида `manual-...` в
`output/_research_seeds/`. Одинаковая нормализованная тема повторно использует тот же seed.
Свободная формулировка задаёт направление, но не считается доказательством.

Хорошая тема содержит предмет и конкретный вопрос или симптом:

- `FSC reset SCU event 38/39`;
- `Download denied при загрузке проекта в R500`;
- `Port FSC not specified после обновления СПО`.

Слишком широкие темы вроде `всё про OPC UA` повышают вероятность неоднородной статьи. В таком
случае лучше сделать несколько узких исследований.

## 6. Генерация по кластеру

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\ticket_clustering\hermes_agent\run_article.ps1 `
  -ClusterId "topic-f57b6834852ca718a67e"
```

Кластер нужен только как seed и контрольный набор тикетов. Агент всё равно выполняет независимый
поиск и не считает название кластера доказательством.

Кандидаты берутся из отчёта, указанного в `candidate_report` внутри Hermes `config.yaml`.

### 6.1. Обычный информационный поиск через Hermes

Для ответа на произвольный вопрос без генерации и сохранения статьи:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\ticket_clustering\hermes_agent\run_search.ps1 `
  -Query "Что означает сообщение FSC session reset и как его устраняли?"
```

Этот режим:

- не использует историю предыдущих вопросов;
- видит только поисковые инструменты `reglab_search`, без сохранения статьи;
- сначала получает короткие snippets, а полный текст читает только у выбранных результатов;
- сохраняет JSON со статистикой токенов в `ticket_clustering/hermes_agent/usage/`.

Правила ответа находятся в `SEARCH_AGENT_PROMPT.md`. Это основа для отдельной модели
`reglab-ai-hermes-search` в OpenWebUI.

Профиль уже встроен в общий OpenAI-совместимый API. Перед первым запуском или после смены
модели Hermes примените конфигурацию:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  ticket_clustering\hermes_agent\configure_provider.py
```

Затем запустите обычный API:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  scripts\api_server.py
```

В `/v1/models` и OpenWebUI появится `reglab-ai-hermes-search`. Каждый вопрос обрабатывается
независимо: предыдущая переписка игнорируется. Hermes запускает агентный one-shot, а его MCP
поиск смонтирован внутри API по `/hermes/mcp` и использует тот же прогретый `RAGEngine`, поэтому
embedding и reranker второй раз не загружаются. MCP-маршрут доступен только с loopback.

Если API работает не на порту 8000, измените `mcp.api_search_url` в Hermes `config.yaml` и снова
запустите `configure_provider.py`. `HERMES_SEARCH_TIMEOUT_SECONDS` задаёт предельное время одного
агентного ответа (по умолчанию 900 секунд).

## 7. Управление числом итераций

По умолчанию разрешено до 80 агентных ходов:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\ticket_clustering\hermes_agent\run_article.ps1 `
  -Topic "FSC reset" `
  -MaxTurns 100
```

`MaxTurns` — страховочный предел, а не требуемое количество вызовов. Агент должен остановиться
после насыщения источников. Уменьшение лимита может помешать сохранить статью, если модель долго
рассуждает или повторяет неудачный tool call.

## 8. Что происходит при запуске

`run_article.ps1` последовательно:

1. читает `llm.active` и применяет профиль к Hermes;
2. проверяет порт `127.0.0.1:8765`;
3. при необходимости запускает MCP скрытым процессом;
4. MCP загружает `RAGEngine`, embedding-модель и reranker;
5. Hermes получает `ARTICLE_AGENT_PROMPT.md` и тему/кластер;
6. агент выполняет несколько разных поисков;
7. поиск возвращает короткие snippets; полный текст читается только по выбранным `result_ref`;
8. важные тикеты перечитываются по точным ID;
9. результат сохраняется через `save_article_draft`.

Первый запуск может несколько минут загружать embedding и reranker. Следующие исследования
быстрее, пока MCP остаётся запущенным. Если одновременно запустить production API, он создаст
второй экземпляр моделей и увеличит расход RAM/VRAM.

## 9. Результаты

Каталог:

```text
ticket_clustering/hermes_agent/output/
```

Для одного исследования создаются:

```text
manual-..._Название.md
manual-..._Название_research.md
manual-..._Название.json
```

или для кластера:

```text
topic-..._Название.md
topic-..._Название_research.md
topic-..._Название.json
```

- основной `.md` — компактная статья EVA;
- `_research.md` — гипотезы, обходы, противоречия, смежные случаи и будущие темы;
- `.json` — основной и полный evidence-наборы, связанные темы, пути и аудит ссылок.

Ключевые поля JSON:

- `origin`: `cluster` или `manual_topic`;
- `research_topic`: исходная свободная тема;
- `article_evidence`: источники основной статьи;
- `research_evidence`: полный полезный набор;
- `citation_audit`: совпадение RL-ссылок статьи с manifest;
- `research_citation_audit`: аналогичная проверка досье;
- `status: draft_not_published`: материал не опубликован.

Чистый аудит выглядит так:

```json
"citation_audit": {
  "cited_not_declared": [],
  "declared_not_cited": []
}
```

Если по одной теме запускалось несколько версий, ориентируйтесь на самый новый комплект по
`LastWriteTime` и проверяйте заголовок в JSON.

## 10. Логи и наблюдение

Логи MCP:

```text
ticket_clustering/hermes_agent/mcp_http_stdout.log
ticket_clustering/hermes_agent/mcp_http_stderr.log
```

Следить за MCP-логом:

```powershell
Get-Content `
  .\ticket_clustering\hermes_agent\mcp_http_stderr.log `
  -Encoding UTF8 -Tail 100 -Wait
```

Проверить listener:

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen
```

Последние результаты:

```powershell
Get-ChildItem .\ticket_clustering\hermes_agent\output -File |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 Name, Length, LastWriteTime
```

Hermes выводит `Session: <id>` после завершения. Сессии хранятся в локальном state store Hermes.
Посмотреть их:

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\hermes.exe" `
  sessions list --limit 20
```

Экспортировать одну сессию для аудита, обязательно с редактированием секретов:

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\hermes.exe" `
  sessions export `
  .\ticket_clustering\hermes_agent\session_debug\exported_session `
  --format md `
  --session-id SESSION_ID `
  --redact `
  --force
```

## 11. Возобновление исследования

Возобновление полезно, когда статья сохранилась, но аудит выявил конкретный недостаток. Не
запускайте retrieval повторно без необходимости: укажите агенту, что нужно использовать уже
собранные источники.

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\hermes.exe" `
  --resume SESSION_ID `
  --toolsets "reglab_articles" `
  --ignore-rules `
  chat --max-turns 30 `
  -q "Не повторяй retrieval. Исправь границы применимости и пересохрани статью через save_article_draft."
```

Если для исправления нужны новые доказательства, прямо задайте проверяемую гипотезу и попросите
сделать дополнительный поиск, а не просто «улучшить статью».

## 12. Остановка и перезапуск MCP

MCP обычно оставляют запущенным между статьями, чтобы не загружать embedding и reranker заново.
Перед запуском production API или после окончания работы его можно остановить безопасно:

```powershell
$listener = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
  $articlePid = ($listener | Select-Object -First 1).OwningProcess
  $process = Get-CimInstance Win32_Process -Filter "ProcessId=$articlePid"
  if ($process.CommandLine -like "*ticket_clustering.hermes_agent.server*") {
    Stop-Process -Id $articlePid
  } else {
    Write-Error "Порт 8765 занят другим процессом: $($process.CommandLine)"
  }
}
```

При следующем `run_article.ps1` сервер запустится автоматически.

Перезапускайте MCP после изменения `server.py`, `service.py` или схемы инструментов. Уже
работающий Python-процесс не подхватит новый код автоматически.

## 13. Настройки retrieval и объёма контекста

В `ticket_clustering/hermes_agent/config.yaml`:

```yaml
rag_profile: "deep"
agent_call_pacing_seconds: 1
search:
  docs_limit: 5
  tickets_limit: 7
  tickets_child_k: 80
  search_snippet_chars: 700
  detail_content_chars: 5000
  max_detail_results: 6
  max_exact_tickets: 12
```

- `rag_profile` — профиль retrieval, не LLM-профиль Hermes;
- `agent_call_pacing_seconds` — минимальная пауза после MCP-вызова;
- `docs_limit` / `tickets_limit` — число результатов в одном поиске;
- `tickets_child_k` — глубина первичного поиска тикетов до финального отбора;
- `search_snippet_chars` — длина краткого фрагмента в поисковой выдаче;
- `detail_content_chars` — предел полного текста при адресном чтении;
- `max_detail_results` — максимум раскрываемых `result_ref` за один вызов;
- `max_exact_tickets` — сколько тикетов можно перечитать за один точный вызов.

Уменьшение snippets и limits экономит токены. Полный текст важных результатов остаётся доступен
через `read_search_results`, поэтому не увеличивайте snippets вместо адресного чтения.

Правила самой статьи находятся в `ARTICLE_AGENT_PROMPT.md`. Не дублируйте одни и те же правила
в prompt и длинных docstring инструментов.

Общий фиксированный prompt Hermes можно оценить офлайн:

```powershell
& "$env:LOCALAPPDATA\hermes\hermes-agent\venv\Scripts\hermes.exe" `
  prompt-size --platform cli --json
```

Этот отчёт показывает обычный CLI. Реальный article-run меньше по набору инструментов, потому что
запускается с `--toolsets reglab_articles --ignore-rules`. Основной рост токенов в длинной сессии
обычно дают накопленные результаты retrieval.

## 14. Типовые проблемы

### PowerShell запрещает запуск `.ps1`

Используйте команду из руководства с:

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ...
```

### `Check: FAILED` или модель не найдена

1. Проверьте `llm.active`.
2. Выполните `configure_provider.py --list`.
3. Для Tailscale проверьте `tailscale ping` и порт 11434.
4. Проверьте точное имя модели через `/v1/models`.

### Qwen возвращает пустой ответ или не вызывает инструмент

Qwen может тратить выходной лимит на внутреннее thinking. Не уменьшайте `max_tokens` ниже 4096
для исследовательского режима без отдельного теста. Tool calling модели `qwen3.6:27b-64k`
проверен через Ollama OpenAI-compatible endpoint.

### MCP не стартует за 180 секунд

Проверьте:

- `mcp_http_stderr.log`;
- Qdrant на 6333;
- пути к коллекциям и parent-store в основном проектном `config.yaml`;
- наличие памяти для embedding и reranker.

### После изменения кода действует старая схема инструмента

На 8765 остался старый MCP-процесс. Безопасно остановите его командой из раздела 12 и запустите
статью снова.

### Ответ модели обрезан по `finish_reason=length`

Если tool call сохранения успел выполниться, сначала проверьте файлы и JSON-аудит. Если статья или
досье отсутствуют, возобновите сессию и попросите пересохранить результат без повторного поиска.

### OOM на GPU

MCP загружает embedding и reranker. Не держите одновременно MCP и production API, если видеопамяти
недостаточно. При необходимости перенесите reranker на CPU через основной проектный конфиг.

### Нет прямой связи Tailscale, используется DERP

Модель продолжит работать, но latency станет выше. Проверьте вывод `tailscale ping`; строка
`via DERP(...)` означает ретрансляцию.

## 15. Рекомендуемый рабочий цикл

1. Выберите LLM и проверьте endpoint.
2. Сформулируйте одну узкую тему.
3. Запустите `run_article.ps1 -Topic ...`.
4. Откройте основной `.md` и `_research.md`.
5. Проверьте оба citation audit в JSON.
6. Сверьте спорные команды, версии и границы применимости с источниками.
7. Если требуется локальная правка исследования, возобновите ту же сессию.
8. Публикуйте материал только после экспертной проверки специалистом ТП.

Hermes создаёт `draft_not_published`: автоматической публикации в EVA в этом контуре нет.
