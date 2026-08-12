# Hermes article agent

Полное руководство по установке, моделям, запуску, результатам и диагностике:
[HERMES_RAG_GUIDE.md](HERMES_RAG_GUIDE.md).

Локальный stdio-MCP для Hermes Agent. Hermes самостоятельно планирует исследование и вызывает
поиск по обеим сторонам существующего `DualRetriever`. Кластеризация предоставляет только темы и
контрольный список тикетов.

## Установка

Hermes Agent на Windows устанавливается отдельно по официальной инструкции. MCP SDK нужен именно
в Python-окружении проекта:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe -m pip install -r ticket_clustering\hermes_agent\requirements-hermes.txt
```

На этой машине сервер зарегистрирован как `reglab_articles` в
`%LOCALAPPDATA%\hermes\config.yaml`. Для другой установки перенесите секцию из
`hermes_config.example.yaml` в активный Hermes `config.yaml`. Qdrant должен быть доступен, а пути
в основном `config.yaml` — указывать на активные коллекции и parent-store.

Для исследования передайте Hermes содержимое `ARTICLE_AGENT_PROMPT.md` и `cluster_id`, задайте
свободную тему либо попросите выбрать тему через `list_article_candidates`.

После настройки модели в Hermes запустите одну тему так:

```powershell
& ticket_clustering\hermes_agent\run_article.ps1 `
  -ClusterId "topic-7147b3350d2cc426e9f3"
```

Свободная тема без предварительной кластеризации:

```powershell
& ticket_clustering\hermes_agent\run_article.ps1 `
  -Topic "FSC reset"
```

Для неё MCP создаёт устойчивый идентификатор `manual-...` в `output/_research_seeds/`. Тема
служит только направлением поиска: факты по-прежнему должны подтверждаться документацией или
точными историческими тикетами.

Для Groq можно безопасно импортировать проектный список `GROQ_API_KEYS` в
credential pool Hermes и включить равномерную ротацию:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  ticket_clustering\hermes_agent\configure_groq.py
```

Скрипт не выводит значения ключей. Он выбирает `llama-3.3-70b-versatile`, создаёт
provider `custom:groq`, ограничивает один ответ модели 2048 токенами и включает
для ключей стратегию `round_robin`.

Для переключения Hermes на приватный Ollama через Tailscale:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  ticket_clustering\hermes_agent\configure_ollama_tailscale.py
```

Адрес приватного Ollama задаётся в локальном `config.yaml`; пример без частных
адресов лежит в `config.example.yaml`. Модель `qwen3.6:27b-64k` использует контекст 65536 и
максимум 4096 выходных токенов. Вернуться на
Yandex DeepSeek можно повторным запуском `configure_yandex.py`.

Все доступные inference-профили хранятся в `config.yaml` в секции `llm`.
Обычно достаточно изменить одну строку:

```yaml
llm:
  active: "tailscale_qwen"  # tailscale_qwen | local_ollama | yandex_deepseek
```

`run_article.ps1` применяет выбранный профиль перед каждым запуском. Профиль также можно выбрать
из командной строки и сохранить как активный:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  ticket_clustering\hermes_agent\configure_provider.py --list

& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  ticket_clustering\hermes_agent\configure_provider.py `
  --profile yandex_deepseek --set-active
```

Проверка Ollama endpoint и наличия модели:

```powershell
& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe `
  ticket_clustering\hermes_agent\configure_provider.py --check
```

Yandex API key и folder ID читаются из `.env`; секреты в `config.yaml` не сохраняются.

Исследовательский запуск допускает до 80 агентных итераций и требует продолжать
поиск до насыщения источников. Минимальная пауза после MCP-вызова задаётся
`agent_call_pacing_seconds` в `config.yaml` и сейчас равна 1 секунде.

Скрипт явно включает только `mcp-reglab_articles`: память, веб, терминал и произвольные файловые
инструменты Hermes в этой сессии недоступны. Запись возможна только через контролируемый
`save_article_draft` в папку `output/`.

Черновики сохраняются только в `ticket_clustering/hermes_agent/output/`. MCP не меняет Qdrant,
аналитическую SQLite или production RAG. Модели embeddings и reranker загружаются один раз за
время жизни MCP-процесса; параллельный запуск отдельного API-сервера создаст второй экземпляр.
