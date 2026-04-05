# 🤖 RegLab RAG Technical Assistant (MVP Edition)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![ChromaDB](https://img.shields.io/badge/Chroma-Vector%20DB-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)

Интеллектуальная система поиска и ответов на вопросы по технической документации компании **"РегЛаб"** (AstraRegul, контроллеры R500 и R500S). 

Эта версия (MVP) построена на базе легковесной локальной векторной базы данных **ChromaDB** и использует классический метод нарезки документов (`RecursiveCharacterTextSplitter`).

---

## ✨ Ключевые особенности (Chroma Edition)

* 🔍 **Гибридный поиск (Hybrid Search):** Совмещение плотных векторов (Dense Embeddings, `Qwen3` / `BGE-m3`) в ChromaDB и лексического поиска (Sparse BM25 через библиотеку `rank_bm25`). Результаты объединяются алгоритмом RRF.
* 🧠 **Schema-Guided Reasoning (SGR):** Использование строгого JSON/Pydantic-формата. Модель обязана сначала выписать "интент" и "извлеченные факты", и только потом генерировать финальный ответ.
* 🎯 **Двойная фильтрация (Reranking):** Найденные документы пересортировываются с помощью локальной модели `CrossEncoder` перед подачей в LLM.
* 🔄 **Умный ингест (MD5):** Интеллектуальный хэшинг файлов пропускает уже загруженные документы при обновлении базы.
* 🌐 **Мульти-LLM:** Поддержка **Groq** (с автоматической ротацией ключей), **GigaChat** и **Gemini**. Переключается одной строкой в `config.yaml`.

---

## 🛠 Установка и настройка

### 1. Подготовка окружения
Убедитесь, что у вас установлен Python версии 3.10 или выше. Рекомендуется использовать виртуальное окружение.

```bash
# Клонирование репозитория
git clone <url_вашего_репозитория>
cd rag-technical-assistant

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Настройка переменных окружения
Создайте файл `.env` в корне проекта и добавьте туда ваши API-ключи:

```env
# API ключи Groq (можно указать несколько через запятую для обхода Rate Limits)
GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3

# API ключ Google (для Gemini)
GOOGLE_API_KEY=AIzaSy...

# Авторизационные данные GigaChat
GIGACHAT_CREDENTIALS=your_base64_credentials_here
```

### 3. Конфигурация проекта
Вся логика работы (выбор базы, LLM, параметров поиска и чанкинга) управляется через файл `config.yaml`:
* **Выбор LLM:** `llm -> active: "groq"` (или `"gigachat"`, `"gemini"`).
* **Выбор базы:** `databases -> active: "v2_unified"`.
* **Пути к файлам:** `paths -> source_dirs`.

---

## 🚀 Основные сценарии использования

### 1. Векторизация базы знаний (Ingest)
Перед тем как задавать вопросы, нужно распарсить документацию (`.docx` и `.html`) и загрузить её в векторную БД (ChromaDB).

```bash
python scripts/run_ingest.py
```
> *Система использует MD5-хэширование, поэтому при повторном запуске обработаются только новые или измененные файлы.*

### 2. Сборка лексического индекса (BM25)
Поскольку MVP-версия использует внешний BM25-индекс (в отличие от Qdrant, где он встроен), **после ингеста необходимо собрать кэш слов**:

```bash
python scripts/build_bm25.py
```
> *Этот скрипт создаст файл `bm25_cache.pkl`, который нужен для работы гибридного поиска.*

### 3. Запуск веб-интерфейса (Streamlit)
Удобный чат-интерфейс для работы с базой знаний, просмотра процесса мышления (SGR Audit) и исходных чанков.

```bash
streamlit run scripts/app.py
```

### 4. Тестирование из консоли (CLI)
Быстрая проверка работы RAG-системы без запуска веб-интерфейса:

```bash
python scripts/run_rag.py
```

---

## 📁 Структура проекта

* `data/` — Исходные документы, ChromaDB, кэш BM25 (`.pkl`), вопросы и логи.
* `result_test_hands/` — Сырые результаты работы автотестов.
* `scripts/` — Исполняемые скрипты:
  * `app.py` — UI интерфейс Streamlit.
  * `run_ingest.py` — Скрипт интеллектуальной загрузки и чанкинга в ChromaDB.
  * `build_bm25.py` — Сборка локального BM25-индекса.
  * `run_rag.py` — Консольный запуск системы для отладки.
  * `check_models.py` — Диагностика доступности API ключей.
* `src/` — Исходный код системы:
  * `config.py` — Парсер настроек (`config.yaml` + `.env`).
  * `logger.py` — Централизованное логирование (`data/logs/rag_system.log`).
  * `document_processing/` — Парсеры DOCX (mammoth) / HTML (BeautifulSoup).
  * `retrieval/` — Логика RAG (BM25_utils, HybridRetriever).
* `config.yaml` — Главный конфигурационный файл.
* `requirements.txt` — Зависимости проекта.

---

## 🛠 Полезные диагностические команды

* **Проверка API-ключей и моделей:**
  ```bash
  python scripts/check_models.py
  ```
* **"Рентген" документа (отладка парсера):**
  ```bash
  python scripts/demo_parsers.py
  ```
* **Статистика векторной базы (ChromaDB):**
  ```bash
  python scripts/inspect_db.py
  ```
