"""Управление конфигурацией RAG системы с валидацией и удобным доступом.

Этот модуль предоставляет глобальный объект `settings` для доступа ко всем параметрам
конфигурации. Конфигурация загружается один раз при импорте и кэшируется.

Использование:
    from src.config import settings
    
    # Доступ к любому параметру
    print(settings.active_llm)
    print(settings.embedding_model_name)
    print(settings.db_url)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Определяем абсолютный путь к корню проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Импортируем валидатор (но ловим ошибки при импорте Pydantic)
try:
    from src.config_validator import ConfigValidator, RAGConfig
    HAS_VALIDATOR = True
except ImportError:
    HAS_VALIDATOR = False

# Логгер
log = logging.getLogger("RegLabRAG")


class Settings:
    """Класс для удобного доступа к конфигурации RAG системы."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Загружает YAML-конфиг проекта и подготавливает удобные свойства доступа."""

        self.config_path = PROJECT_ROOT / config_path
        self._raw_config: Dict[str, Any] = {}
        self._validated_config: Optional['RAGConfig'] = None
        
        self._load_config()
        
        log.info(f"✅ Settings инициализирована (окружение: {self.environment})")
        log.info(f"   LLM: {self.active_llm} ({self.llm_model_name})")
        log.info(f"   Эмбеддинги: {self.embedding_model_name}")
        log.info(f"   БД (docs): {self.active_db_name} | БД (тикеты): {self.second_db_name}")
    
    def _load_config(self) -> None:
        """Читает config.yaml, валидирует структуру и проверяет обязательные env-переменные."""

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please create {self.config_path} based on config.yaml.new"
            )
        
        log.debug(f"Загрузка конфига из: {self.config_path}")
        
        if HAS_VALIDATOR:
            try:
                self._validated_config = ConfigValidator.load_and_validate(str(self.config_path))
                self._raw_config = self._validated_config.dict()
                
                try:
                    ConfigValidator.validate_env_variables(self._validated_config)
                except ValueError as e:
                    log.error(f"❌ Ошибка переменных окружения: {e}")
                    raise
                
                log.debug("✓ Конфиг успешно валидирован")
            except Exception as e:
                log.error(f"❌ Ошибка при валидации конфига: {e}", exc_info=True)
                raise
        else:
            log.warning("⚠️ Pydantic не установлен, используется простой парсинг YAML")
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f)
    
    # ===== ОКРУЖЕНИЕ =====
    @property
    def environment(self) -> str:
        """development | staging | production"""
        return self._raw_config.get("environment", "development")
    
    # ===== LLM КОНФИГУРАЦИЯ =====
    @property
    def active_llm(self) -> str:
        """Активный провайдер LLM (groq | gemini | gigachat | ollama)"""
        return self._raw_config.get("providers", {}).get("llm", {}).get("active", "groq")
    
    @property
    def llm_model_name(self) -> str:
        """Имя модели LLM для активного провайдера"""
        llm_config = self._raw_config.get("providers", {}).get("llm", {})
        configs = llm_config.get("configs", {})
        active = self.active_llm
        return configs.get(active, {}).get("model", "unknown")
    
    @property
    def llm_timeout(self) -> int:
        """Таймаут LLM запросов в секундах"""
        return self._raw_config.get("providers", {}).get("llm", {}).get("timeout_seconds", 30)
    
    @property
    def groq_api_keys(self) -> List[str]:
        """Список API ключей GROQ (из переменной окружения)"""
        keys_str = os.getenv("GROQ_API_KEYS", "")
        return [k.strip() for k in keys_str.split(",") if k.strip()] if keys_str else []
    
    @property
    def google_api_keys(self) -> List[str]:
        """Список API ключей Google (из переменной окружения)"""
        keys_str = os.getenv("GOOGLE_API_KEYS", "")
        return [k.strip() for k in keys_str.split(",") if k.strip()] if keys_str else []
    
    @property
    def gigachat_credentials(self) -> Optional[str]:
        """Credentials для GigaChat (из переменной окружения)"""
        return os.getenv("GIGACHAT_CREDENTIALS")
    
    @property
    def google_api_key(self) -> Optional[str]:
        """Первый ключ Google API (для backward compatibility)"""
        keys = self.google_api_keys
        return keys[0] if keys else None
    
    @property
    def ollama_url(self) -> str:
        """URL для подключения к Ollama"""
        llm_config = self._raw_config.get("providers", {}).get("llm", {})
        configs = llm_config.get("configs", {})
        ollama_config = configs.get("ollama", {})
        url = ollama_config.get("base_url", "http://localhost:11434/v1")
        return url.replace("/v1", "")
    
    # ===== МОДЕЛИ =====
    @property
    def embedding_model_name(self) -> str:
        """Имя модели эмбеддингов"""
        models = self._raw_config.get("models", {})
        embedding = models.get("embedding", {})
        active = embedding.get("active", "qwen3")
        available = embedding.get("available", {})
        return available.get(active, {}).get("name", "Qwen/Qwen3-Embedding-0.6B")
    
    @property
    def reranker_model_name(self) -> str:
        """Имя модели реранкера"""
        models = self._raw_config.get("models", {})
        reranker = models.get("reranker", {})
        active = reranker.get("active", "bge_v2_m3")
        available = reranker.get("available", {})
        return available.get(active, {}).get("name", "BAAI/bge-reranker-v2-m3")
    
    @property
    def device(self) -> str:
        """cuda или cpu"""
        return self._raw_config.get("models", {}).get("embedding", {}).get("device", "cuda")
    
    # ===== БАЗЫ ДАННЫХ =====
    @property
    def db_backends(self) -> Dict[str, Any]:
        """Все бэкенды БД из конфига — единая точка доступа для dual_retriever."""
        return self._raw_config.get("storage", {}).get("vector_db", {}).get("backends", {})

    @property
    def active_db_name(self) -> str:
        """Имя активной (docs) БД"""
        return self._raw_config.get("storage", {}).get("vector_db", {}).get("active", "qdrant_v2_docker")

    @property
    def second_db_name(self) -> str:
        """Имя второй БД (тикеты) для DualRetriever"""
        return self._raw_config.get("storage", {}).get("vector_db", {}).get("second_db", "qdrant_tickets_test")

    @property
    def db_url(self) -> Optional[str]:
        """URL к активной Qdrant (если используется удаленная БД)"""
        return self.db_backends.get(self.active_db_name, {}).get("url")
    
    @property
    def db_path(self) -> Optional[str]:
        """Путь к активной локальной БД"""
        return self.db_backends.get(self.active_db_name, {}).get("path")
    
    @property
    def collection_name(self) -> str:
        """Имя коллекции активной БД"""
        return self.db_backends.get(self.active_db_name, {}).get("collection", "tech_docs")
    
    @property
    def parent_store_path(self) -> str:
        """Путь к SQLite хранилищу родителей активной БД"""
        parent_store = self.db_backends.get(self.active_db_name, {}).get("parent_store", {})
        if isinstance(parent_store, dict):
            return parent_store.get("path", "vector_dbs/parent_docstore.db")
        return str(parent_store) if parent_store else "vector_dbs/parent_docstore.db"
    
    # ===== RETRIEVAL =====
    @property
    def top_k_retrieval(self) -> int:
        """Сколько чанков получить из каждой БД"""
        return self._raw_config.get("retrieval", {}).get("top_k_retrieval", 30)
    
    @property
    def top_k_final(self) -> int:
        """Сколько финальных документов отправить в LLM (суммарно по обеим БД)"""
        return self._raw_config.get("retrieval", {}).get("top_k_final", 5)
    
    @property
    def rerank_threshold(self) -> float:
        """Минимальный score после реранжирования"""
        return self._raw_config.get("retrieval", {}).get("rerank_threshold", 0.05)
    
    @property
    def use_reranker(self) -> bool:
        """Загружать и использовать реранкер.
        False = реранкер не загружается в память,
        результаты идут в LLM в порядке векторного скора из Qdrant."""
        return self._raw_config.get("retrieval", {}).get("use_reranker", True)

    @property
    def use_litm(self) -> bool:
        """Использовать Lost-in-the-Middle сортировку"""
        return self._raw_config.get("retrieval", {}).get("use_litm", True)
    
    @property
    def use_hyde(self) -> bool:
        """Использовать HyDE"""
        return self._raw_config.get("retrieval", {}).get("use_hyde", False)

    @property
    def multi_query_config(self) -> Dict[str, Any]:
        """Параметры RAG-Fusion / Multi-Query retrieval."""
        config = self._raw_config.get("retrieval", {}).get("multi_query", {})
        return config if isinstance(config, dict) else {}

    @property
    def multi_query_enabled(self) -> bool:
        return bool(self.multi_query_config.get("enabled", False))
    
    @property
    def diagnostic_sgr_config(self) -> Dict[str, Any]:
        config = self._raw_config.get("retrieval", {}).get("diagnostic_sgr", {})
        return config if isinstance(config, dict) else {}

    @property
    def diagnostic_sgr_enabled(self) -> bool:
        return bool(self.diagnostic_sgr_config.get("enabled", False))

    @property
    def agentic_rag_config(self) -> Dict[str, Any]:
        config = self._raw_config.get("agentic_rag", {})
        return config if isinstance(config, dict) else {}

    @property
    def agentic_rag_enabled(self) -> bool:
        return bool(self.agentic_rag_config.get("enabled", False))

    # ===== ЧАНКИНГ =====
    @property
    def child_chunk_size(self) -> int:
        """Размер чанка для поиска"""
        return self._raw_config.get("retrieval", {}).get("chunking", {}).get("child_chunk_size", 400)
    
    @property
    def child_chunk_overlap(self) -> int:
        """Перекрытие между чанками"""
        return self._raw_config.get("retrieval", {}).get("chunking", {}).get("child_chunk_overlap", 50)
    
    # ===== ПРИЛОЖЕНИЯ =====
    @property
    def judge_llm(self) -> str:
        """LLM для оценки качества ответов"""
        apps = self._raw_config.get("applications", {})
        return apps.get("evaluation", {}).get("judge_llm", "gemini")
    
    @property
    def evaluation_threshold(self) -> float:
        """Порог косинусного сходства для оценки ответов"""
        apps = self._raw_config.get("applications", {})
        return apps.get("evaluation", {}).get("threshold_cosine", 0.82)
    
    # ===== ДАННЫЕ И ПУТИ =====
    @property
    def source_directories(self) -> Dict[str, str]:
        """Словарь с путями к исходным директориям"""
        return self._raw_config.get("data", {}).get("source_directories", {})
    
    @property
    def source_dirs(self) -> Dict[str, str]:
        """Псевдоним source_directories (для совместимости с run_ingest_qdrant.py)"""
        return self.source_directories

    @property
    def caches(self) -> Dict[str, str]:
        """Словарь с путями к кэшам"""
        return self._raw_config.get("data", {}).get("caches", {})

    @property
    def hash_registry(self) -> str:
        """Путь к реестру хэшей файлов"""
        return self.caches.get("hash_registry", "data/file_hashes_qdrant.json")

    @property
    def images_out_dir(self) -> str:
        """Путь к папке для сохранения изображений из документов"""
        return self.caches.get("images_output", "data/persistent_images")

    @property
    def docs_base_urls(self) -> Dict[str, str]:
        """Словарь base URL для HTML-источников (из data.remote_sources).
        
        Ключи приводятся к формату source_type (без суффикса _url),
        чтобы совпадать с ключами source_directories.
        """
        remote = self._raw_config.get("data", {}).get("remote_sources", {})
        result = {}
        for key, url in remote.items():
            source_key = key[:-4] if key.endswith("_url") else key
            result[source_key] = url
        return result

    # ===== ТИКЕТЫ =====
    @property
    def ticket_active_llm(self) -> str:
        """Провайдер LLM для обработки тикетов"""
        apps = self._raw_config.get("applications", {})
        return apps.get("support_tickets", {}).get("llm", "ollama")

    @property
    def ticket_llm_model_name(self) -> str:
        """Модель LLM для обработки тикетов"""
        llm_config = self._raw_config.get("providers", {}).get("llm", {})
        configs = llm_config.get("configs", {})
        return configs.get(self.ticket_active_llm, {}).get("model", "llama-3.1-8b-instant")

    def ticket_provider_model_name(self, provider: str) -> str:
        """Модель конкретного LLM-провайдера для ticket enrichment."""
        llm_config = self._raw_config.get("providers", {}).get("llm", {})
        configs = llm_config.get("configs", {})
        defaults = {
            "gemini": "gemini-2.5-flash",
            "groq": "llama-3.1-8b-instant",
            "ollama": "qwen3:8b",
        }
        return configs.get(provider, {}).get("model", defaults.get(provider, "unknown"))

    @property
    def ticket_api_provider_order(self) -> List[str]:
        """Порядок облачных провайдеров для обработки тикетов."""
        apps = self._raw_config.get("applications", {})
        processing = apps.get("support_tickets", {}).get("processing", {})
        order = processing.get("api_provider_order", ["gemini", "groq"])
        if isinstance(order, str):
            order = [item.strip() for item in order.split(",") if item.strip()]
        return [str(item).lower() for item in order if str(item).strip()]

    @property
    def ticket_indexing_prefix(self) -> str:
        """Префикс для индексации тикетов"""
        apps = self._raw_config.get("applications", {})
        processing = apps.get("support_tickets", {}).get("processing", {})
        return processing.get(
            "indexing_prefix",
            "Техническое обращение: диагностика и решение проблемы."
        )

    @property
    def enable_smart_metadata(self) -> bool:
        """Использовать LLM для обогащения метаданных тикетов"""
        apps = self._raw_config.get("applications", {})
        processing = apps.get("support_tickets", {}).get("processing", {})
        return processing.get("enable_smart_metadata", False)

    # ===== АВТОТЕСТЫ =====
    @property
    def max_questions_per_file(self) -> Optional[int]:
        """Лимит вопросов на файл при автотестировании. None = без лимита."""
        return self._raw_config.get("testing", {}).get("max_questions_per_file", None)

    # ===== ИНТЕРФЕЙС =====
    @property
    def show_manual_filter(self) -> bool:
        """Показывать фильтр по оборудованию в Streamlit UI"""
        return self._raw_config.get("ui", {}).get("show_manual_filter", True)

    # ===== ОТЛАДКА =====
    @property
    def enable_profiling(self) -> bool:
        """Включить профилирование"""
        return self._raw_config.get("debug", {}).get("enable_profiling", False)
    
    @property
    def log_level(self) -> str:
        """Уровень логирования"""
        return self._raw_config.get("debug", {}).get("log_level", "INFO")
    
    @property
    def llm_max_retries(self) -> int:
        """Максимум повторов при ошибке"""
        return self._raw_config.get("providers", {}).get("llm", {}).get("max_retries", 3)


# === ГЛОБАЛЬНЫЙ ОБЪЕКТ КОНФИГУРАЦИИ ===
try:
    settings = Settings("config.yaml")
except FileNotFoundError:
    log.warning(
        "⚠️  config.yaml не найден!\n"
        "   Используется config.yaml.new как template.\n"
        "   Пожалуйста, переименуйте config.yaml.new в config.yaml и отредактируйте его."
    )
    try:
        settings = Settings("config.yaml.new")
    except Exception as e:
        log.error(f"❌ Не удалось загрузить конфиг: {e}")
        raise
