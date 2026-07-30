"""Валидация и парсинг конфигурации RAG системы с использованием Pydantic v2.

Этот модуль обеспечивает:
- Структурированную валидацию config.yaml
- Проверку наличия обязательных переменных окружения
- Валидацию путей к файлам и URL
- Проверку зависимостей между параметрами

Использование:
    from src.config_validator import ConfigValidator
    config = ConfigValidator.load_and_validate("config.yaml")
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
import yaml
import logging

log = logging.getLogger("RegLabRAG")


class LLMProviderConfig(BaseModel):
    """Конфигурация для одного провайдера LLM."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(..., description="Имя модели")
    base_url: Optional[str] = Field(None, description="Base URL (если требуется)")
    api_keys: Optional[str] = Field(None, description="API ключи (может содержать переменные окружения)")
    credentials: Optional[str] = Field(None, description="Credentials (может быть JSON или переменная)")
    verify_ssl: Optional[bool] = Field(True, description="Проверять ли SSL сертификаты")
    description: Optional[str] = None


class LLMConfig(BaseModel):
    """Конфигурация провайдера LLM."""
    active: str = Field(..., description="Активный провайдер")
    timeout_seconds: int = Field(30, description="Таймаут запросов в секундах")
    max_retries: int = Field(3, description="Максимум повторных попыток")
    
    configs: Dict[str, LLMProviderConfig] = Field(..., description="Конфигурация провайдеров")
    
    @model_validator(mode='after')
    def validate_active(self) -> 'LLMConfig':
        """Проверяет, что активный провайдер существует в configs."""
        if self.active not in self.configs:
            raise ValueError(f"Активный LLM '{self.active}' не найден в configs. Доступные: {list(self.configs.keys())}")
        return self


class VectorDBConfig(BaseModel):
    """Конфигурация для одной векторной БД."""
    type: str = Field("qdrant", description="Тип БД (qdrant, chroma, etc)")
    url: Optional[str] = Field(None, description="URL для удаленной БД")
    path: Optional[str] = Field(None, description="Путь для локальной БД")
    collection: str = Field(..., description="Имя коллекции")
    parent_store: Optional[Dict[str, Any]] = Field(None, description="Конфигурация хранилища родителей")
    description: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_connection(self) -> 'VectorDBConfig':
        """Проверяет, что указана либо url, либо path."""
        if not self.url and not self.path:
            raise ValueError("Должна быть указана либо 'url', либо 'path'")
        
        if self.path and not os.path.exists(self.path) and self.type == "qdrant":
            log.warning(f"⚠️ Путь к Qdrant не существует: {self.path}")
        
        return self


class StorageConfig(BaseModel):
    """Конфигурация хранилища."""
    vector_db: Dict[str, Any] = Field(..., description="Конфигурация векторной БД")
    
    @field_validator("vector_db")
    @classmethod
    def validate_vector_db(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Проверяет, что имеется активная БД и она определена."""
        if "active" not in v:
            raise ValueError("Обязательно должна быть указана 'active' для vector_db")
        
        if "backends" not in v:
            raise ValueError("Обязательно должна быть указана конфигурация 'backends'")
        
        active = v["active"]
        backends = v["backends"]
        
        if active not in backends:
            raise ValueError(f"Активная БД '{active}' не найдена в backends. Доступные: {list(backends.keys())}")
        
        return v


class EmbeddingConfig(BaseModel):
    """Конфигурация модели эмбеддингов."""
    active: str = Field(..., description="Активная модель")
    device: str = Field("cuda", description="cuda или cpu")
    available: Dict[str, Dict[str, Any]] = Field(..., description="Доступные модели")
    
    @model_validator(mode='after')
    def validate_active(self) -> 'EmbeddingConfig':
        """Проверяет, что активная модель существует."""
        if self.active not in self.available:
            raise ValueError(f"Активная модель '{self.active}' не найдена в available")
        return self


class RerankerConfig(BaseModel):
    """Конфигурация реранкера."""
    active: str = Field(..., description="Активный реранкер")
    available: Dict[str, Dict[str, Any]] = Field(..., description="Доступные реранкеры")
    
    @model_validator(mode='after')
    def validate_active(self) -> 'RerankerConfig':
        """Проверяет, что активный реранкер существует."""
        if self.active not in self.available:
            raise ValueError(f"Активный реранкер '{self.active}' не найден в available")
        return self


class ModelsConfig(BaseModel):
    """Конфигурация моделей (эмбеддинги, реранкеры)."""
    embedding: EmbeddingConfig = Field(..., description="Конфигурация эмбеддингов")
    reranker: RerankerConfig = Field(..., description="Конфигурация реранкера")


class RetrievalConfig(BaseModel):
    """Конфигурация retrieval."""
    top_k_retrieval: int = Field(30, ge=1, le=100, description="Кол-во чанков для поиска")
    top_k_final: int = Field(3, ge=1, le=10, description="Кол-во финальных документов")
    rerank_threshold: float = Field(0.05, ge=0.0, le=1.0, description="Порог реранжирования")
    use_reranker: bool = Field(True, description="Загружать и использовать реранкер")
    use_litm: bool = Field(True, description="Использовать LLM для итеративной трансформации")
    use_hyde: bool = Field(False, description="Использовать HyDE")
    multi_query: Dict[str, Any] = Field(default_factory=dict, description="Настройки Multi-Query / RAG-Fusion")
    diagnostic_sgr: Dict[str, Any] = Field(default_factory=dict, description="Настройки диагностического SGR")
    chunking: Dict[str, Any] = Field(..., description="Параметры чанкинга")


class ApplicationConfig(BaseModel):
    """Конфигурация для одного приложения."""
    model_config = ConfigDict(extra='allow')  # Разрешаем дополнительные поля (например, judge_llm)

    name: str
    llm: Optional[str] = Field(None, description="Ссылка на провайдер LLM")
    embedding: Optional[str] = Field(None, description="Ссылка на модель эмбеддинга")
    vector_db: Optional[str] = Field(None, description="Ссылка на векторную БД")
    processing: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class DebugConfig(BaseModel):
    """Конфигурация отладки."""
    enable_profiling: bool = Field(False, description="Включить профилирование")
    log_level: str = Field("INFO", description="Уровень логирования (DEBUG|INFO|WARNING|ERROR)")


class RAGConfig(BaseModel):
    """Основная конфигурация RAG системы."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    environment: str = Field("development", description="development | staging | production")
    providers: Dict[str, Any] = Field(..., description="Конфигурация провайдеров")
    storage: StorageConfig = Field(..., description="Конфигурация хранилища")
    models: ModelsConfig = Field(..., description="Конфигурация моделей")
    retrieval: RetrievalConfig = Field(..., description="Конфигурация retrieval")
    rag_profiles: Dict[str, Any] = Field(default_factory=dict, description="Open WebUI RAG profiles")
    agentic_rag: Dict[str, Any] = Field(default_factory=dict, description="Experimental bounded Agentic RAG")
    applications: Dict[str, ApplicationConfig] = Field(..., description="Конфигурация приложений")
    data: Dict[str, Any] = Field(..., description="Данные и пути")
    debug: DebugConfig = Field(default_factory=DebugConfig, description="Конфигурация отладки")


class ConfigValidator:
    """Валидатор конфигурации с загрузкой и парсингом YAML."""
    
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Загружает YAML конфиг с обработкой переменных окружения."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Обработка переменных окружения (${VAR_NAME})
        config = ConfigValidator._substitute_env_vars(raw_config)
        
        return config
    
    @staticmethod
    def _substitute_env_vars(obj: Any) -> Any:
        """Рекурсивно заменяет ${VAR_NAME} на значения из окружения."""
        if isinstance(obj, dict):
            return {k: ConfigValidator._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConfigValidator._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"Environment variable '{var_name}' not found in .env file")
            return value
        else:
            return obj
    
    @staticmethod
    def validate(config_dict: Dict[str, Any]) -> RAGConfig:
        """Валидирует конфиг с использованием Pydantic."""
        try:
            return RAGConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}") from e
    
    @staticmethod
    def load_and_validate(config_path: str) -> RAGConfig:
        """Загружает и валидирует конфиг за один вызов."""
        config_dict = ConfigValidator.load_yaml(config_path)
        return ConfigValidator.validate(config_dict)
    
    @staticmethod
    def validate_env_variables(config: RAGConfig) -> None:
        """Проверяет наличие необходимых переменных окружения."""
        active_llm = config.providers["llm"]["active"]
        
        required_env_vars = {
            "groq": ["GROQ_API_KEYS"],
            "gemini": ["GOOGLE_API_KEYS"],
            "gigachat": ["GIGACHAT_CREDENTIALS"],
            "ollama": [],  # Не требует API ключей
        }
        
        required = required_env_vars.get(active_llm, [])
        missing = [var for var in required if not os.getenv(var)]
        
        if missing:
            raise ValueError(
                f"Missing environment variables for {active_llm}: {', '.join(missing)}\n"
                f"Please set these variables in .env file"
            )
        
        log.info(f"✅ Все необходимые переменные окружения найдены для {active_llm}")
