import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Автоматически загружаем переменные из .env файла в окружение ОС
load_dotenv()

# Определяем абсолютный путь к корню проекта (на уровень выше папки src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Config:
    """Класс для удобного доступа к настройкам из config.yaml и .env"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_file = PROJECT_ROOT / config_path
        self._raw_config = self._load_yaml()

        # Базы данных
        db_config = self._raw_config.get("databases", {})
        self.active_db_name = db_config.get("active", "v3_unified")  # По умолчанию используем v3_unified, если не указано другое
        
        # Настройки АКТИВНОЙ базы (для обычных скриптов)
        active_db = db_config.get(self.active_db_name, {})
        self.db_path = active_db.get("path", "")
        self.collection_name = active_db.get("collection", "reglab_tech_docs")
        self.bm25_cache = active_db.get("bm25_cache", "")

        # Сохраняем ВСЕ базы в словарь (пригодится для A/B тестирования)
        self.all_databases = {k: v for k, v in db_config.items() if k != "active"}

        # Модели
        models_config = self._raw_config.get("models", {})
        self.device = models_config.get("device", "cuda")
        
        # Узнаем, какие модели сейчас выбраны как активные
        active_emb_key = models_config.get("active_embedding", "bge_m3")
        active_reranker_key = models_config.get("active_reranker", "bge_base")
        
        # Достаем полные пути (HuggingFace) из словарей
        self.embedding_model_name = models_config.get("embeddings", {}).get(active_emb_key, "BAAI/bge-m3")
        self.reranker_model_name = models_config.get("rerankers", {}).get(active_reranker_key, "BAAI/bge-reranker-base")
        
        # Сохраняем все доступные варианты (пригодится для скриптов A/B тестирования)
        self.all_embeddings = models_config.get("embeddings", {})
        self.all_rerankers = models_config.get("rerankers", {})

        # Поиск
        retrieval = self._raw_config.get("retrieval", {})
        self.top_k_retrieval = retrieval.get("top_k_retrieval", 30)
        self.top_k_final = retrieval.get("top_k_final", 3)
        self.rerank_threshold = retrieval.get("rerank_threshold", 0.05)
        self.use_litm = retrieval.get("use_litm", True)

        # Чанкинг
        chunking = self._raw_config.get("chunking", {})
        self.max_chunk_size = chunking.get("max_chunk_size", 2000)
        self.chunk_overlap = chunking.get("chunk_overlap", 200)

        # Пути
        paths = self._raw_config.get("paths", {})
        self.images_out_dir = paths.get("images_out_dir", "")
        self.hash_registry = paths.get("hash_registry", "")
        self.source_dirs = paths.get("source_dirs", {})
        self.docs_base_urls = paths.get("docs_base_urls", {})

    def _load_yaml(self) -> dict:
        if not self.config_file.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {self.config_file}")
        with open(self.config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # --- Секреты (берутся из .env) ---
    @property
    def groq_api_keys(self) -> list:
        """Возвращает список ключей, разбивая строку по запятой"""
        keys_str = os.getenv("GROQ_API_KEYS", "")
        # Разбиваем по запятой и удаляем случайные пробелы, если они есть
        return [k.strip() for k in keys_str.split(",") if k.strip()]

    @property
    def ollama_base_url(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    
    @property
    def gigachat_credentials(self) -> str:
        return os.getenv("GIGACHAT_CREDENTIALS", "")
    
    @property
    def active_llm(self) -> str:
        # ЗАМЕНИЛИ _config на _raw_config
        llm_config = self._raw_config.get("llm") or {}
        return llm_config.get("active", "groq")
    
    @property
    def google_api_key(self) -> str:
        return os.getenv("GOOGLE_API_KEY", "")

# Создаем глобальный экземпляр конфига для импорта
settings = Config()