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
        self.active_db_name = db_config.get("active", "qdrant_v1")  # Переключено на новую БД
        
       # Настройки АКТИВНОЙ базы (для обычных скриптов)
        active_db = db_config.get(self.active_db_name, {})
        self.db_path = active_db.get("path", "")
        self.db_url = active_db.get("url", "") # <--- ДОБАВЬТЕ ЭТУ СТРОКУ
        self.parent_store_path = active_db.get("parent_store", "") 
        self.collection_name = active_db.get("collection", "tech_docs_reglab")

        # Сохраняем ВСЕ базы в словарь (пригодится для A/B тестирования)
        self.all_databases = {k: v for k, v in db_config.items() if k != "active"}

        # Модели
        models_config = self._raw_config.get("models", {})
        self.device = models_config.get("device", "cuda")
        
# -----------------------------------------------------
        # Настройки Ollama (Smart Metadata)
        # -----------------------------------------------------
        ollama_config = self._raw_config.get("ollama", {})
        self.enable_smart_metadata = ollama_config.get("enable_smart_metadata", False)
        self.ollama_metadata_model = ollama_config.get("metadata_model", "deepseek-r1:8b")
        # ollama_base_url уже есть у вас в @property, но можно читать и из yaml
        self.ollama_url = ollama_config.get("base_url", "http://localhost:11434/v1")

        # Узнаем, какие модели сейчас выбраны как активные
        active_emb_key = models_config.get("active_embedding", "qwen3")
        active_reranker_key = models_config.get("active_reranker", "bge_v2_m3")
        
        # Достаем полные пути (HuggingFace) из словарей
        self.embedding_model_name = models_config.get("embeddings", {}).get(active_emb_key, "Qwen/Qwen3-Embedding-0.6B")
        self.reranker_model_name = models_config.get("rerankers", {}).get(active_reranker_key, "BAAI/bge-reranker-v2-m3")
        
        # Сохраняем все доступные варианты (пригодится для скриптов A/B тестирования)
        self.all_embeddings = models_config.get("embeddings", {})
        self.all_rerankers = models_config.get("rerankers", {})

        # Поиск
        retrieval = self._raw_config.get("retrieval", {})
        self.top_k_retrieval = retrieval.get("top_k_retrieval", 30)
        self.top_k_final = retrieval.get("top_k_final", 3)
        self.rerank_threshold = retrieval.get("rerank_threshold", 0.05)
        self.use_litm = retrieval.get("use_litm", True)

        # -----------------------------------------------------
        # Чанкинг Parent-Child (НОВОЕ)
        # -----------------------------------------------------
        chunking_pc = self._raw_config.get("chunking_parents_child", {})
        self.child_chunk_size = chunking_pc.get("child_chunk_size", 400)
        self.child_chunk_overlap = chunking_pc.get("child_chunk_overlap", 50)

        # -----------------------------------------------------
        # Обычный чанкинг (оставлено для совместимости)
        # -----------------------------------------------------
        chunking = self._raw_config.get("chunking", {})
        self.max_chunk_size = chunking.get("max_chunk_size", 2000)
        self.chunk_overlap = chunking.get("chunk_overlap", 200)

        # Пути
        paths = self._raw_config.get("paths", {})
        self.images_out_dir = paths.get("images_out_dir", "")
        self.hash_registry = paths.get("hash_registry", "")
        self.source_dirs = paths.get("source_dirs", {})
        self.docs_base_urls = paths.get("docs_base_urls", {})

        # -----------------------------------------------------
        # Настройки интерфейса (UI)
        # -----------------------------------------------------
        ui_config = self._raw_config.get("ui", {})
        self.show_manual_filter = ui_config.get("show_manual_filter", False)

        # -----------------------------------------------------
        # Отладка и профилирование
        # -----------------------------------------------------
        debug_config = self._raw_config.get("debug", {})
        self.enable_profiling = debug_config.get("enable_profiling", False)

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
        llm_config = self._raw_config.get("llm") or {}
        return llm_config.get("active", "groq")
    
    @property
    def llm_model_name(self) -> str:
        """Возвращает название модели для активного провайдера LLM"""
        llm_config = self._raw_config.get("llm") or {}
        models = llm_config.get("models", {})
        # Возвращаем модель, которая соответствует активному провайдеру (по умолчанию qwen3:8b)
        return models.get(self.active_llm, "qwen3:8b")


    @property
    def google_api_key(self) -> str:
        return os.getenv("GOOGLE_API_KEY", "")

# Создаем глобальный экземпляр конфига для импорта
settings = Config()