import time
import warnings
from threading import Lock
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

warnings.filterwarnings("ignore")

from pydantic import AliasChoices, BaseModel, Field, field_validator
from sentence_transformers import CrossEncoder

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.config import settings
from src.retrieval.dual_retriever import build_dual_retriever
from src.logger import log, log_query_audit

FUNCTION_CALLING_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
}

SUPPORT_DOCS_MAX_CHARS = 5200
SUPPORT_TICKETS_MAX_CHARS = 2600
SUPPORT_DOC_MAX_CHARS = 1800
SUPPORT_TICKET_MAX_CHARS = 1200

RAG_PROFILES: Dict[str, Dict[str, Any]] = {
    "fast": {
        "use_reranker": False,
        "top_k_retrieval": 16,
        "top_k_final": 3,
        "rerank_threshold": None,
        "use_litm": False,
    },
    "deep": {
        "use_reranker": True,
        "top_k_retrieval": 24,
        "top_k_final": 4,
        "rerank_threshold": 0.05,
        "use_litm": True,
    },
}

for _profile_name, _profile_overrides in getattr(settings, "_raw_config", {}).get("rag_profiles", {}).items():
    if isinstance(_profile_overrides, dict):
        base_profile = RAG_PROFILES.get(_profile_name, RAG_PROFILES["deep"])
        RAG_PROFILES[_profile_name] = {**base_profile, **_profile_overrides}

# ==========================================
# 🧩 SGR СХЕМЫ (Schema-Guided Reasoning)
# ==========================================

class FactExtraction(BaseModel):
    """Один извлечённый факт из контекста."""
    source_file: str = Field(description="Название файла-источника из предоставленного контекста")
    fact: str = Field(description="Конкретный технический факт или шаг, полезный для ответа")


class RAGReasoningSchema(BaseModel):
    """Структурированный ответ RAG системы."""
    user_intent: str = Field(description="Кратко переформулируйте, что именно хочет узнать пользователь.")
    extracted_facts: List[FactExtraction] = Field(description="Массив полезных фактов. Пусто, если ничего не найдено.")
    missing_context: str = Field(description="Чего не хватает в контексте для полного ответа.")
    final_answer: str = Field(description="ОБЯЗАТЕЛЬНОЕ ПОЛЕ. Итоговый ответ. Формат Markdown.")
    relevant_images: List[str] = Field(
        default=[],
        description="Массив путей к изображениям (извлекай пути из разметки ![alt](путь) в контексте)."
    )


class QuizAnswerSchema(BaseModel):
    """Структурированный ответ на тестовый вопрос с вариантами ответа."""

    user_intent: str = Field(default="", description="Что проверяет тестовый вопрос")
    extracted_facts: List[FactExtraction] = Field(default_factory=list, description="Факты из документации, на которых основан выбор")
    missing_context: str = Field(default="Всего хватает", description="Чего не хватает для уверенного выбора")
    final_answer: str = Field(description="Точный выбранный вариант ответа и короткое обоснование")
    relevant_images: List[str] = Field(default_factory=list, description="Пути к релевантным изображениям, если они есть")


    @field_validator("extracted_facts", mode="before")
    @classmethod
    def normalize_extracted_facts(cls, value):
        """Принимает факты как строки или объекты, чтобы не ронять автотест."""
        if value is None:
            return []
        if isinstance(value, str):
            return [{"source_file": "", "fact": value}]
        if isinstance(value, list):
            normalized = []
            for item in value:
                if isinstance(item, str):
                    normalized.append({"source_file": "", "fact": item})
                elif isinstance(item, dict):
                    normalized.append({
                        "source_file": item.get("source_file") or item.get("source") or "",
                        "fact": item.get("fact") or item.get("text") or str(item),
                    })
            return normalized
        return value

    @field_validator("relevant_images", mode="before")
    @classmethod
    def normalize_relevant_images(cls, value):
        """Приводит одиночный путь к списку путей."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        return value

    @field_validator("user_intent", "missing_context", "final_answer", mode="before")
    @classmethod
    def normalize_text_fields(cls, value):
        """Заменяет null и другие нестроковые значения на безопасную строку."""
        if value is None:
            return ""
        return str(value)


class SimilarTicketSummary(BaseModel):
    """Краткое описание похожего обращения из базы тикетов."""
    ticket_id: str = Field(default="", validation_alias=AliasChoices("ticket_id", "id"), description="Номер обращения, например RL-12345")
    source_file: str = Field(default="", description="Файл или источник обращения")
    problem_summary: str = Field(default="", validation_alias=AliasChoices("problem_summary", "problem"), description="Краткое саммари проблемы")
    solution_summary: str = Field(default="", validation_alias=AliasChoices("solution_summary", "solution"), description="Краткое саммари решения или действий поддержки")
    relevance_reason: str = Field(default="", validation_alias=AliasChoices("relevance_reason", "similarity_reason"), description="Почему обращение похоже на текущую заявку")
    ticket_url: str = Field(default="", description="URL обращения в портале, если он есть в metadata")


class SourceReference(BaseModel):
    """Ссылка на источник, найденный ретривером."""

    source_id: str = ""
    title: str
    source_file: str
    url: str = ""
    source_type: str = ""
    page_title: str = ""
    release_version: str = ""
    equipment_type: str = ""
    library_name: str = ""
    breadcrumb: str = ""


class EvidenceNote(BaseModel):
    """Проверяемый тезис ответа и источники, которыми он подтверждается."""

    claim: str = Field(default="", validation_alias=AliasChoices("claim", "thesis"), description="Короткий технический тезис или вывод")
    source_ids: List[str] = Field(default_factory=list, description="Метки источников, например D1, D2 или T1")
    comment: str = Field(default="", description="Оговорка, если источник подтверждает тезис не напрямую")


class SupportPrivateResponse(BaseModel):
    """Приватная подсказка ИИ для сотрудника техподдержки."""
    user_intent: str = Field(default="", description="Что клиент, вероятно, хочет решить")
    docs_answer: str = Field(description="Информация из документации. Если прямого решения нет, перечисли релевантные темы.")
    related_topics: List[str] = Field(default=[], description="Темы из документации, которые относятся к обращению")
    similar_tickets: List[SimilarTicketSummary] = Field(default=[], description="Похожие обращения из базы тикетов")
    evidence_notes: List[EvidenceNote] = Field(default_factory=list, description="Ключевые тезисы ответа с привязкой к источникам")
    recommended_questions: List[str] = Field(default_factory=list, description="Что стоит уточнить у клиента или запросить у него")
    internal_notes: List[str] = Field(default_factory=list, description="Внутренние заметки и проверки для инженера техподдержки")
    missing_context: str = Field(default="Не указано", description="Каких данных не хватает для уверенного ответа")
    draft_private_comment: str = Field(description="Готовый приватный комментарий для сотрудника ТП в Markdown")
    confidence: str = Field(default="medium", description="low | medium | high")
    doc_sources: List[SourceReference] = Field(default_factory=list, description="Документные источники, добавленные кодом после поиска")
    ticket_sources: List[SourceReference] = Field(default_factory=list, description="Источники похожих обращений, добавленные кодом после поиска")


# ==========================================
# 🛠 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

    @field_validator("related_topics", "recommended_questions", "internal_notes", mode="before")
    @classmethod
    def normalize_string_lists(cls, value):
        """Allow LLMs to return list fields either as a list or as a single string."""
        if value is None:
            return []
        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []
        if isinstance(value, list):
            return [str(item) for item in value if item is not None and str(item).strip()]
        return [str(value)]


class KeyRotationCallbackHandler(BaseCallbackHandler):
    """Логирует ошибки LLM при ротации API ключей."""
    
    def __init__(self, key_index: int = 0):
        """Запоминает индекс API-ключа, на котором выполняется текущий LLM-вызов."""

        self.key_index = key_index

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        log.warning(f"⚠️ [РОТАЦИЯ] Ошибка на ключе #{self.key_index}: {error}")


class RoundRobinFallbackRunnable(Runnable):
    """Запускает LLM-цепочки по кругу и пробует остальные ключи при ошибке."""

    def __init__(self, runnables: List[Runnable], label: str = "LLM"):
        """Сохраняет список цепочек и индекс следующего стартового ключа."""
        if not runnables:
            raise ValueError("RoundRobinFallbackRunnable requires at least one runnable")
        self.runnables = runnables
        self.label = label
        self._lock = Lock()
        self._next_index = 0

    def _ordered_runnables(self) -> List[tuple[int, Runnable]]:
        """Возвращает порядок попыток: следующий ключ первым, остальные следом."""
        with self._lock:
            start_index = self._next_index
            self._next_index = (self._next_index + 1) % len(self.runnables)

        return [
            (
                (start_index + offset) % len(self.runnables),
                self.runnables[(start_index + offset) % len(self.runnables)],
            )
            for offset in range(len(self.runnables))
        ]

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        """Выполняет запрос через очередной ключ и переключается дальше при сбое."""
        last_error: Optional[Exception] = None
        for key_index, runnable in self._ordered_runnables():
            try:
                log.debug("%s: попытка через ключ #%s", self.label, key_index)
                return runnable.invoke(input, config=config, **kwargs)
            except Exception as error:
                last_error = error
                log.warning("%s: ключ #%s не сработал, пробую следующий: %s", self.label, key_index, error)

        raise last_error or RuntimeError(f"{self.label}: no runnable succeeded")


def trim_text(text: str, max_chars: int) -> str:
    """Обрезает длинный контекст по границе строки, чтобы не переполнять лимит LLM."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit("\n", 1)[0].strip()
    if not cut:
        cut = text[:max_chars].strip()
    return f"{cut}\n\n[...контекст обрезан из-за лимита размера запроса...]"


def format_docs(docs: List, *, max_total_chars: int = 12000, max_doc_chars: int = 3000) -> str:
    """Форматирует документы для передачи в LLM.

    Добавляет заголовок с типом источника: для тикетов — [ТИКЕТ ПОДДЕРЖКИ],
    для документации — [equipment | файл | раздел].
    """
    if not docs:
        log.warning("format_docs: Получен пустой список документов")
        return ""

    formatted = []
    for i, d in enumerate(docs):
        try:
            meta      = d.metadata if hasattr(d, 'metadata') else {}
            content   = d.page_content if hasattr(d, 'page_content') else str(d)
            content = trim_text(content, max_doc_chars)
            db_source = meta.get('db_source', 'docs')

            if db_source == 'tickets':
                header = f"[ТИКЕТ ПОДДЕРЖКИ | Файл: {meta.get('source_file', 'Unknown')}]"
            else:
                source_url = resolve_source_url(meta)
                header = (
                    f"[{meta.get('equipment_type', 'Unknown')} "
                    f"| Library: {meta.get('library_name', 'Unknown')} "
                    f"| Release: {meta.get('release_version', 'Unknown')} "
                    f"| Page: {meta.get('page_title', 'Unknown')} "
                    f"| Файл: {meta.get('source_file', 'Unknown')} "
                    f"| Раздел: {meta.get('breadcrumb_raw', 'No section')} "
                    f"| URL: {source_url}]"
                )

            formatted.append(f"{header}\n{content}")
        except Exception as e:
            log.error(f"Ошибка при форматировании документа {i}: {e}", exc_info=True)
            if hasattr(d, 'page_content'):
                formatted.append(trim_text(d.page_content, max_doc_chars))

    return trim_text("\n\n".join(formatted), max_total_chars)


def format_ticket_docs(docs: List, *, max_total_chars: int = 6000, max_doc_chars: int = 2000) -> str:
    """Formats support tickets with stable IDs for the LLM."""
    if not docs:
        return ""

    formatted = []
    for d in docs:
        meta = d.metadata if hasattr(d, "metadata") else {}
        if not is_ticket_document(d):
            continue
        ticket_id = meta.get("ticket_id") or meta.get("source_file", "Unknown")
        source_file = meta.get("source_file", "Unknown")
        ticket_url = meta.get("ticket_url", "")
        content = d.page_content if hasattr(d, "page_content") else str(d)
        content = trim_text(content, max_doc_chars)
        header = f"[TICKET | id: {ticket_id} | file: {source_file} | url: {ticket_url}]"
        formatted.append(f"{header}\n{content}")

    return trim_text("\n\n".join(formatted), max_total_chars)


def is_ticket_document(doc) -> bool:
    """Проверяет, что найденный документ действительно является обращением, а не документацией."""
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    return bool(meta.get("ticket_id")) or meta.get("format") == "ticket" or meta.get("source_type") == "support_tickets"


def resolve_source_url(meta: dict, *, ticket_only: bool = False) -> str:
    """Строит URL источника из metadata или из корня источника и имени файла."""
    if ticket_only:
        return str(meta.get("ticket_url") or "")

    direct_url = meta.get("source_url") or meta.get("url")
    if direct_url:
        return str(direct_url)

    source_type = str(meta.get("source_type") or "")
    source_file = str(meta.get("source_file") or "")
    base_url = (
        meta.get("source_root")
        or meta.get("source_base_url")
        or meta.get("base_url")
        or meta.get("root_url")
        or settings.docs_base_urls.get(source_type, "")
    )
    if base_url and source_file:
        return urljoin(str(base_url).rstrip("/") + "/", source_file)
    return ""


def source_references(docs: List, *, ticket_only: bool = False, prefix: str = "S") -> List[SourceReference]:
    """Собирает стабильный список источников из metadata найденных документов."""
    result: List[SourceReference] = []
    seen = set()
    for doc in docs:
        if ticket_only and not is_ticket_document(doc):
            continue
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        url = resolve_source_url(meta, ticket_only=ticket_only)
        source_file = meta.get("source_file") or meta.get("ticket_id") or "Unknown"
        title = meta.get("page_title") or meta.get("ticket_id") or meta.get("breadcrumb_raw") or source_file
        key = (source_file, url)
        if key in seen:
            continue
        seen.add(key)
        result.append(SourceReference(
            source_id=f"{prefix}{len(result) + 1}",
            title=str(title),
            source_file=str(source_file),
            url=str(url),
            source_type=str(meta.get("source_type", "")),
            page_title=str(meta.get("page_title", "")),
            release_version=str(meta.get("release_version", "")),
            equipment_type=str(meta.get("equipment_type", "")),
            library_name=str(meta.get("library_name", "")),
            breadcrumb=str(meta.get("breadcrumb_raw", "")),
        ))
    return result


def format_source_references(sources: List[SourceReference]) -> str:
    """Форматирует источники для промпта так, чтобы LLM ссылалась на реальные документы."""
    if not sources:
        return "Источники не найдены."
    lines = []
    for source in sources:
        location = source.url or source.source_file
        details = []
        if source.equipment_type:
            details.append(f"equipment={source.equipment_type}")
        if source.library_name:
            details.append(f"library={source.library_name}")
        if source.release_version:
            details.append(f"release={source.release_version}")
        if source.breadcrumb:
            details.append(f"section={source.breadcrumb}")
        detail_suffix = " | " + " | ".join(details) if details else ""
        lines.append(
            f"[{source.source_id}] {source.title} | file={source.source_file} | url={location}{detail_suffix}"
        )
    return "\n".join(lines)


# ==========================================
# 🚀 ЯДРО RAG-СИСТЕМЫ
# ==========================================

class RAGEngine:
    """Оркестратор RAG системы: поиск по двум БД + LLM + SGR цепочка.

    Attributes:
        retriever:   DualRetriever — параллельный поиск по документации и тикетам
        sgr_chain:   LangChain цепочка для генерации структурированного ответа
        _last_docs:  Документы из последнего вызова (для отображения в Sources без
                     повторного поиска)
    """

    def __init__(self, profile: str = "deep"):
        """Собирает retriever, LLM-цепочки и служебное состояние RAG-движка."""

        try:
            log.info("🚀 Инициализация RAGEngine...")
            self.profile = profile if profile in RAG_PROFILES else "deep"
            self.profile_config = RAG_PROFILES[self.profile]
            log.info(
                "RAGEngine initialization: profile=%s top_k=%s/%s reranker=%s",
                self.profile,
                self.profile_config["top_k_retrieval"],
                self.profile_config["top_k_final"],
                self.profile_config["use_reranker"],
            )
            self.retriever, self.sgr_chain, self.support_chain = self._build_pipeline()
            self._last_docs: List = []
            log.info("✅ RAGEngine успешно инициализирован.")
        except Exception as e:
            log.error(f"❌ Критическая ошибка инициализации RAGEngine: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize RAGEngine: {str(e)}") from e

    def _build_pipeline(self):
        """Строит полный RAG pipeline.

        Returns:
            Tuple[DualRetriever, Runnable]: ретривер и SGR цепочка
        """
        try:
            # === 1. ЭМБЕДДИНГИ ===
            log.info(f"📥 Загрузка эмбеддингов: {settings.embedding_model_name}...")
            dense_embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={'device': settings.device},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'prompt': 'Instruct: Retrieve relevant technical documentation passage to answer the query.\nQuery: ',
                }
            )
            log.debug("✓ Dense embeddings загружены")

            # === 2. РЕРАНКЕР (опционально) ===
            rerank_model = None
            if self.profile_config["use_reranker"]:
                log.info(f"📊 Загрузка реранкера: {settings.reranker_model_name}...")
                rerank_model = CrossEncoder(
                    settings.reranker_model_name,
                    device=settings.device,
                    trust_remote_code=True
                )
                log.debug("✓ Реранкер загружен")
            else:
                log.info("⏭️  Реранкер отключён (use_reranker: false) — используется порядок векторного скора")

            # === 3. DUAL RETRIEVER (документация + тикеты) ===
            log.info("🔀 Построение DualRetriever...")
            retriever = build_dual_retriever(
                dense_embeddings,
                rerank_model,
                top_k_retrieval=self.profile_config["top_k_retrieval"],
                top_k_final=self.profile_config["top_k_final"],
                rerank_threshold=self.profile_config["rerank_threshold"],
                use_litm=self.profile_config["use_litm"],
            )
            log.debug("✓ DualRetriever готов")

            # === 4. LLM С РОТАЦИЕЙ КЛЮЧЕЙ ===
            log.info(f"🤖 Инициализация LLM: {settings.active_llm.upper()} ({settings.llm_model_name})...")

            llm_candidates = []
            if settings.active_llm == "gemini":
                log.debug(f"  Ключей Gemini: {len(settings.google_api_keys)}")
                
                if not settings.google_api_keys:
                    raise ValueError("❌ GOOGLE_API_KEYS не установлена в .env!")
                
                llms = [
                    ChatGoogleGenerativeAI(
                        model=settings.llm_model_name,
                        google_api_key=k,
                        temperature=0.2,
                        timeout=settings.llm_timeout,
                        callbacks=[KeyRotationCallbackHandler(key_index=i)],
                    )
                    for i, k in enumerate(settings.google_api_keys)
                ]
                
                llm_candidates = llms
                if len(llms) > 1:
                    # RoundRobinFallbackRunnable handles per-request fallback.
                    log.info(f"✅ Ротация Gemini ключей включена ({len(llms)} ключей)")
                else:
                    log.warning("⚠️  Только 1 Gemini ключ — ротация отключена")
                    llm = llms[0]

            elif settings.active_llm == "gigachat":
                log.debug("  Подключение к GigaChat...")
                llm = GigaChat(
                    credentials=settings.gigachat_credentials,
                    verify_ssl_certs=False,
                    model=settings.llm_model_name,
                    temperature=0.2
                )
                llm_candidates = [llm]

            elif settings.active_llm == "ollama":
                log.debug(f"  Подключение к Ollama: {settings.ollama_url}...")
                llm = ChatOpenAI(
                    base_url=settings.ollama_url,
                    api_key="ollama",
                    model=settings.llm_model_name,
                    temperature=0.2
                )
                llm_candidates = [llm]

            else:  # GROQ
                log.debug(f"  Ключей GROQ: {len(settings.groq_api_keys)}")
                
                if not settings.groq_api_keys:
                    raise ValueError("❌ GROQ_API_KEYS не установлена в .env!")
                
                llms = [
                    ChatOpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=k,
                        model=settings.llm_model_name,
                        temperature=0.2,
                        timeout=settings.llm_timeout,
                        max_retries=settings.llm_max_retries,
                        callbacks=[KeyRotationCallbackHandler(key_index=i)]
                    )
                    for i, k in enumerate(settings.groq_api_keys)
                ]
                
                llm_candidates = llms
                if len(llms) > 1:
                    # RoundRobinFallbackRunnable handles per-request fallback.
                    log.info(f"✅ Ротация GROQ ключей включена ({len(llms)} ключей)")
                else:
                    log.warning("⚠️  Только 1 GROQ ключ — ротация отключена")
                    llm = llms[0]

            log.debug("✓ LLM инициализирована")

            # === 5. ПРОМПТ ===
            log.info("📝 Создание prompt template...")
            prompt_template = ChatPromptTemplate.from_template("""
Ты технический эксперт компании "РегЛаб" и проходишь автотест по документации.
Твоя задача — выбрать правильный вариант ответа на тестовый вопрос, используя только найденный контекст.

Верни валидный JSON по заданной схеме QuizAnswerSchema.

Правила:
- Если в вопросе есть варианты ответа, final_answer должен начинаться с точного текста выбранного варианта.
- Если правильных вариантов несколько, перечисли все выбранные варианты отдельными строками.
- После выбранного варианта добавь короткое обоснование на 1-3 предложения.
- Не пиши приватную подсказку техподдержки, вопросы клиенту, internal notes или текст для Open WebUI.
- Не используй похожие обращения и тикеты для ответа на учебный тест.
- Не выдумывай факты. Если в контексте нет подтверждения, укажи это в missing_context и выбери наиболее вероятный вариант только с оговоркой.
- extracted_facts заполняй короткими фактами из контекста, которые подтверждают выбор.
- relevant_images оставь пустым списком, если в контексте нет явных путей к изображениям.

Глоссарий:
- R500: стандартные ПЛК РСУ.
- R500S: контроллеры безопасности ПСБ/SIL3.
- AstraRegul: верхний уровень, HMI, Server, Historian.

Контекст:
{context}

Тестовый вопрос:
{input}
""")

            # === 6. STRUCTURED OUTPUT ===
            if settings.active_llm == "gigachat":
                structured_llms = [candidate.with_structured_output(QuizAnswerSchema) for candidate in llm_candidates]
            elif settings.active_llm == "ollama":
                structured_llms = [
                    candidate.with_structured_output(QuizAnswerSchema, method="json_mode")
                    for candidate in llm_candidates
                ]
            else:
                method = "function_calling" if settings.llm_model_name in FUNCTION_CALLING_MODELS else "json_mode"
                structured_llms = [
                    candidate.with_structured_output(QuizAnswerSchema, method=method)
                    for candidate in llm_candidates
                ]

            log.debug(f"✓ Structured LLM готова (метод: {method if settings.active_llm not in ('gigachat', 'ollama') else settings.active_llm})")

            # === 7. СБОРКА ЦЕПОЧКИ ===
            log.info("⛓️  Сборка SGR цепочки...")
            structured_llm = (
                RoundRobinFallbackRunnable(structured_llms, label="Quiz LLM")
                if len(structured_llms) > 1
                else structured_llms[0]
            )

            sgr_chain = (
                {
                    "context": lambda query: format_docs(retriever.retrieve_docs(query)),
                    "input": RunnablePassthrough(),
                }
                | prompt_template
                | structured_llm
            )
            log.debug("✓ SGR цепочка собрана")

            support_prompt_template = ChatPromptTemplate.from_template("""
ВАЖНОЕ ПРАВИЛО КАЧЕСТВА ОТВЕТА:
Ты готовишь приватную подсказку инженеру техподдержки, а не финальный ответ клиенту.
Ответ должен помогать инженеру проверить гипотезу, а не звучать как уверенное решение без доказательств.

Обязательный стиль draft_private_comment:
- Пиши по-русски.
- Разделяй подтвержденные факты и предположения.
- Начинай с короткого вывода: что вероятнее всего и насколько это подтверждено.
- Используй разделы:
  **Что известно из обращения**
  **Что подтверждено источниками**
  **Наиболее вероятно**
  **Проверить сначала**
  **Гипотезы и ограничения**
  **Что запросить у клиента**
- В разделе "Что подтверждено источниками" указывай метки [D1], [D2], [T1] рядом с тезисами.
- В разделе "Гипотезы и ограничения" явно пиши, если вывод основан на похожем тикете или косвенном совпадении, а не на документации.
- Не утверждай существование конкретного поля интерфейса, параметра, чекбокса, вкладки, режима, команды или точного пути меню, если это не подтверждено найденным источником. Если это только предположение, пометь как "гипотеза".
- Не добавляй универсальные советы ради объема. Лучше 3-5 проверок, но привязанных к источникам и тексту обращения.
- Если источники слабо связаны с вопросом, прямо напиши, что прямого подтверждения нет, и снизь confidence до low или medium.
- confidence = high только если есть прямое подтверждение в документации или тикете с тем же симптомом и решением.
- confidence = medium если есть похожие случаи, но нет прямой инструкции.
- confidence = low если есть только общие сведения или не хватает контекста.

Правила для similar_tickets:
- Добавляй тикет только если он реально помогает текущему обращению.
- relevance_reason обязателен: укажи конкретное совпадение симптома, оборудования, версии, ошибки, лога или действия.
- solution_summary должен описывать только то, что действительно есть в найденном тикете. Не превращай опыт прошлого кейса в официальную инструкцию.
- Если похожие тикеты слабые, лучше верни similar_tickets = [].

Правила для evidence_notes:
- Каждая claim должна быть проверяемым тезисом.
- source_ids должны ссылаться только на доступные D/T источники.
- Если тезис является предположением, напиши это в comment.

Ты помощник инженера технической поддержки РегЛаб.

КРИТИЧЕСКИЕ ПРАВИЛА ЯЗЫКА:
- Все строковые поля JSON заполняй только на русском языке.
- Не начинай ответ словами "Engineer", "Client", "According to" и не используй англоязычные служебные заголовки.
- Если источник на русском, пересказывай его по-русски и сохраняй технические обозначения как есть: KEY, RUN/STOP, LD1-LD3, PF.
- draft_private_comment должен быть готовой приватной подсказкой инженеру ТП на русском языке.
- docs_answer должен быть на русском языке.
- similar_tickets.problem_summary, similar_tickets.solution_summary, evidence_notes.claim, recommended_questions и internal_notes тоже должны быть на русском языке.
- Для процедурных запросов сначала дай конкретную последовательность действий, если она есть в источниках.
- Не включай в similar_tickets обращение, если не можешь кратко сформулировать его проблему или решение.

Ты помощник сотрудника технической поддержки. Твой ответ является ПРИВАТНОЙ подсказкой для инженера, а не публичным ответом клиенту.

Нужно строго разделить:
1. Информацию из официальной документации.
2. Похожие обращения из базы тикетов.
3. Черновик приватного комментария для сотрудника.

Правила:
- Не утверждай, что проблема точно решена, если документация или тикеты не дают прямого решения.
- Если документация не дает точного решения, дай полезную общую информацию по темам из обращения.
- Тикеты используй как опыт прошлых кейсов, а не как официальную инструкцию.
- Для каждого похожего тикета укажи номер обращения, кратко проблему, кратко решение и почему он похож.
- Если данных мало, явно напиши, что нужно уточнить.
- Верни валидный JSON по заданной схеме.

Заявка:
{input}

Официальная документация:
{docs_context}

Источники документации:
{doc_sources_context}

Похожие обращения:
{tickets_context}

Источники похожих обращений:
{ticket_sources_context}

Дополнительные обязательные поля JSON:
- evidence_notes: список ключевых технических тезисов. Для каждого тезиса укажи source_ids из доступных источников, например ["D1"] или ["D1", "T1"]. Если тезис является предположением, явно напиши это в comment.
- recommended_questions: конкретные вопросы, файлы, логи, версии ПО или настройки, которые нужно запросить у клиента.
- internal_notes: проверки и действия, которые должен выполнить инженер техподдержки до ответа клиенту.
- Не добавляй публичный ответ клиенту. draft_private_comment должен оставаться приватной подсказкой инженеру.

Важно про похожие обращения:
- Если этот блок пустой, верни similar_tickets = [].
- Не придумывай номера обращений и не превращай документы в обращения.
- В similar_tickets можно включать только элементы, которые явно указаны выше как [TICKET ...].

Важно про источники:
- В docs_answer и draft_private_comment указывай ссылки на источники через метки [D1], [D2], [T1] рядом с подтверждаемыми тезисами.
- Используй только источники из блоков "Источники документации" и "Источники похожих обращений".
- Если подходящего источника нет, явно напиши, что в найденных источниках подтверждения нет.
- Не оставляй evidence_notes, recommended_questions и internal_notes пустыми, если по заявке есть что проверить или уточнить.
""")

            if settings.active_llm == "gigachat":
                support_structured_llms = [
                    candidate.with_structured_output(SupportPrivateResponse)
                    for candidate in llm_candidates
                ]
            else:
                support_structured_llms = [
                    candidate.with_structured_output(SupportPrivateResponse, method="json_mode")
                    for candidate in llm_candidates
                ]
            support_structured_llm = (
                RoundRobinFallbackRunnable(support_structured_llms, label="Support LLM")
                if len(support_structured_llms) > 1
                else support_structured_llms[0]
            )

            support_chain = support_prompt_template | support_structured_llm
            log.debug("✓ Support private chain собрана")

            return retriever, sgr_chain, support_chain

        except Exception as e:
            log.error(f"❌ Ошибка при построении pipeline: {e}", exc_info=True)
            raise RuntimeError(f"Failed to build RAG pipeline: {str(e)}") from e

    def process_support_ticket(self, query: str) -> SupportPrivateResponse:
        """Builds a private support hint with separate docs and ticket retrieval."""
        if not query:
            raise ValueError("query не может быть пустым")
        if not isinstance(query, str):
            raise ValueError(f"query должен быть строкой, получен {type(query).__name__}")
        query = query.strip()
        if not query:
            raise ValueError("query содержит только пробелы")

        start = time.perf_counter()
        docs: List = []
        tickets: List = []

        try:
            log.info(f"🔄 Приватная подсказка ТП: обработка заявки ({len(query)} символов)")

            docs = self.retriever.retrieve_docs(query)
            if settings.second_db_name == settings.active_db_name:
                log.warning(
                    "Ticket retrieval disabled: second_db points to the docs backend (%s)",
                    settings.second_db_name,
                )
                tickets = []
            else:
                tickets = [
                    doc for doc in self.retriever.retrieve_tickets(query)
                    if is_ticket_document(doc)
                ][:2]

            doc_sources = source_references(docs, prefix="D")
            ticket_sources = source_references(tickets, ticket_only=True, prefix="T")
            docs_context = format_docs(
                docs,
                max_total_chars=SUPPORT_DOCS_MAX_CHARS,
                max_doc_chars=SUPPORT_DOC_MAX_CHARS,
            )
            tickets_context = format_ticket_docs(
                tickets,
                max_total_chars=SUPPORT_TICKETS_MAX_CHARS,
                max_doc_chars=SUPPORT_TICKET_MAX_CHARS,
            )
            log.debug(
                "Support context size: docs=%s chars, tickets=%s chars, sources=%s/%s",
                len(docs_context),
                len(tickets_context),
                len(doc_sources),
                len(ticket_sources),
            )

            response = self.support_chain.invoke({
                "input": query,
                "docs_context": docs_context,
                "tickets_context": tickets_context,
                "doc_sources_context": format_source_references(doc_sources),
                "ticket_sources_context": format_source_references(ticket_sources),
            })
            response.doc_sources = doc_sources
            response.ticket_sources = ticket_sources
            if not tickets:
                response.similar_tickets = []
            else:
                response.similar_tickets = [
                    ticket for ticket in response.similar_tickets
                    if (ticket.relevance_reason or "").strip()
                ]

            elapsed = time.perf_counter() - start
            log.info(
                f"✅ Приватная подсказка готова за {elapsed:.2f}с "
                f"| docs={len(docs)} | tickets={len(tickets)}"
            )

            log_query_audit(
                query=query,
                equipment_filter=None,
                retrieved_docs=docs + tickets,
                response=response,
                elapsed_sec=elapsed,
            )

            return response

        except Exception as e:
            elapsed = time.perf_counter() - start
            log.error(f"❌ Ошибка при подготовке приватной подсказки: {e}", exc_info=True)
            log_query_audit(
                query=query,
                equipment_filter=None,
                retrieved_docs=docs + tickets,
                response=None,
                elapsed_sec=elapsed,
                extra={"error": str(e)},
            )
            raise RuntimeError(f"Failed to process support ticket: {str(e)}") from e

    def process_quiz_question(self, query: str) -> QuizAnswerSchema:
        """Обрабатывает учебный тестовый вопрос через legacy/quiz цепочку только по документации."""
        if not query:
            raise ValueError("query не может быть пустым")
        if not isinstance(query, str):
            raise ValueError(f"query должен быть строкой, получен {type(query).__name__}")
        query = query.strip()
        if not query:
            raise ValueError("query содержит только пробелы")

        start = time.perf_counter()
        try:
            log.info("Quiz autotest: обработка вопроса (%s символов)", len(query))
            response = self.sgr_chain.invoke(query)
            elapsed = time.perf_counter() - start
            log.info("Quiz autotest: ответ готов за %.2fс", elapsed)
            return response
        except Exception as e:
            elapsed = time.perf_counter() - start
            log.error("Quiz autotest failed after %.2fс: %s", elapsed, e, exc_info=True)
            raise RuntimeError(f"Failed to process quiz question: {str(e)}") from e

    def process_query(
        self,
        query: str,
        equipment_filter: Optional[List[str]] = None,
    ) -> RAGReasoningSchema:
        """Обрабатывает вопрос пользователя и возвращает структурированный ответ.

        Сохраняет найденные документы в self._last_docs — используйте их в UI
        для блока Sources вместо повторного вызова retriever.invoke().

        После каждого запроса (успешного или нет) записывает запись аудита
        в data/logs/query_audit.jsonl.

        Args:
            query:            Вопрос пользователя (непустая строка)
            equipment_filter: Фильтр по типу оборудования для docs-БД.
                              None или [] — поиск по всей документации.

        Returns:
            RAGReasoningSchema со структурированным ответом

        Raises:
            ValueError:    Если query пустой или не строка
            RuntimeError:  Если произошла ошибка при генерации ответа
        """
        support_response = self.process_support_ticket(query)
        return RAGReasoningSchema(
            user_intent=support_response.user_intent,
            extracted_facts=[
                FactExtraction(source_file="Документация", fact=support_response.docs_answer)
            ],
            missing_context=support_response.missing_context,
            final_answer=support_response.draft_private_comment,
            relevant_images=[],
        )

        # === ВАЛИДАЦИЯ ===
        if not query:
            raise ValueError("query не может быть пустым")
        if not isinstance(query, str):
            raise ValueError(f"query должен быть строкой, получен {type(query).__name__}")
        query = query.strip()
        if not query:
            raise ValueError("query содержит только пробелы")

        # === ФИЛЬТР ===
        if not equipment_filter:
            self.retriever.equipment_filter = ["Все"]
            log.debug("Фильтр оборудования: ВСЕ")
        else:
            if not isinstance(equipment_filter, list):
                raise ValueError(f"equipment_filter должен быть списком, получен {type(equipment_filter).__name__}")
            self.retriever.equipment_filter = equipment_filter
            log.debug(f"Фильтр оборудования: {', '.join(equipment_filter)}")

        # === ОБРАБОТКА ===
        start = time.perf_counter()
        self._last_docs = []

        try:
            log.info(f"🔄 Обработка запроса ({len(query)} символов)")
            log.debug(f"   Запрос: {query[:100]}{'...' if len(query) > 100 else ''}")

            # Поиск документов — сохраняем для UI и передаём в цепочку
            self._last_docs = self.retriever.invoke(query)

            # Генерация ответа (retriever внутри sgr_chain вызовется повторно —
            # это нормально, LangChain chain этого требует)
            response = self.sgr_chain.invoke(query)

            elapsed = time.perf_counter() - start
            log.info(
                f"✅ Ответ сгенерирован за {elapsed:.2f}с "
                f"| документов: {len(self._last_docs)} "
                f"| фактов: {len(response.extracted_facts)}"
            )

            log_query_audit(
                query=query,
                equipment_filter=equipment_filter,
                retrieved_docs=self._last_docs,
                response=response,
                elapsed_sec=elapsed,
            )

            return response

        except Exception as e:
            elapsed = time.perf_counter() - start
            log.error(f"❌ Ошибка при генерации ответа: {e}", exc_info=True)

            log_query_audit(
                query=query,
                equipment_filter=equipment_filter,
                retrieved_docs=self._last_docs,
                response=None,
                elapsed_sec=elapsed,
                extra={"error": str(e)},
            )

            raise RuntimeError(f"Failed to process query: {str(e)}") from e
