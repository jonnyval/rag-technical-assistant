import re
import time
import warnings
from threading import Lock
from typing import Any, Callable, Dict, List, Literal, Optional

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
from src.context_formatting import (
    SourceReference,
    format_adaptive_search_body,
    format_docs,
    format_source_references,
    format_ticket_docs,
    is_ticket_document,
    source_references,
)
from src.evidence_guard import (
    apply_definition_guard,
    apply_entity_citation_guard,
    apply_response_provenance,
    apply_transformation_evidence_guard,
    apply_entity_coverage_guard,
    apply_diagnostic_scope_guard,
    filter_documents_by_requested_series,
    check_and_format_equipment_mismatch_warning,
    strict_evidence_context,
)
from src.module_detection import (
    build_module_enriched_query,
    detect_modules_in_query,
    ensure_module_block,
    format_detected_modules,
    merge_documents,
    module_doc_page_title_hints,
    rank_tickets_for_modules,
)
from src.retrieval.dual_retriever import build_dual_retriever
from src.logger import log, log_query_audit

FUNCTION_CALLING_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
}

SUPPORT_DOCS_MAX_CHARS = 7000
SUPPORT_TICKETS_MAX_CHARS = 2600
SUPPORT_DOC_MAX_CHARS = 2800
SUPPORT_TICKET_MAX_CHARS = 1200
WIKI_DOCS_MAX_CHARS = 9000
WIKI_TICKETS_MAX_CHARS = 4200
WIKI_DOC_MAX_CHARS = 2600
WIKI_TICKET_MAX_CHARS = 1600
WIKI_ITERATIVE_MAX_ROUNDS = 1
WIKI_ITERATIVE_MAX_FOLLOWUP_QUERIES = 2
WIKI_REFLECTION_DOCS_MAX_CHARS = 3200
WIKI_REFLECTION_TICKETS_MAX_CHARS = 1400
WIKI_ITERATIVE_SKIP_MIN_DOCS = 2
WIKI_ITERATIVE_SKIP_MIN_DOC_CHARS = 2200

ADAPTIVE_INFORMATION_PROMPT = """
Ты — информационная поисковая система RegLab для специалистов технической поддержки.
Ты не инженер техподдержки, не советчик и не принимаешь решение по текущей ситуации.
Твоя задача — полно и компактно пересказать факты из найденной документации и отдельно описать, как аналогичные ситуации завершались в исторических обращениях.

Верни валидный JSON по схеме AdaptiveInformationResponse. Все пользовательские поля заполняй по-русски; технические обозначения сохраняй без изменений.

Приоритет источников:
1. Официальная документация — источник фактов, параметров, ограничений и документированных процедур.
2. Тикеты — только исторический опыт: что наблюдалось и как конкретное обращение было решено или закрыто.

Правила доказательности:
- Каждый технический факт сопровождай меткой [D…] или [T…] рядом с утверждением.
- Не добавляй сведения из общих знаний модели и не соединяй разрозненные фрагменты в новую причинно-следственную связь.
- Не переноси сведения между моделями, сериями и версиями без прямого подтверждения совместимости.
- Если прямого ответа нет, точно назови, какой факт не подтверждён найденными источниками.
- Тикет не подтверждает универсальное решение. Формулируй только в прошедшем времени: «в обращении [T…] наблюдалось…», «было решено…».

Стиль информационного поиска:
- Не давай рекомендаций, плана диагностики или чек-листа действий для текущей ситуации.
- Не используй обращения и команды: «проверьте», «сделайте», «убедитесь», «попробуйте», «обратитесь», «соберите», «следует», «рекомендуется».
- Если пользователь спрашивает «что делать», вместо совета сообщи: что прямо описано документацией и что делали в исторических тикетах.
- Документированную процедуру передавай полностью и в исходном порядке, но как описание источника: «Документация задаёт следующий порядок…».
- Не добавляй типовые причины, предупреждения и уточняющие вопросы ради объёма.
- Раскрывай все относящиеся к вопросу различающиеся факты, но не повторяй одну мысль разными словами.
- Если несколько документов описывают одну процедуру, объедини их в один порядок действий и поставь рядом все подтверждающие [D…]. Не пересказывай эту процедуру повторно по каждому источнику; отдельно добавь только реальные различия и дополнительные условия.

Поля JSON:
- docs_answer: самодостаточная фактическая выжимка только из официальной документации. Для процедуры сохраняй Markdown и отдельную строку на каждый шаг.
- similar_tickets: только действительно похожие обращения. Для каждого укажи ticket_id, ситуацию, фактическое решение/исход и конкретное сходство.
- draft_private_comment: та же информационная выжимка, а не черновик ответа и не рекомендация.
- evidence_notes: только проверяемые тезисы с доступными source_ids; если они не нужны, верни [].
- recommended_questions: всегда [].
- internal_notes: всегда [].
- missing_context: одна конкретная фраза только о факте, которого нет в найденных источниках; иначе «Не указано».
- confidence: оцени только полноту и прямоту найденных доказательств.

Вопрос:
{input}

Режим доказательности:
{strict_evidence_context}

Предупреждения о соответствии оборудования:
{equipment_mismatch_context}

Определённые модули:
{module_context}

Официальная документация:
{docs_context}

Источники документации:
{doc_sources_context}

Исторические обращения:
{tickets_context}

Источники исторических обращений:
{ticket_sources_context}
"""

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
    "adaptive": {
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


def apply_adaptive_information_role(response: Any) -> Any:
    """Make Adaptive output a neutral evidence digest instead of support advice."""
    response.recommended_questions = []
    response.internal_notes = []
    response.draft_private_comment = format_adaptive_search_body(response)
    return response

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


class EvidenceNote(BaseModel):
    """Проверяемый тезис ответа и источники, которыми он подтверждается."""

    claim: str = Field(default="", validation_alias=AliasChoices("claim", "thesis"), description="Короткий технический тезис или вывод")
    source_ids: List[str] = Field(default_factory=list, description="Метки источников, например D1, D2 или T1")
    comment: str = Field(default="", description="Оговорка, если источник подтверждает тезис не напрямую")


def _normalize_confidence_value(value: Any) -> str:
    """Normalize provider confidence variants to the public low/medium/high scale."""
    if value is None or isinstance(value, bool):
        return "medium"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized
        try:
            value = float(normalized.replace(",", "."))
        except (TypeError, ValueError):
            return "medium"
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return "medium"
    if score != score or score < 0:
        return "medium"
    if 1 < score <= 100:
        score /= 100
    if score > 1:
        return "medium"
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


class SupportPrivateResponse(BaseModel):
    """Приватная подсказка ИИ для сотрудника техподдержки."""
    user_intent: str = Field(default="", description="Что хочет выяснить или выполнить инженер ТП")
    docs_answer: str = Field(description="Информация из документации. Если прямого решения нет, перечисли релевантные темы.")
    related_topics: List[str] = Field(default_factory=list, description="Темы из документации, которые относятся к обращению")
    similar_tickets: List[SimilarTicketSummary] = Field(default_factory=list, description="Похожие обращения из базы тикетов")
    evidence_notes: List[EvidenceNote] = Field(default_factory=list, description="Ключевые тезисы ответа с привязкой к источникам")
    recommended_questions: List[str] = Field(default_factory=list, description="Какие данные стоит уточнить, только если без них нельзя продолжить")
    internal_notes: List[str] = Field(default_factory=list, description="Внутренний SGR-аудит и проверки для инженера техподдержки")
    missing_context: str = Field(default="Не указано", description="Каких данных не хватает для уверенного ответа")
    draft_private_comment: str = Field(description="Готовый приватный комментарий для сотрудника ТП в Markdown")
    confidence: Literal["low", "medium", "high"] = Field(default="medium", description="low | medium | high")
    doc_sources: List[SourceReference] = Field(default_factory=list, description="Документные источники, добавленные кодом после поиска")
    ticket_sources: List[SourceReference] = Field(default_factory=list, description="Источники похожих обращений, добавленные кодом после поиска")


# ==========================================
# 🛠 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

    @field_validator("similar_tickets", "evidence_notes", mode="before")
    @classmethod
    def normalize_object_lists(cls, value):
        """Treat provider JSON null as an empty list for optional structured fields."""
        return [] if value is None else value
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

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        return _normalize_confidence_value(value)

class AdaptiveSimilarTicketSummary(SimilarTicketSummary):
    """Accept common provider field names without losing retrieved ticket facts."""

    problem_summary: str = Field(
        default="",
        validation_alias=AliasChoices("problem_summary", "problem", "situation"),
        description="Фактическая ситуация из исторического обращения",
    )
    solution_summary: str = Field(
        default="",
        validation_alias=AliasChoices("solution_summary", "solution", "resolution"),
        description="Как это историческое обращение было решено или закрыто",
    )
    relevance_reason: str = Field(
        default="",
        validation_alias=AliasChoices("relevance_reason", "similarity_reason", "similarity"),
        description="Конкретное сходство с текущим вопросом",
    )


class AdaptiveInformationResponse(SupportPrivateResponse):
    """Structured result for the adaptive information-search profile."""

    user_intent: str = Field(default="", description="Краткое описание информационного запроса пользователя")
    docs_answer: str = Field(description="Все релевантные факты из документации с точными ссылками [D...] без советов от модели")
    similar_tickets: List[AdaptiveSimilarTicketSummary] = Field(
        default_factory=list,
        description="Только фактическое описание проблемы и решения из похожих исторических обращений",
    )
    evidence_notes: List[EvidenceNote] = Field(
        default_factory=list,
        description="Проверяемые тезисы с source_ids; пустой список допустим и предпочтителен",
    )
    recommended_questions: List[str] = Field(default_factory=list, description="Всегда пустой список")
    internal_notes: List[str] = Field(default_factory=list, description="Всегда пустой список")
    draft_private_comment: str = Field(
        description="Информационная сводка: факты документации и отдельно опыт похожих обращений; без рекомендаций"
    )

    @field_validator("evidence_notes", mode="before")
    @classmethod
    def normalize_adaptive_evidence_notes(cls, value):
        """Tolerate providers returning compact cited strings instead of objects."""
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        normalized = []
        for item in value:
            if isinstance(item, str):
                source_ids = re.findall(r"\b[DT]\d+\b", item, flags=re.IGNORECASE)
                normalized.append({"claim": item.strip(), "source_ids": source_ids})
            else:
                normalized.append(item)
        return normalized


class WikiChatResponse(BaseModel):
    """Wiki-style chat response: documentation first, tickets as historical cases only."""

    user_intent: str = Field(default="", description="Short Russian restatement of what the user wants to know")
    docs_answer: str = Field(default="", description="Russian summary of all relevant official documentation facts with [D...] citations")
    related_topics: List[str] = Field(default_factory=list, description="Related documentation topics or sections, in Russian")
    similar_tickets: List[SimilarTicketSummary] = Field(default_factory=list, description="Similar historical tickets, not instructions for the current request")
    evidence_notes: List[EvidenceNote] = Field(default_factory=list, description="Verifiable claims with source_ids from documentation or ticket sources")
    missing_context: str = Field(default="", description="Russian description of direct information not found in retrieved sources")
    source_limitations: List[str] = Field(default_factory=list, description="Weak matches, model/version mismatches, or missing direct evidence, in Russian")
    final_answer: str = Field(description="Final Russian Markdown wiki answer for the user")
    confidence: Literal["low", "medium", "high"] = Field(default="medium", description="low | medium | high")
    doc_sources: List[SourceReference] = Field(default_factory=list, description="Documentation sources attached by code after retrieval")
    ticket_sources: List[SourceReference] = Field(default_factory=list, description="Ticket sources attached by code after retrieval")

    @field_validator("related_topics", "source_limitations", mode="before")
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


    @field_validator("user_intent", "docs_answer", "missing_context", "final_answer", mode="before")
    @classmethod
    def normalize_text_fields(cls, value):
        """Replace null and non-string scalar values with safe strings."""
        if value is None:
            return ""
        return str(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        return _normalize_confidence_value(value)


class MultiQueryPlan(BaseModel):
    """Retrieval-only query decomposition produced before RAG-Fusion."""

    queries: List[str] = Field(
        default_factory=list,
        description="Legacy shared search-query alternatives",
    )
    documentation_queries: List[str] = Field(
        default_factory=list,
        description="Focused queries for official documentation",
    )
    ticket_queries: List[str] = Field(
        default_factory=list,
        description="Focused symptom/case queries for historical tickets",
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Exact product, module, parameter, error, path and version anchors",
    )

    @field_validator(
        "queries",
        "documentation_queries",
        "ticket_queries",
        "entities",
        mode="before",
    )
    @classmethod
    def normalize_queries(cls, value):
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        result = []
        seen = set()
        for item in values:
            text = " ".join(str(item or "").split()).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                result.append(text)
        return result

class RetrievalReflection(BaseModel):
    """Internal decision about whether wiki retrieval needs more focused searches."""

    enough_context: bool = Field(default=True, description="True when current retrieved docs/tickets are enough for a cited wiki answer")
    missing_facts: List[str] = Field(default_factory=list, description="Missing facts that should be searched for before answering")
    followup_doc_queries: List[str] = Field(default_factory=list, description="Additional documentation search queries, most important first")
    followup_ticket_queries: List[str] = Field(default_factory=list, description="Additional historical ticket search queries, most important first")
    product_line_filter: str = Field(default="", description="Exact product/model constraint to preserve, for example R500 or R500S")
    source_risks: List[str] = Field(default_factory=list, description="Warnings about weak, mixed, or mismatched retrieved sources")

    @field_validator("missing_facts", "followup_doc_queries", "followup_ticket_queries", "source_risks", mode="before")
    @classmethod
    def normalize_string_lists(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []
        if isinstance(value, list):
            normalized = []
            seen = set()
            for item in value:
                item_text = str(item).strip()
                if not item_text or item_text in seen:
                    continue
                seen.add(item_text)
                normalized.append(item_text)
            return normalized
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

    @staticmethod
    def _failure_policy(error: Exception) -> tuple[bool, float]:
        """Return whether another credential may help and the per-key cooldown."""
        message = str(error).lower()
        permanent_key_or_model_error = any(token in message for token in (
            "401", "403", "404", "not_found", "permission_denied", "invalid api key", "api key not valid",
        ))
        transient_provider_error = any(token in message for token in (
            "429", "500", "502", "503", "504", "unavailable", "timeout", "timed out", "connection",
        ))
        if permanent_key_or_model_error:
            return True, 15 * 60.0
        if transient_provider_error:
            return True, 30.0
        # Schema, validation and prompt errors are deterministic for every key.
        return False, 0.0

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        """Try healthy credentials only; do not multiply deterministic failures."""
        last_error: Optional[Exception] = None
        now = time.monotonic()
        for key_index, runnable in self._ordered_runnables():
            retry_after = getattr(self, "_retry_after", {}).get(key_index, 0.0)
            if retry_after > now:
                log.debug("%s: key #%s is in cooldown for %.0fs", self.label, key_index, retry_after - now)
                continue
            try:
                log.debug("%s: attempt via key #%s", self.label, key_index)
                return runnable.invoke(input, config=config, **kwargs)
            except Exception as error:
                last_error = error
                should_failover, cooldown = self._failure_policy(error)
                if cooldown:
                    if not hasattr(self, "_retry_after"):
                        self._retry_after = {}
                    self._retry_after[key_index] = time.monotonic() + cooldown
                if not should_failover:
                    log.error("%s: deterministic failure on key #%s; fallback skipped: %s", self.label, key_index, error)
                    raise
                log.warning("%s: key #%s failed; trying next healthy key: %s", self.label, key_index, error)

        raise last_error or RuntimeError(f"{self.label}: all credentials are temporarily unavailable")


class RAGEngine:
    """Оркестратор RAG системы: поиск по двум БД + LLM + SGR цепочка.

    Attributes:
        retriever:   DualRetriever — параллельный поиск по документации и тикетам
        sgr_chain:   LangChain цепочка для генерации структурированного ответа
        _last_docs:  Документы из последнего вызова (для отображения в Sources без
                     повторного поиска)
    """

    def __init__(
        self,
        profile: str = "deep",
        *,
        shared_embeddings: Any | None = None,
        shared_reranker: Any | None = None,
        llm_max_completion_tokens: int | None = None,
        llm_reasoning_effort: str | None = None,
    ):
        """Собирает retriever, LLM-цепочки и служебное состояние RAG-движка."""

        try:
            log.info("🚀 Инициализация RAGEngine...")
            self.profile = profile if profile in RAG_PROFILES else "deep"
            self.profile_config = RAG_PROFILES[self.profile]
            self.dense_embeddings = shared_embeddings
            self.rerank_model = shared_reranker
            self.llm_max_completion_tokens = llm_max_completion_tokens
            self.llm_reasoning_effort = llm_reasoning_effort
            log.info(
                "RAGEngine initialization: profile=%s top_k=%s/%s reranker=%s",
                self.profile,
                self.profile_config["top_k_retrieval"],
                self.profile_config["top_k_final"],
                self.profile_config["use_reranker"],
            )
            self.retriever, self.sgr_chain, self.support_chain, self.chat_chain, self.retrieval_reflection_chain, self.multi_query_chain = self._build_pipeline()
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
            dense_embeddings = self.dense_embeddings
            if dense_embeddings is None:
                log.info(f"📥 Загрузка эмбеддингов: {settings.embedding_model_name}...")
                dense_embeddings = HuggingFaceEmbeddings(
                    model_name=settings.embedding_model_name,
                    model_kwargs={'device': settings.device},
                    encode_kwargs={
                        'normalize_embeddings': True,
                        'prompt': 'Instruct: Retrieve relevant technical documentation passage to answer the query.\nQuery: ',
                    }
                )
            else:
                log.info("♻️ Переиспользование уже загруженной embedding-модели")
            self.dense_embeddings = dense_embeddings
            log.debug("✓ Dense embeddings загружены")

            # === 2. РЕРАНКЕР (опционально) ===
            rerank_model = self.rerank_model
            if self.profile_config["use_reranker"]:
                if rerank_model is None:
                    log.info(f"📊 Загрузка реранкера: {settings.reranker_model_name}...")
                    rerank_model = CrossEncoder(
                        settings.reranker_model_name,
                        device=settings.device,
                        trust_remote_code=True
                    )
                else:
                    log.info("♻️ Переиспользование уже загруженного реранкера")
                log.debug("✓ Реранкер загружен")
            else:
                rerank_model = None
                log.info("⏭️  Реранкер отключён (use_reranker: false) — используется порядок векторного скора")
            self.rerank_model = rerank_model

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

                is_gemma4 = settings.llm_model_name.lower().startswith("gemma-4-")
                gemini_request_kwargs = {
                    "timeout": 90 if is_gemma4 else settings.llm_timeout,
                    "max_retries": 0 if is_gemma4 else settings.llm_max_retries,
                }
                if is_gemma4:
                    gemini_request_kwargs.update(
                        thinking_level="minimal",
                        include_thoughts=False,
                    )
                    log.info("Gemma 4: minimal thinking, 90s timeout, retries disabled")

                llms = [
                    ChatGoogleGenerativeAI(
                        model=settings.llm_model_name,
                        google_api_key=k,
                        temperature=0.2,
                        **gemini_request_kwargs,
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

                groq_generation_kwargs = {}
                if self.llm_max_completion_tokens is not None:
                    groq_generation_kwargs["max_completion_tokens"] = self.llm_max_completion_tokens
                if self.llm_reasoning_effort:
                    groq_generation_kwargs["reasoning_effort"] = self.llm_reasoning_effort

                llms = [
                    ChatOpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=k,
                        model=settings.llm_model_name,
                        temperature=0.2,
                        timeout=settings.llm_timeout,
                        max_retries=settings.llm_max_retries,
                        callbacks=[KeyRotationCallbackHandler(key_index=i)],
                        **groq_generation_kwargs,
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
Ты готовишь рабочую подсказку для инженера техподдержки.
Ответ должен помогать инженеру проверить гипотезу, а не звучать как уверенное решение без доказательств.

СТРОГИЙ РЕЖИМ ДОКАЗАТЕЛЬНОСТИ:
- Каждый технический факт связывай с конкретным фрагментом [D…] или [T…]. Если прямого подтверждения нет, кратко сообщи об этом.
- Для вопросов «что это», «что за библиотека», «что означает», «расшифруй» сначала дай определение из источника. Если прямого определения нет, явно отдели это: «В найденной документации RegLab прямого описания <термин> не найдено».
- После такой оговорки можно добавить 1–2 предложения с пометкой «Общее техническое пояснение (проверьте версию и среду):» — только для устойчивого общего назначения термина. Не выдавай в этом блоке точные API, пути меню, совместимость версий, параметры или последовательность действий.
- Не выводи новое правило из разрозненных фрагментов и не переносись между разными моделями/сериями без явной совместимости в источнике.
- Для функций, библиотек и преобразований проверь совпадение действия целиком: экранирование строки, кодировка, символьное представление и массив кодов — разные операции. Не подменяй одну другой по похожему имени функции.
- Для ошибки, самодиагностики, параметра конфигурации или отключения проверки сначала укажи условие применимости. Изменение системного параметра можно описывать только как подтверждённый сценарий для той же причины, а не как универсальное решение.
- Для широкого вопроса о серии контроллеров перечисляй только свойства, прямо относящиеся к этой серии в источнике. Не превращай документацию о модуле, функции или иной серии в описание контроллера.

ФОРМАТ ОТВЕТА:
- Сначала определи тип вопроса. На справочный вопрос ответь кратким определением или перечнем возможностей. Нумерованные шаги используй только для явно запрошенной процедуры.
- Для подтверждённой процедуры передай все найденные шаги в исходном порядке; не заменяй их общей ссылкой на документацию.
- Ставь [D…]/[T…] рядом с фактом или шагом. Похожий тикет описывай как «В историческом обращении [T…] применялось …» и явно указывай, что это пример, а не подтверждённое определение или гарантированное решение для текущего случая.
- Если данных не хватает, назови ровно недостающий факт. Пустые поля JSON возвращай как [] и не заполняй их шаблонными вопросами или проверками.
- Ответ предназначен коллеге-инженеру ТП. Пиши по-русски, коротко и прикладно; не описывай собственное рассуждение.
Правила для similar_tickets:
- Добавляй тикет только если он реально помогает текущему обращению.
- relevance_reason обязателен: укажи конкретное совпадение симптома, оборудования, версии, ошибки, лога или действия.
- solution_summary должен описывать только то, что действительно есть в найденном тикете. Не превращай опыт прошлого кейса в официальную инструкцию.
- Если похожие тикеты слабые, лучше верни similar_tickets = [].

Правила для evidence_notes:
- Каждая claim должна быть проверяемым тезисом.
- source_ids должны ссылаться только на доступные D/T источники.
- Если тезис является предположением, напиши это в comment.
- Не добавляй claim, если рядом с ним нельзя поставить конкретный ID фрагмента из "Официальная документация" или "Похожие обращения".

Ты помощник инженера технической поддержки РегЛаб.

КРИТИЧЕСКИЕ ПРАВИЛА ЯЗЫКА:
- Все строковые поля JSON заполняй только на русском языке.
- Не начинай ответ словами "Engineer", "Client", "According to" и не используй англоязычные служебные заголовки.
- Если источник на русском, пересказывай его по-русски и сохраняй технические обозначения как есть: KEY, RUN/STOP, LD1-LD3, PF.
- draft_private_comment должен быть готовой приватной подсказкой инженеру ТП на русском языке.
- docs_answer должен быть на русском языке.
- docs_answer must keep Markdown line breaks. If documentation contains a procedure or ordered steps, write each step on a separate numbered line. Never compress steps as "1) ...; 2) ...; 3) ..." in one paragraph.
- similar_tickets.problem_summary, similar_tickets.solution_summary, evidence_notes.claim, recommended_questions и internal_notes тоже должны быть на русском языке.
- Для процедурных запросов сначала дай конкретную последовательность действий, если она есть в источниках.
- Не включай в similar_tickets обращение, если не можешь кратко сформулировать его проблему или решение.

Ты помощник сотрудника технической поддержки. Отвечай как рабочий помощник коллеге-инженеру.

Нужно строго разделить:
1. Информацию из официальной документации.
2. Похожие обращения из базы тикетов.
3. Черновик приватного комментария для сотрудника.

Правила:
- Не утверждай, что проблема точно решена, если документация или тикеты не дают прямого решения.
- Если документация не дает точного решения, дай полезную общую информацию по темам из обращения.
- Тикеты используй как опыт прошлых кейсов, а не как официальную инструкцию.
- Для каждого похожего тикета укажи номер обращения, кратко проблему, кратко решение и почему он похож.
- Если данных недостаточно, кратко назови ровно недостающий факт; не перечисляй общие уточнения.
- Верни валидный JSON по заданной схеме.
- Если в блоке "Модули, определенные по вопросу" есть модули [M...], обязательно включи их в docs_answer и draft_private_comment без изменения кода и канонического имени.
- Не заменяй канонический код модуля на похожий. Например, DI032011 должен оставаться R500 DI 32 011, а не DI 32 111.

Заявка:
{input}

Режим доказательности:
{strict_evidence_context}

Проверка совпадения оборудования:
{equipment_mismatch_context}

Модули, определенные по вопросу:
{module_context}

Официальная документация:
{docs_context}

Источники документации:
{doc_sources_context}

Похожие обращения:
{tickets_context}

Источники похожих обращений:
{ticket_sources_context}

Дополнительные поля JSON — это внутренний SGR-аудит, они не являются частью ответа в чате:
- evidence_notes: только проверяемые ключевые тезисы с source_ids.
- recommended_questions: заполняй только при реально недостающих для безопасного действия данных; иначе [].
- internal_notes: только необходимые внутренние проверки; иначе [].
- draft_private_comment — готовая краткая рабочая подсказка коллеге-инженеру, не ответ «для клиента».
Важно про похожие обращения:
- Если этот блок пустой, верни similar_tickets = [].
- Не придумывай номера обращений и не превращай документы в обращения.
- В similar_tickets можно включать только элементы, которые явно указаны выше как [TICKET ...].

Важно про источники:
- В docs_answer и draft_private_comment указывай ссылки на источники через метки [D1], [D2], [T1] рядом с подтверждаемыми тезисами.
- Используй только источники из блоков "Источники документации" и "Источники похожих обращений".
- Если подходящего источника нет, явно напиши, что в найденных источниках подтверждения нет.
- Не заполняй recommended_questions и internal_notes ради полноты: пустой список предпочтительнее выдуманных проверок.
""")

            if self.profile == "adaptive":
                support_prompt_template = ChatPromptTemplate.from_template(
                    ADAPTIVE_INFORMATION_PROMPT
                )

            support_response_schema = (
                AdaptiveInformationResponse
                if self.profile == "adaptive"
                else SupportPrivateResponse
            )
            if settings.active_llm == "gigachat":
                support_structured_llms = [
                    candidate.with_structured_output(support_response_schema)
                    for candidate in llm_candidates
                ]
            else:
                support_structured_llms = [
                    candidate.with_structured_output(support_response_schema, method="json_mode")
                    for candidate in llm_candidates
                ]
            support_structured_llm = (
                RoundRobinFallbackRunnable(support_structured_llms, label="Support LLM")
                if len(support_structured_llms) > 1
                else support_structured_llms[0]
            )

            support_chain = support_prompt_template | support_structured_llm
            log.debug("✓ Support private chain собрана")

            wiki_prompt_template = ChatPromptTemplate.from_template("""
You are RegLab Wiki Chat for OpenWebUI.
Your job is to collect and summarize information found in the provided sources. You are not a support engineer and you must not try to solve the user's incident.

Return valid JSON matching WikiChatResponse.
All user-facing string fields must be written in Russian. Keep technical names, module codes, interface names, parameters and ticket IDs exactly as they appear in sources.

Core policy:
1. Official documentation is the primary source of facts.
2. Support tickets are historical cases only: describe what happened in those tickets and how they were resolved.
3. Never adapt a historical ticket solution as the solution for the current user request.

Evidence rules:
- Every technical fact in docs_answer and final_answer must cite a source marker near the claim: [D1], [D2], [T1].
- Use [T...] only for historical-ticket statements or phrases explicitly marked as a similar ticket case.
- Do not infer definitions of abbreviations, suffixes, letters in markings, parameters or UI controls from indirect matches.
- If there is no direct definition or instruction, explicitly write in Russian that direct confirmation was not found in the retrieved sources.
- Do not merge statements from different sources into a new conclusion unless that link is explicit in the sources.
- Preserve exact canonical module and equipment names. Do not replace them with similar names.

Forbidden in wiki mode:
- Do not write diagnostics such as "check first", "recommended action", "probable cause", or "ask the client" unless this is a documented procedure in an official source.
- Do not provide a troubleshooting plan. If a procedure exists in documentation, present it as a documented reference procedure, not as your recommendation.
- Do not claim that the current issue is solved by a historical ticket.
- Do not include private support notes, internal notes, or a draft reply to a client.
- Do not include a detected-modules/debug block in the final answer.
- Do not add generic advice for volume.

Required final_answer structure:
Write the following three section headings in Russian, exactly matching their meanings:
1. What was found in documentation.
2. Similar historical support tickets.
3. What was not found in the retrieved sources.

In the documentation section, list all relevant documented information: definitions, purpose, parameters, limits, procedures, versions and applicability conditions. If a source refers to a different model, series or version, state that next to the source marker.

In the ticket section, include only truly similar tickets. For each ticket, give ticket ID, problem summary, how that historical ticket was resolved or closed, and why it is similar. Phrase every ticket item as historical fact, not as a recommendation for the current user.

In the missing-context section, state missing direct definitions, instructions, confirmations, limits or applicability details.

JSON field rules:
- docs_answer: only official documentation facts with [D...] markers. Preserve Markdown line breaks; procedures and ordered lists must use one step per line.
- final_answer: preserve Markdown line breaks; never compress numbered steps into a single paragraph.
- similar_tickets: only tickets explicitly present in the provided ticket context. Do not invent ticket IDs or URLs.
- similar_tickets.solution_summary: only how the historical ticket was resolved; do not transfer the solution to the current request.
- evidence_notes: verifiable claims only, with source_ids from available sources.
- source_limitations: weak matches, model/version mismatches, absent direct instruction or absent definition.
- confidence = high only with direct documentation for the requested entity; medium for partial docs or similar tickets; low for weak matches.

Evidence mode:
{strict_evidence_context}

Metadata warnings:
{equipment_mismatch_context}

Detected modules:
{module_context}

Official documentation:
{docs_context}

Documentation sources:
{doc_sources_context}

Similar support tickets:
{tickets_context}

Ticket sources:
{ticket_sources_context}

Chat history:
{chat_history}

User question:
{input}
""")

            if settings.active_llm == "gigachat":
                wiki_structured_llms = [
                    candidate.with_structured_output(WikiChatResponse)
                    for candidate in llm_candidates
                ]
            else:
                wiki_structured_llms = [
                    candidate.with_structured_output(WikiChatResponse, method="json_mode")
                    for candidate in llm_candidates
                ]
            wiki_structured_llm = (
                RoundRobinFallbackRunnable(wiki_structured_llms, label="Wiki Chat LLM")
                if len(wiki_structured_llms) > 1
                else wiki_structured_llms[0]
            )

            chat_chain = wiki_prompt_template | wiki_structured_llm
            log.debug("Wiki chat SGR chain built")

            reflection_prompt_template = ChatPromptTemplate.from_template("""
You are the internal retrieval controller for RegLab Wiki Chat.
Do not answer the user. Decide whether the current source set is enough for a factual, cited wiki answer.
Return valid JSON matching RetrievalReflection.

Rules:
- Official documentation is primary. Tickets are only historical examples.
- Request follow-up searches only when they can likely add direct evidence or clarify a source mismatch.
- Keep follow-up queries short, concrete, and searchable.
- Preserve exact product/model names from the user question. R500 and R500S are different products; never treat them as interchangeable.
- If the user asks about a specific model, follow-up queries must include that exact model name when relevant.
- Do not repeat queries already listed in Already tried queries.
- Use at most three total follow-up queries per round across documentation and tickets.
- Mark enough_context=true when sources already contain direct documentation for the requested entity or when additional retrieval is unlikely to help.

Detected modules:
{module_context}

Already tried queries:
{already_tried_queries}

Current documentation context:
{docs_context}

Current ticket context:
{tickets_context}

Chat history:
{chat_history}

User question:
{input}
""")

            if settings.active_llm == "gigachat":
                reflection_structured_llms = [
                    candidate.with_structured_output(RetrievalReflection)
                    for candidate in llm_candidates
                ]
            else:
                reflection_structured_llms = [
                    candidate.with_structured_output(RetrievalReflection, method="json_mode")
                    for candidate in llm_candidates
                ]
            reflection_structured_llm = (
                RoundRobinFallbackRunnable(reflection_structured_llms, label="Wiki Retrieval Reflection LLM")
                if len(reflection_structured_llms) > 1
                else reflection_structured_llms[0]
            )
            retrieval_reflection_chain = reflection_prompt_template | reflection_structured_llm
            log.debug("Wiki retrieval reflection chain built")

            multi_query_chain = None
            if settings.multi_query_enabled:
                multi_query_prompt = ChatPromptTemplate.from_template("""
You are a retrieval planner for a RegLab technical-support RAG system.
Do not answer the user, diagnose the issue, or propose a solution. Return JSON matching MultiQueryPlan.

Planner mode:
{planner_instructions}

Shared rules:
- The original query is already searched; never repeat it.
- Decompose independent evidence needs instead of producing cosmetic paraphrases.
- Preserve exact product names, module codes, identifiers, error text, paths, commands and versions from the user query.
- Every entity in `entities` must occur verbatim in the original query. Do not infer expansions or related models.
- Do not invent error codes, product models, menu paths, causes, actions or solutions.
- Queries must be short, standalone and useful without conversation history.
- Russian is preferred; keep established English technical tokens unchanged.

For `documentation_queries`, search for definitions, purpose, parameters, limits, configuration, procedures and applicability conditions explicitly requested or implied by the question structure.
For `ticket_queries`, search for historical cases using concrete equipment, symptoms, observed state, error text and operating conditions from the question.
Do not copy the same query to both lists unless its exact anchors are genuinely useful for both source types.

Limits:
- documentation_queries: at most {max_documentation_queries}
- ticket_queries: at most {max_ticket_queries}
- legacy queries: at most {max_queries}

Original query:
{input}

Detected module context:
{module_context}
""")
                if settings.active_llm == "gigachat":
                    expansion_llms = [candidate.with_structured_output(MultiQueryPlan) for candidate in llm_candidates]
                else:
                    expansion_llms = [
                        candidate.with_structured_output(MultiQueryPlan, method="json_mode")
                        for candidate in llm_candidates
                    ]
                expansion_llm = (
                    RoundRobinFallbackRunnable(expansion_llms, label="Multi-Query LLM")
                    if len(expansion_llms) > 1
                    else expansion_llms[0]
                )
                multi_query_chain = multi_query_prompt | expansion_llm
                log.info("Multi-Query RAG-Fusion is enabled")

            return retriever, sgr_chain, support_chain, chat_chain, retrieval_reflection_chain, multi_query_chain

        except Exception as e:
            log.error(f"❌ Ошибка при построении pipeline: {e}", exc_info=True)
            raise RuntimeError(f"Failed to build RAG pipeline: {str(e)}") from e

    @staticmethod
    def _is_incident_query(query: str) -> bool:
        """Return whether a question is likely to need ticket diagnostics."""
        normalized = (query or "").lower().replace("ё", "е")
        markers = (
            "ошиб", "исключ", "отказ", "не работает", "не загруж",
            "download denied", "denied", "сбой", "авари", "не видит",
            "не запуска", "лог", "прошив", "firmware",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _adaptive_ticket_results_are_weak(tickets: List[Any]) -> bool:
        """Use the expensive top-80 rerank only when the first ticket pass is weak."""
        if len(tickets) < 3:
            return True
        scores = [
            float((getattr(ticket, "metadata", {}) or {}).get("rerank_score", 0.0) or 0.0)
            for ticket in tickets
        ]
        return not scores or max(scores) < 0.65

    def _adaptive_doc_results_are_weak(
        self,
        documents: List[Any],
        *,
        query_count: int,
    ) -> bool:
        """Detect a collapsed or weak RRF result before paying for doc reranking."""
        target = min(3, max(1, int(self.profile_config.get("top_k_final", 4))))
        if len(documents) < target:
            return True
        if query_count <= 1:
            return False

        scores = [
            float((getattr(document, "metadata", {}) or {}).get("rerank_score", 0.0) or 0.0)
            for document in documents
        ]
        rrf_k, _ = self._multi_query_options()
        # A strong RRF winner should be supported by at least two query variants.
        minimum_multi_query_score = 1.5 / (rrf_k + 1)
        return not scores or max(scores) < minimum_multi_query_score
    @staticmethod
    def _multi_query_options() -> tuple[int, int]:
        config = settings.multi_query_config
        try:
            rrf_k = max(1, int(config.get("rrf_k", 60)))
            candidate_limit = max(1, int(config.get("max_candidates", 60)))
        except (TypeError, ValueError):
            return 60, 60
        return rrf_k, candidate_limit

    @staticmethod
    def _prepare_query_variants(
        query: str,
        retrieval_query: str,
        variants: Any,
        detected_modules: List[Dict[str, Any]],
        max_queries: int,
    ) -> List[str]:
        """Validate generated queries and preserve deterministic user anchors."""
        requested_series = {
            item.upper()
            for item in re.findall(r"\bR\d{3}S?\b", query, flags=re.IGNORECASE)
        }
        searches = [retrieval_query]
        seen = {re.sub(r"\W+", "", retrieval_query.lower())}
        for variant in list(variants or [])[:max_queries]:
            variant = " ".join(str(variant).split()).strip()
            raw_key = re.sub(r"\W+", "", variant.lower())
            if len(raw_key) < 4 or len(variant) > 360:
                continue
            variant_series = {
                item.upper()
                for item in re.findall(r"\bR\d{3}S?\b", variant, flags=re.IGNORECASE)
            }
            if requested_series and not requested_series.issubset(variant_series):
                variant = f"{variant} {' '.join(sorted(requested_series))}"
            focused_query = build_module_enriched_query(variant, detected_modules)
            key = re.sub(r"\W+", "", focused_query.lower())
            if len(key) < 4 or key in seen:
                continue
            seen.add(key)
            searches.append(focused_query)
        return searches

    def _build_multi_query_searches(
        self,
        query: str,
        retrieval_query: str,
        detected_modules: List[Dict[str, Any]],
        module_context: str,
    ) -> tuple[List[str], List[str]]:
        """Build bounded source-specific searches, preserving the legacy path."""
        if not settings.multi_query_enabled or self.multi_query_chain is None:
            return [retrieval_query], [retrieval_query]

        structured = settings.structured_query_planner_enabled
        try:
            legacy_max = max(
                0,
                min(int(settings.multi_query_config.get("max_generated_queries", 2)), 3),
            )
        except (TypeError, ValueError):
            legacy_max = 2
        structured_config = settings.structured_query_planner_config
        try:
            docs_max = max(
                0,
                min(int(structured_config.get("max_documentation_queries", 3)), 4),
            )
            tickets_max = max(
                0,
                min(int(structured_config.get("max_ticket_queries", 3)), 4),
            )
        except (TypeError, ValueError):
            docs_max, tickets_max = 3, 3
        if not structured and not legacy_max:
            return [retrieval_query], [retrieval_query]
        if structured and not docs_max and not tickets_max:
            return [retrieval_query], [retrieval_query]

        planner_instructions = (
            "Structured mode: fill `entities`, `documentation_queries` and "
            "`ticket_queries`; leave legacy `queries` empty. Each query must cover "
            "a distinct evidence need grounded in the original wording."
            if structured
            else
            "Legacy mode: fill only `queries` with alternative domain/symptom "
            "wording; leave source-specific lists empty."
        )
        try:
            plan = self.multi_query_chain.invoke({
                "input": query,
                "module_context": module_context or "No module was detected.",
                "planner_instructions": planner_instructions,
                "max_queries": legacy_max,
                "max_documentation_queries": docs_max,
                "max_ticket_queries": tickets_max,
            })
        except Exception as error:
            log.warning("Multi-Query generation failed; using original search only: %s", error)
            return [retrieval_query], [retrieval_query]

        if structured:
            doc_variants = getattr(plan, "documentation_queries", None) or []
            ticket_variants = getattr(plan, "ticket_queries", None) or []
            legacy_variants = getattr(plan, "queries", None) or []
            if not doc_variants and not ticket_variants and legacy_variants:
                log.warning("Structured planner returned legacy queries; using them for both sources")
                doc_variants = ticket_variants = legacy_variants
            doc_queries = self._prepare_query_variants(
                query,
                retrieval_query,
                doc_variants,
                detected_modules,
                docs_max,
            )
            ticket_queries = self._prepare_query_variants(
                query,
                retrieval_query,
                ticket_variants,
                detected_modules,
                tickets_max,
            )
            log.info(
                "Structured Query Planner: docs=%s tickets=%s entities=%s",
                len(doc_queries),
                len(ticket_queries),
                len(getattr(plan, "entities", None) or []),
            )
            return doc_queries, ticket_queries

        shared_queries = self._prepare_query_variants(
            query,
            retrieval_query,
            getattr(plan, "queries", None) or [],
            detected_modules,
            legacy_max,
        )
        if len(shared_queries) > 1:
            log.info("Multi-Query RAG-Fusion: %s search variants", len(shared_queries))
        return shared_queries, list(shared_queries)
    def _retrieve_docs_for_queries(
        self,
        queries: List[str],
        retrieval_query: str,
        *,
        use_reranker: bool,
        candidate_filter: Callable[[List[Any]], List[Any]] | None = None,
    ) -> List:
        if len(queries) <= 1:
            return self.retriever.retrieve_docs(
                retrieval_query,
                use_reranker=use_reranker,
                candidate_filter=candidate_filter,
            )
        rrf_k, candidate_limit = self._multi_query_options()
        return self.retriever.retrieve_docs_multi_query(
            queries,
            rerank_query=retrieval_query,
            use_reranker=use_reranker,
            rrf_k=rrf_k,
            candidate_limit=candidate_limit,
            candidate_filter=candidate_filter,
        )

    def _retrieve_tickets_for_queries(
        self,
        queries: List[str],
        retrieval_query: str,
        *,
        final_limit: int | None,
        child_k: int | None,
        use_reranker: bool,
        candidate_filter: Callable[[List[Any]], List[Any]] | None = None,
    ) -> List:
        if len(queries) <= 1:
            return self.retriever.retrieve_tickets(
                retrieval_query,
                final_limit=final_limit,
                child_k=child_k,
                use_reranker=use_reranker,
                candidate_filter=candidate_filter,
            )
        rrf_k, candidate_limit = self._multi_query_options()
        return self.retriever.retrieve_tickets_multi_query(
            queries,
            rerank_query=retrieval_query,
            final_limit=final_limit,
            child_k=child_k,
            use_reranker=use_reranker,
            rrf_k=rrf_k,
            candidate_limit=max(candidate_limit, child_k or 0),
            candidate_filter=candidate_filter,
        )
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

            detected_modules = detect_modules_in_query(query)
            module_context = format_detected_modules(detected_modules)
            retrieval_query = build_module_enriched_query(query, detected_modules)
            if detected_modules:
                log.info(
                    "Detected modules in query: %s",
                    ", ".join(module.get("canonical", "") for module in detected_modules),
                )

            exact_doc_titles = module_doc_page_title_hints(detected_modules)
            exact_module_docs = (
                self.retriever.retrieve_docs_by_page_titles(exact_doc_titles, limit=20)
                if exact_doc_titles and hasattr(self.retriever, "retrieve_docs_by_page_titles")
                else []
            )
            if exact_module_docs:
                log.info(
                    "Added exact module documentation sections: %s",
                    ", ".join(doc.metadata.get("page_title", "") for doc in exact_module_docs[:3]),
                )
            adaptive_mode = self.profile == "adaptive"
            adaptive_incident = adaptive_mode and self._is_incident_query(query)
            doc_retrieval_queries, ticket_retrieval_queries = self._build_multi_query_searches(
                query,
                retrieval_query,
                detected_modules,
                module_context,
            )
            series_candidate_filter = lambda candidates: filter_documents_by_requested_series(
                query,
                candidates,
            )
            docs = merge_documents(
                exact_module_docs,
                self._retrieve_docs_for_queries(
                    doc_retrieval_queries,
                    retrieval_query,
                    use_reranker=not adaptive_mode,
                    candidate_filter=series_candidate_filter,
                ),
            )
            if (
                adaptive_mode
                and self.retriever.reranker_model is not None
                and self._adaptive_doc_results_are_weak(
                    docs,
                    query_count=len(doc_retrieval_queries),
                )
            ):
                log.info(
                    "Adaptive documentation pool is weak or collapsed; enabling reranker"
                )
                docs = merge_documents(
                    exact_module_docs,
                    self._retrieve_docs_for_queries(
                        doc_retrieval_queries,
                        retrieval_query,
                        use_reranker=True,
                        candidate_filter=series_candidate_filter,
                    ),
                )
            if settings.second_db_name == settings.active_db_name:
                log.warning(
                    "Ticket retrieval disabled: second_db points to the docs backend (%s)",
                    settings.second_db_name,
                )
                tickets = []
            else:
                module_codes = [
                    str(module.get("module_code") or "").strip()
                    for module in detected_modules
                    if str(module.get("module_code") or "").strip()
                ]
                exact_module_tickets = (
                    self.retriever.retrieve_tickets_by_module_codes(module_codes, limit=60)
                    if detected_modules and hasattr(self.retriever, "retrieve_tickets_by_module_codes")
                    else []
                )
                ticket_quality_mode = (
                    self.profile in {"deep", "chat_deep"}
                    and self.retriever.reranker_model is not None
                )
                initial_ticket_k = 30 if adaptive_incident else (80 if ticket_quality_mode or detected_modules else None)
                initial_ticket_limit = 4 if adaptive_mode else (8 if ticket_quality_mode else (12 if detected_modules else 2))
                ticket_candidates = [
                    doc for doc in self._retrieve_tickets_for_queries(
                        ticket_retrieval_queries,
                        retrieval_query,
                        final_limit=initial_ticket_limit,
                        child_k=initial_ticket_k,
                        use_reranker=not adaptive_mode or adaptive_incident,
                        candidate_filter=series_candidate_filter,
                    )
                    if is_ticket_document(doc)
                ]
                if adaptive_incident and self._adaptive_ticket_results_are_weak(ticket_candidates):
                    log.info("Adaptive ticket search is weak; expanding rerank pool from 30 to 80 chunks")
                    ticket_candidates = [
                        doc for doc in self._retrieve_tickets_for_queries(
                            ticket_retrieval_queries,
                            retrieval_query,
                            final_limit=8,
                            child_k=80,
                            use_reranker=True,
                            candidate_filter=series_candidate_filter,
                        )
                        if is_ticket_document(doc)
                    ]
                merged_ticket_candidates: List[Any] = []
                seen_ticket_ids = set()
                for doc in [*exact_module_tickets, *ticket_candidates]:
                    if not is_ticket_document(doc):
                        continue
                    ticket_id = doc.metadata.get("ticket_id") if hasattr(doc, "metadata") else None
                    key = ticket_id or id(doc)
                    if key in seen_ticket_ids:
                        continue
                    seen_ticket_ids.add(key)
                    merged_ticket_candidates.append(doc)
                tickets = rank_tickets_for_modules(
                    merged_ticket_candidates,
                    detected_modules,
                    query=query,
                    limit=4 if (ticket_quality_mode or adaptive_mode) else 2,
                )

            docs = filter_documents_by_requested_series(query, docs)
            tickets = filter_documents_by_requested_series(query, tickets)

            # Remove incompatible product-family sources before they reach the LLM. The guard remains as a diagnostic fallback.
            docs = self._filter_wiki_docs_by_requested_product(query, detected_modules, docs)
            tickets = self._filter_wiki_docs_by_requested_product(query, detected_modules, tickets)
            doc_sources = source_references(docs, prefix="D")
            ticket_sources = source_references(tickets, ticket_only=True, prefix="T")
            docs_context = format_docs(
                docs,
                sources=doc_sources,
                max_total_chars=SUPPORT_DOCS_MAX_CHARS,
                max_doc_chars=SUPPORT_DOC_MAX_CHARS,
            )
            tickets_context = format_ticket_docs(
                tickets,
                sources=ticket_sources,
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

            diagnostic_plan_context = "No diagnostic planning was required."
            if settings.diagnostic_sgr_enabled and self._is_incident_query(query):
                try:
                    reflection = self.retrieval_reflection_chain.invoke({
                        "input": query,
                        "chat_history": "No previous chat history.",
                        "module_context": module_context or "No module was detected.",
                        "already_tried_queries": "\n".join([
                            *(f"docs: {item}" for item in doc_retrieval_queries),
                            *(f"tickets: {item}" for item in ticket_retrieval_queries),
                        ]),
                        "docs_context": docs_context[:WIKI_REFLECTION_DOCS_MAX_CHARS],
                        "tickets_context": tickets_context[:WIKI_REFLECTION_TICKETS_MAX_CHARS],
                    })
                    mode = "direct documented answer" if reflection.enough_context else "evidence-bounded diagnostic hypotheses"
                    diagnostic_plan_context = (
                        f"Diagnostic SGR mode: {mode}.\n"
                        f"Missing facts: {'; '.join(reflection.missing_facts[:3]) or 'none recorded'}.\n"
                        f"Source risks: {'; '.join(reflection.source_risks[:3]) or 'none recorded'}.\n"
                        "Separate direct evidence from hypotheses; do not add generic checklists."
                    )
                except Exception as error:
                    log.warning("Diagnostic SGR failed; continuing with initial evidence: %s", error)

            response = self.support_chain.invoke({
                "input": query,
                "strict_evidence_context": strict_evidence_context(query) + "\n\n" + diagnostic_plan_context,
                "equipment_mismatch_context": check_and_format_equipment_mismatch_warning(query, doc_sources),
                "module_context": module_context,
                "docs_context": docs_context,
                "tickets_context": tickets_context,
                "doc_sources_context": format_source_references(doc_sources),
                "ticket_sources_context": format_source_references(ticket_sources),
            })
            response = apply_definition_guard(response, query, docs_context, doc_sources)
            response = apply_transformation_evidence_guard(response, query, docs_context, doc_sources)
            response = apply_entity_coverage_guard(response, query, docs_context)
            response = apply_diagnostic_scope_guard(
                response,
                query,
                docs_context,
                information_mode=adaptive_mode,
            )
            response, doc_sources, ticket_sources = apply_response_provenance(
                response, doc_sources, ticket_sources
            )
            if detected_modules and not adaptive_mode:
                response.docs_answer = ensure_module_block(response.docs_answer, detected_modules)
                response.draft_private_comment = ensure_module_block(response.draft_private_comment, detected_modules)
            response.doc_sources = doc_sources
            response.ticket_sources = ticket_sources
            response.evidence_notes = [
                note for note in response.evidence_notes
                if note.claim.strip() and note.source_ids
            ]
            if not tickets:
                response.similar_tickets = []
            else:
                response.similar_tickets = [
                    ticket for ticket in response.similar_tickets
                    if (ticket.relevance_reason or "").strip()
                ]
            if adaptive_mode:
                response = apply_adaptive_information_role(response)

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

    def _normalize_followup_queries(self, queries: Any, already_tried: set, limit: int) -> List[str]:
        """Return short unique follow-up retrieval queries that were not tried yet."""
        if not queries:
            return []
        if isinstance(queries, str):
            raw_queries = [queries]
        else:
            raw_queries = list(queries)

        normalized: List[str] = []
        for raw_query in raw_queries:
            query = " ".join(str(raw_query or "").split())
            if not query:
                continue
            if len(query) > 220:
                query = query[:220].rsplit(" ", 1)[0] or query[:220]
            key = query.lower()
            if key in already_tried:
                continue
            already_tried.add(key)
            normalized.append(query)
            if len(normalized) >= limit:
                break
        return normalized

    def _merge_ticket_documents(self, current: List, incoming: List) -> List:
        """Merge ticket documents by ticket_id while keeping retrieval order stable."""
        merged: List = []
        seen = set()
        for doc in [*(current or []), *(incoming or [])]:
            if not is_ticket_document(doc):
                continue
            metadata = getattr(doc, "metadata", {}) or {}
            ticket_id = metadata.get("ticket_id")
            key = ("ticket", str(ticket_id)) if ticket_id else ("object", id(doc))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return merged

    def _filter_wiki_docs_by_requested_product(
        self,
        latest_query: str,
        detected_modules: List[Dict[str, Any]],
        docs: List,
    ) -> List:
        """Keep explicitly requested product families separated in wiki chat output."""
        if not docs:
            return docs

        query_lower = (latest_query or "").lower()
        query_mentions_r500s = "r500s" in query_lower
        query_mentions_r500 = bool(re.search(r"(?<![a-z0-9])r500(?!s)(?![a-z0-9])", query_lower))
        if query_mentions_r500 and query_mentions_r500s:
            return docs

        module_families = {
            str(module.get("product_family") or "").strip().lower()
            for module in detected_modules or []
            if str(module.get("product_family") or "").strip()
        }
        module_products = set()
        for family in module_families:
            if "r500s" in family:
                module_products.add("r500s")
            elif re.search(r"(?<![a-z0-9])r500(?!s)(?![a-z0-9])", family):
                module_products.add("r500")

        requested_product = ""
        if query_mentions_r500s:
            requested_product = "r500s"
        elif query_mentions_r500:
            requested_product = "r500"
        elif len(module_products) == 1:
            requested_product = next(iter(module_products))

        if requested_product not in {"r500", "r500s"}:
            return docs

        filtered: List = []
        dropped = 0
        for doc in docs:
            metadata = getattr(doc, "metadata", {}) or {}
            equipment = str(metadata.get("equipment_type") or "").lower()
            source_file = str(metadata.get("source_file") or "").lower()
            source_text = f"{equipment} {source_file}"
            is_r500s_source = "r500s" in source_text
            is_r500_source = bool(re.search(r"(?<![a-z0-9])r500(?!s)(?![a-z0-9])", source_text))

            if requested_product == "r500" and is_r500s_source:
                dropped += 1
                continue
            if requested_product == "r500s" and is_r500_source:
                dropped += 1
                continue
            filtered.append(doc)

        if dropped:
            log.info(
                "Wiki product filter removed %d cross-product docs for requested_product=%s",
                dropped,
                requested_product.upper(),
            )
        return filtered

    def _should_expand_wiki_context(self, latest_query: str, docs: List, tickets: List) -> bool:
        """Skip the extra reflection LLM call when the first-pass context is already strong."""
        if self.profile != "chat_deep":
            return False
        if not docs:
            return True

        query_lower = (latest_query or "").lower()
        high_value_markers = (
            "ticket",
            "r500 and r500s",
            "r500s and r500",
            "r500 и r500s",
            "r500s и r500",
            "тикет",
            "обращен",
            "похож",
            "решал",
            "решили",
            "истор",
            "сравн",
            "отлич",
            "разниц",
        )
        if any(marker in query_lower for marker in high_value_markers):
            return True

        doc_chars = sum(len(getattr(doc, "page_content", "") or "") for doc in docs[:WIKI_ITERATIVE_SKIP_MIN_DOCS])
        if len(docs) < WIKI_ITERATIVE_SKIP_MIN_DOCS or doc_chars < WIKI_ITERATIVE_SKIP_MIN_DOC_CHARS:
            return True

        log.info(
            "Wiki iterative retrieval skipped: first-pass context is sufficient (docs=%d, chars=%d, tickets=%d)",
            len(docs),
            doc_chars,
            len(tickets or []),
        )
        return False

    def _expand_wiki_context_iteratively(
        self,
        latest_query: str,
        history_text: str,
        module_context: str,
        detected_modules: List[Dict[str, Any]],
        retrieval_query: str,
        docs: List,
        tickets: List,
    ):
        """Let chat_deep request a small number of extra vector searches before answering."""
        if self.profile != "chat_deep":
            return docs, tickets, []

        reflection_notes: List[str] = []
        already_tried = {retrieval_query.lower(), latest_query.lower()}
        tickets_enabled = settings.second_db_name != settings.active_db_name

        for round_index in range(WIKI_ITERATIVE_MAX_ROUNDS):
            try:
                # Remove incompatible product-family sources before they reach the LLM. The guard remains as a diagnostic fallback.

                docs = self._filter_wiki_docs_by_requested_product(latest_query, detected_modules, docs)

                tickets = self._filter_wiki_docs_by_requested_product(latest_query, detected_modules, tickets)



                doc_sources = source_references(docs, prefix="D")
                ticket_sources = source_references(tickets, ticket_only=True, prefix="T")
                docs_context = format_docs(
                    docs,
                    sources=doc_sources,
                    max_total_chars=WIKI_REFLECTION_DOCS_MAX_CHARS,
                    max_doc_chars=900,
                )
                tickets_context = format_ticket_docs(
                    tickets,
                    sources=ticket_sources,
                    max_total_chars=WIKI_REFLECTION_TICKETS_MAX_CHARS,
                    max_doc_chars=700,
                )

                reflection = self.retrieval_reflection_chain.invoke({
                    "input": latest_query,
                    "chat_history": history_text or "No previous chat history.",
                    "module_context": module_context,
                    "already_tried_queries": "\n".join(sorted(already_tried)),
                    "docs_context": docs_context,
                    "tickets_context": tickets_context,
                })

                if reflection.source_risks:
                    reflection_notes.extend(reflection.source_risks)
                if reflection.missing_facts:
                    reflection_notes.extend(reflection.missing_facts)

                if reflection.enough_context:
                    log.info("Wiki iterative retrieval stopped at round %d: enough context", round_index + 1)
                    break

                remaining = WIKI_ITERATIVE_MAX_FOLLOWUP_QUERIES
                doc_queries = self._normalize_followup_queries(
                    reflection.followup_doc_queries,
                    already_tried,
                    remaining,
                )
                remaining -= len(doc_queries)
                ticket_queries = self._normalize_followup_queries(
                    reflection.followup_ticket_queries if tickets_enabled else [],
                    already_tried,
                    remaining,
                )

                if not doc_queries and not ticket_queries:
                    log.info("Wiki iterative retrieval stopped at round %d: no new queries", round_index + 1)
                    break

                docs_before = len(docs)
                tickets_before = len(tickets)
                for query in doc_queries:
                    focused_query = build_module_enriched_query(query, detected_modules)
                    docs = merge_documents(docs, self.retriever.retrieve_docs(focused_query))

                if tickets_enabled:
                    ticket_candidates: List = []
                    for query in ticket_queries:
                        focused_query = build_module_enriched_query(query, detected_modules)
                        ticket_candidates.extend([
                            doc for doc in self.retriever.retrieve_tickets(
                                focused_query,
                                final_limit=8 if detected_modules else 4,
                                child_k=60 if detected_modules else None,
                            )
                            if is_ticket_document(doc)
                        ])
                    tickets = self._merge_ticket_documents(tickets, ticket_candidates)
                    tickets = rank_tickets_for_modules(
                        tickets,
                        detected_modules,
                        query=latest_query,
                        limit=4,
                    )

                log.info(
                    "Wiki iterative retrieval round %d: +%d docs, +%d tickets",
                    round_index + 1,
                    max(0, len(docs) - docs_before),
                    max(0, len(tickets) - tickets_before),
                )
                if len(docs) == docs_before and len(tickets) == tickets_before:
                    break
            except Exception as exc:
                log.warning("Wiki iterative retrieval failed, continuing with current context: %s", exc, exc_info=True)
                break

        deduped_notes: List[str] = []
        seen_notes = set()
        for note in reflection_notes:
            note_text = str(note or "").strip()
            if not note_text or note_text in seen_notes:
                continue
            seen_notes.add(note_text)
            deduped_notes.append(note_text)
        return docs, tickets, deduped_notes


    def process_wiki_chat_dialog(
        self,
        messages: List[Dict[str, str]],
        max_history_turns: int = 2,
    ) -> WikiChatResponse:
        """Process an OpenWebUI dialog in wiki mode: docs first, tickets as history."""
        if not messages:
            raise ValueError("messages cannot be empty")

        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise ValueError("user messages were not found")

        latest_query = user_messages[-1].get("content", "").strip()
        if not latest_query:
            raise ValueError("last user message is empty")

        start = time.perf_counter()
        docs: List = []
        tickets: List = []

        # Deliberately ignore all earlier OpenWebUI messages. This makes each
        # request reproducible and prevents prior questions from changing
        # module detection, retrieval, or the final answer.
        history_text = ""
        module_detection_query = latest_query
        try:
            log.info(
                "Wiki chat: processing query (%s chars, profile=%s)",
                len(latest_query),
                self.profile,
            )

            detected_modules = detect_modules_in_query(module_detection_query)
            module_context = format_detected_modules(detected_modules)
            retrieval_query = build_module_enriched_query(latest_query, detected_modules)
            if detected_modules:
                log.info(
                    "Wiki chat detected modules: %s",
                    ", ".join(module.get("canonical", "") for module in detected_modules),
                )

            exact_doc_titles = module_doc_page_title_hints(detected_modules)
            exact_module_docs = (
                self.retriever.retrieve_docs_by_page_titles(exact_doc_titles, limit=20)
                if exact_doc_titles and hasattr(self.retriever, "retrieve_docs_by_page_titles")
                else []
            )
            docs = merge_documents(exact_module_docs, self.retriever.retrieve_docs(retrieval_query))

            if settings.second_db_name == settings.active_db_name:
                log.warning(
                    "Wiki ticket retrieval disabled: second_db points to the docs backend (%s)",
                    settings.second_db_name,
                )
                tickets = []
            else:
                module_codes = [
                    str(module.get("module_code") or "").strip()
                    for module in detected_modules
                    if str(module.get("module_code") or "").strip()
                ]
                exact_module_tickets = (
                    self.retriever.retrieve_tickets_by_module_codes(module_codes, limit=60)
                    if detected_modules and hasattr(self.retriever, "retrieve_tickets_by_module_codes")
                    else []
                )
                ticket_quality_mode = (
                    self.profile in {"deep", "chat_deep"}
                    and self.retriever.reranker_model is not None
                )
                ticket_candidates = [
                    doc for doc in self.retriever.retrieve_tickets(
                        retrieval_query,
                        final_limit=16 if detected_modules else (8 if ticket_quality_mode else 4),
                        child_k=80 if (ticket_quality_mode or detected_modules) else None,
                    )
                    if is_ticket_document(doc)
                ]
                merged_ticket_candidates: List[Any] = []
                seen_ticket_ids = set()
                for doc in [*exact_module_tickets, *ticket_candidates]:
                    if not is_ticket_document(doc):
                        continue
                    ticket_id = doc.metadata.get("ticket_id") if hasattr(doc, "metadata") else None
                    key = ticket_id or id(doc)
                    if key in seen_ticket_ids:
                        continue
                    seen_ticket_ids.add(key)
                    merged_ticket_candidates.append(doc)
                tickets = rank_tickets_for_modules(
                    merged_ticket_candidates,
                    detected_modules,
                    query=latest_query,
                    limit=4,
                )

            docs = self._filter_wiki_docs_by_requested_product(latest_query, detected_modules, docs)

            reflection_notes: List[str] = []
            if self._should_expand_wiki_context(latest_query, docs, tickets):
                docs, tickets, reflection_notes = self._expand_wiki_context_iteratively(
                    latest_query=latest_query,
                    history_text=history_text,
                    module_context=module_context,
                    detected_modules=detected_modules,
                    retrieval_query=retrieval_query,
                    docs=docs,
                    tickets=tickets,
                )
                docs = self._filter_wiki_docs_by_requested_product(latest_query, detected_modules, docs)

            # Remove incompatible product-family sources before they reach the LLM. The guard remains as a diagnostic fallback.


            docs = self._filter_wiki_docs_by_requested_product(latest_query, detected_modules, docs)


            tickets = self._filter_wiki_docs_by_requested_product(latest_query, detected_modules, tickets)





            doc_sources = source_references(docs, prefix="D")
            ticket_sources = source_references(tickets, ticket_only=True, prefix="T")
            docs_context = format_docs(
                docs,
                sources=doc_sources,
                max_total_chars=WIKI_DOCS_MAX_CHARS,
                max_doc_chars=WIKI_DOC_MAX_CHARS,
            )
            tickets_context = format_ticket_docs(
                tickets,
                sources=ticket_sources,
                max_total_chars=WIKI_TICKETS_MAX_CHARS,
                max_doc_chars=WIKI_TICKET_MAX_CHARS,
            )

            response = self.chat_chain.invoke({
                "input": latest_query,
                "chat_history": history_text or "No previous chat history.",
                "strict_evidence_context": strict_evidence_context(latest_query),
                "equipment_mismatch_context": check_and_format_equipment_mismatch_warning(latest_query, doc_sources),
                "module_context": module_context,
                "docs_context": docs_context,
                "tickets_context": tickets_context,
                "doc_sources_context": format_source_references(doc_sources),
                "ticket_sources_context": format_source_references(ticket_sources),
            })

            if detected_modules:
                response.docs_answer = ensure_module_block(response.docs_answer, detected_modules)

            response.final_answer = apply_entity_citation_guard(response.final_answer, latest_query, doc_sources)
            response.doc_sources = doc_sources
            response.ticket_sources = ticket_sources
            if reflection_notes:
                current_limitations = list(response.source_limitations or [])
                seen_limitations = {str(item).strip() for item in current_limitations if str(item).strip()}
                for note in reflection_notes:
                    note_text = str(note or "").strip()
                    if note_text and note_text not in seen_limitations:
                        current_limitations.append(note_text)
                        seen_limitations.add(note_text)
                response.source_limitations = current_limitations
            response.evidence_notes = [
                note for note in response.evidence_notes
                if note.claim.strip() and note.source_ids
            ]
            if not tickets:
                response.similar_tickets = []
            else:
                response.similar_tickets = [
                    ticket for ticket in response.similar_tickets
                    if (ticket.relevance_reason or "").strip()
                    and ((ticket.problem_summary or "").strip() or (ticket.solution_summary or "").strip())
                ]
            if not response.final_answer.strip():
                response.final_answer = response.docs_answer.strip() or "No direct confirmation was found in retrieved sources."

            elapsed = time.perf_counter() - start
            log.info(
                "Wiki chat response generated in %.2fs (profile=%s, docs=%d, tickets=%d)",
                elapsed,
                self.profile,
                len(docs),
                len(tickets),
            )
            log_query_audit(
                query=latest_query,
                equipment_filter=None,
                retrieved_docs=docs + tickets,
                response=response,
                elapsed_sec=elapsed,
            )
            return response

        except Exception as e:
            elapsed = time.perf_counter() - start
            log.error("Wiki chat failed after %.2fs: %s", elapsed, e, exc_info=True)
            log_query_audit(
                query=latest_query,
                equipment_filter=None,
                retrieved_docs=docs + tickets,
                response=None,
                elapsed_sec=elapsed,
                extra={"error": str(e)},
            )
            raise RuntimeError(f"Failed to process wiki chat: {str(e)}") from e


    def process_chat_dialog(
        self,
        messages: List[Dict[str, str]],
        max_history_turns: int = 2,
    ) -> SupportPrivateResponse:
        """Обрабатывает сообщения чата (OpenWebUI) через проверенный 100% точный SGR-движок process_support_ticket."""
        if not messages:
            raise ValueError("messages не может быть пустым")

        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            raise ValueError("Пользовательские сообщения не найдены")

        latest_query = user_messages[-1].get("content", "").strip()
        if not latest_query:
            raise ValueError("Последний вопрос пользователя пуст")

        start = time.perf_counter()

        result = self.process_support_ticket(latest_query)
        elapsed = time.perf_counter() - start
        log.info(
            "Precision SGR chat response generated in %.2fs (profile=%s, user_msgs=%d)",
            elapsed,
            self.profile,
            len(user_messages),
        )

        return result
