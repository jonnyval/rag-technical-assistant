"""FastAPI server for the RegLab RAG support-ticket workflow.

The module exposes two integration surfaces:
- /api/v1/analyze_ticket for the support portal.
- /v1/models and /v1/chat/completions for OpenAI-compatible clients.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
import uvicorn

# Allow running this script directly: python scripts/api_server.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import RAGEngine  # noqa: E402
from src.logger import log  # noqa: E402
from src.context_formatting import format_chat_sources_footer  # noqa: E402
try:
    from faq_pipeline.search.ticket_vector_search import (  # noqa: E402
        DEFAULT_INDEX_DIR,
        TicketSearchConfig,
        TicketVectorSearch,
        preview as ticket_preview,
    )
except ImportError:
    DEFAULT_INDEX_DIR = None
    TicketSearchConfig = None
    TicketVectorSearch = None

    def ticket_preview(items: Any, *, limit: int = 3, max_len: int = 520) -> str:
        values = [str(item).strip() for item in (items or []) if str(item).strip()]
        return "; ".join(values[:limit])[:max_len]


# Kept for backwards-compatible code paths, but not exposed by /v1/models.
MODEL_ID = "reglab-ai"
TICKET_SEARCH_MODEL_ID = "reglab-ticket-search"
EVA_ARTICLE_MODEL_ID = "reglab-eva-article"
CHAT_MODEL_ID = "reglab-ai-chat"
CHAT_DEEP_MODEL_ID = "reglab-ai-chat-deep"
DEEP_MODEL_ID = "reglab-ai-deep"
ADAPTIVE_MODEL_ID = "reglab-ai-adaptive"
MODEL_PROFILES = {
    DEEP_MODEL_ID: "deep",
    ADAPTIVE_MODEL_ID: "adaptive",
}
API_VERSION = "1.1"
DEFAULT_MAX_CONCURRENCY = 2


class SimilarTicketResponse(BaseModel):
    """Short description of a similar historical support ticket."""

    ticket_id: str
    source_file: str
    problem_summary: str
    solution_summary: str
    relevance_reason: str
    ticket_url: str = ""


class SourceResponse(BaseModel):
    """Source reference returned with the generated answer."""

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


class EvidenceNoteResponse(BaseModel):
    """Verifiable answer claim with source references."""

    claim: str
    source_ids: List[str] = Field(default_factory=list)
    comment: str = ""


class TicketRequest(BaseModel):
    """Support portal request for a private AI hint."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ticket_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    equipment: Optional[List[str]] = None


class TicketResponse(BaseModel):
    """Private support hint returned to the portal."""

    ticket_id: str
    user_intent: str
    docs_answer: str
    related_topics: List[str]
    similar_tickets: List[SimilarTicketResponse]
    evidence_notes: List[EvidenceNoteResponse] = Field(default_factory=list)
    recommended_questions: List[str] = Field(default_factory=list)
    internal_notes: List[str] = Field(default_factory=list)
    doc_sources: List[SourceResponse] = Field(default_factory=list)
    ticket_sources: List[SourceResponse] = Field(default_factory=list)
    missing_context: str
    draft_private_comment: str
    confidence: str

    # Backward-compatible fields for older clients.
    final_answer: str
    extracted_facts: List[str]


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    role: Literal["user", "assistant", "system"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI chat completions request used by Open WebUI."""

    model_config = ConfigDict(extra="ignore")

    model: str = DEEP_MODEL_ID
    messages: List[ChatMessage] = Field(..., min_length=1)
    stream: bool = False


class ErrorResponse(BaseModel):
    """Stable error payload for API clients."""

    error: str
    request_id: str


def _max_concurrency() -> int:
    raw_value = os.getenv("RAG_MAX_CONCURRENCY", str(DEFAULT_MAX_CONCURRENCY))
    try:
        return max(1, int(raw_value))
    except ValueError:
        log.warning("Invalid RAG_MAX_CONCURRENCY=%r, using %s", raw_value, DEFAULT_MAX_CONCURRENCY)
        return DEFAULT_MAX_CONCURRENCY


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_engine = None
    app.state.rag_engines = {}
    app.state.ticket_search = None
    app.state.ticket_search_lock = asyncio.Lock()
    app.state.engine_locks = {
        profile: asyncio.Lock()
        for profile in set(MODEL_PROFILES.values())
    }
    app.state.startup_error = None
    app.state.inference_semaphore = asyncio.Semaphore(_max_concurrency())

    try:
        log.info("Starting RegLab RAG API: initializing default deep RAGEngine...")
        app.state.rag_engine = await run_in_threadpool(RAGEngine, "deep")
        app.state.rag_engines["deep"] = app.state.rag_engine
        log.info("RegLab RAG API is ready.")
    except Exception as exc:
        app.state.startup_error = str(exc)
        log.error("RAGEngine initialization failed: %s", exc, exc_info=True)

    yield

    app.state.rag_engine = None
    app.state.rag_engines = {}
    app.state.ticket_search = None
    log.info("RegLab RAG API stopped.")


app = FastAPI(title="RegLab RAG API", version=API_VERSION, lifespan=lifespan)


def _request_id() -> str:
    return uuid4().hex


def _get_engine(request: Request) -> RAGEngine:
    engine = getattr(request.app.state, "rag_engine", None)
    if engine is None:
        detail = getattr(request.app.state, "startup_error", None) or "RAG engine is not initialized"
        log.warning("RAG engine is unavailable: %s", detail)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine is not ready",
        )
    return engine


def _profile_from_model(model_id: str) -> str:
    return MODEL_PROFILES.get(model_id, "deep")


async def _get_engine_for_model(request: Request, model_id: str) -> RAGEngine:
    profile = _profile_from_model(model_id)
    engines: Dict[str, RAGEngine] = getattr(request.app.state, "rag_engines", {})
    engine = engines.get(profile)
    if engine is not None:
        return engine

    locks: Dict[str, asyncio.Lock] = getattr(request.app.state, "engine_locks", {})
    lock = locks.setdefault(profile, asyncio.Lock())
    async with lock:
        engine = engines.get(profile)
        if engine is None:
            log.info("Lazy initialization of RAGEngine profile=%s for model=%s", profile, model_id)
            deep_engine = engines.get("deep")
            shared_embeddings = getattr(deep_engine, "dense_embeddings", None) if deep_engine else None
            shared_reranker = getattr(deep_engine, "rerank_model", None) if deep_engine else None
            if shared_embeddings is not None:
                log.info("Reusing embedding model and reranker from deep profile for profile=%s", profile)
            engine = await run_in_threadpool(
                RAGEngine,
                profile,
                shared_embeddings=shared_embeddings,
                shared_reranker=shared_reranker,
            )
            engines[profile] = engine
            if profile == "deep":
                request.app.state.rag_engine = engine
        return engine


async def _get_ticket_search(request: Request) -> Any:
    if TicketVectorSearch is None or DEFAULT_INDEX_DIR is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ticket vector search is not available in this RAG deployment",
        )

    searcher = getattr(request.app.state, "ticket_search", None)
    if searcher is not None:
        return searcher

    lock: asyncio.Lock = getattr(request.app.state, "ticket_search_lock")
    async with lock:
        searcher = getattr(request.app.state, "ticket_search", None)
        if searcher is None:
            log.info("Lazy initialization of ticket vector search index=%s", DEFAULT_INDEX_DIR)
            searcher = TicketVectorSearch(DEFAULT_INDEX_DIR)
            await run_in_threadpool(searcher.load)
            request.app.state.ticket_search = searcher
        return searcher


def _build_ticket_query(text: str, equipment: Optional[List[str]] = None) -> str:
    cleaned_equipment = [item.strip() for item in equipment or [] if item and item.strip()]
    if not cleaned_equipment:
        return text.strip()
    return f"Equipment: {', '.join(cleaned_equipment)}\n\nTicket text:\n{text.strip()}"


def _ticket_to_response(ticket: Any) -> SimilarTicketResponse:
    if hasattr(ticket, "model_dump"):
        return SimilarTicketResponse(**ticket.model_dump())
    if hasattr(ticket, "dict"):
        return SimilarTicketResponse(**ticket.dict())
    return SimilarTicketResponse(**dict(ticket))


def _source_to_response(source: Any) -> SourceResponse:
    if hasattr(source, "model_dump"):
        return SourceResponse(**source.model_dump())
    if hasattr(source, "dict"):
        return SourceResponse(**source.dict())
    return SourceResponse(**dict(source))


def _evidence_to_response(note: Any) -> EvidenceNoteResponse:
    if hasattr(note, "model_dump"):
        return EvidenceNoteResponse(**note.model_dump())
    if hasattr(note, "dict"):
        return EvidenceNoteResponse(**note.dict())
    return EvidenceNoteResponse(**dict(note))


def _format_list(title: str, items: List[str]) -> str:
    if not items:
        return ""
    lines = "\n".join(f"- {item}" for item in items)
    return f"\n\n**{title}:**\n{lines}"


def _to_mapping(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return vars(value)


def _ticket_has_content(ticket: Any) -> bool:
    problem = getattr(ticket, "problem_summary", "") or ""
    solution = getattr(ticket, "solution_summary", "") or ""
    return bool(problem.strip() or solution.strip())


def _format_similar_ticket(ticket: Any) -> str:
    parts = [f"- **{getattr(ticket, 'ticket_id', '')}**"]
    problem = (getattr(ticket, "problem_summary", "") or "").strip()
    solution = (getattr(ticket, "solution_summary", "") or "").strip()
    relevance = (getattr(ticket, "relevance_reason", "") or "").strip()
    if problem:
        parts.append(f": {problem}")
    if relevance:
        parts.append(f"\n  - Почему похоже: {relevance}")
    if solution:
        parts.append(f"\n  - Что было сделано: {solution}")
    return "".join(parts)


def _format_docs_answer_text(text: Any) -> str:
    value = (
        str(text or "")
        .strip()
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
    )
    if not value:
        return ""

    numbered_items = re.findall(r"(?:^|\s|[;:])\d+\)", value)
    if len(numbered_items) < 2:
        return value

    value = re.sub(r":\s*(1\))", r":\n\1", value)
    value = re.sub(r";\s*(?=\d+\))", "\n", value)
    value = re.sub(r"(?<!\n)\s+(\d+\))", r"\n\1", value)
    value = re.sub(r"(?m)^(\d+)\)\s*", r"\1. ", value)
    return value.strip()

def _format_evidence(notes: List[Any]) -> str:
    if not notes:
        return ""
    lines = []
    for note in notes:
        data = note.model_dump() if hasattr(note, "model_dump") else dict(note)
        claim = data.get("claim") or ""
        source_ids = data.get("source_ids") or []
        comment = data.get("comment") or ""
        suffix = f" [{', '.join(source_ids)}]" if source_ids else ""
        if comment:
            suffix += f" - {comment}"
        lines.append(f"- {claim}{suffix}")
    return "\n\n**Evidence notes:**\n" + "\n".join(lines)


def _strip_wiki_recommendation_sections(text: Any) -> str:
    """Remove support-style recommendation sections from wiki-mode chat output."""
    value = _format_docs_answer_text(text)
    if not value:
        return ""

    recommendation_prefixes = (
        "рекомендац",
        "recommended",
        "recommendations",
        "что делать",
        "что проверить",
        "дальнейшие действия",
    )
    section_prefixes = (
        "что найдено",
        "похожие",
        "что не найдено",
        "not found",
        "documentation sources",
        "ticket sources",
        "источники",
    )

    output_lines = []
    skipping = False
    for line in value.splitlines():
        normalized = line.strip().strip("#* ").lower()
        is_recommendation_heading = any(normalized.startswith(prefix) for prefix in recommendation_prefixes)
        is_next_section = (
            line.strip() == "---"
            or any(normalized.startswith(prefix) for prefix in section_prefixes)
        )
        if is_recommendation_heading:
            skipping = True
            continue
        if skipping and is_next_section:
            skipping = False
        if not skipping:
            output_lines.append(line)

    return "\n".join(output_lines).strip()


def _format_sources(title: str, sources: List[Any]) -> str:
    if not sources:
        return ""
    lines = []
    for source in sources:
        data = source.model_dump() if hasattr(source, "model_dump") else dict(source)
        source_id = data.get("source_id") or ""
        label = data.get("title") or data.get("source_file") or "Источник"
        source_file = data.get("source_file") or ""
        url = data.get("url") or ""
        suffix = f" ({source_file})" if source_file and source_file != label else ""
        prefix = f"[{source_id}] " if source_id else ""
        lines.append(f"- {prefix}{label}{suffix}: {url}" if url else f"- {prefix}{label}{suffix}")
        details = []
        if data.get("equipment_type"):
            details.append(f"Оборудование: {data['equipment_type']}")
        if data.get("library_name"):
            details.append(f"Библиотека: {data['library_name']}")
        if data.get("release_version"):
            details.append(f"Релиз: {data['release_version']}")
        if data.get("breadcrumb"):
            details.append(f"Раздел: {data['breadcrumb']}")
        if details:
            lines.append("  " + " | ".join(details))
    return f"\n\n**{title}:**\n" + "\n".join(lines)


def _format_compact_sources(sources: List[Any], limit: int = 2) -> str:
    """Render only a short source trail for chat; full SGR evidence stays internal."""
    items = []
    for source in sources[:limit]:
        data = _to_mapping(source)
        source_id = str(data.get("source_id") or "").strip()
        title = str(data.get("title") or data.get("source_file") or "Источник").strip()
        if source_id:
            items.append(f"[{source_id}] {title}")
    return "\n\n**Источники:** " + "; ".join(items) if items else ""


def _format_support_answer(result: Any) -> str:
    """Compact engineer-facing answer; structured SGR fields are kept out of chat."""
    answer = _format_docs_answer_text(getattr(result, "draft_private_comment", ""))
    if not answer:
        answer = _format_docs_answer_text(getattr(result, "docs_answer", ""))
    if not answer:
        answer = "В найденных источниках нет подтвержденного ответа на этот вопрос."

    return answer + format_chat_sources_footer(
        getattr(result, "doc_sources", []),
        getattr(result, "ticket_sources", []),
    )

def _format_wiki_chat_answer(result: Any) -> str:
    answer = _strip_wiki_recommendation_sections(getattr(result, "final_answer", ""))
    if not answer:
        answer = _format_docs_answer_text(getattr(result, "docs_answer", ""))
    if not answer:
        answer = "No direct confirmation was found in retrieved sources."

    blocks = [answer]

    tickets = [
        ticket for ticket in getattr(result, "similar_tickets", [])
        if _ticket_has_content(ticket)
    ]
    if tickets and "Similar historical tickets" not in answer and "\u041f\u043e\u0445\u043e\u0436\u0438\u0435 \u0438\u0441\u0442\u043e\u0440\u0438\u0447\u0435\u0441\u043a\u0438\u0435" not in answer:
        ticket_lines = "\n".join(_format_similar_ticket(ticket) for ticket in tickets)
        blocks.append(f"---\n**\u041f\u043e\u0445\u043e\u0436\u0438\u0435 \u0438\u0441\u0442\u043e\u0440\u0438\u0447\u0435\u0441\u043a\u0438\u0435 \u043e\u0431\u0440\u0430\u0449\u0435\u043d\u0438\u044f:**\n{ticket_lines}")

    limitations = [str(item).strip() for item in getattr(result, "source_limitations", []) if str(item).strip()]
    missing_context = (getattr(result, "missing_context", "") or "").strip()
    if (limitations or missing_context) and "Not found in retrieved sources" not in answer and "\u0427\u0442\u043e \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u043e" not in answer:
        lines = []
        if missing_context:
            lines.append(f"- {missing_context}")
        lines.extend(f"- {item}" for item in limitations)
        blocks.append("---\n**\u0427\u0442\u043e \u043d\u0435 \u043d\u0430\u0439\u0434\u0435\u043d\u043e \u0432 \u043f\u043e\u043b\u0443\u0447\u0435\u043d\u043d\u044b\u0445 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0430\u0445:**\n" + "\n".join(lines))

    sources = (
        _format_sources("\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0446\u0438\u0438", getattr(result, "doc_sources", []))
        + _format_sources("\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438 \u043f\u043e\u0445\u043e\u0436\u0438\u0445 \u043e\u0431\u0440\u0430\u0449\u0435\u043d\u0438\u0439", getattr(result, "ticket_sources", []))
    ).strip()
    if sources:
        blocks.append("---\n" + sources)

    return "\n\n".join(block for block in blocks if block.strip())


def _format_ticket_search_answer(query: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        return (
            f"По запросу `{query}` похожие тикеты не найдены.\n\n"
            "Попробуйте ввести точный код ошибки, фрагмент лога или формулировку симптома."
        )

    lines = [
        f"Найдено похожих тикетов: {len(results)}",
        f"Запрос: `{query}`",
        "",
    ]
    for row in results:
        exact = f" | exact: {', '.join(row['exact_matches'])}" if row.get("exact_matches") else ""
        lines.extend(
            [
                f"## {row['rank']}. {row.get('ticket_id') or ''} - {row.get('title') or ''}",
                f"Score: `{row.get('score')}` | word: `{row.get('word_score')}` | char: `{row.get('char_score')}`{exact}",
            ]
        )
        if row.get("ticket_url"):
            lines.append(f"URL: {row['ticket_url']}")
        if row.get("category"):
            lines.append(f"Категория: {row['category']}")
        if row.get("equipment"):
            lines.append(f"Оборудование: {row['equipment']}")

        symptoms = ticket_preview(row.get("symptoms") or [], limit=3, max_len=520)
        solutions = ticket_preview(row.get("solutions") or [], limit=4, max_len=700)
        if symptoms:
            lines.append(f"Симптомы: {symptoms}")
        if solutions:
            lines.append(f"Решение: {solutions}")
        lines.append("")
    return "\n".join(lines).strip()


async def _ticket_search_completion(request: Request, payload: ChatCompletionRequest, request_id: str, query: str) -> dict:
    if TicketSearchConfig is None:
        raise _safe_http_error("Ticket vector search is not available", status.HTTP_503_SERVICE_UNAVAILABLE, request_id)

    searcher = await _get_ticket_search(request)
    config = TicketSearchConfig(top_k=10)
    results = await run_in_threadpool(searcher.search, query, config)
    created = int(time.time())
    full_answer = _format_ticket_search_answer(query, results)
    prompt_tokens = len(query.split())
    completion_tokens = len(full_answer.split())

    if payload.stream:
        async def event_stream():
            chunk = {
                "id": f"chatcmpl-reglab-{request_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": full_answer},
                        "finish_reason": None,
                    }
                ],
            }
            final_chunk = {
                "id": f"chatcmpl-reglab-{request_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-reglab-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _field_value(item: Any, *names: str) -> str:
    for name in names:
        if isinstance(item, dict):
            value = item.get(name)
        else:
            value = getattr(item, name, None)
        if value:
            return str(value).strip()
    return ""


def _compact_text(value: Any, max_chars: int = 1800) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _compact_block_text(value: Any, max_chars: int = 2600) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" (?=(-|\d+\.|Что известно|Что подтверждено|Наиболее вероятно|Проверить сначала|Гипотезы|Что запросить))", "\n", text)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _normalize_eva_article_topic(query: str) -> str:
    text = " ".join(str(query or "").split()).strip(" .")
    patterns = [
        r"^сгенерируй(?:те)?\s+(?:мне\s+)?(?:статью\s+)?(?:eva\s+)?(?:для\s+базы\s+знаний\s+)?(?:по|про|о)\s+",
        r"^создай(?:те)?\s+(?:мне\s+)?(?:статью\s+)?(?:eva\s+)?(?:для\s+базы\s+знаний\s+)?(?:по|про|о)\s+",
        r"^напиши(?:те)?\s+(?:мне\s+)?(?:статью\s+)?(?:eva\s+)?(?:для\s+базы\s+знаний\s+)?(?:по|про|о)\s+",
        r"^подготовь(?:те)?\s+(?:мне\s+)?(?:статью\s+)?(?:eva\s+)?(?:для\s+базы\s+знаний\s+)?(?:по|про|о)\s+",
        r"^сделай(?:те)?\s+(?:мне\s+)?(?:статью\s+)?(?:eva\s+)?(?:для\s+базы\s+знаний\s+)?(?:по|про|о)\s+",
    ]
    lowered = text.lower()
    for pattern in patterns:
        match = re.match(pattern, lowered, flags=re.IGNORECASE)
        if match:
            return text[match.end():].strip(" .")
    return text


def _looks_like_article_command(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in ("сгенерируй", "создай", "напиши", "подготовь", "сделай")) and (
        "стать" in lowered or "eva" in lowered
    )


def _collect_eva_tags(query: str, result: Any) -> List[str]:
    text_parts = [
        query,
        getattr(result, "user_intent", ""),
        getattr(result, "docs_answer", ""),
        getattr(result, "draft_private_comment", ""),
    ]
    for ticket in getattr(result, "similar_tickets", []):
        text_parts.extend(
            [
                _field_value(ticket, "problem_summary"),
                _field_value(ticket, "solution_summary"),
                _field_value(ticket, "source_file"),
            ]
        )
    haystack = " ".join(text_parts).lower()

    tag_rules = [
        ("Astra.IDE", ["astra.ide", "astra ide", "astraide"]),
        ("AstraRegul", ["astraregul"]),
        ("R500S", ["r500s", "regul r500s"]),
        ("R500", ["r500", "regul r500"]),
        ("R050", ["r050", "regul r050"]),
        ("СПО ПЛК", ["спо", "прошив", "firmware"]),
        ("сертификаты", ["сертификат", "certificate"]),
        ("CmpCodeMeter", ["cmpcodemeter", "codemeter"]),
        ("HART", ["hart"]),
        ("Modbus", ["modbus", "модбас"]),
        ("OPC UA", ["opc ua", "opcua"]),
        ("Safety", ["safety"]),
        ("Linux", ["linux", "astra linux"]),
        ("VMWare", ["vmware", "виртуальн"]),
        ("лицензии", ["лиценз", "license"]),
        ("связь", ["связ", "обмен", "опрос"]),
    ]

    tags = ["FAQ", "EVA", "техподдержка"]
    for tag, markers in tag_rules:
        if any(marker in haystack for marker in markers):
            tags.append(tag)

    for source in list(getattr(result, "doc_sources", [])) + list(getattr(result, "ticket_sources", [])):
        for name in ("equipment_type", "library_name", "release_version"):
            value = _field_value(source, name)
            if value and len(value) <= 40:
                tags.append(value)

    unique_tags = []
    seen = set()
    for tag in tags:
        normalized = tag.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_tags.append(tag)
    return unique_tags[:12]


def _build_eva_title(topic: str, result: Any) -> str:
    intent = _compact_text(getattr(result, "user_intent", ""), 140)
    if _looks_like_article_command(intent):
        intent = ""
    base = intent if intent and intent.lower() not in {"unknown", "неизвестно"} else topic
    base = base.splitlines()[0].strip(" .")
    if not base:
        return "Статья базы знаний по обращению"
    lowered = base.lower()
    if lowered.startswith(("ошибке ", "ошибка ", "проблеме ", "проблема ", "сообщении ", "сообщение ")):
        return f"Что делать при {base[:125]}?"
    if lowered.startswith("при "):
        return f"Что делать {base[:128]}?"
    if base.endswith("?"):
        return base[:140]
    if lowered.startswith(("как ", "что ", "почему ", "где ", "когда ")):
        return f"{base[:139]}?"
    return f"Что делать: {base[:125]}?"


def _format_eva_sources(result: Any) -> str:
    lines: List[str] = []
    source_index = 1

    for title, sources in (
        ("Документация", getattr(result, "doc_sources", [])),
        ("Похожие обращения", getattr(result, "ticket_sources", [])),
    ):
        for source in sources:
            source_title = _field_value(source, "title", "page_title", "source_file") or "Источник"
            source_file = _field_value(source, "source_file")
            url = _field_value(source, "url")
            details = [source_title]
            if source_file and source_file != source_title:
                details.append(source_file)
            if url:
                details.append(url)
            lines.append(f"{source_index}. {title}: {' | '.join(details)}")
            source_index += 1

    if not lines:
        return "Источники не найдены в RAG-результате."
    return "\n".join(lines)


def _format_eva_article(result: Any, query: str) -> str:
    topic = _normalize_eva_article_topic(query)
    tags = ", ".join(_collect_eva_tags(topic, result))
    title = _build_eva_title(topic, result)
    raw_problem = getattr(result, "user_intent", "") or topic
    if _looks_like_article_command(raw_problem):
        raw_problem = topic
    problem = _compact_text(raw_problem, 1200)
    docs_answer = _compact_block_text(getattr(result, "docs_answer", ""), 1200)
    solution = _compact_block_text(getattr(result, "draft_private_comment", "") or docs_answer, 3200)
    missing_context = _compact_text(getattr(result, "missing_context", ""), 700)
    confidence = _compact_text(getattr(result, "confidence", ""), 300)

    content_lines = []
    if problem:
        content_lines.append(problem)
    if docs_answer and docs_answer != problem:
        content_lines.append(docs_answer)
    content = "\n\n".join(content_lines) or "Краткое содержание нужно уточнить по источникам."

    comments = []
    if missing_context and missing_context.lower() not in {"none", "нет", "не указано"}:
        comments.append(f"Недостающий контекст: {missing_context}")
    if confidence:
        comments.append(f"Уверенность ответа: {confidence}")
    comments.append("Перед публикацией проверьте актуальность версий ПО, ссылок и применимость решения к конкретной конфигурации.")

    return (
        "Формат написания статьи на портале EVA в базу знаний\n\n"
        f"Теги:\n{tags}\n\n"
        f"Название статьи:\n{title}\n\n"
        f"Содержание статьи:\n{content}\n\n"
        f"Описание проблемы:\n{problem or topic}\n\n"
        f"Решение:\n{solution}\n\n"
        f"Комментарии:\n" + "\n".join(f"- {item}" for item in comments) + "\n\n"
        f"Источники:\n{_format_eva_sources(result)}"
    )


async def _chat_answer_completion(
    payload: ChatCompletionRequest,
    request_id: str,
    query: str,
    full_answer: str,
) -> dict:
    created = int(time.time())
    prompt_tokens = len(query.split())
    completion_tokens = len(full_answer.split())

    if payload.stream:
        async def event_stream():
            chunk = {
                "id": f"chatcmpl-reglab-{request_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": full_answer},
                        "finish_reason": None,
                    }
                ],
            }
            final_chunk = {
                "id": f"chatcmpl-reglab-{request_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": f"chatcmpl-reglab-{request_id}",
        "object": "chat.completion",
        "created": created,
        "model": payload.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _safe_http_error(message: str, status_code: int, request_id: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=ErrorResponse(error=message, request_id=request_id).model_dump(),
    )


@app.get("/health")
async def health() -> dict:
    """Liveness endpoint: the HTTP process is running."""

    return {"status": "ok", "service": "reglab-rag-api", "version": API_VERSION}


@app.get("/ready")
async def ready(request: Request) -> dict:
    """Readiness endpoint: the RAG engine is initialized and can serve requests."""

    if getattr(request.app.state, "rag_engine", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "error": getattr(request.app.state, "startup_error", None) or "RAG engine is not initialized",
            },
        )
    return {"status": "ready", "model": DEEP_MODEL_ID}


@app.post("/api/v1/analyze_ticket", response_model=TicketResponse)
async def analyze_ticket(request: Request, payload: TicketRequest) -> TicketResponse:
    """Generate a private support hint for a new or active support ticket."""

    request_id = _request_id()
    engine = _get_engine(request)
    query = _build_ticket_query(payload.text, payload.equipment)

    try:
        log.info("Analyze ticket request_id=%s ticket_id=%s", request_id, payload.ticket_id)
        async with request.app.state.inference_semaphore:
            result = await run_in_threadpool(engine.process_support_ticket, query)

        return TicketResponse(
            ticket_id=payload.ticket_id,
            user_intent=result.user_intent,
            docs_answer=result.docs_answer,
            related_topics=result.related_topics,
            similar_tickets=[_ticket_to_response(ticket) for ticket in result.similar_tickets],
            evidence_notes=[_evidence_to_response(note) for note in getattr(result, "evidence_notes", [])],
            recommended_questions=list(getattr(result, "recommended_questions", [])),
            internal_notes=list(getattr(result, "internal_notes", [])),
            doc_sources=[_source_to_response(source) for source in getattr(result, "doc_sources", [])],
            ticket_sources=[_source_to_response(source) for source in getattr(result, "ticket_sources", [])],
            missing_context=result.missing_context,
            draft_private_comment=result.draft_private_comment,
            confidence=result.confidence,
            final_answer=result.draft_private_comment,
            extracted_facts=[f"Documentation: {_format_docs_answer_text(result.docs_answer)}"],
        )
    except ValueError as exc:
        log.warning(
            "Invalid ticket request request_id=%s ticket_id=%s: %s",
            request_id,
            payload.ticket_id,
            exc,
        )
        raise _safe_http_error("Invalid ticket request", status.HTTP_400_BAD_REQUEST, request_id) from exc
    except Exception as exc:
        log.error(
            "Ticket processing failed request_id=%s ticket_id=%s: %s",
            request_id,
            payload.ticket_id,
            exc,
            exc_info=True,
        )
        raise _safe_http_error("Ticket processing failed", status.HTTP_500_INTERNAL_SERVER_ERROR, request_id) from exc


@app.get("/v1/models")
async def list_models() -> dict:
    """Return the model list expected by OpenAI-compatible clients."""

    created = int(time.time())
    models = [
        (
            DEEP_MODEL_ID,
            "RegLab AI Deep: maximum ticket-search quality with full reranking",
        ),
        (
            ADAPTIVE_MODEL_ID,
            "RegLab AI Adaptive: fast documentation search and adaptive ticket reranking for incidents",
        ),
    ]
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "reglab",
                "description": description,
            }
            for model_id, description in models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, payload: ChatCompletionRequest) -> dict:
    """OpenAI-compatible non-streaming chat completions endpoint."""

    request_id = _request_id()

    user_messages = [message for message in payload.messages if message.role == "user"]
    if not user_messages:
        raise _safe_http_error("At least one user message is required", status.HTTP_400_BAD_REQUEST, request_id)

    query = user_messages[-1].content.strip()
    if not query:
        raise _safe_http_error("User message is empty", status.HTTP_400_BAD_REQUEST, request_id)

    known_models = set(MODEL_PROFILES)
    if payload.model not in known_models:
        raise _safe_http_error("Unknown model", status.HTTP_404_NOT_FOUND, request_id)

    try:
        if payload.model == TICKET_SEARCH_MODEL_ID:
            log.info(
                "Ticket-search request request_id=%s model=%s query=%r",
                request_id,
                payload.model,
                query[:120],
            )
            async with request.app.state.inference_semaphore:
                return await _ticket_search_completion(request, payload, request_id, query)

        engine = await _get_engine_for_model(request, payload.model)
        if payload.model in (CHAT_MODEL_ID, CHAT_DEEP_MODEL_ID):
            log.info(
                "Wiki chat completion request request_id=%s model=%s profile=%s query=%r messages_count=%d",
                request_id,
                payload.model,
                _profile_from_model(payload.model),
                query[:120],
                len(payload.messages),
            )
            # OpenWebUI may submit the full transcript. The RAG API is
            # intentionally stateless: only the latest user question is sent
            # to the engine, and no prior assistant/user text can affect it.
            raw_messages = [{"role": "user", "content": query}]
            async with request.app.state.inference_semaphore:
                result = await run_in_threadpool(engine.process_wiki_chat_dialog, raw_messages)
            full_answer = _format_wiki_chat_answer(result)
            return await _chat_answer_completion(payload, request_id, query, full_answer)
        if payload.model == EVA_ARTICLE_MODEL_ID:
            log.info(
                "EVA article request request_id=%s model=%s profile=%s query=%r",
                request_id,
                payload.model,
                _profile_from_model(payload.model),
                query[:120],
            )
            async with request.app.state.inference_semaphore:
                result = await run_in_threadpool(engine.process_support_ticket, query)
            full_answer = _format_eva_article(result, query)
            return await _chat_answer_completion(payload, request_id, query, full_answer)

        log.info(
            "OpenAI-compatible request request_id=%s model=%s profile=%s query=%r",
            request_id,
            payload.model,
            _profile_from_model(payload.model),
            query[:120],
        )
        async with request.app.state.inference_semaphore:
            result = await run_in_threadpool(engine.process_support_ticket, query)

        created = int(time.time())
        full_answer = _format_support_answer(result)
        prompt_tokens = len(query.split())
        completion_tokens = len(full_answer.split())

        if payload.stream:
            async def event_stream():
                chunk = {
                    "id": f"chatcmpl-reglab-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": payload.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": full_answer},
                            "finish_reason": None,
                        }
                    ],
                }
                final_chunk = {
                    "id": f"chatcmpl-reglab-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": payload.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return {
            "id": f"chatcmpl-reglab-{request_id}",
            "object": "chat.completion",
            "created": created,
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    except ValueError as exc:
        log.warning("Invalid chat request request_id=%s: %s", request_id, exc)
        raise _safe_http_error("Invalid chat request", status.HTTP_400_BAD_REQUEST, request_id) from exc
    except Exception as exc:
        log.error("OpenAI-compatible request failed request_id=%s: %s", request_id, exc, exc_info=True)
        raise _safe_http_error("Chat completion failed", status.HTTP_500_INTERNAL_SERVER_ERROR, request_id) from exc


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("RAG_API_HOST", "0.0.0.0"),
        port=int(os.getenv("RAG_API_PORT", "8000")),
    )
