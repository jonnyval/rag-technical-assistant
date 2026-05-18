"""FastAPI server for the RegLab RAG support-ticket workflow.

The module exposes two integration surfaces:
- /api/v1/analyze_ticket for the support portal.
- /v1/models and /v1/chat/completions for OpenAI-compatible clients.
"""

from __future__ import annotations

import asyncio
import json
import os
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


MODEL_ID = "reglab-ai"
DEEP_MODEL_ID = "reglab-ai-deep"
MODEL_PROFILES = {
    MODEL_ID: "fast",
    DEEP_MODEL_ID: "deep",
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

    model: str = MODEL_ID
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
    app.state.engine_locks = {
        profile: asyncio.Lock()
        for profile in set(MODEL_PROFILES.values())
    }
    app.state.startup_error = None
    app.state.inference_semaphore = asyncio.Semaphore(_max_concurrency())

    try:
        log.info("Starting RegLab RAG API: initializing default fast RAGEngine...")
        app.state.rag_engine = await run_in_threadpool(RAGEngine, "fast")
        app.state.rag_engines["fast"] = app.state.rag_engine
        log.info("RegLab RAG API is ready.")
    except Exception as exc:
        app.state.startup_error = str(exc)
        log.error("RAGEngine initialization failed: %s", exc, exc_info=True)

    yield

    app.state.rag_engine = None
    app.state.rag_engines = {}
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
    return MODEL_PROFILES.get(model_id, "fast")


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
            engine = await run_in_threadpool(RAGEngine, profile)
            engines[profile] = engine
            if profile == "deep":
                request.app.state.rag_engine = engine
        return engine


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


def _format_support_answer(result: Any) -> str:
    tickets_block = ""
    if result.similar_tickets:
        ticket_lines = "\n".join(
            f"- **{ticket.ticket_id}**: {ticket.problem_summary}. "
            f"Resolution or action: {ticket.solution_summary}"
            for ticket in result.similar_tickets
        )
        tickets_block = f"\n\n---\n**Similar tickets:**\n{ticket_lines}"

    return (
        f"{result.draft_private_comment}"
        f"\n\n---\n**Documentation context:**\n{result.docs_answer}"
        f"{tickets_block}"
        f"{_format_evidence(getattr(result, 'evidence_notes', []))}"
        f"{_format_list('Recommended questions', getattr(result, 'recommended_questions', []))}"
        f"{_format_list('Internal notes', getattr(result, 'internal_notes', []))}"
        f"{_format_sources('Documentation sources', getattr(result, 'doc_sources', []))}"
        f"{_format_sources('Ticket sources', getattr(result, 'ticket_sources', []))}"
        f"\n\n**Confidence:** {result.confidence}"
    )


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
    return "\n\n**Обоснование по источникам:**\n" + "\n".join(lines)


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


def _format_support_answer(result: Any) -> str:
    tickets_block = ""
    if result.similar_tickets:
        ticket_lines = "\n".join(
            f"- **{ticket.ticket_id}**: {ticket.problem_summary}. "
            f"Решение/действие: {ticket.solution_summary}"
            for ticket in result.similar_tickets
        )
        tickets_block = f"\n\n---\n**Похожие обращения:**\n{ticket_lines}"

    internal_notes = getattr(result, "internal_notes", [])
    return (
        f"{result.draft_private_comment}"
        f"\n\n---\n**Информация из документации:**\n{result.docs_answer}"
        f"{tickets_block}"
        f"{_format_evidence(getattr(result, 'evidence_notes', []))}"
        f"{_format_list('Внутренние заметки для инженера', internal_notes)}"
        f"{_format_sources('Источники документации', getattr(result, 'doc_sources', []))}"
        f"{_format_sources('Источники похожих обращений', getattr(result, 'ticket_sources', []))}"
        f"\n\n**Уверенность:** {result.confidence}"
    )


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


def _format_evidence(notes: List[Any]) -> str:
    if not notes:
        return ""
    lines = []
    for note in notes:
        data = _to_mapping(note)
        claim = data.get("claim") or ""
        if not claim.strip():
            continue
        source_ids = data.get("source_ids") or []
        comment = data.get("comment") or ""
        suffix = f" [{', '.join(source_ids)}]" if source_ids else ""
        if comment:
            suffix += f" - {comment}"
        lines.append(f"- {claim}{suffix}")
    return "\n\n**Обоснование по источникам:**\n" + "\n".join(lines) if lines else ""


def _format_sources(title: str, sources: List[Any]) -> str:
    if not sources:
        return ""
    lines = []
    for source in sources:
        data = _to_mapping(source)
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


def _format_support_answer(result: Any) -> str:
    tickets = [
        ticket for ticket in getattr(result, "similar_tickets", [])
        if _ticket_has_content(ticket)
    ]
    tickets_block = ""
    if tickets:
        ticket_lines = "\n".join(
            f"- **{ticket.ticket_id}**: {ticket.problem_summary}. "
            f"Решение/действие: {ticket.solution_summary}"
            for ticket in tickets
        )
        tickets_block = f"\n\n---\n**Похожие обращения:**\n{ticket_lines}"

    return (
        f"{result.draft_private_comment}"
        f"\n\n---\n**Информация из документации:**\n{result.docs_answer}"
        f"{tickets_block}"
        f"{_format_evidence(getattr(result, 'evidence_notes', []))}"
        f"{_format_sources('Источники документации', getattr(result, 'doc_sources', []))}"
        f"{_format_sources('Источники похожих обращений', getattr(result, 'ticket_sources', []))}"
        f"\n\n**Уверенность:** {result.confidence}"
    )


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
    return {"status": "ready", "model": MODEL_ID}


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
            extracted_facts=[f"Documentation: {result.docs_answer}"],
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
            MODEL_ID,
            "RegLab AI: fast support assistant without reranker, lower latency and token use",
        ),
        (
            DEEP_MODEL_ID,
            "RegLab AI Deep: reranker enabled with moderate context for harder questions",
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
    engine = await _get_engine_for_model(request, payload.model)

    user_messages = [message for message in payload.messages if message.role == "user"]
    if not user_messages:
        raise _safe_http_error("At least one user message is required", status.HTTP_400_BAD_REQUEST, request_id)

    query = user_messages[-1].content.strip()
    if not query:
        raise _safe_http_error("User message is empty", status.HTTP_400_BAD_REQUEST, request_id)

    try:
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
