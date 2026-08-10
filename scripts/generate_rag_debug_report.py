"""Generate a local HTML/JSON trace of the production RAG retrieval pipeline.

The script instruments the real ``RAGEngine`` and ``DualRetriever`` methods. It
does not maintain a second implementation of retrieval, so query expansion,
Qdrant search, RRF, reranking, parent lookup, filtering and context formatting
stay aligned with production.

Examples:

    python scripts/generate_rag_debug_report.py --query "заводской сброс R500"
    python scripts/generate_rag_debug_report.py --input questions.json --generate
    python scripts/generate_rag_debug_report.py --query "download denied" \
        --profiles adaptive,deep,agentic --compare reranker,multi_query
    python scripts/generate_rag_debug_report.py --input questions.jsonl \
        --variant no-rerank:reranker=off \
        --variant no-mq:multi_query=off

The HTML is self-contained and performs no network requests. A raw JSON file
with full document text and metadata is written next to it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import re
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

import src.agentic.controller as agentic_controller_module
import src.engine as engine_module
from scripts.api_server import _format_adaptive_answer, _format_support_answer
from src.agentic.pipeline import AgenticRAG
from src.config import settings
from src.engine import (
    AdaptiveInformationResponse,
    RAGEngine,
    SupportPrivateResponse,
)
from src.module_detection import (
    build_module_enriched_query,
    detect_modules_in_query,
    format_detected_modules,
)
from src.retrieval.dual_retriever import DualRetriever


DEFAULT_OUTPUT_DIR = Path("reports/rag_debug")
SUPPORTED_PROFILES = {"adaptive", "deep", "agentic"}
SUPPORTED_SWITCHES = {
    "reranker",
    "multi_query",
    "structured_planner",
    "query_aware_rerank",
}


@dataclass(frozen=True)
class Question:
    question_id: str
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Variant:
    name: str
    switches: dict[str, bool | None] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace the exact RegLab retrieval/context pipeline into local HTML and JSON.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Single question. May be repeated.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Questions from JSON, JSONL or TXT.",
    )
    parser.add_argument(
        "--profiles",
        default="adaptive",
        help="Comma-separated profiles: adaptive,deep,agentic.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Call the final answer LLM and include raw/final responses. Retrieval planners "
            "may still use an LLM without this flag when enabled by configuration."
        ),
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME:KEY=on,KEY=off",
        help=(
            "Repeatable configuration variant. Keys: reranker, multi_query, "
            "structured_planner, query_aware_rerank."
        ),
    )
    parser.add_argument(
        "--compare",
        default="",
        help=(
            "Comma-separated switches to toggle against the current config, e.g. "
            "reranker,multi_query."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="", help="Output basename without extension.")
    parser.add_argument("--limit", type=int, default=None, help="Limit loaded questions.")
    return parser.parse_args()


def _bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "on", "yes", "enabled"}:
        return True
    if normalized in {"0", "false", "off", "no", "disabled"}:
        return False
    raise ValueError(f"Expected on/off value, got: {value!r}")


def parse_profiles(raw: str) -> list[str]:
    profiles = list(dict.fromkeys(item.strip().lower() for item in raw.split(",") if item.strip()))
    unknown = [item for item in profiles if item not in SUPPORTED_PROFILES]
    if unknown:
        raise ValueError(f"Unknown profiles: {', '.join(unknown)}")
    return profiles or ["adaptive"]


def parse_variant(raw: str) -> Variant:
    if ":" not in raw:
        raise ValueError(f"Variant must look like NAME:KEY=on, got: {raw!r}")
    name, assignments = raw.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError("Variant name must not be empty")
    switches: dict[str, bool | None] = {}
    for assignment in assignments.split(","):
        if not assignment.strip():
            continue
        if "=" not in assignment:
            raise ValueError(f"Variant assignment must contain '=': {assignment!r}")
        key, value = (part.strip() for part in assignment.split("=", 1))
        if key not in SUPPORTED_SWITCHES:
            raise ValueError(f"Unknown variant switch {key!r}; supported: {sorted(SUPPORTED_SWITCHES)}")
        switches[key] = _bool_value(value)
    return Variant(name=name, switches=switches)


def current_switches() -> dict[str, bool]:
    return {
        "reranker": True,
        "multi_query": bool(settings.multi_query_enabled),
        "structured_planner": bool(settings.structured_query_planner_enabled),
        "query_aware_rerank": bool(settings.query_aware_rerank_enabled),
    }


def build_variants(raw_variants: list[str], compare: str) -> list[Variant]:
    variants = [Variant("baseline", {})]
    variants.extend(parse_variant(item) for item in raw_variants)
    baseline = current_switches()
    for key in (item.strip() for item in compare.split(",") if item.strip()):
        if key not in SUPPORTED_SWITCHES:
            raise ValueError(f"Unknown --compare switch {key!r}")
        toggled = not baseline[key]
        variants.append(Variant(f"{key}-{'on' if toggled else 'off'}", {key: toggled}))
    result: list[Variant] = []
    seen: set[str] = set()
    for variant in variants:
        if variant.name in seen:
            raise ValueError(f"Duplicate variant name: {variant.name}")
        seen.add(variant.name)
        result.append(variant)
    return result


def _question_from_item(item: Any, index: int) -> Question | None:
    if isinstance(item, str):
        query = item.strip()
        return Question(f"q{index:03d}", query) if query else None
    if not isinstance(item, dict):
        return None
    query = str(item.get("query") or item.get("question") or item.get("text") or "").strip()
    if not query:
        return None
    question_id = str(item.get("id") or item.get("question_id") or f"q{index:03d}")
    metadata = {key: value for key, value in item.items() if key not in {"query", "question", "text"}}
    return Question(question_id, query, metadata)


def load_questions(cli_queries: list[str], input_path: Path | None, limit: int | None) -> list[Question]:
    items: list[Any] = [query for query in cli_queries if str(query).strip()]
    if input_path:
        suffix = input_path.suffix.lower()
        text = input_path.read_text(encoding="utf-8-sig")
        if suffix == ".jsonl":
            items.extend(json.loads(line) for line in text.splitlines() if line.strip())
        elif suffix == ".txt":
            items.extend(line.strip() for line in text.splitlines() if line.strip())
        else:
            payload = json.loads(text)
            if isinstance(payload, dict):
                payload = payload.get("questions", payload.get("items", []))
            if not isinstance(payload, list):
                raise ValueError("JSON input must be a list or contain a 'questions' list")
            items.extend(payload)
    questions = [question for index, item in enumerate(items, start=1) if (question := _question_from_item(item, index))]
    if limit is not None:
        questions = questions[: max(0, limit)]
    if not questions:
        raise ValueError("No questions supplied. Use --query or --input.")
    return questions


def json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return json_safe(value.model_dump(mode="json"))
    if isinstance(value, Document):
        return serialize_document(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def document_key(document: Document) -> str:
    metadata = document.metadata or {}
    raw = "|".join(
        str(value or "")
        for value in (
            metadata.get("db_source"),
            metadata.get("doc_id"),
            metadata.get("ticket_id"),
            metadata.get("source_file"),
            metadata.get("page_title"),
            document.page_content,
        )
    )
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def serialize_document(document: Document, rank: int | None = None) -> dict[str, Any]:
    metadata = copy.deepcopy(getattr(document, "metadata", {}) or {})
    return {
        "key": document_key(document),
        "rank": rank,
        "page_content": str(getattr(document, "page_content", "") or ""),
        "metadata": json_safe(metadata),
        "scores": {
            key: json_safe(metadata.get(key))
            for key in (
                "qdrant_score",
                "qdrant_rank",
                "fusion_score",
                "rerank_score",
                "original_rerank_score",
                "query_rerank_score",
                "origin_query_rank",
            )
            if metadata.get(key) is not None
        },
    }


def serialize_documents(documents: Iterable[Document]) -> list[dict[str, Any]]:
    return [serialize_document(document, rank=index) for index, document in enumerate(documents, start=1)]


def diff_documents(before: Iterable[Document], after: Iterable[Document]) -> list[dict[str, Any]]:
    after_keys = {document_key(document) for document in after}
    return [
        serialize_document(document, rank=index)
        for index, document in enumerate(before, start=1)
        if document_key(document) not in after_keys
    ]


class LLMUsageHandler(BaseCallbackHandler):
    """Collect provider token metadata without exposing credentials."""

    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.errors: list[str] = []
        self.usage: list[dict[str, Any]] = []

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self.starts += 1

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[Any], **kwargs: Any) -> None:
        self.starts += 1

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.errors.append(f"{type(error).__name__}: {error}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.ends += 1
        output = getattr(response, "llm_output", None) or {}
        item: dict[str, Any] = {"llm_output": json_safe(output)}
        generations = getattr(response, "generations", None) or []
        usage_metadata: list[Any] = []
        for generation_group in generations:
            for generation in generation_group:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    usage_metadata.append(json_safe(usage))
        if usage_metadata:
            item["usage_metadata"] = usage_metadata
        self.usage.append(item)

    def snapshot(self) -> dict[str, Any]:
        return {
            "starts": self.starts,
            "ends": self.ends,
            "errors": list(self.errors),
            "usage": copy.deepcopy(self.usage),
        }


def render_chain_prompt(chain: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """Render the first prompt step of a RunnableSequence when available."""
    try:
        first = getattr(chain, "first", None)
        if first is None:
            steps = getattr(chain, "steps", None) or []
            first = steps[0] if steps else None
        if first is None:
            return {"text": "", "messages": [], "error": "Prompt step is not exposed"}
        prompt_value = first.invoke(payload)
        messages = []
        for message in prompt_value.to_messages():
            messages.append({
                "role": getattr(message, "type", message.__class__.__name__),
                "content": str(getattr(message, "content", "")),
            })
        text = prompt_value.to_string()
        return {
            "text": text,
            "messages": messages,
            "characters": len(text),
            "words": len(text.split()),
            "estimated_tokens": max(1, round(len(text) / 4)),
        }
    except Exception as error:
        return {"text": "", "messages": [], "error": f"{type(error).__name__}: {error}"}


class TracedChain:
    """Proxy a LangChain runnable and record prompt, payload, result and usage."""

    def __init__(
        self,
        inner: Any,
        collector: "TraceCollector",
        label: str,
        *,
        invoke_inner: bool = True,
        fallback: Any = None,
        result_cache: dict[str, Any] | None = None,
        cache_key_factory: Any = None,
    ) -> None:
        self.inner = inner
        self.collector = collector
        self.label = label
        self.invoke_inner = invoke_inner
        self.fallback = fallback
        self.result_cache = result_cache
        self.cache_key_factory = cache_key_factory

    def invoke(self, payload: Any, config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        started = time.perf_counter()
        payload_dict = payload if isinstance(payload, dict) else {"input": payload}
        prompt = render_chain_prompt(self.inner, payload_dict)
        usage = LLMUsageHandler()
        call_config = dict(config or {})
        callbacks = list(call_config.get("callbacks") or [])
        callbacks.append(usage)
        call_config["callbacks"] = callbacks
        error_text = ""
        result = None
        replayed = False
        try:
            if self.invoke_inner:
                cache_key = (
                    str(self.cache_key_factory(payload))
                    if self.result_cache is not None and self.cache_key_factory is not None
                    else ""
                )
                if cache_key and cache_key in self.result_cache:
                    result = copy.deepcopy(self.result_cache[cache_key])
                    replayed = True
                else:
                    result = self.inner.invoke(payload, config=call_config, **kwargs)
                    if cache_key:
                        self.result_cache[cache_key] = copy.deepcopy(result)
            else:
                result = copy.deepcopy(self.fallback)
            return result
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.collector.llm_calls.append({
                "label": self.label,
                "executed": self.invoke_inner and not replayed,
                "replayed": replayed,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "payload": json_safe(payload),
                "prompt": prompt,
                "result": json_safe(result),
                "usage": usage.snapshot(),
                "error": error_text,
            })


class TraceCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.formatted_contexts: list[dict[str, Any]] = []
        self.filters: list[dict[str, Any]] = []
        self.agent_trace: list[dict[str, Any]] = []

    def event(
        self,
        stage: str,
        *,
        source: str = "",
        query: str = "",
        inputs: Iterable[Document] = (),
        outputs: Iterable[Document] = (),
        elapsed_seconds: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> None:
        before = list(inputs)
        after = list(outputs)
        self.events.append({
            "index": len(self.events) + 1,
            "stage": stage,
            "source": source,
            "query": query,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "input_count": len(before),
            "output_count": len(after),
            "details": json_safe(details or {}),
            "documents": serialize_documents(after),
            "removed": diff_documents(before, after),
        })


def _dummy_response(profile: str) -> SupportPrivateResponse:
    response_class = AdaptiveInformationResponse if profile == "adaptive" else SupportPrivateResponse
    return response_class(
        user_intent="Generation disabled",
        docs_answer="",
        draft_private_comment="",
        missing_context="Не указано",
        confidence="low",
    )


def _query_from_documents(documents: list[Document]) -> str:
    for document in documents:
        value = str((document.metadata or {}).get("rerank_query") or "").strip()
        if value:
            return value
    return ""


class PipelineInstrumentation:
    """Patch production methods for one run and restore them afterwards."""

    def __init__(self, collector: TraceCollector):
        self.collector = collector
        self.stack = ExitStack()

    def __enter__(self) -> "PipelineInstrumentation":
        collector = self.collector

        original_search = DualRetriever._search_children
        original_fuse = DualRetriever._fuse_multi_query_children
        original_rank = DualRetriever._rank_children
        original_rank_multi = DualRetriever._rank_multi_query_children
        original_diverse = DualRetriever._select_diverse_children
        original_parents = DualRetriever._fetch_parents
        original_exact_docs = DualRetriever.retrieve_docs_by_page_titles
        original_exact_tickets = DualRetriever.retrieve_tickets_by_module_codes
        original_similarity_with_score = QdrantVectorStore.similarity_search_with_score

        def similarity_search_wrapper(
            vectorstore: QdrantVectorStore,
            query: str,
            k: int = 4,
            filter: Any = None,
            search_params: Any = None,
            offset: int = 0,
            score_threshold: float | None = None,
            consistency: Any = None,
            hybrid_fusion: Any = None,
            **kwargs: Any,
        ) -> list[Document]:
            pairs = original_similarity_with_score(
                vectorstore,
                query,
                k=k,
                filter=filter,
                search_params=search_params,
                offset=offset,
                score_threshold=score_threshold,
                consistency=consistency,
                hybrid_fusion=hybrid_fusion,
                **kwargs,
            )
            documents = []
            for document, score in pairs:
                document.metadata["qdrant_score"] = float(score)
                document.metadata["qdrant_rank"] = len(documents) + 1
                document.metadata["qdrant_collection"] = str(vectorstore.collection_name)
                documents.append(document)
            return documents

        def search_wrapper(instance: DualRetriever, retriever: Any, query: str, k: int, apply_filter: bool, db_label: str):
            started = time.perf_counter()
            result = original_search(instance, retriever, query, k, apply_filter, db_label)
            collector.event(
                "qdrant_hybrid_search",
                source=db_label,
                query=query,
                outputs=result,
                elapsed_seconds=time.perf_counter() - started,
                details={
                    "requested_k": k,
                    "equipment_filter_applied": apply_filter,
                    "score_note": (
                        "The production LangChain similarity_search call exposes result order but "
                        "does not expose separate dense/BM25 component scores."
                    ),
                },
            )
            return result

        def fuse_wrapper(instance: DualRetriever, query_results: list[list[Document]], **kwargs: Any):
            started = time.perf_counter()
            before = [document for result in query_results for document in result]
            result = original_fuse(instance, query_results, **kwargs)
            collector.event(
                "rrf_fusion",
                query=" | ".join(kwargs.get("queries") or []),
                inputs=before,
                outputs=result,
                elapsed_seconds=time.perf_counter() - started,
                details={
                    "queries": kwargs.get("queries") or [],
                    "rrf_k": kwargs.get("rrf_k"),
                    "candidate_limit": kwargs.get("candidate_limit"),
                },
            )
            return result

        def rank_wrapper(instance: DualRetriever, query: str, children: list[Document], **kwargs: Any):
            before = [copy.deepcopy(document) for document in children]
            started = time.perf_counter()
            result = original_rank(instance, query, children, **kwargs)
            collector.event(
                "cross_encoder_rerank" if kwargs.get("use_reranker", True) and instance.reranker_model is not None else "vector_order_selection",
                query=query,
                inputs=before,
                outputs=result,
                elapsed_seconds=time.perf_counter() - started,
                details={
                    "requested_limit": kwargs.get("limit"),
                    "reranker_requested": kwargs.get("use_reranker", True),
                    "reranker_available": instance.reranker_model is not None,
                    "threshold": instance.rerank_threshold,
                },
            )
            return result

        def rank_multi_wrapper(instance: DualRetriever, original_query: str, children: list[Document], **kwargs: Any):
            before = [copy.deepcopy(document) for document in children]
            started = time.perf_counter()
            result = original_rank_multi(instance, original_query, children, **kwargs)
            collector.event(
                "query_aware_rerank" if instance.query_aware_rerank else "cross_encoder_rerank",
                query=original_query,
                inputs=before,
                outputs=result,
                elapsed_seconds=time.perf_counter() - started,
                details={
                    "query_aware": instance.query_aware_rerank,
                    "requested_limit": kwargs.get("limit"),
                    "reranker_requested": kwargs.get("use_reranker", True),
                    "reranker_available": instance.reranker_model is not None,
                },
            )
            return result

        def diverse_wrapper(instance: DualRetriever, ranked: list[Document], queries: list[str], **kwargs: Any):
            before = [copy.deepcopy(document) for document in ranked]
            result = original_diverse(instance, ranked, queries, **kwargs)
            collector.event(
                "facet_quota_selection",
                query=" | ".join(queries),
                inputs=before,
                outputs=result,
                details={
                    "limit": kwargs.get("limit"),
                    "max_reserved_queries": instance.max_reserved_queries,
                    "per_query_quota": instance.per_query_quota,
                },
            )
            return result

        def parents_wrapper(instance: DualRetriever, retriever: Any, top_children: list[Document], db_label: str):
            before = [copy.deepcopy(document) for document in top_children]
            started = time.perf_counter()
            result = original_parents(instance, retriever, top_children, db_label)
            collector.event(
                "parent_document_fetch",
                source=db_label,
                query=_query_from_documents(top_children),
                inputs=before,
                outputs=result,
                elapsed_seconds=time.perf_counter() - started,
            )
            return result

        def exact_docs_wrapper(instance: DualRetriever, page_titles: list[str], **kwargs: Any):
            started = time.perf_counter()
            result = original_exact_docs(instance, page_titles, **kwargs)
            collector.event(
                "exact_payload_lookup",
                source="docs",
                query=" | ".join(page_titles),
                outputs=result,
                elapsed_seconds=time.perf_counter() - started,
                details={"field": "metadata.page_title", "limit": kwargs.get("limit")},
            )
            return result

        def exact_tickets_wrapper(instance: DualRetriever, module_codes: list[str], **kwargs: Any):
            started = time.perf_counter()
            result = original_exact_tickets(instance, module_codes, **kwargs)
            collector.event(
                "exact_payload_lookup",
                source="tickets",
                query=" | ".join(module_codes),
                outputs=result,
                elapsed_seconds=time.perf_counter() - started,
                details={"field": "metadata.mentioned_module_codes", "limit": kwargs.get("limit")},
            )
            return result

        self.stack.enter_context(patch.object(DualRetriever, "_search_children", search_wrapper))
        self.stack.enter_context(patch.object(DualRetriever, "_fuse_multi_query_children", fuse_wrapper))
        self.stack.enter_context(patch.object(DualRetriever, "_rank_children", rank_wrapper))
        self.stack.enter_context(patch.object(DualRetriever, "_rank_multi_query_children", rank_multi_wrapper))
        self.stack.enter_context(patch.object(DualRetriever, "_select_diverse_children", diverse_wrapper))
        self.stack.enter_context(patch.object(DualRetriever, "_fetch_parents", parents_wrapper))
        self.stack.enter_context(patch.object(DualRetriever, "retrieve_docs_by_page_titles", exact_docs_wrapper))
        self.stack.enter_context(patch.object(DualRetriever, "retrieve_tickets_by_module_codes", exact_tickets_wrapper))
        self.stack.enter_context(patch.object(QdrantVectorStore, "similarity_search", similarity_search_wrapper))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stack.close()


def _recording_formatter(
    original: Any,
    collector: TraceCollector,
    source: str,
):
    def wrapper(documents: list[Document], *args: Any, **kwargs: Any) -> str:
        result = original(documents, *args, **kwargs)
        collector.formatted_contexts.append({
            "index": len(collector.formatted_contexts) + 1,
            "source": source,
            "characters": len(result),
            "documents": serialize_documents(documents),
            "kwargs": json_safe(kwargs),
            "text": result,
        })
        return result
    return wrapper


def _recording_series_filter(original: Any, collector: TraceCollector, module_name: str):
    def wrapper(query: str, documents: list[Document]):
        before = list(documents)
        result = original(query, documents)
        removed = diff_documents(before, result)
        collector.filters.append({
            "stage": "requested_series_filter",
            "module": module_name,
            "query": query,
            "input_count": len(before),
            "output_count": len(result),
            "removed": removed,
        })
        return result
    return wrapper


def _recording_ticket_ranker(original: Any, collector: TraceCollector, module_name: str):
    def wrapper(tickets: list[Document], modules: list[dict[str, Any]], *, query: str = "", limit: int = 2):
        before = [copy.deepcopy(document) for document in tickets]
        result = original(tickets, modules, query=query, limit=limit)
        collector.event(
            "module_ticket_ranking",
            source="tickets",
            query=query,
            inputs=before,
            outputs=result,
            details={"module": module_name, "modules": modules, "limit": limit},
        )
        return result
    return wrapper


@contextmanager
def instrument_formatting_and_filters(engine: RAGEngine, collector: TraceCollector):
    """Capture exact contexts plus both deterministic source filters."""
    original_product_filter = engine._filter_wiki_docs_by_requested_product
    original_agent_ticket_ranker = agentic_controller_module.AgenticController._rank_agent_tickets

    def product_filter(query: str, modules: list[dict[str, Any]], documents: list[Document]):
        before = list(documents)
        result = original_product_filter(query, modules, documents)
        collector.filters.append({
            "stage": "product_family_filter",
            "query": query,
            "modules": json_safe(modules),
            "input_count": len(before),
            "output_count": len(result),
            "removed": diff_documents(before, result),
        })
        return result

    engine._filter_wiki_docs_by_requested_product = product_filter

    def agent_ticket_ranker(query: str, tickets: list[Document], exact_anchors: list[str], limit: int = 4):
        before = [copy.deepcopy(document) for document in tickets]
        result = original_agent_ticket_ranker(query, tickets, exact_anchors, limit=limit)
        collector.event(
            "agent_ticket_ranking",
            source="tickets",
            query=query,
            inputs=before,
            outputs=result,
            details={"exact_anchors": exact_anchors, "limit": limit},
        )
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(
            engine_module,
            "format_docs",
            _recording_formatter(engine_module.format_docs, collector, "docs"),
        ))
        stack.enter_context(patch.object(
            engine_module,
            "format_ticket_docs",
            _recording_formatter(engine_module.format_ticket_docs, collector, "tickets"),
        ))
        stack.enter_context(patch.object(
            agentic_controller_module,
            "format_docs",
            _recording_formatter(agentic_controller_module.format_docs, collector, "docs"),
        ))
        stack.enter_context(patch.object(
            agentic_controller_module,
            "format_ticket_docs",
            _recording_formatter(agentic_controller_module.format_ticket_docs, collector, "tickets"),
        ))
        stack.enter_context(patch.object(
            engine_module,
            "filter_documents_by_requested_series",
            _recording_series_filter(
                engine_module.filter_documents_by_requested_series,
                collector,
                "src.engine",
            ),
        ))
        stack.enter_context(patch.object(
            agentic_controller_module,
            "filter_documents_by_requested_series",
            _recording_series_filter(
                agentic_controller_module.filter_documents_by_requested_series,
                collector,
                "src.agentic.controller",
            ),
        ))
        stack.enter_context(patch.object(
            engine_module,
            "rank_tickets_for_modules",
            _recording_ticket_ranker(
                engine_module.rank_tickets_for_modules,
                collector,
                "src.engine",
            ),
        ))
        stack.enter_context(patch.object(
            agentic_controller_module,
            "rank_tickets_for_modules",
            _recording_ticket_ranker(
                agentic_controller_module.rank_tickets_for_modules,
                collector,
                "src.agentic.controller",
            ),
        ))
        stack.enter_context(patch.object(
            agentic_controller_module.AgenticController,
            "_rank_agent_tickets",
            staticmethod(agent_ticket_ranker),
        ))
        try:
            yield
        finally:
            engine._filter_wiki_docs_by_requested_product = original_product_filter


def set_nested_switches(raw_config: dict[str, Any], switches: dict[str, bool]) -> None:
    retrieval = raw_config.setdefault("retrieval", {})
    multi_query = retrieval.setdefault("multi_query", {})
    structured = multi_query.setdefault("structured_planner", {})
    query_aware = structured.setdefault("query_aware_rerank", {})
    if "multi_query" in switches:
        multi_query["enabled"] = switches["multi_query"]
    if "structured_planner" in switches:
        structured["enabled"] = switches["structured_planner"]
    if "query_aware_rerank" in switches:
        query_aware["enabled"] = switches["query_aware_rerank"]


@contextmanager
def applied_variant(engine: RAGEngine, variant: Variant):
    original_config = copy.deepcopy(settings._raw_config)
    original_reranker = engine.retriever.reranker_model
    original_query_aware = engine.retriever.query_aware_rerank
    effective = current_switches()
    effective.update({key: value for key, value in variant.switches.items() if value is not None})
    try:
        set_nested_switches(settings._raw_config, effective)
        engine.retriever.reranker_model = original_reranker if effective["reranker"] else None
        engine.retriever.query_aware_rerank = effective["query_aware_rerank"]
        yield effective
    finally:
        settings._raw_config = original_config
        engine.retriever.reranker_model = original_reranker
        engine.retriever.query_aware_rerank = original_query_aware


def _last_context(collector: TraceCollector, source: str) -> dict[str, Any]:
    matches = [item for item in collector.formatted_contexts if item["source"] == source]
    return matches[-1] if matches else {
        "source": source,
        "characters": 0,
        "documents": [],
        "kwargs": {},
        "text": "",
    }


def _final_answer_text(profile: str, response: Any) -> str:
    if response is None:
        return ""
    if profile == "adaptive":
        return _format_adaptive_answer(response)
    return _format_support_answer(response)


def _final_documents(collector: TraceCollector) -> dict[str, list[dict[str, Any]]]:
    return {
        "docs": _last_context(collector, "docs")["documents"],
        "tickets": _last_context(collector, "tickets")["documents"],
    }


def run_standard(
    engine: RAGEngine,
    question: Question,
    profile: str,
    variant: Variant,
    generate: bool,
    planner_cache: dict[str, Any],
) -> dict[str, Any]:
    collector = TraceCollector()
    original_support_chain = engine.support_chain
    original_multi_chain = engine.multi_query_chain
    original_reflection_chain = engine.retrieval_reflection_chain
    support_proxy = TracedChain(
        original_support_chain,
        collector,
        "final_support_generation",
        invoke_inner=generate,
        fallback=_dummy_response(profile),
    )
    engine.support_chain = support_proxy
    if original_multi_chain is not None:
        engine.multi_query_chain = TracedChain(
            original_multi_chain,
            collector,
            "multi_query_planner",
            result_cache=planner_cache,
            cache_key_factory=lambda payload: (
                f"{question.question_id}|{profile}|{question.query}|"
                f"structured={settings.structured_query_planner_enabled}"
            ),
        )
    engine.retrieval_reflection_chain = TracedChain(
        original_reflection_chain,
        collector,
        "diagnostic_reflection",
    )

    modules = detect_modules_in_query(question.query)
    retrieval_query = build_module_enriched_query(question.query, modules)
    started = time.perf_counter()
    response = None
    error = ""
    effective: dict[str, bool] = {}
    try:
        with applied_variant(engine, variant) as effective_switches:
            effective = dict(effective_switches)
            with PipelineInstrumentation(collector), instrument_formatting_and_filters(engine, collector):
                response = engine.process_support_ticket(question.query)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        engine.support_chain = original_support_chain
        engine.multi_query_chain = original_multi_chain
        engine.retrieval_reflection_chain = original_reflection_chain

    final_prompt_call = next(
        (item for item in reversed(collector.llm_calls) if item["label"] == "final_support_generation"),
        None,
    )
    return {
        "run_id": f"{question.question_id}:{profile}:{variant.name}",
        "question_id": question.question_id,
        "query": question.query,
        "question_metadata": json_safe(question.metadata),
        "profile": profile,
        "variant": variant.name,
        "switches": effective,
        "generate": generate,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "query_processing": {
            "detected_modules": json_safe(modules),
            "module_context": format_detected_modules(modules),
            "retrieval_query": retrieval_query,
        },
        "events": collector.events,
        "filters": collector.filters,
        "formatted_contexts": collector.formatted_contexts,
        "final_context": {
            "docs": _last_context(collector, "docs"),
            "tickets": _last_context(collector, "tickets"),
        },
        "final_documents": _final_documents(collector),
        "llm_calls": collector.llm_calls,
        "generation": {
            "executed": generate,
            "prompt": (final_prompt_call or {}).get("prompt", {}),
            "input_payload": (final_prompt_call or {}).get("payload", {}),
            "raw_response": (final_prompt_call or {}).get("result") if generate else None,
            "post_guard_response": json_safe(response) if generate else None,
            "rendered_answer": _final_answer_text(profile, response) if generate else "",
        },
        "error": error,
    }


def run_agentic(
    engine: RAGEngine,
    question: Question,
    variant: Variant,
    generate: bool,
    planner_cache: dict[str, Any],
) -> dict[str, Any]:
    collector = TraceCollector()
    agent = AgenticRAG(shared_engine=engine)
    original_support_chain = engine.support_chain
    original_multi_chain = engine.multi_query_chain
    original_reflection_chain = engine.retrieval_reflection_chain
    engine.support_chain = TracedChain(
        original_support_chain,
        collector,
        "final_support_generation",
        invoke_inner=generate,
        fallback=_dummy_response("deep"),
    )
    if original_multi_chain is not None:
        engine.multi_query_chain = TracedChain(
            original_multi_chain,
            collector,
            "multi_query_planner",
            result_cache=planner_cache,
            cache_key_factory=lambda payload: (
                f"{question.question_id}|agentic|{question.query}|"
                f"structured={settings.structured_query_planner_enabled}"
            ),
        )
    engine.retrieval_reflection_chain = TracedChain(
        original_reflection_chain,
        collector,
        "agentic_reflection",
    )

    modules = detect_modules_in_query(question.query)
    retrieval_query = build_module_enriched_query(question.query, modules)
    started = time.perf_counter()
    result = None
    response = None
    error = ""
    effective: dict[str, bool] = {}
    try:
        with applied_variant(engine, variant) as effective_switches:
            effective = dict(effective_switches)
            with PipelineInstrumentation(collector), instrument_formatting_and_filters(engine, collector):
                result = agent.run(question.query)
                response = result.answer
                collector.agent_trace = json_safe(result.trace)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        engine.support_chain = original_support_chain
        engine.multi_query_chain = original_multi_chain
        engine.retrieval_reflection_chain = original_reflection_chain

    final_prompt_call = next(
        (item for item in reversed(collector.llm_calls) if item["label"] == "final_support_generation"),
        None,
    )
    return {
        "run_id": f"{question.question_id}:agentic:{variant.name}",
        "question_id": question.question_id,
        "query": question.query,
        "question_metadata": json_safe(question.metadata),
        "profile": "agentic",
        "variant": variant.name,
        "switches": effective,
        "generate": generate,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "query_processing": {
            "detected_modules": json_safe(modules),
            "module_context": format_detected_modules(modules),
            "retrieval_query": retrieval_query,
        },
        "events": collector.events,
        "filters": collector.filters,
        "formatted_contexts": collector.formatted_contexts,
        "final_context": {
            "docs": _last_context(collector, "docs"),
            "tickets": _last_context(collector, "tickets"),
        },
        "final_documents": _final_documents(collector),
        "llm_calls": collector.llm_calls,
        "agent": {
            "trace": collector.agent_trace,
            "stop_reason": getattr(result, "stop_reason", "") if result else "",
            "rounds_used": getattr(result, "rounds_used", 0) if result else 0,
            "tool_calls_used": getattr(result, "tool_calls_used", 0) if result else 0,
            "diagnostics": json_safe(getattr(result, "diagnostics", {})) if result else {},
        },
        "generation": {
            "executed": generate,
            "prompt": (final_prompt_call or {}).get("prompt", {}),
            "input_payload": (final_prompt_call or {}).get("payload", {}),
            "raw_response": (final_prompt_call or {}).get("result") if generate else None,
            "post_guard_response": json_safe(response) if generate else None,
            "rendered_answer": _final_answer_text("deep", response) if generate else "",
        },
        "error": error,
    }


def _best_stage_rank(run: dict[str, Any], stage_pattern: str) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for event in run.get("events", []):
        if stage_pattern not in str(event.get("stage", "")):
            continue
        for document in event.get("documents", []):
            key = document.get("key", "")
            rank = int(document.get("rank") or 0)
            if key and rank and (key not in ranks or rank < ranks[key]):
                ranks[key] = rank
    return ranks


def build_comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    question_ids = list(dict.fromkeys(run["question_id"] for run in runs))
    for question_id in question_ids:
        selected_runs = [run for run in runs if run["question_id"] == question_id]
        columns = [f"{run['profile']} / {run['variant']}" for run in selected_runs]
        identities: dict[str, dict[str, Any]] = {}
        run_maps: list[dict[str, Any]] = []
        for run in selected_runs:
            final_ranks: dict[str, int] = {}
            for source in ("docs", "tickets"):
                for document in run.get("final_documents", {}).get(source, []):
                    key = document.get("key", "")
                    final_ranks[key] = int(document.get("rank") or 0)
                    metadata = document.get("metadata", {})
                    identities.setdefault(key, {
                        "key": key,
                        "source": source,
                        "title": (
                            metadata.get("ticket_id")
                            or metadata.get("page_title")
                            or metadata.get("title")
                            or metadata.get("source_file")
                            or key
                        ),
                    })
            vector_ranks = _best_stage_rank(run, "qdrant_hybrid_search")
            rerank_ranks = _best_stage_rank(run, "rerank")
            for event in run.get("events", []):
                for document in event.get("documents", []):
                    metadata = document.get("metadata", {})
                    key = document.get("key", "")
                    identities.setdefault(key, {
                        "key": key,
                        "source": event.get("source", ""),
                        "title": (
                            metadata.get("ticket_id")
                            or metadata.get("page_title")
                            or metadata.get("title")
                            or metadata.get("source_file")
                            or key
                        ),
                    })
            run_maps.append({
                "final": final_ranks,
                "vector": vector_ranks,
                "rerank": rerank_ranks,
            })

        rows = []
        for key, identity in identities.items():
            cells = []
            for mapping in run_maps:
                cells.append({
                    "final_rank": mapping["final"].get(key),
                    "vector_rank": mapping["vector"].get(key),
                    "rerank_rank": mapping["rerank"].get(key),
                })
            if any(cell["final_rank"] for cell in cells):
                rows.append({**identity, "cells": cells})
        rows.sort(key=lambda row: (
            min((cell["final_rank"] for cell in row["cells"] if cell["final_rank"]), default=9999),
            str(row["title"]),
        ))
        comparisons.append({
            "question_id": question_id,
            "query": selected_runs[0]["query"],
            "columns": columns,
            "rows": rows,
            "legend": "F = final context rank, V = best Qdrant result rank, R = best reranker rank",
        })
    return comparisons


def h(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _score_text(document: dict[str, Any]) -> str:
    scores = document.get("scores", {}) or {}
    if not scores:
        return "scores unavailable"
    return " · ".join(f"{h(key)}={h(round(value, 6) if isinstance(value, float) else value)}" for key, value in scores.items())


def render_document(document: dict[str, Any], css_class: str = "") -> str:
    metadata = document.get("metadata", {}) or {}
    title = (
        metadata.get("ticket_id")
        or metadata.get("page_title")
        or metadata.get("title")
        or metadata.get("source_file")
        or document.get("key")
    )
    search_text = " ".join((
        str(title or ""),
        str(document.get("page_content") or ""),
        json.dumps(metadata, ensure_ascii=False),
    )).lower()
    return f"""
    <details class="document {h(css_class)}" data-search="{h(search_text)}">
      <summary>
        <span class="rank">#{h(document.get('rank') or '—')}</span>
        <strong>{h(title)}</strong>
        <span class="score">{_score_text(document)}</span>
      </summary>
      <div class="document-grid">
        <section><h5>Полный текст</h5><pre>{h(document.get('page_content', ''))}</pre></section>
        <section><h5>Все метаданные</h5><pre>{h(json.dumps(metadata, ensure_ascii=False, indent=2))}</pre></section>
      </div>
      <div class="key">stable key: {h(document.get('key'))}</div>
    </details>
    """


def render_event(event: dict[str, Any]) -> str:
    documents = "".join(render_document(item) for item in event.get("documents", []))
    removed = "".join(render_document(item, "removed") for item in event.get("removed", []))
    removed_block = f"<details><summary>Исключено на этапе: {len(event.get('removed', []))}</summary>{removed}</details>" if removed else ""
    return f"""
    <article class="stage">
      <header>
        <span class="stage-index">{h(event.get('index'))}</span>
        <div><h4>{h(event.get('stage'))}</h4>
        <div class="muted">{h(event.get('source'))} · {h(event.get('elapsed_seconds'))} с ·
        {h(event.get('input_count'))} → {h(event.get('output_count'))}</div></div>
      </header>
      <div class="query"><b>Запрос:</b> {h(event.get('query') or '—')}</div>
      <details><summary>Параметры этапа</summary><pre>{h(json.dumps(event.get('details', {}), ensure_ascii=False, indent=2))}</pre></details>
      <div class="documents">{documents or '<p class="empty">Документы отсутствуют</p>'}</div>
      {removed_block}
    </article>
    """


def render_filter(item: dict[str, Any]) -> str:
    removed = "".join(render_document(document, "removed") for document in item.get("removed", []))
    return f"""
    <details class="filter-block">
      <summary>{h(item.get('stage'))}: {h(item.get('input_count'))} → {h(item.get('output_count'))}; исключено {len(item.get('removed', []))}</summary>
      <pre>{h(json.dumps({key: value for key, value in item.items() if key != 'removed'}, ensure_ascii=False, indent=2))}</pre>
      {removed or '<p class="empty">Ничего не исключено</p>'}
    </details>
    """


def render_context(context: dict[str, Any]) -> str:
    documents = "".join(render_document(item) for item in context.get("documents", []))
    return f"""
    <article class="context-card">
      <h4>{h(context.get('source'))}: {h(context.get('characters'))} символов</h4>
      <details open><summary>Точный текст, переданный LLM</summary><pre>{h(context.get('text', ''))}</pre></details>
      <details><summary>Документы, из которых собран контекст ({len(context.get('documents', []))})</summary>{documents}</details>
      <details><summary>Ограничения форматтера</summary><pre>{h(json.dumps(context.get('kwargs', {}), ensure_ascii=False, indent=2))}</pre></details>
    </article>
    """


def render_llm_call(call: dict[str, Any]) -> str:
    if call.get("replayed"):
        badge = "воспроизведён из baseline"
    else:
        badge = "выполнен" if call.get("executed") else "пропущен"
    return f"""
    <details class="llm-call">
      <summary>{h(call.get('label'))} · {badge} · {h(call.get('elapsed_seconds'))} с</summary>
      <h5>Отрендеренный prompt</h5><pre>{h(call.get('prompt', {}).get('text', ''))}</pre>
      <h5>Входные поля</h5><pre>{h(json.dumps(call.get('payload', {}), ensure_ascii=False, indent=2))}</pre>
      <h5>Сырой структурированный результат</h5><pre>{h(json.dumps(call.get('result'), ensure_ascii=False, indent=2))}</pre>
      <h5>Токены и callback-события</h5><pre>{h(json.dumps(call.get('usage', {}), ensure_ascii=False, indent=2))}</pre>
      {f'<p class="error">{h(call.get("error"))}</p>' if call.get('error') else ''}
    </details>
    """


def render_run(run: dict[str, Any], index: int) -> str:
    switches = "".join(
        f'<span class="pill">{h(key)}={"on" if value else "off"}</span>'
        for key, value in run.get("switches", {}).items()
    )
    events = "".join(render_event(event) for event in run.get("events", []))
    filters = "".join(render_filter(item) for item in run.get("filters", []))
    contexts = "".join(render_context(run.get("final_context", {}).get(source, {})) for source in ("docs", "tickets"))
    llm_calls = "".join(render_llm_call(call) for call in run.get("llm_calls", []))
    generation = run.get("generation", {})
    agent = run.get("agent")
    agent_block = ""
    if agent:
        agent_block = f"""
        <section class="panel"><h3>Agentic trace</h3>
        <div class="metrics"><span>stop={h(agent.get('stop_reason'))}</span><span>rounds={h(agent.get('rounds_used'))}</span><span>tools={h(agent.get('tool_calls_used'))}</span></div>
        <pre>{h(json.dumps(agent.get('trace', []), ensure_ascii=False, indent=2))}</pre></section>
        """
    error = f'<p class="error">{h(run.get("error"))}</p>' if run.get("error") else ""
    return f"""
    <section class="run" id="run-{index}">
      <header class="run-header">
        <div><div class="eyebrow">{h(run.get('question_id'))} · {h(run.get('run_id'))}</div>
        <h2>{h(run.get('query'))}</h2></div>
        <div class="run-badges"><span class="profile">{h(run.get('profile'))}</span><span class="variant">{h(run.get('variant'))}</span></div>
      </header>
      <div class="metrics"><span>{h(run.get('elapsed_seconds'))} с</span>{switches}<span>generation={'on' if run.get('generate') else 'off'}</span></div>
      {error}
      <section class="panel"><h3>1. Обработка запроса</h3><pre>{h(json.dumps(run.get('query_processing', {}), ensure_ascii=False, indent=2))}</pre></section>
      <section class="panel"><h3>2. Этапы retrieval</h3>
        <input class="candidate-search" placeholder="Фильтр документов по тексту или метаданным…" oninput="filterDocuments(this)">
        {events or '<p class="empty">Retrieval events отсутствуют</p>'}
      </section>
      <section class="panel"><h3>3. Детерминированные фильтры</h3>{filters or '<p class="empty">Фильтры не вызывались</p>'}</section>
      <section class="panel"><h3>4. Финальный контекст для answer LLM</h3>{contexts}</section>
      <section class="panel"><h3>5. LLM-вызовы</h3>{llm_calls or '<p class="empty">LLM-вызовы не зафиксированы</p>'}</section>
      {agent_block}
      <section class="panel"><h3>6. Генерация и post-guards</h3>
        <details open><summary>Итоговый текст интерфейса</summary><pre>{h(generation.get('rendered_answer', '') or 'Генерация отключена')}</pre></details>
        <details><summary>Сырой ответ LLM</summary><pre>{h(json.dumps(generation.get('raw_response'), ensure_ascii=False, indent=2))}</pre></details>
        <details><summary>Ответ после guards</summary><pre>{h(json.dumps(generation.get('post_guard_response'), ensure_ascii=False, indent=2))}</pre></details>
      </section>
    </section>
    """


def render_comparison(comparison: dict[str, Any]) -> str:
    headers = "".join(f"<th>{h(column)}</th>" for column in comparison.get("columns", []))
    rows = []
    for row in comparison.get("rows", []):
        cells = []
        for cell in row.get("cells", []):
            text = " · ".join(
                value
                for value in (
                    f"F{cell['final_rank']}" if cell.get("final_rank") else "",
                    f"V{cell['vector_rank']}" if cell.get("vector_rank") else "",
                    f"R{cell['rerank_rank']}" if cell.get("rerank_rank") else "",
                )
                if value
            ) or "—"
            cells.append(f"<td>{h(text)}</td>")
        rows.append(f"<tr><td>{h(row.get('source'))}</td><td>{h(row.get('title'))}</td>{''.join(cells)}</tr>")
    return f"""
    <section class="comparison-block">
      <h3>{h(comparison.get('question_id'))}: {h(comparison.get('query'))}</h3>
      <p class="muted">{h(comparison.get('legend'))}</p>
      <div class="table-scroll"><table><thead><tr><th>Источник</th><th>Документ</th>{headers}</tr></thead>
      <tbody>{''.join(rows) or '<tr><td colspan="99">Финальные документы отсутствуют</td></tr>'}</tbody></table></div>
    </section>
    """


REPORT_CSS = r"""
:root { color-scheme:dark; --bg:#0b1020; --panel:#131b2e; --panel2:#19243a; --line:#2b3a58; --text:#e7edf7; --muted:#94a3b8; --accent:#65d5c5; --blue:#7bb8ff; --warn:#ffcc66; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.5 Inter,Segoe UI,Arial,sans-serif; }
a { color:var(--blue); }
pre { white-space:pre-wrap; overflow-wrap:anywhere; margin:.7rem 0; padding:1rem; background:#09101e; border:1px solid var(--line); border-radius:8px; max-height:36rem; overflow:auto; }
summary { cursor:pointer; padding:.45rem 0; }
h1,h2,h3,h4,h5 { margin:.25rem 0 .65rem; }
.topbar { position:sticky; top:0; z-index:10; padding:1rem 1.5rem; background:rgba(11,16,32,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }
.topbar h1 { font-size:1.25rem; }
.layout { display:grid; grid-template-columns:260px minmax(0,1fr); gap:1.25rem; max-width:1800px; margin:auto; padding:1.25rem; }
nav { position:sticky; top:90px; align-self:start; max-height:calc(100vh - 110px); overflow:auto; padding:1rem; background:var(--panel); border:1px solid var(--line); border-radius:12px; }
nav a { display:block; padding:.42rem .55rem; text-decoration:none; border-radius:6px; }
nav a:hover { background:var(--panel2); }
.content { min-width:0; }
.intro,.run,.comparison { margin-bottom:1.25rem; padding:1.25rem; background:var(--panel); border:1px solid var(--line); border-radius:14px; }
.run-header { display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; padding-bottom:1rem; border-bottom:1px solid var(--line); }
.run-header h2 { font-size:1.25rem; max-width:1100px; }
.eyebrow,.muted,.key { color:var(--muted); font-size:.85rem; }
.run-badges,.metrics { display:flex; gap:.5rem; flex-wrap:wrap; align-items:center; }
.profile,.variant,.pill,.metrics>span { padding:.22rem .55rem; border:1px solid var(--line); border-radius:999px; background:var(--panel2); white-space:nowrap; }
.profile { color:var(--accent); } .variant { color:var(--warn); }
.metrics { margin:.9rem 0; }
.panel { margin:1rem 0; padding:1rem; background:#10182a; border:1px solid var(--line); border-radius:10px; }
.stage { margin:.9rem 0; padding:1rem; background:var(--panel2); border:1px solid var(--line); border-radius:9px; }
.stage>header { display:flex; gap:.75rem; align-items:center; }
.stage-index,.rank { display:inline-grid; place-items:center; min-width:2rem; height:2rem; border-radius:50%; background:#243653; color:var(--accent); }
.query { margin:.65rem 0; padding:.6rem .8rem; border-left:3px solid var(--blue); background:#111b2e; }
.document { margin:.45rem 0; padding:.3rem .7rem; border:1px solid #334765; border-radius:8px; background:#0e1728; }
.document summary { display:flex; align-items:center; gap:.65rem; }
.document summary strong { flex:1; } .score { color:var(--warn); font-size:.82rem; }
.document-grid { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(320px,1fr); gap:.75rem; }
.document.removed { border-color:#75404a; opacity:.86; }
.context-card { margin:.8rem 0; padding:.8rem; border:1px solid var(--line); border-radius:8px; }
.candidate-search { width:100%; padding:.72rem; color:var(--text); background:#09101e; border:1px solid var(--line); border-radius:8px; }
.error { color:#ffd7da; background:#3a1720; border:1px solid #74303c; padding:.75rem; border-radius:8px; }
.empty { color:var(--muted); font-style:italic; }
.local-note { color:var(--accent); }
.comparison-block { margin:1rem 0; } .table-scroll { overflow:auto; }
table { width:100%; border-collapse:collapse; background:#0d1627; } th,td { padding:.55rem; border:1px solid var(--line); text-align:left; vertical-align:top; } th { position:sticky; top:0; background:#1a2841; }
.hidden-by-filter { display:none; }
@media(max-width:1000px) { .layout { grid-template-columns:1fr; } nav { position:static; max-height:none; } .document-grid { grid-template-columns:1fr; } }
"""


def render_html(report: dict[str, Any]) -> str:
    runs = report.get("runs", [])
    comparisons = report.get("comparisons", [])
    navigation = "".join(
        f'<a href="#run-{index}">{h(run.get("question_id"))} · {h(run.get("profile"))}/{h(run.get("variant"))}</a>'
        for index, run in enumerate(runs, start=1)
    )
    comparison_html = "".join(render_comparison(item) for item in comparisons)
    runs_html = "".join(render_run(run, index) for index, run in enumerate(runs, start=1))
    metadata_json = json.dumps(report.get("metadata", {}), ensure_ascii=False, indent=2)
    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RegLab RAG retrieval debug report</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="topbar"><h1>RegLab RAG · внутренний отчёт retrieval и LLM-контекста</h1></div>
<div class="layout">
<nav><b>Прогоны</b>{navigation}<hr><a href="#comparison">Сравнение</a></nav>
<main class="content">
  <section class="intro">
    <h2>О запуске</h2>
    <p class="local-note">Отчёт самодостаточный и не загружает внешние ресурсы. Он может содержать полный текст внутренних документов и тикетов.</p>
    <p>Отдельные dense/BM25 scores не доступны из текущего production-вызова LangChain/Qdrant: фиксируются фактический порядок hybrid search, RRF score и CrossEncoder score, когда они присутствуют.</p>
    <details><summary>Метаданные запуска</summary><pre>{h(metadata_json)}</pre></details>
  </section>
  <section class="comparison" id="comparison"><h2>Сравнение профилей и вариантов</h2>{comparison_html}</section>
  {runs_html}
</main>
</div>
<script>
function filterDocuments(input) {{
  const value = input.value.trim().toLowerCase();
  const panel = input.closest('.panel');
  panel.querySelectorAll('.document').forEach(el => {{
    el.classList.toggle('hidden-by-filter', value && !el.dataset.search.includes(value));
  }});
}}
</script>
</body></html>"""


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def initialize_engines(profiles: list[str]) -> dict[str, RAGEngine]:
    """Load local models once and ensure a planner chain exists for variant toggles."""
    original_config = copy.deepcopy(settings._raw_config)
    settings._raw_config.setdefault("retrieval", {}).setdefault("multi_query", {})["enabled"] = True
    try:
        first_profile = "adaptive" if "adaptive" in profiles else "deep"
        first = RAGEngine(first_profile)
        engines = {first_profile: first}
        shared_embeddings = first.dense_embeddings
        shared_reranker = first.rerank_model
        if "deep" in profiles or "agentic" in profiles:
            if first_profile != "deep":
                engines["deep"] = RAGEngine(
                    "deep",
                    shared_embeddings=shared_embeddings,
                    shared_reranker=shared_reranker,
                )
        if "adaptive" in profiles and first_profile != "adaptive":
            engines["adaptive"] = RAGEngine(
                "adaptive",
                shared_embeddings=shared_embeddings,
                shared_reranker=shared_reranker,
            )
        return engines
    finally:
        settings._raw_config = original_config


def config_snapshot() -> dict[str, Any]:
    return {
        "active_llm": settings.active_llm,
        "llm_model": settings.llm_model_name,
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
        "embedding_device": settings.device,
        "docs_backend": settings.active_db_name,
        "tickets_backend": settings.second_db_name,
        "retrieval": json_safe(settings._raw_config.get("retrieval", {})),
    }


def main() -> None:
    args = parse_args()
    profiles = parse_profiles(args.profiles)
    questions = load_questions(args.query, args.input, args.limit)
    variants = build_variants(args.variant, args.compare)
    snapshot = config_snapshot()
    engines = initialize_engines(profiles)

    runs: list[dict[str, Any]] = []
    planner_cache: dict[str, Any] = {}
    total = len(questions) * len(profiles) * len(variants)
    counter = 0
    for question in questions:
        for profile in profiles:
            for variant in variants:
                counter += 1
                print(f"[{counter}/{total}] {question.question_id} | {profile} | {variant.name}", flush=True)
                if profile == "agentic":
                    run = run_agentic(
                        engines["deep"],
                        question,
                        variant,
                        args.generate,
                        planner_cache,
                    )
                else:
                    run = run_standard(
                        engines[profile],
                        question,
                        profile,
                        variant,
                        args.generate,
                        planner_cache,
                    )
                runs.append(run)
                if run.get("error"):
                    print(f"  ERROR: {run['error']}", flush=True)
                else:
                    docs_count = len(run.get("final_documents", {}).get("docs", []))
                    tickets_count = len(run.get("final_documents", {}).get("tickets", []))
                    print(f"  OK: {run['elapsed_seconds']}s docs={docs_count} tickets={tickets_count}", flush=True)

    report = {
        "metadata": {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "working_directory": str(Path.cwd()),
            "profiles": profiles,
            "variants": [json_safe(variant.__dict__) for variant in variants],
            "generate": args.generate,
            "question_count": len(questions),
            "run_count": len(runs),
            "config": snapshot,
            "notes": [
                "HTML is local and self-contained.",
                "Full internal document/ticket text and metadata may be present.",
                "Without --generate only final answer synthesis is skipped; enabled planners/reflections may still call an LLM.",
                "Current production similarity_search does not expose separate dense and sparse/BM25 component scores.",
            ],
        },
        "questions": [json_safe(question.__dict__) for question in questions],
        "runs": runs,
        "comparisons": build_comparisons(runs),
    }
    basename = args.name.strip() or f"rag_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_path = args.output_dir / f"{basename}.json"
    html_path = args.output_dir / f"{basename}.html"
    atomic_write(json_path, json.dumps(report, ensure_ascii=False, indent=2))
    atomic_write(html_path, render_html(report))
    print(f"JSON: {json_path.resolve()}")
    print(f"HTML: {html_path.resolve()}")


if __name__ == "__main__":
    main()
