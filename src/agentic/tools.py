"""Safe retrieval tools exposed to the bounded agent controller."""

from __future__ import annotations

import time
from typing import Any, List

from src.agentic.schemas import AgentTraceStep
from src.context_formatting import is_ticket_document
from src.evidence_guard import filter_documents_by_requested_series


def _source_keys(documents: List[Any]) -> List[str]:
    result = []
    for document in documents[:8]:
        metadata = getattr(document, "metadata", {}) or {}
        key = metadata.get("ticket_id") or metadata.get("page_title") or metadata.get("source_file")
        if key:
            result.append(str(key))
    return result


class AgenticRetrievalTools:
    def __init__(self, engine: Any):
        self.engine = engine

    @staticmethod
    def _multi_trace_steps(
        *,
        queries: List[str],
        round_index: int,
        tool: str,
        documents: List[Any],
        elapsed_seconds: float,
    ) -> List[AgentTraceStep]:
        source_keys = _source_keys(documents)
        return [
            AgentTraceStep(
                round_index=round_index,
                phase="tool",
                tool=tool,
                query=query,
                reason="initial structured query plan with RRF fusion",
                result_count=len(documents),
                source_keys=source_keys,
                elapsed_seconds=elapsed_seconds if index == 0 else 0.0,
            )
            for index, query in enumerate(queries)
        ]

    def search_docs_multi(
        self,
        queries: List[str],
        *,
        rerank_query: str,
        round_index: int,
        use_reranker: bool = True,
        source_query: str | None = None,
    ):
        started = time.perf_counter()
        candidate_filter = (
            lambda candidates: filter_documents_by_requested_series(source_query, candidates)
        ) if source_query else None
        retrieval_options = {"use_reranker": use_reranker}
        if candidate_filter is not None:
            retrieval_options["candidate_filter"] = candidate_filter
        documents = self.engine._retrieve_docs_for_queries(
            queries,
            rerank_query,
            **retrieval_options,
        )
        elapsed = round(time.perf_counter() - started, 3)
        return documents, self._multi_trace_steps(
            queries=queries,
            round_index=round_index,
            tool="search_docs",
            documents=documents,
            elapsed_seconds=elapsed,
        )

    def search_tickets_multi(
        self,
        queries: List[str],
        *,
        rerank_query: str,
        round_index: int,
        limit: int = 8,
        child_k: int = 60,
        source_query: str | None = None,
    ):
        started = time.perf_counter()
        candidate_filter = (
            lambda candidates: filter_documents_by_requested_series(source_query, candidates)
        ) if source_query else None
        retrieval_options = {
            "final_limit": limit,
            "child_k": child_k,
            "use_reranker": True,
        }
        if candidate_filter is not None:
            retrieval_options["candidate_filter"] = candidate_filter
        documents = [
            document
            for document in self.engine._retrieve_tickets_for_queries(
                queries,
                rerank_query,
                **retrieval_options,
            )
            if is_ticket_document(document)
        ]
        elapsed = round(time.perf_counter() - started, 3)
        return documents, self._multi_trace_steps(
            queries=queries,
            round_index=round_index,
            tool="search_tickets",
            documents=documents,
            elapsed_seconds=elapsed,
        )
    def search_docs(
        self,
        query: str,
        *,
        round_index: int,
        use_reranker: bool = True,
        source_query: str | None = None,
    ):
        started = time.perf_counter()
        candidate_filter = (
            lambda candidates: filter_documents_by_requested_series(source_query, candidates)
        ) if source_query else None
        documents = self.engine.retriever.retrieve_docs(
            query,
            use_reranker=use_reranker,
            candidate_filter=candidate_filter,
        )
        trace = AgentTraceStep(
            round_index=round_index,
            phase="tool",
            tool="search_docs",
            query=query,
            result_count=len(documents),
            source_keys=_source_keys(documents),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        return documents, trace

    def search_tickets(
        self,
        query: str,
        *,
        round_index: int,
        limit: int = 8,
        child_k: int = 60,
        source_query: str | None = None,
    ):
        started = time.perf_counter()
        candidate_filter = (
            lambda candidates: filter_documents_by_requested_series(source_query, candidates)
        ) if source_query else None
        documents = [
            document
            for document in self.engine.retriever.retrieve_tickets(
                query,
                final_limit=limit,
                child_k=child_k,
                use_reranker=True,
                candidate_filter=candidate_filter,
            )
            if is_ticket_document(document)
        ]
        trace = AgentTraceStep(
            round_index=round_index,
            phase="tool",
            tool="search_tickets",
            query=query,
            result_count=len(documents),
            source_keys=_source_keys(documents),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        return documents, trace
