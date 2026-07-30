"""Safe retrieval tools exposed to the bounded agent controller."""

from __future__ import annotations

import time
from typing import Any, List

from src.agentic.schemas import AgentTraceStep
from src.context_formatting import is_ticket_document


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

    def search_docs(self, query: str, *, round_index: int, use_reranker: bool = True):
        started = time.perf_counter()
        documents = self.engine.retriever.retrieve_docs(query, use_reranker=use_reranker)
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

    def search_tickets(self, query: str, *, round_index: int, limit: int = 8, child_k: int = 60):
        started = time.perf_counter()
        documents = [
            document
            for document in self.engine.retriever.retrieve_tickets(
                query,
                final_limit=limit,
                child_k=child_k,
                use_reranker=True,
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
