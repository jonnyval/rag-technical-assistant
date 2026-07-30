"""Public construction entry point for the experimental Agentic RAG pipeline."""

from __future__ import annotations

from typing import Any

from src.agentic.controller import AgenticController
from src.config import settings
from src.engine import RAGEngine


class AgenticRAG:
    def __init__(self, *, shared_engine: Any | None = None):
        self.engine = shared_engine or RAGEngine("deep")
        config = settings.agentic_rag_config
        self.controller = AgenticController(
            self.engine,
            max_rounds=config.get("max_rounds", 3),
            max_tool_calls=config.get("max_tool_calls", 6),
            max_queries_per_round=config.get("max_queries_per_round", 2),
            timeout_seconds=config.get("timeout_seconds", 90),
        )

    def run(self, query: str):
        return self.controller.run(query)
