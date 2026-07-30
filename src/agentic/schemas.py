"""Structured public results and trace records for Agentic RAG."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AgentTraceStep(BaseModel):
    round_index: int = 0
    phase: str
    tool: str = ""
    query: str = ""
    reason: str = ""
    result_count: int = 0
    source_keys: List[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class AgenticRAGResult(BaseModel):
    answer: Any
    trace: List[AgentTraceStep] = Field(default_factory=list)
    stop_reason: str
    rounds_used: int = 0
    tool_calls_used: int = 0
    elapsed_seconds: float = 0.0
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
