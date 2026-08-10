"""Regression checks for the local RAG debug report generator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from scripts.generate_rag_debug_report import (
    Question,
    Variant,
    build_comparisons,
    build_variants,
    parse_profiles,
    parse_variant,
    render_html,
    serialize_document,
)


def test_cli_parsing() -> None:
    assert parse_profiles("adaptive,deep,agentic,adaptive") == ["adaptive", "deep", "agentic"]
    variant = parse_variant("no-rerank:reranker=off,multi_query=on")
    assert variant == Variant("no-rerank", {"reranker": False, "multi_query": True})
    variants = build_variants(["no-mq:multi_query=off"], "reranker")
    assert variants[0].name == "baseline"
    assert any(item.name == "no-mq" for item in variants)


def test_document_serialization_keeps_full_metadata() -> None:
    document = Document(
        page_content="полный текст",
        metadata={
            "ticket_id": "RL-1",
            "qdrant_score": 0.75,
            "llm_solution": ["решение"],
        },
    )
    serialized = serialize_document(document, rank=2)
    assert serialized["rank"] == 2
    assert serialized["page_content"] == "полный текст"
    assert serialized["metadata"]["llm_solution"] == ["решение"]
    assert serialized["scores"]["qdrant_score"] == 0.75


def test_comparison_and_html() -> None:
    document = serialize_document(
        Document(page_content="контекст", metadata={"page_title": "Раздел", "db_source": "docs"}),
        rank=1,
    )
    run = {
        "run_id": "q1:adaptive:baseline",
        "question_id": "q1",
        "query": "вопрос",
        "profile": "adaptive",
        "variant": "baseline",
        "switches": {},
        "generate": False,
        "elapsed_seconds": 1.0,
        "query_processing": {},
        "events": [{
            "index": 1,
            "stage": "qdrant_hybrid_search",
            "source": "docs",
            "query": "вопрос",
            "elapsed_seconds": 0.1,
            "input_count": 0,
            "output_count": 1,
            "details": {},
            "documents": [document],
            "removed": [],
        }],
        "filters": [],
        "final_context": {
            "docs": {"source": "docs", "characters": 7, "documents": [document], "kwargs": {}, "text": "контекст"},
            "tickets": {"source": "tickets", "characters": 0, "documents": [], "kwargs": {}, "text": ""},
        },
        "final_documents": {"docs": [document], "tickets": []},
        "llm_calls": [],
        "generation": {},
        "error": "",
    }
    comparisons = build_comparisons([run])
    assert comparisons[0]["rows"][0]["cells"][0]["final_rank"] == 1
    report = {"metadata": {}, "runs": [run], "comparisons": comparisons}
    output = render_html(report)
    assert "Точный текст, переданный LLM" in output
    assert "Все метаданные" in output
    assert "контекст" in output


if __name__ == "__main__":
    test_cli_parsing()
    test_document_serialization_keeps_full_metadata()
    test_comparison_and_html()
    print("RAG debug report checks: OK")
