"""Fast regression checks for deterministic RAG guards.

Run with the project interpreter:
    & ...\python.exe scripts\test_engine_guards.py
"""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context_formatting import SourceReference
from src.engine import RAGEngine, RoundRobinFallbackRunnable
from src.evidence_guard import apply_response_provenance


def test_product_filter_keeps_exact_series() -> None:
    r500 = SimpleNamespace(metadata={"equipment_type": "R500", "source_file": "R500_guide.htm"})
    r500s = SimpleNamespace(metadata={"equipment_type": "R500S", "source_file": "R500S_guide.htm"})
    result = RAGEngine._filter_wiki_docs_by_requested_product(
        None, "как выполнить сброс R500?", [], [r500, r500s]
    )
    assert result == [r500]


def test_provenance_removes_unknown_sources() -> None:
    response = SimpleNamespace(
        docs_answer="Подтверждённый факт [D1], выдуманный [D99].",
        draft_private_comment="Исторический тикет [T1], выдуманный [T9].",
        evidence_notes=[
            SimpleNamespace(claim="ok", source_ids=["D1"]),
            SimpleNamespace(claim="bad", source_ids=["D9"]),
        ],
        similar_tickets=[
            SimpleNamespace(ticket_id="RL-1", source_file="ticket.json"),
            SimpleNamespace(ticket_id="RL-X", source_file="unknown.json"),
        ],
    )
    docs = [SourceReference(source_id="D1", title="guide", source_file="guide.htm")]
    tickets = [SourceReference(source_id="T1", title="RL-1", source_file="ticket.json")]
    response, used_docs, used_tickets = apply_response_provenance(response, docs, tickets)
    assert "[D99]" not in response.docs_answer
    assert "[T9]" not in response.draft_private_comment
    assert len(response.evidence_notes) == 1
    assert len(response.similar_tickets) == 1
    assert [source.source_id for source in used_docs] == ["D1"]
    assert [source.source_id for source in used_tickets] == ["T1"]



def test_retry_policy_does_not_repeat_schema_errors() -> None:
    assert RoundRobinFallbackRunnable._failure_policy(RuntimeError("503 UNAVAILABLE")) == (True, 30.0)
    assert RoundRobinFallbackRunnable._failure_policy(RuntimeError("404 NOT_FOUND")) == (True, 900.0)
    assert RoundRobinFallbackRunnable._failure_policy(ValueError("schema validation error")) == (False, 0.0)


if __name__ == "__main__":
    test_product_filter_keeps_exact_series()
    test_provenance_removes_unknown_sources()
    test_retry_policy_does_not_repeat_schema_errors()
    print("Engine guard regression checks: OK")