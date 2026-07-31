"""Fast regression checks for deterministic RAG guards.

Run with the project interpreter:
    & ...\python.exe scripts\test_engine_guards.py
"""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context_formatting import SourceReference
from src.engine import MultiQueryPlan, RAGEngine, RoundRobinFallbackRunnable
from src.evidence_guard import (
    apply_response_provenance, apply_transformation_evidence_guard,
    apply_entity_coverage_guard, apply_diagnostic_scope_guard,
    filter_documents_by_requested_series,
)


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




def test_ascii_transformation_guard_rejects_lookalike_api() -> None:
    response = SimpleNamespace(
        docs_answer="EscapeString решает задачу.",
        draft_private_comment="Используйте EscapeString.",
        evidence_notes=[],
        confidence="high",
    )
    result = apply_transformation_evidence_guard(
        response,
        "преобразование строки в набор ASCII кодов",
        "В документации описана функция EscapeString для SQL-строки.",
        [SourceReference(source_id="D1", title="functions", source_file="guide.docx")],
    )
    assert "нет прямого подтверждения" in result.draft_private_comment
    assert result.confidence == "low"


def test_series_filter_drops_r500_for_r400_request() -> None:
    r400 = SimpleNamespace(metadata={"equipment_type": "R400"}, page_content="R400 service mode")
    r500 = SimpleNamespace(metadata={"equipment_type": "R500"}, page_content="R500 reset")
    result = filter_documents_by_requested_series("как сделать сброс R400?", [r400, r500])
    assert result == [r400]


def test_entity_and_diagnostic_guards_require_direct_scope() -> None:
    response = SimpleNamespace(docs_answer="R500 details", draft_private_comment="R500 details", evidence_notes=[], confidence="high")
    result = apply_entity_coverage_guard(response, "чем отличается R400 от R500?", "source about R500 only")
    assert "R400" in result.draft_private_comment and result.confidence == "low"

    response = SimpleNamespace(docs_answer="generic", draft_private_comment="generic", evidence_notes=[], confidence="high")
    result = apply_diagnostic_scope_guard(response, "ошибка self-diagnostic что делать?", "general runtime guide")
    assert "точный текст сообщения" in result.draft_private_comment and result.confidence == "low"
def test_structured_query_planner_separates_source_queries() -> None:
    class FakePlanner:
        def invoke(self, payload):
            assert "documentation_queries" in payload["planner_instructions"]
            return MultiQueryPlan(
                entities=["R500", "IS_ACTIVE"],
                documentation_queries=[
                    "назначение IS_ACTIVE резервирование R500",
                    "параметры CPU_A CPU_B R500",
                ],
                ticket_queries=[
                    "R500 после перезагрузки сбрасывается IS_ACTIVE",
                ],
            )

    from src.config import settings

    structured = settings.multi_query_config.setdefault("structured_planner", {})
    previous = structured.get("enabled", False)
    structured["enabled"] = True
    try:
        engine = RAGEngine.__new__(RAGEngine)
        engine.multi_query_chain = FakePlanner()
        docs, tickets = engine._build_multi_query_searches(
            "После перезагрузки R500 сбрасывается IS_ACTIVE",
            "После перезагрузки R500 сбрасывается IS_ACTIVE",
            [],
            "No module was detected.",
        )
    finally:
        structured["enabled"] = previous

    assert len(docs) == 3
    assert len(tickets) == 2
    assert docs != tickets
    assert all("R500" in query for query in [*docs, *tickets])
    assert any("назначение" in query for query in docs)
    assert any("сбрасывается" in query for query in tickets)


def test_legacy_multi_query_remains_shared() -> None:
    class FakePlanner:
        def invoke(self, payload):
            assert "Legacy mode" in payload["planner_instructions"]
            return MultiQueryPlan(queries=["ошибка загрузки проекта R500"])

    from src.config import settings

    structured = settings.multi_query_config.setdefault("structured_planner", {})
    previous = structured.get("enabled", False)
    structured["enabled"] = False
    try:
        engine = RAGEngine.__new__(RAGEngine)
        engine.multi_query_chain = FakePlanner()
        docs, tickets = engine._build_multi_query_searches(
            "download denied R500",
            "download denied R500",
            [],
            "No module was detected.",
        )
    finally:
        structured["enabled"] = previous

    assert docs == tickets
    assert docs == ["download denied R500", "ошибка загрузки проекта R500"]

def test_query_variant_validation_preserves_requested_series() -> None:
    queries = RAGEngine._prepare_query_variants(
        "заводской сброс R400",
        "заводской сброс R400",
        ["сервисный режим", "сервисный режим", "x"],
        [],
        3,
    )
    assert queries == ["заводской сброс R400", "сервисный режим R400"]

def test_retry_policy_does_not_repeat_schema_errors() -> None:
    assert RoundRobinFallbackRunnable._failure_policy(RuntimeError("503 UNAVAILABLE")) == (True, 30.0)
    assert RoundRobinFallbackRunnable._failure_policy(RuntimeError("404 NOT_FOUND")) == (True, 900.0)
    assert RoundRobinFallbackRunnable._failure_policy(ValueError("schema validation error")) == (False, 0.0)


if __name__ == "__main__":
    test_product_filter_keeps_exact_series()
    test_provenance_removes_unknown_sources()
    test_ascii_transformation_guard_rejects_lookalike_api()
    test_series_filter_drops_r500_for_r400_request()
    test_entity_and_diagnostic_guards_require_direct_scope()
    test_structured_query_planner_separates_source_queries()
    test_legacy_multi_query_remains_shared()
    test_query_variant_validation_preserves_requested_series()
    test_retry_policy_does_not_repeat_schema_errors()
    print("Engine guard regression checks: OK")