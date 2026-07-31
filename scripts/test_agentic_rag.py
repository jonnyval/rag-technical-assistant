"""Fast unit checks for the experimental Agentic RAG layer."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agentic.controller import AgenticController
from src.agentic.tools import AgenticRetrievalTools
from src.context_formatting import SourceReference
from src.evidence_guard import apply_response_provenance


def test_agentic_diagnostic_classifier_covers_connection_loss() -> None:
    engine = SimpleNamespace(_is_incident_query=lambda query: False)
    controller = AgenticController.__new__(AgenticController)
    controller.engine = engine
    assert controller._is_diagnostic_query(
        "После перезагрузки сбрасываются флаги и теряется связь. Как исправить?"
    )


def test_exact_identifier_query_is_derived_from_user_question() -> None:
    query = AgenticController._build_exact_identifier_query(
        "Как преобразовать SYS_MEM_COPY в массив ASCII кодов?",
        ["SYS_MEM_COPY", "ASCII"],
        [],
    )
    assert query.startswith("SYS_MEM_COPY ASCII")
    assert "преобразовать" in query and "массив" in query
    assert "перезагрузка" not in query and "резервирование" not in query


def test_ticket_ranking_uses_exact_and_cyrillic_terms() -> None:
    exact = SimpleNamespace(
        page_content="R500: после перезагрузки сбрасываются IS_ACTIVE и IS_CPU_A",
        metadata={"rerank_score": 0.1},
    )
    generic = SimpleNamespace(
        page_content="Общие сведения о резервировании контроллеров",
        metadata={"rerank_score": 0.9},
    )
    ranked = AgenticController._rank_agent_tickets(
        "R500 перезагрузка: сбрасываются IS_ACTIVE и IS_CPU_A",
        [generic, exact],
        ["IS_ACTIVE", "IS_CPU_A"],
        limit=2,
    )
    assert ranked == [exact, generic]


def test_multi_query_tool_trace_preserves_each_search() -> None:
    document = SimpleNamespace(
        page_content="evidence",
        metadata={"page_title": "R500 flags"},
    )
    calls = []

    def retrieve(queries, rerank_query, *, use_reranker):
        calls.append((queries, rerank_query, use_reranker))
        return [document]

    tools = AgenticRetrievalTools(
        SimpleNamespace(_retrieve_docs_for_queries=retrieve)
    )
    documents, trace = tools.search_docs_multi(
        ["original R500", "IS_ACTIVE назначение R500"],
        rerank_query="original R500",
        round_index=0,
    )
    assert documents == [document]
    assert calls == [
        (["original R500", "IS_ACTIVE назначение R500"], "original R500", True)
    ]
    assert [step.query for step in trace] == [
        "original R500",
        "IS_ACTIVE назначение R500",
    ]
    assert all(step.tool == "search_docs" for step in trace)

def test_ticket_provenance_recognizes_id_in_url() -> None:
    response = SimpleNamespace(
        docs_answer="",
        draft_private_comment="Исторический тикет [T1].",
        evidence_notes=[],
        similar_tickets=[
            SimpleNamespace(ticket_id="RL-8318", source_file="legacy-export.json"),
        ],
    )
    sources = [
        SourceReference(
            source_id="T1",
            title="Диагностика резервирования",
            source_file="ticket.json",
            url="https://support.example/Task/RL-8318",
        )
    ]
    response, _, used_tickets = apply_response_provenance(response, [], sources)
    assert len(response.similar_tickets) == 1
    assert [source.source_id for source in used_tickets] == ["T1"]


if __name__ == "__main__":
    test_agentic_diagnostic_classifier_covers_connection_loss()
    test_exact_identifier_query_is_derived_from_user_question()
    test_ticket_ranking_uses_exact_and_cyrillic_terms()
    test_multi_query_tool_trace_preserves_each_search()
    test_ticket_provenance_recognizes_id_in_url()
    print("Agentic RAG unit checks: OK")
