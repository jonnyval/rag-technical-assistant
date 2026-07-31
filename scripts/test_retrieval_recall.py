"""Fast checks for the retrieval recall diagnostic summary."""

from diagnose_retrieval_recall import (
    build_summary,
    local_recall_judgement,
    trace_query_groups,
)


def test_reranked_hit_implies_vector_hit() -> None:
    summary = build_summary(
        [
            {
                "judgement": {
                    "points": [
                        {
                            "vector_candidates": False,
                            "reranked_selection": True,
                            "final_context": True,
                            "adaptive_answer": False,
                            "agentic_answer": True,
                        }
                    ]
                }
            }
        ]
    )

    assert summary["stage_recall"]["vector_candidates"]["hits"] == 1
    assert summary["question_coverage"]["full_candidate_recall"] == 1
    assert "absent_from_vector_candidates" not in summary["losses"]


def test_loss_categories_are_separated() -> None:
    summary = build_summary(
        [
            {
                "judgement": {
                    "points": [
                        {
                            "vector_candidates": True,
                            "reranked_selection": False,
                            "final_context": True,
                            "adaptive_answer": False,
                            "agentic_answer": False,
                        },
                        {
                            "vector_candidates": True,
                            "reranked_selection": True,
                            "final_context": False,
                            "adaptive_answer": False,
                            "agentic_answer": False,
                        },
                    ]
                }
            }
        ]
    )

    losses = summary["losses"]
    assert losses["recovered_by_parent_expansion"] == 1
    assert losses["lost_after_rerank_before_context"] == 1
    assert losses["candidate_fact_absent_from_final_context"] == 1


def test_trace_separates_initial_rrf_from_followups() -> None:
    item = {
        "runs": {
            "agentic": {
                "agent": {
                    "trace": [
                        {
                            "tool": "search_docs",
                            "query": "initial docs",
                            "reason": "initial structured query plan with RRF fusion",
                        },
                        {
                            "tool": "search_tickets",
                            "query": "initial tickets",
                            "reason": "initial structured query plan with RRF fusion",
                        },
                        {
                            "tool": "search_docs",
                            "query": "follow-up",
                            "reason": "documentation follow-up requested by evidence planner",
                        },
                    ]
                }
            }
        }
    }
    groups = trace_query_groups(item, "fallback")
    assert groups["docs_initial"] == ["initial docs"]
    assert groups["tickets_initial"] == ["initial tickets"]
    assert groups["docs_followup"] == ["follow-up"]
    assert groups["tickets_followup"] == []


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0] for text in texts]


class CountingReranker:
    def __init__(self):
        self.calls = 0

    def predict(self, pairs, show_progress_bar=False):
        self.calls += 1
        return [0.9 if query.lower() in text.lower() else 0.1 for query, text in pairs]


def test_local_judge_batches_cross_encoder_pairs() -> None:
    reranker = CountingReranker()
    result = local_recall_judgement(
        reranker,
        FakeEmbeddings(),
        ["alpha"],
        [{"id": "V001", "content": "alpha evidence"}],
        [{"id": "R001", "content": "alpha evidence"}],
        "alpha context",
        "alpha adaptive",
        "alpha agentic",
        threshold=0.25,
        prefilter_limit=12,
    )
    assert reranker.calls == 1
    assert result["points"][0]["vector_candidates"] is True
    assert result["points"][0]["final_context"] is True


if __name__ == "__main__":
    test_reranked_hit_implies_vector_hit()
    test_loss_categories_are_separated()
    test_trace_separates_initial_rrf_from_followups()
    test_local_judge_batches_cross_encoder_pairs()
    print("retrieval recall tests: OK")
