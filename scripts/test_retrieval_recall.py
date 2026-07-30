"""Fast checks for the retrieval recall diagnostic summary."""

from diagnose_retrieval_recall import build_summary


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


if __name__ == "__main__":
    test_reranked_hit_implies_vector_hit()
    test_loss_categories_are_separated()
    print("retrieval recall tests: OK")
