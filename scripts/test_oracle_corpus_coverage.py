"""Focused tests for oracle corpus-coverage accounting."""

from diagnose_oracle_corpus_coverage import (
    build_summary,
    classify_point,
    original_vector_hits,
)


def test_classification() -> None:
    assert classify_point(True, True) == "retrieved_by_agent_query"
    assert classify_point(True, False) == "retrieved_by_agent_query"
    assert classify_point(False, True) == "query_formulation_gap"
    assert classify_point(False, False, True) == "ambiguous_semantic_match"
    assert classify_point(False, False) == "likely_corpus_or_index_gap"


def test_original_vector_causal_correction() -> None:
    recall_item = {
        "judgement": {
            "points": [
                {
                    "point_index": 1,
                    "vector_candidates": False,
                    "reranked_selection": True,
                },
                {
                    "point_index": 2,
                    "vector_candidates": False,
                    "reranked_selection": False,
                },
            ]
        }
    }
    assert original_vector_hits(recall_item, 3) == [True, False, False]


def test_summary() -> None:
    summary = build_summary(
        [
            {
                "points": [
                    {
                        "oracle_hit": True,
                        "classification": "retrieved_by_agent_query",
                        "best_match": {"source": "docs"},
                    },
                    {
                        "oracle_hit": True,
                        "classification": "query_formulation_gap",
                        "best_match": {"source": "tickets"},
                    },
                ]
            },
            {
                "points": [
                    {
                        "oracle_hit": False,
                        "classification": "likely_corpus_or_index_gap",
                        "best_match": {},
                    }
                ]
            },
        ]
    )
    assert summary["questions_evaluated"] == 2
    assert summary["reference_points"] == 3
    assert summary["oracle_hits"] == 2
    assert summary["oracle_recall"] == 0.667
    assert summary["classification"]["query_formulation_gap"] == 1
    assert summary["best_hit_source"] == {"docs": 1, "tickets": 1}
    assert summary["question_coverage"] == {
        "zero_oracle_recall": 1,
        "full_oracle_recall": 1,
    }


if __name__ == "__main__":
    test_classification()
    test_original_vector_causal_correction()
    test_summary()
    print("oracle corpus coverage tests: OK")
