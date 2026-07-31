"""Focused checks for query-aware RRF reranking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from src.retrieval.dual_retriever import DualRetriever


class FakeReranker:
    def predict(self, pairs):
        scores = []
        for query, content in pairs:
            if query == "original":
                scores.append({"main": 0.90, "facet-a": 0.10, "facet-b": 0.08}[content])
            elif query == "query a":
                scores.append(0.85 if content == "facet-a" else 0.05)
            elif query == "query b":
                scores.append(0.80 if content == "facet-b" else 0.05)
            else:
                scores.append(0.01)
        return scores


def make_retriever(**overrides) -> DualRetriever:
    values = {
        "docs_retriever": object(),
        "tickets_retriever": object(),
        "reranker_model": FakeReranker(),
        "top_k_final": 3,
        "rerank_threshold": 0.05,
        "query_aware_rerank": True,
        "max_reserved_queries": 2,
        "per_query_quota": 1,
    }
    values.update(overrides)
    return DualRetriever.model_construct(**values)


def document(content: str) -> Document:
    return Document(page_content=content, metadata={"db_source": "docs", "doc_id": content})


def test_fusion_keeps_query_provenance() -> None:
    retriever = make_retriever()
    main, facet = document("main"), document("facet-a")
    fused = retriever._fuse_multi_query_children(
        [[main], [facet, main]],
        queries=["original", "query a"],
        rrf_k=60,
        candidate_limit=10,
    )
    by_content = {item.page_content: item for item in fused}
    assert by_content["main"].metadata["matched_queries"] == ["original", "query a"]
    assert by_content["facet-a"].metadata["origin_query"] == "query a"


def test_generated_query_can_recover_facet_candidate() -> None:
    retriever = make_retriever()
    items = [document("main"), document("facet-a"), document("facet-b")]
    items[0].metadata["matched_queries"] = ["original"]
    items[1].metadata.update(matched_queries=["query a"], origin_query="query a")
    items[2].metadata.update(matched_queries=["query b"], origin_query="query b")
    ranked = retriever._rank_multi_query_children(
        "original", items, limit=3, use_reranker=True
    )
    assert [item.page_content for item in ranked] == ["main", "facet-a", "facet-b"]
    assert ranked[1].metadata["rerank_query"] == "query a"
    assert ranked[1].metadata["original_rerank_score"] == 0.10
    assert ranked[1].metadata["rerank_score"] == 0.85


def test_diversity_reserves_generated_query_facets() -> None:
    retriever = make_retriever()
    main, facet_a, facet_b = document("main"), document("facet-a"), document("facet-b")
    main.metadata.update(rerank_score=0.95, matched_queries=["original"])
    facet_a.metadata.update(rerank_score=0.40, matched_queries=["query a"], rerank_query="query a")
    facet_b.metadata.update(rerank_score=0.30, matched_queries=["query b"], rerank_query="query b")
    selected = retriever._select_diverse_children(
        [main, facet_a, facet_b],
        ["original", "query a", "query b"],
        limit=2,
    )
    assert [item.page_content for item in selected] == ["facet-a", "facet-b"]


def test_disabled_mode_delegates_to_original_query() -> None:
    retriever = make_retriever(query_aware_rerank=False)
    items = [document("main"), document("facet-a")]
    items[1].metadata["matched_queries"] = ["query a"]
    ranked = retriever._rank_multi_query_children(
        "original", items, limit=2, use_reranker=True
    )
    assert [item.page_content for item in ranked] == ["main", "facet-a"]
    assert ranked[1].metadata["rerank_score"] == 0.10


if __name__ == "__main__":
    test_fusion_keeps_query_provenance()
    test_generated_query_can_recover_facet_candidate()
    test_diversity_reserves_generated_query_facets()
    test_disabled_mode_delegates_to_original_query()
    print("query-aware rerank tests: OK")