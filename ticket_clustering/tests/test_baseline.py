from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import networkx as nx


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ticket_clustering.article_readiness import evaluate_article_readiness
from ticket_clustering.data import SymptomOccurrence, build_nodes, select_eligible_nodes
from ticket_clustering.entities import adjusted_similarity, detect_entities, profile_nodes
from ticket_clustering.graph import TopicCluster, build_mutual_knn_graph, cluster_graph
from ticket_clustering.vectors import load_or_compute_exact_knn
from ticket_clustering.subclustering import stricter_induced_graph


def occurrence(ticket_id: str, ordinal: int, text: str, normalized: str) -> SymptomOccurrence:
    return SymptomOccurrence(ticket_id, ordinal, text, normalized, "R500", "Справочный вопрос")


def main() -> None:
    r500 = detect_entities("Ошибка OPC UA на R500", ["Regul R050, R100, R200, R400, R500, R600"])
    r500s = detect_entities("Ошибка OPC UA на R500S", ["Regul R500S"])
    assert r500.series == frozenset({"R500"})
    assert r500s.series == frozenset({"R500S"})
    assert r500.protocols == frozenset({"opc_ua"})
    _, conflict = adjusted_similarity(0.95, r500, r500s, {"block_series_conflicts": True})
    assert conflict == "series_conflict"
    no_entities = detect_entities("Неизвестная ошибка")
    adjusted, status = adjusted_similarity(
        0.81,
        r500,
        no_entities,
        {"penalties": {"unmatched_anchor": 0.02}},
    )
    assert status == "ok" and abs(adjusted - 0.79) < 1e-9

    article_occurrences = [
        occurrence(f"RL-A{index}", 1, text, text.casefold())
        for index, text in enumerate(
            [
                "Ошибка подписки OPC UA",
                "Разрыв сессии OPC UA",
                "Лимит подключений OPC UA",
                "Сертификат OPC UA",
                "Таймаут OPC UA",
            ],
            start=1,
        )
    ]
    article_nodes = build_nodes(article_occurrences)
    article_cluster = TopicCluster(
        cluster_id="topic-test",
        label="OPC UA",
        representative_index=0,
        member_indices=list(range(5)),
        unique_symptoms=5,
        occurrence_count=5,
        ticket_count=5,
        equipment_count=1,
        mean_edge_similarity=0.88,
        minimum_edge_similarity=0.84,
    )
    readiness = evaluate_article_readiness(
        article_cluster,
        article_nodes,
        profile_nodes(article_nodes),
        {
            "min_tickets": 5,
            "ready_threshold": 0.68,
            "review_threshold": 0.48,
            "min_topic_quality": 0.18,
            "weak_components": ["astraregul", "astraide"],
            "actionable_issue_types": [],
        },
    )
    assert readiness["status"] == "ready"
    assert readiness["features"]["strong_anchor"]["dominant"] == "protocol:opc_ua"

    parent_graph = nx.Graph()
    parent_graph.add_edge(0, 1, weight=0.84)
    parent_graph.add_edge(1, 2, weight=0.82)
    strict_graph = stricter_induced_graph(parent_graph, [0, 1, 2], similarity_threshold=0.83)
    assert strict_graph.has_edge(0, 1)
    assert not strict_graph.has_edge(1, 2)

    occurrences = [
        occurrence("RL-1", 1, "Нет связи с ПЛК", "нет связи с плк"),
        occurrence("RL-2", 1, "Нет связи с ПЛК", "нет связи с плк"),
        occurrence("RL-3", 1, "Download denied", "download denied"),
        occurrence("RL-4", 1, "Заводской сброс R500", "заводской сброс r500"),
    ]
    nodes = build_nodes(occurrences)
    assert len(nodes) == 3
    assert next(node for node in nodes if node.normalized_text == "нет связи с плк").ticket_count == 2
    eligible, excluded = select_eligible_nodes(nodes, min_chars=4, min_tokens=1, min_informativeness=0.0)
    assert len(eligible) == 3
    assert not excluded

    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        indices, similarities, _ = load_or_compute_exact_knn(
            embeddings,
            embedding_fingerprint="test-fingerprint",
            max_neighbors=2,
            block_size=2,
            device="cpu",
            cache_dir=Path(temp_dir),
        )
        assert int(indices[0, 0]) == 1
        assert float(similarities[0, 0]) > 0.99

    graph = build_mutual_knn_graph(
        node_count=3,
        neighbor_indices=indices,
        neighbor_similarities=similarities,
        graph_neighbors=1,
        similarity_threshold=0.9,
        mutual_only=True,
    )
    assert graph.number_of_edges() == 1
    result = cluster_graph(
        graph,
        eligible,
        category="Справочное",
        resolution=1.0,
        seed=42,
        min_unique_symptoms=2,
        min_tickets=2,
    )
    assert len(result.clusters) == 1
    assert result.clusters[0].unique_symptoms == 2
    assert len(result.unclustered_indices) == 1
    print("ticket_clustering baseline tests: OK")


if __name__ == "__main__":
    main()
