from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from ticket_clustering.article_readiness import evaluate_article_readiness
from ticket_clustering.data import SymptomNode
from ticket_clustering.entities import EntityProfile
from ticket_clustering.graph import TopicCluster, cluster_graph


@dataclass
class ParentSubclustering:
    parent_cluster_id: str
    parent_label: str
    parent_ticket_count: int
    parent_unique_symptoms: int
    child_clusters: list[TopicCluster]
    unclustered_indices: list[int]
    node_coverage: float


def stricter_induced_graph(
    graph: nx.Graph,
    member_indices: list[int],
    similarity_threshold: float,
) -> nx.Graph:
    result = graph.subgraph(member_indices).copy()
    weak_edges = [
        (source, target)
        for source, target, data in result.edges(data=True)
        if float(data.get("weight", 0.0)) < similarity_threshold
    ]
    result.remove_edges_from(weak_edges)
    result.graph.update(graph.graph)
    return result


def build_review_subclusters(
    graph: nx.Graph,
    nodes: list[SymptomNode],
    profiles: list[EntityProfile],
    primary_clusters: list[TopicCluster],
    category: str,
    readiness_config: dict[str, Any],
    config: dict[str, Any],
) -> list[ParentSubclustering]:
    if not bool(config.get("enabled", False)):
        return []

    target_statuses = {str(value) for value in config.get("statuses", ["review"])}
    min_parent_tickets = int(config.get("min_parent_tickets", 15))
    min_parent_symptoms = int(config.get("min_parent_symptoms", 12))
    min_children = int(config.get("min_children", 2))
    min_node_coverage = float(config.get("min_node_coverage", 0.50))
    results: list[ParentSubclustering] = []

    for parent in primary_clusters:
        readiness = evaluate_article_readiness(parent, nodes, profiles, readiness_config)
        if readiness["status"] not in target_statuses:
            continue
        if parent.ticket_count < min_parent_tickets or parent.unique_symptoms < min_parent_symptoms:
            continue

        subgraph = stricter_induced_graph(
            graph,
            parent.member_indices,
            similarity_threshold=float(config.get("similarity_threshold", 0.83)),
        )
        child_result = cluster_graph(
            subgraph,
            nodes,
            category=f"{category}:{parent.cluster_id}",
            resolution=float(config.get("resolution", 5.0)),
            seed=int(config.get("seed", 42)),
            min_unique_symptoms=int(config.get("min_unique_symptoms", 3)),
            min_tickets=int(config.get("min_tickets", 3)),
        )
        clustered_nodes = sum(child.unique_symptoms for child in child_result.clusters)
        node_coverage = clustered_nodes / parent.unique_symptoms if parent.unique_symptoms else 0.0
        if len(child_result.clusters) < min_children or node_coverage < min_node_coverage:
            continue
        if len(child_result.clusters) == 1 and child_result.clusters[0].unique_symptoms == parent.unique_symptoms:
            continue

        results.append(
            ParentSubclustering(
                parent_cluster_id=parent.cluster_id,
                parent_label=parent.label,
                parent_ticket_count=parent.ticket_count,
                parent_unique_symptoms=parent.unique_symptoms,
                child_clusters=child_result.clusters,
                unclustered_indices=child_result.unclustered_indices,
                node_coverage=node_coverage,
            )
        )

    results.sort(key=lambda item: (-item.parent_ticket_count, item.parent_cluster_id))
    return results
