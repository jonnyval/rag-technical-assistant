from __future__ import annotations

import hashlib
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ticket_clustering.data import SymptomNode
from ticket_clustering.entities import EntityProfile, adjusted_similarity


@dataclass
class TopicCluster:
    cluster_id: str
    label: str
    representative_index: int
    member_indices: list[int]
    unique_symptoms: int
    occurrence_count: int
    ticket_count: int
    equipment_count: int
    mean_edge_similarity: float
    minimum_edge_similarity: float


@dataclass
class GraphClusteringResult:
    clusters: list[TopicCluster]
    unclustered_indices: list[int]
    edge_count: int
    isolated_count: int
    community_count_before_min_size: int
    blocked_entity_conflicts: int = 0
    entity_adjusted_candidates: int = 0


def build_mutual_knn_graph(
    node_count: int,
    neighbor_indices: np.ndarray,
    neighbor_similarities: np.ndarray,
    graph_neighbors: int,
    similarity_threshold: float,
    mutual_only: bool,
    entity_profiles: list[EntityProfile] | None = None,
    entity_config: dict | None = None,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    effective_neighbors = min(graph_neighbors, neighbor_indices.shape[1])
    directed: list[dict[int, float]] = []
    blocked_conflicts = 0
    adjusted_candidates = 0
    for index in range(node_count):
        values: dict[int, float] = {}
        for position in range(effective_neighbors):
            target = int(neighbor_indices[index, position])
            similarity = float(neighbor_similarities[index, position])
            if entity_profiles is not None and entity_config is not None:
                similarity, status = adjusted_similarity(
                    similarity,
                    entity_profiles[index],
                    entity_profiles[target],
                    entity_config,
                )
                adjusted_candidates += 1
                if status == "series_conflict":
                    blocked_conflicts += 1
                    continue
            if similarity < similarity_threshold:
                continue
            values[target] = similarity
        directed.append(values)

    for source, targets in enumerate(directed):
        for target, similarity in targets.items():
            if source >= target:
                continue
            reverse = directed[target].get(source)
            if mutual_only and reverse is None:
                continue
            weight = (similarity + reverse) / 2.0 if reverse is not None else similarity
            graph.add_edge(source, target, weight=weight)
    graph.graph["blocked_entity_conflicts"] = blocked_conflicts
    graph.graph["entity_adjusted_candidates"] = adjusted_candidates
    return graph


def stable_cluster_id(category: str, representative: SymptomNode) -> str:
    raw = f"{category}\0{representative.normalized_text}".encode("utf-8", errors="ignore")
    return f"topic-{hashlib.sha256(raw).hexdigest()[:20]}"


def representative_index(graph: nx.Graph, members: set[int], nodes: list[SymptomNode]) -> int:
    def score(index: int) -> tuple[float, int, float, str]:
        weighted_degree = sum(
            float(data.get("weight", 1.0))
            for _, target, data in graph.edges(index, data=True)
            if target in members
        )
        return (
            weighted_degree,
            nodes[index].ticket_count,
            nodes[index].informativeness,
            nodes[index].normalized_text,
        )

    return max(sorted(members), key=score)


def cluster_graph(
    graph: nx.Graph,
    nodes: list[SymptomNode],
    category: str,
    resolution: float,
    seed: int,
    min_unique_symptoms: int,
    min_tickets: int,
) -> GraphClusteringResult:
    if graph.number_of_edges():
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )
    else:
        communities = [{index} for index in graph.nodes]

    clusters: list[TopicCluster] = []
    unclustered: list[int] = []
    for members_raw in communities:
        members = set(int(index) for index in members_raw)
        tickets = {
            occurrence.ticket_id
            for index in members
            for occurrence in nodes[index].occurrences
        }
        if len(members) < min_unique_symptoms or len(tickets) < min_tickets:
            unclustered.extend(sorted(members))
            continue

        representative = representative_index(graph, members, nodes)
        subgraph = graph.subgraph(members)
        edge_weights = [float(data.get("weight", 1.0)) for _, _, data in subgraph.edges(data=True)]
        equipment = {
            occurrence.equipment
            for index in members
            for occurrence in nodes[index].occurrences
            if occurrence.equipment
        }
        clusters.append(
            TopicCluster(
                cluster_id=stable_cluster_id(category, nodes[representative]),
                label=nodes[representative].display_text,
                representative_index=representative,
                member_indices=sorted(members),
                unique_symptoms=len(members),
                occurrence_count=sum(len(nodes[index].occurrences) for index in members),
                ticket_count=len(tickets),
                equipment_count=len(equipment),
                mean_edge_similarity=sum(edge_weights) / len(edge_weights) if edge_weights else 0.0,
                minimum_edge_similarity=min(edge_weights) if edge_weights else 0.0,
            )
        )

    clusters.sort(key=lambda item: (-item.ticket_count, -item.unique_symptoms, item.cluster_id))
    return GraphClusteringResult(
        clusters=clusters,
        unclustered_indices=sorted(unclustered),
        edge_count=graph.number_of_edges(),
        isolated_count=sum(1 for node in graph.nodes if graph.degree(node) == 0),
        community_count_before_min_size=len(communities),
        blocked_entity_conflicts=int(graph.graph.get("blocked_entity_conflicts", 0)),
        entity_adjusted_candidates=int(graph.graph.get("entity_adjusted_candidates", 0)),
    )
