from __future__ import annotations

import html
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from ticket_clustering.data import SymptomNode
from ticket_clustering.graph import GraphClusteringResult, TopicCluster
from ticket_clustering.entities import EntityProfile
from ticket_clustering.article_readiness import evaluate_article_readiness
from ticket_clustering.subclustering import ParentSubclustering


def atomic_write_text(path: Path, text: str, replace: bool) -> None:
    path = path.resolve()
    if path.exists() and not replace:
        raise FileExistsError(f"Output already exists: {path}. Use --replace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def node_sample(node: SymptomNode, entity_profile: EntityProfile | None = None) -> dict[str, Any]:
    equipment = Counter(item.equipment for item in node.occurrences if item.equipment)
    payload = {
        "node_id": node.node_id,
        "text": node.display_text,
        "normalized_text": node.normalized_text,
        "informativeness": node.informativeness,
        "occurrences": len(node.occurrences),
        "tickets": sorted({item.ticket_id for item in node.occurrences})[:8],
        "equipment": [{"name": name, "count": count} for name, count in equipment.most_common(5)],
    }
    if entity_profile is not None:
        payload["entities"] = entity_profile.as_dict()
    return payload


def cluster_payload(
    cluster: TopicCluster,
    nodes: list[SymptomNode],
    sample_count: int,
    entity_profiles: list[EntityProfile] | None = None,
    readiness_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node_positions = {node.node_id: index for index, node in enumerate(nodes)}
    member_nodes = [nodes[index] for index in cluster.member_indices]
    member_nodes.sort(
        key=lambda node: (
            node.node_id != nodes[cluster.representative_index].node_id,
            -node.ticket_count,
            -node.informativeness,
            node.normalized_text,
        )
    )
    equipment = Counter(
        occurrence.equipment
        for node in member_nodes
        for occurrence in node.occurrences
        if occurrence.equipment
    )
    ticket_ids = sorted(
        {
            occurrence.ticket_id
            for node in member_nodes
            for occurrence in node.occurrences
        }
    )
    payload = {
        "cluster_id": cluster.cluster_id,
        "label": cluster.label,
        "unique_symptoms": cluster.unique_symptoms,
        "occurrences": cluster.occurrence_count,
        "tickets": cluster.ticket_count,
        "ticket_ids": ticket_ids,
        "equipment_count": cluster.equipment_count,
        "mean_edge_similarity": round(cluster.mean_edge_similarity, 6),
        "minimum_edge_similarity": round(cluster.minimum_edge_similarity, 6),
        "equipment": [{"name": name, "count": count} for name, count in equipment.most_common(8)],
        "samples": [
            node_sample(node, entity_profiles[node_positions[node.node_id]] if entity_profiles else None)
            for node in member_nodes[:sample_count]
        ],
    }
    if entity_profiles is not None and readiness_config is not None:
        payload["article_readiness"] = evaluate_article_readiness(
            cluster,
            nodes,
            entity_profiles,
            readiness_config,
        )
    return payload


def build_payload(
    category: str,
    config: dict[str, Any],
    all_nodes: list[SymptomNode],
    eligible_nodes: list[SymptomNode],
    excluded_nodes: list[SymptomNode],
    result: GraphClusteringResult,
    embedding_cache: Path,
    knn_cache: Path,
    entity_profiles: list[EntityProfile] | None = None,
    subclusters: list[ParentSubclustering] | None = None,
) -> dict[str, Any]:
    clustered_node_count = sum(cluster.unique_symptoms for cluster in result.clusters)
    clustered_ticket_ids = {
        occurrence.ticket_id
        for cluster in result.clusters
        for index in cluster.member_indices
        for occurrence in eligible_nodes[index].occurrences
    }
    eligible_ticket_ids = {
        occurrence.ticket_id for node in eligible_nodes for occurrence in node.occurrences
    }
    sample_count = int(config["report"]["samples_per_cluster"])
    max_clusters = int(config["report"]["max_clusters"])
    readiness_config = config.get("article_readiness")
    ticket_sizes = sorted(cluster.ticket_count for cluster in result.clusters)
    symptom_sizes = sorted(cluster.unique_symptoms for cluster in result.clusters)
    total_cluster_edges_weight = sum(
        cluster.mean_edge_similarity * max(cluster.unique_symptoms - 1, 1)
        for cluster in result.clusters
    )
    total_cluster_edges_proxy = sum(max(cluster.unique_symptoms - 1, 1) for cluster in result.clusters)

    def percentile(values: list[int], fraction: float) -> int:
        if not values:
            return 0
        index = round((len(values) - 1) * fraction)
        return int(values[index])

    cluster_payloads = [
        cluster_payload(
            cluster,
            eligible_nodes,
            sample_count,
            entity_profiles,
            readiness_config,
        )
        for cluster in result.clusters
    ]
    if readiness_config:
        status_order = {"ready": 0, "review": 1, "not_ready": 2}
        cluster_payloads.sort(
            key=lambda item: (
                status_order.get(item.get("article_readiness", {}).get("status", "not_ready"), 9),
                -float(item.get("article_readiness", {}).get("score", 0.0)),
                -int(item["tickets"]),
            )
        )
    readiness_counts = Counter(
        item.get("article_readiness", {}).get("status", "unrated")
        for item in cluster_payloads
    )
    subcluster_payloads: list[dict[str, Any]] = []
    for split in subclusters or []:
        children = [
            cluster_payload(
                child,
                eligible_nodes,
                sample_count,
                entity_profiles,
                readiness_config,
            )
            for child in split.child_clusters
        ]
        children.sort(
            key=lambda item: (
                {"ready": 0, "review": 1, "not_ready": 2}.get(
                    item.get("article_readiness", {}).get("status", "not_ready"), 9
                ),
                -float(item.get("article_readiness", {}).get("score", 0.0)),
                -int(item["tickets"]),
            )
        )
        subcluster_payloads.append(
            {
                "parent_cluster_id": split.parent_cluster_id,
                "parent_label": split.parent_label,
                "parent_tickets": split.parent_ticket_count,
                "parent_unique_symptoms": split.parent_unique_symptoms,
                "node_coverage": round(split.node_coverage, 6),
                "unclustered_unique_symptoms": len(split.unclustered_indices),
                "children": children,
            }
        )
    child_readiness_counts = Counter(
        child.get("article_readiness", {}).get("status", "unrated")
        for split in subcluster_payloads
        for child in split["children"]
    )
    article_candidates = [
        {
            "origin": "primary",
            "parent_cluster_id": "",
            "cluster_id": item["cluster_id"],
            "label": item["label"],
            "score": item["article_readiness"]["score"],
            "ticket_ids": item["ticket_ids"],
        }
        for item in cluster_payloads
        if item.get("article_readiness", {}).get("status") == "ready"
    ]
    article_candidates.extend(
        {
            "origin": "subcluster",
            "parent_cluster_id": split["parent_cluster_id"],
            "cluster_id": child["cluster_id"],
            "label": child["label"],
            "score": child["article_readiness"]["score"],
            "ticket_ids": child["ticket_ids"],
        }
        for split in subcluster_payloads
        for child in split["children"]
        if child.get("article_readiness", {}).get("status") == "ready"
    )
    article_candidates.sort(key=lambda item: (-float(item["score"]), item["label"], item["cluster_id"]))
    return {
        "category": category,
        "config": config,
        "cache": {
            "embeddings": str(embedding_cache.resolve()),
            "knn": str(knn_cache.resolve()),
        },
        "summary": {
            "unique_symptoms_total": len(all_nodes),
            "unique_symptoms_eligible": len(eligible_nodes),
            "unique_symptoms_excluded": len(excluded_nodes),
            "unique_symptoms_clustered": clustered_node_count,
            "unique_symptoms_unclustered": len(result.unclustered_indices),
            "eligible_tickets": len(eligible_ticket_ids),
            "clustered_tickets": len(clustered_ticket_ids),
            "coverage_unique_symptoms": round(clustered_node_count / len(eligible_nodes), 6) if eligible_nodes else 0.0,
            "coverage_tickets": round(len(clustered_ticket_ids) / len(eligible_ticket_ids), 6) if eligible_ticket_ids else 0.0,
            "graph_edges": result.edge_count,
            "isolated_nodes": result.isolated_count,
            "communities_before_min_size": result.community_count_before_min_size,
            "accepted_clusters": len(result.clusters),
            "blocked_entity_conflicts": result.blocked_entity_conflicts,
            "entity_adjusted_candidates": result.entity_adjusted_candidates,
            "clusters_in_report": min(len(result.clusters), max_clusters),
            "cluster_ticket_size_median": round(float(median(ticket_sizes)), 2) if ticket_sizes else 0.0,
            "cluster_ticket_size_p90": percentile(ticket_sizes, 0.90),
            "cluster_ticket_size_max": max(ticket_sizes, default=0),
            "cluster_symptom_size_median": round(float(median(symptom_sizes)), 2) if symptom_sizes else 0.0,
            "cluster_symptom_size_p90": percentile(symptom_sizes, 0.90),
            "cluster_symptom_size_max": max(symptom_sizes, default=0),
            "cluster_mean_edge_similarity_weighted": round(
                total_cluster_edges_weight / total_cluster_edges_proxy,
                6,
            ) if total_cluster_edges_proxy else 0.0,
            "article_ready_clusters": readiness_counts.get("ready", 0),
            "article_review_clusters": readiness_counts.get("review", 0),
            "article_not_ready_clusters": readiness_counts.get("not_ready", 0),
            "review_parents_split": len(subcluster_payloads),
            "article_ready_subclusters": child_readiness_counts.get("ready", 0),
            "article_review_subclusters": child_readiness_counts.get("review", 0),
            "article_not_ready_subclusters": child_readiness_counts.get("not_ready", 0),
            "article_candidates_total": len(article_candidates),
        },
        "excluded": [
            {**node_sample(node), "reason": node.excluded_reason}
            for node in excluded_nodes[:500]
        ],
        "unclustered": [node_sample(eligible_nodes[index]) for index in result.unclustered_indices[:1000]],
        "clusters": cluster_payloads[:max_clusters],
        "subclusters": subcluster_payloads,
        "article_candidates": article_candidates,
    }


def render_html(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    cards = []
    for cluster in payload["clusters"]:
        readiness = cluster.get("article_readiness", {})
        status = readiness.get("status", "unrated")
        status_labels = {
            "ready": "готов для статьи",
            "review": "нужна проверка",
            "not_ready": "не готов",
            "unrated": "без оценки",
        }
        reasons = "; ".join(readiness.get("reasons", [])) or "—"
        samples = "".join(
            "<li><b>{}</b><br><small>тикеты: {} · информативность: {:.3f}</small></li>".format(
                html.escape(sample["text"]),
                html.escape(", ".join(sample["tickets"])),
                float(sample["informativeness"]),
            )
            for sample in cluster["samples"]
        )
        equipment = ", ".join(f"{item['name']} ({item['count']})" for item in cluster["equipment"]) or "—"
        cards.append(
            f"""
            <details>
              <summary><b>{html.escape(cluster['label'])}</b> — {cluster['tickets']} тикетов / {cluster['unique_symptoms']} формулировок</summary>
              <p><span class="status {html.escape(status)}">{html.escape(status_labels.get(status, status))}</span>
              Оценка: <b>{float(readiness.get('score', 0.0)):.3f}</b>; {html.escape(reasons)}</p>
              <p>Средняя similarity рёбер: <b>{cluster['mean_edge_similarity']:.3f}</b>; минимальная: {cluster['minimum_edge_similarity']:.3f}</p>
              <p>Оборудование: {html.escape(equipment)}</p>
              <ol>{samples}</ol>
            </details>
            """
        )
    split_sections = []
    for split in payload.get("subclusters", []):
        children = []
        for child in split["children"]:
            readiness = child.get("article_readiness", {})
            status = readiness.get("status", "unrated")
            reasons = "; ".join(readiness.get("reasons", [])) or "—"
            samples = "".join(
                f"<li>{html.escape(sample['text'])}</li>"
                for sample in child["samples"][:6]
            )
            children.append(
                f"<details><summary><span class=\"status {html.escape(status)}\">{html.escape(status)}</span> "
                f"<b>{html.escape(child['label'])}</b> — {child['tickets']} тикетов, "
                f"оценка {float(readiness.get('score', 0.0)):.3f}</summary>"
                f"<p>{html.escape(reasons)}</p><ol>{samples}</ol></details>"
            )
        split_sections.append(
            f"<section><h3>{html.escape(split['parent_label'])}</h3>"
            f"<p>Родитель: {split['parent_tickets']} тикетов; покрытие формулировок подкластерами: "
            f"{float(split['node_coverage']) * 100:.1f}%.</p>{''.join(children)}</section>"
        )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Baseline кластеризации — {html.escape(payload['category'])}</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1300px;margin:24px auto;padding:0 20px;color:#20242a}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}}
.kpi{{border:1px solid #d9dee6;border-radius:8px;padding:12px}}.kpi b{{font-size:1.4rem;display:block}}
details{{border:1px solid #d9dee6;border-radius:8px;padding:10px;margin:8px 0}}summary{{cursor:pointer}}
small{{color:#667085}}code{{word-break:break-all}}
.status{{display:inline-block;border-radius:999px;padding:2px 8px;margin-right:8px;background:#e5e7eb}}
.status.ready{{background:#d1fadf;color:#05603a}}.status.review{{background:#fef0c7;color:#7a2e0e}}
.status.not_ready{{background:#fee4e2;color:#912018}}
</style></head><body>
<h1>Baseline тематических подкластеров</h1>
<p>Категория: <b>{html.escape(payload['category'])}</b>. Названия обращений и решения при кластеризации не использовались.</p>
<div class="kpis">
  <div class="kpi"><b>{summary['accepted_clusters']}</b>кластеров</div>
  <div class="kpi"><b>{summary['coverage_unique_symptoms']*100:.1f}%</b>покрытие формулировок</div>
  <div class="kpi"><b>{summary['coverage_tickets']*100:.1f}%</b>покрытие тикетов</div>
  <div class="kpi"><b>{summary['isolated_nodes']}</b>изолированных симптомов</div>
  <div class="kpi"><b>{summary.get('article_ready_clusters', 0)}</b>готовы для статьи</div>
  <div class="kpi"><b>{summary.get('article_review_clusters', 0)}</b>нужна проверка</div>
  <div class="kpi"><b>{summary.get('article_ready_subclusters', 0)}</b>готовых подкластеров</div>
  <div class="kpi"><b>{summary.get('article_candidates_total', 0)}</b>кандидатов всего</div>
</div>
<p>Уникальных симптомов: {summary['unique_symptoms_total']}; допущено: {summary['unique_symptoms_eligible']};
рёбер графа: {summary['graph_edges']}; сообществ до минимального размера: {summary['communities_before_min_size']}.</p>
<h2>Кластеры</h2>{''.join(cards)}
<h2>Повторное разбиение review-кластеров</h2>{''.join(split_sections) or '<p>Подходящие разбиения не найдены.</p>'}
</body></html>"""


def write_report(payload: dict[str, Any], html_path: Path, json_path: Path, replace: bool) -> None:
    if not replace:
        for path in (html_path, json_path):
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}. Use --replace.")
    atomic_write_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2), replace=replace)
    atomic_write_text(html_path, render_html(payload), replace=replace)
