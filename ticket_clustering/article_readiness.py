from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ticket_clustering.data import SymptomNode
from ticket_clustering.entities import EntityProfile
from ticket_clustering.graph import TopicCluster


@dataclass(frozen=True)
class GroupStats:
    coverage: float
    purity: float
    dominant: str
    dominant_support: int
    nodes_with_values: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": round(self.coverage, 6),
            "purity": round(self.purity, 6),
            "dominant": self.dominant,
            "dominant_support": self.dominant_support,
            "nodes_with_values": self.nodes_with_values,
        }


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def group_stats(profiles: list[EntityProfile], field: str) -> GroupStats:
    support: Counter[str] = Counter()
    nodes_with_values = 0
    for profile in profiles:
        values = getattr(profile, field)
        if values:
            nodes_with_values += 1
            support.update(values)
    if not support:
        return GroupStats(0.0, 0.0, "", 0, 0)
    dominant, dominant_support = sorted(support.items(), key=lambda item: (-item[1], item[0]))[0]
    return GroupStats(
        coverage=nodes_with_values / len(profiles) if profiles else 0.0,
        purity=dominant_support / nodes_with_values if nodes_with_values else 0.0,
        dominant=dominant,
        dominant_support=dominant_support,
        nodes_with_values=nodes_with_values,
    )


def combined_strong_anchor_stats(
    profiles: list[EntityProfile],
    weak_components: set[str],
) -> GroupStats:
    support: Counter[str] = Counter()
    nodes_with_values = 0
    for profile in profiles:
        values = {
            *(f"module:{value}" for value in profile.modules),
            *(f"protocol:{value}" for value in profile.protocols),
            *(
                f"component:{value}"
                for value in profile.components
                if value not in weak_components
            ),
        }
        if values:
            nodes_with_values += 1
            support.update(values)
    if not support:
        return GroupStats(0.0, 0.0, "", 0, 0)
    dominant, dominant_support = sorted(support.items(), key=lambda item: (-item[1], item[0]))[0]
    return GroupStats(
        coverage=nodes_with_values / len(profiles),
        purity=dominant_support / nodes_with_values,
        dominant=dominant,
        dominant_support=dominant_support,
        nodes_with_values=nodes_with_values,
    )


def issue_quality(stats: GroupStats, actionable_issue_types: set[str]) -> float:
    if not stats.dominant or stats.dominant not in actionable_issue_types:
        return 0.0
    return clamp(stats.coverage * stats.purity)


def evaluate_article_readiness(
    cluster: TopicCluster,
    nodes: list[SymptomNode],
    profiles: list[EntityProfile],
    config: dict[str, Any],
) -> dict[str, Any]:
    member_nodes = [nodes[index] for index in cluster.member_indices]
    member_profiles = [profiles[index] for index in cluster.member_indices]
    weak_components = {str(value) for value in config.get("weak_components", [])}
    actionable_issue_types = {str(value) for value in config.get("actionable_issue_types", [])}

    stats = {
        field: group_stats(member_profiles, field)
        for field in ("series", "modules", "protocols", "components", "issue_types")
    }
    strong = combined_strong_anchor_stats(member_profiles, weak_components)
    anchor_quality = clamp(strong.coverage * strong.purity)
    issue_topic_quality = issue_quality(stats["issue_types"], actionable_issue_types)

    cohesion_floor = float(config.get("cohesion_floor", 0.80))
    cohesion_full = float(config.get("cohesion_full", 0.88))
    cohesion_quality = clamp(
        (cluster.mean_edge_similarity - cohesion_floor) / max(cohesion_full - cohesion_floor, 1e-9)
    )
    mean_informativeness = (
        sum(node.informativeness for node in member_nodes) / len(member_nodes)
        if member_nodes else 0.0
    )
    specificity_floor = float(config.get("specificity_floor", 0.62))
    specificity_full = float(config.get("specificity_full", 0.84))
    specificity_quality = clamp(
        (mean_informativeness - specificity_floor) / max(specificity_full - specificity_floor, 1e-9)
    )

    preferred_max = int(config.get("preferred_max_tickets", 100))
    absolute_max = int(config.get("absolute_max_tickets", 220))
    if cluster.ticket_count <= preferred_max:
        size_quality = 1.0
    else:
        size_quality = clamp(
            1.0 - (cluster.ticket_count - preferred_max) / max(absolute_max - preferred_max, 1)
        )

    weights = config.get("weights", {})
    score = (
        float(weights.get("cohesion", 0.30)) * cohesion_quality
        + float(weights.get("strong_anchor", 0.30)) * anchor_quality
        + float(weights.get("actionable_issue", 0.20)) * issue_topic_quality
        + float(weights.get("specificity", 0.10)) * specificity_quality
        + float(weights.get("size", 0.10)) * size_quality
    )
    score = clamp(score)

    min_tickets = int(config.get("min_tickets", 5))
    ready_threshold = float(config.get("ready_threshold", 0.68))
    review_threshold = float(config.get("review_threshold", 0.48))
    min_topic_quality = float(config.get("min_topic_quality", 0.18))
    topic_quality = max(anchor_quality, issue_topic_quality)
    if cluster.ticket_count < min_tickets:
        status = "not_ready"
    elif score >= ready_threshold and topic_quality >= min_topic_quality:
        status = "ready"
    elif score >= review_threshold:
        status = "review"
    else:
        status = "not_ready"

    reasons: list[str] = []
    if strong.dominant and anchor_quality >= 0.20:
        reasons.append(f"предметный якорь: {strong.dominant}")
    elif issue_topic_quality >= 0.20:
        reasons.append(f"техническое событие: {stats['issue_types'].dominant}")
    else:
        reasons.append("нет устойчивого предметного якоря")
    if cohesion_quality < 0.40:
        reasons.append("слабая внутренняя связность")
    if cluster.ticket_count > preferred_max:
        reasons.append("кластер желательно дополнительно разделить")
    if cluster.ticket_count < min_tickets:
        reasons.append("недостаточно обращений")

    return {
        "status": status,
        "score": round(score, 6),
        "reasons": reasons,
        "features": {
            "cohesion_quality": round(cohesion_quality, 6),
            "strong_anchor_quality": round(anchor_quality, 6),
            "actionable_issue_quality": round(issue_topic_quality, 6),
            "specificity_quality": round(specificity_quality, 6),
            "size_quality": round(size_quality, 6),
            "topic_quality": round(topic_quality, 6),
            "mean_informativeness": round(mean_informativeness, 6),
            "strong_anchor": strong.as_dict(),
            "groups": {name: value.as_dict() for name, value in stats.items()},
        },
    }
