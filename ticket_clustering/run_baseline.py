from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ticket_clustering.data import (
    build_nodes,
    load_category_occurrences,
    open_source_readonly,
    select_eligible_nodes,
)
from ticket_clustering.graph import build_mutual_knn_graph, cluster_graph
from ticket_clustering.entities import profile_nodes
from ticket_clustering.reporting import build_payload, write_report
from ticket_clustering.subclustering import build_review_subclusters
from ticket_clustering.vectors import load_or_compute_exact_knn, load_or_encode


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "baseline_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline graph clustering inside one ticket category.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--category", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="")
    parser.add_argument("--graph-neighbors", type=int, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument("--resolution", type=float, default=None)
    parser.add_argument("--strategy", choices=("semantic-only", "semantic+entities"), default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return data


def effective_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if args.category:
        config["category"] = args.category
    if args.device:
        config["embedding"]["device"] = args.device
    if args.graph_neighbors is not None:
        config["neighbors"]["graph_neighbors"] = args.graph_neighbors
    if args.similarity_threshold is not None:
        config["neighbors"]["similarity_threshold"] = args.similarity_threshold
    if args.resolution is not None:
        config["clustering"]["resolution"] = args.resolution
    if args.strategy:
        config["strategy"] = args.strategy
    return config


def category_slug(category: str) -> str:
    known = {
        "Справочное": "reference",
        "ОшибкаНастройкиЧужая": "configuration_external",
        "ОшибкаНастройкиПрософт": "configuration_internal",
        "ОтказПрогПрософт": "software_failure_internal",
        "ОтказОборПрософт": "hardware_failure_internal",
    }
    return known.get(category, f"category_{hashlib.sha256(category.encode('utf-8')).hexdigest()[:10]}")


def output_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    neighbors = config["neighbors"]
    clustering = config["clustering"]
    similarity = f"{float(neighbors['similarity_threshold']):.3f}".replace(".", "")
    resolution = f"{float(clustering['resolution']):.2f}".replace(".", "")
    strategy = str(config.get("strategy", "semantic-only")).replace("+", "_plus_").replace("-", "_")
    suffix = f"{strategy}_n{int(neighbors['graph_neighbors'])}_s{similarity}_r{resolution}"
    base = PACKAGE_ROOT / "reports" / f"baseline_{category_slug(str(config['category']))}_{suffix}"
    return base.with_suffix(".html"), base.with_suffix(".json")


def validate_config(config: dict[str, Any]) -> None:
    neighbors = config["neighbors"]
    if int(neighbors["graph_neighbors"]) < 1:
        raise ValueError("graph_neighbors must be positive")
    if int(neighbors["max_cached_neighbors"]) < int(neighbors["graph_neighbors"]):
        raise ValueError("max_cached_neighbors must be >= graph_neighbors")
    threshold = float(neighbors["similarity_threshold"])
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1]")


def main() -> None:
    args = parse_args()
    config = effective_config(args)
    validate_config(config)
    category = str(config["category"])
    source_path = (ROOT / str(config["source_db"])).resolve()
    html_path, json_path = output_paths(config)
    if not args.replace and not args.dry_run:
        for path in (html_path, json_path):
            if path.exists():
                raise SystemExit(f"Output already exists: {path}. Use --replace.")

    connection = open_source_readonly(source_path)
    try:
        occurrences = load_category_occurrences(connection, category)
    finally:
        connection.close()
    nodes = build_nodes(occurrences)
    preprocessing = config["preprocessing"]
    eligible, excluded = select_eligible_nodes(
        nodes,
        min_chars=int(preprocessing["min_chars"]),
        min_tokens=int(preprocessing["min_tokens"]),
        min_informativeness=float(preprocessing["min_informativeness"]),
    )
    scope = {
        "category": category,
        "occurrences": len(occurrences),
        "tickets": len({item.ticket_id for item in occurrences}),
        "unique_symptoms": len(nodes),
        "eligible_unique_symptoms": len(eligible),
        "excluded_unique_symptoms": len(excluded),
    }
    print(json.dumps(scope, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete: source was read-only; no model loaded and no files written.")
        return
    if len(eligible) < 2:
        raise SystemExit("Not enough eligible symptoms for clustering")

    cache_dir = PACKAGE_ROOT / "artifacts"
    embedding = config["embedding"]
    embeddings, embedding_fingerprint, embedding_cache = load_or_encode(
        eligible,
        model_name=str(embedding["model"]),
        device=str(embedding["device"]),
        batch_size=int(embedding["batch_size"]),
        local_files_only=bool(embedding["local_files_only"]),
        cache_dir=cache_dir,
    )
    neighbors = config["neighbors"]
    indices, similarities, knn_cache = load_or_compute_exact_knn(
        embeddings,
        embedding_fingerprint=embedding_fingerprint,
        max_neighbors=int(neighbors["max_cached_neighbors"]),
        block_size=int(neighbors["block_size"]),
        device=str(embedding["device"]),
        cache_dir=cache_dir,
    )
    strategy = str(config.get("strategy", "semantic-only"))
    entity_profiles = profile_nodes(eligible)
    graph_entity_profiles = entity_profiles if strategy == "semantic+entities" else None
    graph = build_mutual_knn_graph(
        node_count=len(eligible),
        neighbor_indices=indices,
        neighbor_similarities=similarities,
        graph_neighbors=int(neighbors["graph_neighbors"]),
        similarity_threshold=float(neighbors["similarity_threshold"]),
        mutual_only=bool(neighbors["mutual_only"]),
        entity_profiles=graph_entity_profiles,
        entity_config=config.get("entities") if graph_entity_profiles is not None else None,
    )
    clustering = config["clustering"]
    result = cluster_graph(
        graph,
        eligible,
        category=category,
        resolution=float(clustering["resolution"]),
        seed=int(clustering["seed"]),
        min_unique_symptoms=int(clustering["min_unique_symptoms"]),
        min_tickets=int(clustering["min_tickets"]),
    )
    subclusters = build_review_subclusters(
        graph,
        eligible,
        entity_profiles,
        result.clusters,
        category,
        config.get("article_readiness", {}),
        config.get("subclustering", {}),
    )
    payload = build_payload(
        category,
        config,
        nodes,
        eligible,
        excluded,
        result,
        embedding_cache,
        knn_cache,
        entity_profiles,
        subclusters,
    )
    write_report(payload, html_path, json_path, replace=args.replace)
    print(f"HTML: {html_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
