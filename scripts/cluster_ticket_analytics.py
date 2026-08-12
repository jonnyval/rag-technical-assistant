from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "analytics" / "tp_analyze" / "server_data" / "ticket_analytics.sqlite3"
DEFAULT_OUTPUTS = {
    "category": ROOT / "analytics" / "tp_analyze" / "server_data" / "clusters_by_category.sqlite3",
    "symptom": ROOT / "analytics" / "tp_analyze" / "server_data" / "clusters_by_symptom.sqlite3",
}
CLUSTER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SymptomRow:
    ticket_id: str
    ordinal: int
    text: str
    normalized_text: str
    category: str
    canonical_type_code: str
    canonical_type_label: str
    equipment: str


@dataclass
class ClusterResult:
    cluster_id: str
    label: str
    representative_text: str
    is_noise: bool
    members: list[tuple[SymptomRow, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparable category-based or symptom-based ticket clusters.")
    parser.add_argument("--method", choices=("category", "symptom"), required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Read-only ticket analytics snapshot.")
    parser.add_argument("--output", type=Path, default=None, help="Cluster SQLite output.")
    parser.add_argument("--replace", action="store_true", help="Atomically replace an existing cluster output.")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and show scope without clustering or writes.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Embedding device for symptom mode.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=400, help="Number of semantic symptom clusters.")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Hugging Face network access if the embedding model is not cached locally.",
    )
    return parser.parse_args()


def readonly_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Analytics snapshot not found: {resolved}")
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def validate_source(connection: sqlite3.Connection) -> None:
    required_tables = {"tickets", "ticket_symptoms", "ticket_product_groups", "ticket_modules"}
    actual = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    missing = sorted(required_tables - actual)
    if missing:
        raise RuntimeError(f"Analytics snapshot is missing tables: {', '.join(missing)}")
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    if integrity != "ok":
        raise RuntimeError(f"Source integrity check failed: {integrity}")


def load_symptoms(connection: sqlite3.Connection) -> list[SymptomRow]:
    rows = connection.execute(
        """
        SELECT
            s.ticket_id,
            s.ordinal,
            s.text,
            s.normalized_text,
            COALESCE(NULLIF(t.raw_category, ''), NULLIF(t.qdrant_category, ''), t.canonical_type_label) AS category,
            t.canonical_type_code,
            t.canonical_type_label,
            t.equipment
        FROM ticket_symptoms AS s
        JOIN tickets AS t ON t.ticket_id = s.ticket_id
        ORDER BY s.ticket_id, s.ordinal
        """
    ).fetchall()
    return [
        SymptomRow(
            ticket_id=str(row["ticket_id"]),
            ordinal=int(row["ordinal"]),
            text=str(row["text"]),
            normalized_text=str(row["normalized_text"]),
            category=str(row["category"] or "Не классифицировано"),
            canonical_type_code=str(row["canonical_type_code"]),
            canonical_type_label=str(row["canonical_type_label"]),
            equipment=str(row["equipment"] or ""),
        )
        for row in rows
    ]


def stable_id(prefix: str, value: str, size: int = 16) -> str:
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:size]
    return f"{prefix}-{digest}"


def category_clusters(rows: list[SymptomRow]) -> list[ClusterResult]:
    grouped: dict[str, list[SymptomRow]] = defaultdict(list)
    for row in rows:
        grouped[row.category].append(row)
    return [
        ClusterResult(
            cluster_id=stable_id("category", category.casefold()),
            label=category,
            representative_text=category,
            is_noise=False,
            members=[(row, 1.0) for row in members],
        )
        for category, members in sorted(grouped.items(), key=lambda item: item[0].casefold())
    ]


def symptom_clusters(
    rows: list[SymptomRow],
    model_name: str,
    device: str,
    batch_size: int,
    random_state: int,
    n_clusters: int,
    local_files_only: bool,
) -> tuple[list[ClusterResult], dict[str, Any]]:
    print("Importing sentence-transformers...", flush=True)
    from sentence_transformers import SentenceTransformer

    print("Importing MiniBatchKMeans...", flush=True)
    from sklearn.cluster import MiniBatchKMeans

    if not rows:
        return [], {}

    texts = [row.text for row in rows]
    print(f"Loading embedding model: {model_name} ({device})", flush=True)
    model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
    print(f"Encoding {len(texts)} symptoms...", flush=True)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    effective_clusters = max(2, min(n_clusters, len(rows)))
    print(f"Clustering normalized symptom embeddings with MiniBatchKMeans: k={effective_clusters}", flush=True)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    clusterer = MiniBatchKMeans(
        n_clusters=effective_clusters,
        random_state=random_state,
        batch_size=max(1024, batch_size * 8),
        n_init=3,
        max_iter=200,
        reassignment_ratio=0.01,
        verbose=0,
    )
    labels = clusterer.fit_predict(embeddings)
    centroids = np.asarray(clusterer.cluster_centers_, dtype=np.float32)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(centroid_norms, 1e-12)
    similarities = np.einsum("ij,ij->i", embeddings, centroids[labels])

    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        grouped_indices[int(label)].append(index)

    results: list[ClusterResult] = []
    for numeric_label, indices in sorted(grouped_indices.items()):
        cluster_similarities = similarities[indices]
        best_similarity = float(cluster_similarities.max())
        representative_index = min(
            index for index in indices if float(similarities[index]) == best_similarity
        )
        representative = rows[representative_index]
        cluster_id = stable_id("symptom", representative.normalized_text)
        results.append(
            ClusterResult(
                cluster_id=cluster_id,
                label=representative.text,
                representative_text=representative.text,
                is_noise=False,
                members=[
                    (rows[index], max(0.0, min(1.0, float(similarities[index]))))
                    for index in indices
                ],
            )
        )

    details = {
        "embedding_model": model_name,
        "device": device,
        "batch_size": batch_size,
        "random_state": random_state,
        "n_clusters": effective_clusters,
        "algorithm": "MiniBatchKMeans",
        "local_files_only": local_files_only,
    }
    return results, details


def medoid_index(indices: list[int], embeddings: np.ndarray) -> int:
    if len(indices) == 1:
        return indices[0]
    vectors = embeddings[indices]
    centroid = vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm:
        centroid = centroid / norm
    similarities = vectors @ centroid
    best_score = float(similarities.max())
    candidates = [indices[pos] for pos, score in enumerate(similarities) if float(score) == best_score]
    return min(candidates)


OUTPUT_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE run (
    run_id TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    symptom_count INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL,
    noise_count INTEGER NOT NULL
);

CREATE TABLE clusters (
    run_id TEXT NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    cluster_id TEXT NOT NULL,
    label TEXT NOT NULL,
    representative_text TEXT NOT NULL,
    is_noise INTEGER NOT NULL CHECK (is_noise IN (0, 1)),
    member_count INTEGER NOT NULL,
    ticket_count INTEGER NOT NULL,
    category_count INTEGER NOT NULL,
    canonical_type_count INTEGER NOT NULL,
    equipment_count INTEGER NOT NULL,
    mean_confidence REAL NOT NULL,
    PRIMARY KEY (run_id, cluster_id)
);

CREATE TABLE cluster_members (
    run_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    symptom_ordinal INTEGER NOT NULL,
    symptom_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    category TEXT NOT NULL,
    canonical_type_code TEXT NOT NULL,
    canonical_type_label TEXT NOT NULL,
    equipment TEXT NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY (run_id, ticket_id, symptom_ordinal),
    FOREIGN KEY (run_id, cluster_id) REFERENCES clusters(run_id, cluster_id) ON DELETE CASCADE
);

CREATE INDEX idx_clusters_size ON clusters(member_count DESC);
CREATE INDEX idx_members_cluster ON cluster_members(run_id, cluster_id);
CREATE INDEX idx_members_ticket ON cluster_members(ticket_id);
CREATE INDEX idx_members_category ON cluster_members(category);
CREATE INDEX idx_members_type ON cluster_members(canonical_type_code);
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_statistics(cluster: ClusterResult) -> tuple[int, int, int, int, float]:
    tickets = {row.ticket_id for row, _ in cluster.members}
    categories = {row.category for row, _ in cluster.members}
    canonical_types = {row.canonical_type_code for row, _ in cluster.members}
    equipment = {row.equipment for row, _ in cluster.members if row.equipment}
    confidences = [confidence for _, confidence in cluster.members]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return len(tickets), len(categories), len(canonical_types), len(equipment), mean_confidence


def write_clusters_atomic(
    output: Path,
    source: Path,
    method: str,
    rows: list[SymptomRow],
    clusters: list[ClusterResult],
    parameters: dict[str, Any],
    replace: bool,
) -> dict[str, int]:
    output = output.resolve()
    if output.exists() and not replace:
        raise FileExistsError(f"Output already exists: {output}. Use --replace to replace it atomically.")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        connection = sqlite3.connect(temp_path)
        try:
            connection.executescript(OUTPUT_SCHEMA_SQL)
            connection.execute(f"PRAGMA user_version = {CLUSTER_SCHEMA_VERSION}")
            run_id = stable_id(method, json.dumps(parameters, ensure_ascii=False, sort_keys=True), size=20)
            noise_count = sum(len(cluster.members) for cluster in clusters if cluster.is_noise)
            with connection:
                connection.execute(
                    """INSERT INTO run(
                           run_id, method, created_at, source_path, source_sha256, parameters_json,
                           symptom_count, cluster_count, noise_count
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        method,
                        datetime.now().isoformat(timespec="seconds"),
                        str(source.resolve()),
                        sha256_file(source),
                        json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                        len(rows),
                        sum(1 for cluster in clusters if not cluster.is_noise),
                        noise_count,
                    ),
                )
                for cluster in clusters:
                    ticket_count, category_count, type_count, equipment_count, mean_confidence = cluster_statistics(cluster)
                    connection.execute(
                        """INSERT INTO clusters(
                               run_id, cluster_id, label, representative_text, is_noise, member_count,
                               ticket_count, category_count, canonical_type_count, equipment_count, mean_confidence
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            run_id,
                            cluster.cluster_id,
                            cluster.label,
                            cluster.representative_text,
                            int(cluster.is_noise),
                            len(cluster.members),
                            ticket_count,
                            category_count,
                            type_count,
                            equipment_count,
                            mean_confidence,
                        ),
                    )
                    connection.executemany(
                        """INSERT INTO cluster_members(
                               run_id, cluster_id, ticket_id, symptom_ordinal, symptom_text, normalized_text,
                               category, canonical_type_code, canonical_type_label, equipment, confidence
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        [
                            (
                                run_id,
                                cluster.cluster_id,
                                row.ticket_id,
                                row.ordinal,
                                row.text,
                                row.normalized_text,
                                row.category,
                                row.canonical_type_code,
                                row.canonical_type_label,
                                row.equipment,
                                confidence,
                            )
                            for row, confidence in cluster.members
                        ],
                    )
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            foreign_key_errors = len(connection.execute("PRAGMA foreign_key_check").fetchall())
            member_count = int(connection.execute("SELECT COUNT(*) FROM cluster_members").fetchone()[0])
            if integrity != "ok" or foreign_key_errors or member_count != len(rows):
                raise RuntimeError(
                    f"Cluster output validation failed: integrity={integrity}, "
                    f"foreign_key_errors={foreign_key_errors}, members={member_count}/{len(rows)}"
                )
        finally:
            connection.close()
        os.replace(temp_path, output)
        return {
            "symptoms": len(rows),
            "clusters": sum(1 for cluster in clusters if not cluster.is_noise),
            "noise": sum(len(cluster.members) for cluster in clusters if cluster.is_noise),
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()


def preview(rows: list[SymptomRow]) -> dict[str, Any]:
    return {
        "symptoms": len(rows),
        "tickets": len({row.ticket_id for row in rows}),
        "categories": len({row.category for row in rows}),
        "canonical_types": dict(Counter(row.canonical_type_label for row in rows).most_common()),
    }


def main() -> None:
    args = parse_args()
    output = args.output or DEFAULT_OUTPUTS[args.method]
    if args.batch_size < 1 or args.n_clusters < 2:
        raise SystemExit("Invalid clustering parameters")
    if output.exists() and not args.replace and not args.dry_run:
        raise SystemExit(f"Output already exists: {output}. Use --replace to replace it atomically.")

    source_connection = readonly_connection(args.source)
    try:
        validate_source(source_connection)
        rows = load_symptoms(source_connection)
    finally:
        source_connection.close()

    print(json.dumps(preview(rows), ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete: no model was loaded and no files were created or changed.")
        return

    parameters: dict[str, Any] = {"method": args.method}
    if args.method == "category":
        clusters = category_clusters(rows)
    else:
        import sys

        sys.path.insert(0, str(ROOT))
        from src.config import settings

        device = args.device or settings.device
        clusters, symptom_parameters = symptom_clusters(
            rows=rows,
            model_name=settings.embedding_model_name,
            device=device,
            batch_size=args.batch_size,
            random_state=args.random_state,
            n_clusters=args.n_clusters,
            local_files_only=not args.allow_model_download,
        )
        parameters.update(symptom_parameters)

    stats = write_clusters_atomic(output, args.source, args.method, rows, clusters, parameters, args.replace)
    print(f"Cluster output written atomically: {output.resolve()}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
