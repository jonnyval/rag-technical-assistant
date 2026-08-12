from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SERVER_DATA = ROOT / "analytics" / "tp_analyze" / "server_data"
DEFAULT_CATEGORY = SERVER_DATA / "clusters_by_category.sqlite3"
DEFAULT_SYMPTOM = SERVER_DATA / "clusters_by_symptom.sqlite3"
DEFAULT_HTML = ROOT / "reports" / "ticket_clustering_comparison.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare category and symptom ticket clustering outputs.")
    parser.add_argument("--category", type=Path, default=DEFAULT_CATEGORY)
    parser.add_argument("--symptom", type=Path, default=DEFAULT_SYMPTOM)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def connect_readonly(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    connection = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        connection.close()
        raise RuntimeError(f"Cluster database integrity check failed: {resolved}")
    return connection


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def weighted_purity(connection: sqlite3.Connection, field: str) -> float:
    if field not in {"category", "canonical_type_code"}:
        raise ValueError(field)
    rows = connection.execute(
        f"""
        SELECT cluster_id, {field}, COUNT(*) AS amount
        FROM cluster_members
        GROUP BY cluster_id, {field}
        """
    ).fetchall()
    totals: Counter[str] = Counter()
    maxima: Counter[str] = Counter()
    for row in rows:
        cluster_id = str(row["cluster_id"])
        amount = int(row["amount"])
        totals[cluster_id] += amount
        maxima[cluster_id] = max(maxima[cluster_id], amount)
    denominator = sum(totals.values())
    return sum(maxima.values()) / denominator if denominator else 0.0


def distributions(connection: sqlite3.Connection, cluster_id: str, field: str, limit: int = 5) -> list[dict[str, Any]]:
    if field not in {"category", "canonical_type_label", "equipment"}:
        raise ValueError(field)
    rows = connection.execute(
        f"""
        SELECT {field} AS name, COUNT(*) AS amount
        FROM cluster_members
        WHERE cluster_id = ? AND {field} != ''
        GROUP BY {field}
        ORDER BY amount DESC, name
        LIMIT ?
        """,
        (cluster_id, limit),
    ).fetchall()
    return [{"name": str(row["name"]), "count": int(row["amount"])} for row in rows]


def sample_members(connection: sqlite3.Connection, cluster_id: str, limit: int) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT ticket_id, symptom_text, category, canonical_type_label, equipment, confidence
        FROM cluster_members
        WHERE cluster_id = ?
        ORDER BY confidence DESC, ticket_id, symptom_ordinal
        LIMIT ?
        """,
        (cluster_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def analyze(path: Path, top: int, samples: int) -> dict[str, Any]:
    connection = connect_readonly(path)
    try:
        run = dict(connection.execute("SELECT * FROM run").fetchone())
        cluster_rows = connection.execute(
            """
            SELECT * FROM clusters
            ORDER BY is_noise, member_count DESC, cluster_id
            """
        ).fetchall()
        sizes = [int(row["member_count"]) for row in cluster_rows if not row["is_noise"]]
        total = sum(sizes)
        metrics = {
            "method": run["method"],
            "source_sha256": run["source_sha256"],
            "symptoms": int(run["symptom_count"]),
            "clusters": int(run["cluster_count"]),
            "noise": int(run["noise_count"]),
            "smallest_cluster": min(sizes, default=0),
            "p25_cluster": round(percentile(sizes, 0.25), 2),
            "median_cluster": round(statistics.median(sizes), 2) if sizes else 0,
            "p75_cluster": round(percentile(sizes, 0.75), 2),
            "largest_cluster": max(sizes, default=0),
            "largest_share": round(max(sizes, default=0) / total, 4) if total else 0,
            "category_purity": round(weighted_purity(connection, "category"), 4),
            "canonical_type_purity": round(weighted_purity(connection, "canonical_type_code"), 4),
            "mean_confidence": round(
                sum(float(row["mean_confidence"]) * int(row["member_count"]) for row in cluster_rows) / total,
                4,
            ) if total else 0,
        }
        clusters = []
        for row in cluster_rows[:top]:
            cluster_id = str(row["cluster_id"])
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "label": str(row["label"]),
                    "member_count": int(row["member_count"]),
                    "ticket_count": int(row["ticket_count"]),
                    "category_count": int(row["category_count"]),
                    "canonical_type_count": int(row["canonical_type_count"]),
                    "equipment_count": int(row["equipment_count"]),
                    "mean_confidence": round(float(row["mean_confidence"]), 4),
                    "categories": distributions(connection, cluster_id, "category"),
                    "canonical_types": distributions(connection, cluster_id, "canonical_type_label"),
                    "equipment": distributions(connection, cluster_id, "equipment"),
                    "samples": sample_members(connection, cluster_id, samples),
                }
            )
        return {"path": str(path.resolve()), "run": run, "metrics": metrics, "top_clusters": clusters}
    finally:
        connection.close()


def fmt_distribution(values: list[dict[str, Any]]) -> str:
    return ", ".join(f"{item['name']} ({item['count']})" for item in values) or "—"


def render_clusters(section: dict[str, Any]) -> str:
    cards = []
    for cluster in section["top_clusters"]:
        samples = "".join(
            "<li><code>{}</code> {} <small>{} · confidence {:.3f}</small></li>".format(
                html.escape(str(sample["ticket_id"])),
                html.escape(str(sample["symptom_text"])),
                html.escape(str(sample["category"])),
                float(sample["confidence"]),
            )
            for sample in cluster["samples"]
        )
        cards.append(
            f"""
            <details>
              <summary><b>{html.escape(cluster['label'])}</b> — {cluster['member_count']} симптомов / {cluster['ticket_count']} тикетов</summary>
              <p>Категории: {html.escape(fmt_distribution(cluster['categories']))}</p>
              <p>Типы: {html.escape(fmt_distribution(cluster['canonical_types']))}</p>
              <p>Оборудование: {html.escape(fmt_distribution(cluster['equipment']))}</p>
              <p>Средняя близость к центру: {cluster['mean_confidence']:.3f}</p>
              <ol>{samples}</ol>
            </details>
            """
        )
    return "\n".join(cards)


def render_html(payload: dict[str, Any]) -> str:
    category = payload["category"]
    symptom = payload["symptom"]
    metric_names = [
        ("symptoms", "Симптомов"),
        ("clusters", "Кластеров"),
        ("median_cluster", "Медианный размер"),
        ("largest_cluster", "Максимальный размер"),
        ("largest_share", "Доля крупнейшего"),
        ("category_purity", "Чистота по категории"),
        ("canonical_type_purity", "Чистота по типу"),
        ("mean_confidence", "Средняя близость"),
    ]
    comparison_rows = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{category['metrics'][key]}</td><td>{symptom['metrics'][key]}</td></tr>"
        for key, label in metric_names
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Сравнение кластеризации обращений</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#20242a}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #d9dee6;padding:8px;text-align:left}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}details{{border:1px solid #d9dee6;border-radius:8px;padding:10px;margin:8px 0}}
summary{{cursor:pointer}}small{{color:#667085}}code{{white-space:nowrap}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Сравнение кластеризации обращений</h1>
<p>Обе версии построены на одном snapshot: <code>{html.escape(payload['source_sha256'])}</code>.</p>
<table><thead><tr><th>Метрика</th><th>По категории</th><th>По симптомам</th></tr></thead><tbody>{comparison_rows}</tbody></table>
<p><b>Важно:</b> чистота по исходной категории не является оценкой тематической корректности. Для symptom-кластеров смешение категорий ожидаемо. Качество темы проверяется по примерам внутри кластеров.</p>
<div class="grid"><section><h2>Кластеры по категории</h2>{render_clusters(category)}</section>
<section><h2>Кластеры по симптомам</h2>{render_clusters(symptom)}</section></div>
</body></html>"""


def atomic_write(path: Path, content: str, replace: bool) -> None:
    path = path.resolve()
    if path.exists() and not replace:
        raise FileExistsError(f"Output already exists: {path}. Use --replace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp = Path(temp_name)
    try:
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> None:
    args = parse_args()
    json_path = args.json or args.html.with_suffix(".json")
    if args.top < 1 or args.samples < 1:
        raise SystemExit("--top and --samples must be positive")
    if not args.replace:
        existing = [path for path in (args.html, json_path) if path.exists()]
        if existing:
            raise SystemExit(f"Output already exists: {existing[0]}. Use --replace.")

    category = analyze(args.category, args.top, args.samples)
    symptom = analyze(args.symptom, args.top, args.samples)
    if category["metrics"]["source_sha256"] != symptom["metrics"]["source_sha256"]:
        raise RuntimeError("Cluster outputs were built from different analytics snapshots")
    payload = {
        "source_sha256": category["metrics"]["source_sha256"],
        "category": category,
        "symptom": symptom,
    }
    atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2), args.replace)
    try:
        atomic_write(args.html, render_html(payload), args.replace)
    except Exception:
        if json_path.exists() and not args.replace:
            json_path.unlink()
        raise
    print(f"HTML: {args.html.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    print(json.dumps({"category": category["metrics"], "symptom": symptom["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
