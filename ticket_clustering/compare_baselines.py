from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from ticket_clustering.reporting import atomic_write_text


PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline clustering JSON reports.")
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        help="JSON reports. By default all baseline_*.json files are used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "reports" / "baseline_comparison.html",
    )
    parser.add_argument("--top-clusters", type=int, default=12)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def load_reports(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    loaded = []
    for path in paths:
        resolved = path.resolve()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        loaded.append((resolved, payload))
    return loaded


def config_label(payload: dict[str, Any]) -> str:
    config = payload["config"]
    neighbors = config["neighbors"]
    clustering = config["clustering"]
    return (
        f"{config.get('strategy', 'semantic-only')}: k={neighbors['graph_neighbors']}, "
        f"threshold={float(neighbors['similarity_threshold']):.3f}, "
        f"resolution={float(clustering['resolution']):.2f}"
    )


def render(reports: list[tuple[Path, dict[str, Any]]], top_clusters: int) -> str:
    rows = []
    sections = []
    for path, payload in reports:
        summary = payload["summary"]
        label = config_label(payload)
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(label)}</b><br><small>{html.escape(path.name)}</small></td>"
            f"<td>{summary['coverage_tickets'] * 100:.1f}%</td>"
            f"<td>{summary['coverage_unique_symptoms'] * 100:.1f}%</td>"
            f"<td>{summary['accepted_clusters']}</td>"
            f"<td>{summary.get('cluster_ticket_size_median', '—')}</td>"
            f"<td>{summary.get('cluster_ticket_size_p90', '—')}</td>"
            f"<td>{summary.get('cluster_ticket_size_max', '—')}</td>"
            f"<td>{summary.get('cluster_mean_edge_similarity_weighted', '—')}</td>"
            f"<td>{summary.get('article_ready_clusters', '—')}</td>"
            f"<td>{summary.get('article_review_clusters', '—')}</td>"
            f"<td>{summary.get('article_ready_subclusters', '—')}</td>"
            f"<td>{summary['isolated_nodes']}</td>"
            "</tr>"
        )
        cluster_items = []
        for cluster in payload["clusters"][:top_clusters]:
            samples = "".join(
                f"<li>{html.escape(sample['text'])}</li>"
                for sample in cluster["samples"][:6]
            )
            cluster_items.append(
                "<details>"
                f"<summary><b>{html.escape(cluster['label'])}</b> — "
                f"{cluster['tickets']} тикетов, similarity {cluster['mean_edge_similarity']:.3f}</summary>"
                f"<ol>{samples}</ol></details>"
            )
        sections.append(
            f"<section><h2>{html.escape(label)}</h2>{''.join(cluster_items)}</section>"
        )

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Сравнение baseline-кластеризации</title>
<style>
body{{font:14px/1.45 system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#20242a}}
table{{border-collapse:collapse;width:100%;position:relative}}th,td{{border:1px solid #d9dee6;padding:8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{background:#f5f7fa;position:sticky;top:0}}small{{color:#667085}}
details{{border:1px solid #d9dee6;border-radius:8px;padding:8px;margin:6px 0}}summary{{cursor:pointer}}
section{{margin-top:28px}}
</style></head><body>
<h1>Сравнение baseline-кластеризации</h1>
<p>Числа показывают компромисс между покрытием и укрупнением кластеров. Они не заменяют смысловую проверку примеров ниже.</p>
<table><thead><tr><th>Параметры</th><th>Тикеты</th><th>Формулировки</th><th>Кластеры</th>
<th>Медиана тикетов</th><th>P90</th><th>Максимум</th><th>Связность</th><th>Готовы</th><th>Проверка</th><th>Готовые подкластеры</th><th>Изоляты</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
{''.join(sections)}
</body></html>"""


def main() -> None:
    args = parse_args()
    paths = args.reports or sorted((PACKAGE_ROOT / "reports").glob("baseline_*.json"))
    if not paths:
        raise SystemExit("No baseline JSON reports found")
    reports = load_reports(paths)
    output = args.output.resolve()
    atomic_write_text(output, render(reports, args.top_clusters), replace=args.replace)
    print(f"Compared reports: {len(reports)}")
    print(f"HTML: {output}")


if __name__ == "__main__":
    main()
