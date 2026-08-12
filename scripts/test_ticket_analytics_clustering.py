from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cluster_ticket_analytics import (
    SymptomRow,
    category_clusters,
    load_symptoms,
    medoid_index,
    readonly_connection,
    validate_source,
    write_clusters_atomic,
)
from scripts.build_ticket_analytics_db import TicketAggregate, write_snapshot_atomic


def main() -> None:
    rows = [
        SymptomRow("RL-1", 1, "Ошибка загрузки", "ошибка загрузки", "Справочное", "reference", "Справочный вопрос", "R500"),
        SymptomRow("RL-2", 1, "Download denied", "download denied", "ОшибкаНастройкиПрософт", "configuration_error", "Ошибка конфигурации", "R500"),
        SymptomRow("RL-3", 1, "Нет связи", "нет связи", "Справочное", "reference", "Справочный вопрос", "R400"),
    ]
    clusters = category_clusters(rows)
    assert len(clusters) == 2
    assert sorted(len(cluster.members) for cluster in clusters) == [1, 2]

    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32)
    assert medoid_index([0, 1], embeddings) in (0, 1)

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "source.sqlite3"
        tickets = {}
        for row in rows:
            ticket = TicketAggregate(ticket_id=row.ticket_id)
            ticket.fields.update(
                {
                    "csv_found": "1",
                    "raw_category": row.category,
                    "equipment": row.equipment,
                    "title": row.text,
                }
            )
            ticket.symptoms.append(row.text)
            tickets[row.ticket_id] = ticket
        write_snapshot_atomic(source, tickets, {"test": "true"}, replace=False)

        connection = readonly_connection(source)
        try:
            validate_source(connection)
            loaded = load_symptoms(connection)
        finally:
            connection.close()
        assert len(loaded) == 3

        output = root / "category.sqlite3"
        stats = write_clusters_atomic(
            output,
            source,
            "category",
            loaded,
            category_clusters(loaded),
            {"method": "category"},
            replace=False,
        )
        assert stats == {"symptoms": 3, "clusters": 2, "noise": 0}

        connection = sqlite3.connect(output)
        try:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert connection.execute("SELECT COUNT(*) FROM cluster_members").fetchone()[0] == 3
        finally:
            connection.close()

        original = output.read_bytes()
        try:
            write_clusters_atomic(
                output,
                source,
                "category",
                loaded,
                category_clusters(loaded),
                {"method": "category"},
                replace=False,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("Cluster output must not be replaced without --replace")
        assert output.read_bytes() == original

    print("ticket analytics clustering tests: OK")


if __name__ == "__main__":
    main()
