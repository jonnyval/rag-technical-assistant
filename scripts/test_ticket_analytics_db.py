from __future__ import annotations

import csv
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_ticket_analytics_db import (
    TicketAggregate,
    canonical_ticket_type,
    load_csv_tickets,
    merge_qdrant_payload,
    normalize_ticket_id,
    values_list,
    write_snapshot_atomic,
)


def main() -> None:
    assert normalize_ticket_id("[rl_00123] example.json") == "RL-123"
    assert values_list("['one', 'two']") == ["one", "two"]
    assert canonical_ticket_type("ОшибкаНастройкиПрософт", "") == (
        "configuration_error",
        "Ошибка конфигурации",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        csv_path = root / "tickets.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "Код",
                    "Наименование",
                    "Текст без html",
                    "Результат без html",
                    "Категория обращения РегЛаб",
                    "Статус.Имя статуса",
                ],
                delimiter=";",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "Код": "RL-42",
                    "Наименование": "Test ticket",
                    "Текст без html": "Description",
                    "Результат без html": "Result",
                    "Категория обращения РегЛаб": "Справочное",
                    "Статус.Имя статуса": "Closed",
                }
            )

        tickets = load_csv_tickets(csv_path)
        ticket = tickets["RL-42"]
        merge_qdrant_payload(
            ticket,
            {
                "metadata": {
                    "ticket_id": "RL-42",
                    "ticket_url": "https://example.invalid/RL-42",
                    "llm_symptoms": ["Ошибка Ё", "ошибка е"],
                    "llm_solution": ["Перезапустить контроллер"],
                    "quality_tags": ["PLC"],
                    "ticket_product_groups": ["r500", "astraide"],
                    "ticket_product_group_titles": ["REGUL R500", "AstraIDE"],
                    "mentioned_module_codes": ["CU 00 051"],
                    "mentioned_modules": ["R500 CU 00 051"],
                    "mentioned_module_families": ["REGUL R500"],
                    "mentioned_module_functions": ["central_processing_unit"],
                }
            },
        )

        output = root / "analytics.sqlite3"
        stats = write_snapshot_atomic(output, tickets, {"test": "true"}, replace=False)
        assert stats["tickets"] == 1
        assert stats["joined_tickets"] == 1
        assert stats["symptoms"] == 1
        assert stats["solutions"] == 1
        assert stats["product_groups"] == 2
        assert stats["modules"] == 1

        original = output.read_bytes()
        try:
            write_snapshot_atomic(output, tickets, {"test": "true"}, replace=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing snapshot must not be replaced without --replace")
        assert output.read_bytes() == original

        connection = sqlite3.connect(output)
        try:
            row = connection.execute(
                "SELECT canonical_type_code, csv_found, qdrant_found FROM tickets WHERE ticket_id = 'RL-42'"
            ).fetchone()
            assert row == ("reference", 1, 1)
        finally:
            connection.close()

    print("ticket analytics database tests: OK")


if __name__ == "__main__":
    main()
