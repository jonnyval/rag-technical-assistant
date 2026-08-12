from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import yaml
from langchain_core.documents import Document


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ticket_clustering.hermes_agent.service import ArticleResearchService


class FakeRetriever:
    def retrieve_docs(self, query, **kwargs):
        return [Document(page_content=f"doc {query}\n" * 300, metadata={"page_title": "Manual", "source_file": "manual.htm"})]

    def retrieve_tickets(self, query, **kwargs):
        return [Document(page_content=f"ticket {query}\n" * 300, metadata={"ticket_id": "RL-1", "source_file": "RL-1.json"})]


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        report = root / "report.json"
        report.write_text(
            json.dumps(
                {
                    "article_candidates": [{"cluster_id": "topic-1", "label": "OPC UA", "ticket_ids": ["RL-1"]}],
                    "clusters": [{
                        "cluster_id": "topic-1",
                        "label": "OPC UA",
                        "tickets": 1,
                        "ticket_ids": ["RL-1"],
                        "samples": [{"text": "Ошибка OPC UA", "entities": {"protocols": ["opc_ua"]}}],
                        "article_readiness": {"status": "ready", "score": 0.8},
                    }],
                    "subclusters": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        database = root / "analytics.sqlite3"
        connection = sqlite3.connect(database)
        connection.executescript(
            """
            CREATE TABLE tickets(ticket_id TEXT PRIMARY KEY, equipment TEXT, canonical_type_label TEXT,
                status TEXT, created_at TEXT, closed_at TEXT, ticket_url TEXT);
            CREATE TABLE ticket_symptoms(ticket_id TEXT, ordinal INTEGER, text TEXT);
            CREATE TABLE ticket_solutions(ticket_id TEXT, ordinal INTEGER, text TEXT);
            INSERT INTO tickets VALUES('RL-1','R500','Справочное','closed','','','https://ticket/RL-1');
            INSERT INTO ticket_symptoms VALUES('RL-1',1,'Ошибка OPC UA');
            INSERT INTO ticket_solutions VALUES('RL-1',1,'Настроен сертификат');
            """
        )
        connection.commit()
        connection.close()
        config = root / "config.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "candidate_report": str(report),
                    "analytics_db": str(database),
                    "output_dir": str(root / "output"),
                    "search": {
                        "search_snippet_chars": 300,
                        "detail_content_chars": 700,
                        "max_detail_results": 1,
                        "max_exact_tickets": 10,
                    },
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        service = ArticleResearchService(
            config,
            engine_factory=lambda: SimpleNamespace(retriever=FakeRetriever()),
        )
        assert service.list_article_candidates()["count"] == 1
        assert service.get_article_seed("topic-1")["verification_ticket_ids"] == ["RL-1"]
        search = service.search_dual("OPC UA")
        assert search["documentation"]["count"] == 1
        doc_result = search["documentation"]["results"][0]
        ticket_result = search["tickets"]["results"][0]
        assert len(doc_result["snippet"]) <= 300
        assert len(ticket_result["snippet"]) <= 300
        assert "content" not in doc_result
        expanded = service.read_search_results(
            [doc_result["result_ref"], ticket_result["result_ref"]],
        )
        assert expanded["count"] == 1
        assert len(expanded["results"][0]["content"]) <= 700
        assert service.get_tickets_by_ids(["RL-1"])["tickets"][0]["solutions"] == ["Настроен сертификат"]
        saved = service.save_article_draft("topic-1", "OPC UA", "# Draft", ["RL-1"], ["manual.htm"])
        assert saved["saved"] and Path(saved["markdown_path"]).is_file()
    print("Hermes article service tests: OK")


if __name__ == "__main__":
    main()
