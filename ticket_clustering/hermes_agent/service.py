from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.yaml")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def safe_slug(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", value).strip("_")
    return value[:100] or "article"


class ArticleResearchService:
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        *,
        engine_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        self.report_path = (ROOT / self.config["candidate_report"]).resolve()
        self.analytics_db = (ROOT / self.config["analytics_db"]).resolve()
        self.output_dir = (ROOT / self.config["output_dir"]).resolve()
        self.manual_seed_dir = (self.output_dir / "_research_seeds").resolve()
        self.search_config = self.config.get("search", {})
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._lock = threading.RLock()
        self._result_cache: OrderedDict[str, tuple[Any, str]] = OrderedDict()
        self._result_cache_limit = max(20, int(self.search_config.get("result_cache_size", 200)))

    def _load_report(self) -> dict[str, Any]:
        return json.loads(self.report_path.read_text(encoding="utf-8"))

    def _get_engine(self) -> Any:
        with self._lock:
            if self._engine is None:
                if self._engine_factory is not None:
                    self._engine = self._engine_factory()
                else:
                    from src.engine import RAGEngine

                    self._engine = RAGEngine(profile=str(self.config.get("rag_profile", "deep")))
            return self._engine

    def warm_up(self) -> None:
        """Initialize model-backed retrieval on the MCP process main thread."""
        self._get_engine()

    def list_article_candidates(self, limit: int = 30) -> dict[str, Any]:
        report = self._load_report()
        bounded = max(1, min(int(limit), 100))
        candidates = report.get("article_candidates", [])[:bounded]
        return {"count": len(candidates), "candidates": candidates}

    @staticmethod
    def _find_cluster(report: dict[str, Any], cluster_id: str) -> tuple[dict[str, Any] | None, str]:
        for cluster in report.get("clusters", []):
            if cluster.get("cluster_id") == cluster_id:
                return cluster, ""
        for split in report.get("subclusters", []):
            for child in split.get("children", []):
                if child.get("cluster_id") == cluster_id:
                    return child, str(split.get("parent_label") or "")
        return None, ""

    def get_article_seed(self, cluster_id: str) -> dict[str, Any]:
        report = self._load_report()
        cluster, parent_label = self._find_cluster(report, cluster_id)
        if cluster is None:
            manual_seed = self._load_manual_seed(cluster_id)
            if manual_seed is not None:
                return manual_seed
            return {"error": "unknown_research_id", "research_id": cluster_id}
        readiness = cluster.get("article_readiness", {}) or {}
        return {
            "cluster_id": cluster_id,
            "label": cluster.get("label"),
            "parent_label": parent_label,
            "ticket_count": cluster.get("tickets"),
            "readiness": {
                "status": readiness.get("status"),
                "score": readiness.get("score"),
                "reasons": readiness.get("reasons", []),
            },
            "representative_symptoms": [item.get("text") for item in cluster.get("samples", [])],
            "verification_ticket_ids": cluster.get("ticket_ids", []),
            "instruction": (
                "Use this only as a research seed. Search documentation and tickets independently; "
                "then use verification_ticket_ids to audit recall and contradictions."
            ),
        }

    def create_research_seed(self, topic: str) -> dict[str, Any]:
        """Create a persistent deterministic seed for a user-supplied research topic."""
        normalized_topic = " ".join(str(topic).split()).strip()
        if not normalized_topic:
            return {"error": "empty_topic"}
        if len(normalized_topic) > 500:
            return {"error": "topic_too_long", "max_chars": 500}
        digest = hashlib.sha256(normalized_topic.casefold().encode("utf-8")).hexdigest()[:20]
        research_id = f"manual-{digest}"
        seed = {
            "research_id": research_id,
            "cluster_id": research_id,
            "origin": "manual_topic",
            "label": normalized_topic,
            "parent_label": "",
            "ticket_count": None,
            "readiness": {"status": "manual", "score": None, "reasons": []},
            "representative_symptoms": [normalized_topic],
            "verification_ticket_ids": [],
            "instruction": (
                "This is a user-supplied research direction, not evidence. Resolve ambiguous terms "
                "from retrieved sources, search documentation and tickets independently, and do not "
                "claim recurrence or frequency without supporting data."
            ),
        }
        seed_path = (self.manual_seed_dir / f"{research_id}.json").resolve()
        if self.manual_seed_dir not in seed_path.parents:
            raise ValueError("Unsafe manual seed path")
        if seed_path.is_file():
            return json.loads(seed_path.read_text(encoding="utf-8"))
        atomic_write(seed_path, json.dumps(seed, ensure_ascii=False, indent=2))
        return seed

    def _load_manual_seed(self, research_id: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"manual-[0-9a-f]{20}", str(research_id)):
            return None
        seed_path = (self.manual_seed_dir / f"{research_id}.json").resolve()
        if self.manual_seed_dir not in seed_path.parents or not seed_path.is_file():
            return None
        return json.loads(seed_path.read_text(encoding="utf-8"))

    def _cache_document(self, document: Any, source: str) -> str:
        metadata = getattr(document, "metadata", {}) or {}
        content = str(getattr(document, "page_content", "") or "")
        identity = "|".join(
            (
                source,
                str(metadata.get("ticket_id") or ""),
                str(metadata.get("doc_id") or ""),
                str(metadata.get("source_file") or ""),
                content,
            )
        )
        result_ref = f"{source[0]}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
        with self._lock:
            self._result_cache[result_ref] = (document, source)
            self._result_cache.move_to_end(result_ref)
            while len(self._result_cache) > self._result_cache_limit:
                self._result_cache.popitem(last=False)
        return result_ref

    @staticmethod
    def _put_if_value(payload: dict[str, Any], key: str, value: Any) -> None:
        if value not in (None, "", [], {}):
            payload[key] = value

    @staticmethod
    def _compact_text(text: str, max_chars: int) -> str:
        """Fit agent-facing text into a hard character budget."""
        marker = "\n[…обрезано]"
        normalized = str(text or "").strip()
        if max_chars <= 0 or len(normalized) <= max_chars:
            return normalized
        body_limit = max(1, max_chars - len(marker))
        body = normalized[:body_limit].rsplit("\n", 1)[0].strip()
        if not body:
            body = normalized[:body_limit].strip()
        return f"{body}{marker}"

    def _serialize_documents(self, documents: list[Any], source: str, limit: int) -> list[dict[str, Any]]:
        from src.context_formatting import resolve_source_url

        max_chars = max(200, int(self.search_config.get("search_snippet_chars", 700)))
        result = []
        for document in documents[:limit]:
            metadata = getattr(document, "metadata", {}) or {}
            content = str(getattr(document, "page_content", "") or "")
            ticket = source == "tickets"
            payload: dict[str, Any] = {
                "result_ref": self._cache_document(document, source),
                "title": str(metadata.get("page_title") or metadata.get("ticket_id") or metadata.get("source_file") or ""),
                "snippet": self._compact_text(content, max_chars),
            }
            self._put_if_value(payload, "ticket_id", str(metadata.get("ticket_id") or ""))
            self._put_if_value(payload, "source_file", str(metadata.get("source_file") or ""))
            self._put_if_value(payload, "url", resolve_source_url(metadata, ticket_only=ticket))
            self._put_if_value(payload, "equipment", str(metadata.get("equipment_type") or ""))
            self._put_if_value(payload, "release", str(metadata.get("release_version") or ""))
            score = float(metadata.get("rerank_score", 0.0) or 0.0)
            if score:
                payload["score"] = round(score, 4)
            result.append(payload)
        return result

    def read_search_results(self, result_refs: list[str], max_chars: int | None = None) -> dict[str, Any]:
        """Expand selected search snippets from the bounded in-memory result cache."""
        from src.context_formatting import resolve_source_url

        maximum = max(1, min(int(self.search_config.get("max_detail_results", 6)), 12))
        refs = list(dict.fromkeys(str(value).strip() for value in result_refs if str(value).strip()))[:maximum]
        content_limit = max(
            500,
            min(int(max_chars or self.search_config.get("detail_content_chars", 5000)), 12_000),
        )
        results: list[dict[str, Any]] = []
        missing: list[str] = []
        with self._lock:
            cached_results = []
            for result_ref in refs:
                cached = self._result_cache.get(result_ref)
                if cached is None:
                    missing.append(result_ref)
                    continue
                self._result_cache.move_to_end(result_ref)
                cached_results.append((result_ref, *cached))
        for result_ref, document, source in cached_results:
            metadata = getattr(document, "metadata", {}) or {}
            content = str(getattr(document, "page_content", "") or "")
            ticket = source == "tickets"
            payload: dict[str, Any] = {
                "result_ref": result_ref,
                "source": source,
                "title": str(metadata.get("page_title") or metadata.get("ticket_id") or metadata.get("source_file") or ""),
                "content": self._compact_text(content, content_limit),
            }
            self._put_if_value(payload, "ticket_id", str(metadata.get("ticket_id") or ""))
            self._put_if_value(payload, "source_file", str(metadata.get("source_file") or ""))
            self._put_if_value(payload, "url", resolve_source_url(metadata, ticket_only=ticket))
            self._put_if_value(payload, "equipment", str(metadata.get("equipment_type") or ""))
            self._put_if_value(payload, "release", str(metadata.get("release_version") or ""))
            self._put_if_value(payload, "breadcrumb", str(metadata.get("breadcrumb_raw") or ""))
            results.append(payload)
        return {"count": len(results), "results": results, "missing_refs": missing}

    def search_documentation(self, query: str, limit: int | None = None) -> dict[str, Any]:
        query = " ".join(str(query).split())
        if not query:
            return {"error": "empty_query"}
        bounded = max(1, min(int(limit or self.search_config.get("docs_limit", 8)), 20))
        with self._lock:
            docs = self._get_engine().retriever.retrieve_docs(query, use_reranker=True)
        return {"query": query, "count": min(len(docs), bounded), "results": self._serialize_documents(docs, "docs", bounded)}

    def search_tickets(self, query: str, limit: int | None = None) -> dict[str, Any]:
        query = " ".join(str(query).split())
        if not query:
            return {"error": "empty_query"}
        bounded = max(1, min(int(limit or self.search_config.get("tickets_limit", 10)), 20))
        child_k = max(bounded, min(int(self.search_config.get("tickets_child_k", 80)), 150))
        with self._lock:
            docs = self._get_engine().retriever.retrieve_tickets(
                query,
                final_limit=bounded,
                child_k=child_k,
                use_reranker=True,
            )
        return {"query": query, "count": min(len(docs), bounded), "results": self._serialize_documents(docs, "tickets", bounded)}

    def search_dual(self, query: str, docs_limit: int | None = None, tickets_limit: int | None = None) -> dict[str, Any]:
        return {
            "documentation": self.search_documentation(query, docs_limit),
            "tickets": self.search_tickets(query, tickets_limit),
        }

    def get_tickets_by_ids(self, ticket_ids: list[str]) -> dict[str, Any]:
        maximum = max(1, int(self.search_config.get("max_exact_tickets", 30)))
        ids = list(dict.fromkeys(str(value).strip() for value in ticket_ids if str(value).strip()))[:maximum]
        if not ids:
            return {"count": 0, "tickets": []}
        placeholders = ",".join("?" for _ in ids)
        connection = sqlite3.connect(f"file:{self.analytics_db.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            tickets = {
                row["ticket_id"]: dict(row)
                for row in connection.execute(
                    f"""SELECT ticket_id, equipment, canonical_type_label AS category, status,
                    created_at, closed_at, ticket_url FROM tickets WHERE ticket_id IN ({placeholders})""",
                    ids,
                )
            }
            symptoms: dict[str, list[str]] = {ticket_id: [] for ticket_id in ids}
            solutions: dict[str, list[str]] = {ticket_id: [] for ticket_id in ids}
            for row in connection.execute(
                f"SELECT ticket_id, text FROM ticket_symptoms WHERE ticket_id IN ({placeholders}) ORDER BY ticket_id, ordinal",
                ids,
            ):
                symptoms[row["ticket_id"]].append(row["text"])
            for row in connection.execute(
                f"SELECT ticket_id, text FROM ticket_solutions WHERE ticket_id IN ({placeholders}) ORDER BY ticket_id, ordinal",
                ids,
            ):
                solutions[row["ticket_id"]].append(row["text"])
        finally:
            connection.close()
        rows = []
        for ticket_id in ids:
            if ticket_id not in tickets:
                continue
            rows.append({**tickets[ticket_id], "symptoms": symptoms[ticket_id], "solutions": solutions[ticket_id]})
        return {"count": len(rows), "tickets": rows, "missing_ids": [value for value in ids if value not in tickets]}

    def save_article_draft(
        self,
        cluster_id: str,
        title: str,
        markdown: str,
        evidence_ticket_ids: list[str],
        evidence_document_sources: list[str],
        research_notes_markdown: str = "",
        research_ticket_ids: list[str] | None = None,
        research_document_sources: list[str] | None = None,
        related_article_topics: list[str] | None = None,
    ) -> dict[str, Any]:
        report = self._load_report()
        cluster, _ = self._find_cluster(report, cluster_id)
        origin = "cluster"
        research_topic = str(cluster.get("label") or "") if cluster is not None else ""
        if cluster is None:
            manual_seed = self._load_manual_seed(cluster_id)
            if manual_seed is None:
                return {"error": "unknown_research_id", "research_id": cluster_id}
            origin = "manual_topic"
            research_topic = str(manual_seed.get("label") or "")
        if not str(markdown).strip():
            return {"error": "empty_markdown"}
        base = f"{safe_slug(cluster_id)}_{safe_slug(title)}"
        markdown_path = (self.output_dir / f"{base}.md").resolve()
        metadata_path = (self.output_dir / f"{base}.json").resolve()
        research_path = (self.output_dir / f"{base}_research.md").resolve()
        if any(
            self.output_dir not in path.parents
            for path in (markdown_path, metadata_path, research_path)
        ):
            raise ValueError("Unsafe output path")
        atomic_write(markdown_path, str(markdown).strip() + "\n")

        research_notes = str(research_notes_markdown or "").strip()
        if research_notes:
            atomic_write(research_path, research_notes + "\n")

        article_evidence_ids = sorted(set(str(value).strip() for value in evidence_ticket_ids if str(value).strip()))
        all_research_ticket_ids = sorted(
            set(str(value).strip() for value in (research_ticket_ids or []) if str(value).strip())
        )
        article_cited_ticket_ids = sorted(set(re.findall(r"\bRL-\d+\b", str(markdown))))
        citation_warnings = {
            "cited_not_declared": sorted(set(article_cited_ticket_ids) - set(article_evidence_ids)),
            "declared_not_cited": sorted(set(article_evidence_ids) - set(article_cited_ticket_ids)),
        }
        research_cited_ticket_ids = sorted(set(re.findall(r"\bRL-\d+\b", research_notes)))
        research_citation_warnings = {
            "cited_not_declared": sorted(set(research_cited_ticket_ids) - set(all_research_ticket_ids)),
            "declared_not_cited": sorted(set(all_research_ticket_ids) - set(research_cited_ticket_ids)),
        }
        metadata = {
            "cluster_id": cluster_id,
            "origin": origin,
            "research_topic": research_topic,
            "title": title,
            "article_evidence": {
                "ticket_ids": article_evidence_ids,
                "document_sources": sorted(set(evidence_document_sources)),
            },
            "research_evidence": {
                "ticket_ids": all_research_ticket_ids,
                "document_sources": sorted(set(research_document_sources or [])),
            },
            "related_article_topics": list(dict.fromkeys(
                str(value).strip() for value in (related_article_topics or []) if str(value).strip()
            )),
            "citation_audit": citation_warnings,
            "research_citation_audit": research_citation_warnings,
            "research_notes_path": str(research_path) if research_notes else "",
            "status": "draft_not_published",
        }
        atomic_write(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2))
        return {
            "saved": True,
            "markdown_path": str(markdown_path),
            "metadata_path": str(metadata_path),
            "research_notes_path": str(research_path) if research_notes else "",
            "citation_audit": citation_warnings,
            "research_citation_audit": research_citation_warnings,
        }
