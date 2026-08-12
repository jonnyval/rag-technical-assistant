from __future__ import annotations

import argparse
import time

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as error:  # pragma: no cover - exercised only before optional install
    raise SystemExit(
        "MCP SDK is not installed. Run: pip install -r "
        "ticket_clustering/hermes_agent/requirements-hermes.txt"
    ) from error

from ticket_clustering.hermes_agent.service import ArticleResearchService


mcp = FastMCP(
    name="RegLab Article Research",
    instructions=(
        "Read-only research tools for RegLab knowledge articles. Start from an article candidate, "
        "search documentation and historical tickets with multiple focused queries, verify exact "
        "cluster tickets, distinguish evidence from hypotheses, and save only a draft."
    ),
    host="127.0.0.1",
    port=8765,
)
service = ArticleResearchService()


def _paced(result: dict) -> dict:
    """Space agent iterations so round-robin Groq keys do not recycle inside one TPM window."""
    delay = max(0.0, float(service.config.get("agent_call_pacing_seconds", 0) or 0))
    if delay:
        time.sleep(delay)
    return result


@mcp.tool()
def list_article_candidates(limit: int = 30) -> dict:
    """List recurring topics selected by deterministic clustering; this is a topic map, not evidence."""
    return _paced(service.list_article_candidates(limit))


@mcp.tool()
def get_article_seed(cluster_id: str) -> dict:
    """Get a persisted cluster or manual research seed by its identifier."""
    return _paced(service.get_article_seed(cluster_id))


@mcp.tool()
def create_research_seed(topic: str) -> dict:
    """Create a persistent research seed from a freely supplied topic; the topic is not evidence."""
    return _paced(service.create_research_seed(topic))


@mcp.tool()
def search_documentation(query: str, limit: int = 8) -> dict:
    """Search official documentation through the documentation side of DualRetriever."""
    return _paced(service.search_documentation(query, limit))


@mcp.tool()
def search_historical_tickets(query: str, limit: int = 10) -> dict:
    """Search historical support cases through the ticket side of DualRetriever."""
    return _paced(service.search_tickets(query, limit))


@mcp.tool()
def search_dual_retriever(query: str, docs_limit: int = 8, tickets_limit: int = 10) -> dict:
    """Run one focused query against both documentation and ticket collections."""
    return _paced(service.search_dual(query, docs_limit, tickets_limit))


@mcp.tool()
def read_search_results(result_refs: list[str], max_chars: int = 5000) -> dict:
    """Read full text only for selected result_ref values returned by search tools."""
    return _paced(service.read_search_results(result_refs, max_chars))


@mcp.tool()
def get_cluster_tickets(ticket_ids: list[str]) -> dict:
    """Read structured symptoms and solutions for exact ticket IDs to audit recall and contradictions."""
    return _paced(service.get_tickets_by_ids(ticket_ids))


@mcp.tool()
def save_article_draft(
    cluster_id: str,
    title: str,
    markdown: str,
    evidence_ticket_ids: list[str],
    evidence_document_sources: list[str],
    research_notes_markdown: str = "",
    research_ticket_ids: list[str] | None = None,
    research_document_sources: list[str] | None = None,
    related_article_topics: list[str] | None = None,
) -> dict:
    """Save an atomic EVA draft plus a separate research dossier for adjacent findings."""
    return service.save_article_draft(
        cluster_id,
        title,
        markdown,
        evidence_ticket_ids,
        evidence_document_sources,
        research_notes_markdown,
        research_ticket_ids,
        research_document_sources,
        related_article_topics,
    )


if __name__ == "__main__":
    # On Windows, loading Torch/SentenceTransformer lazily from FastMCP's
    # worker thread may stall.  Initialize the shared engine on the process
    # main thread before FastMCP starts dispatching synchronous tools.
    service.warm_up()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)
