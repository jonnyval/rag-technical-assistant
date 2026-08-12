"""Search-only MCP application mounted into the production RAG API."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from ticket_clustering.hermes_agent.service import ArticleResearchService


_shared_engine: Any | None = None


def set_shared_engine(engine: Any | None) -> None:
    """Bind the MCP service to the API-owned, already initialized RAGEngine."""
    global _shared_engine
    _shared_engine = engine
    service._engine = engine


def _engine_factory() -> Any:
    if _shared_engine is None:
        raise RuntimeError("The API RAGEngine is not ready")
    return _shared_engine


mcp = FastMCP(
    name="RegLab Hermes Search",
    instructions=(
        "Search-only access to RegLab documentation and historical support tickets. "
        "Search results are compact; expand only selected result_ref values."
    ),
    stateless_http=True,
)
service = ArticleResearchService(engine_factory=_engine_factory)


@mcp.tool()
def search_documentation(query: str, limit: int = 5) -> dict:
    """Search official RegLab documentation and return compact candidate snippets."""
    return service.search_documentation(query, limit)


@mcp.tool()
def search_historical_tickets(query: str, limit: int = 7) -> dict:
    """Search historical support cases and return compact candidate snippets."""
    return service.search_tickets(query, limit)


@mcp.tool()
def search_dual_retriever(query: str, docs_limit: int = 5, tickets_limit: int = 7) -> dict:
    """Run one focused query against documentation and historical tickets."""
    return service.search_dual(query, docs_limit, tickets_limit)


@mcp.tool()
def read_search_results(result_refs: list[str], max_chars: int = 5000) -> dict:
    """Read full text only for selected result_ref values from earlier searches."""
    return service.read_search_results(result_refs, max_chars)


@mcp.tool()
def get_cluster_tickets(ticket_ids: list[str]) -> dict:
    """Read exact structured ticket symptoms and solutions to verify important cases."""
    return service.get_tickets_by_ids(ticket_ids)


http_app = mcp.streamable_http_app()
