"""Run the experimental stateless Agentic RAG pipeline from PowerShell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic import AgenticRAG
from src.config import settings
from src.context_formatting import format_chat_sources_footer


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Experimental bounded Agentic RAG")
    parser.add_argument("query", help="One independent support-engineer question")
    parser.add_argument("--no-trace", action="store_true", help="Do not print the internal retrieval trace")
    args = parser.parse_args()

    if not settings.agentic_rag_enabled:
        raise RuntimeError("agentic_rag.enabled is false in config.yaml")

    result = AgenticRAG().run(args.query)
    print(
        result.answer.draft_private_comment
        + format_chat_sources_footer(result.answer.doc_sources, result.answer.ticket_sources)
    )
    if not args.no_trace and settings.agentic_rag_config.get("expose_trace", True):
        print("\n--- AGENT TRACE ---")
        print(json.dumps(
            {
                "stop_reason": result.stop_reason,
                "rounds_used": result.rounds_used,
                "tool_calls_used": result.tool_calls_used,
                "elapsed_seconds": result.elapsed_seconds,
                "steps": [step.model_dump() for step in result.trace],
            },
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
