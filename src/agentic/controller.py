"""Bounded stateless agent loop over the existing RegLab retriever."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from src.agentic.schemas import AgentTraceStep, AgenticRAGResult
from src.agentic.tools import AgenticRetrievalTools
from src.config import settings
from src.context_formatting import (
    format_docs,
    format_source_references,
    format_ticket_docs,
    source_references,
)
from src.engine import (
    WIKI_REFLECTION_DOCS_MAX_CHARS,
    WIKI_REFLECTION_TICKETS_MAX_CHARS,
)
from src.evidence_guard import (
    apply_definition_guard,
    apply_diagnostic_scope_guard,
    apply_entity_coverage_guard,
    apply_response_provenance,
    apply_transformation_evidence_guard,
    check_and_format_equipment_mismatch_warning,
    filter_documents_by_requested_series,
    strict_evidence_context,
)
from src.logger import log
from src.module_detection import (
    build_module_enriched_query,
    detect_modules_in_query,
    ensure_module_block,
    format_detected_modules,
    merge_documents,
    rank_tickets_for_modules,
)


class AgenticController:
    def __init__(
        self,
        engine: Any,
        *,
        max_rounds: int = 3,
        max_tool_calls: int = 6,
        max_queries_per_round: int = 2,
        timeout_seconds: float = 90.0,
    ):
        self.engine = engine
        self.tools = AgenticRetrievalTools(engine)
        self.max_rounds = max(1, min(int(max_rounds), 5))
        self.max_tool_calls = max(2, min(int(max_tool_calls), 12))
        self.max_queries_per_round = max(1, min(int(max_queries_per_round), 3))
        self.timeout_seconds = max(15.0, float(timeout_seconds))
        config = settings.agentic_rag_config
        self.docs_context_max_chars = max(1500, int(config.get("docs_context_max_chars", 4000)))
        self.tickets_context_max_chars = max(1200, int(config.get("tickets_context_max_chars", 3000)))
        self.doc_max_chars = max(800, int(config.get("doc_max_chars", 1800)))
        self.ticket_max_chars = max(800, int(config.get("ticket_max_chars", 1400)))

    def _filter_sources(self, query: str, modules: List[Dict[str, Any]], documents: List[Any]):
        documents = filter_documents_by_requested_series(query, documents)
        return self.engine._filter_wiki_docs_by_requested_product(query, modules, documents)

    def _is_diagnostic_query(self, query: str) -> bool:
        if self.engine._is_incident_query(query):
            return True
        normalized = (query or "").lower().replace("ё", "е")
        markers = (
            "сбрасыва", "теряется связь", "потеря связи", "нестабил",
            "неисправ", "самодиагност", "self-diagnostic", "что делать",
            "как исправ",
        )
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _build_exact_identifier_query(query: str, exact_anchors: List[str], requested_series: List[str]) -> str:
        stopwords = {
            "который", "которая", "которые", "какой", "какая", "после", "этого",
            "из-за", "чего", "может", "быть", "связан", "связано", "подскажите",
            "пожалуйста", "сделать",
        }
        reserved = {item.lower() for item in [*exact_anchors, *requested_series]}
        salient: List[str] = []
        for token in re.findall(r"[\w-]+", query.lower(), flags=re.UNICODE):
            if len(token) < 4 or token in stopwords or token in reserved or token in salient:
                continue
            salient.append(token)
        return " ".join([*exact_anchors, *requested_series, *salient[:10]]).strip()

    @staticmethod
    def _rank_agent_tickets(query: str, tickets: List[Any], exact_anchors: List[str], limit: int = 4):
        query_tokens = {token for token in re.findall(r"\w+", query.lower(), flags=re.UNICODE) if len(token) >= 5}
        def score(document: Any):
            content = str(getattr(document, "page_content", "") or "")
            metadata = getattr(document, "metadata", {}) or {}
            return (
                sum(1 for anchor in exact_anchors if anchor in content.upper()),
                sum(1 for token in query_tokens if token in content.lower()),
                float(metadata.get("rerank_score", 0.0) or 0.0),
            )
        return sorted(tickets, key=score, reverse=True)[:limit]

    def _format_context(self, documents: List[Any], tickets: List[Any]):
        doc_sources = source_references(documents, prefix="D")
        ticket_sources = source_references(tickets, ticket_only=True, prefix="T")
        docs_context = format_docs(
            documents,
            sources=doc_sources,
            max_total_chars=self.docs_context_max_chars,
            max_doc_chars=self.doc_max_chars,
        )
        tickets_context = format_ticket_docs(
            tickets,
            sources=ticket_sources,
            max_total_chars=self.tickets_context_max_chars,
            max_doc_chars=self.ticket_max_chars,
        )
        return doc_sources, ticket_sources, docs_context, tickets_context

    def run(self, query: str) -> AgenticRAGResult:
        query = str(query or "").strip()
        if not query:
            raise ValueError("query must not be empty")

        started = time.perf_counter()
        deadline = started + self.timeout_seconds
        trace: List[AgentTraceStep] = []
        tool_calls = 0
        rounds_used = 0
        stop_reason = "round_limit"
        modules = detect_modules_in_query(query)
        module_context = format_detected_modules(modules)
        retrieval_query = build_module_enriched_query(query, modules)
        tried_docs = {retrieval_query.lower(), query.lower()}
        tried_tickets = {retrieval_query.lower(), query.lower()}
        requested_series_set = {item.upper() for item in re.findall(r"\bR\d{3}S?\b", query, flags=re.IGNORECASE)}
        exact_anchors = sorted(set(re.findall(r"\b[A-Z][A-Z0-9_]{3,}\b", query)) - requested_series_set)
        unresolved_anchors: List[str] = []

        docs, step = self.tools.search_docs(retrieval_query, round_index=0, use_reranker=True)
        trace.append(step)
        tool_calls += 1
        tickets: List[Any] = []
        tickets_enabled = self.engine.retriever.tickets_retriever is not None
        if tickets_enabled and tool_calls < self.max_tool_calls:
            tickets, step = self.tools.search_tickets(retrieval_query, round_index=0)
            trace.append(step)
            tool_calls += 1

        if exact_anchors and tool_calls < self.max_tool_calls:
            requested_series = sorted(set(re.findall(r"\bR\d{3}S?\b", query, flags=re.IGNORECASE)))
            exact_query = self._build_exact_identifier_query(query, exact_anchors, requested_series)
            incoming, step = self.tools.search_docs(exact_query, round_index=0, use_reranker=True)
            step.reason = "deterministic exact-identifier search"
            trace.append(step)
            tool_calls += 1
            docs = merge_documents(incoming, docs)
            if tickets_enabled and tool_calls < self.max_tool_calls:
                incoming, step = self.tools.search_tickets(exact_query, round_index=0)
                step.reason = "deterministic exact-identifier ticket search"
                trace.append(step)
                tool_calls += 1
                tickets = self.engine._merge_ticket_documents(incoming, tickets)

        for round_index in range(1, self.max_rounds + 1):
            rounds_used = round_index
            docs = self._filter_sources(query, modules, docs)
            tickets = self._filter_sources(query, modules, tickets)
            _, _, docs_context, tickets_context = self._format_context(docs, tickets)

            plan_started = time.perf_counter()
            reflection = self.engine.retrieval_reflection_chain.invoke({
                "input": query,
                "chat_history": "No previous chat history. This run is stateless.",
                "module_context": module_context or "No module was detected.",
                "already_tried_queries": "\n".join([
                    *(f"docs: {item}" for item in sorted(tried_docs)),
                    *(f"tickets: {item}" for item in sorted(tried_tickets)),
                ]),
                "docs_context": docs_context[:WIKI_REFLECTION_DOCS_MAX_CHARS],
                "tickets_context": tickets_context[:WIKI_REFLECTION_TICKETS_MAX_CHARS],
            })
            combined_context = f"{docs_context}\n{tickets_context}".upper()
            unresolved_anchors = [anchor for anchor in exact_anchors if anchor not in combined_context]
            if reflection.enough_context and unresolved_anchors:
                exact_query = " ".join([*unresolved_anchors, *sorted(set(re.findall(r"\bR\d{3}S?\b", query, flags=re.IGNORECASE)))]).strip()
                reflection.enough_context = False
                reflection.missing_facts = [f"Direct evidence for exact identifiers: {', '.join(unresolved_anchors)}", *reflection.missing_facts]
                if exact_query:
                    reflection.followup_doc_queries = [exact_query, *reflection.followup_doc_queries]
                    reflection.followup_ticket_queries = [exact_query, *reflection.followup_ticket_queries]
            trace.append(AgentTraceStep(
                round_index=round_index,
                phase="plan",
                reason=(
                    f"enough={reflection.enough_context}; "
                    f"missing={'; '.join(reflection.missing_facts[:3])}; "
                    f"risks={'; '.join(reflection.source_risks[:3])}"
                ),
                elapsed_seconds=round(time.perf_counter() - plan_started, 3),
            ))

            if reflection.enough_context:
                stop_reason = "enough_evidence"
                break
            if time.perf_counter() >= deadline:
                stop_reason = "timeout"
                break
            if tool_calls >= self.max_tool_calls:
                stop_reason = "tool_budget"
                break

            doc_queries = self.engine._normalize_followup_queries(
                reflection.followup_doc_queries, tried_docs, self.max_queries_per_round
            )
            ticket_queries = self.engine._normalize_followup_queries(
                reflection.followup_ticket_queries,
                tried_tickets,
                self.max_queries_per_round,
            )
            if not doc_queries and not ticket_queries:
                stop_reason = "no_new_actions"
                break

            before_count = len(docs) + len(tickets)
            for followup in doc_queries:
                if tool_calls >= self.max_tool_calls or time.perf_counter() >= deadline:
                    break
                focused = build_module_enriched_query(followup, modules)
                incoming, step = self.tools.search_docs(focused, round_index=round_index, use_reranker=True)
                step.reason = "documentation follow-up requested by evidence planner"
                trace.append(step)
                tool_calls += 1
                docs = merge_documents(docs, incoming)

            for followup in ticket_queries:
                if not tickets_enabled or tool_calls >= self.max_tool_calls or time.perf_counter() >= deadline:
                    break
                focused = build_module_enriched_query(followup, modules)
                incoming, step = self.tools.search_tickets(focused, round_index=round_index)
                step.reason = "ticket follow-up requested by evidence planner"
                trace.append(step)
                tool_calls += 1
                tickets = self.engine._merge_ticket_documents(tickets, incoming)

            if len(docs) + len(tickets) <= before_count:
                stop_reason = "no_new_evidence"
                break

        docs = self._filter_sources(query, modules, docs)
        tickets = self._filter_sources(query, modules, tickets)
        tickets = rank_tickets_for_modules(tickets, modules, query=query, limit=len(tickets) or 1)
        tickets = self._rank_agent_tickets(query, tickets, exact_anchors, limit=4)
        doc_sources, ticket_sources, docs_context, tickets_context = self._format_context(docs, tickets)
        agent_context = (
            f"Agentic retrieval stop reason: {stop_reason}; rounds={rounds_used}; tool calls={tool_calls}. "
            f"Exact identifiers still absent from evidence: {', '.join(unresolved_anchors) or 'none'}. "
            "Use only the supplied evidence. If an exact identifier is absent, explicitly say that direct evidence was not found and do not invent a procedure. "
            "A ticket solution is historical evidence only: never turn it into an imperative for the current incident; phrase it as 'In historical ticket [T...]'. "
            "Do not recommend Factory reset or another procedure merely because the user says that a value resets. "
            "A documented post-load step is not proof of the current root cause; present it as a documented check, not a confirmed fix. "
            "For an incident question, start by stating whether the root cause is directly confirmed. Then separate: documented facts, historical ticket observations, and conditional checks. "
            "Never phrase a hypothesis as a confirmed cause. General model knowledge may appear only in a clearly labelled 'General technical hypothesis' block without a source citation. "
            "Do not mention internal planning or tool calls in the user-facing answer."
        )
        final_started = time.perf_counter()
        response = self.engine.support_chain.invoke({
            "input": query,
            "strict_evidence_context": strict_evidence_context(query) + "\n\n" + agent_context,
            "equipment_mismatch_context": check_and_format_equipment_mismatch_warning(query, doc_sources),
            "module_context": module_context,
            "docs_context": docs_context,
            "tickets_context": tickets_context,
            "doc_sources_context": format_source_references(doc_sources),
            "ticket_sources_context": format_source_references(ticket_sources),
        })
        trace.append(AgentTraceStep(
            round_index=rounds_used,
            phase="final",
            reason="evidence-bounded synthesis",
            elapsed_seconds=round(time.perf_counter() - final_started, 3),
        ))
        response = apply_definition_guard(response, query, docs_context, doc_sources)
        response = apply_transformation_evidence_guard(response, query, docs_context, doc_sources)
        response = apply_entity_coverage_guard(response, query, docs_context)
        response = apply_diagnostic_scope_guard(response, query, docs_context)
        response, doc_sources, ticket_sources = apply_response_provenance(response, doc_sources, ticket_sources)
        if modules:
            response.docs_answer = ensure_module_block(response.docs_answer, modules)
            response.draft_private_comment = ensure_module_block(response.draft_private_comment, modules)
        if self._is_diagnostic_query(query) and stop_reason != "enough_evidence":
            limitation = (
                "Прямого подтверждения причины именно для описанного случая в найденных источниках нет. "
                "Ниже приведены документированные проверки и опыт похожих обращений, "
                "а не подтверждённое решение.\n\n"
            )
            if not response.draft_private_comment.startswith("Прямого подтверждения причины"):
                response.draft_private_comment = limitation + response.draft_private_comment.lstrip()
            response.confidence = "low"
        response.doc_sources = doc_sources
        response.ticket_sources = ticket_sources
        if not tickets:
            response.similar_tickets = []

        elapsed = time.perf_counter() - started
        log.info(
            "AgenticRAG finished in %.2fs rounds=%s calls=%s stop=%s docs=%s tickets=%s",
            elapsed, rounds_used, tool_calls, stop_reason, len(docs), len(tickets),
        )
        return AgenticRAGResult(
            answer=response,
            trace=trace,
            stop_reason=stop_reason,
            rounds_used=rounds_used,
            tool_calls_used=tool_calls,
            elapsed_seconds=round(elapsed, 3),
            diagnostics={"docs": len(docs), "tickets": len(tickets)},
        )
