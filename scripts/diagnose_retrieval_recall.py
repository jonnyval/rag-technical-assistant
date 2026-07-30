"""Diagnose where reference facts disappear in the Agentic RAG pipeline.

The script replays the exact search queries stored in an Agentic benchmark
trace and asks an independent judge whether every expected point is present at
five stages:

1. Qdrant vector candidates;
2. children selected by the reranker;
3. the final formatted context;
4. the Adaptive answer;
5. the Agentic answer.

Production request handling is not modified. Retrieval is replayed locally;
only the semantic recall judgement uses an LLM API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.context_formatting import (
    format_docs,
    format_ticket_docs,
    source_references,
)
from src.evidence_guard import filter_documents_by_requested_series
from src.logger import log
from src.module_detection import (
    build_module_enriched_query,
    detect_modules_in_query,
    merge_documents,
    rank_tickets_for_modules,
)


DEFAULT_INPUT = Path("data/agentic_vs_adaptive_questions.json")
DEFAULT_BENCHMARK = Path("benchmark_results/agentic_vs_adaptive_cache/results.json")
DEFAULT_OUTPUT = Path("benchmark_results/agentic_vs_adaptive_cache/retrieval_recall")


class PointRecall(BaseModel):
    point_index: int = Field(ge=1)
    vector_candidates: bool
    reranked_selection: bool
    final_context: bool
    adaptive_answer: bool
    agentic_answer: bool
    supporting_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    explanation: str = ""


class RecallJudgement(BaseModel):
    points: list[PointRecall]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-by-stage semantic retrieval recall audit.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--benchmark-results", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--judge-provider",
        choices=("local", "gemini", "groq"),
        default="gemini",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--local-threshold",
        type=float,
        default=0.25,
        help="CrossEncoder semantic recall threshold for --judge-provider local.",
    )
    parser.add_argument(
        "--local-prefilter-limit",
        type=int,
        default=12,
        help="Embedding top-N passed to CrossEncoder per point and stage.",
    )
    parser.add_argument("--candidate-chars", type=int, default=900)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    items = payload.get("questions", []) if isinstance(payload, dict) else payload
    return [dict(item) for item in items]


def document_key(document: Document) -> str:
    metadata = document.metadata or {}
    identity = "|".join(
        str(value or "")
        for value in (
            metadata.get("db_source"),
            metadata.get("doc_id"),
            metadata.get("ticket_id"),
            metadata.get("source_file"),
            metadata.get("page_title"),
            document.page_content,
        )
    )
    return hashlib.sha1(identity.encode("utf-8", errors="ignore")).hexdigest()[:16]


def deduplicate(documents: list[Document]) -> list[Document]:
    result: list[Document] = []
    seen: set[str] = set()
    for document in documents:
        key = document_key(document)
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


def serialize_documents(
    documents: list[Document],
    *,
    stage_prefix: str,
    max_chars: int,
) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    blocks: list[str] = []
    for index, document in enumerate(deduplicate(documents), start=1):
        metadata = document.metadata or {}
        item_id = f"{stage_prefix}{index:03d}"
        content = " ".join(str(document.page_content or "").split())
        if len(content) > max_chars:
            content = content[:max_chars].rsplit(" ", 1)[0] + " […]"
        title = (
            metadata.get("ticket_id")
            or metadata.get("page_title")
            or metadata.get("title")
            or metadata.get("source_file")
            or "unknown"
        )
        record = {
            "id": item_id,
            "source": str(metadata.get("db_source") or ""),
            "title": str(title),
            "source_file": str(metadata.get("source_file") or ""),
            "rerank_score": float(metadata.get("rerank_score", 0.0) or 0.0),
            "fusion_score": float(metadata.get("fusion_score", 0.0) or 0.0),
            "content": content,
        }
        records.append(record)
        blocks.append(
            f"[{item_id}] source={record['source']} title={record['title']} "
            f"file={record['source_file']}\n{content}"
        )
    return records, "\n\n".join(blocks)


def trace_queries(
    benchmark_item: dict[str, Any],
    fallback_query: str,
) -> tuple[list[str], list[str]]:
    trace = (
        benchmark_item.get("runs", {})
        .get("agentic", {})
        .get("agent", {})
        .get("trace", [])
    )
    docs: list[str] = []
    tickets: list[str] = []
    for step in trace:
        query = " ".join(str(step.get("query") or "").split()).strip()
        if not query:
            continue
        if step.get("tool") == "search_docs" and query not in docs:
            docs.append(query)
        elif step.get("tool") == "search_tickets" and query not in tickets:
            tickets.append(query)
    return docs or [fallback_query], tickets or [fallback_query]


def replay_retrieval(
    engine: Any,
    question: str,
    benchmark_item: dict[str, Any],
) -> dict[str, Any]:
    modules = detect_modules_in_query(question)
    retrieval_query = build_module_enriched_query(question, modules)
    doc_queries, ticket_queries = trace_queries(benchmark_item, retrieval_query)
    retriever = engine.retriever

    vector_candidates: list[Document] = []
    reranked_selection: list[Document] = []
    parent_docs: list[Document] = []
    parent_tickets: list[Document] = []

    docs_k = int(retriever.docs_retriever.search_kwargs.get("k", 24))
    for query in doc_queries:
        children = retriever._search_children(
            retriever.docs_retriever,
            query,
            docs_k,
            False,
            "docs",
        )
        vector_candidates.extend(children)
        ranked = retriever._rank_children(
            query,
            children,
            limit=len(children),
            use_reranker=True,
        )
        selected = ranked[: retriever.top_k_final]
        reranked_selection.extend(selected)
        parent_docs = merge_documents(
            parent_docs,
            retriever._fetch_parents(retriever.docs_retriever, selected, "docs"),
        )

    tickets_k = 60
    for query in ticket_queries:
        children = retriever._search_children(
            retriever.tickets_retriever,
            query,
            tickets_k,
            False,
            "tickets",
        )
        vector_candidates.extend(children)
        ranked = retriever._rank_children(
            query,
            children,
            limit=len(children),
            use_reranker=True,
        )
        selected = retriever._unique_parent_children(ranked, 8)
        reranked_selection.extend(selected)
        incoming = retriever._fetch_parents(
            retriever.tickets_retriever,
            selected,
            "tickets",
        )
        parent_tickets = engine._merge_ticket_documents(parent_tickets, incoming)

    parent_docs = filter_documents_by_requested_series(question, parent_docs)
    parent_tickets = filter_documents_by_requested_series(question, parent_tickets)
    parent_docs = engine._filter_wiki_docs_by_requested_product(question, modules, parent_docs)
    parent_tickets = engine._filter_wiki_docs_by_requested_product(
        question,
        modules,
        parent_tickets,
    )
    parent_tickets = rank_tickets_for_modules(
        parent_tickets,
        modules,
        query=question,
        limit=4,
    )

    doc_sources = source_references(parent_docs, prefix="D")
    ticket_sources = source_references(parent_tickets, ticket_only=True, prefix="T")
    docs_context = format_docs(
        parent_docs,
        sources=doc_sources,
        max_total_chars=4000,
        max_doc_chars=1800,
    )
    tickets_context = format_ticket_docs(
        parent_tickets,
        sources=ticket_sources,
        max_total_chars=3000,
        max_doc_chars=1400,
    )
    return {
        "queries": {"docs": doc_queries, "tickets": ticket_queries},
        "vector_candidates": deduplicate(vector_candidates),
        "reranked_selection": deduplicate(reranked_selection),
        "final_documents": {
            "docs": parent_docs,
            "tickets": parent_tickets,
        },
        "final_context": f"{docs_context}\n\n{tickets_context}".strip(),
    }


def build_judge(provider: str, model: str):
    from src.config import settings
    from src.engine import RoundRobinFallbackRunnable

    prompt = ChatPromptTemplate.from_template(
        """You audit semantic recall in a technical-support RAG pipeline.

For every numbered reference point, independently determine whether its core
technical fact or action is explicitly supported at each stage. Topical
similarity, matching product names, and vague troubleshooting do NOT count.
Equivalent technically precise wording does count.

Stages:
- vector_candidates: raw child chunks returned by Qdrant;
- reranked_selection: child chunks retained for parent expansion;
- final_context: exact text available to the answer model after filtering and truncation;
- adaptive_answer and agentic_answer: final generated answers.

Booleans are independent. Parent expansion can make final_context=true even if
a short vector child did not contain the fact. An answer can also state a fact
not present in final_context; mark the answer true but do not mark context true.
supporting_ids may contain only IDs visible in vector/reranked sections.
Return exactly one result for each reference point, preserving point_index.

Question:
{question}

Reference points:
{expected_points}

VECTOR CANDIDATES:
{vector_candidates}

RERANKED SELECTION:
{reranked_selection}

FINAL CONTEXT:
{final_context}

ADAPTIVE ANSWER:
{adaptive_answer}

AGENTIC ANSWER:
{agentic_answer}
"""
    )

    llms: list[Any]
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llms = [
            ChatGoogleGenerativeAI(
                model=model,
                google_api_key=key,
                temperature=0.0,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
            )
            for key in settings.google_api_keys
        ]
    else:
        from langchain_openai import ChatOpenAI

        llms = [
            ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=key,
                model=model,
                temperature=0.0,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
            )
            for key in settings.groq_api_keys
        ]
    structured = [llm.with_structured_output(RecallJudgement) for llm in llms]
    return prompt | RoundRobinFallbackRunnable(
        structured,
        label=f"Retrieval recall judge ({provider})",
    )


def split_text(text: str, *, chunk_chars: int = 1200, overlap: int = 150) -> list[str]:
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_chars)
        chunk = normalized[start:end]
        if end < len(normalized):
            chunk = chunk.rsplit(" ", 1)[0] or chunk
        chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def local_recall_judgement(
    reranker: Any,
    embeddings: Any,
    expected_points: list[str],
    vector_records: list[dict[str, Any]],
    reranked_records: list[dict[str, Any]],
    final_context: str,
    adaptive_answer: str,
    agentic_answer: str,
    *,
    threshold: float,
    prefilter_limit: int,
) -> dict[str, Any]:
    import numpy as np

    stages = {
        "vector_candidates": [(record["id"], record["content"]) for record in vector_records],
        "reranked_selection": [(record["id"], record["content"]) for record in reranked_records],
        "final_context": [
            (f"C{index:03d}", chunk)
            for index, chunk in enumerate(split_text(final_context), start=1)
        ],
        "adaptive_answer": [
            (f"A{index:03d}", chunk)
            for index, chunk in enumerate(split_text(adaptive_answer), start=1)
        ],
        "agentic_answer": [
            (f"G{index:03d}", chunk)
            for index, chunk in enumerate(split_text(agentic_answer), start=1)
        ],
    }
    stage_vectors = {
        stage: (
            np.asarray(
                embeddings.embed_documents([text for _, text in entries]),
                dtype=np.float32,
            )
            if entries
            else None
        )
        for stage, entries in stages.items()
    }
    points: list[dict[str, Any]] = []
    for point_index, point in enumerate(expected_points, start=1):
        point_vector = np.asarray(embeddings.embed_query(point), dtype=np.float32)
        stage_hits: dict[str, bool] = {}
        stage_scores: dict[str, float] = {}
        supporting_ids: list[str] = []
        for stage, entries in stages.items():
            if not entries:
                stage_hits[stage] = False
                stage_scores[stage] = 0.0
                continue
            similarities = stage_vectors[stage] @ point_vector
            selected_indices = np.argsort(similarities)[::-1][
                : max(1, min(prefilter_limit, len(entries)))
            ]
            selected_entries = [entries[int(index)] for index in selected_indices]
            scores = reranker.predict(
                [[point, text] for _, text in selected_entries],
                show_progress_bar=False,
            )
            scored = [
                (entry_id, float(score))
                for (entry_id, _), score in zip(selected_entries, scores)
            ]
            best_id, best_score = max(scored, key=lambda item: item[1])
            stage_scores[stage] = round(best_score, 4)
            stage_hits[stage] = best_score >= threshold
            if stage in {"vector_candidates", "reranked_selection"} and best_score >= threshold:
                supporting_ids.append(best_id)
        points.append(
            {
                "point_index": point_index,
                **stage_hits,
                "supporting_ids": supporting_ids,
                "confidence": "medium",
                "explanation": (
                    "Local CrossEncoder semantic-relevance proxy; "
                    f"threshold={threshold:.3f}; embedding prefilter={prefilter_limit}."
                ),
                "stage_scores": stage_scores,
            }
        )
    return {
        "points": points,
        "method": "embedding_prefilter_cross_encoder",
        "threshold": threshold,
        "prefilter_limit": prefilter_limit,
    }

def build_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    stages = (
        "vector_candidates",
        "reranked_selection",
        "final_context",
        "adaptive_answer",
        "agentic_answer",
    )
    stage_hits = Counter()
    losses = Counter()
    total_points = 0
    questions_complete = 0
    zero_candidate_questions = 0
    full_candidate_questions = 0
    for item in items:
        judgement = item.get("judgement") or {}
        points = judgement.get("points") or []
        if not points:
            continue
        questions_complete += 1
        question_vector_hits = 0
        for point in points:
            total_points += 1
            reranked = bool(point.get("reranked_selection"))
            # Reranked children are a physical subset of vector candidates.
            # This causal invariant corrects occasional local prefilter misses.
            vector = bool(point.get("vector_candidates")) or reranked
            context = bool(point.get("final_context"))
            adaptive = bool(point.get("adaptive_answer"))
            agentic = bool(point.get("agentic_answer"))
            question_vector_hits += int(vector)
            effective = {
                "vector_candidates": vector,
                "reranked_selection": reranked,
                "final_context": context,
                "adaptive_answer": adaptive,
                "agentic_answer": agentic,
            }
            for stage in stages:
                stage_hits[stage] += int(effective[stage])
            if not vector:
                losses["absent_from_vector_candidates"] += 1
            if vector and not reranked:
                losses["candidate_fact_not_in_selected_child"] += 1
            if vector and not context:
                losses["candidate_fact_absent_from_final_context"] += 1
            if not reranked and context:
                losses["recovered_by_parent_expansion"] += 1
            if reranked and not context:
                losses["lost_after_rerank_before_context"] += 1
            if context and not adaptive:
                losses["adaptive_generation_loss"] += 1
            if context and not agentic:
                losses["agentic_generation_loss"] += 1
            if not context and adaptive:
                losses["adaptive_answer_without_replay_context"] += 1
            if not context and agentic:
                losses["agentic_answer_without_replay_context"] += 1
        if question_vector_hits == 0:
            zero_candidate_questions += 1
        if question_vector_hits == len(points):
            full_candidate_questions += 1

    return {
        "questions_evaluated": questions_complete,
        "reference_points": total_points,
        "question_coverage": {
            "zero_candidate_recall": zero_candidate_questions,
            "full_candidate_recall": full_candidate_questions,
        },
        "stage_recall": {
            stage: {
                "hits": stage_hits[stage],
                "total": total_points,
                "recall": round(stage_hits[stage] / total_points, 3) if total_points else 0.0,
            }
            for stage in stages
        },
        "losses": dict(losses),
    }

def main() -> None:
    args = parse_args()
    all_questions = load_questions(Path(args.input))
    questions = all_questions
    if args.limit is not None:
        questions = all_questions[: max(0, args.limit)]
    benchmark_payload = json.loads(
        Path(args.benchmark_results).read_text(encoding="utf-8")
    )
    benchmark_by_id = {
        item["id"]: item for item in benchmark_payload.get("questions", [])
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    stored = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else {"metadata": {}, "questions": []}
    )
    results_by_id = {item["id"]: item for item in stored.get("questions", [])}
    if args.summarize_only:
        ordered = [
            results_by_id[item["id"]]
            for item in all_questions
            if item["id"] in results_by_id
        ]
        atomic_write_json(output_dir / "summary.json", build_summary(ordered))
        print(f"Retrieval recall summary updated: {output_dir.resolve()}")
        return

    from src.config import settings
    from src.engine import RAGEngine

    provider_configs = (
        settings._raw_config.get("providers", {}).get("llm", {}).get("configs", {})
    )
    judge_model = (
        "BAAI/bge-reranker-v2-m3"
        if args.judge_provider == "local"
        else args.judge_model
        or provider_configs.get(args.judge_provider, {}).get("model", "")
    )
    if args.judge_provider != "local" and not judge_model:
        raise ValueError(f"No judge model configured for {args.judge_provider}")

    log.info("Initialize retrieval engine...")
    engine = RAGEngine("deep")
    judge = (
        None
        if args.judge_provider == "local"
        else build_judge(args.judge_provider, judge_model)
    )

    for index, question in enumerate(questions, start=1):
        question_id = str(question["id"])
        existing = results_by_id.get(question_id, {})
        if existing.get("judgement") and not existing.get("error") and not args.force:
            log.info("Skip completed recall audit %s", question_id)
            continue
        benchmark_item = benchmark_by_id.get(question_id)
        if not benchmark_item:
            log.warning("No benchmark result for %s", question_id)
            continue

        log.info("Recall audit %s/%s id=%s", index, len(questions), question_id)
        started = time.perf_counter()
        try:
            replay = replay_retrieval(engine, question["question"], benchmark_item)
            vector_records, vector_text = serialize_documents(
                replay["vector_candidates"],
                stage_prefix="V",
                max_chars=max(200, args.candidate_chars),
            )
            reranked_records, reranked_text = serialize_documents(
                replay["reranked_selection"],
                stage_prefix="R",
                max_chars=max(200, args.candidate_chars),
            )
            runs = benchmark_item.get("runs", {})
            expected_points = list(question.get("expected_points") or [])
            adaptive_answer = runs.get("adaptive", {}).get("answer", "") or ""
            agentic_answer = runs.get("agentic", {}).get("answer", "") or ""
            if args.judge_provider == "local":
                judgement_payload = local_recall_judgement(
                    engine.rerank_model,
                    engine.dense_embeddings,
                    expected_points,
                    vector_records,
                    reranked_records,
                    replay["final_context"],
                    adaptive_answer,
                    agentic_answer,
                    threshold=args.local_threshold,
                    prefilter_limit=args.local_prefilter_limit,
                )
            else:
                judgement = judge.invoke(
                    {
                        "question": question["question"],
                        "expected_points": "\n".join(
                            f"{point_index}. {point}"
                            for point_index, point in enumerate(expected_points, start=1)
                        ),
                        "vector_candidates": vector_text or "(empty)",
                        "reranked_selection": reranked_text or "(empty)",
                        "final_context": replay["final_context"] or "(empty)",
                        "adaptive_answer": adaptive_answer or "(empty)",
                        "agentic_answer": agentic_answer or "(empty)",
                    }
                )
                judgement_payload = judgement.model_dump()
            if len(judgement_payload["points"]) != len(expected_points):
                raise ValueError(
                    f"Judge returned {len(judgement_payload['points'])} points, "
                    f"expected {len(expected_points)}"
                )
            result = {
                **question,
                "queries": replay["queries"],
                "counts": {
                    "vector_candidates": len(vector_records),
                    "reranked_selection": len(reranked_records),
                    "final_docs": len(replay["final_documents"]["docs"]),
                    "final_tickets": len(replay["final_documents"]["tickets"]),
                },
                "vector_candidates": vector_records,
                "reranked_selection": reranked_records,
                "final_context": replay["final_context"],
                "judgement": judgement_payload,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": "",
            }
        except Exception as error:
            log.error("Recall audit failed for %s: %s", question_id, error, exc_info=True)
            result = {
                **question,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        results_by_id[question_id] = result
        ordered = [
            results_by_id[item["id"]]
            for item in all_questions
            if item["id"] in results_by_id
        ]
        stored["metadata"] = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input": str(args.input),
            "benchmark_results": str(args.benchmark_results),
            "judge_provider": args.judge_provider,
            "judge_model": judge_model,
            "candidate_chars": args.candidate_chars,
            "local_threshold": args.local_threshold,
            "local_prefilter_limit": args.local_prefilter_limit,
        }
        stored["questions"] = ordered
        atomic_write_json(results_path, stored)
        atomic_write_json(output_dir / "summary.json", build_summary(ordered))
        if args.sleep > 0 and index < len(questions):
            time.sleep(args.sleep)

    print(f"Retrieval recall audit finished: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
