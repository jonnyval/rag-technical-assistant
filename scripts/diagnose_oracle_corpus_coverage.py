"""Estimate whether benchmark facts are retrievable from the indexed corpus.

Each expected benchmark point is used as an ideal ("oracle") search query
against documentation and ticket collections. Results separate facts already
found by the original query, facts lost because of query formulation, and
facts likely absent from the current corpus/index.

This is a local semantic proxy, not proof of corpus absence. It does not call
an external LLM and does not modify production retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from scripts.diagnose_retrieval_recall import (
    atomic_write_json,
    deduplicate,
    document_key,
    load_questions,
    split_text,
)
from src.logger import log


DEFAULT_INPUT = Path("data/agentic_vs_adaptive_questions.json")
DEFAULT_RECALL_RESULTS = Path(
    "benchmark_results/agentic_vs_adaptive_cache/retrieval_recall/results.json"
)
DEFAULT_OUTPUT = DEFAULT_RECALL_RESULTS.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle@k corpus/index coverage audit for expected benchmark facts."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--recall-results", default=str(DEFAULT_RECALL_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--ambiguous-threshold", type=float, default=0.25)
    parser.add_argument(
        "--prefilter-method",
        choices=("lexical", "embedding"),
        default="lexical",
    )
    parser.add_argument("--prefilter-limit", type=int, default=12)
    parser.add_argument("--parent-chunk-limit", type=int, default=6)
    parser.add_argument("--match-limit", type=int, default=5)
    parser.add_argument("--candidate-chars", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def load_recall_by_id(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["id"]): dict(item)
        for item in payload.get("questions", [])
        if item.get("id")
    }


def original_vector_hits(
    recall_item: dict[str, Any] | None,
    point_count: int,
) -> list[bool]:
    """Return causally corrected original-query vector hits."""
    points = ((recall_item or {}).get("judgement") or {}).get("points") or []
    by_index = {
        int(point.get("point_index", index)): point
        for index, point in enumerate(points, start=1)
    }
    hits: list[bool] = []
    for point_index in range(1, point_count + 1):
        point = by_index.get(point_index, {})
        hits.append(
            bool(point.get("vector_candidates"))
            or bool(point.get("reranked_selection"))
        )
    return hits


def metadata_title(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("ticket_id")
        or metadata.get("page_title")
        or metadata.get("title")
        or metadata.get("source_file")
        or "unknown"
    )


def make_snippet(
    document: Document,
    text: str,
    *,
    kind: str,
    part_index: int,
) -> dict[str, Any]:
    metadata = dict(document.metadata or {})
    return {
        "key": f"{document_key(document)}:{kind}:{part_index}",
        "source": str(metadata.get("db_source") or ""),
        "kind": kind,
        "retrieval_rank": part_index if kind == "child" else 0,
        "title": metadata_title(metadata),
        "ticket_id": str(metadata.get("ticket_id") or ""),
        "source_file": str(metadata.get("source_file") or ""),
        "content": " ".join(str(text or "").split()),
    }


def select_parent_chunks(text: str, point: str, limit: int) -> list[tuple[int, str]]:
    """Cheap lexical oracle prefilter before expensive embedding/reranking."""
    chunks = split_text(text, chunk_chars=1400, overlap=180)
    if len(chunks) <= limit:
        return list(enumerate(chunks, start=1))
    point_tokens = set(re.findall(r"(?u)\b[\w.-]{4,}\b", point.lower()))
    scored: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_tokens = set(re.findall(r"(?u)\b[\w.-]{4,}\b", chunk.lower()))
        scored.append((len(point_tokens & chunk_tokens), index, chunk))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [(index, chunk) for _, index, chunk in scored[: max(1, limit)]]


def build_oracle_snippets(
    engine: Any,
    point: str,
    top_k: int,
    parent_chunk_limit: int,
) -> list[dict[str, Any]]:
    """Search both collections and expand returned children to their parents."""
    sources = (
        ("docs", engine.retriever.docs_retriever),
        ("tickets", engine.retriever.tickets_retriever),
    )
    snippets: list[dict[str, Any]] = []
    for db_label, retriever in sources:
        children = engine.retriever._search_children(
            retriever,
            point,
            top_k,
            False,
            db_label,
        )
        children = deduplicate(children)
        for child in children:
            child.metadata.setdefault("rerank_score", 0.0)
        for child_index, child in enumerate(children, start=1):
            snippets.append(
                make_snippet(
                    child,
                    child.page_content,
                    kind="child",
                    part_index=child_index,
                )
            )

        parents = engine.retriever._fetch_parents(retriever, children, db_label)
        for parent_index, parent in enumerate(deduplicate(parents), start=1):
            chunks = select_parent_chunks(
                parent.page_content,
                point,
                parent_chunk_limit,
            )
            for chunk_index, chunk in chunks:
                snippets.append(
                    make_snippet(
                        parent,
                        chunk,
                        kind="parent",
                        part_index=parent_index * 1000 + chunk_index,
                    )
                )
    return snippets


def score_snippets(
    engine: Any,
    point: str,
    snippets: list[dict[str, Any]],
    *,
    embedding_cache: dict[str, np.ndarray],
    prefilter_method: str,
    prefilter_limit: int,
    match_limit: int,
    candidate_chars: int,
) -> list[dict[str, Any]]:
    """Prefilter snippets, then score them with the local CrossEncoder."""
    if not snippets:
        return []
    texts = [item["content"] for item in snippets]
    if prefilter_method == "embedding":
        missing_texts = list(
            dict.fromkeys(text for text in texts if text not in embedding_cache)
        )
        if missing_texts:
            missing_vectors = engine.dense_embeddings.embed_documents(missing_texts)
            embedding_cache.update(
                {
                    text: np.asarray(vector, dtype=np.float32)
                    for text, vector in zip(missing_texts, missing_vectors)
                }
            )
        vectors = np.stack([embedding_cache[text] for text in texts])
        query_vector = np.asarray(
            engine.dense_embeddings.embed_query(point),
            dtype=np.float32,
        )
        prefilter_scores = vectors @ query_vector
    else:
        point_tokens = set(re.findall(r"(?u)\b[\w.-]{4,}\b", point.lower()))
        prefilter_scores = np.asarray(
            [
                len(
                    point_tokens
                    & set(re.findall(r"(?u)\b[\w.-]{4,}\b", text.lower()))
                )
                for text in texts
            ],
            dtype=np.float32,
        )
    selected_indices = list(
        np.argsort(prefilter_scores)[::-1][
            : max(1, min(prefilter_limit, len(snippets)))
        ]
    )
    if prefilter_method == "lexical":
        selected_indices.extend(
            index
            for index, item in enumerate(snippets)
            if item["kind"] == "child" and item["retrieval_rank"] <= 6
        )
    selected_indices = list(dict.fromkeys(int(index) for index in selected_indices))
    selected = [snippets[index] for index in selected_indices]
    scores = engine.rerank_model.predict(
        [[point, item["content"]] for item in selected],
        show_progress_bar=False,
    )
    ranked: list[dict[str, Any]] = []
    for item, score in zip(selected, scores):
        record = {key: value for key, value in item.items() if key != "key"}
        record["score"] = round(float(score), 4)
        content = record["content"]
        if len(content) > candidate_chars:
            content = content[:candidate_chars].rsplit(" ", 1)[0] + " […]"
        record["content"] = content
        ranked.append(record)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: max(1, match_limit)]


def classify_point(
    original_hit: bool,
    oracle_hit: bool,
    ambiguous_match: bool = False,
) -> str:
    if original_hit:
        return "retrieved_by_agent_query"
    if oracle_hit:
        return "query_formulation_gap"
    if ambiguous_match:
        return "ambiguous_semantic_match"
    return "likely_corpus_or_index_gap"


def build_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    classifications: Counter[str] = Counter()
    best_sources: Counter[str] = Counter()
    total_points = 0
    oracle_hits = 0
    zero_oracle_questions = 0
    full_oracle_questions = 0
    for item in items:
        points = item.get("points") or []
        if not points:
            continue
        question_hits = 0
        for point in points:
            total_points += 1
            hit = bool(point.get("oracle_hit"))
            oracle_hits += int(hit)
            question_hits += int(hit)
            classifications[str(point.get("classification") or "unknown")] += 1
            best_match = point.get("best_match") or {}
            if hit:
                best_sources[str(best_match.get("source") or "unknown")] += 1
        if question_hits == 0:
            zero_oracle_questions += 1
        if question_hits == len(points):
            full_oracle_questions += 1
    return {
        "questions_evaluated": sum(bool(item.get("points")) for item in items),
        "reference_points": total_points,
        "oracle_hits": oracle_hits,
        "oracle_recall": round(oracle_hits / total_points, 3) if total_points else 0.0,
        "classification": dict(classifications),
        "best_hit_source": dict(best_sources),
        "question_coverage": {
            "zero_oracle_recall": zero_oracle_questions,
            "full_oracle_recall": full_oracle_questions,
        },
        "interpretation": {
            "retrieved_by_agent_query": (
                "The original Agentic query already retrieved evidence for the fact."
            ),
            "query_formulation_gap": (
                "Oracle@k found evidence, but the original Agentic query did not."
            ),
            "likely_corpus_or_index_gap": (
                "Oracle@k did not find evidence; this is not proof of corpus absence."
            ),

            "ambiguous_semantic_match": (
                "A related passage was found, but its score is below the conservative "
                "oracle-hit threshold and requires review."
            ),        },
    }


def ordered_results(
    questions: list[dict[str, Any]],
    results_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        results_by_id[str(question["id"])]
        for question in questions
        if str(question["id"]) in results_by_id
    ]


def main() -> None:
    args = parse_args()
    all_questions = load_questions(Path(args.input))
    questions = all_questions
    if args.limit is not None:
        questions = all_questions[: max(0, args.limit)]
    recall_by_id = load_recall_by_id(Path(args.recall_results))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "oracle_results.json"
    stored = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else {"metadata": {}, "questions": []}
    )
    results_by_id = {
        str(item["id"]): item
        for item in stored.get("questions", [])
        if item.get("id")
    }
    if args.summarize_only:
        ordered = ordered_results(all_questions, results_by_id)
        atomic_write_json(output_dir / "oracle_summary.json", build_summary(ordered))
        print(f"Oracle coverage summary updated: {output_dir.resolve()}")
        return

    from src.engine import RAGEngine

    log.info("Initialize retrieval engine for oracle@%s audit...", args.top_k)
    engine = RAGEngine("deep")
    if engine.rerank_model is None:
        raise RuntimeError("The deep profile must provide a local reranker.")
    embedding_cache: dict[str, np.ndarray] = {}

    for question_index, question in enumerate(questions, start=1):
        question_id = str(question["id"])
        existing = results_by_id.get(question_id, {})
        if existing.get("points") and not existing.get("error") and not args.force:
            log.info("Skip completed oracle audit %s", question_id)
            continue
        expected_points = list(question.get("expected_points") or [])
        original_hits = original_vector_hits(
            recall_by_id.get(question_id),
            len(expected_points),
        )
        log.info(
            "Oracle audit %s/%s id=%s points=%s",
            question_index,
            len(questions),
            question_id,
            len(expected_points),
        )
        started = time.perf_counter()
        point_results: list[dict[str, Any]] = []
        error = ""
        try:
            for point_index, (point, original_hit) in enumerate(
                zip(expected_points, original_hits),
                start=1,
            ):
                log.info(
                    "Oracle point %s/%s for %s",
                    point_index,
                    len(expected_points),
                    question_id,
                )
                snippets = build_oracle_snippets(
                    engine,
                    point,
                    max(1, args.top_k),
                    max(1, args.parent_chunk_limit),
                )
                matches = score_snippets(
                    engine,
                    point,
                    snippets,
                    embedding_cache=embedding_cache,
                    prefilter_method=args.prefilter_method,
                    prefilter_limit=max(1, args.prefilter_limit),
                    match_limit=max(1, args.match_limit),
                    candidate_chars=max(200, args.candidate_chars),
                )
                best_match = matches[0] if matches else {}
                best_score = float(best_match.get("score", 0.0))
                oracle_hit = best_score >= args.threshold
                ambiguous_match = (
                    not oracle_hit and best_score >= args.ambiguous_threshold
                )
                point_results.append(
                    {
                        "point_index": point_index,
                        "expected_point": point,
                        "original_vector_hit": original_hit,
                        "oracle_hit": oracle_hit,
                        "oracle_score": round(best_score, 4),
                        "ambiguous_match": ambiguous_match,
                        "classification": classify_point(
                            original_hit,
                            oracle_hit,
                            ambiguous_match,
                        ),
                        "best_match": best_match,
                        "top_matches": matches,
                        "candidate_snippets": len(snippets),
                    }
                )
        except Exception as exc:
            error = str(exc)
            log.error("Oracle audit failed for %s: %s", question_id, exc, exc_info=True)

        results_by_id[question_id] = {
            "id": question_id,
            "category": question.get("category", ""),
            "question": question.get("question", ""),
            "points": point_results,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "error": error,
        }
        ordered = ordered_results(all_questions, results_by_id)
        payload = {
            "metadata": {
                "method": "expected_point_oracle_search_hybrid_prefilter_cross_encoder",
                "top_k_per_collection": args.top_k,
                "threshold": args.threshold,
                "ambiguous_threshold": args.ambiguous_threshold,
                "prefilter_method": args.prefilter_method,
                "prefilter_limit": args.prefilter_limit,
                "parent_chunk_limit": args.parent_chunk_limit,
                "match_limit": args.match_limit,
                "scope_note": (
                    "Oracle@k is a semantic proxy over retrieved child chunks and "
                    "their parent documents; misses do not prove corpus absence."
                ),
            },
            "questions": ordered,
        }
        atomic_write_json(results_path, payload)
        atomic_write_json(output_dir / "oracle_summary.json", build_summary(ordered))

    ordered = ordered_results(all_questions, results_by_id)
    atomic_write_json(output_dir / "oracle_summary.json", build_summary(ordered))
    print(f"Oracle coverage audit complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
