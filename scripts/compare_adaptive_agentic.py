"""Compare Adaptive and Agentic RAG over the same support questions.

The runner:
- loads both engines once and shares embedding/reranker models;
- alternates execution order to reduce order bias;
- records exact OpenAI-compatible token usage, latency, sources and agent trace;
- saves after every question and resumes an existing output directory;
- runs a blind pairwise LLM judge against reference points;
- creates a blind A/B CSV for optional manual quality review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from src.logger import log


DEFAULT_INPUT = Path("data/agentic_vs_adaptive_questions.json")
DEFAULT_SEED = "reglab-agentic-v1"
REVIEW_SCORE_COLUMNS = (
    "correctness_A_0_2",
    "completeness_A_0_2",
    "groundedness_A_0_2",
    "usefulness_A_0_2",
    "correctness_B_0_2",
    "completeness_B_0_2",
    "groundedness_B_0_2",
    "usefulness_B_0_2",
    "preferred_A_B_tie",
    "review_comment",
)


class AutomaticAnswerScore(BaseModel):
    correctness: int = Field(ge=0, le=2)
    completeness: int = Field(ge=0, le=2)
    groundedness: int = Field(ge=0, le=2)
    usefulness: int = Field(ge=0, le=2)
    critical_errors: list[str] = Field(default_factory=list)
    comment: str = ""


class AutomaticPairwiseJudgement(BaseModel):
    answer_a: AutomaticAnswerScore
    answer_b: AutomaticAnswerScore
    preferred: Literal["A", "B", "tie"]
    confidence: Literal["high", "medium", "low"]
    rationale: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blind quality comparison of reglab-ai-adaptive and experimental Agentic RAG."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="JSON or JSONL questions file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Result directory. Reuse the same path to resume; default creates a timestamped directory.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N questions.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pause between questions.")
    parser.add_argument(
        "--rag-max-completion-tokens",
        type=int,
        default=2048,
        help="Groq completion budget for benchmark RAG; production defaults are unchanged.",
    )
    parser.add_argument(
        "--rag-reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
        help="Groq reasoning effort for benchmark RAG.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run both modes even if results exist.")
    parser.add_argument(
        "--no-auto-judge",
        action="store_true",
        help="Do not run the blind LLM judge after both answers are ready.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        choices=("gemini", "groq", "ollama"),
        help="Judge provider; default uses applications.evaluation.judge_llm (currently Gemini).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model; default uses the selected provider model from config.yaml.",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED, help="Stable seed for blind A/B mapping.")
    parser.add_argument(
        "--init-template",
        action="store_true",
        help="Create an example questions file at --input and exit.",
    )
    parser.add_argument(
        "--summarize-review",
        default=None,
        metavar="OUTPUT_DIR",
        help="Summarize a filled blind review.csv in OUTPUT_DIR and exit.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def create_template(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing questions file: {path}")
    payload = [
        {
            "id": "q001",
            "category": "diagnostics",
            "question": "Замените этот текст первым тестовым вопросом.",
            "expected_points": [
                "Необязательно: факты или действия, которые хороший ответ должен содержать."
            ],
            "notes": "Необязательно: контекст для проверяющего, который нельзя передавать RAG.",
        }
    ]
    atomic_write_json(path, payload)
    print(f"Template created: {path.resolve()}")


def read_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {path}. Run with --init-template to create one."
        )
    if path.suffix.lower() == ".jsonl":
        raw_items = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        raw_items = payload.get("questions", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError("Questions file must contain a JSON array or {'questions': [...]}.")

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items, start=1):
        item = {"question": raw_item} if isinstance(raw_item, str) else dict(raw_item)
        question = str(item.get("question") or item.get("Вопрос") or "").strip()
        if not question:
            raise ValueError(f"Question #{index} is empty.")
        question_id = str(item.get("id") or f"q{index:03d}").strip()
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question id: {question_id}")
        seen_ids.add(question_id)
        expected = item.get("expected_points") or item.get("expected") or []
        if isinstance(expected, str):
            expected = [expected]
        questions.append(
            {
                "id": question_id,
                "category": str(item.get("category") or "").strip(),
                "question": question,
                "expected_points": [str(value).strip() for value in expected if str(value).strip()],
                "notes": str(item.get("notes") or "").strip(),
            }
        )
    return questions


def serialize_models(values: list[Any]) -> list[dict[str, Any]]:
    result = []
    for value in values or []:
        if hasattr(value, "model_dump"):
            result.append(value.model_dump())
        elif isinstance(value, dict):
            result.append(value)
        else:
            result.append({"value": str(value)})
    return result


def serialize_response(response: Any) -> dict[str, Any]:
    from src.context_formatting import format_chat_sources_footer

    doc_sources = list(getattr(response, "doc_sources", []) or [])
    ticket_sources = list(getattr(response, "ticket_sources", []) or [])
    draft = str(getattr(response, "draft_private_comment", "") or "")
    return {
        "answer": draft + format_chat_sources_footer(doc_sources, ticket_sources),
        "draft_private_comment": draft,
        "docs_answer": str(getattr(response, "docs_answer", "") or ""),
        "confidence": str(getattr(response, "confidence", "") or ""),
        "doc_sources": serialize_models(doc_sources),
        "ticket_sources": serialize_models(ticket_sources),
        "evidence_notes": serialize_models(list(getattr(response, "evidence_notes", []) or [])),
        "similar_tickets": serialize_models(list(getattr(response, "similar_tickets", []) or [])),
    }


def run_measured(mode: str, query: str, adaptive: RAGEngine, agentic: AgenticRAG) -> dict[str, Any]:
    from langchain_community.callbacks.manager import get_openai_callback

    started = time.perf_counter()
    callback = None
    try:
        with get_openai_callback() as callback:
            if mode == "adaptive":
                response = adaptive.process_support_ticket(query)
                result = serialize_response(response)
            else:
                agent_result = agentic.run(query)
                result = serialize_response(agent_result.answer)
                result["agent"] = {
                    "stop_reason": agent_result.stop_reason,
                    "rounds_used": agent_result.rounds_used,
                    "tool_calls_used": agent_result.tool_calls_used,
                    "diagnostics": agent_result.diagnostics,
                    "trace": [step.model_dump() for step in agent_result.trace],
                }
        result["error"] = ""
    except Exception as error:
        log.error("%s failed for query: %s", mode, error, exc_info=True)
        result = {"error": f"{type(error).__name__}: {error}", "answer": ""}

    result["metrics"] = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prompt_tokens": int(getattr(callback, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(callback, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(callback, "total_tokens", 0) or 0),
        "llm_requests": int(getattr(callback, "successful_requests", 0) or 0),
    }
    return result


def build_automatic_judge(provider: str, model: str):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    from src.config import settings
    from src.engine import RoundRobinFallbackRunnable

    prompt = ChatPromptTemplate.from_template(
        """You are a strict, impartial evaluator of two technical-support RAG answers.
The answers are anonymized as A and B. Never infer or favor their implementation.

Evaluate each answer independently against the question and reference points.
The reference points are derived from the resolved support case and are authoritative,
but equivalent technically correct wording is acceptable.

Score every criterion from 0 to 2:
- correctness: technical claims and proposed actions are correct;
- completeness: all important reference points needed for the question are covered;
- groundedness: uncertainty is explicit and unsupported claims are avoided;
- usefulness: a support engineer can use the answer with little or no rewriting.

Do not reward verbosity, formatting, number of citations, or mentioning a ticket by itself.
Penalize dangerous actions, invented root causes, wrong product/series, and confident claims
that are absent from the reference points. A concise answer may receive the maximum score.
Choose preferred=A/B only for a meaningful quality advantage; otherwise choose tie.
Use confidence=low when the reference points are ambiguous or both answers are hard to compare.

Question:
{question}

Reference points:
{expected_points}

Answer A:
{answer_a}

Answer B:
{answer_b}
"""
    )

    llms = []
    if provider == "groq":
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
    elif provider == "ollama":
        llms = [
            ChatOpenAI(
                base_url=settings.ollama_url,
                api_key="ollama",
                model=model,
                temperature=0.0,
            )
        ]
    elif provider == "gemini":
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
        raise RuntimeError(
            f"Automatic benchmark judge is not configured for provider: {provider}"
        )
    if not llms:
        raise RuntimeError("No LLM credentials available for automatic benchmark judge")

    chains = []
    for llm in llms:
        if provider in ("groq", "ollama"):
            structured = llm.with_structured_output(
                AutomaticPairwiseJudgement,
                method="json_mode",
            )
        else:
            structured = llm.with_structured_output(AutomaticPairwiseJudgement)
        chains.append(prompt | structured)
    return RoundRobinFallbackRunnable(chains, label="Benchmark Judge LLM")


def run_automatic_judge(
    judge_chain: Any,
    question: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    mapping: dict[str, str],
) -> dict[str, Any]:
    from langchain_community.callbacks.manager import get_openai_callback

    started = time.perf_counter()
    callback = None
    try:
        with get_openai_callback() as callback:
            judgement = judge_chain.invoke(
                {
                    "question": question["question"],
                    "expected_points": "\n".join(
                        f"- {point}" for point in question.get("expected_points", [])
                    ) or "No explicit reference points were supplied.",
                    "answer_a": runs[mapping["A"]].get("answer", ""),
                    "answer_b": runs[mapping["B"]].get("answer", ""),
                }
            )
        result = judgement.model_dump()
        result["error"] = ""
    except Exception as error:
        log.error("Automatic benchmark judge failed: %s", error, exc_info=True)
        result = {"error": f"{type(error).__name__}: {error}"}
    result["metrics"] = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prompt_tokens": int(getattr(callback, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(callback, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(callback, "total_tokens", 0) or 0),
        "llm_requests": int(getattr(callback, "successful_requests", 0) or 0),
    }
    return result


def blind_mapping(question_id: str, seed: str) -> dict[str, str]:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).digest()
    if digest[0] % 2:
        return {"A": "agentic", "B": "adaptive"}
    return {"A": "adaptive", "B": "agentic"}


def load_existing_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            row["id"]: row
            for row in csv.DictReader(file, delimiter=";")
            if row.get("id")
        }


def write_blind_review(
    path: Path,
    questions: list[dict[str, Any]],
    results_by_id: dict[str, dict[str, Any]],
    mappings: dict[str, dict[str, str]],
) -> None:
    old_rows = load_existing_review(path)
    fieldnames = [
        "id",
        "category",
        "question",
        "expected_points",
        "answer_A",
        "answer_B",
        *REVIEW_SCORE_COLUMNS,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        for question in questions:
            result = results_by_id.get(question["id"], {})
            mapping = mappings[question["id"]]
            row = {
                "id": question["id"],
                "category": question["category"],
                "question": question["question"],
                "expected_points": "\n".join(question["expected_points"]),
                "answer_A": result.get("runs", {}).get(mapping["A"], {}).get("answer", ""),
                "answer_B": result.get("runs", {}).get(mapping["B"], {}).get("answer", ""),
            }
            previous = old_rows.get(question["id"], {})
            for column in REVIEW_SCORE_COLUMNS:
                row[column] = previous.get(column, "")
            writer.writerow(row)


def metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"total": 0, "mean": 0, "median": 0}
    return {
        "total": round(sum(values), 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
    }


def build_summary(results_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in ("adaptive", "agentic"):
        runs = [
            item.get("runs", {}).get(mode, {})
            for item in results_by_id.values()
            if mode in item.get("runs", {})
        ]
        successful = [run for run in runs if not run.get("error")]
        summary[mode] = {
            "runs": len(runs),
            "successful": len(successful),
            "errors": len(runs) - len(successful),
            "prompt_tokens": metric_summary(
                [run.get("metrics", {}).get("prompt_tokens", 0) for run in successful]
            ),
            "completion_tokens": metric_summary(
                [run.get("metrics", {}).get("completion_tokens", 0) for run in successful]
            ),
            "total_tokens": metric_summary(
                [run.get("metrics", {}).get("total_tokens", 0) for run in successful]
            ),
            "elapsed_seconds": metric_summary(
                [run.get("metrics", {}).get("elapsed_seconds", 0) for run in successful]
            ),
            "llm_requests": metric_summary(
                [run.get("metrics", {}).get("llm_requests", 0) for run in successful]
            ),
        }
    return summary


def build_automatic_quality_summary(
    results_by_id: dict[str, dict[str, Any]],
    mappings: dict[str, dict[str, str]],
) -> dict[str, Any]:
    criteria = ("correctness", "completeness", "groundedness", "usefulness")
    scores = {
        mode: {criterion: [] for criterion in criteria}
        for mode in ("adaptive", "agentic")
    }
    preferences = {"adaptive": 0, "agentic": 0, "tie": 0}
    confidence = {"high": 0, "medium": 0, "low": 0}
    errors = []
    judged = 0
    judge_tokens = 0

    for question_id, item in results_by_id.items():
        judgement = item.get("automatic_judgement", {})
        if not judgement:
            continue
        if judgement.get("error"):
            errors.append({"id": question_id, "error": judgement["error"]})
            continue
        mapping = mappings.get(question_id)
        if not mapping:
            errors.append({"id": question_id, "error": "missing blind mapping"})
            continue
        judged += 1
        judge_tokens += int(judgement.get("metrics", {}).get("total_tokens", 0) or 0)
        confidence[str(judgement.get("confidence", "low"))] += 1
        for side_key, side in (("answer_a", "A"), ("answer_b", "B")):
            mode = mapping[side]
            side_score = judgement.get(side_key, {})
            for criterion in criteria:
                scores[mode][criterion].append(float(side_score.get(criterion, 0)))
        preferred = str(judgement.get("preferred", "tie"))
        if preferred in ("A", "B"):
            preferences[mapping[preferred]] += 1
        else:
            preferences["tie"] += 1

    return {
        "judged_questions": judged,
        "preferences": preferences,
        "judge_confidence": confidence,
        "judge_total_tokens": judge_tokens,
        "scores": {
            mode: {
                criterion: {
                    "mean": round(statistics.mean(values), 3) if values else 0,
                    "total": round(sum(values), 3),
                    "count": len(values),
                }
                for criterion, values in mode_scores.items()
            }
            for mode, mode_scores in scores.items()
        },
        "errors": errors,
    }


def summarize_review(output_dir: Path) -> dict[str, Any]:
    review_path = output_dir / "review.csv"
    key_path = output_dir / "review_key.json"
    if not review_path.exists() or not key_path.exists():
        raise FileNotFoundError(f"review.csv or review_key.json not found in {output_dir}")

    mappings = json.loads(key_path.read_text(encoding="utf-8"))
    with review_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter=";"))

    criteria = ("correctness", "completeness", "groundedness", "usefulness")
    scores = {
        mode: {criterion: [] for criterion in criteria}
        for mode in ("adaptive", "agentic")
    }
    preferences = {"adaptive": 0, "agentic": 0, "tie": 0, "unscored": 0}
    fully_scored = 0
    invalid: list[str] = []

    for row in rows:
        question_id = row.get("id", "")
        mapping = mappings.get(question_id)
        if not mapping:
            invalid.append(f"{question_id}: missing blind mapping")
            continue
        row_complete = True
        pending: list[tuple[str, str, float]] = []
        for side in ("A", "B"):
            mode = mapping[side]
            for criterion in criteria:
                raw_value = str(row.get(f"{criterion}_{side}_0_2", "")).strip()
                try:
                    value = float(raw_value)
                except ValueError:
                    row_complete = False
                    continue
                if value < 0 or value > 2:
                    invalid.append(f"{question_id}: {criterion}_{side} must be between 0 and 2")
                    row_complete = False
                    continue
                pending.append((mode, criterion, value))
        if row_complete and len(pending) == 8:
            fully_scored += 1
            for mode, criterion, value in pending:
                scores[mode][criterion].append(value)

            preferred = str(row.get("preferred_A_B_tie", "")).strip().lower()
            if preferred in ("a", "b"):
                preferences[mapping[preferred.upper()]] += 1
            elif preferred == "tie":
                preferences["tie"] += 1
            else:
                preferences["unscored"] += 1

    summary = {
        "review_rows": len(rows),
        "fully_scored_rows": fully_scored,
        "preferences": preferences,
        "scores": {
            mode: {
                criterion: {
                    "mean": round(statistics.mean(values), 3) if values else 0,
                    "total": round(sum(values), 3),
                    "count": len(values),
                }
                for criterion, values in mode_scores.items()
            }
            for mode, mode_scores in scores.items()
        },
        "invalid": invalid,
    }
    atomic_write_json(output_dir / "quality_summary.json", summary)
    return summary


def write_readme(
    path: Path,
    input_path: Path,
    judge_provider: str,
    judge_model: str,
) -> None:
    from src.config import settings

    path.write_text(
        fr"""# Adaptive vs Agentic RAG

Input: `{input_path}`
RAG LLM: `{settings.active_llm}/{settings.llm_model_name}`
Judge LLM: `{judge_provider}/{judge_model}`
Embeddings: `{settings.embedding_model_name}`
Reranker: `{settings.reranker_model_name}`

## Automatic blind evaluation

By default, an LLM judge evaluates anonymized answers A/B against `expected_points`.
The automatic result is saved to `automatic_quality_summary.json`; per-question
scores, confidence and rationale are stored in `results.json`.
Use `--no-auto-judge` only when automatic evaluation is not required.

## Optional manual blind review

Open `review.csv` without opening `review_key.json`. Score A and B independently:

- correctness: 0 = incorrect, 1 = partly correct, 2 = correct;
- completeness: 0 = misses the task, 1 = useful but incomplete, 2 = sufficiently complete;
- groundedness: 0 = unsupported claims, 1 = mixed, 2 = claims are supported and uncertainty is explicit;
- usefulness: 0 = unusable, 1 = needs substantial editing, 2 = usable by a support engineer.

Set `preferred_A_B_tie` to `A`, `B` or `tie`. Add free-form notes to `review_comment`.
Only after the review is complete, use `review_key.json` to reveal the modes.

`results.json` contains full non-blind outputs, sources, token usage and Agentic RAG trace.
`summary.json` contains operational totals and averages.
`automatic_quality_summary.json` contains automatic quality scores and winners.

After filling `review.csv`, calculate quality totals:

`& C:\Users\e.valov\AppData\Local\anaconda3\envs\rag_langchain\python.exe scripts\compare_adaptive_agentic.py --summarize-review "{path.parent}"`
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.summarize_review:
        summary = summarize_review(Path(args.summarize_review))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    input_path = Path(args.input)
    if args.init_template:
        create_template(input_path)
        return

    from src.agentic import AgenticRAG
    from src.config import settings
    from src.engine import RAGEngine

    judge_provider = str(args.judge_provider or settings.judge_llm).strip().lower()
    provider_configs = (
        settings._raw_config.get("providers", {}).get("llm", {}).get("configs", {})
    )
    judge_model = str(
        args.judge_model
        or provider_configs.get(judge_provider, {}).get("model", "")
    ).strip()
    if not args.no_auto_judge and not judge_model:
        raise ValueError(f"No model configured for judge provider: {judge_provider}")

    questions = read_questions(input_path)
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]
    if not questions:
        raise ValueError("No questions selected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("benchmark_results") / f"agentic_vs_adaptive_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    review_path = output_dir / "review.csv"
    key_path = output_dir / "review_key.json"

    stored = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else {"metadata": {}, "questions": []}
    )
    results_by_id = {item["id"]: item for item in stored.get("questions", [])}
    mappings = {question["id"]: blind_mapping(question["id"], args.seed) for question in questions}

    stored["metadata"] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "provider": settings.active_llm,
        "model": settings.llm_model_name,
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
        "docs_db": settings.active_db_name,
        "tickets_db": settings.second_db_name,
        "blind_seed": args.seed,
        "automatic_judge": not args.no_auto_judge,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "rag_max_completion_tokens": args.rag_max_completion_tokens,
        "rag_reasoning_effort": args.rag_reasoning_effort,
    }
    write_readme(
        output_dir / "README.md",
        input_path,
        judge_provider,
        judge_model,
    )
    atomic_write_json(key_path, mappings)

    log.info("Initialize Adaptive engine and shared models...")
    adaptive = RAGEngine(
        "adaptive",
        llm_max_completion_tokens=args.rag_max_completion_tokens,
        llm_reasoning_effort=args.rag_reasoning_effort,
    )
    deep = RAGEngine(
        "deep",
        shared_embeddings=adaptive.dense_embeddings,
        shared_reranker=adaptive.rerank_model,
        llm_max_completion_tokens=args.rag_max_completion_tokens,
        llm_reasoning_effort=args.rag_reasoning_effort,
    )
    agentic = AgenticRAG(shared_engine=deep)
    judge_chain = (
        None
        if args.no_auto_judge
        else build_automatic_judge(judge_provider, judge_model)
    )

    for index, question in enumerate(questions, start=1):
        item = results_by_id.setdefault(question["id"], {**question, "runs": {}})
        item.update(question)
        item.setdefault("runs", {})
        order = ("adaptive", "agentic") if index % 2 else ("agentic", "adaptive")
        log.info("Question %s/%s id=%s order=%s", index, len(questions), question["id"], order)
        for mode in order:
            if (
                mode in item["runs"]
                and not item["runs"][mode].get("error")
                and not args.force
            ):
                log.info("Skip completed %s/%s", question["id"], mode)
                continue
            item["runs"][mode] = run_measured(mode, question["question"], adaptive, agentic)
            stored["questions"] = [results_by_id[q["id"]] for q in questions if q["id"] in results_by_id]
            atomic_write_json(results_path, stored)
            atomic_write_json(output_dir / "summary.json", build_summary(results_by_id))
            write_blind_review(review_path, questions, results_by_id, mappings)

        successful_runs = all(
            mode in item["runs"] and not item["runs"][mode].get("error")
            for mode in ("adaptive", "agentic")
        )
        if judge_chain is not None and successful_runs:
            if (
                args.force
                or not item.get("automatic_judgement")
                or item.get("automatic_judgement", {}).get("error")
            ):
                item["automatic_judgement"] = run_automatic_judge(
                    judge_chain,
                    question,
                    item["runs"],
                    mappings[question["id"]],
                )
                stored["questions"] = [
                    results_by_id[q["id"]] for q in questions if q["id"] in results_by_id
                ]
                atomic_write_json(results_path, stored)
                atomic_write_json(
                    output_dir / "automatic_quality_summary.json",
                    build_automatic_quality_summary(results_by_id, mappings),
                )

        if args.sleep > 0 and index < len(questions):
            time.sleep(args.sleep)

    stored["questions"] = [results_by_id[question["id"]] for question in questions]
    atomic_write_json(results_path, stored)
    atomic_write_json(output_dir / "summary.json", build_summary(results_by_id))
    atomic_write_json(
        output_dir / "automatic_quality_summary.json",
        build_automatic_quality_summary(results_by_id, mappings),
    )
    write_blind_review(review_path, questions, results_by_id, mappings)
    print(f"Comparison finished: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
