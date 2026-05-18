"""Batch autotest runner for RAGEngine.

The script reads JSON files with test questions, sends unanswered questions to
RAGEngine, and writes enriched JSON files with model answers and audit fields.
It is designed for long runs: results are saved after every question and an
existing output file is resumed by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.config import settings
from src.engine import RAGEngine
from src.logger import log


QUESTION_KEY = "Вопрос"
OPTIONS_KEY = "Варианты ответа"
RAG_ANSWER_KEY = "Ответ RAG"
CORRECTNESS_KEY = "Правильность"
AUDIT_KEY = "SGR_Audit"
ERROR_PREFIX = "ОШИБКА:"


def parse_args() -> argparse.Namespace:
    """Parses CLI options for a resumable autotest run."""
    parser = argparse.ArgumentParser(description="Run RAGEngine autotests over JSON question files.")
    parser.add_argument("--input-dir", default="data/50_questions", help="Directory with input JSON files.")
    parser.add_argument("--output-dir", default=None, help="Directory for result JSON files.")
    parser.add_argument("--limit", type=int, default=None, help="Max questions per file for this run.")
    parser.add_argument("--sleep", type=float, default=None, help="Delay between LLM calls in seconds.")
    parser.add_argument("--force", action="store_true", help="Reprocess questions that already have answers.")
    parser.add_argument("--file", default=None, help="Process only one JSON filename from input-dir.")
    return parser.parse_args()


def provider_default_sleep() -> float:
    """Returns a conservative provider-specific pause between requests."""
    if settings.active_llm == "gemini":
        return 5.0
    if settings.active_llm == "ollama":
        return 0.0
    return 10.0


def read_json(path: Path) -> List[Dict[str, Any]]:
    """Reads a JSON array from disk and validates its top-level shape."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def atomic_write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """Writes JSON through a temporary file and atomically replaces the target."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    tmp_path.replace(path)


def generate_test_readme(output_dir: Path) -> None:
    """Creates a short README with the active autotest configuration."""
    readme_path = output_dir / "README.md"
    content = f"""# Отчет об автотестировании RAGEngine

Дата запуска: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Модели
- LLM provider: `{settings.active_llm}`
- LLM model: `{settings.llm_model_name}`
- Embeddings: `{settings.embedding_model_name}`
- Reranker: `{settings.reranker_model_name}`

## База и retrieval
- Docs DB: `{settings.active_db_name}`
- Tickets DB: `{settings.second_db_name}`
- Collection: `{settings.collection_name}`
- top_k_retrieval: `{settings.top_k_retrieval}`
- top_k_final: `{settings.top_k_final}`
- rerank_threshold: `{settings.rerank_threshold}`
- use_reranker: `{settings.use_reranker}`
- use_litm: `{settings.use_litm}`
- use_hyde: `{settings.use_hyde}`
- child_chunk_size: `{settings.child_chunk_size}`
- child_chunk_overlap: `{settings.child_chunk_overlap}`
"""
    readme_path.write_text(content, encoding="utf-8")
    log.info("Autotest README saved: %s", readme_path)


def build_prompt(item: Dict[str, Any]) -> str:
    """Builds a stable prompt from question and answer options."""
    question = str(item.get(QUESTION_KEY, "")).strip()
    options = str(item.get(OPTIONS_KEY, "")).strip()
    if options:
        return f"{question}\n\nВарианты ответа:\n{options}"
    return question


def has_answer(item: Dict[str, Any]) -> bool:
    """Checks whether a question already has a non-empty successful RAG answer."""
    answer = str(item.get(RAG_ANSWER_KEY, "")).strip()
    return bool(answer) and not answer.startswith(ERROR_PREFIX)


def response_to_item(response: Any) -> Dict[str, Any]:
    """Converts a quiz/general RAG response into fields stored inside the test item."""
    return {
        RAG_ANSWER_KEY: getattr(response, "final_answer", "") or "",
        CORRECTNESS_KEY: "",
        AUDIT_KEY: {
            "intent": getattr(response, "user_intent", ""),
            "missing_context": getattr(response, "missing_context", ""),
            "extracted_facts": [
                fact.model_dump() if hasattr(fact, "model_dump") else dict(fact)
                for fact in (getattr(response, "extracted_facts", []) or [])
            ],
            "relevant_images": list(getattr(response, "relevant_images", []) or []),
        },
    }


def process_file(
    engine: RAGEngine,
    input_file: Path,
    output_file: Path,
    *,
    limit: int | None,
    force: bool,
    sleep_seconds: float,
) -> Dict[str, int]:
    """Processes one JSON file and returns counters for the summary."""
    if output_file.exists():
        data = read_json(output_file)
        log.info("Resume existing result file: %s", output_file.name)
    else:
        data = read_json(input_file)
        log.info("Start new result file: %s", input_file.name)

    if limit is not None:
        data = data[:limit]

    counters = {"processed": 0, "skipped": 0, "errors": 0}

    for index, item in enumerate(data, start=1):
        if has_answer(item) and not force:
            counters["skipped"] += 1
            log.info("Skip answered question %s/%s", index, len(data))
            continue

        prompt = build_prompt(item)
        if not prompt:
            counters["errors"] += 1
            item[RAG_ANSWER_KEY] = f"{ERROR_PREFIX} empty question"
            atomic_write_json(output_file, data)
            continue

        log.info("Process question %s/%s (%s chars)", index, len(data), len(prompt))
        try:
            response = engine.process_quiz_question(prompt)
            item.update(response_to_item(response))
            counters["processed"] += 1
            log.info("Question %s processed successfully", index)
        except Exception as exc:
            counters["errors"] += 1
            log.error("Question %s failed: %s", index, exc, exc_info=True)
            item[RAG_ANSWER_KEY] = f"{ERROR_PREFIX} {exc}"
            item[CORRECTNESS_KEY] = ""

        atomic_write_json(output_file, data)

        if sleep_seconds > 0 and index < len(data):
            log.info("Sleep %.1f seconds before next request", sleep_seconds)
            time.sleep(sleep_seconds)

    return counters


def main() -> None:
    """Runs autotests across all selected JSON files."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("result_test_auto") / settings.active_db_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        log.error("Input directory not found: %s", input_dir)
        return

    sleep_seconds = args.sleep
    if sleep_seconds is None:
        sleep_seconds = float(os.getenv("AUTOTEST_SLEEP_SECONDS", provider_default_sleep()))

    limit = args.limit if args.limit is not None else settings.max_questions_per_file
    generate_test_readme(output_dir)

    log.info("Initialize RAGEngine for autotests...")
    engine = RAGEngine()

    json_files = sorted(input_dir.glob("*.json"))
    if args.file:
        json_files = [path for path in json_files if path.name == args.file]
        if not json_files:
            log.error("Requested file not found in %s: %s", input_dir, args.file)
            return
    log.info("Found input files: %s", len(json_files))
    for path in json_files:
        log.info("Selected autotest file: %s", path.name)

    total = {"processed": 0, "skipped": 0, "errors": 0}
    for file_index, input_file in enumerate(json_files, start=1):
        log.info("Start file %s/%s: %s", file_index, len(json_files), input_file.name)
        output_file = output_dir / f"{settings.active_db_name}_{input_file.name}"
        counters = process_file(
            engine,
            input_file,
            output_file,
            limit=limit,
            force=args.force,
            sleep_seconds=sleep_seconds,
        )
        for key, value in counters.items():
            total[key] += value
        log.info("Finished file %s/%s: %s -> %s", file_index, len(json_files), input_file.name, counters)

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(total, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(total, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Autotest finished: %s", total)
    log.info("Summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
