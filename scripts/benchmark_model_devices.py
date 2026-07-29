"""Benchmark embedding and reranking latency for CPU/GPU device combinations.

Run each combination in a separate process so CUDA allocations from one run do
not affect the next one.  The benchmark deliberately excludes Qdrant and the
LLM: it measures only the two local inference stages used by RAG.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings


QUERIES = [
    "Как выполнить заводской сброс контроллера R500?",
    "Что означает сообщение self-diagnostic и где искать логи?",
    "Ошибка download denied при загрузке проекта в ПЛК",
    "Как проверить целостность файловой системы контроллера?",
    "Как преобразовать строку в набор ASCII-кодов?",
]

PASSAGES = [
    "Сервисный режим контроллера позволяет выполнить сброс до заводского состояния.",
    "Для проверки целостности файловой системы используйте команду firmware check on.",
    "При ошибке загрузки проекта проверьте совместимость IDE и версии прошивки.",
    "Журнал системных событий содержит диагностические сообщения контроллера.",
    "Библиотеки проекта подключаются в настройках среды разработки.",
] * 6  # 30 candidates: same order of magnitude as production retrieval pool.


def timed(callable_):
    started = time.perf_counter()
    callable_()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - started


def median_seconds(samples: list[float]) -> float:
    return round(statistics.median(samples), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--reranker-device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if "cuda" in (args.embedding_device, args.reranker_device) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    result: dict[str, object] = {
        "embedding_device": args.embedding_device,
        "reranker_device": args.reranker_device,
        "embedding_model": settings.embedding_model_name,
        "reranker_model": settings.reranker_model_name,
        "iterations": args.iterations,
    }

    try:
        result["embedding_load_s"] = round(timed(lambda: None), 4)
        started = time.perf_counter()
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": args.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["embedding_load_s"] = round(time.perf_counter() - started, 4)

        started = time.perf_counter()
        reranker = CrossEncoder(
            settings.reranker_model_name,
            device=args.reranker_device,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result["reranker_load_s"] = round(time.perf_counter() - started, 4)

        # Warm-up avoids measuring CUDA kernel compilation and allocator setup.
        embeddings.embed_query(QUERIES[0])
        reranker.predict([(QUERIES[0], PASSAGES[0])])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        embedding_times = []
        rerank_times = []
        total_times = []
        for index in range(args.iterations):
            query = QUERIES[index % len(QUERIES)]
            embedding_times.append(timed(lambda: embeddings.embed_query(query)))
            rerank_times.append(timed(lambda: reranker.predict([(query, passage) for passage in PASSAGES])))
            total_times.append(embedding_times[-1] + rerank_times[-1])

        result["embedding_query_median_s"] = median_seconds(embedding_times)
        result["rerank_30_candidates_median_s"] = median_seconds(rerank_times)
        result["local_rag_stages_median_s"] = median_seconds(total_times)
        if torch.cuda.is_available():
            result["cuda_memory_allocated_mib"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
            result["cuda_memory_reserved_mib"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
    except torch.cuda.OutOfMemoryError as error:
        result["status"] = "oom"
        result["error"] = str(error).splitlines()[0]
    except Exception as error:  # keep a failed device combination comparable in the report
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}".splitlines()[0]
    else:
        result["status"] = "ok"
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
