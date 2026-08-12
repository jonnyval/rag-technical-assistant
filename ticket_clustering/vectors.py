from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ticket_clustering.data import SymptomNode


def input_fingerprint(nodes: list[SymptomNode], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for node in nodes:
        digest.update(b"\0")
        digest.update(node.node_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(node.normalized_text.encode("utf-8"))
    return digest.hexdigest()


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("wb") as output:
            np.savez(output, **arrays)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_or_encode(
    nodes: list[SymptomNode],
    model_name: str,
    device: str,
    batch_size: int,
    local_files_only: bool,
    cache_dir: Path,
) -> tuple[np.ndarray, str, Path]:
    fingerprint = input_fingerprint(nodes, model_name)
    cache_path = cache_dir / f"embeddings_{fingerprint[:20]}.npz"
    node_ids = np.asarray([node.node_id for node in nodes])
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_fingerprint = str(cached["fingerprint"].item())
            cached_ids = cached["node_ids"]
            embeddings = cached["embeddings"].astype(np.float32, copy=False)
        if cached_fingerprint == fingerprint and np.array_equal(cached_ids, node_ids):
            print(f"Using embedding cache: {cache_path}", flush=True)
            return embeddings, fingerprint, cache_path
        raise RuntimeError(f"Invalid embedding cache: {cache_path}")

    print(f"Loading embedding model: {model_name} ({device})", flush=True)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
    print(f"Encoding {len(nodes)} unique symptoms...", flush=True)
    embeddings = model.encode(
        [node.display_text for node in nodes],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    atomic_save_npz(
        cache_path,
        fingerprint=np.asarray(fingerprint),
        node_ids=node_ids,
        embeddings=embeddings,
    )
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return embeddings, fingerprint, cache_path


def knn_fingerprint(embedding_fingerprint: str, max_neighbors: int) -> str:
    payload = json.dumps(
        {"embedding_fingerprint": embedding_fingerprint, "max_neighbors": max_neighbors},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_compute_exact_knn(
    embeddings: np.ndarray,
    embedding_fingerprint: str,
    max_neighbors: int,
    block_size: int,
    device: str,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, Path]:
    count = len(embeddings)
    effective_neighbors = max(0, min(max_neighbors, count - 1))
    fingerprint = knn_fingerprint(embedding_fingerprint, effective_neighbors)
    cache_path = cache_dir / f"knn_{fingerprint[:20]}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) != fingerprint:
                raise RuntimeError(f"Invalid kNN cache: {cache_path}")
            indices = cached["indices"].astype(np.int32, copy=False)
            similarities = cached["similarities"].astype(np.float32, copy=False)
        print(f"Using exact kNN cache: {cache_path}", flush=True)
        return indices, similarities, cache_path

    if effective_neighbors == 0:
        indices = np.empty((count, 0), dtype=np.int32)
        similarities = np.empty((count, 0), dtype=np.float32)
        atomic_save_npz(
            cache_path,
            fingerprint=np.asarray(fingerprint),
            indices=indices,
            similarities=similarities,
        )
        return indices, similarities, cache_path

    import torch

    actual_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
    print(
        f"Computing exact cosine kNN: nodes={count}, k={effective_neighbors}, block={block_size}, device={actual_device}",
        flush=True,
    )
    all_vectors = torch.from_numpy(np.ascontiguousarray(embeddings)).to(actual_device)
    indices = np.empty((count, effective_neighbors), dtype=np.int32)
    similarities = np.empty((count, effective_neighbors), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, count, block_size):
            end = min(count, start + block_size)
            scores = all_vectors[start:end] @ all_vectors.T
            local_rows = torch.arange(end - start, device=actual_device)
            global_rows = torch.arange(start, end, device=actual_device)
            scores[local_rows, global_rows] = -float("inf")
            values, positions = torch.topk(scores, k=effective_neighbors, dim=1, largest=True, sorted=True)
            indices[start:end] = positions.cpu().numpy().astype(np.int32, copy=False)
            similarities[start:end] = values.cpu().numpy().astype(np.float32, copy=False)
            if start == 0 or end == count or (start // block_size) % 10 == 0:
                print(f"kNN progress: {end}/{count}", flush=True)
            del scores, values, positions
    del all_vectors
    if actual_device == "cuda":
        torch.cuda.empty_cache()

    atomic_save_npz(
        cache_path,
        fingerprint=np.asarray(fingerprint),
        indices=indices,
        similarities=similarities,
    )
    return indices, similarities, cache_path
