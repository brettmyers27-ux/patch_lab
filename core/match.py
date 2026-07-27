"""Synth-agnostic brute-force cosine retrieval over normalized numpy matrices."""

from __future__ import annotations

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def cosine_topk(
    queries: np.ndarray, matrix: np.ndarray, k: int = 10, *, normalized: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return descending cosine scores and row indices without FAISS."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not normalized:
        queries = l2_normalize(queries)
        matrix = l2_normalize(matrix)
    else:
        queries = np.asarray(queries, dtype=np.float32)
        matrix = np.asarray(matrix, dtype=np.float32)
    k = min(k, matrix.shape[0])
    scores = queries @ matrix.T
    candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    return (
        np.take_along_axis(candidate_scores, order, axis=1),
        np.take_along_axis(candidates, order, axis=1),
    )
