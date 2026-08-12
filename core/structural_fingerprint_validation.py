"""Integrity checks for controlled structural-fingerprint matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class SelfRetrievalResult:
    queried: int
    passed: int
    failed: int
    failures: tuple[tuple[str, str, float], ...]


@dataclass(frozen=True, slots=True)
class DistinctnessResult:
    threshold: float
    pair_count: int
    cluster_count: int
    clustered_members: int
    largest_cluster: int
    singleton_count: int


def distinctness_components(
    features: np.ndarray, *, threshold: float, block_size: int = 256
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic non-singleton cosine-distance components."""

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("features must be a matrix")
    if threshold < 0 or block_size <= 0:
        raise ValueError("threshold must be non-negative and block_size positive")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    count = len(matrix)
    parent = np.arange(count, dtype=np.int64)
    sizes = np.ones(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    for start in range(0, count, block_size):
        similarities = matrix[start : start + block_size] @ matrix.T
        for local_index, row in enumerate(similarities):
            index = start + local_index
            neighbors = (
                np.flatnonzero((1.0 - row[index + 1 :]) <= threshold)
                + index
                + 1
            )
            for neighbor in neighbors:
                union(index, int(neighbor))

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return tuple(
        tuple(members)
        for _root, members in sorted(groups.items())
        if len(members) > 1
    )


def deterministic_sample_indices(count: int, requested: int = 20) -> np.ndarray:
    """Return stable, evenly-spaced indices without duplicating short sets."""

    if count < 0 or requested <= 0:
        raise ValueError("count must be non-negative and requested must be positive")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(
        np.linspace(0, count - 1, min(count, requested), dtype=np.int64)
    )


def self_retrieval(
    features: np.ndarray,
    stable_ids: Sequence[str],
    indices: Sequence[int] | None = None,
) -> SelfRetrievalResult:
    """Query rows against their own index with production-compatible tie breaks."""

    matrix = np.asarray(features, dtype=np.float32)
    identifiers = np.asarray([str(value) for value in stable_ids])
    if matrix.ndim != 2 or len(matrix) != len(identifiers):
        raise ValueError("features and stable_ids must be aligned")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    selected = (
        np.arange(len(matrix), dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    failures: list[tuple[str, str, float]] = []
    for index in selected:
        query = matrix[int(index)].copy()
        query /= max(float(np.linalg.norm(query)), 1e-8)
        scores = matrix @ query
        # ControlledFingerprintIndex orders by descending similarity and then
        # stable ID, so ties between collapsed descriptors are deterministic.
        order = np.lexsort((identifiers, -scores))
        winner = int(order[0])
        if winner != int(index):
            failures.append(
                (
                    str(identifiers[int(index)]),
                    str(identifiers[winner]),
                    float(scores[winner]),
                )
            )
    return SelfRetrievalResult(
        queried=len(selected),
        passed=len(selected) - len(failures),
        failed=len(failures),
        failures=tuple(failures),
    )


def distinctness_clusters(
    features: np.ndarray, *, threshold: float, block_size: int = 256
) -> DistinctnessResult:
    """Cluster rows connected by cosine distance at or below ``threshold``."""

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("features must be a matrix")
    if threshold < 0 or block_size <= 0:
        raise ValueError("threshold must be non-negative and block_size positive")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)
    count = len(matrix)
    parent = np.arange(count, dtype=np.int64)
    sizes = np.ones(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    pair_count = 0
    for start in range(0, count, block_size):
        similarities = matrix[start : start + block_size] @ matrix.T
        for local_index, row in enumerate(similarities):
            index = start + local_index
            neighbors = (
                np.flatnonzero((1.0 - row[index + 1 :]) <= threshold)
                + index
                + 1
            )
            pair_count += len(neighbors)
            for neighbor in neighbors:
                union(index, int(neighbor))

    groups: dict[int, int] = {}
    for index in range(count):
        root = find(index)
        groups[root] = groups.get(root, 0) + 1
    multi = [size for size in groups.values() if size > 1]
    return DistinctnessResult(
        threshold=threshold,
        pair_count=pair_count,
        cluster_count=len(multi),
        clustered_members=sum(multi),
        largest_cluster=max(multi, default=1),
        singleton_count=sum(size == 1 for size in groups.values()),
    )
