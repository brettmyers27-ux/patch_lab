"""Deterministic on-manifold perturbations around real Serum presets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _subset(
    rng: np.random.Generator, indices: Sequence[int], low: float = 0.10, high: float = 0.40
) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.size == 0:
        return values
    count = max(1, int(round(values.size * float(rng.uniform(low, high)))))
    return rng.choice(values, size=min(count, values.size), replace=False)


def _nearby_position(rng: np.random.Generator, current: int, count: int) -> int:
    if count < 2:
        return current
    choices = []
    if current > 0:
        choices.append(current - 1)
    if current + 1 < count:
        choices.append(current + 1)
    return int(rng.choice(choices))


def perturb_serum1(
    base: np.ndarray,
    mask: np.ndarray,
    mapping: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Jitter a real Serum 1 vector while retaining its audible topology."""

    result = np.asarray(base, dtype=np.float32).copy()
    present = np.asarray(mask, dtype=np.bool_)
    continuous = [
        int(field["index"])
        for field in mapping
        if present[int(field["index"])] and not bool(field.get("stepped"))
    ]
    changed = _subset(rng, continuous)
    sigma = float(rng.uniform(0.05, 0.15))
    result[changed] += rng.normal(0.0, sigma, size=changed.size).astype(np.float32)

    enum_changed = False
    if rng.random() < 0.10:
        candidates = [
            field
            for field in mapping
            if present[int(field["index"])] and len(field.get("step_values", ())) > 1
        ]
        if candidates:
            field = candidates[int(rng.integers(len(candidates)))]
            index = int(field["index"])
            choices = np.asarray(field["step_values"], dtype=np.float32)
            position = int(np.argmin(np.abs(choices - result[index])))
            result[index] = choices[_nearby_position(rng, position, len(choices))]
            enum_changed = True
    np.clip(result, 0.0, 1.0, out=result)
    return result, {
        "continuous_changed": int(changed.size),
        "continuous_available": len(continuous),
        "sigma": sigma,
        "enum_changed": enum_changed,
    }


def perturb_serum2(
    base: np.ndarray,
    mask: np.ndarray,
    schema: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Jitter present numeric fields and occasionally swap one nearby category."""

    result = np.asarray(base, dtype=np.float32).copy()
    present = np.asarray(mask, dtype=np.bool_)
    numeric = [
        int(field["index"])
        for field in schema["fields"]
        if field["encoding"] == "minmax_float" and present[int(field["index"])]
    ]
    changed = _subset(rng, numeric)
    sigma = float(rng.uniform(0.05, 0.15))
    result[changed] += rng.normal(0.0, sigma, size=changed.size).astype(np.float32)

    enum_changed = False
    if rng.random() < 0.10:
        candidates = [
            field
            for field in schema["fields"]
            if field["encoding"] == "one_hot"
            and present[int(field["index"])]
            and int(field["width"]) > 1
        ]
        if candidates:
            field = candidates[int(rng.integers(len(candidates)))]
            start, width = int(field["index"]), int(field["width"])
            current = int(np.argmax(result[start : start + width]))
            replacement = _nearby_position(rng, current, width)
            result[start : start + width] = 0.0
            result[start + replacement] = 1.0
            enum_changed = True
    np.clip(result, 0.0, 1.0, out=result)
    return result, {
        "continuous_changed": int(changed.size),
        "continuous_available": len(numeric),
        "sigma": sigma,
        "enum_changed": enum_changed,
    }
