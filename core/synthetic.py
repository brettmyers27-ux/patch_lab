"""Serum 1 random-patch augmentation with audible synthesis priors."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np


MODULATION_PATTERN = re.compile(r"^(?:Mod\s*\d+|LFO Bus)", re.IGNORECASE)


def sample_serum1_patch(
    rng: np.random.Generator,
    initial: np.ndarray,
    mapping: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Sample one normalized patch while preserving routing and audibility."""

    values = np.asarray(initial, dtype=np.float32).copy()
    for field in mapping:
        index = int(field["index"])
        name = str(field["name"])
        if MODULATION_PATTERN.search(name):
            continue
        if field.get("stepped") and field.get("step_values"):
            choices = np.asarray(field["step_values"], dtype=np.float32)
            values[index] = float(rng.choice(choices))
        else:
            values[index] = float(rng.beta(1.5, 1.5))

    # Stable audible priors. At least one source is active and comfortably loud.
    values[0] = rng.uniform(0.65, 0.95)  # MasterVol
    on_indices = np.asarray([212, 213, 214, 215])
    level_indices = np.asarray([1, 14, 27, 33])
    active = rng.random(4) < np.asarray([0.8, 0.55, 0.2, 0.3])
    if not np.any(active):
        active[int(rng.integers(0, 2))] = True
    values[on_indices] = active.astype(np.float32)
    values[level_indices[active]] = rng.uniform(0.4, 0.9, int(np.sum(active)))
    values[35] = min(values[35], 0.35)  # Env1 attack
    values[38] = max(values[38], 0.45)  # Env1 sustain
    values[39] = max(values[39], 0.1)  # Env1 release
    if values[216] >= 0.5:  # Filter On
        values[45] = max(values[45], 0.2)
        values[49] = max(values[49], 0.35)
    values[315] = 0.0  # Host bypass off
    return np.clip(values, 0.0, 1.0)
