"""Display-value calibration and inverse lookup for plug-in parameters."""

from __future__ import annotations

import difflib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CALIBRATION_SCHEMA_VERSION = 1
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
UNIT_RE = re.compile(r"^[\s\[\(]*([%°A-Za-zµμ/]+)")


def canonical_display(value: object) -> str:
    text = str(value).strip().casefold()
    text = text.replace("μ", "µ").replace("−", "-")
    return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class NumericToken:
    value: float
    unit: str


def numeric_tokens(value: object) -> list[NumericToken]:
    text = str(value).replace("−", "-")
    result: list[NumericToken] = []
    for match in NUMBER_RE.finditer(text):
        try:
            number = float(match.group())
        except ValueError:
            continue
        unit_match = UNIT_RE.match(text[match.end() :])
        unit = canonical_display(unit_match.group(1)) if unit_match else ""
        result.append(NumericToken(number, unit))
    return result


def classify_samples(samples: list[list[object]]) -> str:
    displays = {canonical_display(sample[1]) for sample in samples}
    if len(displays) <= 1:
        return "constant"
    if len(displays) <= 64:
        return "stepped"
    return "continuous"


@dataclass(frozen=True, slots=True)
class CalibrationMatch:
    normalized: float
    display: str
    method: str
    score: float


def _midpoint_for_display(samples: list[list[object]], display: str) -> float:
    values = [float(sample[0]) for sample in samples if canonical_display(sample[1]) == display]
    return (min(values) + max(values)) / 2.0


def inverse_lookup(entry: dict[str, Any], target: object) -> CalibrationMatch:
    """Resolve a human-readable target to the closest sampled normalized value."""

    samples = entry.get("samples", [])
    if not samples:
        raise ValueError(f"Calibration entry {entry.get('name')!r} has no samples")
    target_text = canonical_display(target)
    exact = [sample for sample in samples if canonical_display(sample[1]) == target_text]
    if exact:
        normalized = _midpoint_for_display(samples, target_text)
        return CalibrationMatch(normalized, str(exact[0][1]), "exact-display", 1.0)

    target_numbers = numeric_tokens(target)
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        target_numbers = [NumericToken(float(target), "")]
    if target_numbers:
        best: tuple[float, list[object]] | None = None
        for sample in samples:
            candidates = numeric_tokens(sample[1])
            if not candidates:
                continue
            distances: list[float] = []
            for wanted in target_numbers:
                matching_units = [
                    item
                    for item in candidates
                    if not wanted.unit or item.unit == wanted.unit
                ]
                if not matching_units:
                    continue
                scale = max(abs(wanted.value), 1.0)
                distances.append(
                    min(abs(item.value - wanted.value) / scale for item in matching_units)
                )
            if not distances:
                continue
            distance = min(distances)
            if best is None or distance < best[0]:
                best = (distance, sample)
        if best is not None:
            distance, sample = best
            return CalibrationMatch(
                float(sample[0]), str(sample[1]), "numeric-nearest", 1.0 / (1.0 + distance)
            )

    scored = [
        (
            difflib.SequenceMatcher(None, target_text, canonical_display(sample[1])).ratio(),
            sample,
        )
        for sample in samples
    ]
    score, sample = max(scored, key=lambda item: item[0])
    display_key = canonical_display(sample[1])
    return CalibrationMatch(
        _midpoint_for_display(samples, display_key), str(sample[1]), "fuzzy-display", score
    )


def _numeric_for_target(display: object, wanted: NumericToken) -> float | None:
    candidates = numeric_tokens(display)
    if wanted.unit:
        candidates = [item for item in candidates if item.unit == wanted.unit]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item.value - wanted.value)).value


def refine_numeric_live(
    entry: dict[str, Any],
    target: object,
    processor: Any,
    *,
    normalized_tolerance: float = 1e-4,
    max_iterations: int = 20,
) -> CalibrationMatch:
    """Biselect a numeric display target against a live plug-in parameter.

    The cached table supplies a safe local bracket. The plug-in is touched only
    for the selected parameter and its original normalized value is restored.
    """

    cached = inverse_lookup(entry, target)
    if entry.get("kind") != "continuous":
        return cached
    wanted_tokens = numeric_tokens(target)
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        wanted_tokens = [NumericToken(float(target), "")]
    if not wanted_tokens:
        return cached
    wanted = wanted_tokens[0]

    points: list[tuple[float, float, str]] = []
    for normalized, display in entry.get("samples", []):
        numeric = _numeric_for_target(display, wanted)
        if numeric is not None and math.isfinite(numeric):
            points.append((float(normalized), numeric, str(display)))
    points.sort(key=lambda item: item[0])
    brackets: list[tuple[float, tuple[float, float, str], tuple[float, float, str]]] = []
    for left, right in zip(points, points[1:]):
        left_delta = left[1] - wanted.value
        right_delta = right[1] - wanted.value
        if left_delta == 0:
            return CalibrationMatch(left[0], left[2], "cached-numeric-exact", 1.0)
        if right_delta == 0:
            return CalibrationMatch(right[0], right[2], "cached-numeric-exact", 1.0)
        if left_delta * right_delta < 0:
            brackets.append((abs(left_delta) + abs(right_delta), left, right))
    if not brackets:
        return cached
    _span, left, right = min(brackets, key=lambda item: item[0])
    low_norm, low_value, low_display = left
    high_norm, high_value, high_display = right
    if low_norm > high_norm:
        low_norm, high_norm = high_norm, low_norm
        low_value, high_value = high_value, low_value
        low_display, high_display = high_display, low_display

    from core.plugin_host import dawdreamer_parameter_display

    index = int(entry["index"])
    original = float(processor.get_parameter(index))
    best_norm = cached.normalized
    best_display = cached.display
    best_distance = math.inf
    try:
        for _ in range(max_iterations):
            midpoint = (low_norm + high_norm) / 2.0
            processor.set_parameter(index, midpoint)
            actual = float(processor.get_parameter(index))
            display = dawdreamer_parameter_display(processor, index, actual)
            numeric = _numeric_for_target(display, wanted)
            if numeric is None:
                return cached
            distance = abs(numeric - wanted.value)
            if distance < best_distance:
                best_norm, best_display, best_distance = actual, display, distance
            if distance == 0 or canonical_display(display) == canonical_display(target):
                break
            if high_norm - low_norm < normalized_tolerance:
                break
            # Works for either increasing or decreasing display functions.
            increasing = high_value >= low_value
            if (numeric < wanted.value) == increasing:
                low_norm, low_value, low_display = midpoint, numeric, display
            else:
                high_norm, high_value, high_display = midpoint, numeric, display
    finally:
        processor.set_parameter(index, original)
    scale = max(abs(wanted.value), 1.0)
    score = 1.0 / (1.0 + best_distance / scale)
    return CalibrationMatch(best_norm, best_display, "live-bisection", score)


class CalibrationTable:
    def __init__(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported calibration schema: {payload.get('schema_version')}")
        self.payload = payload

    @classmethod
    def load(cls, path: Path) -> "CalibrationTable":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def entries(self, name: str) -> list[dict[str, Any]]:
        entries = self.payload.get("parameters", {}).get(name, [])
        return list(entries) if isinstance(entries, list) else []

    def entry(self, name: str, index: int | None = None) -> dict[str, Any]:
        entries = self.entries(name)
        if index is not None:
            entries = [entry for entry in entries if int(entry["index"]) == index]
        if len(entries) != 1:
            raise KeyError(
                f"Parameter {name!r} resolves to {len(entries)} entries; provide an index"
            )
        return entries[0]

    def inverse(self, name: str, target: object, index: int | None = None) -> CalibrationMatch:
        return inverse_lookup(self.entry(name, index), target)

    def inverse_live(
        self,
        processor: Any,
        name: str,
        target: object,
        index: int | None = None,
        *,
        normalized_tolerance: float = 1e-4,
    ) -> CalibrationMatch:
        return refine_numeric_live(
            self.entry(name, index),
            target,
            processor,
            normalized_tolerance=normalized_tolerance,
        )


def calibration_stats(payload: dict[str, Any]) -> dict[str, int]:
    entries = [
        entry
        for group in payload.get("parameters", {}).values()
        for entry in group
    ]
    kinds: dict[str, int] = {}
    for entry in entries:
        kind = str(entry.get("kind", "unknown"))
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "parameters": len(entries),
        "observations": sum(len(entry.get("samples", [])) for entry in entries),
        **{f"kind_{key}": value for key, value in sorted(kinds.items())},
    }
