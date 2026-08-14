"""Serializable one- or two-preset stacks and deterministic mix polishing."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Callable, Literal, Sequence

import numpy as np

from core.matcher import loudness_normalize


STACK_VERSION = 1
LayerRole = Literal["dominant", "residual"]
BatchScorer = Callable[[Sequence[np.ndarray]], Sequence[float]]


@dataclass(frozen=True, slots=True)
class PresetLayer:
    synth: str
    base_preset_id: int
    state_reference: str
    candidate_state_sha256: str
    decoded_audio_sha256: str
    gain_db: float
    timing_offset_ms: float
    role: LayerRole
    match_score: float
    midi_note: int
    origin: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PresetLayer":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class PresetStack:
    target_synth: str
    layers: tuple[PresetLayer, ...]
    combined_final_score: float
    residual_energy_ratio: float
    second_layer_selected: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    stack_version: int = STACK_VERSION

    def __post_init__(self) -> None:
        if self.stack_version != STACK_VERSION:
            raise ValueError(f"Unsupported preset stack version {self.stack_version}")
        if len(self.layers) not in (1, 2):
            raise ValueError("A Stage 3I preset stack must contain one or two layers")
        if self.layers[0].role != "dominant":
            raise ValueError("Layer 1 must have the dominant role")
        if len(self.layers) == 2 and self.layers[1].role != "residual":
            raise ValueError("Layer 2 must have the residual role")
        if self.second_layer_selected != (len(self.layers) == 2):
            raise ValueError("Layer count and second-layer selection disagree")

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["layer_count"] = self.layer_count
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PresetStack":
        payload = dict(value)
        declared_count = payload.pop("layer_count", None)
        payload["layers"] = tuple(PresetLayer.from_dict(item) for item in payload["layers"])
        result = cls(**payload)
        if declared_count is not None and int(declared_count) != result.layer_count:
            raise ValueError("Serialized preset stack layer count is inconsistent")
        return result

    @classmethod
    def from_json(cls, value: str) -> "PresetStack":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, slots=True)
class MixPolishResult:
    audio: np.ndarray = field(repr=False, compare=False)
    score: float
    layer1_gain_db: float
    layer2_gain_db: float | None
    layer2_timing_offset_ms: float
    second_layer_selected: bool
    gain_combination_count: int
    mixture_evaluation_count: int
    search_trace: tuple[dict[str, float], ...]


def scale_to_db(scale: float, *, floor_db: float = -80.0) -> float:
    if scale <= 0.0:
        return float(floor_db)
    return float(max(floor_db, 20.0 * math.log10(scale)))


def apply_timing_offset(
    audio: np.ndarray,
    offset_ms: float,
    sample_rate: int,
    *,
    length: int | None = None,
) -> np.ndarray:
    """Shift audio without wrapping; positive offsets delay Layer 2."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    output_length = len(values) if length is None else int(length)
    output = np.zeros(output_length, dtype=np.float32)
    samples = int(round(float(offset_ms) * sample_rate / 1000.0))
    if samples >= 0:
        count = min(len(values), max(0, output_length - samples))
        if count:
            output[samples : samples + count] = values[:count]
    else:
        source_start = min(len(values), -samples)
        count = min(len(values) - source_start, output_length)
        if count:
            output[:count] = values[source_start : source_start + count]
    return output


def mix_layers(
    layer1_audio: np.ndarray,
    layer2_audio: np.ndarray | None,
    *,
    sample_rate: int,
    layer1_gain_db: float = 0.0,
    layer2_gain_db: float | None = None,
    layer2_timing_offset_ms: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Mix a stack, preserving the exact Layer 1 waveform for the null path."""

    layer1 = np.asarray(layer1_audio, dtype=np.float32).reshape(-1)
    if layer2_audio is None or layer2_gain_db is None or math.isinf(layer2_gain_db):
        return np.ascontiguousarray(layer1.copy(), dtype=np.float32)
    shifted = apply_timing_offset(
        layer2_audio,
        layer2_timing_offset_ms,
        sample_rate,
        length=len(layer1),
    )
    mixed = (
        layer1 * (10.0 ** (float(layer1_gain_db) / 20.0))
        + shifted * (10.0 ** (float(layer2_gain_db) / 20.0))
    ).astype(np.float32)
    return loudness_normalize(mixed) if normalize else np.ascontiguousarray(mixed)


def deterministic_mix_polish(
    *,
    layer1_audio: np.ndarray,
    layer2_audio: np.ndarray,
    sample_rate: int,
    initial_layer1_scale: float,
    initial_layer2_scale: float,
    one_layer_score: float,
    score_batch: BatchScorer,
) -> MixPolishResult:
    """Search only stack gains/timing with an exact mandatory null candidate.

    The deterministic schedule evaluates nine coarse gain pairs, all eleven
    required Layer 2 timing offsets for the best coarse pair, then at most 25
    local gain pairs. This is at most 34 distinct gain combinations and 45
    mixture evaluations, comfortably below Stage 3I's 100-combination cap.
    """

    layer1 = np.asarray(layer1_audio, dtype=np.float32).reshape(-1)
    layer2 = np.asarray(layer2_audio, dtype=np.float32).reshape(-1)
    initial1_db = scale_to_db(initial_layer1_scale)
    initial2_db = scale_to_db(initial_layer2_scale)
    trace: list[dict[str, float]] = []
    gain_pairs: set[tuple[float, float]] = set()

    def evaluate(
        pairs: Sequence[tuple[float, float, float]], stage: str
    ) -> tuple[float, tuple[float, float, float], np.ndarray]:
        audios = [
            mix_layers(
                layer1,
                layer2,
                sample_rate=sample_rate,
                layer1_gain_db=gain1,
                layer2_gain_db=gain2,
                layer2_timing_offset_ms=offset,
            )
            for gain1, gain2, offset in pairs
        ]
        scores = [float(value) for value in score_batch(audios)]
        if len(scores) != len(pairs):
            raise RuntimeError("Mix scorer returned the wrong number of scores")
        for (gain1, gain2, offset), score in zip(pairs, scores, strict=True):
            gain_pairs.add((gain1, gain2))
            trace.append(
                {
                    "stage": stage,
                    "layer1_gain_db": gain1,
                    "layer2_gain_db": gain2,
                    "layer2_timing_offset_ms": offset,
                    "score": score,
                }
            )
        winner = max(range(len(scores)), key=lambda index: (scores[index], -index))
        return scores[winner], pairs[winner], audios[winner]

    coarse = [
        (initial1_db + left, initial2_db + right, 0.0)
        for left, right in product((-6.0, 0.0, 6.0), repeat=2)
    ]
    best_score, best_parameters, best_audio = evaluate(coarse, "coarse-gain")
    gain1, gain2, _offset = best_parameters
    timing = [(gain1, gain2, float(offset)) for offset in range(-25, 26, 5)]
    timing_score, timing_parameters, timing_audio = evaluate(timing, "timing")
    if timing_score > best_score:
        best_score, best_parameters, best_audio = timing_score, timing_parameters, timing_audio

    gain1, gain2, offset = best_parameters
    refined_pairs: list[tuple[float, float, float]] = []
    for left, right in product((-3.0, -1.5, 0.0, 1.5, 3.0), repeat=2):
        candidate1 = float(np.clip(gain1 + left, initial1_db - 6.0, initial1_db + 6.0))
        candidate2 = float(np.clip(gain2 + right, initial2_db - 6.0, initial2_db + 6.0))
        value = (candidate1, candidate2, offset)
        if value not in refined_pairs:
            refined_pairs.append(value)
    refined_score, refined_parameters, refined_audio = evaluate(
        refined_pairs, "refined-gain"
    )
    if refined_score > best_score:
        best_score, best_parameters, best_audio = (
            refined_score,
            refined_parameters,
            refined_audio,
        )

    selected = bool(best_score > float(one_layer_score))
    if not selected:
        return MixPolishResult(
            audio=np.ascontiguousarray(layer1.copy(), dtype=np.float32),
            score=float(one_layer_score),
            layer1_gain_db=0.0,
            layer2_gain_db=None,
            layer2_timing_offset_ms=0.0,
            second_layer_selected=False,
            gain_combination_count=len(gain_pairs),
            mixture_evaluation_count=len(trace),
            search_trace=tuple(trace),
        )
    gain1, gain2, offset = best_parameters
    return MixPolishResult(
        audio=np.ascontiguousarray(best_audio, dtype=np.float32),
        score=float(best_score),
        layer1_gain_db=float(gain1),
        layer2_gain_db=float(gain2),
        layer2_timing_offset_ms=float(offset),
        second_layer_selected=True,
        gain_combination_count=len(gain_pairs),
        mixture_evaluation_count=len(trace),
        search_trace=tuple(trace),
    )
