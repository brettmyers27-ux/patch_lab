"""Audio-guided shortlist estimators for Serum 2 structural choices.

These estimators rank choices; they never form an allow-list.  A matcher can
fall back to the full structural vocabulary when confidence is low.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import librosa
import numpy as np


@dataclass(frozen=True, slots=True)
class RankedChoice:
    value: str
    confidence: float


@dataclass(frozen=True, slots=True)
class EstimatorMetrics:
    samples: int
    classes: int
    top1: float
    top5: float
    common_top1: float
    common_top5: float
    adopted: bool


def audio_descriptor(audio: np.ndarray, sample_rate: int = 24_000, *, mode: str = "full") -> np.ndarray:
    """Compact timbre + trajectory descriptor used by all structural heads."""

    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mode == "noise":
        _harmonic, y = librosa.effects.hpss(y)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    y = y / max(peak, 1e-7)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_fft=1024, hop_length=256, n_mels=64, fmin=30.0
    )
    logmel = librosa.power_to_db(mel + 1e-10, ref=np.max)
    centroid = librosa.feature.spectral_centroid(S=np.sqrt(mel), sr=sample_rate)[0]
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    trajectory = np.concatenate(
        [
            np.quantile(centroid / max(sample_rate / 2, 1), [0.1, 0.25, 0.5, 0.75, 0.9]),
            np.quantile(rms, [0.1, 0.25, 0.5, 0.75, 0.9]),
            [np.std(centroid) / max(sample_rate / 2, 1), np.std(rms)],
        ]
    )
    descriptor = np.concatenate(
        [np.mean(logmel, axis=1) / 80.0, np.std(logmel, axis=1) / 40.0, trajectory]
    ).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    return descriptor / max(norm, 1e-8)


class NearestStructuralEstimator:
    """Cosine nearest-exemplar ranker with class-frequency calibration."""

    def __init__(self, *, mode: str = "full") -> None:
        self.mode = mode
        self.features = np.empty((0, 140), dtype=np.float32)
        self.labels: list[str] = []
        self.counts: Counter[str] = Counter()
        self.enabled = False

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> "NearestStructuralEstimator":
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != len(labels) or not len(labels):
            raise ValueError("features and labels must be non-empty aligned arrays")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.features = matrix / np.maximum(norms, 1e-8)
        self.labels = [str(label) for label in labels]
        self.counts = Counter(self.labels)
        return self

    def rank_descriptor(self, descriptor: np.ndarray, top_k: int = 5) -> list[RankedChoice]:
        if not len(self.labels):
            return []
        query = np.asarray(descriptor, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-8)
        similarities = self.features @ query
        by_label: dict[str, float] = defaultdict(lambda: -1.0)
        for label, score in zip(self.labels, similarities, strict=True):
            by_label[label] = max(by_label[label], float(score))
        ordered = sorted(by_label.items(), key=lambda item: (-item[1], -self.counts[item[0]], item[0]))
        if not ordered:
            return []
        scores = np.asarray([score for _label, score in ordered[:top_k]], dtype=np.float64)
        probabilities = np.exp((scores - scores.max()) * 8.0)
        probabilities /= max(float(probabilities.sum()), 1e-12)
        return [
            RankedChoice(label, float(confidence))
            for (label, _score), confidence in zip(ordered[:top_k], probabilities, strict=True)
        ]

    def rank(self, audio: np.ndarray, sample_rate: int = 24_000, top_k: int = 5) -> list[RankedChoice]:
        return self.rank_descriptor(audio_descriptor(audio, sample_rate, mode=self.mode), top_k)


def evaluate_estimator(
    estimator: NearestStructuralEstimator,
    test_features: np.ndarray,
    test_labels: Sequence[str],
    *,
    top_k: int = 5,
) -> EstimatorMetrics:
    labels = [str(value) for value in test_labels]
    predictions = [estimator.rank_descriptor(row, top_k) for row in test_features]
    top1 = float(np.mean([bool(items) and items[0].value == truth for items, truth in zip(predictions, labels, strict=True)]))
    top5 = float(np.mean([truth in {item.value for item in items} for items, truth in zip(predictions, labels, strict=True)]))
    common = [label for label, _count in estimator.counts.most_common(top_k)]
    common_top1 = float(np.mean([truth == common[0] for truth in labels])) if common else 0.0
    common_top5 = float(np.mean([truth in common for truth in labels])) if common else 0.0
    adopted = top1 > common_top1
    estimator.enabled = adopted
    return EstimatorMetrics(
        samples=len(labels),
        classes=len(set(estimator.labels) | set(labels)),
        top1=top1,
        top5=top5,
        common_top1=common_top1,
        common_top5=common_top5,
        adopted=adopted,
    )


def deterministic_split(
    ids: Iterable[int], *, holdout_modulus: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(list(ids), dtype=np.int64)
    test = values % holdout_modulus == 0
    return np.flatnonzero(~test), np.flatnonzero(test)
