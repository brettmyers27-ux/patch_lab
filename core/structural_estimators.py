"""Audio-guided shortlist estimators for Serum 2 structural choices.

These estimators rank choices; they never form an allow-list.  A matcher can
fall back to the full structural vocabulary when confidence is low.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
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


def modulation_descriptor(audio: np.ndarray, sample_rate: int = 24_000) -> np.ndarray:
    """Describe periodic amplitude, brightness, pitch, and flux trajectories.

    Unlike :func:`audio_descriptor`, this deliberately discards the mean
    spectrum.  A modulation route is identified by what moves over time, not
    by the neutral carrier used to expose that movement.
    """

    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(y):
        return np.zeros(108, dtype=np.float32)
    y = y / max(float(np.max(np.abs(y))), 1e-7)
    spectrum = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    rms = librosa.feature.rms(S=spectrum, frame_length=1024, hop_length=256)[0]
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=sample_rate)[0]
    flux = np.sqrt(np.sum(np.diff(spectrum, axis=1, prepend=spectrum[:, :1]) ** 2, axis=0))
    trajectories = (
        rms,
        centroid / max(sample_rate / 2, 1),
        flux / max(float(np.max(flux)), 1e-7),
    )
    pieces: list[np.ndarray] = []
    for trajectory in trajectories:
        values = np.asarray(trajectory, dtype=np.float32)
        values = values - float(np.mean(values))
        scale = max(float(np.std(values)), 1e-7)
        normalized = values / scale
        spectrum_1d = np.abs(np.fft.rfft(normalized, n=64))[1:25]
        spectrum_1d /= max(float(np.linalg.norm(spectrum_1d)), 1e-8)
        stats = np.asarray(
            [
                np.std(trajectory),
                np.mean(np.abs(np.diff(trajectory))) if len(trajectory) > 1 else 0.0,
                *np.quantile(trajectory, [0.1, 0.25, 0.5, 0.75, 0.9]),
                np.max(trajectory) - np.min(trajectory),
                np.mean(trajectory),
                np.sqrt(np.mean(np.square(trajectory))),
                np.mean(np.abs(normalized)),
                np.max(np.abs(normalized)),
            ],
            dtype=np.float32,
        )
        pieces.extend((spectrum_1d.astype(np.float32), stats))
    descriptor = np.concatenate(pieces).astype(np.float32)
    return descriptor / max(float(np.linalg.norm(descriptor)), 1e-8)


def controlled_descriptor(
    audio: np.ndarray, category: str, sample_rate: int = 24_000
) -> np.ndarray:
    """Return the category-specific Stage 3B controlled fingerprint."""

    if category == "mod_route":
        return modulation_descriptor(audio, sample_rate)
    return audio_descriptor(
        audio,
        sample_rate,
        mode="noise" if category == "noise_sample" else "full",
    )


class ControlledFingerprintIndex:
    """Direct stable-ID lookup over neutral-patch structural renders."""

    def __init__(
        self,
        features: dict[str, np.ndarray],
        stable_ids: dict[str, Sequence[str]],
        labels: dict[str, Sequence[str]],
        *,
        adopted: Iterable[str] = (),
        prior_strengths: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.features: dict[str, np.ndarray] = {}
        self.stable_ids = {key: [str(value) for value in values] for key, values in stable_ids.items()}
        self.labels = {key: [str(value) for value in values] for key, values in labels.items()}
        for category, matrix in features.items():
            values = np.asarray(matrix, dtype=np.float32)
            if values.ndim != 2:
                raise ValueError(f"{category} fingerprints must be a matrix")
            if len(values) != len(self.stable_ids.get(category, ())) or len(values) != len(
                self.labels.get(category, ())
            ):
                raise ValueError(f"{category} fingerprint rows and IDs are not aligned")
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            self.features[category] = values / np.maximum(norms, 1e-8)
        self.adopted = frozenset(str(value) for value in adopted)
        self.prior_strengths = {
            str(key): float(value) for key, value in (prior_strengths or {}).items()
        }
        self.metadata = dict(metadata or {})

    def rank_descriptor(
        self,
        category: str,
        descriptor: np.ndarray,
        *,
        top_k: int = 5,
        log_priors: dict[str, float] | None = None,
    ) -> list[RankedChoice]:
        matrix = self.features.get(category)
        if matrix is None or not len(matrix):
            return []
        query = np.asarray(descriptor, dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-8)
        scores = matrix @ query
        strength = self.prior_strengths.get(category, 0.0)
        by_id: dict[str, float] = {}
        for identifier, label, score in zip(
            self.stable_ids[category], self.labels[category], scores, strict=True
        ):
            adjusted = float(score) + strength * float((log_priors or {}).get(label, 0.0))
            by_id[identifier] = max(by_id.get(identifier, -np.inf), adjusted)
        ordered = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        if not ordered:
            return []
        raw = np.asarray([score for _identifier, score in ordered], dtype=np.float64)
        confidence = np.exp((raw - raw.max()) * 8.0)
        confidence /= max(float(confidence.sum()), 1e-12)
        return [
            RankedChoice(identifier, float(probability))
            for (identifier, _score), probability in zip(ordered, confidence, strict=True)
        ]

    def rank(
        self,
        category: str,
        audio: np.ndarray,
        sample_rate: int = 24_000,
        *,
        top_k: int = 5,
        log_priors: dict[str, float] | None = None,
    ) -> list[RankedChoice]:
        return self.rank_descriptor(
            category,
            controlled_descriptor(audio, category, sample_rate),
            top_k=top_k,
            log_priors=log_priors,
        )

    def save(self, path: Path) -> None:
        """Persist without pickle so the private artifact is safe to validate."""

        payload: dict[str, Any] = {
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        **self.metadata,
                        "adopted": sorted(self.adopted),
                        "prior_strengths": self.prior_strengths,
                    },
                    sort_keys=True,
                )
            )
        }
        for category in sorted(self.features):
            payload[f"{category}__features"] = self.features[category]
            payload[f"{category}__stable_ids"] = np.asarray(self.stable_ids[category])
            payload[f"{category}__labels"] = np.asarray(self.labels[category])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(path).with_name(Path(path).name + ".tmp.npz")
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ControlledFingerprintIndex":
        archive = np.load(Path(path), allow_pickle=False)
        metadata = json.loads(str(archive["metadata_json"]))
        categories = sorted(
            name.removesuffix("__features")
            for name in archive.files
            if name.endswith("__features")
        )
        return cls(
            {category: archive[f"{category}__features"] for category in categories},
            {category: archive[f"{category}__stable_ids"].tolist() for category in categories},
            {category: archive[f"{category}__labels"].tolist() for category in categories},
            adopted=metadata.pop("adopted", ()),
            prior_strengths=metadata.pop("prior_strengths", {}),
            metadata=metadata,
        )


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
