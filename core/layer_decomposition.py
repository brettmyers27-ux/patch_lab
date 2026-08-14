"""Phase-robust two-layer residual extraction for the Stage 3I experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import librosa
import numpy as np

from core.matcher import loudness_normalize
from core.preset_stack import apply_timing_offset, scale_to_db


DEFAULT_FFT_SIZE = 2048
DEFAULT_HOP_LENGTH = 512


@dataclass(frozen=True, slots=True)
class ResidualDiagnostics:
    alignment_offset_ms: float
    layer1_magnitude_scale: float
    layer1_gain_db: float
    residual_rms_ratio: float
    residual_spectral_energy_ratio: float
    residual_duration_s: float
    residual_centroid_hz: float
    residual_noisiness: float
    residual_harmonic_percussive_ratio: float
    residual_tail_energy_fraction: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResidualResult:
    residual_audio: np.ndarray = field(repr=False, compare=False)
    normalized_residual_audio: np.ndarray = field(repr=False, compare=False)
    residual_magnitude: np.ndarray = field(repr=False, compare=False)
    target_magnitude: np.ndarray = field(repr=False, compare=False)
    aligned_layer1_audio: np.ndarray = field(repr=False, compare=False)
    diagnostics: ResidualDiagnostics


def stft_magnitude(
    audio: np.ndarray,
    *,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    return np.abs(
        librosa.stft(
            values,
            n_fft=fft_size,
            hop_length=hop_length,
            window="hann",
        )
    ).astype(np.float32)


def fit_nonnegative_magnitude_scale(
    target_magnitude: np.ndarray,
    layer_magnitude: np.ndarray,
    *,
    maximum: float = 2.0,
) -> float:
    target = np.asarray(target_magnitude, dtype=np.float64)
    layer = np.asarray(layer_magnitude, dtype=np.float64)
    denominator = float(np.sum(np.square(layer)))
    if denominator <= 1e-20:
        return 0.0
    scale = float(np.sum(target * layer) / denominator)
    return float(np.clip(scale, 0.0, maximum))


def subtract_magnitudes(
    target_magnitude: np.ndarray,
    layer_magnitude: np.ndarray,
    scale: float,
) -> np.ndarray:
    return np.maximum(
        np.asarray(target_magnitude, dtype=np.float32)
        - float(scale) * np.asarray(layer_magnitude, dtype=np.float32),
        0.0,
    ).astype(np.float32)


def _alignment_offsets(maximum_ms: int, step_ms: int) -> tuple[int, ...]:
    values = list(range(-maximum_ms, maximum_ms + 1, step_ms))
    return tuple(sorted(values, key=lambda value: (abs(value), value)))


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def phase_robust_residual(
    target_audio: np.ndarray,
    layer1_audio: np.ndarray,
    sample_rate: int,
    *,
    maximum_alignment_ms: int = 100,
    alignment_step_ms: int = 5,
    fft_size: int = DEFAULT_FFT_SIZE,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> ResidualResult:
    """Subtract an aligned, fitted Layer 1 STFT magnitude from the target."""

    target = np.asarray(target_audio, dtype=np.float32).reshape(-1)
    layer1 = np.asarray(layer1_audio, dtype=np.float32).reshape(-1)
    if not len(target):
        raise ValueError("Target audio must not be empty")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if maximum_alignment_ms < 0 or alignment_step_ms <= 0:
        raise ValueError("Alignment bounds must be non-negative with a positive step")
    target_stft = librosa.stft(
        target,
        n_fft=fft_size,
        hop_length=hop_length,
        window="hann",
    )
    target_magnitude = np.abs(target_stft).astype(np.float32)
    target_energy = float(np.sum(np.square(target_magnitude, dtype=np.float64)))

    best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
    if _rms(layer1) <= 1e-12:
        offsets = (0,)
    else:
        offsets = _alignment_offsets(maximum_alignment_ms, alignment_step_ms)
    for offset_ms in offsets:
        aligned = apply_timing_offset(layer1, offset_ms, sample_rate, length=len(target))
        layer_magnitude = stft_magnitude(
            aligned, fft_size=fft_size, hop_length=hop_length
        )
        scale = fit_nonnegative_magnitude_scale(target_magnitude, layer_magnitude)
        residual_magnitude = subtract_magnitudes(
            target_magnitude, layer_magnitude, scale
        )
        ratio = float(
            np.sum(np.square(residual_magnitude, dtype=np.float64))
            / max(target_energy, 1e-20)
        )
        candidate = (ratio, int(offset_ms), scale, aligned, residual_magnitude)
        if best is None or candidate[0] < best[0] - 1e-15:
            best = candidate
    assert best is not None
    energy_ratio, offset_ms, scale, aligned, residual_magnitude = best
    residual_complex = residual_magnitude * np.exp(1j * np.angle(target_stft))
    residual = librosa.istft(
        residual_complex,
        hop_length=hop_length,
        window="hann",
        length=len(target),
    ).astype(np.float32)
    residual_rms = _rms(residual)
    target_rms = _rms(target)
    if float(np.sum(residual_magnitude)) > 1e-12:
        centroid = librosa.feature.spectral_centroid(
            S=residual_magnitude, sr=sample_rate
        )[0]
        flatness = librosa.feature.spectral_flatness(S=residual_magnitude)[0]
        harmonic, percussive = librosa.decompose.hpss(residual_magnitude)
        harmonic_energy = float(np.mean(np.square(harmonic, dtype=np.float64)))
        percussive_energy = float(np.mean(np.square(percussive, dtype=np.float64)))
        centroid_hz = float(np.mean(centroid))
        noisiness = float(np.mean(flatness))
        hp_ratio = harmonic_energy / max(percussive_energy, 1e-12)
    else:
        centroid_hz = 0.0
        noisiness = 0.0
        hp_ratio = 0.0
    split = max(1, int(round(len(residual) * 0.75)))
    total_time_energy = float(np.sum(np.square(residual, dtype=np.float64)))
    tail_fraction = float(
        np.sum(np.square(residual[split:], dtype=np.float64))
        / max(total_time_energy, 1e-20)
    )
    diagnostics = ResidualDiagnostics(
        alignment_offset_ms=float(offset_ms),
        layer1_magnitude_scale=float(scale),
        layer1_gain_db=scale_to_db(scale),
        residual_rms_ratio=residual_rms / max(target_rms, 1e-20),
        residual_spectral_energy_ratio=float(energy_ratio),
        residual_duration_s=len(residual) / sample_rate,
        residual_centroid_hz=centroid_hz,
        residual_noisiness=noisiness,
        residual_harmonic_percussive_ratio=hp_ratio,
        residual_tail_energy_fraction=tail_fraction,
    )
    return ResidualResult(
        residual_audio=np.ascontiguousarray(residual, dtype=np.float32),
        normalized_residual_audio=loudness_normalize(residual),
        residual_magnitude=np.ascontiguousarray(residual_magnitude, dtype=np.float32),
        target_magnitude=np.ascontiguousarray(target_magnitude, dtype=np.float32),
        aligned_layer1_audio=np.ascontiguousarray(aligned, dtype=np.float32),
        diagnostics=diagnostics,
    )
