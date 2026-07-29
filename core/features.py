"""LAION-CLAP embeddings and deterministic handcrafted audio features."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf
import torch

from core.platform_env import PlatformEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "models"
HF_DIR = Path(
    os.environ.get("PATCHLAB_MODEL_CACHE", str(MODEL_DIR / "huggingface"))
).expanduser()
CLAP_CHECKPOINT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_CHECKPOINT = MODEL_DIR / CLAP_CHECKPOINT_NAME
CLAP_SAMPLE_RATE = 48_000
CLAP_DIMENSIONS = 512
HANDCRAFTED_NAMES = (
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "zero_crossing_rate_mean",
    "log_attack_time",
    "harmonic_percussive_energy_ratio",
)


def configure_model_cache() -> None:
    HF_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_DIR / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(MODEL_DIR / "torch"))
    # All required assets are populated by scripts/cache_clap.py. Avoid remote
    # HEAD probes (and their retry backoff) during the hours-long feature pass.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    waveform: np.ndarray
    sample_rate: int = CLAP_SAMPLE_RATE


def load_audio_48k_mono(path: Path) -> PreparedAudio:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if sample_rate != CLAP_SAMPLE_RATE:
        mono = librosa.resample(
            mono,
            orig_sr=sample_rate,
            target_sr=CLAP_SAMPLE_RATE,
            res_type="soxr_hq",
        ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(mono)):
        raise ValueError(f"Non-finite audio samples in {path}")
    # Deliberately do not clip peaks above 1.0. The tensor CLAP API bypasses
    # package-level int16 quantization and preserves the waveform's shape.
    return PreparedAudio(np.ascontiguousarray(mono, dtype=np.float32))


def _log_attack_time(audio: np.ndarray, sample_rate: int) -> float:
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    peak = float(np.max(rms, initial=0.0))
    if peak <= 1e-12:
        return float(np.log10(1e-6))
    low = np.flatnonzero(rms >= peak * 0.1)
    high = np.flatnonzero(rms >= peak * 0.9)
    if low.size == 0 or high.size == 0:
        return float(np.log10(1e-6))
    start = int(low[0])
    end_candidates = high[high >= start]
    end = int(end_candidates[0]) if end_candidates.size else start
    seconds = max((end - start) * 512 / sample_rate, 1e-6)
    return float(np.log10(seconds))


def handcrafted_features(audio: np.ndarray, sample_rate: int = CLAP_SAMPLE_RATE) -> np.ndarray:
    # Reuse one STFT for all spectral statistics. For H/P energy we only need
    # the separated spectrogram energy, so avoid two unnecessary inverse STFTs.
    magnitude = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    harmonic, percussive = librosa.decompose.hpss(magnitude)
    harmonic_energy = float(np.mean(np.square(harmonic, dtype=np.float64)))
    percussive_energy = float(np.mean(np.square(percussive, dtype=np.float64)))
    ratio = harmonic_energy / max(percussive_energy, 1e-12)
    result = np.asarray(
        [
            np.mean(centroid),
            np.std(centroid),
            np.mean(rolloff),
            np.std(rolloff),
            np.mean(flatness),
            np.std(flatness),
            np.mean(zcr),
            _log_attack_time(audio, sample_rate),
            ratio,
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("Handcrafted feature calculation produced non-finite values")
    return result


class ClapEmbedder:
    """One frozen LAION-CLAP HTSAT-base model reused across all batches."""

    def __init__(self, platform_env: PlatformEnv, checkpoint: Path = CLAP_CHECKPOINT) -> None:
        configure_model_cache()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing {checkpoint}; run scripts/cache_clap.py before analysis"
            )
        import laion_clap

        if bool(getattr(__import__("sys"), "frozen", False)):
            # LAION-CLAP wraps every HTSAT construction error in one generic
            # RuntimeError, which makes a missing frozen dependency impossible
            # to diagnose. Use its pinned constructor directly in the bundle.
            import importlib

            # laion_clap.hook imports ``clap_module`` as a top-level package,
            # while normal callers can also import ``laion_clap.clap_module``.
            # Patch both module identities because each owns a separate CLAP
            # class and constructor global in a frozen process.
            for prefix in ("clap_module", "laion_clap.clap_module"):
                clap_model = importlib.import_module(f"{prefix}.model")
                htsat = importlib.import_module(f"{prefix}.htsat")

                def create_frozen_htsat(
                    audio_cfg,  # type: ignore[no-untyped-def]
                    enable_fusion: bool = False,
                    fusion_type: str = "None",
                    *,
                    _transformer=htsat.HTSAT_Swin_Transformer,
                ):
                    model_shapes = {
                        "tiny": (96, [2, 2, 6, 2]),
                        "base": (128, [2, 2, 12, 2]),
                        "large": (256, [2, 2, 12, 2]),
                    }
                    embed_dim, depths = model_shapes[audio_cfg.model_name]
                    return _transformer(
                        spec_size=256,
                        patch_size=4,
                        patch_stride=(4, 4),
                        num_classes=audio_cfg.class_num,
                        embed_dim=embed_dim,
                        depths=depths,
                        num_heads=[4, 8, 16, 32],
                        window_size=8,
                        config=audio_cfg,
                        enable_fusion=enable_fusion,
                        fusion_type=fusion_type,
                    )

                clap_model.create_htsat_model = create_frozen_htsat

        self.device = platform_env.compute_backend
        self.model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel="HTSAT-base",
            device=self.device,
        )
        self.model.load_ckpt(str(checkpoint), verbose=False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def embed(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        tensors = [torch.from_numpy(np.asarray(waveform, dtype=np.float32)) for waveform in waveforms]
        with torch.inference_mode():
            result = self.model.get_audio_embedding_from_data(tensors, use_tensor=True)
        array = result.detach().cpu().numpy().astype(np.float32, copy=False)
        if array.shape != (len(waveforms), CLAP_DIMENSIONS):
            raise RuntimeError(f"Unexpected CLAP embedding shape {array.shape}")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)
