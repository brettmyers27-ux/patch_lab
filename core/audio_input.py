"""User-audio decoding shared by the UI worker and its integration gates."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    path: Path
    mono: np.ndarray
    sample_rate: int
    source_duration_s: float
    start_offset_s: float
    used_duration_s: float
    rms_dbfs: float
    silent: bool
    decoder: str


def _levels(audio: np.ndarray) -> tuple[float, bool]:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0
    dbfs = float(20.0 * np.log10(max(rms, np.finfo(np.float64).tiny)))
    return dbfs, dbfs <= -60.0


def _slice(
    audio: np.ndarray, sample_rate: int, offset_s: float, maximum_s: float
) -> tuple[np.ndarray, float]:
    start = min(int(round(offset_s * sample_rate)), len(audio))
    stop = min(start + int(round(maximum_s * sample_rate)), len(audio))
    return np.ascontiguousarray(audio[start:stop], dtype=np.float32), len(audio) / sample_rate


def _soundfile(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return np.mean(audio, axis=1, dtype=np.float32), int(rate)


def _torchaudio(path: Path) -> tuple[np.ndarray, int]:
    import torch
    import torchaudio

    tensor, rate = torchaudio.load(str(path))
    mono = torch.mean(tensor.to(dtype=torch.float32), dim=0)
    return mono.cpu().numpy(), int(rate)


def _bundled_ffmpeg(
    path: Path, offset_s: float, maximum_s: float
) -> tuple[np.ndarray, int]:
    import imageio_ffmpeg

    executable = imageio_ffmpeg.get_ffmpeg_exe()
    # A closed path inside a private directory works on Windows too;
    # NamedTemporaryFile cannot be reopened by ffmpeg while held open there.
    with tempfile.TemporaryDirectory(prefix="patchlab-decode-") as directory:
        output = Path(directory) / "decoded.wav"
        command = [
            executable,
            "-v",
            "error",
            "-ss",
            f"{offset_s:.6f}",
            "-i",
            str(path),
            "-t",
            f"{maximum_s:.6f}",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-f",
            "wav",
            "-y",
            str(output),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode:
            raise RuntimeError(
                "Bundled ffmpeg could not decode the audio: "
                + completed.stderr.strip()[-1000:]
            )
        audio, rate = _soundfile(output)
    return audio, rate


def decode_audio_file(
    path: Path, *, start_offset_s: float = 0.0, maximum_s: float = 10.0
) -> DecodedAudio:
    path = Path(path).expanduser().resolve()
    if path.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(
            f"Unsupported audio type {path.suffix!r}; choose WAV, MP3, FLAC, OGG, or AIFF"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    if start_offset_s < 0 or maximum_s <= 0:
        raise ValueError("Audio offset must be non-negative and duration positive")

    errors: list[str] = []
    for label, decoder in (("soundfile", _soundfile), ("torchaudio", _torchaudio)):
        try:
            full, sample_rate = decoder(path)
            selected, source_duration = _slice(
                full, sample_rate, start_offset_s, maximum_s
            )
            if not len(selected):
                raise ValueError("The selected start offset is beyond the end of the file")
            rms_dbfs, silent = _levels(selected)
            return DecodedAudio(
                path=path,
                mono=selected,
                sample_rate=sample_rate,
                source_duration_s=source_duration,
                start_offset_s=start_offset_s,
                used_duration_s=len(selected) / sample_rate,
                rms_dbfs=rms_dbfs,
                silent=silent,
                decoder=label,
            )
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if path.suffix.casefold() == ".mp3":
        try:
            selected, sample_rate = _bundled_ffmpeg(
                path, start_offset_s, maximum_s
            )
            if not len(selected):
                raise ValueError("The selected MP3 segment is empty")
            rms_dbfs, silent = _levels(selected)
            return DecodedAudio(
                path=path,
                mono=selected,
                sample_rate=sample_rate,
                source_duration_s=start_offset_s + len(selected) / sample_rate,
                start_offset_s=start_offset_s,
                used_duration_s=len(selected) / sample_rate,
                rms_dbfs=rms_dbfs,
                silent=silent,
                decoder="bundled-ffmpeg",
            )
        except Exception as exc:
            errors.append(f"bundled-ffmpeg: {type(exc).__name__}: {exc}")
    raise RuntimeError("Audio decoding failed — " + " | ".join(errors))
