from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.benchmark_suite import (
    BenchmarkFactoryPreset,
    _musical_gate,
    benchmark_audio_files,
    target_synth_for_name,
)
from core.factory_bundle import FactoryPreset


def test_benchmark_audio_files_keeps_real_aiff_and_wav_only(tmp_path: Path) -> None:
    audio = np.zeros(128, dtype=np.float32)
    sf.write(tmp_path / "Bass.aif", audio, 48_000, format="AIFF")
    sf.write(tmp_path / "Lead.wav", audio, 48_000, format="WAV")
    (tmp_path / "._Bass.aif").write_bytes(b"AppleDouble")
    (tmp_path / "Bass.aif.asd").write_bytes(b"Ableton")
    (tmp_path / ".DS_Store").write_bytes(b"metadata")
    nested = tmp_path / "resampled"
    nested.mkdir()
    sf.write(nested / "Bass.wav", audio, 48_000, format="WAV")

    assert [path.name for path in benchmark_audio_files(tmp_path)] == [
        "Bass.aif",
        "Lead.wav",
    ]


def test_target_synth_uses_bam_source_format_and_honors_filename_marker() -> None:
    assert target_synth_for_name("Dill Bass 1.aif") == "serum1"
    assert target_synth_for_name("Later target.wav") == "serum2"
    assert target_synth_for_name("Example [S1].wav") == "serum1"
    assert target_synth_for_name("Example Serum 2.aif") == "serum2"


def test_musical_gate_is_deterministic_and_changes_amplitude() -> None:
    audio = np.ones(48_000 * 2, dtype=np.float32)
    first = _musical_gate(audio, bpm=160.0, division=8)
    second = _musical_gate(audio, bpm=160.0, division=8)

    assert np.array_equal(first, second)
    assert first.shape == audio.shape
    assert 0.0 < float(np.mean(first)) < 1.0
    assert float(np.min(first)) == 0.0


def test_benchmark_factory_identity_keeps_bundle_and_catalog_ids_distinct() -> None:
    bundle = FactoryPreset(
        id=12,
        content_hash="abc",
        name="Bass",
        synth="serum2",
        relative_path="Bass.serumpreset",
        extension=".serumpreset",
        searchable=True,
    )

    selected = BenchmarkFactoryPreset(bundle=bundle, catalog_id=712)

    assert selected.bundle.id == 12
    assert selected.catalog_id == 712
