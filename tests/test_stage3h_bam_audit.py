from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.matcher import _DeterministicRenderPool
from scripts.stage3h_bam_audit import canonical_audio_sha256, repeat_summary


def test_canonical_audio_hash_ignores_wav_container_metadata(tmp_path: Path) -> None:
    audio = np.linspace(-0.5, 0.5, 256, dtype=np.float32)
    left = tmp_path / "left.wav"
    right = tmp_path / "right.wav"
    sf.write(left, audio, 48_000, subtype="FLOAT")
    sf.write(right, audio, 48_000, subtype="FLOAT")

    assert canonical_audio_sha256(left) == canonical_audio_sha256(right)


def test_repeat_summary_detects_winner_and_audio_instability() -> None:
    rows = [
        {
            "clap_similarity": 0.8,
            "base_preset_id": 1,
            "origin": "cma",
            "candidate_state_sha256": "state-a",
            "decoded_audio_sha256": "audio-a",
            "wav_file_sha256": "wav-a",
        },
        {
            "clap_similarity": 0.7,
            "base_preset_id": 2,
            "origin": "mutation-1-1",
            "candidate_state_sha256": "state-b",
            "decoded_audio_sha256": "audio-b",
            "wav_file_sha256": "wav-b",
        },
    ]

    summary = repeat_summary(rows)

    assert summary["score_span"] == pytest.approx(0.1)
    assert summary["unique_states"] == 2
    assert summary["unique_decoded_audio"] == 2
    assert summary["unique_winners"] == 2


class _ImmediateJob:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def get(self) -> list[int]:
        return self._values


class _FakePool:
    def __init__(self, worker: int) -> None:
        self.worker = worker
        self.seen: list[list[int]] = []

    def map_async(self, function, values: list[int]) -> _ImmediateJob:
        self.seen.append(values)
        return _ImmediateJob([function((self.worker, value)) for value in values])

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


class _FakeContext:
    def __init__(self) -> None:
        self.pools: list[_FakePool] = []

    def Pool(self, _processes, *, initializer, initargs) -> _FakePool:
        pool = _FakePool(len(self.pools))
        self.pools.append(pool)
        return pool


def test_deterministic_pool_pins_positions_to_workers() -> None:
    context = _FakeContext()
    pool = _DeterministicRenderPool(context, 3, lambda: None, ())

    first = pool.map(lambda item: item, list(range(8)))
    second = pool.map(lambda item: item, list(range(8, 16)))

    assert first == [
        (0, 0),
        (1, 1),
        (2, 2),
        (0, 3),
        (1, 4),
        (2, 5),
        (0, 6),
        (1, 7),
    ]
    assert second == [
        (0, 8),
        (1, 9),
        (2, 10),
        (0, 11),
        (1, 12),
        (2, 13),
        (0, 14),
        (1, 15),
    ]
