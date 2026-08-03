import numpy as np

from scripts.benchmark_serum2_structural_mutations import _distances, _get, _set


def test_nested_structural_path_round_trip() -> None:
    graph = {"FXRack0": {"FX": [{"type": 0}]}}
    path = ("FXRack0", "FX", 0, "type")
    assert _get(graph, path) == 0
    _set(graph, path, 6)
    assert _get(graph, path) == 6


def test_identical_audio_does_not_count_as_audible_mutation() -> None:
    audio = np.sin(np.linspace(0, 20, 44_100)).astype(np.float32)
    result = _distances(audio, audio.copy(), "wavetable")
    assert result["audibly_changed"] is False
    assert result["directionally_correct"] is False
