from __future__ import annotations

import numpy as np
import pytest

from core.layer_decomposition import (
    fit_nonnegative_magnitude_scale,
    phase_robust_residual,
    subtract_magnitudes,
)
from core.preset_stack import (
    PresetLayer,
    PresetStack,
    deterministic_mix_polish,
    mix_layers,
)


def _tone(frequency: float = 220.0, seconds: float = 0.25) -> np.ndarray:
    samples = int(round(48_000 * seconds))
    time = np.arange(samples, dtype=np.float32) / 48_000.0
    return np.sin(2.0 * np.pi * frequency * time).astype(np.float32)


def test_stft_residual_magnitude_cannot_be_negative() -> None:
    target = np.asarray([[1.0, 0.25], [0.0, 2.0]], dtype=np.float32)
    layer = np.asarray([[4.0, 0.5], [1.0, 3.0]], dtype=np.float32)

    residual = subtract_magnitudes(target, layer, 0.75)

    assert np.all(residual >= 0.0)


def test_identical_target_and_layer_produce_near_zero_residual() -> None:
    target = _tone()

    result = phase_robust_residual(target, target, 48_000)

    assert result.diagnostics.alignment_offset_ms == 0.0
    assert result.diagnostics.layer1_magnitude_scale == pytest.approx(1.0, abs=1e-6)
    assert float(np.max(np.abs(result.residual_audio))) < 1e-5
    assert result.diagnostics.residual_spectral_energy_ratio < 1e-10


def test_zero_layer_preserves_original_target_magnitude() -> None:
    target = _tone(330.0)
    zero = np.zeros_like(target)

    result = phase_robust_residual(target, zero, 48_000)

    assert result.diagnostics.layer1_magnitude_scale == 0.0
    assert np.allclose(result.residual_magnitude, result.target_magnitude)
    assert np.allclose(result.residual_audio, target, atol=2e-5)


def test_residual_generation_is_deterministic() -> None:
    target = _tone(220.0) + 0.25 * _tone(880.0)
    layer = _tone(220.0)

    first = phase_robust_residual(target, layer, 48_000)
    second = phase_robust_residual(target, layer, 48_000)

    assert first.diagnostics == second.diagnostics
    assert np.array_equal(first.residual_audio, second.residual_audio)
    assert np.array_equal(first.residual_magnitude, second.residual_magnitude)


def test_magnitude_scale_is_nonnegative_and_bounded() -> None:
    target = np.ones((2, 2), dtype=np.float32) * 10.0
    layer = np.ones((2, 2), dtype=np.float32)

    assert fit_nonnegative_magnitude_scale(target, layer) == 2.0
    assert fit_nonnegative_magnitude_scale(-target, layer) == 0.0


def test_stack_serialization_round_trip() -> None:
    layer = PresetLayer(
        synth="serum2",
        base_preset_id=42,
        state_reference="candidate.npz",
        candidate_state_sha256="state",
        decoded_audio_sha256="audio",
        gain_db=0.0,
        timing_offset_ms=0.0,
        role="dominant",
        match_score=0.8,
        midi_note=36,
        origin="cma",
    )
    stack = PresetStack(
        target_synth="serum2",
        layers=(layer,),
        combined_final_score=0.8,
        residual_energy_ratio=0.2,
        second_layer_selected=False,
        diagnostics={"selection_reason": "null candidate won"},
    )

    decoded = PresetStack.from_json(stack.to_json())

    assert decoded == stack
    assert decoded.layer_count == 1


def test_one_layer_null_candidate_reproduces_layer1_exactly() -> None:
    layer1 = _tone()

    mixed = mix_layers(
        layer1,
        _tone(440.0),
        sample_rate=48_000,
        layer1_gain_db=6.0,
        layer2_gain_db=None,
        layer2_timing_offset_ms=25.0,
    )

    assert np.array_equal(mixed, layer1)


def test_mix_polish_is_deterministic_and_can_keep_null() -> None:
    layer1 = _tone()
    layer2 = _tone(440.0)

    def score_batch(audios):
        return [float(np.dot(audio, layer1) / len(layer1)) for audio in audios]

    first = deterministic_mix_polish(
        layer1_audio=layer1,
        layer2_audio=layer2,
        sample_rate=48_000,
        initial_layer1_scale=1.0,
        initial_layer2_scale=0.25,
        one_layer_score=2.0,
        score_batch=score_batch,
    )
    second = deterministic_mix_polish(
        layer1_audio=layer1,
        layer2_audio=layer2,
        sample_rate=48_000,
        initial_layer1_scale=1.0,
        initial_layer2_scale=0.25,
        one_layer_score=2.0,
        score_batch=score_batch,
    )

    assert first.score == second.score == 2.0
    assert first.second_layer_selected is False
    assert first.layer2_gain_db is None
    assert np.array_equal(first.audio, layer1)
    assert first.search_trace == second.search_trace
    assert first.gain_combination_count <= 100


def test_disabled_layer2_path_does_not_change_single_preset_output() -> None:
    layer1 = np.linspace(-0.25, 0.25, 2048, dtype=np.float32)
    layer2 = np.ones_like(layer1)

    output = mix_layers(
        layer1,
        layer2,
        sample_rate=48_000,
        layer2_gain_db=float("-inf"),
    )

    assert output.dtype == np.float32
    assert np.array_equal(output, layer1)
