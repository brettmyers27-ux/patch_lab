from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from scripts.stage2_generate_training_data import (
    _base_definition,
    provenance_key,
    training_variants,
)


def test_training_variants_are_deterministic_and_total_thirteen() -> None:
    waveform = np.linspace(-0.5, 0.5, 48_000, dtype=np.float32)

    first = training_variants(waveform, seed=123)
    second = training_variants(waveform, seed=123)

    assert len(first) == 13
    assert [name for name, _audio in first] == [name for name, _audio in second]
    for (_name_a, audio_a), (_name_b, audio_b) in zip(first, second, strict=True):
        assert np.array_equal(audio_a, audio_b)


def test_provenance_key_changes_for_note_or_seed() -> None:
    base = provenance_key(
        content_hash="abc", synth="serum1", perturb_seed=10, midi_note=60
    )

    assert base == provenance_key(
        content_hash="abc", synth="serum1", perturb_seed=10, midi_note=60
    )
    assert base != provenance_key(
        content_hash="abc", synth="serum1", perturb_seed=11, midi_note=60
    )
    assert base != provenance_key(
        content_hash="abc", synth="serum1", perturb_seed=10, midi_note=72
    )


def test_three_note_group_shares_exact_perturbed_patch() -> None:
    store = SimpleNamespace(
        preset_row={5: 0},
        vectors=np.asarray([[0.25, 0.75]], dtype=np.float32),
        masks=np.asarray([[True, True]], dtype=bool),
        mapping=[
            {"index": 0, "stepped": False},
            {"index": 1, "stepped": False},
        ],
    )
    definitions = [
        _base_definition(
            index,
            attempt=0,
            seed=123,
            serum1=store,
            serum2=store,
            serum2_schema={"fields": []},
            hashes={5: "abc"},
            eligible_ids={1: np.asarray([5]), 2: np.asarray([5])},
            forced_synth_code=1,
        )
        for index in range(3)
    ]
    assert len({item.perturb_seed for item in definitions}) == 1
    assert len({item.midi_note for item in definitions}) == 3
    assert all(np.array_equal(definitions[0].vector, item.vector) for item in definitions[1:])
