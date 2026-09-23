"""Real Serum 2: the preset PatchLab writes is built on the preset the user heard.

Run separately (``pytest tests/ -m realserum``).  For catalog presets chosen to
cover every collision shape of the old numeric lookup, this writes the preset
through the production writer, reconstructs it headlessly into the installed
Serum 2 VST3, renders it, and compares it with the shipped render state of the
same catalog id -- which is exactly what the Match audition played.

Where the old lookup would have picked a different Serum 2 preset, that preset
is rendered too, to prove the comparison can tell them apart.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.realserum

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "data" / "runtime" / "v1-legacy-stock-clap"


def _mono(audio) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[0] != 2 and audio.shape[1] == 2:
        audio = audio.T
    return np.mean(audio, axis=0)


def _signature(audio, bands: int = 32) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(_mono(audio)[: 1 << 16]))
    edges = np.geomspace(20, len(spectrum) - 1, bands + 1).astype(int)
    return np.asarray(
        [np.log1p(spectrum[edges[i] : max(edges[i] + 1, edges[i + 1])].mean()) for i in range(bands)],
        dtype=np.float32,
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.fixture(scope="module")
def serum2():
    if not (RUNTIME / "dist" / "factory_bundle.sqlite").is_file():
        pytest.skip("shipped runtime family not present")
    from core.renderer_selection import RendererUnavailableError, open_renderer

    try:
        engine, processor, _selection = open_renderer("serum2")
    except RendererUnavailableError as exc:
        pytest.skip(f"Serum 2 not installed: {exc}")
    return engine, processor


def _render(serum2, state_path: Path) -> np.ndarray:
    from core.plugin_host import render_dawdreamer_note

    engine, processor = serum2
    assert processor.load_vst3_preset(str(state_path)) is not False
    if hasattr(processor, "clear_midi"):
        processor.clear_midi()
    return render_dawdreamer_note(engine, processor, duration=2.0)


def _headless_state(written: Path, catalog_id: int, scratch: Path):
    from core.serum2_preset import parse_serum2_preset
    from core.serum2_state_reconstruct import decode_host_template, reconstruct_vstpreset
    from core.synthesis_assets import resolve_synthesis_assets

    template = resolve_synthesis_assets().render_states / f"{catalog_id}.vstpreset"
    state, partition = reconstruct_vstpreset(
        parse_serum2_preset(written), decode_host_template(template.read_bytes())
    )
    path = scratch / f"{written.stem}.vstpreset"
    path.write_bytes(state)
    return path, partition.coverage, template


# (catalog id, what the old numeric lookup would have done)
CASES = [
    (1, "bundle id 1 is a Serum 1 preset"),
    (171, "bundle id 171 is a Serum 1 preset (the beta tester's failure)"),
    (458, "bundle id 458 is a DIFFERENT Serum 2 preset (silent wrong preset)"),
    (4831, "bundle has no id 4831 (the beta tester's other failure)"),
]


@pytest.mark.parametrize("catalog_id,old_behaviour", CASES)
def test_the_written_preset_sounds_like_the_auditioned_preset(serum2, tmp_path, catalog_id, old_behaviour) -> None:
    from core.base_preset_identity import resolve_serum2_base
    from core.factory_bundle import FactoryBundle
    from core.plugin_host import audio_levels as levels
    from core.serum2_preset import parse_serum2_preset
    from core.serum2_preset_writer import write_serum2_preset

    base = resolve_serum2_base(catalog_id)
    written = tmp_path / f"{catalog_id}.SerumPreset"
    result = write_serum2_preset(
        written, base_preset_id=catalog_id, vector=base.base_vector,
        mask=np.ones_like(base.base_vector, dtype=bool), meaningfully_modified=False,
    )
    assert result.base_identity == base.identity
    assert parse_serum2_preset(written).data == base.settings

    state, coverage, auditioned = _headless_state(written, catalog_id, tmp_path)
    assert coverage >= 0.85
    audio = _render(serum2, state)
    assert levels(audio)[1] > -60.0, "the reloaded preset makes sound"
    written_sig = _signature(audio)
    auditioned_sig = _signature(_render(serum2, auditioned))
    assert _cosine(written_sig, auditioned_sig) >= 0.98, old_behaviour

    bundle = FactoryBundle(RUNTIME / "dist" / "factory_bundle.sqlite")
    try:
        colliding = bundle.preset_by_id(catalog_id)
    except KeyError:
        colliding = None
    if colliding is not None:
        assert bundle.settings(colliding.id)[0] != base.settings, "never the numerically colliding preset"
    if colliding is not None and colliding.synth == "serum2":
        with sqlite3.connect(RUNTIME / "models" / "patchlab-synthesis-catalog.sqlite") as connection:
            (wrong_catalog_id,) = connection.execute(
                "SELECT id FROM presets WHERE content_hash=?", (colliding.content_hash,)
            ).fetchone()
        wrong_sig = _signature(
            _render(serum2, auditioned.parent / f"{wrong_catalog_id}.vstpreset")
        )
        right, wrong = _cosine(written_sig, auditioned_sig), _cosine(written_sig, wrong_sig)
        assert right - wrong >= 0.05, f"right {right:.4f} vs wrong {wrong:.4f}"


@pytest.mark.parametrize("catalog_id", [458, 4831])
def test_a_modified_preset_reloads_intact_on_the_right_base(serum2, tmp_path, catalog_id) -> None:
    from core.base_preset_identity import resolve_serum2_base
    from core.plugin_host import audio_levels as levels
    from core.serum2_preset import parse_serum2_preset
    from core.serum2_preset_writer import write_serum2_preset
    import json

    from core.synthesis_assets import resolve_synthesis_assets

    base = resolve_serum2_base(catalog_id)
    assets = resolve_synthesis_assets()
    schema = json.loads(assets.serum2_schema.read_text(encoding="utf-8"))
    with np.load(assets.serum2_targets) as stored:
        row = int(np.flatnonzero(stored["preset_ids"] == catalog_id)[0])
        present = np.asarray(stored["masks"][row], dtype=bool)
    # Nudge continuous parameters this preset actually has, as the optimiser would.
    vector = base.base_vector.copy()
    mask = present.copy()
    changed = 0
    for field in schema["fields"]:
        index = int(field["index"])
        if field["encoding"] == "one_hot" or not present[index]:
            continue
        if not str(field["name"]).split(".")[-1].startswith("kParam"):
            continue
        vector[index] = float(np.clip(vector[index] + (0.04 if vector[index] < 0.5 else -0.04), 0.0, 1.0))
        changed += 1
        if changed == 12:
            break
    assert changed > 0
    written = tmp_path / f"modified-{catalog_id}.SerumPreset"
    result = write_serum2_preset(
        written, base_preset_id=catalog_id, vector=vector, mask=mask, meaningfully_modified=True,
    )
    assert result.mode == "optimized-overlay" and result.applied_fields > 0
    assert result.base_identity.content_hash == base.identity.content_hash
    graph = parse_serum2_preset(written).data
    assert set(graph) == set(base.settings), "same module topology as the identified base"

    state, coverage, auditioned = _headless_state(written, catalog_id, tmp_path)
    assert coverage >= 0.85
    audio = _render(serum2, state)
    assert levels(audio)[1] > -60.0
    assert _cosine(_signature(audio), _signature(_render(serum2, auditioned))) >= 0.9
