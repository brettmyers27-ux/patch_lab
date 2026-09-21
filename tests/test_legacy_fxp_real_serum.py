"""Real headless evidence for the legacy-.fxp conclusion.

Unit tests cannot settle "can Serum 2 open a .fxp?". These run against the real
installed plug-ins, headlessly (DawDreamer / pedalboard, no DAW and no GUI), and
skip cleanly on a machine that lacks the relevant Serum or preset library.

The trap these guard against is the one the investigation actually hit: every
naive attempt *appears* to succeed. ``load_preset`` returns True and no exception
is raised, but the plug-in keeps its init patch. So a load is only accepted here
when the rendered **audio** changes and two different presets render differently.

A second, subtler trap is also encoded: ``changed_parameter_count`` is useless
for Serum 2, which keeps its patch in an opaque state chunk and exposes no
per-parameter values. A verification that relied on parameter deltas would
reject every valid Serum 2 load.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.platform_env import ENV


#: Deselected from the default unit run: instantiating a real Serum plug-in in
#: this process makes a later PySide6 QEventLoop.exec() segfault. Run with
#: `pytest tests/ -m realserum`.
pytestmark = pytest.mark.realserum

SERUM2_LIBRARY = Path("/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets")
#: How many renders are averaged per state. Serum's oscillators and LFOs are
#: free-running and are not reset between renders, so a single render of an
#: unchanged patch varies measurably (observed cosine 0.97-1.00 on this machine).
#: Averaging suppresses that so a real difference is unambiguous.
RENDERS_PER_STATE = 3

#: Tolerance around the measured same-state envelope. A similarity at or above
#: ``envelope - SEPARATION_MARGIN`` is treated as "the state did not change".
#: Calibrated per run rather than hardcoded, because the envelope is a property
#: of the installed Serum build.
SEPARATION_MARGIN = 0.02


def _candidate(synth: str, plugin_format: str):
    return next(
        (item for item in ENV.plugins_for(synth) if item.format == plugin_format),
        None,
    )


def _mono(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        if values.shape[0] != 2 and values.shape[1] == 2:
            values = values.T
        values = np.mean(values, axis=0)
    return np.ascontiguousarray(values, dtype=np.float32)


def _averaged_signature(engine, processor, *, renders: int = RENDERS_PER_STATE) -> np.ndarray:
    """Mean signature over several renders of the current plug-in state.

    Serum does not reset its free-running modulators between renders, so one
    render is a noisy sample of the patch. The mean is stable enough to compare.
    """

    from core.plugin_host import render_dawdreamer_note

    signatures = [
        _signature(render_dawdreamer_note(engine, processor, duration=2.0))
        for _ in range(renders)
    ]
    return np.mean(np.asarray(signatures, dtype=np.float32), axis=0)


def _same_state_envelope(engine, processor, *, samples: int = 3) -> float:
    """Lowest similarity between two averaged renders of an UNCHANGED state.

    This is the noise floor. Any comparison at or above it is indistinguishable
    from "the plug-in state did not change at all".
    """

    signatures = [_averaged_signature(engine, processor) for _ in range(samples)]
    return min(
        _cosine(signatures[i], signatures[j])
        for i in range(len(signatures))
        for j in range(i + 1, len(signatures))
    )


def _signature(audio: np.ndarray, bands: int = 24) -> np.ndarray:
    """Coarse log-magnitude band signature: enough to tell timbres apart."""

    values = _mono(audio)
    if values.size < 2048:
        return np.zeros(bands, dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(values[: 1 << 16]))
    edges = np.geomspace(20, len(spectrum) - 1, bands + 1).astype(int)
    return np.asarray(
        [
            float(np.log1p(spectrum[edges[i] : max(edges[i] + 1, edges[i + 1])].mean()))
            for i in range(bands)
        ],
        dtype=np.float32,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation of two band signatures.

    Mean-centring matters: a plain cosine over log-magnitudes is dominated by
    the shared DC offset and compresses every comparison into 0.95-1.00, which
    is too coarse to separate "a different patch" from "the same patch rendered
    twice". Centred, the separation on this machine is unambiguous -- an
    unchanged state sits near 0.92-1.00 while a genuinely different patch falls
    to 0.34-0.85.
    """

    left = np.asarray(left, dtype=np.float64) - float(np.mean(left))
    right = np.asarray(right, dtype=np.float64) - float(np.mean(right))
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm < 1e-9 or right_norm < 1e-9:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _diverse_presets(pattern: str, count: int) -> list[Path]:
    """One preset from each of several different packs, not N from one pack."""

    if not SERUM2_LIBRARY.is_dir():
        return []
    by_pack: dict[str, list[Path]] = {}
    for path in sorted(SERUM2_LIBRARY.rglob(pattern)):
        by_pack.setdefault(path.parent.name, []).append(path)
    picked: list[Path] = []
    for pack in sorted(by_pack):
        picked.append(by_pack[pack][0])
        if len(picked) >= count:
            break
    return picked


requires_serum2 = pytest.mark.skipif(
    _candidate("serum2", "VST3") is None,
    reason="real Serum 2 VST3 is not installed on this machine",
)
requires_serum1 = pytest.mark.skipif(
    _candidate("serum1", "VST2") is None,
    reason="real Serum 1 VST2 is not installed on this machine",
)
requires_library = pytest.mark.skipif(
    not SERUM2_LIBRARY.is_dir(),
    reason="no real Serum 2 factory preset library on this machine",
)


# ---------------------------------------------------------------------------
# The measured library shape
# ---------------------------------------------------------------------------


@requires_library
def test_real_serum2_library_is_mostly_legacy_fxp() -> None:
    """The population that made a Serum-2-only machine look broken."""

    fxp = list(SERUM2_LIBRARY.rglob("*.fxp"))
    native = list(SERUM2_LIBRARY.rglob("*.SerumPreset")) + list(
        SERUM2_LIBRARY.rglob("*.serumpreset")
    )
    assert fxp, "expected legacy .fxp content inside the Serum 2 library"
    assert native, "expected native .serumpreset content too"
    # Not an incidental handful: legacy content is the majority here.
    assert len(fxp) > len(native), (
        f"{len(fxp)} .fxp vs {len(native)} .serumpreset -- the whole point is that "
        "legacy content dominates a Serum 2 factory library"
    )


# ---------------------------------------------------------------------------
# Positive control: the harness can see a REAL Serum 2 load
# ---------------------------------------------------------------------------


@requires_serum2
@requires_library
def test_native_serumpreset_loads_and_changes_the_audio() -> None:
    """Control. Without this, a negative .fxp result would prove nothing."""

    from pedalboard import load_plugin

    from core.plugin_host import make_dawdreamer_processor
    from core.serum2_preset import parse_serum2_preset
    from core.serum2_state_reconstruct import decode_host_template, reconstruct_vstpreset

    presets = _diverse_presets("*.SerumPreset", 3) or _diverse_presets("*.serumpreset", 3)
    if len(presets) < 2:
        pytest.skip("need at least two native Serum 2 presets")

    candidate = _candidate("serum2", "VST3")
    live = load_plugin(str(candidate.path), plugin_name="Serum 2")
    template = decode_host_template(bytes(live.preset_data))

    engine, processor = make_dawdreamer_processor(candidate)
    envelope = _same_state_envelope(engine, processor)
    init_signature = _averaged_signature(engine, processor)

    signatures: list[np.ndarray] = []
    for preset in presets:
        decoded = parse_serum2_preset(preset)
        state, _partition = reconstruct_vstpreset(decoded, template)
        target = Path(tempfile.mkdtemp()) / f"{preset.stem}.vstpreset"
        target.write_bytes(state)
        assert processor.load_vst3_preset(str(target)) is not False
        signatures.append(_averaged_signature(engine, processor))

    threshold = envelope
    # Each preset is audibly different from the init patch...
    for index, signature in enumerate(signatures):
        assert _cosine(signature, init_signature) < threshold, (
            f"{presets[index].name} rendered within the same-state envelope "
            f"({envelope:.4f}) of the init patch, i.e. it did not really load"
        )
    # ...and from each other.
    for left in range(len(signatures)):
        for right in range(left + 1, len(signatures)):
            assert _cosine(signatures[left], signatures[right]) < threshold, (
                f"{presets[left].name} and {presets[right].name} rendered the same"
            )


@requires_serum2
@requires_library
def test_parameter_deltas_are_not_a_valid_signal_for_serum2() -> None:
    """Serum 2 hides its patch in an opaque chunk, so only audio is evidence.

    Recorded as a test because a future change that verified Serum 2 loads by
    counting changed parameters would reject every valid load.
    """

    from pedalboard import load_plugin

    from core.plugin_host import (
        changed_parameter_count,
        dump_dawdreamer_parameters,
        make_dawdreamer_processor,
    )
    from core.serum2_preset import parse_serum2_preset
    from core.serum2_state_reconstruct import decode_host_template, reconstruct_vstpreset

    presets = _diverse_presets("*.SerumPreset", 1) or _diverse_presets("*.serumpreset", 1)
    if not presets:
        pytest.skip("need a native Serum 2 preset")
    candidate = _candidate("serum2", "VST3")
    live = load_plugin(str(candidate.path), plugin_name="Serum 2")
    template = decode_host_template(bytes(live.preset_data))
    engine, processor = make_dawdreamer_processor(candidate)
    initial = dump_dawdreamer_parameters(processor)
    envelope = _same_state_envelope(engine, processor)
    before = _averaged_signature(engine, processor)

    decoded = parse_serum2_preset(presets[0])
    state, _partition = reconstruct_vstpreset(decoded, template)
    target = Path(tempfile.mkdtemp()) / "control.vstpreset"
    target.write_bytes(state)
    processor.load_vst3_preset(str(target))

    changed = changed_parameter_count(initial, dump_dawdreamer_parameters(processor))
    after = _averaged_signature(engine, processor)

    assert _cosine(before, after) < envelope, (
        "the control load must change the audio"
    )
    assert changed == 0, (
        "Serum 2 exposes no per-parameter state, so a parameter-delta check would "
        f"reject this valid load (changed={changed})"
    )


# ---------------------------------------------------------------------------
# The negative result, proven properly
# ---------------------------------------------------------------------------


@requires_serum2
@requires_library
@pytest.mark.parametrize("plugin_format", ["VST3", "AU"])
def test_serum2_silently_ignores_legacy_fxp(plugin_format: str) -> None:
    """Every headless route "succeeds" while changing nothing.

    This is the evidence behind
    :data:`core.preset_identity.FXP_SERUM2_FINDING`. The assertion is framed
    against the *measured* same-state envelope, so it cannot be fooled by
    Serum's ordinary render-to-render variance in either direction.
    """

    from core.fxp import parse_fxp
    from core.plugin_host import make_dawdreamer_processor

    candidate = _candidate("serum2", plugin_format)
    if candidate is None:
        pytest.skip(f"Serum 2 {plugin_format} is not installed")
    presets = _diverse_presets("*.fxp", 3)
    if len(presets) < 2:
        pytest.skip("need at least two legacy .fxp presets")

    engine, processor = make_dawdreamer_processor(candidate)
    envelope = _same_state_envelope(engine, processor)
    init_signature = _averaged_signature(engine, processor)
    # Below this, the state genuinely changed. The .fxp attempts must all land
    # at or above it -- i.e. no more different from init than init is from
    # itself across two renders.
    threshold = envelope - SEPARATION_MARGIN

    def load_state(blob: bytes) -> None:
        with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as handle:
            handle.write(blob)
            handle.flush()
            name = handle.name
        processor.load_state(name)

    loaded: list[tuple[str, str, float]] = []
    refused: list[str] = []
    for preset in presets:
        for label, action in (
            ("load_preset", lambda p=preset: processor.load_preset(str(p))),
            ("load_state_payload", lambda p=preset: load_state(parse_fxp(p).payload)),
            ("load_state_whole", lambda p=preset: load_state(p.read_bytes())),
        ):
            try:
                action()
            except Exception as exc:  # an explicit refusal is also a valid answer
                refused.append(f"{preset.name}/{label}: {type(exc).__name__}")
                continue
            similarity = _cosine(_averaged_signature(engine, processor), init_signature)
            if similarity < threshold:
                loaded.append((preset.name, label, similarity))

    assert not loaded, (
        "Serum 2 unexpectedly loaded a legacy .fxp (audio moved outside the "
        f"same-state envelope {envelope:.4f}): {loaded}. If this is genuine, "
        "core.preset_identity.COMPATIBLE_RENDERERS and FXP_SERUM2_FINDING must "
        "be updated to allow serum2 for .fxp."
    )


@requires_serum2
@requires_library
def test_fxp_is_not_a_serum2_preset_container() -> None:
    """Structural incompatibility, independent of any host behaviour."""

    from core.serum2_preset import parse_serum2_preset

    presets = _diverse_presets("*.fxp", 3)
    if not presets:
        pytest.skip("need a legacy .fxp preset")
    for preset in presets:
        with pytest.raises(Exception) as caught:
            parse_serum2_preset(preset)
        assert "magic" in str(caught.value).casefold()


# ---------------------------------------------------------------------------
# Serum 1 really can render these presets
# ---------------------------------------------------------------------------


@requires_serum1
@requires_library
def test_serum1_genuinely_renders_the_legacy_fxp_presets() -> None:
    """The flip side: routing .fxp to Serum 1 is correct, not a cop-out.

    Confirms the compatible-renderer set for .fxp really is ("serum1",) rather
    than "nothing can open these".
    """

    from core.plugin_host import (
        audio_levels,
        changed_parameter_count,
        dump_dawdreamer_parameters,
        make_dawdreamer_processor,
        render_dawdreamer_note,
    )

    presets = _diverse_presets("*.fxp", 3)
    if len(presets) < 2:
        pytest.skip("need at least two legacy .fxp presets")

    candidate = _candidate("serum1", "VST2")
    engine, processor = make_dawdreamer_processor(candidate)
    initial = dump_dawdreamer_parameters(processor)
    envelope = _same_state_envelope(engine, processor)
    init_signature = _averaged_signature(engine, processor)
    threshold = envelope

    signatures: list[np.ndarray] = []
    for preset in presets:
        assert processor.load_preset(str(preset)) is not False, preset.name
        changed = changed_parameter_count(initial, dump_dawdreamer_parameters(processor))
        _peak, rms = audio_levels(render_dawdreamer_note(engine, processor, duration=2.0))
        # Serum 1 *does* expose parameters, so both signals are available here.
        assert changed >= 5, f"{preset.name}: only {changed} parameters changed"
        assert rms > -60.0, f"{preset.name}: rendered silence at {rms:.1f} dBFS"
        signatures.append(_averaged_signature(engine, processor))

    for index, signature in enumerate(signatures):
        assert _cosine(signature, init_signature) < threshold, presets[index].name
    for left in range(len(signatures)):
        for right in range(left + 1, len(signatures)):
            assert _cosine(signatures[left], signatures[right]) < threshold, (
                f"{presets[left].name} and {presets[right].name} rendered the same"
            )


@requires_serum1
@requires_library
def test_legacy_fxp_loading_is_deterministic_and_does_not_leak() -> None:
    """Repeatability and isolation, on the renderer that actually works.

    Loading A then B then A again must give A's sound back both times, which is
    what proves no previous preset state leaks into the next one. Judged against
    the measured same-state envelope, because Serum's free-running modulators
    mean two renders of one patch are never bit-identical.
    """

    from core.plugin_host import make_dawdreamer_processor

    presets = _diverse_presets("*.fxp", 2)
    if len(presets) < 2:
        pytest.skip("need two legacy .fxp presets")

    candidate = _candidate("serum1", "VST2")
    engine, processor = make_dawdreamer_processor(candidate)
    envelope = _same_state_envelope(engine, processor)

    def render(preset: Path) -> np.ndarray:
        processor.load_preset(str(preset))
        return _averaged_signature(engine, processor)

    first_a = render(presets[0])
    first_b = render(presets[1])
    second_a = render(presets[0])
    second_b = render(presets[1])

    separation = _cosine(first_a, first_b)
    assert separation < envelope, (
        f"the two presets must sound different (similarity {separation:.4f} vs "
        f"envelope {envelope:.4f})"
    )
    # Reproducible within the same envelope that an unchanged patch occupies,
    # and markedly closer to itself than to the other preset.
    for label, before, after in (("A", first_a, second_a), ("B", first_b, second_b)):
        repeat = _cosine(before, after)
        assert repeat > separation + SEPARATION_MARGIN, (
            f"preset {label} did not reproduce: repeat similarity {repeat:.4f} is "
            f"not clearly above cross-preset separation {separation:.4f}"
        )
