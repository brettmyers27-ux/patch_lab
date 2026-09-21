"""Serum generation and plug-in format routing.

These lock in the shared defect behind both reported production bugs: a
renderer requirement that did not depend on the work actually requested, and a
plug-in format that was hardcoded rather than selected from PatchLab's verified
hierarchy.

The fixture machine mirrors the reporting user's Mac exactly -- Serum 2 present
as AU and VST3, Serum 1 absent in every format -- because that is the one
configuration in which the shipped build failed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.diagnostics import DiagnosticRecorder, set_recorder
from core.platform_env import PluginCandidate
from core.renderer_selection import (
    PREFERRED_FORMATS,
    RendererUnavailableError,
    preflight_renderers,
    renderer_inventory,
    require_renderer,
    select_renderer,
)


MACOS_CANDIDATE_SHAPE = (
    ("serum1", "AU", "user/Components/Serum.component"),
    ("serum2", "AU", "user/Components/Serum2.component"),
    ("serum1", "AU", "system/Components/Serum.component"),
    ("serum2", "AU", "system/Components/Serum2.component"),
    ("serum1", "VST2", "user/VST/Serum.vst"),
    ("serum1", "VST2", "system/VST/Serum.vst"),
    ("serum1", "VST3", "user/VST3/Serum.vst3"),
    ("serum2", "VST3", "user/VST3/Serum2.vst3"),
    ("serum1", "VST3", "system/VST3/Serum.vst3"),
    ("serum2", "VST3", "system/VST3/Serum2.vst3"),
)


@pytest.fixture(autouse=True)
def isolated_recorder(tmp_path: Path):
    """Every test records into its own directory and never touches ~/Library."""

    recorder = DiagnosticRecorder(root=tmp_path / "diagnostics", session_id="test")
    set_recorder(recorder)
    yield recorder
    recorder.close()
    set_recorder(None)


def build_env(tmp_path: Path, present: set[tuple[str, str, str]]):
    """Return a PlatformEnv whose only existing plug-ins are ``present``."""

    from core.platform_env import ENV

    candidates = []
    for synth, plugin_format, relative in MACOS_CANDIDATE_SHAPE:
        path = tmp_path / "plugins" / relative
        if (synth, plugin_format, relative) in present:
            path.mkdir(parents=True, exist_ok=True)
            binary = path / "Contents" / "MacOS"
            binary.mkdir(parents=True, exist_ok=True)
            (binary / path.stem).write_bytes(b"\xca\xfe\xba\xbe" + b"\x00" * 64)
            info = path / "Contents" / "Info.plist"
            info.write_text(
                '<?xml version="1.0"?><!DOCTYPE plist><plist version="1.0"><dict>'
                "<key>CFBundleShortVersionString</key><string>2.0.1</string>"
                "</dict></plist>",
                encoding="utf-8",
            )
        candidates.append(PluginCandidate(synth, plugin_format, path))
    return dataclasses.replace(
        ENV, branch="macos", machine="arm64", plugin_candidates=tuple(candidates)
    )


def serum2_only_env(tmp_path: Path):
    """The reporting user's machine: Serum 2 AU + VST3, no Serum 1 at all."""

    return build_env(
        tmp_path,
        {
            ("serum2", "AU", "system/Components/Serum2.component"),
            ("serum2", "VST3", "system/VST3/Serum2.vst3"),
        },
    )


def serum1_only_env(tmp_path: Path):
    return build_env(tmp_path, {("serum1", "VST2", "system/VST/Serum.vst")})


def both_env(tmp_path: Path):
    return build_env(
        tmp_path,
        {
            ("serum1", "VST2", "system/VST/Serum.vst"),
            ("serum2", "VST3", "system/VST3/Serum2.vst3"),
        },
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_serum2_selects_vst3_when_serum1_is_completely_absent(tmp_path: Path) -> None:
    """The exact configuration in which the shipped build hung."""

    env = serum2_only_env(tmp_path)
    selection = select_renderer("serum2", env=env)
    assert selection.available
    assert selection.renderer == "serum2/VST3"
    assert "highest-preference" in selection.reason


def test_absent_serum1_does_not_affect_a_serum2_selection(tmp_path: Path) -> None:
    env = serum2_only_env(tmp_path)
    assert select_renderer("serum2", env=env).available
    assert not select_renderer("serum1", env=env).available


def test_preferred_format_is_unchanged_where_it_exists(tmp_path: Path) -> None:
    """No accuracy drift: a healthy machine picks exactly what it always did.

    Serum 1 -> VST2 and Serum 2 -> VST3 remain first preference, so render
    output on any machine where the previous code worked is identical.
    """

    env = both_env(tmp_path)
    assert select_renderer("serum1", env=env).plugin_format == "VST2"
    assert select_renderer("serum2", env=env).plugin_format == "VST3"
    assert PREFERRED_FORMATS["serum1"][0] == "VST2"
    assert PREFERRED_FORMATS["serum2"][0] == "VST3"


def test_user_plugin_wins_over_system_when_both_exist(tmp_path: Path) -> None:
    """Ties must keep the shipped candidate order, not sort by path.

    ``plugin_candidates`` lists the user location before the system one for each
    format, and the previous ``next(...)`` therefore picked the user copy. Those
    can be different plug-in versions, so silently flipping to the system copy
    would change render output.
    """

    env = build_env(
        tmp_path,
        {
            ("serum1", "VST2", "user/VST/Serum.vst"),
            ("serum1", "VST2", "system/VST/Serum.vst"),
            ("serum2", "VST3", "user/VST3/Serum2.vst3"),
            ("serum2", "VST3", "system/VST3/Serum2.vst3"),
        },
    )
    assert "/user/" in select_renderer("serum1", env=env).selected.path
    assert "/user/" in select_renderer("serum2", env=env).selected.path
    # And the system copy is reported as usable-but-outranked, not missing.
    serum1 = select_renderer("serum1", env=env)
    system = next(
        item for item in serum1.candidates if "/system/" in item.path and item.format == "VST2"
    )
    assert system.exists and not system.accepted
    assert "higher preference" in system.rejection_reason


def test_serum1_falls_back_to_vst3_then_au_when_vst2_is_missing(tmp_path: Path) -> None:
    """Serum 1 installed only as VST3 must no longer be treated as absent.

    core/plugin_host.load_preset already implements S2-dawdreamer-vst3-fxp-state
    for exactly this case; only SequentialSerum1VST2 refused to use it.
    """

    vst3_only = build_env(tmp_path, {("serum1", "VST3", "system/VST3/Serum.vst3")})
    selection = select_renderer("serum1", env=vst3_only)
    assert selection.available
    assert selection.plugin_format == "VST3"
    assert "fallback" in selection.reason

    au_only = build_env(
        tmp_path / "au", {("serum1", "AU", "system/Components/Serum.component")}
    )
    assert select_renderer("serum1", env=au_only).plugin_format == "AU"


def test_selection_records_every_rejection_with_a_reason(tmp_path: Path) -> None:
    """A support bundle must explain why an available renderer was not chosen."""

    env = serum2_only_env(tmp_path)
    selection = select_renderer("serum1", env=env)
    assert not selection.available
    rejections = selection.rejections()
    assert rejections
    for item in rejections:
        assert item.rejection_reason, f"{item.label} has no rejection reason"
        assert "does not exist" in item.rejection_reason

    chosen = select_renderer("serum2", env=env)
    au = next(item for item in chosen.candidates if item.format == "AU" and item.exists)
    # AU exists and is usable; it must say it lost on preference, not pretend to
    # be missing.
    assert "higher preference" in au.rejection_reason


def test_require_renderer_raises_a_structured_error_not_stopiteration(
    tmp_path: Path,
) -> None:
    """The shipped code raised a bare StopIteration from ``next()``.

    StopIteration carries no message, no candidate list and -- inside a
    multiprocessing pool initializer -- no usable diagnosis at all.
    """

    env = serum2_only_env(tmp_path)
    with pytest.raises(RendererUnavailableError) as caught:
        require_renderer("serum1", env=env, context="testing")
    error = caught.value
    assert not isinstance(error, StopIteration)
    assert "serum1" in str(error)
    assert error.selection.candidates
    assert "Serum 1" in error.user_message
    assert "install" in error.user_message.casefold()


# ---------------------------------------------------------------------------
# Preflight across generations
# ---------------------------------------------------------------------------


def test_serum2_only_library_preflight_passes_for_serum2(tmp_path: Path) -> None:
    preflight = preflight_renderers(["serum2"], env=serum2_only_env(tmp_path))
    assert preflight.fully_supported
    assert preflight.supported == ("serum2",)


def test_serum1_only_library_preflight_passes_for_serum1(tmp_path: Path) -> None:
    preflight = preflight_renderers(["serum1"], env=serum1_only_env(tmp_path))
    assert preflight.fully_supported


def test_mixed_library_reports_a_processable_subset(tmp_path: Path) -> None:
    """A Serum 2 folder full of legacy .fxp must not block its Serum 2 subset."""

    preflight = preflight_renderers(
        ["serum1", "serum2"], env=serum2_only_env(tmp_path)
    )
    assert not preflight.fully_supported
    assert preflight.any_supported
    assert preflight.supported == ("serum2",)
    assert preflight.unsupported == ("serum1",)
    message = preflight.user_message()
    assert "Serum 1" in message
    assert "waiting" in message, "unprocessable presets are waiting, not discarded"
    assert "Serum 2" in message
    assert "renderer" not in message.casefold(), "no internal jargon in user text"


def test_mixed_library_fully_supported_when_both_exist(tmp_path: Path) -> None:
    preflight = preflight_renderers(["serum1", "serum2"], env=both_env(tmp_path))
    assert preflight.fully_supported
    assert preflight.unsupported == ()


def test_preflight_is_fast_enough_to_run_before_expensive_work(tmp_path: Path) -> None:
    """The 36-minute discovery only makes sense if preflight is cheap."""

    import time

    env = serum2_only_env(tmp_path)
    started = time.monotonic()
    for _ in range(20):
        preflight_renderers(["serum1", "serum2"], env=env)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"20 preflights took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Inventory for the support bundle
# ---------------------------------------------------------------------------


def test_inventory_covers_every_candidate_with_a_verdict(tmp_path: Path) -> None:
    env = serum2_only_env(tmp_path)
    inventory = renderer_inventory(env=env)
    assert len(inventory) == len(MACOS_CANDIDATE_SHAPE)
    for entry in inventory:
        assert entry["renderer"]
        assert entry["path"]
        assert "exists" in entry
        assert "accepted" in entry
        if not entry["accepted"]:
            assert entry["rejection_reason"]
    accepted = [entry["renderer"] for entry in inventory if entry["accepted"]]
    assert accepted == ["serum2/VST3"]


def test_inventory_records_plugin_version_and_architecture(tmp_path: Path) -> None:
    env = serum2_only_env(tmp_path)
    inventory = renderer_inventory(env=env)
    present = [entry for entry in inventory if entry["exists"]]
    assert present
    for entry in present:
        assert entry["plugin_version"] == "2.0.1"
        assert entry["architecture_compatible"] is True


def test_decision_log_captures_the_reasoning(tmp_path: Path, isolated_recorder) -> None:
    """PART 9: record WHY, not just WHAT."""

    env = serum2_only_env(tmp_path)
    select_renderer("serum2", env=env, operation_id="op-1")
    events = isolated_recorder.recent_events()
    decisions = [item for item in events if item["event_type"] == "decision"]
    assert decisions
    decision = decisions[-1]
    assert decision["operation_id"] == "op-1"
    assert decision["decision_reason"]
    assert decision["fields"]["outcome"] == "serum2/VST3"
    candidates = decision["fields"]["candidates"]
    assert any(item["accepted"] for item in candidates)
    assert any(item["rejection_reason"] for item in candidates)
