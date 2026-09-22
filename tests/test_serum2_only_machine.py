"""The reporting beta tester's Mac: Serum 2 present, Serum 1 absent everywhere.

Every failure they hit was the same defect in different clothes -- a code path
that assumed a renderer instead of asking PatchLab's verified selector for one:

* audition/preview died with ``KeyError: 'hosts'`` because the preview worker
  demanded BOTH generations, failed to initialise without Serum 1, and then
  indexed the host table that initialisation never created;
* Export Preset and Load in Serum died with ``StopIteration`` because the export
  verifier opened a Serum 1 VST2 host up front, for a Serum 2 export;
* library rendering could queue work for a generation this machine cannot host.

Matrix: A preview · B/C/D export · E library render · F Serum-1 absence ·
G no renderer at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core import matcher as matcher_module
from core import render as render_module
from core.db import Database
from core.platform_env import PluginCandidate
from core.plugin_host import ParameterValue
from core.preset_scan import sha1_file
from core.renderer_selection import (
    RendererUnavailableError,
    open_renderer,
    preflight_renderers,
    renderer_candidate,
    require_renderer,
)
from tests.test_renderer_routing import build_env

SERUM2_VST3 = ("serum2", "VST3", "system/VST3/Serum2.vst3")
SERUM2_AU = ("serum2", "AU", "system/Components/Serum2.component")
SERUM1_VST2 = ("serum1", "VST2", "system/VST/Serum.vst")


@pytest.fixture()
def serum2_only(tmp_path: Path):
    """Exactly the tester's machine."""

    return build_env(tmp_path, {SERUM2_VST3, SERUM2_AU})


@pytest.fixture()
def no_synths(tmp_path: Path):
    return build_env(tmp_path, set())


# ---------------------------------------------------------------------------
# The shared rule: one authoritative answer per generation
# ---------------------------------------------------------------------------


def test_serum2_resolves_and_serum1_raises_an_explained_error(serum2_only) -> None:
    selection = require_renderer("serum2", env=serum2_only, context="test")
    assert selection.available and selection.selected.format == "VST3"
    assert renderer_candidate(selection).path.name == "Serum2.vst3"

    with pytest.raises(RendererUnavailableError) as raised:
        require_renderer("serum1", env=serum2_only, context="auditioning a preset")
    assert "Serum 1" in raised.value.user_message
    assert "StopIteration" not in repr(raised.value)


def test_no_user_facing_path_still_hand_rolls_renderer_discovery() -> None:
    """Regression: a bare next() over a hardcoded plug-in format is the defect."""

    import re

    offenders = []
    for path in (
        Path("core/render.py"), Path("core/preset_export.py"), Path("core/match_workflow.py"),
        Path("core/preset_scan.py"), Path("scripts/render_factory_preview.py"),
        Path("scripts/render_recommendation_preview.py"),
    ):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"next\(\s*\n?\s*item\s*\n?\s*for item in [^)]*plugin", source):
            offenders.append(f"{path}:{source[:match.start()].count(chr(10)) + 1}")
    assert not offenders, "these still search for a plug-in themselves: " + ", ".join(offenders)


# --- A. preview / audition ---------------------------------------------------


def test_A_preview_worker_hosts_only_the_generation_it_renders(serum2_only, tmp_path: Path, monkeypatch) -> None:
    """The KeyError: 'hosts' bug: a Serum 2 audition must not need Serum 1."""

    monkeypatch.setattr("core.renderer_selection.ENV", serum2_only)
    opened: list[str] = []

    def fake_processor(candidate: PluginCandidate):
        opened.append(f"{candidate.synth}/{candidate.format}")
        return MagicMock(name="engine"), MagicMock(name="processor")

    monkeypatch.setattr("core.plugin_host.make_dawdreamer_processor", fake_processor)
    monkeypatch.setattr(matcher_module, "_serum1_targets", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(matcher_module, "_serum2_targets", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr(matcher_module, "resolve_synthesis_assets", lambda: MagicMock(
        serum2_schema=MagicMock(read_text=lambda **_k: "{}"), library_db=tmp_path / "x.db"))

    matcher_module._RENDER.clear()
    matcher_module._init_render_worker(str(tmp_path), required_synths=("serum2",))
    assert opened == ["serum2/VST3"], "only the requested generation is hosted"
    assert "init_failure" not in matcher_module._RENDER
    assert set(matcher_module._RENDER["hosts"]) == {"serum2"}
    matcher_module._RENDER.clear()


def test_A_a_worker_that_never_initialised_explains_itself(monkeypatch, tmp_path: Path, serum2_only) -> None:
    """Never 'KeyError: hosts' again: the recorded reason is what surfaces."""

    monkeypatch.setattr("core.renderer_selection.ENV", serum2_only)
    matcher_module._RENDER.clear()
    matcher_module._init_render_worker(str(tmp_path), required_synths=("serum1",))
    assert "hosts" not in matcher_module._RENDER and "init_failure" in matcher_module._RENDER

    with pytest.raises(matcher_module.RenderWorkerNotReady) as raised:
        matcher_module._require_hosts()
    assert "Serum 1" in raised.value.user_message
    payload = matcher_module._render_candidate(
        (MagicMock(synth="serum1"), 60, 1.0)
    )
    assert payload[2] and payload[2].startswith(matcher_module.WORKER_INIT_FAILURE_PREFIX)
    matcher_module._RENDER.clear()


def test_A_preview_script_asks_for_only_the_candidate_generation() -> None:
    source = Path("scripts/render_recommendation_preview.py").read_text(encoding="utf-8")
    assert "required_synths=(candidate.synth,)" in source


# --- B/C/D. export, closest-match export, Load in Serum ----------------------


def test_B_export_verifier_opens_only_the_exported_generation(serum2_only, monkeypatch) -> None:
    """The StopIteration bug: verifying a Serum 2 export must not need Serum 1."""

    from core import preset_export

    monkeypatch.setattr("core.renderer_selection.ENV", serum2_only)
    monkeypatch.setattr(preset_export, "ClapEmbedder", lambda *_a, **_k: MagicMock())
    opened: list[str] = []
    monkeypatch.setattr(
        "core.plugin_host.make_dawdreamer_processor",
        lambda c: (opened.append(f"{c.synth}/{c.format}"), (MagicMock(), MagicMock()))[1],
    )
    verifier = preset_export.PresetExportVerifier()
    assert opened == [], "constructing the verifier opens nothing"
    verifier.host("serum2")
    verifier.host("serum2")
    assert opened == ["serum2/VST3"], "opened once, for the exported generation only"
    with pytest.raises(RendererUnavailableError):
        verifier.host("serum1")
    verifier.close()


def test_D_load_in_serum_and_export_share_one_implementation() -> None:
    """Load in Serum is the same verified export, written to the Serum folder."""

    source = Path("app/ui.py").read_text(encoding="utf-8")
    body = source.split("def load_in_serum(")[1].split("def _start_preset_export(")[0]
    assert "_start_preset_export(" in body
    assert "self.export_runner.start" not in body, "it must not re-implement export"


# --- E. Render Sound Library on this machine ---------------------------------


def _library(tmp_path: Path) -> tuple[Database, dict[str, int]]:
    database = Database(tmp_path / "library.db")
    ids = {}
    for name, synth in (("LegacyLead", "serum1"), ("ModernPad", "serum2")):
        path = tmp_path / f"{name}{'.fxp' if synth == 'serum1' else '.SerumPreset'}"
        path.write_bytes(b"CcnK" + name.encode())
        preset_id, _ = database.insert_preset(path=path, name=name, synth=synth, content_hash=sha1_file(path))
        database.replace_params(preset_id, [ParameterValue(0, "Master", 0.5, "50%")], "test")
        ids[synth] = preset_id
    return database, ids


def test_E_library_render_skips_generations_this_machine_cannot_host(serum2_only, tmp_path: Path, monkeypatch) -> None:
    database, ids = _library(tmp_path)
    monkeypatch.setattr("core.renderer_selection.ENV", serum2_only)
    started: list[tuple] = []
    monkeypatch.setattr(render_module.mp, "get_context", lambda _k: (_ for _ in ()).throw(
        AssertionError("no worker pool should start for an unhostable generation")))

    # Only the Serum 1 preset is unrenderable here, and the Serum 2 one has all
    # its notes already, so no pool is needed and none may be created.
    for note in render_module.MIDI_NOTES:
        database.upsert_renders([render_module.RenderRecord(
            preset_id=ids["serum2"], midi_note=note, wav_path=str(tmp_path / f"{note}.wav"),
            peak_dbfs=-6.0, rms_dbfs=-18.0, duration_s=1.0)])
    summary = render_module.render_library(
        db_path=database.path, audio_root=tmp_path / "audio", state_dir=tmp_path / "states",
        processes=1, log=lambda _m: None,
    )
    assert summary.skipped_unrenderable_presets == 1
    assert summary.unrenderable_generations == "serum1"
    assert summary.selected_presets == 1, "only the Serum 2 preset was selected"


def test_E_no_serum1_worker_is_created_for_a_serum2_only_library(serum2_only, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("core.renderer_selection.ENV", serum2_only)
    opened: list[str] = []
    monkeypatch.setattr(
        "core.plugin_host.make_dawdreamer_processor",
        lambda c: (opened.append(str(c.synth)), (MagicMock(), MagicMock()))[1],
    )
    render_module._HOSTS.clear()
    engine, processor = render_module._worker_host("serum2")
    assert opened == ["serum2"]
    with pytest.raises(RendererUnavailableError):
        render_module._worker_host("serum1")
    assert opened == ["serum2"], "no Serum 1 host was ever constructed"
    render_module._HOSTS.clear()


# --- F. Serum 1 absence breaks nothing ---------------------------------------


def test_F_every_generation_aware_path_reports_serum1_absence_the_same_way(serum2_only) -> None:
    preflight = preflight_renderers(("serum1", "serum2"), env=serum2_only)
    assert preflight.supported == ("serum2",) and preflight.unsupported == ("serum1",)
    assert preflight.any_supported and not preflight.fully_supported
    message = preflight.user_message()
    assert "Serum 1" in message and "Serum 2" in message and "StopIteration" not in message


def test_F_opening_serum2_never_touches_serum1(serum2_only, monkeypatch) -> None:
    monkeypatch.setattr("core.renderer_selection.ENV", serum2_only)
    seen: list[str] = []
    monkeypatch.setattr(
        "core.plugin_host.make_dawdreamer_processor",
        lambda c: (seen.append(str(c.synth)), (MagicMock(), MagicMock()))[1],
    )
    _engine, _processor, selection = open_renderer("serum2", env=serum2_only, context="test")
    assert seen == ["serum2"] and selection.selected.format == "VST3"


# --- G. no renderer at all ----------------------------------------------------


def test_G_no_synth_installed_fails_cleanly_before_any_worker(no_synths, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("core.renderer_selection.ENV", no_synths)
    for synth in ("serum1", "serum2"):
        with pytest.raises(RendererUnavailableError) as raised:
            open_renderer(synth, env=no_synths, context="rendering")
        assert raised.value.user_message and "StopIteration" not in raised.value.user_message

    preflight = preflight_renderers(("serum1", "serum2"), env=no_synths)
    assert not preflight.any_supported
    assert preflight.user_message()


def test_G_a_failed_worker_initialisation_never_raises_into_the_pool(no_synths, tmp_path: Path, monkeypatch) -> None:
    """A raising initializer makes multiprocessing.Pool respawn forever."""

    monkeypatch.setattr("core.renderer_selection.ENV", no_synths)
    matcher_module._RENDER.clear()
    matcher_module._init_render_worker(str(tmp_path), required_synths=("serum2",))  # must not raise
    assert matcher_module._RENDER["init_failure"]["exception_type"] == "RendererUnavailableError"
    matcher_module._RENDER.clear()


# --- structured diagnostics ---------------------------------------------------


def test_worker_failures_carry_enough_detail_to_diagnose_remotely(serum2_only, monkeypatch) -> None:
    from core.worker_failure import report_worker_failure

    monkeypatch.setattr("core.renderer_selection.ENV", serum2_only)
    try:
        require_renderer("serum1", env=serum2_only, context="auditioning")
    except RendererUnavailableError as exc:
        detail = report_worker_failure(
            exc, subsystem="preview", operation="auditioning the generated preset",
            synth="serum1", requested_renderer="serum1",
        )
    for key in ("exception_type", "message", "traceback", "code_location", "operation",
                "serum_generation", "requested_renderer", "user_message", "renderer_selection"):
        assert detail.get(key), f"missing {key}"
    assert detail["exception_type"] == "RendererUnavailableError"
    assert ".py:" in detail["code_location"]
    blob = json.dumps(detail)
    assert "CcnK" not in blob and "password" not in blob.lower()


# ---------------------------------------------------------------------------
# What the user is told, and what the support bundle carries
# ---------------------------------------------------------------------------


def test_preview_and_export_workers_print_a_sentence_not_an_exception_name() -> None:
    """The tester saw 'KeyError: hosts' and 'StopIteration'. Never again."""

    for script, prefix in (
        ("scripts/render_recommendation_preview.py", "PREVIEW_ERROR="),
        ("scripts/render_factory_preview.py", "PREVIEW_ERROR="),
        ("scripts/export_match.py", "EXPORT_ERROR="),
    ):
        source = Path(script).read_text(encoding="utf-8")
        assert 'f"' + prefix + '{type(exc).__name__}: {exc}"' not in source, (
            f"{script} still prints the raw exception class to the user"
        )
        assert "report_worker_failure(" in source, f"{script} must record structured detail"
        assert prefix + '" + detail["user_message"]' in source
        assert prefix.rstrip("=") + '_DETAIL=' in source


def test_the_recorded_detail_excludes_preset_and_audio_content(serum2_only, monkeypatch, tmp_path: Path) -> None:
    from core.worker_failure import report_worker_failure

    monkeypatch.setattr("core.renderer_selection.ENV", serum2_only)
    preset = tmp_path / "Secret.fxp"
    preset.write_bytes(b"CcnK" + b"SECRET-PRESET-BYTES" * 8)
    try:
        raise RuntimeError(f"Serum rejected {preset}")
    except RuntimeError as exc:
        detail = report_worker_failure(
            exc, subsystem="preview", operation="auditioning", synth="serum2",
            preset_path=str(preset),
        )
    blob = json.dumps(detail)
    assert "SECRET-PRESET-BYTES" not in blob and "CcnK" not in blob
    assert str(preset) in blob, "the path is useful; the bytes are not"
