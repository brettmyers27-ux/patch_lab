"""Bug B regression: Render Sound Library must not demand an irrelevant Serum.

Reported behaviour (PatchLab 1.5.3, commit 7223b349):

* the linked folder was ``/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets``;
* PatchLab cataloged 5,634 files over roughly 36 minutes;
* then ``core/preset_scan.py`` line 81 raised
  ``RuntimeError("Verified Serum 1 VST2 binary is unavailable.")``;
* and the UI showed the Link step running again instead of reporting that
  Render had failed.

The cause is that Serum 2's factory library is mostly legacy ``.fxp`` content --
on a representative install, 4,826 ``.fxp`` against 1,857 ``.serumpreset`` --
which ``synth_for`` classifies as ``serum1``, and the ingestor for that
generation was hardcoded to require a Serum 1 **VST2** binary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.diagnostics import DiagnosticRecorder, set_recorder
from core.preset_scan import (
    classify_generations,
    classify_preset,
    discover_presets,
    log_classification_summary,
    synth_for,
)
from core.renderer_selection import RendererUnavailableError, preflight_renderers
from tests.test_renderer_routing import both_env, serum1_only_env, serum2_only_env


@pytest.fixture(autouse=True)
def isolated_recorder(tmp_path: Path):
    recorder = DiagnosticRecorder(root=tmp_path / "diagnostics", session_id="test")
    set_recorder(recorder)
    yield recorder
    recorder.close()
    set_recorder(None)


def make_library(
    root: Path, *, fxp: int = 0, serumpreset: int = 0, noise: int = 0
) -> Path:
    """Build a preset folder shaped like a real Serum library."""

    root.mkdir(parents=True, exist_ok=True)
    # Contents must differ per file: PatchLab deduplicates by content hash, so
    # identical bytes would collapse the whole library into one catalog row.
    for index in range(fxp):
        (root / f"legacy-{index}.fxp").write_bytes(
            b"CcnK" + f"legacy-{index}".encode() + bytes(64)
        )
    for index in range(serumpreset):
        # Real files use mixed case; classification must be case-insensitive.
        (root / f"modern-{index}.SerumPreset").write_bytes(
            b"SERUM" + f"modern-{index}".encode() + bytes(64)
        )
    for index in range(noise):
        (root / f"readme-{index}.txt").write_text("not a preset", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_serum2_factory_folder_is_mostly_fxp(tmp_path: Path) -> None:
    """The shape that caused the bug, asserted explicitly."""

    root = make_library(tmp_path / "Serum 2 Presets", fxp=48, serumpreset=18, noise=5)
    paths = discover_presets(root)
    counts = classify_generations(paths)
    assert counts == {"serum1": 48, "serum2": 18}
    assert len(paths) == 66, "non-preset files must not be cataloged"


def test_classification_is_case_insensitive(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    root.mkdir()
    for name in ("a.FXP", "b.fxp", "c.SerumPreset", "d.serumpreset"):
        (root / name).write_bytes(b"x")
    assert synth_for(root / "a.FXP") == "serum1"
    assert synth_for(root / "c.SerumPreset") == "serum2"


def test_classification_explains_itself(tmp_path: Path) -> None:
    """PART 9: why a preset was classified serum1 vs serum2."""

    fxp = classify_preset(tmp_path / "bass.fxp")
    assert fxp.generation == "serum1"
    assert "Serum 1's patch format" in fxp.reason
    # The reason must say that a Serum 2 folder legitimately contains these,
    # because that is the fact the bug turned on.
    assert "Serum 2 factory libraries also ship .fxp" in fxp.reason
    assert fxp.evidence == "extension=.fxp"

    modern = classify_preset(tmp_path / "lead.serumpreset")
    assert modern.generation == "serum2"
    assert "Serum 2's patch format" in modern.reason

    other = classify_preset(tmp_path / "notes.txt")
    assert other.generation is None


def test_classification_is_logged_once_not_per_file(
    tmp_path: Path, isolated_recorder
) -> None:
    """A 5,000-preset library must not produce 5,000 log lines."""

    root = make_library(tmp_path / "lib", fxp=300, serumpreset=120)
    counts = log_classification_summary(
        discover_presets(root), operation_id="op-1", linked_folder=root
    )
    assert counts == {"serum1": 300, "serum2": 120}
    summaries = [
        item
        for item in isolated_recorder.recent_events()
        if item["event_type"] == "classification_summary"
    ]
    assert len(summaries) == 1, "classification must be aggregated, not per file"
    event = summaries[0]
    assert event["fields"]["counts"] == {"serum1": 300, "serum2": 120}
    assert event["fields"]["total_files"] == 420
    assert event["decision_reason"]
    assert "serum1" in event["fields"]["reasons"]


# ---------------------------------------------------------------------------
# 1-4. Per-generation routing
# ---------------------------------------------------------------------------


def test_serum2_only_library_with_no_serum1_vst2(tmp_path: Path) -> None:
    """Item 1: the reported configuration must be processable."""

    root = make_library(tmp_path / "lib", serumpreset=20)
    counts = classify_generations(discover_presets(root))
    preflight = preflight_renderers(sorted(counts), env=serum2_only_env(tmp_path))
    assert preflight.fully_supported
    assert preflight.supported == ("serum2",)


def test_serum1_only_library(tmp_path: Path) -> None:
    root = make_library(tmp_path / "lib", fxp=20)
    counts = classify_generations(discover_presets(root))
    preflight = preflight_renderers(sorted(counts), env=serum1_only_env(tmp_path))
    assert preflight.fully_supported


def test_mixed_library_routes_each_generation(tmp_path: Path) -> None:
    root = make_library(tmp_path / "lib", fxp=30, serumpreset=10)
    counts = classify_generations(discover_presets(root))
    preflight = preflight_renderers(sorted(counts), env=both_env(tmp_path))
    assert preflight.fully_supported
    assert preflight.selection_for("serum1").plugin_format == "VST2"
    assert preflight.selection_for("serum2").plugin_format == "VST3"


def test_mixed_library_without_serum1_processes_the_serum2_subset(
    tmp_path: Path,
) -> None:
    """The heart of Bug B: do not block valid Serum 2 work because Serum 1 is absent."""

    root = make_library(tmp_path / "lib", fxp=40, serumpreset=12)
    counts = classify_generations(discover_presets(root))
    preflight = preflight_renderers(sorted(counts), env=serum2_only_env(tmp_path))
    assert not preflight.fully_supported
    assert preflight.any_supported
    assert preflight.supported == ("serum2",)


# ---------------------------------------------------------------------------
# 5. Deterministic prerequisite failure happens BEFORE the expensive work
# ---------------------------------------------------------------------------


def test_capability_preflight_runs_before_any_file_is_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_recorder
) -> None:
    """Item 5: the renderer requirement must not wait for a 36-minute scan.

    The requirement is now *reported* early rather than aborting early, because
    every discovered preset must still be catalogued so a later Serum install
    needs no rediscovery. What must not happen is discovering the requirement
    only after the whole library has been hashed -- so this asserts the coverage
    decision is recorded before the first hash.
    """

    import core.local_library as library_module
    import core.renderer_selection as selection_module
    import core.preset_scan as scan_module

    root = make_library(tmp_path / "lib", fxp=25)
    env = serum2_only_env(tmp_path)
    monkeypatch.setattr(selection_module, "ENV", env)

    order: list[str] = []

    def counting_sha1(path: Path, chunk_size: int = 0) -> str:
        order.append("hash")
        return f"h{len(order)}"

    monkeypatch.setattr(library_module, "sha1_file", counting_sha1)
    monkeypatch.setattr(scan_module, "sha1_file", counting_sha1)

    library_module.process_linked_folder(
        root,
        db_path=tmp_path / "library.db",
        audio_root=tmp_path / "audio",
        state_dir=tmp_path / "states",
        env=env,
        compact_mode=True,
        log=lambda _message: order.append("log"),
    )
    events = isolated_recorder.recent_events()
    coverage = [item for item in events if item["fields"].get("decision") == "library_coverage"]
    assert coverage, "the coverage decision must be recorded"
    assert order and order[0] == "log", (
        "coverage must be reported before the first file is hashed; "
        f"first action was {order[0]}"
    )


def test_unprocessable_library_is_catalogued_not_aborted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Part 7: discovered-but-unprocessable presets must be preserved.

    The previous behaviour raised, which lost the catalog entirely and meant a
    later Serum 1 install had to rediscover thousands of files.
    """

    import core.local_library as library_module
    import core.renderer_selection as selection_module
    from core.db import Database
    from core.preset_identity import PENDING_UNSUPPORTED_LEGACY_FORMAT

    root = make_library(tmp_path / "lib", fxp=12)
    env = serum2_only_env(tmp_path)
    monkeypatch.setattr(selection_module, "ENV", env)
    db_path = tmp_path / "library.db"

    summary = library_module.process_linked_folder(
        root,
        db_path=db_path,
        audio_root=tmp_path / "audio",
        state_dir=tmp_path / "states",
        env=env,
        compact_mode=True,
        log=lambda _m: None,
    )
    assert summary.found == 12
    assert summary.params_dumped == 0
    database = Database(db_path)
    coverage = database.library_coverage()
    assert coverage["discovered"] == 12
    assert coverage["pending"] == 12
    # A .fxp inside a user folder reports serum1_not_installed; these fixtures
    # live outside a factory root, so that is the expected code.
    assert set(coverage["pending_by_reason"]) <= {
        "serum1_not_installed",
        PENDING_UNSUPPORTED_LEGACY_FORMAT,
    }
    assert all(row.file_format == "fxp" for row in database.pending_presets())
    assert all(
        row.compatible_renderers == "serum1" for row in database.pending_presets()
    )


def test_partially_supported_library_processes_the_supported_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mixed library must catalog everything and process what it can."""

    import core.local_library as library_module
    import core.renderer_selection as selection_module
    from core.db import Database

    root = make_library(tmp_path / "lib", fxp=30, serumpreset=7)
    env = serum2_only_env(tmp_path)
    monkeypatch.setattr(selection_module, "ENV", env)

    logged: list[str] = []
    # Stop at the Serum 2 host so this exercises routing, not plug-in hosting.
    monkeypatch.setattr(
        library_module,
        "decode_host_template",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop-after-catalog")),
    )
    with pytest.raises(Exception) as caught:
        library_module.process_linked_folder(
            root,
            db_path=tmp_path / "library.db",
            audio_root=tmp_path / "audio",
            state_dir=tmp_path / "states",
            env=env,
            compact_mode=True,
            log=logged.append,
        )
    assert not isinstance(caught.value, RendererUnavailableError), (
        "the Serum 2 subset must be attempted, not blocked by absent Serum 1"
    )
    # Everything was catalogued, including the 30 unprocessable legacy presets.
    coverage = Database(tmp_path / "library.db").library_coverage()
    assert coverage["discovered"] == 37
    assert coverage["pending"] == 30
    assert any("pending because" in line for line in logged), logged


def test_pending_subset_is_reported_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_recorder
) -> None:
    """Item 12: diagnostics must say which renderer and WHY."""

    import core.local_library as library_module
    import core.renderer_selection as selection_module

    root = make_library(tmp_path / "lib", fxp=9, serumpreset=3)
    env = serum2_only_env(tmp_path)
    monkeypatch.setattr(selection_module, "ENV", env)
    monkeypatch.setattr(
        library_module,
        "decode_host_template",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    with pytest.raises(Exception):
        library_module.process_linked_folder(
            root,
            db_path=tmp_path / "library.db",
            audio_root=tmp_path / "audio",
            state_dir=tmp_path / "states",
            env=env,
            compact_mode=True,
            log=lambda _m: None,
        )
    events = isolated_recorder.recent_events()
    pending = [item for item in events if item["event_type"] == "presets_pending"]
    assert pending
    assert sum(pending[0]["fields"]["pending_by_reason"].values()) == 9
    assert "without rediscovery" in pending[0]["decision_reason"]

    coverage = [
        item for item in events if item["fields"].get("decision") == "library_coverage"
    ]
    assert coverage
    fields = coverage[0]["fields"]
    assert fields["by_file_format"] == {"fxp": 9, "serumpreset": 3}
    assert fields["pending"] == 9
    assert fields["processable"] == 3
    assert "only Serum 1 can render headlessly" in coverage[0]["decision_reason"]


# ---------------------------------------------------------------------------
# Ingestor no longer demands VST2 specifically
# ---------------------------------------------------------------------------


def test_serum1_ingestor_accepts_vst3_when_vst2_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact line that produced the reported RuntimeError."""

    import core.preset_scan as scan_module
    from tests.test_renderer_routing import build_env

    env = build_env(tmp_path, {("serum1", "VST3", "system/VST3/Serum.vst3")})
    opened: list[str] = []

    def fake_processor(candidate):
        opened.append(f"{candidate.synth}/{candidate.format}")
        return ("engine", _StubProcessor())

    monkeypatch.setattr(scan_module, "make_dawdreamer_processor", fake_processor)
    monkeypatch.setattr(scan_module, "dump_dawdreamer_parameters", lambda _p: [])

    ingestor = scan_module.SequentialSerum1Ingestor(env)
    assert opened == ["serum1/VST3"]
    assert ingestor.candidate.format == "VST3"
    assert ingestor.strategy_label == "VST3/S2-dawdreamer-vst3-fxp-state"


def test_serum1_ingestor_raises_structured_error_when_serum1_is_absent(
    tmp_path: Path,
) -> None:
    import core.preset_scan as scan_module

    with pytest.raises(RendererUnavailableError) as caught:
        scan_module.SequentialSerum1Ingestor(serum2_only_env(tmp_path))
    # The old message was a bare RuntimeError naming only VST2.
    assert "VST2 binary is unavailable" not in str(caught.value)
    assert caught.value.selection.candidates


def test_legacy_alias_still_resolves() -> None:
    from core.preset_scan import SequentialSerum1Ingestor, SequentialSerum1VST2

    assert SequentialSerum1VST2 is SequentialSerum1Ingestor


class _StubProcessor:
    def get_parameters_description(self):
        return []

    def get_parameter(self, _index):
        return 0.0


# ---------------------------------------------------------------------------
# 10-11. Completed work and compact cleanup safety
# ---------------------------------------------------------------------------


def test_successfully_learned_presets_survive_a_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 10: a failure must preserve the work already durably committed."""

    import core.local_library as library_module
    import core.renderer_selection as selection_module
    from core.db import Database

    root = make_library(tmp_path / "lib", serumpreset=6)
    env = serum2_only_env(tmp_path)
    monkeypatch.setattr(selection_module, "ENV", env)
    db_path = tmp_path / "library.db"

    # Pre-populate the catalog, then make the Serum 2 host fail.
    monkeypatch.setattr(
        library_module,
        "decode_host_template",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("host unavailable")),
    )
    with pytest.raises(Exception):
        library_module.process_linked_folder(
            root,
            db_path=db_path,
            audio_root=tmp_path / "audio",
            state_dir=tmp_path / "states",
            env=env,
            compact_mode=True,
            log=lambda _m: None,
        )
    # The rows written before the failure are still there: nothing was rolled
    # back or deleted by the failure path.
    with Database(db_path).connect() as connection:
        count = int(connection.execute("SELECT COUNT(*) FROM presets").fetchone()[0])
    assert count == 6


def test_compact_pipeline_order_is_preserved() -> None:
    """PART 27: learn and durably commit BEFORE deleting regenerable audio.

    Asserted against the source because the ordering is the safety property:
    fingerprints are written by fingerprint_batch and only then does
    compact_render_library remove WAVs.
    """

    import ast
    import inspect

    import core.local_library as library_module

    # Scope the check to the per-batch loop inside process_linked_folder: the
    # module has other compact_render_library() calls (the standalone compaction
    # entry point and the final sweep) that are not part of this ordering.
    source = inspect.getsource(library_module._process_linked_folder)
    tree = ast.parse(inspect.cleandoc(source))
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "id", "") == "compact_render_library"
            for inner in ast.walk(node)
        )
    )
    # ast.walk is breadth-first, so order by source position explicitly.
    calls = sorted(
        (
            (node.lineno, node.col_offset, getattr(node.func, "id", ""))
            for node in ast.walk(loop)
            if isinstance(node, ast.Call)
        )
    )
    names = [name for _line, _col, name in calls]
    assert "fingerprint_batch" in names
    assert "compact_render_library" in names
    assert names.index("fingerprint_batch") < names.index("compact_render_library"), (
        "temporary renders must never be deleted before the learned state is "
        f"committed; call order was {names}"
    )
