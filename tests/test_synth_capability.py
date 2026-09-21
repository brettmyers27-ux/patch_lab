"""The installation / library behaviour matrix.

Covers every sensible combination of Serum 1 / Serum 2 installation against
every combination of preset ownership, plus the dynamic-availability and
capability-gating behaviour built on top.

The governing distinction, asserted repeatedly below because it is the one that
would break users if it regressed:

* **synth capability** decides whether PatchLab can CREATE a requested output;
* **library processing** decides which existing presets participate in search.

They are never allowed to gate each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.capability_ux import (
    MATERIAL_COUNT_DELTA,
    acknowledge,
    evaluate_output_target,
    load_notice_state,
    missing_synth_notice,
    record_pending_exclusion,
    synth_available_notice,
)
from core.db import Database
from core.diagnostics import DiagnosticRecorder, set_recorder
from core.preset_identity import (
    COMPATIBLE_RENDERERS,
    FXP_SERUM2_FINDING,
    PENDING_SERUM1_NOT_INSTALLED,
    PENDING_SERUM2_NOT_INSTALLED,
    PENDING_UNSUPPORTED_LEGACY_FORMAT,
    identify_preset,
    summarise_identities,
)
from core.synth_capability import (
    GENERATIONS,
    candidate_signature,
    capability_for,
    diff_capabilities,
    refresh_capabilities,
)
from tests.test_renderer_routing import (
    MACOS_CANDIDATE_SHAPE,
    both_env,
    build_env,
    serum1_only_env,
    serum2_only_env,
)


def neither_env(tmp_path: Path):
    return build_env(tmp_path, set())


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Recorder, capability cache and notice state all isolated per test."""

    import core.capability_ux as ux
    import core.platform_env as platform_env

    recorder = DiagnosticRecorder(root=tmp_path / "diagnostics", session_id="test")
    set_recorder(recorder)
    monkeypatch.setattr(ux, "_state_path", lambda env=None: tmp_path / "notices.json")
    yield recorder
    recorder.close()
    set_recorder(None)


def make_library(root: Path, *, fxp: int = 0, serumpreset: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(fxp):
        (root / f"legacy-{index}.fxp").write_bytes(b"CcnK" + f"{index}".encode() + bytes(32))
    for index in range(serumpreset):
        (root / f"modern-{index}.SerumPreset").write_bytes(
            b"SERUM" + f"{index}".encode() + bytes(32)
        )
    return root


class _StubIngestor:
    """Stands in for a real Serum 1 host so these tests need no plug-in."""

    def __init__(self, env, *, operation_id: str = "") -> None:
        from core.renderer_selection import require_renderer

        # Still goes through real renderer selection, so an absent Serum 1
        # raises exactly as production would.
        self.selection = require_renderer(
            "serum1", env=env, operation_id=operation_id, context="stub ingest"
        )
        self.candidate = type("C", (), {"format": self.selection.plugin_format})()

    strategy_label = "stub/serum1"

    def ingest(self, path: Path):
        from core.plugin_host import ParameterValue

        return [ParameterValue(0, "stub", 0.5, "stub")], -18.0, self.strategy_label


def install_host_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace real plug-in hosting with stubs, keeping all routing real."""

    import pedalboard

    import core.local_library as library_module

    monkeypatch.setattr(library_module, "SequentialSerum1Ingestor", _StubIngestor)
    monkeypatch.setattr(
        pedalboard,
        "load_plugin",
        lambda *_a, **_k: type("P", (), {"preset_data": b"stub-template"})(),
    )
    monkeypatch.setattr(
        library_module, "decode_host_template", lambda _data: "stub-template"
    )

    def stub_store_serum2(database, preset_id, path, template, state_dir):
        from core.plugin_host import ParameterValue

        Path(state_dir).mkdir(parents=True, exist_ok=True)
        (Path(state_dir) / f"{preset_id}.vstpreset").write_bytes(b"stub")
        database.replace_params(
            preset_id,
            [ParameterValue(0, "Serum 2 complete settings", 1.0, "available")],
            "stub/serum2",
        )

    monkeypatch.setattr(library_module, "_store_serum2", stub_store_serum2)


def catalogue(
    tmp_path: Path,
    env,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fxp: int = 0,
    serumpreset: int = 0,
    db_name: str = "library.db",
) -> tuple[Database, Path]:
    """Catalogue a library through the real pipeline with stubbed hosting.

    Routing, identity, capability and pending logic are all the production code
    paths; only the plug-in hosting and rendering are stubbed.
    """

    import core.local_library as library_module

    install_host_stubs(monkeypatch)
    root = make_library(tmp_path / "linked", fxp=fxp, serumpreset=serumpreset)
    db_path = tmp_path / db_name
    # Rendering needs a real plug-in; stop after ingest so these tests stay fast.
    monkeypatch.setattr(
        library_module,
        "render_library",
        lambda **_kwargs: (_ for _ in ()).throw(_StopAfterIngest()),
    )
    try:
        library_module.process_linked_folder(
            root,
            db_path=db_path,
            audio_root=tmp_path / "audio",
            state_dir=tmp_path / "states",
            env=env,
            compact_mode=True,
            log=lambda _m: None,
        )
    except _StopAfterIngest:
        pass
    return Database(db_path), root


class _StopAfterIngest(Exception):
    """Sentinel: cataloguing and ingest are done, rendering is out of scope."""


# ---------------------------------------------------------------------------
# Capability model
# ---------------------------------------------------------------------------


def test_capability_distinguishes_the_four_states(tmp_path: Path) -> None:
    """not_installed / not_hostable / renderer_invalid / ready must differ."""

    absent = capability_for("serum2", env=neither_env(tmp_path))
    assert absent.status == "not_installed"
    assert not absent.available
    assert not absent.installed
    assert "no Serum 2 plug-in exists" in absent.reason

    present = capability_for("serum2", env=serum2_only_env(tmp_path))
    assert present.status == "present_unvalidated"
    assert present.available, "tier-1 presence is enough to start work"
    assert present.installed
    assert present.preferred_renderer == "serum2/VST3"

    # A binary that exists but is not in a hostable format.
    import dataclasses

    from core.platform_env import PluginCandidate

    clap = tmp_path / "clap" / "Serum2.clap"
    clap.mkdir(parents=True)
    unhostable = dataclasses.replace(
        serum2_only_env(tmp_path / "u"),
        plugin_candidates=(PluginCandidate("serum2", "CLAP", clap, False),),
    )
    state = capability_for("serum2", env=unhostable)
    assert state.status == "not_hostable"
    assert state.installed
    assert not state.available


def test_deep_validation_failure_is_reported_as_renderer_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.synth_capability as capability_module

    monkeypatch.setattr(
        capability_module,
        "_validate_renderer",
        lambda selection: (False, "plug-in crashed during init render"),
    )
    state = capability_for("serum2", env=serum2_only_env(tmp_path), deep=True)
    assert state.status == "renderer_invalid"
    assert not state.available
    assert "crashed" in state.validation_detail
    assert "couldn't start" in state.user_message()


def test_signature_detects_installation_without_opening_a_plugin(
    tmp_path: Path,
) -> None:
    """Tier 1 must notice a new plug-in from stat calls alone."""

    before = candidate_signature("serum2", env=neither_env(tmp_path))
    after = candidate_signature("serum2", env=serum2_only_env(tmp_path))
    assert before != after
    # And it is stable when nothing changed.
    assert after == candidate_signature("serum2", env=serum2_only_env(tmp_path))


def test_quick_capability_check_is_cheap(tmp_path: Path) -> None:
    """Safe to run on every click: no plug-in is opened."""

    import time

    env = serum2_only_env(tmp_path)
    started = time.perf_counter()
    for _ in range(25):
        refresh_capabilities(env=env)
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, f"25 quick refreshes took {elapsed:.2f}s"


def test_capability_refresh_is_recorded_with_reasons(tmp_path: Path, isolated) -> None:
    refresh_capabilities(env=serum2_only_env(tmp_path), reason="unit test")
    events = isolated.recent_events()
    kinds = {item["event_type"] for item in events}
    assert "capability_refresh_started" in kinds
    assert "capability_refresh_completed" in kinds
    decision = next(
        item for item in events if item["fields"].get("decision") == "capability_refresh"
    )
    assert "serum1" in decision["fields"]
    assert "serum2" in decision["fields"]
    assert decision["fields"]["serum2"]["status"] == "present_unvalidated"
    assert decision["decision_reason"]


def test_capability_change_detection(tmp_path: Path, isolated) -> None:
    before = refresh_capabilities(env=serum2_only_env(tmp_path))
    after = refresh_capabilities(env=both_env(tmp_path / "both"))
    change = diff_capabilities(before, after)
    assert change.became_available == ("serum1",)
    assert change.became_unavailable == ()
    assert change.changed
    assert any(
        item["event_type"] == "capability_changed" for item in isolated.recent_events()
    )


def test_no_change_reports_nothing(tmp_path: Path) -> None:
    env = serum2_only_env(tmp_path)
    change = diff_capabilities(refresh_capabilities(env=env), refresh_capabilities(env=env))
    assert not change.changed


# ---------------------------------------------------------------------------
# PART 1: the six separated concepts
# ---------------------------------------------------------------------------


def test_identity_separates_format_origin_provenance_and_compatibility(
    tmp_path: Path,
) -> None:
    """`.fxp` is a format, not a verdict about which synth the user needs."""

    legacy = identify_preset(tmp_path / "bass.fxp")
    assert legacy.file_format == "fxp"
    assert legacy.origin_generation == "serum1"
    assert legacy.provenance == "user_folder"
    assert legacy.compatible_renderers == ("serum1",)
    # The compatibility claim must carry the measured evidence for it.
    assert FXP_SERUM2_FINDING in legacy.compatibility_reason
    assert "GUI browser feature" in legacy.compatibility_reason

    modern = identify_preset(tmp_path / "lead.serumpreset")
    assert modern.file_format == "serumpreset"
    assert modern.compatible_renderers == ("serum2",)


def test_compatibility_table_is_data_not_extension_logic() -> None:
    assert COMPATIBLE_RENDERERS["fxp"] == ("serum1",)
    assert COMPATIBLE_RENDERERS["serumpreset"] == ("serum2",)


def test_legacy_fxp_in_a_serum2_library_is_named_precisely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The population that made a Serum-2-only machine look broken."""

    import core.preset_identity as identity_module

    monkeypatch.setattr(
        identity_module, "provenance_for", lambda path, env=None: "serum2_factory"
    )
    identity = identify_preset(tmp_path / "factory.fxp")
    assert identity.is_legacy_in_serum2_library
    serum2_only = {
        "serum1": capability_for("serum1", env=neither_env(tmp_path)),
        "serum2": capability_for("serum2", env=serum2_only_env(tmp_path)),
    }
    # "Install Serum 1" is surprising advice for a Serum 2 owner, so the reason
    # code is the specific legacy-format one.
    assert identity.pending_reason(serum2_only) == PENDING_UNSUPPORTED_LEGACY_FORMAT


def test_summary_counts_answer_why_only_n_of_m(tmp_path: Path) -> None:
    identities = [identify_preset(tmp_path / f"a{i}.fxp") for i in range(7)] + [
        identify_preset(tmp_path / f"b{i}.serumpreset") for i in range(3)
    ]
    capabilities = {
        "serum1": capability_for("serum1", env=neither_env(tmp_path)),
        "serum2": capability_for("serum2", env=serum2_only_env(tmp_path)),
    }
    summary = summarise_identities(identities, capabilities)
    assert summary["discovered"] == 10
    assert summary["by_file_format"] == {"fxp": 7, "serumpreset": 3}
    assert summary["processable"] == 3
    assert summary["pending"] == 7
    assert summary["pending_by_reason"] == {PENDING_SERUM1_NOT_INSTALLED: 7}
    assert summary["processable_by_renderer"] == {"serum2": 3}


# ---------------------------------------------------------------------------
# PART 6: the installation matrix (cases A-D)
# ---------------------------------------------------------------------------


def test_case_a_serum1_only_with_serum1_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """1. Serum 1 installed, Serum 2 absent, Serum 1 presets only."""

    env = serum1_only_env(tmp_path)
    database, _root = catalogue(tmp_path, env, monkeypatch, fxp=6)
    coverage = database.library_coverage()
    assert coverage["discovered"] == 6
    assert coverage["pending"] == 0, "Serum 1 presets must be processable"


def test_case_a_serum1_only_with_mixed_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """2. Mixed library on a Serum-1-only machine: never fail the library."""

    env = serum1_only_env(tmp_path)
    database, _root = catalogue(tmp_path, env, monkeypatch, fxp=5, serumpreset=4)
    coverage = database.library_coverage()
    assert coverage["discovered"] == 9
    assert coverage["pending"] == 4
    assert coverage["pending_by_reason"] == {PENDING_SERUM2_NOT_INSTALLED: 4}
    # The Serum 1 subset is not blocked.
    assert len(database.presets_needing_generation("serum2")) == 4
    assert len(database.presets_needing_generation("serum1")) == 0


def test_case_b_serum2_only_with_native_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """3. Serum 2 installed, Serum 1 absent, native Serum 2 presets."""

    database, _root = catalogue(tmp_path, serum2_only_env(tmp_path), monkeypatch, serumpreset=5)
    coverage = database.library_coverage()
    assert coverage["discovered"] == 5
    assert coverage["pending"] == 0


def test_case_b_serum2_only_with_legacy_fxp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """4. Legacy .fxp inside a Serum 2 library on a Serum-2-only machine.

    The exact reported configuration. It must catalogue, not crash, not lie.
    """

    database, _root = catalogue(tmp_path, serum2_only_env(tmp_path), monkeypatch, fxp=8, serumpreset=2)
    coverage = database.library_coverage()
    assert coverage["discovered"] == 10
    assert coverage["pending"] == 8
    assert coverage["learned"] == 0
    # Nothing was marked processed that was not processed.
    pending = database.pending_presets()
    assert {row.file_format for row in pending} == {"fxp"}
    assert all(row.status == "scanned" for row in pending)


def test_case_c_both_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """5. Both synths installed: everything valid is processable."""

    database, _root = catalogue(tmp_path, both_env(tmp_path), monkeypatch, fxp=4, serumpreset=4)
    coverage = database.library_coverage()
    assert coverage["discovered"] == 8
    assert coverage["pending"] == 0


def test_case_c_does_not_convert_serum1_presets_to_serum2(tmp_path: Path) -> None:
    """A .fxp must route to Serum 1 when Serum 1 can already render it."""

    env = both_env(tmp_path)
    identity = identify_preset(tmp_path / "x.fxp")
    capabilities = {name: capability_for(name, env=env) for name in GENERATIONS}
    assert identity.pending_reason(capabilities) is None
    assert identity.renderable_with(["serum1", "serum2"]) == ("serum1",)


def test_case_d_neither_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """6. Neither synth: catalogue, never pretend, never crash."""

    database, _root = catalogue(tmp_path, neither_env(tmp_path), monkeypatch, fxp=4, serumpreset=3)
    coverage = database.library_coverage()
    assert coverage["discovered"] == 7
    assert coverage["pending"] == 7
    assert coverage["learned"] == 0
    assert coverage["processed_params"] == 0
    assert set(coverage["pending_by_reason"]) == {
        PENDING_SERUM1_NOT_INSTALLED,
        PENDING_SERUM2_NOT_INSTALLED,
    }


def test_case_d_tells_the_user_which_synth_is_needed(tmp_path: Path) -> None:
    for generation in GENERATIONS:
        state = capability_for(generation, env=neither_env(tmp_path))
        message = state.user_message()
        assert ("Serum 2" if generation == "serum2" else "Serum 1") in message
        assert "Install" in message


# ---------------------------------------------------------------------------
# PART 8 / 10: later installation needs no rediscovery
# ---------------------------------------------------------------------------


def test_pending_presets_become_processable_without_rediscovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """9/10. Installing the missing synth must not re-walk the library."""

    import core.local_library as library_module

    # Catalogue on a Serum-2-only machine: the .fxp subset is pending.
    database, root = catalogue(tmp_path, serum2_only_env(tmp_path), monkeypatch, fxp=9, serumpreset=2)
    assert database.pending_counts()
    pending_ids = {row.id for row in database.presets_needing_generation("serum1")}
    assert len(pending_ids) == 9

    # Serum 1 appears. Re-evaluating must not touch the filesystem tree.
    walked: list[Path] = []
    monkeypatch.setattr(
        library_module,
        "discover_presets",
        lambda path: walked.append(Path(path)) or [],
    )
    hashed: list[Path] = []
    monkeypatch.setattr(
        library_module, "sha1_file", lambda p, chunk_size=0: hashed.append(p) or "x"
    )

    counts = library_module.refresh_pending_reasons(
        db_path=tmp_path / "library.db", env=both_env(tmp_path / "both")
    )
    assert walked == [], "no filesystem walk may be needed"
    assert hashed == [], "no file may be re-hashed"
    # Nothing is blocked any more; the presets are simply not processed yet, and
    # they must stay discoverable so Process Now can pick them up.
    from core.preset_identity import BLOCKED_PENDING_REASONS, PENDING_AWAITING_PROCESSING

    assert set(counts) == {PENDING_AWAITING_PROCESSING}
    assert not set(counts) & BLOCKED_PENDING_REASONS
    assert counts[PENDING_AWAITING_PROCESSING] == 9
    # And the same preset rows are still there, ready to process.
    assert {row.id for row in database.presets_with_status(("scanned",))} >= pending_ids


def test_only_the_relevant_pending_generation_is_affected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """14. Installing Serum 2 must not disturb pending Serum 1 work."""

    import core.local_library as library_module

    database, _root = catalogue(tmp_path, neither_env(tmp_path), monkeypatch, fxp=5, serumpreset=4)
    assert database.pending_counts() == {
        PENDING_SERUM1_NOT_INSTALLED: 5,
        PENDING_SERUM2_NOT_INSTALLED: 4,
    }
    # Serum 2 arrives only.
    library_module.refresh_pending_reasons(
        db_path=_db_path(tmp_path), env=serum2_only_env(tmp_path / "s2")
    )
    from core.preset_identity import PENDING_AWAITING_PROCESSING

    # The Serum 2 subset is now merely unprocessed; the Serum 1 subset is still
    # blocked on a synth that is still absent, and was not disturbed.
    assert database.pending_counts() == {
        PENDING_SERUM1_NOT_INSTALLED: 5,
        PENDING_AWAITING_PROCESSING: 4,
    }
    assert len(database.presets_needing_generation("serum1")) == 5
    assert len(database.presets_needing_generation("serum2")) == 4
    blocked = [
        row
        for row in database.pending_presets()
        if row.pending_reason == PENDING_SERUM1_NOT_INSTALLED
    ]
    assert all(row.file_format == "fxp" for row in blocked)


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "library.db"


def test_pending_query_selects_only_matching_generation(tmp_path: Path) -> None:
    database = Database(tmp_path / "q.db")
    for index, (fmt, compatible) in enumerate(
        [("fxp", ("serum1",)), ("serumpreset", ("serum2",)), ("fxp", ("serum1", "serum2"))]
    ):
        pid, _ = database.insert_preset(
            path=tmp_path / f"p{index}.{fmt}",
            name=f"p{index}",
            synth="serum1" if fmt == "fxp" else "serum2",
            content_hash=f"h{index}",
        )
        database.record_identity(
            pid, file_format=fmt, provenance="user_folder", compatible_renderers=compatible
        )
        database.set_pending_reason(pid, "serum1_not_installed")
    assert len(database.presets_needing_generation("serum1")) == 2
    assert len(database.presets_needing_generation("serum2")) == 2


def test_processing_failure_keeps_the_preset_pending(tmp_path: Path) -> None:
    """19. Pending counts must stay accurate after partial failures."""

    from core.preset_identity import PENDING_RENDER_FAILED

    database = Database(tmp_path / "f.db")
    pid, _ = database.insert_preset(
        path=tmp_path / "a.fxp", name="a", synth="serum1", content_hash="h"
    )
    database.record_identity(
        pid, file_format="fxp", provenance="user_folder", compatible_renderers=("serum1",)
    )
    database.set_pending_reason(pid, PENDING_RENDER_FAILED, error="boom")
    assert database.pending_counts() == {PENDING_RENDER_FAILED: 1}
    row = database.pending_presets()[0]
    assert row.error == "boom"
    assert row.last_attempt_at
    # The preset is preserved, not deleted.
    assert database.library_coverage()["discovered"] == 1


# ---------------------------------------------------------------------------
# PART 9: output-generation gating
# ---------------------------------------------------------------------------


def test_requesting_serum2_output_without_serum2_is_blocked(tmp_path: Path) -> None:
    """15/17. A fresh check that still fails prevents an impossible operation."""

    decision, _snapshot = evaluate_output_target("serum2", env=neither_env(tmp_path))
    assert not decision.allowed
    assert decision.rechecked, "the check must be fresh, not cached startup state"
    assert "Serum 2 is required to create Serum 2 presets" in decision.message
    assert "Install Serum 2" in decision.message


def test_fresh_check_discovers_newly_installed_serum2(tmp_path: Path) -> None:
    """16. A synth installed after launch must be picked up without a restart."""

    before, _ = evaluate_output_target("serum2", env=neither_env(tmp_path))
    assert not before.allowed
    _stale = refresh_capabilities(env=neither_env(tmp_path))
    after, _snapshot = evaluate_output_target(
        "serum2", env=serum2_only_env(tmp_path), previous=_stale
    )
    assert after.allowed
    assert after.became_available, "the transition must be detected"
    assert after.message == ""


def test_serum1_output_gating_is_symmetric(tmp_path: Path) -> None:
    """18. Serum 1 behaves the same way."""

    blocked, _ = evaluate_output_target("serum1", env=serum2_only_env(tmp_path))
    assert not blocked.allowed
    assert "Serum 1 is required to create Serum 1 presets" in blocked.message

    allowed, _ = evaluate_output_target("serum1", env=serum1_only_env(tmp_path))
    assert allowed.allowed


def test_output_gating_ignores_library_processing_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """24. Output capability must not depend on the library being processed.

    A user who just installed Serum 2 may create a Serum 2 preset even though
    none of their Serum 2 library has been rendered yet.
    """

    database, _root = catalogue(tmp_path, serum2_only_env(tmp_path), monkeypatch, fxp=50, serumpreset=40)
    coverage = database.library_coverage()
    assert coverage["learned"] == 0, "nothing has been learned in this fixture"
    assert coverage["pending"] == 50

    decision, _snapshot = evaluate_output_target(
        "serum2", env=serum2_only_env(tmp_path)
    )
    assert decision.allowed, (
        "an unprocessed library must never block creating a Serum 2 preset"
    )


def test_output_block_is_recorded_with_a_reason(tmp_path: Path, isolated) -> None:
    evaluate_output_target("serum2", env=neither_env(tmp_path))
    events = isolated.recent_events()
    blocked = [item for item in events if item["event_type"] == "output_blocked"]
    assert blocked
    assert blocked[0]["fields"]["generation"] == "serum2"
    assert "would produce no result" in blocked[0]["decision_reason"]
    check = next(
        item
        for item in events
        if item["fields"].get("decision") == "output_target_capability_check"
    )
    assert check["fields"]["outcome"] == "blocked"
    assert check["decision_reason"]


# ---------------------------------------------------------------------------
# PART 11 / 12: Match is never blocked; notices do not nag
# ---------------------------------------------------------------------------


def test_match_is_not_blocked_by_pending_presets(tmp_path: Path, isolated) -> None:
    """11/12. 4,000 learned + 1,000 pending must still allow Match."""

    record_pending_exclusion(
        pending_counts={PENDING_SERUM1_NOT_INSTALLED: 1000}, learned=4000
    )
    event = next(
        item
        for item in isolated.recent_events()
        if item["event_type"] == "pending_presets_excluded_from_retrieval"
    )
    assert event["fields"]["pending_total"] == 1000
    assert event["fields"]["learned"] == 4000
    reason = event["decision_reason"]
    assert "never blocked by pending work" in reason
    assert "capability determines what PatchLab can create" in reason


def test_missing_synth_notice_shows_once(tmp_path: Path) -> None:
    """22. The informational popup must not repeat every launch."""

    snapshot = refresh_capabilities(env=serum2_only_env(tmp_path))
    pending = {PENDING_SERUM1_NOT_INSTALLED: 327}

    first = missing_synth_notice(snapshot=snapshot, pending_counts=pending)
    assert first is not None
    assert "327" in first.body
    assert "still processed" in first.body
    assert "matching keeps working" in first.body

    acknowledge(first, choice="seen")
    again = missing_synth_notice(snapshot=snapshot, pending_counts=pending)
    assert again is None, "an acknowledged notice must not be shown again"


def test_missing_synth_notice_returns_when_the_count_changes_materially(
    tmp_path: Path,
) -> None:
    snapshot = refresh_capabilities(env=serum2_only_env(tmp_path))
    first = missing_synth_notice(
        snapshot=snapshot, pending_counts={PENDING_SERUM1_NOT_INSTALLED: 100}
    )
    assert first is not None
    acknowledge(first)
    # A trivial change stays quiet.
    assert (
        missing_synth_notice(
            snapshot=snapshot,
            pending_counts={PENDING_SERUM1_NOT_INSTALLED: 100 + MATERIAL_COUNT_DELTA - 1},
        )
        is None
    )
    # A material change is genuinely new information.
    assert (
        missing_synth_notice(
            snapshot=snapshot,
            pending_counts={PENDING_SERUM1_NOT_INSTALLED: 100 + MATERIAL_COUNT_DELTA},
        )
        is not None
    )


def test_no_notice_when_the_synth_is_available(tmp_path: Path) -> None:
    snapshot = refresh_capabilities(env=both_env(tmp_path))
    assert (
        missing_synth_notice(
            snapshot=snapshot, pending_counts={PENDING_SERUM1_NOT_INSTALLED: 50}
        )
        is None
    )


def test_new_synth_prompt_offers_processing_once(tmp_path: Path, isolated) -> None:
    """23. A one-time prompt when a synth becomes available."""

    notice = synth_available_notice(generation="serum2", pending_count=327)
    assert notice is not None
    assert notice.offers_processing
    assert "Serum 2 is now available" in notice.title
    assert "327" in notice.body
    assert "keep matching sounds either way" in notice.body

    acknowledge(notice, choice="not_now")
    assert synth_available_notice(generation="serum2", pending_count=327) is None

    offered = [
        item
        for item in isolated.recent_events()
        if item["event_type"] == "pending_processing_offered"
    ]
    assert offered
    assert "without rediscovery" in offered[0]["decision_reason"]


def test_not_now_choice_is_recorded(tmp_path: Path, isolated) -> None:
    """11. Choosing Not Now is recorded and changes nothing else."""

    notice = synth_available_notice(generation="serum2", pending_count=10)
    assert notice is not None
    acknowledge(notice, choice="not_now")
    event = next(
        item
        for item in isolated.recent_events()
        if item["event_type"] == "notice_acknowledged"
    )
    assert event["fields"]["choice"] == "not_now"
    state = load_notice_state()
    assert state["synth_now_available:serum2"]["choice"] == "not_now"


def test_no_prompt_when_nothing_is_pending(tmp_path: Path) -> None:
    assert synth_available_notice(generation="serum2", pending_count=0) is None


# ---------------------------------------------------------------------------
# PART 13: coverage counts
# ---------------------------------------------------------------------------


def test_coverage_distinguishes_every_required_category(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database, _root = catalogue(tmp_path, serum2_only_env(tmp_path), monkeypatch, fxp=6, serumpreset=4)
    coverage = database.library_coverage()
    for key in (
        "discovered",
        "by_status",
        "by_generation",
        "by_file_format",
        "pending_by_reason",
        "pending",
        "processed_params",
        "learned",
        "failed",
    ):
        assert key in coverage, key
    assert coverage["by_generation"] == {"serum1": 6, "serum2": 4}
    assert coverage["by_file_format"] == {"fxp": 6, "serumpreset": 4}


# ---------------------------------------------------------------------------
# PART 16: data safety
# ---------------------------------------------------------------------------


def test_original_preset_files_are_never_modified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PART 16: PatchLab reads user presets; it never writes to them."""

    import core.local_library as library_module

    install_host_stubs(monkeypatch)
    monkeypatch.setattr(
        library_module,
        "render_library",
        lambda **_kwargs: (_ for _ in ()).throw(_StopAfterIngest()),
    )
    root = make_library(tmp_path / "linked", fxp=5, serumpreset=3)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime) for path in sorted(root.iterdir())}
    try:
        library_module.process_linked_folder(
            root,
            db_path=tmp_path / "safety.db",
            audio_root=tmp_path / "audio",
            state_dir=tmp_path / "states",
            env=both_env(tmp_path / "both"),
            compact_mode=True,
            log=lambda _m: None,
        )
    except _StopAfterIngest:
        pass
    after = {path.name: (path.read_bytes(), path.stat().st_mtime) for path in sorted(root.iterdir())}
    assert before == after, "user preset files must never be written to or touched"


def test_pending_presets_are_never_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    database, _root = catalogue(tmp_path, neither_env(tmp_path), monkeypatch, fxp=4)
    ids = {row.id for row in database.pending_presets()}
    assert len(ids) == 4
    import core.local_library as library_module

    library_module.refresh_pending_reasons(
        db_path=_db_path(tmp_path), env=neither_env(tmp_path)
    )
    assert {row.id for row in database.pending_presets()} == ids
    assert database.library_coverage()["discovered"] == 4


def test_pending_serum2_presets_become_processable_without_rediscovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 9, the Serum 2 direction of the same guarantee."""

    import core.local_library as library_module

    # Catalogue on a Serum-1-only machine: the .serumpreset subset is pending.
    database, _root = catalogue(
        tmp_path, serum1_only_env(tmp_path), monkeypatch, fxp=3, serumpreset=7
    )
    assert database.pending_counts() == {PENDING_SERUM2_NOT_INSTALLED: 7}
    pending_ids = {row.id for row in database.presets_needing_generation("serum2")}
    assert len(pending_ids) == 7

    walked: list[Path] = []
    hashed: list[Path] = []
    monkeypatch.setattr(
        library_module, "discover_presets", lambda p: walked.append(Path(p)) or []
    )
    monkeypatch.setattr(
        library_module, "sha1_file", lambda p, chunk_size=0: hashed.append(p) or "x"
    )
    counts = library_module.refresh_pending_reasons(
        db_path=_db_path(tmp_path), env=both_env(tmp_path / "both")
    )
    assert walked == [] and hashed == []
    from core.preset_identity import BLOCKED_PENDING_REASONS, PENDING_AWAITING_PROCESSING

    assert set(counts) == {PENDING_AWAITING_PROCESSING}
    assert not set(counts) & BLOCKED_PENDING_REASONS
    assert {row.id for row in database.presets_with_status(("scanned",))} >= pending_ids


def test_process_now_actually_processes_only_the_pending_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 13: choosing Process Now does the work, without a rescan."""

    import core.local_library as library_module

    database, _root = catalogue(
        tmp_path, serum1_only_env(tmp_path), monkeypatch, fxp=3, serumpreset=5
    )
    assert len(database.presets_needing_generation("serum2")) == 5

    install_host_stubs(monkeypatch)
    walked: list[Path] = []
    monkeypatch.setattr(
        library_module, "discover_presets", lambda p: walked.append(Path(p)) or []
    )
    # Stop after ingest; rendering needs a real plug-in.
    monkeypatch.setattr(
        library_module,
        "render_library",
        lambda **_k: (_ for _ in ()).throw(_StopAfterIngest()),
    )
    library_module.refresh_pending_reasons(
        db_path=_db_path(tmp_path), env=both_env(tmp_path / "both")
    )
    try:
        summary = library_module.process_pending_for_generation(
            "serum2",
            db_path=_db_path(tmp_path),
            audio_root=tmp_path / "audio",
            state_dir=tmp_path / "states",
            env=both_env(tmp_path / "both"),
            compact_mode=True,
            log=lambda _m: None,
        )
    except _StopAfterIngest:
        summary = None
    assert walked == [], "Process Now must never re-walk the library"
    # All five Serum 2 presets were ingested and are no longer pending.
    assert database.pending_counts() == {}
    coverage = database.library_coverage()
    assert coverage["processed_params"] >= 5


def test_processing_pending_refuses_when_the_synth_is_still_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never start work whose renderer is missing; say why instead."""

    import core.local_library as library_module

    database, _root = catalogue(
        tmp_path, serum1_only_env(tmp_path), monkeypatch, serumpreset=4
    )
    with pytest.raises(RuntimeError) as caught:
        library_module.process_pending_for_generation(
            "serum2",
            db_path=_db_path(tmp_path),
            audio_root=tmp_path / "audio",
            state_dir=tmp_path / "states",
            env=serum1_only_env(tmp_path),
            log=lambda _m: None,
        )
    assert "Serum 2" in str(caught.value)
    # The presets are untouched and still pending.
    assert database.pending_counts() == {PENDING_SERUM2_NOT_INSTALLED: 4}


def test_missing_preset_file_stays_pending_not_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A moved or deleted file must not be invented into a failure."""

    import core.local_library as library_module

    database, root = catalogue(
        tmp_path, serum1_only_env(tmp_path), monkeypatch, serumpreset=3
    )
    for path in root.glob("*.SerumPreset"):
        path.unlink()
    install_host_stubs(monkeypatch)
    monkeypatch.setattr(
        library_module, "render_library", lambda **_k: (_ for _ in ()).throw(_StopAfterIngest())
    )
    library_module.refresh_pending_reasons(
        db_path=_db_path(tmp_path), env=both_env(tmp_path / "both")
    )
    summary = library_module.process_pending_for_generation(
        "serum2",
        db_path=_db_path(tmp_path),
        audio_root=tmp_path / "audio",
        state_dir=tmp_path / "states",
        env=both_env(tmp_path / "both"),
        compact_mode=True,
        log=lambda _m: None,
    )
    assert summary.missing_on_disk == 3
    assert summary.params_dumped == 0
    assert summary.failed_load == 0, "a missing file is not a load failure"
    # The catalog rows survive.
    assert database.library_coverage()["discovered"] == 3
