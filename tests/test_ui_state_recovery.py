"""A failure in a later stage must never erase a successful earlier stage.

The reported symptom was "clicking render sound library did not render sound
library, it just restarted 'link my preset folder'".  Two distinct wiring bugs
produced it:

1. ``start_render`` set the activity on the **link** card, and the linked-folder
   progress stream reports ``stage="scan"``, which
   ``_local_library_progress_changed`` also routed to the link card.  So a
   Render click visibly re-ran Link for the whole 36-minute catalog pass.
2. That path's failure handler is ``_scan_failed``, which never set
   ``_render_failure_detail``, so the Render card never showed a failure and
   looked as though nothing had been attempted.

These tests exercise the routing logic and the resolver directly, without a
QApplication, so they stay fast and deterministic.
"""

from __future__ import annotations

import ast
import inspect
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from core.privacy import PrivacyChoice
from core.workflow_state import WorkflowActivity, resolve_workflow_state


def _factory_bundle(path: Path) -> Path:
    # Idempotent: a single test may resolve the workflow several times.
    if path.exists():
        return path
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE presets (id INTEGER PRIMARY KEY, searchable INTEGER);
        CREATE TABLE preset_embeddings (
          preset_id INTEGER PRIMARY KEY,
          embedding_f16 BLOB NOT NULL
        );
        INSERT INTO presets(id,searchable) VALUES (1,1);
        """
    )
    connection.execute(
        "INSERT INTO preset_embeddings(preset_id,embedding_f16) VALUES (?,?)",
        (1, bytes(512 * 2)),
    )
    connection.commit()
    connection.close()
    return path


def _library(tmp_path: Path, *, presets: int, fingerprinted: int = 0) -> Path:
    """A per-user library with ``presets`` catalog rows under the linked folder."""

    from core.db import Database

    linked = tmp_path / "Serum 2 Presets"
    linked.mkdir(parents=True, exist_ok=True)
    database = Database(tmp_path / "app-data" / "library.db")
    for index in range(presets):
        path = linked / f"preset-{index}.serumpreset"
        path.write_bytes(b"SERUM" + str(index).encode())
        preset_id, _ = database.insert_preset(
            path=path, name=path.stem, synth="serum2", content_hash=f"hash{index}"
        )
        if index < fingerprinted:
            database.upsert_fingerprint(
                preset_id, 0, bytes(512 * 4), bytes(64)
            )
    return linked


def _resolve(
    tmp_path: Path,
    *,
    linked: Path | None,
    activities: dict[str, WorkflowActivity] | None = None,
    render_failed_detail: str = "",
    compact_mode: bool = True,
    audio_selected: bool = False,
):
    privacy = PrivacyChoice(
        use_and_share_own_presets=linked is not None,
        linked_folder=linked,
    )
    with patch("core.workflow_state.validate_model_assets"):
        return resolve_workflow_state(
            privacy=privacy,
            local_database_path=tmp_path / "app-data" / "library.db",
            factory_bundle_path=_factory_bundle(tmp_path / "factory.sqlite"),
            audio_selected=audio_selected,
            activities=activities,
            render_failed_detail=render_failed_detail,
            compact_mode=compact_mode,
        )


# ---------------------------------------------------------------------------
# Link stays linked
# ---------------------------------------------------------------------------


def test_render_failure_leaves_link_complete(tmp_path: Path) -> None:
    """Item 6/7: Render failing must not send the user back to Link."""

    linked = _library(tmp_path, presets=12)
    state = _resolve(
        tmp_path,
        linked=linked,
        render_failed_detail=(
            "PatchLab couldn't start the Serum 1 renderer, so Serum 1 presets "
            "were skipped. Your preset folder is still linked."
        ),
    )
    assert state.link.phase == "complete"
    assert "Linked:" in state.link.text
    assert "12 presets" in state.link.text
    # And Render reports a retryable failure, not "not started" and not complete.
    assert state.render.phase == "failed"
    assert "retry" in state.render.text.casefold()
    assert "still linked" in state.render.detail


def test_render_failure_does_not_mark_render_complete(tmp_path: Path) -> None:
    linked = _library(tmp_path, presets=8)
    state = _resolve(tmp_path, linked=linked, render_failed_detail="boom")
    assert state.render.phase != "complete"
    assert state.render.current == 0
    assert state.render.total == 8


def test_render_activity_clears_on_failure(tmp_path: Path) -> None:
    """Item 8: no terminal failure may leave a stale activity state."""

    linked = _library(tmp_path, presets=5)
    # While running, the render card is in-progress...
    running = _resolve(
        tmp_path,
        linked=linked,
        activities={"render": WorkflowActivity(2, 5, "Rendering 2 of 5")},
        render_failed_detail="boom",
    )
    assert running.render.phase == "in-progress"
    # ...and once the activity is cleared the failure becomes visible.
    stopped = _resolve(tmp_path, linked=linked, render_failed_detail="boom")
    assert stopped.render.phase == "failed"


def test_link_only_regresses_when_the_folder_actually_goes_away(
    tmp_path: Path,
) -> None:
    linked = _library(tmp_path, presets=4)
    assert _resolve(tmp_path, linked=linked).link.phase == "complete"
    # Unlinking is the user's explicit action.
    assert _resolve(tmp_path, linked=None).link.phase == "needs-action"


def test_partially_learned_library_reports_progress_not_failure(
    tmp_path: Path,
) -> None:
    """Item 10: presets already learned must stay counted after a failure."""

    linked = _library(tmp_path, presets=10, fingerprinted=6)
    state = _resolve(tmp_path, linked=linked, render_failed_detail="renderer missing")
    assert state.link.phase == "complete"
    assert state.render.phase == "failed"
    assert state.render.current == 6, "learned presets must remain counted"
    assert state.render.total == 10


def test_fully_learned_library_reads_complete_even_after_a_stale_failure(
    tmp_path: Path,
) -> None:
    linked = _library(tmp_path, presets=6, fingerprinted=6)
    state = _resolve(tmp_path, linked=linked, render_failed_detail="old failure")
    assert state.render.phase == "complete"
    assert "learned" in state.render.text


# ---------------------------------------------------------------------------
# Card ownership: the Render button must drive the Render card
# ---------------------------------------------------------------------------


def _ui_source() -> str:
    return (Path(__file__).resolve().parents[1] / "app" / "ui.py").read_text(
        encoding="utf-8"
    )


def test_compact_render_sets_the_render_activity_not_link() -> None:
    """The literal wiring bug behind "it just restarted link my preset folder"."""

    source = _ui_source()
    start_render = source[source.index("def start_render(") :]
    start_render = start_render[: start_render.index("def toggle_render_pause(")]
    assert '_set_workflow_activity(\n                "render"' in start_render or (
        '"render", 0, 0, "Starting compact preset-library processing' in start_render
    ), "Render Sound Library must set the render card's activity"
    assert (
        '"link", 0, 0, "Starting compact preset-library processing' not in start_render
    ), "the compact render path must no longer drive the Link card"
    assert "_compact_render_active = True" in start_render


def test_scan_progress_routes_to_render_while_render_owns_the_job() -> None:
    source = _ui_source()
    handler = source[source.index("def _local_library_progress_changed(") :]
    handler = handler[: handler.index("def _render_library_complete(")]
    # The scan stage must go to the render card when Render started the job.
    assert '_compact_render_active' in handler
    assert handler.index("_compact_render_active") < handler.index(
        '_set_workflow_activity("link"'
    ), "the compact-render branch must be checked before falling back to Link"


def test_scan_failure_sets_render_failure_detail_when_render_started_it() -> None:
    source = _ui_source()
    handler = source[source.index("def _scan_failed(") :]
    handler = handler[: handler.index("def _user_facing_error(")]
    assert "_compact_render_active" in handler
    assert "_render_failure_detail = self._user_facing_error(error)" in handler
    assert "render_button.setEnabled(True)" in handler, "retry must stay available"
    # That the handler does not unlink the folder is asserted structurally in
    # test_no_failure_handler_unlinks_the_preset_folder, which parses the AST and
    # is therefore not confused by comments mentioning these names.


def test_no_failure_handler_unlinks_the_preset_folder() -> None:
    """Belt and braces: no failure path may clear the linked folder."""

    source = _ui_source()
    tree = ast.parse(source)
    handlers = {
        "_scan_failed",
        "_render_failed",
        "_match_failed",
        "_analyze_failed",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in handlers:
            continue
        body = ast.unparse(node)
        assert "save_privacy_choice" not in body, node.name
        assert "PrivacyChoice(" not in body, node.name
        assert "linked_folder=" not in body, node.name


# ---------------------------------------------------------------------------
# Match recovery
# ---------------------------------------------------------------------------


def test_match_failure_restores_controls_and_clears_activity() -> None:
    source = _ui_source()
    handler = source[source.index("def _match_failed(") :]
    handler = handler[: handler.index("def report_model_asset_error(")]
    assert '_workflow_activities.pop("match", None)' in handler, "activity must clear"
    assert "match_start_button.setEnabled(" in handler, "retry must be possible"
    assert "match_cancel_button.setEnabled(False)" in handler
    assert "match_synth.setEnabled(True)" in handler
    assert "match_budget.setEnabled(True)" in handler
    assert "match_offset.setEnabled(True)" in handler
    # The selected source audio must be preserved, not cleared.
    assert "_match_audio_path = None" not in handler


def test_match_failure_clears_the_loading_state(tmp_path: Path) -> None:
    linked = _library(tmp_path, presets=3)
    # With no match activity the card must not read in-progress.
    state = _resolve(tmp_path, linked=linked, audio_selected=True, activities={})
    assert state.match.phase != "in-progress"
    assert "ready" in state.match.text.casefold()


def test_match_runner_has_a_liveness_watchdog() -> None:
    """The guarantee that an output-less worker still terminates."""

    from app import workers

    source = inspect.getsource(workers.MatchProcessRunner)
    assert "_liveness_timer" in source
    assert "_liveness_timed_out" in source
    assert "_touch_liveness" in source
    # It must kill the process and emit a terminal failure.
    timed_out = source[source.index("def _liveness_timed_out") :]
    timed_out = timed_out[: timed_out.index("def start(")]
    assert "self.process.kill()" in timed_out
    assert "self.failed.emit(" in timed_out
    # Evidence before recovery.
    assert timed_out.index("write_postmortem") < timed_out.index("self.process.kill()")


def test_liveness_budget_is_generous_enough_not_to_kill_real_matches() -> None:
    from app.workers import DEFAULT_MATCH_INACTIVITY_TIMEOUT_MS

    # A best-quality structural search is bounded at 900s plus model loading, so
    # the GUI backstop must sit well above that.
    assert DEFAULT_MATCH_INACTIVITY_TIMEOUT_MS >= 15 * 60 * 1000


# ---------------------------------------------------------------------------
# User-facing errors (PART 26)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("internal", "expected"),
    [
        (
            "RendererUnavailableError: No usable serum2 renderer is available. "
            "Tried: serum2/AU: path does not exist",
            "couldn't start Serum 2",
        ),
        (
            "RuntimeError: No usable serum1 renderer is available while processing",
            "couldn't start Serum 1",
        ),
        (
            "WorkerRespawnStormError: Render workers died and were replaced 42 times",
            "Serum kept stopping",
        ),
        ("OSError: [Errno 28] No space left on device", "disk space"),
        ("PermissionError: [Errno 13] Permission denied", "permissions"),
        ("RuntimeError: match stalled with no measurable progress", "stopped making progress"),
    ],
)
def test_user_facing_errors_are_actionable_not_tracebacks(
    internal: str, expected: str
) -> None:
    from app.ui import MainWindow

    message = MainWindow._user_facing_error(internal)
    assert expected.casefold() in message.casefold()
    # No internal class names, tracebacks or plug-in jargon.
    assert "Error:" not in message
    assert "Traceback" not in message
    for jargon in ("renderer", "worker", "VST", "serum1", "serum2"):
        assert jargon.casefold() not in message.casefold(), jargon
    assert len(message) < 240


def test_renderer_errors_reassure_that_setup_is_intact() -> None:
    from app.ui import MainWindow

    for internal in (
        "No usable serum2 renderer is available",
        "No usable serum1 renderer is available",
        "renderer unavailable",
    ):
        message = MainWindow._user_facing_error(internal)
        assert "still linked" in message, message


# ---------------------------------------------------------------------------
# PART 6 / PART 12: the one informational notice, at the right moment
# ---------------------------------------------------------------------------


def _method(class_name: str, method: str) -> ast.FunctionDef:
    """Locate a method by class and name, independent of file order."""

    tree = ast.parse(_ui_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return item
    raise AssertionError(f"{class_name}.{method} not found")


def _calls(node: ast.AST) -> list[str]:
    return [
        ast.unparse(inner.func)
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
    ]


def test_scan_completion_triggers_the_pending_notice() -> None:
    """Case A's popup fires after processing, not at an arbitrary moment."""

    assert "self._notify_pending_after_scan" in _calls(
        _method("LegacyMainWindow", "_scan_completed")
    )
    notify = _calls(_method("MainWindow", "_notify_pending_after_scan"))
    # A fresh capability check, so a synth installed during the scan is reflected
    # rather than a stale startup snapshot.
    assert "refresh_capabilities" in notify
    assert "self._maybe_show_missing_synth_notice" in notify


def test_match_start_gates_before_any_side_effect() -> None:
    """A refused Match must not cancel background work, disable controls or spin.

    Compares statement order inside the real ``start_match``: the capability gate
    has to precede the background-scan cancel, every control being disabled, and
    the busy state. (An earlier version of this test only compared against the
    busy state, which let the gate sit *after* the controls were disabled -- a
    refused Match left the user with dead controls.)
    """

    body = _method("LegacyMainWindow", "start_match").body

    def first(predicate) -> int:
        for index, statement in enumerate(body):
            if predicate(_calls(statement), ast.unparse(statement)):
                return index
        raise AssertionError("statement not found")

    gate = first(lambda calls, _s: "self._check_output_capability" in calls)
    cancel = first(lambda calls, _s: "self.runner.cancel" in calls or "self.runner.running" in _s)
    disable = first(lambda _c, source: "match_start_button.setEnabled(False)" in source)
    busy = first(lambda _c, source: "_set_workflow_activity('match'" in source)
    launch = first(lambda calls, _s: "self.match_runner.start" in calls)
    assert gate < cancel < disable < busy < launch, (gate, cancel, disable, busy, launch)


def test_capability_gate_only_refuses_for_a_missing_synth() -> None:
    """Library processing state may be *offered* work, but never blocks output.

    The single ``return False`` in the gate must be guarded by the capability
    decision, and nothing that reads pending/learned counts may sit in that
    condition.
    """

    function = _method("MainWindow", "_check_output_capability")
    refusals = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
    ]
    assert len(refusals) == 1
    guards = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If) and refusals[0] in list(ast.walk(node))
    ]
    assert guards and "decision.allowed" in ast.unparse(guards[-1].test)
    for forbidden in ("pending", "coverage", "learned"):
        assert forbidden not in ast.unparse(guards[-1].test)


def test_pending_offer_gives_both_choices_and_never_blocks_match() -> None:
    source = _ui_source()
    offer = source[source.index("def _offer_pending_processing(") :]
    offer = offer[: offer.index("def _last_operation_name(")]
    assert '"Process Now"' in offer
    assert '"Not Now"' in offer
    # Not Now must simply return; nothing may disable matching.
    not_now = offer[offer.index("if not chose_process:") :]
    for forbidden in ("setEnabled(False)", "match_start_button", "_model_asset_error"):
        assert forbidden not in not_now, (
            f"declining the offer must not touch {forbidden}"
        )
    # Process Now must use the no-rescan path.
    assert "pending_generation=generation" in offer
