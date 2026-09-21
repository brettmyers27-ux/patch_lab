"""Stall detection for a hang that never throws.

The reported Match failure produced no exception in the parent at all: a pool
whose workers never reached a task, and a ``map`` call with no timeout.  A
watchdog is the only thing that can turn that into a terminal state, and it has
to be able to tell "wedged" apart from "legitimately slow but progressing" or it
would kill healthy long renders.

Everything here uses an injected clock, so nothing depends on wall-clock timing
and nothing is flaky.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.diagnostics import DiagnosticRecorder, set_recorder
from core.operation_state import (
    DEFAULT_PHASE_STALL_SECONDS,
    TERMINAL_PHASES,
    StallDetector,
    StallWatchdog,
    capture_postmortem,
    capture_thread_stacks,
    stall_budget_for,
    start_operation,
    write_postmortem,
)


@pytest.fixture(autouse=True)
def isolated_recorder(tmp_path: Path):
    recorder = DiagnosticRecorder(root=tmp_path / "diagnostics", session_id="test")
    set_recorder(recorder)
    yield recorder
    recorder.close()
    set_recorder(None)


class FakeClock:
    """A controllable monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def tracker_at(phase: str, clock: FakeClock):
    tracker = start_operation(
        "match", subsystem="match", operation_id="op-hang", clock=clock
    )
    if phase != "accepted":
        tracker.enter_phase(phase, reason="test setup")
    return tracker


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


def test_tracker_records_phase_history_and_reasons() -> None:
    clock = FakeClock()
    tracker = tracker_at("accepted", clock)
    clock.advance(2.0)
    tracker.enter_phase("decoding", reason="decoding the user's audio")
    clock.advance(3.0)
    tracker.enter_phase("model-loading", reason="loading CLAP")
    clock.advance(5.0)

    snapshot = tracker.snapshot()
    assert snapshot["operation_id"] == "op-hang"
    assert snapshot["current_phase"] == "model-loading"
    assert snapshot["previous_phase"] == "decoding"
    assert snapshot["last_successful_phase"] == "decoding"
    assert snapshot["elapsed_seconds"] == 10.0
    assert snapshot["phase_elapsed_seconds"] == 5.0
    history = [item["phase"] for item in snapshot["phase_history"]]
    assert history == ["accepted", "decoding", "model-loading"]
    assert snapshot["phase_history"][1]["duration_seconds"] == 3.0
    assert snapshot["phase_history"][1]["completed"] is True


def test_terminal_states_are_recorded_with_a_reason() -> None:
    clock = FakeClock()
    tracker = tracker_at("evaluation", clock)
    tracker.complete("produced a result")
    assert tracker.phase == "complete"
    assert tracker.terminal
    assert tracker.terminal_reason == "produced a result"
    assert "complete" in TERMINAL_PHASES


def test_failure_preserves_the_exception_chain() -> None:
    """PART 21: nothing may disappear between worker and UI."""

    clock = FakeClock()
    tracker = tracker_at("worker-pool-initialization", clock)
    try:
        try:
            next(item for item in () if True)
        except StopIteration as inner:
            raise RuntimeError("render worker initialization failed") from inner
    except RuntimeError as exc:
        marker = tracker.fail(exc)

    assert tracker.phase == "failed"
    assert marker.digest
    failure = tracker.failure()
    assert failure is not None
    assert failure["phase"] == "worker-pool-initialization"
    chain = failure["exception_chain"]
    assert [item["type"] for item in chain] == ["RuntimeError", "StopIteration"]
    assert chain[1]["link"] == "cause"
    assert chain[0]["traceback"]


# ---------------------------------------------------------------------------
# Stall vs slow
# ---------------------------------------------------------------------------


def test_slow_but_progressing_work_is_not_a_stall() -> None:
    """A long render that keeps reporting progress must never be killed."""

    clock = FakeClock()
    tracker = tracker_at("rendering", clock)
    detector = StallDetector(tracker, clock=clock)
    budget = stall_budget_for("rendering")

    # Ten times the budget of elapsed time, but progress every half-budget.
    for _ in range(20):
        clock.advance(budget * 0.5)
        tracker.mark_progress(text="batch rendered")
        verdict = detector.evaluate()
        assert not verdict.stalled
        assert verdict.progressing

    assert tracker.elapsed() > budget * 9
    assert detector.evaluate().reason.startswith("last progress")


def test_silence_beyond_the_phase_budget_is_a_stall() -> None:
    clock = FakeClock()
    tracker = tracker_at("evaluation", clock)
    detector = StallDetector(tracker, clock=clock)

    clock.advance(stall_budget_for("evaluation") - 1.0)
    assert not detector.evaluate().stalled

    clock.advance(2.0)
    verdict = detector.evaluate()
    assert verdict.stalled
    assert not verdict.progressing
    assert verdict.phase == "evaluation"
    assert "no measurable progress" in verdict.reason
    assert "inactivity budget" in verdict.reason


def test_budgets_are_per_phase() -> None:
    """Loading an 800 MB checkpoint is silent for much longer than discovery."""

    assert (
        DEFAULT_PHASE_STALL_SECONDS["model-loading"]
        > DEFAULT_PHASE_STALL_SECONDS["renderer-discovery"]
    )
    clock = FakeClock()
    tracker = tracker_at("renderer-discovery", clock)
    detector = StallDetector(tracker, clock=clock)
    clock.advance(DEFAULT_PHASE_STALL_SECONDS["renderer-discovery"] + 1.0)
    assert detector.evaluate().stalled

    clock2 = FakeClock()
    slow = tracker_at("model-loading", clock2)
    slow_detector = StallDetector(slow, clock=clock2)
    clock2.advance(DEFAULT_PHASE_STALL_SECONDS["renderer-discovery"] + 1.0)
    assert not slow_detector.evaluate().stalled


def test_terminal_operations_are_never_reported_as_stalled() -> None:
    clock = FakeClock()
    tracker = tracker_at("evaluation", clock)
    tracker.complete()
    clock.advance(100_000.0)
    verdict = StallDetector(tracker, clock=clock).evaluate()
    assert not verdict.stalled
    assert "terminal phase" in verdict.reason


# ---------------------------------------------------------------------------
# The hang test: no exception, detected, snapshotted, then recovered
# ---------------------------------------------------------------------------


def test_silent_hang_is_detected_and_snapshotted_before_recovery(
    tmp_path: Path, isolated_recorder
) -> None:
    """PART 31: a worker stops reporting progress without throwing anything.

    Asserts the ordering that makes the evidence useful: the postmortem is
    captured BEFORE recovery runs, because recovery tears down the pool and
    destroys the worker state that explains the hang.
    """

    clock = FakeClock()
    tracker = start_operation(
        "match",
        subsystem="match",
        operation_id="op-silent",
        clock=clock,
        target_synth="serum2",
    )
    tracker.enter_phase("worker-pool-initialization", reason="creating render pool")

    order: list[str] = []
    recovered: dict[str, object] = {}

    def postmortem_factory(_verdict):
        order.append("postmortem")
        return capture_postmortem(
            tracker,
            trigger="stall",
            workers=[
                {"worker_id": "render-1", "pid": 111, "alive": True, "exitcode": None},
                {"worker_id": "render-2", "pid": 112, "alive": False, "exitcode": 1},
            ],
            queue_state={"pending_tasks": 16, "completed_tasks": 0},
            pending_callbacks=["progress_callback"],
            disk_paths=[tmp_path],
        )

    def on_stall(verdict, postmortem):
        order.append("recover")
        recovered["verdict"] = verdict
        recovered["postmortem"] = postmortem

    watchdog = StallWatchdog(
        tracker, on_stall=on_stall, postmortem_factory=postmortem_factory
    )

    # No progress, no exception -- exactly the reported shape.
    clock.advance(stall_budget_for("worker-pool-initialization") + 1.0)
    verdict = watchdog.check_once()

    assert verdict.stalled
    assert order == ["postmortem", "recover"], (
        "state must be frozen before recovery destroys it"
    )

    postmortem = recovered["postmortem"]
    assert postmortem["trigger"] == "stall"
    assert postmortem["operation_id"] == "op-silent"
    assert postmortem["current_phase"] == "worker-pool-initialization"
    assert postmortem["previous_phase"] == "accepted"
    assert postmortem["seconds_since_last_progress"] > postmortem["stall_budget_seconds"]
    # A hang produces no traceback, so thread stacks are the evidence.
    assert postmortem["thread_stacks"]
    assert any(item.get("stack") for item in postmortem["thread_stacks"])
    assert len(postmortem["workers"]) == 2
    assert postmortem["queue_state"]["pending_tasks"] == 16
    assert postmortem["pending_callbacks"] == ["progress_callback"]
    assert postmortem["resources"]["disk"]
    assert postmortem["operation_state_machine"]["phase_history"]
    assert postmortem["cancellation_state"]["cancelled"] is False

    # And the report can tell stalled from progressing.
    events = isolated_recorder.recent_events()
    detected = [item for item in events if item["event_type"] == "stall_detected"]
    assert detected
    assert detected[0]["fields"]["stalled"] is True
    assert detected[0]["fields"]["progressing"] is False
    assert detected[0]["decision_reason"]


def test_watchdog_fires_only_once() -> None:
    clock = FakeClock()
    tracker = tracker_at("evaluation", clock)
    calls: list[int] = []
    watchdog = StallWatchdog(
        tracker,
        on_stall=lambda _v, _p: calls.append(1),
        postmortem_factory=lambda _v: {},
    )
    clock.advance(stall_budget_for("evaluation") + 10.0)
    watchdog.check_once()
    watchdog.check_once()
    watchdog.check_once()
    assert calls == [1], "recovery must not be attempted repeatedly"


def test_postmortem_survives_a_capture_failure(tmp_path: Path) -> None:
    """Evidence capture must never block recovery."""

    clock = FakeClock()
    tracker = tracker_at("evaluation", clock)
    recovered: list[str] = []

    def exploding_factory(_verdict):
        raise RuntimeError("snapshot failed")

    watchdog = StallWatchdog(
        tracker,
        on_stall=lambda _v, _p: recovered.append("recovered"),
        postmortem_factory=exploding_factory,
    )
    clock.advance(stall_budget_for("evaluation") + 1.0)
    watchdog.check_once()
    assert recovered == ["recovered"]
    assert watchdog.postmortem is None


def test_postmortem_is_written_to_disk(tmp_path: Path) -> None:
    clock = FakeClock()
    tracker = tracker_at("evaluation", clock)
    payload = capture_postmortem(tracker, trigger="stall")
    path = write_postmortem(payload)
    assert path is not None
    assert path.name == "postmortem.json"
    import json

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["operation_id"] == "op-hang"
    assert reloaded["trigger"] == "stall"


def test_thread_stacks_are_capturable() -> None:
    stacks = capture_thread_stacks()
    assert stacks
    current = [item for item in stacks if item.get("is_current")]
    assert current
    assert any("test_thread_stacks_are_capturable" in line for line in current[0]["stack"])


def test_successful_runs_capture_no_stacks(isolated_recorder) -> None:
    """Stack capture is failure-only, so it costs nothing on the happy path."""

    clock = FakeClock()
    tracker = tracker_at("evaluation", clock)
    for _ in range(50):
        clock.advance(1.0)
        tracker.mark_progress(text="evaluated")
    tracker.complete()
    events = isolated_recorder.recent_events()
    assert not [item for item in events if item["event_type"] == "postmortem_captured"]


# ---------------------------------------------------------------------------
# PART 20: heartbeat-aware stall policy
# ---------------------------------------------------------------------------


def test_busy_heartbeat_extends_the_budget_but_does_not_remove_it() -> None:
    """A long render batch must not be killed; a wedged one must still trip.

    Heartbeats are evidence of life, not of progress, so they buy a bounded
    extension rather than immunity.
    """

    from core.operation_state import HEARTBEAT_BUDGET_MULTIPLIER, BUSY_WORKER_STATES

    clock = FakeClock()
    tracker = tracker_at("rendering", clock)
    detector = StallDetector(tracker, clock=clock)
    budget = stall_budget_for("rendering")

    # Past the plain budget with no completed work, but a worker is rendering.
    clock.advance(budget + 10.0)
    tracker.note_heartbeat("render-1", "rendering")
    verdict = detector.evaluate()
    assert not verdict.stalled, "a busy worker must not be killed mid-batch"
    assert not verdict.progressing, "but this is not real progress either"
    assert "budget is extended" in verdict.reason
    assert verdict.budget_seconds == budget * HEARTBEAT_BUDGET_MULTIPLIER

    # Heartbeating forever without completing anything still trips eventually.
    while tracker.since_progress() <= budget * HEARTBEAT_BUDGET_MULTIPLIER:
        clock.advance(30.0)
        tracker.note_heartbeat("render-1", "rendering")
    final = detector.evaluate()
    assert final.stalled, "an endlessly-heartbeating worker must still be caught"
    assert "nothing has completed" in final.reason


def test_idle_worker_heartbeat_does_not_extend_the_budget() -> None:
    """A worker sitting in waiting-for-task is not evidence of useful work."""

    clock = FakeClock()
    tracker = tracker_at("rendering", clock)
    detector = StallDetector(tracker, clock=clock)
    clock.advance(stall_budget_for("rendering") + 10.0)
    tracker.note_heartbeat("render-1", "waiting-for-task")
    verdict = detector.evaluate()
    assert verdict.stalled
    assert "no worker has reported a busy state" in verdict.reason


def test_stale_busy_heartbeat_does_not_extend_the_budget() -> None:
    """A worker that went quiet mid-render is exactly the hang case."""

    clock = FakeClock()
    tracker = tracker_at("rendering", clock)
    detector = StallDetector(tracker, clock=clock)
    tracker.note_heartbeat("render-1", "rendering")
    clock.advance(stall_budget_for("rendering") * 1.5)
    verdict = detector.evaluate()
    assert verdict.stalled
    assert "last busy worker heartbeat was" in verdict.reason


def test_real_progress_still_beats_everything() -> None:
    """Completed work resets the budget outright, no extension needed."""

    clock = FakeClock()
    tracker = tracker_at("rendering", clock)
    detector = StallDetector(tracker, clock=clock)
    for _ in range(10):
        clock.advance(stall_budget_for("rendering") * 0.9)
        tracker.mark_progress(text="batch complete")
        verdict = detector.evaluate()
        assert not verdict.stalled
        assert verdict.progressing


def test_postmortem_reports_per_worker_liveness() -> None:
    """A reader must be able to tell a busy worker from a vanished one."""

    clock = FakeClock()
    tracker = tracker_at("rendering", clock)
    tracker.note_heartbeat("render-1", "rendering")
    clock.advance(45.0)
    tracker.note_heartbeat("render-2", "waiting-for-task")
    payload = capture_postmortem(tracker, trigger="stall")
    beats = payload["worker_heartbeats"]
    assert beats["render-1"]["busy"] is True
    assert beats["render-1"]["seconds_since_heartbeat"] == 45.0
    assert beats["render-2"]["busy"] is False
    assert beats["render-2"]["seconds_since_heartbeat"] == 0.0
