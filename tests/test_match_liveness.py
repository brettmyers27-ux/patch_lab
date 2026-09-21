"""Bug A regression: Match must never load indefinitely.

Reported behaviour on the user's Mac (PatchLab 1.5.3, commit 7223b349):

* progress reached ``decoding`` then ``loading-models`` and stopped;
* ``core/matcher.py`` line 285 raised a bare ``StopIteration`` in every spawned
  render worker, because ``next()`` searched for a Serum 1 VST2 binary that the
  machine did not have -- for a Serum **2** match;
* ``multiprocessing.Pool`` replaced each dead worker forever (measured on this
  machine: 551 distinct worker PIDs in 8 seconds, and 200 in the user's log);
* ``pool.map`` blocked with no timeout because no worker ever consumed a task;
* so no terminal state ever reached the UI and the spinner never stopped.

Every test here targets one link in that chain.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import pytest

from core.diagnostics import DiagnosticRecorder, set_recorder
from core.matcher import (
    MAX_WORKER_REPLACEMENTS,
    WORKER_INIT_FAILURE_PREFIX,
    RenderWorkerInitializationError,
    WorkerRespawnStormError,
    _init_render_worker,
    _PoolSupervisor,
    _render_candidate,
    _worker_probe,
    decode_worker_init_failure,
)
from core.renderer_selection import RendererUnavailableError
from tests.test_renderer_routing import serum2_only_env


@pytest.fixture(autouse=True)
def isolated_recorder(tmp_path: Path):
    recorder = DiagnosticRecorder(root=tmp_path / "diagnostics", session_id="test")
    set_recorder(recorder)
    yield recorder
    recorder.close()
    set_recorder(None)


# ---------------------------------------------------------------------------
# 1-2. No valid renderer / the StopIteration site
# ---------------------------------------------------------------------------


def test_worker_initializer_does_not_raise_when_a_renderer_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The initializer must not die: a dying initializer is what makes Pool churn."""

    import core.renderer_selection as selection_module

    env = serum2_only_env(tmp_path)
    monkeypatch.setattr(selection_module, "ENV", env)

    # Must return normally even though serum1 cannot be hosted.
    _init_render_worker(str(tmp_path), None, ("serum1",), {})

    from core.matcher import _RENDER

    assert "init_failure" in _RENDER
    detail = _RENDER["init_failure"]
    assert detail["exception_type"] == "RendererUnavailableError"
    assert "serum1" in detail["message"]
    assert detail["traceback"]
    assert detail["fingerprint"]["digest"]
    assert "Serum 1" in detail["user_message"]
    _RENDER.clear()


def test_worker_initializer_only_hosts_the_required_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Serum 2 match must not require Serum 1 at all.

    This is the smallest correct functional fix: ``_render_candidate_unsafe``
    only ever indexes ``hosts[candidate.synth]``, so hosting the other
    generation imposed a requirement the work did not have.
    """

    import core.matcher as matcher_module
    import core.renderer_selection as selection_module

    env = serum2_only_env(tmp_path)
    monkeypatch.setattr(selection_module, "ENV", env)

    opened: list[str] = []

    def fake_processor(candidate):
        opened.append(f"{candidate.synth}/{candidate.format}")
        return ("engine", "processor")

    monkeypatch.setattr(
        "core.plugin_host.make_dawdreamer_processor", fake_processor
    )
    monkeypatch.setattr(
        matcher_module, "resolve_synthesis_assets", lambda: _StubAssets(tmp_path)
    )
    monkeypatch.setattr(matcher_module, "_serum1_targets", lambda *_a, **_k: "s1")
    monkeypatch.setattr(matcher_module, "_serum2_targets", lambda *_a, **_k: "s2")

    _init_render_worker(str(tmp_path), None, ("serum2",), {})

    from core.matcher import _RENDER

    assert "init_failure" not in _RENDER
    assert opened == ["serum2/VST3"], "only the requested generation may be hosted"
    assert _RENDER["hosted_synths"] == ("serum2",)
    _RENDER.clear()


class _StubAssets:
    def __init__(self, root: Path) -> None:
        self.library_db = root / "library.db"
        self.serum2_targets = root / "targets.npz"
        self.serum2_schema = root / "schema.json"
        self.serum2_schema.write_text("{}", encoding="utf-8")
        self.factory_mapping = None


# ---------------------------------------------------------------------------
# 3. The failure reaches the parent
# ---------------------------------------------------------------------------


def test_init_failure_reaches_the_parent_through_the_result_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import core.renderer_selection as selection_module
    from core.matcher import _RENDER

    monkeypatch.setattr(selection_module, "ENV", serum2_only_env(tmp_path))
    _init_render_worker(str(tmp_path), None, ("serum1",), {})

    _waveform, _coverage, error = _render_candidate((object(), 60, 1.0))
    assert error is not None
    assert error.startswith(WORKER_INIT_FAILURE_PREFIX)

    detail = decode_worker_init_failure(error)
    assert detail is not None
    assert detail["exception_type"] == "RendererUnavailableError"
    assert detail["renderer_selection"]["candidates"]
    assert detail["fingerprint"]["code_location"]
    _RENDER.clear()


def test_decode_ignores_ordinary_render_errors() -> None:
    assert decode_worker_init_failure(None) is None
    assert decode_worker_init_failure("RuntimeError: preset rejected") is None


# ---------------------------------------------------------------------------
# 4. The pool does not infinitely respawn
# ---------------------------------------------------------------------------


def _storming_initializer() -> None:
    """Reproduce the shipped defect exactly: a bare next() over nothing."""

    next(item for item in () if True)


@pytest.mark.parametrize("processes", [2])
def test_shipped_initializer_shape_storms_without_the_supervisor(
    processes: int,
) -> None:
    """Document the defect: Pool() succeeds and then churns without bound.

    This is the *unsupervised* behaviour, asserted so the regression is visible
    if anyone reintroduces a raising initializer.
    """

    context = mp.get_context("spawn")
    pool = context.Pool(processes, initializer=_storming_initializer)
    try:
        # A healthy pool holds exactly `processes` workers for its whole life, so
        # seeing more distinct PIDs than that is proof of replacement churn.
        # Poll until the condition rather than counting inside a fixed window:
        # the storm rate is enormous (measured ~70 new processes/second), so a
        # generous deadline keeps this deterministic even under test-suite load.
        target = processes * 3
        seen: set[int] = set()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and len(seen) <= target:
            for process in list(getattr(pool, "_pool", ())):
                if process.pid is not None:
                    seen.add(process.pid)
            time.sleep(0.02)
        assert len(seen) > target, (
            "expected the shipped raising-initializer shape to churn workers; "
            f"saw only {len(seen)} distinct PIDs"
        )
    finally:
        pool.terminate()


def test_supervisor_bounds_worker_replacement() -> None:
    """The bounded-replacement policy turns an infinite loop into one error."""

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.name = f"worker-{pid}"
            self.exitcode = 1

        def is_alive(self) -> bool:
            return False

    class _FakePool:
        def __init__(self) -> None:
            self._pool = [_FakeProcess(1), _FakeProcess(2)]
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

    pool = _FakePool()
    supervisor = _PoolSupervisor(pool, expected_workers=2, max_replacements=3)
    supervisor.check()  # initial workers are not replacements

    next_pid = 3
    with pytest.raises(WorkerRespawnStormError) as caught:
        for _ in range(20):
            pool._pool = [_FakeProcess(next_pid), _FakeProcess(next_pid + 1)]
            next_pid += 2
            supervisor.check()
    assert caught.value.replacements > 3
    assert pool.terminated, "the pool must be shut down, not left churning"
    assert supervisor.replacements <= 12, "replacement must stop promptly"
    assert "render workers kept stopping" in caught.value.user_message.casefold()


def test_default_replacement_bound_is_small() -> None:
    assert 1 <= MAX_WORKER_REPLACEMENTS <= 32


# ---------------------------------------------------------------------------
# 5-7. Match terminates, activity clears, controls recover
# ---------------------------------------------------------------------------


def test_matcher_construction_fails_fast_before_spawning_any_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The primary fix: preflight in the parent, before Pool() exists.

    Asserted by proving no multiprocessing context is ever asked for a pool.
    """

    import core.matcher as matcher_module
    import core.renderer_selection as selection_module

    env = serum2_only_env(tmp_path)
    monkeypatch.setattr(selection_module, "ENV", env)
    monkeypatch.setattr(matcher_module, "ENV", env)

    created: list[int] = []

    class _Context:
        def Pool(self, processes, **_kwargs):  # noqa: N802 - mp API shape
            created.append(processes)
            raise AssertionError("no pool may be created when preflight fails")

    monkeypatch.setattr(matcher_module.mp, "get_context", lambda _name: _Context())

    with pytest.raises(RendererUnavailableError) as caught:
        matcher_module.AnalysisBySynthesisMatcher(
            processes=4, required_synths=("serum1",)
        )
    assert created == [], "the pool must never be created"
    assert "serum1" in str(caught.value)
    assert caught.value.user_message


def test_serum2_match_preflight_passes_without_any_serum1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 12: a Serum 2 match must work when Serum 1 VST2 is irrelevant.

    Construction is allowed to proceed past the renderer preflight; it then
    fails on model assets, which proves the renderer gate no longer blocks it.
    """

    import core.matcher as matcher_module
    import core.renderer_selection as selection_module
    from core.renderer_selection import preflight_renderers

    env = serum2_only_env(tmp_path)
    monkeypatch.setattr(selection_module, "ENV", env)
    monkeypatch.setattr(matcher_module, "ENV", env)

    preflight = preflight_renderers(("serum2",), env=env)
    assert preflight.fully_supported, (
        "a Serum 2 match must preflight cleanly on a machine with no Serum 1"
    )

    marker = {"created": False}

    class _Context:
        def Pool(self, processes, **_kwargs):  # noqa: N802
            marker["created"] = True
            raise _StopAfterPool

    class _StopAfterPool(Exception):
        pass

    monkeypatch.setattr(matcher_module.mp, "get_context", lambda _name: _Context())
    monkeypatch.setattr(
        matcher_module, "resolve_synthesis_assets", lambda: _StubAssets(tmp_path)
    )
    with pytest.raises(_StopAfterPool):
        matcher_module.AnalysisBySynthesisMatcher(
            processes=2, required_synths=("serum2",)
        )
    assert marker["created"], (
        "preflight must let a valid Serum 2 request reach pool creation"
    )


# ---------------------------------------------------------------------------
# 8-11. Diagnostics: root cause, aggregation, correlation, phase history
# ---------------------------------------------------------------------------


def test_repeated_worker_failures_are_aggregated_not_repeated(
    isolated_recorder,
) -> None:
    """PART 22: 200 identical tracebacks must become one plus a count."""

    for index in range(200):
        try:
            next(item for item in () if True)
        except StopIteration as exc:
            isolated_recorder.record_failure(
                "render-worker",
                "worker_init_failed",
                exc,
                message="render worker initialization failed",
                operation_id="op-1",
                phase="worker-pool-initialization",
                worker_id=f"render-{1000 + index}",
            )

    events = isolated_recorder.recent_events()
    failures = [item for item in events if item["event_type"] == "worker_init_failed"]
    assert len(failures) == 200, "each occurrence is still counted"
    with_full_traceback = [
        item for item in failures if "exception" in item.get("fields", {})
    ]
    assert len(with_full_traceback) == 1, (
        "exactly one full traceback should be retained, not 200"
    )

    groups = isolated_recorder.repeat_summary()
    assert len(groups) == 1
    group = groups[0]
    assert group["total_occurrences"] == 200
    assert group["first_timestamp"]
    assert group["latest_timestamp"]
    assert group["window_seconds"] >= 0.0
    assert len(group["worker_examples"]) <= 10
    assert group["first_full_event"]["fields"]["exception"]["chain"]
    # And the aggregated later events say so in one line.
    assert "repeated" in failures[-1]["message"]


def test_operation_id_propagates_into_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PART 10: one operation must be reconstructable across process boundaries."""

    import core.renderer_selection as selection_module
    from core.diagnostics import child_environment, recorder
    from core.matcher import _RENDER

    monkeypatch.setattr(selection_module, "ENV", serum2_only_env(tmp_path))
    correlation = child_environment("match-abc123", PATCHLAB_POOL_ID="pool-xyz")
    assert correlation["PATCHLAB_OPERATION_ID"] == "match-abc123"
    assert correlation["PATCHLAB_SESSION_ID"]

    _init_render_worker(str(tmp_path), None, ("serum1",), correlation)
    assert _RENDER["operation_id"] == "match-abc123"
    assert _RENDER["pool_id"] == "pool-xyz"
    assert _RENDER["init_failure"]["operation_id"] == "match-abc123"

    events = recorder().recent_events()
    assert any(item["operation_id"] == "match-abc123" for item in events)
    _RENDER.clear()


def test_worker_probe_reports_init_state() -> None:
    from core.matcher import _RENDER

    _RENDER.clear()
    _RENDER["worker_id"] = "render-42"
    _RENDER["hosted_synths"] = ("serum2",)
    probe = _worker_probe()
    assert probe["worker_id"] == "render-42"
    assert probe["hosted_synths"] == ["serum2"]
    assert probe["init_failure"] is None
    _RENDER.clear()


def test_worker_heartbeats_record_state_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_recorder
) -> None:
    """PART 14: alive-and-busy must be distinguishable from disappeared."""

    import core.renderer_selection as selection_module
    from core.matcher import _RENDER

    monkeypatch.setattr(selection_module, "ENV", serum2_only_env(tmp_path))
    _init_render_worker(str(tmp_path), None, ("serum1",), {})
    beats = [
        item
        for item in isolated_recorder.recent_events()
        if item["event_type"] == "heartbeat"
    ]
    states = [item["fields"]["worker_state"] for item in beats]
    assert "initializing-renderer" in states
    assert "initialization-failed" in states
    for beat in beats:
        assert beat["fields"]["worker_pid"] == os.getpid()
    _RENDER.clear()


def test_heartbeats_are_rate_limited(tmp_path: Path) -> None:
    """Heartbeats must not become log spam."""

    from core.operation_state import emit_worker_heartbeat

    recorder = DiagnosticRecorder(
        root=tmp_path / "hb", session_id="t", heartbeat_interval_seconds=60.0
    )
    set_recorder(recorder)
    try:
        for _ in range(500):
            emit_worker_heartbeat(
                operation_id="op", worker_id="w1", state="rendering"
            )
        beats = [
            item
            for item in recorder.recent_events()
            if item["event_type"] == "heartbeat"
        ]
        assert len(beats) == 1, f"expected rate limiting, got {len(beats)} heartbeats"
        assert beats[0]["fields"].get("suppressed_since_last") is None
    finally:
        recorder.close()
        set_recorder(None)
