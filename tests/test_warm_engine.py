"""The warm engine: cache, server loop, and the GUI-side connection.

Three layers, each tested on its own:

* ``WarmMatcherCache`` -- when a loaded matcher may be reused and when it must go;
* ``scripts.engine_server`` -- the line protocol a long-lived worker speaks;
* ``EngineConnection`` + the runners -- real ``QProcess`` talking to a fake
  engine, covering crashes, cancel, fallback to one-shot workers and the idle-exit
  race.  Nothing here loads a model or a plug-in.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication

import core.match_workflow as match_workflow
from core.match_workflow import WarmMatcherCache


# --------------------------------------------------------------------------------
# WarmMatcherCache
# --------------------------------------------------------------------------------


class _FakeMatcher:
    instances: list["_FakeMatcher"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.embedder = kwargs.get("embedder") or object()
        self.usable = True
        self.closed = False
        self.operations: list[str] = []
        self.tracker = "stale"
        _FakeMatcher.instances.append(self)

    def is_usable(self) -> bool:
        return self.usable and not self.closed

    def begin_operation(self, operation_id: str) -> None:
        self.operations.append(operation_id)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> WarmMatcherCache:
    _FakeMatcher.instances = []
    monkeypatch.setattr(match_workflow, "AnalysisBySynthesisMatcher", _FakeMatcher)
    return WarmMatcherCache(embedder=object())


def _acquire(cache: WarmMatcherCache, synth: str = "serum2", processes: int = 4, op: str = "op"):
    return cache.acquire(
        processes=processes,
        deterministic_render_dispatch=False,
        required_synths=(synth,),
        operation_id=op,
    )


def test_a_healthy_matcher_is_reused_for_the_next_match(cache: WarmMatcherCache) -> None:
    first = _acquire(cache, op="one")
    cache.release(first, healthy=True)
    second = _acquire(cache, op="two")
    assert second is first and len(_FakeMatcher.instances) == 1
    assert first.operations == ["two"], "a reused matcher is re-labelled for the new operation"
    assert cache.reuse_count == 1


def test_the_clap_model_is_shared_with_the_matcher(cache: WarmMatcherCache) -> None:
    matcher = _acquire(cache)
    assert matcher.kwargs["embedder"] is cache.embedder


def test_the_first_matcher_loads_clap_itself_and_the_cache_then_shares_it(monkeypatch) -> None:
    """So the render workers spawn while CLAP loads, exactly as in a one-shot Match."""

    _FakeMatcher.instances = []
    monkeypatch.setattr(match_workflow, "AnalysisBySynthesisMatcher", _FakeMatcher)
    fresh = WarmMatcherCache()
    first = _acquire(fresh)
    assert first.kwargs["embedder"] is None, "the matcher was left to load CLAP in parallel with its workers"
    assert fresh.embedder is first.embedder, "and the export verifier now shares that one model"
    fresh.release(first, healthy=True)
    first.usable = False
    assert _acquire(fresh).kwargs["embedder"] is first.embedder, "a rebuilt matcher keeps the loaded model"


def test_a_different_generation_or_worker_count_rebuilds(cache: WarmMatcherCache) -> None:
    first = _acquire(cache, "serum2", 4)
    cache.release(first, healthy=True)
    other_synth = _acquire(cache, "serum1", 4)
    assert other_synth is not first and first.closed
    cache.release(other_synth, healthy=True)
    other_count = _acquire(cache, "serum1", 6)
    assert other_count is not other_synth and other_synth.closed


def test_a_failed_match_discards_the_matcher(cache: WarmMatcherCache) -> None:
    first = _acquire(cache)
    cache.release(first, healthy=False)
    assert first.closed and cache.matcher is None
    assert _acquire(cache) is not first


def test_a_damaged_pool_is_never_reused(cache: WarmMatcherCache) -> None:
    first = _acquire(cache)
    cache.release(first, healthy=True)
    first.usable = False  # e.g. a render worker died between matches
    second = _acquire(cache)
    assert second is not first and first.closed


def test_releasing_an_unknown_matcher_closes_it(cache: WarmMatcherCache) -> None:
    stranger = _FakeMatcher()
    cache.release(stranger, healthy=True)
    assert stranger.closed


def test_close_releases_everything(cache: WarmMatcherCache) -> None:
    matcher = _acquire(cache)
    cache.close()
    assert matcher.closed and cache.matcher is None


def test_without_a_cache_a_match_still_builds_and_closes_its_own_matcher() -> None:
    matcher = SimpleNamespace(tracker="t", closed=False, close=lambda: setattr(matcher, "closed", True))
    match_workflow._retire_matcher(matcher, None, healthy=True)
    assert matcher.closed and matcher.tracker is None


# --------------------------------------------------------------------------------
# The engine server's line protocol
# --------------------------------------------------------------------------------


def _serve(monkeypatch: pytest.MonkeyPatch, capsys, commands: list[dict], **handlers):
    import scripts.engine_server as server

    for name, function in handlers.items():
        monkeypatch.setattr(server.Engine, name, function)
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("".join(json.dumps(item) + "\n" for item in commands))
    )
    code = server.main()
    return code, capsys.readouterr().out.splitlines()


def test_each_job_ends_with_a_done_line_carrying_its_id_and_exit_code(monkeypatch, capsys) -> None:
    code, lines = _serve(
        monkeypatch,
        capsys,
        [
            {"id": "1", "kind": "match", "argv": ["a.wav"]},
            {"id": "2", "kind": "export", "argv": ["r.json", "o"]},
        ],
        run_match=lambda self, argv: (print("MATCH_RESULT=/r.json"), 0)[1],
        run_export=lambda self, argv: (print("EXPORT_ERROR=disk full"), 1)[1],
    )
    assert code == 0
    assert lines == [
        "MATCH_RESULT=/r.json",
        'ENGINE_JOB_DONE={"id": "1", "exit_code": 0}',
        "EXPORT_ERROR=disk full",
        'ENGINE_JOB_DONE={"id": "2", "exit_code": 1}',
    ]


def test_jobs_get_their_own_operation_id(monkeypatch, capsys) -> None:
    seen: list[str] = []
    _serve(
        monkeypatch,
        capsys,
        [
            {"id": "1", "kind": "match", "argv": [], "operation_id": "match-aaa"},
            {"id": "2", "kind": "match", "argv": [], "operation_id": "match-bbb"},
        ],
        run_match=lambda self, argv: (seen.append(os.environ["PATCHLAB_OPERATION_ID"]), 0)[1],
    )
    assert seen == ["match-aaa", "match-bbb"]


def test_a_job_that_raises_fails_cleanly_and_discards_the_warm_matcher(monkeypatch, capsys) -> None:
    discarded: list[bool] = []

    def boom(self, argv):
        self.cache.discard = lambda: discarded.append(True)
        raise RuntimeError("renderer vanished")

    code, lines = _serve(
        monkeypatch,
        capsys,
        [{"id": "1", "kind": "match", "argv": []}, {"id": "2", "kind": "warm", "argv": []}],
        run_match=boom,
        warm=lambda self, argv: 0,
    )
    assert code == 0, "one bad job must not take the engine down"
    assert lines[0] == "MATCH_ERROR=RuntimeError: renderer vanished"
    assert lines[1] == 'ENGINE_JOB_DONE={"id": "1", "exit_code": 1}'
    assert lines[2] == 'ENGINE_JOB_DONE={"id": "2", "exit_code": 0}'
    assert discarded[:1] == [True], "the failed job's matcher is discarded straight away"


def test_unknown_job_kinds_are_reported_not_ignored(monkeypatch, capsys) -> None:
    _, lines = _serve(monkeypatch, capsys, [{"id": "9", "kind": "dance", "argv": []}])
    assert lines[0].startswith("ENGINE_ERROR=unknown job kind")
    assert lines[1] == 'ENGINE_JOB_DONE={"id": "9", "exit_code": 2}'


def test_shutdown_stops_before_later_commands(monkeypatch, capsys) -> None:
    ran: list[str] = []
    _, lines = _serve(
        monkeypatch,
        capsys,
        [{"id": "1", "kind": "shutdown"}, {"id": "2", "kind": "match", "argv": []}],
        run_match=lambda self, argv: (ran.append("late"), 0)[1],
    )
    assert ran == [] and lines == []


def test_the_engine_exits_when_its_parent_closes_stdin(monkeypatch, capsys) -> None:
    code, lines = _serve(monkeypatch, capsys, [])
    assert code == 0 and lines == []


def test_the_idle_limit_is_configurable_with_a_floor(monkeypatch) -> None:
    import scripts.engine_server as server

    monkeypatch.setenv("PATCHLAB_ENGINE_IDLE_SECONDS", "90")
    assert server.idle_limit_seconds() == 90.0
    monkeypatch.setenv("PATCHLAB_ENGINE_IDLE_SECONDS", "1")
    assert server.idle_limit_seconds() == 5.0
    monkeypatch.setenv("PATCHLAB_ENGINE_IDLE_SECONDS", "soon")
    assert server.idle_limit_seconds() == server.DEFAULT_IDLE_SECONDS


def test_the_engine_is_a_registered_worker() -> None:
    from core.worker_runtime import WORKER_ENTRY_POINTS

    assert WORKER_ENTRY_POINTS["engine"].module == "scripts.engine_server"


# --------------------------------------------------------------------------------
# EngineConnection + runners against a fake engine process (real QProcess)
# --------------------------------------------------------------------------------

FAKE_ENGINE = r'''
import json, os, sys, time
home = os.environ["FAKE_ENGINE_HOME"]
with open(os.path.join(home, "launches.txt"), "a") as handle:
    handle.write("launch\n")
if os.environ.get("FAKE_ENGINE_NEVER_READY") == "1":
    sys.exit(3)
print("PATCHLAB_WORKER_READY=engine", flush=True)
for raw in sys.stdin:
    command = json.loads(raw)
    with open(os.path.join(home, "commands.jsonl"), "a") as handle:
        handle.write(raw)
    kind, argv, job = command["kind"], command["argv"], command["id"]
    if kind == "shutdown":
        break
    tag = argv[0] if argv else ""
    marker = os.path.join(home, "idled")
    if tag == "idle-once" and not os.path.exists(marker):
        open(marker, "w").close()
        print("ENGINE_IDLE_EXIT=1", flush=True)
        os._exit(0)
    if tag == "crash":
        os._exit(7)
    if tag == "slow":
        time.sleep(60)
    code = 0
    if kind == "match":
        print('MATCH_PROGRESS={"phase":"decoding","evaluations":0}', flush=True)
        if tag == "fail":
            print("MATCH_ERROR=no confident sound", flush=True)
            code = 1
        else:
            print("MATCH_RESULT=/tmp/result.json", flush=True)
    elif kind == "export":
        print('EXPORT_RESULT={"path": "/tmp/saved.SerumPreset"}', flush=True)
    print("ENGINE_JOB_DONE=" + json.dumps({"id": job, "exit_code": code}), flush=True)
'''


@pytest.fixture
def qt():
    # A QApplication, never a bare QCoreApplication: the GUI tests in this suite
    # reuse whichever application already exists, and widgets abort the process
    # if all they find is a core application.
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _pump(condition, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


class Rig:
    """An EngineConnection and both runners wired to a fake engine process."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.workers as workers

        self.home = tmp_path / "engine-home"
        self.home.mkdir()
        script = tmp_path / "fake_engine.py"
        script.write_text(FAKE_ENGINE)
        monkeypatch.setenv("FAKE_ENGINE_HOME", str(self.home))
        monkeypatch.setenv("PATCHLAB_WARM_ENGINE", "1")
        monkeypatch.setattr(
            workers,
            "worker_invocation",
            lambda name, arguments=(): (sys.executable, [str(script), *map(str, arguments)]),
        )
        self.one_shot: list[tuple[str, list[str]]] = []
        for cls, name in ((workers.MatchProcessRunner, "match"), (workers.ExportProcessRunner, "export")):
            monkeypatch.setattr(
                cls,
                "_start_worker",
                lambda runner, worker, arguments, _name=name: self.one_shot.append((worker, list(arguments))),
            )
        self.engine = workers.EngineConnection()
        self.match = workers.MatchProcessRunner(engine=self.engine)
        self.export = workers.ExportProcessRunner(engine=self.engine)
        self.matches: list[str] = []
        self.match_errors: list[str] = []
        self.progress: list[dict] = []
        self.exports: list[dict] = []
        self.export_errors: list[str] = []
        self.match.completed.connect(self.matches.append)
        self.match.failed.connect(self.match_errors.append)
        self.match.progress.connect(self.progress.append)
        self.export.completed.connect(self.exports.append)
        self.export.failed.connect(self.export_errors.append)

    @property
    def launches(self) -> int:
        path = self.home / "launches.txt"
        return len(path.read_text().split()) if path.is_file() else 0

    def commands(self) -> list[dict]:
        path = self.home / "commands.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []

    def start_match(self, tag: str = "ok", **kwargs) -> None:
        self.match.start(Path(tag), target_synth="serum2", budget="quick", offset=0.0, **kwargs)

    def start_export(self, tag: str = "ok") -> None:
        self.export.start(Path(tag), Path("/tmp/out.SerumPreset"))

    def close(self) -> None:
        self.engine.stop()


@pytest.fixture
def rig(qt, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    made = Rig(tmp_path, monkeypatch)
    yield made
    made.close()


def test_a_match_runs_through_the_engine_and_reports_like_a_one_shot_worker(rig: Rig) -> None:
    rig.start_match()
    assert rig.match.running, "the runner is busy as soon as the job is submitted"
    assert _pump(lambda: rig.matches)
    assert rig.matches == ["/tmp/result.json"]
    assert [item["phase"] for item in rig.progress] == ["decoding"]
    assert not rig.match.running
    assert rig.one_shot == [], "no one-shot worker was started"


def test_match_and_export_share_one_engine_process_and_run_in_order(rig: Rig) -> None:
    # A save is quick, so a Match submitted right behind it simply queues.
    rig.start_export()
    rig.start_match()
    assert _pump(lambda: rig.matches and rig.exports)
    assert rig.exports == [{"path": "/tmp/saved.SerumPreset"}]
    assert [item["kind"] for item in rig.commands()] == ["export", "match"]
    assert rig.launches == 1 and rig.one_shot == []


def test_the_second_match_reuses_the_running_engine(rig: Rig) -> None:
    rig.start_match()
    assert _pump(lambda: len(rig.matches) == 1)
    rig.start_match()
    assert _pump(lambda: len(rig.matches) == 2)
    assert rig.launches == 1


def test_an_engine_error_line_fails_the_job_but_keeps_the_engine(rig: Rig) -> None:
    rig.start_match("fail")
    assert _pump(lambda: rig.match_errors)
    assert rig.match_errors == ["no confident sound"]
    rig.start_match()
    assert _pump(lambda: rig.matches)
    assert rig.launches == 1 and rig.engine.usable


def test_each_job_carries_its_own_operation_id(rig: Rig) -> None:
    rig.start_match()
    assert _pump(lambda: rig.matches)
    first_id = rig.match.operation_id
    rig.start_match()
    assert _pump(lambda: len(rig.matches) == 2)
    ids = [item["operation_id"] for item in rig.commands()]
    assert len(set(ids)) == 2 and all(ids) and ids[0] == first_id and ids[1] == rig.match.operation_id


def test_a_crashing_engine_fails_the_job_once_and_is_replaced(rig: Rig) -> None:
    rig.start_match("crash")
    assert _pump(lambda: rig.match_errors)
    assert len(rig.match_errors) == 1 and "stopped unexpectedly" in rig.match_errors[0]
    assert not rig.match.running
    rig.start_match()
    assert _pump(lambda: rig.matches)
    assert rig.launches == 2, "the next job got a fresh engine"


def test_repeated_crashes_switch_to_one_shot_workers(rig: Rig) -> None:
    for _ in range(2):
        rig.start_match("crash")
        assert _pump(lambda: len(rig.match_errors) == _ + 1)
    assert not rig.engine.usable
    rig.start_match()
    assert rig.one_shot and rig.one_shot[-1][0] == "match", "back to the original worker per job"
    assert rig.launches == 2


def test_cancelling_a_running_job_stops_it_without_counting_a_crash(rig: Rig) -> None:
    rig.start_match("slow")
    assert _pump(lambda: rig.commands())
    rig.match.cancel()
    assert _pump(lambda: rig.match_errors)
    assert rig.match_errors == ["Match stopped."]
    assert rig.engine.usable and not rig.match.running
    rig.start_match()
    assert _pump(lambda: rig.matches)


def test_cancelling_a_queued_job_just_removes_it(rig: Rig) -> None:
    rig.start_match("slow")
    rig.start_export()
    assert _pump(lambda: rig.commands())
    rig.export.cancel()
    assert not rig.export.running and not rig.engine.has_job(rig.export)
    rig.match.cancel()
    assert _pump(lambda: rig.match_errors)
    assert rig.exports == [] and rig.export_errors == []


def test_disabled_by_environment_uses_the_original_workers(qt, tmp_path, monkeypatch) -> None:
    made = Rig(tmp_path, monkeypatch)
    monkeypatch.setenv("PATCHLAB_WARM_ENGINE", "0")
    import app.workers as workers

    engine = workers.EngineConnection()
    runner = workers.MatchProcessRunner(engine=engine)
    runner.start(Path("a.wav"), target_synth="serum2", budget="quick", offset=0.0)
    assert not engine.usable and not engine.process_running
    assert made.one_shot and made.one_shot[-1][0] == "match"


def test_an_engine_that_cannot_start_falls_back_to_a_one_shot_worker(rig: Rig, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_ENGINE_NEVER_READY", "1")
    rig.start_match()
    assert _pump(lambda: rig.one_shot)
    assert rig.one_shot[-1][0] == "match"
    assert rig.engine._failures == 1 and rig.match_errors == []


def test_an_idle_exit_at_the_moment_a_job_arrives_is_retried_transparently(rig: Rig) -> None:
    rig.start_match("idle-once")
    assert _pump(lambda: rig.matches, timeout=40)
    assert rig.matches == ["/tmp/result.json"] and rig.match_errors == []
    assert rig.launches == 2 and rig.engine._failures == 0


def test_factory_only_matches_never_use_the_engine(rig: Rig) -> None:
    rig.start_match(factory_only=True)
    assert rig.one_shot and rig.one_shot[-1][0] == "match"
    assert rig.engine.process_running is False


def test_warming_loads_the_engine_once_and_touches_no_runner(rig: Rig) -> None:
    rig.engine.warm("serum2")
    assert _pump(lambda: not rig.engine.busy and rig.commands())
    assert [item["kind"] for item in rig.commands()] == ["warm"]
    rig.engine.warm("serum2")
    assert not rig.engine.busy, "already warm for this generation: nothing is sent"
    assert rig.launches == 1 and rig.matches == [] and rig.match_errors == []


def test_stopping_the_engine_ends_the_process(rig: Rig) -> None:
    rig.start_match()
    assert _pump(lambda: rig.matches)
    assert rig.engine.process_running
    rig.engine.stop()
    assert not rig.engine.process_running


def test_an_export_does_not_wait_behind_a_running_match(rig: Rig) -> None:
    rig.start_match("slow")
    assert _pump(lambda: rig.commands())
    rig.start_export()
    assert rig.one_shot and rig.one_shot[-1][0] == "export", "the save ran as its own worker"
    assert [item["kind"] for item in rig.commands()] == ["match"]
    rig.match.cancel()
    assert _pump(lambda: rig.match_errors)


def test_an_export_after_the_match_finished_uses_the_engine(rig: Rig) -> None:
    rig.start_match()
    assert _pump(lambda: rig.matches)
    rig.start_export()
    assert _pump(lambda: rig.exports)
    assert rig.one_shot == [] and rig.launches == 1
