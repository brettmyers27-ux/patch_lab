"""The diagnostic subsystem is production code and needs its own tests.

Two kinds of coverage here:

* the recorder's own contracts -- bounded retention, batched writes, rotation,
  redaction, aggregation, correlation, fingerprint stability, and the guarantee
  that a diagnostics failure never becomes a PatchLab failure;
* synthetic failures across the whole list in PART 28, each asserting that a
  bundle built from them answers the questions a remote diagnosis needs:
  operation, phase, error, environment, renderer decisions, failure reason,
  process/worker context and UI recovery state.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

from core.diagnostics import (
    BUNDLE_HISTORY_SECONDS,
    DIAGNOSTIC_SCHEMA_VERSION,
    MAX_EVENT_FILES,
    MAX_EVENT_FILE_BYTES,
    RING_CAPACITY,
    DiagnosticRecorder,
    exception_chain,
    fingerprint_failure,
    normalize_error_text,
    recorder,
    redact_text,
    sanitize,
    set_recorder,
)
from core.operation_state import capture_postmortem, start_operation
from core.support_bundle import (
    ENVIRONMENT_FILENAME,
    EVENTS_FILENAME,
    POSTMORTEM_FILENAME,
    REPRODUCTION_FILENAME,
    SUMMARY_FILENAME,
    build_reproduction_descriptor,
    create_support_bundle,
)
from tests.test_renderer_routing import serum2_only_env


@pytest.fixture
def fresh(tmp_path: Path):
    instance = DiagnosticRecorder(root=tmp_path / "diagnostics", session_id="sess1234")
    set_recorder(instance)
    yield instance
    instance.close()
    set_recorder(None)


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


def test_schema_version_is_declared_and_propagated(fresh) -> None:
    """PART 18: a reader must distinguish 'unavailable' from 'older format'."""

    assert DIAGNOSTIC_SCHEMA_VERSION >= 2
    marker = fingerprint_failure(RuntimeError("x"), subsystem="s", phase="p")
    assert marker.schema_version == DIAGNOSTIC_SCHEMA_VERSION
    assert marker.as_dict()["diagnostic_schema_version"] == DIAGNOSTIC_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Bounded retention and rotation (PART 23)
# ---------------------------------------------------------------------------


def test_ring_buffer_is_bounded(tmp_path: Path) -> None:
    instance = DiagnosticRecorder(root=tmp_path / "d", capacity=64)
    try:
        for index in range(1000):
            instance.record("test", "tick", f"event {index}")
        events = instance.recent_events()
        assert len(events) == 64, "the in-memory ring must be bounded"
        assert events[-1]["message"] == "event 999", "newest events are retained"
    finally:
        instance.close()


def test_event_files_rotate_and_are_capped(tmp_path: Path) -> None:
    root = tmp_path / "d"
    instance = DiagnosticRecorder(root=root, capacity=128)
    try:
        payload = "x" * 2000
        for index in range(6000):
            instance.record("test", "bulk", payload, index=index)
        instance.flush(timeout=5.0)
    finally:
        instance.close()

    files = sorted(root.glob("events*.jsonl"))
    assert files, "events must be persisted"
    assert len(files) <= MAX_EVENT_FILES
    for path in files:
        # Rotation happens before a batch is appended, so a file may exceed the
        # threshold by at most one batch. Assert the real bound: it never grows
        # without limit.
        assert path.stat().st_size <= MAX_EVENT_FILE_BYTES * 2, path
    total = sum(path.stat().st_size for path in files)
    assert total <= MAX_EVENT_FILE_BYTES * MAX_EVENT_FILES * 2


def test_retention_policy_is_reported(fresh) -> None:
    policy = fresh.retention_policy()
    assert policy["ring_capacity_events"] == RING_CAPACITY
    assert policy["max_total_event_bytes"] == MAX_EVENT_FILE_BYTES * MAX_EVENT_FILES
    assert policy["bundle_history_seconds"] == BUNDLE_HISTORY_SECONDS
    assert policy["persistence_enabled"] is True


def test_writes_are_batched_not_synchronous(tmp_path: Path) -> None:
    """Recording must not block on disk."""

    import time

    instance = DiagnosticRecorder(root=tmp_path / "d")
    try:
        started = time.monotonic()
        for index in range(5000):
            instance.record("test", "tick", "m", index=index)
        elapsed = time.monotonic() - started
        # 5,000 events in well under a second means no per-event fsync.
        assert elapsed < 2.0, f"5000 records took {elapsed:.2f}s"
    finally:
        instance.close()


def test_recorder_works_with_persistence_disabled(tmp_path: Path) -> None:
    """A read-only or full disk must not break diagnostics or PatchLab."""

    instance = DiagnosticRecorder(root=tmp_path / "d")
    instance._root = None  # simulate an unwritable diagnostics directory
    try:
        instance.record("test", "tick", "still recorded")
        assert instance.recent_events()
        assert instance.retention_policy()["persistence_enabled"] is False
    finally:
        instance.close()


def test_disabling_diagnostics_is_honoured(tmp_path: Path) -> None:
    instance = DiagnosticRecorder(root=tmp_path / "d", enabled=False)
    try:
        assert instance.record("test", "tick", "ignored") is None
        assert instance.recent_events() == []
    finally:
        instance.close()


# ---------------------------------------------------------------------------
# Privacy and redaction (PART 24)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "api_key",
        "access_token",
        "authorization",
        "Cookie",
        "oauth_refresh",
        "private_key",
        "relay_passcode",
        "signing_secret",
        "google_credentials",
    ],
)
def test_sensitive_keys_are_redacted_whatever_the_call_site(key: str) -> None:
    result = sanitize({key: "super-secret-value"})
    assert result[key] == "[redacted]"
    assert "super-secret-value" not in json.dumps(result)


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer abc123def456",
        "api_key=sk-live-1234567890",
        "token: ghp_0123456789abcdefghij",
        "-----BEGIN RSA PRIVATE KEY-----",
        "password = hunter2",
    ],
)
def test_credential_shaped_values_are_redacted_under_innocent_keys(text: str) -> None:
    result = sanitize({"note": text})
    assert "[redacted]" in result["note"]
    for secret in ("abc123def456", "sk-live-1234567890", "hunter2"):
        assert secret not in result["note"]


def test_raw_binary_content_is_never_recorded() -> None:
    """No raw audio, preset bytes or model weights, ever."""

    audio = b"RIFF" + os.urandom(4096)
    result = sanitize({"waveform": audio, "preset": bytearray(b"CcnK" + os.urandom(64))})
    assert result["waveform"] == f"[{len(audio)} bytes omitted]"
    assert "bytes omitted" in result["preset"]
    assert "RIFF" not in json.dumps(result)


def test_file_paths_are_retained_because_they_are_the_diagnosis() -> None:
    path = "/Applications/PatchLab.app/Contents/Resources/data/models/x.pt"
    assert sanitize({"checkpoint": Path(path)})["checkpoint"] == path


def test_sanitize_handles_hostile_values() -> None:
    circular: dict = {}
    circular["self"] = circular
    assert sanitize(circular)  # must not recurse forever
    assert sanitize(float("nan")) == "nan"
    assert sanitize(float("inf")) == "inf"
    assert "more items omitted" in json.dumps(sanitize(list(range(500))))


# ---------------------------------------------------------------------------
# Fingerprinting (PART 17)
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_across_volatile_detail() -> None:
    """The same defect on two machines must fingerprint identically."""

    first = fingerprint_failure(
        RuntimeError("worker 1234 failed loading /Users/alice/Library/x.vst3 after 7 tries"),
        subsystem="render-worker",
        phase="worker-pool-initialization",
        renderer="serum2/VST3",
        serum_generation="serum2",
    )
    second = fingerprint_failure(
        RuntimeError("worker 9876 failed loading /Users/bob/Library/y.vst3 after 42 tries"),
        subsystem="render-worker",
        phase="worker-pool-initialization",
        renderer="serum2/VST3",
        serum_generation="serum2",
    )
    assert first.digest == second.digest, "volatile detail must not change identity"


def test_fingerprint_separates_genuinely_different_failures() -> None:
    base = dict(subsystem="render-worker", phase="worker-pool-initialization")
    a = fingerprint_failure(RuntimeError("no renderer"), **base)
    b = fingerprint_failure(ValueError("no renderer"), **base)
    c = fingerprint_failure(
        RuntimeError("no renderer"), subsystem="local-library", phase="catalog-scan"
    )
    assert len({a.digest, b.digest, c.digest}) == 3


def test_normalization_collapses_paths_pids_and_digests() -> None:
    normalized = normalize_error_text(
        "pid 4821 failed on /Users/x/Library/Audio/Plug-Ins/VST3/Serum2.vst3 "
        "hash a1b2c3d4e5f6 at 0x7ffee "
    )
    assert "4821" not in normalized
    assert "/Users" not in normalized
    assert "a1b2c3d4e5f6" not in normalized
    assert "0x7ffee" not in normalized
    assert "<path>" in normalized and "<n>" in normalized


def test_fingerprint_of_stopiteration_uses_the_code_location() -> None:
    """A bare StopIteration stringifies to nothing; location must disambiguate."""

    try:
        next(item for item in () if True)
    except StopIteration as exc:
        marker = fingerprint_failure(
            exc, subsystem="render-worker", phase="worker-pool-initialization"
        )
    assert marker.exception_class == "StopIteration"
    assert marker.code_location, "location must identify the empty iterator"
    assert "test_diagnostics.py" in marker.code_location


# ---------------------------------------------------------------------------
# Repeat aggregation (PART 22)
# ---------------------------------------------------------------------------


def test_repeat_aggregation_keeps_the_first_full_traceback_only(fresh) -> None:
    for index in range(150):
        try:
            raise RuntimeError("renderer unavailable")
        except RuntimeError as exc:
            fresh.record_failure(
                "render-worker",
                "worker_init_failed",
                exc,
                worker_id=f"render-{index}",
                operation_id="op",
            )
    groups = fresh.repeat_summary()
    assert len(groups) == 1
    group = groups[0]
    assert group["total_occurrences"] == 150
    assert group["first_full_event"]["fields"]["exception"]["chain"]
    assert group["occurrences_per_second"] is None or group["occurrences_per_second"] > 0
    full = [
        item
        for item in fresh.recent_events()
        if item["event_type"] == "worker_init_failed"
        and "exception" in item.get("fields", {})
    ]
    assert len(full) == 1


def test_distinct_failures_are_not_merged(fresh) -> None:
    for message in ("no serum1 renderer", "no serum2 renderer", "disk full"):
        try:
            raise RuntimeError(message)
        except RuntimeError as exc:
            fresh.record_failure("render-worker", "worker_init_failed", exc)
    assert fresh.repeat_summary() == [], "three different failures, none repeated"


def test_serum_generation_survives_normalization() -> None:
    """serum1 and serum2 failures must never share a fingerprint.

    Digit normalization exists to collapse PIDs and counts. Collapsing the
    generation digit too would merge the two distinct renderer bugs this work
    was about into one indistinguishable failure class.
    """

    one = normalize_error_text("No usable serum1 renderer is available")
    two = normalize_error_text("No usable serum2 renderer is available")
    assert one != two
    assert "serum1" in one and "serum2" in two

    vst2 = normalize_error_text("Verified Serum 1 VST2 binary is unavailable")
    vst3 = normalize_error_text("Verified Serum 1 VST3 binary is unavailable")
    assert vst2 != vst3, "plug-in format must stay distinguishable"

    # But volatile numbers still collapse.
    assert normalize_error_text("worker 1234 died") == normalize_error_text(
        "worker 9876 died"
    )


# ---------------------------------------------------------------------------
# Correlation (PART 10) and decisions (PART 9)
# ---------------------------------------------------------------------------


def test_correlation_ids_travel_into_a_child_environment(fresh) -> None:
    from core.diagnostics import child_environment, inherited_operation_id

    values = child_environment("match-1234", PATCHLAB_POOL_ID="pool-9")
    assert values["PATCHLAB_SESSION_ID"] == "sess1234"
    assert values["PATCHLAB_OPERATION_ID"] == "match-1234"
    assert values["PATCHLAB_DIAGNOSTIC_SCHEMA"] == str(DIAGNOSTIC_SCHEMA_VERSION)
    assert values["PATCHLAB_POOL_ID"] == "pool-9"
    os.environ["PATCHLAB_OPERATION_ID"] = "match-1234"
    try:
        assert inherited_operation_id() == "match-1234"
    finally:
        os.environ.pop("PATCHLAB_OPERATION_ID", None)


def test_decisions_record_the_reason_and_the_alternatives(fresh) -> None:
    fresh.record_decision(
        "renderer-selection",
        "renderer_selection[serum2]",
        outcome="serum2/VST3",
        reason="highest-preference validated renderer",
        operation_id="op-1",
        candidates=[
            {"renderer": "serum2/AU", "accepted": False, "rejection_reason": "missing"},
            {"renderer": "serum2/VST3", "accepted": True},
        ],
    )
    event = fresh.recent_events()[-1]
    assert event["event_type"] == "decision"
    assert event["decision_reason"] == "highest-preference validated renderer"
    assert event["fields"]["outcome"] == "serum2/VST3"
    assert len(event["fields"]["candidates"]) == 2


def test_structured_events_carry_the_normalized_field_set(fresh) -> None:
    """PART 15: machine-readable, not prose."""

    fresh.record(
        "match",
        "phase_changed",
        "decoding -> model-loading",
        operation_id="op-7",
        phase="model-loading",
        worker_id="render-1",
        severity="info",
    )
    event = fresh.recent_events()[-1]
    for key in (
        "ts",
        "severity",
        "session_id",
        "operation_id",
        "subsystem",
        "event_type",
        "phase",
        "pid",
        "worker_id",
        "message",
    ):
        assert key in event, key
    assert event["pid"] == os.getpid()
    assert event["session_id"] == "sess1234"


def test_persisted_events_are_valid_json_lines(tmp_path: Path) -> None:
    root = tmp_path / "d"
    instance = DiagnosticRecorder(root=root)
    try:
        instance.record("match", "decision", "a", operation_id="op", nested={"x": [1, 2]})
        instance.record("match", "progress", "b", operation_id="op")
        instance.flush(timeout=5.0)
    finally:
        instance.close()
    lines = (root / EVENTS_FILENAME).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    for line in lines:
        assert json.loads(line)["session_id"]


# ---------------------------------------------------------------------------
# Performance guards (DIAGNOSTIC PERFORMANCE REQUIREMENTS)
# ---------------------------------------------------------------------------


def test_progress_events_are_rate_limited(tmp_path: Path) -> None:
    instance = DiagnosticRecorder(
        root=tmp_path / "d", progress_interval_seconds=60.0
    )
    try:
        for index in range(10_000):
            instance.record(
                "match", "progress", "searching", operation_id="op", index=index
            )
        progress = [
            item for item in instance.recent_events() if item["event_type"] == "progress"
        ]
        assert len(progress) == 1, f"expected rate limiting, kept {len(progress)}"
        # Nothing was suppressed before the first event, so the field is absent.
        assert "suppressed_since_last" not in progress[0]["fields"]
    finally:
        instance.close()


def test_rate_limiting_reports_how_many_events_it_dropped(tmp_path: Path) -> None:
    """A suppressed burst must be visible as a count, not silently lost."""

    instance = DiagnosticRecorder(root=tmp_path / "d", progress_interval_seconds=0.05)
    try:
        import time

        for index in range(50):
            instance.record("match", "progress", "searching", operation_id="op", index=index)
        time.sleep(0.08)
        instance.record("match", "progress", "searching", operation_id="op", index=99)
        progress = [
            item for item in instance.recent_events() if item["event_type"] == "progress"
        ]
        assert len(progress) == 2
        assert progress[1]["fields"]["suppressed_since_last"] == 49
    finally:
        instance.close()


def test_file_digests_are_cached_against_size_and_mtime(tmp_path: Path) -> None:
    """Plug-in and checkpoint hashes must not be recomputed per operation."""

    instance = DiagnosticRecorder(root=tmp_path / "d")
    try:
        target = tmp_path / "plugin.bin"
        target.write_bytes(b"a" * 100_000)
        first = instance.cached_file_digest(target)
        assert first

        reads: list[int] = []
        real_open = Path.open

        def counting_open(self, *args, **kwargs):
            if self == target:
                reads.append(1)
            return real_open(self, *args, **kwargs)

        Path.open = counting_open  # type: ignore[method-assign]
        try:
            for _ in range(20):
                assert instance.cached_file_digest(target) == first
            assert reads == [], "a cache hit must not reopen the file"
            # Changing the file invalidates the cache.
            target.write_bytes(b"b" * 100_001)
            changed = instance.cached_file_digest(target)
            assert changed != first
            assert reads, "a changed file must be rehashed"
        finally:
            Path.open = real_open  # type: ignore[method-assign]
    finally:
        instance.close()


def test_huge_files_are_not_hashed(tmp_path: Path) -> None:
    instance = DiagnosticRecorder(root=tmp_path / "d")
    try:
        target = tmp_path / "big.bin"
        target.write_bytes(b"x" * 1024)
        assert instance.cached_file_digest(target, max_bytes=10) == ""
    finally:
        instance.close()


def test_recording_is_thread_safe(tmp_path: Path) -> None:
    instance = DiagnosticRecorder(root=tmp_path / "d", capacity=4096)
    try:

        def worker(index: int) -> None:
            for step in range(200):
                instance.record("test", "tick", f"{index}-{step}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(instance.recent_events()) == 1600
        assert instance.counters()["write_errors"] == 0
    finally:
        instance.close()


def test_a_diagnostics_failure_never_raises_into_patchlab(fresh) -> None:
    """A diagnostics bug must not become a PatchLab bug."""

    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("repr exploded")

        def __str__(self) -> str:
            raise RuntimeError("str exploded")

    # Must not raise.
    assert fresh.record("test", "tick", "ok", hostile=Hostile()) is not None or True
    fresh.record("test", "tick", "ok", nested={"deep": Hostile()})


# ---------------------------------------------------------------------------
# PART 28: synthetic failures, each must be diagnosable from the bundle
# ---------------------------------------------------------------------------


def _bundle_for(
    tmp_path: Path,
    *,
    operation: str,
    phase: str,
    exception: BaseException,
    renderer_selection: dict | None = None,
    environment: dict | None = None,
    ui_recovery: dict | None = None,
):
    tracker = start_operation(
        operation,
        subsystem=operation,
        operation_id=f"{operation}-synthetic",
        target_synth="serum2",
    )
    tracker.enter_phase(phase, reason="synthetic failure setup")
    tracker.mark_progress(text="some work happened")
    postmortem = capture_postmortem(
        tracker,
        exception=exception,
        trigger="failure",
        workers=[{"worker_id": "render-1", "pid": 4242, "alive": False, "exitcode": 1}],
        disk_paths=[tmp_path],
        extra={"ui_recovery": ui_recovery or {}},
    )
    tracker.fail(exception)
    return create_support_bundle(
        ticket_id="a" * 32,
        comments="synthetic failure",
        directory=tmp_path / "bundle",
        operation=operation,
        operation_id=tracker.operation_id,
        environment=environment or {},
        postmortem=postmortem,
        renderer_selection=renderer_selection,
        settings={"ui_recovery": ui_recovery or {}},
        archive=False,
    )


def _assert_diagnosable(bundle, *, operation: str, phase: str, exception_type: str):
    """Every synthetic failure must answer the same standing questions."""

    directory = bundle.directory
    summary = (directory / SUMMARY_FILENAME).read_text(encoding="utf-8")
    postmortem = json.loads((directory / POSTMORTEM_FILENAME).read_text(encoding="utf-8"))
    reproduction = json.loads(
        (directory / REPRODUCTION_FILENAME).read_text(encoding="utf-8")
    )
    events = [
        json.loads(line)
        for line in (directory / EVENTS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # operation, phase, error
    assert postmortem["operation"] == operation
    assert postmortem["current_phase"] == phase
    assert postmortem["exception"]["type"] == exception_type
    # failure reason and fingerprint
    assert postmortem["exception"]["fingerprint"]["digest"]
    assert postmortem["exception"]["traceback"]
    # process/worker context
    assert postmortem["workers"]
    assert postmortem["runtime"]["pid"] == os.getpid()
    # resource conditions
    assert postmortem["resources"]["disk"]
    # progressing vs stalled
    assert "seconds_since_last_progress" in postmortem
    assert "stall_budget_seconds" in postmortem
    # state machine / phase history
    assert postmortem["operation_state_machine"]["phase_history"]
    # human summary names the operation, phase and fingerprint
    assert operation in summary
    assert phase in summary
    assert exception_type in summary
    assert postmortem["exception"]["fingerprint"]["digest"] in summary
    # a reproduction descriptor exists and excludes sensitive content
    assert reproduction["operation"] == operation
    assert reproduction["failure"]["phase"] == phase
    assert reproduction["diagnostic_schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    # lead-up events
    assert events
    assert any(item["event_type"] == "phase_changed" for item in events)
    return summary, postmortem, reproduction, events


def test_synthetic_missing_serum1(tmp_path: Path, fresh) -> None:
    from core.renderer_selection import RendererUnavailableError, select_renderer

    selection = select_renderer("serum1", env=serum2_only_env(tmp_path))
    error = RendererUnavailableError(selection, context="rendering the library")
    bundle = _bundle_for(
        tmp_path,
        operation="render-sound-library",
        phase="renderer-preflight",
        exception=error,
        renderer_selection=selection.as_dict(),
        environment={"renderer_inventory": [item.as_dict() for item in selection.candidates]},
    )
    summary, postmortem, reproduction, _events = _assert_diagnosable(
        bundle,
        operation="render-sound-library",
        phase="renderer-preflight",
        exception_type="RendererUnavailableError",
    )
    # Which renderer was required, and WHY each candidate was rejected.
    assert "serum1" in reproduction["renderer"]["selection"]["requested_serum"]
    assert reproduction["renderer"]["selection"]["available"] is False
    rejected = reproduction["renderer"]["installed"]
    assert rejected and all(item["rejection_reason"] for item in rejected)
    assert "RENDERER CANDIDATES" in summary


def test_synthetic_missing_serum2(tmp_path: Path, fresh) -> None:
    from core.renderer_selection import RendererUnavailableError, select_renderer
    from tests.test_renderer_routing import serum1_only_env

    selection = select_renderer("serum2", env=serum1_only_env(tmp_path))
    bundle = _bundle_for(
        tmp_path,
        operation="match",
        phase="renderer-validation",
        exception=RendererUnavailableError(selection, context="preparing a serum2 match"),
        renderer_selection=selection.as_dict(),
    )
    _assert_diagnosable(
        bundle,
        operation="match",
        phase="renderer-validation",
        exception_type="RendererUnavailableError",
    )


def test_synthetic_empty_renderer_candidate_list(tmp_path: Path, fresh) -> None:
    import dataclasses

    from core.platform_env import ENV
    from core.renderer_selection import RendererUnavailableError, select_renderer

    empty = dataclasses.replace(ENV, plugin_candidates=())
    selection = select_renderer("serum2", env=empty)
    assert not selection.candidates
    bundle = _bundle_for(
        tmp_path,
        operation="match",
        phase="renderer-discovery",
        exception=RendererUnavailableError(selection),
        renderer_selection=selection.as_dict(),
    )
    _assert_diagnosable(
        bundle,
        operation="match",
        phase="renderer-discovery",
        exception_type="RendererUnavailableError",
    )
    assert "no candidate locations" in str(RendererUnavailableError(selection))


def test_synthetic_wrong_plugin_format(tmp_path: Path, fresh) -> None:
    """Serum 1 installed only as AU: must be selected, not reported missing."""

    from core.renderer_selection import select_renderer
    from tests.test_renderer_routing import build_env

    env = build_env(tmp_path, {("serum1", "AU", "system/Components/Serum.component")})
    selection = select_renderer("serum1", env=env)
    assert selection.available and selection.plugin_format == "AU"
    assert "fallback" in selection.reason


def test_synthetic_plugin_validation_failure(tmp_path: Path, fresh) -> None:
    """An unreadable plug-in must be rejected with a permissions reason."""

    from core.renderer_selection import select_renderer
    from tests.test_renderer_routing import build_env

    env = build_env(tmp_path, {("serum2", "VST3", "system/VST3/Serum2.vst3")})
    target = Path(
        next(
            item.path
            for item in env.plugin_candidates
            if item.synth == "serum2" and item.format == "VST3" and item.path.exists()
        )
    )
    mode = target.stat().st_mode
    os.chmod(target, 0o000)
    try:
        if os.access(target, os.R_OK):
            pytest.skip("running as a user that bypasses permission checks")
        selection = select_renderer("serum2", env=env)
        assert not selection.available
        reasons = [item.rejection_reason for item in selection.rejections()]
        assert any("not readable" in reason for reason in reasons)
        assert "permission" in selection.user_message().casefold()
    finally:
        os.chmod(target, mode)


def test_synthetic_stopiteration_worker_initializer(tmp_path: Path, fresh) -> None:
    """The exact shipped failure, end to end through a bundle."""

    try:
        next(item for item in () if True)
    except StopIteration as exc:
        error = exc
    bundle = _bundle_for(
        tmp_path,
        operation="match",
        phase="worker-pool-initialization",
        exception=error,
    )
    summary, postmortem, _reproduction, _events = _assert_diagnosable(
        bundle,
        operation="match",
        phase="worker-pool-initialization",
        exception_type="StopIteration",
    )
    assert "test_diagnostics.py" in postmortem["exception"]["fingerprint"]["code_location"]


def test_synthetic_unexpected_worker_death(tmp_path: Path, fresh) -> None:
    from core.matcher import WorkerRespawnStormError

    error = WorkerRespawnStormError(
        42, {"replacements": 42, "worker_states": [{"pid": 1, "exitcode": -11}]}
    )
    bundle = _bundle_for(
        tmp_path,
        operation="match",
        phase="worker-pool-initialization",
        exception=error,
    )
    _assert_diagnosable(
        bundle,
        operation="match",
        phase="worker-pool-initialization",
        exception_type="WorkerRespawnStormError",
    )
    assert "42" in str(error)
    assert error.user_message


def test_synthetic_worker_stall(tmp_path: Path, fresh) -> None:
    tracker = start_operation(
        "match", subsystem="match", operation_id="match-stall", target_synth="serum2"
    )
    tracker.enter_phase("evaluation", reason="searching")
    postmortem = capture_postmortem(tracker, trigger="stall")
    bundle = create_support_bundle(
        ticket_id="b" * 32,
        directory=tmp_path / "bundle-stall",
        operation="match",
        postmortem=postmortem,
        archive=False,
    )
    summary = (bundle.directory / SUMMARY_FILENAME).read_text(encoding="utf-8")
    reloaded = json.loads(
        (bundle.directory / POSTMORTEM_FILENAME).read_text(encoding="utf-8")
    )
    assert reloaded["trigger"] == "stall"
    assert reloaded["thread_stacks"], "a hang has no traceback; stacks are the evidence"
    assert "Trigger: stall" in summary


def test_synthetic_failed_subprocess(tmp_path: Path, fresh) -> None:
    import subprocess

    try:
        raise subprocess.CalledProcessError(1, ["patchlab", "--worker", "local-library"])
    except subprocess.CalledProcessError as exc:
        error = exc
    bundle = _bundle_for(
        tmp_path,
        operation="render-sound-library",
        phase="catalog-scan",
        exception=error,
    )
    _assert_diagnosable(
        bundle,
        operation="render-sound-library",
        phase="catalog-scan",
        exception_type="CalledProcessError",
    )


def test_synthetic_database_error(tmp_path: Path, fresh) -> None:
    try:
        raise sqlite3.OperationalError("database is locked")
    except sqlite3.OperationalError as exc:
        error = exc
    bundle = _bundle_for(
        tmp_path, operation="render-sound-library", phase="durable-save", exception=error
    )
    _assert_diagnosable(
        bundle,
        operation="render-sound-library",
        phase="durable-save",
        exception_type="OperationalError",
    )


def test_synthetic_corrupt_database_is_reported_not_crashed(tmp_path: Path, fresh) -> None:
    from core.diagnostic_env import capture_environment

    corrupt = tmp_path / "library.db"
    corrupt.write_bytes(b"this is not a sqlite database at all")
    snapshot = capture_environment(operation="match", db_path=corrupt).as_dict()
    database = snapshot["database"]
    assert database["available"] is False
    assert "reason" in database
    assert "DatabaseError" in database["reason"] or "Error" in database["reason"]


def test_synthetic_missing_database_is_distinguishable(tmp_path: Path, fresh) -> None:
    from core.diagnostic_env import capture_environment

    snapshot = capture_environment(
        operation="match", db_path=tmp_path / "absent.db"
    ).as_dict()
    assert snapshot["database"]["available"] is False
    assert "does not exist" in snapshot["database"]["reason"]


def test_synthetic_permission_denial(tmp_path: Path, fresh) -> None:
    from core.diagnostic_env import capture_environment

    locked = tmp_path / "locked"
    locked.mkdir()
    mode = locked.stat().st_mode
    os.chmod(locked, 0o000)
    try:
        snapshot = capture_environment(
            operation="render-sound-library", linked_folder=locked
        ).as_dict()
        access = snapshot["path_access"]["linked_folder"]
        assert access["exists"] is True
        if os.access(locked, os.R_OK):
            pytest.skip("running as a user that bypasses permission checks")
        assert access["readable"] is False
    finally:
        os.chmod(locked, mode)


def test_synthetic_disk_state_is_recorded(tmp_path: Path, fresh) -> None:
    """Near-full disks are reported without needing a real full disk."""

    from core.operation_state import resource_snapshot

    snapshot = resource_snapshot([tmp_path, Path("/definitely/not/here")])
    entry = snapshot["disk"][str(tmp_path)]
    assert entry["total_bytes"] > 0
    assert "free_bytes" in entry
    assert "error" in snapshot["disk"]["/definitely/not/here"]


def test_synthetic_missing_model(tmp_path: Path, fresh) -> None:
    try:
        raise FileNotFoundError(2, "No such file", "/Resources/data/models/clap.pt")
    except FileNotFoundError as exc:
        error = exc
    bundle = _bundle_for(
        tmp_path, operation="match", phase="model-loading", exception=error
    )
    _assert_diagnosable(
        bundle, operation="match", phase="model-loading", exception_type="FileNotFoundError"
    )


def test_synthetic_model_load_exception(tmp_path: Path, fresh) -> None:
    try:
        try:
            raise ValueError("unexpected checkpoint keys")
        except ValueError as inner:
            raise RuntimeError("model load failed") from inner
    except RuntimeError as exc:
        error = exc
    bundle = _bundle_for(
        tmp_path, operation="match", phase="model-loading", exception=error
    )
    _summary, postmortem, _repro, _events = _assert_diagnosable(
        bundle, operation="match", phase="model-loading", exception_type="RuntimeError"
    )
    chain = postmortem["exception"]["chain"]
    assert [item["type"] for item in chain] == ["RuntimeError", "ValueError"]
    assert chain[1]["link"] == "cause"


def test_synthetic_inaccessible_linked_folder(tmp_path: Path, fresh) -> None:
    try:
        raise NotADirectoryError(tmp_path / "gone")
    except NotADirectoryError as exc:
        error = exc
    bundle = _bundle_for(
        tmp_path,
        operation="render-sound-library",
        phase="linked-folder-validation",
        exception=error,
    )
    _assert_diagnosable(
        bundle,
        operation="render-sound-library",
        phase="linked-folder-validation",
        exception_type="NotADirectoryError",
    )


def test_synthetic_disconnected_storage(tmp_path: Path, fresh) -> None:
    try:
        raise OSError(5, "Input/output error", "/Volumes/External/audio")
    except OSError as exc:
        error = exc
    bundle = _bundle_for(
        tmp_path, operation="render-sound-library", phase="durable-save", exception=error
    )
    _assert_diagnosable(
        bundle,
        operation="render-sound-library",
        phase="durable-save",
        exception_type="OSError",
    )


def test_synthetic_unexpected_exception_in_match(tmp_path: Path, fresh) -> None:
    try:
        raise ZeroDivisionError("division by zero")
    except ZeroDivisionError as exc:
        error = exc
    bundle = _bundle_for(tmp_path, operation="match", phase="ranking", exception=error)
    _assert_diagnosable(
        bundle, operation="match", phase="ranking", exception_type="ZeroDivisionError"
    )


def test_synthetic_unexpected_exception_in_render(tmp_path: Path, fresh) -> None:
    try:
        raise KeyError("preset_id")
    except KeyError as exc:
        error = exc
    bundle = _bundle_for(
        tmp_path, operation="render-sound-library", phase="rendering", exception=error
    )
    _assert_diagnosable(
        bundle,
        operation="render-sound-library",
        phase="rendering",
        exception_type="KeyError",
    )


def test_bundle_records_ui_recovery_state(tmp_path: Path, fresh) -> None:
    """A report must answer: was earlier workflow state preserved?"""

    ui = {
        "link": {"phase": "complete", "text": "Linked: Serum 2 Presets · 5,122 presets"},
        "render": {"phase": "failed", "text": "Rendering stopped — click to retry"},
        "match": {"phase": "needs-action", "text": "Ready · choose an audio file"},
    }
    bundle = _bundle_for(
        tmp_path,
        operation="render-sound-library",
        phase="renderer-preflight",
        exception=RuntimeError("no renderer"),
        ui_recovery=ui,
    )
    reproduction = json.loads(
        (bundle.directory / REPRODUCTION_FILENAME).read_text(encoding="utf-8")
    )
    recovered = reproduction["settings"]["ui_recovery"]
    assert recovered["link"]["phase"] == "complete", "Link must still read complete"
    assert recovered["render"]["phase"] == "failed"
    postmortem = json.loads(
        (bundle.directory / POSTMORTEM_FILENAME).read_text(encoding="utf-8")
    )
    assert postmortem["extra"]["ui_recovery"]["link"]["phase"] == "complete"


# ---------------------------------------------------------------------------
# Bundle structure and the local-save guarantee (PART 16, PART 35)
# ---------------------------------------------------------------------------


def test_bundle_has_the_expected_files(tmp_path: Path, fresh) -> None:
    fresh.record("match", "decision", "picked something", operation_id="op")
    bundle = create_support_bundle(
        ticket_id="c" * 32,
        comments="it broke",
        directory=tmp_path / "bundle",
        operation="match",
    )
    names = {path.name for path in bundle.files}
    assert names == {
        SUMMARY_FILENAME,
        EVENTS_FILENAME,
        ENVIRONMENT_FILENAME,
        POSTMORTEM_FILENAME,
        REPRODUCTION_FILENAME,
        "repeats.json",
    }
    assert bundle.archive is not None and bundle.archive.is_file()
    import zipfile

    with zipfile.ZipFile(bundle.archive) as handle:
        assert len(handle.namelist()) == len(names)


def test_bundle_is_written_without_any_support_service(tmp_path: Path, fresh) -> None:
    """PART 35: the bundle must not depend on the private support transport."""

    bundle = create_support_bundle(
        ticket_id="d" * 32, directory=tmp_path / "bundle", archive=False
    )
    assert (bundle.directory / SUMMARY_FILENAME).is_file()
    assert "PATCHLAB SUPPORT BUNDLE" in (
        bundle.directory / SUMMARY_FILENAME
    ).read_text(encoding="utf-8")


def test_bundle_events_are_size_capped(tmp_path: Path) -> None:
    from core.support_bundle import MAX_BUNDLE_EVENT_BYTES

    instance = DiagnosticRecorder(root=tmp_path / "d", capacity=4096)
    set_recorder(instance)
    try:
        for index in range(4096):
            instance.record("test", "bulk", "y" * 3000, index=index)
        bundle = create_support_bundle(
            ticket_id="e" * 32, directory=tmp_path / "bundle", archive=False
        )
        size = (bundle.directory / EVENTS_FILENAME).stat().st_size
        assert size <= MAX_BUNDLE_EVENT_BYTES, size
    finally:
        instance.close()
        set_recorder(None)


def test_reproduction_descriptor_excludes_sensitive_content(tmp_path: Path, fresh) -> None:
    descriptor = build_reproduction_descriptor(
        operation="match",
        environment={
            "application": {"patchlab_version": "1.5.4"},
            "system": {"branch": "macos", "architecture": "arm64"},
            "patchlab_state": {"target_serum_version": "serum2"},
        },
        settings={"relay_token": "secret-value", "quality": "balanced"},
    )
    blob = json.dumps(descriptor)
    assert "secret-value" not in blob
    assert descriptor["settings"]["relay_token"] == "[redacted]"
    assert descriptor["request"]["serum_generation"] == "serum2"
    assert "raw user audio" in descriptor["excluded_by_policy"]


def test_environment_snapshot_covers_the_required_sections(tmp_path: Path, fresh) -> None:
    from core.diagnostic_env import capture_environment

    snapshot = capture_environment(
        operation="match",
        operation_id="op-1",
        env=serum2_only_env(tmp_path),
        target_synth="serum2",
        quality_mode="balanced",
        library_counts={"presets": 5122, "renders": 0},
    ).as_dict()
    for section in (
        "application",
        "system",
        "dependencies",
        "patchlab_state",
        "database",
        "model",
        "storage",
        "renderer_inventory",
        "path_access",
        "flight_recorder",
        "resources",
    ):
        assert section in snapshot, section
    assert snapshot["application"]["diagnostic_schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert snapshot["system"]["architecture"]
    assert snapshot["patchlab_state"]["library_counts"]["presets"] == 5122
    assert snapshot["renderer_inventory"]
    accepted = [item for item in snapshot["renderer_inventory"] if item["accepted"]]
    assert [item["renderer"] for item in accepted] == ["serum2/VST3"]
    assert "torch" in snapshot["dependencies"]


def test_environment_snapshot_is_captured_once_not_continuously(
    tmp_path: Path, fresh
) -> None:
    from core.diagnostic_env import capture_environment

    capture_environment(operation="match", env=serum2_only_env(tmp_path))
    snapshots = [
        item
        for item in fresh.recent_events()
        if item["event_type"] == "environment_snapshot"
    ]
    assert len(snapshots) == 1


# ---------------------------------------------------------------------------
# PART 35: the local bundle never depends on the support service
# ---------------------------------------------------------------------------


def test_bug_report_worker_writes_the_bundle_before_any_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh
) -> None:
    """A failed or absent support service must never cost the user evidence."""

    import core.bug_report as bug_report_module
    import core.local_library as library_module
    import scripts.submit_bug_report as worker

    def reports_root() -> Path:
        root = tmp_path / "reports"
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(bug_report_module, "reports_root", reports_root)
    import core.submission_upload as upload_module

    monkeypatch.setattr(upload_module, "resolve_relay", lambda: (None, "local_only"))
    fresh.record("match", "decision", "picked something", operation_id="op")

    request_path = bug_report_module.create_request(
        comments="render sound library did nothing", logs="diagnostics"
    )
    ticket = bug_report_module.load_request(request_path).ticket_id

    monkeypatch.setattr(sys, "argv", ["submit_bug_report", "--request", str(request_path)])
    # The relay is unavailable, so the worker reports failure...
    assert worker.main() == 1
    # ...but the readable ticket and the structured bundle both survive.
    assert request_path.is_file()
    directory = (
        tmp_path / "reports" / f"PatchLab Bug Report {ticket} diagnostics"
    )
    assert (directory / SUMMARY_FILENAME).is_file()
    assert (directory / EVENTS_FILENAME).is_file()
    assert (directory / REPRODUCTION_FILENAME).is_file()


def test_bundle_creation_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fresh
) -> None:
    """Re-running must not duplicate or clobber an existing bundle."""

    import core.bug_report as bug_report_module
    import scripts.submit_bug_report as worker

    def reports_root() -> Path:
        root = tmp_path / "reports"
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(bug_report_module, "reports_root", reports_root)
    request_path = bug_report_module.create_request(comments="x", logs="y")
    request = bug_report_module.load_request(request_path)

    first = worker._ensure_bundle(request)
    assert first is not None
    marker = first / SUMMARY_FILENAME
    stamp = marker.stat().st_mtime_ns
    second = worker._ensure_bundle(request)
    assert second == first
    assert marker.stat().st_mtime_ns == stamp, "an existing bundle must be left alone"
    # Exactly one bundle directory, not two.
    assert len(list((tmp_path / "reports").glob("*diagnostics"))) == 1


# ---------------------------------------------------------------------------
# The bundle is built in the GUI process; the evidence lives in worker processes
# ---------------------------------------------------------------------------


def _run_worker_that_records(root: Path, message: str, *, count: int = 1, worker: str = "render-1") -> None:
    """Record events from a *different process*, as a real worker would."""

    import subprocess
    import sys as _sys

    project = Path(__file__).resolve().parents[1]
    code = f"""
import sys; sys.path.insert(0, {str(project)!r})
from core.diagnostics import recorder
r = recorder()
for _ in range({count}):
    try:
        raise RuntimeError({message!r})
    except RuntimeError as exc:
        r.record_failure('render-worker', 'worker_init_failed', exc,
                         operation_id='op-x', phase='worker-pool-initialization', worker_id={worker!r})
r.flush(timeout=5); r.close()
"""
    env = dict(os.environ, PATCHLAB_DIAGNOSTICS_DIR=str(root), PATCHLAB_SESSION_ID="cross-proc")
    subprocess.run([_sys.executable, "-c", code], check=True, env=env)


def test_bundle_contains_events_written_by_other_processes(tmp_path: Path) -> None:
    """A GUI-built bundle must include what the worker processes recorded."""

    root = tmp_path / "shared"
    instance = DiagnosticRecorder(root=root, session_id="cross-proc")
    set_recorder(instance)
    try:
        _run_worker_that_records(root, "worker exploded")
        instance.record("gui", "render_failed", "GUI-SIDE EVENT", severity="warning")
        bundle = create_support_bundle(ticket_id="d" * 32, directory=tmp_path / "b", archive=False)
        rows = [
            json.loads(line)
            for line in (bundle.directory / EVENTS_FILENAME).read_text().splitlines()
            if line.strip()
        ]
        pids = {row["pid"] for row in rows}
        assert len(pids) == 2 and os.getpid() in pids
        worker_rows = [row for row in rows if row["pid"] != os.getpid()]
        assert any(row["event_type"] == "worker_init_failed" for row in worker_rows)
        assert any(row["message"] == "GUI-SIDE EVENT" for row in rows)
        # One chronological timeline, not two concatenated ones.
        stamps = [row["ts"] for row in rows]
        assert stamps == sorted(stamps)
        # And the human summary's timeline shows the worker's failure.
        summary = (bundle.directory / SUMMARY_FILENAME).read_text()
        assert "worker_init_failed" in summary
    finally:
        instance.close()
        set_recorder(None)


def test_bundle_does_not_duplicate_events_present_on_disk_and_in_memory(tmp_path: Path) -> None:
    instance = DiagnosticRecorder(root=tmp_path / "d", session_id="dedupe")
    set_recorder(instance)
    try:
        for index in range(20):
            instance.record("gui", "tick", f"event {index}")
        bundle = create_support_bundle(ticket_id="e" * 32, directory=tmp_path / "b", archive=False)
        rows = [
            json.loads(line)
            for line in (bundle.directory / EVENTS_FILENAME).read_text().splitlines()
            if line.strip()
        ]
        ticks = [row for row in rows if row["event_type"] == "tick"]
        assert len(ticks) == 20, "each event must appear once, not once per source"
    finally:
        instance.close()
        set_recorder(None)


def test_bundle_window_excludes_old_history(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    root = tmp_path / "d"
    root.mkdir()
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    (root / "events.jsonl").write_text(
        json.dumps({"ts": old, "pid": 1, "event_type": "ancient", "message": "old", "severity": "info"}) + "\n"
        "{ torn json line from a killed worker\n",
        encoding="utf-8",
    )
    instance = DiagnosticRecorder(root=root, session_id="window")
    set_recorder(instance)
    try:
        instance.record("gui", "recent", "now")
        bundle = create_support_bundle(ticket_id="f" * 32, directory=tmp_path / "b", archive=False)
        types = {
            json.loads(line)["event_type"]
            for line in (bundle.directory / EVENTS_FILENAME).read_text().splitlines()
            if line.strip()
        }
        assert "recent" in types and "ancient" not in types
    finally:
        instance.close()
        set_recorder(None)


def test_repeated_failures_across_many_processes_are_aggregated_in_the_bundle(tmp_path: Path) -> None:
    """The reported ticket: the same failure from many worker processes."""

    root = tmp_path / "shared"
    instance = DiagnosticRecorder(root=root, session_id="cross-proc")
    set_recorder(instance)
    try:
        for index in range(4):
            _run_worker_that_records(root, "renderer unavailable", count=3, worker=f"render-{index}")
        bundle = create_support_bundle(ticket_id="a" * 32, directory=tmp_path / "b", archive=False)
        groups = json.loads((bundle.directory / "repeats.json").read_text())
        assert len(groups) == 1
        group = groups[0]
        assert group["total_occurrences"] == 12
        assert len(group["worker_examples"]) == 4
        assert len(group["pid_examples"]) == 4, "one PID per worker process"
        assert group["first_full_event"]["fields"]["exception"]["chain"]
    finally:
        instance.close()
        set_recorder(None)


def test_summary_states_capability_coverage_and_ui_recovery(tmp_path: Path, fresh) -> None:
    """A cold reader gets synth state, library coverage and UI recovery in one file."""

    fresh.record(
        "gui", "render_failed", "Render Sound Library failed; UI recovered", severity="warning",
        link_phase="complete", render_phase="failed", retry_available=True, stale_activities=[],
    )
    environment = {
        "application": {"patchlab_version": "1.5.4", "source_commit": "abc", "frozen": True},
        "system": {"architecture": "arm64"},
        "synth_capability": {
            "capabilities": {
                "serum1": {"status": "not_installed", "preferred_renderer": "",
                            "reason": "no Serum 1 plug-in exists at any known location"},
                "serum2": {"status": "ready", "preferred_renderer": "serum2/VST3", "reason": "ok"},
            }
        },
        "library_coverage": {
            "available": True, "discovered": 6683, "learned": 1857, "processed_params": 1857,
            "pending": 4826, "failed": 0,
            "by_file_format": {"fxp": 4826, "serumpreset": 1857},
            "by_generation": {"serum1": 4826, "serum2": 1857},
            "pending_by_reason": {"unsupported_legacy_format": 4826},
        },
    }
    bundle = create_support_bundle(
        ticket_id="b" * 32, directory=tmp_path / "b", archive=False, environment=environment
    )
    summary = (bundle.directory / SUMMARY_FILENAME).read_text()
    assert "INSTALLED SYNTHS" in summary and "serum1: not_installed" in summary
    assert "serum2: ready" in summary
    assert "LIBRARY COVERAGE" in summary
    assert "discovered 6683" in summary and "pending 4826" in summary
    assert "unsupported_legacy_format" in summary
    assert "UI RECOVERY" in summary
    assert "'link_phase': 'complete'" in summary and "'retry_available': True" in summary


def test_a_burst_of_match_failures_does_not_hide_an_earlier_render_failure(tmp_path: Path, fresh) -> None:
    """Regression: the summary once kept only the last four UI events overall."""

    fresh.record(
        "gui", "render_failed", "Render Sound Library failed; UI recovered", severity="warning",
        link_phase="complete", render_phase="failed", retry_available=True, stale_activities=[],
    )
    for _ in range(6):
        fresh.record(
            "gui", "match_failed", "Match failed; UI recovered", severity="warning",
            controls_recovered=True, stale_activities=[], source_audio_kept=True,
        )
    bundle = create_support_bundle(ticket_id="c" * 32, directory=tmp_path / "c", archive=False, environment={})
    summary = (bundle.directory / SUMMARY_FILENAME).read_text()
    recovery = summary.split("UI RECOVERY", 1)[1].split("=== OPERATION", 1)[0]
    assert "render_failed" in recovery and "'link_phase': 'complete'" in recovery
    assert recovery.count("match_failed") == 2, "bounded: latest two of each kind"


# ---------------------------------------------------------------------------
# Durability: what was recorded must reach disk
# ---------------------------------------------------------------------------


def test_flush_is_a_real_durability_barrier(tmp_path: Path) -> None:
    """After flush() returns, every earlier event is readable from disk.

    No close(), no sleep: the previous flush() returned when the queue was empty,
    before the writer had written its in-hand batch.
    """

    root = tmp_path / "d"
    instance = DiagnosticRecorder(root=root, session_id="durable")
    try:
        for index in range(50):
            instance.record("test", "tick", f"event {index}")
        instance.flush(timeout=5.0)
        lines = (root / EVENTS_FILENAME).read_text().splitlines()
        assert len([l for l in lines if l.strip()]) == 50
    finally:
        instance.close()


def test_a_process_that_exits_without_close_still_persists_its_events(tmp_path: Path) -> None:
    """The GUI and the packaged gates never call close(); the exit hook must."""

    import subprocess
    import sys as _sys

    project = Path(__file__).resolve().parents[1]
    root = tmp_path / "d"
    code = (
        f"import sys; sys.path.insert(0, {str(project)!r})\n"
        "from core.diagnostics import record\n"
        "record('gui', 'render_failed', 'lost-if-not-drained', severity='warning')\n"
    )  # deliberately no flush(), no close()
    env = dict(os.environ, PATCHLAB_DIAGNOSTICS_DIR=str(root))
    subprocess.run([_sys.executable, "-c", code], check=True, env=env)
    rows = [json.loads(l) for l in (root / EVENTS_FILENAME).read_text().splitlines() if l.strip()]
    assert any(r["message"] == "lost-if-not-drained" for r in rows)
