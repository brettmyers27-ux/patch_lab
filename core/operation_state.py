"""Explicit phases, liveness and postmortem capture for long-running work.

PatchLab's primary reliability invariant is that a long-running operation must
eventually make measurable forward progress, complete, or fail cleanly with a
useful error.  It must never sit in an indefinite loading state.

The reported Match hang violated that in the worst possible way: there was no
exception in the parent at all.  A pool initializer died in every child, the
pool silently replaced them forever, and ``pool.map`` blocked with no timeout,
so no traceback and no terminal state ever reached the UI.  A hang that
produces no exception needs a detector and a state snapshot, not just better
logging.

This module provides:

* :class:`OperationTracker` -- an operation's ID, phase history, phase start,
  last meaningful progress, counts and terminal reason.
* :class:`StallDetector` -- "no meaningful progress for abnormally long *given
  the current phase*", not simply "took a long time", so a legitimately slow
  render or search is never killed.
* postmortem capture -- thread stacks, child-process state, resource usage,
  the state machine itself -- taken **before** recovery destroys the evidence.

Clocks are injectable throughout so the tests are deterministic rather than
wall-clock races.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.diagnostics import (
    DIAGNOSTIC_SCHEMA_VERSION,
    FailureFingerprint,
    exception_chain,
    fingerprint_failure,
    new_operation_id,
    recorder,
    sanitize,
)


Clock = Callable[[], float]


# ---------------------------------------------------------------------------
# Phase vocabularies
# ---------------------------------------------------------------------------

MATCH_PHASES: tuple[str, ...] = (
    "accepted",
    "decoding",
    "target-analysis",
    "model-loading",
    "renderer-discovery",
    "renderer-validation",
    "worker-pool-initialization",
    "candidate-preparation",
    "evaluation",
    "ranking",
    "result-construction",
    "complete",
    "failed",
    "cancelled",
)

LIBRARY_PHASES: tuple[str, ...] = (
    "accepted",
    "linked-folder-validation",
    "preset-classification",
    "renderer-preflight",
    "catalog-scan",
    "batch-preparation",
    "rendering",
    "analysis",
    "durable-save",
    "temporary-cleanup",
    "complete",
    "failed",
    "cancelled",
)

TERMINAL_PHASES = frozenset({"complete", "failed", "cancelled"})

#: Per-phase stall budgets in seconds.  These are *inactivity* budgets: the
#: clock only matters when nothing has reported progress.  Phases that are
#: legitimately long and silent (loading an 800 MB checkpoint, opening Serum)
#: get generous budgets; phases that should be near-instant get tight ones.
#:
#: ``worker-pool-initialization`` is the one that mattered: a deterministic
#: initializer failure is caught by preflight long before this fires, and this
#: budget exists only so an *unforeseen* variant of the same hang still
#: terminates instead of spinning forever.
DEFAULT_PHASE_STALL_SECONDS: Mapping[str, float] = {
    "accepted": 60.0,
    "decoding": 180.0,
    "target-analysis": 180.0,
    "model-loading": 900.0,
    "renderer-discovery": 120.0,
    "renderer-validation": 300.0,
    "worker-pool-initialization": 300.0,
    "candidate-preparation": 600.0,
    "evaluation": 900.0,
    "ranking": 300.0,
    "result-construction": 300.0,
    "linked-folder-validation": 120.0,
    "preset-classification": 900.0,
    "renderer-preflight": 120.0,
    "catalog-scan": 900.0,
    "batch-preparation": 600.0,
    "rendering": 1800.0,
    "analysis": 1800.0,
    "durable-save": 600.0,
    "temporary-cleanup": 600.0,
}

DEFAULT_STALL_SECONDS = 900.0

#: Worker states that mean "alive and doing real work". A heartbeat in one of
#: these is corroborating evidence of liveness when the parent has not yet seen
#: a completed unit of work -- a single render batch can legitimately run longer
#: than a phase budget while every worker is busy.
BUSY_WORKER_STATES: frozenset[str] = frozenset(
    {"initializing-renderer", "loading-preset", "rendering", "analyzing"}
)

#: How far a busy heartbeat may extend an inactivity budget, as a multiple.
#: Bounded on purpose: a worker that heartbeats "rendering" forever without ever
#: completing anything is still stuck, and must eventually be captured. Evidence
#: of life buys time; it does not buy immunity.
HEARTBEAT_BUDGET_MULTIPLIER = 2.0


def stall_budget_for(phase: str, overrides: Mapping[str, float] | None = None) -> float:
    if overrides and phase in overrides:
        return float(overrides[phase])
    return float(DEFAULT_PHASE_STALL_SECONDS.get(phase, DEFAULT_STALL_SECONDS))


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseRecord:
    name: str
    started_monotonic: float
    started_utc: str
    ended_monotonic: float | None = None
    reason: str = ""

    def duration(self, now: float) -> float:
        end = self.ended_monotonic if self.ended_monotonic is not None else now
        return max(end - self.started_monotonic, 0.0)

    def as_dict(self, now: float) -> dict[str, Any]:
        return {
            "phase": self.name,
            "started_utc": self.started_utc,
            "duration_seconds": round(self.duration(now), 3),
            "completed": self.ended_monotonic is not None,
            "reason": self.reason,
        }


@dataclass(slots=True)
class OperationTracker:
    """One long-running operation's identity, phases and liveness.

    Thread-safe: the render/search code, a heartbeat reader and a watchdog all
    touch it.
    """

    operation: str
    operation_id: str
    subsystem: str
    phases: tuple[str, ...] = ()
    clock: Clock = time.monotonic
    stall_overrides: Mapping[str, float] = field(default_factory=dict)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _history: list[PhaseRecord] = field(default_factory=list)
    _started_monotonic: float = 0.0
    _started_utc: str = ""
    _last_progress_monotonic: float = 0.0
    _last_progress_detail: dict[str, Any] = field(default_factory=dict)
    _progress_events: int = 0
    _counts: dict[str, int] = field(default_factory=dict)
    _terminal_reason: str = ""
    _failure: dict[str, Any] | None = None
    _cancelled: bool = False
    _metadata: dict[str, Any] = field(default_factory=dict)
    #: worker_id -> (monotonic timestamp, worker state)
    _worker_heartbeats: dict[str, tuple[float, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        now = float(self.clock())
        self._started_monotonic = now
        self._started_utc = datetime.now(timezone.utc).isoformat()
        self._last_progress_monotonic = now
        self._history.append(
            PhaseRecord("accepted", now, self._started_utc)
        )
        recorder().record(
            self.subsystem,
            "operation_started",
            f"{self.operation} accepted",
            operation_id=self.operation_id,
            phase="accepted",
            operation=self.operation,
        )

    # -- identity ----------------------------------------------------------

    @property
    def phase(self) -> str:
        with self._lock:
            return self._history[-1].name

    @property
    def previous_phase(self) -> str:
        with self._lock:
            return self._history[-2].name if len(self._history) > 1 else ""

    @property
    def last_successful_phase(self) -> str:
        """The most recent phase that completed without becoming terminal."""

        with self._lock:
            for record in reversed(self._history[:-1]):
                if record.ended_monotonic is not None and record.name not in TERMINAL_PHASES:
                    return record.name
            return ""

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def elapsed(self) -> float:
        return max(float(self.clock()) - self._started_monotonic, 0.0)

    def phase_elapsed(self) -> float:
        with self._lock:
            return self._history[-1].duration(float(self.clock()))

    def since_progress(self) -> float:
        with self._lock:
            return max(float(self.clock()) - self._last_progress_monotonic, 0.0)

    # -- transitions -------------------------------------------------------

    def enter_phase(self, name: str, *, reason: str = "", **fields: Any) -> None:
        """Move to ``name``, recording why.

        A phase transition is a decision, so the reason is logged with it.
        """

        now = float(self.clock())
        with self._lock:
            if self._history and self._history[-1].name == name:
                return
            previous = self._history[-1]
            self._history[-1] = PhaseRecord(
                previous.name,
                previous.started_monotonic,
                previous.started_utc,
                ended_monotonic=now,
                reason=previous.reason,
            )
            self._history.append(
                PhaseRecord(name, now, datetime.now(timezone.utc).isoformat(), reason=reason)
            )
            # Entering a new phase is itself forward progress.
            self._last_progress_monotonic = now
            from_phase = previous.name
            from_phase_seconds = round(now - previous.started_monotonic, 3)
        recorder().record(
            self.subsystem,
            "phase_changed",
            f"{from_phase} -> {name}",
            operation_id=self.operation_id,
            phase=name,
            decision_reason=reason or f"{self.operation} advanced to {name}",
            previous_phase=from_phase,
            previous_phase_seconds=from_phase_seconds,
            **fields,
        )

    def mark_progress(self, detail: Mapping[str, Any] | None = None, **fields: Any) -> None:
        """Record measurable forward progress.

        This is what keeps the stall detector quiet.  Only call it when
        something real advanced -- a candidate rendered, a batch committed, an
        evaluation completed -- never from a timer, or the detector becomes
        worthless.
        """

        payload = dict(detail or {})
        payload.update(fields)
        now = float(self.clock())
        with self._lock:
            self._last_progress_monotonic = now
            self._progress_events += 1
            if payload:
                self._last_progress_detail = dict(payload)
            phase = self._history[-1].name
        recorder().record(
            self.subsystem,
            "progress",
            payload.get("text", "") or f"{self.operation} progress",
            operation_id=self.operation_id,
            phase=phase,
            **payload,
        )

    def note_heartbeat(self, worker_id: str, state: str) -> None:
        """Record a worker heartbeat.

        Deliberately *not* treated as progress: a heartbeat says a worker is
        alive, not that anything was accomplished. The stall detector uses it
        only to extend an inactivity budget within a bounded multiple, so a
        long-but-healthy render batch is not killed while a genuinely wedged
        worker still eventually trips.
        """

        now = float(self.clock())
        with self._lock:
            self._worker_heartbeats[str(worker_id)] = (now, str(state))

    def busy_heartbeat_age(self) -> float | None:
        """Seconds since the most recent heartbeat in a busy state, if any."""

        now = float(self.clock())
        with self._lock:
            busy = [
                stamp
                for stamp, state in self._worker_heartbeats.values()
                if state in BUSY_WORKER_STATES
            ]
        if not busy:
            return None
        return max(now - max(busy), 0.0)

    def worker_heartbeats(self) -> dict[str, dict[str, Any]]:
        """Per-worker liveness view, for postmortems."""

        now = float(self.clock())
        with self._lock:
            return {
                worker_id: {
                    "state": state,
                    "seconds_since_heartbeat": round(now - stamp, 3),
                    "busy": state in BUSY_WORKER_STATES,
                }
                for worker_id, (stamp, state) in self._worker_heartbeats.items()
            }

    def bump(self, key: str, amount: int = 1) -> int:
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + int(amount)
            return self._counts[key]

    def set_count(self, key: str, value: int) -> None:
        with self._lock:
            self._counts[key] = int(value)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def annotate(self, **values: Any) -> None:
        """Attach operation-scoped facts (target synth, quality mode, ...)."""

        with self._lock:
            self._metadata.update(sanitize(values))

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._metadata)

    # -- terminal states ---------------------------------------------------

    def complete(self, reason: str = "finished successfully", **fields: Any) -> None:
        self.enter_phase("complete", reason=reason, **fields)
        with self._lock:
            self._terminal_reason = reason
        recorder().record(
            self.subsystem,
            "operation_complete",
            f"{self.operation} complete",
            operation_id=self.operation_id,
            phase="complete",
            decision_reason=reason,
            elapsed_seconds=round(self.elapsed(), 3),
            counts=self.counts(),
        )

    def fail(
        self,
        exception: BaseException | None = None,
        *,
        reason: str = "",
        **fields: Any,
    ) -> FailureFingerprint:
        """Enter ``failed``, preserving the exception chain and traceback."""

        failing_phase = self.phase
        message = reason or (
            f"{type(exception).__name__}: {exception}" if exception is not None else "failed"
        )
        marker = fingerprint_failure(
            exception,
            subsystem=self.subsystem,
            phase=failing_phase,
            renderer=str(self.metadata().get("renderer", "")),
            serum_generation=str(self.metadata().get("target_synth", "")),
            message=message,
        )
        with self._lock:
            self._terminal_reason = message
            self._failure = {
                "phase": failing_phase,
                "message": message,
                "fingerprint": marker.as_dict(),
                "exception_chain": exception_chain(exception) if exception else [],
            }
        self.enter_phase("failed", reason=message, **fields)
        recorder().record(
            self.subsystem,
            "operation_failed",
            message,
            severity="error",
            operation_id=self.operation_id,
            phase="failed",
            decision_reason=f"terminal failure during {failing_phase}",
            exception=exception,
            fingerprint=marker.digest,
            failed_phase=failing_phase,
            failure_fingerprint=marker.as_dict(),
            elapsed_seconds=round(self.elapsed(), 3),
            counts=self.counts(),
        )
        return marker

    def cancel(self, reason: str = "cancelled by the user") -> None:
        with self._lock:
            self._cancelled = True
            self._terminal_reason = reason
        self.enter_phase("cancelled", reason=reason)

    def failure(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._failure) if self._failure else None

    @property
    def terminal_reason(self) -> str:
        with self._lock:
            return self._terminal_reason

    # -- snapshot ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The complete state-machine view, for progress and for postmortems."""

        now = float(self.clock())
        with self._lock:
            history = [record.as_dict(now) for record in self._history]
            current = self._history[-1]
            return {
                "operation": self.operation,
                "operation_id": self.operation_id,
                "subsystem": self.subsystem,
                "current_phase": current.name,
                "previous_phase": (
                    self._history[-2].name if len(self._history) > 1 else ""
                ),
                "last_successful_phase": self.last_successful_phase,
                "phase_started_utc": current.started_utc,
                "phase_elapsed_seconds": round(current.duration(now), 3),
                "operation_started_utc": self._started_utc,
                "elapsed_seconds": round(now - self._started_monotonic, 3),
                "seconds_since_last_progress": round(
                    now - self._last_progress_monotonic, 3
                ),
                "progress_events": self._progress_events,
                "last_progress_detail": dict(self._last_progress_detail),
                "counts": dict(self._counts),
                "metadata": dict(self._metadata),
                "cancelled": self._cancelled,
                "terminal": current.name in TERMINAL_PHASES,
                "terminal_reason": self._terminal_reason,
                "worker_heartbeats": {
                    worker_id: {
                        "state": state,
                        "seconds_since_heartbeat": round(now - stamp, 3),
                    }
                    for worker_id, (stamp, state) in self._worker_heartbeats.items()
                },
                "failure": dict(self._failure) if self._failure else None,
                "phase_history": history,
                "stall_budget_seconds": stall_budget_for(
                    current.name, self.stall_overrides
                ),
                "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            }


def start_operation(
    operation: str,
    *,
    subsystem: str,
    phases: Sequence[str] = (),
    operation_id: str = "",
    clock: Clock = time.monotonic,
    stall_overrides: Mapping[str, float] | None = None,
    **metadata: Any,
) -> OperationTracker:
    tracker = OperationTracker(
        operation=operation,
        operation_id=operation_id or new_operation_id(operation),
        subsystem=subsystem,
        phases=tuple(phases),
        clock=clock,
        stall_overrides=dict(stall_overrides or {}),
    )
    if metadata:
        tracker.annotate(**metadata)
    return tracker


# ---------------------------------------------------------------------------
# Environment / resource probes
# ---------------------------------------------------------------------------


def _process_rss_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Darwin reports bytes, Linux kilobytes.
        return int(usage) if sys.platform == "darwin" else int(usage) * 1024
    except Exception:
        return None


def resource_snapshot(paths: Sequence[Path] = ()) -> dict[str, Any]:
    """Cheap CPU/memory/disk state.  Safe to call on failure paths."""

    snapshot: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "load_average": None,
        "process_rss_bytes": _process_rss_bytes(),
        "disk": {},
    }
    try:
        snapshot["load_average"] = [round(value, 3) for value in os.getloadavg()]
    except (OSError, AttributeError):
        pass
    for path in paths:
        try:
            usage = shutil.disk_usage(Path(path))
            snapshot["disk"][str(path)] = {
                "total_bytes": usage.total,
                "free_bytes": usage.free,
                "used_bytes": usage.used,
            }
        except OSError as exc:
            snapshot["disk"][str(path)] = {"error": f"{type(exc).__name__}: {exc}"}
    return snapshot


def compute_snapshot() -> dict[str, Any]:
    """Torch/MPS/CUDA availability without forcing an expensive import."""

    detail: dict[str, Any] = {"torch_imported": "torch" in sys.modules}
    torch = sys.modules.get("torch")
    if torch is None:
        return detail
    try:
        detail["torch_version"] = str(getattr(torch, "__version__", ""))
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is not None:
            detail["mps_available"] = bool(mps.is_available())
            detail["mps_built"] = bool(mps.is_built())
        cuda = getattr(torch, "cuda", None)
        if cuda is not None:
            detail["cuda_available"] = bool(cuda.is_available())
            if detail["cuda_available"]:
                detail["cuda_device_count"] = int(cuda.device_count())
    except Exception as exc:
        detail["error"] = f"{type(exc).__name__}: {exc}"
    return detail


def capture_thread_stacks(*, limit: int = 40) -> list[dict[str, Any]]:
    """Every live thread's stack.

    This is the evidence a hang otherwise never produces.  ``sys._current_frames``
    is a read-only snapshot and does not stop or interfere with the threads.
    """

    stacks: list[dict[str, Any]] = []
    try:
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        frames = sys._current_frames()
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    for ident, frame in list(frames.items())[:limit]:
        try:
            stacks.append(
                {
                    "thread_id": ident,
                    "thread_name": names.get(ident, "unknown"),
                    "is_current": ident == threading.get_ident(),
                    "stack": [
                        line.rstrip()
                        for line in traceback.format_stack(frame, limit=25)
                    ],
                }
            )
        except Exception as exc:
            stacks.append({"thread_id": ident, "error": f"{type(exc).__name__}: {exc}"})
    return stacks


def capture_child_processes(children: Sequence[Any] = ()) -> list[dict[str, Any]]:
    """Status of multiprocessing children without touching their internals."""

    result: list[dict[str, Any]] = []
    candidates = list(children)
    if not candidates:
        try:
            import multiprocessing

            candidates = list(multiprocessing.active_children())
        except Exception:
            candidates = []
    for child in candidates:
        try:
            result.append(
                {
                    "name": getattr(child, "name", ""),
                    "pid": getattr(child, "pid", None),
                    "alive": bool(child.is_alive()) if hasattr(child, "is_alive") else None,
                    "exitcode": getattr(child, "exitcode", None),
                    "daemon": getattr(child, "daemon", None),
                }
            )
        except Exception as exc:
            result.append({"error": f"{type(exc).__name__}: {exc}"})
    return result


def runtime_identity() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "frozen": bool(getattr(sys, "frozen", False)),
        "pid": os.getpid(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


# ---------------------------------------------------------------------------
# Postmortem
# ---------------------------------------------------------------------------


def capture_postmortem(
    tracker: OperationTracker,
    *,
    exception: BaseException | None = None,
    trigger: str = "failure",
    workers: Sequence[Mapping[str, Any]] = (),
    children: Sequence[Any] = (),
    queue_state: Mapping[str, Any] | None = None,
    pending_callbacks: Sequence[str] = (),
    disk_paths: Sequence[Path] = (),
    include_thread_stacks: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze everything useful *before* cleanup destroys it.

    Called only on failure or stall -- never on a successful path -- so the cost
    of walking thread stacks and probing disks is irrelevant to throughput.
    """

    if include_thread_stacks is None:
        include_thread_stacks = trigger in {"stall", "failure", "worker-death"}
    snapshot = tracker.snapshot()
    payload: dict[str, Any] = {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "trigger": trigger,
        "operation_id": tracker.operation_id,
        "operation": tracker.operation,
        "subsystem": tracker.subsystem,
        "current_phase": snapshot["current_phase"],
        "previous_phase": snapshot["previous_phase"],
        "last_successful_phase": snapshot["last_successful_phase"],
        "elapsed_seconds": snapshot["elapsed_seconds"],
        "phase_elapsed_seconds": snapshot["phase_elapsed_seconds"],
        "seconds_since_last_progress": snapshot["seconds_since_last_progress"],
        "stall_budget_seconds": snapshot["stall_budget_seconds"],
        "last_progress_detail": snapshot["last_progress_detail"],
        "counts": snapshot["counts"],
        "cancellation_state": {
            "cancelled": snapshot["cancelled"],
            "terminal": snapshot["terminal"],
            "terminal_reason": snapshot["terminal_reason"],
        },
        "operation_state_machine": snapshot,
        "runtime": runtime_identity(),
        "resources": resource_snapshot(disk_paths),
        "compute": compute_snapshot(),
        "workers": sanitize(list(workers)),
        # Per-worker liveness: distinguishes "busy" from "disappeared".
        "worker_heartbeats": tracker.worker_heartbeats(),
        "child_processes": capture_child_processes(children),
        "queue_state": sanitize(dict(queue_state or {})),
        "pending_callbacks": list(pending_callbacks),
        "recorder": recorder().counters(),
        "repeated_failures": recorder().repeat_summary(),
    }
    if exception is not None:
        marker = fingerprint_failure(
            exception,
            subsystem=tracker.subsystem,
            phase=snapshot["current_phase"],
            renderer=str(snapshot["metadata"].get("renderer", "")),
            serum_generation=str(snapshot["metadata"].get("target_synth", "")),
        )
        payload["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception),
            "chain": exception_chain(exception),
            "traceback": [
                line.rstrip()
                for line in traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            ],
            "fingerprint": marker.as_dict(),
        }
    elif snapshot.get("failure"):
        payload["exception"] = snapshot["failure"]
    if include_thread_stacks:
        payload["thread_stacks"] = capture_thread_stacks()
    if extra:
        payload["extra"] = sanitize(dict(extra))

    recorder().record(
        tracker.subsystem,
        "postmortem_captured",
        f"postmortem captured for {tracker.operation} ({trigger})",
        severity="error" if trigger != "manual" else "info",
        operation_id=tracker.operation_id,
        phase=snapshot["current_phase"],
        decision_reason=(
            f"{trigger} detected during {snapshot['current_phase']}; state frozen "
            "before recovery"
        ),
        trigger=trigger,
        thread_count=len(payload.get("thread_stacks", []) or []),
        worker_count=len(payload["workers"]),
        child_count=len(payload["child_processes"]),
    )
    return payload


def write_postmortem(payload: Mapping[str, Any], *, root: Path | None = None) -> Path | None:
    """Persist a postmortem next to the flight recorder."""

    import json

    try:
        target_root = Path(root) if root is not None else recorder().root
        if target_root is None:
            return None
        target_root.mkdir(parents=True, exist_ok=True)
        path = target_root / "postmortem.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(sanitize(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path
    except (OSError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Stall detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StallVerdict:
    """Why an operation is or is not considered stalled.

    ``progressing`` exists so a report can say "legitimately slow but
    progressing" instead of leaving a reader to guess.
    """

    stalled: bool
    phase: str
    seconds_since_progress: float
    budget_seconds: float
    reason: str
    progressing: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "stalled": self.stalled,
            "progressing": self.progressing,
            "phase": self.phase,
            "seconds_since_progress": round(self.seconds_since_progress, 3),
            "budget_seconds": round(self.budget_seconds, 3),
            "reason": self.reason,
        }


class StallDetector:
    """Decide whether an operation has stopped making progress.

    A stall is *not* "the operation took a long time".  Rendering 5,000 presets
    legitimately takes hours.  A stall is no meaningful progress for abnormally
    long **given the current phase**, which is why the budget is per phase and
    why every real advance calls :meth:`OperationTracker.mark_progress`.
    """

    def __init__(
        self,
        tracker: OperationTracker,
        *,
        overrides: Mapping[str, float] | None = None,
        clock: Clock | None = None,
        grace_multiplier: float = 1.0,
    ) -> None:
        self._tracker = tracker
        self._overrides = dict(overrides or tracker.stall_overrides or {})
        self._clock = clock or tracker.clock
        self._grace = max(1.0, float(grace_multiplier))

    def budget(self) -> float:
        return stall_budget_for(self._tracker.phase, self._overrides) * self._grace

    def evaluate(self) -> StallVerdict:
        phase = self._tracker.phase
        if phase in TERMINAL_PHASES:
            return StallVerdict(
                stalled=False,
                phase=phase,
                seconds_since_progress=0.0,
                budget_seconds=0.0,
                reason=f"operation already reached terminal phase {phase}",
                progressing=False,
            )
        idle = self._tracker.since_progress()
        budget = self.budget()
        if idle <= budget:
            return StallVerdict(
                stalled=False,
                phase=phase,
                seconds_since_progress=idle,
                budget_seconds=budget,
                reason=(
                    f"last progress {idle:.1f}s ago, within the {budget:.0f}s "
                    f"inactivity budget for {phase}"
                ),
                progressing=True,
            )

        # Past the budget with no completed work. Before declaring a stall, look
        # for corroborating evidence that workers are alive and busy: one render
        # batch can legitimately outlast a phase budget. The extension is
        # bounded, so a worker that heartbeats "rendering" forever without ever
        # finishing anything is still caught.
        heartbeat_age = self._tracker.busy_heartbeat_age()
        extended = budget * HEARTBEAT_BUDGET_MULTIPLIER
        if heartbeat_age is not None and heartbeat_age <= budget and idle <= extended:
            return StallVerdict(
                stalled=False,
                phase=phase,
                seconds_since_progress=idle,
                budget_seconds=extended,
                reason=(
                    f"no completed work for {idle:.1f}s during {phase}, but a "
                    f"worker reported a busy state {heartbeat_age:.1f}s ago, so "
                    f"the budget is extended to {extended:.0f}s"
                ),
                progressing=False,
            )
        if heartbeat_age is None:
            detail = "no worker has reported a busy state"
        elif heartbeat_age > budget:
            detail = f"the last busy worker heartbeat was {heartbeat_age:.1f}s ago"
        else:
            detail = (
                f"workers are heartbeating but nothing has completed within "
                f"{extended:.0f}s"
            )
        return StallVerdict(
            stalled=True,
            phase=phase,
            seconds_since_progress=idle,
            budget_seconds=budget,
            reason=(
                f"no measurable progress for {idle:.1f}s during {phase}, which "
                f"exceeds its {budget:.0f}s inactivity budget; {detail}"
            ),
            progressing=False,
        )


class StallWatchdog:
    """Background thread that detects a stall, snapshots it, then recovers.

    Ordering is deliberate and load-bearing: the postmortem is captured
    **before** ``on_stall`` runs, because recovery terminates the pool and
    destroys exactly the worker and stack state that explains the hang.
    """

    def __init__(
        self,
        tracker: OperationTracker,
        *,
        on_stall: Callable[[StallVerdict, dict[str, Any]], None],
        interval_seconds: float = 5.0,
        overrides: Mapping[str, float] | None = None,
        postmortem_factory: Callable[[StallVerdict], dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._tracker = tracker
        self._detector = StallDetector(tracker, overrides=overrides)
        self._on_stall = on_stall
        self._interval = max(0.01, float(interval_seconds))
        self._postmortem_factory = postmortem_factory
        self._sleep = sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fired = False
        self.verdict: StallVerdict | None = None
        self.postmortem: dict[str, Any] | None = None

    def check_once(self) -> StallVerdict:
        """Evaluate once; fire recovery if stalled.  Used directly by tests."""

        verdict = self._detector.evaluate()
        if not verdict.stalled or self.fired:
            return verdict
        self.fired = True
        self.verdict = verdict
        recorder().record(
            self._tracker.subsystem,
            "stall_detected",
            verdict.reason,
            severity="error",
            operation_id=self._tracker.operation_id,
            phase=verdict.phase,
            decision_reason=(
                "inactivity exceeded this phase's budget; capturing state before "
                "recovery"
            ),
            # Nested, not splatted: the verdict carries its own `phase` key and
            # would collide with the event's phase argument.
            **{
                key: value
                for key, value in verdict.as_dict().items()
                if key != "phase"
            },
        )
        try:
            self.postmortem = (
                self._postmortem_factory(verdict)
                if self._postmortem_factory is not None
                else capture_postmortem(self._tracker, trigger="stall")
            )
        except Exception as exc:  # evidence capture must not block recovery
            recorder().record(
                self._tracker.subsystem,
                "postmortem_failed",
                f"stall postmortem capture failed: {type(exc).__name__}: {exc}",
                severity="warning",
                operation_id=self._tracker.operation_id,
                phase=verdict.phase,
                exception=exc,
            )
            self.postmortem = None
        self._on_stall(verdict, self.postmortem or {})
        return verdict

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            name=f"patchlab-stall-watchdog-{self._tracker.operation_id}",
            daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._tracker.terminal:
                return
            try:
                verdict = self.check_once()
            except Exception:
                return
            if verdict.stalled:
                return
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def __enter__(self) -> "StallWatchdog":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Worker heartbeats
# ---------------------------------------------------------------------------

WORKER_STATES = (
    "starting",
    "initializing-renderer",
    "renderer-ready",
    "waiting-for-task",
    "loading-preset",
    "rendering",
    "analyzing",
    "initialization-failed",
    "shutting-down",
)


def emit_worker_heartbeat(
    *,
    operation_id: str,
    worker_id: str,
    state: str,
    subsystem: str = "render-worker",
    phase: str = "",
    action: str = "",
    completed: int | None = None,
    **fields: Any,
) -> None:
    """Emit one lightweight worker heartbeat.

    Rate limited inside the recorder (``event_type="heartbeat"``), so calling
    this from a render loop is safe and cannot become log spam.  The parent uses
    it to tell "alive and busy" apart from "disappeared or deadlocked".
    """

    recorder().record(
        subsystem,
        "heartbeat",
        action or state,
        operation_id=operation_id,
        worker_id=worker_id,
        phase=phase,
        worker_state=state,
        worker_pid=os.getpid(),
        completed=completed,
        **fields,
    )


__all__ = [
    "DEFAULT_PHASE_STALL_SECONDS",
    "LIBRARY_PHASES",
    "MATCH_PHASES",
    "OperationTracker",
    "PhaseRecord",
    "StallDetector",
    "StallVerdict",
    "StallWatchdog",
    "TERMINAL_PHASES",
    "WORKER_STATES",
    "capture_child_processes",
    "capture_postmortem",
    "capture_thread_stacks",
    "compute_snapshot",
    "emit_worker_heartbeat",
    "resource_snapshot",
    "runtime_identity",
    "stall_budget_for",
    "start_operation",
    "write_postmortem",
]
