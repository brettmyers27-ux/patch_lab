"""Always-on bounded diagnostic flight recorder.

PatchLab's hardest bugs happen on somebody else's Mac: a plug-in that is not
installed, a frozen-bundle path that does not resolve, a render worker that
dies before it ever consumes a task.  Reproducing those locally is usually
impossible, so a single submitted support bundle has to carry enough evidence
to diagnose them.

This module records a structured, machine-readable timeline of what PatchLab
did, *what it believed*, and **why it decided what it decided**.  It is
deliberately cheap:

* every event goes into an in-memory ring buffer first;
* a background writer thread batches them to a rotating JSON Lines file, so no
  user-visible code path ever waits on disk;
* identical repeated failures are aggregated by fingerprint instead of writing
  the same traceback hundreds of times;
* progress and heartbeat events are rate limited;
* expensive facts (file hashes) are cached against size+mtime.

Nothing here may raise into a caller.  A diagnostics bug must never become a
PatchLab bug, so every public entry point is failure-tolerant.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Version 1 was the single unstructured "Application log" pane captured by
# app/ui.py's diagnostic report.  Version 2 adds the structured flight
# recorder, environment/postmortem snapshots, decision records, correlation
# IDs, failure fingerprints and the reproduction descriptor.
#
# A future debugging agent reads this to tell "the field was unavailable" apart
# from "the user ran an older diagnostics format".  Bump it whenever the
# meaning or presence of bundle fields changes.
DIAGNOSTIC_SCHEMA_VERSION = 2

SEVERITIES = ("debug", "info", "warning", "error", "critical")


# ---------------------------------------------------------------------------
# Retention limits.  These are the numbers quoted in the support bundle so a
# reader knows exactly how much history they are looking at.
# ---------------------------------------------------------------------------

#: Events kept in memory.  At PatchLab's observed steady-state rate (a few
#: events per second during Match/Render, near zero when idle) this comfortably
#: covers the most recent 10-30 minutes of activity, which is the window that
#: matters for reconstructing a hang or a failure.
RING_CAPACITY = 4096

#: One event file never exceeds this, and at most MAX_EVENT_FILES are retained,
#: so the flight recorder's disk budget is hard-capped at 16 MiB per install.
MAX_EVENT_FILE_BYTES = 2 * 1024 * 1024
MAX_EVENT_FILES = 8

#: Rotated files older than this are pruned on startup so an install that ran
#: months ago does not keep stale evidence around forever.
MAX_EVENT_FILE_AGE_SECONDS = 72 * 60 * 60

#: How much history a support bundle freezes out of the recorder.
BUNDLE_HISTORY_SECONDS = 30 * 60

#: Writer batching.  The background thread drains everything available and
#: flushes at most this often, so a burst of events costs one write.
FLUSH_INTERVAL_SECONDS = 1.0
FLUSH_BATCH_EVENTS = 256

#: A single normalized failure is reported in full once, then counted.
REPEAT_FULL_DETAIL_LIMIT = 1

EVENTS_FILENAME = "events.jsonl"
HASH_CACHE_FILENAME = "file-hash-cache.json"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

#: Field names whose values are never recorded, whatever the caller passes.
#: Centralised on purpose: no logging call site is expected to remember what is
#: sensitive.  Matching is substring-based on the case-folded key.
_SECRET_KEY_TOKENS = (
    "password",
    "passcode",
    "secret",
    "token",
    "credential",
    "authorization",
    "auth_header",
    "cookie",
    "session_key",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "signing",
    "certificate",
    "bearer",
    "oauth",
)

#: Values that look like credentials are redacted even under an innocent key.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)\b(?:api[-_]?key|token|passcode|password)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
)

REDACTED = "[redacted]"

#: Raw content PatchLab must never place in diagnostics even by accident.
#: Callers pass paths and counts; bytes-like payloads are dropped here.
_MAX_VALUE_CHARACTERS = 4_000
_MAX_COLLECTION_ITEMS = 64


def _looks_secret_key(key: str) -> bool:
    lowered = str(key).casefold()
    return any(token in lowered for token in _SECRET_KEY_TOKENS)


def redact_text(value: str) -> str:
    """Strip credential-shaped substrings from free text."""

    result = str(value)
    for pattern in _SECRET_VALUE_PATTERNS:
        result = pattern.sub(REDACTED, result)
    return result


def sanitize(value: Any, *, _depth: int = 0) -> Any:
    """Return a JSON-serialisable, credential-free copy of ``value``.

    File paths are intentionally retained: on a remote-machine bug the plug-in
    path or bundle resource path is usually *the* answer.  Keeping that policy
    in this one function is the point -- privacy behaviour can later be
    tightened centrally (for example hashing home directories) without
    revisiting a single logging call site.
    """

    if _depth > 6:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        # NaN/inf are not valid JSON and silently break a whole bundle.
        return value if -1e308 < value < 1e308 else str(value)
    if isinstance(value, str):
        text = redact_text(value)
        return text if len(text) <= _MAX_VALUE_CHARACTERS else text[:_MAX_VALUE_CHARACTERS] + "…"
    if isinstance(value, (bytes, bytearray, memoryview)):
        # Raw audio, preset binaries and model weights all arrive as bytes.
        # Record only the shape, never the content.
        return f"[{len(bytes(value))} bytes omitted]"
    if isinstance(value, Path):
        return redact_text(str(value))
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {redact_text(str(value))}"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                result["…"] = f"[{len(value) - _MAX_COLLECTION_ITEMS} more keys omitted]"
                break
            name = str(key)
            result[name] = REDACTED if _looks_secret_key(name) else sanitize(item, _depth=_depth + 1)
        return result
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        items = list(value)
        trimmed = [sanitize(item, _depth=_depth + 1) for item in items[:_MAX_COLLECTION_ITEMS]]
        if len(items) > _MAX_COLLECTION_ITEMS:
            trimmed.append(f"[{len(items) - _MAX_COLLECTION_ITEMS} more items omitted]")
        return trimmed
    if hasattr(value, "as_dict") and callable(value.as_dict):
        try:
            return sanitize(value.as_dict(), _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dataclass_fields__"):
        try:
            from dataclasses import asdict

            return sanitize(asdict(value), _depth=_depth + 1)
        except Exception:
            pass
    return sanitize(repr(value), _depth=_depth + 1)


# ---------------------------------------------------------------------------
# Failure fingerprinting
# ---------------------------------------------------------------------------

#: Collapse numbers that are volatile (PIDs, counts, retry totals, sizes) while
#: preserving digits glued to a word, because those are product identity, not
#: noise: "serum1" and "serum2" -- and "VST2" vs "VST3" -- must never normalize
#: to the same fingerprint, which is exactly the distinction both reported bugs
#: turn on.
_DIGITS = re.compile(r"(?<![0-9A-Za-z])\d+")
_HEXISH = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
_QUOTED = re.compile(r"""(['"])(?:(?!\1).)*\1""")
_PATHISH = re.compile(r"(?:/[^/\s:,)]+){2,}")
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def normalize_error_text(text: str) -> str:
    """Collapse the volatile parts of an error message.

    Two occurrences of the same defect must normalize to the same string, so
    paths, PIDs, counts, hex digests and pointer addresses are replaced with
    stable placeholders.  Without this every worker failure would fingerprint
    uniquely and repeat aggregation could never fire.
    """

    result = redact_text(str(text)).strip()
    result = _ADDRESS.sub("<addr>", result)
    result = _PATHISH.sub("<path>", result)
    result = _QUOTED.sub("<str>", result)
    result = _HEXISH.sub("<hex>", result)
    result = _DIGITS.sub("<n>", result)
    return re.sub(r"\s+", " ", result)[:400]


@dataclass(frozen=True, slots=True)
class FailureFingerprint:
    """A stable identity for one failure *class*.

    Deliberately excludes timestamps, PIDs, counts and absolute paths so a new
    ticket can be recognised as "probably the same failure as a previous one".
    """

    digest: str
    exception_class: str
    normalized_error: str
    subsystem: str
    phase: str
    code_location: str
    renderer: str
    serum_generation: str
    schema_version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "exception_class": self.exception_class,
            "normalized_error": self.normalized_error,
            "subsystem": self.subsystem,
            "phase": self.phase,
            "code_location": self.code_location,
            "renderer": self.renderer,
            "serum_generation": self.serum_generation,
            "diagnostic_schema_version": self.schema_version,
        }


def _code_location(exception: BaseException | None) -> str:
    """Return ``module.py:line`` for the deepest PatchLab frame."""

    if exception is None or exception.__traceback__ is None:
        return ""
    chosen = ""
    for frame, lineno in traceback.walk_tb(exception.__traceback__):
        filename = frame.f_code.co_filename.replace("\\", "/")
        name = frame.f_code.co_name
        short = "/".join(filename.rsplit("/", 2)[-2:])
        entry = f"{short}:{lineno}:{name}"
        # Prefer PatchLab's own frames over interpreter/library frames: a
        # StopIteration raised inside multiprocessing/pool.py is useless, the
        # core/matcher.py frame that produced it is the answer.
        if "/core/" in filename or "/app/" in filename or "/scripts/" in filename:
            chosen = entry
        elif not chosen:
            chosen = entry
    return chosen


def fingerprint_failure(
    exception: BaseException | None,
    *,
    subsystem: str = "",
    phase: str = "",
    renderer: str = "",
    serum_generation: str = "",
    message: str = "",
    code_location: str = "",
) -> FailureFingerprint:
    exception_class = type(exception).__name__ if exception is not None else "None"
    raw = message or (str(exception) if exception is not None else "")
    # A bare StopIteration stringifies to "", which would fingerprint every
    # exhausted-iterator bug identically. Fall back to the code location.
    normalized = normalize_error_text(raw)
    location = code_location or _code_location(exception)
    if not location:
        # An exception captured before it was raised has no traceback. Say so
        # rather than leaving the field blank, which a reader cannot tell apart
        # from "this diagnostics version did not collect it".
        location = f"unavailable (no traceback) in {subsystem or '?'}/{phase or '?'}"
    parts = (
        exception_class,
        normalized,
        str(subsystem),
        str(phase),
        location,
        str(renderer),
        str(serum_generation),
        str(DIAGNOSTIC_SCHEMA_VERSION),
    )
    digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()[:16]
    return FailureFingerprint(
        digest=digest,
        exception_class=exception_class,
        normalized_error=normalized,
        subsystem=str(subsystem),
        phase=str(phase),
        code_location=location,
        renderer=str(renderer),
        serum_generation=str(serum_generation),
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
    )


def exception_chain(exception: BaseException | None) -> list[dict[str, Any]]:
    """Return the full ``__cause__``/``__context__`` chain with tracebacks.

    Preserving this is the difference between "a worker failed" and "a worker
    failed *because* no Serum 1 VST2 binary exists at this path".
    """

    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = exception
    # ``link`` describes how this entry relates to the one *before* it, so the
    # first (outermost) exception has no link and each following entry says
    # whether it was an explicit ``raise ... from`` cause or an implicit context.
    link = ""
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "module": type(current).__module__,
                "message": redact_text(str(current)),
                "traceback": [
                    redact_text(line.rstrip())
                    for line in traceback.format_tb(current.__traceback__)
                ],
                "link": link,
            }
        )
        if current.__cause__ is not None:
            current, link = current.__cause__, "cause"
        elif not current.__suppress_context__ and current.__context__ is not None:
            current, link = current.__context__, "context"
        else:
            break
    return chain


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiagnosticEvent:
    """One structured flight-recorder entry."""

    timestamp: str
    monotonic: float
    severity: str
    session_id: str
    operation_id: str
    subsystem: str
    event_type: str
    phase: str
    process_id: int
    worker_id: str
    message: str
    fields: dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""
    fingerprint: str = ""
    repeat_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ts": self.timestamp,
            "mono": round(self.monotonic, 6),
            "severity": self.severity,
            "session_id": self.session_id,
            "operation_id": self.operation_id,
            "subsystem": self.subsystem,
            "event_type": self.event_type,
            "phase": self.phase,
            "pid": self.process_id,
            "worker_id": self.worker_id,
            "message": self.message,
        }
        if self.fields:
            payload["fields"] = self.fields
        if self.decision_reason:
            payload["decision_reason"] = self.decision_reason
        if self.fingerprint:
            payload["fingerprint"] = self.fingerprint
        if self.repeat_count != 1:
            payload["repeat_count"] = self.repeat_count
        return payload


@dataclass(slots=True)
class RepeatGroup:
    """Aggregation state for one repeatedly occurring failure."""

    fingerprint: str
    first_event: dict[str, Any]
    first_timestamp: str
    first_monotonic: float
    latest_timestamp: str
    latest_monotonic: float
    count: int = 1
    worker_examples: list[str] = field(default_factory=list)
    pid_examples: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        window = max(self.latest_monotonic - self.first_monotonic, 0.0)
        return {
            "fingerprint": self.fingerprint,
            "total_occurrences": self.count,
            "first_timestamp": self.first_timestamp,
            "latest_timestamp": self.latest_timestamp,
            "window_seconds": round(window, 3),
            "occurrences_per_second": (
                round(self.count / window, 3) if window > 0.05 else None
            ),
            "worker_examples": self.worker_examples[:10],
            "pid_examples": self.pid_examples[:10],
            "first_full_event": self.first_event,
        }

    def summary_line(self) -> str:
        window = max(self.latest_monotonic - self.first_monotonic, 0.0)
        repeats = self.count - REPEAT_FULL_DETAIL_LIMIT
        event_type = str(self.first_event.get("event_type", "event"))
        return (
            f"{event_type}: same fingerprint {self.fingerprint} repeated "
            f"{repeats} additional time(s) over {window:.1f}s "
            f"({self.count} total)"
        )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Allow one event per key per interval; count what was dropped."""

    def __init__(self, interval_seconds: float) -> None:
        self._interval = float(interval_seconds)
        self._last: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}

    def allow(self, key: str, now: float) -> int | None:
        """Return the suppressed count to report, or None to drop the event."""

        last = self._last.get(key)
        if last is None or (now - last) >= self._interval:
            self._last[key] = now
            return self._suppressed.pop(key, 0)
        self._suppressed[key] = self._suppressed.get(key, 0) + 1
        return None


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


def diagnostics_root() -> Path:
    """Return the directory holding flight-recorder state.

    ``PATCHLAB_DIAGNOSTICS_DIR`` overrides it, which is what the test suite and
    the spawned worker processes use so a child writes into the same place as
    its parent.
    """

    override = os.environ.get("PATCHLAB_DIAGNOSTICS_DIR", "").strip()
    if override:
        root = Path(override).expanduser()
    else:
        from core.platform_env import ENV

        root = ENV.app_data_dir / "diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    return root


class DiagnosticRecorder:
    """Bounded, always-on, append-mostly structured event recorder.

    Threading model: ``record`` only touches the in-memory ring plus a queue,
    both under a short lock, and never performs I/O.  A daemon writer thread
    batches to disk.  That keeps the recorder off the UI thread's critical path
    and stops it from competing with Serum rendering or model inference.
    """

    def __init__(
        self,
        *,
        root: Path | None = None,
        session_id: str | None = None,
        capacity: int = RING_CAPACITY,
        enabled: bool | None = None,
        progress_interval_seconds: float = 1.0,
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        self._lock = threading.RLock()
        self._ring: deque[DiagnosticEvent] = deque(maxlen=max(16, int(capacity)))
        self._repeats: dict[str, RepeatGroup] = {}
        self._queue: queue.Queue[DiagnosticEvent | None] = queue.Queue(maxsize=8192)
        self._writer: threading.Thread | None = None
        self._stopping = threading.Event()
        self._dropped_events = 0
        self._write_errors = 0
        self._bytes_written = 0
        self._progress_limit = _RateLimiter(progress_interval_seconds)
        self._heartbeat_limit = _RateLimiter(heartbeat_interval_seconds)
        self._started_monotonic = time.monotonic()
        self._started_utc = datetime.now(timezone.utc).isoformat()
        self._hash_cache: dict[str, Any] | None = None
        self._hash_cache_dirty = False

        if enabled is None:
            enabled = os.environ.get("PATCHLAB_DIAGNOSTICS", "1").strip() != "0"
        self._enabled = bool(enabled)

        # The GUI creates the session; every worker inherits it through the
        # environment so one operation can be reconstructed across processes.
        self.session_id = (
            session_id
            or os.environ.get("PATCHLAB_SESSION_ID", "").strip()
            or uuid.uuid4().hex[:16]
        )
        os.environ.setdefault("PATCHLAB_SESSION_ID", self.session_id)

        self._root: Path | None = None
        if self._enabled:
            try:
                self._root = Path(root) if root is not None else diagnostics_root()
                self._root.mkdir(parents=True, exist_ok=True)
                self._prune_old_files()
            except OSError:
                # A full or read-only disk disables persistence but keeps the
                # in-memory ring, which is what a bundle needs most.
                self._root = None

    # -- properties --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def root(self) -> Path | None:
        return self._root

    @property
    def events_path(self) -> Path | None:
        return None if self._root is None else self._root / EVENTS_FILENAME

    def retention_policy(self) -> dict[str, Any]:
        """Describe the exact bounds, for the support bundle."""

        return {
            "ring_capacity_events": self._ring.maxlen,
            "max_event_file_bytes": MAX_EVENT_FILE_BYTES,
            "max_event_files": MAX_EVENT_FILES,
            "max_total_event_bytes": MAX_EVENT_FILE_BYTES * MAX_EVENT_FILES,
            "max_event_file_age_seconds": MAX_EVENT_FILE_AGE_SECONDS,
            "bundle_history_seconds": BUNDLE_HISTORY_SECONDS,
            "flush_interval_seconds": FLUSH_INTERVAL_SECONDS,
            "flush_batch_events": FLUSH_BATCH_EVENTS,
            "repeat_full_detail_limit": REPEAT_FULL_DETAIL_LIMIT,
            "persistence_enabled": self._root is not None,
            "events_path": str(self.events_path) if self.events_path else None,
        }

    def counters(self) -> dict[str, int]:
        with self._lock:
            return {
                "events_in_ring": len(self._ring),
                "dropped_events": self._dropped_events,
                "write_errors": self._write_errors,
                "bytes_written": self._bytes_written,
                "repeat_groups": len(self._repeats),
            }

    # -- recording ---------------------------------------------------------

    def record(
        self,
        subsystem: str,
        event_type: str,
        message: str = "",
        *,
        severity: str = "info",
        operation_id: str = "",
        phase: str = "",
        worker_id: str = "",
        decision_reason: str = "",
        fingerprint: str = "",
        exception: BaseException | None = None,
        **fields: Any,
    ) -> DiagnosticEvent | None:
        """Record one event.  Never raises."""

        if not self._enabled:
            return None
        try:
            return self._record_unsafe(
                subsystem,
                event_type,
                message,
                severity=severity,
                operation_id=operation_id,
                phase=phase,
                worker_id=worker_id,
                decision_reason=decision_reason,
                fingerprint=fingerprint,
                exception=exception,
                fields=fields,
            )
        except Exception:
            # Diagnostics must never take PatchLab down with them.
            with self._lock:
                self._write_errors += 1
            return None

    def _record_unsafe(
        self,
        subsystem: str,
        event_type: str,
        message: str,
        *,
        severity: str,
        operation_id: str,
        phase: str,
        worker_id: str,
        decision_reason: str,
        fingerprint: str,
        exception: BaseException | None,
        fields: dict[str, Any],
    ) -> DiagnosticEvent | None:
        now = time.monotonic()
        payload = sanitize(fields) if fields else {}
        if exception is not None:
            payload["exception"] = {
                "type": type(exception).__name__,
                "message": redact_text(str(exception)),
                "chain": exception_chain(exception),
            }
            if not fingerprint:
                fingerprint = fingerprint_failure(
                    exception,
                    subsystem=subsystem,
                    phase=phase,
                    renderer=str(fields.get("renderer", "")),
                    serum_generation=str(fields.get("serum_generation", "")),
                    message=message,
                ).digest

        event = DiagnosticEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            monotonic=now - self._started_monotonic,
            severity=severity if severity in SEVERITIES else "info",
            session_id=self.session_id,
            operation_id=str(operation_id or ""),
            subsystem=str(subsystem),
            event_type=str(event_type),
            phase=str(phase or ""),
            process_id=os.getpid(),
            worker_id=str(worker_id or ""),
            message=redact_text(message),
            fields=payload,
            decision_reason=redact_text(decision_reason) if decision_reason else "",
            fingerprint=fingerprint,
        )

        # Rate-limit the two genuinely high-frequency families.  A dropped
        # progress tick is invisible; a dropped decision or failure is not, so
        # only these two are limited.
        if event_type in {"progress", "heartbeat"}:
            limiter = (
                self._progress_limit if event_type == "progress" else self._heartbeat_limit
            )
            # The key includes the phase and the worker state on purpose. Routine
            # repetition ("rendering" 500 times) collapses to one event, while a
            # genuine transition ("renderer-ready" -> "initialization-failed")
            # always gets through -- suppressing a state change would destroy
            # exactly the signal needed to tell a busy worker from a dead one.
            discriminator = str(payload.get("worker_state", "")) or str(phase or "")
            key = f"{operation_id}|{subsystem}|{worker_id}|{event_type}|{discriminator}"
            suppressed = limiter.allow(key, now)
            if suppressed is None:
                return None
            if suppressed:
                event.fields["suppressed_since_last"] = suppressed

        with self._lock:
            if fingerprint:
                # Group by event type as well as fingerprint: a heartbeat that
                # merely references a failure's fingerprint must never be
                # aggregated into that failure's group and have its own fields
                # replaced.
                group_key = f"{event.event_type}|{fingerprint}"
                group = self._repeats.get(group_key)
                if group is None:
                    self._repeats[group_key] = RepeatGroup(
                        fingerprint=fingerprint,
                        first_event=event.as_dict(),
                        first_timestamp=event.timestamp,
                        first_monotonic=event.monotonic,
                        latest_timestamp=event.timestamp,
                        latest_monotonic=event.monotonic,
                        worker_examples=[event.worker_id] if event.worker_id else [],
                        pid_examples=[event.process_id],
                    )
                else:
                    group.count += 1
                    group.latest_timestamp = event.timestamp
                    group.latest_monotonic = event.monotonic
                    if event.worker_id and event.worker_id not in group.worker_examples:
                        group.worker_examples.append(event.worker_id)
                    if event.process_id not in group.pid_examples:
                        group.pid_examples.append(event.process_id)
                    if group.count > REPEAT_FULL_DETAIL_LIMIT:
                        # Keep the first complete traceback and stop writing
                        # the identical one again.  The count, the window and
                        # the affected workers are all preserved above.
                        event.repeat_count = group.count
                        event.fields = {
                            "aggregated": True,
                            "total_occurrences": group.count,
                            "first_timestamp": group.first_timestamp,
                            "detail": "suppressed; identical to first occurrence",
                        }
                        event.message = group.summary_line()
            self._ring.append(event)

        self._enqueue(event)
        return event

    def record_decision(
        self,
        subsystem: str,
        decision: str,
        *,
        outcome: str,
        reason: str,
        operation_id: str = "",
        phase: str = "",
        candidates: Iterable[Any] = (),
        **fields: Any,
    ) -> DiagnosticEvent | None:
        """Record *why* PatchLab chose what it chose.

        The reason and the rejected alternatives are the point.  "selected
        Serum2/VST3" is not diagnosable; "selected Serum2/VST3 because it is
        the highest-preference validated renderer, Serum2/AU rejected: path
        does not exist, Serum1/VST2 not applicable to a serum2 request" is.
        """

        return self.record(
            subsystem,
            "decision",
            f"{decision} -> {outcome}",
            severity="info",
            operation_id=operation_id,
            phase=phase,
            decision_reason=reason,
            decision=decision,
            outcome=outcome,
            candidates=list(candidates),
            **fields,
        )

    def record_failure(
        self,
        subsystem: str,
        event_type: str,
        exception: BaseException,
        *,
        message: str = "",
        operation_id: str = "",
        phase: str = "",
        worker_id: str = "",
        renderer: str = "",
        serum_generation: str = "",
        **fields: Any,
    ) -> FailureFingerprint:
        """Record a failure with its full chain and return its fingerprint."""

        marker = fingerprint_failure(
            exception,
            subsystem=subsystem,
            phase=phase,
            renderer=renderer,
            serum_generation=serum_generation,
            message=message,
        )
        self.record(
            subsystem,
            event_type,
            message or f"{type(exception).__name__}: {exception}",
            severity="error",
            operation_id=operation_id,
            phase=phase,
            worker_id=worker_id,
            exception=exception,
            fingerprint=marker.digest,
            failure_fingerprint=marker.as_dict(),
            renderer=renderer,
            serum_generation=serum_generation,
            **fields,
        )
        return marker

    # -- reading -----------------------------------------------------------

    def recent_events(
        self, *, since_seconds: float | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Freeze the recent in-memory history."""

        with self._lock:
            events = list(self._ring)
        if since_seconds is not None:
            cutoff = (time.monotonic() - self._started_monotonic) - float(since_seconds)
            events = [event for event in events if event.monotonic >= cutoff]
        if limit is not None and len(events) > limit:
            events = events[-limit:]
        return [event.as_dict() for event in events]

    def repeat_summary(self) -> list[dict[str, Any]]:
        """Aggregated view of every repeated failure, most frequent first."""

        with self._lock:
            groups = [group.as_dict() for group in self._repeats.values() if group.count > 1]
        groups.sort(key=lambda item: -int(item.get("total_occurrences", 0)))
        return groups

    # -- persistence -------------------------------------------------------

    def _enqueue(self, event: DiagnosticEvent) -> None:
        if self._root is None:
            return
        self._ensure_writer()
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Prefer losing the newest disk copy over blocking a render loop.
            # The ring buffer still holds it, and the counter makes the loss
            # visible in the bundle instead of silent.
            with self._lock:
                self._dropped_events += 1

    def _ensure_writer(self) -> None:
        if self._writer is not None and self._writer.is_alive():
            return
        with self._lock:
            if self._writer is not None and self._writer.is_alive():
                return
            self._stopping.clear()
            self._writer = threading.Thread(
                target=self._writer_loop,
                name="patchlab-diagnostics-writer",
                daemon=True,
            )
            self._writer.start()

    def _writer_loop(self) -> None:
        pending: list[DiagnosticEvent] = []
        while True:
            timeout = FLUSH_INTERVAL_SECONDS if pending else None
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._flush(pending)
                pending = []
                continue
            if item is None:
                self._flush(pending)
                return
            if isinstance(item, threading.Event):
                # A flush() barrier: everything queued before it must be on disk
                # before the caller is released.
                self._flush(pending)
                pending = []
                item.set()
                continue
            pending.append(item)
            if len(pending) >= FLUSH_BATCH_EVENTS:
                self._flush(pending)
                pending = []

    def _flush(self, events: list[DiagnosticEvent]) -> None:
        if not events or self._root is None:
            return
        path = self._root / EVENTS_FILENAME
        try:
            blob = "".join(
                json.dumps(event.as_dict(), separators=(",", ":"), default=str) + "\n"
                for event in events
            ).encode("utf-8")
            self._rotate_if_needed(path, len(blob))
            with path.open("ab") as handle:
                handle.write(blob)
            with self._lock:
                self._bytes_written += len(blob)
        except (OSError, ValueError, TypeError):
            with self._lock:
                self._write_errors += 1

    def _rotate_if_needed(self, path: Path, incoming: int) -> None:
        try:
            size = path.stat().st_size if path.exists() else 0
        except OSError:
            return
        if size + incoming <= MAX_EVENT_FILE_BYTES:
            return
        # events.jsonl -> events.1.jsonl -> ... -> events.{N-1}.jsonl, oldest dropped.
        for index in range(MAX_EVENT_FILES - 1, 0, -1):
            source = path.with_name(f"events.{index}.jsonl")
            if not source.exists():
                continue
            if index == MAX_EVENT_FILES - 1:
                source.unlink(missing_ok=True)
            else:
                source.replace(path.with_name(f"events.{index + 1}.jsonl"))
        path.replace(path.with_name("events.1.jsonl"))

    def rotated_event_files(self) -> list[Path]:
        """Newest first: the live file, then rotated history."""

        if self._root is None:
            return []
        files = [self._root / EVENTS_FILENAME]
        files.extend(
            self._root / f"events.{index}.jsonl" for index in range(1, MAX_EVENT_FILES)
        )
        return [path for path in files if path.is_file()]

    def _prune_old_files(self) -> None:
        if self._root is None:
            return
        cutoff = time.time() - MAX_EVENT_FILE_AGE_SECONDS
        for path in self._root.glob("events.*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue

    def flush(self, timeout: float = 2.0) -> None:
        """Block until every event recorded so far is on disk (or ``timeout``).

        A real barrier, not a heuristic: a marker is queued behind the pending
        events and the writer sets it only after writing them. (An earlier
        version returned as soon as the *queue* was empty, by which time the
        writer had already moved events into a local batch it would not write for
        another second -- so a process that flushed and then exited lost them.)
        """

        if self._root is None:
            return
        self._ensure_writer()
        marker = threading.Event()
        try:
            self._queue.put(marker, timeout=max(0.01, float(timeout)))
        except queue.Full:
            return
        marker.wait(max(0.0, float(timeout)))

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        writer = self._writer
        if writer is not None and writer.is_alive():
            writer.join(timeout=2.0)
        self._save_hash_cache()

    # -- cached expensive facts -------------------------------------------

    def _load_hash_cache(self) -> dict[str, Any]:
        if self._hash_cache is not None:
            return self._hash_cache
        cache: dict[str, Any] = {}
        if self._root is not None:
            path = self._root / HASH_CACHE_FILENAME
            try:
                if path.is_file():
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        cache = raw
            except (OSError, ValueError):
                cache = {}
        self._hash_cache = cache
        return cache

    def _save_hash_cache(self) -> None:
        if self._root is None or not self._hash_cache_dirty or self._hash_cache is None:
            return
        try:
            path = self._root / HASH_CACHE_FILENAME
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self._hash_cache, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(path)
            self._hash_cache_dirty = False
        except OSError:
            pass

    def cached_file_digest(self, path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> str:
        """Return a sha1 of ``path``, cached against its size and mtime.

        Plug-in bundles and model checkpoints are large and never change
        between runs, so hashing them on every Match would be pure waste.  A
        cache miss only happens when the underlying file actually changed.
        """

        try:
            resolved = Path(path)
            stat = resolved.stat()
        except OSError:
            return ""
        key = str(resolved)
        cache = self._load_hash_cache()
        entry = cache.get(key)
        if (
            isinstance(entry, dict)
            and int(entry.get("size", -1)) == stat.st_size
            and abs(float(entry.get("mtime", -1.0)) - stat.st_mtime) < 1e-6
        ):
            return str(entry.get("sha1", ""))
        if stat.st_size > max_bytes:
            digest = ""
        else:
            try:
                hasher = hashlib.sha1()
                with resolved.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
            except OSError:
                digest = ""
        cache[key] = {"size": stat.st_size, "mtime": stat.st_mtime, "sha1": digest}
        self._hash_cache_dirty = True
        self._save_hash_cache()
        return digest


# ---------------------------------------------------------------------------
# Process-wide recorder
# ---------------------------------------------------------------------------

_RECORDER: DiagnosticRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def _close_process_recorder() -> None:
    """Drain and close whatever recorder this process ended with.

    Registered once. The GUI, the packaged gates and any worker that forgets an
    explicit ``close()`` would otherwise lose up to the last second of events --
    which is exactly the lead-up to a failure a support bundle needs.
    """

    current = _RECORDER
    if current is None:
        return
    try:
        current.flush(timeout=2.0)
        current.close()
    except Exception:
        pass


_EXIT_HOOK_REGISTERED = False


def recorder() -> DiagnosticRecorder:
    """Return this process's recorder, creating it on first use."""

    global _RECORDER, _EXIT_HOOK_REGISTERED
    if _RECORDER is not None:
        return _RECORDER
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = DiagnosticRecorder()
        if not _EXIT_HOOK_REGISTERED:
            atexit.register(_close_process_recorder)
            _EXIT_HOOK_REGISTERED = True
    return _RECORDER


def set_recorder(value: DiagnosticRecorder | None) -> None:
    """Install a recorder.  Tests use this; workers use it after spawn."""

    global _RECORDER
    with _RECORDER_LOCK:
        _RECORDER = value


def reset_recorder() -> None:
    set_recorder(None)


def record(subsystem: str, event_type: str, message: str = "", **kwargs: Any) -> Any:
    """Module-level shorthand used throughout PatchLab."""

    return recorder().record(subsystem, event_type, message, **kwargs)


def record_decision(subsystem: str, decision: str, **kwargs: Any) -> Any:
    return recorder().record_decision(subsystem, decision, **kwargs)


def record_failure(
    subsystem: str, event_type: str, exception: BaseException, **kwargs: Any
) -> FailureFingerprint:
    return recorder().record_failure(subsystem, event_type, exception, **kwargs)


def new_operation_id(prefix: str) -> str:
    """Return a short, human-quotable correlation ID for one operation."""

    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def worker_identity(kind: str = "worker") -> str:
    """Stable per-process worker ID, usable as a correlation key."""

    return f"{kind}-{os.getpid()}"


def child_environment(operation_id: str = "", **extra: str) -> dict[str, str]:
    """Return environment overrides that carry correlation IDs across a spawn.

    This is how one operation stays reconstructable through QProcess workers
    and multiprocessing children.
    """

    current = recorder()
    values = {
        "PATCHLAB_SESSION_ID": current.session_id,
        "PATCHLAB_DIAGNOSTIC_SCHEMA": str(DIAGNOSTIC_SCHEMA_VERSION),
    }
    if operation_id:
        values["PATCHLAB_OPERATION_ID"] = operation_id
    if current.root is not None:
        values["PATCHLAB_DIAGNOSTICS_DIR"] = str(current.root)
    values.update({key: str(value) for key, value in extra.items()})
    return values


def inherited_operation_id() -> str:
    """The operation ID this process was spawned for, if any."""

    return os.environ.get("PATCHLAB_OPERATION_ID", "").strip()


def apply_child_environment(values: Mapping[str, str]) -> None:
    """Install inherited correlation IDs inside a freshly spawned process."""

    for key, value in values.items():
        if value:
            os.environ[key] = str(value)
    reset_recorder()


__all__ = [
    "BUNDLE_HISTORY_SECONDS",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "DiagnosticEvent",
    "DiagnosticRecorder",
    "FailureFingerprint",
    "MAX_EVENT_FILES",
    "MAX_EVENT_FILE_BYTES",
    "RING_CAPACITY",
    "RepeatGroup",
    "apply_child_environment",
    "child_environment",
    "diagnostics_root",
    "exception_chain",
    "fingerprint_failure",
    "inherited_operation_id",
    "new_operation_id",
    "normalize_error_text",
    "record",
    "record_decision",
    "record_failure",
    "recorder",
    "redact_text",
    "reset_recorder",
    "sanitize",
    "set_recorder",
    "worker_identity",
]
