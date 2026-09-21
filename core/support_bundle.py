"""Assemble the diagnostic support bundle that ships with a bug report.

The acceptance test for this module is a question: given only what a user's one
button-click produces, could another developer -- or a capable AI with no access
to that machine -- determine what was attempted, on what build, in what
environment, which renderer was selected and *why*, which were rejected and
why, which phase failed, which worker, the exact exception, the lead-up, whether
it was progressing or stalled, and enough configuration to write a regression
test?

Layout (a directory beside the existing plain-text ticket, plus a zip):

    summary.txt        human-readable overview and timeline
    events.jsonl       chronological structured flight-recorder events
    environment.json   machine/build/plug-in/model/library snapshot
    postmortem.json    failure/stall/process/worker/state snapshot
    repeats.json       aggregated repeated failures
    reproduction.json  sanitized machine-readable reproduction descriptor

The existing one-button user flow is unchanged: ``core.bug_report.create_request``
still writes its single readable ``.txt`` first, and this bundle is attached
next to it.  Nothing here depends on the private support service being
reachable.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.diagnostics import (
    BUNDLE_HISTORY_SECONDS,
    DIAGNOSTIC_SCHEMA_VERSION,
    recorder,
    redact_text,
    sanitize,
)


BUNDLE_DIRNAME_SUFFIX = " diagnostics"
SUMMARY_FILENAME = "summary.txt"
EVENTS_FILENAME = "events.jsonl"
ENVIRONMENT_FILENAME = "environment.json"
POSTMORTEM_FILENAME = "postmortem.json"
REPEATS_FILENAME = "repeats.json"
REPRODUCTION_FILENAME = "reproduction.json"
TICKET_FILENAME = "ticket.txt"

#: Hard cap on the events file inside a bundle, so a report is always sendable.
MAX_BUNDLE_EVENT_BYTES = 6 * 1024 * 1024


# ---------------------------------------------------------------------------
# Reproduction descriptor
# ---------------------------------------------------------------------------


def build_reproduction_descriptor(
    *,
    operation: str,
    operation_id: str = "",
    environment: Mapping[str, Any] | None = None,
    postmortem: Mapping[str, Any] | None = None,
    renderer_selection: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    feature_flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return just enough sanitized configuration to reconstruct the code path.

    Contains no raw audio, no preset bytes, no credentials and no model
    contents -- only the switches, counts and selection outcomes an engineer
    needs to write a regression test on a different machine.
    """

    env = dict(environment or {})
    post = dict(postmortem or {})
    app = dict(env.get("application") or {})
    system = dict(env.get("system") or {})
    state = dict(env.get("patchlab_state") or {})
    database = dict(env.get("database") or {})
    model = dict(env.get("model") or {})

    inventory = list(env.get("renderer_inventory") or [])
    descriptor: dict[str, Any] = {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "operation_id": operation_id or post.get("operation_id", ""),
        "platform": {
            "branch": system.get("branch"),
            "architecture": system.get("architecture"),
            "macos_version": system.get("macos_version"),
            "compute_backend": system.get("compute_backend"),
            "python_version": system.get("python_version"),
            "frozen": app.get("frozen"),
        },
        "build": {
            "patchlab_version": app.get("patchlab_version"),
            "source_commit": app.get("source_commit"),
            "distribution_mode": app.get("distribution_mode"),
        },
        "request": {
            "serum_generation": state.get("target_serum_version"),
            "quality_mode": state.get("quality_mode"),
            "linked_folder_present": bool(state.get("linked_folder")),
        },
        "renderer": {
            "selection": sanitize(dict(renderer_selection or {})),
            # Only shape, never the user's absolute plug-in paths' contents.
            "installed": [
                {
                    "renderer": item.get("renderer"),
                    "exists": item.get("exists"),
                    "readable": item.get("readable"),
                    "accepted": item.get("accepted"),
                    "rejection_reason": item.get("rejection_reason"),
                    "plugin_version": item.get("plugin_version"),
                }
                for item in inventory
            ],
        },
        "library_state": {
            "presets": database.get("presets"),
            "renders": database.get("renders"),
            "fingerprinted": database.get("fingerprinted"),
            "presets_by_synth_status": database.get("presets_by_synth_status"),
            "db_schema_version": database.get("schema_version"),
            "counts": state.get("library_counts"),
        },
        "model": {
            "checkpoint_size_bytes": model.get("checkpoint_size_bytes"),
            "checkpoint_sha1": model.get("checkpoint_sha1"),
            "device": model.get("device"),
        },
        "failure": {
            "phase": post.get("current_phase"),
            "previous_phase": post.get("previous_phase"),
            "last_successful_phase": post.get("last_successful_phase"),
            "trigger": post.get("trigger"),
            "exception_type": (post.get("exception") or {}).get("type"),
            "fingerprint": (
                ((post.get("exception") or {}).get("fingerprint") or {}).get("digest")
            ),
            "worker_count": len(post.get("workers") or []),
            "seconds_since_last_progress": post.get("seconds_since_last_progress"),
            "stall_budget_seconds": post.get("stall_budget_seconds"),
        },
        "settings": sanitize(dict(settings or {})),
        "feature_flags": sanitize(dict(feature_flags or {})),
        "excluded_by_policy": [
            "raw user audio",
            "raw preset bytes",
            "model weights",
            "credentials and tokens",
        ],
    }
    return descriptor


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _timeline_lines(events: Sequence[Mapping[str, Any]], *, limit: int = 60) -> list[str]:
    """A concise, chronological, human-readable timeline.

    Prefers the events that explain a failure -- phase changes, decisions,
    failures, stalls -- over routine progress ticks.
    """

    interesting = [
        event
        for event in events
        if str(event.get("event_type"))
        in {
            "operation_started",
            "phase_changed",
            "decision",
            "operation_failed",
            "operation_complete",
            "stall_detected",
            "postmortem_captured",
            "worker_init_failed",
            "worker_died",
            "renderer_unavailable",
            "preflight_failed",
            "classification_summary",
            "pool_shutdown",
            "batch_committed",
            # GUI-level decisions: what the user asked for and how the UI recovered.
            "render_requested",
            "render_started",
            "render_completed",
            "render_failed",
            "match_requested",
            "match_started",
            "match_completed",
            "match_failed",
            "output_blocked",
            "pending_processing_offered",
            "process_pending_started",
            "notice_acknowledged",
            "capability_changed",
            "presets_pending",
        }
        or str(event.get("severity")) in {"error", "critical", "warning"}
    ]
    chosen = interesting[-limit:] if len(interesting) > limit else interesting
    lines: list[str] = []
    for event in chosen:
        stamp = str(event.get("ts", ""))[11:19]
        phase = str(event.get("phase", ""))
        message = str(event.get("message", ""))
        reason = str(event.get("decision_reason", ""))
        repeats = event.get("repeat_count")
        suffix = f" (x{repeats})" if isinstance(repeats, int) and repeats > 1 else ""
        line = f"  [{stamp}] {event.get('subsystem','')}/{event.get('event_type','')}"
        if phase:
            line += f" ({phase})"
        line += f": {message}{suffix}"
        lines.append(line)
        if reason:
            lines.append(f"            why: {reason}")
    return lines


def build_summary(
    *,
    ticket_id: str,
    comments: str,
    environment: Mapping[str, Any] | None,
    postmortem: Mapping[str, Any] | None,
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Mapping[str, Any]],
) -> str:
    env = dict(environment or {})
    post = dict(postmortem or {})
    app = dict(env.get("application") or {})
    system = dict(env.get("system") or {})
    state = dict(env.get("patchlab_state") or {})
    exception = dict(post.get("exception") or {})
    fingerprint = dict(exception.get("fingerprint") or {})

    lines: list[str] = [
        "PATCHLAB SUPPORT BUNDLE",
        f"Ticket ID: {ticket_id}",
        f"Diagnostic schema version: {DIAGNOSTIC_SCHEMA_VERSION}",
        f"Created (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "=== USER COMMENTS ===",
        comments.strip() or "(none)",
        "",
        "=== BUILD ===",
        f"PatchLab {app.get('patchlab_version') or 'unknown'} "
        f"({'frozen' if app.get('frozen') else 'source'})",
        f"Source commit: {app.get('source_commit') or 'unknown'}"
        f"{' (dirty)' if app.get('source_dirty') else ''}",
        f"Built at: {app.get('built_at_utc') or 'unknown'}",
        f"Session ID: {app.get('session_id') or recorder().session_id}",
        "",
        "=== ENVIRONMENT ===",
        f"{system.get('system') or '?'} {system.get('macos_version') or system.get('os_release') or ''} "
        f"({system.get('macos_build') or ''}) · {system.get('architecture') or '?'}",
        f"Python {system.get('python_version') or '?'} · compute {system.get('compute_backend') or '?'}",
        f"CPU count: {system.get('cpu_count')} · RAM: {system.get('total_memory_bytes')} bytes",
    ]

    inventory = list(env.get("renderer_inventory") or [])
    if inventory:
        lines.extend(["", "=== RENDERER CANDIDATES (accept/reject with reasons) ==="])
        for item in inventory:
            verdict = "ACCEPTED" if item.get("accepted") else "rejected"
            lines.append(
                f"  {item.get('renderer')}: {verdict}"
                f"{' — ' + str(item.get('rejection_reason')) if item.get('rejection_reason') else ''}"
            )
            lines.append(
                f"      path={item.get('path')} exists={item.get('exists')} "
                f"readable={item.get('readable')} version={item.get('plugin_version') or '?'}"
            )

    capability = dict(env.get("synth_capability") or {})
    if capability.get("capabilities"):
        lines.extend(["", "=== INSTALLED SYNTHS (what PatchLab believed it could run) ==="])
        for name, item in dict(capability["capabilities"]).items():
            lines.append(
                f"  {name}: {item.get('status')} -- preferred "
                f"{item.get('preferred_renderer') or 'none'}; {item.get('reason')}"
            )

    coverage = dict(env.get("library_coverage") or {})
    if coverage.get("available"):
        lines.extend(
            [
                "",
                "=== LIBRARY COVERAGE (why only N of my M presets appear) ===",
                f"  discovered {coverage.get('discovered')} | learned {coverage.get('learned')} | "
                f"processed {coverage.get('processed_params')} | pending {coverage.get('pending')} | "
                f"failed {coverage.get('failed')}",
                f"  by format: {coverage.get('by_file_format')}",
                f"  by generation: {coverage.get('by_generation')}",
                f"  pending by reason: {coverage.get('pending_by_reason')}",
            ]
        )

    ui_events = [
        event
        for event in events
        if str(event.get("event_type")) in {"render_failed", "match_failed", "output_blocked"}
    ]
    if ui_events:
        lines.extend(["", "=== UI RECOVERY (what the user saw after each failure) ==="])
        # Keep the latest few of each kind, so a burst of one failure type
        # cannot push a different failure out of the summary.
        by_type: dict[str, list] = {}
        for event in ui_events:
            by_type.setdefault(str(event.get("event_type")), []).append(event)
        kept = {id(e) for group in by_type.values() for e in group[-2:]}
        for event in (e for e in ui_events if id(e) in kept):
            fields = dict(event.get("fields") or {})
            keep = {
                key: fields[key]
                for key in (
                    "link_phase", "render_phase", "retry_available", "controls_recovered",
                    "stale_activities", "source_audio_kept", "shown_to_user", "generation",
                )
                if key in fields
            }
            lines.append(f"  {event.get('event_type')}: {keep}")

    lines.extend(
        [
            "",
            "=== OPERATION ===",
            f"Operation: {post.get('operation') or env.get('operation') or 'unknown'}",
            f"Operation ID: {post.get('operation_id') or env.get('operation_id') or 'unknown'}",
            f"Requested Serum generation: {state.get('target_serum_version') or 'unknown'}",
            f"Quality mode: {state.get('quality_mode') or 'unknown'}",
            f"Linked folder: {state.get('linked_folder') or 'none'}",
        ]
    )

    if post:
        progressing = post.get("seconds_since_last_progress")
        budget = post.get("stall_budget_seconds")
        verdict = "unknown"
        if str(post.get("trigger")) == "failure" and exception:
            # An exception terminated this run, so "was it stalled?" is the wrong
            # question -- saying "progressing" here would mislead a reader into
            # looking for a hang that never happened.
            verdict = "not applicable — terminated by an exception, not a stall"
        elif isinstance(progressing, (int, float)) and isinstance(budget, (int, float)):
            verdict = (
                "STALLED (no meaningful progress within this phase's budget)"
                if progressing > budget
                else "progressing (slow but advancing, not stalled)"
            )
        lines.extend(
            [
                "",
                "=== FAILURE ===",
                f"Trigger: {post.get('trigger') or 'unknown'}",
                f"Failed during phase: {post.get('current_phase') or 'unknown'}",
                f"Previous phase: {post.get('previous_phase') or 'unknown'}",
                f"Last successful phase: {post.get('last_successful_phase') or 'none'}",
                f"Elapsed: {post.get('elapsed_seconds')}s "
                f"(phase {post.get('phase_elapsed_seconds')}s)",
                f"Liveness: {verdict} "
                f"(last progress {progressing}s ago, budget {budget}s)",
                f"Exception: {exception.get('type') or 'none'}: "
                f"{exception.get('message') or ''}",
                f"Failure fingerprint: {fingerprint.get('digest') or 'none'}",
                f"  normalized: {fingerprint.get('normalized_error') or ''}",
                f"  location: {fingerprint.get('code_location') or ''}",
                f"  subsystem/phase: {fingerprint.get('subsystem') or ''}/"
                f"{fingerprint.get('phase') or ''}",
                f"UI recovery state: {post.get('ui_recovery') or (post.get('extra') or {}).get('ui_recovery') or 'see postmortem.json'}",
            ]
        )
        workers = list(post.get("workers") or [])
        if workers:
            lines.append(f"Workers ({len(workers)}):")
            for worker in workers[:12]:
                lines.append(f"  {json.dumps(worker, default=str)[:300]}")

    if repeats:
        lines.extend(["", "=== REPEATED FAILURES (aggregated) ==="])
        for group in repeats:
            lines.append(
                f"  {group.get('fingerprint')}: {group.get('total_occurrences')} occurrence(s) "
                f"over {group.get('window_seconds')}s "
                f"(rate {group.get('occurrences_per_second')}/s), "
                f"workers {group.get('worker_examples')}"
            )

    timeline = _timeline_lines(events)
    if timeline:
        lines.extend(["", "=== TIMELINE (decisions, phases and failures) ==="])
        lines.extend(timeline)

    lines.extend(
        [
            "",
            "=== BUNDLE CONTENTS ===",
            f"  {SUMMARY_FILENAME}       this file",
            f"  {EVENTS_FILENAME}       structured flight-recorder events (JSON Lines)",
            f"  {ENVIRONMENT_FILENAME}   machine/build/plug-in/model/library snapshot",
            f"  {POSTMORTEM_FILENAME}    failure/stall/process/worker/state snapshot",
            f"  {REPEATS_FILENAME}       aggregated repeated failures",
            f"  {REPRODUCTION_FILENAME}  sanitized reproduction descriptor",
            "",
            "=== RETENTION ===",
            json.dumps(recorder().retention_policy(), indent=2, sort_keys=True),
            "",
            f"Keep Ticket ID {ticket_id} if you contact PatchLab support.",
            "",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupportBundle:
    directory: Path
    archive: Path | None
    files: tuple[Path, ...]
    reproduction: dict[str, Any]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "archive": str(self.archive) if self.archive else None,
            "files": [path.name for path in self.files],
            "failure_fingerprint": self.fingerprint,
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        }


def _read_persisted(name: str) -> dict[str, Any] | None:
    root = recorder().root
    if root is None:
        return None
    path = root / name
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except (OSError, ValueError):
        return None


def _parse_ts(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def merged_events(current, history_seconds: float) -> list[dict[str, Any]]:
    """The recent timeline across EVERY PatchLab process, oldest first.

    The recorder's in-memory ring only holds this process's events, but a bundle
    is built in the GUI process while the events that explain a failure -- phase
    history, renderer decisions, worker heartbeats, worker failures -- are
    written by short-lived worker processes into the shared on-disk
    ``events.jsonl`` files. A bundle built from the ring alone therefore
    contained none of them. This merges the disk files (all processes) with the
    ring (anything not yet flushed), de-duplicated and time-windowed.
    """

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(0.0, history_seconds))
    seen: dict[tuple, dict[str, Any]] = {}

    def add(row: dict[str, Any]) -> None:
        stamp = _parse_ts(row.get("ts"))
        if stamp is None or stamp < cutoff:
            return
        key = (row.get("ts"), row.get("pid"), row.get("event_type"), row.get("message"))
        seen.setdefault(key, row)

    for path in reversed(current.rotated_event_files()):  # oldest file first
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a torn final line from a killed worker
                if isinstance(row, dict):
                    add(row)
        except OSError:
            continue
    for row in current.recent_events():
        add(row)
    return sorted(seen.values(), key=lambda row: (str(row.get("ts")), int(row.get("pid") or 0)))


def repeat_groups_from(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate repeated failures across processes from the merged timeline.

    Each worker aggregates only in its own memory, so the recorder's live
    summary cannot see 200 identical failures spread over 200 processes. Every
    occurrence is still one event line on disk, so the groups are rebuilt here:
    the first complete event, the count, the window and example workers/PIDs.
    """

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for event in events:
        fingerprint = str(event.get("fingerprint") or "")
        if fingerprint:
            groups.setdefault((str(event.get("event_type")), fingerprint), []).append(event)
    result: list[dict[str, Any]] = []
    for (event_type, fingerprint), items in groups.items():
        if len(items) < 2:
            continue
        first, last = items[0], items[-1]
        started, ended = _parse_ts(first.get("ts")), _parse_ts(last.get("ts"))
        window = (ended - started).total_seconds() if started and ended else 0.0
        result.append(
            {
                "fingerprint": fingerprint,
                "event_type": event_type,
                "total_occurrences": len(items),
                "first_timestamp": first.get("ts"),
                "latest_timestamp": last.get("ts"),
                "window_seconds": round(window, 3),
                "occurrences_per_second": round(len(items) / window, 3) if window > 0.05 else None,
                "worker_examples": sorted({str(i.get("worker_id")) for i in items if i.get("worker_id")})[:10],
                "pid_examples": sorted({int(i.get("pid")) for i in items if i.get("pid")})[:10],
                "first_full_event": first,
            }
        )
    result.sort(key=lambda item: -int(item["total_occurrences"]))
    return result


def _events_blob(events: Sequence[Mapping[str, Any]]) -> str:
    """Serialize events newest-last, trimming the oldest if over the cap."""

    lines = [
        json.dumps(sanitize(event), separators=(",", ":"), default=str)
        for event in events
    ]
    blob = "\n".join(lines)
    while len(blob.encode("utf-8")) > MAX_BUNDLE_EVENT_BYTES and len(lines) > 100:
        # Drop the oldest quarter; the lead-up nearest the failure matters most.
        lines = lines[len(lines) // 4 :]
        blob = "\n".join(lines)
    return blob + "\n" if blob else ""


def archive_bundle_directory(directory: Path) -> Path | None:
    """Zip an already-written bundle directory next to it (no re-collection).

    Used both when the bundle is first created and when an upload needs the
    archive of a bundle that was saved earlier.
    """

    directory = Path(directory)
    archive_path = directory.with_suffix(".zip")
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(directory.iterdir()):
                if path.is_file() and not path.name.startswith("."):
                    handle.write(path, arcname=f"{directory.name}/{path.name}")
        temporary.replace(archive_path)
        return archive_path
    except (OSError, zipfile.BadZipFile):
        temporary.unlink(missing_ok=True)
        return None


def create_support_bundle(
    *,
    ticket_id: str,
    comments: str = "",
    directory: Path | None = None,
    operation: str = "",
    operation_id: str = "",
    environment: Mapping[str, Any] | None = None,
    postmortem: Mapping[str, Any] | None = None,
    renderer_selection: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    feature_flags: Mapping[str, Any] | None = None,
    history_seconds: float = BUNDLE_HISTORY_SECONDS,
    archive: bool = True,
    ticket_path: Path | None = None,
) -> SupportBundle:
    """Freeze the recent diagnostic history into a bundle directory.

    This is the *local* save.  It must always succeed before any upload is
    attempted, and must work whether or not the private support service is
    reachable.
    """

    current = recorder()
    current.flush(timeout=1.5)

    # Fall back to whatever the last operation persisted, so a bug report filed
    # minutes after a failure still carries that failure's snapshots.
    environment = environment if environment is not None else _read_persisted(
        ENVIRONMENT_FILENAME
    )
    postmortem = postmortem if postmortem is not None else _read_persisted(
        POSTMORTEM_FILENAME
    )
    if not environment:
        # No operation has run in this session, or its snapshot rotated away.
        # Capture one now rather than shipping a bundle with no machine
        # fingerprint -- the plug-in inventory and build identity are the most
        # valuable part of a report about somebody else's Mac, and they are the
        # cheapest to obtain (one pass of stat calls plus cached hashes).
        try:
            from core.diagnostic_env import capture_environment
            from core.local_library import default_local_paths

            environment = capture_environment(
                operation=operation or "bug-report",
                operation_id=operation_id,
                db_path=default_local_paths()["db"],
            ).as_dict()
        except Exception as exc:
            current.record(
                "support-bundle",
                "environment_capture_failed",
                f"could not capture an environment snapshot: {type(exc).__name__}: {exc}",
                severity="warning",
                exception=exc,
            )
            environment = {}

    events = merged_events(current, history_seconds)
    if not events:
        events = current.recent_events()
    live_repeats = current.repeat_summary()
    disk_repeats = repeat_groups_from(events)
    # Prefer whichever view saw more occurrences of each failure.
    combined = {item["fingerprint"]: item for item in live_repeats}
    for item in disk_repeats:
        known = combined.get(item["fingerprint"])
        if known is None or item["total_occurrences"] > int(known.get("total_occurrences", 0)):
            combined[item["fingerprint"]] = item
    repeats = sorted(combined.values(), key=lambda item: -int(item.get("total_occurrences", 0)))

    reproduction = build_reproduction_descriptor(
        operation=operation or str((postmortem or {}).get("operation", "")),
        operation_id=operation_id,
        environment=environment,
        postmortem=postmortem,
        renderer_selection=renderer_selection,
        settings=settings,
        feature_flags=feature_flags,
    )
    fingerprint = str(reproduction.get("failure", {}).get("fingerprint") or "")

    if directory is None:
        from core.bug_report import reports_root

        directory = reports_root() / f"PatchLab Bug Report {ticket_id}{BUNDLE_DIRNAME_SUFFIX}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    def _write(name: str, text: str) -> None:
        path = directory / name
        temporary = path.with_name(f".{name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
        written.append(path)

    _write(
        SUMMARY_FILENAME,
        build_summary(
            ticket_id=ticket_id,
            comments=comments,
            environment=environment,
            postmortem=postmortem,
            events=events,
            repeats=repeats,
        ),
    )
    _write(EVENTS_FILENAME, _events_blob(events))
    _write(
        ENVIRONMENT_FILENAME,
        json.dumps(sanitize(environment or {}), indent=2, sort_keys=True, default=str),
    )
    _write(
        POSTMORTEM_FILENAME,
        json.dumps(sanitize(postmortem or {}), indent=2, sort_keys=True, default=str),
    )
    _write(
        REPEATS_FILENAME,
        json.dumps(sanitize(repeats), indent=2, sort_keys=True, default=str),
    )
    _write(
        REPRODUCTION_FILENAME,
        json.dumps(reproduction, indent=2, sort_keys=True, default=str),
    )

    if ticket_path is not None:
        # The readable ticket (the user's description and the visible app log)
        # travels with the bundle so support gets one complete file.
        try:
            _write(TICKET_FILENAME, redact_text(Path(ticket_path).read_text(encoding="utf-8")))
        except OSError:
            pass

    archive_path: Path | None = None
    if archive:
        archive_path = archive_bundle_directory(directory)

    current.record(
        "support-bundle",
        "bundle_created",
        f"support bundle saved locally for ticket {ticket_id}",
        decision_reason=(
            "the local bundle is always written before any upload is attempted, so a "
            "transport failure can never destroy the saved report"
        ),
        ticket_id=ticket_id,
        directory=str(directory),
        archive=str(archive_path) if archive_path else None,
        event_count=len(events),
        repeat_groups=len(repeats),
        failure_fingerprint=fingerprint or None,
    )
    return SupportBundle(
        directory=directory,
        archive=archive_path,
        files=tuple(written),
        reproduction=reproduction,
        fingerprint=fingerprint,
    )


__all__ = [
    "ENVIRONMENT_FILENAME",
    "EVENTS_FILENAME",
    "POSTMORTEM_FILENAME",
    "REPRODUCTION_FILENAME",
    "REPEATS_FILENAME",
    "SUMMARY_FILENAME",
    "SupportBundle",
    "TICKET_FILENAME",
    "archive_bundle_directory",
    "build_reproduction_descriptor",
    "build_summary",
    "create_support_bundle",
]
