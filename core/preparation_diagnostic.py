"""Portable, privacy-preserving inspection of PatchLab preparation failures.

This module deliberately uses only the Python standard library.  It is shared
by the macOS diagnostic app and the future Windows frozen executable; platform
code is limited to finding PatchLab's documented per-user data directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath, PureWindowsPath
from typing import Iterable, Mapping
from urllib.parse import quote


COLLECTOR_VERSION = "1.0.0"
OUTPUT_FILENAME = "patchlab-1.6.6-preparation-diagnostics.json"
FAILURE_STATUSES = ("failed_load", "failed_silent", "incomplete", "stale")


@dataclass(frozen=True)
class DiscoveryResult:
    status: str
    database: Path | None
    candidate_count: int
    valid_count: int
    reason: str = ""


def platform_data_candidates(
    *,
    system_name: str | None = None,
    home: PurePath | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[PurePath, ...]:
    """Return conventional PatchLab database locations without scanning disks."""
    system_name = system_name or platform.system()
    environ = environ or os.environ
    if system_name == "Darwin":
        user_home = home or Path.home()
        return (user_home / "Library" / "Application Support" / "Patch Lab" / "library.db",)
    if system_name == "Windows":
        user_home = home or PureWindowsPath(environ.get("USERPROFILE", str(Path.home())))
        local = PureWindowsPath(environ.get("LOCALAPPDATA", str(user_home / "AppData" / "Local")))
        # ``Patch Lab`` is the current app convention.  The no-space form is
        # retained only to detect an old installation safely; ambiguity stops
        # collection rather than guessing which database belongs to PatchLab.
        return (
            local / "Patch Lab" / "library.db",
            local / "PatchLab" / "library.db",
        )
    return ()


def _readonly_connection(database: Path) -> sqlite3.Connection:
    uri = "file:" + quote(str(database.resolve())) + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _valid_patchlab_database(candidate: Path) -> bool:
    try:
        with _readonly_connection(candidate) as connection:
            return {"id", "synth", "status"}.issubset(_columns(connection, "presets"))
    except (OSError, sqlite3.Error):
        return False


def discover_database() -> DiscoveryResult:
    candidates = [Path(item) for item in platform_data_candidates()]
    present = [item for item in candidates if item.is_file()]
    valid = [item for item in present if _valid_patchlab_database(item)]
    if not valid:
        return DiscoveryResult("not_found", None, len(present), 0, "no readable PatchLab database found")
    if len(valid) != 1:
        return DiscoveryResult("ambiguous", None, len(present), len(valid), "multiple valid PatchLab databases found")
    return DiscoveryResult("found", valid[0], len(present), 1)


def _error_family(value: object) -> str:
    """Classify errors without returning their message, paths, or identifiers."""
    message = str(value or "").casefold()
    if not message:
        return "no_stored_error"
    if "silent rendered midi notes" in message:
        notes = sorted(set(re.findall(r"\b(?:24|36|48|60|72|84|96)\b", message)))
        return "silent_notes:" + (",".join(notes) if notes else "unspecified")
    known = (
        ("seven_note_validation", "rendering did not produce seven valid note files"),
        ("render_state_missing", "filenotfounderror"),
        ("render_state_rejected", "serum 2 render state was rejected"),
        ("state_application", "parameters changed from init"),
        ("state_application", "load_vst3_preset"),
        ("state_decode", "xferjson"),
        ("state_decode", "cbor"),
        ("state_decode", "zstd"),
        ("worker_pipe", "brokenpipeerror"),
        ("worker_timeout", "timeout"),
        ("worker_exit", "exit code"),
    )
    for label, marker in known:
        if marker in message:
            return label
    exception = re.search(r"\b([a-z_]+(?:error|exception))\b", message)
    if exception:
        return "exception:" + exception.group(1)
    return "unknown:" + hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]


def _metric_range(values: Iterable[object]) -> dict[str, float | int] | None:
    numbers = [float(item) for item in values if item is not None and math.isfinite(float(item))]
    if not numbers:
        return None
    return {"count": len(numbers), "minimum": min(numbers), "maximum": max(numbers)}


def _attempt_range(values: Iterable[object]) -> dict[str, int] | None:
    numbers = [int(item) for item in values if item is not None]
    return {"minimum": min(numbers), "maximum": max(numbers)} if numbers else None


def _correlation_id(database: Path, preset_id: int) -> str:
    """Return an opaque database-local identifier without exposing the row ID."""
    salt = hashlib.sha256(str(database.resolve()).encode("utf-8")).digest()
    return "local-" + hashlib.sha256(salt + str(preset_id).encode("ascii")).hexdigest()[:12]


def collect(database: Path) -> dict[str, object]:
    """Read a PatchLab database and return aggregate-only diagnostic evidence."""
    unavailable: list[str] = [
        "PatchLab version: not stored by the 1.6.6 library database",
        "plugin format: not stored by the 1.6.6 library database",
        "earliest per-preset error history: not stored by the 1.6.6 library database",
        "worker exit status: not stored unless a later schema adds it",
        "operation correlation: not stored by the 1.6.6 library database",
    ]
    with _readonly_connection(database) as connection:
        preset_columns = _columns(connection, "presets")
        required = {"id", "synth", "status"}
        if not required.issubset(preset_columns):
            raise RuntimeError("database does not have the required PatchLab preset schema")
        job_columns = _columns(connection, "preparation_jobs")
        render_columns = _columns(connection, "renders")
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        error_column = "p.error" if "error" in preset_columns else "NULL"
        if "error" not in preset_columns:
            unavailable.append("stored per-preset error: unavailable in this schema")
        job_state = "j.state" if "state" in job_columns else "NULL"
        job_error = "j.last_error" if "last_error" in job_columns else "NULL"
        attempt_count = "j.attempt_count" if "attempt_count" in job_columns else "NULL"
        job_join = "LEFT JOIN preparation_jobs AS j ON j.preset_id=p.id" if job_columns else ""
        if not job_columns:
            unavailable.append("preparation stage and retry count: unavailable in this schema")
        placeholders = ",".join("?" for _ in FAILURE_STATUSES)
        rows = connection.execute(
            f"""
            SELECT p.id,p.synth,p.status,{error_column} AS preset_error,
                   {job_state} AS job_state,{job_error} AS job_error,
                   {attempt_count} AS attempt_count
            FROM presets AS p
            {job_join}
            WHERE p.status IN ({placeholders})
            """,
            FAILURE_STATUSES,
        ).fetchall()
        failure_ids = [int(row["id"]) for row in rows]
        render_data: dict[int, list[sqlite3.Row]] = defaultdict(list)
        required_render = {"preset_id", "midi_note", "peak_dbfs", "rms_dbfs"}
        if failure_ids and required_render.issubset(render_columns):
            placeholders = ",".join("?" for _ in failure_ids)
            for row in connection.execute(
                f"SELECT preset_id,midi_note,peak_dbfs,rms_dbfs FROM renders WHERE preset_id IN ({placeholders})",
                failure_ids,
            ):
                render_data[int(row["preset_id"])].append(row)
        elif failure_ids:
            unavailable.append("note peak/RMS and MIDI metrics: unavailable in this schema")

    grouped: dict[tuple[str, str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        error = _error_family(row["preset_error"] or row["job_error"])
        grouped[(str(row["synth"] or "unknown"), str(row["status"]), str(row["job_state"] or "not_stored"), error)].append(row)
    groups: list[dict[str, object]] = []
    for (synth, status, stage, error), members in sorted(grouped.items()):
        metrics = [item for row in members for item in render_data.get(int(row["id"]), [])]
        notes = sorted({int(item["midi_note"]) for item in metrics if item["midi_note"] is not None})
        groups.append(
            {
                "synth": synth,
                "status": status,
                "preparation_stage": stage,
                "sanitized_error_family": error,
                "count": len(members),
                "attempt_range": _attempt_range(row["attempt_count"] for row in members),
                "recorded_note_count_range": _attempt_range(len(render_data.get(int(row["id"]), [])) for row in members),
                "recorded_midi_notes": notes,
                "peak_dbfs": _metric_range(item["peak_dbfs"] for item in metrics),
                "rms_dbfs": _metric_range(item["rms_dbfs"] for item in metrics),
                "representative_correlation_ids": [
                    _correlation_id(database, int(row["id"])) for row in members[:3]
                ],
            }
        )
    return {
        "collector_version": COLLECTOR_VERSION,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "os_version": platform.version(),
        "patchlab_version": "not_stored_by_1.6.6_library_database",
        "database": {"schema_user_version": user_version, "opened_read_only": True},
        "totals": {"failure_rows": len(rows), "failure_groups": len(groups)},
        "failure_groups": groups,
        "unavailable_fields": unavailable,
        "privacy": {
            "network_calls": False,
            "database_writes": False,
            "includes_preset_names": False,
            "includes_file_paths": False,
            "includes_preset_bytes": False,
            "includes_audio": False,
            "includes_credentials": False,
            "includes_raw_error_messages": False,
        },
    }


def output_path() -> Path:
    candidates = [Path.home() / "Desktop", Path(sys_executable_parent()), Path(tempfile.gettempdir())]
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".patchlab-diagnostic-write-probe"
            probe.write_bytes(b"")
            probe.unlink()
            return directory / OUTPUT_FILENAME
        except OSError:
            continue
    raise RuntimeError("no writable location is available for the diagnostic output")


def sys_executable_parent() -> Path:
    import sys

    return Path(sys.executable).resolve().parent


def write_output(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".patchlab-diagnostic-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def run_collector() -> tuple[Path | None, str]:
    discovery = discover_database()
    if discovery.status != "found" or discovery.database is None:
        return None, "No unambiguous readable PatchLab database was found. PatchLab data was not changed."
    try:
        return write_output(output_path(), collect(discovery.database)), ""
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        return None, "The diagnostic could not read PatchLab data safely: " + str(exc)
