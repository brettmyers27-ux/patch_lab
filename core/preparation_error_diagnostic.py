"""Read-only follow-up collector for semantic preparation-error evidence."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from core.preparation_diagnostic import (
    _columns,
    _readonly_connection,
    discover_database,
    output_path,
    write_output,
)


OUTPUT_FILENAME = "patchlab-1.6.6-preparation-error-details.json"
_PATH = re.compile(r"(?:[A-Za-z]:\\|/)(?:[^\s/:]+[/\\]){2,}[^\s:,)]*")
_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_UUID = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)
_SECRET = re.compile(r"(?i)\b(token|password|secret|key)\s*[=:]\s*\S+")
_NOTES = (24, 36, 48, 60, 72, 84, 96)


def sanitize_error(value: object) -> str:
    """Keep a bounded technical message while removing private identifiers."""

    text = str(value or "").replace("\\", "/")
    text = _PATH.sub("<path>", text)
    text = _EMAIL.sub("<email>", text)
    text = _UUID.sub("<uuid>", text)
    text = _SECRET.sub(r"\1=<redacted>", text)
    return re.sub(r"\s+", " ", text).strip()[:400]


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def collect(database: Path) -> dict[str, object]:
    """Aggregate retained errors without emitting identifiers, paths, or names."""

    with _readonly_connection(database) as connection:
        presets = _columns(connection, "presets")
        jobs = _columns(connection, "preparation_jobs")
        if not {"id", "status", "synth", "error"}.issubset(presets):
            raise RuntimeError("database lacks the 1.6.6 preparation-error columns")
        join = "LEFT JOIN preparation_jobs j ON j.preset_id=p.id" if jobs else ""
        job_fields = "j.state AS job_state,j.last_error AS job_error" if jobs else "NULL AS job_state,NULL AS job_error"
        rows = connection.execute(
            f"""SELECT p.status,p.synth,p.error AS preset_error,{job_fields}
                FROM presets p {join}
                WHERE p.status IN ('failed_load','failed_silent')"""
        ).fetchall()

    groups: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    source_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    silent_notes: Counter[int] = Counter()
    for row in rows:
        status = str(row["status"])
        job_error = str(row["job_error"] or "")
        preset_error = str(row["preset_error"] or "")
        raw, source = (job_error, "preparation_jobs.last_error") if job_error else (preset_error, "presets.error")
        clean = sanitize_error(raw) or "no retained error text"
        stage = str(row["job_state"] or "not_stored")
        source_counts[source] += 1
        stage_counts[stage] += 1
        groups[(str(row["synth"]), status, stage, source, clean)] += 1
        if status == "failed_silent":
            for note in _NOTES:
                if re.search(rf"\b{note}\b", clean):
                    silent_notes[note] += 1

    failure_groups = [
        {
            "synth": synth,
            "status": status,
            "preparation_stage": stage,
            "error_source": source,
            "semantic_error": message,
            "semantic_fingerprint": _fingerprint(message),
            "count": count,
        }
        for (synth, status, stage, source, message), count in groups.items()
    ]
    failure_groups.sort(key=lambda item: (-int(item["count"]), str(item["semantic_fingerprint"])))
    return {
        "collector_version": "1.0.0",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": {"opened_read_only": True},
        "totals": {"failure_rows": len(rows), "semantic_groups": len(failure_groups)},
        "relationship": {
            "one_job_row_per_preset_schema": "preparation_jobs.preset_id is PRIMARY KEY",
            "rows_with_job_state": sum(count for stage, count in stage_counts.items() if stage != "not_stored"),
            "rows_without_job_state": stage_counts["not_stored"],
            "error_sources": dict(sorted(source_counts.items())),
        },
        "failure_groups": failure_groups,
        "silent_note_frequency": {str(note): silent_notes[note] for note in _NOTES},
        "privacy": {
            "network_calls": False,
            "database_writes": False,
            "includes_preset_names": False,
            "includes_file_paths": False,
            "includes_preset_ids": False,
            "includes_audio": False,
            "includes_credentials": False,
            "error_text": "semantic messages only; paths, email, UUIDs, and secrets redacted",
        },
    }


def run_collector() -> tuple[Path | None, str]:
    discovery = discover_database()
    if discovery.status != "found" or discovery.database is None:
        return None, "No unambiguous readable PatchLab database was found. PatchLab data was not changed."
    try:
        return write_output(output_path().with_name(OUTPUT_FILENAME), collect(discovery.database)), ""
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        return None, "The diagnostic could not read PatchLab data safely: " + str(exc)
