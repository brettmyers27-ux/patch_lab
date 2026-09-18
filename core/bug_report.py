"""Durable, user-readable PatchLab support tickets."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.platform_env import ENV


MAX_COMMENT_CHARACTERS = 20_000
MAX_LOG_CHARACTERS = 500_000
COMMENTS_MARKER = "=== USER COMMENTS ==="
DIAGNOSTICS_MARKER = "=== PATCHLAB DIAGNOSTICS ==="


@dataclass(frozen=True, slots=True)
class BugReportRequest:
    ticket_id: str
    comments: str
    logs: str
    report_path: Path


def reports_root() -> Path:
    """Return the visible Desktop ticket folder, with a safe fallback."""

    desktop = Path.home() / "Desktop" / "PatchLab Bug Reports"
    try:
        desktop.mkdir(parents=True, exist_ok=True)
        if os.access(desktop, os.R_OK | os.W_OK):
            return desktop
    except OSError:
        pass
    fallback = ENV.app_data_dir / "diagnostics" / "bug-reports"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _report_contents(*, ticket_id: str, comments: str, logs: str) -> str:
    return (
        "PATCHLAB BUG REPORT\n"
        f"Ticket ID: {ticket_id}\n"
        f"Created (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        "Keep this Ticket ID if you contact PatchLab support about this report.\n\n"
        f"{COMMENTS_MARKER}\n"
        f"{comments}\n\n"
        f"{DIAGNOSTICS_MARKER}\n"
        f"{logs.rstrip()}\n"
    )


def create_request(*, comments: str, logs: str) -> Path:
    """Create the one durable local ticket before attempting any network call."""

    cleaned_comments = comments.strip()
    if not cleaned_comments:
        raise ValueError("A description is required before sending a bug report.")
    ticket_id = uuid.uuid4().hex
    root = reports_root()
    path = root / f"PatchLab Bug Report {ticket_id}.txt"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        _report_contents(
            ticket_id=ticket_id,
            comments=cleaned_comments[:MAX_COMMENT_CHARACTERS],
            logs=logs[-MAX_LOG_CHARACTERS:],
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_request(path: Path) -> BugReportRequest:
    """Read the durable ticket format without creating a hidden second copy."""

    report_path = Path(path).expanduser().resolve()
    raw = report_path.read_text(encoding="utf-8")
    if COMMENTS_MARKER not in raw or DIAGNOSTICS_MARKER not in raw:
        raise ValueError("Bug report file is incomplete or has an unknown format.")
    header, rest = raw.split(COMMENTS_MARKER, 1)
    comments_text, logs_text = rest.split(DIAGNOSTICS_MARKER, 1)
    ticket_line = next(
        (line for line in header.splitlines() if line.startswith("Ticket ID: ")),
        "",
    )
    ticket_id = ticket_line.removeprefix("Ticket ID: ").strip()
    comments = comments_text.strip()
    logs = logs_text.strip()
    if len(ticket_id) != 32 or any(char not in "0123456789abcdef" for char in ticket_id):
        raise ValueError("Bug report ticket identifier is invalid.")
    if not comments:
        raise ValueError("A description is required before sending a bug report.")
    return BugReportRequest(
        ticket_id=ticket_id,
        comments=comments[:MAX_COMMENT_CHARACTERS],
        logs=logs[-MAX_LOG_CHARACTERS:],
        report_path=report_path,
    )
