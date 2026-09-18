"""Consent-driven diagnostic report requests for the private PatchLab relay."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.platform_env import ENV


MAX_COMMENT_CHARACTERS = 20_000
MAX_LOG_CHARACTERS = 500_000


@dataclass(frozen=True, slots=True)
class BugReportRequest:
    ticket_id: str
    comments: str
    logs: str


def pending_reports_root() -> Path:
    return ENV.app_data_dir / "diagnostics" / "pending-bug-reports"


def create_request(*, comments: str, logs: str) -> Path:
    """Persist one user-approved report until its background upload finishes."""

    cleaned_comments = comments.strip()
    if not cleaned_comments:
        raise ValueError("A description is required before sending a bug report.")
    request = BugReportRequest(
        ticket_id=uuid.uuid4().hex,
        comments=cleaned_comments[:MAX_COMMENT_CHARACTERS],
        logs=logs[-MAX_LOG_CHARACTERS:],
    )
    root = pending_reports_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{request.ticket_id}.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "ticket_id": request.ticket_id,
                "comments": request.comments,
                "logs": request.logs,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_request(path: Path) -> BugReportRequest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ticket_id = str(raw.get("ticket_id", ""))
    comments = str(raw.get("comments", "")).strip()
    logs = str(raw.get("logs", ""))
    if len(ticket_id) != 32 or any(char not in "0123456789abcdef" for char in ticket_id):
        raise ValueError("Bug report ticket identifier is invalid.")
    if not comments:
        raise ValueError("A description is required before sending a bug report.")
    return BugReportRequest(
        ticket_id=ticket_id,
        comments=comments[:MAX_COMMENT_CHARACTERS],
        logs=logs[-MAX_LOG_CHARACTERS:],
    )
