"""Safe cleanup helpers for interrupted Match-a-Sound scratch state."""

from __future__ import annotations

import re
import shutil
import tempfile
import time
from pathlib import Path


STALE_MATCH_PATTERN = re.compile(
    r"patchlab-match-(?:session-)?[A-Za-z0-9_-]+$"
)


def cleanup_stale_match_scratch(*, minimum_age_s: float = 3600.0) -> list[Path]:
    """Remove only Patch Lab match scratch directories old enough to be inactive."""

    root = Path(tempfile.gettempdir()).resolve()
    now = time.time()
    removed: list[Path] = []
    for path in root.glob("patchlab-match-*"):
        resolved = path.resolve()
        if (
            resolved.parent != root
            or not resolved.is_dir()
            or not STALE_MATCH_PATTERN.fullmatch(resolved.name)
        ):
            continue
        try:
            age = now - resolved.stat().st_mtime
        except FileNotFoundError:
            continue
        if age < minimum_age_s:
            continue
        shutil.rmtree(resolved)
        removed.append(resolved)
    return removed
