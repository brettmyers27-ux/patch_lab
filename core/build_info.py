"""Read and report the source identity embedded in a packaged build."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BuildInfo:
    source_commit: str
    built_at_utc: str
    source_dirty: bool
    frozen: bool

    @property
    def short_commit(self) -> str:
        return self.source_commit[:12] if self.source_commit else "unknown"

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "source_commit": self.source_commit,
            "built_at_utc": self.built_at_utc,
            "source_dirty": self.source_dirty,
            "frozen": self.frozen,
        }


def _runtime_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(str(frozen_root)).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _development_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def current_build_info() -> BuildInfo:
    frozen = bool(getattr(sys, "frozen", False))
    root = _runtime_root()
    bundled = root / "patchlab-build-info.json"
    if bundled.is_file():
        raw = json.loads(bundled.read_text(encoding="utf-8"))
        return BuildInfo(
            source_commit=str(raw.get("source_commit", "")),
            built_at_utc=str(raw.get("built_at_utc", "")),
            source_dirty=bool(raw.get("source_dirty", False)),
            frozen=frozen,
        )
    return BuildInfo(
        source_commit=_development_commit(root),
        built_at_utc=datetime.now(timezone.utc).isoformat(),
        source_dirty=False,
        frozen=frozen,
    )


def assert_packaged_commit(expected_commit: str) -> BuildInfo:
    info = current_build_info()
    if not info.frozen:
        raise RuntimeError("Build identity verification must run in the packaged app")
    if info.source_commit != expected_commit:
        raise RuntimeError(
            "Stale PatchLab build: packaged source commit "
            f"{info.source_commit or 'unknown'} does not match expected "
            f"{expected_commit}."
        )
    if info.source_dirty:
        raise RuntimeError(
            f"PatchLab build {info.short_commit} was made from dirty source"
        )
    return info
