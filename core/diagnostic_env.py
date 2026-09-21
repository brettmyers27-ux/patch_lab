"""Pre-operation environment fingerprint.

For a bug that only happens on one person's Mac, the environment *is* the bug
report.  The two tickets this module was written for differ from a healthy
machine in exactly one respect -- Serum 1 is not installed in any format -- and
nothing in the old text diagnostics made that actionable, because it printed a
flat `serum1/VST2=missing` list without saying which renderer any operation
actually needed.

Captured once per significant operation, never continuously.  Expensive facts
(plug-in and checkpoint hashes) go through the recorder's size+mtime cache, so
a repeat Match re-reads nothing from disk.
"""

from __future__ import annotations

import os
import platform
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.diagnostics import (
    DIAGNOSTIC_SCHEMA_VERSION,
    recorder,
    sanitize,
)
from core.operation_state import compute_snapshot, resource_snapshot, runtime_identity
from core.platform_env import ENV, PlatformEnv
from core.renderer_selection import renderer_inventory


SUBSYSTEM = "environment"

_PROCESS_START = datetime.now(timezone.utc)


def _application_section() -> dict[str, Any]:
    from core.build_info import current_build_info

    try:
        from app.__version__ import __version__ as version
    except Exception:
        version = ""
    build = current_build_info()
    return {
        "patchlab_version": version,
        "source_commit": build.source_commit,
        "source_commit_short": build.short_commit,
        "source_dirty": build.source_dirty,
        "built_at_utc": build.built_at_utc,
        "frozen": build.frozen,
        "mode": "frozen" if build.frozen else "source",
        "distribution_mode": os.environ.get("PATCHLAB_DISTRIBUTION_MODE", "0") == "1",
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "session_id": recorder().session_id,
        "executable": sys.executable,
        "argv0": sys.argv[0] if sys.argv else "",
        "resource_root": str(getattr(sys, "_MEIPASS", "")) or str(
            Path(__file__).resolve().parents[1]
        ),
        "process_started_utc": _PROCESS_START.isoformat(),
        "uptime_seconds": round(
            (datetime.now(timezone.utc) - _PROCESS_START).total_seconds(), 3
        ),
    }


def _macos_build() -> str:
    if sys.platform != "darwin":
        return ""
    try:
        import subprocess

        return subprocess.check_output(
            ["sw_vers", "-buildVersion"], text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except Exception:
        return ""


def _physical_memory_bytes() -> int | None:
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and size > 0:
                return int(pages) * int(size)
    except (OSError, ValueError):
        pass
    return None


def _system_section(env: PlatformEnv) -> dict[str, Any]:
    section: dict[str, Any] = {
        "system": env.system_name,
        "branch": env.branch,
        "architecture": env.machine,
        "os_release": platform.release(),
        "os_version": platform.version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": _physical_memory_bytes(),
        "compute_backend": env.compute_backend,
        "compute_warning": env.compute_warning,
    }
    if env.branch == "macos":
        section["macos_version"] = platform.mac_ver()[0]
        section["macos_build"] = _macos_build()
    section.update(runtime_identity())
    section["compute"] = compute_snapshot()
    return section


def _dependency_versions() -> dict[str, str]:
    """Versions of the dependencies that actually affect these code paths.

    Deliberately not a full software inventory: only PatchLab's own render,
    audio and model stack.
    """

    from importlib import metadata

    names = (
        "torch",
        "torchaudio",
        "numpy",
        "librosa",
        "soundfile",
        "dawdreamer",
        "pedalboard",
        "PySide6",
        "transformers",
        "laion-clap",
        "cma",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = ""
    return versions


def _database_section(db_path: Path | None) -> dict[str, Any]:
    if db_path is None:
        return {"path": None, "available": False, "reason": "no database path supplied"}
    path = Path(db_path)
    section: dict[str, Any] = {"path": str(path), "available": path.is_file()}
    if not path.is_file():
        section["reason"] = "database file does not exist yet"
        return section
    try:
        section["size_bytes"] = path.stat().st_size
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0) as connection:
            section["schema_version"] = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            for label, statement in (
                ("presets", "SELECT COUNT(*) FROM presets"),
                ("renders", "SELECT COUNT(*) FROM renders"),
            ):
                try:
                    section[label] = int(connection.execute(statement).fetchone()[0])
                except sqlite3.Error:
                    section[label] = None
            try:
                rows = connection.execute(
                    "SELECT synth, status, COUNT(*) FROM presets GROUP BY synth, status"
                ).fetchall()
                section["presets_by_synth_status"] = {
                    f"{row[0]}/{row[1]}": int(row[2]) for row in rows
                }
            except sqlite3.Error:
                section["presets_by_synth_status"] = None
            try:
                section["fingerprinted"] = int(
                    connection.execute(
                        "SELECT COUNT(DISTINCT preset_id) FROM fingerprints "
                        "WHERE midi_note=0"
                    ).fetchone()[0]
                )
            except sqlite3.Error:
                section["fingerprinted"] = None
            try:
                section["tables"] = sorted(
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                )
            except sqlite3.Error:
                section["tables"] = None
    except (OSError, sqlite3.Error) as exc:
        section["available"] = False
        section["reason"] = f"{type(exc).__name__}: {exc}"
    return section


def _model_section() -> dict[str, Any]:
    section: dict[str, Any] = {}
    try:
        from core.model_assets import resolve_model_assets

        assets = resolve_model_assets()
        checkpoint = Path(getattr(assets, "checkpoint", ""))
        section["checkpoint_path"] = str(checkpoint)
        section["checkpoint_exists"] = checkpoint.is_file()
        if checkpoint.is_file():
            stat = checkpoint.stat()
            section["checkpoint_size_bytes"] = stat.st_size
            section["checkpoint_mtime_utc"] = datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat()
            # Cached against size+mtime: hashed once per install, not per Match.
            section["checkpoint_sha1"] = recorder().cached_file_digest(checkpoint)
        section["device"] = ENV.compute_backend
        for name in ("cache_dir", "tokenizer_dir", "model_id"):
            value = getattr(assets, name, None)
            if value is not None:
                section[name] = str(value)
    except Exception as exc:
        section["error"] = f"{type(exc).__name__}: {exc}"
    return section


def _storage_section() -> dict[str, Any]:
    section: dict[str, Any] = {}
    try:
        from core.storage import storage_status

        status = storage_status()
        section = {
            "root": str(status.root),
            "available": bool(status.available),
            "configured_external": bool(status.configured_external),
            "reason": status.reason or "",
        }
        try:
            usage = shutil.disk_usage(status.root)
            section["total_bytes"] = usage.total
            section["free_bytes"] = usage.free
        except OSError as exc:
            section["disk_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        section["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from core.storage import load_storage_preferences

        section["compact_mode"] = bool(load_storage_preferences().compact_mode)
    except Exception:
        pass
    return section


def _path_access(paths: Mapping[str, Path | str | None]) -> dict[str, Any]:
    """Permission and existence facts for the paths an operation depends on."""

    result: dict[str, Any] = {}
    for label, value in paths.items():
        if value in (None, ""):
            result[label] = {"path": None, "exists": False}
            continue
        path = Path(str(value))
        try:
            exists = path.exists()
            result[label] = {
                "path": str(path),
                "exists": exists,
                "is_dir": path.is_dir() if exists else None,
                "readable": os.access(path, os.R_OK) if exists else False,
                "writable": os.access(path, os.W_OK) if exists else False,
            }
        except OSError as exc:
            result[label] = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return result


def _capability_section(env: PlatformEnv) -> dict[str, Any]:
    """Cheap tier-1 capability view. Never opens a plug-in."""

    try:
        from core.synth_capability import refresh_capabilities

        snapshot = refresh_capabilities(env=env, reason="environment snapshot")
        return snapshot.as_dict()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _coverage_section(db_path: Path | None) -> dict[str, Any]:
    """Library counts split by status, generation, format and pending reason."""

    if db_path is None or not Path(db_path).is_file():
        return {"available": False, "reason": "no library database yet"}
    try:
        from core.db import Database

        coverage = dict(Database(Path(db_path)).library_coverage())
        coverage["available"] = True
        return coverage
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def capture_environment(
    *,
    operation: str = "",
    operation_id: str = "",
    env: PlatformEnv = ENV,
    db_path: Path | None = None,
    linked_folder: Path | str | None = None,
    target_synth: str = "",
    quality_mode: str = "",
    library_counts: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    include_plugin_hashes: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> EnvironmentSnapshot:
    """Capture the machine/build/plug-in/model/library fingerprint once."""

    payload: dict[str, Any] = {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "operation_id": operation_id,
        "application": _application_section(),
        "system": _system_section(env),
        "dependencies": _dependency_versions(),
        "patchlab_state": {
            "linked_folder": str(linked_folder) if linked_folder else None,
            "target_serum_version": target_synth or None,
            "quality_mode": quality_mode or None,
            "app_data_dir": str(env.app_data_dir),
            "library_counts": sanitize(dict(library_counts or {})),
            "settings": sanitize(dict(settings or {})),
            "preset_roots": [str(path) for path in env.preset_roots],
            "existing_preset_roots": [str(path) for path in env.existing_preset_roots],
        },
        "database": _database_section(db_path),
        "model": _model_section(),
        "storage": _storage_section(),
        # The accept/reject table for every candidate, with the reason each was
        # rejected.  This is the single most valuable section for these bugs.
        "renderer_inventory": renderer_inventory(
            env=env, include_hash=include_plugin_hashes
        ),
        # Current synth capability, so a report says what PatchLab believed it
        # could run -- not just which files exist on disk.
        "synth_capability": _capability_section(env),
        # Discovered / processed / pending split, so "why are only N of my M
        # presets appearing?" is answerable without guessing.
        "library_coverage": _coverage_section(db_path),
        "path_access": _path_access(
            {
                "app_data_dir": env.app_data_dir,
                "linked_folder": linked_folder,
                "database": db_path,
                "diagnostics": recorder().root,
            }
        ),
        "flight_recorder": {
            "retention": recorder().retention_policy(),
            "counters": recorder().counters(),
            "session_id": recorder().session_id,
        },
        "resources": resource_snapshot(
            [path for path in (env.app_data_dir,) if path is not None]
        ),
    }
    if extra:
        payload["extra"] = sanitize(dict(extra))

    recorder().record(
        SUBSYSTEM,
        "environment_snapshot",
        f"environment captured for {operation or 'operation'}",
        operation_id=operation_id,
        phase="accepted",
        decision_reason="one snapshot per operation; plug-in and model hashes are cached",
        serum1_renderers_available=sum(
            1
            for item in payload["renderer_inventory"]
            if item.get("synth") == "serum1" and item.get("exists")
        ),
        serum2_renderers_available=sum(
            1
            for item in payload["renderer_inventory"]
            if item.get("synth") == "serum2" and item.get("exists")
        ),
        target_synth=target_synth or None,
    )
    return EnvironmentSnapshot(payload=payload)


def write_environment(
    snapshot: EnvironmentSnapshot | Mapping[str, Any], *, root: Path | None = None
) -> Path | None:
    """Persist the latest environment snapshot beside the flight recorder."""

    import json

    try:
        target_root = Path(root) if root is not None else recorder().root
        if target_root is None:
            return None
        target_root.mkdir(parents=True, exist_ok=True)
        payload = (
            snapshot.as_dict() if isinstance(snapshot, EnvironmentSnapshot) else dict(snapshot)
        )
        path = target_root / "environment.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(sanitize(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path
    except (OSError, ValueError, TypeError):
        return None


__all__ = [
    "EnvironmentSnapshot",
    "capture_environment",
    "write_environment",
]
