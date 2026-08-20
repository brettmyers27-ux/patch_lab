"""Cross-platform storage preferences, migration, and compact-cache cleanup."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from core.platform_env import ENV, PlatformEnv


STORAGE_SCHEMA = 1
DEFAULT_PREVIEW_CACHE_MB = 512
MIN_PREVIEW_CACHE_MB = 128
MAX_PREVIEW_CACHE_MB = 4096
SETTINGS_FILENAME = "storage-settings.json"
MARKER_FILENAME = ".patchlab-audio-storage.json"
LogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict[str, int | str]], None]


@dataclass(frozen=True, slots=True)
class StoragePreferences:
    audio_root: str | None = None
    compact_mode: bool = True
    preview_cache_mb: int = DEFAULT_PREVIEW_CACHE_MB
    schema: int = STORAGE_SCHEMA


@dataclass(frozen=True, slots=True)
class StorageStatus:
    root: Path
    configured_external: bool
    available: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class StorageOperationSummary:
    files: int = 0
    bytes: int = 0
    render_rows: int = 0
    preset_count: int = 0


def settings_path(env: PlatformEnv = ENV) -> Path:
    return Path(env.app_data_dir) / SETTINGS_FILENAME


def default_audio_root(env: PlatformEnv = ENV) -> Path:
    return (Path(env.app_data_dir) / "audio").expanduser().resolve()


def preview_cache_root(env: PlatformEnv = ENV) -> Path:
    """Root consumed by preview_cache_path(); audio lives below this root."""

    return (Path(env.app_data_dir) / "preview-cache").expanduser().resolve()


def load_storage_preferences(env: PlatformEnv = ENV) -> StoragePreferences:
    path = settings_path(env)
    if not path.is_file():
        return StoragePreferences()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        root = str(raw.get("audio_root") or "").strip() or None
        limit = min(
            max(int(raw.get("preview_cache_mb", DEFAULT_PREVIEW_CACHE_MB)), MIN_PREVIEW_CACHE_MB),
            MAX_PREVIEW_CACHE_MB,
        )
        return StoragePreferences(
            audio_root=root,
            compact_mode=bool(raw.get("compact_mode", True)),
            preview_cache_mb=limit,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # A damaged preference must not redirect writes to an unknown place.
        return StoragePreferences()


def save_storage_preferences(
    preferences: StoragePreferences,
    env: PlatformEnv = ENV,
) -> Path:
    path = settings_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(preferences)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def configured_audio_root(env: PlatformEnv = ENV) -> Path:
    override = os.environ.get("PATCHLAB_AUDIO_STORAGE", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    configured = load_storage_preferences(env).audio_root
    return (
        Path(configured).expanduser().resolve()
        if configured
        else default_audio_root(env)
    )


def storage_status(env: PlatformEnv = ENV) -> StorageStatus:
    preferences = load_storage_preferences(env)
    override = os.environ.get("PATCHLAB_AUDIO_STORAGE", "").strip()
    root = configured_audio_root(env)
    external = bool(override or preferences.audio_root)
    if not external:
        return StorageStatus(root, False, True)
    if not root.exists():
        return StorageStatus(
            root,
            True,
            False,
            "The configured audio drive or folder is not connected.",
        )
    if not root.is_dir():
        return StorageStatus(root, True, False, "The configured audio path is not a folder.")
    if not os.access(root, os.R_OK | os.W_OK):
        return StorageStatus(root, True, False, "The configured audio folder is not writable.")
    return StorageStatus(root, True, True)


def prepare_audio_root(root: Path) -> Path:
    """Create and prove a user-selected audio root is writable."""

    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / MARKER_FILENAME
    probe = root / f".patchlab-write-test-{time.time_ns()}"
    try:
        probe.write_bytes(b"PatchLab")
        probe.unlink()
        if not marker.exists():
            marker.write_text(
                json.dumps({"schema": STORAGE_SCHEMA, "purpose": "PatchLab audio"}),
                encoding="utf-8",
            )
    except OSError as exc:
        raise OSError(f"PatchLab cannot write to {root}: {exc}") from exc
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        if _sha256(destination) == _sha256(source):
            return
    temporary = destination.with_name(f".{destination.name}.patchlab-part")
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != source.stat().st_size or _sha256(temporary) != _sha256(source):
        temporary.unlink(missing_ok=True)
        raise OSError(f"Verification failed while copying {source.name}")
    temporary.replace(destination)


def _files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.name != MARKER_FILENAME)


def migrate_audio_storage(
    source_root: Path,
    destination_root: Path,
    *,
    database_path: Path | None = None,
    env: PlatformEnv = ENV,
    log: LogCallback = lambda _message: None,
    progress: ProgressCallback | None = None,
) -> StorageOperationSummary:
    """Move audio resumably across volumes, then commit the new preference.

    Each destination is checksum-verified before its database path is changed
    and before the source is removed. A crash can leave a duplicate, never the
    only copy missing.
    """

    source = Path(source_root).expanduser().resolve()
    destination = prepare_audio_root(destination_root)
    if source == destination:
        preferences = load_storage_preferences(env)
        save_storage_preferences(
            StoragePreferences(
                str(destination),
                preferences.compact_mode,
                preferences.preview_cache_mb,
            ),
            env,
        )
        return StorageOperationSummary()
    files = _files_under(source)
    required = 0
    for path in files:
        target = destination / path.relative_to(source)
        if (
            not target.is_file()
            or target.stat().st_size != path.stat().st_size
            or _sha256(target) != _sha256(path)
        ):
            required += path.stat().st_size
    free = shutil.disk_usage(destination).free
    if required > free:
        raise OSError(
            f"Not enough free space at {destination}: need {required:,} bytes, "
            f"only {free:,} bytes are available."
        )
    connection = (
        sqlite3.connect(database_path)
        if database_path and Path(database_path).is_file()
        else None
    )
    moved_bytes = 0
    try:
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(source)
            target = destination / relative
            size = path.stat().st_size
            _copy_verified(path, target)
            if connection is not None:
                connection.execute(
                    "UPDATE renders SET wav_path=? WHERE wav_path=?",
                    (str(target), str(path)),
                )
                connection.commit()
            path.unlink()
            moved_bytes += size
            log(f"Moved {relative}")
            if progress is not None:
                progress(
                    {
                        "phase": "move",
                        "current": index,
                        "total": len(files),
                        "bytes": moved_bytes,
                    }
                )
    finally:
        if connection is not None:
            connection.close()
    # Remove empty directories only after every contained file is safe.
    if source.is_dir():
        for directory in sorted(
            (path for path in source.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
    preferences = load_storage_preferences(env)
    save_storage_preferences(
        StoragePreferences(
            str(destination),
            preferences.compact_mode,
            preferences.preview_cache_mb,
        ),
        env,
    )
    return StorageOperationSummary(files=len(files), bytes=moved_bytes)


def compact_render_library(
    database_path: Path,
    audio_root: Path,
    *,
    log: LogCallback = lambda _message: None,
) -> StorageOperationSummary:
    """Remove regenerable full renders while retaining fingerprints and labels."""

    database_path = Path(database_path).expanduser().resolve()
    audio_root = Path(audio_root).expanduser().resolve()
    if not database_path.is_file():
        return StorageOperationSummary()
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT r.preset_id,r.wav_path
        FROM renders r
        WHERE EXISTS (
          SELECT 1 FROM fingerprints f
          WHERE f.preset_id=r.preset_id AND f.midi_note=0
        )
        """
    ).fetchall()
    paths_by_preset: dict[int, list[Path]] = {}
    unsafe_presets: set[int] = set()
    for row in rows:
        preset_id = int(row["preset_id"])
        path = Path(str(row["wav_path"])).expanduser().resolve()
        try:
            path.relative_to(audio_root)
        except ValueError:
            unsafe_presets.add(preset_id)
            log(
                "Skipped preset with a render outside the configured audio root: "
                f"{path}"
            )
            continue
        paths_by_preset.setdefault(preset_id, []).append(path)
    preset_ids = sorted(set(paths_by_preset).difference(unsafe_presets))
    removed_files = 0
    removed_bytes = 0
    for preset_id in preset_ids:
        for path in paths_by_preset[preset_id]:
            if path.is_file():
                size = path.stat().st_size
                path.unlink()
                removed_files += 1
                removed_bytes += size
    if preset_ids:
        placeholders = ",".join("?" for _ in preset_ids)
        connection.execute(
            f"DELETE FROM renders WHERE preset_id IN ({placeholders})",
            tuple(preset_ids),
        )
        connection.execute(
            f"UPDATE presets SET status='embedded',error=NULL WHERE id IN ({placeholders})",
            tuple(preset_ids),
        )
    connection.commit()
    connection.close()
    for preset_id in preset_ids:
        directory = audio_root / str(preset_id)
        try:
            directory.rmdir()
        except OSError:
            pass
    return StorageOperationSummary(
        files=removed_files,
        bytes=removed_bytes,
        render_rows=sum(len(paths_by_preset[preset_id]) for preset_id in preset_ids),
        preset_count=len(preset_ids),
    )


def prune_preview_cache(
    root: Path,
    limit_mb: int,
    *,
    protected: Iterable[Path] = (),
) -> StorageOperationSummary:
    """Keep the internal preview cache under a hard LRU size ceiling."""

    audio = Path(root).expanduser().resolve() / "audio"
    if not audio.is_dir():
        return StorageOperationSummary()
    limit = min(max(int(limit_mb), MIN_PREVIEW_CACHE_MB), MAX_PREVIEW_CACHE_MB) * 1024 * 1024
    protected_paths = {Path(path).expanduser().resolve() for path in protected}
    files = [path for path in audio.rglob("*.wav") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    removed = 0
    removed_bytes = 0
    for path in sorted(files, key=lambda item: item.stat().st_mtime):
        if total <= limit:
            break
        if path.resolve() in protected_paths:
            continue
        size = path.stat().st_size
        path.unlink()
        total -= size
        removed += 1
        removed_bytes += size
    for directory in sorted(
        (path for path in audio.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    return StorageOperationSummary(files=removed, bytes=removed_bytes)
