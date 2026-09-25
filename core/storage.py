"""Cross-platform storage preferences, migration, and compact-cache cleanup."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from core.platform_env import ENV, PlatformEnv
from core.prepared_state import prepared_predicate


STORAGE_SCHEMA = 1
DEFAULT_PREVIEW_CACHE_MB = 512
MIN_PREVIEW_CACHE_MB = 128
MAX_PREVIEW_CACHE_MB = 4096
SETTINGS_FILENAME = "storage-settings.json"
MARKER_FILENAME = ".patchlab-audio-storage.json"
PATCHLAB_RENDER_NOTES = frozenset((24, 36, 48, 60, 72, 84, 96))
_RENDER_TEMPORARY_NAME = re.compile(r"^\.(24|36|48|60|72|84|96)\.\d+\.tmp\.wav$")
_HASH_CACHE_DIRECTORY = re.compile(r"^[0-9a-f]{40}$")
_STALE_TEMPORARY_AGE_SECONDS = 60 * 60
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


@dataclass(frozen=True, slots=True)
class LegacyAdoptionSummary:
    """Outcome of reconciling an orphaned legacy render folder against the DB.

    Every count here is a *preset*, not a file, since adoption is decided per
    preset (all its expected notes must be present and valid) even though the
    underlying copy happens note by note.
    """

    presets_adopted: int = 0
    notes_copied: int = 0
    bytes_copied: int = 0
    presets_already_complete: int = 0
    presets_uncataloged: int = 0
    presets_partial_conflict: int = 0
    folders_unclassified: int = 0


def settings_path(env: PlatformEnv = ENV) -> Path:
    return Path(env.app_data_dir) / SETTINGS_FILENAME


def default_audio_root(env: PlatformEnv = ENV) -> Path:
    return (Path(env.app_data_dir) / "audio").expanduser().resolve()


def preview_cache_root(env: PlatformEnv = ENV) -> Path:
    """Root consumed by preview_cache_path(); audio lives below this root."""

    return (Path(env.app_data_dir) / "preview-cache").expanduser().resolve()


def preview_cache_usage(root: Path) -> int:
    """Return bytes owned by the preview cache, never by analysis work."""

    audio = Path(root).expanduser().resolve() / "audio"
    if not audio.is_dir():
        return 0
    return sum(path.stat().st_size for path in audio.rglob("*.wav") if path.is_file())


def clear_preview_cache(root: Path) -> StorageOperationSummary:
    """Delete only disposable preview-cache WAVs and their empty directories."""

    audio = Path(root).expanduser().resolve() / "audio"
    if not audio.is_dir():
        return StorageOperationSummary()
    files = 0
    byte_count = 0
    for path in audio.rglob("*.wav"):
        if not path.is_file():
            continue
        byte_count += path.stat().st_size
        path.unlink()
        files += 1
    for directory in sorted((path for path in audio.rglob("*") if path.is_dir()),
                            key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        audio.rmdir()
    except OSError:
        pass
    return StorageOperationSummary(files=files, bytes=byte_count)


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


def audio_root_size(root: Path) -> int:
    """Approximate on-disk size of everything under an audio root.

    Informational only (Settings display) -- never gates a decision, so an
    unreadable or still-mounting drive just reports 0 rather than raising.
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


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
    prepared_sql, prepared_parameters = prepared_predicate("p")
    rows = connection.execute(
        f"""
        SELECT r.preset_id,r.wav_path,p.status,
          ({prepared_sql}) AS is_prepared
        FROM renders r
        JOIN presets p ON p.id=r.preset_id
        WHERE ({prepared_sql})
        """,
        prepared_parameters,
    ).fetchall()
    paths_by_preset: dict[int, list[Path]] = {}
    learned_preset_ids: set[int] = set()
    unsafe_presets: set[int] = set()
    for row in rows:
        preset_id = int(row["preset_id"])
        if bool(row["is_prepared"]):
            learned_preset_ids.add(preset_id)
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
    learned_preset_ids.intersection_update(preset_ids)
    # First make the durable database state authoritative.  Every selected
    # preset already has its mean fingerprint, so a crash after this commit
    # can at worst leave regenerable WAVs behind; it can never leave database
    # rows that falsely claim deleted WAVs still exist and therefore block a
    # future render resume.
    if preset_ids:
        placeholders = ",".join("?" for _ in preset_ids)
        connection.execute(
            f"DELETE FROM renders WHERE preset_id IN ({placeholders})",
            tuple(preset_ids),
        )
        if learned_preset_ids:
            learned_placeholders = ",".join("?" for _ in learned_preset_ids)
            connection.execute(
                f"UPDATE presets SET status='embedded',error=NULL "
                f"WHERE id IN ({learned_placeholders})",
                tuple(sorted(learned_preset_ids)),
            )
    connection.commit()
    cleanupable_preset_ids = {
        int(row[0])
        for row in connection.execute(
            f"""
            SELECT p.id FROM presets p
            WHERE ({prepared_sql})
            """,
            prepared_parameters,
        ).fetchall()
    }
    connection.close()

    removed_files = 0
    removed_bytes = 0
    for preset_id in preset_ids:
        for path in paths_by_preset[preset_id]:
            try:
                if path.is_file():
                    size = path.stat().st_size
                    path.unlink()
                    removed_files += 1
                    removed_bytes += size
            except OSError as exc:
                log(f"Will retry cleanup of regenerable render {path.name}: {exc}")

    # A worker writes to a hidden, per-process temporary name before the
    # atomic rename.  A forced quit can strand one.  These have no database
    # row and are never a completed render, so reclaiming only old files with
    # this exact private name is safe.  Likewise, old releases stored preview
    # cache folders under audio_root; current releases use preview-cache/.
    # Restrict the migration cleanup to PatchLab's own generated/hash naming
    # schemes and never touch numeric folders unless the preset is durable.
    cleanup_files, cleanup_bytes = _cleanup_stale_audio_residue(
        audio_root, durable_preset_ids=cleanupable_preset_ids, log=log
    )
    removed_files += cleanup_files
    removed_bytes += cleanup_bytes
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


def _cleanup_stale_audio_residue(
    audio_root: Path,
    *,
    durable_preset_ids: set[int],
    log: LogCallback,
) -> tuple[int, int]:
    """Reclaim only provably regenerable residue in PatchLab's audio root.

    This intentionally leaves unknown numeric folders and all non-PatchLab
    filenames alone.  It is safe to run repeatedly after compacting a batch.
    """

    if not audio_root.is_dir():
        return (0, 0)
    cutoff = time.time() - _STALE_TEMPORARY_AGE_SECONDS
    removed_files = 0
    removed_bytes = 0

    def remove_file(path: Path) -> None:
        nonlocal removed_files, removed_bytes
        try:
            size = path.stat().st_size
            path.unlink()
            removed_files += 1
            removed_bytes += size
        except OSError as exc:
            log(f"Will retry cleanup of temporary audio {path.name}: {exc}")

    for directory in audio_root.iterdir():
        if not directory.is_dir():
            continue
        # Pre-compact preview layouts: generated recommendations and
        # content-hash lookup folders.  The current preview cache is a
        # separate bounded directory, so these legacy copies are unused.
        if directory.name.startswith("generated-") or _HASH_CACHE_DIRECTORY.fullmatch(
            directory.name
        ):
            for path in directory.rglob("*"):
                if path.is_file():
                    remove_file(path)
            for child in sorted(
                (path for path in directory.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                try:
                    child.rmdir()
                except OSError:
                    pass
            try:
                directory.rmdir()
            except OSError:
                pass
            continue
        if not directory.name.isdigit():
            continue
        preset_id = int(directory.name)
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if _RENDER_TEMPORARY_NAME.fullmatch(path.name):
                try:
                    if path.stat().st_mtime <= cutoff:
                        remove_file(path)
                except OSError:
                    continue
            elif (
                preset_id in durable_preset_ids
                and path.stem.isdigit()
                and int(path.stem) in PATCHLAB_RENDER_NOTES
                and path.suffix == ".wav"
            ):
                # A file can survive a filesystem error after the database
                # commit above.  It is no longer a required library asset.
                remove_file(path)
        try:
            directory.rmdir()
        except OSError:
            pass
    return removed_files, removed_bytes


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


def adopt_legacy_renders(
    legacy_root: Path,
    *,
    database_path: Path,
    audio_root: Path,
    log: LogCallback = lambda _message: None,
    progress: ProgressCallback | None = None,
) -> LegacyAdoptionSummary:
    """Reconcile an orphaned legacy render folder against the current library.

    Handles a folder the database has never heard of -- e.g. an old manual
    backup -- as distinct from migrate_audio_storage(), which only relocates
    renders the database already tracks. Never deletes or modifies anything
    under legacy_root: this is copy-only, and only for notes the database does
    not already have. A preset already fully covered internally is left
    untouched rather than re-copied, so this is safe to re-run.

    Only numeric preset-id folders are adopted -- PatchLab's renderer always
    names a preset's folder after its integer id (core/render.py's
    _write_note: `audio_root / str(task.preset_id)`). Any other folder name
    (e.g. a content-hash-named synthesis-experiment folder) is a different,
    unverified identity and is reported as unclassified rather than guessed.
    """

    import numpy as np
    import soundfile as sf

    from core.db import Database, RenderRecord
    from core.render import MIDI_NOTES
    from core.render import _dbfs as render_dbfs

    legacy_root = Path(legacy_root).expanduser().resolve()
    audio_root = Path(audio_root).expanduser().resolve()
    database = Database(Path(database_path).expanduser().resolve())
    with database.connect() as connection:
        known_ids = {int(row["id"]) for row in connection.execute("SELECT id FROM presets")}
    existing = database.existing_render_notes()

    summary = {
        "presets_adopted": 0,
        "notes_copied": 0,
        "bytes_copied": 0,
        "presets_already_complete": 0,
        "presets_uncataloged": 0,
        "presets_partial_conflict": 0,
        "folders_unclassified": 0,
    }
    candidates = sorted(
        p for p in legacy_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    for index, folder in enumerate(candidates, start=1):
        if progress is not None:
            progress({"phase": "adopt", "current": index, "total": len(candidates)})
        if not folder.name.isdigit():
            summary["folders_unclassified"] += 1
            continue
        preset_id = int(folder.name)
        if preset_id not in known_ids:
            summary["presets_uncataloged"] += 1
            log(f"Skipped preset {preset_id}: no longer in the catalog")
            continue
        have = existing.get(preset_id, set())
        missing_notes = [note for note in MIDI_NOTES if note not in have]
        if not missing_notes:
            summary["presets_already_complete"] += 1
            continue

        # Validate every missing note before writing anything. A preset is
        # adopted all-or-nothing so a partial/corrupt legacy folder can never
        # leave stray copied files with no matching database row.
        planned: list[tuple[int, Path, Path, np.ndarray, int]] = []
        valid = True
        for note in missing_notes:
            source = folder / f"{note}.wav"
            if not source.is_file():
                valid = False
                log(
                    f"Skipped preset {preset_id}: legacy folder is missing note "
                    f"{note}, so nothing was adopted for this preset"
                )
                break
            try:
                info = sf.info(str(source))
                if info.frames == 0 or info.samplerate == 0:
                    raise OSError("empty or zero-rate audio")
                data, rate = sf.read(str(source), dtype="float32", always_2d=True)
            except Exception as exc:
                valid = False
                log(f"Skipped preset {preset_id}: note {note} is unreadable ({exc})")
                break
            destination = audio_root / str(preset_id) / f"{note}.wav"
            if destination.is_file() and destination.stat().st_size != source.stat().st_size:
                # Something already occupies this exact slot even though the
                # database has no row for it -- do not silently overwrite.
                valid = False
                log(
                    f"Skipped preset {preset_id}: an unregistered file already "
                    f"exists at note {note}'s destination with different content"
                )
                break
            planned.append((note, source, destination, data, rate))
        if not valid or not planned:
            summary["presets_partial_conflict"] += 1
            continue

        rows: list[RenderRecord] = []
        for note, source, destination, data, rate in planned:
            if not destination.is_file():
                _copy_verified(source, destination)
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)))) if data.size else 0.0
            rows.append(
                RenderRecord(
                    preset_id=preset_id,
                    midi_note=note,
                    wav_path=destination,
                    peak_dbfs=render_dbfs(peak),
                    rms_dbfs=render_dbfs(rms),
                    duration_s=data.shape[0] / rate,
                )
            )
        database.upsert_renders(rows)
        database.finalize_render_status(preset_id, MIDI_NOTES)
        summary["presets_adopted"] += 1
        summary["notes_copied"] += len(rows)
        summary["bytes_copied"] += sum(
            row.wav_path.stat().st_size for row in rows if row.wav_path.is_file()
        )
        log(f"Adopted preset {preset_id}: {len(rows)} note(s) recovered from legacy storage")
    return LegacyAdoptionSummary(**summary)
