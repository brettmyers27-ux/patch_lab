"""Crash-safe, per-preset streaming preparation for personal Match data."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, get_ident
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import soundfile as sf

from core.db import Database, PresetRecord
from core.features import ClapEmbedder, handcrafted_features, load_audio_48k_mono
from core.library_state import get_presets_needing_preparation
from core.platform_env import ENV, PlatformEnv
from core.prepared_state import (
    CURRENT_PREPARED_REVISION,
    HANDCRAFTED_FLOAT_COUNT,
    PreparedRevision,
    has_required_permanent_data,
    is_preset_prepared,
    prepared_revision_token,
    record_prepared_revision,
)
from core.preset_scan import sha1_file
from core.render import MIDI_NOTES, SAMPLE_RATE, RenderSummary, render_library


ANALYSIS_DIRECTORY_NAME = "analysis-work"
ANALYSIS_MARKER_NAME = ".patchlab-analysis-work.json"
MINIMUM_WORKING_FREE_BYTES = 512 * 1024 * 1024

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]
StageHook = Callable[[str, int], None]
FreeSpaceProvider = Callable[[Path], int]
UnlinkFile = Callable[[Path], None]
RenderFunction = Callable[..., RenderSummary]
FingerprintFunction = Callable[[Database, Any, int], bool]


@dataclass(slots=True)
class PreparationSummary:
    queued: int = 0
    attempted: int = 0
    prepared: int = 0
    rendered_notes: int = 0
    reused_render_notes: int = 0
    fingerprints_created: int = 0
    failed: int = 0
    cleanup_failures: int = 0
    cleaned_files: int = 0
    cleaned_bytes: int = 0
    peak_temp_bytes: int = 0
    cancelled: bool = False


@dataclass(slots=True)
class RecoverySummary:
    incomplete_temp_files_removed: int = 0
    prepared_presets_cleaned: int = 0
    cleaned_files: int = 0
    cleanup_failures: int = 0


@dataclass(frozen=True, slots=True)
class _PipelineItem:
    """One durable job moving through the bounded render/analyze handoff."""

    preset: PresetRecord
    index: int
    temp_dir: Path


def analysis_temp_root(env: PlatformEnv = ENV) -> Path:
    return (Path(env.app_data_dir) / ANALYSIS_DIRECTORY_NAME).expanduser().resolve()


def _ensure_analysis_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ANALYSIS_MARKER_NAME
    if not marker.exists():
        # Two bounded pipeline children can initialize the workspace together.
        # Give the atomic marker write a private temporary name before replace.
        temporary = root / f".{ANALYSIS_MARKER_NAME}.{get_ident()}.tmp"
        temporary.write_text(
            json.dumps({"owner": "PatchLab", "purpose": "temporary preset analysis"}),
            encoding="utf-8",
        )
        temporary.replace(marker)
    return root


def _is_within(path: Path, roots: Iterable[Path]) -> bool:
    resolved = path.expanduser().resolve()
    for root in roots:
        try:
            resolved.relative_to(Path(root).expanduser().resolve())
            return True
        except ValueError:
            continue
    return False


def _valid_render(path: Path) -> bool:
    try:
        info = sf.info(path)
    except (OSError, RuntimeError):
        return False
    return info.samplerate == SAMPLE_RATE and info.channels == 2 and info.frames > 0


def _tree_size(root: Path) -> int:
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


def _job_row(database: Database, preset_id: int):
    with database.connect() as connection:
        return connection.execute(
            "SELECT * FROM preparation_jobs WHERE preset_id=?", (preset_id,)
        ).fetchone()


def _set_job(
    database: Database,
    preset: PresetRecord,
    *,
    state: str,
    temp_dir: Path,
    revision_token: str,
    error: str | None = None,
    cleanup_needed: bool = False,
    increment_attempt: bool = False,
) -> None:
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO preparation_jobs(
              preset_id,expected_content_hash,target_revision,state,temp_dir,
              cleanup_needed,attempt_count,last_error
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(preset_id) DO UPDATE SET
              expected_content_hash=excluded.expected_content_hash,
              target_revision=excluded.target_revision,
              state=excluded.state,
              temp_dir=excluded.temp_dir,
              cleanup_needed=excluded.cleanup_needed,
              attempt_count=preparation_jobs.attempt_count + ?,
              last_error=excluded.last_error,
              updated_at=CURRENT_TIMESTAMP
            """,
            (
                preset.id,
                preset.content_hash,
                revision_token,
                state,
                str(temp_dir),
                1 if cleanup_needed else 0,
                1 if increment_attempt else 0,
                error[:4000] if error else None,
                1 if increment_attempt else 0,
            ),
        )


def _verified_source(database: Database, preset: PresetRecord) -> Path | None:
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT normalized_path FROM preset_sources "
            "WHERE preset_id=? AND content_hash=? AND active=1 "
            "ORDER BY normalized_path",
            (preset.id, preset.content_hash),
        ).fetchall()
    for row in rows:
        path = Path(str(row[0]))
        try:
            if path.is_file() and sha1_file(path) == preset.content_hash:
                with database.connect() as connection:
                    connection.execute(
                        "UPDATE presets SET path=?,name=? WHERE id=?",
                        (str(path.resolve()), path.stem, preset.id),
                    )
                return path.resolve()
        except OSError:
            continue
    return None


def _reconcile_render_rows(
    database: Database,
    preset_id: int,
    *,
    allowed_roots: Sequence[Path],
) -> dict[int, Path]:
    valid: dict[int, Path] = {}
    invalid_notes: list[int] = []
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT midi_note,wav_path FROM renders WHERE preset_id=?",
            (preset_id,),
        ).fetchall()
    for row in rows:
        note = int(row["midi_note"])
        path = Path(str(row["wav_path"])).expanduser().resolve()
        if (
            note in MIDI_NOTES
            and _is_within(path, allowed_roots)
            and path.is_file()
            and _valid_render(path)
        ):
            valid[note] = path
        else:
            invalid_notes.append(note)
    if invalid_notes:
        with database.connect() as connection:
            connection.executemany(
                "DELETE FROM renders WHERE preset_id=? AND midi_note=?",
                [(preset_id, note) for note in invalid_notes],
            )
    return valid


def _discard_incompatible_renders(
    database: Database,
    preset_id: int,
    *,
    allowed_roots: Sequence[Path],
    unlink_file: UnlinkFile,
) -> None:
    """Remove PatchLab-owned renders produced by an obsolete render contract."""

    with database.connect() as connection:
        rows = connection.execute(
            "SELECT midi_note,wav_path FROM renders WHERE preset_id=?",
            (preset_id,),
        ).fetchall()
    removable: list[int] = []
    for row in rows:
        note = int(row["midi_note"])
        path = Path(str(row["wav_path"])).expanduser().resolve()
        if _is_within(path, allowed_roots) and path.is_file():
            # If deletion fails, keep the row so the useful old render remains
            # recoverable. A later invocation can retry this reset safely.
            unlink_file(path)
        removable.append(note)
    if removable:
        with database.connect() as connection:
            connection.executemany(
                "DELETE FROM renders WHERE preset_id=? AND midi_note=?",
                [(preset_id, note) for note in removable],
            )


def fingerprint_render_rows(
    database: Database,
    embedder: Any,
    preset_id: int,
) -> bool:
    """Write the established seven per-note features and aggregate fingerprint."""

    with database.connect() as connection:
        paths = {
            int(row["midi_note"]): Path(str(row["wav_path"]))
            for row in connection.execute(
                "SELECT midi_note,wav_path FROM renders WHERE preset_id=?",
                (preset_id,),
            ).fetchall()
            if int(row["midi_note"]) in MIDI_NOTES
        }
    if set(paths) != set(MIDI_NOTES):
        return False
    prepared_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for note in MIDI_NOTES:
        prepared = load_audio_48k_mono(paths[note])
        prepared_rows.append(
            (note, prepared.waveform, handcrafted_features(prepared.waveform))
        )
    embeddings = embedder.embed([row[1] for row in prepared_rows])
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for (note, _waveform, handcrafted), embedding in zip(
        prepared_rows, embeddings, strict=True
    ):
        database.upsert_fingerprint(
            preset_id,
            note,
            np.ascontiguousarray(embedding, dtype=np.float32).tobytes(),
            np.ascontiguousarray(handcrafted, dtype=np.float32).tobytes(),
        )
        rows.append((note, embedding, handcrafted))
    mean_embedding = np.mean([row[1] for row in rows], axis=0)
    mean_embedding /= max(float(np.linalg.norm(mean_embedding)), 1e-12)
    mean_handcrafted = np.mean([row[2] for row in rows], axis=0)
    database.upsert_fingerprint(
        preset_id,
        0,
        np.ascontiguousarray(mean_embedding, dtype=np.float32).tobytes(),
        np.ascontiguousarray(mean_handcrafted, dtype=np.float32).tobytes(),
    )
    return len(mean_handcrafted) == HANDCRAFTED_FLOAT_COUNT


def _commit_prepared(
    database: Database,
    preset: PresetRecord,
    *,
    temp_dir: Path,
    revision: PreparedRevision,
    revision_token: str,
) -> bool:
    """Validate, record PREPARED, and persist cleanup-needed in one transaction."""

    with database.connect() as connection:
        connection.execute(
            "UPDATE preparation_jobs SET state='committing',last_error=NULL,"
            "updated_at=CURRENT_TIMESTAMP WHERE preset_id=?",
            (preset.id,),
        )
        if not record_prepared_revision(connection, preset.id, revision=revision):
            return False
        if not is_preset_prepared(connection, preset.id, revision=revision):
            connection.execute(
                "DELETE FROM prepared_presets WHERE preset_id=?", (preset.id,)
            )
            return False
        connection.execute(
            "UPDATE presets SET status='embedded',error=NULL WHERE id=?", (preset.id,)
        )
        connection.execute(
            """
            UPDATE preparation_jobs
            SET expected_content_hash=?,target_revision=?,state='prepared',
                temp_dir=?,cleanup_needed=1,last_error=NULL,
                updated_at=CURRENT_TIMESTAMP
            WHERE preset_id=?
            """,
            (preset.content_hash, revision_token, str(temp_dir), preset.id),
        )
    return True


def _cleanup_preset(
    database: Database,
    preset_id: int,
    *,
    allowed_roots: Sequence[Path],
    unlink_file: UnlinkFile,
) -> tuple[int, int, list[str]]:
    files = 0
    byte_count = 0
    errors: list[str] = []
    removable_notes: list[int] = []
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT midi_note,wav_path FROM renders WHERE preset_id=?",
            (preset_id,),
        ).fetchall()
    for row in rows:
        note = int(row["midi_note"])
        path = Path(str(row["wav_path"])).expanduser().resolve()
        if not _is_within(path, allowed_roots):
            errors.append(f"refused cleanup outside PatchLab storage: {path}")
            continue
        try:
            size = path.stat().st_size if path.is_file() else 0
            if path.is_file():
                unlink_file(path)
                files += 1
                byte_count += size
            removable_notes.append(note)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    if removable_notes:
        with database.connect() as connection:
            connection.executemany(
                "DELETE FROM renders WHERE preset_id=? AND midi_note=?",
                [(preset_id, note) for note in removable_notes],
            )
    return files, byte_count, errors


def _finish_cleanup(
    database: Database,
    preset: PresetRecord,
    *,
    analysis_root: Path,
    legacy_audio_root: Path,
    unlink_file: UnlinkFile,
) -> tuple[int, int, bool]:
    files, byte_count, errors = _cleanup_preset(
        database,
        preset.id,
        allowed_roots=(analysis_root, legacy_audio_root),
        unlink_file=unlink_file,
    )
    temp_dir = analysis_root / "jobs" / str(preset.id)
    try:
        if temp_dir.is_dir() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    except OSError as exc:
        errors.append(str(exc))
    with database.connect() as connection:
        connection.execute(
            "UPDATE preparation_jobs SET state=?,cleanup_needed=?,last_error=?,"
            "updated_at=CURRENT_TIMESTAMP WHERE preset_id=?",
            (
                "prepared" if errors else "cleanup_complete",
                1 if errors else 0,
                "\n".join(errors)[:4000] if errors else None,
                preset.id,
            ),
        )
    return files, byte_count, bool(errors)


def recover_preparation_state(
    database: Database,
    *,
    analysis_root: Path,
    legacy_audio_root: Path,
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
    unlink_file: UnlinkFile = Path.unlink,
) -> RecoverySummary:
    """Perform lightweight, idempotent cleanup without rendering or analysis."""

    root = _ensure_analysis_root(analysis_root)
    summary = RecoverySummary()
    for path in root.rglob(".*.tmp.wav"):
        try:
            unlink_file(path)
            summary.incomplete_temp_files_removed += 1
        except OSError:
            pass
    with database.connect() as connection:
        candidate_ids = {
            int(row[0])
            for row in connection.execute(
                "SELECT preset_id FROM preparation_jobs WHERE cleanup_needed=1 "
                "UNION SELECT DISTINCT preset_id FROM renders"
            ).fetchall()
        }
        preset_rows = {
            int(row["id"]): Database._preset(row)
            for row in connection.execute("SELECT * FROM presets").fetchall()
        }
    for preset_id in sorted(candidate_ids):
        preset = preset_rows.get(preset_id)
        if preset is None:
            continue
        with database.connect() as connection:
            prepared = is_preset_prepared(connection, preset_id, revision=revision)
        if not prepared:
            continue
        files, _bytes, failed = _finish_cleanup(
            database,
            preset,
            analysis_root=root,
            legacy_audio_root=legacy_audio_root,
            unlink_file=unlink_file,
        )
        summary.cleaned_files += files
        if failed:
            summary.cleanup_failures += 1
        else:
            summary.prepared_presets_cleaned += 1
    return summary


def prepare_work_queue(
    *,
    db_path: Path,
    analysis_root: Path,
    legacy_audio_root: Path,
    state_dir: Path,
    env: PlatformEnv = ENV,
    preset_ids: Sequence[int] | None = None,
    render_processes: int = 1,
    allow_render: bool = True,
    log: LogCallback = print,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck = lambda: False,
    stage_hook: StageHook = lambda _stage, _preset_id: None,
    render_function: RenderFunction = render_library,
    fingerprint_function: FingerprintFunction = fingerprint_render_rows,
    embedder_factory: Callable[[PlatformEnv], Any] = ClapEmbedder,
    free_space_provider: FreeSpaceProvider = lambda path: int(shutil.disk_usage(path).free),
    minimum_free_bytes: int = MINIMUM_WORKING_FREE_BYTES,
    unlink_file: UnlinkFile = Path.unlink,
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
) -> PreparationSummary:
    """Run a capacity-one render handoff over the durable serial lifecycle.

    Each child still uses the established per-preset transaction and recovery
    path.  Two adjacent children are permitted in flight: locks ensure that
    only one owns Serum and only one owns CLAP analysis.  Consequently the
    second child's render can overlap the first child's analysis, while no
    unbounded rendered backlog or simultaneous Serum host is possible.
    """

    database = Database(Path(db_path).expanduser().resolve())
    queue = get_presets_needing_preparation(database, revision=revision)
    if preset_ids is not None:
        wanted = {int(item) for item in preset_ids}
        queue = [preset for preset in queue if preset.id in wanted]
    if len(queue) < 2:
        return _prepare_work_queue_serial(
            db_path=db_path, analysis_root=analysis_root, legacy_audio_root=legacy_audio_root,
            state_dir=state_dir, env=env, preset_ids=preset_ids,
            render_processes=render_processes, allow_render=allow_render, log=log,
            progress=progress, cancel_check=cancel_check, stage_hook=stage_hook,
            render_function=render_function, fingerprint_function=fingerprint_function,
            embedder_factory=embedder_factory, free_space_provider=free_space_provider,
            minimum_free_bytes=minimum_free_bytes, unlink_file=unlink_file, revision=revision,
        )

    render_lock = Lock()
    analysis_lock = Lock()
    state_lock = Lock()
    shared_embedder: Any | None = None
    terminal = 0
    aggregate = PreparationSummary(queued=len(queue))

    def shared_embedder_factory(requested_env: PlatformEnv) -> Any:
        nonlocal shared_embedder
        with analysis_lock:
            if shared_embedder is None:
                shared_embedder = embedder_factory(requested_env)
            return shared_embedder

    def one_renderer(**kwargs: Any) -> RenderSummary:
        with render_lock:
            # A successor may already be waiting at the capacity-one handoff
            # when consent is withdrawn. Never begin that renderer.
            if cancel_check():
                return RenderSummary(cancelled=True)
            return render_function(**kwargs)

    def one_analyzer(target_db: Database, embedder: Any, preset_id: int) -> bool:
        with analysis_lock:
            return fingerprint_function(target_db, embedder, preset_id)

    def merge(summary: PreparationSummary) -> None:
        for name in (
            "attempted", "prepared", "rendered_notes", "reused_render_notes",
            "fingerprints_created", "failed", "cleanup_failures", "cleaned_files",
            "cleaned_bytes",
        ):
            setattr(aggregate, name, getattr(aggregate, name) + getattr(summary, name))
        aggregate.peak_temp_bytes = max(aggregate.peak_temp_bytes, summary.peak_temp_bytes)
        aggregate.cancelled = aggregate.cancelled or summary.cancelled

    def run_one(preset: PresetRecord) -> PreparationSummary:
        def child_progress(detail: dict[str, Any]) -> None:
            nonlocal terminal
            stage = str(detail.get("current_stage", ""))
            with state_lock:
                if stage in {"complete", "failed"}:
                    terminal += 1
                current = terminal
            if progress is not None:
                copied = dict(detail)
                copied["current"] = current
                copied["total"] = len(queue)
                progress(copied)

        return _prepare_work_queue_serial(
            db_path=db_path, analysis_root=analysis_root, legacy_audio_root=legacy_audio_root,
            state_dir=state_dir, env=env, preset_ids=[preset.id],
            render_processes=render_processes, allow_render=allow_render, log=log,
            progress=child_progress, cancel_check=cancel_check, stage_hook=stage_hook,
            render_function=one_renderer, fingerprint_function=one_analyzer,
            embedder_factory=shared_embedder_factory, free_space_provider=free_space_provider,
            minimum_free_bytes=minimum_free_bytes, unlink_file=unlink_file, revision=revision,
        )

    next_index = 0
    active: dict[Future[PreparationSummary], PresetRecord] = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="patchlab-preparation") as executor:
        while (next_index < len(queue) or active) and not aggregate.cancelled:
            while next_index < len(queue) and len(active) < 2 and not cancel_check():
                preset = queue[next_index]
                next_index += 1
                active[executor.submit(run_one, preset)] = preset
            if not active:
                aggregate.cancelled = cancel_check()
                break
            future = next(iter(active))
            active.pop(future)
            result = future.result()  # Preserve BaseException crash semantics.
            with state_lock:
                merge(result)
            if result.cancelled:
                aggregate.cancelled = True
        # Executor shutdown waits only for the one other bounded item.  Its
        # durable state remains recoverable even when cancellation won the race.
        for future in active:
            result = future.result()
            with state_lock:
                merge(result)
    return aggregate


def _disk_free(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _prepare_work_queue_serial(
    *,
    db_path: Path,
    analysis_root: Path,
    legacy_audio_root: Path,
    state_dir: Path,
    env: PlatformEnv = ENV,
    preset_ids: Sequence[int] | None = None,
    render_processes: int = 1,
    allow_render: bool = True,
    log: LogCallback = print,
    progress: ProgressCallback | None = None,
    cancel_check: CancelCheck = lambda: False,
    stage_hook: StageHook = lambda _stage, _preset_id: None,
    render_function: RenderFunction = render_library,
    fingerprint_function: FingerprintFunction = fingerprint_render_rows,
    embedder_factory: Callable[[PlatformEnv], Any] = ClapEmbedder,
    free_space_provider: FreeSpaceProvider = _disk_free,
    minimum_free_bytes: int = MINIMUM_WORKING_FREE_BYTES,
    unlink_file: UnlinkFile = Path.unlink,
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
) -> PreparationSummary:
    """Stream only Phase-2 queue entries through render, commit, and cleanup."""

    # The lifecycle intentionally processes one preset at a time. Keep this
    # compatibility argument for existing callers while Phase 5 owns any
    # future concurrency work.
    _ = render_processes

    database = Database(Path(db_path).expanduser().resolve())
    root = _ensure_analysis_root(analysis_root)
    jobs_root = root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    revision_token = prepared_revision_token(revision)
    recover_preparation_state(
        database,
        analysis_root=root,
        legacy_audio_root=legacy_audio_root,
        revision=revision,
        unlink_file=unlink_file,
    )
    queue = get_presets_needing_preparation(database, revision=revision)
    if preset_ids is not None:
        wanted = set(int(item) for item in preset_ids)
        queue = [preset for preset in queue if preset.id in wanted]
    summary = PreparationSummary(queued=len(queue))
    embedder: Any | None = None

    def report(stage: str, preset: PresetRecord, completed: int) -> None:
        if progress is None:
            return
        labels = {
            "render": "Rendering",
            "analyze": "Analyzing",
            "commit": "Saving",
            "cleanup": "Cleaning up",
            "complete": "Prepared",
            "failed": "Couldn't prepare",
        }
        progress(
            {
                "stage": "prepare",
                "current": completed,
                "total": len(queue),
                "current_stage": stage,
                "preset_id": preset.id,
                "preset_name": preset.name,
                "text": f"{labels[stage]} {preset.name}",
            }
        )

    for index, preset in enumerate(queue, start=1):
        if cancel_check():
            summary.cancelled = True
            break
        summary.attempted += 1
        temp_dir = jobs_root / str(preset.id)
        temp_dir.mkdir(parents=True, exist_ok=True)
        existing_job = _job_row(database, preset.id)
        _set_job(
            database,
            preset,
            state="pending",
            temp_dir=temp_dir,
            revision_token=revision_token,
            increment_attempt=True,
        )
        try:
            if _verified_source(database, preset) is None:
                raise RuntimeError(
                    "no active source still matches the expected content hash"
                )
            if free_space_provider(root) < minimum_free_bytes:
                raise OSError(
                    f"temporary analysis needs at least {minimum_free_bytes} free bytes"
                )

            can_resume_features = bool(
                existing_job is not None
                and str(existing_job["target_revision"]) == revision_token
                and str(existing_job["expected_content_hash"]) == preset.content_hash
                and str(existing_job["state"]) in {"analyzing", "committing"}
            )
            if can_resume_features:
                with database.connect() as connection:
                    complete = has_required_permanent_data(connection, preset.id)
                source_still_matches = _verified_source(database, preset) is not None
                if complete and source_still_matches and _commit_prepared(
                    database,
                    preset,
                    temp_dir=temp_dir,
                    revision=revision,
                    revision_token=revision_token,
                ):
                    summary.prepared += 1
                    stage_hook("after_prepared_commit", preset.id)
                    files, bytes_removed, cleanup_failed = _finish_cleanup(
                        database,
                        preset,
                        analysis_root=root,
                        legacy_audio_root=legacy_audio_root,
                        unlink_file=unlink_file,
                    )
                    summary.cleaned_files += files
                    summary.cleaned_bytes += bytes_removed
                    summary.cleanup_failures += int(cleanup_failed)
                    report("complete", preset, index)
                    continue

            with database.connect() as connection:
                old_revision = connection.execute(
                    "SELECT * FROM prepared_presets WHERE preset_id=?", (preset.id,)
                ).fetchone()
            if old_revision is not None:
                serum_revision = (
                    str(old_revision["serum1_schema_revision"])
                    if preset.synth == "serum1"
                    else str(old_revision["serum2_schema_revision"])
                )
                expected_serum_revision = (
                    revision.serum1 if preset.synth == "serum1" else revision.serum2
                )
                if serum_revision != expected_serum_revision:
                    raise RuntimeError(
                        "stored Serum data uses an incompatible schema and must be "
                        "re-extracted before preparation"
                    )

                current_job_in_progress = bool(
                    existing_job is not None
                    and str(existing_job["target_revision"]) == revision_token
                    and str(existing_job["expected_content_hash"])
                    == preset.content_hash
                    and str(existing_job["state"])
                    in {
                        "rendering",
                        "rendered",
                        "analyzing",
                        "committing",
                        "failed",
                        "cancelled",
                    }
                )
                if (
                    str(old_revision["render_revision"]) != revision.render
                    and not current_job_in_progress
                ):
                    _discard_incompatible_renders(
                        database,
                        preset.id,
                        allowed_roots=(
                            root,
                            Path(legacy_audio_root).expanduser().resolve(),
                        ),
                        unlink_file=unlink_file,
                    )

                feature_revision_changed = any(
                    (
                        str(old_revision["render_revision"]) != revision.render,
                        str(old_revision["fingerprint_revision"])
                        != revision.fingerprint,
                        str(old_revision["clap_revision"]) != revision.clap,
                        str(old_revision["handcrafted_revision"])
                        != revision.handcrafted,
                    )
                )
                # Remove old feature rows before analysis. This makes a partial
                # rewrite visibly incomplete to the authoritative validator.
                # The render rows remain available until the new prepared
                # revision has committed.
                if feature_revision_changed:
                    with database.connect() as connection:
                        connection.execute(
                            "DELETE FROM fingerprints WHERE preset_id=?", (preset.id,)
                        )

            if _verified_source(database, preset) is None:
                raise RuntimeError("source changed while preparation was running")

            allowed_roots = (root, Path(legacy_audio_root).expanduser().resolve())
            valid = _reconcile_render_rows(
                database, preset.id, allowed_roots=allowed_roots
            )
            summary.reused_render_notes += len(valid)
            if set(valid) != set(MIDI_NOTES):
                if not allow_render:
                    raise RuntimeError("complete valid renders are not available")
                _set_job(
                    database,
                    preset,
                    state="rendering",
                    temp_dir=temp_dir,
                    revision_token=revision_token,
                )
                report("render", preset, index - 1)
                stage_hook("before_render", preset.id)
                rendered = render_function(
                    db_path=database.path,
                    audio_root=jobs_root,
                    state_dir=state_dir,
                    preset_ids=[preset.id],
                    processes=1,
                    log=log,
                    progress=None,
                )
                summary.rendered_notes += int(rendered.rendered_note_pairs)
                summary.peak_temp_bytes = max(summary.peak_temp_bytes, _tree_size(root))
                stage_hook("after_render", preset.id)
                if rendered.cancelled:
                    _set_job(
                        database,
                        preset,
                        state="cancelled",
                        temp_dir=temp_dir,
                        revision_token=revision_token,
                    )
                    summary.cancelled = True
                    break
                valid = _reconcile_render_rows(
                    database, preset.id, allowed_roots=allowed_roots
                )
                if set(valid) != set(MIDI_NOTES):
                    raise RuntimeError("rendering did not produce seven valid note files")

            _set_job(
                database,
                preset,
                state="rendered",
                temp_dir=temp_dir,
                revision_token=revision_token,
            )
            report("analyze", preset, index - 1)
            if cancel_check():
                _set_job(
                    database,
                    preset,
                    state="cancelled",
                    temp_dir=temp_dir,
                    revision_token=revision_token,
                )
                summary.cancelled = True
                break
            _set_job(
                database,
                preset,
                state="analyzing",
                temp_dir=temp_dir,
                revision_token=revision_token,
            )
            stage_hook("before_fingerprint", preset.id)
            if embedder is None:
                embedder = embedder_factory(env)
            if not fingerprint_function(database, embedder, preset.id):
                raise RuntimeError(
                    "fingerprinting did not produce the complete feature contract"
                )
            summary.fingerprints_created += 1
            summary.peak_temp_bytes = max(summary.peak_temp_bytes, _tree_size(root))
            stage_hook("after_features", preset.id)
            if _verified_source(database, preset) is None:
                raise RuntimeError("source changed while preparation was running")
            report("commit", preset, index - 1)
            if not _commit_prepared(
                database,
                preset,
                temp_dir=temp_dir,
                revision=revision,
                revision_token=revision_token,
            ):
                raise RuntimeError("authoritative prepared-state validation failed")
            summary.prepared += 1
            stage_hook("after_prepared_commit", preset.id)
            report("cleanup", preset, index - 1)
            files, bytes_removed, cleanup_failed = _finish_cleanup(
                database,
                preset,
                analysis_root=root,
                legacy_audio_root=legacy_audio_root,
                unlink_file=unlink_file,
            )
            summary.cleaned_files += files
            summary.cleaned_bytes += bytes_removed
            summary.cleanup_failures += int(cleanup_failed)
            stage_hook("after_cleanup", preset.id)
            report("complete", preset, index)
        except Exception as exc:
            summary.failed += 1
            _set_job(
                database,
                preset,
                state="failed",
                temp_dir=temp_dir,
                revision_token=revision_token,
                error=f"{type(exc).__name__}: {exc}",
            )
            log(f"Preparation failed for {preset.name}: {exc}")
            report("failed", preset, index)
    return summary
