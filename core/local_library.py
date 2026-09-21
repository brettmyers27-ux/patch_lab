"""Consent-gated local processing and contribution orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from core.contribution_bundle import LEDGER_NAME
from core.db import Database
from core.factory_bundle import DEFAULT_FACTORY_BUNDLE, FactoryBundle
from core.features import ClapEmbedder, handcrafted_features, load_audio_48k_mono
from core.platform_env import ENV, PlatformEnv
from core.plugin_host import ParameterValue
from core.preset_scan import (
    SequentialSerum1Ingestor,
    SilentPresetError,
    discover_presets,
    sha1_file,
    synth_for,
)
from core.render import MIDI_NOTES, render_library, summary_dict
from core.serum2_preset import parse_serum2_preset
from core.serum2_state_reconstruct import decode_host_template, reconstruct_vstpreset
from core.storage import (
    compact_render_library,
    configured_audio_root,
    load_storage_preferences,
    preview_cache_root,
)


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict[str, Any]], None]

AUTO_SCAN_MARKER_FILENAME = "last-auto-link-scan.json"
DEFAULT_AUTO_SCAN_INTERVAL_HOURS = 24.0


def _auto_scan_marker_path(env: PlatformEnv = ENV) -> Path:
    return Path(env.app_data_dir) / AUTO_SCAN_MARKER_FILENAME


def auto_scan_due(
    env: PlatformEnv = ENV,
    *,
    min_interval_hours: float = DEFAULT_AUTO_SCAN_INTERVAL_HOURS,
) -> bool:
    """True once enough time has passed since the last automatic linked-folder scan.

    A damaged or missing marker means due -- silently never checking again
    because of one corrupt file would be worse than an extra scan.
    """

    path = _auto_scan_marker_path(env)
    if not path.is_file():
        return True
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        last = datetime.fromisoformat(str(raw["last_scan_at"]))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - last >= timedelta(hours=min_interval_hours)


def record_auto_scan(env: PlatformEnv = ENV, *, at: datetime | None = None) -> None:
    """Mark that an automatic linked-folder scan is starting now.

    Recorded before the scan runs, not after it finishes, so an interrupted
    or slow scan can't cause a second one to fire the next time the app
    happens to launch within the same day.
    """

    path = _auto_scan_marker_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = at or datetime.now(timezone.utc)
    path.write_text(
        json.dumps({"last_scan_at": timestamp.isoformat()}), encoding="utf-8"
    )


class RelayProtocol(Protocol):
    def post_submission(
        self,
        *,
        kind: str,
        submission_id: str,
        path: Path,
        sha256: str,
        version: str,
        timeout: float,
        on_body: Any = None,
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class LocalLibrarySummary:
    found: int = 0
    deduped_local: int = 0
    params_dumped: int = 0
    failed_load: int = 0
    failed_silent: int = 0
    factory_skipped_upload: int = 0
    fingerprints_created: int = 0
    searchable_local: int = 0
    relay_already_present: int = 0
    relay_uploaded: int = 0
    relay_upload_failed: int = 0
    relay_disabled: int = 0
    relay_disabled_after_failures: int = 0
    audio_bytes_uploaded: int = 0
    compacted_render_files: int = 0
    compacted_render_bytes: int = 0
    #: Presets that were skipped because this machine has no renderer for their
    #: Serum generation.  A mixed library processes the subsets it can and
    #: reports the rest, rather than aborting everything.
    skipped_unsupported_generation: int = 0
    unsupported_generations: str = ""


def default_local_paths(env: PlatformEnv = ENV) -> dict[str, Path]:
    base = env.app_data_dir
    return {
        "db": base / "library.db",
        "audio": configured_audio_root(env),
        "preview_root": preview_cache_root(env),
        "states": base / "serum2-render-states",
        "matches": base / "match_library",
    }


def _store_serum2(
    database: Database,
    preset_id: int,
    path: Path,
    template: Any,
    state_dir: Path,
) -> None:
    decoded = parse_serum2_preset(path)
    metadata_json = json.dumps(decoded.metadata, separators=(",", ":"), ensure_ascii=False)
    settings_json = json.dumps(decoded.data, separators=(",", ":"), ensure_ascii=False)
    database.replace_serum2_full_settings(
        preset_id,
        metadata_json=metadata_json,
        settings_json=settings_json,
        settings_sha256=hashlib.sha256(settings_json.encode("utf-8")).hexdigest(),
        payload_version=decoded.payload_version,
        cbor_length=decoded.cbor_length,
        compressed_length=decoded.compressed_length,
    )
    state, _partition = reconstruct_vstpreset(decoded, template)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{preset_id}.vstpreset").write_bytes(state)
    database.replace_params(
        preset_id,
        [ParameterValue(0, "Serum 2 complete settings", 1.0, "available")],
        "VST3/S2-partitioned-state-local-v1",
    )


def _fingerprint_preset(
    database: Database,
    embedder: ClapEmbedder,
    audio_root: Path,
    preset_id: int,
) -> bool:
    """Embed and store fingerprints for one already-rendered preset.

    Shared by the inline linked-folder pipeline and fingerprint_pending_presets()
    so both ever compute a preset's fingerprint the same way. Returns False (no
    database write) if none of the expected notes are on disk yet.
    """

    prepared_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for note in MIDI_NOTES:
        wav = Path(audio_root) / str(preset_id) / f"{note}.wav"
        if not wav.is_file():
            continue
        prepared = load_audio_48k_mono(wav)
        prepared_rows.append(
            (note, prepared.waveform, handcrafted_features(prepared.waveform))
        )
    embeddings = (
        embedder.embed([row[1] for row in prepared_rows])
        if prepared_rows
        else np.empty((0, 512), dtype=np.float32)
    )
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
    if not rows:
        return False
    mean_embedding = np.mean([row[1] for row in rows], axis=0)
    mean_embedding /= max(float(np.linalg.norm(mean_embedding)), 1e-12)
    mean_handcrafted = np.mean([row[2] for row in rows], axis=0)
    database.upsert_fingerprint(
        preset_id,
        0,
        np.ascontiguousarray(mean_embedding, dtype=np.float32).tobytes(),
        np.ascontiguousarray(mean_handcrafted, dtype=np.float32).tobytes(),
    )
    return True


def fingerprint_pending_presets(
    db_path: Path,
    audio_root: Path,
    *,
    env: PlatformEnv = ENV,
    log: LogCallback = print,
    progress: ProgressCallback | None = None,
    compact_mode: bool | None = None,
) -> LocalLibrarySummary:
    """Fingerprint every fully-rendered preset that has no fingerprint yet.

    process_linked_folder() only fingerprints as an inline step of its own
    render batches, so audio rendered any other way -- a standalone
    render-library run, a recovered legacy library -- never gets fingerprinted
    on its own. This is the catch-up path: safe to run any time, independent
    of how the audio came to exist on disk. Never retrains any model; it only
    runs the shipped CLAP encoder over audio that already exists.
    """

    database = Database(Path(db_path).expanduser().resolve())
    if compact_mode is None:
        compact_mode = load_storage_preferences(env).compact_mode
    with database.connect() as connection:
        already_fingerprinted = {
            int(row[0])
            for row in connection.execute(
                "SELECT preset_id FROM fingerprints WHERE midi_note=0"
            ).fetchall()
        }
        rows = connection.execute(
            "SELECT id,name FROM presets WHERE status IN ('rendered','embedded')"
        ).fetchall()
    targets = [
        (int(row["id"]), str(row["name"]))
        for row in rows
        if int(row["id"]) not in already_fingerprinted
    ]
    summary = LocalLibrarySummary()
    total = len(targets)
    if total == 0:
        log("LOCAL_LIBRARY_SUMMARY=" + json.dumps(asdict(summary), sort_keys=True))
        return summary

    embedder = ClapEmbedder(env)
    completed = 0
    batch_size = 24 if compact_mode else total
    for batch_start in range(0, total, batch_size):
        for preset_id, name in targets[batch_start : batch_start + batch_size]:
            if _fingerprint_preset(database, embedder, audio_root, preset_id):
                summary.fingerprints_created += 1
                log(f"Local fingerprint ready: {name}")
            completed += 1
            if progress is not None:
                progress(
                    {
                        "stage": "analyze",
                        "current": completed,
                        "total": total,
                        "text": f"Learning {completed:,} of {total:,} rendered presets",
                    }
                )
        if compact_mode:
            compacted = compact_render_library(db_path, audio_root, log=log)
            summary.compacted_render_files += compacted.files
            summary.compacted_render_bytes += compacted.bytes
            if compacted.files:
                log(
                    "Compact storage retained fingerprints and removed "
                    f"{compacted.files:,} regenerable WAV files "
                    f"({compacted.bytes / (1024 ** 3):.2f} GiB)."
                )
    log("LOCAL_LIBRARY_SUMMARY=" + json.dumps(asdict(summary), sort_keys=True))
    return summary


def _fingerprint_payload(database: Database, preset_id: int) -> dict[str, Any]:
    with database.connect() as connection:
        preset = connection.execute(
            "SELECT * FROM presets WHERE id=?", (preset_id,)
        ).fetchone()
        features = connection.execute(
            "SELECT midi_note,embedding_f32,handcrafted_f32 FROM fingerprints "
            "WHERE preset_id=? ORDER BY midi_note",
            (preset_id,),
        ).fetchall()
        params = connection.execute(
            "SELECT param_index,param_name,norm_value,display_value FROM params "
            "WHERE preset_id=? ORDER BY param_index",
            (preset_id,),
        ).fetchall()
        settings = connection.execute(
            "SELECT metadata_json,settings_json,payload_version FROM serum2_full_settings "
            "WHERE preset_id=?",
            (preset_id,),
        ).fetchone()
    payload: dict[str, Any] = {
        "schema": 1,
        "content_hash": str(preset["content_hash"]),
        "name": str(preset["name"]),
        "synth": str(preset["synth"]),
        "embeddings": {
            str(int(row["midi_note"])): np.frombuffer(
                row["embedding_f32"], dtype=np.float32
            ).tolist()
            for row in features
        },
        "handcrafted": {
            str(int(row["midi_note"])): np.frombuffer(
                row["handcrafted_f32"], dtype=np.float32
            ).tolist()
            for row in features
        },
        "params": [dict(row) for row in params],
    }
    if settings is not None:
        payload["serum2"] = {
            "metadata": json.loads(settings["metadata_json"]),
            "settings": json.loads(settings["settings_json"]),
            "payload_version": int(settings["payload_version"]),
        }
    return payload


@dataclass(slots=True)
class PendingProcessSummary:
    """Outcome of processing already-catalogued pending presets."""

    generation: str = ""
    considered: int = 0
    missing_on_disk: int = 0
    params_dumped: int = 0
    failed_load: int = 0
    failed_silent: int = 0
    fingerprints_created: int = 0
    still_pending: int = 0
    compacted_render_files: int = 0
    compacted_render_bytes: int = 0


def process_pending_for_generation(
    generation: str,
    *,
    db_path: Path,
    audio_root: Path,
    state_dir: Path,
    env: PlatformEnv = ENV,
    log: LogCallback = print,
    progress: ProgressCallback | None = None,
    render_processes: int = 4,
    compact_mode: bool | None = None,
    operation_id: str = "",
    limit: int | None = None,
) -> PendingProcessSummary:
    """Process presets already catalogued as pending for ``generation``.

    This is the whole point of persisting pending state: PatchLab already knows
    these files exist, so installing a missing Serum later must not cost another
    full filesystem walk and re-hash of a multi-thousand-file library. Nothing
    here calls ``discover_presets``.

    Only presets whose recorded compatible-renderer set includes ``generation``
    are touched; a Serum 2 install never disturbs pending Serum 1 work.
    """

    from core.diagnostics import new_operation_id, record, record_decision
    from core.preset_identity import PENDING_RENDER_FAILED, identify_preset
    from core.synth_capability import capability_for, display_name

    operation_id = operation_id or new_operation_id("pending")
    generation = str(generation)
    summary = PendingProcessSummary(generation=generation)
    if compact_mode is None:
        compact_mode = load_storage_preferences(env).compact_mode
    database = Database(db_path)

    capability = capability_for(generation, env=env, operation_id=operation_id)
    if not capability.available:
        record(
            "local-library",
            "pending_processing_refused",
            f"cannot process pending {generation} presets: {capability.status}",
            severity="warning",
            operation_id=operation_id,
            phase="renderer-preflight",
            decision_reason=capability.reason,
            generation=generation,
        )
        raise RuntimeError(capability.user_message())

    candidates = database.presets_needing_generation(generation)
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]
    summary.considered = len(candidates)
    record_decision(
        "local-library",
        "process_pending_presets",
        outcome=f"{len(candidates)} pending {generation} preset(s) selected",
        reason=(
            f"{display_name(generation)} became usable, and these presets were "
            "already catalogued with a pending reason, so no rediscovery is needed"
        ),
        operation_id=operation_id,
        phase="batch-preparation",
        generation=generation,
        renderer=capability.preferred_renderer,
    )
    if not candidates:
        log(f"No pending {display_name(generation)} presets to process.")
        return summary

    # ---- ingest ---------------------------------------------------------
    ingestor = None
    template = None
    ready_ids: list[int] = []
    for index, record_row in enumerate(candidates, start=1):
        path = Path(record_row.path)
        if not path.is_file():
            # The file moved or was deleted. Keep the row pending rather than
            # inventing a failure, and never delete the user's catalog entry.
            summary.missing_on_disk += 1
            continue
        identity = identify_preset(path, env=env)
        try:
            if generation == "serum1":
                if ingestor is None:
                    ingestor = SequentialSerum1Ingestor(env, operation_id=operation_id)
                parameters, _rms, strategy = ingestor.ingest(path)
                database.replace_params(record_row.id, parameters, strategy)
            else:
                if template is None:
                    from pedalboard import load_plugin

                    from core.renderer_selection import require_renderer

                    selection = require_renderer(
                        generation,
                        env=env,
                        operation_id=operation_id,
                        phase="batch-preparation",
                        context="reading the Serum 2 host template",
                    )
                    assert selection.selected is not None
                    live = load_plugin(
                        str(selection.selected.path), plugin_name="Serum 2"
                    )
                    template = decode_host_template(bytes(live.preset_data))
                _store_serum2(
                    database, record_row.id, path, template, Path(state_dir)
                )
            database.set_pending_reason(record_row.id, None)
            summary.params_dumped += 1
            ready_ids.append(record_row.id)
        except SilentPresetError as exc:
            database.mark_failed(record_row.id, "failed_silent", str(exc))
            database.set_pending_reason(record_row.id, None)
            summary.failed_silent += 1
        except Exception as exc:
            # A processing failure keeps the preset pending with an accurate
            # reason instead of silently losing it.
            database.mark_failed(record_row.id, "failed_load", repr(exc))
            database.set_pending_reason(
                record_row.id, PENDING_RENDER_FAILED, error=repr(exc)
            )
            summary.failed_load += 1
            log(f"Pending preset failed: {path.name}: {exc}")
        if progress is not None:
            progress(
                {
                    "stage": "scan",
                    "current": index,
                    "total": len(candidates),
                    "text": f"Preparing {index:,} of {len(candidates):,} pending presets",
                }
            )

    # ---- render, learn, durably commit, then clean up -------------------
    embedder = ClapEmbedder(env) if ready_ids else None
    total_notes = len(ready_ids) * len(MIDI_NOTES)
    completed_notes = 0
    batch_size = 24 if compact_mode else max(len(ready_ids), 1)
    for start in range(0, len(ready_ids), batch_size):
        batch = ready_ids[start : start + batch_size]

        def render_progress(detail: dict[str, Any]) -> None:
            if progress is None:
                return
            current = completed_notes + int(detail.get("completed_note_pairs", 0))
            progress(
                {
                    **detail,
                    "stage": "render",
                    "current": current,
                    "total": total_notes,
                    "text": f"Rendering {current:,} of {total_notes:,} notes",
                }
            )

        render_summary = render_library(
            db_path=db_path,
            audio_root=audio_root,
            state_dir=state_dir,
            preset_ids=batch,
            processes=render_processes,
            log=log,
            progress=render_progress,
        )
        log(
            "Pending render batch summary: "
            + json.dumps(summary_dict(render_summary), sort_keys=True)
        )
        completed_notes += len(batch) * len(MIDI_NOTES)
        with database.connect() as connection:
            audible = {
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM presets WHERE status='rendered'"
                ).fetchall()
            }
        for preset_id in batch:
            if preset_id not in audible or embedder is None:
                continue
            if _fingerprint_preset(database, embedder, audio_root, preset_id):
                summary.fingerprints_created += 1
            if progress is not None:
                progress(
                    {
                        "stage": "analyze",
                        "current": summary.fingerprints_created,
                        "total": len(ready_ids),
                        "text": (
                            f"Learning {summary.fingerprints_created:,} of "
                            f"{len(ready_ids):,} presets"
                        ),
                    }
                )
        # Only after the learned state is durably committed above.
        if compact_mode:
            compacted = compact_render_library(db_path, audio_root, log=log)
            summary.compacted_render_files += compacted.files
            summary.compacted_render_bytes += compacted.bytes

    summary.still_pending = len(database.presets_needing_generation(generation))
    record(
        "local-library",
        "pending_processing_complete",
        f"processed {summary.params_dumped} pending {generation} preset(s); "
        f"{summary.still_pending} still pending",
        operation_id=operation_id,
        phase="complete",
        generation=generation,
        **{
            key: getattr(summary, key)
            for key in (
                "considered",
                "missing_on_disk",
                "params_dumped",
                "failed_load",
                "failed_silent",
                "fingerprints_created",
                "still_pending",
            )
        },
    )
    return summary


def refresh_pending_reasons(
    *,
    db_path: Path,
    env: PlatformEnv = ENV,
    operation_id: str = "",
) -> dict[str, int]:
    """Re-evaluate every pending preset against current synth capability.

    Cheap: one capability refresh plus one indexed query and one bulk update.
    No filesystem walk, no hashing. This is what turns "Serum 2 is now
    installed" into "these 327 presets are now processable".
    """

    from core.diagnostics import new_operation_id, record
    from core.preset_identity import identify_preset
    from core.synth_capability import refresh_capabilities

    from core.preset_identity import PENDING_AWAITING_PROCESSING

    operation_id = operation_id or new_operation_id("capability")
    database = Database(db_path)
    capability = refresh_capabilities(
        env=env, operation_id=operation_id, reason="re-evaluating pending presets"
    )
    pending = database.pending_presets()
    updates: dict[int, str | None] = {}
    for row in pending:
        identity = identify_preset(Path(row.path), env=env)
        reason = identity.pending_reason(capability.capabilities)
        if reason is None:
            # Nothing is missing any more, but the preset still has not been
            # processed. Clearing the reason outright would lose that fact and
            # make the row invisible to process_pending_for_generation -- the
            # preset would silently never be picked up. Mark it ready instead,
            # and only clear it once the work actually completes.
            reason = (
                None
                if row.status not in {"scanned", "failed_load"}
                else PENDING_AWAITING_PROCESSING
            )
        updates[row.id] = reason
    database.set_pending_reasons(updates)
    cleared = sum(
        1
        for value in updates.values()
        if value is None or value == PENDING_AWAITING_PROCESSING
    )
    counts = database.pending_counts()
    record(
        "local-library",
        "pending_reasons_refreshed",
        f"{cleared} preset(s) became processable; {sum(counts.values())} still pending",
        operation_id=operation_id,
        phase="capability-refresh",
        decision_reason=(
            "pending reasons are derived from current capability, so a newly "
            "installed engine clears them without rediscovering any file"
        ),
        became_processable=cleared,
        pending_by_reason=counts,
    )
    return counts


def _coverage_log_lines(coverage: Mapping[str, Any]) -> list[str]:
    """Plain-language coverage lines for the visible application log."""

    from core.preset_identity import PENDING_REASON_LABELS

    lines = [
        f"Discovered {coverage['discovered']:,} preset(s): "
        + ", ".join(
            f"{count:,} {name}"
            for name, count in sorted(coverage.get("by_file_format", {}).items())
        )
    ]
    legacy = int(coverage.get("legacy_fxp_in_serum2_library", 0) or 0)
    if legacy:
        lines.append(
            f"{legacy:,} of those are legacy .fxp presets shipped inside a Serum 2 "
            "library; those need Serum 1 to render."
        )
    pending = coverage.get("pending_by_reason") or {}
    for reason, count in sorted(pending.items()):
        label = PENDING_REASON_LABELS.get(reason, reason)
        lines.append(f"{count:,} preset(s) pending because {label}.")
    if coverage.get("processable"):
        lines.append(f"{int(coverage['processable']):,} preset(s) can be processed now.")
    return lines


def process_linked_folder(
    root: Path,
    *,
    db_path: Path,
    audio_root: Path,
    state_dir: Path,
    bundle_path: Path = DEFAULT_FACTORY_BUNDLE,
    env: PlatformEnv = ENV,
    relay: RelayProtocol | None = None,
    log: LogCallback = print,
    progress: ProgressCallback | None = None,
    render_processes: int = 4,
    compact_mode: bool | None = None,
    operation_id: str = "",
    upload_sleep: Callable[[float], None] = time.sleep,
) -> LocalLibrarySummary:
    """Always process locally first, then share new presets in a few bundles."""

    from core.diagnostics import new_operation_id, record, record_decision
    from core.preset_identity import identify_preset, summarise_identities
    from core.synth_capability import refresh_capabilities

    operation_id = operation_id or new_operation_id("library")

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    database = Database(db_path)
    if compact_mode is None:
        compact_mode = load_storage_preferences(env).compact_mode
    known_factory = FactoryBundle(bundle_path).known_hashes()
    paths = discover_presets(root)
    summary = LocalLibrarySummary(found=len(paths))

    # ---------------------------------------------------------------------
    # IDENTIFY AND PREFLIGHT BEFORE THE EXPENSIVE SCAN.
    #
    # The reported failure spent ~36 minutes hashing 5,634 files and writing
    # database rows, then raised "Verified Serum 1 VST2 binary is unavailable."
    # from the ingestor constructor.  That requirement was deterministic and
    # knowable the moment the file list existed, so it is resolved here.
    #
    # `discover_presets` has already walked the tree, so identifying its result
    # costs a suffix comparison plus a prefix test per path and adds no second
    # filesystem scan.
    # ---------------------------------------------------------------------
    identities = {path: identify_preset(path, env=env) for path in paths}
    capability = refresh_capabilities(
        env=env, operation_id=operation_id, reason="linked-folder processing preflight"
    )
    coverage = summarise_identities(
        list(identities.values()), capability.capabilities
    )
    record_decision(
        "local-library",
        "library_coverage",
        outcome=(
            f"{coverage.get('processable', 0)} processable, "
            f"{coverage.get('pending', 0)} pending of {coverage['discovered']} discovered"
        ),
        reason=(
            "renderer compatibility is a property of the container format and the "
            "installed engines, not of the folder the preset lives in; a Serum 2 "
            "factory library legitimately contains legacy .fxp content that only "
            "Serum 1 can render headlessly"
        ),
        operation_id=operation_id,
        phase="preset-classification",
        linked_folder=str(root),
        **coverage,
    )
    for line in _coverage_log_lines(coverage):
        log(line)

    # Every discovered preset is catalogued, including ones this machine cannot
    # render. Dropping them would mean rediscovering the whole library after the
    # user installs the missing Serum.
    processable_paths = [
        path
        for path, identity in identities.items()
        if identity.pending_reason(capability.capabilities) is None
    ]
    pending_paths = [path for path in paths if path not in set(processable_paths)]
    summary.skipped_unsupported_generation = len(pending_paths)
    summary.unsupported_generations = ",".join(
        sorted(
            {
                str(identities[path].origin_generation)
                for path in pending_paths
                if identities[path].origin_generation
            }
        )
    )
    if not processable_paths and pending_paths:
        # Nothing can be processed right now. This is not a crash: the catalog
        # below still records every preset so a later install can pick them up.
        record(
            "local-library",
            "preflight_no_processable_presets",
            "no installed Serum engine can process any of the linked presets yet",
            severity="warning",
            operation_id=operation_id,
            phase="renderer-preflight",
            decision_reason=(
                "presets are catalogued as pending rather than failed, so "
                "installing the missing Serum later needs no rediscovery"
            ),
            **coverage,
        )

    id_to_path: dict[int, Path] = {}
    new_or_pending: list[int] = []
    pending_updates: dict[int, str | None] = {}
    if progress is not None:
        progress(
            {
                "stage": "scan",
                "current": 0,
                "total": len(paths),
                "text": f"Scanning 0 of {len(paths):,} presets",
            }
        )
    for discovered_index, path in enumerate(paths, start=1):
        identity = identities[path]
        synth = identity.origin_generation
        assert synth is not None
        digest = sha1_file(path)
        preset_id, inserted = database.insert_preset(
            path=path, name=path.stem, synth=synth, content_hash=digest
        )
        if not inserted:
            summary.deduped_local += 1
        database.set_factory_status(preset_id, digest in known_factory)
        database.record_identity(
            preset_id,
            file_format=identity.file_format,
            provenance=identity.provenance,
            compatible_renderers=identity.compatible_renderers,
        )
        pending_updates[preset_id] = identity.pending_reason(capability.capabilities)
        id_to_path[preset_id] = path
        with database.connect() as connection:
            status = str(
                connection.execute(
                    "SELECT status FROM presets WHERE id=?", (preset_id,)
                ).fetchone()[0]
            )
        if status in {"scanned", "failed_load"} and pending_updates[preset_id] is None:
            new_or_pending.append(preset_id)
        if progress is not None:
            progress(
                {
                    "stage": "scan",
                    "current": discovered_index,
                    "total": len(paths),
                    "text": (
                        f"Scanning {discovered_index:,} of {len(paths):,} presets"
                    ),
                }
            )
    # One transaction records every pending reason, so a 5,000-preset library
    # costs one write rather than 5,000.
    database.set_pending_reasons(pending_updates)
    pending_counts = database.pending_counts()
    if pending_counts:
        record(
            "local-library",
            "presets_pending",
            "pending presets by reason: "
            + ", ".join(f"{count} {reason}" for reason, count in sorted(pending_counts.items())),
            severity="info",
            operation_id=operation_id,
            phase="preset-classification",
            decision_reason=(
                "these presets are discovered and catalogued but not processable on "
                "this machine; they are retained so a later Serum install can "
                "process them without rediscovery"
            ),
            pending_by_reason=pending_counts,
        )
    log(f"Local catalog: {len(paths)} files; {summary.deduped_local} already known locally")

    # Route by the identity's compatible renderer, not by extension alone.
    serum1_ids = [
        preset_id
        for preset_id in new_or_pending
        if "serum1" in identities[id_to_path[preset_id]].compatible_renderers
    ]
    serum2_ids = [
        preset_id
        for preset_id in new_or_pending
        if "serum2" in identities[id_to_path[preset_id]].compatible_renderers
    ]
    if serum1_ids:
        # Renderer availability was already established by the preflight above,
        # so this can no longer be the first place a missing Serum 1 surfaces --
        # and it now honours PatchLab's real verified format hierarchy rather
        # than demanding VST2 specifically.
        ingestor = SequentialSerum1Ingestor(env, operation_id=operation_id)
        record(
            "local-library",
            "ingestor_ready",
            f"Serum 1 ingestion via {ingestor.candidate.format}",
            operation_id=operation_id,
            phase="preset-classification",
            decision_reason=ingestor.selection.reason,
            renderer=f"serum1/{ingestor.candidate.format}",
            strategy=ingestor.strategy_label,
            pending_presets=len(serum1_ids),
        )
        for ingest_index, preset_id in enumerate(serum1_ids, start=1):
            try:
                parameters, rms, strategy = ingestor.ingest(id_to_path[preset_id])
                database.replace_params(preset_id, parameters, strategy)
                summary.params_dumped += 1
                log(f"Local params ready: {id_to_path[preset_id].name} ({rms:.1f} dBFS)")
            except SilentPresetError as exc:
                database.mark_failed(preset_id, "failed_silent", str(exc))
                summary.failed_silent += 1
            except Exception as exc:
                database.mark_failed(preset_id, "failed_load", repr(exc))
                summary.failed_load += 1
                log(f"Local load failed: {id_to_path[preset_id].name}: {exc}")
            if progress is not None:
                progress(
                    {
                        "stage": "scan",
                        "current": ingest_index,
                        "total": len(serum1_ids) + len(serum2_ids),
                        "text": (
                            f"Preparing {ingest_index:,} of "
                            f"{len(serum1_ids) + len(serum2_ids):,} presets"
                        ),
                    }
                )

    if serum2_ids:
        from pedalboard import load_plugin

        from core.renderer_selection import require_renderer

        # Same defect class as the matcher's line 285: a bare `next()` over a
        # hardcoded format raised StopIteration on a machine without that exact
        # binary.  Selection is now explicit, ordered and explained.
        serum2_selection = require_renderer(
            "serum2",
            env=env,
            operation_id=operation_id,
            phase="preset-classification",
            context="reading the Serum 2 host template",
        )
        assert serum2_selection.selected is not None
        record(
            "local-library",
            "ingestor_ready",
            f"Serum 2 ingestion via {serum2_selection.plugin_format}",
            operation_id=operation_id,
            phase="preset-classification",
            decision_reason=serum2_selection.reason,
            renderer=serum2_selection.renderer,
            pending_presets=len(serum2_ids),
        )
        live = load_plugin(str(serum2_selection.selected.path), plugin_name="Serum 2")
        template = decode_host_template(bytes(live.preset_data))
        for serum2_index, preset_id in enumerate(serum2_ids, start=1):
            try:
                _store_serum2(
                    database, preset_id, id_to_path[preset_id], template, Path(state_dir)
                )
                summary.params_dumped += 1
                log(f"Local Serum 2 state ready: {id_to_path[preset_id].name}")
            except Exception as exc:
                database.mark_failed(preset_id, "failed_load", repr(exc))
                summary.failed_load += 1
                log(f"Local Serum 2 load failed: {id_to_path[preset_id].name}: {exc}")
            if progress is not None:
                complete = len(serum1_ids) + serum2_index
                progress(
                    {
                        "stage": "scan",
                        "current": complete,
                        "total": len(serum1_ids) + len(serum2_ids),
                        "text": (
                            f"Preparing {complete:,} of "
                            f"{len(serum1_ids) + len(serum2_ids):,} presets"
                        ),
                    }
                )

    renderable = [
        record.id
        for record in database.renderable_presets()
        if record.id in id_to_path and record.status != "failed_silent"
    ]
    with database.connect() as connection:
        already_fingerprinted = {
            int(row[0])
            for row in connection.execute(
                "SELECT preset_id FROM fingerprints WHERE midi_note=0"
            ).fetchall()
        }
    # Compact mode treats fingerprints as the durable result. Re-scanning an
    # already learned preset must not recreate seven large WAV files merely to
    # delete them again. Turning compact mode off intentionally restores them.
    render_targets = (
        [preset_id for preset_id in renderable if preset_id not in already_fingerprinted]
        if compact_mode
        else renderable
    )
    feature_targets = {
        preset_id
        for preset_id in render_targets
        if preset_id not in already_fingerprinted
    }
    total_notes = len(render_targets) * len(MIDI_NOTES)
    total_features = len(feature_targets)
    completed_render_notes = 0
    completed_features = 0
    embedder = ClapEmbedder(env) if feature_targets else None

    def fingerprint_batch(preset_ids: list[int]) -> None:
        nonlocal completed_features
        for preset_id in preset_ids:
            if preset_id not in feature_targets:
                continue
            assert embedder is not None
            if _fingerprint_preset(database, embedder, audio_root, preset_id):
                summary.fingerprints_created += 1
                log(f"Local fingerprint ready: {id_to_path[preset_id].name}")
            completed_features += 1
            if progress is not None:
                progress(
                    {
                        "stage": "analyze",
                        "current": completed_features,
                        "total": total_features,
                        "text": (
                            f"Learning {completed_features:,} of "
                            f"{total_features:,} linked presets"
                        ),
                    }
                )

    # Compact mode deliberately limits the temporary render working set. Each
    # batch is rendered, embedded, and removed before the next one begins, so
    # a full library never needs tens of gigabytes of free local space.
    batch_size = 24 if compact_mode else max(len(render_targets), 1)
    for batch_start in range(0, len(render_targets), batch_size):
        batch = render_targets[batch_start : batch_start + batch_size]

        def render_progress(detail: dict[str, Any]) -> None:
            if progress is None:
                return
            current = completed_render_notes + int(
                detail.get("completed_note_pairs", 0)
            )
            progress(
                {
                    **detail,
                    "stage": "render",
                    "current": current,
                    "total": total_notes,
                    "text": f"Rendering {current:,} of {total_notes:,} notes",
                }
            )

        render_summary = render_library(
            db_path=db_path,
            audio_root=audio_root,
            state_dir=state_dir,
            preset_ids=batch,
            processes=render_processes,
            log=log,
            progress=render_progress,
        )
        log(
            "Local render batch summary: "
            + json.dumps(summary_dict(render_summary), sort_keys=True)
        )
        completed_render_notes += len(batch) * len(MIDI_NOTES)
        with database.connect() as connection:
            audible_ids = {
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM presets WHERE status='rendered'"
                ).fetchall()
            }
        fingerprint_batch([preset_id for preset_id in batch if preset_id in audible_ids])
        if compact_mode:
            compacted = compact_render_library(db_path, audio_root, log=log)
            summary.compacted_render_files += compacted.files
            summary.compacted_render_bytes += compacted.bytes
            if compacted.files:
                log(
                    "Compact storage retained fingerprints and removed "
                    f"{compacted.files:,} regenerable WAV files "
                    f"({compacted.bytes / (1024 ** 3):.2f} GiB)."
                )

    with database.connect() as connection:
        searchable_ids = {
            int(row[0])
            for row in connection.execute(
                "SELECT preset_id FROM fingerprints WHERE midi_note=0"
            ).fetchall()
        }
        presets = connection.execute(
            "SELECT id,content_hash,is_factory FROM presets "
            "WHERE status IN ('rendered','embedded') ORDER BY id"
        ).fetchall()
    eligible = [row for row in presets if int(row["id"]) in id_to_path and int(row["id"]) in searchable_ids]
    summary.searchable_local = len(eligible)
    _contribute_presets(
        database=database,
        relay=relay,
        root=root,
        rows=eligible,
        id_to_path=id_to_path,
        ledger_path=db_path.parent / LEDGER_NAME,
        summary=summary,
        log=log,
        sleep=upload_sleep,
    )
    if compact_mode:
        compacted = compact_render_library(db_path, audio_root, log=log)
        summary.compacted_render_files += compacted.files
        summary.compacted_render_bytes += compacted.bytes
        if compacted.files:
            log(
                "Compact storage retained fingerprints and removed "
                f"{compacted.files:,} regenerable WAV files "
                f"({compacted.bytes / (1024 ** 3):.2f} GiB)."
            )
    log("LOCAL_LIBRARY_SUMMARY=" + json.dumps(asdict(summary), sort_keys=True))
    return summary


def _contribute_presets(
    *,
    database: Database,
    relay: RelayProtocol | None,
    root: Path,
    rows: list,
    id_to_path: Mapping[int, Path],
    ledger_path: Path,
    summary: LocalLibrarySummary,
    log: LogCallback,
    sleep: Callable[[float], None],
) -> None:
    """Send presets this Mac has not contributed yet, a few large bundles at a time.

    Local processing is already finished, so nothing here can affect it.  The
    original preset files are only ever read.  A failed bundle is left for the
    next scan (same bytes, same id, so the service treats it as one submission).
    """

    from app.__version__ import __version__
    from core.contribution_bundle import (
        MAX_BUNDLES_PER_SCAN,
        Candidate,
        ContributionLedger,
        build_bundles,
    )
    from core.submission_upload import PRESET_CONTRIBUTION, UploadError, upload_file

    ledger = ContributionLedger(ledger_path)
    pending: list[Candidate] = []
    for row in rows:
        preset_id = int(row["id"])
        if bool(row["is_factory"]):
            summary.factory_skipped_upload += 1
            continue
        if relay is None:
            summary.relay_disabled += 1
            continue
        digest = str(row["content_hash"])
        if digest in ledger:
            summary.relay_already_present += 1
            continue
        path = id_to_path[preset_id]
        pending.append(Candidate(digest, path, path.relative_to(root).as_posix(), preset_id))
    if not pending or relay is None:
        return
    sent = 0
    with tempfile.TemporaryDirectory(prefix="patchlab-contribution-") as scratch:
        bundles = build_bundles(
            pending,
            fingerprint_for=lambda item: _fingerprint_payload(database, item.preset_id),
            work_dir=Path(scratch),
        )
        for bundle in bundles:
            if sent >= MAX_BUNDLES_PER_SCAN:
                log(
                    "More presets are waiting to be shared; they will be sent on a "
                    "later scan."
                )
                break
            try:
                result = upload_file(
                    relay,
                    kind=PRESET_CONTRIBUTION,
                    submission_id=bundle.submission_id,
                    path=bundle.path,
                    version=__version__,
                    sleep=sleep,
                )
            except (UploadError, OSError) as exc:
                summary.relay_upload_failed += len(bundle.content_hashes)
                summary.relay_disabled_after_failures += 1
                code = getattr(exc, "code", type(exc).__name__)
                log(
                    "Preset sharing skipped (will retry next scan): "
                    f"{len(bundle.content_hashes)} presets, {code}"
                )
                log(
                    "Sharing paused for the remainder of this scan; local preset "
                    "processing is unaffected."
                )
                break
            ledger.add_many(bundle.content_hashes)
            summary.relay_uploaded += len(bundle.content_hashes)
            sent += 1
            log(
                f"Shared {len(bundle.content_hashes)} presets in one upload "
                f"(receipt {result.receipt_id})."
            )
            bundle.discard()


def relay_from_environment() -> RelayProtocol | None:
    from core.submission_upload import resolve_relay

    return resolve_relay()[0]
