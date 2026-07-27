"""Consent-gated local processing and contribution orchestration."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from core.db import Database
from core.factory_bundle import DEFAULT_FACTORY_BUNDLE, FactoryBundle
from core.features import ClapEmbedder, handcrafted_features, load_audio_48k_mono
from core.platform_env import ENV, PlatformEnv
from core.plugin_host import ParameterValue
from core.preset_scan import (
    SequentialSerum1VST2,
    SilentPresetError,
    discover_presets,
    sha1_file,
    synth_for,
)
from core.render import MIDI_NOTES, render_library, summary_dict
from core.serum2_preset import parse_serum2_preset
from core.serum2_state_reconstruct import decode_host_template, reconstruct_vstpreset


LogCallback = Callable[[str], None]


class RelayProtocol(Protocol):
    def check_hash(self, content_hash: str) -> bool: ...

    def upload(
        self,
        *,
        preset_path: Path,
        relative_path: str,
        content_hash: str,
        fingerprint: dict[str, Any],
    ) -> Any: ...


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


def default_local_paths(env: PlatformEnv = ENV) -> dict[str, Path]:
    base = env.app_data_dir
    return {
        "db": base / "library.db",
        "audio": base / "audio",
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
    render_processes: int = 4,
) -> LocalLibrarySummary:
    """Always process locally first, then perform storage-only relay dedup."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    database = Database(db_path)
    known_factory = FactoryBundle(bundle_path).known_hashes()
    paths = discover_presets(root)
    summary = LocalLibrarySummary(found=len(paths))
    id_to_path: dict[int, Path] = {}
    new_or_pending: list[int] = []
    for path in paths:
        synth = synth_for(path)
        assert synth is not None
        digest = sha1_file(path)
        preset_id, inserted = database.insert_preset(
            path=path, name=path.stem, synth=synth, content_hash=digest
        )
        if not inserted:
            summary.deduped_local += 1
        database.set_factory_status(preset_id, digest in known_factory)
        id_to_path[preset_id] = path
        with database.connect() as connection:
            status = str(
                connection.execute(
                    "SELECT status FROM presets WHERE id=?", (preset_id,)
                ).fetchone()[0]
            )
        if status in {"scanned", "failed_load"}:
            new_or_pending.append(preset_id)
    log(f"Local catalog: {len(paths)} files; {summary.deduped_local} already known locally")

    serum1_ids = [
        preset_id
        for preset_id in new_or_pending
        if synth_for(id_to_path[preset_id]) == "serum1"
    ]
    if serum1_ids:
        ingestor = SequentialSerum1VST2(env)
        for preset_id in serum1_ids:
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

    serum2_ids = [
        preset_id
        for preset_id in new_or_pending
        if synth_for(id_to_path[preset_id]) == "serum2"
    ]
    if serum2_ids:
        from pedalboard import load_plugin

        candidate = next(
            item for item in env.plugins_for("serum2") if item.format == "VST3"
        )
        live = load_plugin(str(candidate.path), plugin_name="Serum 2")
        template = decode_host_template(bytes(live.preset_data))
        for preset_id in serum2_ids:
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

    renderable = [
        record.id
        for record in database.renderable_presets()
        if record.id in id_to_path
    ]
    render_summary = render_library(
        db_path=db_path,
        audio_root=audio_root,
        state_dir=state_dir,
        preset_ids=renderable,
        processes=render_processes,
        log=log,
    )
    log("Local render summary: " + json.dumps(summary_dict(render_summary), sort_keys=True))

    with database.connect() as connection:
        audible_ids = {
            int(row[0])
            for row in connection.execute(
                "SELECT id FROM presets WHERE status='rendered'"
            ).fetchall()
        }
        already = {
            int(row[0])
            for row in connection.execute(
                "SELECT preset_id FROM fingerprints WHERE midi_note=0"
            ).fetchall()
        }
    needs_features = [
        preset_id
        for preset_id in renderable
        if preset_id in audible_ids and preset_id not in already
    ]
    if needs_features:
        embedder = ClapEmbedder(env)
        for preset_id in needs_features:
            prepared_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
            for note in MIDI_NOTES:
                wav = Path(audio_root) / str(preset_id) / f"{note}.wav"
                if not wav.is_file():
                    continue
                prepared = load_audio_48k_mono(wav)
                handcrafted = handcrafted_features(prepared.waveform)
                prepared_rows.append((note, prepared.waveform, handcrafted))
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
            if rows:
                mean_embedding = np.mean([row[1] for row in rows], axis=0)
                mean_embedding /= max(float(np.linalg.norm(mean_embedding)), 1e-12)
                mean_handcrafted = np.mean([row[2] for row in rows], axis=0)
                database.upsert_fingerprint(
                    preset_id,
                    0,
                    np.ascontiguousarray(mean_embedding, dtype=np.float32).tobytes(),
                    np.ascontiguousarray(mean_handcrafted, dtype=np.float32).tobytes(),
                )
                summary.fingerprints_created += 1
                log(f"Local fingerprint ready: {id_to_path[preset_id].name}")

    with database.connect() as connection:
        searchable_ids = {
            int(row[0])
            for row in connection.execute(
                "SELECT preset_id FROM fingerprints WHERE midi_note=0"
            ).fetchall()
        }
        presets = connection.execute(
            "SELECT id,content_hash,is_factory FROM presets WHERE status='rendered' ORDER BY id"
        ).fetchall()
    eligible = [row for row in presets if int(row["id"]) in id_to_path and int(row["id"]) in searchable_ids]
    summary.searchable_local = len(eligible)
    consecutive_relay_failures = 0
    relay_disabled_by_failures = False
    relay_failure_limit = 3
    for row in eligible:
        preset_id = int(row["id"])
        if bool(row["is_factory"]):
            summary.factory_skipped_upload += 1
            continue
        if relay is None:
            if not relay_disabled_by_failures:
                summary.relay_disabled += 1
            continue
        digest = str(row["content_hash"])
        path = id_to_path[preset_id]
        try:
            if relay.check_hash(digest):
                summary.relay_already_present += 1
                consecutive_relay_failures = 0
                continue
            relay.upload(
                preset_path=path,
                relative_path=path.relative_to(root).as_posix(),
                content_hash=digest,
                fingerprint=_fingerprint_payload(database, preset_id),
            )
            summary.relay_uploaded += 1
            consecutive_relay_failures = 0
        except Exception as exc:
            summary.relay_upload_failed += 1
            consecutive_relay_failures += 1
            log(
                "Relay upload skipped (will retry next scan): "
                f"{path.name}: {exc}"
            )
            if consecutive_relay_failures >= relay_failure_limit:
                relay = None
                relay_disabled_by_failures = True
                summary.relay_disabled_after_failures += 1
                log(
                    "Relay disabled for the remainder of this scan after "
                    f"{consecutive_relay_failures} consecutive failures; "
                    "local preset processing will continue."
                )
            continue
    log("LOCAL_LIBRARY_SUMMARY=" + json.dumps(asdict(summary), sort_keys=True))
    return summary


def relay_from_environment() -> RelayProtocol | None:
    if os.environ.get("PATCHLAB_DISABLE_RELAY", "").strip() == "1":
        return None
    url = os.environ.get("PATCHLAB_RELAY_URL", "").strip()
    password = os.environ.get("PATCHLAB_RELAY_PASSWORD", "")
    token = None
    if url and not password:
        from core.access_gate import stored_relay_credential

        password, token = stored_relay_credential()
        password = password or ""
    if not url or (not password and not token):
        return None
    from core.relay_client import RelayClient

    return RelayClient(url, password, token=token)
