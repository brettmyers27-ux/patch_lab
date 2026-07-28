"""Instant matching against the shipped, preset-file-free factory bundle."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable

import librosa
import numpy as np

from core.audio_input import decode_audio_file
from core.branding import display_match_name, generated_preset_name
from core.factory_bundle import DEFAULT_FACTORY_BUNDLE, FactoryBundle
from core.features import CLAP_SAMPLE_RATE, ClapEmbedder
from core.match import cosine_topk
from core.matcher import detect_midi_note, loudness_normalize
from core.platform_env import ENV
from core.serum2_targets import encode_graph_with_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = PROJECT_ROOT / "data" / "matches"
SECTION_LABELS = {
    "oscillators": "Oscillators",
    "filters": "Filter",
    "envelopes": "Envelopes",
    "lfos": "LFOs",
    "fx": "FX",
    "mod_matrix": "Mod Matrix",
    "macros": "Macros",
    "global_other": "Global",
}


def _load_mapping(path: Path | None) -> dict[str, str]:
    if path is None or not Path(path).is_file():
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(key): str(value)
        for key, value in raw.get("local_paths_by_hash", {}).items()
    }


def _settings_summary(bundle: FactoryBundle, preset_id: int) -> dict[str, dict[str, Any]]:
    preset = bundle.preset_by_id(preset_id)
    _vector, mask = bundle.parameters(preset_id)
    if preset.synth == "serum1":
        settings, _metadata, _version = bundle.settings(preset_id)
        groups: dict[str, dict[str, Any]] = {}
        for item in settings["parameters"]:
            name = str(item["name"])
            lower = name.casefold()
            if "filter" in lower:
                section = "Filter"
            elif "env" in lower:
                section = "Envelopes"
            elif "lfo" in lower:
                section = "LFOs"
            elif "osc" in lower:
                section = "Oscillators"
            elif "mod " in lower:
                section = "Mod Matrix"
            elif any(token in lower for token in ("fx", "delay", "reverb", "dist", "chorus")):
                section = "FX"
            else:
                section = "Global"
            group = groups.setdefault(section, {"changed": [], "matches_base_count": 0})
            group["matches_base_count"] += 1
        return dict(sorted(groups.items()))
    schema = bundle.schema("serum2")
    groups: dict[str, dict[str, Any]] = {}
    for field in schema["fields"]:
        if not bool(mask[int(field["index"])]):
            continue
        section = SECTION_LABELS.get(str(field["category"]), "Global")
        group = groups.setdefault(section, {"changed": [], "matches_base_count": 0})
        group["matches_base_count"] += 1
    return dict(sorted(groups.items()))


def _local_search_rows(
    database_path: Path | None,
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    if database_path is None or not Path(database_path).is_file():
        return None, []
    database_path = Path(database_path).resolve()
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT p.id,p.content_hash,p.name,p.synth,p.path,f.embedding_f32
        FROM presets p JOIN fingerprints f ON f.preset_id=p.id
        WHERE f.midi_note=0 AND p.is_factory=0 AND p.status='rendered'
        ORDER BY p.id
        """
    ).fetchall()
    connection.close()
    if not rows:
        return None, []
    matrix = np.stack(
        [np.frombuffer(row["embedding_f32"], dtype=np.float32).copy() for row in rows]
    )
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    audio_root = database_path.parent / "audio"
    return matrix, [
        {
            "kind": "local",
            "preset_id": int(row["id"]),
            "content_hash": str(row["content_hash"]),
            "name": str(row["name"]),
            "synth": str(row["synth"]),
            "path": str(row["path"]),
            "audition_path": str(audio_root / str(int(row["id"])) / "60.wav"),
            "database_path": str(database_path),
        }
        for row in rows
    ]


def _local_parameters(
    bundle: FactoryBundle, item: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    connection = sqlite3.connect(item["database_path"])
    connection.row_factory = sqlite3.Row
    preset_id = int(item["preset_id"])
    if item["synth"] == "serum1":
        rows = connection.execute(
            "SELECT param_name,norm_value FROM params WHERE preset_id=? ORDER BY param_index",
            (preset_id,),
        ).fetchall()
        length = int(bundle.schema("serum1")["vector_length"])
        vector = np.zeros(length, dtype=np.float32)
        mask = np.zeros(length, dtype=np.bool_)
        for index, row in enumerate(rows[:length]):
            vector[index] = float(row["norm_value"])
            mask[index] = True
        settings = {
            "Local Serum 1": {"changed": [], "matches_base_count": int(mask.sum())}
        }
    else:
        row = connection.execute(
            "SELECT settings_json FROM serum2_full_settings WHERE preset_id=?",
            (preset_id,),
        ).fetchone()
        if row is None:
            connection.close()
            raise RuntimeError("Local Serum 2 fingerprint has no complete settings")
        schema = bundle.schema("serum2")
        vector, mask, coverage = encode_graph_with_schema(
            json.loads(row["settings_json"]), schema
        )
        settings = {
            "Local Serum 2": {
                "changed": [],
                "matches_base_count": int(coverage["matched_fields"]),
            }
        }
    connection.close()
    return vector, mask, settings


def run_factory_match_file(
    input_path: Path,
    *,
    target_synth: str = "serum2",
    start_offset_s: float = 0.0,
    bundle_path: Path = DEFAULT_FACTORY_BUNDLE,
    mapping_path: Path | None = None,
    local_db_path: Path | None = None,
    session_root: Path = DEFAULT_SESSION_ROOT,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    session = Path(session_root) / uuid.uuid4().hex
    session.mkdir(parents=True, exist_ok=False)
    if progress_callback:
        progress_callback({"phase": "decoding", "evaluations": 0})
    decoded = decode_audio_file(input_path, start_offset_s=start_offset_s)
    result_path = session / "result.json"
    if decoded.silent:
        result_path.write_text(
            json.dumps(
                {
                    "status": "no_confident_match",
                    "no_confident_match": True,
                    "message": "No confident match — the selected audio is silent.",
                    "source": {
                        "path": str(decoded.path),
                        "decoder": decoded.decoder,
                        "source_duration_s": decoded.source_duration_s,
                        "start_offset_s": decoded.start_offset_s,
                        "used_duration_s": decoded.used_duration_s,
                        "rms_dbfs": decoded.rms_dbfs,
                    },
                    "target_synth": target_synth,
                    "existing_matches": [],
                    "recommendation": None,
                    "factory_only": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result_path
    if progress_callback:
        progress_callback({"phase": "loading-factory-fingerprints", "evaluations": 0})
    bundle = FactoryBundle(bundle_path)
    matrix, presets = bundle.search_index()
    runtime: list[dict[str, Any]] = [
        {"kind": "factory", "preset": preset} for preset in presets
    ]
    local_matrix, local_rows = _local_search_rows(local_db_path)
    if local_matrix is not None:
        matrix = np.concatenate([matrix, local_matrix], axis=0)
        runtime.extend(local_rows)
    embedder = ClapEmbedder(ENV)
    audio = decoded.mono
    if decoded.sample_rate != CLAP_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=decoded.sample_rate,
            target_sr=CLAP_SAMPLE_RATE,
            res_type="soxr_hq",
        ).astype(np.float32)
    embedding = embedder.embed([loudness_normalize(audio)])[0]
    scores, positions = cosine_topk(
        embedding[None, :], matrix, k=len(presets), normalized=True
    )
    local = _load_mapping(mapping_path)
    existing = []
    for score, position in zip(scores[0, :10], positions[0, :10], strict=True):
        item = runtime[int(position)]
        if item["kind"] == "factory":
            preset = item["preset"]
            path = local.get(preset.content_hash)
            preset_id = preset.id
            content_hash = preset.content_hash
            name = preset.name
            synth = preset.synth
            source_path = path or f"Factory/{preset.relative_path}"
            audition_path = None
            bundle_id: int | None = preset.id
        else:
            preset_id = int(item["preset_id"])
            content_hash = str(item["content_hash"])
            name = str(item["name"])
            synth = str(item["synth"])
            path = str(item["path"])
            source_path = path
            audition_path = item["audition_path"]
            bundle_id = None
        existing.append(
            {
                "preset_id": preset_id,
                "content_hash": content_hash,
                "name": display_match_name(name, len(existing) + 1),
                "synth": synth,
                "source_path": source_path,
                "local_source_available": bool(path),
                "similarity": float(score),
                "similarity_percent": 100.0 * float(score),
                "audition_midi_note": 60 if audition_path else None,
                "audition_path": audition_path,
                "preview_source_path": path if path and not audition_path else None,
                "factory_bundle_id": bundle_id,
            }
        )
    target_rows = [
        index
        for index, item in enumerate(runtime)
        if (
            item["preset"].synth
            if item["kind"] == "factory"
            else item["synth"]
        )
        == target_synth
    ]
    target_scores = matrix[target_rows] @ embedding
    target_position = target_rows[int(np.argmax(target_scores))]
    recommendation_item = runtime[target_position]
    recommendation_score = float(target_scores[int(np.argmax(target_scores))])
    if recommendation_item["kind"] == "factory":
        recommendation_preset = recommendation_item["preset"]
        vector, mask = bundle.parameters(recommendation_preset.id)
        settings = _settings_summary(bundle, recommendation_preset.id)
        recommendation_id = recommendation_preset.id
        recommendation_hash = recommendation_preset.content_hash
        recommendation_name = recommendation_preset.name
        recommendation_synth = recommendation_preset.synth
        local_source = local.get(recommendation_preset.content_hash)
        bundle_id = recommendation_preset.id
        origin = "factory-fingerprint"
    else:
        vector, mask, settings = _local_parameters(bundle, recommendation_item)
        recommendation_id = int(recommendation_item["preset_id"])
        recommendation_hash = str(recommendation_item["content_hash"])
        recommendation_name = str(recommendation_item["name"])
        recommendation_synth = str(recommendation_item["synth"])
        local_source = str(recommendation_item["path"])
        bundle_id = None
        origin = "local-fingerprint"
    candidate_path = session / "candidate.npz"
    np.savez_compressed(candidate_path, vector=vector, mask=mask, base_vector=vector)
    acoustic_note, hz, unpitched = detect_midi_note(decoded.mono, decoded.sample_rate)
    no_confident = recommendation_score < 0.65
    payload = {
        "status": "complete",
        "factory_only": True,
        "no_confident_match": no_confident,
        "message": (
            "Low confidence — this sound may not be well-suited to Serum."
            if no_confident
            else "Factory fingerprint match complete"
        ),
        "source": {
            "path": str(decoded.path),
            "decoder": decoded.decoder,
            "source_duration_s": decoded.source_duration_s,
            "start_offset_s": decoded.start_offset_s,
            "used_duration_s": decoded.used_duration_s,
            "rms_dbfs": decoded.rms_dbfs,
        },
        "target_synth": target_synth,
        "detected": {
            "midi_note": acoustic_note,
            "acoustic_midi_note": acoustic_note,
            "frequency_hz": hz,
            "unpitched_fallback": unpitched,
        },
        "existing_matches": existing,
        "recommendation": {
            "synth": recommendation_synth,
            "base_preset_id": recommendation_id,
            "factory_bundle_id": bundle_id,
            "content_hash": recommendation_hash,
            "factory_source_path": local_source,
            "export_available": bool(local_source),
            "base_name": generated_preset_name(recommendation_synth),
            "origin": origin,
            "meaningfully_modified": False,
            "clap_similarity": recommendation_score,
            "similarity_percent": 100.0 * recommendation_score,
            "stft_loss": None,
            "objective": 1.0 - recommendation_score,
            "evaluations": 0,
            "elapsed_s": 0.0,
            "winner_audio_path": None,
            "preview_source_path": local_source,
            "preview_midi_note": acoustic_note,
            "candidate_path": str(candidate_path),
            "settings": settings,
            "objective_trace": [],
        },
    }
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if progress_callback:
        progress_callback(
            {
                "phase": "complete",
                "evaluations": 0,
                "best_clap_cosine": recommendation_score,
            }
        )
    return result_path
