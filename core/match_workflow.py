"""File-to-result workflow shared by the Match UI and integration gates."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import soundfile as sf

from core.audio_input import DecodedAudio, decode_audio_file
from core.branding import display_match_name, generated_preset_name
from core.matcher import AnalysisBySynthesisMatcher, Candidate, SearchConfig
from core.serum2_preset_writer import vector_was_modified


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = PROJECT_ROOT / "data" / "matches"
BUDGETS: dict[str, SearchConfig] = {
    "quick": SearchConfig(max_evaluations=51, max_seconds=30.0),
    "balanced": SearchConfig(max_evaluations=300, max_seconds=120.0),
    "best": SearchConfig(max_evaluations=600, max_seconds=300.0),
}
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


def _emit(
    callback: Callable[[dict[str, Any]], None] | None, detail: dict[str, Any]
) -> None:
    if callback is not None:
        callback(detail)


def _section_for_serum1(name: str) -> str:
    value = name.casefold()
    if any(token in value for token in ("osc a", "a osc", "oscillator a")):
        return "Osc A"
    if any(token in value for token in ("osc b", "b osc", "oscillator b")):
        return "Osc B"
    if "sub" in value:
        return "Sub"
    if "noise" in value:
        return "Noise"
    if "filter" in value:
        return "Filter"
    if "env" in value or "envelope" in value:
        return "Envelopes"
    if "lfo" in value:
        return "LFOs"
    if any(
        token in value
        for token in (
            "fx",
            "hyper",
            "dist",
            "flang",
            "chorus",
            "delay",
            "reverb",
            "compress",
            "eq ",
        )
    ):
        return "FX"
    if "mod " in value or "matrix" in value:
        return "Mod Matrix"
    return "Global"


def _preset_details(
    preset_ids: list[int],
    database_path: Path,
) -> dict[int, dict[str, Any]]:
    placeholders = ",".join("?" for _ in preset_ids)
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            f"SELECT id,name,synth,path,content_hash FROM presets "
            f"WHERE id IN ({placeholders})",
            tuple(preset_ids),
        ).fetchall()
    return {
        int(row[0]): {
            "name": str(row[1]),
            "synth": str(row[2]),
            "source_path": str(row[3]),
            "content_hash": str(row[4]),
        }
        for row in rows
    }


def _nearest_render(
    preset_id: int,
    midi_note: int,
    audio_root: Path,
) -> tuple[int, Path | None]:
    notes = (24, 36, 48, 60, 72, 84, 96)
    ordered = sorted(notes, key=lambda value: abs(value - midi_note))
    for note in ordered:
        path = audio_root / str(preset_id) / f"{note}.wav"
        if path.is_file():
            return note, path
    return ordered[0], None


def _serum1_settings(
    candidate: Candidate,
    base_vector: np.ndarray,
    database_path: Path,
) -> dict[str, dict[str, Any]]:
    from core.platform_env import ENV
    from core.plugin_host import dump_dawdreamer_parameters, make_dawdreamer_processor

    plugin = next(
        item
        for item in ENV.plugins_for("serum1")
        if item.format == "VST2" and item.hostable
    )
    _engine, processor = make_dawdreamer_processor(plugin)
    with sqlite3.connect(database_path) as connection:
        path = connection.execute(
            "SELECT path FROM presets WHERE id=?", (candidate.base_preset_id,)
        ).fetchone()[0]
    if processor.load_preset(str(path)) is False:
        raise RuntimeError("Could not format Serum 1 candidate settings")
    for index, value in enumerate(candidate.vector):
        processor.set_parameter(index, float(value))
    dumped = dump_dawdreamer_parameters(processor)
    groups: dict[str, dict[str, Any]] = {}
    for item in dumped:
        group = groups.setdefault(
            _section_for_serum1(item.name), {"changed": [], "matches_base_count": 0}
        )
        changed = abs(float(candidate.vector[item.index]) - float(base_vector[item.index])) >= 0.02
        if changed:
            group["changed"].append(
                {
                    "index": item.index,
                    "name": item.name,
                    "value": item.display_value,
                    "normalized": float(candidate.vector[item.index]),
                }
            )
        else:
            group["matches_base_count"] += 1
    return dict(sorted(groups.items()))


def _schema_value(field: Mapping[str, Any], vector: np.ndarray) -> Any:
    index = int(field["index"])
    if field["encoding"] == "one_hot":
        width = int(field["width"])
        return field["categories"][int(np.argmax(vector[index : index + width]))]
    minimum, maximum = float(field["minimum"]), float(field["maximum"])
    return minimum + float(np.clip(vector[index], 0.0, 1.0)) * (maximum - minimum)


def _display_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return "None"
    return str(value)


def _serum2_settings(
    candidate: Candidate, base_vector: np.ndarray, schema: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for field in schema["fields"]:
        index = int(field["index"])
        width = int(field.get("width", 1))
        if not bool(candidate.mask[index]):
            continue
        group_name = SECTION_LABELS.get(
            str(field.get("category", "global_other")), "Global"
        )
        group = groups.setdefault(
            group_name, {"changed": [], "matches_base_count": 0}
        )
        if field["encoding"] == "one_hot":
            changed = int(
                np.argmax(candidate.vector[index : index + width])
            ) != int(np.argmax(base_vector[index : index + width]))
        else:
            changed = (
                abs(float(candidate.vector[index]) - float(base_vector[index]))
                >= 0.02
            )
        if changed:
            value = _schema_value(field, candidate.vector)
            group["changed"].append(
                {
                    "index": index,
                    "name": str(field["name"]),
                    "value": _display_value(value),
                    "normalized": float(candidate.vector[index]),
                }
            )
        else:
            group["matches_base_count"] += 1
    return dict(sorted(groups.items()))


def _candidate_payload(
    matcher: AnalysisBySynthesisMatcher, candidate: Candidate
) -> tuple[dict[str, Any], np.ndarray]:
    code = 1 if candidate.synth == "serum1" else 2
    store = matcher.stores[code]
    row = store.preset_row[candidate.base_preset_id]
    base = np.asarray(store.vectors[row], dtype=np.float32)
    if candidate.synth == "serum1":
        settings = _serum1_settings(
            candidate,
            base,
            matcher.assets.library_db,
        )
    else:
        settings = _serum2_settings(
            candidate, base, matcher.absolute_checkpoint["serum2_schema"]
        )
    return settings, base


def _silence_result(decoded: DecodedAudio, session: Path, target_synth: str) -> Path:
    payload = {
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
    }
    result_path = session / "result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result_path


def run_match_file(
    input_path: Path,
    *,
    target_synth: str = "serum2",
    budget: str = "balanced",
    start_offset_s: float = 0.0,
    session_root: Path = DEFAULT_SESSION_ROOT,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Decode a user file and persist the complete UI result artifact."""

    if target_synth not in {"serum1", "serum2"}:
        raise ValueError("target_synth must be serum1 or serum2")
    if budget not in BUDGETS:
        raise ValueError(f"Unknown search budget {budget!r}")
    session = Path(session_root) / uuid.uuid4().hex
    session.mkdir(parents=True, exist_ok=False)
    _emit(progress_callback, {"phase": "decoding", "evaluations": 0})
    decoded = decode_audio_file(input_path, start_offset_s=start_offset_s)
    if decoded.silent:
        _emit(
            progress_callback,
            {"phase": "complete", "evaluations": 0, "best_clap_cosine": 0.0},
        )
        return _silence_result(decoded, session, target_synth)

    _emit(progress_callback, {"phase": "loading-models", "evaluations": 0})
    matcher = AnalysisBySynthesisMatcher(processes=4)
    try:
        embedding = matcher.query_embedding(decoded.mono, decoded.sample_rate)
        retrieval = matcher.retrieve_existing(embedding, 10)
        detail = _preset_details(
            [preset_id for preset_id, _score in retrieval],
            matcher.assets.library_db,
        )

        def search_progress(value: dict[str, Any]) -> None:
            _emit(progress_callback, {"phase": "searching", **value})

        result = matcher.match(
            decoded.mono,
            decoded.sample_rate,
            synth_hint=target_synth,
            config=BUDGETS[budget],
            target_embedding=embedding,
            progress_callback=search_progress,
        )
        existing = []
        for preset_id, score in retrieval:
            note, wav_path = _nearest_render(
                preset_id,
                result.midi_note,
                matcher.audio_root,
            )
            existing.append(
                {
                    "preset_id": preset_id,
                    **detail[preset_id],
                    "similarity": score,
                    "similarity_percent": 100.0 * score,
                    "audition_midi_note": note,
                    "audition_path": str(wav_path) if wav_path else None,
                }
            )
        settings, base_vector = _candidate_payload(matcher, result.best)
        modified = (
            not result.best.exact_base
            and vector_was_modified(result.best.vector, base_vector, result.best.mask)
        )
        winner_path = session / "winner.wav"
        if result.best.waveform is None:
            raise RuntimeError("Matcher returned no winner audition waveform")
        sf.write(
            winner_path,
            result.best.waveform,
            48_000,
            subtype="FLOAT",
            format="WAV",
        )
        candidate_path = session / "candidate.npz"
        np.savez_compressed(
            candidate_path,
            vector=np.asarray(result.best.vector, dtype=np.float32),
            mask=np.asarray(result.best.mask, dtype=np.bool_),
            base_vector=base_vector,
        )
        no_confident = result.best.clap_cosine < 0.65
        payload = {
            "status": "complete",
            "no_confident_match": no_confident,
            "message": (
                "Low confidence — this sound may not be well-suited to Serum."
                if no_confident
                else "Match complete"
            ),
            "source": {
                "path": str(decoded.path),
                "decoder": decoded.decoder,
                "source_duration_s": decoded.source_duration_s,
                "start_offset_s": decoded.start_offset_s,
                "used_duration_s": decoded.used_duration_s,
                "rms_dbfs": decoded.rms_dbfs,
            },
            "budget": budget,
            "target_synth": target_synth,
            "detected": {
                "midi_note": result.midi_note,
                "acoustic_midi_note": result.acoustic_midi_note,
                "frequency_hz": result.detected_hz,
                "pyin_confidence": result.pitch_confidence,
                "sub_bass_fraction": result.sub_bass_fraction,
                "unpitched_fallback": result.unpitched_fallback,
                "note_hypotheses": list(result.note_hypotheses),
                "comparison_duration_s": result.comparison_duration_s,
            },
            "existing_matches": [
                {
                    **item,
                    "name": display_match_name(item.get("name"), index),
                }
                for index, item in enumerate(existing, start=1)
            ],
            "recommendation": {
                "synth": result.best.synth,
                "base_preset_id": result.best.base_preset_id,
                "base_name": generated_preset_name(result.best.synth),
                "origin": result.best.origin,
                "meaningfully_modified": modified,
                "clap_similarity": result.best.clap_cosine,
                "similarity_percent": 100.0 * result.best.clap_cosine,
                "stft_loss": result.best.stft_loss,
                "objective": result.best.objective,
                "stft_weight": result.stft_weight,
                "clap_weight": result.clap_weight,
                "evaluations": result.evaluations,
                "elapsed_s": result.elapsed_s,
                "winner_audio_path": str(winner_path),
                "candidate_path": str(candidate_path),
                "settings": settings,
                "objective_trace": result.objective_trace,
            },
        }
        result_path = session / "result.json"
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _emit(
            progress_callback,
            {
                "phase": "complete",
                "evaluations": result.evaluations,
                "budget": BUDGETS[budget].max_evaluations,
                "best_clap_cosine": result.best.clap_cosine,
            },
        )
        return result_path
    finally:
        matcher.close()
