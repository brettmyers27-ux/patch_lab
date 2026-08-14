#!/usr/bin/env python3
"""Resumable Stage 3I residual, pilot, and full-suite runner."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_input import decode_audio_file
from core.features import CLAP_SAMPLE_RATE, ClapEmbedder
from core.layer_decomposition import (
    fit_nonnegative_magnitude_scale,
    phase_robust_residual,
    stft_magnitude,
)
from core.match_workflow import run_match_file
from core.matcher import embedding_comparison_audio, prepare_query_audio
from core.model_assets import configure_model_environment
from core.platform_env import ENV
from core.preset_stack import (
    PresetLayer,
    PresetStack,
    deterministic_mix_polish,
)
from core.synthesis_assets import resolve_synthesis_assets
from scripts.benchmark_suite import (
    ADOPTED_STAGE2B_BAM_DETAILS,
    _detail_name,
    target_synth_for_name,
)
from scripts.stage3h_bam_audit import (
    candidate_state_sha256,
    canonical_audio_sha256,
)


DEFAULT_CONTROL = PROJECT_ROOT / "data" / "stage3i" / "control-arm-a-valid" / "bam"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "stage3i" / "two-layer"
SEED = 20260802


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _control_rows(directory: Path) -> list[dict[str, Any]]:
    rows = [_read_json(path) for path in Path(directory).glob("*.json")]
    rows.sort(key=lambda row: str(row["source_name"]).casefold())
    if len(rows) != 99:
        raise RuntimeError(f"Stage 3I requires 99 control rows, found {len(rows)}")
    if any(row.get("status") != "complete" for row in rows):
        raise RuntimeError("The Stage 3I control contains failed rows")
    if sum(target_synth_for_name(str(row["source_name"])) == "serum1" for row in rows) != 47:
        raise RuntimeError("The fixed BAM split no longer contains 47 Serum 1 rows")
    if sum(target_synth_for_name(str(row["source_name"])) == "serum2" for row in rows) != 52:
        raise RuntimeError("The fixed BAM split no longer contains 52 Serum 2 rows")
    if any(row.get("render_dispatch") != "deterministic-dedicated-workers-v1" for row in rows):
        raise RuntimeError("The Stage 3H pinned host assignment is not preserved")
    return rows


def _layer1_result_path(control: dict[str, Any]) -> Path:
    direct = Path(str(control.get("result_path") or ""))
    if direct.is_file():
        return direct
    adopted = ADOPTED_STAGE2B_BAM_DETAILS / _detail_name(Path(control["source"]))
    row = _read_json(adopted)
    result = Path(str(row.get("result_path") or ""))
    if not result.is_file():
        raise FileNotFoundError(f"Layer 1 result is unavailable for {control['source_name']}")
    return result


def _mono_file(path: Path, *, length: int | None = None) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if sample_rate != CLAP_SAMPLE_RATE:
        mono = librosa.resample(
            mono,
            orig_sr=int(sample_rate),
            target_sr=CLAP_SAMPLE_RATE,
            res_type="soxr_hq",
        ).astype(np.float32)
    if length is not None:
        mono = mono[:length]
        if len(mono) < length:
            mono = np.pad(mono, (0, length - len(mono)))
    return np.ascontiguousarray(mono, dtype=np.float32)


def _prepared_target(source: Path) -> tuple[np.ndarray, float]:
    decoded = decode_audio_file(source)
    return prepare_query_audio(decoded.mono, decoded.sample_rate, adaptive=True)


def _preset_name(preset_id: int, database: Path) -> str:
    with sqlite3.connect(database) as connection:
        row = connection.execute("SELECT name FROM presets WHERE id=?", (preset_id,)).fetchone()
    return str(row[0]) if row else f"preset-{preset_id}"


def prepare_residuals(control_dir: Path, output_root: Path) -> dict[str, Any]:
    rows = _control_rows(control_dir)
    residual_dir = output_root / "residuals"
    detail_dir = output_root / "residual-details"
    index_rows: list[dict[str, Any]] = []
    for position, control in enumerate(rows, start=1):
        key = Path(_detail_name(Path(control["source"]))).stem
        detail_path = detail_dir / f"{key}.json"
        residual_path = residual_dir / f"{key}.wav"
        expected = {
            "schema_version": 1,
            "stage": "3I",
            "source_sha256": control["source_sha256"],
            "control_score": float(control["clap_similarity"]),
            "residual_method": "aligned-stft-magnitude-subtraction-v1",
        }
        existing = _read_json(detail_path) if detail_path.is_file() else None
        if (
            existing is not None
            and all(existing.get(name) == value for name, value in expected.items())
            and residual_path.is_file()
        ):
            row = existing
        else:
            source = Path(control["source"])
            target, duration = _prepared_target(source)
            layer1_result_path = _layer1_result_path(control)
            layer1_result = _read_json(layer1_result_path)
            recommendation = layer1_result["recommendation"]
            winner = Path(recommendation["winner_audio_path"])
            candidate = Path(recommendation["candidate_path"])
            layer1_audio = _mono_file(winner, length=len(target))
            residual = phase_robust_residual(target, layer1_audio, CLAP_SAMPLE_RATE)
            residual_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                residual_path,
                residual.normalized_residual_audio,
                CLAP_SAMPLE_RATE,
                subtype="FLOAT",
                format="WAV",
            )
            row = {
                **expected,
                "source_name": control["source_name"],
                "source": str(source.resolve()),
                "source_group": target_synth_for_name(str(control["source_name"])),
                "comparison_duration_s": float(duration),
                "prepared_samples": len(target),
                "residual_path": str(residual_path.resolve()),
                "layer1": {
                    "result_path": str(layer1_result_path.resolve()),
                    "synth": recommendation["synth"],
                    "base_preset_id": int(recommendation["base_preset_id"]),
                    "origin": str(recommendation["origin"]),
                    "midi_note": int(layer1_result["detected"]["midi_note"]),
                    "match_score": float(control["clap_similarity"]),
                    "candidate_path": str(candidate.resolve()),
                    "winner_audio_path": str(winner.resolve()),
                    "candidate_state_sha256": candidate_state_sha256(candidate),
                    "decoded_audio_sha256": canonical_audio_sha256(winner),
                },
                "residual": residual.diagnostics.to_dict(),
            }
            _write_json(detail_path, row)
        index_rows.append(row)
        print(
            f"RESIDUAL={position}/{len(rows)} target={control['source_name']} "
            f"energy={row['residual']['residual_spectral_energy_ratio']:.9f}",
            flush=True,
        )
    ordered_high = sorted(
        index_rows,
        key=lambda row: (
            -float(row["residual"]["residual_spectral_energy_ratio"]),
            str(row["source_name"]).casefold(),
        ),
    )
    high = ordered_high[:12]
    high_hashes = {row["source_sha256"] for row in high}
    low = [
        row
        for row in sorted(
            index_rows,
            key=lambda row: (
                float(row["residual"]["residual_spectral_energy_ratio"]),
                str(row["source_name"]).casefold(),
            ),
        )
        if row["source_sha256"] not in high_hashes
    ][:4]
    payload = {
        "schema_version": 1,
        "stage": "3I",
        "seed": SEED,
        "control_detail_dir": str(control_dir.resolve()),
        "residual_count": len(index_rows),
        "rows": index_rows,
        "pilot": [
            {
                "source_sha256": row["source_sha256"],
                "source_name": row["source_name"],
                "selection": "highest-residual" if row in high else "lowest-residual-control",
                "residual_spectral_energy_ratio": row["residual"]["residual_spectral_energy_ratio"],
            }
            for row in high + low
        ],
    }
    _write_json(output_root / "residual-index.json", payload)
    return payload


def _run_layer2_match(
    row: dict[str, Any],
    *,
    source: Path,
    arm: str,
    output_root: Path,
) -> dict[str, Any]:
    key = Path(_detail_name(Path(row["source"]))).stem
    detail_path = output_root / "layer2-matches" / arm / f"{key}.json"
    expected = {
        "schema_version": 1,
        "stage": "3I",
        "arm": arm,
        "source_sha256": row["source_sha256"],
        "target_synth": row["layer1"]["synth"],
        "render_dispatch": "deterministic-dedicated-workers-v1",
        "structural_search": False,
        "seed": SEED,
    }
    if detail_path.is_file():
        existing = _read_json(detail_path)
        result_path = Path(str(existing.get("result_path") or ""))
        if all(existing.get(name) == value for name, value in expected.items()) and result_path.is_file():
            return existing
    started = time.monotonic()
    result_path = run_match_file(
        source,
        target_synth=row["layer1"]["synth"],
        budget="balanced",
        session_root=output_root / "layer2-sessions" / arm,
        matcher_processes=4,
        deterministic_render_dispatch=True,
    )
    payload = {
        **expected,
        "source_name": row["source_name"],
        "match_input": "full-target" if arm == "pilot-b" else "phase-robust-residual",
        "result_path": str(result_path.resolve()),
        "wall_clock_s": time.monotonic() - started,
    }
    _write_json(detail_path, payload)
    return payload


class _ClapMixScorer:
    def __init__(self) -> None:
        self.embedder = ClapEmbedder(ENV)

    def for_target(self, target: np.ndarray, duration_s: float):
        target_audio = embedding_comparison_audio(target, duration_s, adaptive=True)
        target_embedding = self.embedder.embed([target_audio])[0]

        def score_batch(audios: Sequence[np.ndarray]) -> list[float]:
            prepared = [
                embedding_comparison_audio(audio, duration_s, adaptive=True)
                for audio in audios
            ]
            scores: list[float] = []
            for start in range(0, len(prepared), 16):
                embeddings = self.embedder.embed(prepared[start : start + 16])
                scores.extend(
                    float(np.clip(np.dot(target_embedding, item), -1.0, 1.0))
                    for item in embeddings
                )
            return scores

        return score_batch


def _polish(
    row: dict[str, Any],
    match: dict[str, Any],
    *,
    arm: str,
    scorer: _ClapMixScorer,
    output_root: Path,
) -> dict[str, Any]:
    key = Path(_detail_name(Path(row["source"]))).stem
    output_path = output_root / "polished" / arm / f"{key}.json"
    expected = {
        "schema_version": 1,
        "stage": "3I",
        "arm": arm,
        "source_sha256": row["source_sha256"],
        "mix_search": "deterministic-coarse-timing-refine-v1",
    }
    if output_path.is_file():
        existing = _read_json(output_path)
        if all(existing.get(name) == value for name, value in expected.items()):
            return existing
    started = time.monotonic()
    target, duration = _prepared_target(Path(row["source"]))
    layer1 = _mono_file(Path(row["layer1"]["winner_audio_path"]), length=len(target))
    layer2_result = _read_json(Path(match["result_path"]))
    recommendation = layer2_result["recommendation"]
    layer2_winner = Path(recommendation["winner_audio_path"])
    layer2_candidate = Path(recommendation["candidate_path"])
    layer2 = _mono_file(layer2_winner, length=len(target))
    reference = (
        stft_magnitude(target)
        if arm == "pilot-b"
        else stft_magnitude(_mono_file(Path(row["residual_path"]), length=len(target)))
    )
    layer2_scale = fit_nonnegative_magnitude_scale(reference, stft_magnitude(layer2))
    polish = deterministic_mix_polish(
        layer1_audio=layer1,
        layer2_audio=layer2,
        sample_rate=CLAP_SAMPLE_RATE,
        initial_layer1_scale=float(row["residual"]["layer1_magnitude_scale"]),
        initial_layer2_scale=layer2_scale,
        one_layer_score=float(row["control_score"]),
        score_batch=scorer.for_target(target, duration),
    )
    mixed_path = output_root / "mixed-audio" / arm / f"{key}.wav"
    mixed_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(mixed_path, polish.audio, CLAP_SAMPLE_RATE, subtype="FLOAT", format="WAV")
    layer1_descriptor = PresetLayer(
        synth=row["layer1"]["synth"],
        base_preset_id=int(row["layer1"]["base_preset_id"]),
        state_reference=row["layer1"]["candidate_path"],
        candidate_state_sha256=row["layer1"]["candidate_state_sha256"],
        decoded_audio_sha256=row["layer1"]["decoded_audio_sha256"],
        gain_db=polish.layer1_gain_db,
        timing_offset_ms=0.0,
        role="dominant",
        match_score=float(row["control_score"]),
        midi_note=int(row["layer1"]["midi_note"]),
        origin=row["layer1"]["origin"],
    )
    layer2_state_hash = candidate_state_sha256(layer2_candidate)
    layer2_audio_hash = canonical_audio_sha256(layer2_winner)
    layer2_descriptor = PresetLayer(
        synth=recommendation["synth"],
        base_preset_id=int(recommendation["base_preset_id"]),
        state_reference=str(layer2_candidate.resolve()),
        candidate_state_sha256=layer2_state_hash,
        decoded_audio_sha256=layer2_audio_hash,
        gain_db=float(polish.layer2_gain_db or 0.0),
        timing_offset_ms=polish.layer2_timing_offset_ms,
        role="residual",
        match_score=float(recommendation["clap_similarity"]),
        midi_note=int(layer2_result["detected"]["midi_note"]),
        origin=str(recommendation["origin"]),
    )
    same_base = layer1_descriptor.base_preset_id == layer2_descriptor.base_preset_id
    same_state = layer1_descriptor.candidate_state_sha256 == layer2_state_hash
    relative_gain = (
        float(polish.layer2_gain_db - polish.layer1_gain_db)
        if polish.layer2_gain_db is not None
        else float("-inf")
    )
    delta = float(polish.score - row["control_score"])
    diagnostics = {
        "selection_reason": (
            "two-layer score strictly exceeded exact Layer 1 null"
            if polish.second_layer_selected
            else "exact Layer 1 null candidate won"
        ),
        "matching_input": match["match_input"],
        "one_layer_score": float(row["control_score"]),
        "score_delta": delta,
        "initial_layer1_scale": float(row["residual"]["layer1_magnitude_scale"]),
        "initial_layer2_scale": layer2_scale,
        "residual_alignment_offset_ms": float(row["residual"]["alignment_offset_ms"]),
        "same_base_preset": same_base,
        "same_candidate_state": same_state,
        "negligible_layer2_gain": relative_gain <= -30.0,
        "extremely_small_improvement": 0.0 < delta < 0.001,
        "noise_or_tail_flag": (
            float(row["residual"]["residual_noisiness"]) >= 0.5
            or float(row["residual"]["residual_tail_energy_fraction"]) >= 0.6
        ),
        "layer2_relative_gain_db": relative_gain,
        "gain_combination_count": polish.gain_combination_count,
        "mixture_evaluation_count": polish.mixture_evaluation_count,
    }
    stack = PresetStack(
        target_synth=row["layer1"]["synth"],
        layers=(layer1_descriptor, layer2_descriptor)
        if polish.second_layer_selected
        else (layer1_descriptor,),
        combined_final_score=polish.score,
        residual_energy_ratio=float(row["residual"]["residual_spectral_energy_ratio"]),
        second_layer_selected=polish.second_layer_selected,
        diagnostics=diagnostics,
    )
    payload = {
        **expected,
        "source_name": row["source_name"],
        "source_group": row["source_group"],
        "control_score": float(row["control_score"]),
        "combined_score": polish.score,
        "delta": delta,
        "second_layer_selected": polish.second_layer_selected,
        "residual": row["residual"],
        "layer2_match": {
            "result_path": match["result_path"],
            "wall_clock_s": float(match["wall_clock_s"]),
            "base_preset_id": int(recommendation["base_preset_id"]),
            "origin": str(recommendation["origin"]),
            "match_score": float(recommendation["clap_similarity"]),
            "candidate_state_sha256": layer2_state_hash,
            "decoded_audio_sha256": layer2_audio_hash,
        },
        "stack": stack.to_dict(),
        "mixed_audio_path": str(mixed_path.resolve()),
        "mix_polish_wall_clock_s": time.monotonic() - started,
        "mix_search_trace": list(polish.search_trace),
    }
    _write_json(output_path, payload)
    return payload


def run_pilot(control_dir: Path, output_root: Path) -> dict[str, Any]:
    index_path = output_root / "residual-index.json"
    index = _read_json(index_path) if index_path.is_file() else prepare_residuals(control_dir, output_root)
    by_hash = {row["source_sha256"]: row for row in index["rows"]}
    membership = index["pilot"]
    matches: dict[tuple[str, str], dict[str, Any]] = {}
    for position, selected in enumerate(membership, start=1):
        row = by_hash[selected["source_sha256"]]
        for arm, source in (
            ("pilot-b", Path(row["source"])),
            ("pilot-c", Path(row["residual_path"])),
        ):
            matches[(row["source_sha256"], arm)] = _run_layer2_match(
                row, source=source, arm=arm, output_root=output_root
            )
        print(f"PILOT_MATCH={position}/{len(membership)} target={row['source_name']}", flush=True)
    scorer = _ClapMixScorer()
    rows = []
    for position, selected in enumerate(membership, start=1):
        row = by_hash[selected["source_sha256"]]
        arm_b = _polish(
            row,
            matches[(row["source_sha256"], "pilot-b")],
            arm="pilot-b",
            scorer=scorer,
            output_root=output_root,
        )
        arm_c = _polish(
            row,
            matches[(row["source_sha256"], "pilot-c")],
            arm="pilot-c",
            scorer=scorer,
            output_root=output_root,
        )
        rows.append(
            {
                "source_name": row["source_name"],
                "source_sha256": row["source_sha256"],
                "selection": selected["selection"],
                "residual_energy_ratio": float(row["residual"]["residual_spectral_energy_ratio"]),
                "arm_a_score": float(row["control_score"]),
                "arm_b_score": float(arm_b["combined_score"]),
                "arm_c_score": float(arm_c["combined_score"]),
                "arm_b_selected_layer2": bool(arm_b["second_layer_selected"]),
                "arm_c_selected_layer2": bool(arm_c["second_layer_selected"]),
                "arm_b_minus_a": float(arm_b["delta"]),
                "arm_c_minus_a": float(arm_c["delta"]),
                "arm_c_minus_b": float(arm_c["combined_score"] - arm_b["combined_score"]),
            }
        )
        print(f"PILOT_POLISH={position}/{len(membership)} target={row['source_name']}", flush=True)
    high = [row for row in rows if row["selection"] == "highest-residual"]
    low = [row for row in rows if row["selection"] == "lowest-residual-control"]
    high_a = mean(row["arm_a_score"] for row in high)
    high_b = mean(row["arm_b_score"] for row in high)
    high_c = mean(row["arm_c_score"] for row in high)
    conditions = {
        "arm_c_mean_exceeds_arm_a_on_high_residual": high_c > high_a,
        "arm_c_mean_exceeds_arm_b_on_high_residual": high_c > high_b,
        "at_least_6_of_12_high_select_layer2": sum(row["arm_c_selected_layer2"] for row in high) >= 6,
        "at_most_2_of_4_low_select_layer2": sum(row["arm_c_selected_layer2"] for row in low) <= 2,
        "no_infrastructure_failure": True,
    }
    passed = all(conditions.values())
    failure_modes = []
    if not conditions["arm_c_mean_exceeds_arm_a_on_high_residual"]:
        failure_modes.append("residual extraction or residual retrieval did not improve the dominant-layer control")
    if not conditions["arm_c_mean_exceeds_arm_b_on_high_residual"]:
        failure_modes.append("compute-matched duplicate stacking performed at least as well as residual matching")
    if not conditions["at_least_6_of_12_high_select_layer2"]:
        failure_modes.append("residual-matched patches did not reconstruct enough high-residual targets")
    if not conditions["at_most_2_of_4_low_select_layer2"]:
        failure_modes.append("the selection gate over-selected Layer 2 on low-residual controls")
    payload = {
        "schema_version": 1,
        "stage": "3I",
        "seed": SEED,
        "rows": rows,
        "high_residual": {
            "count": len(high),
            "arm_a_mean": high_a,
            "arm_b_mean": high_b,
            "arm_c_mean": high_c,
            "arm_b_selected": sum(row["arm_b_selected_layer2"] for row in high),
            "arm_c_selected": sum(row["arm_c_selected_layer2"] for row in high),
        },
        "low_residual": {
            "count": len(low),
            "arm_a_mean": mean(row["arm_a_score"] for row in low),
            "arm_b_mean": mean(row["arm_b_score"] for row in low),
            "arm_c_mean": mean(row["arm_c_score"] for row in low),
            "arm_b_selected": sum(row["arm_b_selected_layer2"] for row in low),
            "arm_c_selected": sum(row["arm_c_selected_layer2"] for row in low),
        },
        "conditions": conditions,
        "passed": passed,
        "failure_modes": failure_modes,
    }
    _write_json(output_root / "pilot-summary.json", payload)
    return payload


def _percentile(values: Sequence[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def run_full(control_dir: Path, output_root: Path) -> dict[str, Any]:
    pilot = _read_json(output_root / "pilot-summary.json")
    if not pilot.get("passed"):
        raise RuntimeError("Pilot gate failed; the full 99-file benchmark is prohibited")
    index = _read_json(output_root / "residual-index.json")
    rows = index["rows"]
    matches = []
    for position, row in enumerate(rows, start=1):
        matches.append(
            _run_layer2_match(
                row,
                source=Path(row["residual_path"]),
                arm="full-b",
                output_root=output_root,
            )
        )
        print(f"FULL_MATCH={position}/{len(rows)} target={row['source_name']}", flush=True)
    scorer = _ClapMixScorer()
    results = []
    for position, (row, match) in enumerate(zip(rows, matches, strict=True), start=1):
        results.append(
            _polish(
                row,
                match,
                arm="full-b",
                scorer=scorer,
                output_root=output_root,
            )
        )
        print(f"FULL_POLISH={position}/{len(rows)} target={row['source_name']}", flush=True)
    arm_a = [float(row["control_score"]) for row in rows]
    arm_b = [float(row["combined_score"]) for row in results]
    deltas = [right - left for left, right in zip(arm_a, arm_b, strict=True)]
    selected = [row for row in results if row["second_layer_selected"]]
    selected_deltas = [float(row["delta"]) for row in selected]
    residual_energy = [float(row["residual"]["residual_spectral_energy_ratio"]) for row in results]
    correlation = float(np.corrcoef(residual_energy, deltas)[0, 1])
    groups = {}
    for group in ("serum1", "serum2"):
        positions = [index for index, row in enumerate(rows) if row["source_group"] == group]
        group_a = [arm_a[index] for index in positions]
        group_b = [arm_b[index] for index in positions]
        groups[group] = {
            "count": len(positions),
            "arm_a_mean": mean(group_a),
            "arm_b_mean": mean(group_b),
            "delta": mean(group_b) - mean(group_a),
        }
        if group == "serum2":
            groups[group].update(
                arm_a_median=median(group_a),
                arm_b_median=median(group_b),
                arm_a_minimum=min(group_a),
                arm_b_minimum=min(group_b),
            )
    arm_a_wall = [float(row.get("wall_clock_s") or 0.0) for row in _control_rows(control_dir)]
    arm_b_wall = [
        float(row["layer2_match"]["wall_clock_s"]) + float(row["mix_polish_wall_clock_s"])
        for row in results
    ]
    payload = {
        "schema_version": 1,
        "stage": "3I",
        "seed": SEED,
        "count": len(rows),
        "arm_a_whole_mean": mean(arm_a),
        "arm_b_whole_mean": mean(arm_b),
        "paired_mean_delta": mean(deltas),
        "improved": sum(delta > 1e-12 for delta in deltas),
        "regressed": sum(delta < -1e-12 for delta in deltas),
        "unchanged": sum(abs(delta) <= 1e-12 for delta in deltas),
        "selected_layer2": len(selected),
        "selection_rate": len(selected) / len(rows),
        "selected_mean_delta": mean(selected_deltas) if selected_deltas else 0.0,
        "selected_median_delta": median(selected_deltas) if selected_deltas else 0.0,
        "selected_at_least_0_005": (
            mean(delta >= 0.005 for delta in selected_deltas) if selected_deltas else 0.0
        ),
        "maximum_regression": min(deltas),
        "residual_energy_benefit_correlation": correlation,
        "residual_energy_distribution": {
            "mean": mean(residual_energy),
            "p50": median(residual_energy),
            "p95": _percentile(residual_energy, 95),
        },
        "wall_clock": {
            "arm_a": {"mean": mean(arm_a_wall), "p50": median(arm_a_wall), "p95": _percentile(arm_a_wall, 95)},
            "arm_b": {"mean": mean(arm_b_wall), "p50": median(arm_b_wall), "p95": _percentile(arm_b_wall, 95)},
        },
        "by_source_group": groups,
        "rows": results,
    }
    _write_json(output_root / "full-summary.json", payload)
    return payload


def export_public_diagnostics(
    control_dir: Path, output_root: Path, public_dir: Path
) -> dict[str, Any]:
    """Commit-safe measured rows without private paths, audio, or preset state."""

    controls = _control_rows(control_dir)
    control_scores = [float(row["clap_similarity"]) for row in controls]
    control_groups: dict[str, Any] = {}
    for group in ("serum1", "serum2"):
        scores = [
            float(row["clap_similarity"])
            for row in controls
            if target_synth_for_name(str(row["source_name"])) == group
        ]
        control_groups[group] = {
            "count": len(scores),
            "mean": mean(scores),
            "median": median(scores),
            "minimum": min(scores),
        }
    control_payload = {
        "schema_version": 1,
        "stage": "3I",
        "seed": SEED,
        "render_dispatch": "deterministic-dedicated-workers-v1",
        "structural_search": False,
        "whole_set": {
            "count": len(control_scores),
            "mean": mean(control_scores),
            "median": median(control_scores),
            "minimum": min(control_scores),
        },
        "by_source_group": control_groups,
        "rows": [
            {
                "source_name": row["source_name"],
                "source_sha256": row["source_sha256"],
                "source_group": target_synth_for_name(str(row["source_name"])),
                "score": float(row["clap_similarity"]),
                "reused_unaffected": bool(row.get("reused_unaffected_arm")),
            }
            for row in controls
        ],
    }
    index = _read_json(output_root / "residual-index.json")
    membership = {
        row["source_sha256"]: row["selection"] for row in index["pilot"]
    }
    residual_payload = {
        "schema_version": 1,
        "stage": "3I",
        "method": "aligned-stft-magnitude-subtraction-v1",
        "fft_size": 2048,
        "hop_length": 512,
        "window": "hann",
        "layer1_scale_bounds": [0.0, 2.0],
        "rows": [
            {
                "source_name": row["source_name"],
                "source_sha256": row["source_sha256"],
                "source_group": row["source_group"],
                "pilot_selection": membership.get(row["source_sha256"]),
                **row["residual"],
            }
            for row in index["rows"]
        ],
    }
    pilot = _read_json(output_root / "pilot-summary.json")
    assets = resolve_synthesis_assets()
    detail_rows = []
    by_hash = {row["source_sha256"]: row for row in index["rows"]}
    for summary in pilot["rows"]:
        source = by_hash[summary["source_sha256"]]
        key = Path(_detail_name(Path(source["source"]))).stem
        row: dict[str, Any] = dict(summary)
        row["layer1"] = {
            "base_preset_id": int(source["layer1"]["base_preset_id"]),
            "base_preset_name": _preset_name(
                int(source["layer1"]["base_preset_id"]), assets.library_db
            ),
            "candidate_state_sha256": source["layer1"]["candidate_state_sha256"],
            "decoded_audio_sha256": source["layer1"]["decoded_audio_sha256"],
            "synth": source["layer1"]["synth"],
            "match_score": float(source["layer1"]["match_score"]),
        }
        for arm, label in (("pilot-b", "arm_b"), ("pilot-c", "arm_c")):
            detail = _read_json(output_root / "polished" / arm / f"{key}.json")
            layer2 = detail["layer2_match"]
            diagnostics = detail["stack"]["diagnostics"]
            layers = detail["stack"]["layers"]
            row[label] = {
                "layer2_base_preset_id": int(layer2["base_preset_id"]),
                "layer2_base_preset_name": _preset_name(
                    int(layer2["base_preset_id"]), assets.library_db
                ),
                "layer2_candidate_state_sha256": layer2["candidate_state_sha256"],
                "layer2_decoded_audio_sha256": layer2["decoded_audio_sha256"],
                "layer2_match_score": float(layer2["match_score"]),
                "selected_layer2": bool(detail["second_layer_selected"]),
                "combined_score": float(detail["combined_score"]),
                "delta": float(detail["delta"]),
                "layer1_gain_db": float(layers[0]["gain_db"]),
                "layer2_gain_db": (
                    float(layers[1]["gain_db"]) if len(layers) == 2 else None
                ),
                "layer2_offset_ms": (
                    float(layers[1]["timing_offset_ms"]) if len(layers) == 2 else 0.0
                ),
                "same_base_preset": bool(diagnostics["same_base_preset"]),
                "same_candidate_state": bool(diagnostics["same_candidate_state"]),
                "negligible_layer2_gain": bool(diagnostics["negligible_layer2_gain"]),
                "extremely_small_improvement": bool(
                    diagnostics["extremely_small_improvement"]
                ),
                "noise_or_tail_flag": bool(diagnostics["noise_or_tail_flag"]),
                "layer2_relative_gain_db": (
                    float(diagnostics["layer2_relative_gain_db"])
                    if np.isfinite(diagnostics["layer2_relative_gain_db"])
                    else None
                ),
                "match_wall_clock_s": float(layer2["wall_clock_s"]),
                "mix_wall_clock_s": float(detail["mix_polish_wall_clock_s"]),
                "gain_combination_count": int(diagnostics["gain_combination_count"]),
                "mixture_evaluation_count": int(
                    diagnostics["mixture_evaluation_count"]
                ),
            }
        detail_rows.append(row)
    pilot_payload = {**pilot, "rows": detail_rows}
    outputs = {
        "control": public_dir / "stage3i-control.json",
        "residuals": public_dir / "stage3i-residuals.json",
        "pilot": public_dir / "stage3i-pilot.json",
    }
    _write_json(outputs["control"], control_payload)
    _write_json(outputs["residuals"], residual_payload)
    _write_json(outputs["pilot"], pilot_payload)
    return {name: str(path.resolve()) for name, path in outputs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "full", "export"))
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--public-dir", type=Path, default=PROJECT_ROOT / "docs" / "benchmarks"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    os.environ["PATCHLAB_SERUM2_STRUCTURAL_SEARCH"] = "0"
    os.environ["PATCHLAB_SERUM2_STRUCTURAL_ROUTES"] = "0"
    configure_model_environment()
    control_dir = args.control_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        result = prepare_residuals(control_dir, output_root)
    elif args.phase == "pilot":
        result = run_pilot(control_dir, output_root)
    elif args.phase == "full":
        result = run_full(control_dir, output_root)
    else:
        result = export_public_diagnostics(
            control_dir, output_root, args.public_dir.expanduser().resolve()
        )
    print("STAGE3I_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
