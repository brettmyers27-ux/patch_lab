#!/usr/bin/env python3
"""Run the held-out and transformed-audio analysis-by-synthesis gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import FEATURE_DIR
from core.matcher import AnalysisBySynthesisMatcher, MatchResult, SearchConfig


REPORT = PROJECT_ROOT / "data" / "models" / "analysis_by_synthesis_gate_report.json"


def _selected() -> dict[str, list[int]]:
    checkpoint = __import__("torch").load(
        PROJECT_ROOT / "data" / "models" / "param_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    rng = np.random.default_rng(int(checkpoint["split"]["seed"]))
    result = {}
    for synth in ("serum1", "serum2"):
        candidates = np.asarray(checkpoint["split"]["validation_preset_ids"][synth])
        result[synth] = list(map(int, rng.choice(candidates, size=20, replace=False)))
    return result


def _row_lookup() -> tuple[Any, dict[tuple[int, int], int], np.ndarray]:
    manifest = np.load(FEATURE_DIR / "note_manifest.npz")
    lookup = {
        (int(preset_id), int(note)): index
        for index, (preset_id, note) in enumerate(
            zip(manifest["preset_ids"], manifest["midi_notes"], strict=True)
        )
    }
    embeddings = np.load(FEATURE_DIR / "note_embeddings.npy", mmap_mode="r")
    return manifest, lookup, embeddings


def _summary(result: MatchResult, query_id: int) -> dict[str, object]:
    excluding = result.best_excluding_preset
    return {
        "query_preset_id": query_id,
        "selected_midi_note": result.midi_note,
        "acoustic_midi_note": result.acoustic_midi_note,
        "pitch_confidence": result.pitch_confidence,
        "sub_bass_fraction": result.sub_bass_fraction,
        "note_hypotheses": result.note_hypotheses,
        "comparison_duration_s": result.comparison_duration_s,
        "clap_weight": result.clap_weight,
        "stft_weight": result.stft_weight,
        "unpitched_fallback": result.unpitched_fallback,
        "retrieved_preset_ids": result.retrieved_preset_ids,
        "best_base_preset_id": result.best.base_preset_id,
        "best_origin": result.best.origin,
        "best_clap_cosine": result.best.clap_cosine,
        "best_stft_loss": result.best.stft_loss,
        "best_objective": result.best.objective,
        "excluding_target_base_preset_id": excluding.base_preset_id if excluding else None,
        "excluding_target_clap_cosine": excluding.clap_cosine if excluding else None,
        "excluding_target_objective": excluding.objective if excluding else None,
        "evaluations": result.evaluations,
        "elapsed_s": result.elapsed_s,
        "evaluations_per_second": result.evaluations_per_second,
        "objective_trace": result.objective_trace,
    }


def _transform(audio: np.ndarray, sample_rate: int, index: int) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(
        np.asarray(audio, dtype=np.float32), sr=sample_rate, n_steps=3.0 if index % 2 == 0 else -3.0
    )
    spectrum = np.fft.rfft(shifted)
    frequency = np.linspace(0.0, 1.0, len(spectrum))
    tilt = np.power(10.0, ((frequency - 0.5) * (6.0 if index % 3 else -6.0)) / 20.0)
    tilted = np.fft.irfft(spectrum * tilt, n=len(shifted)).astype(np.float32)
    distorted = np.tanh(1.6 * tilted).astype(np.float32)
    delay = max(1, int(round(sample_rate * (0.011 + 0.002 * (index % 3)))))
    chorus = distorted.copy()
    chorus[delay:] = 0.78 * distorted[delay:] + 0.22 * distorted[:-delay]
    peak = float(np.max(np.abs(chorus)))
    return chorus / max(peak, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--novel-budget", type=int, default=300)
    parser.add_argument("--skip-novel", action="store_true")
    args = parser.parse_args()
    selected = _selected()
    manifest, lookup, embeddings = _row_lookup()
    matcher = AnalysisBySynthesisMatcher(processes=4)
    held_out: dict[str, list[dict[str, object]]] = {"serum1": [], "serum2": []}
    novel: list[dict[str, object]] = []
    try:
        total = 40
        completed = 0
        for synth in ("serum1", "serum2"):
            for preset_id in selected[synth]:
                row = lookup[(preset_id, 60)]
                audio, rate = sf.read(str(manifest["wav_paths"][row]), dtype="float32", always_2d=True)
                result = matcher.match(
                    np.mean(audio, axis=1),
                    rate,
                    synth_hint=synth,
                    config=SearchConfig(max_evaluations=args.budget),
                    target_embedding=np.asarray(embeddings[row]),
                    exclude_preset_id=preset_id,
                )
                held_out[synth].append(_summary(result, preset_id))
                completed += 1
                print(f"SELF_RECREATION_PROGRESS={completed}/{total}", flush=True)
        if not args.skip_novel:
            fixtures = [("serum1", value) for value in selected["serum1"][:5]] + [
                ("serum2", value) for value in selected["serum2"][:5]
            ]
            for index, (synth, preset_id) in enumerate(fixtures):
                row = lookup[(preset_id, 60)]
                audio, rate = sf.read(str(manifest["wav_paths"][row]), dtype="float32", always_2d=True)
                transformed = _transform(np.mean(audio, axis=1), rate, index)
                result = matcher.match(
                    transformed,
                    rate,
                    synth_hint=synth,
                    config=SearchConfig(max_evaluations=args.novel_budget),
                    exclude_preset_id=preset_id,
                )
                item = _summary(result, preset_id)
                item["transformation"] = {
                    "pitch_shift_semitones": 3 if index % 2 == 0 else -3,
                    "eq_tilt_db": 6 if index % 3 else -6,
                    "tanh_drive": 1.6,
                    "chorus_delay_ms": 11 + 2 * (index % 3),
                }
                novel.append(item)
                print(f"NOVEL_REPORT_PROGRESS={index + 1}/10", flush=True)
    finally:
        matcher.close()
    by_synth = {}
    thresholds = {"serum1": 0.90, "serum2": 0.80}
    for synth in ("serum1", "serum2"):
        rows = held_out[synth]
        similarities = [float(row["best_clap_cosine"]) for row in rows]
        excluding = [
            float(row["excluding_target_clap_cosine"])
            for row in rows
            if row["excluding_target_clap_cosine"] is not None
        ]
        times = [float(row["elapsed_s"]) for row in rows]
        mean = float(np.mean(similarities))
        by_synth[synth] = {
            "count": len(rows),
            "mean_clap_cosine": mean,
            "minimum_clap_cosine": float(np.min(similarities)),
            "threshold": thresholds[synth],
            "pass": mean >= thresholds[synth],
            "target_own_preset_wins": sum(
                int(row["best_base_preset_id"]) == int(row["query_preset_id"]) for row in rows
            ),
            "mean_excluding_target_clap_cosine": float(np.mean(excluding)),
            "median_wall_clock_s": float(np.median(times)),
            "mean_evaluations_per_second": float(
                np.mean([float(row["evaluations_per_second"]) for row in rows])
            ),
        }
    one_shot = {"serum1": 0.46867061853408815, "serum2": 0.19319253629073502}
    report = {
        "by_synth": by_synth,
        "before_after": {
            synth: {
                "milestone3_one_shot_clap": one_shot[synth],
                "analysis_by_synthesis_clap": by_synth[synth]["mean_clap_cosine"],
            }
            for synth in ("serum1", "serum2")
        },
        "held_out": held_out,
        "novel_transformed": novel,
        "novel_informational": {
            "count": len(novel),
            "mean_clap_cosine": float(
                np.mean([float(row["best_clap_cosine"]) for row in novel])
            )
            if novel
            else None,
            "mean_objective": float(np.mean([float(row["best_objective"]) for row in novel]))
            if novel
            else None,
        },
        "gate_pass": all(row["pass"] for row in by_synth.values()),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("MATCHER_GATE_SUMMARY=" + json.dumps(report, sort_keys=True))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
