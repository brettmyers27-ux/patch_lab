#!/usr/bin/env python3
"""Benchmark Patch Lab against a folder of real short production samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_input import SUPPORTED_AUDIO_SUFFIXES, decode_audio_file
from core.features import CLAP_SAMPLE_RATE
from core.matcher import (
    AnalysisBySynthesisMatcher,
    SearchConfig,
    prepare_query_audio,
)


DEFAULT_INPUT = Path.home() / "Music" / "Samples" / "BAM"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluations" / "real_samples"
MAX_DURATION_S = 4.0
CSV_FIELDS = (
    "file",
    "path",
    "duration_s",
    "duration_bucket",
    "detected_midi_note",
    "detected_hz",
    "pyin_confidence",
    "pitch_confidence_bucket",
    "unpitched_fallback",
    "onset_count",
    "sub_bass_fraction",
    "sub_bass_bucket",
    "top5_retrieval_scores",
    "top5_retrieval_preset_ids",
    "note_hypotheses",
    "selected_midi_note",
    "best_base_preset_id",
    "best_origin",
    "best_origin_class",
    "optimized_clap",
    "optimized_stft",
    "optimized_objective",
    "stft_weight",
    "clap_weight",
    "comparison_duration_s",
    "evaluations",
    "wall_clock_s",
    "target_audition",
    "result_audition",
)


def _audio_files(folder: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_AUDIO_SUFFIXES
        ),
        key=lambda value: str(value).casefold(),
    )


def _duration(path: Path) -> float:
    return float(sf.info(path).duration)


def _duration_bucket(duration_s: float) -> str:
    if duration_s < 0.5:
        return "<0.5s"
    if duration_s < 1.5:
        return "0.5-1.5s"
    return "1.5-4s"


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.35:
        return "low <0.35"
    if confidence < 0.70:
        return "medium 0.35-0.70"
    return "high >=0.70"


def _sub_bass_bucket(fraction: float) -> str:
    if fraction < 0.25:
        return "low <0.25"
    if fraction < 0.60:
        return "medium 0.25-0.60"
    return "high >=0.60"


def _pitch_diagnostic(
    audio: np.ndarray, sample_rate: int
) -> tuple[int, float | None, float, bool]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if (
        len(values) < 2048
        or float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) < 1e-4
    ):
        return 60, None, 0.0, True
    f0, voiced, probabilities = librosa.pyin(
        values,
        fmin=float(librosa.note_to_hz("C1")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sample_rate,
        frame_length=2048,
    )
    usable = np.isfinite(f0) & voiced
    if int(np.count_nonzero(usable)) < 3:
        confidence = float(
            np.nanmedian(probabilities[np.isfinite(probabilities)])
        ) if np.any(np.isfinite(probabilities)) else 0.0
        return 60, None, confidence, True
    hz = float(np.median(f0[usable]))
    confidence = float(np.median(probabilities[usable]))
    midi = int(np.clip(round(float(librosa.hz_to_midi(hz))), 24, 96))
    return midi, hz, confidence, False


def _onset_count(audio: np.ndarray, sample_rate: int) -> int:
    onsets = librosa.onset.onset_detect(
        y=np.asarray(audio, dtype=np.float32),
        sr=sample_rate,
        units="frames",
        backtrack=False,
    )
    return int(len(onsets))


def _sub_bass_fraction(audio: np.ndarray, sample_rate: int) -> float:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(values) < 2:
        return 0.0
    values = values - float(np.mean(values))
    spectrum = np.fft.rfft(values)
    energy = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(len(values), 1.0 / sample_rate)
    total = float(np.sum(energy[frequencies >= 20.0]))
    if total <= 1e-20:
        return 0.0
    return float(
        np.sum(energy[(frequencies >= 20.0) & (frequencies < 100.0)])
        / total
    )


def _preprocessed_target(
    audio: np.ndarray,
    sample_rate: int,
    *,
    adaptive: bool,
) -> np.ndarray:
    values, _duration = prepare_query_audio(
        audio,
        sample_rate,
        adaptive=adaptive,
    )
    return values


def _audition_stem(path: Path, relative: Path) -> str:
    raw = relative.with_suffix("").as_posix()
    safe = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in raw
    ).strip("_")
    safe = "_".join(part for part in safe.split("_") if part)[:96]
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{safe or 'sample'}_{digest}"


def _write_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    clap = np.asarray([float(row["optimized_clap"]) for row in rows])
    stft = np.asarray([float(row["optimized_stft"]) for row in rows])
    return {
        "count": len(rows),
        "mean_clap": float(np.mean(clap)),
        "median_clap": float(np.median(clap)),
        "p10_clap": float(np.quantile(clap, 0.10)),
        "p90_clap": float(np.quantile(clap, 0.90)),
        "mean_stft": float(np.mean(stft)),
        "failure_below_0_65": int(np.count_nonzero(clap < 0.65)),
        "above_0_80": int(np.count_nonzero(clap >= 0.80)),
        "above_0_90": int(np.count_nonzero(clap >= 0.90)),
    }


def _grouped(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    values = sorted({str(row[key]) for row in rows})
    return {
        value: _numeric_summary(
            [row for row in rows if str(row[key]) == value]
        )
        for value in values
    }


def _correlation(
    rows: list[dict[str, Any]], key: str
) -> float | None:
    if len(rows) < 3:
        return None
    left = np.asarray([float(row[key]) for row in rows])
    right = np.asarray([float(row["optimized_clap"]) for row in rows])
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _tier_counts(
    rows: list[dict[str, Any]],
    thresholds: tuple[float, float, float] = (0.65, 0.80, 0.90),
) -> dict[str, int]:
    low, good, high = thresholds
    counts = Counter()
    for row in rows:
        score = float(row["optimized_clap"])
        if score >= high:
            counts["high"] += 1
        elif score >= good:
            counts["good"] += 1
        elif score >= low:
            counts["fair"] += 1
        else:
            counts["low"] += 1
    return {key: counts[key] for key in ("high", "good", "fair", "low")}


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], ensure_ascii=False)
                        if isinstance(row.get(key), (list, dict))
                        else row.get(key)
                    )
                    for key in CSV_FIELDS
                }
            )


def _summary(
    rows: list[dict[str, Any]],
    *,
    input_folder: Path,
    output: Path,
    behavior: str,
    all_files: int,
    skipped_long: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    budget: int,
    target_synth: str,
) -> dict[str, Any]:
    return {
        "behavior": behavior,
        "input_folder": str(input_folder),
        "output_folder": str(output),
        "maximum_duration_s": MAX_DURATION_S,
        "all_supported_files": all_files,
        "processed_short_files": len(rows),
        "skipped_over_4s_count": len(skipped_long),
        "skipped_over_4s": skipped_long,
        "error_count": len(errors),
        "errors": errors,
        "search_budget": budget,
        "target_synth": target_synth,
        "overall": _numeric_summary(rows),
        "confidence_tiers_reference": {
            "high": ">=0.90",
            "good": "0.80-0.90",
            "fair": "0.65-0.80",
            "low": "<0.65",
            "counts": _tier_counts(rows),
        },
        "by_duration": _grouped(rows, "duration_bucket"),
        "by_pitch_confidence": _grouped(
            rows, "pitch_confidence_bucket"
        ),
        "by_sub_bass_fraction": _grouped(rows, "sub_bass_bucket"),
        "pearson_score_correlations": {
            "duration_s": _correlation(rows, "duration_s"),
            "pyin_confidence": _correlation(rows, "pyin_confidence"),
            "sub_bass_fraction": _correlation(rows, "sub_bass_fraction"),
        },
        "ten_worst": sorted(
            rows, key=lambda row: float(row["optimized_clap"])
        )[:10],
        "median_wall_clock_s": (
            float(np.median([float(row["wall_clock_s"]) for row in rows]))
            if rows
            else None
        ),
        "total_wall_clock_s": float(
            np.sum([float(row["wall_clock_s"]) for row in rows])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT / "baseline",
    )
    parser.add_argument(
        "--behavior",
        choices=("baseline", "adaptive"),
        default="baseline",
    )
    parser.add_argument("--budget", type=int, default=51)
    parser.add_argument(
        "--target-synth",
        choices=("serum1", "serum2"),
        default="serum2",
    )
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    adaptive = args.behavior == "adaptive"

    input_folder = args.folder.expanduser().resolve()
    output = args.output.expanduser().resolve()
    auditions = output / "auditions"
    output.mkdir(parents=True, exist_ok=True)
    auditions.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.jsonl"
    errors_path = output / "errors.jsonl"
    if args.fresh:
        rows_path.unlink(missing_ok=True)
        errors_path.unlink(missing_ok=True)

    files = _audio_files(input_folder)
    duration_rows: list[tuple[Path, float]] = []
    skipped_long: list[dict[str, Any]] = []
    metadata_errors: list[dict[str, Any]] = []
    for path in files:
        try:
            duration_s = _duration(path)
        except Exception as exc:
            metadata_errors.append(
                {
                    "path": str(path),
                    "phase": "metadata",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if duration_s > MAX_DURATION_S:
            skipped_long.append(
                {"path": str(path), "duration_s": duration_s}
            )
        else:
            duration_rows.append((path, duration_s))
    if args.limit is not None:
        duration_rows = duration_rows[: max(args.limit, 0)]

    completed_rows = _load_jsonl(rows_path)
    completed = {str(row["path"]) for row in completed_rows}
    matcher = AnalysisBySynthesisMatcher(processes=args.processes)
    try:
        for index, (path, duration_s) in enumerate(duration_rows, start=1):
            if str(path) in completed:
                print(
                    f"REAL_SAMPLE_PROGRESS={index}/{len(duration_rows)} resumed {path.name}",
                    flush=True,
                )
                continue
            started = time.monotonic()
            try:
                decoded = decode_audio_file(path, maximum_s=MAX_DURATION_S)
                midi, hz, pitch_confidence, unpitched = _pitch_diagnostic(
                    decoded.mono, decoded.sample_rate
                )
                onset_count = _onset_count(
                    decoded.mono, decoded.sample_rate
                )
                sub_bass = _sub_bass_fraction(
                    decoded.mono, decoded.sample_rate
                )
                target = _preprocessed_target(
                    decoded.mono,
                    decoded.sample_rate,
                    adaptive=adaptive,
                )
                embedding = matcher.query_embedding(
                    decoded.mono,
                    decoded.sample_rate,
                    adaptive_preprocessing=adaptive,
                )
                retrieval = matcher.retrieve_existing(embedding, 5)
                result = matcher.match(
                    decoded.mono,
                    decoded.sample_rate,
                    synth_hint=args.target_synth,
                    config=SearchConfig(
                        max_evaluations=args.budget,
                        max_seconds=30.0,
                        adaptive_preprocessing=adaptive,
                    ),
                    target_embedding=embedding,
                )
                relative = path.relative_to(input_folder)
                stem = _audition_stem(path, relative)
                target_path = auditions / f"{stem}_target.wav"
                result_path = auditions / f"{stem}_result.wav"
                sf.write(
                    target_path,
                    target,
                    CLAP_SAMPLE_RATE,
                    subtype="FLOAT",
                    format="WAV",
                )
                sf.write(
                    result_path,
                    np.asarray(result.best.waveform, dtype=np.float32),
                    CLAP_SAMPLE_RATE,
                    subtype="FLOAT",
                    format="WAV",
                )
                row = {
                    "file": path.name,
                    "path": str(path),
                    "duration_s": duration_s,
                    "duration_bucket": _duration_bucket(duration_s),
                    "detected_midi_note": midi,
                    "detected_hz": hz,
                    "pyin_confidence": pitch_confidence,
                    "pitch_confidence_bucket": _confidence_bucket(
                        pitch_confidence
                    ),
                    "unpitched_fallback": unpitched,
                    "onset_count": onset_count,
                    "sub_bass_fraction": sub_bass,
                    "sub_bass_bucket": _sub_bass_bucket(sub_bass),
                    "top5_retrieval_scores": [
                        float(score) for _preset_id, score in retrieval
                    ],
                    "top5_retrieval_preset_ids": [
                        int(preset_id) for preset_id, _score in retrieval
                    ],
                    "note_hypotheses": list(result.note_hypotheses),
                    "selected_midi_note": result.midi_note,
                    "best_base_preset_id": result.best.base_preset_id,
                    "best_origin": result.best.origin,
                    "best_origin_class": (
                        "retrieved"
                        if result.best.exact_base
                        else "optimized"
                    ),
                    "optimized_clap": result.best.clap_cosine,
                    "optimized_stft": result.best.stft_loss,
                    "optimized_objective": result.best.objective,
                    "stft_weight": result.stft_weight,
                    "clap_weight": result.clap_weight,
                    "comparison_duration_s": result.comparison_duration_s,
                    "evaluations": result.evaluations,
                    "wall_clock_s": time.monotonic() - started,
                    "target_audition": str(target_path),
                    "result_audition": str(result_path),
                }
                _write_jsonl(rows_path, row)
                completed_rows.append(row)
            except Exception as exc:
                error = {
                    "path": str(path),
                    "phase": "match",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _write_jsonl(errors_path, error)
                print(
                    "REAL_SAMPLE_ERROR=" + json.dumps(error),
                    flush=True,
                )
            print(
                f"REAL_SAMPLE_PROGRESS={index}/{len(duration_rows)} {path.name}",
                flush=True,
            )
    finally:
        matcher.close()

    rows = _load_jsonl(rows_path)
    errors = metadata_errors + _load_jsonl(errors_path)
    summary = _summary(
        rows,
        input_folder=input_folder,
        output=output,
        behavior=args.behavior,
        all_files=len(files),
        skipped_long=skipped_long,
        errors=errors,
        budget=args.budget,
        target_synth=args.target_synth,
    )
    _write_csv(output / "per_file.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "REAL_SAMPLE_SUMMARY="
        + json.dumps(
            {
                key: summary[key]
                for key in (
                    "behavior",
                    "target_synth",
                    "all_supported_files",
                    "processed_short_files",
                    "skipped_over_4s_count",
                    "error_count",
                    "overall",
                    "by_duration",
                    "by_pitch_confidence",
                    "by_sub_bass_fraction",
                    "pearson_score_correlations",
                    "median_wall_clock_s",
                    "total_wall_clock_s",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
