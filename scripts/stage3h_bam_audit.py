#!/usr/bin/env python3
"""Stage 3H BAM reproducibility runner and repeat verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from statistics import mean, median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.match_workflow import run_match_file
from core.model_assets import configure_model_environment
from scripts.benchmark_suite import target_synth_for_name


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_audio_sha256(path: Path) -> str:
    """Hash decoded float32 samples, excluding mutable WAV container metadata."""

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    header = np.asarray([int(sample_rate), *audio.shape], dtype=np.int64).tobytes()
    samples = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    return _sha256_bytes(header + samples)


def candidate_state_sha256(path: Path) -> str:
    with np.load(path, allow_pickle=False) as payload:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(payload["vector"], dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(payload["mask"], dtype=np.bool_).tobytes())
        if "structural_overrides_json" in payload:
            digest.update(str(payload["structural_overrides_json"].item()).encode("utf-8"))
        return digest.hexdigest()


def audit_result(result_path: Path) -> dict[str, Any]:
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    recommendation = result["recommendation"]
    winner = Path(recommendation["winner_audio_path"])
    candidate = Path(recommendation["candidate_path"])
    return {
        "source_name": Path(result["source"]["path"]).name,
        "target_synth": result["target_synth"],
        "clap_similarity": float(recommendation["clap_similarity"]),
        "objective": float(recommendation["objective"]),
        "base_preset_id": int(recommendation["base_preset_id"]),
        "origin": str(recommendation["origin"]),
        "candidate_state_sha256": candidate_state_sha256(candidate),
        "decoded_audio_sha256": canonical_audio_sha256(winner),
        "wav_file_sha256": file_sha256(winner),
        "midi_note": int(result["detected"]["midi_note"]),
        "evaluations": int(recommendation["evaluations"]),
    }


def repeat_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["clap_similarity"]) for row in rows]
    return {
        "runs": len(rows),
        "score_minimum": min(scores),
        "score_maximum": max(scores),
        "score_span": max(scores) - min(scores),
        "unique_states": len({row["candidate_state_sha256"] for row in rows}),
        "unique_decoded_audio": len({row["decoded_audio_sha256"] for row in rows}),
        "unique_wav_files": len({row["wav_file_sha256"] for row in rows}),
        "unique_winners": len(
            {
                (row["base_preset_id"], row["origin"], row["candidate_state_sha256"])
                for row in rows
            }
        ),
    }


def _load_bam_rows(directory: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in Path(directory).glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[str(row["source_sha256"])] = row
    return rows


def _result_identity(row: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(str(row.get("result_path") or ""))
    if not result_path.is_file():
        return {
            "base_preset_id": None,
            "origin": row.get("origin"),
            "candidate_state_sha256": None,
            "decoded_audio_sha256": None,
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    recommendation = result.get("recommendation") or {}
    candidate = Path(str(recommendation.get("candidate_path") or ""))
    winner = Path(str(recommendation.get("winner_audio_path") or ""))
    return {
        "base_preset_id": recommendation.get("base_preset_id"),
        "origin": recommendation.get("origin"),
        "candidate_state_sha256": (
            candidate_state_sha256(candidate) if candidate.is_file() else None
        ),
        "decoded_audio_sha256": (
            canonical_audio_sha256(winner) if winner.is_file() else None
        ),
    }


def compare_bam_rows(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left = _load_bam_rows(left_dir)
    right = _load_bam_rows(right_dir)
    if set(left) != set(right):
        raise RuntimeError("BAM row memberships or source hashes differ")
    rows = []
    for source_hash in sorted(left, key=lambda key: left[key]["source_name"].casefold()):
        before = left[source_hash]
        after = right[source_hash]
        before_identity = _result_identity(before)
        after_identity = _result_identity(after)
        delta = float(after["clap_similarity"]) - float(before["clap_similarity"])
        rows.append(
            {
                "target": str(before["source_name"]),
                "source_sha256": source_hash,
                "serum_version": target_synth_for_name(str(before["source_name"])),
                "historical_bam": float(before["clap_similarity"]),
                "stage3g_bam": float(after["clap_similarity"]),
                "delta": delta,
                "historical": before_identity,
                "stage3g": after_identity,
                "same_base_preset": (
                    before_identity["base_preset_id"]
                    == after_identity["base_preset_id"]
                    if before_identity["base_preset_id"] is not None
                    and after_identity["base_preset_id"] is not None
                    else None
                ),
                "same_candidate_state": (
                    before_identity["candidate_state_sha256"]
                    == after_identity["candidate_state_sha256"]
                    if before_identity["candidate_state_sha256"] is not None
                    and after_identity["candidate_state_sha256"] is not None
                    else None
                ),
                "same_decoded_audio": (
                    before_identity["decoded_audio_sha256"]
                    == after_identity["decoded_audio_sha256"]
                    if before_identity["decoded_audio_sha256"] is not None
                    and after_identity["decoded_audio_sha256"] is not None
                    else None
                ),
            }
        )
    changed = [row for row in rows if abs(row["delta"]) > 1e-12]
    return {
        "count": len(rows),
        "exactly_unchanged": len(rows) - len(changed),
        "changed": len(changed),
        "mean_absolute_delta": mean(abs(row["delta"]) for row in rows),
        "largest_positive_delta": max(row["delta"] for row in rows),
        "largest_negative_delta": min(row["delta"] for row in rows),
        "ten_largest_absolute_changes": [
            {"target": row["target"], "delta": row["delta"]}
            for row in sorted(rows, key=lambda item: abs(item["delta"]), reverse=True)[:10]
        ],
        "rows": rows,
    }


def _aggregate(directory: Path) -> dict[str, Any]:
    rows = list(_load_bam_rows(directory).values())
    serum2 = [
        row
        for row in rows
        if target_synth_for_name(str(row["source_name"])) == "serum2"
    ]
    scores = [float(row["clap_similarity"]) for row in rows]
    serum2_scores = [float(row["clap_similarity"]) for row in serum2]
    return {
        "total_count": len(rows),
        "serum1_count": len(rows) - len(serum2),
        "serum2_count": len(serum2),
        "whole_set_mean": mean(scores),
        "serum2_mean": mean(serum2_scores),
        "serum2_median": median(serum2_scores),
        "serum2_minimum": min(serum2_scores),
    }


def historical_comparison(args: argparse.Namespace) -> int:
    arm_a = compare_bam_rows(args.stage2b_arm_a, args.stage3g_arm_a)
    arm_b = compare_bam_rows(args.stage3d_arm_b, args.stage3g_arm_b)
    stage3g = {
        "arm_a": _aggregate(args.stage3g_arm_a),
        "arm_b": _aggregate(args.stage3g_arm_b),
        "arm_c": _aggregate(args.stage3g_arm_c),
    }
    b_rows = _load_bam_rows(args.stage3g_arm_b)
    c_rows = _load_bam_rows(args.stage3g_arm_c)
    deltas = [
        float(c_rows[key]["clap_similarity"]) - float(b_rows[key]["clap_similarity"])
        for key in b_rows
        if target_synth_for_name(str(b_rows[key]["source_name"])) == "serum2"
    ]
    payload = {
        "schema_version": 1,
        "stage": "3H",
        "stage3g_recalculated": stage3g,
        "stage3g_arm_c_vs_b": {
            "count": len(deltas),
            "paired_mean_delta": mean(deltas),
            "improved": sum(delta > 1e-12 for delta in deltas),
            "regressed": sum(delta < -1e-12 for delta in deltas),
            "unchanged": sum(abs(delta) <= 1e-12 for delta in deltas),
        },
        "stage2b_vs_stage3g_arm_a": arm_a,
        "stage3d_vs_stage3g_arm_b": arm_b,
    }
    _write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "stage3g": stage3g}, sort_keys=True))
    return 0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_target(args: argparse.Namespace) -> int:
    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    os.environ["PATCHLAB_SERUM2_STRUCTURAL_SEARCH"] = "0"
    os.environ["PATCHLAB_SERUM2_STRUCTURAL_ROUTES"] = "0"
    configure_model_environment()
    result = run_match_file(
        args.source,
        target_synth="serum2",
        budget="balanced",
        session_root=args.session_root,
        matcher_processes=args.matcher_processes,
        deterministic_render_dispatch=args.deterministic_render_dispatch,
    )
    row = audit_result(result)
    row["matcher_processes"] = args.matcher_processes
    row["render_dispatch"] = (
        "deterministic-dedicated-workers-v1"
        if args.deterministic_render_dispatch
        else "legacy-shared-pool"
    )
    _write_json(args.output, row)
    print(json.dumps(row, sort_keys=True))
    return 0


def verify_repeats(args: argparse.Namespace) -> int:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_dir.glob("*.json"))
    ]
    if not rows:
        raise RuntimeError(f"No repeat JSON files found in {args.input_dir}")
    summary = repeat_summary(rows)
    if args.output:
        _write_json(args.output, {"summary": summary, "rows": rows})
    print(json.dumps(summary, sort_keys=True))
    stable = (
        summary["score_span"] <= args.max_score_span
        and summary["unique_states"] == 1
        and summary["unique_decoded_audio"] == 1
    )
    return 0 if stable else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-target")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--session-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--matcher-processes", type=int, default=4)
    run.add_argument("--deterministic-render-dispatch", action="store_true")
    run.set_defaults(function=run_target)
    verify = subparsers.add_parser("verify-repeats")
    verify.add_argument("--input-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    verify.add_argument("--max-score-span", type=float, default=0.0)
    verify.set_defaults(function=verify_repeats)
    compare = subparsers.add_parser("historical-comparison")
    compare.add_argument("--stage2b-arm-a", type=Path, required=True)
    compare.add_argument("--stage3d-arm-b", type=Path, required=True)
    compare.add_argument("--stage3g-arm-a", type=Path, required=True)
    compare.add_argument("--stage3g-arm-b", type=Path, required=True)
    compare.add_argument("--stage3g-arm-c", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(function=historical_comparison)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if getattr(args, "matcher_processes", 1) <= 0:
        raise ValueError("--matcher-processes must be positive")
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
