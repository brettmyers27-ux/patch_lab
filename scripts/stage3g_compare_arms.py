"""Create the sanitized fixed-corpus Stage 3G three-arm comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-a", type=Path, required=True)
    parser.add_argument("--arm-b", type=Path, required=True)
    parser.add_argument("--arm-c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rows(directory: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in (directory / "bam").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows[str(row["source_sha256"])] = row
    if len(rows) != 99:
        raise RuntimeError(f"Expected 99 rows under {directory}, found {len(rows)}")
    errors = [row for row in rows.values() if row.get("status") != "complete"]
    if errors:
        raise RuntimeError(f"Found {len(errors)} incomplete rows under {directory}")
    return rows


def _metrics(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    all_scores = [float(row["clap_similarity"]) for row in rows.values()]
    serum2 = [
        float(row["clap_similarity"])
        for row in rows.values()
        if row["target_synth"] == "serum2"
    ]
    if len(serum2) != 52:
        raise RuntimeError(f"Expected 52 Serum 2 scores, found {len(serum2)}")
    return {
        "completed": len(all_scores),
        "errors": 0,
        "whole_set_mean": mean(all_scores),
        "serum2_count": len(serum2),
        "serum2_mean": mean(serum2),
        "serum2_median": median(serum2),
        "serum2_minimum": min(serum2),
    }


def _paired(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    keys = sorted(
        key for key, row in left.items() if row["target_synth"] == "serum2"
    )
    deltas = [
        float(right[key]["clap_similarity"]) - float(left[key]["clap_similarity"])
        for key in keys
    ]
    return {
        "count": len(deltas),
        "mean_delta": mean(deltas),
        "improved": sum(delta > 0.0 for delta in deltas),
        "regressed": sum(delta < 0.0 for delta in deltas),
        "unchanged": sum(delta == 0.0 for delta in deltas),
    }


def _structural_timing(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = []
    route_times = []
    route_rates = []
    evaluation_fallbacks = 0
    time_fallbacks = 0
    routes_searched = 0
    target_rows = []
    serum2_rows = sorted(
        (row for row in rows.values() if row["target_synth"] == "serum2"),
        key=lambda row: str(row.get("source_name", "")).casefold(),
    )
    for target_index, row in enumerate(serum2_rows, start=1):
        if not row.get("result_path"):
            raise RuntimeError("Serum 2 structural row is missing its result path")
        result = json.loads(Path(row["result_path"]).read_text(encoding="utf-8"))
        trace = result["recommendation"]["objective_trace"]
        stages = [
            item
            for item in trace
            if item.get("stage")
            in (
                "structural-wavetable",
                "structural-fx_type",
                "structural-noise_sample",
                "structural-mod_route",
            )
        ]
        structural_seconds = sum(
            float(item.get("elapsed_s", 0.0)) for item in stages
        )
        totals.append(structural_seconds)
        narrowing = next(
            (
                item
                for item in trace
                if item.get("stage") == "structural-mod-route-narrowing"
            ),
            None,
        )
        if narrowing and narrowing.get("hierarchical_fallback"):
            evaluation_fallbacks += 1
        route = next(
            (item for item in stages if item.get("stage") == "structural-mod_route"),
            None,
        )
        if route and int(route.get("evaluated_count", 0)) > 0:
            routes_searched += 1
            route_times.append(float(route["elapsed_s"]))
            route_rates.append(float(route["evaluations_per_minute"]))
            projection = route.get("time_projection") or {}
            if projection.get("time_hierarchical_fallback"):
                time_fallbacks += 1
        projection = (route or {}).get("time_projection") or {}
        target_rows.append(
            {
                "target_index": target_index,
                "route_candidates_after_motion_narrowing": int(
                    (narrowing or {}).get("surviving_candidates", 0)
                ),
                "complete_structural_evaluations": int(
                    (narrowing or {}).get("full_structural_evaluations", 0)
                ),
                "evaluation_fit_structural_evaluations": int(
                    (narrowing or {}).get("selected_structural_evaluations", 0)
                ),
                "evaluation_hierarchical_fallback": bool(
                    (narrowing or {}).get("hierarchical_fallback", False)
                ),
                "time_fit_route_evaluations_before": int(
                    (route or {}).get("full_candidate_count", 0)
                ),
                "time_fit_route_evaluations_after": int(
                    (route or {}).get("evaluated_count", 0)
                ),
                "time_hierarchical_fallback": bool(
                    projection.get("time_hierarchical_fallback", False)
                ),
                "time_destination_groups_before": int(
                    projection.get("destination_groups_before", 0)
                ),
                "time_destination_groups_after": int(
                    projection.get("destination_groups_after", 0)
                ),
                "structural_wall_clock_s": structural_seconds,
                "within_8192_evaluations": int(
                    (narrowing or {}).get("selected_structural_evaluations", 0)
                )
                <= 8192,
                "within_900_seconds": structural_seconds <= 900.0,
            }
        )
    return {
        "targets_with_structural_timing": len(totals),
        "targets_with_routes_searched": routes_searched,
        "targets_with_evaluation_hierarchical_fallback": evaluation_fallbacks,
        "targets_with_time_hierarchical_fallback": time_fallbacks,
        "structural_wall_clock_s": {
            "mean": mean(totals),
            "median": median(totals),
            "maximum": max(totals),
        },
        "route_wall_clock_s": {
            "mean": mean(route_times),
            "median": median(route_times),
            "maximum": max(route_times),
        },
        "route_evaluations_per_minute": {
            "minimum": min(route_rates),
            "median": median(route_rates),
            "maximum": max(route_rates),
        },
        "targets": target_rows,
    }


def main() -> int:
    args = parse_args()
    arm_a = _rows(args.arm_a)
    arm_b = _rows(args.arm_b)
    arm_c = _rows(args.arm_c)
    if set(arm_a) != set(arm_b) or set(arm_a) != set(arm_c):
        raise RuntimeError("Arm membership/hash mismatch")
    payload = {
        "schema_version": 1,
        "stage": "3G",
        "seed": 20260802,
        "corpus": {"serum1_aiff": 47, "serum2_wav": 52, "total": 99},
        "serum1_reuse": {
            "rows": 47,
            "source": "adopted Stage 2B details",
            "justification": "Stage 3G changes execute only in the Serum 2 structural branch.",
        },
        "arms": {
            "a_production": _metrics(arm_a),
            "b_fx_wavetable_noise": _metrics(arm_b),
            "c_fx_wavetable_noise_routes": _metrics(arm_c),
        },
        "serum2_paired": {
            "b_vs_a": _paired(arm_a, arm_b),
            "c_vs_a": _paired(arm_a, arm_c),
            "c_vs_b": _paired(arm_b, arm_c),
        },
        "arm_c_compute": _structural_timing(arm_c),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
