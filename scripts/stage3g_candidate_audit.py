"""Reproduce the sanitized Stage 3G per-target structural workload audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.matcher import _state_file
from core.serum2_state_reconstruct import decode_host_template
from core.structural_search import (
    discover_structural_fields,
    fit_mod_route_ids,
    load_search_policy,
    load_vocabulary,
)
from core.synthesis_assets import resolve_synthesis_assets


def _trace(result: dict[str, Any], stage: str) -> dict[str, Any]:
    for row in result["recommendation"]["objective_trace"]:
        if row.get("stage") == stage:
            return row
    raise RuntimeError(f"Missing {stage} trace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage3d-details",
        type=Path,
        default=PROJECT_ROOT / "data" / "stage3d" / "benchmark-b" / "bam",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vocabulary = load_vocabulary(
        PROJECT_ROOT / "data" / "models" / "serum2_structural_space.json"
    )
    policy = load_search_policy(
        PROJECT_ROOT / "data" / "models" / "serum2_structural_search_policy.json"
    )
    permitted = set(policy.get("allowed_ids", {}).get("mod_route", ()))
    routes = [
        row
        for row in vocabulary.get("categories", {})
        .get("mod_route", {})
        .get("entries", ())
        if not permitted or str(row.get("id", "")) in permitted
    ]
    assets = resolve_synthesis_assets()
    private_rows = []
    for detail_path in args.stage3d_details.glob("*.json"):
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        if detail.get("target_synth") == "serum2" and detail.get("status") == "complete":
            private_rows.append(detail)
    private_rows.sort(key=lambda row: str(row.get("source_name", "")).casefold())

    rows = []
    for target_index, detail in enumerate(private_rows, start=1):
        result = json.loads(Path(detail["result_path"]).read_text(encoding="utf-8"))
        recommendation = result["recommendation"]
        template = decode_host_template(
            _state_file(assets, int(recommendation["base_preset_id"])).read_bytes()
        )
        fields = discover_structural_fields(template.component.data)
        wavetable = int(_trace(result, "structural-wavetable")["candidate_count"])
        fx = int(_trace(result, "structural-fx_type")["candidate_count"])
        noise = int(_trace(result, "structural-noise_sample")["candidate_count"])
        narrowing = _trace(result, "structural-mod-route-narrowing")
        selected, fit = fit_mod_route_ids(
            routes,
            narrowing["movement"],
            field_count=len(fields["mod_route"]),
            non_route_evaluations=wavetable + fx + noise,
        )
        rows.append(
            {
                "target_index": target_index,
                "wavetable_evaluations": wavetable,
                "fx_evaluations": fx,
                "noise_evaluations": noise,
                "route_candidates_before_motion_narrowing": int(
                    narrowing["input_candidates"]
                ),
                "route_candidates_after_motion_narrowing": int(
                    narrowing["surviving_candidates"]
                ),
                "route_fields": len(fields["mod_route"]),
                "projected_complete_route_evaluations": int(
                    fit["full_route_evaluations"]
                ),
                "projected_complete_structural_evaluations": int(
                    fit["full_structural_evaluations"]
                ),
                "selected_route_candidates_after_evaluation_fit": len(selected),
                "selected_route_evaluations_after_evaluation_fit": int(
                    fit["selected_route_evaluations"]
                ),
                "selected_structural_evaluations_after_evaluation_fit": int(
                    fit["selected_structural_evaluations"]
                ),
                "structural_evaluation_budget": int(fit["structural_budget"]),
                "evaluation_hierarchical_fallback": bool(
                    fit["hierarchical_fallback"]
                ),
                "destination_groups_before": int(
                    fit["destination_groups_before"]
                ),
                "destination_groups_after": int(fit["destination_groups_after"]),
            }
        )
    if len(rows) != 52:
        raise RuntimeError(f"Expected 52 Serum 2 rows, found {len(rows)}")

    complete_totals = [row["projected_complete_structural_evaluations"] for row in rows]
    selected_totals = [
        row["selected_structural_evaluations_after_evaluation_fit"] for row in rows
    ]
    payload = {
        "schema_version": 1,
        "stage": "3G",
        "seed": 20260802,
        "target_count": len(rows),
        "source": "Stage 3D traces plus unchanged clean Stage 3D candidate/index artifacts",
        "summary": {
            "route_candidates_before_motion_narrowing": 2485,
            "route_candidates_after_motion_narrowing_min": min(
                row["route_candidates_after_motion_narrowing"] for row in rows
            ),
            "route_candidates_after_motion_narrowing_median": median(
                row["route_candidates_after_motion_narrowing"] for row in rows
            ),
            "route_candidates_after_motion_narrowing_max": max(
                row["route_candidates_after_motion_narrowing"] for row in rows
            ),
            "projected_complete_structural_evaluations_min": min(complete_totals),
            "projected_complete_structural_evaluations_median": median(
                complete_totals
            ),
            "projected_complete_structural_evaluations_max": max(complete_totals),
            "selected_structural_evaluations_min": min(selected_totals),
            "selected_structural_evaluations_median": median(selected_totals),
            "selected_structural_evaluations_max": max(selected_totals),
            "targets_with_complete_workload_at_or_below_4096": sum(
                value <= 4096 for value in complete_totals
            ),
            "targets_with_complete_workload_at_or_below_8192": sum(
                value <= 8192 for value in complete_totals
            ),
            "targets_requiring_evaluation_hierarchical_fallback": sum(
                row["evaluation_hierarchical_fallback"] for row in rows
            ),
        },
        "targets": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
