#!/usr/bin/env python3
"""Read-only verifier for the analysis-by-synthesis upgrade gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "models"


def _load(name: str) -> dict[str, Any]:
    path = MODEL_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _row(label: str, passed: bool, detail: str) -> bool:
    print(f"{'PASS' if passed else 'FAIL':4}  {label:<29} {detail}")
    return passed


def main() -> int:
    checks: list[bool] = []
    try:
        perturb = _load("milestone3_perturbation_report.json")
        neighbor = _load("delta_neighbor_report.json")
        delta = _load("delta_training_report.json")
        search = _load("analysis_by_synthesis_gate_report.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"FAIL  reports                       {exc}")
        return 1

    for synth in ("serum1", "serum2"):
        row = perturb[synth]
        passed = (
            bool(row["gate_pass"])
            and int(row["completed"]) == 20_000
            and float(row["discard_rate"]) < 0.15
            and int(row["spot_render_count"]) == 20
            and bool(row["plausible_distribution"])
        )
        checks.append(
            _row(
                f"{synth} perturbations",
                passed,
                f"20,000; discard={100 * row['discard_rate']:.3f}%; "
                f"enum={100 * row['enum_change_rate']:.2f}%",
            )
        )

    no_self = all(
        int(neighbor["by_synth"][synth]["self_neighbors"]) == 0
        for synth in ("serum1", "serum2")
    )
    checks.append(_row("leakage-safe neighbors", no_self, "zero self-neighbors"))

    # This stage is diagnostic: it is retained as an auxiliary seed even though
    # its validation MAE did not beat the already accepted absolute model.
    for synth in ("serum1", "serum2"):
        row = delta["validation"][synth]
        reported = all(
            key in row
            for key in ("mae", "milestone3_absolute_mae", "improvement_vs_milestone3")
        )
        checks.append(
            _row(
                f"{synth} delta diagnostic",
                reported,
                f"MAE={row['mae']:.6f} vs absolute={row['milestone3_absolute_mae']:.6f} "
                f"({100 * row['improvement_vs_milestone3']:+.2f}%)",
            )
        )

    for synth, threshold in (("serum1", 0.90), ("serum2", 0.80)):
        row = search["by_synth"][synth]
        traces = search["held_out"][synth]
        passed = (
            int(row["count"]) == 20
            and float(row["mean_clap_cosine"]) >= threshold
            and bool(row["pass"])
            and len(traces) == 20
            and all(item.get("objective_trace") for item in traces)
            and all(int(item["evaluations"]) <= 300 for item in traces)
        )
        checks.append(
            _row(
                f"{synth} synthesis gate",
                passed,
                f"mean={row['mean_clap_cosine']:.6f} (>= {threshold:.2f}); "
                f"own wins={row['target_own_preset_wins']}/20; "
                f"median={row['median_wall_clock_s']:.2f}s",
            )
        )

    novel = search.get("novel_transformed", [])
    novel_summary = search.get("novel_informational", {})
    novel_pass = (
        len(novel) == 10
        and int(novel_summary.get("count", 0)) == 10
        and all(item.get("objective_trace") for item in novel)
        and all(int(item["evaluations"]) <= 300 for item in novel)
        and any(len(item["objective_trace"]) > 1 for item in novel)
    )
    checks.append(
        _row(
            "transformed-audio set",
            novel_pass,
            f"10/10; mean CLAP={novel_summary.get('mean_clap_cosine', float('nan')):.6f}",
        )
    )

    overall = bool(search.get("gate_pass")) and all(checks)
    print(f"\n{'PASS' if overall else 'FAIL'}  analysis-by-synthesis upgrade")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
