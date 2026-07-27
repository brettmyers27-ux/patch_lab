#!/usr/bin/env python3
"""Create the Phase C before/after report for the real-sample benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "evaluations" / "real_samples"
THRESHOLDS = {"high": 0.90, "good": 0.80, "fair": 0.65}


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["path"])] = row
    return result


def _scores(rows: list[dict[str, Any]], side: str) -> dict[str, float]:
    values = np.asarray([float(row[f"{side}_clap"]) for row in rows])
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _tiers(rows: list[dict[str, Any]], side: str) -> dict[str, int]:
    counts = {"high": 0, "good": 0, "fair": 0, "low": 0}
    for row in rows:
        score = float(row[f"{side}_clap"])
        if score >= THRESHOLDS["high"]:
            counts["high"] += 1
        elif score >= THRESHOLDS["good"]:
            counts["good"] += 1
        elif score >= THRESHOLDS["fair"]:
            counts["fair"] += 1
        else:
            counts["low"] += 1
    return counts


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "file",
            "duration_s",
            "duration_bucket",
            "before_clap",
            "after_clap",
            "clap_delta",
            "before_stft",
            "after_stft",
            "before_note",
            "after_note",
            "note_hypotheses",
            "baseline_target",
            "baseline_result",
            "adaptive_target",
            "adaptive_result",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_ROOT / "baseline",
    )
    parser.add_argument(
        "--adaptive",
        type=Path,
        default=DEFAULT_ROOT / "adaptive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT,
    )
    args = parser.parse_args()
    baseline = _rows(args.baseline / "rows.jsonl")
    adaptive = _rows(args.adaptive / "rows.jsonl")
    shared = sorted(set(baseline) & set(adaptive))
    if set(baseline) != set(adaptive):
        raise RuntimeError(
            f"Benchmark sets differ: baseline={len(baseline)}, adaptive={len(adaptive)}, shared={len(shared)}"
        )
    rows = []
    for path in shared:
        before = baseline[path]
        after = adaptive[path]
        rows.append(
            {
                "file": before["file"],
                "path": path,
                "duration_s": before["duration_s"],
                "duration_bucket": before["duration_bucket"],
                "before_clap": before["optimized_clap"],
                "after_clap": after["optimized_clap"],
                "clap_delta": float(after["optimized_clap"])
                - float(before["optimized_clap"]),
                "before_stft": before["optimized_stft"],
                "after_stft": after["optimized_stft"],
                "before_note": before["selected_midi_note"],
                "after_note": after["selected_midi_note"],
                "note_hypotheses": after.get("note_hypotheses", []),
                "baseline_target": before["target_audition"],
                "baseline_result": before["result_audition"],
                "adaptive_target": after["target_audition"],
                "adaptive_result": after["result_audition"],
            }
        )
    by_duration = {}
    for bucket in ("<0.5s", "0.5-1.5s", "1.5-4s"):
        selected = [row for row in rows if row["duration_bucket"] == bucket]
        before = _scores(selected, "before")
        after = _scores(selected, "after")
        by_duration[bucket] = {
            "count": len(selected),
            "before_mean": before["mean"],
            "after_mean": after["mean"],
            "mean_delta": after["mean"] - before["mean"],
            "before_median": before["median"],
            "after_median": after["median"],
            "median_delta": after["median"] - before["median"],
        }
    report = {
        "count": len(rows),
        "confidence_thresholds": {
            "high": ">=0.90",
            "good": "0.80-0.90",
            "fair": "0.65-0.80",
            "low": "<0.65",
        },
        "overall": {
            "before": _scores(rows, "before"),
            "after": _scores(rows, "after"),
            "mean_delta": float(
                np.mean([float(row["clap_delta"]) for row in rows])
            ),
        },
        "by_duration": by_duration,
        "tier_counts": {
            "before": _tiers(rows, "before"),
            "after": _tiers(rows, "after"),
        },
        "ten_most_improved": [
            _compact(row)
            for row in sorted(
                rows,
                key=lambda item: float(item["clap_delta"]),
                reverse=True,
            )[:10]
        ],
        "ten_still_worst": [
            _compact(row)
            for row in sorted(
                rows,
                key=lambda item: float(item["after_clap"]),
            )[:10]
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    print(
        "REAL_SAMPLE_COMPARISON=" + json.dumps(report, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
