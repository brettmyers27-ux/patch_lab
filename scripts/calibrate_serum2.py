#!/usr/bin/env python3
"""Build the one-time Serum 2 normalized-to-display calibration table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.param_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationTable,
    calibration_stats,
    canonical_display,
    classify_samples,
)
from core.platform_env import ENV
from core.plugin_host import (
    dawdreamer_parameter_display,
    dump_dawdreamer_parameters,
    make_dawdreamer_processor,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "serum2_param_calibration.json"


def sample_parameter(processor: Any, index: int, base_steps: int) -> tuple[list[list[object]], list[str]]:
    errors: list[str] = []
    observations: list[list[object]] = []

    def observe(requested: float) -> tuple[float, str] | None:
        try:
            processor.set_parameter(index, requested)
            actual = float(processor.get_parameter(index))
            display = dawdreamer_parameter_display(processor, index, actual)
        except Exception as exc:
            errors.append(f"{requested:.9f}: {exc!r}")
            return None
        return actual, display

    for step in range(base_steps + 1):
        observed = observe(step / base_steps)
        if observed is not None:
            observations.append([observed[0], observed[1]])

    # Refine display transitions only for genuinely stepped/enum controls.
    if classify_samples(observations) == "stepped":
        refinements: list[list[object]] = []
        for left, right in zip(observations, observations[1:]):
            left_text = canonical_display(left[1])
            right_text = canonical_display(right[1])
            if left_text == right_text:
                continue
            low, high = float(left[0]), float(right[0])
            low_text = left_text
            for _ in range(10):
                midpoint = (low + high) / 2.0
                observed = observe(midpoint)
                if observed is None:
                    break
                actual, display = observed
                refinements.append([actual, display])
                if canonical_display(display) == low_text:
                    low = midpoint
                else:
                    high = midpoint
        observations.extend(refinements)

    unique: dict[tuple[float, str], list[object]] = {}
    for normalized, display in observations:
        key = (round(float(normalized), 9), str(display))
        unique[key] = [float(normalized), str(display)]
    return sorted(unique.values(), key=lambda sample: (float(sample[0]), str(sample[1]))), errors


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def build(output: Path, base_steps: int) -> dict[str, Any]:
    candidates = [item for item in ENV.plugins_for("serum2") if item.format == "VST3"]
    if not candidates:
        raise RuntimeError("No Serum 2 VST3 candidate is available")
    candidate = candidates[0]
    _engine, processor = make_dawdreamer_processor(candidate)
    initial = dump_dawdreamer_parameters(processor)
    initial_by_index = {item.index: item for item in initial}
    parameter_signature = hashlib.sha256(
        "\n".join(f"{item.index}\0{item.name}" for item in initial).encode("utf-8")
    ).hexdigest()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for position, parameter in enumerate(initial, start=1):
        samples, errors = sample_parameter(processor, parameter.index, base_steps)
        try:
            processor.set_parameter(parameter.index, parameter.norm_value)
        except Exception as exc:
            errors.append(f"restore: {exc!r}")
        entry = {
            "index": parameter.index,
            "name": parameter.name,
            "initial_normalized": parameter.norm_value,
            "initial_display": parameter.display_value,
            "kind": classify_samples(samples),
            "unique_displays": len({canonical_display(item[1]) for item in samples}),
            "samples": samples,
            "errors": errors,
        }
        grouped[parameter.name].append(entry)
        if position == 1 or position % 50 == 0 or position == len(initial):
            print(f"CALIBRATION_PROGRESS={position}/{len(initial)}", flush=True)

    # Restore the complete init vector once more in case any parameter was coupled.
    restore_errors: list[str] = []
    for index, parameter in initial_by_index.items():
        try:
            processor.set_parameter(index, parameter.norm_value)
        except Exception as exc:
            restore_errors.append(f"{index} {parameter.name}: {exc!r}")

    payload: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "synth": "serum2",
        "plugin": {
            "format": candidate.format,
            "path": str(candidate.path),
            "size": candidate.path.stat().st_size,
            "mtime_ns": candidate.path.stat().st_mtime_ns,
            "parameter_signature_sha256": parameter_signature,
            "parameter_count": len(initial),
        },
        "base_steps": base_steps,
        "parameters": dict(grouped),
        "restore_errors": restore_errors,
    }
    payload["stats"] = calibration_stats(payload)
    write_json(output, payload)
    return payload


def validate(path: Path) -> dict[str, Any]:
    table = CalibrationTable.load(path)
    payload = table.payload
    stats = calibration_stats(payload)
    expected = int(payload["stats"]["parameters"])
    if stats["parameters"] != expected:
        raise RuntimeError(f"Parameter count mismatch: {stats['parameters']} != {expected}")
    if payload.get("restore_errors"):
        raise RuntimeError(f"Restore errors: {payload['restore_errors'][:5]}")

    probes = [
        ("Mono Toggle", "On", None),
        ("A Loop Mode", "Tailed", None),
        ("A WT Pos", 128, None),
        ("Porta Time", "120 ms", None),
        ("Main Vol", "50%", None),
    ]
    results = []
    for name, target, index in probes:
        match = table.inverse(name, target, index)
        if not math.isfinite(match.normalized) or not 0.0 <= match.normalized <= 1.0:
            raise RuntimeError(f"Invalid inverse result for {name}: {match}")
        results.append(
            {
                "name": name,
                "target": target,
                "normalized": match.normalized,
                "display": match.display,
                "method": match.method,
                "score": match.score,
            }
        )
    return {"stats": stats, "inverse_probes": results, "bytes": path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    reuse = args.output.is_file() and not args.force
    if not args.validate_only and not reuse:
        payload = build(args.output, args.steps)
        print("CALIBRATION_STATS=" + json.dumps(payload["stats"], sort_keys=True))
        print("CALIBRATION_CACHE=BUILT")
    elif not args.validate_only:
        print("CALIBRATION_CACHE=REUSED")
    report = validate(args.output)
    print("CALIBRATION_VALIDATION=" + json.dumps(report, sort_keys=True))
    print("STEP_2A_GATE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
