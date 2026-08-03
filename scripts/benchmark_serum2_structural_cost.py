#!/usr/bin/env python3
"""Measure direct Serum 2 automation and reconstructed-state evaluation cost."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import _serum2_targets
from core.matcher import Candidate, _init_render_worker, _render_candidate
from core.platform_env import ENV
from core.plugin_host import audio_levels, make_dawdreamer_processor, render_dawdreamer_note
from core.serum2_preset import Serum2Preset
from core.serum2_state_reconstruct import (
    decode_host_template,
    load_render_state,
    reconstruct_partial_vstpreset,
)
from core.serum2_targets import decode_vector
from core.synthesis_assets import resolve_synthesis_assets


DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage3a" / "structural-cost.json"


def design_decision(ratio: float) -> str:
    if ratio < 5.0:
        return "direct-shortlist-search-viable"
    if ratio > 20.0:
        return "neural-surrogate-prerequisite"
    return "direct-search-only-with-tight-shortlists"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _levels_are_audible(audio: np.ndarray) -> tuple[float, float]:
    peak, rms = audio_levels(audio)
    if not np.isfinite(audio).all() or rms <= -75.0:
        raise RuntimeError(f"Cost-probe render was silent or non-finite: rms={rms:.3f} dBFS")
    return peak, rms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--pool-trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--parameters", type=int, default=32)
    parser.add_argument("--midi-note", type=int, default=60)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--preset-id", type=int)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.trials < 50:
        raise ValueError("Stage 3A requires at least 50 trials per timing path")
    if args.pool_trials < 50:
        raise ValueError("Stage 3A requires at least 50 four-pool structural evaluations")
    if args.parameters < 1:
        raise ValueError("--parameters must be positive")

    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    assets = resolve_synthesis_assets()
    store = _serum2_targets(assets.serum2_targets, assets.serum2_schema)
    preset_id = args.preset_id or min(store.preset_row)
    if preset_id not in store.preset_row:
        raise ValueError(f"Serum 2 preset {preset_id} is absent from the target store")
    state_path = assets.find_render_state(preset_id)
    if state_path is None:
        raise FileNotFoundError(f"No render state for Serum 2 preset {preset_id}")
    row = store.preset_row[preset_id]
    schema = json.loads(assets.serum2_schema.read_text(encoding="utf-8"))
    graph = decode_vector(store.vectors[row], schema, store.masks[row])
    template = decode_host_template(state_path.read_bytes())
    predicted = Serum2Preset(
        path=state_path,
        metadata={"presetName": "PatchLab Stage 3A cost probe"},
        data=graph,
        metadata_length=0,
        cbor_length=0,
        payload_version=0,
        compressed_length=0,
    )

    plugin = next(
        item
        for item in ENV.plugins_for("serum2")
        if item.format == "VST3" and item.hostable
    )
    engine, processor = make_dawdreamer_processor(plugin)
    load_render_state(processor, preset_id, state_path.parent)
    parameter_count = len(processor.get_parameters_description())
    selected_parameters = tuple(range(min(args.parameters, parameter_count)))

    for _ in range(args.warmup):
        for index in selected_parameters:
            processor.set_parameter(index, float(processor.get_parameter(index)))
        _levels_are_audible(
            render_dawdreamer_note(
                engine, processor, midi_note=args.midi_note, duration=args.duration
            )
        )
    automation_seconds: list[float] = []
    automation_rms: list[float] = []
    for _ in range(args.trials):
        started = time.perf_counter()
        for index in selected_parameters:
            processor.set_parameter(index, float(processor.get_parameter(index)))
        audio = render_dawdreamer_note(
            engine, processor, midi_note=args.midi_note, duration=args.duration
        )
        automation_seconds.append(time.perf_counter() - started)
        automation_rms.append(_levels_are_audible(audio)[1])

    structural_seconds: list[float] = []
    structural_rms: list[float] = []
    state_load_successes = 0
    partition_coverage = 0.0
    with tempfile.TemporaryDirectory(prefix="patchlab-stage3a-cost-") as scratch:
        output = Path(scratch) / "candidate.vstpreset"
        for _ in range(args.warmup):
            blob, _partition = reconstruct_partial_vstpreset(
                predicted, template, merge_matching_lists=True
            )
            output.write_bytes(blob)
            if processor.load_vst3_preset(str(output)) is False:
                raise RuntimeError("Serum 2 rejected a structural warmup state")
            _levels_are_audible(
                render_dawdreamer_note(
                    engine, processor, midi_note=args.midi_note, duration=args.duration
                )
            )
        for _ in range(args.trials):
            started = time.perf_counter()
            blob, partition = reconstruct_partial_vstpreset(
                predicted, template, merge_matching_lists=True
            )
            output.write_bytes(blob)
            if processor.load_vst3_preset(str(output)) is False:
                raise RuntimeError("Serum 2 rejected a reconstructed structural state")
            state_load_successes += 1
            audio = render_dawdreamer_note(
                engine, processor, midi_note=args.midi_note, duration=args.duration
            )
            structural_seconds.append(time.perf_counter() - started)
            structural_rms.append(_levels_are_audible(audio)[1])
            partition_coverage = partition.coverage

    candidate = Candidate(
        "serum2",
        preset_id,
        np.asarray(store.vectors[row], dtype=np.float32).copy(),
        np.asarray(store.masks[row], dtype=np.bool_).copy(),
        "stage3a-cost-pool",
        exact_base=False,
        midi_note=args.midi_note,
    )
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="patchlab-stage3a-pool-") as scratch:
        with context.Pool(
            4, initializer=_init_render_worker, initargs=(scratch, assets)
        ) as pool:
            warm = pool.map(
                _render_candidate,
                [(candidate, args.midi_note, args.duration)] * 4,
                chunksize=1,
            )
            if any(error for _audio, _coverage, error in warm):
                raise RuntimeError(f"Four-pool warmup failed: {warm}")
            started = time.perf_counter()
            results = pool.map(
                _render_candidate,
                [(candidate, args.midi_note, args.duration)] * args.pool_trials,
                chunksize=1,
            )
            pool_elapsed = time.perf_counter() - started
    pool_errors = [error for _audio, _coverage, error in results if error]
    if pool_errors:
        raise RuntimeError(f"Four-pool structural evaluation failed: {pool_errors[0]}")

    automation_median = statistics.median(automation_seconds)
    structural_median = statistics.median(structural_seconds)
    ratio = structural_median / automation_median
    report: dict[str, Any] = {
        "status": "complete",
        "preset_id": preset_id,
        "midi_note": args.midi_note,
        "duration_seconds": args.duration,
        "parameters_set_per_automation_trial": len(selected_parameters),
        "automation": {
            "trials": args.trials,
            "median_seconds": automation_median,
            "minimum_seconds": min(automation_seconds),
            "maximum_seconds": max(automation_seconds),
            "median_rms_dbfs": statistics.median(automation_rms),
            "seconds": automation_seconds,
        },
        "structural": {
            "trials": args.trials,
            "median_seconds": structural_median,
            "minimum_seconds": min(structural_seconds),
            "maximum_seconds": max(structural_seconds),
            "median_rms_dbfs": statistics.median(structural_rms),
            "seconds": structural_seconds,
            "state_load_successes": state_load_successes,
            "same_instance_reused": state_load_successes == args.trials,
            "reinstantiation_required": False,
            "partition_coverage": partition_coverage,
        },
        "structural_to_automation_ratio": ratio,
        "four_process_pool": {
            "evaluations": args.pool_trials,
            "elapsed_seconds": pool_elapsed,
            "evaluations_per_minute": args.pool_trials / pool_elapsed * 60.0,
            "failures": 0,
        },
        "decision_rule": {
            "ratio_below_5": "direct shortlist search is viable",
            "ratio_above_20": "a learned surrogate is a prerequisite",
            "otherwise": "direct search requires tightly bounded shortlists",
        },
        "design_decision": design_decision(ratio),
    }
    _atomic_json(args.report.expanduser().resolve(), report)
    print(
        "STAGE3A_STRUCTURAL_COST="
        + json.dumps(
            {
                "automation_median_seconds": automation_median,
                "structural_median_seconds": structural_median,
                "ratio": ratio,
                "evaluations_per_minute": report["four_process_pool"]["evaluations_per_minute"],
                "same_instance_reused": True,
                "design_decision": report["design_decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
