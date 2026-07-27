#!/usr/bin/env python3
"""Read-only aggregate verification for every Milestone 3 acceptance gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "models"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"


def _report(name: str) -> dict[str, Any]:
    path = MODEL_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _line(name: str, passed: bool, detail: str) -> bool:
    print(f"{'PASS' if passed else 'FAIL':4}  {name:28} {detail}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-deep", action="store_true")
    args = parser.parse_args()
    outcomes: list[bool] = []

    target = _report("serum2_target_report.json")
    outcomes.append(
        _line(
            "Serum 2 target vector",
            bool(target.get("gate_pass")) and int(target.get("vector_length", 0)) > 0,
            f"{target.get('preset_count')} presets; "
            f"{target.get('conceptual_field_count')} fields -> "
            f"{target.get('vector_length')} outputs",
        )
    )
    embedding = _report("milestone3_embedding_report.json")
    complete = np.load(FEATURE_DIR / "note_complete.npy", mmap_mode="r")
    embedding_ok = (
        bool(embedding.get("gate_pass"))
        and bool(np.all(complete))
        and int(embedding.get("render_count", 0)) == 39_053
        and int(embedding.get("preset_count", 0)) == 5_579
    )
    outcomes.append(
        _line(
            "Audio features",
            embedding_ok,
            f"{int(np.sum(complete)):,}/39,053 notes; {embedding.get('preset_count')} presets",
        )
    )

    similarity = _report("milestone3_similarity_report.json")
    self_gate = similarity.get("self_retrieval", {})
    outcomes.append(
        _line(
            "Combined self-retrieval",
            bool(similarity.get("combined_synth_index")) and bool(self_gate.get("pass")),
            f"{100.0 * float(self_gate.get('rate', 0.0)):.3f}% (required 99%)",
        )
    )
    octave = similarity.get("octave_generalization", {})
    outcomes.append(
        _line(
            "Octave generalization",
            int(octave.get("total", 0)) == 5_579,
            f"{100.0 * float(octave.get('rate', 0.0)):.3f}% (70% reporting target)",
        )
    )

    training = _report("milestone3_training_report.json")
    for synth in ("serum1", "serum2"):
        result = training.get("by_synth", {}).get(synth, {})
        improvement = float(result.get("improvement", -1.0))
        outcomes.append(
            _line(
                f"{synth.title()} parameter model",
                improvement >= 0.20,
                f"MAE {float(result.get('mae', 0.0)):.6f}; "
                f"baseline {float(result.get('baseline_mae', 0.0)):.6f}; "
                f"improvement {100.0 * improvement:.2f}%",
            )
        )

    roundtrip = _report("milestone3_roundtrip_report.json")
    for synth in ("serum1", "serum2"):
        result = roundtrip.get("by_synth", {}).get(synth, {})
        outcomes.append(
            _line(
                f"{synth.title()} round trip",
                bool(roundtrip.get("gate_pass"))
                and int(result.get("count", 0)) == 20
                and int(result.get("silent_renders", -1)) == 0,
                f"20-preset mean CLAP cosine "
                f"{float(result.get('mean_clap_cosine_similarity', 0.0)):.6f}; "
                f"silent {int(result.get('silent_renders', 0))}",
            )
        )

    if args.require_deep:
        synthetic = _report("milestone3_synthetic_report.json")
        outcomes.append(
            _line(
                "Deep-training augmentation",
                bool(synthetic.get("gate_pass")) and int(synthetic.get("completed", 0)) == 20_000,
                f"{int(synthetic.get('completed', 0)):,}/20,000 patches",
            )
        )

    passed = all(outcomes)
    print(f"\nMILESTONE_3={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
