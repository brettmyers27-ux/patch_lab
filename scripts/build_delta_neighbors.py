#!/usr/bin/env python3
"""Build deterministic nearest-real-preset neighbors for delta training."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import FEATURE_DIR


OUTPUT = FEATURE_DIR / "delta_neighbors.npz"
REPORT = PROJECT_ROOT / "data" / "models" / "delta_neighbor_report.json"


def _nearest(
    query: np.ndarray, candidates: np.ndarray, query_ids: np.ndarray, candidate_ids: np.ndarray
) -> np.ndarray:
    result = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), 256):
        stop = min(start + 256, len(query))
        scores = np.asarray(query[start:stop]) @ np.asarray(candidates).T
        for row, preset_id in enumerate(query_ids[start:stop]):
            scores[row, candidate_ids == preset_id] = -np.inf
        result[start:stop] = candidate_ids[np.argmax(scores, axis=1)]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    feature_dir = args.feature_dir.expanduser().resolve()
    output = (args.output or feature_dir / "delta_neighbors.npz").expanduser().resolve()
    report_path = (args.report or REPORT).expanduser().resolve()
    manifest = np.load(feature_dir / "preset_manifest.npz")
    embeddings = np.load(feature_dir / "preset_embeddings.npy", mmap_mode="r")
    preset_ids = manifest["preset_ids"].astype(np.int64)
    synths = manifest["synths"].astype(np.uint8)
    rng = np.random.default_rng(args.seed)
    train_by_name: dict[str, list[int]] = {}
    validation_by_name: dict[str, list[int]] = {}
    for code, synth in ((1, "serum1"), (2, "serum2")):
        values = np.asarray(sorted(map(int, preset_ids[synths == code])), dtype=np.int64)
        rng.shuffle(values)
        validation_count = max(1, int(round(len(values) * 0.1)))
        validation_by_name[synth] = sorted(map(int, values[:validation_count]))
        train_by_name[synth] = sorted(map(int, values[validation_count:]))
    neighbor_ids = np.empty_like(preset_ids)
    report: dict[str, object] = {"seed": args.seed, "by_synth": {}}
    for code, synth in ((1, "serum1"), (2, "serum2")):
        selected = np.flatnonzero(synths == code)
        ids = preset_ids[selected]
        train_ids = np.asarray(train_by_name[synth], dtype=np.int64)
        train_rows = selected[np.isin(ids, train_ids)]
        if len(train_rows) < 2:
            raise RuntimeError(f"Not enough training neighbors for {synth}")
        neighbor_ids[selected] = _nearest(
            embeddings[selected], embeddings[train_rows], ids, preset_ids[train_rows]
        )
        similarities = np.sum(
            np.asarray(embeddings[selected])
            * np.asarray(
                embeddings[
                    [int(np.flatnonzero(preset_ids == value)[0]) for value in neighbor_ids[selected]]
                ]
            ),
            axis=1,
        )
        report["by_synth"][synth] = {
            "preset_count": len(selected),
            "training_neighbor_pool": len(train_rows),
            "self_neighbors": int(np.sum(neighbor_ids[selected] == ids)),
            "mean_cosine": float(np.mean(similarities)),
            "minimum_cosine": float(np.min(similarities)),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, preset_ids=preset_ids, synths=synths, neighbor_preset_ids=neighbor_ids)
    report["gate_pass"] = all(
        row["self_neighbors"] == 0 for row in report["by_synth"].values()  # type: ignore[union-attr]
    )
    report["split"] = {
        "train_preset_ids": train_by_name,
        "validation_preset_ids": validation_by_name,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("DELTA_NEIGHBORS=" + json.dumps(report, sort_keys=True))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
