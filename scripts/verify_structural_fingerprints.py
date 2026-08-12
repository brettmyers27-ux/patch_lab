#!/usr/bin/env python3
"""Run the Stage 3C Phase 0 hard gate on the private Stage 3B index."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.structural_fingerprint_validation import (
    SelfRetrievalResult,
    deterministic_sample_indices,
    distinctness_clusters,
    self_retrieval,
)


CATEGORIES = ("fx_type", "wavetable", "mod_route", "noise_sample")
DEFAULT_INDEX = (
    PROJECT_ROOT / "data" / "models" / "serum2_structural_fingerprints.npz"
)
DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage3c" / "phase0-fingerprint-gate.json"
FAILURE_EXAMPLE_LIMIT = 20


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _self_retrieval_payload(result: SelfRetrievalResult) -> dict[str, object]:
    """Keep private diagnostic output useful without dumping thousands of IDs."""

    return {
        "queried": result.queried,
        "passed": result.passed,
        "failed": result.failed,
        "failure_examples": [
            {
                "query_stable_id": query_id,
                "rank_1_stable_id": winner_id,
                "cosine_similarity": similarity,
            }
            for query_id, winner_id, similarity in result.failures[
                :FAILURE_EXAMPLE_LIMIT
            ]
        ],
        "failure_examples_truncated": result.failed > FAILURE_EXAMPLE_LIMIT,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    archive = np.load(args.index.expanduser().resolve(), allow_pickle=False)
    categories: dict[str, object] = {}
    hard_gate_passed = True
    for category in CATEGORIES:
        features = np.asarray(archive[f"{category}__features"], dtype=np.float32)
        stable_ids = archive[f"{category}__stable_ids"].tolist()
        sample_indices = deterministic_sample_indices(len(features), args.sample_count)
        sampled = self_retrieval(features, stable_ids, sample_indices)
        exhaustive = self_retrieval(features, stable_ids)
        distinctness = {
            name: asdict(distinctness_clusters(features, threshold=threshold))
            for name, threshold in (
                ("exact_1e-7", 1e-7),
                ("near_1e-6", 1e-6),
                ("near_1e-4", 1e-4),
            )
        }
        zero_descriptors = int(
            np.count_nonzero(np.linalg.norm(features, axis=1) <= 1e-12)
        )
        category_passed = sampled.failed == 0 and exhaustive.failed == 0
        hard_gate_passed &= category_passed
        categories[category] = {
            "candidates": len(features),
            "descriptor_dimensions": features.shape[1],
            "zero_descriptors": zero_descriptors,
            "sampled_self_retrieval": _self_retrieval_payload(sampled),
            "exhaustive_self_retrieval": _self_retrieval_payload(exhaustive),
            "distinctness": distinctness,
            "hard_gate_passed": category_passed,
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "index": str(args.index.expanduser().resolve()),
        "self_retrieval_rule": (
            "own descriptor must rank its own stable ID first; descending cosine "
            "similarity with stable-ID tie break"
        ),
        "distinctness_rule": "connected components by cosine-distance threshold",
        "categories": categories,
        "hard_gate_passed": hard_gate_passed,
        "decision": (
            "proceed"
            if hard_gate_passed
            else "stop before in-context search; Phase 0 integrity gate failed"
        ),
    }
    _atomic_json(args.report.expanduser().resolve(), payload)
    print("STAGE3C_PHASE0=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if hard_gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
