#!/usr/bin/env python3
"""Repair Stage 3B noise/route identity sets without publishing private data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.structural_estimators import ControlledFingerprintIndex
from core.structural_fingerprint_validation import distinctness_components


DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "models" / "serum2_structural_fingerprints.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "serum2_structural_fingerprints_stage3d.npz"
)
DEFAULT_POLICY = (
    PROJECT_ROOT / "data" / "models" / "serum2_structural_search_policy.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage3d" / "fingerprint-repair.json"
DEFAULT_DIAGNOSTICS = (
    PROJECT_ROOT / "data" / "stage3c" / "phase0-fingerprint-gate.json"
)
DEFAULT_VOCABULARY = (
    PROJECT_ROOT / "data" / "models" / "serum2_structural_space.json"
)
CATEGORIES = ("fx_type", "wavetable", "mod_route", "noise_sample")
REPAIR_THRESHOLD = 1e-4


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resource_key(value: str) -> str:
    current = value.replace("\\", "/").strip()
    while current.startswith("/"):
        current = current[1:]
    return current.casefold()


def _resolve_noise_resource(value: str) -> Path | None:
    xfer = Path.home() / "Documents" / "Xfer"
    factory = xfer / "Serum 2 Presets" / "Samples" / "Factory Non-Tonal"
    normalized = value.replace("\\", "/").strip().lstrip("/")
    candidates = [
        factory / "Noises" / normalized,
        factory / normalized.removeprefix("../"),
        xfer / "Serum Presets" / "Noises" / normalized,
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _decoded_audio_equivalent(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    left_audio, left_rate = sf.read(left, always_2d=True, dtype="float32")
    right_audio, right_rate = sf.read(right, always_2d=True, dtype="float32")
    return bool(
        left_rate == right_rate
        and left_audio.shape == right_audio.shape
        and np.max(np.abs(left_audio - right_audio), initial=0.0) <= 1.0 / 32768.0
    )


def _deduplicate(
    features: np.ndarray,
    stable_ids: list[str],
    observed_counts: dict[str, int],
    *,
    exclude_zero: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
    norms = np.linalg.norm(features, axis=1)
    excluded = set(np.flatnonzero(norms <= 1e-12).tolist()) if exclude_zero else set()
    available = np.asarray(
        [index for index in range(len(features)) if index not in excluded],
        dtype=np.int64,
    )
    dropped: set[int] = set(excluded)
    cluster_rows: list[dict[str, Any]] = []
    # Re-normalizing a persisted float32 subset can expose a boundary pair that
    # sat microscopically above the threshold in the parent matrix. Iterate to
    # a fixed point so the repaired artifact itself satisfies the same gate.
    while True:
        components = distinctness_components(
            features[available], threshold=REPAIR_THRESHOLD
        )
        if not components:
            break
        round_dropped: set[int] = set()
        for component in components:
            source_indices = [int(available[position]) for position in component]
            representative = min(
                source_indices,
                key=lambda index: (
                    -observed_counts.get(stable_ids[index], 0),
                    stable_ids[index],
                ),
            )
            members = [stable_ids[index] for index in source_indices]
            round_dropped.update(
                index for index in source_indices if index != representative
            )
            cluster_rows.append(
                {
                    "representative": stable_ids[representative],
                    "members": members,
                    "member_count": len(members),
                }
            )
        dropped.update(round_dropped)
        available = np.asarray(
            [index for index in available if int(index) not in round_dropped],
            dtype=np.int64,
        )
    retained = np.asarray(
        [index for index in range(len(features)) if index not in dropped],
        dtype=np.int64,
    )
    return retained, cluster_rows, [stable_ids[index] for index in sorted(excluded)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    args = parser.parse_args()

    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    observed_counts = {
        category: {
            str(entry["id"]): int(entry.get("observed_count", 0))
            for entry in vocabulary["categories"][category]["entries"]
        }
        for category in CATEGORIES
    }
    features: dict[str, np.ndarray] = {}
    stable_ids: dict[str, list[str]] = {}
    labels: dict[str, list[str]] = {}
    category_report: dict[str, Any] = {}
    with np.load(args.input, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        for category in CATEGORIES:
            source_features = np.asarray(
                archive[f"{category}__features"], dtype=np.float32
            )
            source_ids = archive[f"{category}__stable_ids"].astype(str).tolist()
            source_labels = archive[f"{category}__labels"].astype(str).tolist()
            if category in {"noise_sample", "mod_route"}:
                retained, clusters, zero_ids = _deduplicate(
                    source_features,
                    source_ids,
                    observed_counts[category],
                    exclude_zero=category == "mod_route",
                )
            else:
                retained = np.arange(len(source_features), dtype=np.int64)
                clusters, zero_ids = [], []
            features[category] = source_features[retained]
            stable_ids[category] = [source_ids[index] for index in retained]
            labels[category] = [source_labels[index] for index in retained]
            category_report[category] = {
                "input_candidates": len(source_ids),
                "retained_candidates": len(retained),
                "excluded_zero_descriptors": len(zero_ids),
                "collapsed_clusters": len(clusters),
                "collapsed_members": sum(row["member_count"] for row in clusters),
                "largest_cluster": max(
                    (row["member_count"] for row in clusters), default=1
                ),
                "clusters": clusters,
                "excluded_zero_stable_ids": zero_ids,
            }

    noise_by_id = {
        str(entry["id"]): str(entry["value"])
        for entry in vocabulary["categories"]["noise_sample"]["entries"]
    }
    failures = diagnostics["categories"]["noise_sample"][
        "exhaustive_self_retrieval"
    ]["failure_examples"]
    resource_rows = []
    for failure in failures:
        query_id = str(failure["query_stable_id"])
        winner_id = str(failure["rank_1_stable_id"])
        query_value, winner_value = noise_by_id[query_id], noise_by_id[winner_id]
        query_path = _resolve_noise_resource(query_value)
        winner_path = _resolve_noise_resource(winner_value)
        resource_rows.append(
            {
                "query_stable_id": query_id,
                "winner_stable_id": winner_id,
                "query_value": query_value,
                "winner_value": winner_value,
                "query_source_exists": query_path is not None,
                "winner_source_exists": winner_path is not None,
                "same_normalized_resource_key": _resource_key(query_value)
                == _resource_key(winner_value),
                "decoded_audio_equivalent": _decoded_audio_equivalent(
                    query_path, winner_path
                ),
            }
        )

    repaired = ControlledFingerprintIndex(
        features,
        stable_ids,
        labels,
        metadata={
            **metadata,
            "schema_version": 2,
            "stage3d_repair_threshold": REPAIR_THRESHOLD,
            "stage3d_policy": (
                "exclude zero mod-route descriptors; collapse connected "
                "near-duplicate noise and route components"
            ),
        },
    )
    repaired.save(args.output)
    policy = {
        "schema_version": 1,
        "source_index": args.output.name,
        "enabled_categories": list(CATEGORIES),
        "allowed_ids": {
            category: stable_ids[category] for category in CATEGORIES
        },
        "repair_threshold": REPAIR_THRESHOLD,
    }
    _atomic_json(args.policy, policy)
    missing_overlap = sum(not row["query_source_exists"] for row in resource_rows)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "policy": str(args.policy.resolve()),
        "categories": category_report,
        "noise_diagnosis": {
            "failing_rows": len(resource_rows),
            "missing_source_overlap": missing_overlap,
            "resource_rows": resource_rows,
        },
    }
    _atomic_json(args.report, payload)
    print(
        "STAGE3D_FINGERPRINT_REPAIR="
        + json.dumps(
            {
                "counts": {
                    category: row["retained_candidates"]
                    for category, row in category_report.items()
                },
                "noise_missing_overlap": missing_overlap,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
