#!/usr/bin/env python3
"""Validate a completed instant factory-only result without rerunning CLAP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.factory_bundle import FactoryBundle


REPORT = PROJECT_ROOT / "data" / "models" / "factory_only_match_report.json"


def latest_result() -> Path:
    candidates = []
    for path in (PROJECT_ROOT / "data" / "matches").glob("*/result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("factory_only") and all(
            item.get("factory_bundle_id") is not None
            for item in payload.get("existing_matches", [])
        ):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No completed pure factory-only match result")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    path = args.result or latest_result()
    result = json.loads(path.read_text(encoding="utf-8"))
    known = FactoryBundle().known_hashes()
    existing = result.get("existing_matches", [])
    recommendation = result.get("recommendation") or {}
    payload = {
        "result_path": str(path),
        "factory_only": bool(result.get("factory_only")),
        "result_count": len(existing),
        "all_result_hashes_factory": all(
            str(item.get("content_hash")) in known for item in existing
        ),
        "recommendation_hash_factory": (
            str(recommendation.get("content_hash")) in known
        ),
        "evaluations": int(recommendation.get("evaluations", -1)),
        "winner_audio_path": recommendation.get("winner_audio_path"),
        "on_demand_preview_count": sum(
            bool(item.get("preview_source_path")) for item in existing
        ),
        "recommendation_preview_available": bool(
            recommendation.get("preview_source_path")
        ),
        "plugin_render_required": False,
        "local_export_available": bool(recommendation.get("export_available")),
    }
    payload["gate_pass"] = (
        payload["factory_only"]
        and payload["result_count"] == 10
        and payload["all_result_hashes_factory"]
        and payload["recommendation_hash_factory"]
        and payload["evaluations"] == 0
        and payload["winner_audio_path"] is None
        and payload["on_demand_preview_count"] == 10
        and payload["recommendation_preview_available"]
    )
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("FACTORY_ONLY_MATCH_REPORT=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
