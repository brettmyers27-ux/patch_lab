#!/usr/bin/env python3
"""Aggregate all persisted Milestone 6 gate reports."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "data" / "models" / "milestone6_report.json"
SOURCES = {
    "factory_bundle": PROJECT_ROOT / "data" / "models" / "factory_bundle_report.json",
    "factory_only_match": PROJECT_ROOT / "data" / "models" / "factory_only_match_report.json",
    "consent_ui": PROJECT_ROOT / "data" / "models" / "milestone6_ui_report.json",
    "local_processing": PROJECT_ROOT / "data" / "models" / "milestone6_local_processing_report.json",
    "audio_lifecycle": PROJECT_ROOT / "data" / "models" / "milestone6_audio_lifecycle_report.json",
    "relay": PROJECT_ROOT.parent / "patchlab-relay" / "relay-test-report.json",
}


def main() -> int:
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in SOURCES.items()
    }
    payload = {
        "gates": {
            name: bool(report.get("gate_pass")) for name, report in reports.items()
        },
        "factory_bundle": {
            "classified_factory": reports["factory_bundle"]["classified_factory_count"],
            "shipped_presets": reports["factory_bundle"]["preset_count"],
            "note_embeddings": reports["factory_bundle"]["note_embedding_count"],
            "size_mib": reports["factory_bundle"]["file_size_mib"],
            "contains_audio": reports["factory_bundle"]["contains_audio"],
            "contains_preset_files": reports["factory_bundle"]["contains_preset_files"],
        },
        "local_processing": reports["local_processing"]["first_run"],
        "local_rerun": reports["local_processing"]["second_run"],
        "relay_routes": reports["relay"]["routes"],
        "relay_audio_files": reports["relay"]["audio_files_stored"],
        "cma_new_scratch_paths": reports["audio_lifecycle"]["new_scratch_paths"],
    }
    payload["gate_pass"] = all(payload["gates"].values())
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("MILESTONE6_REPORT=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
