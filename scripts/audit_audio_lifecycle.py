#!/usr/bin/env python3
"""Static audio-write inventory plus completed-CMA scratch cleanup gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT = PROJECT_ROOT / "data" / "models" / "milestone6_audio_lifecycle_report.json"


INVENTORY = [
    {
        "location": "core/render.py",
        "kind": "permanent-local-only",
        "purpose": "User-owned library renders for local search and audition",
        "cleanup": "Retained on the user's disk; never enters relay payloads",
    },
    {
        "location": "core/match_workflow.py",
        "kind": "application-session-temporary",
        "purpose": "Winning optimization render before the completed match is archived",
        "cleanup": "The UI archives the winner/result then removes remaining session scratch on close",
    },
    {
        "location": "scripts/render_recommendation_preview.py",
        "kind": "permanent-local-only",
        "purpose": "On-demand C1-C7 recommendation audition",
        "cleanup": "Content-addressed under app-data audio; generated keys are removed after their final Match Library reference",
    },
    {
        "location": "scripts/render_factory_preview.py",
        "kind": "permanent-local-only",
        "purpose": "User-requested factory audition rendered on first click",
        "cleanup": "Cached locally for later audition; never enters relay payloads",
    },
    {
        "location": "core/audio_input.py",
        "kind": "temporary-cleanup-required",
        "purpose": "Bundled-ffmpeg decode fallback",
        "cleanup": "TemporaryDirectory context removes decoded WAV",
    },
    {
        "location": "core/preset_export.py",
        "kind": "temporary-cleanup-required",
        "purpose": "Export verification render/state",
        "cleanup": "TemporaryDirectory/try-finally removes verification artifacts",
    },
    {
        "location": "scripts/generate_perturbations.py",
        "kind": "developer-diagnostic",
        "purpose": "First 20 explicitly retained training spot-check WAVs",
        "cleanup": "Development data only; not part of end-user Match a Sound",
    },
    {
        "location": "scripts/analyze_serum2_partitioned_audio.py",
        "kind": "developer-diagnostic",
        "purpose": "Milestone state-reconstruction fixture renders",
        "cleanup": "Development report artifacts; excluded from distribution",
    },
]


def scratch_paths() -> set[str]:
    root = Path(tempfile.gettempdir())
    return {
        str(path)
        for pattern in ("patchlab-match-session-*", "patchlab-match-*")
        for path in root.glob(pattern)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-cma", action="store_true")
    parser.add_argument(
        "--fixture", type=Path, default=PROJECT_ROOT / "data" / "audio" / "67" / "60.wav"
    )
    args = parser.parse_args()
    before = scratch_paths()
    completed_query = False
    result_path: str | None = None
    if args.run_cma:
        process = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "match_sound.py"),
                str(args.fixture),
                "--target-synth",
                "serum1",
                "--budget",
                "quick",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        for line in process.stdout.splitlines():
            if line.startswith("MATCH_RESULT="):
                result_path = line.split("=", 1)[1]
        completed_query = process.returncode == 0 and result_path is not None
        if not completed_query:
            print(process.stdout)
            print(process.stderr, file=sys.stderr)
    after = scratch_paths()
    leaked = sorted(after - before)
    relay_root = PROJECT_ROOT.parent / "patchlab-relay"
    relay_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in relay_root.rglob("*.py")
        if path.is_file()
    )
    relay_has_audio_route = any(
        token in relay_sources for token in ('@app.post("/audio', '@app.get("/audio')
    )
    payload = {
        "inventory": INVENTORY,
        "cma_query_requested": args.run_cma,
        "cma_query_completed": completed_query,
        "cma_result_path": result_path,
        "scratch_before": sorted(before),
        "scratch_after": sorted(after),
        "new_scratch_paths": leaked,
        "candidate_audio_storage": "in-memory numpy arrays",
        "retained_match_audio": "octave previews persist in one content-addressed local cache; generated entries are reference-cleaned",
        "relay_has_audio_route": relay_has_audio_route,
        "gate_pass": (not args.run_cma or completed_query)
        and not leaked
        and not relay_has_audio_route,
    }
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("AUDIO_LIFECYCLE_REPORT=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
