#!/usr/bin/env python3
"""Stream the complete resumable Milestone 3 workflow for the Qt worker."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
from pathlib import Path

from core.worker_runtime import worker_invocation_for_script

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE: subprocess.Popen[str] | None = None


def _stop(_signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.terminate()
    raise SystemExit(130)


def run(label: str, arguments: list[str]) -> None:
    global ACTIVE
    print(f"MILESTONE3_PHASE={label}", flush=True)
    program, invocation = worker_invocation_for_script(arguments)
    ACTIVE = subprocess.Popen(
        [program, *invocation],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert ACTIVE.stdout is not None
    for line in ACTIVE.stdout:
        print(line, end="", flush=True)
    code = ACTIVE.wait()
    ACTIVE = None
    if code:
        raise RuntimeError(f"{label} exited with code {code}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-training", action="store_true")
    parser.add_argument("--packaged-smoke-db", type=Path)
    parser.add_argument("--packaged-smoke-feature-dir", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    if args.packaged_smoke_db is not None:
        if args.packaged_smoke_feature_dir is None:
            parser.error(
                "--packaged-smoke-feature-dir is required with "
                "--packaged-smoke-db"
            )
        report = args.packaged_smoke_feature_dir / "embedding-report.json"
        run(
            "embeddings",
            [
                "scripts/analyze_library.py",
                "--db",
                str(args.packaged_smoke_db),
                "--batch-size",
                "1",
                "--feature-workers",
                "1",
                "--limit",
                "1",
                "--expected-count",
                "7",
                "--feature-dir",
                str(args.packaged_smoke_feature_dir),
                "--report",
                str(report),
            ],
        )
        summary = {
            "deep_training": False,
            "complete": True,
            "packaged_smoke": True,
            "embedding_report": str(report),
        }
        print(
            "MILESTONE3_SUMMARY=" + json.dumps(summary, sort_keys=True),
            flush=True,
        )
        return 0
    run("serum2-targets", ["scripts/build_serum2_targets.py"])
    run("embeddings", ["scripts/analyze_library.py", "--batch-size", "64", "--feature-workers", "8"])
    run("similarity", ["scripts/build_similarity_index.py"])
    if args.deep_training:
        run("synthetic-serum1", ["scripts/generate_synthetic_serum1.py"])
    train = ["scripts/train_param_model.py"]
    if args.deep_training:
        train.append("--deep-training")
    run("training", train)
    run("roundtrip", ["scripts/roundtrip_param_model.py"])
    summary = {"deep_training": args.deep_training, "complete": True}
    print("MILESTONE3_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
