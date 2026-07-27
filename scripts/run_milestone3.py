#!/usr/bin/env python3
"""Stream the complete resumable Milestone 3 workflow for the Qt worker."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE: subprocess.Popen[str] | None = None


def _stop(_signum: int, _frame) -> None:  # type: ignore[no-untyped-def]
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.terminate()
    raise SystemExit(130)


def run(label: str, arguments: list[str]) -> None:
    global ACTIVE
    print(f"MILESTONE3_PHASE={label}", flush=True)
    ACTIVE = subprocess.Popen(
        [sys.executable, *arguments],
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
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
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
