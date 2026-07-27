#!/usr/bin/env python3
"""Train the Milestone 3 two-head parameter inference model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import add_serum1_synthetic, load_training_bundle
from core.platform_env import ENV
from core.train import train_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deep-training", action="store_true")
    args = parser.parse_args()
    bundle = load_training_bundle(seed=args.seed)
    if args.deep_training:
        bundle = add_serum1_synthetic(bundle)
    report = train_model(
        bundle,
        ENV,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
    )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
