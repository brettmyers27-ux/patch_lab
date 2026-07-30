#!/usr/bin/env python3
"""One-command cross-platform PatchLab model-cache validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_assets import ModelAssetsError, validate_model_assets  # noqa: E402


def main() -> int:
    try:
        assets = validate_model_assets()
    except ModelAssetsError as exc:
        print(
            "MODEL_ASSETS_GATE="
            + json.dumps(
                {"gate_pass": False, "error": str(exc)},
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 1
    print(
        "MODEL_ASSETS_GATE="
        + json.dumps(
            {
                "gate_pass": True,
                "model_cache": str(assets.cache_dir),
                "checkpoint": str(assets.checkpoint),
                "checkpoint_bytes": assets.checkpoint.stat().st_size,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
