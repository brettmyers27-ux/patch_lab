#!/usr/bin/env python3
"""Build the private Serum 2 structural vocabulary artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.platform_env import ENV
from core.serum2_preset import parse_serum2_preset
from core.serum2_structural_space import build_structural_space


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "serum2_structural_space.json",
    )
    args = parser.parse_args()
    package = args.package.expanduser().resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        row
        for row in manifest["presets_by_hash"].values()
        if row.get("synth") == "serum2"
    ]
    presets = [parse_serum2_preset(package / row["relative_path"]) for row in rows]
    roots = ENV.factory_roots_for("serum2", existing_only=True)
    payload = build_structural_space(presets, roots).to_json()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    summary = {name: value["count"] for name, value in payload["categories"].items()}
    print("SERUM2_STRUCTURAL_SPACE=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
