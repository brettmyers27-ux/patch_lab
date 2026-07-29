#!/usr/bin/env python3
"""Refresh the macOS reference used by the real-Windows parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.factory_bundle import DEFAULT_FACTORY_BUNDLE, FactoryBundle  # noqa: E402
from core.platform_env import ENV  # noqa: E402
from core.plugin_host import (  # noqa: E402
    dump_dawdreamer_parameters,
    make_dawdreamer_processor,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "core" / "reference" / "macos_plugin_parity.json"
FIXTURE_IDS = (1, 100, 458, 600)


def _parameter_reference(synth: str, required_format: str) -> dict[str, object]:
    candidate = next(
        item
        for item in ENV.plugins_for(synth)
        if item.format == required_format and item.hostable
    )
    _engine, processor = make_dawdreamer_processor(candidate)
    parameters = dump_dawdreamer_parameters(processor)
    signature_source = "\n".join(
        f"{item.index}\0{item.name}" for item in parameters
    ).encode("utf-8")
    return {
        "format": required_format,
        "plugin_path": str(candidate.path),
        "parameter_count": len(parameters),
        "index_name_sha256": hashlib.sha256(signature_source).hexdigest(),
        "parameters": [
            {
                "index": item.index,
                "name": item.name,
                "normalized_value": item.norm_value,
                "display_value": item.display_value,
            }
            for item in parameters
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_FACTORY_BUNDLE)
    args = parser.parse_args()
    if ENV.branch != "macos":
        raise RuntimeError("The reference must be generated from the proven macOS build")
    bundle = FactoryBundle(args.bundle)
    for preset_id in FIXTURE_IDS:
        bundle.preset_by_id(preset_id)
        bundle.note_embedding(preset_id, 60)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": f"{ENV.system_name}/{ENV.machine}",
        "plugins": {
            "serum1": _parameter_reference("serum1", "VST2"),
            "serum2": _parameter_reference("serum2", "VST3"),
        },
        # The actual names, hashes, and macOS embeddings stay in the
        # passcode-gated factory bundle rather than being copied into source.
        "factory_fixture_ids": list(FIXTURE_IDS),
        "factory_fixture_midi_note": 60,
        "thresholds": {
            "minimum_fixture_clap_cosine": 0.80,
            "minimum_mean_clap_cosine": 0.90,
            "maximum_value_mismatch_fraction": 0.05,
            "normalized_value_tolerance": 1e-4,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output}: "
        f"Serum 1={payload['plugins']['serum1']['parameter_count']}, "
        f"Serum 2={payload['plugins']['serum2']['parameter_count']}, "
        f"fixtures={len(payload['factory_fixture_ids'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
