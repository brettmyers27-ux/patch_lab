#!/usr/bin/env python3
"""Read-only verification of persisted Milestone 4 gate artifacts."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.fxp import parse_fxp
from core.serum2_preset import parse_serum2_preset


MODEL_DIR = PROJECT_ROOT / "data" / "models"


def _row(label: str, passed: bool, detail: str) -> bool:
    print(f"{'PASS' if passed else 'FAIL':4}  {label:<28} {detail}")
    return passed


def main() -> int:
    export = json.loads(
        (MODEL_DIR / "milestone4_export_report.json").read_text(encoding="utf-8")
    )
    ui = json.loads(
        (MODEL_DIR / "milestone4_ui_gate_report.json").read_text(encoding="utf-8")
    )
    checks: list[bool] = []

    rows = export["rows"]
    coverage = all(
        sum(row["synth"] == synth and row["case"] == case for row in rows) > 0
        for synth in ("serum1", "serum2")
        for case in ("copied", "optimized")
    )
    checks.append(
        _row(
            "export case coverage",
            len(rows) == 10 and coverage,
            "10 files; copied + optimized for both synths",
        )
    )

    decoded = True
    hashes = True
    assets = True
    minimum_clap = 1.0
    for item in rows:
        path = Path(item["path"])
        if not path.is_file():
            decoded = False
            continue
        minimum_clap = min(minimum_clap, float(item["clap_similarity"]))
        if item["synth"] == "serum1":
            try:
                parse_fxp(path)
            except Exception:
                decoded = False
        else:
            try:
                preset = parse_serum2_preset(path)
                raw = path.read_bytes()
                metadata_length = struct.unpack_from("<Q", raw, 9)[0]
                compressed = raw[17 + metadata_length + 8 :]
                hashes &= preset.metadata.get("hash") == hashlib.md5(compressed).hexdigest()
            except Exception:
                decoded = False
            assets &= bool(item["assets_retained"])
    checks.append(
        _row(
            "native decode/reload",
            decoded and all(row["pass"] for row in rows),
            f"10/10; minimum audio CLAP={minimum_clap:.6f}",
        )
    )
    checks.append(
        _row(
            "Serum 2 hash/assets",
            hashes and assets,
            "compressed-payload MD5 valid; base references retained",
        )
    )

    checks.append(
        _row(
            "raw upload path",
            bool(ui["raw"]["pass"]) and int(ui["raw"]["own_preset_rank"]) == 1,
            f"own preset rank #{ui['raw']['own_preset_rank']}; "
            f"{ui['raw']['progress_updates']} progress updates",
        )
    )
    checks.append(
        _row(
            "lossy MP3 upload path",
            bool(ui["lossy_mp3"]["pass"])
            and int(ui["lossy_mp3"]["own_preset_rank"]) <= 5,
            f"own preset rank #{ui['lossy_mp3']['own_preset_rank']}",
        )
    )
    checks.append(
        _row(
            "silence handling",
            bool(ui["silence"]["pass"]),
            str(ui["silence"]["ui_message"]),
        )
    )
    checks.append(
        _row(
            "target/budget controls",
            bool(ui["controls"]["pass"]),
            "Serum 2 default; Quick, Balanced, Best Quality",
        )
    )
    overall = (
        bool(export["gate_pass"]) and bool(ui["gate_pass"]) and all(checks)
    )
    print(f"\n{'PASS' if overall else 'FAIL'}  Milestone 4")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
