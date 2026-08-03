#!/usr/bin/env python3
"""Render a deterministic sample of formerly missing Serum 1 source presets.

Stage 2B must prove the transferred hash-to-path map works in the actual
headless Serum render worker before asking it to render the whole index.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import _serum1_targets
from core.matcher import Candidate, _init_render_worker, _render_candidate
from core.synthesis_assets import resolve_synthesis_assets


DEFAULT_PREFLIGHT = PROJECT_ROOT / "data" / "stage2" / "embedding-rebuild-preflight-ft.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage2" / "preset-resolution-proof.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _previously_missing_serum1(report: Path) -> list[int]:
    payload = json.loads(report.read_text(encoding="utf-8"))
    rows = payload.get("renderability", {}).get("missing_presets", [])
    values = sorted(
        {
            int(row["preset_id"])
            for row in rows
            if row.get("synth") == "serum1"
        }
    )
    if not values:
        raise RuntimeError(f"No previously missing Serum 1 presets in {report}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--midi-note", type=int, default=60)
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()
    if args.sample_size < 25:
        raise ValueError("Stage 2B requires a sample size of at least 25")

    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    assets = resolve_synthesis_assets()
    previous = _previously_missing_serum1(args.preflight_report.expanduser().resolve())
    if len(previous) < args.sample_size:
        raise RuntimeError(f"Only {len(previous)} formerly missing Serum 1 presets are available")
    selected = sorted(random.Random(args.seed).sample(previous, args.sample_size))
    store = _serum1_targets(assets.library_db)

    with sqlite3.connect(assets.library_db) as connection:
        catalog = {
            int(preset_id): str(content_hash or "")
            for preset_id, content_hash in connection.execute(
                "SELECT id,content_hash FROM presets WHERE id IN ({})".format(
                    ",".join("?" for _ in selected)
                ),
                selected,
            )
        }
    mapping_payload = json.loads(Path(assets.factory_mapping).read_text(encoding="utf-8"))
    mapping = mapping_payload.get("local_paths_by_hash", {})
    unresolved = [
        preset_id
        for preset_id in selected
        if not catalog.get(preset_id) or not Path(str(mapping.get(catalog[preset_id], ""))).is_file()
    ]
    rows: list[dict[str, Any]] = []
    if not unresolved:
        candidates = [
            Candidate(
                "serum1",
                preset_id,
                np.asarray(store.vectors[store.preset_row[preset_id]], dtype=np.float32),
                np.asarray(store.masks[store.preset_row[preset_id]], dtype=bool),
                "stage2b-preset-resolution-proof",
                exact_base=True,
                midi_note=args.midi_note,
            )
            for preset_id in selected
        ]
        context = mp.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="patchlab-stage2b-resolution-") as scratch:
            with context.Pool(4, initializer=_init_render_worker, initargs=(scratch, assets)) as pool:
                results = pool.map(
                    _render_candidate,
                    [(candidate, args.midi_note, args.duration) for candidate in candidates],
                    chunksize=1,
                )
        for preset_id, (audio, coverage, error) in zip(selected, results, strict=True):
            rms = float(np.sqrt(np.mean(np.square(audio)))) if audio is not None and len(audio) else 0.0
            rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
            rows.append(
                {
                    "preset_id": preset_id,
                    "content_hash": catalog[preset_id],
                    "path": str(mapping[catalog[preset_id]]),
                    "coverage": float(coverage),
                    "rms_dbfs": rms_dbfs,
                    "error": error,
                    "non_silent": error is None and rms_dbfs > -75.0,
                }
            )
    failures = unresolved + [row["preset_id"] for row in rows if not row["non_silent"]]
    payload = {
        "status": "complete" if not failures else "failed",
        "sample_size": args.sample_size,
        "seed": args.seed,
        "midi_note": args.midi_note,
        "duration_s": args.duration,
        "mapping": str(assets.factory_mapping),
        "previously_missing_serum1_count": len(previous),
        "unresolved_preset_ids": unresolved,
        "failures": failures,
        "renders": rows,
    }
    _atomic_json(args.report.expanduser().resolve(), payload)
    print(
        "STAGE2B_PRESET_RESOLUTION="
        + json.dumps(
            {
                "status": payload["status"],
                "sample_size": args.sample_size,
                "failures": len(failures),
                "report": str(args.report.expanduser().resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
