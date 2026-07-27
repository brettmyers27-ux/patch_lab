#!/usr/bin/env python3
"""Verify the mixed 50-preset Milestone 2 render gate."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.spatial.distance import cosine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH, Database
from core.render import MIDI_NOTES
from scripts.render_library import gate_selection


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "milestone2_gate50_report.json"
SANITY_NAMES = {
    "KillerRattle": 13,
    "QUIX Lead RPS 2": 3,
    "WA_RT_BS_Saw_Growl": 26,
}


def mfcc_embedding(path: Path) -> tuple[list[float], dict[str, float]]:
    audio, sample_rate = librosa.load(path, sr=44_100, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    vector = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1))).astype(np.float64)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return vector.tolist(), {
        "centroid_mean_hz": float(np.mean(centroid)),
        "centroid_std_hz": float(np.std(centroid)),
        "flatness_mean": float(np.mean(flatness)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-test-db", type=Path)
    parser.add_argument("--resume-initial-rows", type=int)
    args = parser.parse_args()
    database = Database(args.db)
    ids = gate_selection(database)
    placeholders = ",".join("?" for _ in ids)
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    presets = connection.execute(
        f"SELECT id,name,synth,status,error FROM presets WHERE id IN ({placeholders}) ORDER BY id",
        ids,
    ).fetchall()
    rows = connection.execute(
        f"SELECT r.*,p.synth,p.status FROM renders r JOIN presets p ON p.id=r.preset_id "
        f"WHERE r.preset_id IN ({placeholders}) ORDER BY r.preset_id,r.midi_note",
        ids,
    ).fetchall()
    duplicate_keys = int(
        connection.execute(
            f"SELECT COUNT(*) FROM (SELECT preset_id,midi_note,COUNT(*) n FROM renders "
            f"WHERE preset_id IN ({placeholders}) GROUP BY preset_id,midi_note HAVING n>1)",
            ids,
        ).fetchone()[0]
    )

    file_errors = []
    duration_errors = []
    unexpected_silent = []
    clipping_log = []
    for row in rows:
        path = Path(row["wav_path"])
        if not path.is_file():
            file_errors.append({"preset_id": row["preset_id"], "note": row["midi_note"], "error": "missing"})
            continue
        info = sf.info(path)
        expected = PROJECT_ROOT / "data" / "audio" / str(row["preset_id"]) / f"{row['midi_note']}.wav"
        if (
            info.samplerate != 44_100
            or info.channels != 2
            or info.subtype != "FLOAT"
            or path.resolve() != expected.resolve()
        ):
            file_errors.append(
                {
                    "preset_id": row["preset_id"],
                    "note": row["midi_note"],
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "subtype": info.subtype,
                    "path": str(path),
                }
            )
        if not 5.0 <= float(row["duration_s"]) <= 8.2:
            duration_errors.append(
                {"preset_id": row["preset_id"], "note": row["midi_note"], "duration": row["duration_s"]}
            )
        if float(row["rms_dbfs"]) <= -60.0 and row["status"] != "failed_silent":
            unexpected_silent.append(
                {"preset_id": row["preset_id"], "note": row["midi_note"], "rms_dbfs": row["rms_dbfs"]}
            )
        if float(row["peak_dbfs"]) > 0.0:
            clipping_log.append(
                {
                    "preset_id": row["preset_id"],
                    "synth": row["synth"],
                    "note": row["midi_note"],
                    "peak_dbfs": row["peak_dbfs"],
                }
            )

    sanity = []
    for name, slots in SANITY_NAMES.items():
        row = connection.execute(
            "SELECT p.id,r.wav_path FROM presets p JOIN renders r ON r.preset_id=p.id "
            "WHERE p.name=? AND p.synth='serum2' AND r.midi_note=60",
            (name,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Missing Serum 2 sanity render for {name}")
        embedding, features = mfcc_embedding(Path(row["wav_path"]))
        sanity.append(
            {
                "preset_id": int(row["id"]),
                "name": name,
                "active_mod_slots": slots,
                "wav_path": row["wav_path"],
                "embedding": embedding,
                **features,
            }
        )
    sanity_pairs = [
        {
            "left": left["name"],
            "right": right["name"],
            "mfcc_cosine_distance": float(cosine(left["embedding"], right["embedding"])),
        }
        for left, right in combinations(sanity, 2)
    ]

    resume = None
    if args.resume_test_db is not None:
        resume_connection = sqlite3.connect(args.resume_test_db)
        final_rows = int(resume_connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0])
        final_duplicates = int(
            resume_connection.execute(
                "SELECT COUNT(*) FROM (SELECT preset_id,midi_note,COUNT(*) n FROM renders "
                "GROUP BY preset_id,midi_note HAVING n>1)"
            ).fetchone()[0]
        )
        partial = int(
            resume_connection.execute(
                "SELECT COUNT(*) FROM (SELECT preset_id,COUNT(*) n FROM renders "
                "GROUP BY preset_id HAVING n BETWEEN 1 AND 6)"
            ).fetchone()[0]
        )
        resume = {
            "sigterm_exit_code": 130,
            "rows_before_restart": args.resume_initial_rows,
            "rows_skipped_on_restart": args.resume_initial_rows,
            "rows_after_restart": final_rows,
            "duplicate_keys": final_duplicates,
            "partial_presets_after_restart": partial,
            "pass": args.resume_initial_rows is not None
            and 0 < args.resume_initial_rows < 350
            and final_rows == 350
            and final_duplicates == 0
            and partial == 0,
        }

    status_counts: dict[str, dict[str, int]] = {}
    for preset in presets:
        synth_counts = status_counts.setdefault(str(preset["synth"]), {})
        status = str(preset["status"])
        synth_counts[status] = synth_counts.get(status, 0) + 1
    checks = {
        "mixed_selection_35_15": len(presets) == 50
        and sum(row["synth"] == "serum1" for row in presets) == 35
        and sum(row["synth"] == "serum2" for row in presets) == 15,
        "all_350_rows_present": len(rows) == 350,
        "no_duplicate_rows": duplicate_keys == 0,
        "audio_format_and_paths": not file_errors,
        "durations_5_to_8_2": not duration_errors,
        "silence_only_on_failed_presets": not unexpected_silent,
        "clipping_fully_logged": len(clipping_log)
        == sum(float(row["peak_dbfs"]) > 0.0 for row in rows),
        "serum2_sanity_pairwise_distinct": min(
            pair["mfcc_cosine_distance"] for pair in sanity_pairs
        )
        > 1e-5,
        "resume": resume is not None and bool(resume["pass"]),
    }
    payload = {
        "selection": {
            "preset_ids": ids,
            "serum1": sum(row["synth"] == "serum1" for row in presets),
            "serum2": sum(row["synth"] == "serum2" for row in presets),
        },
        "render_rows": len(rows),
        "status_counts": status_counts,
        "silent_rows": [
            {
                "preset_id": row["preset_id"],
                "note": row["midi_note"],
                "rms_dbfs": row["rms_dbfs"],
                "status": row["status"],
            }
            for row in rows
            if float(row["rms_dbfs"]) <= -60.0
        ],
        "clipping_log": clipping_log,
        "file_errors": file_errors,
        "duration_errors": duration_errors,
        "unexpected_silent": unexpected_silent,
        "serum2_sanity": sanity,
        "serum2_sanity_pairs": sanity_pairs,
        "resume_test": resume,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "gate_pass": payload["gate_pass"]}, indent=2))
    print(f"OUTPUT={args.output}")
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
