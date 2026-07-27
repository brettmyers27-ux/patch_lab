#!/usr/bin/env python3
"""Verify the completed Milestone 2 library and write a durable report."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH
from core.render import MIDI_NOTES


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "milestone2_full_render_report.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    presets = connection.execute(
        "SELECT p.id,p.synth,p.status,p.error FROM presets p "
        "WHERE EXISTS (SELECT 1 FROM params pa WHERE pa.preset_id=p.id) ORDER BY p.id"
    ).fetchall()
    rows = connection.execute(
        "SELECT r.*,p.synth,p.status FROM renders r "
        "JOIN presets p ON p.id=r.preset_id ORDER BY r.preset_id,r.midi_note"
    ).fetchall()

    preset_synth = {int(row["id"]): str(row["synth"]) for row in presets}
    notes_by_preset: dict[int, list[int]] = defaultdict(list)
    rows_by_synth = Counter()
    silent_rows_by_synth = Counter()
    clipped_rows_by_synth = Counter()
    file_errors: list[dict[str, object]] = []
    duration_errors: list[dict[str, object]] = []
    unexpected_silent: list[dict[str, object]] = []
    min_duration = float("inf")
    max_duration = 0.0
    max_peak_by_synth: dict[str, float] = defaultdict(lambda: float("-inf"))

    for row in rows:
        preset_id = int(row["preset_id"])
        note = int(row["midi_note"])
        synth = str(row["synth"])
        notes_by_preset[preset_id].append(note)
        rows_by_synth[synth] += 1
        duration = float(row["duration_s"])
        peak = float(row["peak_dbfs"])
        rms = float(row["rms_dbfs"])
        min_duration = min(min_duration, duration)
        max_duration = max(max_duration, duration)
        max_peak_by_synth[synth] = max(max_peak_by_synth[synth], peak)
        if rms <= -60.0:
            silent_rows_by_synth[synth] += 1
            if str(row["status"]) != "failed_silent":
                unexpected_silent.append(
                    {"preset_id": preset_id, "note": note, "rms_dbfs": rms, "status": row["status"]}
                )
        if peak > 0.0:
            clipped_rows_by_synth[synth] += 1
        if not 5.0 <= duration <= 8.2:
            duration_errors.append({"preset_id": preset_id, "note": note, "duration_s": duration})

        path = Path(str(row["wav_path"]))
        expected = PROJECT_ROOT / "data" / "audio" / str(preset_id) / f"{note}.wav"
        if not path.is_file():
            file_errors.append({"preset_id": preset_id, "note": note, "error": "missing", "path": str(path)})
            continue
        try:
            info = sf.info(path)
        except Exception as exc:
            file_errors.append({"preset_id": preset_id, "note": note, "error": str(exc), "path": str(path)})
            continue
        if (
            path.resolve() != expected.resolve()
            or info.samplerate != 44_100
            or info.channels != 2
            or info.subtype != "FLOAT"
        ):
            file_errors.append(
                {
                    "preset_id": preset_id,
                    "note": note,
                    "path": str(path),
                    "expected_path": str(expected),
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "subtype": info.subtype,
                }
            )

    expected_notes = sorted(MIDI_NOTES)
    note_set_errors = [
        {"preset_id": preset_id, "synth": preset_synth[preset_id], "notes": sorted(notes_by_preset[preset_id])}
        for preset_id in preset_synth
        if sorted(notes_by_preset[preset_id]) != expected_notes
    ]
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    failed_load_by_synth = Counter()
    failed_silent_presets_by_synth = Counter()
    for preset in presets:
        synth = str(preset["synth"])
        status = str(preset["status"])
        status_counts[synth][status] += 1
        if status == "failed_load":
            failed_load_by_synth[synth] += 1
        if status == "failed_silent":
            failed_silent_presets_by_synth[synth] += 1

    temporary_files = [str(path) for path in (PROJECT_ROOT / "data" / "audio").rglob("*.tmp")]
    expected_rows = len(presets) * len(MIDI_NOTES)
    checks = {
        "all_expected_rows_present": len(rows) == expected_rows == 39_053,
        "all_presets_have_exact_note_set": not note_set_errors,
        "audio_format_and_id_paths": not file_errors,
        "durations_5_to_8_2": not duration_errors,
        "silence_only_on_failed_silent_presets": not unexpected_silent,
        "no_failed_load_presets": not failed_load_by_synth,
        "terminal_status_for_every_preset": sum(
            count
            for counts in status_counts.values()
            for status, count in counts.items()
            if status in {"rendered", "failed_silent"}
        )
        == len(presets),
        "no_temporary_audio_files": not temporary_files,
    }
    payload = {
        "preset_count": len(presets),
        "expected_render_rows": expected_rows,
        "render_rows": len(rows),
        "render_rows_by_synth": dict(rows_by_synth),
        "status_counts_by_synth": {synth: dict(counts) for synth, counts in status_counts.items()},
        "failed_silent_presets_by_synth": dict(failed_silent_presets_by_synth),
        "failed_load_presets_by_synth": dict(failed_load_by_synth),
        "silent_render_rows_by_synth": dict(silent_rows_by_synth),
        "clipped_render_rows_by_synth": dict(clipped_rows_by_synth),
        "max_peak_dbfs_by_synth": dict(max_peak_by_synth),
        "duration_range_s": [min_duration, max_duration],
        "note_set_errors": note_set_errors,
        "file_errors": file_errors,
        "duration_errors": duration_errors,
        "unexpected_silent": unexpected_silent,
        "temporary_files": temporary_files,
        "checks": checks,
        "gate_pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"OUTPUT={args.output}")
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
