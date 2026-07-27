#!/usr/bin/env python3
"""Render the resumable Serum library from the command line or GUI QProcess."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH, Database, PresetRecord
from core.render import DEFAULT_AUDIO_ROOT, RenderControl, render_library, summary_dict


REFERENCE_SERUM2_NAMES = {
    "KillerRattle",
    "BAM [BA] Just Hold E",
    "QUIX Lead RPS 2",
    "WA_RT_BS_Saw_Growl",
    "WA_RT_LD_Arpeggio",
}
_CONTROL: RenderControl | None = None


def evenly_spaced(records: list[PresetRecord], count: int) -> list[PresetRecord]:
    if count >= len(records):
        return records
    if count <= 1:
        return records[:count]
    indices = [round(index * (len(records) - 1) / (count - 1)) for index in range(count)]
    return [records[index] for index in indices]


def gate_selection(database: Database) -> list[int]:
    serum1 = evenly_spaced(database.renderable_presets("serum1"), 35)
    serum2_all = database.renderable_presets("serum2")
    references = [record for record in serum2_all if record.name in REFERENCE_SERUM2_NAMES]
    chosen = list(references)
    for record in evenly_spaced(serum2_all, 15):
        if record.id not in {item.id for item in chosen}:
            chosen.append(record)
        if len(chosen) == 15:
            break
    if len(chosen) < 15:
        for record in serum2_all:
            if record.id not in {item.id for item in chosen}:
                chosen.append(record)
            if len(chosen) == 15:
                break
    return [record.id for record in serum1 + chosen]


def control_ready(control: RenderControl) -> None:
    global _CONTROL
    _CONTROL = control

    def read_commands() -> None:
        for raw in sys.stdin:
            command = raw.strip().upper()
            if command == "PAUSE":
                control.pause()
                print("RENDER_CONTROL=paused", flush=True)
            elif command == "RESUME":
                control.resume()
                print("RENDER_CONTROL=resumed", flush=True)
            elif command == "CANCEL":
                control.cancel()
                print("RENDER_CONTROL=cancelling", flush=True)
                break

    threading.Thread(target=read_commands, name="render-command-reader", daemon=True).start()


def handle_signal(signum: int, _frame: object) -> None:
    if _CONTROL is not None:
        print(f"RENDER_CONTROL=signal-{signum}-cancelling", flush=True)
        _CONTROL.cancel()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preset-id", type=int, action="append")
    parser.add_argument("--gate50", action="store_true")
    parser.add_argument("--interrupt-after-presets", type=int)
    parser.add_argument("--pause-after-presets", type=int)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    args = parser.parse_args()
    print(f"RENDER_PROCESS_PID={os.getpid()}", flush=True)
    if args.gate50 and args.preset_id:
        parser.error("--gate50 and --preset-id are mutually exclusive")
    database = Database(args.db)
    preset_ids = gate_selection(database) if args.gate50 else args.preset_id
    if preset_ids is not None:
        selected = [record for record in database.renderable_presets() if record.id in set(preset_ids)]
        print(
            "RENDER_SELECTION="
            + json.dumps(
                {
                    "presets": len(selected),
                    "serum1": sum(record.synth == "serum1" for record in selected),
                    "serum2": sum(record.synth == "serum2" for record in selected),
                    "ids": [record.id for record in selected],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)

    interruption_sent = False
    pause_sent = False

    def progress(detail: dict[str, object]) -> None:
        nonlocal interruption_sent, pause_sent
        print("RENDER_PROGRESS=" + json.dumps(detail, sort_keys=True), flush=True)
        processed = int(detail.get("processed_presets", 0))
        if (
            args.pause_after_presets is not None
            and processed >= args.pause_after_presets
            and not pause_sent
            and _CONTROL is not None
        ):
            pause_sent = True
            _CONTROL.pause()
            print("RENDER_CONTROL=test-paused", flush=True)

            def resume_after_delay() -> None:
                time.sleep(max(args.pause_seconds, 0.0))
                if _CONTROL is not None:
                    _CONTROL.resume()
                    print("RENDER_CONTROL=test-resumed", flush=True)

            threading.Thread(target=resume_after_delay, daemon=True).start()
        if (
            args.interrupt_after_presets is not None
            and processed >= args.interrupt_after_presets
            and not interruption_sent
        ):
            interruption_sent = True
            os.kill(os.getpid(), signal.SIGTERM)

    summary = render_library(
        db_path=args.db,
        audio_root=args.audio_root,
        state_dir=args.state_dir,
        preset_ids=preset_ids,
        processes=args.workers,
        log=lambda message: print(message, flush=True),
        progress=progress,
        on_control_ready=control_ready,
    )
    result = summary_dict(summary)
    print("RENDER_SUMMARY=" + json.dumps(result, sort_keys=True), flush=True)
    return 130 if summary.cancelled else 0


if __name__ == "__main__":
    raise SystemExit(main())
