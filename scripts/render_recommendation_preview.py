#!/usr/bin/env python3
"""Render one selected octave for the current generated recommendation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.features import CLAP_SAMPLE_RATE
from core.match_library import resolve_result_path
from core.matcher import (
    Candidate,
    _init_render_worker,
    _render_candidate_unsafe,
)


PREVIEW_NOTES = (24, 36, 48, 60, 72, 84, 96)


def render_recommendation(result_path: Path, midi_note: int) -> Path:
    result_path = Path(result_path).expanduser().resolve()
    if midi_note not in PREVIEW_NOTES:
        raise ValueError(f"Unsupported preview note {midi_note}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    recommendation = result.get("recommendation")
    if not isinstance(recommendation, dict):
        raise RuntimeError("The match has no generated recommendation")
    candidate_path = resolve_result_path(
        result_path, recommendation["candidate_path"]
    )
    stored = np.load(candidate_path)
    candidate = Candidate(
        synth=str(recommendation["synth"]),
        base_preset_id=int(recommendation["base_preset_id"]),
        vector=np.asarray(stored["vector"], dtype=np.float32),
        mask=np.asarray(stored["mask"], dtype=np.bool_),
        origin="audition-preview",
        exact_base=not bool(recommendation.get("meaningfully_modified", False)),
    )
    output = result_path.parent / f"recommendation-{midi_note}.wav"
    if output.is_file():
        return output
    with tempfile.TemporaryDirectory(
        prefix="patchlab-recommendation-preview-"
    ) as scratch:
        _init_render_worker(scratch)
        waveform, _coverage = _render_candidate_unsafe(
            (candidate, midi_note, 4.0)
        )
    temporary = output.with_suffix(".tmp.wav")
    sf.write(
        temporary,
        np.asarray(waveform, dtype=np.float32),
        CLAP_SAMPLE_RATE,
        subtype="FLOAT",
        format="WAV",
    )
    temporary.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--note", type=int, choices=PREVIEW_NOTES, required=True)
    args = parser.parse_args()
    try:
        output = render_recommendation(args.result, args.note)
    except Exception as exc:
        print(f"PREVIEW_ERROR={type(exc).__name__}: {exc}", flush=True)
        return 1
    print(
        "PREVIEW_RESULT="
        + json.dumps({"path": str(output), "midi_note": args.note}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
