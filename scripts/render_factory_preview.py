#!/usr/bin/env python3
"""Render one locally installed factory preset only when the user auditions it."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.platform_env import ENV
from core.plugin_host import make_dawdreamer_processor
from core.preview_cache import preview_cache_path
from core.preset_scan import sha1_file
from core.render import SAMPLE_RATE, _render_audio, _trim_tail
from core.serum2_preset import parse_serum2_preset
from core.serum2_state_reconstruct import decode_host_template, reconstruct_vstpreset
from core.synthesis_assets import SynthesisAssets, resolve_synthesis_assets


def _catalog_render_state(
    content_hash: str, assets: SynthesisAssets | None = None
) -> Path | None:
    """Resolve a shipped Serum 2 state by stable content hash, not bundle ID."""

    resolved = assets or resolve_synthesis_assets()
    with closing(sqlite3.connect(resolved.library_db)) as connection:
        row = connection.execute(
            "SELECT id FROM presets WHERE synth='serum2' AND content_hash=?",
            (content_hash,),
        ).fetchone()
    return resolved.find_render_state(int(row[0])) if row is not None else None


def render_preview(
    source: Path,
    synth: str,
    midi_note: int,
    content_hash: str,
    *,
    output_root: Path | None = None,
) -> Path:
    source = source.expanduser().resolve()
    if sha1_file(source) != content_hash:
        raise RuntimeError("Local factory preset no longer matches its verified fingerprint")
    output = preview_cache_path(
        Path(output_root).resolve() if output_root else ENV.app_data_dir,
        content_hash,
        midi_note,
    )
    if output.is_file():
        return output
    required = "VST2" if synth == "serum1" else "VST3"
    candidate = next(
        item
        for item in ENV.plugins_for(synth)
        if item.format == required and item.hostable
    )
    engine, processor = make_dawdreamer_processor(candidate)
    if synth == "serum1":
        if processor.load_preset(str(source)) is False:
            raise RuntimeError("Serum 1 rejected the local factory preset")
    else:
        shipped_state = _catalog_render_state(content_hash)
        if shipped_state is not None:
            if processor.load_vst3_preset(str(shipped_state.resolve())) is False:
                raise RuntimeError("Serum 2 rejected the shipped factory state")
            audio = _trim_tail(_render_audio(engine, processor, midi_note))
        else:
            # User-linked presets do not have a shipped state. Retain the
            # reconstruction path for them, but factory previews and the
            # installer parity gate use the same verified DawDreamer state as
            # production synthesis and do not depend on Pedalboard VST3 scan.
            from pedalboard import load_plugin

            live = load_plugin(str(candidate.path), plugin_name="Serum 2")
            template = decode_host_template(bytes(live.preset_data))
            state, _partition = reconstruct_vstpreset(
                parse_serum2_preset(source), template
            )
            with tempfile.TemporaryDirectory(
                prefix="patchlab-factory-preview-"
            ) as temporary:
                state_path = Path(temporary) / "state.vstpreset"
                state_path.write_bytes(state)
                if processor.load_vst3_preset(str(state_path)) is False:
                    raise RuntimeError("Serum 2 rejected the reconstructed factory state")
                audio = _trim_tail(_render_audio(engine, processor, midi_note))
    if synth == "serum1":
        audio = _trim_tail(_render_audio(engine, processor, midi_note))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(".tmp.wav")
    sf.write(
        temporary_output,
        audio.T,
        SAMPLE_RATE,
        subtype="FLOAT",
        format="WAV",
    )
    temporary_output.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--synth", choices=("serum1", "serum2"), required=True)
    parser.add_argument("--note", type=int, default=60)
    parser.add_argument("--content-hash", required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        output = render_preview(
            args.source,
            args.synth,
            args.note,
            args.content_hash,
            output_root=args.output_root,
        )
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
