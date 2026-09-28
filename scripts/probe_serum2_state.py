"""Development-only, read-only headless probe for one local Serum 2 preset.

Prints stages and aggregate counts, never the preset name, source path, state
bytes, or rendered audio. The temporary reconstructed state stays alive until
the spawned render worker exits.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys
import tempfile
from pathlib import Path

from core.diagnostics import normalize_error_text
from core.plugin_host import inspect_vstpreset_bytes
from core.renderer_selection import open_renderer, require_renderer
from core.serum2_preset import parse_serum2_preset
from core.serum2_state_reconstruct import (
    decode_host_template,
    load_render_state,
    reconstruct_vstpreset,
)


def _worker(state_dir: str, result_queue: object) -> None:
    stage = "host"
    try:
        # Native plug-ins can write user sample paths to process stderr. The
        # probe reports only the caught stage and sanitized exception instead.
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        from core.render import _render_audio

        engine, processor, _selection = open_renderer("serum2")
        stage = "load"
        load_render_state(processor, 1, Path(state_dir))
        stage = "render"
        audio = _render_audio(engine, processor, 60)
        peak = float(abs(audio).max()) if audio.size else 0.0
        result_queue.put(("PASS", peak, ""))
    except Exception as exc:
        result_queue.put((stage.upper(), 0.0, normalize_error_text(f"{type(exc).__name__}: {exc}")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", type=Path, help="one locally available .SerumPreset file")
    args = parser.parse_args()
    try:
        source = args.preset.expanduser().resolve(strict=True)
        if source.suffix.casefold() != ".serumpreset":
            raise ValueError("expected a .SerumPreset file")
        selection = require_renderer("serum2", context="probing a local preset")
        from pedalboard import load_plugin

        assert selection.path is not None
        live = load_plugin(str(selection.path), plugin_name="Serum 2")
        template = decode_host_template(bytes(live.preset_data))
        preset = parse_serum2_preset(source)
        state, partition = reconstruct_vstpreset(preset, template)
        structure = inspect_vstpreset_bytes(state)
        chunks = {entry["id"] for entry in structure["entries"]}
        if chunks != {"Comp", "Cont"}:
            raise ValueError("reconstructed VST3 state lacks Comp or Cont")
    except Exception as exc:
        print("PREPARATION PROBE: FAILED BEFORE WORKER")
        print(normalize_error_text(f"{type(exc).__name__}: {exc}"))
        return 1

    print("PREPARATION PROBE: PARSE AND RECONSTRUCTION PASSED")
    print(f"GRAPH LEAF COVERAGE: {partition.matched_leaves}/{partition.total_leaves}")
    print(f"VST3 STATE BYTES: {len(state)}")
    with tempfile.TemporaryDirectory(prefix="patchlab-serum2-probe-") as directory:
        Path(directory, "1.vstpreset").write_bytes(state)
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        worker = context.Process(target=_worker, args=(directory, result_queue))
        worker.start()
        worker.join(timeout=90)
        if worker.is_alive():
            worker.terminate()
            worker.join()
            print("WORKER: TIMED OUT")
            return 1
        try:
            stage, peak, error = result_queue.get(timeout=2)
        except queue.Empty:
            print(f"WORKER: EXITED WITHOUT RESULT (code {worker.exitcode})")
            return 1
        if stage != "PASS":
            print(f"WORKER {stage}: FAILED")
            print(error)
            return 1
        print("WORKER VST3 LOAD: PASSED")
        print(f"WORKER NOTE 60 RENDER: {'AUDIBLE' if peak > 0.001 else 'SILENT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
