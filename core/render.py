"""Spawn-based, resumable offline rendering for Serum 1 and Serum 2."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf

from core.db import DEFAULT_DB_PATH, Database, PresetRecord, RenderRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_ROOT = PROJECT_ROOT / "data" / "audio"
MIDI_NOTES = (24, 36, 48, 60, 72, 84, 96)
SAMPLE_RATE = 44_100
HOLD_SECONDS = 4.0
TAIL_SECONDS = 4.0
FULL_DURATION_SECONDS = HOLD_SECONDS + TAIL_SECONDS
MIN_DURATION_SECONDS = 5.0
SILENCE_DBFS = -60.0
SILENCE_AMPLITUDE = 10.0 ** (SILENCE_DBFS / 20.0)


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class RenderTask:
    preset_id: int
    source_path: Path
    synth: str
    midi_notes: tuple[int, ...]
    audio_root: Path
    state_dir: Path


@dataclass(frozen=True, slots=True)
class RenderedNote:
    midi_note: int
    wav_path: Path
    peak_dbfs: float
    rms_dbfs: float
    duration_s: float


@dataclass(slots=True)
class PresetRenderResult:
    preset_id: int
    synth: str
    requested_notes: tuple[int, ...]
    rows: list[RenderedNote] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False


@dataclass(slots=True)
class RenderSummary:
    selected_presets: int = 0
    queued_presets: int = 0
    total_note_pairs: int = 0
    skipped_note_pairs: int = 0
    rendered_note_pairs: int = 0
    silent_note_pairs: int = 0
    clipped_note_pairs: int = 0
    failed_load_serum1: int = 0
    failed_load_serum2: int = 0
    failed_silent_serum1: int = 0
    failed_silent_serum2: int = 0
    elapsed_s: float = 0.0
    cancelled: bool = False


class RenderControl:
    """Cross-process pause/cancel flags exposed to CLI or GUI orchestration."""

    def __init__(self, pause_event: Any, cancel_event: Any) -> None:
        self.pause_event = pause_event
        self.cancel_event = cancel_event

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def cancel(self) -> None:
        self.cancel_event.set()


_HOSTS: dict[str, tuple[Any, Any]] = {}


def _worker_host(synth: str) -> tuple[Any, Any]:
    cached = _HOSTS.get(synth)
    if cached is not None:
        return cached
    from core.platform_env import ENV
    from core.plugin_host import make_dawdreamer_processor

    required_format = "VST2" if synth == "serum1" else "VST3"
    candidate = next(
        item for item in ENV.plugins_for(synth) if item.format == required_format and item.hostable
    )
    cached = make_dawdreamer_processor(candidate)
    _HOSTS[synth] = cached
    return cached


def _wait_if_paused(pause_event: Any, cancel_event: Any) -> bool:
    while pause_event.is_set() and not cancel_event.is_set():
        time.sleep(0.1)
    return cancel_event.is_set()


def _load_task_state(task: RenderTask, processor: Any) -> None:
    if task.synth == "serum1":
        if processor.load_preset(str(task.source_path.resolve())) is False:
            raise RuntimeError("DawDreamer load_preset returned False")
        return
    if task.synth == "serum2":
        from core.serum2_state_reconstruct import load_render_state

        load_render_state(processor, task.preset_id, task.state_dir)
        return
    raise ValueError(f"Unsupported synth {task.synth!r}")


def _render_audio(engine: Any, processor: Any, midi_note: int) -> np.ndarray:
    if hasattr(processor, "clear_midi"):
        processor.clear_midi()
    processor.add_midi_note(midi_note, 100, 0.0, HOLD_SECONDS)
    engine.render(FULL_DURATION_SECONDS)
    audio = np.asarray(engine.get_audio(), dtype=np.float32)
    if audio.ndim != 2:
        raise RuntimeError(f"Expected stereo matrix, received shape {audio.shape}")
    if audio.shape[0] != 2 and audio.shape[1] == 2:
        audio = audio.T
    if audio.shape[0] != 2:
        raise RuntimeError(f"Expected two channels, received shape {audio.shape}")
    return audio


def _trim_tail(audio: np.ndarray) -> np.ndarray:
    above = np.flatnonzero(np.max(np.abs(audio), axis=0) > SILENCE_AMPLITUDE)
    last_signal = int(above[-1] + 1) if above.size else 0
    minimum = min(audio.shape[1], int(round(MIN_DURATION_SECONDS * SAMPLE_RATE)))
    end = min(audio.shape[1], max(minimum, last_signal))
    return np.ascontiguousarray(audio[:, :end], dtype=np.float32)


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def _write_note(task: RenderTask, midi_note: int, audio: np.ndarray) -> RenderedNote:
    trimmed = _trim_tail(audio)
    peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(trimmed, dtype=np.float64)))) if trimmed.size else 0.0
    directory = task.audio_root / str(task.preset_id)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{midi_note}.wav"
    temporary = directory / f".{midi_note}.{os.getpid()}.tmp.wav"
    sf.write(temporary, trimmed.T, SAMPLE_RATE, subtype="FLOAT", format="WAV")
    temporary.replace(output)
    return RenderedNote(
        midi_note=midi_note,
        wav_path=output.resolve(),
        peak_dbfs=_dbfs(peak),
        rms_dbfs=_dbfs(rms),
        duration_s=trimmed.shape[1] / SAMPLE_RATE,
    )


def _process_task(task: RenderTask, pause_event: Any, cancel_event: Any) -> PresetRenderResult:
    result = PresetRenderResult(task.preset_id, task.synth, task.midi_notes)
    try:
        if _wait_if_paused(pause_event, cancel_event):
            result.cancelled = True
            return result
        engine, processor = _worker_host(task.synth)
        _load_task_state(task, processor)
        for midi_note in task.midi_notes:
            if _wait_if_paused(pause_event, cancel_event):
                result.cancelled = True
                break
            row = _write_note(task, midi_note, _render_audio(engine, processor, midi_note))
            result.rows.append(row)
            if row.rms_dbfs <= SILENCE_DBFS:
                result.warnings.append(
                    f"SILENT preset={task.preset_id} synth={task.synth} note={midi_note} "
                    f"RMS={row.rms_dbfs:.2f} dBFS"
                )
            if row.peak_dbfs > 0.0:
                result.warnings.append(
                    f"CLIPPING preset={task.preset_id} synth={task.synth} note={midi_note} "
                    f"peak={row.peak_dbfs:.2f} dBFS"
                )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
    return result


def _worker_loop(task_queue: Any, result_queue: Any, pause_event: Any, cancel_event: Any) -> None:
    try:
        while not cancel_event.is_set():
            task = task_queue.get()
            if task is None:
                break
            result_queue.put(_process_task(task, pause_event, cancel_event))
    finally:
        result_queue.put(("worker_done", os.getpid()))


def _select_records(database: Database, preset_ids: Sequence[int] | None) -> list[PresetRecord]:
    records = database.renderable_presets()
    if preset_ids is None:
        return records
    wanted = set(preset_ids)
    selected = [record for record in records if record.id in wanted]
    missing = wanted - {record.id for record in selected}
    if missing:
        raise KeyError(f"Preset ids are not renderable: {sorted(missing)}")
    return selected


def render_library(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    audio_root: Path = DEFAULT_AUDIO_ROOT,
    state_dir: Path | None = None,
    preset_ids: Sequence[int] | None = None,
    processes: int = 4,
    log: LogCallback = print,
    progress: ProgressCallback | None = None,
    on_control_ready: Callable[[RenderControl], None] | None = None,
) -> RenderSummary:
    """Render all missing note pairs and commit results only in the parent process."""

    if processes < 1:
        raise ValueError("processes must be positive")
    if state_dir is None:
        from core.serum2_state_reconstruct import DEFAULT_RENDER_STATE_DIR

        state_dir = DEFAULT_RENDER_STATE_DIR
    database = Database(db_path)
    records = _select_records(database, preset_ids)
    all_existing = database.existing_render_notes()
    tasks = []
    skipped = 0
    for record in records:
        existing = all_existing.get(record.id, set())
        missing = tuple(note for note in MIDI_NOTES if note not in existing)
        skipped += len(MIDI_NOTES) - len(missing)
        if missing:
            tasks.append(
                RenderTask(
                    record.id,
                    record.path,
                    record.synth,
                    missing,
                    Path(audio_root).resolve(),
                    Path(state_dir).resolve(),
                )
            )

    summary = RenderSummary(
        selected_presets=len(records),
        queued_presets=len(tasks),
        total_note_pairs=len(records) * len(MIDI_NOTES),
        skipped_note_pairs=skipped,
    )
    if not tasks:
        for record in records:
            database.finalize_render_status(record.id, MIDI_NOTES)
        return summary

    context = mp.get_context("spawn")
    pause_event = context.Event()
    cancel_event = context.Event()
    control = RenderControl(pause_event, cancel_event)
    if on_control_ready is not None:
        on_control_ready(control)
    task_queue = context.Queue()
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=_worker_loop,
            args=(task_queue, result_queue, pause_event, cancel_event),
            name=f"patchlab-render-{index + 1}",
        )
        for index in range(processes)
    ]
    for worker in workers:
        worker.start()
    for task in tasks:
        task_queue.put(task)
    for _worker in workers:
        task_queue.put(None)

    started = time.monotonic()
    done_workers = 0
    processed_tasks = 0
    resolved_pairs = skipped
    try:
        while done_workers < len(workers):
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                if not any(worker.is_alive() for worker in workers):
                    break
                continue
            if isinstance(message, tuple) and message and message[0] == "worker_done":
                done_workers += 1
                continue
            result: PresetRenderResult = message
            processed_tasks += 1
            rows = [
                RenderRecord(
                    preset_id=result.preset_id,
                    midi_note=row.midi_note,
                    wav_path=row.wav_path,
                    peak_dbfs=row.peak_dbfs,
                    rms_dbfs=row.rms_dbfs,
                    duration_s=row.duration_s,
                )
                for row in result.rows
            ]
            database.upsert_renders(rows)
            summary.rendered_note_pairs += len(rows)
            summary.silent_note_pairs += sum(row.rms_dbfs <= SILENCE_DBFS for row in rows)
            summary.clipped_note_pairs += sum(row.peak_dbfs > 0.0 for row in rows)
            for warning in result.warnings:
                log(warning)
            if result.error:
                database.mark_failed(result.preset_id, "failed_load", result.error)
                field_name = f"failed_load_{result.synth}"
                setattr(summary, field_name, getattr(summary, field_name) + 1)
                log(f"FAILED_LOAD preset={result.preset_id} synth={result.synth}: {result.error}")
            else:
                status = database.finalize_render_status(result.preset_id, MIDI_NOTES)
                if status == "failed_silent":
                    field_name = f"failed_silent_{result.synth}"
                    setattr(summary, field_name, getattr(summary, field_name) + 1)
                if result.cancelled:
                    summary.cancelled = True
            resolved_pairs += len(result.requested_notes) if not result.cancelled else len(rows)
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = summary.rendered_note_pairs / elapsed
            remaining = max(summary.total_note_pairs - resolved_pairs, 0)
            detail = {
                "processed_presets": processed_tasks,
                "queued_presets": len(tasks),
                "completed_note_pairs": resolved_pairs,
                "total_note_pairs": summary.total_note_pairs,
                "new_renders": summary.rendered_note_pairs,
                "renders_per_second": rate,
                "eta_seconds": remaining / rate if rate > 0 else None,
            }
            if progress is not None:
                progress(detail)
            if cancel_event.is_set():
                summary.cancelled = True
    except KeyboardInterrupt:
        summary.cancelled = True
        control.cancel()
    finally:
        if cancel_event.is_set():
            summary.cancelled = True
        for worker in workers:
            worker.join(timeout=10.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5.0)
        task_queue.close()
        result_queue.close()
        summary.elapsed_s = time.monotonic() - started
    return summary


def summary_dict(summary: RenderSummary) -> dict[str, Any]:
    return asdict(summary)
