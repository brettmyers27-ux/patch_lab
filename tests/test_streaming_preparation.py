"""Crash and restart guarantees for per-preset streaming preparation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from threading import Lock
import time

from core.db import Database, RenderRecord
from core.factory_match import _local_search_rows
from core.library_state import (
    is_preset_prepared,
    mark_preset_prepared,
    reconcile_source_tree,
)
from core.plugin_host import ParameterValue
from core.preparation import fingerprint_render_rows, prepare_work_queue
from core.prepared_state import PreparedRevision, REQUIRED_FINGERPRINT_NOTES
from core.render import MIDI_NOTES, RenderSummary


class SimulatedCrash(BaseException):
    pass


class FakeEmbedder:
    def __init__(self, _env) -> None:
        pass


def _catalog(
    tmp_path: Path,
    count: int = 1,
    *,
    synth: str = "serum1",
) -> tuple[Database, Path, list[int]]:
    root = tmp_path / "presets"
    root.mkdir(parents=True)
    suffix = ".fxp" if synth == "serum1" else ".serumpreset"
    for index in range(count):
        (root / f"Preset {index:03d}{suffix}").write_bytes(
            f"preset-{synth}-{index}".encode()
        )
    database = Database(tmp_path / "library.db")
    entries = reconcile_source_tree(root, database).entries
    ids = [entry.preset_id for entry in entries]
    if synth == "serum1":
        for preset_id in ids:
            database.replace_params(
                preset_id,
                [ParameterValue(0, "Volume", 0.5, "50%")],
                "test",
            )
    else:
        for preset_id in ids:
            database.replace_serum2_full_settings(
                preset_id,
                metadata_json="{}",
                settings_json="{}",
                settings_sha256="sha",
                payload_version=1,
                cbor_length=2,
                compressed_length=2,
            )
    return database, root, ids


def _write_render(path: Path, *, frames: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.full((frames, 2), 0.2, dtype=np.float32)
    sf.write(path, audio, 44_100, subtype="FLOAT", format="WAV")


def _renderer(counter: list[tuple[int, int]] | None = None, *, fail_id: int | None = None):
    def render(**kwargs):
        database = Database(kwargs["db_path"])
        preset_id = int(kwargs["preset_ids"][0])
        if preset_id == fail_id:
            raise RuntimeError("deterministic render failure")
        existing = database.existing_render_notes([preset_id]).get(preset_id, set())
        missing = [note for note in MIDI_NOTES if note not in existing]
        if counter is not None:
            counter.append((preset_id, len(missing)))
        rows = []
        for note in missing:
            wav = Path(kwargs["audio_root"]) / str(preset_id) / f"{note}.wav"
            _write_render(wav)
            rows.append(RenderRecord(preset_id, note, wav, -3.0, -18.0, 0.01))
        database.upsert_renders(rows)
        database.finalize_render_status(preset_id, MIDI_NOTES)
        return RenderSummary(
            selected_presets=1,
            queued_presets=1 if missing else 0,
            rendered_note_pairs=len(missing),
        )

    return render


def _fingerprinter(counter: list[int] | None = None, *, fail_id: int | None = None):
    def fingerprint(database: Database, _embedder, preset_id: int) -> bool:
        if counter is not None:
            counter.append(preset_id)
        if preset_id == fail_id:
            raise RuntimeError("deterministic fingerprint failure")
        for note in REQUIRED_FINGERPRINT_NOTES:
            database.upsert_fingerprint(
                preset_id,
                note,
                np.full(512, preset_id, dtype=np.float32).tobytes(),
                np.full(9, note, dtype=np.float32).tobytes(),
            )
        return True

    return fingerprint


def _run(
    tmp_path: Path,
    database: Database,
    *,
    render=None,
    fingerprint=None,
    **kwargs,
):
    embedder_factory = kwargs.pop("embedder_factory", FakeEmbedder)
    return prepare_work_queue(
        db_path=database.path,
        analysis_root=tmp_path / "analysis",
        legacy_audio_root=tmp_path / "legacy-audio",
        state_dir=tmp_path / "states",
        render_function=render or _renderer(),
        fingerprint_function=fingerprint or _fingerprinter(),
        embedder_factory=embedder_factory,
        **kwargs,
    )


def test_new_preset_streams_to_prepared_then_removes_wavs(tmp_path: Path) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)

    summary = _run(tmp_path, database)

    assert summary.prepared == 1
    assert summary.rendered_notes == 7
    assert summary.cleaned_files == 7
    assert is_preset_prepared(database, preset_id)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 0
    assert not list((tmp_path / "analysis").rglob("*.wav"))


def test_second_run_does_zero_expensive_work(tmp_path: Path) -> None:
    database, _root, _ids = _catalog(tmp_path)
    _run(tmp_path, database)
    render_calls: list[tuple[int, int]] = []
    fingerprint_calls: list[int] = []

    second = _run(
        tmp_path,
        database,
        render=_renderer(render_calls),
        fingerprint=_fingerprinter(fingerprint_calls),
    )

    assert second.queued == 0
    assert render_calls == []
    assert fingerprint_calls == []


def test_one_hundred_interrupt_after_sixty_then_only_forty_resume(
    tmp_path: Path,
) -> None:
    database, _root, ids = _catalog(tmp_path, 100)

    def cancelled() -> bool:
        with database.connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM prepared_presets").fetchone()[0] >= 60

    first = _run(tmp_path, database, cancel_check=cancelled)
    with database.connect() as connection:
        completed_before_resume = {
            int(row[0]) for row in connection.execute("SELECT preset_id FROM prepared_presets")
        }
    render_calls: list[tuple[int, int]] = []
    second = _run(tmp_path, database, render=_renderer(render_calls))

    # The bounded handoff can finish one already-scheduled neighbor before
    # cancellation takes effect; no later preset is started.
    assert first.prepared in {60, 61} and first.cancelled
    assert second.queued == 100 - first.prepared
    assert second.prepared == second.queued
    assert {preset_id for preset_id, _count in render_calls}.isdisjoint(completed_before_resume)


@pytest.mark.parametrize(
    "crash_stage",
    [
        "before_render",
        "after_render",
        "before_fingerprint",
        "after_features",
        "after_prepared_commit",
    ],
)
def test_crash_boundaries_recover_idempotently(
    tmp_path: Path,
    crash_stage: str,
) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)

    def crash(stage: str, _preset_id: int) -> None:
        if stage == crash_stage:
            raise SimulatedCrash(stage)

    with pytest.raises(SimulatedCrash):
        _run(tmp_path, database, stage_hook=crash)

    render_calls: list[tuple[int, int]] = []
    fingerprint_calls: list[int] = []
    recovered = _run(
        tmp_path,
        database,
        render=_renderer(render_calls),
        fingerprint=_fingerprinter(fingerprint_calls),
    )

    assert is_preset_prepared(database, preset_id)
    assert not list((tmp_path / "analysis").rglob("*.wav"))
    if crash_stage in {"after_render", "before_fingerprint"}:
        assert render_calls == []
    if crash_stage == "after_features":
        assert render_calls == [] and fingerprint_calls == []
    if crash_stage == "after_prepared_commit":
        assert recovered.queued == 0 and render_calls == [] and fingerprint_calls == []


def test_crash_during_render_reuses_completed_note(tmp_path: Path) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)

    def crash_render(**kwargs):
        wav = Path(kwargs["audio_root"]) / str(preset_id) / f"{MIDI_NOTES[0]}.wav"
        _write_render(wav)
        Database(kwargs["db_path"]).upsert_renders(
            [RenderRecord(preset_id, MIDI_NOTES[0], wav, -3.0, -18.0, 0.01)]
        )
        raise SimulatedCrash("during render")

    with pytest.raises(SimulatedCrash):
        _run(tmp_path, database, render=crash_render)
    calls: list[tuple[int, int]] = []

    _run(tmp_path, database, render=_renderer(calls))

    assert calls == [(preset_id, 6)]
    assert is_preset_prepared(database, preset_id)


def test_crash_during_render_removes_abandoned_atomic_temp_file(
    tmp_path: Path,
) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)
    temporary = (
        tmp_path / "analysis" / "jobs" / str(preset_id) / ".24.123.tmp.wav"
    )

    def crash_render(**_kwargs):
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(b"incomplete")
        raise SimulatedCrash("during atomic render")

    with pytest.raises(SimulatedCrash):
        _run(tmp_path, database, render=crash_render)

    _run(tmp_path, database)

    assert not temporary.exists()
    assert is_preset_prepared(database, preset_id)


def test_crash_during_fingerprinting_retries_without_rerender(tmp_path: Path) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)

    def crash_fingerprint(db: Database, _embedder, target: int) -> bool:
        db.upsert_fingerprint(
            target,
            MIDI_NOTES[0],
            np.zeros(512, dtype=np.float32).tobytes(),
            np.zeros(9, dtype=np.float32).tobytes(),
        )
        raise SimulatedCrash("during fingerprint")

    with pytest.raises(SimulatedCrash):
        _run(tmp_path, database, fingerprint=crash_fingerprint)
    render_calls: list[tuple[int, int]] = []

    _run(tmp_path, database, render=_renderer(render_calls))

    assert render_calls == []
    assert is_preset_prepared(database, preset_id)


def test_cleanup_failure_keeps_prepared_and_retries_without_analysis(
    tmp_path: Path,
) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)

    def refuse(_path: Path) -> None:
        raise OSError("busy")

    first = _run(tmp_path, database, unlink_file=refuse)
    render_calls: list[tuple[int, int]] = []
    fingerprint_calls: list[int] = []
    second = _run(
        tmp_path,
        database,
        render=_renderer(render_calls),
        fingerprint=_fingerprinter(fingerprint_calls),
    )

    assert first.cleanup_failures == 1
    assert is_preset_prepared(database, preset_id)
    assert second.queued == 0
    assert render_calls == [] and fingerprint_calls == []
    assert not list((tmp_path / "analysis").rglob("*.wav"))


def test_valid_legacy_renders_fingerprint_without_rendering(tmp_path: Path) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)
    legacy = tmp_path / "legacy-audio" / str(preset_id)
    rows = []
    for note in MIDI_NOTES:
        wav = legacy / f"{note}.wav"
        _write_render(wav)
        rows.append(RenderRecord(preset_id, note, wav, -3.0, -18.0, 0.01))
    database.upsert_renders(rows)
    database.finalize_render_status(preset_id, MIDI_NOTES)

    summary = _run(
        tmp_path,
        database,
        render=lambda **_kwargs: pytest.fail("legacy renders should be reused"),
    )

    assert summary.reused_render_notes == 7
    assert summary.rendered_notes == 0
    assert is_preset_prepared(database, preset_id)
    assert not legacy.exists() or not list(legacy.glob("*.wav"))


def test_missing_wav_despite_db_row_is_rerendered(tmp_path: Path) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)
    legacy = tmp_path / "legacy-audio" / str(preset_id)
    rows = []
    for note in MIDI_NOTES:
        wav = legacy / f"{note}.wav"
        if note != MIDI_NOTES[-1]:
            _write_render(wav)
        rows.append(RenderRecord(preset_id, note, wav, -3.0, -18.0, 0.01))
    database.upsert_renders(rows)
    calls: list[tuple[int, int]] = []

    _run(tmp_path, database, render=_renderer(calls))

    assert calls == [(preset_id, 1)]
    assert is_preset_prepared(database, preset_id)


def test_compatible_stale_fingerprint_revision_reuses_legacy_renders(
    tmp_path: Path,
) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)
    legacy = tmp_path / "legacy-audio" / str(preset_id)
    rows = []
    for note in MIDI_NOTES:
        wav = legacy / f"{note}.wav"
        _write_render(wav)
        rows.append(RenderRecord(preset_id, note, wav, -3.0, -18.0, 0.01))
    database.upsert_renders(rows)
    assert _fingerprinter()(database, None, preset_id)
    assert mark_preset_prepared(
        database,
        preset_id,
        revision=PreparedRevision(fingerprint="legacy-fingerprint-v0"),
    )
    fingerprint_calls: list[int] = []

    summary = _run(
        tmp_path,
        database,
        render=lambda **_kwargs: pytest.fail("compatible renders should be reused"),
        fingerprint=_fingerprinter(fingerprint_calls),
    )

    assert summary.reused_render_notes == 7
    assert fingerprint_calls == [preset_id]
    assert is_preset_prepared(database, preset_id)


def test_stale_render_revision_forces_safe_rerender(tmp_path: Path) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)
    legacy = tmp_path / "legacy-audio" / str(preset_id)
    rows = []
    for note in MIDI_NOTES:
        wav = legacy / f"{note}.wav"
        _write_render(wav)
        rows.append(RenderRecord(preset_id, note, wav, -3.0, -18.0, 0.01))
    database.upsert_renders(rows)
    assert _fingerprinter()(database, None, preset_id)
    assert mark_preset_prepared(
        database,
        preset_id,
        revision=PreparedRevision(render="legacy-render-v0"),
    )
    render_calls: list[tuple[int, int]] = []

    summary = _run(tmp_path, database, render=_renderer(render_calls))

    assert render_calls == [(preset_id, 7)]
    assert summary.reused_render_notes == 0
    assert is_preset_prepared(database, preset_id)


def test_incompatible_serum_schema_is_not_relabelled_or_cleaned(
    tmp_path: Path,
) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path)
    legacy = tmp_path / "legacy-audio" / str(preset_id)
    rows = []
    for note in MIDI_NOTES:
        wav = legacy / f"{note}.wav"
        _write_render(wav)
        rows.append(RenderRecord(preset_id, note, wav, -3.0, -18.0, 0.01))
    database.upsert_renders(rows)
    assert _fingerprinter()(database, None, preset_id)
    assert mark_preset_prepared(
        database,
        preset_id,
        revision=PreparedRevision(serum1="legacy-serum-schema-v0"),
    )

    summary = _run(tmp_path, database)

    assert summary.failed == 1
    assert not is_preset_prepared(database, preset_id)
    assert len(list(legacy.glob("*.wav"))) == 7
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM fingerprints WHERE preset_id=?", (preset_id,)
        ).fetchone()[0] == 8
        assert connection.execute(
            "SELECT serum1_schema_revision FROM prepared_presets WHERE preset_id=?",
            (preset_id,),
        ).fetchone()[0] == "legacy-serum-schema-v0"


def test_one_failure_does_not_stop_other_presets(tmp_path: Path) -> None:
    database, _root, ids = _catalog(tmp_path, 3)

    summary = _run(tmp_path, database, render=_renderer(fail_id=ids[1]))

    assert summary.failed == 1 and summary.prepared == 2
    assert is_preset_prepared(database, ids[0])
    assert not is_preset_prepared(database, ids[1])
    assert is_preset_prepared(database, ids[2])


def test_bounded_handoff_overlaps_one_render_with_one_analysis(tmp_path: Path) -> None:
    database, _root, ids = _catalog(tmp_path, 3)
    events: list[tuple[str, int, float]] = []
    guard = Lock()
    active_renders = 0
    maximum_renders = 0

    def render(**kwargs):
        nonlocal active_renders, maximum_renders
        preset_id = int(kwargs["preset_ids"][0])
        with guard:
            active_renders += 1
            maximum_renders = max(maximum_renders, active_renders)
            events.append(("render-start", preset_id, time.monotonic()))
        result = _renderer()(**kwargs)
        time.sleep(0.08)
        with guard:
            active_renders -= 1
            events.append(("render-end", preset_id, time.monotonic()))
        return result

    def fingerprint(db: Database, embedder, preset_id: int) -> bool:
        with guard:
            events.append(("analyze-start", preset_id, time.monotonic()))
        time.sleep(0.08)
        return _fingerprinter()(db, embedder, preset_id)

    summary = _run(tmp_path, database, render=render, fingerprint=fingerprint)

    assert summary.prepared == 3
    assert maximum_renders == 1
    render_starts = {event[1]: event[2] for event in events if event[0] == "render-start"}
    render_ends = {event[1]: event[2] for event in events if event[0] == "render-end"}
    analyze_starts = [event for event in events if event[0] == "analyze-start"]
    assert any(
        render_starts[preset_id] < at < render_ends[preset_id]
        for _kind, analyzed_id, at in analyze_starts
        for preset_id in render_starts
        if preset_id != analyzed_id
    )
    assert all(is_preset_prepared(database, preset_id) for preset_id in ids)


def test_cancel_keeps_completed_work_and_current_renders_recoverable(
    tmp_path: Path,
) -> None:
    database, _root, ids = _catalog(tmp_path, 3)
    checks = 0

    def cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    first = _run(tmp_path, database, cancel_check=cancel)
    second = _run(tmp_path, database)

    assert first.prepared <= 1 and first.cancelled
    assert second.queued == 3 - first.prepared
    assert second.prepared == second.queued
    assert all(is_preset_prepared(database, preset_id) for preset_id in ids)


def test_five_gb_and_bounded_temp_storage(tmp_path: Path) -> None:
    single_root = tmp_path / "single"
    single_database, _root, _ids = _catalog(single_root)
    single = _run(
        single_root,
        single_database,
        free_space_provider=lambda _path: 5 * 1024**3,
    )
    library_root = tmp_path / "library"
    database, _root, ids = _catalog(library_root, 20)

    summary = _run(
        library_root,
        database,
        free_space_provider=lambda _path: 5 * 1024**3,
    )

    assert summary.prepared == len(ids)
    # One render may wait while one prior preset is analyzed, but the handoff
    # never retains more than two preset work directories.
    assert single.peak_temp_bytes <= summary.peak_temp_bytes <= 2 * single.peak_temp_bytes
    assert summary.peak_temp_bytes < 5 * 1024**2
    assert not list((library_root / "analysis").rglob("*.wav"))


def test_streaming_fingerprints_equal_established_pipeline_output(
    tmp_path: Path,
) -> None:
    class DeterministicEmbedder:
        def __init__(self, _env=None) -> None:
            pass

        def embed(self, waveforms):
            rows = []
            for index, waveform in enumerate(waveforms):
                base = float(np.mean(waveform)) + index / 100.0
                rows.append(np.linspace(base, base + 1.0, 512, dtype=np.float32))
            return np.stack(rows)

    established_root = tmp_path / "established"
    established_db, _root, (established_id,) = _catalog(established_root)
    _renderer()(
        db_path=established_db.path,
        audio_root=established_root / "legacy-audio",
        state_dir=established_root / "states",
        preset_ids=[established_id],
        processes=1,
    )
    assert fingerprint_render_rows(
        established_db, DeterministicEmbedder(), established_id
    )

    streaming_root = tmp_path / "streaming"
    streaming_db, _root, (streaming_id,) = _catalog(streaming_root)
    _run(
        streaming_root,
        streaming_db,
        fingerprint=fingerprint_render_rows,
        embedder_factory=DeterministicEmbedder,
    )

    def permanent_features(database: Database, preset_id: int):
        with database.connect() as connection:
            return [
                (int(row[0]), bytes(row[1]), bytes(row[2]))
                for row in connection.execute(
                    "SELECT midi_note,embedding_f32,handcrafted_f32 "
                    "FROM fingerprints WHERE preset_id=? ORDER BY midi_note",
                    (preset_id,),
                )
            ]

    assert permanent_features(established_db, established_id) == permanent_features(
        streaming_db, streaming_id
    )
    assert is_preset_prepared(streaming_db, streaming_id)


def test_serum1_params_survive_preparation_and_cleanup(tmp_path: Path) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path, synth="serum1")

    _run(tmp_path, database)

    with database.connect() as connection:
        params = connection.execute(
            "SELECT param_name,norm_value FROM params WHERE preset_id=?",
            (preset_id,),
        ).fetchall()
    assert [(row[0], row[1]) for row in params] == [("Volume", 0.5)]
    assert is_preset_prepared(database, preset_id)


def test_serum2_contract_and_match_survive_analysis_wav_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, _root, (preset_id,) = _catalog(tmp_path, synth="serum2")
    monkeypatch.setattr("core.factory_match.user_presets_enabled", lambda: True)

    _run(tmp_path, database)
    matrix, rows = _local_search_rows(database.path, tmp_path / "analysis")

    assert is_preset_prepared(database, preset_id)
    assert matrix is not None and matrix.shape == (1, 512)
    assert rows[0]["preset_id"] == preset_id
    assert rows[0]["audition_path"] is None
