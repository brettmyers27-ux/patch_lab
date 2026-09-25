from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import soundfile as sf

from core.db import Database
from core.factory_match import _local_search_rows
from core.library_state import mark_preset_prepared, reconcile_source_tree
from core.plugin_host import ParameterValue
from core.prepared_state import REQUIRED_FINGERPRINT_NOTES
from core.render import MIDI_NOTES
from core.storage import (
    StoragePreferences,
    adopt_legacy_renders,
    audio_root_size,
    compact_render_library,
    configured_audio_root,
    load_storage_preferences,
    migrate_audio_storage,
    prune_preview_cache,
    save_storage_preferences,
    storage_status,
)


def _write_wav(path: Path, *, amplitude: float = 0.5, seconds: float = 0.05) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(int(44_100 * seconds), 1)
    audio = np.full((frames, 2), amplitude, dtype=np.float32)
    sf.write(path, audio, 44_100, subtype="FLOAT", format="WAV")


def _environment(tmp_path: Path):
    return SimpleNamespace(app_data_dir=tmp_path / "app-data")


def test_preferences_are_cross_platform_paths_and_disconnection_is_explicit(
    tmp_path: Path,
) -> None:
    env = _environment(tmp_path)
    external = tmp_path / "removable" / "PatchLab Audio"
    preferences = StoragePreferences(str(external), True, 768)
    save_storage_preferences(preferences, env)

    assert load_storage_preferences(env) == preferences
    assert configured_audio_root(env) == external.resolve()
    unavailable = storage_status(env)
    assert unavailable.configured_external
    assert not unavailable.available
    assert "not connected" in unavailable.reason

    external.mkdir(parents=True)
    assert storage_status(env).available


def test_migration_verifies_updates_database_and_commits_preference_last(
    tmp_path: Path,
) -> None:
    env = _environment(tmp_path)
    source = tmp_path / "old-audio"
    destination = tmp_path / "external" / "PatchLab Audio"
    wav = source / "12" / "60.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(os.urandom(4096))
    database = Database(tmp_path / "library.db")
    preset_id, _ = database.insert_preset(
        path=tmp_path / "Preset.fxp",
        name="Preset",
        synth="serum1",
        content_hash="hash-12",
    )
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO renders VALUES (?,?,?,?,?,?)",
            (preset_id, 60, str(wav.resolve()), -1.0, -12.0, 5.0),
        )

    summary = migrate_audio_storage(
        source,
        destination,
        database_path=database.path,
        env=env,
    )

    moved = destination / "12" / "60.wav"
    assert summary.files == 1
    assert moved.read_bytes()
    assert not wav.exists()
    with database.connect() as connection:
        stored = connection.execute("SELECT wav_path FROM renders").fetchone()[0]
    assert Path(stored) == moved
    assert configured_audio_root(env) == destination.resolve()


def test_failed_migration_keeps_source_and_old_preference(tmp_path: Path) -> None:
    env = _environment(tmp_path)
    source = tmp_path / "old"
    source.mkdir()
    original = source / "60.wav"
    original.write_bytes(b"original")
    destination = tmp_path / "new"

    with patch("core.storage._copy_verified", side_effect=OSError("copy failed")):
        with pytest.raises(OSError, match="copy failed"):
            migrate_audio_storage(source, destination, env=env)

    assert original.read_bytes() == b"original"
    assert load_storage_preferences(env).audio_root is None


def test_compact_cleanup_only_removes_learned_renders_inside_root(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio"
    database = Database(tmp_path / "library.db")
    ids = []
    for index in range(2):
        preset_id, _ = database.insert_preset(
            path=tmp_path / f"Preset-{index}.fxp",
            name=f"Preset {index}",
            synth="serum1",
            content_hash=f"hash-{index}",
        )
        ids.append(preset_id)
        wav = audio / str(preset_id) / "60.wav"
        wav.parent.mkdir(parents=True)
        wav.write_bytes(b"audio" * 100)
        with database.connect() as connection:
            connection.execute(
                "INSERT INTO renders VALUES (?,?,?,?,?,?)",
                (preset_id, 60, str(wav.resolve()), -1.0, -12.0, 5.0),
            )
            connection.execute(
                "UPDATE presets SET status='rendered' WHERE id=?", (preset_id,)
            )
    database.upsert_fingerprint(
        ids[0],
        0,
        np.zeros(512, dtype=np.float32).tobytes(),
        np.zeros(10, dtype=np.float32).tobytes(),
    )

    summary = compact_render_library(database.path, audio)

    assert summary.files == 1
    assert not (audio / str(ids[0]) / "60.wav").exists()
    assert (audio / str(ids[1]) / "60.wav").is_file()
    with database.connect() as connection:
        statuses = dict(connection.execute("SELECT id,status FROM presets"))
        render_ids = {row[0] for row in connection.execute("SELECT preset_id FROM renders")}
    assert statuses[ids[0]] == "embedded"
    assert statuses[ids[1]] == "rendered"
    assert render_ids == {ids[1]}


def test_compact_cleanup_reclaims_only_safe_interrupted_and_legacy_residue(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio"
    database = Database(tmp_path / "library.db")
    preset_id, _ = database.insert_preset(
        path=tmp_path / "Preset.fxp",
        name="Preset",
        synth="serum1",
        content_hash="residue-hash",
    )
    render = audio / str(preset_id) / "60.wav"
    render.parent.mkdir(parents=True)
    render.write_bytes(b"render")
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO renders VALUES (?,?,?,?,?,?)",
            (preset_id, 60, str(render.resolve()), -1.0, -12.0, 5.0),
        )
        connection.execute("UPDATE presets SET status='rendered' WHERE id=?", (preset_id,))
    database.upsert_fingerprint(
        preset_id,
        0,
        np.zeros(512, dtype=np.float32).tobytes(),
        np.zeros(10, dtype=np.float32).tobytes(),
    )
    old_temporary = audio / "999" / ".60.123.tmp.wav"
    old_temporary.parent.mkdir(parents=True)
    old_temporary.write_bytes(b"interrupted")
    os.utime(old_temporary, (1, 1))
    recent_temporary = audio / "999" / ".60.456.tmp.wav"
    recent_temporary.write_bytes(b"active")
    legacy_generated = audio / "generated-legacy-preview" / "60.wav"
    legacy_hash = audio / ("a" * 40) / "60.wav"
    legacy_generated.parent.mkdir(parents=True)
    legacy_hash.parent.mkdir(parents=True)
    legacy_generated.write_bytes(b"preview")
    legacy_hash.write_bytes(b"preview")
    unknown_complete = audio / "999" / "60.wav"
    unknown_complete.write_bytes(b"keep")

    summary = compact_render_library(database.path, audio)

    assert summary.files == 4  # tracked render, old temp, two legacy previews
    assert not render.exists()
    assert not old_temporary.exists()
    assert recent_temporary.exists()
    assert not legacy_generated.exists()
    assert not legacy_hash.exists()
    assert unknown_complete.read_bytes() == b"keep"
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 0
        assert connection.execute(
            "SELECT status FROM presets WHERE id=?", (preset_id,)
        ).fetchone()[0] == "embedded"


def test_compact_cleanup_discards_failed_silent_wavs_but_keeps_the_failure(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio"
    database = Database(tmp_path / "library.db")
    preset_id, _ = database.insert_preset(
        path=tmp_path / "Silent.fxp",
        name="Silent",
        synth="serum1",
        content_hash="silent-hash",
    )
    render = audio / str(preset_id) / "60.wav"
    render.parent.mkdir(parents=True)
    render.write_bytes(b"silent-render")
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO renders VALUES (?,?,?,?,?,?)",
            (preset_id, 60, str(render.resolve()), -90.0, -90.0, 5.0),
        )
        connection.execute(
            "UPDATE presets SET status='failed_silent',error='Silent rendered MIDI notes: 60' "
            "WHERE id=?",
            (preset_id,),
        )

    summary = compact_render_library(database.path, audio)

    assert summary.files == 1
    assert not render.exists()
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 0
        row = connection.execute(
            "SELECT status,error FROM presets WHERE id=?", (preset_id,)
        ).fetchone()
    assert tuple(row) == ("failed_silent", "Silent rendered MIDI notes: 60")


def test_compaction_keeps_database_durable_when_wav_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio"
    database = Database(tmp_path / "library.db")
    preset_id, _ = database.insert_preset(
        path=tmp_path / "Preset.fxp",
        name="Preset",
        synth="serum1",
        content_hash="unlink-failure-hash",
    )
    render = audio / str(preset_id) / "60.wav"
    render.parent.mkdir(parents=True)
    render.write_bytes(b"render")
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO renders VALUES (?,?,?,?,?,?)",
            (preset_id, 60, str(render.resolve()), -1.0, -12.0, 5.0),
        )
        connection.execute("UPDATE presets SET status='rendered' WHERE id=?", (preset_id,))
    database.upsert_fingerprint(
        preset_id,
        0,
        np.zeros(512, dtype=np.float32).tobytes(),
        np.zeros(10, dtype=np.float32).tobytes(),
    )
    original_unlink = Path.unlink

    def fail_render_unlink(path: Path, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if path == render:
            raise OSError("simulated removable-drive interruption")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_render_unlink)
    compact_render_library(database.path, audio)

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 0
        assert connection.execute(
            "SELECT status FROM presets WHERE id=?", (preset_id,)
        ).fetchone()[0] == "embedded"
    assert render.exists()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    compact_render_library(database.path, audio)
    assert not render.exists()


def test_preview_cache_has_a_hard_lru_limit(tmp_path: Path) -> None:
    audio = tmp_path / "preview" / "audio" / "hash"
    audio.mkdir(parents=True)
    older = audio / "48.wav"
    newer = audio / "60.wav"
    older.write_bytes(b"a" * 700_000)
    newer.write_bytes(b"b" * 700_000)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    with patch("core.storage.MIN_PREVIEW_CACHE_MB", 1):
        summary = prune_preview_cache(tmp_path / "preview", 1)

    assert summary.files == 1
    assert not older.exists()
    assert newer.exists()


def test_compacted_local_match_stays_searchable_without_a_full_render(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "app-data" / "library.db")
    source = tmp_path / "Preset.fxp"
    source.write_bytes(b"preset")
    preset_id = reconcile_source_tree(tmp_path, database).entries[0].preset_id
    database.replace_params(
        preset_id,
        [ParameterValue(0, "Volume", 0.5, "50%")],
        "test",
    )
    for note in REQUIRED_FINGERPRINT_NOTES:
        database.upsert_fingerprint(
            preset_id,
            note,
            np.ones(512, dtype=np.float32).tobytes(),
            np.zeros(9, dtype=np.float32).tobytes(),
        )
    with database.connect() as connection:
        connection.execute(
            "UPDATE presets SET status='embedded',is_factory=0,name='Real Preset Name' "
            "WHERE id=?",
            (preset_id,),
        )
    assert mark_preset_prepared(database, preset_id)

    matrix, rows = _local_search_rows(database.path, tmp_path / "external-audio")

    assert matrix is not None and matrix.shape == (1, 512)
    assert rows[0]["name"] == "Real Preset Name"
    assert rows[0]["audition_path"] is None
    assert rows[0]["path"] == str(source)


def test_adopt_legacy_renders_recovers_an_uncataloged_but_still_known_preset(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "app-data" / "library.db")
    preset_id, _ = database.insert_preset(
        path=tmp_path / "Preset.fxp",
        name="Recoverable Preset",
        synth="serum1",
        content_hash="recoverable-hash",
    )
    legacy = tmp_path / "legacy backup" / str(preset_id)
    for note in MIDI_NOTES:
        _write_wav(legacy / f"{note}.wav")
    audio_root = tmp_path / "audio"

    summary = adopt_legacy_renders(
        legacy.parent, database_path=database.path, audio_root=audio_root
    )

    assert summary.presets_adopted == 1
    assert summary.notes_copied == len(MIDI_NOTES)
    assert summary.bytes_copied > 0
    for note in MIDI_NOTES:
        assert (audio_root / str(preset_id) / f"{note}.wav").is_file()
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT midi_note FROM renders WHERE preset_id=? ORDER BY midi_note", (preset_id,)
        ).fetchall()
        status = connection.execute(
            "SELECT status FROM presets WHERE id=?", (preset_id,)
        ).fetchone()[0]
    assert [row[0] for row in rows] == sorted(MIDI_NOTES)
    assert status == "rendered"
    # The legacy source is untouched -- adoption is copy-only.
    for note in MIDI_NOTES:
        assert (legacy / f"{note}.wav").is_file()


def test_adopt_legacy_renders_never_touches_an_already_complete_preset(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "app-data" / "library.db")
    preset_id, _ = database.insert_preset(
        path=tmp_path / "Preset.fxp",
        name="Already Rendered",
        synth="serum1",
        content_hash="already-rendered-hash",
    )
    audio_root = tmp_path / "audio"
    current = audio_root / str(preset_id)
    for note in MIDI_NOTES:
        _write_wav(current / f"{note}.wav", amplitude=0.9)
    with database.connect() as connection:
        connection.executemany(
            "INSERT INTO renders VALUES (?,?,?,?,?,?)",
            [
                (preset_id, note, str((current / f"{note}.wav").resolve()), -1.0, -12.0, 0.05)
                for note in MIDI_NOTES
            ],
        )
        connection.execute("UPDATE presets SET status='rendered' WHERE id=?", (preset_id,))
    original_bytes = {
        note: (current / f"{note}.wav").read_bytes() for note in MIDI_NOTES
    }

    legacy = tmp_path / "legacy backup" / str(preset_id)
    for note in MIDI_NOTES:
        _write_wav(legacy / f"{note}.wav", amplitude=0.1)  # deliberately different content

    summary = adopt_legacy_renders(
        legacy.parent, database_path=database.path, audio_root=audio_root
    )

    assert summary.presets_adopted == 0
    assert summary.presets_already_complete == 1
    assert summary.notes_copied == 0
    # The current, already-tracked renders must be byte-for-byte untouched.
    for note in MIDI_NOTES:
        assert (current / f"{note}.wav").read_bytes() == original_bytes[note]


def test_adopt_legacy_renders_skips_presets_no_longer_in_the_catalog(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "app-data" / "library.db")
    legacy = tmp_path / "legacy backup" / "999999"
    for note in MIDI_NOTES:
        _write_wav(legacy / f"{note}.wav")

    summary = adopt_legacy_renders(
        legacy.parent, database_path=database.path, audio_root=tmp_path / "audio"
    )

    assert summary.presets_uncataloged == 1
    assert summary.presets_adopted == 0
    assert not (tmp_path / "audio" / "999999").exists()


def test_adopt_legacy_renders_leaves_non_numeric_folders_unclassified(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "app-data" / "library.db")
    legacy_parent = tmp_path / "legacy backup"
    _write_wav(legacy_parent / "generated-abc123hash" / "24.wav")
    _write_wav(legacy_parent / "generated-abc123hash" / "36.wav")

    summary = adopt_legacy_renders(
        legacy_parent, database_path=database.path, audio_root=tmp_path / "audio"
    )

    assert summary.folders_unclassified == 1
    assert summary.presets_adopted == 0


def test_adopt_legacy_renders_does_not_partially_register_an_incomplete_folder(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "app-data" / "library.db")
    preset_id, _ = database.insert_preset(
        path=tmp_path / "Preset.fxp",
        name="Partial Preset",
        synth="serum1",
        content_hash="partial-hash",
    )
    legacy = tmp_path / "legacy backup" / str(preset_id)
    # Only two of the seven expected notes are present.
    _write_wav(legacy / f"{MIDI_NOTES[0]}.wav")
    _write_wav(legacy / f"{MIDI_NOTES[1]}.wav")

    summary = adopt_legacy_renders(
        legacy.parent, database_path=database.path, audio_root=tmp_path / "audio"
    )

    assert summary.presets_adopted == 0
    assert summary.presets_partial_conflict == 1
    with database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM renders WHERE preset_id=?", (preset_id,)
        ).fetchone()[0]
    assert count == 0
    assert not (tmp_path / "audio" / str(preset_id)).exists()


def test_audio_root_size_sums_real_files_and_tolerates_a_missing_root(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "audio"
    (audio / "1").mkdir(parents=True)
    (audio / "1" / "24.wav").write_bytes(b"a" * 1000)
    (audio / "1" / "36.wav").write_bytes(b"b" * 2000)

    assert audio_root_size(audio) == 3000
    assert audio_root_size(tmp_path / "does-not-exist") == 0
