from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from core.db import Database
from core.factory_match import _local_search_rows
from core.storage import (
    StoragePreferences,
    compact_render_library,
    configured_audio_root,
    load_storage_preferences,
    migrate_audio_storage,
    prune_preview_cache,
    save_storage_preferences,
    storage_status,
)


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
    preset_id, _ = database.insert_preset(
        path=source,
        name="Real Preset Name",
        synth="serum1",
        content_hash="searchable-hash",
    )
    database.upsert_fingerprint(
        preset_id,
        0,
        np.ones(512, dtype=np.float32).tobytes(),
        np.zeros(10, dtype=np.float32).tobytes(),
    )
    with database.connect() as connection:
        connection.execute(
            "UPDATE presets SET status='embedded',is_factory=0 WHERE id=?",
            (preset_id,),
        )

    matrix, rows = _local_search_rows(database.path, tmp_path / "external-audio")

    assert matrix is not None and matrix.shape == (1, 512)
    assert rows[0]["name"] == "Real Preset Name"
    assert rows[0]["audition_path"] is None
    assert rows[0]["path"] == str(source)
