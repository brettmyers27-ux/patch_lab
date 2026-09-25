"""Upgrading an existing library to schema 8 must never lose user data.

A user upgrading has thousands of learned presets and a Match library. This is a
release blocker: the migration must succeed automatically, keep every row, keep
learned state, mark nothing as pending that was not pending, be idempotent, and
if it fails part-way leave the library exactly as it was.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

import core.db as db_module
from core.db import SCHEMA_SQL, SCHEMA_VERSION, Database

NEW_COLUMNS = (
    "file_format",
    "provenance",
    "compatible_renderers",
    "pending_reason",
    "last_attempt_at",
)

#: The columns a pre-1.5.4 (schema 5) ``presets`` table already had.
OLD_PRESET_COLUMNS = (
    "id",
    "path",
    "name",
    "synth",
    "content_hash",
    "load_strategy",
    "status",
    "error",
    "is_factory",
)

TABLES = (
    "presets",
    "params",
    "renders",
    "serum2_full_settings",
    "fingerprints",
    "favorites",
    "match_batches",
    "match_library",
)


def build_schema5_library(path: Path, *, presets: int = 40) -> None:
    """A populated library in exactly the layout the last production build wrote.

    The current base schema still describes every legacy table. New state tables
    are removed below so the fixture reproduces the old layout.
    """

    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA_SQL)
    # SCHEMA_SQL describes the current new-database shape. A real schema-5
    # database predates the incremental and preparation state tables.
    connection.execute("DROP TABLE preparation_jobs")
    connection.execute("DROP TABLE prepared_presets")
    connection.execute("DROP TABLE preset_sources")
    connection.execute("INSERT INTO schema_migrations(version) VALUES (5)")
    statuses = ("scanned", "params_dumped", "rendered", "embedded", "failed_load", "failed_silent")
    for index in range(1, presets + 1):
        status = statuses[index % len(statuses)]
        synth = "serum1" if index % 3 else "serum2"
        connection.execute(
            "INSERT INTO presets(id,path,name,synth,content_hash,load_strategy,status,error,is_factory)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                index,
                f"/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets/p{index}."
                + ("fxp" if synth == "serum1" else "SerumPreset"),
                f"Preset {index}",
                synth,
                f"hash-{index:04d}",
                "S1-dawdreamer-vst2-fxp" if synth == "serum1" else "VST3/S2",
                status,
                "boom" if status.startswith("failed") else None,
                index % 2,
            ),
        )
        if status in {"params_dumped", "rendered", "embedded"}:
            connection.executemany(
                "INSERT INTO params VALUES (?,?,?,?,?)",
                [(index, n, f"p{n}", 0.1 * n, f"{n}") for n in range(4)],
            )
        if status in {"rendered", "embedded"}:
            for note in (24, 36, 48):
                connection.execute(
                    "INSERT INTO renders VALUES (?,?,?,?,?,?)",
                    (index, note, f"/audio/{index}/{note}.wav", -3.0, -18.0, 2.0),
                )
        if status == "embedded":
            connection.execute(
                "INSERT INTO fingerprints(preset_id,midi_note,embedding_f32,handcrafted_f32)"
                " VALUES (?,?,?,?)",
                (index, 0, bytes(512 * 4), bytes(64)),
            )
        if synth == "serum2" and status in {"params_dumped", "rendered", "embedded"}:
            connection.execute(
                "INSERT INTO serum2_full_settings(preset_id,metadata_json,settings_json,"
                "settings_sha256,payload_version,cbor_length,compressed_length)"
                " VALUES (?,?,?,?,?,?,?)",
                (index, "{}", "{}", "sha", 1, 10, 5),
            )
    connection.execute("INSERT INTO favorites(content_hash) VALUES ('hash-0003')")
    connection.execute(
        "INSERT INTO match_batches(id,folder_name,source_folder,export_folder,target_synth,budget,status)"
        " VALUES (1,'b','/s','/e','serum2','balanced','complete')"
    )
    connection.execute(
        "INSERT INTO match_library(match_uid,source_name,source_audio_path,source_content_hash,"
        "result_json_path,target_synth,budget,base_name,recommendation_synth,batch_id)"
        " VALUES ('uid-1','bass','/a.wav','h','/r.json','serum2','balanced','Base','serum2',1)"
    )
    connection.commit()
    connection.close()


def dump(path: Path, *, preset_columns: tuple[str, ...] = OLD_PRESET_COLUMNS) -> dict[str, list]:
    """Every row of every table, using only the columns a schema-5 library had."""

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result: dict[str, list] = {}
        for table in TABLES:
            columns = ",".join(preset_columns) if table == "presets" else "*"
            result[table] = connection.execute(
                f"SELECT {columns} FROM {table} ORDER BY rowid"
            ).fetchall()
        return result
    finally:
        connection.close()


def columns_of(path: Path) -> set[str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {row[1] for row in connection.execute("PRAGMA table_info(presets)")}
    finally:
        connection.close()


def versions_of(path: Path) -> list[int]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return [row[0] for row in connection.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    finally:
        connection.close()


def tables_of(path: Path) -> set[str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        connection.close()


# ---------------------------------------------------------------------------


def test_upgrade_preserves_every_row_and_adds_the_new_columns(tmp_path: Path) -> None:
    path = tmp_path / "library.db"
    build_schema5_library(path)
    before = dump(path)
    assert not columns_of(path) & set(NEW_COLUMNS), "fixture must start as schema 5"
    assert versions_of(path) == [5]

    Database(path)

    assert dump(path) == before, "no existing row may change in any table"
    assert set(NEW_COLUMNS) <= columns_of(path)
    assert {"preset_sources", "prepared_presets", "preparation_jobs"} <= tables_of(
        path
    )
    assert versions_of(path) == [5, SCHEMA_VERSION]
    assert SCHEMA_VERSION == 8


def test_new_fields_start_empty_and_nothing_is_marked_pending(tmp_path: Path) -> None:
    """A migration must not reinterpret the library: no false pending state."""

    path = tmp_path / "library.db"
    build_schema5_library(path)
    database = Database(path)

    coverage = database.library_coverage()
    assert coverage["pending"] == 0
    assert database.pending_counts() == {}
    assert database.pending_presets() == []
    connection = sqlite3.connect(path)
    for column in NEW_COLUMNS:
        assert connection.execute(
            f"SELECT COUNT(*) FROM presets WHERE {column} IS NOT NULL"
        ).fetchone()[0] == 0, column
    connection.close()


def test_learned_and_match_state_survive_the_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "library.db"
    build_schema5_library(path)
    connection = sqlite3.connect(path)
    learned_before = connection.execute(
        "SELECT COUNT(DISTINCT preset_id) FROM fingerprints WHERE midi_note=0"
    ).fetchone()[0]
    renderable_before = connection.execute(
        "SELECT COUNT(DISTINCT preset_id) FROM params"
    ).fetchone()[0]
    connection.close()
    assert learned_before > 0 and renderable_before > 0

    database = Database(path)
    coverage = database.library_coverage()
    assert coverage["learned"] == learned_before
    assert coverage["processed_params"] == renderable_before
    assert len(database.renderable_presets()) == renderable_before
    # Match-time accessors still work on migrated rows.
    for record in database.presets_with_status(("embedded", "rendered")):
        assert record.path.suffix in {".fxp", ".SerumPreset"}
        assert record.pending_reason is None
        assert record.file_format is None, "unknown until the next scan, not guessed"


def test_migration_is_idempotent_and_a_current_library_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "library.db"
    build_schema5_library(path)
    Database(path)
    settled = dump(path, preset_columns=OLD_PRESET_COLUMNS + NEW_COLUMNS)
    versions = versions_of(path)

    statements: list[str] = []
    real_connect = sqlite3.connect

    def tracing_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(db_module.sqlite3, "connect", tracing_connect)
    for _ in range(3):
        Database(path)
    monkeypatch.undo()

    assert dump(path, preset_columns=OLD_PRESET_COLUMNS + NEW_COLUMNS) == settled
    assert versions_of(path) == versions
    ddl = [s for s in statements if s.lstrip().upper().startswith(("ALTER", "DROP", "DELETE"))]
    assert ddl == [], ddl
    assert not [s for s in statements if "BEGIN IMMEDIATE" in s.upper()], (
        "an already-current library must not take the migration write lock"
    )


def test_a_failure_part_way_rolls_the_whole_step_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic: a crash after some ALTERs must not leave a half-migrated table."""

    path = tmp_path / "library.db"
    build_schema5_library(path)
    before = dump(path)

    good = Database._PRESET_COLUMNS_ADDED_LATER
    # The last column is invalid (NOT NULL without a default cannot be added), so
    # the valid ones before it have already been applied when SQLite refuses.
    monkeypatch.setattr(
        Database,
        "_PRESET_COLUMNS_ADDED_LATER",
        good + (("poisoned", "INTEGER NOT NULL"),),
    )
    with pytest.raises(sqlite3.Error):
        Database(path)

    assert not columns_of(path) & set(NEW_COLUMNS), (
        "the columns added before the failure must have been rolled back"
    )
    assert "poisoned" not in columns_of(path)
    assert versions_of(path) == [5], "the schema version must not advance"
    assert dump(path) == before, "no data may change when the migration fails"

    # Recoverable: the next launch (with the defect gone) completes normally.
    monkeypatch.setattr(Database, "_PRESET_COLUMNS_ADDED_LATER", good)
    Database(path)
    assert set(NEW_COLUMNS) <= columns_of(path)
    assert versions_of(path) == [5, SCHEMA_VERSION]
    assert dump(path) == before


def test_a_very_old_library_without_is_factory_upgrades_too(tmp_path: Path) -> None:
    path = tmp_path / "library.db"
    build_schema5_library(path)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        ALTER TABLE presets RENAME TO presets_new;
        CREATE TABLE presets (
          id INTEGER PRIMARY KEY, path TEXT NOT NULL, name TEXT NOT NULL,
          synth TEXT NOT NULL CHECK (synth IN ('serum1','serum2')),
          content_hash TEXT NOT NULL UNIQUE, load_strategy TEXT,
          status TEXT NOT NULL DEFAULT 'scanned', error TEXT);
        INSERT INTO presets SELECT id,path,name,synth,content_hash,load_strategy,status,error FROM presets_new;
        DROP TABLE presets_new;
        """
    )
    connection.commit()
    connection.close()
    assert "is_factory" not in columns_of(path)

    Database(path)
    assert {"is_factory", *NEW_COLUMNS} <= columns_of(path)
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM presets").fetchone()[0] == 40
    assert connection.execute("SELECT DISTINCT is_factory FROM presets").fetchall() == [(0,)]
    connection.close()


def test_a_new_library_gets_the_same_final_shape(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.db"
    upgraded = tmp_path / "upgraded.db"
    Database(fresh)
    build_schema5_library(upgraded)
    Database(upgraded)
    assert columns_of(fresh) == columns_of(upgraded)
    assert versions_of(fresh) == [SCHEMA_VERSION]


def test_migration_can_be_retried_by_a_second_process_safely(tmp_path: Path) -> None:
    """Two PatchLab processes opening the same old library both end up correct."""

    path = tmp_path / "library.db"
    build_schema5_library(path)
    before = dump(path)
    first, second = Database(path), Database(path)
    assert first.path == second.path
    assert dump(path) == before
    assert versions_of(path) == [5, SCHEMA_VERSION]


REAL_BACKUP = (
    Path.home()
    / "Library/Application Support/Patch Lab/library.db.pre-adoption-backup-20260825174942"
)


@pytest.mark.skipif(
    not REAL_BACKUP.is_file(), reason="no real schema-5 library backup on this machine"
)
def test_a_copy_of_a_real_pre_upgrade_library_migrates_cleanly(tmp_path: Path) -> None:
    """The real thing: thousands of presets from an actual user library.

    Works on a copy; the original backup is only read.
    """

    path = tmp_path / "real-copy.db"
    shutil.copy2(REAL_BACKUP, path)
    assert 7 not in versions_of(path) and not columns_of(path) & set(NEW_COLUMNS)
    before = dump(path)
    assert len(before["presets"]) > 1000

    database = Database(path)

    assert dump(path) == before
    coverage = database.library_coverage()
    assert coverage["pending"] == 0
    assert coverage["discovered"] == len(before["presets"])
    assert set(NEW_COLUMNS) <= columns_of(path)
    assert versions_of(path)[-1] == SCHEMA_VERSION
    # Idempotent on real data as well.
    Database(path)
    assert dump(path) == before
