"""Persistent source snapshots, preparation revisions, and Match eligibility."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from core.db import Database
from core.factory_match import _local_search_rows
from core.library_state import (
    active_source_count,
    get_presets_needing_preparation,
    is_preset_prepared,
    mark_preset_prepared,
    reconcile_source_tree,
)
from core.plugin_host import ParameterValue
from core.prepared_state import (
    CURRENT_PREPARED_REVISION,
    RENDER_MIDI_NOTES,
    REQUIRED_FINGERPRINT_NOTES,
)
from core.preset_scan import sha1_file
from core.render import MIDI_NOTES


EMBEDDING = np.zeros(512, dtype=np.float32).tobytes()
HANDCRAFTED = np.zeros(9, dtype=np.float32).tobytes()


def test_prepared_contract_uses_the_existing_render_note_set() -> None:
    assert RENDER_MIDI_NOTES == MIDI_NOTES


def _prepare(database: Database, preset_id: int, *, synth: str = "serum1") -> None:
    if synth == "serum1":
        database.replace_params(
            preset_id,
            [ParameterValue(0, "Volume", 0.5, "50%")],
            "test",
        )
    else:
        database.replace_serum2_full_settings(
            preset_id,
            metadata_json="{}",
            settings_json="{}",
            settings_sha256="sha",
            payload_version=1,
            cbor_length=2,
            compressed_length=2,
        )
    for note in REQUIRED_FINGERPRINT_NOTES:
        database.upsert_fingerprint(preset_id, note, EMBEDDING, HANDCRAFTED)
    with database.connect() as connection:
        connection.execute(
            "UPDATE presets SET status='embedded' WHERE id=?", (preset_id,)
        )
    assert mark_preset_prepared(database, preset_id)


def test_fresh_then_unchanged_scan_hashes_only_once(tmp_path: Path) -> None:
    root = tmp_path / "presets"
    root.mkdir()
    source = root / "Bass.fxp"
    source.write_bytes(b"bass")
    database = Database(tmp_path / "library.db")
    hashed: list[Path] = []

    def counted(path: Path) -> str:
        from core.preset_scan import sha1_file

        hashed.append(path)
        return sha1_file(path)

    first = reconcile_source_tree(root, database, hash_file=counted)
    second = reconcile_source_tree(root, database, hash_file=counted)

    assert first.hashes_computed == 1
    assert second.hashes_computed == 0
    assert len(hashed) == 1
    assert active_source_count(database, first.entries[0].preset_id) == 1
    assert [row.id for row in get_presets_needing_preparation(database)] == [
        first.entries[0].preset_id
    ]


def test_prepared_unchanged_and_duplicate_paths_never_requeue(tmp_path: Path) -> None:
    root = tmp_path / "presets"
    (root / "one").mkdir(parents=True)
    original = root / "one" / "Lead.fxp"
    original.write_bytes(b"same-content")
    database = Database(tmp_path / "library.db")
    first = reconcile_source_tree(root, database)
    preset_id = first.entries[0].preset_id
    _prepare(database, preset_id)

    (root / "two").mkdir()
    duplicate = root / "two" / "Lead Copy.fxp"
    duplicate.write_bytes(original.read_bytes())
    copied = reconcile_source_tree(root, database)

    assert copied.hashes_computed == 1
    assert {entry.preset_id for entry in copied.entries} == {preset_id}
    assert active_source_count(database, preset_id) == 2
    assert get_presets_needing_preparation(database) == []

    original.unlink()
    removed_one = reconcile_source_tree(root, database)
    assert removed_one.sources_deactivated == 1
    assert active_source_count(database, preset_id) == 1
    assert is_preset_prepared(database, preset_id)


def test_move_or_rename_hashes_new_path_but_does_not_reprocess(tmp_path: Path) -> None:
    root = tmp_path / "presets"
    root.mkdir()
    old = root / "Old.fxp"
    old.write_bytes(b"move-me")
    database = Database(tmp_path / "library.db")
    preset_id = reconcile_source_tree(root, database).entries[0].preset_id
    _prepare(database, preset_id)

    new = root / "Renamed.fxp"
    old.rename(new)
    moved = reconcile_source_tree(root, database)

    assert moved.hashes_computed == 1
    assert moved.sources_deactivated == 1
    assert moved.entries[0].preset_id == preset_id
    assert active_source_count(database, preset_id) == 1
    assert get_presets_needing_preparation(database) == []


def test_modified_path_retires_old_mapping_and_queues_only_new_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "presets"
    root.mkdir()
    source = root / "Bass.fxp"
    source.write_bytes(b"version-a")
    database = Database(tmp_path / "library.db")
    old_id = reconcile_source_tree(root, database).entries[0].preset_id
    _prepare(database, old_id)

    source.write_bytes(b"version-b-longer")
    changed = reconcile_source_tree(root, database)
    new_id = changed.entries[0].preset_id

    assert changed.hashes_computed == 1
    assert new_id != old_id
    assert active_source_count(database, old_id) == 0
    assert active_source_count(database, new_id) == 1
    assert [row.id for row in get_presets_needing_preparation(database)] == [new_id]
    with database.connect() as connection:
        source_history = connection.execute(
            "SELECT preset_id,active FROM preset_sources "
            "WHERE normalized_path=? ORDER BY id",
            (str(source.resolve()),),
        ).fetchall()
    assert [(int(row[0]), int(row[1])) for row in source_history] == [
        (old_id, 0),
        (new_id, 1),
    ]


def test_removed_content_is_inactive_and_absent_from_personal_match(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "presets"
    root.mkdir()
    source = root / "Pad.fxp"
    source.write_bytes(b"pad")
    database = Database(tmp_path / "library.db")
    preset_id = reconcile_source_tree(root, database).entries[0].preset_id
    _prepare(database, preset_id)
    monkeypatch.setattr("core.factory_match.user_presets_enabled", lambda: True)
    assert _local_search_rows(database.path)[1]

    source.unlink()
    removed = reconcile_source_tree(root, database)

    assert removed.sources_deactivated == 1
    assert active_source_count(database, preset_id) == 0
    assert get_presets_needing_preparation(database) == []
    assert _local_search_rows(database.path) == (None, [])


def test_serum1_and_serum2_permanent_state_requirements(tmp_path: Path) -> None:
    root = tmp_path / "presets"
    root.mkdir()
    (root / "One.fxp").write_bytes(b"s1")
    (root / "Two.serumpreset").write_bytes(b"s2")
    database = Database(tmp_path / "library.db")
    entries = reconcile_source_tree(root, database).entries
    by_suffix = {entry.path.suffix.casefold(): entry.preset_id for entry in entries}

    for preset_id in by_suffix.values():
        for note in REQUIRED_FINGERPRINT_NOTES:
            database.upsert_fingerprint(preset_id, note, EMBEDDING, HANDCRAFTED)
    assert not mark_preset_prepared(database, by_suffix[".fxp"])
    assert not mark_preset_prepared(database, by_suffix[".serumpreset"])

    database.replace_params(
        by_suffix[".fxp"], [ParameterValue(0, "Volume", 0.5, "50%")], "test"
    )
    database.replace_serum2_full_settings(
        by_suffix[".serumpreset"],
        metadata_json="{}",
        settings_json="{}",
        settings_sha256="sha",
        payload_version=1,
        cbor_length=2,
        compressed_length=2,
    )
    assert mark_preset_prepared(database, by_suffix[".fxp"])
    assert mark_preset_prepared(database, by_suffix[".serumpreset"])


def test_prepared_revision_selectively_invalidates(tmp_path: Path) -> None:
    root = tmp_path / "presets"
    root.mkdir()
    (root / "Ready.fxp").write_bytes(b"ready")
    database = Database(tmp_path / "library.db")
    preset_id = reconcile_source_tree(root, database).entries[0].preset_id
    _prepare(database, preset_id)

    assert is_preset_prepared(database, preset_id)
    assert get_presets_needing_preparation(database) == []

    changed_fingerprint = replace(
        CURRENT_PREPARED_REVISION, fingerprint="future-fingerprint-v2"
    )
    assert not is_preset_prepared(database, preset_id, revision=changed_fingerprint)
    assert [row.id for row in get_presets_needing_preparation(
        database, revision=changed_fingerprint
    )] == [preset_id]

    changed_clap = replace(CURRENT_PREPARED_REVISION, clap="future-clap-checkpoint")
    assert not is_preset_prepared(database, preset_id, revision=changed_clap)
    changed_serum2_only = replace(
        CURRENT_PREPARED_REVISION, serum2="future-serum2-decoder"
    )
    assert is_preset_prepared(database, preset_id, revision=changed_serum2_only), (
        "a Serum 2 decoder change must not stale compatible Serum 1 data"
    )
    assert is_preset_prepared(database, preset_id), (
        "unrelated application version changes cannot affect this data revision"
    )


def test_legacy_schema6_complete_migrates_but_partial_stays_incomplete(
    tmp_path: Path,
) -> None:
    path = tmp_path / "library.db"
    complete_source = tmp_path / "Complete.fxp"
    partial_source = tmp_path / "Partial.fxp"
    complete_source.write_bytes(b"complete")
    partial_source.write_bytes(b"partial")
    database = Database(path)
    complete_id, _ = database.insert_preset(
        path=complete_source, name="Complete", synth="serum1", content_hash="complete"
    )
    partial_id, _ = database.insert_preset(
        path=partial_source, name="Partial", synth="serum1", content_hash="partial"
    )
    missing_id, _ = database.insert_preset(
        path=tmp_path / "Missing.fxp",
        name="Missing",
        synth="serum1",
        content_hash="missing",
    )
    database.replace_params(
        complete_id, [ParameterValue(0, "Volume", 0.5, "50%")], "legacy"
    )
    database.replace_params(
        partial_id, [ParameterValue(0, "Volume", 0.5, "50%")], "legacy"
    )
    database.replace_params(
        missing_id, [ParameterValue(0, "Volume", 0.5, "50%")], "legacy"
    )
    for note in REQUIRED_FINGERPRINT_NOTES:
        database.upsert_fingerprint(complete_id, note, EMBEDDING, HANDCRAFTED)
        database.upsert_fingerprint(missing_id, note, EMBEDDING, HANDCRAFTED)
    database.upsert_fingerprint(partial_id, 0, EMBEDDING, HANDCRAFTED)
    with database.connect() as connection:
        connection.execute("DROP TABLE prepared_presets")
        connection.execute("DROP TABLE preset_sources")
        connection.execute("DELETE FROM schema_migrations")
        connection.execute("INSERT INTO schema_migrations(version) VALUES (6)")

    migrated = Database(path)
    assert is_preset_prepared(migrated, complete_id)
    assert not is_preset_prepared(migrated, partial_id)
    assert [row.id for row in get_presets_needing_preparation(migrated)] == [partial_id]
    assert active_source_count(migrated, missing_id) == 0
    assert not is_preset_prepared(migrated, missing_id)
    with migrated.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM prepared_presets WHERE preset_id=?", (missing_id,)
        ).fetchone()[0] == 1, "complete learned data is retained while its source is inactive"
    Database(path)
    assert is_preset_prepared(migrated, complete_id), "repeat migration is idempotent"


def test_legacy_modified_path_migration_activates_only_current_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "library.db"
    source = tmp_path / "Changed.fxp"
    source.write_bytes(b"current-content")
    database = Database(path)
    old_id, _ = database.insert_preset(
        path=source, name="Old", synth="serum1", content_hash="old-content-hash"
    )
    current_id, _ = database.insert_preset(
        path=source, name="Current", synth="serum1", content_hash=sha1_file(source)
    )
    with database.connect() as connection:
        connection.execute("DROP TABLE prepared_presets")
        connection.execute("DROP TABLE preset_sources")
        connection.execute("DELETE FROM schema_migrations")
        connection.execute("INSERT INTO schema_migrations(version) VALUES (6)")

    migrated = Database(path)

    assert active_source_count(migrated, old_id) == 0
    assert active_source_count(migrated, current_id) == 1


def test_five_thousand_prepared_plus_one_hundred_new_queues_only_new(
    tmp_path: Path,
) -> None:
    root = tmp_path / "presets"
    root.mkdir()
    for index in range(5_000):
        (root / f"p{index:04d}.fxp").write_bytes(f"preset-{index}".encode())
    database = Database(tmp_path / "library.db")
    initial = reconcile_source_tree(root, database)
    ids = [entry.preset_id for entry in initial.entries]
    revision = CURRENT_PREPARED_REVISION
    with database.connect() as connection:
        connection.executemany(
            "INSERT INTO params VALUES (?,0,'Volume',0.5,'50%')",
            [(preset_id,) for preset_id in ids],
        )
        for note in REQUIRED_FINGERPRINT_NOTES:
            connection.executemany(
                "INSERT INTO fingerprints(preset_id,midi_note,embedding_f32,handcrafted_f32) "
                "VALUES (?,?,zeroblob(2048),zeroblob(36))",
                [(preset_id, note) for preset_id in ids],
            )
        connection.executemany(
            "INSERT INTO prepared_presets VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
            [
                (
                    preset_id,
                    revision.render,
                    revision.fingerprint,
                    revision.clap,
                    revision.handcrafted,
                    revision.serum1,
                    revision.serum2,
                )
                for preset_id in ids
            ],
        )
    for index in range(5_000, 5_100):
        (root / f"p{index:04d}.fxp").write_bytes(f"preset-{index}".encode())

    update = reconcile_source_tree(root, database)
    queue = get_presets_needing_preparation(database)

    assert update.hashes_computed == 100
    assert len(queue) == 100
    assert {row.name for row in queue} == {f"p{index:04d}" for index in range(5_000, 5_100)}
