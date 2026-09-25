"""Phase 4 status is derived from Phase 2/3 durable state only."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.db import Database
from core.library_state import mark_preset_prepared, reconcile_source_tree
from core.library_status import preparation_queue_ids, preset_library_status, refresh_preset_library
from core.plugin_host import ParameterValue
from core.prepared_state import REQUIRED_FINGERPRINT_NOTES


def _prepared(database: Database, preset_id: int) -> None:
    database.replace_params(
        preset_id, [ParameterValue(0, "Master", 0.5, "50%")], "test"
    )
    for note in REQUIRED_FINGERPRINT_NOTES:
        database.upsert_fingerprint(
            preset_id,
            note,
            np.zeros(512, dtype=np.float32).tobytes(),
            np.zeros(9, dtype=np.float32).tobytes(),
        )
    assert mark_preset_prepared(database, preset_id)


def test_status_counts_only_the_current_phase2_queue(tmp_path: Path) -> None:
    root = tmp_path / "presets"
    root.mkdir()
    for index in range(12):
        (root / f"Preset {index}.fxp").write_bytes(f"preset-{index}".encode())
    database = Database(tmp_path / "library.db")
    discovery = reconcile_source_tree(root, database)
    for entry in discovery.entries[:10]:
        _prepared(database, entry.preset_id)

    status = preset_library_status(database.path)

    assert (status.found, status.ready, status.needs_preparation) == (12, 10, 2)
    assert preparation_queue_ids(database.path) == [
        entry.preset_id for entry in discovery.entries[10:]
    ]


def test_refresh_detects_new_changed_and_removed_without_preparation(tmp_path: Path) -> None:
    root = tmp_path / "presets"
    root.mkdir()
    first = root / "First.fxp"
    second = root / "Second.fxp"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    database = Database(tmp_path / "library.db")
    first_refresh = refresh_preset_library(root, database.path)
    first.write_bytes(b"changed")
    second.unlink()
    (root / "Third.fxp").write_bytes(b"three")

    refreshed = refresh_preset_library(root, database.path)

    assert first_refresh["new"] == 2
    assert refreshed["new"] == 2
    assert refreshed["changed"] == 1
    assert refreshed["removed"] == 1
    assert refreshed["found"] == 2
    assert refreshed["needs_preparation"] == 2
