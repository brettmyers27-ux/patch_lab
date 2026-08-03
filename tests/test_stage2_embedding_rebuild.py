from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from scripts.stage2_rebuild_embeddings import (
    _close_memmaps,
    _copy_runtime_targets,
    renderability_inventory,
)


def test_renderability_inventory_reports_missing_catalog_rows(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "catalog.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE presets(id INTEGER PRIMARY KEY,path TEXT,content_hash TEXT)")
    connection.execute("INSERT INTO presets VALUES (1,?,?)", (str(tmp_path / "gone.fxp"), "abc"))
    connection.commit()
    connection.close()

    class Assets:
        factory_mapping = tmp_path / "mapping.json"

        @staticmethod
        def find_render_state(_preset_id: int):
            return None

    class Verification:
        local_paths_by_hash = {}

    monkeypatch.setattr("scripts.stage2_rebuild_embeddings.resolve_synthesis_assets", lambda: Assets())
    monkeypatch.setattr("scripts.stage2_rebuild_embeddings.verify_local_factory_install", lambda **_kwargs: Verification())
    result = renderability_inventory([1, 2], [1, 2], library_db=database)
    assert result["available"] == 0
    assert result["missing"] == 2
    assert result["complete_embedding_world_possible"] is False


def test_renderability_inventory_reads_transferred_mapping_without_overwriting_it(
    tmp_path: Path, monkeypatch
) -> None:
    preset = tmp_path / "transferred.fxp"
    preset.write_bytes(b"fixture")
    mapping = tmp_path / "preset-paths.json"
    mapping.write_text(
        json.dumps({"local_paths_by_hash": {"abc": str(preset)}}), encoding="utf-8"
    )
    database = tmp_path / "catalog.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE presets(id INTEGER PRIMARY KEY,path TEXT,content_hash TEXT)")
    connection.execute("INSERT INTO presets VALUES (1,?,?)", (str(tmp_path / "gone.fxp"), "abc"))
    connection.commit()
    connection.close()

    class Assets:
        factory_mapping = mapping

        @staticmethod
        def find_render_state(_preset_id: int):
            return None

    class Verification:
        local_paths_by_hash = {}

    monkeypatch.setattr("scripts.stage2_rebuild_embeddings.resolve_synthesis_assets", lambda: Assets())
    monkeypatch.setattr("scripts.stage2_rebuild_embeddings.verify_local_factory_install", lambda: Verification())

    result = renderability_inventory([1], [1], library_db=database)

    assert result["available"] == 1
    assert json.loads(mapping.read_text(encoding="utf-8"))["local_paths_by_hash"]["abc"] == str(preset)


def test_rebuild_copies_serum2_runtime_targets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    payload = b"runtime targets"
    (source / "serum2_targets.npz").write_bytes(payload)

    _copy_runtime_targets(source, output)

    assert (output / "serum2_targets.npz").read_bytes() == payload


def test_rebuild_closes_memmaps_before_windows_child_processes(tmp_path: Path) -> None:
    path = tmp_path / "mapped.npy"
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(4,))
    mapped[:] = 1.0

    _close_memmaps(mapped)

    replacement = np.full(4, 2.0, dtype=np.float32)
    np.save(path, replacement)
    assert np.array_equal(np.load(path), replacement)
