from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.stage2_rebuild_embeddings import renderability_inventory


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
