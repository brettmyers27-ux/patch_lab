"""Read-only library status and lightweight Phase 2 source refresh helpers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from core.db import Database
from core.library_state import get_presets_needing_preparation, reconcile_source_tree
from core.prepared_state import prepared_predicate
from core.preset_scan import discover_presets


@dataclass(frozen=True, slots=True)
class PresetLibraryStatus:
    """Authoritative counts used by the Prepare Preset Library UI."""

    found: int = 0
    ready: int = 0
    needs_preparation: int = 0
    failed: int = 0
    pending: int = 0


def preset_library_status(db_path: Path) -> PresetLibraryStatus:
    """Read current Phase 2 prepared state without scanning or processing."""

    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        return PresetLibraryStatus()
    try:
        database = Database(path)
        queue = get_presets_needing_preparation(database)
        with database.connect() as connection:
            prepared_sql, parameters = prepared_predicate("p")
            found = int(
                connection.execute(
                    "SELECT COUNT(*) FROM presets p WHERE p.is_factory=0 "
                    "AND EXISTS (SELECT 1 FROM preset_sources ps "
                    "WHERE ps.preset_id=p.id AND ps.active=1)"
                ).fetchone()[0]
            )
            ready = int(
                connection.execute(
                    "SELECT COUNT(*) FROM presets p WHERE p.is_factory=0 "
                    f"AND {prepared_sql}",
                    parameters,
                ).fetchone()[0]
            )
            failed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM preparation_jobs j "
                    "JOIN presets p ON p.id=j.preset_id "
                    "WHERE p.is_factory=0 AND j.state='failed'"
                ).fetchone()[0]
            )
            pending = int(
                connection.execute(
                    "SELECT COUNT(*) FROM presets p WHERE p.is_factory=0 "
                    "AND p.pending_reason IS NOT NULL"
                ).fetchone()[0]
            )
        return PresetLibraryStatus(
            found=found,
            ready=ready,
            needs_preparation=len(queue),
            failed=failed,
            pending=pending,
        )
    except (OSError, sqlite3.Error):
        return PresetLibraryStatus()


def preparation_queue_ids(db_path: Path) -> list[int]:
    """Snapshot exactly the Phase 2 items eligible for one preparation run."""

    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        return []
    return [record.id for record in get_presets_needing_preparation(Database(path))]


def refresh_preset_library(root: Path, db_path: Path) -> dict[str, int]:
    """Reconcile a linked folder without Serum, rendering, or analysis."""

    database = Database(Path(db_path).expanduser().resolve())
    result = reconcile_source_tree(Path(root), database, paths=discover_presets(Path(root)))
    status = preset_library_status(database.path)
    return {
        "found": status.found,
        "ready": status.ready,
        "needs_preparation": status.needs_preparation,
        "new": result.new_content,
        "changed": result.changed_sources,
        "removed": result.sources_deactivated,
    }
