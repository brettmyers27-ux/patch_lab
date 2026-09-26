"""Persistent incremental membership state for personal preset libraries."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from core.db import Database, PresetRecord
from core.prepared_state import (
    CURRENT_PREPARED_REVISION,
    PreparedRevision,
    is_preset_prepared as _is_preset_prepared,
    prepared_predicate,
    record_prepared_revision,
)
from core.preset_scan import discover_presets, sha1_file, synth_for


HashFile = Callable[[Path], str]
DiscoveryProgress = Callable[[int, int], None]


def normalize_source_path(path: Path) -> str:
    """Return the stable absolute path key used by source snapshots."""

    return os.path.normcase(str(Path(path).expanduser().resolve()))


@dataclass(frozen=True, slots=True)
class DiscoveredPreset:
    path: Path
    preset_id: int
    content_hash: str
    file_size: int
    mtime_ns: int
    hashed: bool
    new_content: bool


@dataclass(slots=True)
class DiscoveryResult:
    scan_id: str
    root: Path
    entries: list[DiscoveredPreset] = field(default_factory=list)
    hashes_computed: int = 0
    new_content: int = 0
    new_sources: int = 0
    changed_sources: int = 0
    unchanged_sources: int = 0
    sources_deactivated: int = 0


def _under_root(path_text: str, root: Path) -> bool:
    try:
        Path(path_text).relative_to(root)
    except ValueError:
        return False
    return True


def reconcile_source_tree(
    root: Path,
    database: Database,
    *,
    paths: Sequence[Path] | None = None,
    hash_file: HashFile = sha1_file,
    progress: DiscoveryProgress | None = None,
) -> DiscoveryResult:
    """Reconcile one tree without Serum, rendering, or feature extraction."""

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    discovered_paths = list(paths) if paths is not None else discover_presets(root)
    scan_id = uuid.uuid4().hex
    result = DiscoveryResult(scan_id=scan_id, root=root)

    with database.connect() as connection:
        source_rows = {
            str(row["normalized_path"]): row
            for row in connection.execute(
                "SELECT * FROM preset_sources WHERE active=1"
            ).fetchall()
        }
        preset_rows = {
            str(row["content_hash"]): row
            for row in connection.execute("SELECT * FROM presets").fetchall()
        }

    staged: list[tuple[Path, str, int, int, str, bool, bool, int | None]] = []
    seen_paths: set[str] = set()
    total = len(discovered_paths)
    for index, raw_path in enumerate(discovered_paths, start=1):
        path = Path(raw_path).expanduser().resolve()
        normalized = normalize_source_path(path)
        seen_paths.add(normalized)
        stat = path.stat()
        old = source_rows.get(normalized)
        unchanged = (
            old is not None
            and int(old["file_size"]) == int(stat.st_size)
            and int(old["mtime_ns"]) == int(stat.st_mtime_ns)
        )
        if unchanged:
            digest = str(old["content_hash"])
            preset_id = int(old["preset_id"])
            result.unchanged_sources += 1
            staged.append(
                (
                    path,
                    normalized,
                    stat.st_size,
                    stat.st_mtime_ns,
                    digest,
                    False,
                    False,
                    preset_id,
                )
            )
            if progress is not None:
                progress(index, total)
            continue
        digest = hash_file(path)
        result.hashes_computed += 1
        is_new_source = old is None
        if is_new_source:
            result.new_sources += 1
        else:
            result.changed_sources += 1
        known = preset_rows.get(digest)
        new_content = known is None
        staged.append(
            (
                path,
                normalized,
                stat.st_size,
                stat.st_mtime_ns,
                digest,
                True,
                new_content,
                int(known["id"]) if known is not None else None,
            )
        )
        if progress is not None:
            progress(index, total)

    affected_ids: set[int] = set()
    with database.connect() as connection:
        for (
            path,
            normalized,
            size,
            mtime_ns,
            digest,
            hashed,
            new_content,
            preset_id,
        ) in staged:
            old = source_rows.get(normalized)
            if old is not None:
                affected_ids.add(int(old["preset_id"]))
            if preset_id is None:
                known_now = connection.execute(
                    "SELECT id FROM presets WHERE content_hash=?", (digest,)
                ).fetchone()
                if known_now is not None:
                    preset_id = int(known_now["id"])
                    new_content = False
                else:
                    generation = synth_for(path)
                    if generation is None:
                        continue
                    cursor = connection.execute(
                        "INSERT INTO presets(path,name,synth,content_hash) VALUES (?,?,?,?)",
                        (normalized, path.stem, generation, digest),
                    )
                    preset_id = int(cursor.lastrowid)
                    result.new_content += 1
                    new_content = True
            affected_ids.add(preset_id)
            connection.execute(
                "UPDATE preset_sources SET active=0,updated_at=CURRENT_TIMESTAMP "
                "WHERE normalized_path=? AND content_hash<>? AND active=1",
                (normalized, digest),
            )
            connection.execute(
                """
                INSERT INTO preset_sources(
                  preset_id,normalized_path,source_root,file_size,mtime_ns,
                  content_hash,active,last_seen_scan
                ) VALUES (?,?,?,?,?,?,1,?)
                ON CONFLICT(normalized_path,content_hash) DO UPDATE SET
                  preset_id=excluded.preset_id,
                  source_root=excluded.source_root,
                  file_size=excluded.file_size,
                  mtime_ns=excluded.mtime_ns,
                  content_hash=excluded.content_hash,
                  active=1,
                  last_seen_scan=excluded.last_seen_scan,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (
                    preset_id,
                    normalized,
                    normalize_source_path(root),
                    size,
                    mtime_ns,
                    digest,
                    scan_id,
                ),
            )
            result.entries.append(
                DiscoveredPreset(
                    path=path,
                    preset_id=preset_id,
                    content_hash=digest,
                    file_size=size,
                    mtime_ns=mtime_ns,
                    hashed=hashed,
                    new_content=new_content,
                )
            )

        for normalized, old in source_rows.items():
            if (
                bool(old["active"])
                and normalized not in seen_paths
                and _under_root(normalized, root)
            ):
                connection.execute(
                    "UPDATE preset_sources SET active=0,updated_at=CURRENT_TIMESTAMP "
                    "WHERE normalized_path=?",
                    (normalized,),
                )
                affected_ids.add(int(old["preset_id"]))
                result.sources_deactivated += 1

        # Keep the legacy presets.path field useful to rendering and diagnostics.
        for preset_id in affected_ids:
            active = connection.execute(
                "SELECT normalized_path FROM preset_sources "
                "WHERE preset_id=? AND active=1 ORDER BY normalized_path LIMIT 1",
                (preset_id,),
            ).fetchone()
            if active is not None:
                connection.execute(
                    "UPDATE presets SET path=?,name=? WHERE id=?",
                    (str(active[0]), Path(str(active[0])).stem, preset_id),
                )
    return result


def is_preset_prepared(
    database: Database,
    preset_id: int,
    *,
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
) -> bool:
    with database.connect() as connection:
        return _is_preset_prepared(connection, preset_id, revision=revision)


def mark_preset_prepared(
    database: Database,
    preset_id: int,
    *,
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
) -> bool:
    with database.connect() as connection:
        recorded = record_prepared_revision(connection, preset_id, revision=revision)
        if not recorded:
            return False
        return _is_preset_prepared(connection, preset_id, revision=revision)


def get_presets_needing_preparation(
    database: Database,
    *,
    revision: PreparedRevision = CURRENT_PREPARED_REVISION,
) -> list[PresetRecord]:
    """Return active, supported content whose permanent Match state needs work."""

    prepared, parameters = prepared_predicate(revision=revision)
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT p.* FROM presets p "
            "WHERE EXISTS (SELECT 1 FROM preset_sources ps "
            "              WHERE ps.preset_id=p.id AND ps.active=1) "
            "AND p.is_factory=0 "
            "AND p.status!='failed_silent' "
            "AND (p.pending_reason IS NULL OR p.pending_reason IN "
            "     ('awaiting_processing','render_failed')) "
            f"AND NOT ({prepared}) "
            "ORDER BY p.id",
            parameters,
        ).fetchall()
    return [Database._preset(row) for row in rows]


def active_source_count(database: Database, preset_id: int) -> int:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM preset_sources WHERE preset_id=? AND active=1",
            (preset_id,),
        ).fetchone()
    return int(row[0])
