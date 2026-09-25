"""SQLite schema, migrations, and typed library accessors."""

from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from core.privacy import user_presets_enabled
from core.plugin_host import ParameterValue


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "library.db"
SCHEMA_VERSION = 8


@dataclass(frozen=True, slots=True)
class PresetRecord:
    id: int
    path: Path
    name: str
    synth: str
    content_hash: str
    load_strategy: str | None
    status: str
    error: str | None
    is_factory: bool
    # Identity facts kept separate from `synth` (the origin generation), because
    # collapsing format, origin, provenance and renderer compatibility into one
    # column is what made a Serum 2 library demand a Serum 1 renderer.
    file_format: str | None = None
    provenance: str | None = None
    compatible_renderers: str | None = None
    #: Stable machine-readable code from core.preset_identity, or None when the
    #: preset is processable on this machine.
    pending_reason: str | None = None
    last_attempt_at: str | None = None


@dataclass(frozen=True, slots=True)
class RenderRecord:
    preset_id: int
    midi_note: int
    wav_path: Path
    peak_dbfs: float
    rms_dbfs: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class MatchLibraryRecord:
    id: int
    match_uid: str
    source_name: str
    source_audio_path: Path
    source_content_hash: str
    result_json_path: Path
    target_synth: str
    budget: str
    similarity_percent: float
    base_name: str
    recommendation_synth: str
    no_confident_match: bool
    batch_id: int | None
    exported_preset_path: Path | None
    created_at: str


@dataclass(frozen=True, slots=True)
class MatchBatchRecord:
    id: int
    folder_name: str
    source_folder: str
    export_folder: str
    target_synth: str
    budget: str
    total_files: int
    completed_files: int
    failed_files: int
    status: str
    created_at: str


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS presets (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  name TEXT NOT NULL,
  synth TEXT NOT NULL CHECK (synth IN ('serum1','serum2')),
  content_hash TEXT NOT NULL UNIQUE,
  load_strategy TEXT,
  status TEXT NOT NULL DEFAULT 'scanned',
  error TEXT,
  is_factory INTEGER NOT NULL DEFAULT 0 CHECK (is_factory IN (0,1))
);
CREATE TABLE IF NOT EXISTS params (
  preset_id INTEGER REFERENCES presets(id) ON DELETE CASCADE,
  param_index INTEGER,
  param_name TEXT,
  norm_value REAL,
  display_value TEXT,
  PRIMARY KEY (preset_id, param_index)
);
CREATE TABLE IF NOT EXISTS renders (
  preset_id INTEGER REFERENCES presets(id) ON DELETE CASCADE,
  midi_note INTEGER,
  wav_path TEXT,
  peak_dbfs REAL,
  rms_dbfs REAL,
  duration_s REAL,
  PRIMARY KEY (preset_id, midi_note)
);
CREATE TABLE IF NOT EXISTS serum2_full_settings (
  preset_id INTEGER PRIMARY KEY REFERENCES presets(id) ON DELETE CASCADE,
  metadata_json TEXT NOT NULL,
  settings_json TEXT NOT NULL,
  settings_sha256 TEXT NOT NULL,
  payload_version INTEGER NOT NULL,
  cbor_length INTEGER NOT NULL,
  compressed_length INTEGER NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS fingerprints (
  preset_id INTEGER NOT NULL REFERENCES presets(id) ON DELETE CASCADE,
  midi_note INTEGER NOT NULL,
  embedding_f32 BLOB NOT NULL,
  handcrafted_f32 BLOB NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (preset_id,midi_note)
);
CREATE TABLE IF NOT EXISTS preset_sources (
  id INTEGER PRIMARY KEY,
  preset_id INTEGER NOT NULL REFERENCES presets(id) ON DELETE CASCADE,
  normalized_path TEXT NOT NULL,
  source_root TEXT NOT NULL,
  file_size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0,1)),
  last_seen_scan TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(normalized_path,content_hash)
);
CREATE TABLE IF NOT EXISTS prepared_presets (
  preset_id INTEGER PRIMARY KEY REFERENCES presets(id) ON DELETE CASCADE,
  render_revision TEXT NOT NULL,
  fingerprint_revision TEXT NOT NULL,
  clap_revision TEXT NOT NULL,
  handcrafted_revision TEXT NOT NULL,
  serum1_schema_revision TEXT NOT NULL,
  serum2_schema_revision TEXT NOT NULL,
  prepared_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS preparation_jobs (
  preset_id INTEGER PRIMARY KEY REFERENCES presets(id) ON DELETE CASCADE,
  expected_content_hash TEXT NOT NULL,
  target_revision TEXT NOT NULL,
  state TEXT NOT NULL CHECK (state IN (
    'pending','rendering','rendered','analyzing','committing',
    'prepared','cleanup_complete','failed','cancelled'
  )),
  temp_dir TEXT NOT NULL,
  cleanup_needed INTEGER NOT NULL DEFAULT 0 CHECK (cleanup_needed IN (0,1)),
  attempt_count INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS favorites (
  content_hash TEXT PRIMARY KEY,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS match_batches (
  id INTEGER PRIMARY KEY,
  folder_name TEXT NOT NULL,
  source_folder TEXT NOT NULL,
  export_folder TEXT NOT NULL,
  target_synth TEXT NOT NULL CHECK (target_synth IN ('serum1','serum2')),
  budget TEXT NOT NULL CHECK (budget IN ('quick','balanced','best')),
  total_files INTEGER NOT NULL DEFAULT 0,
  completed_files INTEGER NOT NULL DEFAULT 0,
  failed_files INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL CHECK (status IN ('running','cancelled','complete')),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS match_library (
  id INTEGER PRIMARY KEY,
  match_uid TEXT NOT NULL UNIQUE,
  source_name TEXT NOT NULL,
  source_audio_path TEXT NOT NULL,
  source_content_hash TEXT NOT NULL,
  result_json_path TEXT NOT NULL,
  target_synth TEXT NOT NULL CHECK (target_synth IN ('serum1','serum2')),
  budget TEXT NOT NULL CHECK (budget IN ('quick','balanced','best')),
  similarity_percent REAL NOT NULL DEFAULT 0,
  base_name TEXT NOT NULL,
  recommendation_synth TEXT NOT NULL CHECK (recommendation_synth IN ('serum1','serum2')),
  no_confident_match INTEGER NOT NULL DEFAULT 0 CHECK (no_confident_match IN (0,1)),
  batch_id INTEGER REFERENCES match_batches(id) ON DELETE SET NULL,
  exported_preset_path TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_presets_status ON presets(status);
CREATE INDEX IF NOT EXISTS idx_presets_synth ON presets(synth);
CREATE INDEX IF NOT EXISTS idx_serum2_full_settings_sha256
  ON serum2_full_settings(settings_sha256);
CREATE INDEX IF NOT EXISTS idx_renders_preset ON renders(preset_id);
CREATE INDEX IF NOT EXISTS idx_fingerprints_preset ON fingerprints(preset_id);
CREATE INDEX IF NOT EXISTS idx_preset_sources_preset_active
  ON preset_sources(preset_id,active);
CREATE INDEX IF NOT EXISTS idx_preset_sources_root_active
  ON preset_sources(source_root,active);
CREATE UNIQUE INDEX IF NOT EXISTS idx_preset_sources_one_active_path
  ON preset_sources(normalized_path) WHERE active=1;
CREATE INDEX IF NOT EXISTS idx_preparation_jobs_cleanup
  ON preparation_jobs(cleanup_needed,state);
CREATE INDEX IF NOT EXISTS idx_match_library_created ON match_library(created_at DESC,id DESC);
CREATE INDEX IF NOT EXISTS idx_match_library_batch ON match_library(batch_id);
CREATE INDEX IF NOT EXISTS idx_match_library_hash ON match_library(source_content_hash);
CREATE INDEX IF NOT EXISTS idx_match_batches_created ON match_batches(created_at DESC,id DESC);
"""



def _factory_only(alias: str = "") -> str:
    """SQL fragment that removes the user's own presets while consent is OFF.

    User presets are rows with ``is_factory=0``.  They stay in the database --
    turning consent off never deletes anything -- but every query that decides
    what to process must treat them as inactive, so a later render, analysis or
    upload cannot pick them up.  Evaluated per call, so withdrawing consent
    takes effect on the very next query.
    """

    return "" if user_presets_enabled() else f" AND {alias}is_factory=1"


class Database:
    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    #: Columns added to ``presets`` after schema 5, in the order they are added.
    _PRESET_COLUMNS_ADDED_LATER: tuple[tuple[str, str], ...] = (
        (
            "is_factory",
            "INTEGER NOT NULL DEFAULT 0 CHECK (is_factory IN (0,1))",
        ),
        # Schema 6: preserve discovered-but-unprocessable presets with the reason,
        # so installing a missing Serum later does not require rediscovering a
        # multi-thousand-file library from scratch.
        ("file_format", "TEXT"),
        ("provenance", "TEXT"),
        ("compatible_renderers", "TEXT"),
        ("pending_reason", "TEXT"),
        ("last_attempt_at", "TEXT"),
    )

    def migrate(self) -> None:
        """Bring any older library up to the current schema, atomically.

        ``executescript`` first (every statement is ``CREATE ... IF NOT EXISTS``,
        so it only fills in what is missing and never alters existing data).
        Then, only if a column is actually missing, the ``ALTER`` statements run
        inside ONE explicit transaction. Python's sqlite3 does not open a
        transaction for DDL on its own, so without ``BEGIN`` a failure between two
        ``ALTER``s would leave a half-migrated table; with it, SQLite rolls the
        whole step back and the next launch simply retries. Already-current
        libraries take no write lock and change nothing.
        """

        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)
            current_version = int(
                connection.execute(
                    "SELECT COALESCE(MAX(version),0) FROM schema_migrations"
                ).fetchone()[0]
            )
            existing = self._preset_columns(connection)
            missing = [
                (name, definition)
                for name, definition in self._PRESET_COLUMNS_ADDED_LATER
                if name not in existing
            ]
            needs_schema_7 = current_version < 7
            if missing or needs_schema_7:
                connection.execute("BEGIN IMMEDIATE")
                # Re-read under the lock: another PatchLab process may have
                # finished the same migration while this one waited.
                existing = self._preset_columns(connection)
                for name, definition in self._PRESET_COLUMNS_ADDED_LATER:
                    if name not in existing:
                        connection.execute(
                            f"ALTER TABLE presets ADD COLUMN {name} {definition}"
                        )
                if needs_schema_7:
                    self._migrate_schema_7_state(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS presets_pending_reason "
                "ON presets(pending_reason)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

    @staticmethod
    def _migrate_schema_7_state(connection: sqlite3.Connection) -> None:
        """Seed source snapshots and justified revisions for schema-6 rows."""

        import hashlib
        import os

        from core.prepared_state import record_prepared_revision

        rows = connection.execute(
            "SELECT id,path,content_hash FROM presets ORDER BY id"
        ).fetchall()
        by_path: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            path = Path(str(row["path"])).expanduser().resolve()
            normalized = os.path.normcase(str(path))
            by_path.setdefault(normalized, []).append(row)

        for normalized, path_rows in by_path.items():
            path = Path(normalized)
            try:
                stat = path.stat()
            except OSError:
                size, mtime_ns, current_hash, file_exists = 0, 0, None, False
            else:
                size, mtime_ns = stat.st_size, stat.st_mtime_ns
                file_exists = True
                current_hash = None
                if len(path_rows) > 1:
                    try:
                        digest = hashlib.sha1()
                        with path.open("rb") as handle:
                            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                                digest.update(chunk)
                    except OSError:
                        file_exists = False
                    else:
                        current_hash = digest.hexdigest()
            for row in path_rows:
                # Old databases can contain several content rows with the same
                # path after an in-place edit. Hash only those ambiguous paths;
                # activate the row that still matches the bytes on disk.
                active = int(
                    file_exists
                    and (
                        len(path_rows) == 1
                        or str(row["content_hash"]) == current_hash
                    )
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO preset_sources(
                      preset_id,normalized_path,source_root,file_size,mtime_ns,
                      content_hash,active,last_seen_scan
                    ) VALUES (?,?,?,?,?,?,?,NULL)
                    """,
                    (
                        int(row["id"]),
                        normalized,
                        os.path.normcase(str(path.parent)),
                        int(size),
                        int(mtime_ns),
                        str(row["content_hash"]),
                        active,
                    ),
                )
                # Fully learned legacy data receives the current revision without
                # rerendering. Partial data deliberately remains unprepared.
                record_prepared_revision(connection, int(row["id"]))

    @staticmethod
    def _preset_columns(connection: sqlite3.Connection) -> set[str]:
        return {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(presets)").fetchall()
        }

    def insert_preset(
        self, *, path: Path, name: str, synth: str, content_hash: str
    ) -> tuple[int, bool]:
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM presets WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            if existing:
                return int(existing["id"]), False
            cursor = connection.execute(
                "INSERT INTO presets(path,name,synth,content_hash) VALUES (?,?,?,?)",
                (str(path.resolve()), name, synth, content_hash),
            )
            return int(cursor.lastrowid), True

    def presets_with_status(self, statuses: Sequence[str]) -> list[PresetRecord]:
        placeholders = ",".join("?" for _ in statuses)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM presets WHERE status IN ({placeholders}){_factory_only()} ORDER BY id", tuple(statuses)
            ).fetchall()
        return [self._preset(row) for row in rows]

    def renderable_presets(self, synth: str | None = None) -> list[PresetRecord]:
        """Return presets with a completed parameter record, regardless of later render status."""

        where = "WHERE p.synth=?" if synth is not None else "WHERE 1=1"
        arguments: tuple[object, ...] = (synth,) if synth is not None else ()
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT p.* FROM presets p JOIN params pa ON pa.preset_id=p.id "
                f"{where}{_factory_only('p.')} GROUP BY p.id HAVING COUNT(pa.param_index)>0 ORDER BY p.id",
                arguments,
            ).fetchall()
        return [self._preset(row) for row in rows]

    def replace_params(
        self, preset_id: int, parameters: Sequence[ParameterValue], strategy: str
    ) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM params WHERE preset_id = ?", (preset_id,))
            connection.executemany(
                "INSERT INTO params(preset_id,param_index,param_name,norm_value,display_value) "
                "VALUES (?,?,?,?,?)",
                (
                    (preset_id, item.index, item.name, item.norm_value, item.display_value)
                    for item in parameters
                ),
            )
            connection.execute(
                "UPDATE presets SET load_strategy=?, status='params_dumped', error=NULL WHERE id=?",
                (strategy, preset_id),
            )

    def mark_failed(self, preset_id: int, status: str, error: str) -> None:
        if status not in {"failed_load", "failed_silent"}:
            raise ValueError(f"Invalid failure status {status!r}")
        with self.connect() as connection:
            connection.execute(
                "UPDATE presets SET status=?, error=? WHERE id=?", (status, error[:4000], preset_id)
            )

    def existing_render_notes(self, preset_ids: Sequence[int] | None = None) -> dict[int, set[int]]:
        with self.connect() as connection:
            if preset_ids is None:
                rows = connection.execute("SELECT preset_id,midi_note FROM renders").fetchall()
            elif not preset_ids:
                return {}
            else:
                placeholders = ",".join("?" for _ in preset_ids)
                rows = connection.execute(
                    f"SELECT preset_id,midi_note FROM renders WHERE preset_id IN ({placeholders})",
                    tuple(preset_ids),
                ).fetchall()
        result: dict[int, set[int]] = {}
        for row in rows:
            result.setdefault(int(row["preset_id"]), set()).add(int(row["midi_note"]))
        return result

    def upsert_renders(self, rows: Sequence[RenderRecord]) -> None:
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO renders(preset_id,midi_note,wav_path,peak_dbfs,rms_dbfs,duration_s)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(preset_id,midi_note) DO UPDATE SET
                  wav_path=excluded.wav_path,
                  peak_dbfs=excluded.peak_dbfs,
                  rms_dbfs=excluded.rms_dbfs,
                  duration_s=excluded.duration_s
                """,
                (
                    (
                        row.preset_id,
                        row.midi_note,
                        str(row.wav_path),
                        row.peak_dbfs,
                        row.rms_dbfs,
                        row.duration_s,
                    )
                    for row in rows
                ),
            )

    def finalize_render_status(self, preset_id: int, expected_notes: Sequence[int]) -> str:
        """Set rendered/failed_silent after inspecting all persisted rows for one preset."""

        with self.connect() as connection:
            placeholders = ",".join("?" for _ in expected_notes)
            rows = connection.execute(
                f"SELECT midi_note,rms_dbfs FROM renders WHERE preset_id=? "
                f"AND midi_note IN ({placeholders})",
                (preset_id, *expected_notes),
            ).fetchall()
            if len(rows) < len(expected_notes):
                return "partial"
            silent = [int(row["midi_note"]) for row in rows if float(row["rms_dbfs"]) <= -60.0]
            if silent:
                detail = "Silent rendered MIDI notes: " + ", ".join(map(str, silent))
                connection.execute(
                    "UPDATE presets SET status='failed_silent',error=? WHERE id=?",
                    (detail, preset_id),
                )
                return "failed_silent"
            connection.execute(
                "UPDATE presets SET status='rendered',error=NULL WHERE id=?", (preset_id,)
            )
            return "rendered"

    def param_vector(self, preset_id: int) -> list[float]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT norm_value FROM params WHERE preset_id=? ORDER BY param_index", (preset_id,)
            ).fetchall()
        return [float(row[0]) for row in rows]

    def replace_serum2_full_settings(
        self,
        preset_id: int,
        *,
        metadata_json: str,
        settings_json: str,
        settings_sha256: str,
        payload_version: int,
        cbor_length: int,
        compressed_length: int,
    ) -> None:
        """Store complete decoded Serum 2 state without changing mapped parameters."""

        with self.connect() as connection:
            preset = connection.execute(
                "SELECT synth FROM presets WHERE id=?", (preset_id,)
            ).fetchone()
            if preset is None:
                raise KeyError(f"Unknown preset id {preset_id}")
            if preset["synth"] != "serum2":
                raise ValueError(f"Preset id {preset_id} is not Serum 2")
            connection.execute(
                """
                INSERT INTO serum2_full_settings(
                  preset_id, metadata_json, settings_json, settings_sha256,
                  payload_version, cbor_length, compressed_length
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(preset_id) DO UPDATE SET
                  metadata_json=excluded.metadata_json,
                  settings_json=excluded.settings_json,
                  settings_sha256=excluded.settings_sha256,
                  payload_version=excluded.payload_version,
                  cbor_length=excluded.cbor_length,
                  compressed_length=excluded.compressed_length,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (
                    preset_id,
                    metadata_json,
                    settings_json,
                    settings_sha256,
                    payload_version,
                    cbor_length,
                    compressed_length,
                ),
            )

    def serum2_full_settings(self, preset_id: int) -> dict[str, object]:
        """Return the authoritative decoded metadata/settings graph for one preset."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM serum2_full_settings WHERE preset_id=?", (preset_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"No full Serum 2 settings for preset id {preset_id}")
        return {
            "preset_id": int(row["preset_id"]),
            "metadata": json.loads(row["metadata_json"]),
            "settings": json.loads(row["settings_json"]),
            "settings_sha256": str(row["settings_sha256"]),
            "payload_version": int(row["payload_version"]),
            "cbor_length": int(row["cbor_length"]),
            "compressed_length": int(row["compressed_length"]),
            "updated_at": str(row["updated_at"]),
        }

    def serum2_full_settings_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM serum2_full_settings").fetchone()
        return int(row[0])

    def status_counts(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM presets GROUP BY status"
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def upsert_fingerprint(
        self,
        preset_id: int,
        midi_note: int,
        embedding: bytes,
        handcrafted: bytes,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO fingerprints(
                  preset_id,midi_note,embedding_f32,handcrafted_f32
                ) VALUES (?,?,?,?)
                ON CONFLICT(preset_id,midi_note) DO UPDATE SET
                  embedding_f32=excluded.embedding_f32,
                  handcrafted_f32=excluded.handcrafted_f32,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (preset_id, midi_note, embedding, handcrafted),
            )

    def set_favorite(self, content_hash: str, favorited: bool) -> None:
        with self.connect() as connection:
            if favorited:
                connection.execute(
                    "INSERT OR IGNORE INTO favorites(content_hash) VALUES (?)",
                    (content_hash,),
                )
            else:
                connection.execute(
                    "DELETE FROM favorites WHERE content_hash=?", (content_hash,)
                )

    def favorite_hashes(self) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute("SELECT content_hash FROM favorites").fetchall()
        return {str(row["content_hash"]) for row in rows}

    def insert_match_library(
        self,
        *,
        match_uid: str,
        source_name: str,
        source_audio_path: Path,
        source_content_hash: str,
        result_json_path: Path,
        target_synth: str,
        budget: str,
        similarity_percent: float,
        base_name: str,
        recommendation_synth: str,
        no_confident_match: bool,
        batch_id: int | None = None,
        exported_preset_path: Path | None = None,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO match_library(
                  match_uid,source_name,source_audio_path,source_content_hash,
                  result_json_path,target_synth,budget,similarity_percent,
                  base_name,recommendation_synth,no_confident_match,batch_id,
                  exported_preset_path
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    match_uid,
                    source_name,
                    str(source_audio_path),
                    source_content_hash,
                    str(result_json_path),
                    target_synth,
                    budget,
                    float(similarity_percent),
                    base_name,
                    recommendation_synth,
                    1 if no_confident_match else 0,
                    batch_id,
                    str(exported_preset_path) if exported_preset_path else None,
                ),
            )
            return int(cursor.lastrowid)

    def list_match_library(self) -> list[MatchLibraryRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM match_library ORDER BY created_at DESC,id DESC"
            ).fetchall()
        return [self._match_library(row) for row in rows]

    def get_match_library(self, match_uid: str) -> MatchLibraryRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM match_library WHERE match_uid=?", (match_uid,)
            ).fetchone()
        return self._match_library(row) if row is not None else None

    def delete_match_library(self, match_uid: str) -> MatchLibraryRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM match_library WHERE match_uid=?", (match_uid,)
            ).fetchone()
            if row is None:
                return None
            record = self._match_library(row)
            connection.execute(
                "DELETE FROM match_library WHERE match_uid=?", (match_uid,)
            )
            return record

    def set_match_exported_path(
        self, match_uid: str, exported_preset_path: Path
    ) -> bool:
        """Record where a match's preset was saved; True only if a row now says so.

        An UPDATE that matches no row succeeds silently in SQLite, so the value
        is read back rather than trusted: the Library must never point at a
        path the caller did not actually record.
        """

        with self.connect() as connection:
            connection.execute(
                "UPDATE match_library SET exported_preset_path=? WHERE match_uid=?",
                (str(exported_preset_path), match_uid),
            )
            row = connection.execute(
                "SELECT exported_preset_path FROM match_library WHERE match_uid=?",
                (match_uid,),
            ).fetchone()
        return row is not None and str(row[0]) == str(exported_preset_path)

    def create_match_batch(
        self,
        *,
        folder_name: str,
        source_folder: Path,
        export_folder: Path,
        target_synth: str,
        budget: str,
        total_files: int,
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO match_batches(
                  folder_name,source_folder,export_folder,target_synth,budget,
                  total_files,status
                ) VALUES (?,?,?,?,?,?,'running')
                """,
                (
                    folder_name,
                    str(source_folder.resolve()),
                    str(export_folder.resolve()),
                    target_synth,
                    budget,
                    int(total_files),
                ),
            )
            return int(cursor.lastrowid)

    def find_match_batch(
        self,
        *,
        source_folder: Path,
        export_folder: Path,
        target_synth: str,
        budget: str,
    ) -> MatchBatchRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM match_batches
                WHERE source_folder=? AND export_folder=?
                  AND target_synth=? AND budget=?
                ORDER BY id DESC LIMIT 1
                """,
                (
                    str(source_folder.resolve()),
                    str(export_folder.resolve()),
                    target_synth,
                    budget,
                ),
            ).fetchone()
        return self._match_batch(row) if row is not None else None

    def update_match_batch(
        self,
        batch_id: int,
        *,
        completed_files: int,
        failed_files: int,
        status: str,
        total_files: int | None = None,
    ) -> None:
        if status not in {"running", "cancelled", "complete"}:
            raise ValueError(f"Invalid match batch status {status!r}")
        with self.connect() as connection:
            if total_files is None:
                connection.execute(
                    """
                    UPDATE match_batches
                    SET completed_files=?,failed_files=?,status=?
                    WHERE id=?
                    """,
                    (completed_files, failed_files, status, batch_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE match_batches
                    SET total_files=?,completed_files=?,failed_files=?,status=?
                    WHERE id=?
                    """,
                    (total_files, completed_files, failed_files, status, batch_id),
                )

    def get_match_batch(self, batch_id: int) -> MatchBatchRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM match_batches WHERE id=?", (batch_id,)
            ).fetchone()
        return self._match_batch(row) if row is not None else None

    def list_match_batches(self) -> list[MatchBatchRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM match_batches ORDER BY created_at DESC,id DESC"
            ).fetchall()
        return [self._match_batch(row) for row in rows]

    def batch_completed_hashes(self, batch_id: int) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT source_content_hash FROM match_library WHERE batch_id=? "
                "AND (exported_preset_path IS NOT NULL OR no_confident_match=1)",
                (batch_id,),
            ).fetchall()
        return {str(row["source_content_hash"]) for row in rows}

    def record_identity(
        self,
        preset_id: int,
        *,
        file_format: str | None,
        provenance: str | None,
        compatible_renderers: Sequence[str],
    ) -> None:
        """Persist the orthogonal identity facts for one preset."""

        with self.connect() as connection:
            connection.execute(
                "UPDATE presets SET file_format=?, provenance=?, "
                "compatible_renderers=? WHERE id=?",
                (
                    file_format,
                    provenance,
                    ",".join(str(item) for item in compatible_renderers) or None,
                    preset_id,
                ),
            )

    def set_pending_reason(
        self, preset_id: int, reason: str | None, *, error: str | None = None
    ) -> None:
        """Mark a discovered preset pending, or clear it once processable.

        A pending preset is *kept*, not discarded: this is what lets a later
        Serum install pick up thousands of already-catalogued presets without
        rediscovering them.
        """

        stamp = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            if error is None:
                connection.execute(
                    "UPDATE presets SET pending_reason=?, last_attempt_at=? WHERE id=?",
                    (reason, stamp, preset_id),
                )
            else:
                connection.execute(
                    "UPDATE presets SET pending_reason=?, last_attempt_at=?, error=? "
                    "WHERE id=?",
                    (reason, stamp, error[:4000], preset_id),
                )

    def set_pending_reasons(self, updates: Mapping[int, str | None]) -> None:
        """Bulk pending update; one transaction for a whole library."""

        if not updates:
            return
        stamp = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            connection.executemany(
                "UPDATE presets SET pending_reason=?, last_attempt_at=? WHERE id=?",
                [(reason, stamp, preset_id) for preset_id, reason in updates.items()],
            )

    def pending_presets(
        self, reasons: Sequence[str] | None = None
    ) -> list[PresetRecord]:
        """Discovered presets that are not processable yet."""

        with self.connect() as connection:
            if reasons:
                placeholders = ",".join("?" for _ in reasons)
                rows = connection.execute(
                    f"SELECT * FROM presets WHERE pending_reason IN ({placeholders})"
                    f"{_factory_only()} ORDER BY id",
                    tuple(reasons),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM presets WHERE pending_reason IS NOT NULL"
                    f"{_factory_only()} ORDER BY id"
                ).fetchall()
        return [self._preset(row) for row in rows]

    def pending_counts(self) -> dict[str, int]:
        """Pending totals per stable reason code, for diagnostics and the UI."""

        with self.connect() as connection:
            rows = connection.execute(
                "SELECT pending_reason, COUNT(*) FROM presets "
                "WHERE pending_reason IS NOT NULL GROUP BY pending_reason"
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def presets_needing_generation(
        self, generation: str, reasons: Sequence[str] | None = None
    ) -> list[PresetRecord]:
        """Pending presets whose compatible renderer set includes ``generation``.

        This is the query that makes a later Serum install cheap: PatchLab
        already knows these files exist, so it never needs to re-walk and
        re-hash the whole library to find them again.

        ``reasons`` narrows to specific pending reasons -- the GUI uses it to
        offer only presets that were genuinely *waiting on this synth*, not ones
        that failed to render for an unrelated reason.
        """

        clause = ""
        arguments: list[object] = [
            generation,
            f"{generation},%",
            f"%,{generation}",
            f"%,{generation},%",
        ]
        if reasons:
            clause = " AND pending_reason IN (" + ",".join("?" for _ in reasons) + ")"
            arguments.extend(reasons)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM presets WHERE pending_reason IS NOT NULL AND ("
                "  compatible_renderers = ?"
                "  OR compatible_renderers LIKE ? OR compatible_renderers LIKE ?"
                "  OR compatible_renderers LIKE ?"
                f"){clause}{_factory_only()} ORDER BY id",
                tuple(arguments),
            ).fetchall()
        return [self._preset(row) for row in rows]

    def library_coverage(self) -> dict[str, object]:
        """Discovered / processed / pending counts, split by generation.

        Exists so a support bundle can answer "why are only 3,842 of my 5,000
        presets appearing?" without anyone having to guess.
        """

        with self.connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM presets").fetchone()[0])
            by_status = {
                f"{row[0]}": int(row[1])
                for row in connection.execute(
                    "SELECT status, COUNT(*) FROM presets GROUP BY status"
                ).fetchall()
            }
            by_generation = {
                f"{row[0]}": int(row[1])
                for row in connection.execute(
                    "SELECT synth, COUNT(*) FROM presets GROUP BY synth"
                ).fetchall()
            }
            by_format = {
                f"{row[0] or 'unknown'}": int(row[1])
                for row in connection.execute(
                    "SELECT file_format, COUNT(*) FROM presets GROUP BY file_format"
                ).fetchall()
            }
            pending = {
                f"{row[0]}": int(row[1])
                for row in connection.execute(
                    "SELECT pending_reason, COUNT(*) FROM presets "
                    "WHERE pending_reason IS NOT NULL GROUP BY pending_reason"
                ).fetchall()
            }
            learned = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT preset_id) FROM fingerprints WHERE midi_note=0"
                ).fetchone()[0]
            )
            with_params = int(
                connection.execute(
                    "SELECT COUNT(DISTINCT preset_id) FROM params"
                ).fetchone()[0]
            )
        return {
            "discovered": total,
            "by_status": by_status,
            "by_generation": by_generation,
            "by_file_format": by_format,
            "pending_by_reason": pending,
            "pending": sum(pending.values()),
            "processed_params": with_params,
            "learned": learned,
            "failed": by_status.get("failed_load", 0) + by_status.get("failed_silent", 0),
        }

    def set_factory_status(self, preset_id: int, is_factory: bool) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE presets SET is_factory=? WHERE id=?",
                (1 if is_factory else 0, preset_id),
            )

    @staticmethod
    def _preset(row: sqlite3.Row) -> PresetRecord:
        return PresetRecord(
            id=int(row["id"]),
            path=Path(row["path"]),
            name=str(row["name"]),
            synth=str(row["synth"]),
            content_hash=str(row["content_hash"]),
            load_strategy=row["load_strategy"],
            status=str(row["status"]),
            error=row["error"],
            is_factory=bool(row["is_factory"]),
            file_format=Database._optional(row, "file_format"),
            provenance=Database._optional(row, "provenance"),
            compatible_renderers=Database._optional(row, "compatible_renderers"),
            pending_reason=Database._optional(row, "pending_reason"),
            last_attempt_at=Database._optional(row, "last_attempt_at"),
        )

    @staticmethod
    def _optional(row: sqlite3.Row, column: str) -> str | None:
        """Read a column that may not exist in an older row projection."""

        try:
            value = row[column]
        except (IndexError, KeyError):
            return None
        return None if value is None else str(value)

    @staticmethod
    def _match_library(row: sqlite3.Row) -> MatchLibraryRecord:
        return MatchLibraryRecord(
            id=int(row["id"]),
            match_uid=str(row["match_uid"]),
            source_name=str(row["source_name"]),
            source_audio_path=Path(str(row["source_audio_path"])),
            source_content_hash=str(row["source_content_hash"]),
            result_json_path=Path(str(row["result_json_path"])),
            target_synth=str(row["target_synth"]),
            budget=str(row["budget"]),
            similarity_percent=float(row["similarity_percent"]),
            base_name=str(row["base_name"]),
            recommendation_synth=str(row["recommendation_synth"]),
            no_confident_match=bool(row["no_confident_match"]),
            batch_id=int(row["batch_id"]) if row["batch_id"] is not None else None,
            exported_preset_path=(
                Path(str(row["exported_preset_path"]))
                if row["exported_preset_path"]
                else None
            ),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _match_batch(row: sqlite3.Row) -> MatchBatchRecord:
        return MatchBatchRecord(
            id=int(row["id"]),
            folder_name=str(row["folder_name"]),
            source_folder=str(row["source_folder"]),
            export_folder=str(row["export_folder"]),
            target_synth=str(row["target_synth"]),
            budget=str(row["budget"]),
            total_files=int(row["total_files"]),
            completed_files=int(row["completed_files"]),
            failed_files=int(row["failed_files"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
        )
