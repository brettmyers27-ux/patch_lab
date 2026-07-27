"""SQLite schema, migrations, and typed library accessors."""

from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from core.plugin_host import ParameterValue


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "library.db"
SCHEMA_VERSION = 5


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
CREATE INDEX IF NOT EXISTS idx_match_library_created ON match_library(created_at DESC,id DESC);
CREATE INDEX IF NOT EXISTS idx_match_library_batch ON match_library(batch_id);
CREATE INDEX IF NOT EXISTS idx_match_library_hash ON match_library(source_content_hash);
CREATE INDEX IF NOT EXISTS idx_match_batches_created ON match_batches(created_at DESC,id DESC);
"""


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

    def migrate(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(presets)").fetchall()
            }
            if "is_factory" not in columns:
                connection.execute(
                    "ALTER TABLE presets ADD COLUMN is_factory INTEGER NOT NULL DEFAULT 0 "
                    "CHECK (is_factory IN (0,1))"
                )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (SCHEMA_VERSION,)
            )

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
                f"SELECT * FROM presets WHERE status IN ({placeholders}) ORDER BY id", tuple(statuses)
            ).fetchall()
        return [self._preset(row) for row in rows]

    def renderable_presets(self, synth: str | None = None) -> list[PresetRecord]:
        """Return presets with a completed parameter record, regardless of later render status."""

        where = "WHERE p.synth=?" if synth is not None else ""
        arguments: tuple[object, ...] = (synth,) if synth is not None else ()
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT p.* FROM presets p JOIN params pa ON pa.preset_id=p.id "
                f"{where} GROUP BY p.id HAVING COUNT(pa.param_index)>0 ORDER BY p.id",
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
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE match_library SET exported_preset_path=? WHERE match_uid=?",
                (str(exported_preset_path), match_uid),
            )

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
        )

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
