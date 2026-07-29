"""Compact, preset-file-free factory fingerprint bundle."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zstandard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTORY_BUNDLE = PROJECT_ROOT / "data" / "dist" / "factory_bundle.sqlite"
BUNDLE_SCHEMA_VERSION = 1
_COMPRESSOR = zstandard.ZstdCompressor(level=9)
_DECOMPRESSOR = zstandard.ZstdDecompressor()


BUNDLE_SCHEMA = """
CREATE TABLE bundle_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE schemas (
  synth TEXT PRIMARY KEY CHECK (synth IN ('serum1','serum2')),
  schema_json TEXT NOT NULL
);
CREATE TABLE presets (
  id INTEGER PRIMARY KEY,
  content_hash TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  synth TEXT NOT NULL CHECK (synth IN ('serum1','serum2')),
  relative_path TEXT NOT NULL,
  extension TEXT NOT NULL,
  searchable INTEGER NOT NULL CHECK (searchable IN (0,1)),
  parameter_vector BLOB NOT NULL,
  parameter_mask BLOB NOT NULL,
  settings_zstd BLOB NOT NULL,
  metadata_zstd BLOB,
  payload_version INTEGER
);
CREATE TABLE preset_embeddings (
  preset_id INTEGER PRIMARY KEY REFERENCES presets(id) ON DELETE CASCADE,
  embedding_f16 BLOB NOT NULL
);
CREATE TABLE note_embeddings (
  preset_id INTEGER NOT NULL REFERENCES presets(id) ON DELETE CASCADE,
  midi_note INTEGER NOT NULL,
  embedding_f16 BLOB NOT NULL,
  PRIMARY KEY (preset_id,midi_note)
);
CREATE INDEX idx_factory_hash ON presets(content_hash);
CREATE INDEX idx_factory_synth ON presets(synth);
"""


@dataclass(frozen=True, slots=True)
class FactoryPreset:
    id: int
    content_hash: str
    name: str
    synth: str
    relative_path: str
    extension: str
    searchable: bool


def compress_json(value: Any) -> bytes:
    raw = json.dumps(
        value, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    return _COMPRESSOR.compress(raw)


def decompress_json(value: bytes) -> Any:
    return json.loads(_DECOMPRESSOR.decompress(value).decode("utf-8"))


def compress_array(value: np.ndarray) -> bytes:
    return _COMPRESSOR.compress(
        np.ascontiguousarray(value, dtype=np.float32).tobytes()
    )


def decompress_array(value: bytes, length: int) -> np.ndarray:
    raw = _DECOMPRESSOR.decompress(value)
    result = np.frombuffer(raw, dtype=np.float32)
    if len(result) != length:
        raise ValueError(f"Expected {length} floats, decoded {len(result)}")
    return result.copy()


def compress_mask(value: np.ndarray) -> bytes:
    packed = np.packbits(np.asarray(value, dtype=np.bool_), bitorder="little")
    return _COMPRESSOR.compress(packed.tobytes())


def decompress_mask(value: bytes, length: int) -> np.ndarray:
    packed = np.frombuffer(_DECOMPRESSOR.decompress(value), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little")[:length].astype(np.bool_)


class FactoryBundle:
    def __init__(self, path: Path = DEFAULT_FACTORY_BUNDLE) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def metadata(self) -> dict[str, str]:
        with self.connect() as connection:
            rows = connection.execute("SELECT key,value FROM bundle_metadata").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def known_hashes(self) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute("SELECT content_hash FROM presets").fetchall()
        return {str(row[0]) for row in rows}

    def presets(self, *, searchable_only: bool = False) -> list[FactoryPreset]:
        where = "WHERE searchable=1" if searchable_only else ""
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT id,content_hash,name,synth,relative_path,extension,searchable "
                f"FROM presets {where} ORDER BY id"
            ).fetchall()
        return [
            FactoryPreset(
                id=int(row["id"]),
                content_hash=str(row["content_hash"]),
                name=str(row["name"]),
                synth=str(row["synth"]),
                relative_path=str(row["relative_path"]),
                extension=str(row["extension"]),
                searchable=bool(row["searchable"]),
            )
            for row in rows
        ]

    def search_index(self) -> tuple[np.ndarray, list[FactoryPreset]]:
        presets = self.presets(searchable_only=True)
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT embedding_f16 FROM preset_embeddings ORDER BY preset_id"
            ).fetchall()
        matrix = np.stack(
            [
                np.frombuffer(row["embedding_f16"], dtype=np.float16).astype(np.float32)
                for row in rows
            ]
        )
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        if len(matrix) != len(presets):
            raise RuntimeError("Factory bundle preset/index row count mismatch")
        return matrix, presets

    def note_embedding(self, preset_id: int, midi_note: int) -> np.ndarray:
        """Return one normalized macOS reference embedding for parity checks."""

        with self.connect() as connection:
            row = connection.execute(
                "SELECT embedding_f16 FROM note_embeddings "
                "WHERE preset_id=? AND midi_note=?",
                (preset_id, midi_note),
            ).fetchone()
        if row is None:
            raise KeyError((preset_id, midi_note))
        value = np.frombuffer(row["embedding_f16"], dtype=np.float16).astype(
            np.float32
        )
        return value / max(float(np.linalg.norm(value)), 1e-12)

    def schema(self, synth: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT schema_json FROM schemas WHERE synth=?", (synth,)
            ).fetchone()
        if row is None:
            raise KeyError(synth)
        return json.loads(str(row[0]))

    def parameters(self, preset_id: int) -> tuple[np.ndarray, np.ndarray]:
        schema = self.schema(
            self.preset_by_id(preset_id).synth
        )
        length = int(schema["vector_length"])
        with self.connect() as connection:
            row = connection.execute(
                "SELECT parameter_vector,parameter_mask FROM presets WHERE id=?",
                (preset_id,),
            ).fetchone()
        if row is None:
            raise KeyError(preset_id)
        return (
            decompress_array(row["parameter_vector"], length),
            decompress_mask(row["parameter_mask"], length),
        )

    def preset_by_id(self, preset_id: int) -> FactoryPreset:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id,content_hash,name,synth,relative_path,extension,searchable "
                "FROM presets WHERE id=?",
                (preset_id,),
            ).fetchone()
        if row is None:
            raise KeyError(preset_id)
        return FactoryPreset(
            id=int(row["id"]),
            content_hash=str(row["content_hash"]),
            name=str(row["name"]),
            synth=str(row["synth"]),
            relative_path=str(row["relative_path"]),
            extension=str(row["extension"]),
            searchable=bool(row["searchable"]),
        )

    def settings(self, preset_id: int) -> tuple[Any, Any | None, int | None]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT settings_zstd,metadata_zstd,payload_version FROM presets WHERE id=?",
                (preset_id,),
            ).fetchone()
        if row is None:
            raise KeyError(preset_id)
        return (
            decompress_json(row["settings_zstd"]),
            decompress_json(row["metadata_zstd"]) if row["metadata_zstd"] else None,
            int(row["payload_version"]) if row["payload_version"] is not None else None,
        )
