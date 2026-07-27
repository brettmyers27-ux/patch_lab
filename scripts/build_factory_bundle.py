#!/usr/bin/env python3
"""Build the shippable, factory-only fingerprint bundle."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import FEATURE_DIR, _serum1_targets, _serum2_targets
from core.db import DEFAULT_DB_PATH, Database
from core.factory_bundle import (
    BUNDLE_SCHEMA,
    BUNDLE_SCHEMA_VERSION,
    DEFAULT_FACTORY_BUNDLE,
    FactoryBundle,
    compress_array,
    compress_json,
    compress_mask,
)
from core.platform_env import ENV


REPORT = PROJECT_ROOT / "data" / "models" / "factory_bundle_report.json"


def _relative(synth: str, path: Path) -> str:
    for root in ENV.factory_roots_for(synth):
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    raise ValueError(f"{path} is not under a {synth} factory root")


def main() -> int:
    output = DEFAULT_FACTORY_BUNDLE
    output.parent.mkdir(parents=True, exist_ok=True)
    database = Database(DEFAULT_DB_PATH)
    stores = {1: _serum1_targets(DEFAULT_DB_PATH), 2: _serum2_targets()}
    note_manifest = np.load(FEATURE_DIR / "similarity_manifest.npz")
    note_index = np.load(FEATURE_DIR / "note_index.npy", mmap_mode="r")
    preset_index = np.load(FEATURE_DIR / "preset_index.npy", mmap_mode="r")
    note_rows: dict[int, list[int]] = {}
    for index, preset_id in enumerate(note_manifest["note_preset_ids"]):
        note_rows.setdefault(int(preset_id), []).append(index)
    preset_rows = {
        int(preset_id): index
        for index, preset_id in enumerate(note_manifest["preset_ids"])
    }
    with database.connect() as connection:
        all_source_rows = connection.execute(
            "SELECT id,path,name,synth,content_hash FROM presets "
            "WHERE is_factory=1 ORDER BY synth,id"
        ).fetchall()
        s1_parameter_rows = {
            int(preset_id): []
            for preset_id, in connection.execute(
                "SELECT id FROM presets WHERE synth='serum1' AND is_factory=1"
            )
        }
        for row in connection.execute(
            "SELECT pa.preset_id,pa.param_index,pa.param_name,pa.norm_value,pa.display_value "
            "FROM params pa JOIN presets p ON p.id=pa.preset_id "
            "WHERE p.synth='serum1' AND p.is_factory=1 "
            "ORDER BY pa.preset_id,pa.param_index"
        ):
            s1_parameter_rows[int(row["preset_id"])].append(
                {
                    "index": int(row["param_index"]),
                    "name": str(row["param_name"]),
                    "normalized": float(row["norm_value"]),
                    "display": str(row["display_value"]),
                }
            )
    source_rows = [
        row
        for row in all_source_rows
        if int(row["id"])
        in stores[1 if str(row["synth"]) == "serum1" else 2].preset_row
    ]
    excluded_rows = [
        {
            "preset_id": int(row["id"]),
            "name": str(row["name"]),
            "synth": str(row["synth"]),
            "reason": "no complete parameter target/fingerprint",
        }
        for row in all_source_rows
        if int(row["id"])
        not in stores[1 if str(row["synth"]) == "serum1" else 2].preset_row
    ]

    fd, temporary_name = tempfile.mkstemp(
        prefix=".factory_bundle.", suffix=".sqlite", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA page_size=4096")
        connection.executescript(BUNDLE_SCHEMA)
        schemas = {
            "serum1": {
                "vector_length": stores[1].dimension,
                "mapping": stores[1].mapping,
            },
            "serum2": json.loads(
                (PROJECT_ROOT / "data" / "models" / "serum2_target_schema.json").read_text(
                    encoding="utf-8"
                )
            ),
        }
        connection.executemany(
            "INSERT INTO schemas(synth,schema_json) VALUES (?,?)",
            [
                (synth, json.dumps(schema, separators=(",", ":"), sort_keys=True))
                for synth, schema in schemas.items()
            ],
        )
        searchable = 0
        note_count = 0
        for bundle_id, row in enumerate(source_rows, start=1):
            original_id = int(row["id"])
            synth = str(row["synth"])
            code = 1 if synth == "serum1" else 2
            store = stores[code]
            target_row = store.preset_row[original_id]
            vector = np.asarray(store.vectors[target_row], dtype=np.float32)
            mask = np.asarray(store.masks[target_row], dtype=np.bool_)
            if synth == "serum1":
                settings = {"parameters": s1_parameter_rows[original_id]}
                metadata = None
                payload_version = None
            else:
                full = database.serum2_full_settings(original_id)
                settings = full["settings"]
                metadata = full["metadata"]
                payload_version = int(full["payload_version"])
            has_embedding = original_id in preset_rows and original_id in note_rows
            connection.execute(
                "INSERT INTO presets("
                "id,content_hash,name,synth,relative_path,extension,searchable,"
                "parameter_vector,parameter_mask,settings_zstd,metadata_zstd,payload_version"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    bundle_id,
                    str(row["content_hash"]),
                    str(row["name"]),
                    synth,
                    _relative(synth, Path(str(row["path"]))),
                    Path(str(row["path"])).suffix,
                    int(has_embedding),
                    compress_array(vector),
                    compress_mask(mask),
                    compress_json(settings),
                    compress_json(metadata) if metadata is not None else None,
                    payload_version,
                ),
            )
            if has_embedding:
                searchable += 1
                preset_embedding = np.asarray(
                    preset_index[preset_rows[original_id]], dtype=np.float16
                )
                connection.execute(
                    "INSERT INTO preset_embeddings(preset_id,embedding_f16) VALUES (?,?)",
                    (bundle_id, preset_embedding.tobytes()),
                )
                for note_row in note_rows[original_id]:
                    note = int(note_manifest["note_midi_notes"][note_row])
                    embedding = np.asarray(note_index[note_row], dtype=np.float16)
                    connection.execute(
                        "INSERT INTO note_embeddings(preset_id,midi_note,embedding_f16) "
                        "VALUES (?,?,?)",
                        (bundle_id, note, embedding.tobytes()),
                    )
                    note_count += 1
        meta = {
            "schema_version": str(BUNDLE_SCHEMA_VERSION),
            "preset_count": str(len(source_rows)),
            "searchable_preset_count": str(searchable),
            "note_embedding_count": str(note_count),
            "embedding_dimensions": "512",
            "embedding_storage": "float16 normalized CLAP",
            "contains_audio": "false",
            "contains_preset_files": "false",
            "source_scope": "factory-only",
        }
        connection.executemany(
            "INSERT INTO bundle_metadata(key,value) VALUES (?,?)", meta.items()
        )
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        temporary.replace(output)
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise

    bundle = FactoryBundle(output)
    matrix, searchable_presets = bundle.search_index()
    factory_hashes = bundle.known_hashes()
    with database.connect() as source:
        nonfactory_hashes = {
            str(row[0])
            for row in source.execute(
                "SELECT content_hash FROM presets WHERE is_factory=0"
            )
        }
    metadata = bundle.metadata()
    size = output.stat().st_size
    report = {
        "path": str(output),
        "file_size_bytes": size,
        "file_size_mib": size / (1024**2),
        "preset_count": int(metadata["preset_count"]),
        "classified_factory_count": len(all_source_rows),
        "excluded_factory_presets": excluded_rows,
        "searchable_preset_count": int(metadata["searchable_preset_count"]),
        "unsearchable_preset_count": int(metadata["preset_count"])
        - int(metadata["searchable_preset_count"]),
        "note_embedding_count": int(metadata["note_embedding_count"]),
        "embedding_dimensions": int(metadata["embedding_dimensions"]),
        "factory_hash_count": len(factory_hashes),
        "nonfactory_hash_leaks": len(factory_hashes & nonfactory_hashes),
        "search_index_shape": list(matrix.shape),
        "unexpectedly_large": size > 100 * 1024**2,
        "contains_audio": False,
        "contains_preset_files": False,
        "gate_pass": (
            len(factory_hashes) == len(source_rows)
            and not (factory_hashes & nonfactory_hashes)
            and len(searchable_presets) == searchable
            and size <= 100 * 1024**2
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("FACTORY_BUNDLE_REPORT=" + json.dumps(report, sort_keys=True))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
