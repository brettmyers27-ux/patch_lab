"""Native Serum 2 ``.SerumPreset`` writing with structural verification."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sqlite3
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cbor2
import numpy as np
import zstandard

from core.db import DEFAULT_DB_PATH, Database
from core.branding import (
    APP_NAME,
    GENERATED_PRESET_DESCRIPTION,
    GENERATED_PRESET_TAGS,
    generated_preset_name,
)
from core.serum2_preset import MAGIC, Serum2Preset, parse_serum2_preset
from core.serum2_targets import ASSET_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "data" / "models" / "serum2_target_schema.json"
DEFAULT_TARGET_PATH = PROJECT_ROOT / "data" / "features" / "serum2_targets.npz"
MEANINGFUL_DELTA = 1e-4


@dataclass(frozen=True, slots=True)
class Serum2WriteResult:
    path: Path
    mode: str
    base_preset_id: int
    applied_fields: int
    skipped_fields: tuple[str, ...]
    graph_sha256: str
    asset_references: tuple[str, ...]


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field_value(
    field: Mapping[str, Any], vector: np.ndarray
) -> Any:
    index = int(field["index"])
    if field["encoding"] == "one_hot":
        width = int(field["width"])
        position = int(np.argmax(vector[index : index + width]))
        return copy.deepcopy(field["categories"][position])
    minimum = float(field["minimum"])
    maximum = float(field["maximum"])
    normalized = float(np.clip(vector[index], 0.0, 1.0))
    return minimum + normalized * (maximum - minimum)


def _plain_params(component: dict[str, Any]) -> dict[str, Any]:
    params = component.get("plainParams")
    if not isinstance(params, dict):
        params = {}
        component["plainParams"] = params
    return params


def _set_existing_dict_path(graph: dict[str, Any], parts: list[str], value: Any) -> bool:
    cursor: Any = graph
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        child = cursor[part]
        if child == "default" and part == "plainParams":
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            return False
        cursor = child
    if not isinstance(cursor, dict):
        return False
    leaf = parts[-1]
    # Schema fields came from a real corpus graph. Do not invent unsupported
    # topology when the selected base preset does not contain the field.
    if leaf not in cursor:
        return False
    cursor[leaf] = copy.deepcopy(value)
    return True


def _apply_field(graph: dict[str, Any], name: str, value: Any) -> bool:
    parts = name.split(".")
    root = parts[0]

    if root.startswith("ModSlot") and root[7:].isdigit():
        slot = graph.get(root)
        if not isinstance(slot, dict):
            return False
        suffix = ".".join(parts[1:])
        if suffix == "source":
            if "source" not in slot:
                return False
            slot["source"] = copy.deepcopy(value)
            return True
        if suffix == "destination":
            if not isinstance(value, dict):
                return False
            keys = {
                "destModuleTypeString": "module_type",
                "destModuleID": "module_id",
                "destModuleParamID": "param_id",
                "destModuleParamName": "param_name",
            }
            if not any(key in slot for key in keys):
                return False
            for target, source in keys.items():
                if target in slot:
                    slot[target] = copy.deepcopy(value.get(source))
            return True
        if len(parts) == 3 and parts[1] == "plainParams":
            params = _plain_params(slot)
            if parts[2] not in params:
                return False
            params[parts[2]] = copy.deepcopy(value)
            return True
        return False

    if root.startswith("Macro") and len(parts) == 3 and parts[1] == "plainParams":
        macro = graph.get(root)
        if not isinstance(macro, dict):
            return False
        params = _plain_params(macro)
        if parts[2] not in params:
            return False
        params[parts[2]] = copy.deepcopy(value)
        return True

    if root.startswith("FXRack") and len(parts) >= 3 and parts[1].startswith("slot"):
        rack = graph.get(root)
        if not isinstance(rack, dict):
            return False
        items = rack.get("FX")
        try:
            slot_index = int(parts[1][4:])
        except ValueError:
            return False
        if not isinstance(items, list) or slot_index >= len(items):
            return False
        item = items[slot_index]
        if not isinstance(item, dict):
            return False
        if parts[2] == "module":
            # The optimizer freezes topology. Keeping the base module object
            # also retains private per-module state not represented in schema.
            return True
        module_name = parts[2]
        module = item.get(module_name)
        if not isinstance(module, dict):
            return False
        if len(parts) == 5 and parts[3] == "plainParams":
            params = _plain_params(module)
            if parts[4] not in params:
                return False
            params[parts[4]] = copy.deepcopy(value)
            return True
        if len(parts) == 4 and parts[3] in ASSET_KEYS:
            # Assets are deliberately inherited from the known-good base.
            return True
        return False

    if any(name.endswith(f".{key}") for key in ASSET_KEYS):
        # Factory/custom asset references must remain exactly as the real seed.
        return True
    return _set_existing_dict_path(graph, parts, value)


def overlay_vector(
    base_graph: Mapping[str, Any],
    schema: Mapping[str, Any],
    vector: np.ndarray,
    mask: np.ndarray,
    base_vector: np.ndarray | None = None,
    threshold: float = MEANINGFUL_DELTA,
) -> tuple[dict[str, Any], int, tuple[str, ...]]:
    """Overlay schema-covered values while retaining all private base state."""

    result = copy.deepcopy(dict(base_graph))
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if vector.shape != mask.shape or len(vector) != int(schema["vector_length"]):
        raise ValueError("Serum 2 candidate vector/mask does not match the schema")
    if base_vector is not None:
        base_vector = np.asarray(base_vector, dtype=np.float32).reshape(-1)
        if base_vector.shape != vector.shape:
            raise ValueError("Serum 2 base vector does not match the candidate")
    applied = 0
    skipped: list[str] = []
    for field in schema["fields"]:
        index = int(field["index"])
        if not bool(mask[index]):
            continue
        width = int(field.get("width", 1))
        if base_vector is not None and not np.any(
            np.abs(vector[index : index + width] - base_vector[index : index + width])
            > threshold
        ):
            continue
        name = str(field["name"])
        value = _field_value(field, vector)
        if _apply_field(result, name, value):
            applied += 1
        else:
            skipped.append(name)
    return result, applied, tuple(skipped)


def asset_references(value: Any) -> tuple[str, ...]:
    found: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in ASSET_KEYS and isinstance(child, str) and child:
                    found.append(child)
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return tuple(sorted(set(found)))


def encode_serum2_preset(
    metadata: Mapping[str, Any], graph: Any, payload_version: int
) -> bytes:
    raw_cbor = cbor2.dumps(graph)
    compressed = zstandard.ZstdCompressor().compress(raw_cbor)
    output_metadata = copy.deepcopy(dict(metadata))
    output_metadata["hash"] = hashlib.md5(compressed).hexdigest()
    encoded_metadata = json.dumps(
        output_metadata, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return (
        MAGIC
        + struct.pack("<Q", len(encoded_metadata))
        + encoded_metadata
        + struct.pack("<II", len(raw_cbor), int(payload_version))
        + compressed
    )


def branded_serum2_metadata(source: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    """Retain required format fields while removing inherited pack branding."""

    metadata = {
        key: copy.deepcopy(source[key])
        for key in ("fileType", "product", "productVersion", "version")
        if key in source
    }
    metadata.update(
        {
            "presetName": name,
            "presetAuthor": APP_NAME,
            "presetDescription": GENERATED_PRESET_DESCRIPTION,
            "tags": list(GENERATED_PRESET_TAGS),
            "vendor": APP_NAME,
        }
    )
    return metadata


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _resolve_serum2_base(
    base_preset_id: int, db_path: Path
) -> tuple[Path, dict[str, Any]]:
    """Return the base preset's file path and decoded settings.

    A developer checkout answers both from `data/library.db`. A packaged or
    git-clone install has no such database, and its synthesis catalog carries
    only Serum 1 automation targets — so every Serum 2 export failed there with
    "Unknown Serum 2 base preset". The already-shipped factory bundle holds the
    same settings/metadata/payload_version this needs, and the locally scanned
    factory mapping resolves the file itself, so fall back to both rather than
    shipping a second copy of a 177 MB table.
    """

    from core.synthesis_assets import resolve_synthesis_assets

    resolved_db = Path(db_path)
    if resolved_db.is_file():
        try:
            database = Database(resolved_db)
            with database.connect() as connection:
                row = connection.execute(
                    "SELECT path FROM presets WHERE id=? AND synth='serum2'",
                    (base_preset_id,),
                ).fetchone()
            if row is not None:
                return (
                    Path(str(row["path"])).resolve(),
                    database.serum2_full_settings(base_preset_id),
                )
        except (KeyError, sqlite3.Error):
            pass  # fall through to the bundle

    assets = resolve_synthesis_assets()
    from core.factory_bundle import DEFAULT_FACTORY_BUNDLE, FactoryBundle

    bundle_path = DEFAULT_FACTORY_BUNDLE
    if not Path(bundle_path).is_file():
        raise KeyError(f"Unknown Serum 2 base preset {base_preset_id}")
    bundle = FactoryBundle(bundle_path)
    try:
        preset = bundle.preset_by_id(base_preset_id)
        settings, metadata, payload_version = bundle.settings(base_preset_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise KeyError(
            f"Unknown Serum 2 base preset {base_preset_id}"
        ) from exc

    local_path = _factory_path_for_hash(
        assets.factory_mapping, str(preset.content_hash)
    )
    if local_path is None:
        raise KeyError(
            f"Serum 2 base preset {base_preset_id} is not installed on this "
            "machine; its factory preset file could not be located by content hash"
        )
    return local_path, {
        "settings": settings,
        "metadata": metadata if metadata is not None else {},
        "payload_version": int(payload_version or 0),
    }


def _factory_path_for_hash(mapping_path: Path | None, content_hash: str) -> Path | None:
    if mapping_path is None or not Path(mapping_path).is_file():
        return None
    try:
        raw = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = raw.get("local_paths_by_hash", {}).get(content_hash)
    if not value:
        return None
    candidate = Path(str(value))
    return candidate.resolve() if candidate.is_file() else None


def write_serum2_preset(
    output_path: Path,
    *,
    base_preset_id: int,
    vector: np.ndarray,
    mask: np.ndarray,
    meaningfully_modified: bool,
    name: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    target_path: Path = DEFAULT_TARGET_PATH,
) -> Serum2WriteResult:
    """Write a native preset and decode it back before returning success."""

    base_path, base = _resolve_serum2_base(base_preset_id, db_path)
    base_graph = base["settings"]
    base_assets = asset_references(base_graph)
    output_path = Path(output_path).expanduser().resolve()

    output_name = name or generated_preset_name("serum2")
    if not meaningfully_modified:
        payload = encode_serum2_preset(
            branded_serum2_metadata(base["metadata"], name=output_name),
            base_graph,
            int(base["payload_version"]),
        )
        _atomic_write(output_path, payload)
        parsed = parse_serum2_preset(output_path)
        if parsed.data != base_graph:
            output_path.unlink(missing_ok=True)
            raise RuntimeError("Copied Serum 2 preset did not decode to its authoritative graph")
        return Serum2WriteResult(
            path=output_path,
            mode="copied-native-branded",
            base_preset_id=base_preset_id,
            applied_fields=0,
            skipped_fields=(),
            graph_sha256=_json_fingerprint(parsed.data),
            asset_references=asset_references(parsed.data),
        )

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    stored_targets = np.load(target_path)
    target_ids = np.asarray(stored_targets["preset_ids"], dtype=np.int64)
    matches = np.flatnonzero(target_ids == base_preset_id)
    if len(matches) != 1:
        raise RuntimeError(f"Serum 2 target store has no unique row for {base_preset_id}")
    base_vector = np.asarray(stored_targets["vectors"][int(matches[0])], dtype=np.float32)
    intended, applied, skipped = overlay_vector(
        base_graph, schema, vector, mask, base_vector
    )
    intended_assets = asset_references(intended)
    if intended_assets != base_assets:
        raise RuntimeError("Candidate overlay changed a base preset asset reference")

    metadata = branded_serum2_metadata(base["metadata"], name=output_name)

    payload = encode_serum2_preset(metadata, intended, int(base["payload_version"]))
    _atomic_write(output_path, payload)
    parsed = parse_serum2_preset(output_path)
    if parsed.data != intended:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Written Serum 2 preset failed decoded-graph equality")
    if parsed.metadata.get("hash") != hashlib.md5(
        payload[-parsed.compressed_length :]
    ).hexdigest():
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Written Serum 2 preset has an invalid payload hash")
    if asset_references(parsed.data) != base_assets:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Written Serum 2 preset did not retain base asset references")
    return Serum2WriteResult(
        path=output_path,
        mode="optimized-overlay",
        base_preset_id=base_preset_id,
        applied_fields=applied,
        skipped_fields=skipped,
        graph_sha256=_json_fingerprint(parsed.data),
        asset_references=asset_references(parsed.data),
    )


def vector_was_modified(
    vector: np.ndarray, base_vector: np.ndarray, mask: np.ndarray, threshold: float = MEANINGFUL_DELTA
) -> bool:
    values = np.asarray(vector, dtype=np.float32)
    base = np.asarray(base_vector, dtype=np.float32)
    active = np.asarray(mask, dtype=np.bool_)
    return bool(np.any(np.abs(values[active] - base[active]) > threshold))
