"""Resolve a generated preset's base by stable identity, never by a raw row id.

PatchLab's matcher, its Serum 2 target store and its render states all number
presets in one space: the synthesis catalog's ``presets.id`` (the "catalog
id").  The shipped factory bundle, which carries the full Serum 2 settings
graph a preset is written from, was built by renumbering those same presets
``1..N`` (``scripts/build_factory_bundle.py``).  The two stores therefore share
no numeric identity at all: catalog id 171 is the Serum 2 preset "DR - Shake",
while bundle id 171 is an unrelated Serum 1 preset, and 172 catalog ids fall
inside the bundle's own Serum 2 range where they silently named a *different*
Serum 2 preset.

The identity they do share is the preset file's content hash (SHA-1 of its
bytes), which both stores recorded from the same file.  Resolution is:

    catalog id -> catalog row (synth, name, content hash)
               -> the one bundle row with that content hash
               -> cross-checked by name and by the stored parameter vector
               -> settings graph used to write the preset

Anything that does not line up fails closed with ``BaseIdentityError``; no
caller ever receives a plausible-looking but different preset.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_SHA1 = re.compile(r"^[0-9a-f]{40}$")
#: Stored vectors are float32 round trips of the same values.
VECTOR_TOLERANCE = 1e-6


class BaseIdentityError(RuntimeError):
    """The base preset of a generated result cannot be resolved safely."""

    user_message = (
        "PatchLab couldn't confirm which Serum preset this result is built on, "
        "so it did not save a file."
    )

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.detail = detail


@dataclass(frozen=True, slots=True)
class BaseIdentity:
    """Who a generated preset's base is, in terms that survive any renumbering."""

    synth: str
    catalog_id: int
    content_hash: str
    name: str
    bundle_id: int | None = None
    source_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "synth": self.synth,
            "catalog_id": self.catalog_id,
            "content_hash": self.content_hash,
            "name": self.name,
            "bundle_id": self.bundle_id,
            "source_path": self.source_path,
        }


@dataclass(frozen=True, slots=True)
class ResolvedSerum2Base:
    identity: BaseIdentity
    settings: Any
    metadata: dict[str, Any]
    payload_version: int
    #: The target-store vector of the catalog id, verified equal to the bundle's.
    base_vector: np.ndarray
    settings_sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedSerum1Base:
    identity: BaseIdentity
    path: Path


def _runtime_paths() -> dict[str, Path]:
    from core.runtime_compatibility import runtime_data_root
    from core.synthesis_assets import SYNTHESIS_CATALOG_NAME, SERUM2_TARGETS_NAME

    root = runtime_data_root()
    return {
        "catalog": root / "models" / SYNTHESIS_CATALOG_NAME,
        "bundle": root / "dist" / "factory_bundle.sqlite",
        "targets": root / "features" / SERUM2_TARGETS_NAME,
    }


def _catalog_row(catalog_path: Path, catalog_id: int) -> tuple[str, str, str, str]:
    if not Path(catalog_path).is_file():
        raise BaseIdentityError(
            f"the synthesis catalog is missing: {catalog_path}", catalog_path=str(catalog_path)
        )
    with sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT synth,content_hash,name,path FROM presets WHERE id=?", (int(catalog_id),)
        ).fetchone()
    if row is None:
        raise BaseIdentityError(
            f"catalog preset {catalog_id} does not exist", catalog_id=int(catalog_id)
        )
    synth, content_hash, name, path = (str(value or "") for value in row)
    if not _SHA1.fullmatch(content_hash):
        raise BaseIdentityError(
            f"catalog preset {catalog_id} has no usable content hash",
            catalog_id=int(catalog_id),
        )
    return synth, content_hash, name, path


def resolve_serum2_base(
    catalog_id: int,
    *,
    catalog_path: Path | None = None,
    bundle_path: Path | None = None,
    targets_path: Path | None = None,
) -> ResolvedSerum2Base:
    """Return the verified settings graph a Serum 2 result is built on."""

    from core.factory_bundle import FactoryBundle

    defaults = _runtime_paths()
    catalog_path = Path(catalog_path or defaults["catalog"])
    bundle_path = Path(bundle_path or defaults["bundle"])
    targets_path = Path(targets_path or defaults["targets"])
    catalog_id = int(catalog_id)

    synth, content_hash, name, path = _catalog_row(catalog_path, catalog_id)
    if synth != "serum2":
        raise BaseIdentityError(
            f"catalog preset {catalog_id} is a {synth} preset, not Serum 2",
            catalog_id=catalog_id, catalog_synth=synth,
        )
    if not bundle_path.is_file():
        raise BaseIdentityError(
            f"the factory bundle is missing: {bundle_path}", bundle_path=str(bundle_path)
        )
    bundle = FactoryBundle(bundle_path)
    # By content hash only.  The bundle's own ``id`` is a different numbering
    # and is never compared with, or substituted for, the catalog id.
    with bundle.connect() as connection:
        rows = connection.execute(
            "SELECT id,synth,name FROM presets WHERE content_hash=?", (content_hash,)
        ).fetchall()
    if len(rows) != 1:
        raise BaseIdentityError(
            f"the factory bundle has {len(rows)} presets with content hash "
            f"{content_hash} (catalog preset {catalog_id}); expected exactly one",
            catalog_id=catalog_id, content_hash=content_hash, bundle_matches=len(rows),
        )
    bundle_id, bundle_synth, bundle_name = int(rows[0]["id"]), str(rows[0]["synth"]), str(rows[0]["name"])
    identity = BaseIdentity("serum2", catalog_id, content_hash, name, bundle_id, path)
    if bundle_synth != "serum2" or bundle_name != name:
        raise BaseIdentityError(
            f"bundle preset {bundle_id} ({bundle_synth} {bundle_name!r}) disagrees with "
            f"catalog preset {catalog_id} (serum2 {name!r}) despite a shared content hash",
            **identity.as_dict(),
        )

    # Independent second check: the parameter vector the matcher optimised from
    # (keyed by catalog id) must be the one the bundle stored for this preset.
    if not targets_path.is_file():
        raise BaseIdentityError(
            f"the Serum 2 target store is missing: {targets_path}", **identity.as_dict()
        )
    with np.load(targets_path) as stored:
        target_ids = np.asarray(stored["preset_ids"], dtype=np.int64)
        positions = np.flatnonzero(target_ids == catalog_id)
        if len(positions) != 1:
            raise BaseIdentityError(
                f"the Serum 2 target store has {len(positions)} rows for catalog preset "
                f"{catalog_id}; expected exactly one",
                **identity.as_dict(),
            )
        base_vector = np.asarray(stored["vectors"][int(positions[0])], dtype=np.float32)
        base_mask = np.asarray(stored["masks"][int(positions[0])], dtype=np.bool_)
    bundle_vector, bundle_mask = bundle.parameters(bundle_id)
    if not (
        np.array_equal(np.asarray(bundle_mask, dtype=np.bool_), base_mask)
        and np.allclose(bundle_vector, base_vector, atol=VECTOR_TOLERANCE)
    ):
        raise BaseIdentityError(
            f"bundle preset {bundle_id} does not carry the parameters of catalog "
            f"preset {catalog_id} ({name!r})",
            **identity.as_dict(),
        )

    settings, metadata, payload_version = bundle.settings(bundle_id)
    if not isinstance(settings, dict) or not settings:
        raise BaseIdentityError(
            f"bundle preset {bundle_id} has no Serum 2 settings graph", **identity.as_dict()
        )
    if payload_version is None:
        raise BaseIdentityError(
            f"bundle preset {bundle_id} has no Serum 2 payload version", **identity.as_dict()
        )
    settings_sha256 = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return ResolvedSerum2Base(
        identity=identity,
        settings=settings,
        metadata=dict(metadata or {}),
        payload_version=int(payload_version),
        base_vector=base_vector,
        settings_sha256=settings_sha256,
    )


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _factory_paths_by_hash(mapping_path: Path | None) -> dict[str, str]:
    if mapping_path is None or not Path(mapping_path).is_file():
        return {}
    try:
        raw = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(key): str(value) for key, value in raw.get("local_paths_by_hash", {}).items()}


def resolve_serum1_base(
    catalog_id: int,
    *,
    catalog_path: Path | None = None,
    factory_mapping: Path | None = None,
) -> ResolvedSerum1Base:
    """Return the local ``.fxp`` a Serum 1 result is built on, verified by content.

    Serum 1 is written by loading the base file into the plug-in, so the file
    itself is required.  The catalog's stored path came from the machine that
    built it, so it is accepted only when its bytes hash to the catalog's
    content hash; otherwise this Mac's scanned factory mapping is consulted.
    """

    catalog_path = Path(catalog_path or _runtime_paths()["catalog"])
    if factory_mapping is None:
        from core.synthesis_assets import resolve_synthesis_assets

        factory_mapping = resolve_synthesis_assets().factory_mapping
    catalog_id = int(catalog_id)
    synth, content_hash, name, stored = _catalog_row(catalog_path, catalog_id)
    if synth != "serum1":
        raise BaseIdentityError(
            f"catalog preset {catalog_id} is a {synth} preset, not Serum 1",
            catalog_id=catalog_id, catalog_synth=synth,
        )
    candidates = [stored, _factory_paths_by_hash(factory_mapping).get(content_hash, "")]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            if path.is_file() and _sha1_file(path) == content_hash:
                return ResolvedSerum1Base(
                    BaseIdentity("serum1", catalog_id, content_hash, name, None, str(path)),
                    path.resolve(),
                )
        except OSError:
            continue
    raise BaseIdentityError(
        f"Serum 1 preset {name!r} (catalog {catalog_id}) is not on this Mac with "
        "matching content",
        catalog_id=catalog_id, content_hash=content_hash, name=name,
    )
