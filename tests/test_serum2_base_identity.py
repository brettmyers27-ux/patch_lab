"""Family B: a generated preset's base is resolved by identity, never by row number.

The shipped synthesis catalog (which the matcher, target store and render
states all number presets by) and the shipped factory bundle (which holds the
settings a preset is written from) use independent integer ids.  A beta
tester's saves failed deterministically because catalog id 171 (Serum 2
"DR - Shake") was looked up as bundle id 171 (an unrelated Serum 1 preset), and
catalog id 4831 is beyond the bundle's range entirely.  Worse, 172 catalog ids
land inside the bundle's Serum 2 range, where the old lookup silently wrote a
*different* Serum 2 preset.

These fixtures reproduce every one of those shapes with a small catalog and
bundle built in the real formats.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from core.base_preset_identity import (
    BaseIdentityError,
    resolve_serum1_base,
    resolve_serum2_base,
)
from core.factory_bundle import BUNDLE_SCHEMA, compress_array, compress_json, compress_mask
from core.serum2_preset import parse_serum2_preset
from core.serum2_preset_writer import write_serum2_preset


VECTOR_LENGTH = 6
REAL = Path(__file__).resolve().parents[1] / "data" / "runtime" / "v1-legacy-stock-clap"


def _hash(label: str) -> str:
    return hashlib.sha1(label.encode()).hexdigest()


def _graph(label: str) -> dict:
    return {"Oscillator0": {"plainParams": {"kParamVolume": 0.5, "kParamTag": label}}, "Global0": {"name": label}}


def _vector(seed: int) -> np.ndarray:
    return np.linspace(0.05 * seed, 0.05 * seed + 0.5, VECTOR_LENGTH, dtype=np.float32) % 1.0


#: catalog id -> (name, synth, content label, target seed)
CATALOG = {
    171: ("DR - Shake", "serum2", "shake", 1),        # bundle 171 is an unrelated Serum 1 preset
    5: ("PL - Five", "serum2", "five", 2),            # bundle 5 is a different Serum 2 preset
    900: ("BS - Nine Hundred", "serum2", "nine", 3),  # inside the bundle's Serum 2 range
    4831: ("WA_RT_BS_Reese_Wobble", "serum2", "reese", 4),  # beyond the bundle's range
    777: ("Orphan", "serum2", "orphan", 5),           # its content is not in the bundle at all
    12: ("LD Serum One", "serum1", "s1", 6),
}

#: bundle id -> (name, synth, content label, vector seed)
BUNDLE = {
    171: ("LD Heavy Metal[SD]", "serum1", "heavy-metal", 20),
    5: ("PAD - Some Other Pad", "serum2", "other-pad", 21),
    628: ("DR - Shake", "serum2", "shake", 1),
    700: ("PL - Five", "serum2", "five", 2),
    900: ("KY - Wrong Keys", "serum2", "wrong-keys", 22),
    901: ("BS - Nine Hundred", "serum2", "nine", 3),
    1142: ("WA_RT_BS_Reese_Wobble", "serum2", "reese", 4),
    777: ("KY - Also Not Orphan", "serum2", "not-orphan", 23),
}


def build_fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    catalog = root / "catalog.sqlite"
    with sqlite3.connect(catalog) as connection:
        connection.execute(
            "CREATE TABLE presets (id INTEGER PRIMARY KEY, path TEXT, name TEXT, synth TEXT, content_hash TEXT)"
        )
        for preset_id, (name, synth, label, _seed) in CATALOG.items():
            connection.execute(
                "INSERT INTO presets VALUES (?,?,?,?,?)",
                (preset_id, f"/nowhere/{name}.SerumPreset", name, synth, _hash(label)),
            )
    bundle = root / "bundle.sqlite"
    with sqlite3.connect(bundle) as connection:
        connection.executescript(BUNDLE_SCHEMA)
        for synth in ("serum1", "serum2"):
            connection.execute(
                "INSERT INTO schemas VALUES (?,?)", (synth, json.dumps({"vector_length": VECTOR_LENGTH}))
            )
        for preset_id, (name, synth, label, seed) in BUNDLE.items():
            connection.execute(
                "INSERT INTO presets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    preset_id, _hash(label), name, synth, f"Presets/{name}", ".SerumPreset", 1,
                    compress_array(_vector(seed)), compress_mask(np.ones(VECTOR_LENGTH, bool)),
                    compress_json(_graph(label)),
                    compress_json({"fileType": "SerumPreset", "presetName": name}) if synth == "serum2" else None,
                    2 if synth == "serum2" else None,
                ),
            )
    targets = root / "serum2_targets.npz"
    ids = [cid for cid, row in CATALOG.items() if row[1] == "serum2"]
    np.savez(
        targets,
        preset_ids=np.asarray(ids, dtype=np.int64),
        vectors=np.stack([_vector(CATALOG[cid][3]) for cid in ids]),
        masks=np.ones((len(ids), VECTOR_LENGTH), dtype=bool),
    )
    return {"catalog_path": catalog, "bundle_path": bundle, "targets_path": targets}


@pytest.fixture
def stores(tmp_path: Path) -> dict[str, Path]:
    return build_fixture(tmp_path / "stores")


# --- A. catalog id collides with a Serum 1 bundle row -------------------------


def test_A_collision_with_a_serum1_bundle_row_resolves_the_right_serum2_preset(stores) -> None:
    base = resolve_serum2_base(171, **stores)
    assert base.identity.name == "DR - Shake"
    assert base.identity.bundle_id == 628, "resolved by content hash, not by the number 171"
    assert base.settings == _graph("shake")
    assert base.settings != _graph("heavy-metal"), "never the unrelated Serum 1 preset"


# --- B. catalog id collides with a different Serum 2 bundle row ---------------


@pytest.mark.parametrize("catalog_id,bundle_id,label", [(5, 700, "five"), (900, 901, "nine")])
def test_B_collision_with_a_different_serum2_preset_resolves_by_identity(stores, catalog_id, bundle_id, label) -> None:
    base = resolve_serum2_base(catalog_id, **stores)
    assert base.identity.bundle_id == bundle_id
    assert base.settings == _graph(label)
    wrong = BUNDLE[catalog_id]
    assert base.settings != _graph(wrong[2]), f"must not be bundle row {catalog_id} ({wrong[0]})"


# --- C. catalog id beyond the bundle's range ---------------------------------


def test_C_catalog_id_outside_the_bundle_range_still_resolves(stores) -> None:
    base = resolve_serum2_base(4831, **stores)
    assert (base.identity.bundle_id, base.identity.name) == (1142, "WA_RT_BS_Reese_Wobble")


def test_C_missing_content_fails_closed_even_when_the_number_exists(stores) -> None:
    """Bundle id 777 exists and is Serum 2, which is exactly the trap."""

    with pytest.raises(BaseIdentityError) as caught:
        resolve_serum2_base(777, **stores)
    assert "0 presets with content hash" in str(caught.value)


def test_unknown_catalog_id_fails_closed(stores) -> None:
    with pytest.raises(BaseIdentityError):
        resolve_serum2_base(99_999, **stores)


def test_a_serum1_catalog_row_is_never_used_as_a_serum2_base(stores) -> None:
    with pytest.raises(BaseIdentityError, match="serum1 preset, not Serum 2"):
        resolve_serum2_base(12, **stores)


# --- defence in depth: every cross-check can independently refuse -------------


def test_a_name_disagreement_under_a_shared_hash_fails_closed(stores) -> None:
    with sqlite3.connect(stores["bundle_path"]) as connection:
        connection.execute("UPDATE presets SET name='Tampered' WHERE id=628")
    with pytest.raises(BaseIdentityError, match="disagrees"):
        resolve_serum2_base(171, **stores)


def test_a_parameter_vector_disagreement_fails_closed(stores) -> None:
    """The target store (catalog ids) and the bundle must describe the same preset."""

    with sqlite3.connect(stores["bundle_path"]) as connection:
        connection.execute(
            "UPDATE presets SET parameter_vector=? WHERE id=628", (compress_array(_vector(20)),)
        )
    with pytest.raises(BaseIdentityError, match="does not carry the parameters"):
        resolve_serum2_base(171, **stores)


def test_a_missing_target_row_fails_closed(stores, tmp_path) -> None:
    targets = tmp_path / "empty.npz"
    np.savez(targets, preset_ids=np.asarray([5], dtype=np.int64),
             vectors=_vector(2)[None, :], masks=np.ones((1, VECTOR_LENGTH), bool))
    with pytest.raises(BaseIdentityError, match="0 rows for catalog preset 171"):
        resolve_serum2_base(171, **{**stores, "targets_path": targets})


def test_the_old_numeric_lookup_is_gone_from_the_writer() -> None:
    source = Path("core/serum2_preset_writer.py").read_text(encoding="utf-8")
    assert "preset_by_id" not in source
    assert "_resolve_serum2_base" not in source
    assert "serum2_full_settings" not in source


# --- the writer produces the RIGHT preset, bytes and all ----------------------


def test_the_written_preset_is_built_on_the_identified_base(stores, tmp_path) -> None:
    output = tmp_path / "out" / "Generated.SerumPreset"
    result = write_serum2_preset(
        output, base_preset_id=171, vector=_vector(1), mask=np.ones(VECTOR_LENGTH, bool),
        meaningfully_modified=False, **{k: v for k, v in stores.items() if k != "targets_path"},
        target_path=stores["targets_path"],
    )
    parsed = parse_serum2_preset(output)
    assert parsed.data == _graph("shake")
    assert result.base_identity is not None and result.base_identity.content_hash == _hash("shake")
    assert parsed.metadata["presetName"] == "PatchLab Serum 2 Match"


def test_a_modified_preset_overlays_onto_the_identified_base(stores, tmp_path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({
        "vector_length": VECTOR_LENGTH,
        "fields": [{"index": 0, "name": "Oscillator0.plainParams.kParamVolume",
                    "encoding": "continuous", "minimum": 0.0, "maximum": 1.0}],
    }))
    vector = _vector(4).copy()
    vector[0] = 0.9
    output = tmp_path / "Modified.SerumPreset"
    write_serum2_preset(
        output, base_preset_id=4831, vector=vector, mask=np.ones(VECTOR_LENGTH, bool),
        meaningfully_modified=True, schema_path=schema,
        catalog_path=stores["catalog_path"], bundle_path=stores["bundle_path"],
        target_path=stores["targets_path"],
    )
    data = parse_serum2_preset(output).data
    assert data["Global0"]["name"] == "reese", "untouched state comes from the right base"
    assert data["Oscillator0"]["plainParams"]["kParamVolume"] == pytest.approx(0.9)


def test_an_unresolvable_base_writes_nothing(stores, tmp_path) -> None:
    output = tmp_path / "never.SerumPreset"
    with pytest.raises(BaseIdentityError):
        write_serum2_preset(
            output, base_preset_id=777, vector=_vector(5), mask=np.ones(VECTOR_LENGTH, bool),
            meaningfully_modified=False, catalog_path=stores["catalog_path"],
            bundle_path=stores["bundle_path"], target_path=stores["targets_path"],
        )
    assert not output.exists()


# --- Serum 1: the local file must be the catalog's content --------------------


def test_serum1_base_is_accepted_only_with_matching_content(stores, tmp_path) -> None:
    right = tmp_path / "right.fxp"
    right.write_bytes(b"s1")  # sha1("s1") is the catalog hash of preset 12
    mapping = tmp_path / "factory-paths.json"
    mapping.write_text(json.dumps({"local_paths_by_hash": {_hash("s1"): str(right)}}))
    resolved = resolve_serum1_base(12, catalog_path=stores["catalog_path"], factory_mapping=mapping)
    assert resolved.path == right.resolve()

    right.write_bytes(b"edited by the user")
    with pytest.raises(BaseIdentityError, match="matching content"):
        resolve_serum1_base(12, catalog_path=stores["catalog_path"], factory_mapping=mapping)


def test_serum1_resolution_refuses_a_serum2_catalog_row(stores) -> None:
    with pytest.raises(BaseIdentityError, match="not Serum 1"):
        resolve_serum1_base(171, catalog_path=stores["catalog_path"], factory_mapping=None)


# --- the shipped runtime family itself -----------------------------------------


needs_runtime = pytest.mark.skipif(
    not (REAL / "dist" / "factory_bundle.sqlite").is_file(),
    reason="shipped runtime family not present in this checkout",
)


@needs_runtime
@pytest.mark.parametrize(
    "catalog_id,name",
    [
        (1, None),                       # low catalog id
        (171, "DR - Shake"),             # bundle 171 is a Serum 1 preset
        (458, "PL - Membrane Harp"),     # bundle 458 is a different Serum 2 preset
        (4831, "WA_RT_BS_Reese_Wobble"),  # beyond the bundle's range
    ],
)
def test_shipped_stores_resolve_the_beta_testers_presets(catalog_id, name) -> None:
    base = resolve_serum2_base(catalog_id)
    with sqlite3.connect(REAL / "models" / "patchlab-synthesis-catalog.sqlite") as connection:
        catalog_hash, catalog_name = connection.execute(
            "SELECT content_hash,name FROM presets WHERE id=?", (catalog_id,)
        ).fetchone()
    assert base.identity.content_hash == catalog_hash
    assert base.identity.name == catalog_name == (name or catalog_name)
    assert base.identity.bundle_id != catalog_id, "the two numberings never coincide for these"


@needs_runtime
def test_every_shipped_serum2_catalog_preset_resolves_uniquely() -> None:
    with sqlite3.connect(REAL / "models" / "patchlab-synthesis-catalog.sqlite") as connection:
        ids = [row[0] for row in connection.execute("SELECT id FROM presets WHERE synth='serum2'")]
    bundle_ids = {resolve_serum2_base(catalog_id).identity.bundle_id for catalog_id in ids}
    assert len(ids) == 710 and len(bundle_ids) == 710
