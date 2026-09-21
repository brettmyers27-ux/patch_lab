from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import core.runtime_compatibility as compatibility


def test_runtime_family_rejects_missing_or_mixed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(compatibility, "runtime_root", lambda: tmp_path)
    with pytest.raises(compatibility.RuntimeCompatibilityError, match="missing"):
        compatibility.validate_runtime_family()


def test_runtime_family_accepts_only_complete_fixture_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = b"legacy factory"
    second = b"legacy index"
    artifacts = (
        compatibility.RuntimeArtifact("dist/factory_bundle.sqlite", hashlib.sha256(first).hexdigest()),
        compatibility.RuntimeArtifact("features/preset_index.npy", hashlib.sha256(second).hexdigest()),
    )
    monkeypatch.setattr(compatibility, "runtime_root", lambda: tmp_path)
    monkeypatch.setattr(compatibility, "REQUIRED_ARTIFACTS", artifacts)
    monkeypatch.setattr(compatibility, "MIN_RENDER_STATE_FILES", 1)
    manifest = (
        "".join(
            (
                hashlib.sha256(b"state").hexdigest(),
                "  ./artifacts/data/models/serum2_render_states/1.vstpreset\n",
            )
        ).encode("utf-8")
    )
    monkeypatch.setattr(
        compatibility, "STATE_CHECKSUMS_SHA256", hashlib.sha256(manifest).hexdigest()
    )
    root = compatibility.runtime_data_root()
    (root / "dist").mkdir(parents=True)
    (root / "features").mkdir(parents=True)
    (root / "models" / "serum2_render_states").mkdir(parents=True)
    (root / "dist" / "factory_bundle.sqlite").write_bytes(first)
    (root / "features" / "preset_index.npy").write_bytes(second)
    (root / "models" / "serum2_render_states" / "1.vstpreset").write_bytes(b"state")
    (root / compatibility.STATE_CHECKSUMS_NAME).write_bytes(manifest)

    assert compatibility.validate_runtime_family() == root
    (root / "features" / "preset_index.npy").write_bytes(b"mixed")
    with pytest.raises(compatibility.RuntimeCompatibilityError, match="hash mismatch"):
        compatibility.validate_runtime_family()
