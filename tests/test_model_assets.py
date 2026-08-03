from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

from core.build_info import assert_packaged_commit, current_build_info
from core.model_assets import (
    CLAP_CHECKPOINT_NAME,
    MIN_CHECKPOINT_BYTES,
    TOKENIZER_REQUIREMENTS,
    ModelAssetsError,
    configure_model_environment,
    validate_model_assets,
)
import core.model_assets as model_assets


def _populate_assets(cache: Path, checkpoint: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("wb") as handle:
        handle.truncate(MIN_CHECKPOINT_BYTES)
    for model_name, filenames in TOKENIZER_REQUIREMENTS.items():
        snapshot = cache / "transformers" / model_name / "snapshots" / "fixture"
        snapshot.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            (snapshot / filename).write_text("fixture", encoding="utf-8")


def test_shared_model_resolution_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    checkpoint = tmp_path / "checkpoint.pt"
    _populate_assets(cache, checkpoint)
    monkeypatch.setenv("PATCHLAB_MODEL_CACHE", str(cache))
    monkeypatch.setenv("PATCHLAB_CLAP_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "wrong"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "wrong-transformers"))

    assets = validate_model_assets()

    assert assets.cache_dir == cache.resolve()
    assert assets.checkpoint == checkpoint.resolve()
    assert os.environ["HF_HOME"] == str(cache.resolve())
    assert os.environ["TRANSFORMERS_CACHE"] == str(
        cache.resolve() / "transformers"
    )


def test_default_model_resolution_uses_adopted_stage2b_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PATCHLAB_CLAP_CHECKPOINT", raising=False)
    monkeypatch.setattr(model_assets, "runtime_root", lambda: tmp_path)

    assets = model_assets.resolve_model_assets()

    assert CLAP_CHECKPOINT_NAME == "patchlab_clap_ft_v1.pt"
    assert assets.checkpoint == (tmp_path / "data" / "models" / CLAP_CHECKPOINT_NAME)


def test_missing_assets_error_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "empty-cache"
    checkpoint = tmp_path / "missing-checkpoint.pt"
    monkeypatch.setenv("PATCHLAB_MODEL_CACHE", str(cache))
    monkeypatch.setenv("PATCHLAB_CLAP_CHECKPOINT", str(checkpoint))

    with pytest.raises(ModelAssetsError) as captured:
        validate_model_assets()

    message = str(captured.value)
    assert str(cache.resolve()) in message
    assert str(checkpoint.resolve()) in message
    assert "install.sh" in message
    assert "install.ps1" in message
    assert "scripts/cache_clap.py" in message


def test_smaller_authenticated_finetuned_checkpoint_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    checkpoint = tmp_path / "patchlab_clap_ft_v1.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                "audio_branch.fixture": torch.ones(1),
                "audio_projection.fixture": torch.ones(1),
            },
            "patchlab_metadata": {"format": "patchlab_clap_ft_v1"},
        },
        checkpoint,
    )
    for model_name, filenames in TOKENIZER_REQUIREMENTS.items():
        snapshot = cache / "transformers" / model_name / "snapshots" / "fixture"
        snapshot.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            (snapshot / filename).write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(model_assets, "MIN_FINETUNED_CHECKPOINT_BYTES", 0)
    monkeypatch.setenv("PATCHLAB_MODEL_CACHE", str(cache))
    monkeypatch.setenv("PATCHLAB_CLAP_CHECKPOINT", str(checkpoint))

    assets = validate_model_assets()

    assert assets.checkpoint == checkpoint.resolve()


def test_offline_default_allows_explicit_diagnostic_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATCHLAB_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("PATCHLAB_CLAP_CHECKPOINT", str(tmp_path / "checkpoint.pt"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    configure_model_environment()

    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_packaged_build_identity_detects_stale_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "source_commit": "a" * 40,
        "built_at_utc": "2026-07-29T00:00:00+00:00",
        "source_dirty": False,
    }
    (tmp_path / "patchlab-build-info.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    info = current_build_info()
    assert info.source_commit == "a" * 40
    assert_packaged_commit("a" * 40)
    with pytest.raises(RuntimeError, match="Stale PatchLab build"):
        assert_packaged_commit("b" * 40)
