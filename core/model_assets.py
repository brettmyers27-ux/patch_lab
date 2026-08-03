"""Shared, platform-neutral resolution and validation for CLAP model assets."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


CLAP_CHECKPOINT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
MIN_CHECKPOINT_BYTES = 1_000_000_000
MIN_FINETUNED_CHECKPOINT_BYTES = 750_000_000
TOKENIZER_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "models--bert-base-uncased": (
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
    ),
    "models--roberta-base": (
        "config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.json",
    ),
    "models--facebook--bart-base": (
        "config.json",
        "merges.txt",
        "vocab.json",
    ),
}


class ModelAssetsError(RuntimeError):
    """Raised before CLAP initialization when its offline assets are incomplete."""


def _is_patchlab_finetuned_checkpoint(path: Path) -> bool:
    """Recognize the smaller audio-tower-only Stage 2 checkpoint safely."""

    if not path.is_file() or path.stat().st_size < MIN_FINETUNED_CHECKPOINT_BYTES:
        return False
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload.get("patchlab_metadata", {})
        state = payload.get("state_dict", {})
        return (
            metadata.get("format") == "patchlab_clap_ft_v1"
            and isinstance(state, dict)
            and any(key.startswith("audio_branch.") for key in state)
            and any(key.startswith("audio_projection.") for key in state)
        )
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError):
        return False


@dataclass(frozen=True, slots=True)
class ModelAssets:
    runtime_root: Path
    model_dir: Path
    cache_dir: Path
    checkpoint: Path


def runtime_root() -> Path:
    """Return the source checkout or PyInstaller's bundled runtime root."""

    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(str(frozen_root)).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def resolve_model_assets() -> ModelAssets:
    """Resolve all model paths from the same knobs on every supported OS."""

    root = runtime_root()
    model_dir = root / "data" / "models"
    cache = Path(
        os.environ.get(
            "PATCHLAB_MODEL_CACHE",
            str(model_dir / "huggingface"),
        )
    ).expanduser().resolve()
    checkpoint = Path(
        os.environ.get(
            "PATCHLAB_CLAP_CHECKPOINT",
            str(model_dir / CLAP_CHECKPOINT_NAME),
        )
    ).expanduser().resolve()
    return ModelAssets(root, model_dir, cache, checkpoint)


def configure_model_environment() -> ModelAssets:
    """Make Hugging Face consume the resolved PatchLab cache exclusively."""

    assets = resolve_model_assets()
    assets.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PATCHLAB_MODEL_CACHE"] = str(assets.cache_dir)
    os.environ["PATCHLAB_CLAP_CHECKPOINT"] = str(assets.checkpoint)
    os.environ["HF_HOME"] = str(assets.cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(assets.cache_dir / "transformers")
    os.environ["TORCH_HOME"] = str(assets.model_dir / "torch")
    # Long matching/batch sessions must never stall on hidden network retries.
    # Operators can explicitly set either variable to "0" for diagnostics.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return assets


def _snapshot_has_files(model_root: Path, required: tuple[str, ...]) -> bool:
    snapshots = model_root / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        snapshot.is_dir()
        and all((snapshot / filename).is_file() for filename in required)
        for snapshot in snapshots.iterdir()
    )


def validate_model_assets(
    *,
    checkpoint: Path | None = None,
) -> ModelAssets:
    """Fail early with one actionable description of every missing asset."""

    assets = configure_model_environment()
    resolved_checkpoint = (
        checkpoint.expanduser().resolve() if checkpoint is not None else assets.checkpoint
    )
    problems: list[str] = []
    if not resolved_checkpoint.is_file() or (
        resolved_checkpoint.stat().st_size < MIN_CHECKPOINT_BYTES
        and not _is_patchlab_finetuned_checkpoint(resolved_checkpoint)
    ):
        problems.append(
            "the pinned CLAP checkpoint is missing or incomplete at "
            f"{resolved_checkpoint}"
        )

    transformer_cache = assets.cache_dir / "transformers"
    for model_name, required in TOKENIZER_REQUIREMENTS.items():
        if not _snapshot_has_files(transformer_cache / model_name, required):
            readable = model_name.removeprefix("models--").replace("--", "/")
            problems.append(
                f"the required {readable} tokenizer/model cache is incomplete under "
                f"{transformer_cache / model_name}"
            )

    if problems:
        detail = "\n• ".join(problems)
        raise ModelAssetsError(
            "PatchLab model assets are unavailable, so matching cannot start.\n"
            f"Resolved model cache: {assets.cache_dir}\n"
            f"Resolved CLAP checkpoint: {resolved_checkpoint}\n"
            f"• {detail}\n"
            "Run the PatchLab installer again (install.sh on macOS or install.ps1 "
            "on Windows). For a development checkout, run "
            "`python scripts/cache_clap.py`, then restart PatchLab."
        )
    return ModelAssets(
        assets.runtime_root,
        assets.model_dir,
        assets.cache_dir,
        resolved_checkpoint,
    )
