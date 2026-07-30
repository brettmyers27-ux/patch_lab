#!/usr/bin/env python3
"""Download and validate the pinned LAION-CLAP music checkpoint and HF assets."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_assets import configure_model_environment  # noqa: E402


MODEL_ASSETS = configure_model_environment()
MODEL_DIR = MODEL_ASSETS.model_dir
HF_DIR = MODEL_ASSETS.cache_dir
CHECKPOINT = MODEL_ASSETS.checkpoint
CHECKPOINT_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/"
    "music_audioset_epoch_15_esc_90.14.pt"
)


def _configure_cache() -> None:
    configure_model_environment()


def _download() -> None:
    if CHECKPOINT.is_file() and CHECKPOINT.stat().st_size > 100_000_000:
        print(f"CHECKPOINT_CACHED={CHECKPOINT} bytes={CHECKPOINT.stat().st_size}")
        return
    temporary = CHECKPOINT.with_suffix(CHECKPOINT.suffix + ".part")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    def progress(blocks: int, block_size: int, total: int) -> None:
        received = blocks * block_size
        if blocks % 128 == 0:
            percent = 100.0 * received / total if total > 0 else 0.0
            print(f"CHECKPOINT_DOWNLOAD={received}/{total} ({percent:.1f}%)", flush=True)

    urllib.request.urlretrieve(CHECKPOINT_URL, temporary, progress)
    temporary.replace(CHECKPOINT)
    print(f"CHECKPOINT_DOWNLOADED={CHECKPOINT} bytes={CHECKPOINT.stat().st_size}")


def main() -> int:
    _configure_cache()
    _download()
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device="cpu")
    model.load_ckpt(str(CHECKPOINT), verbose=False)
    dimensions = int(model.model.audio_projection[-1].out_features)
    print(f"CLAP_CACHE_PASS checkpoint={CHECKPOINT.name} embedding_dimensions={dimensions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
