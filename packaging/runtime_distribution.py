"""PyInstaller runtime hook for platform-neutral packaged runtime paths."""

import os
import sys
from pathlib import Path

os.environ["PATCHLAB_DISTRIBUTION_MODE"] = "1"
runtime_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
model_dir = runtime_root / "data" / "models"
os.environ.setdefault(
    "PATCHLAB_MODEL_CACHE",
    str(model_dir / "huggingface"),
)
os.environ.setdefault(
    "PATCHLAB_CLAP_CHECKPOINT",
    str(model_dir / "music_audioset_epoch_15_esc_90.14.pt"),
)
