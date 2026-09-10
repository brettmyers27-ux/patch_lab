"""PyInstaller runtime hook for platform-neutral packaged runtime paths."""

import os
import sys
from pathlib import Path

os.environ["PATCHLAB_DISTRIBUTION_MODE"] = "1"
os.environ["PATCHLAB_PACKAGED_INSTALLER"] = "1"
# The old Finder launcher supplied this non-secret endpoint. Frozen releases
# need the same configuration so the existing first-run access gate and its
# already-consented contribution workflow behave exactly as they do today.
os.environ.setdefault(
    "PATCHLAB_RELAY_URL",
    "https://patchlab-relay-482507024870.us-central1.run.app",
)
runtime_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
model_dir = runtime_root / "data" / "models"
os.environ.setdefault(
    "PATCHLAB_MODEL_CACHE",
    str(model_dir / "huggingface"),
)
os.environ.setdefault(
    "PATCHLAB_CLAP_CHECKPOINT",
    str(model_dir / "patchlab_clap_ft_v1.pt"),
)

# A flat PKG installs this bundle in /Applications. Retain the bundle path for
# diagnostics and prevent source-checkout update code from assuming that the
# packaged app has an adjacent install.sh checkout.
executable = Path(sys.executable).resolve()
if executable.parent.name == "MacOS" and executable.parent.parent.name == "Contents":
    os.environ.setdefault("PATCHLAB_APP_BUNDLE", str(executable.parent.parent.parent))
