"""Console-free Windows launcher used by the installed shortcuts."""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / ".patchlab-launcher.json"


def main() -> None:
    # Windows PowerShell 5.1 writes UTF-8 with a BOM; utf-8-sig accepts both
    # that form and PowerShell 7's BOM-free UTF-8.
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    os.environ["PATCHLAB_DISTRIBUTION_MODE"] = "1"
    os.environ["PATCHLAB_RELAY_URL"] = str(config["relay_url"])
    os.environ["PATCHLAB_MODEL_CACHE"] = str(config["model_cache"])
    os.chdir(PROJECT_ROOT)
    runpy.run_path(str(PROJECT_ROOT / "app" / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
