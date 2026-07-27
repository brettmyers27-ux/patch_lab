"""PyInstaller runtime hook: packaged builds always use distribution behavior."""

import os

os.environ["PATCHLAB_DISTRIBUTION_MODE"] = "1"
