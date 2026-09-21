"""Test isolation: the suite must never read or write a real user's PatchLab data.

``core.platform_env.ENV`` resolves ``app_data_dir`` once, at import time, from
``PATCHLAB_APP_DATA``. Setting it here -- conftest is imported before any test
module -- points *every* default path (library.db, privacy settings, storage
settings, diagnostics, notice state) at a throw-away directory.

Why this matters, learned the hard way: once the library schema gained columns,
every test that constructed a real ``MainWindow`` (which opens the default
library) silently migrated the developer's live ``Patch Lab/library.db`` in
place, and the flight recorder appended test events to the developer's real
diagnostics folder -- which would then have shipped inside their next support
bundle.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

ISOLATED_APP_DATA = tempfile.mkdtemp(prefix="patchlab-test-appdata-")
os.environ["PATCHLAB_APP_DATA"] = ISOLATED_APP_DATA
atexit.register(shutil.rmtree, ISOLATED_APP_DATA, ignore_errors=True)
