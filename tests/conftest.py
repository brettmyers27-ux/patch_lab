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

``PATCHLAB_DISABLE_KEYCHAIN`` gets the same treatment for the same reason: a
frozen PatchLab is ad-hoc signed, so every rebuild has a new code identity,
and macOS's Keychain "Always Allow" cannot survive that. Any test that
constructs a real ``AccessStore``/``AccessManager`` (directly, or via
``core.access_gate.stored_passcode``) would otherwise trigger a real
authorization prompt on this machine every time the suite runs after a
rebuild. A test that specifically needs to exercise the real keychain
integration injects its own ``keyring_backend`` (see ``AccessStore``), which
this does not override.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile

ISOLATED_APP_DATA = tempfile.mkdtemp(prefix="patchlab-test-appdata-")
os.environ["PATCHLAB_APP_DATA"] = ISOLATED_APP_DATA
os.environ["PATCHLAB_DISABLE_KEYCHAIN"] = "1"
atexit.register(shutil.rmtree, ISOLATED_APP_DATA, ignore_errors=True)
