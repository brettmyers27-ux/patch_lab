"""Device-session migration, persistence, revocation, and OS-store isolation."""

from __future__ import annotations

import json
import os
import stat
import time
import urllib.error
from pathlib import Path

from core.access_gate import AccessManager, AccessStore, DEVICE_SESSION_FILENAME


def _issued(passcode: str) -> dict:
    if passcode != "beta-secret":
        raise urllib.error.HTTPError("", 401, "", {}, None)
    return {"device_credential": "opaque-device-credential", "expires_at": int(time.time()) + 90 * 86400,
            "access_token": f"{int(time.time()) + 3600}.access"}


def _manager(store: AccessStore, *, refresher=None) -> AccessManager:
    return AccessManager(store, relay_url="https://relay.invalid",
                         validator=lambda _url, password: _issued(password),
                         refresher=refresher or (lambda _url, _credential: {
                             "access_token": f"{int(time.time()) + 3600}.renewed"}))


def test_fresh_auth_relaunch_upgrade_signout_and_no_passcode_on_disk(tmp_path: Path) -> None:
    store = AccessStore(marker_path=tmp_path / "access-state.json")
    manager = _manager(store)
    assert manager.needs_prompt()
    assert not manager.authenticate("wrong")[0]
    assert manager.authenticate("beta-secret")[0]
    assert not manager.needs_prompt()
    assert store.device_path.name == DEVICE_SESSION_FILENAME
    # POSIX mode bits are meaningful on macOS; Windows protects the per-user
    # AppData directory with its profile ACL instead.  Requiring 0600 on NTFS
    # makes a valid Windows device-session run look like a security failure.
    if os.name == "posix":
        assert stat.S_IMODE(store.device_path.stat().st_mode) == 0o600
    assert store.load().device_credential == "opaque-device-credential"
    assert "beta-secret" not in store.device_path.read_text()
    assert "beta-secret" not in store.marker_path.read_text()
    assert "access" not in store.device_path.read_text()
    assert "access_token" not in store.marker_path.read_text()
    store.accept_license()
    # A new AccessStore models a new launch or an app replacement.
    restarted = AccessStore(marker_path=store.marker_path)
    assert not _manager(restarted).needs_prompt()
    assert not _manager(restarted).needs_prompt()
    restarted.clear()
    assert not restarted.device_path.exists()
    assert restarted.needs_license_agreement() is False
    assert _manager(restarted).needs_prompt()


def test_legacy_marker_never_migrates_by_reading_os_credentials(tmp_path: Path) -> None:
    marker = tmp_path / "access-state.json"
    marker.write_text(json.dumps({"authenticated_once": True, "token": "old-token",
                                  "agreed_to_license": True, "license_accepted_at": "2026-01-01"}))
    store = AccessStore(marker_path=marker)
    assert _manager(store).needs_prompt()
    assert store.passcode() is None
    assert _manager(store).authenticate("beta-secret")[0]
    assert "old-token" not in marker.read_text()


def test_invalid_device_session_prompts_but_network_failure_preserves_it(tmp_path: Path) -> None:
    store = AccessStore(marker_path=tmp_path / "access-state.json")
    assert _manager(store).authenticate("beta-secret")[0]
    # Force the access token to be stale without touching the device credential.
    store.refresh_token("1.expired")
    offline = _manager(store, refresher=lambda _u, _c: (_ for _ in ()).throw(OSError("offline")))
    assert not offline.needs_prompt()
    assert store.device_path.exists()
    invalid = _manager(store, refresher=lambda _u, _c: (_ for _ in ()).throw(
        urllib.error.HTTPError("", 401, "", {}, None)))
    assert invalid.needs_prompt()
    assert not store.device_path.exists()


def test_local_only_and_windows_profile_storage_shape(tmp_path: Path, monkeypatch) -> None:
    store = AccessStore(marker_path=tmp_path / "AppData" / "PatchLab" / "access-state.json")
    assert store.device_path.parent == tmp_path / "AppData" / "PatchLab"
    manager = _manager(store)
    manager.continue_locally()
    assert not manager.needs_prompt()
    assert os.environ["PATCHLAB_DISABLE_RELAY"] == "1"
    monkeypatch.delenv("PATCHLAB_DISABLE_RELAY")
    assert manager.authenticate("beta-secret")[0]
    assert "PATCHLAB_DISABLE_RELAY" not in os.environ
