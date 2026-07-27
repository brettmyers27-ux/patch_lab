from __future__ import annotations

import os
import tempfile
import unittest
import urllib.error
from pathlib import Path

from core.access_gate import ACCOUNT, SERVICE, AccessManager, AccessStore
from core.privacy import PrivacyStore


class MemoryKeyring:
    def __init__(self, *, fail: bool = False) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.fail = fail

    def get_password(self, service: str, account: str) -> str | None:
        if self.fail:
            raise RuntimeError("keychain unavailable")
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        if self.fail:
            raise RuntimeError("keychain unavailable")
        self.values[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


class AccessGateTest(unittest.TestCase):
    def test_first_success_second_skips_and_signout_preserves_terms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keyring = MemoryKeyring()
            store = AccessStore(
                marker_path=root / "access.json", keyring_backend=keyring
            )
            manager = AccessManager(
                store,
                relay_url="http://relay.invalid",
                validator=lambda _url, value: (
                    "token" if value == "correct" else (_ for _ in ()).throw(
                        urllib.error.HTTPError("", 401, "", {}, None)
                    )
                ),
            )
            privacy = PrivacyStore(root / "privacy.json")
            self.assertTrue(manager.needs_prompt())
            self.assertIsNone(privacy.load().use_and_share_own_presets)
            ok, _message, _offline = manager.authenticate("correct")
            self.assertTrue(ok)
            privacy.save(True)
            self.assertFalse(manager.needs_prompt())
            self.assertEqual(keyring.get_password(SERVICE, ACCOUNT), "correct")
            store.clear()
            self.assertTrue(manager.needs_prompt())
            self.assertTrue(privacy.load().use_and_share_own_presets)

    def test_wrong_unreachable_local_and_keychain_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = AccessStore(
                marker_path=root / "fallback.json",
                keyring_backend=MemoryKeyring(fail=True),
            )
            manager = AccessManager(
                fallback,
                relay_url="http://relay.invalid",
                validator=lambda _url, _value: "token",
            )
            ok, message, _offline = manager.authenticate("correct")
            self.assertTrue(ok)
            self.assertIn("keychain was unavailable", message)
            self.assertFalse(manager.needs_prompt())
            self.assertNotIn("correct", fallback.marker_path.read_text())

            local_store = AccessStore(
                marker_path=root / "local.json", keyring_backend=MemoryKeyring()
            )
            local = AccessManager(local_store, relay_url="")
            ok, _message, offline = local.authenticate("anything")
            self.assertFalse(ok)
            self.assertTrue(offline)
            local.continue_locally()
            self.assertFalse(local.needs_prompt())
            self.assertEqual(os.environ.get("PATCHLAB_DISABLE_RELAY"), "1")
            os.environ.pop("PATCHLAB_DISABLE_RELAY", None)
            restarted = AccessManager(local_store, relay_url="http://relay.invalid")
            self.assertFalse(restarted.needs_prompt())
            self.assertEqual(os.environ.get("PATCHLAB_DISABLE_RELAY"), "1")
            os.environ.pop("PATCHLAB_DISABLE_RELAY", None)


if __name__ == "__main__":
    unittest.main()
