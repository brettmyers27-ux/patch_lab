from __future__ import annotations

import os
import tempfile
import unittest
import urllib.error
from datetime import datetime
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
    def test_launch_validation_keeps_the_fresh_token_for_background_workers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AccessStore(
                marker_path=Path(directory) / "access.json", keyring_backend=MemoryKeyring()
            )
            AccessManager(
                store, relay_url="https://relay.invalid", validator=lambda _u, _p: "first-token"
            ).authenticate("group-passcode")
            self.assertEqual(store.load().token, "first-token")
            manager = AccessManager(
                store, relay_url="https://relay.invalid", validator=lambda _u, _p: "second-token"
            )
            self.assertFalse(manager.needs_prompt())
            self.assertEqual(store.load().token, "second-token")
            self.assertTrue(store.load().authenticated_once)

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
            self.assertTrue(store.needs_license_agreement())
            accepted = store.accept_license()
            self.assertTrue(accepted.agreed_to_license)
            self.assertIsNotNone(accepted.license_accepted_at)
            datetime.fromisoformat(str(accepted.license_accepted_at))
            self.assertTrue(manager.needs_prompt())
            self.assertIsNone(privacy.load().use_and_share_own_presets)
            ok, _message, _offline = manager.authenticate("correct")
            self.assertTrue(ok)
            privacy.save(True)
            self.assertFalse(manager.needs_prompt())
            self.assertEqual(keyring.get_password(SERVICE, ACCOUNT), "correct")
            store.clear()
            self.assertTrue(manager.needs_prompt())
            self.assertFalse(store.needs_license_agreement())
            self.assertIsNotNone(store.load().license_accepted_at)
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


    def test_rotated_passcode_locks_out_a_previously_authenticated_machine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keyring = MemoryKeyring()
            store = AccessStore(
                marker_path=root / "access.json", keyring_backend=keyring
            )
            accepted = {"value": "old-passcode"}

            def validator(_url: str, value: str) -> str:
                if value == accepted["value"]:
                    return "token"
                raise urllib.error.HTTPError("", 401, "", {}, None)

            manager = AccessManager(
                store, relay_url="http://relay.invalid", validator=validator
            )
            ok, _message, _offline = manager.authenticate("old-passcode")
            self.assertTrue(ok)
            self.assertFalse(manager.needs_prompt())

            accepted["value"] = "new-passcode"
            self.assertTrue(manager.needs_prompt())
            self.assertIsNone(keyring.get_password(SERVICE, ACCOUNT))
            self.assertFalse(store.load().authenticated_once)

            ok, _message, _offline = manager.authenticate("new-passcode")
            self.assertTrue(ok)
            self.assertFalse(manager.needs_prompt())

    def test_unreachable_relay_does_not_lock_out_an_authenticated_machine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = AccessStore(
                marker_path=root / "access.json", keyring_backend=MemoryKeyring()
            )
            manager = AccessManager(
                store, relay_url="http://relay.invalid", validator=lambda _u, _v: "token"
            )
            ok, _message, _offline = manager.authenticate("correct")
            self.assertTrue(ok)

            def offline_validator(_url: str, _value: str) -> str:
                raise OSError("network unreachable")

            manager.validator = offline_validator
            self.assertFalse(manager.needs_prompt())
            self.assertTrue(store.load().authenticated_once)


class DisableKeychainEnvTest(unittest.TestCase):
    """PATCHLAB_DISABLE_KEYCHAIN: how the whole suite avoids the real login keychain.

    A frozen PatchLab is ad-hoc signed, so every rebuild gets a new code
    identity and macOS's "Always Allow" cannot survive that -- an automated
    workflow that rebuilds and relaunches PatchLab repeatedly would otherwise
    hit a real authorization prompt every run. This is set for the whole
    suite in conftest.py; these tests pin down exactly what it does.
    """

    def setUp(self) -> None:
        self._had = os.environ.get("PATCHLAB_DISABLE_KEYCHAIN")

    def tearDown(self) -> None:
        if self._had is None:
            os.environ.pop("PATCHLAB_DISABLE_KEYCHAIN", None)
        else:
            os.environ["PATCHLAB_DISABLE_KEYCHAIN"] = self._had

    def test_the_real_keyring_module_is_never_imported_when_set(self) -> None:
        os.environ["PATCHLAB_DISABLE_KEYCHAIN"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            store = AccessStore(marker_path=Path(directory) / "access.json")
        self.assertIsNone(store.keyring)

    def test_a_disabled_keychain_reads_back_as_no_stored_passcode(self) -> None:
        os.environ["PATCHLAB_DISABLE_KEYCHAIN"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            store = AccessStore(marker_path=Path(directory) / "access.json")
            self.assertIsNone(store.passcode())
            # Writing is a no-op too, never an exception, never a real touch.
            store.save_success("a-passcode", "a-token")
            self.assertIsNone(store.passcode())
            self.assertEqual(store.load().token, "a-token")

    def test_unset_still_behaves_like_before(self) -> None:
        os.environ.pop("PATCHLAB_DISABLE_KEYCHAIN", None)
        with tempfile.TemporaryDirectory() as directory:
            store = AccessStore(marker_path=Path(directory) / "access.json")
        # The real keyring module may or may not be installed in this
        # environment; either way it must have actually been *attempted*,
        # not silently skipped, when the override is absent.
        try:
            import keyring as real_keyring
        except Exception:
            real_keyring = None
        self.assertIs(store.keyring, real_keyring)

    def test_an_explicit_backend_still_wins_over_the_env_var(self) -> None:
        os.environ["PATCHLAB_DISABLE_KEYCHAIN"] = "1"
        fake = MemoryKeyring()
        with tempfile.TemporaryDirectory() as directory:
            store = AccessStore(marker_path=Path(directory) / "access.json", keyring_backend=fake)
        self.assertIs(store.keyring, fake)

    def test_stored_passcode_never_blocks_when_disabled(self) -> None:
        from core.access_gate import stored_passcode

        os.environ["PATCHLAB_DISABLE_KEYCHAIN"] = "1"
        self.assertIsNone(stored_passcode(timeout=5.0))


if __name__ == "__main__":
    unittest.main()
