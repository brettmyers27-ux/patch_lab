"""Distribution-only first-run relay access gate and secure credential storage."""

from __future__ import annotations

import json
import os
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from core.platform_env import ENV
from core.relay_client import RelayClient


SERVICE = "com.patchlab.desktop"
ACCOUNT = "private-group-passcode"


@dataclass(frozen=True, slots=True)
class AccessState:
    authenticated_once: bool = False
    token: str | None = None
    local_only: bool = False
    agreed_to_license: bool = False
    license_accepted_at: str | None = None


class AccessStore:
    def __init__(self, *, marker_path: Path | None = None, keyring_backend=None) -> None:
        override = os.environ.get("PATCHLAB_ACCESS_STATE")
        self.marker_path = Path(
            override or marker_path or (ENV.app_data_dir / "access-state.json")
        ).expanduser().resolve()
        if keyring_backend is None:
            try:
                import keyring

                keyring_backend = keyring
            except Exception:
                keyring_backend = None
        self.keyring = keyring_backend

    def load(self) -> AccessState:
        if not self.marker_path.is_file():
            return AccessState()
        try:
            raw = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return AccessState()
        return AccessState(
            bool(raw.get("authenticated_once")),
            str(raw["token"]) if raw.get("token") else None,
            bool(raw.get("local_only")),
            bool(raw.get("agreed_to_license")),
            (
                str(raw["license_accepted_at"])
                if raw.get("license_accepted_at")
                else None
            ),
        )

    def needs_license_agreement(self) -> bool:
        state = self.load()
        return not state.agreed_to_license or not state.license_accepted_at

    def accept_license(self, *, accepted_at: str | None = None) -> AccessState:
        current = self.load()
        timestamp = accepted_at or datetime.now(timezone.utc).isoformat()
        state = AccessState(
            current.authenticated_once,
            current.token,
            current.local_only,
            True,
            timestamp,
        )
        self._write(state)
        return state

    def passcode(self) -> str | None:
        if self.keyring is None:
            return None
        try:
            return self.keyring.get_password(SERVICE, ACCOUNT)
        except Exception:
            return None

    def save_success(self, passcode: str, token: str) -> bool:
        keychain_saved = False
        if self.keyring is not None:
            try:
                self.keyring.set_password(SERVICE, ACCOUNT, passcode)
                keychain_saved = True
            except Exception:
                pass
        current = self.load()
        self._write(
            AccessState(
                True,
                token,
                False,
                current.agreed_to_license,
                current.license_accepted_at,
            )
        )
        return keychain_saved

    def refresh_token(self, token: str) -> None:
        """Keep a freshly validated relay token so background workers can use it."""

        current = self.load()
        if not current.authenticated_once:
            return
        self._write(
            AccessState(
                True,
                token,
                current.local_only,
                current.agreed_to_license,
                current.license_accepted_at,
            )
        )

    def save_local_only(self) -> None:
        current = self.load()
        self._write(
            AccessState(
                False,
                None,
                True,
                current.agreed_to_license,
                current.license_accepted_at,
            )
        )

    def clear(self) -> None:
        if self.keyring is not None:
            try:
                self.keyring.delete_password(SERVICE, ACCOUNT)
            except Exception:
                pass
        current = self.load()
        if current.agreed_to_license and current.license_accepted_at:
            self._write(
                AccessState(
                    False,
                    None,
                    False,
                    True,
                    current.license_accepted_at,
                )
            )
        else:
            self.marker_path.unlink(missing_ok=True)

    def _write(self, state: AccessState) -> None:
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(
            json.dumps(
                {
                    "authenticated_once": state.authenticated_once,
                    "token": state.token,
                    "local_only": state.local_only,
                    "agreed_to_license": state.agreed_to_license,
                    "license_accepted_at": state.license_accepted_at,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


class AccessManager:
    def __init__(
        self,
        store: AccessStore | None = None,
        *,
        relay_url: str | None = None,
        validator: Callable[[str, str], str] | None = None,
    ) -> None:
        self.store = store or AccessStore()
        self.relay_url = (
            relay_url
            if relay_url is not None
            else os.environ.get("PATCHLAB_RELAY_URL", "").strip()
        )
        self.validator = validator or self._validate

    def needs_prompt(self) -> bool:
        state = self.store.load()
        if state.local_only:
            os.environ["PATCHLAB_DISABLE_RELAY"] = "1"
            return False
        if not state.authenticated_once:
            return True
        return self._revoked_since_authentication()

    def _revoked_since_authentication(self) -> bool:
        """Re-check a previously-accepted passcode against the live relay.

        Without this, "authenticated once" means "trusted forever" on this
        machine, and rotating the group passcode to end a beta would do
        nothing for anyone who already unlocked the app. Only an explicit
        401/403 from the relay counts as revoked; an unreachable relay or a
        missing local passcode fails open so offline use still works.
        """

        if not self.relay_url:
            return False
        passcode = self.store.passcode()
        if not passcode:
            return False
        try:
            token = self.validator(self.relay_url, passcode)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                self.store.clear()
                return True
            return False
        except Exception:
            return False
        try:
            self.store.refresh_token(str(token))
        except Exception:
            pass
        return False

    def authenticate(self, passcode: str) -> tuple[bool, str, bool]:
        if not passcode:
            return False, "Enter the group passcode and try again.", False
        if not self.relay_url:
            return False, "The private sharing service is not configured.", True
        try:
            token = self.validator(self.relay_url, passcode)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return False, "That passcode was not accepted. Please try again.", False
            return False, f"The sharing service returned HTTP {exc.code}.", True
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            return False, f"The sharing service is unavailable ({exc}).", True
        except Exception as exc:
            return False, f"Could not contact the sharing service ({exc}).", True
        keychain_saved = self.store.save_success(passcode, token)
        # A user who earlier chose "continue without sharing" is signing in on
        # purpose now; the rest of this session (and its workers) must see that.
        os.environ.pop("PATCHLAB_DISABLE_RELAY", None)
        return (
            True,
            "Passcode accepted and saved securely."
            if keychain_saved
            else "Passcode accepted. The keychain was unavailable, so only a non-secret success marker and relay token were retained.",
            False,
        )

    def continue_locally(self) -> None:
        self.store.save_local_only()
        os.environ["PATCHLAB_DISABLE_RELAY"] = "1"

    @staticmethod
    def _validate(url: str, passcode: str) -> str:
        return RelayClient(url, passcode, timeout=10.0).token()


#: How long a background process will wait for the keychain before giving up.
#: A first read by a newly-signed app can sit behind an "allow access" prompt
#: that nobody can see from a worker, so the wait must be bounded.
KEYCHAIN_TIMEOUT_S = 15.0


def ensure_relay_token(
    *, store: AccessStore | None = None, timeout: float = KEYCHAIN_TIMEOUT_S
) -> bool:
    """Make sure a usable support token exists, using the passcode already saved.

    PatchLab's own sign-in and its support-service sign-in are the same passcode,
    so a member who unlocked the app should never be told later that PatchLab is
    "not signed in to the support service". That happened because the token is
    what background workers use, older versions stored only the passcode, and a
    keychain read from a worker can be refused or delayed.

    Called once at launch, off the critical path: it mints and stores a token
    from the saved passcode, and reports whether one is now available. Never
    raises and never prompts.
    """

    store = store or AccessStore()
    state = store.load()
    if state.local_only or not state.authenticated_once:
        return False
    if state.token:
        return True
    url = os.environ.get("PATCHLAB_RELAY_URL", "").strip()
    passcode = stored_passcode(timeout)
    if not url or not passcode:
        return False
    try:
        token = RelayClient(url, passcode, timeout=15.0).token()
    except Exception:
        return False
    try:
        store.refresh_token(str(token))
    except Exception:
        return False
    return True


def stored_passcode(timeout: float = KEYCHAIN_TIMEOUT_S) -> str | None:
    """Read the saved passcode, but never wait on the keychain indefinitely."""

    import threading
    import time

    box: dict[str, str | None] = {}

    def read() -> None:
        try:
            box["value"] = AccessStore().passcode()
        except Exception:
            box["value"] = None

    started = time.monotonic()
    thread = threading.Thread(target=read, name="keychain-read", daemon=True)
    thread.start()
    thread.join(timeout)
    finished = not thread.is_alive()
    try:
        from core.diagnostics import recorder

        recorder().record(
            "relay-auth",
            "keychain_read",
            "saved passcode read from the keychain"
            if finished
            else "keychain did not answer in time; continuing without it",
            severity="info" if finished else "warning",
            elapsed_s=round(time.monotonic() - started, 3),
            timed_out=not finished,
            found=bool(box.get("value")),
        )
    except Exception:
        pass
    return box.get("value") if finished else None


def stored_token() -> str | None:
    """The cached relay token (a plain file read; never touches the keychain)."""

    try:
        return AccessStore().load().token
    except Exception:
        return None


def stored_relay_credential() -> tuple[str | None, str | None]:
    return stored_passcode(), stored_token()
