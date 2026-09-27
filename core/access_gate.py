"""PatchLab-owned device sessions; never touch OS credential stores."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.platform_env import ENV
from core.relay_client import RelayClient

DEVICE_SESSION_FILENAME = "device-session.json"
WORKER_ACCESS_TOKEN_ENV = "PATCHLAB_WORKER_ACCESS_TOKEN"
_access_token: str | None = None
_token_owner: Path | None = None


@dataclass(frozen=True, slots=True)
class AccessState:
    authenticated_once: bool = False
    token: str | None = None
    local_only: bool = False
    agreed_to_license: bool = False
    license_accepted_at: str | None = None
    device_credential: str | None = None
    device_expires_at: int | None = None


class AccessStore:
    def __init__(self, *, marker_path: Path | None = None) -> None:
        override = os.environ.get("PATCHLAB_ACCESS_STATE")
        self.marker_path = Path(override or marker_path or (ENV.app_data_dir / "access-state.json")).expanduser().resolve()
        self.device_path = self.marker_path.with_name(DEVICE_SESSION_FILENAME)

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def load(self) -> AccessState:
        marker = self._read(self.marker_path)
        device = self._read(self.device_path)
        credential, expires = device.get("device_credential"), device.get("expires_at")
        valid = isinstance(credential, str) and bool(credential) and isinstance(expires, int)
        return AccessState(
            authenticated_once=valid,
            token=_access_token if valid and _token_owner == self.device_path else None,
            local_only=bool(marker.get("local_only")),
            agreed_to_license=bool(marker.get("agreed_to_license")),
            license_accepted_at=str(marker["license_accepted_at"]) if marker.get("license_accepted_at") else None,
            device_credential=credential if valid else None,
            device_expires_at=expires if valid else None,
        )

    def needs_license_agreement(self) -> bool:
        state = self.load()
        return not state.agreed_to_license or not state.license_accepted_at

    def accept_license(self, *, accepted_at: str | None = None) -> AccessState:
        marker = self._read(self.marker_path)
        marker["agreed_to_license"] = True
        marker["license_accepted_at"] = accepted_at or datetime.now(timezone.utc).isoformat()
        self._write(self.marker_path, marker)
        return self.load()

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name == "posix":
                temporary.chmod(0o600)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def save_device(self, credential: str, expires_at: int, access_token: str) -> None:
        global _access_token, _token_owner
        if not credential or not access_token or expires_at <= 0:
            raise ValueError("invalid device session")
        self._write(self.device_path, {"device_credential": credential, "expires_at": expires_at})
        marker = self._read(self.marker_path)
        marker.pop("token", None)
        marker.pop("authenticated_once", None)
        marker["local_only"] = False
        self._write(self.marker_path, marker)
        _access_token = access_token
        _token_owner = self.device_path

    def refresh_token(self, token: str) -> None:
        global _access_token, _token_owner
        if self.load().authenticated_once:
            _access_token = token
            _token_owner = self.device_path

    def save_local_only(self) -> None:
        marker = self._read(self.marker_path)
        marker["local_only"] = True
        self._write(self.marker_path, marker)

    def clear(self) -> None:
        global _access_token, _token_owner
        _access_token = None
        _token_owner = None
        self.device_path.unlink(missing_ok=True)
        marker = self._read(self.marker_path)
        marker.pop("token", None)
        marker.pop("authenticated_once", None)
        marker["local_only"] = False
        self._write(self.marker_path, marker)

    def passcode(self) -> None:
        return None


class AccessManager:
    def __init__(self, store: AccessStore | None = None, *, relay_url: str | None = None,
                 validator: Callable[[str, str], dict[str, Any]] | None = None,
                 refresher: Callable[[str, str], dict[str, Any]] | None = None) -> None:
        self.store = store or AccessStore()
        self.relay_url = relay_url if relay_url is not None else os.environ.get("PATCHLAB_RELAY_URL", "").strip()
        self.validator = validator or self._register
        self.refresher = refresher or self._refresh
        self.prompt_message = ""

    def needs_prompt(self) -> bool:
        state = self.store.load()
        if state.local_only:
            os.environ["PATCHLAB_DISABLE_RELAY"] = "1"
            return False
        if not state.device_credential:
            return True
        if state.device_expires_at is not None and state.device_expires_at < int(datetime.now(timezone.utc).timestamp()):
            self.store.clear()
            self.prompt_message = "Your PatchLab session has expired. Enter the beta passcode to continue."
            return True
        if not self.relay_url:
            return False
        self.access_token()
        return not self.store.load().authenticated_once

    def access_token(self, *, force_refresh: bool = False) -> str | None:
        state = self.store.load()
        if not state.device_credential:
            return None
        if state.token and not force_refresh:
            try:
                if int(state.token.split(".", 1)[0]) > int(datetime.now(timezone.utc).timestamp()) + 60:
                    return state.token
            except (ValueError, IndexError):
                pass
        if not self.relay_url:
            return None
        try:
            response = self.refresher(self.relay_url, state.device_credential)
            token = str(response["access_token"])
            self.store.refresh_token(token)
            return token
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                self.store.clear()
                self.prompt_message = "Your PatchLab session has expired. Enter the beta passcode to continue."
            return None
        except Exception:
            # Connectivity loss cannot revoke an otherwise valid session.
            return None

    def authenticate(self, passcode: str) -> tuple[bool, str, bool]:
        if not passcode:
            return False, "Enter the group passcode and try again.", False
        if not self.relay_url:
            return False, "The private sharing service is not configured.", True
        try:
            response = self.validator(self.relay_url, passcode)
            self.store.save_device(str(response["device_credential"]), int(response["expires_at"]),
                                   str(response["access_token"]))
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return False, "That passcode was not accepted. Please try again.", False
            return False, "The sharing service is unavailable. Try again later.", True
        except (OSError, TimeoutError, urllib.error.URLError):
            return False, "The sharing service is unavailable. Try again later.", True
        except Exception:
            return False, "Could not establish a PatchLab session. Try again later.", True
        os.environ.pop("PATCHLAB_DISABLE_RELAY", None)
        return True, "Passcode accepted. This device is signed in to PatchLab.", False

    def continue_locally(self) -> None:
        self.store.save_local_only()
        os.environ["PATCHLAB_DISABLE_RELAY"] = "1"

    @staticmethod
    def _register(url: str, passcode: str) -> dict[str, Any]:
        return RelayClient(url, "", timeout=10.0).register_device(passcode)

    @staticmethod
    def _refresh(url: str, credential: str) -> dict[str, Any]:
        return RelayClient(url, "", timeout=8.0).refresh_device(credential)


def ensure_relay_token(*, store: AccessStore | None = None, timeout: float = 8.0) -> bool:
    return bool(AccessManager(store=store).access_token())


def stored_passcode(timeout: float = 0.0) -> None:
    return None


def stored_token() -> str | None:
    return os.environ.get(WORKER_ACCESS_TOKEN_ENV) or _access_token


def stored_relay_credential() -> tuple[None, str | None]:
    return None, stored_token()
