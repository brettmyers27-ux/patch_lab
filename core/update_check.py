"""Update-check preferences and version comparison for the distributed app.

This module only ever decides *whether* a newer version exists. Applying an
update is a separate, explicit step (scripts/apply_update.sh) triggered only
after the user chooses "Update Now" in the dialog this drives -- nothing here
touches the filesystem beyond a tiny preferences file.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from core.platform_env import ENV, PlatformEnv


SETTINGS_FILENAME = "update-preferences.json"
DEFAULT_VERSION_URL = (
    "https://raw.githubusercontent.com/brettmyers27-ux/patch_lab/main/"
    "app/__version__.py"
)
_VERSION_PATTERN = re.compile(r'__version__\s*=\s*"([0-9]+(?:\.[0-9]+)*)"')
_MACOS_PACKAGE_PATTERN = re.compile(
    r"^PatchLab-(?P<version>[0-9]+(?:\.[0-9]+){2})-macOS\.pkg$"
)


@dataclass(frozen=True, slots=True)
class UpdatePreferences:
    auto_check: bool = True
    skipped_version: str | None = None


@dataclass(frozen=True, slots=True)
class MacOSPackageRelease:
    """One signed-by-checksum private macOS installer advertised by the relay."""

    name: str
    version: str
    size: int
    sha256: str


class UpdateCheckState(str, Enum):
    """The four outcomes a version check can actually have.

    A tester on 1.5.5 hit a real ``TimeoutError`` reaching the private
    release catalog, and PatchLab reported ``update_available: false`` --
    indistinguishable from genuinely being current. Whatever produces a
    check result must say which of these four happened, never collapse
    a failure into "no update".
    """

    CURRENT = "current"
    UPDATE_AVAILABLE = "update_available"
    CHECK_FAILED = "check_failed"
    AUTH_REQUIRED = "auth_required"


#: Why a CHECK_FAILED happened, for diagnostics and for picking a message.
#: Never a raw exception class or traceback -- see UpdateCheckOutcome.user_message.
FailureCategory = str  # one of: "", "timeout", "network", "invalid_response", "unknown"

_FAILURE_MESSAGES: dict[str, str] = {
    "timeout": (
        "PatchLab couldn't reach the update service in time. "
        "Check your internet connection and try again."
    ),
    "network": (
        "PatchLab couldn't check for updates right now. "
        "Check your internet connection and try again."
    ),
    "invalid_response": (
        "PatchLab couldn't understand the update service's response. Try again later."
    ),
    "unknown": (
        "PatchLab couldn't check for updates right now. Try again in a moment."
    ),
}
_DEFAULT_CHECK_FAILED_MESSAGE = _FAILURE_MESSAGES["unknown"]
_AUTH_REQUIRED_MESSAGE = (
    "PatchLab needs you to sign in to the support service to check for updates."
)


@dataclass(frozen=True, slots=True)
class UpdateCheckOutcome:
    """One update check's result: exactly one of the four states above.

    ``as_dict()`` is what workers print as ``UPDATE_CHECK_RESULT=``. It keeps
    the historical ``update_available`` boolean for any older reader, but
    ``state`` is authoritative -- a boolean alone can never distinguish
    "current" from "the check failed".
    """

    state: UpdateCheckState
    current_version: str
    remote_version: str | None = None
    package: dict[str, Any] | None = None
    failure_category: FailureCategory = ""
    user_message: str = ""

    @classmethod
    def current(cls, current_version: str) -> "UpdateCheckOutcome":
        return cls(UpdateCheckState.CURRENT, current_version)

    @classmethod
    def available(
        cls, current_version: str, remote_version: str, package: dict[str, Any] | None = None
    ) -> "UpdateCheckOutcome":
        return cls(
            UpdateCheckState.UPDATE_AVAILABLE, current_version,
            remote_version=remote_version, package=package,
        )

    @classmethod
    def failed(
        cls, current_version: str, *, category: FailureCategory = "unknown"
    ) -> "UpdateCheckOutcome":
        message = _FAILURE_MESSAGES.get(category, _DEFAULT_CHECK_FAILED_MESSAGE)
        return cls(
            UpdateCheckState.CHECK_FAILED, current_version,
            failure_category=category or "unknown", user_message=message,
        )

    @classmethod
    def auth_required(cls, current_version: str) -> "UpdateCheckOutcome":
        return cls(
            UpdateCheckState.AUTH_REQUIRED, current_version,
            user_message=_AUTH_REQUIRED_MESSAGE,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": self.state.value,
            "current_version": self.current_version,
            "remote_version": self.remote_version,
            # Historical field: only ever true for UPDATE_AVAILABLE. A reader
            # that still only checks this boolean degrades safely to "no
            # update to offer" for CHECK_FAILED/AUTH_REQUIRED, same as before
            # -- it just no longer *claims* those states mean "current".
            "update_available": self.state is UpdateCheckState.UPDATE_AVAILABLE,
        }
        if self.package is not None:
            payload["package"] = self.package
        if self.failure_category:
            payload["failure_category"] = self.failure_category
        if self.user_message:
            payload["user_message"] = self.user_message
        return payload


def classify_update_check_exception(exc: BaseException) -> FailureCategory:
    """Sort a relay-check exception into one plain, non-secret category.

    Never returns exception text -- ``str(exc)`` for an HTTP or connection
    error can include the request URL, which stays in structured diagnostics
    only (see ``scripts/check_for_update.py``), never in what this returns.
    """

    # A read/connect timeout surfaces either as a bare TimeoutError (Python's
    # http.client raising directly) or wrapped in URLError(reason=timeout()).
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, urllib.error.HTTPError):
        # 401 means the stored passcode/token can no longer authenticate --
        # nothing a retry can fix. Any other status is a service problem.
        return "auth" if exc.code == 401 else "network"
    if isinstance(exc, urllib.error.URLError):
        if isinstance(exc.reason, TimeoutError):
            return "timeout"
        return "network"
    if isinstance(exc, (ConnectionError, OSError)):
        return "network"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "invalid_response"
    return "unknown"


def macos_package_releases(rows: Iterable[dict[str, Any]]) -> list[MacOSPackageRelease]:
    """Return only well-formed private Mac installer records.

    The runtime-artifact catalog is also used by first installation.  An
    explicit ``kind`` prevents a package from ever being mistaken for a model
    download, while the filename/version checks prevent a malformed catalog
    entry from becoming an update prompt.
    """

    releases: list[MacOSPackageRelease] = []
    for row in rows:
        if row.get("kind") != "macos-package":
            continue
        name = str(row.get("name") or "")
        match = _MACOS_PACKAGE_PATTERN.fullmatch(name)
        version = str(row.get("version") or "")
        sha256 = str(row.get("sha256") or "")
        try:
            size = int(row.get("size"))
        except (TypeError, ValueError):
            continue
        if (
            match is None
            or match.group("version") != version
            or size <= 0
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            continue
        releases.append(MacOSPackageRelease(name, version, size, sha256))
    return sorted(releases, key=lambda release: parse_version(release.version))


def newest_macos_package(
    rows: Iterable[dict[str, Any]], current_version: str
) -> MacOSPackageRelease | None:
    """Select the newest private package that is genuinely newer than this app."""

    available = [
        release
        for release in macos_package_releases(rows)
        if update_available(current_version, release.version)
    ]
    return available[-1] if available else None


def settings_path(env: PlatformEnv = ENV) -> Path:
    return Path(env.app_data_dir) / SETTINGS_FILENAME


def load_update_preferences(env: PlatformEnv = ENV) -> UpdatePreferences:
    path = settings_path(env)
    if not path.is_file():
        return UpdatePreferences()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        skipped = raw.get("skipped_version")
        return UpdatePreferences(
            auto_check=bool(raw.get("auto_check", True)),
            skipped_version=str(skipped) if skipped else None,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # A damaged preference must not silently disable update checks.
        return UpdatePreferences()


def save_update_preferences(
    preferences: UpdatePreferences, env: PlatformEnv = ENV
) -> Path:
    path = settings_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(asdict(preferences), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def parse_version(text: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable tuple.

    Falls back to (0,) for anything unparseable so a malformed or truncated
    remote response can never look newer than a real local version.
    """

    try:
        parts = tuple(int(part) for part in text.strip().split("."))
        return parts if parts else (0,)
    except ValueError:
        return (0,)


def extract_version(source_text: str) -> str | None:
    """Pull the version string out of an app/__version__.py file's contents."""

    match = _VERSION_PATTERN.search(source_text)
    return match.group(1) if match else None


def fetch_remote_version(
    url: str = DEFAULT_VERSION_URL, *, timeout: float = 10.0
) -> str | None:
    """Fetch the currently published version string. None on any failure.

    Never raises: an unreachable network, a GitHub outage, or a malformed
    response must never crash or block the app -- it just means no update is
    reported this time, exactly like every other relay-adjacent network call
    in this codebase.
    """

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return extract_version(body)


#: A downloaded installer needs room for itself, for macOS Installer's expanded
#: payload, and for the replacement app to exist beside the current one while
#: the swap happens. Measured against a 3.4 GB package whose payload expands to
#: a ~4.6 GB app bundle: 3.4 (download) + 4.6 (expansion/replacement) is already
#: over 2x, and Installer needs scratch on top.
UPDATE_SPACE_MULTIPLIER = 2.5
#: Absolute headroom so a "just barely fits" disk is still refused.
UPDATE_SPACE_HEADROOM_BYTES = 2 * 1024 ** 3


@dataclass(frozen=True, slots=True)
class UpdateSpaceCheck:
    """Whether this Mac can actually complete a package update."""

    required_bytes: int
    available_bytes: int
    package_bytes: int
    path: str

    @property
    def sufficient(self) -> bool:
        return self.available_bytes >= self.required_bytes

    @staticmethod
    def _gb(value: int) -> str:
        return f"{value / 1e9:.1f} GB"

    def user_message(self) -> str:
        """One sentence naming both numbers, because 'failed' is not actionable."""

        return (
            f"PatchLab needs about {self._gb(self.required_bytes)} free to download "
            f"and install this update, and this disk has {self._gb(self.available_bytes)} "
            f"available. The installer alone is {self._gb(self.package_bytes)}, and macOS "
            "needs room to expand it and replace the current app. Free up some space and "
            "try Check for Updates again."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_bytes": self.required_bytes,
            "available_bytes": self.available_bytes,
            "package_bytes": self.package_bytes,
            "path": self.path,
            "sufficient": self.sufficient,
        }


def update_space_preflight(
    package_bytes: int,
    *,
    destination: Path | None = None,
    env: PlatformEnv = ENV,
) -> UpdateSpaceCheck:
    """Decide BEFORE downloading whether a multi-gigabyte update can complete.

    The reporting tester had ~4.4 GiB free for a 3.4 GB package: enough to store
    the download and nothing else, so an update that started would have failed
    somewhere the user could not interpret.
    """

    import shutil

    target = Path(destination or Path(env.app_data_dir) / "updates")
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        available = int(shutil.disk_usage(probe).free)
    except OSError:
        available = 0
    package_bytes = max(int(package_bytes), 0)
    required = int(package_bytes * UPDATE_SPACE_MULTIPLIER) + UPDATE_SPACE_HEADROOM_BYTES
    return UpdateSpaceCheck(required, available, package_bytes, str(target))


def update_available(current: str, remote: str | None) -> bool:
    if not remote:
        return False
    return parse_version(remote) > parse_version(current)
