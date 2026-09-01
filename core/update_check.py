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
from pathlib import Path

from core.platform_env import ENV, PlatformEnv


SETTINGS_FILENAME = "update-preferences.json"
DEFAULT_VERSION_URL = (
    "https://raw.githubusercontent.com/brettmyers27-ux/patch_lab/main/"
    "app/__version__.py"
)
_VERSION_PATTERN = re.compile(r'__version__\s*=\s*"([0-9]+(?:\.[0-9]+)*)"')


@dataclass(frozen=True, slots=True)
class UpdatePreferences:
    auto_check: bool = True
    skipped_version: str | None = None


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


def update_available(current: str, remote: str | None) -> bool:
    if not remote:
        return False
    return parse_version(remote) > parse_version(current)
