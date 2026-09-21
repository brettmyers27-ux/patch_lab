"""Persisted, reversible consent for using the user's own presets.

One field, ``use_and_share_own_presets``, is the only authority.  ON permits
both local use (scanning, analysing, rendering, matching) of the user's own
presets and contributing them; OFF permits neither.  Factory presets never
depend on it.  ``user_presets_enabled()`` fails closed: anything other than an
explicit ``true`` on disk -- no file, no answer yet, a damaged file -- is OFF.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from core.platform_env import ENV


def distribution_mode() -> bool:
    return os.environ.get("PATCHLAB_DISTRIBUTION_MODE", "0").strip() == "1"


@dataclass(frozen=True, slots=True)
class PrivacyChoice:
    use_and_share_own_presets: bool | None
    linked_folder: str | None = None


class PrivacyStore:
    def __init__(self, path: Path | None = None) -> None:
        override = os.environ.get("PATCHLAB_PRIVACY_SETTINGS")
        self.path = Path(
            override
            if override
            else path or ENV.app_data_dir / "privacy-settings.json"
        ).expanduser().resolve()

    def load(self) -> PrivacyChoice:
        if not self.path.is_file():
            return PrivacyChoice(None, None)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return PrivacyChoice(None, None)
        value = raw.get("use_and_share_own_presets")
        choice = value if isinstance(value, bool) else None
        linked = raw.get("linked_folder")
        return PrivacyChoice(choice, str(linked) if linked else None)

    def save(
        self, use_and_share: bool, *, linked_folder: Path | str | None = None
    ) -> PrivacyChoice:
        current = self.load()
        linked = str(linked_folder) if linked_folder is not None else current.linked_folder
        choice = PrivacyChoice(bool(use_and_share), linked)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "use_and_share_own_presets": choice.use_and_share_own_presets,
                    "linked_folder": choice.linked_folder,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return choice


class UserPresetsDisabled(RuntimeError):
    """The user has not (or no longer) allowed PatchLab to use their own presets."""

    def __init__(self, stage: str = "") -> None:
        super().__init__(
            "Personal presets are turned off"
            + (f" (stopped before: {stage})" if stage else "")
        )
        self.stage = stage


_ACTIVE_STORE: PrivacyStore | None = None


def set_active_store(store: PrivacyStore | None) -> None:
    """Make the window's own store authoritative inside this process.

    The GUI may be handed a store at a non-default path; anything running in the
    same process must then read the same file the user's choice was written to.
    Worker processes read the default (or ``PATCHLAB_PRIVACY_SETTINGS``) path.
    """

    global _ACTIVE_STORE
    _ACTIVE_STORE = store


def user_presets_enabled() -> bool:
    """True only when the user explicitly allowed use of their own presets.

    Read from disk on every call, so a choice withdrawn while a job is running
    is honoured at that job's next boundary rather than at its next launch.
    Developer builds do not apply distribution consent, matching the UI.
    """

    if not distribution_mode():
        return True
    try:
        store = _ACTIVE_STORE or PrivacyStore()
        return store.load().use_and_share_own_presets is True
    except Exception:
        return False


def require_user_presets(stage: str = "") -> None:
    """Raise :class:`UserPresetsDisabled` unless the user's presets are allowed."""

    if not user_presets_enabled():
        raise UserPresetsDisabled(stage)
