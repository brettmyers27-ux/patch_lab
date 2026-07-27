"""Persisted, reversible consent for local linking plus contribution upload."""

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
