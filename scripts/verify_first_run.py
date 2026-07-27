#!/usr/bin/env python3
"""Automated clean-profile passcode/terms/sign-out verification."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.access_gate import AccessManager, AccessStore
from core.privacy import PrivacyStore


class MemoryKeyring:
    def __init__(self) -> None:
        self.value: str | None = None

    def get_password(self, _service: str, _account: str) -> str | None:
        return self.value

    def set_password(self, _service: str, _account: str, value: str) -> None:
        self.value = value

    def delete_password(self, _service: str, _account: str) -> None:
        self.value = None


def profile_case(root: Path, accept: bool) -> dict:
    keyring = MemoryKeyring()
    access_store = AccessStore(
        marker_path=root / "access-state.json", keyring_backend=keyring
    )
    privacy = PrivacyStore(root / "privacy-settings.json")
    manager = AccessManager(
        access_store,
        relay_url="http://local-test-relay",
        validator=lambda _url, password: (
            "test-token"
            if password == "group-passcode"
            else (_ for _ in ()).throw(ValueError("wrong passcode"))
        ),
    )
    first_passcode = manager.needs_prompt()
    first_terms = privacy.load().use_and_share_own_presets is None
    wrong_ok, _wrong_message, _ = manager.authenticate("wrong")
    success, _message, _ = manager.authenticate("group-passcode")
    privacy.save(accept)
    second_passcode = manager.needs_prompt()
    second_terms = privacy.load().use_and_share_own_presets is None
    persisted_choice = privacy.load().use_and_share_own_presets
    access_store.clear()
    signout_prompts = manager.needs_prompt()
    signout_terms = privacy.load().use_and_share_own_presets
    return {
        "first_passcode_prompt": first_passcode,
        "first_terms_prompt": first_terms,
        "wrong_passcode_rejected": not wrong_ok,
        "success": success,
        "second_passcode_prompt": second_passcode,
        "second_terms_prompt": second_terms,
        "persisted_choice": persisted_choice,
        "signout_prompts_again": signout_prompts,
        "signout_preserved_terms": signout_terms,
    }


def main() -> int:
    real_privacy = PrivacyStore().path
    real_before = (
        (real_privacy.stat().st_mtime_ns, real_privacy.stat().st_size)
        if real_privacy.exists()
        else None
    )
    with tempfile.TemporaryDirectory(prefix="patchlab-clean-profile-") as directory:
        root = Path(directory)
        accepted = profile_case(root / "accepted", True)
        declined = profile_case(root / "declined", False)
    real_after = (
        (real_privacy.stat().st_mtime_ns, real_privacy.stat().st_size)
        if real_privacy.exists()
        else None
    )
    payload = {
        "accepted": accepted,
        "declined": declined,
        "real_profile_untouched": real_before == real_after,
        "decline_disables_upload": declined["persisted_choice"] is False,
        "accept_enables_upload": accepted["persisted_choice"] is True,
    }
    payload["gate_pass"] = all(
        (
            accepted["first_passcode_prompt"],
            accepted["first_terms_prompt"],
            accepted["wrong_passcode_rejected"],
            accepted["success"],
            not accepted["second_passcode_prompt"],
            not accepted["second_terms_prompt"],
            accepted["signout_prompts_again"],
            accepted["signout_preserved_terms"] is True,
            declined["persisted_choice"] is False,
            declined["signout_preserved_terms"] is False,
            payload["real_profile_untouched"],
        )
    )
    print("FIRST_RUN_GATE=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
