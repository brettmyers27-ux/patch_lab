#!/usr/bin/env python3
"""Automated clean-profile license/passcode/sharing/sign-out verification."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.access_gate import AccessManager, AccessStore
from core.launch_gates import run_distribution_gates
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
    first_license = access_store.needs_license_agreement()
    first_passcode = manager.needs_prompt()
    first_terms = privacy.load().use_and_share_own_presets is None
    wrong_ok, _wrong_message, _ = manager.authenticate("wrong")
    events: list[str] = []

    def license_prompt(store: AccessStore) -> bool:
        events.append("license")
        store.accept_license()
        return True

    def passcode_prompt(access: AccessManager) -> bool:
        events.append("passcode")
        success, _message, _offline = access.authenticate("group-passcode")
        return success

    reached_main = run_distribution_gates(
        manager,
        license_prompt=license_prompt,
        passcode_prompt=passcode_prompt,
    )
    accepted_state = access_store.load()
    timestamp_valid = False
    if accepted_state.license_accepted_at:
        try:
            datetime.fromisoformat(accepted_state.license_accepted_at)
            timestamp_valid = True
        except ValueError:
            pass
    privacy.save(accept)
    second_events: list[str] = []
    second_reached_main = run_distribution_gates(
        manager,
        license_prompt=lambda _store: second_events.append("license") or False,
        passcode_prompt=lambda _manager: second_events.append("passcode") or False,
    )
    second_license = access_store.needs_license_agreement()
    second_passcode = manager.needs_prompt()
    second_terms = privacy.load().use_and_share_own_presets is None
    persisted_choice = privacy.load().use_and_share_own_presets
    access_store.clear()
    signout_license = access_store.needs_license_agreement()
    signout_prompts = manager.needs_prompt()
    signout_terms = privacy.load().use_and_share_own_presets
    return {
        "first_license_prompt": first_license,
        "first_passcode_prompt": first_passcode,
        "first_terms_prompt": first_terms,
        "gate_order": events,
        "wrong_passcode_rejected": not wrong_ok,
        "success": reached_main,
        "license_timestamp_valid": timestamp_valid,
        "second_reached_main": second_reached_main,
        "second_gate_prompts": second_events,
        "second_license_prompt": second_license,
        "second_passcode_prompt": second_passcode,
        "second_terms_prompt": second_terms,
        "persisted_choice": persisted_choice,
        "signout_license_prompt": signout_license,
        "signout_prompts_again": signout_prompts,
        "signout_preserved_terms": signout_terms,
    }


def declined_license_case(root: Path) -> dict:
    store = AccessStore(
        marker_path=root / "access-state.json", keyring_backend=MemoryKeyring()
    )
    manager = AccessManager(
        store,
        relay_url="http://local-test-relay",
        validator=lambda _url, _password: "unexpected-token",
    )
    events: list[str] = []
    reached_main = run_distribution_gates(
        manager,
        license_prompt=lambda _store: events.append("license") or False,
        passcode_prompt=lambda _manager: events.append("passcode") or True,
    )
    return {
        "gate_order": events,
        "reached_main": reached_main,
        "license_persisted": not store.needs_license_agreement(),
        "passcode_reached": "passcode" in events,
    }


def _file_state(path: Path) -> tuple[int, int] | None:
    return (
        (path.stat().st_mtime_ns, path.stat().st_size)
        if path.exists()
        else None
    )


def main() -> int:
    real_privacy = PrivacyStore().path
    real_access = AccessStore().marker_path
    real_before = {
        "privacy": _file_state(real_privacy),
        "access": _file_state(real_access),
    }
    with tempfile.TemporaryDirectory(prefix="patchlab-clean-profile-") as directory:
        root = Path(directory)
        accepted = profile_case(root / "accepted", True)
        declined = profile_case(root / "declined", False)
        license_declined = declined_license_case(root / "license-declined")
    real_after = {
        "privacy": _file_state(real_privacy),
        "access": _file_state(real_access),
    }
    payload = {
        "accepted": accepted,
        "declined": declined,
        "license_declined": license_declined,
        "real_profile_untouched": real_before == real_after,
        "decline_disables_upload": declined["persisted_choice"] is False,
        "accept_enables_upload": accepted["persisted_choice"] is True,
    }
    payload["gate_pass"] = all(
        (
            accepted["first_license_prompt"],
            accepted["first_passcode_prompt"],
            accepted["first_terms_prompt"],
            accepted["gate_order"] == ["license", "passcode"],
            accepted["wrong_passcode_rejected"],
            accepted["success"],
            accepted["license_timestamp_valid"],
            accepted["second_reached_main"],
            accepted["second_gate_prompts"] == [],
            not accepted["second_license_prompt"],
            not accepted["second_passcode_prompt"],
            not accepted["second_terms_prompt"],
            not accepted["signout_license_prompt"],
            accepted["signout_prompts_again"],
            accepted["signout_preserved_terms"] is True,
            declined["persisted_choice"] is False,
            declined["signout_preserved_terms"] is False,
            license_declined["gate_order"] == ["license"],
            not license_declined["reached_main"],
            not license_declined["license_persisted"],
            not license_declined["passcode_reached"],
            payload["real_profile_untouched"],
        )
    )
    print("FIRST_RUN_GATE=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
