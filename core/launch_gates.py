"""Ordered distribution-only first-launch gates."""

from __future__ import annotations

from collections.abc import Callable

from core.access_gate import AccessManager, AccessStore


LicensePrompt = Callable[[AccessStore], bool]
PasscodePrompt = Callable[[AccessManager], bool]


def run_distribution_gates(
    manager: AccessManager,
    *,
    license_prompt: LicensePrompt,
    passcode_prompt: PasscodePrompt,
) -> bool:
    """Require legal acceptance before attempting private-group authentication."""

    if manager.store.needs_license_agreement():
        if not license_prompt(manager.store):
            return False
        if manager.store.needs_license_agreement():
            return False
    if manager.needs_prompt() and not passcode_prompt(manager):
        return False
    return True
