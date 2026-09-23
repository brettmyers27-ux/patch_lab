#!/usr/bin/env python3
"""Check whether a real, checksum-verified PatchLab update is available.

A tester on 1.5.5 hit a real network timeout reaching the private release
catalog and PatchLab reported "no update available" -- indistinguishable
from actually being current. This always resolves to one of four states
(``core.update_check.UpdateCheckState``): CURRENT, UPDATE_AVAILABLE,
CHECK_FAILED, or AUTH_REQUIRED. A manual check must never fail silently.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.__version__ import __version__ as CURRENT_VERSION  # noqa: E402
from core.update_check import (  # noqa: E402
    UpdateCheckOutcome,
    classify_update_check_exception,
    fetch_remote_version,
    newest_macos_package,
    update_available,
)


#: Bounded so a manual click never sits apparently frozen. Two network round
#: trips (token mint, then the artifact list) can happen per attempt, so the
#: worst single attempt is ~2x this, well short of anything a user reads as
#: "frozen".
RELAY_TIMEOUT_S = 8.0
#: A transient network/timeout failure gets exactly one retry -- never a
#: loop. An auth failure is not retried: a fresh attempt cannot fix it.
MAX_ATTEMPTS = 2


def _record_check_failure(
    exc: BaseException, *, category: str, attempt: int, elapsed_s: float, operation: str
) -> None:
    try:
        from core.diagnostics import record_failure

        record_failure(
            "update-check",
            "update_check_failed",
            exc,
            message=f"update check ({operation}) failed on attempt {attempt}: {category}",
            phase=operation,
            failure_category=category,
            attempt=attempt,
            elapsed_s=round(elapsed_s, 3),
        )
    except Exception:
        pass


def _private_package_result() -> UpdateCheckOutcome:
    """Return an update only when its private PKG exists in the relay catalog."""

    from core.access_gate import stored_relay_credential
    from core.relay_client import RelayClient

    current = CURRENT_VERSION
    url = os.environ.get("PATCHLAB_RELAY_URL", "").strip()
    if os.environ.get("PATCHLAB_DISABLE_RELAY", "").strip() == "1" or not url:
        # The private update mechanism is intentionally off for this build or
        # environment -- not a failure, so there is nothing to report.
        return UpdateCheckOutcome.current(current)
    password, token = stored_relay_credential()
    if not password and not token:
        return UpdateCheckOutcome.auth_required(current)

    last_category = "unknown"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            client = RelayClient(url, password or "", timeout=RELAY_TIMEOUT_S, token=token)
            rows = client.artifact_manifest()
            release = newest_macos_package(rows, current)
        except Exception as exc:
            elapsed = time.monotonic() - started
            category = classify_update_check_exception(exc)
            print(
                f"UPDATE_CHECK_NOTE=Private release catalog unavailable "
                f"(attempt {attempt}/{MAX_ATTEMPTS}, {category}): "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            _record_check_failure(
                exc, category=category, attempt=attempt, elapsed_s=elapsed,
                operation="artifact_manifest",
            )
            last_category = category
            if category == "auth":
                return UpdateCheckOutcome.auth_required(current)
            # Only a transient network/timeout failure earns the one retry.
            if attempt < MAX_ATTEMPTS and category in ("timeout", "network"):
                continue
            return UpdateCheckOutcome.failed(current, category=category)
        else:
            if release is None:
                return UpdateCheckOutcome.current(current)
            return UpdateCheckOutcome.available(
                current, release.version,
                package={
                    "name": release.name, "version": release.version,
                    "size": release.size, "sha256": release.sha256,
                },
            )
    return UpdateCheckOutcome.failed(current, category=last_category)


def _public_source_result() -> UpdateCheckOutcome:
    """The unauthenticated GitHub-source check used outside distribution mode."""

    remote = fetch_remote_version()
    if remote is None:
        # fetch_remote_version() never raises -- it already swallows network
        # detail. Treated as a failure so a manual check still says so,
        # rather than silently agreeing with "you're current".
        return UpdateCheckOutcome.failed(CURRENT_VERSION, category="network")
    if update_available(CURRENT_VERSION, remote):
        return UpdateCheckOutcome.available(CURRENT_VERSION, remote)
    return UpdateCheckOutcome.current(CURRENT_VERSION)


def main() -> int:
    if os.environ.get("PATCHLAB_PACKAGED_INSTALLER") == "1":
        outcome = _private_package_result()
    else:
        outcome = _public_source_result()
    print("UPDATE_CHECK_RESULT=" + json.dumps(outcome.as_dict(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
