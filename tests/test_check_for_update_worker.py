"""scripts/check_for_update.py: the regression matrix from the diagnostic pass.

The worker's own retry/classification logic (real ``_private_package_result``,
a fake ``RelayClient``) -- distinct from the UI-level tests in
``tests/test_gui_release_flows`` (or wherever the Match window's manual/auto
handling is tested), which cover what the app *does* with these results.
"""

from __future__ import annotations

import socket
import urllib.error
from unittest.mock import patch

import pytest

import scripts.check_for_update as check_for_update
from core.update_check import UpdateCheckState


PACKAGE_ROW = {
    "name": "PatchLab-1.5.7-macOS.pkg", "version": "1.5.7", "size": 123,
    "sha256": "a" * 64, "kind": "macos-package",
}


class FakeClient:
    """Replaces RelayClient: each call consumes the next scripted response."""

    def __init__(self, base_url, password, *, timeout, token):
        self.timeout = timeout
        self.calls = 0

    def artifact_manifest(self):
        self.calls += 1
        outcome = FakeClient.script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    script: list = []


@pytest.fixture(autouse=True)
def _stub_relay(monkeypatch):
    monkeypatch.setenv("PATCHLAB_RELAY_URL", "https://relay.example")
    monkeypatch.delenv("PATCHLAB_DISABLE_RELAY", raising=False)
    monkeypatch.setattr("core.relay_client.RelayClient", FakeClient)
    monkeypatch.setattr(
        "core.access_gate.stored_relay_credential", lambda: ("secret", "tok"),
        raising=False,
    )
    monkeypatch.setattr(check_for_update, "CURRENT_VERSION", "1.5.6")
    FakeClient.script = []
    yield
    FakeClient.script = []


def _check():
    return check_for_update._private_package_result()


# --- A/B: real success paths ------------------------------------------------


def test_A_current_when_no_newer_package_exists() -> None:
    FakeClient.script = [[{
        "name": "PatchLab-1.5.6-macOS.pkg", "version": "1.5.6", "size": 1,
        "sha256": "b" * 64, "kind": "macos-package",
    }]]
    outcome = _check()
    assert outcome.state is UpdateCheckState.CURRENT
    assert outcome.as_dict()["update_available"] is False


def test_B_update_available_when_a_newer_package_exists() -> None:
    FakeClient.script = [[PACKAGE_ROW]]
    outcome = _check()
    assert outcome.state is UpdateCheckState.UPDATE_AVAILABLE
    assert outcome.remote_version == "1.5.7"
    assert outcome.package == {"name": "PatchLab-1.5.7-macOS.pkg", "version": "1.5.7",
                                "size": 123, "sha256": "a" * 64}


# --- C/D: real failures must say so, distinctly from "current" -------------


def test_C_a_timeout_on_both_attempts_is_check_failed_with_no_raw_traceback(capsys) -> None:
    FakeClient.script = [TimeoutError("timed out"), TimeoutError("timed out")]
    outcome = _check()
    assert outcome.state is UpdateCheckState.CHECK_FAILED
    assert outcome.failure_category == "timeout"
    assert "TimeoutError" not in outcome.user_message
    printed = capsys.readouterr().out
    assert "TimeoutError" in printed, "the technical detail still reaches diagnostics"


def test_D_connection_refused_is_check_failed() -> None:
    FakeClient.script = [
        urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
        urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
    ]
    outcome = _check()
    assert outcome.state is UpdateCheckState.CHECK_FAILED
    assert outcome.failure_category == "network"


def test_E_an_expired_token_is_auth_required_not_check_failed() -> None:
    FakeClient.script = [
        urllib.error.HTTPError("https://relay.example/artifacts", 401, "Unauthorized", {}, None)
    ]
    outcome = _check()
    assert outcome.state is UpdateCheckState.AUTH_REQUIRED


def test_no_stored_credential_at_all_is_auth_required(monkeypatch) -> None:
    monkeypatch.setattr("core.access_gate.stored_relay_credential", lambda: (None, None), raising=False)
    outcome = _check()
    assert outcome.state is UpdateCheckState.AUTH_REQUIRED
    assert FakeClient.script == [], "never even attempted a network call"


# --- G: exactly one retry for a transient failure, then success ------------


def test_G_a_transient_timeout_then_success_reports_update_available() -> None:
    FakeClient.script = [TimeoutError("timed out"), [PACKAGE_ROW]]
    outcome = _check()
    assert outcome.state is UpdateCheckState.UPDATE_AVAILABLE
    assert outcome.remote_version == "1.5.7"


def test_only_one_retry_is_attempted_never_a_loop() -> None:
    calls = {"n": 0}

    class CountingClient(FakeClient):
        def artifact_manifest(self):
            calls["n"] += 1
            raise TimeoutError("timed out")

    with patch("core.relay_client.RelayClient", CountingClient):
        outcome = _check()
    assert calls["n"] == check_for_update.MAX_ATTEMPTS == 2
    assert outcome.state is UpdateCheckState.CHECK_FAILED


def test_an_auth_failure_is_never_retried() -> None:
    calls = {"n": 0}

    class CountingAuthClient(FakeClient):
        def artifact_manifest(self):
            calls["n"] += 1
            raise urllib.error.HTTPError("https://x/artifacts", 401, "Unauthorized", {}, None)

    with patch("core.relay_client.RelayClient", CountingAuthClient):
        outcome = _check()
    assert calls["n"] == 1, "an expired token cannot be fixed by trying again"
    assert outcome.state is UpdateCheckState.AUTH_REQUIRED


# --- timeout bound: a manual check cannot sit apparently frozen ------------


def test_the_client_is_given_a_bounded_timeout() -> None:
    seen = {}

    class RecordingClient(FakeClient):
        def __init__(self, base_url, password, *, timeout, token):
            seen["timeout"] = timeout
            super().__init__(base_url, password, timeout=timeout, token=token)

    FakeClient.script = [[]]
    with patch("core.relay_client.RelayClient", RecordingClient):
        _check()
    assert 0 < seen["timeout"] <= 10.0


def test_disabled_relay_is_current_not_a_failure(monkeypatch) -> None:
    monkeypatch.setenv("PATCHLAB_DISABLE_RELAY", "1")
    outcome = _check()
    assert outcome.state is UpdateCheckState.CURRENT
    assert FakeClient.script == []


def test_missing_relay_url_is_current_not_a_failure(monkeypatch) -> None:
    monkeypatch.delenv("PATCHLAB_RELAY_URL", raising=False)
    outcome = _check()
    assert outcome.state is UpdateCheckState.CURRENT


# --- diagnostics carry the technical detail; the result never does --------


def test_no_secrets_reach_the_printed_result(capsys) -> None:
    FakeClient.script = [
        urllib.error.URLError("password=hunter2 leaked"),
        urllib.error.URLError("password=hunter2 leaked"),
    ]
    outcome = _check()
    assert "hunter2" not in outcome.as_dict().get("user_message", "")


def test_socket_timeout_via_urlerror_is_classified_as_timeout() -> None:
    FakeClient.script = [
        urllib.error.URLError(socket.timeout("timed out")),
        urllib.error.URLError(socket.timeout("timed out")),
    ]
    outcome = _check()
    assert outcome.failure_category == "timeout"
