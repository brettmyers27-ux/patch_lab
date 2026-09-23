from __future__ import annotations

import socket
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.update_check import (
    macos_package_releases,
    newest_macos_package,
    UpdateCheckOutcome,
    UpdateCheckState,
    UpdatePreferences,
    classify_update_check_exception,
    extract_version,
    fetch_remote_version,
    load_update_preferences,
    parse_version,
    save_update_preferences,
    update_available,
)


def test_parse_version_orders_naturally_not_lexically() -> None:
    assert parse_version("1.4.9") < parse_version("1.4.10")
    assert parse_version("1.5.0") > parse_version("1.4.99")
    assert parse_version("2.0.0") > parse_version("1.99.99")


def test_parse_version_falls_open_on_garbage() -> None:
    # A malformed remote value must never look newer than a real version.
    assert parse_version("not-a-version") == (0,)
    assert parse_version("") == (0,)


def test_extract_version_reads_the_real_file_format() -> None:
    source = '"""Docstring."""\n\n__version__ = "1.4.5"\n'
    assert extract_version(source) == "1.4.5"


def test_extract_version_returns_none_when_absent() -> None:
    assert extract_version("no version marker here") is None


def test_update_available_compares_correctly() -> None:
    assert update_available("1.4.5", "1.4.6") is True
    assert update_available("1.4.5", "1.4.5") is False
    assert update_available("1.4.5", "1.4.4") is False


def test_update_available_is_false_when_remote_is_unknown() -> None:
    assert update_available("1.4.5", None) is False


def test_private_macos_release_requires_explicit_kind_and_valid_metadata() -> None:
    rows = [
        {"name": "PatchLab-1.5.3-macOS.pkg", "version": "1.5.3", "size": 9,
         "sha256": "a" * 64, "kind": "macos-package"},
        {"name": "PatchLab-9.9.9-macOS.pkg", "version": "9.9.9", "size": 9,
         "sha256": "b" * 64, "kind": "runtime"},
        {"name": "PatchLab-1.5.4-macOS.pkg", "version": "1.5.3", "size": 9,
         "sha256": "c" * 64, "kind": "macos-package"},
    ]
    releases = macos_package_releases(rows)
    assert [release.version for release in releases] == ["1.5.3"]
    assert newest_macos_package(rows, "1.5.2").version == "1.5.3"
    assert newest_macos_package(rows, "1.5.3") is None


def test_fetch_remote_version_never_raises_on_network_failure() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("network unreachable")):
        assert fetch_remote_version(timeout=1.0) is None


def test_fetch_remote_version_parses_a_real_response(tmp_path: Path) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'__version__ = "9.9.9"'

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        assert fetch_remote_version() == "9.9.9"


def test_preferences_round_trip(tmp_path: Path) -> None:
    env = SimpleNamespace(app_data_dir=tmp_path)
    preferences = UpdatePreferences(auto_check=False, skipped_version="1.4.6")
    save_update_preferences(preferences, env)
    assert load_update_preferences(env) == preferences


def test_preferences_default_when_missing(tmp_path: Path) -> None:
    env = SimpleNamespace(app_data_dir=tmp_path / "does-not-exist")
    assert load_update_preferences(env) == UpdatePreferences()


def test_preferences_default_when_corrupt(tmp_path: Path) -> None:
    env = SimpleNamespace(app_data_dir=tmp_path)
    (tmp_path / "update-preferences.json").write_text("not json", encoding="utf-8")
    assert load_update_preferences(env) == UpdatePreferences()


# --- UpdateCheckOutcome: a failure is never encoded as "no update" ----------


def test_current_and_update_available_never_look_like_a_failure() -> None:
    current = UpdateCheckOutcome.current("1.5.6")
    assert current.state is UpdateCheckState.CURRENT
    payload = current.as_dict()
    assert payload["update_available"] is False and "failure_category" not in payload

    available = UpdateCheckOutcome.available("1.5.5", "1.5.6", package={"name": "x"})
    assert available.as_dict()["update_available"] is True
    assert available.as_dict()["package"] == {"name": "x"}


def test_check_failed_and_current_are_distinguishable() -> None:
    """The exact bug: a real failure must not collapse into update_available=false."""

    failed = UpdateCheckOutcome.failed("1.5.4", category="timeout")
    current = UpdateCheckOutcome.current("1.5.4")
    # Old readers checking only this boolean see the same "no update" value...
    assert failed.as_dict()["update_available"] == current.as_dict()["update_available"] is False
    # ...but the state field -- what anything written against this fix reads -- differs.
    assert failed.as_dict()["state"] != current.as_dict()["state"]
    assert failed.state is UpdateCheckState.CHECK_FAILED
    assert "timed out" in failed.user_message.lower() or "connection" in failed.user_message.lower()
    assert "TimeoutError" not in failed.user_message and "Traceback" not in failed.user_message


def test_auth_required_carries_no_technical_detail() -> None:
    outcome = UpdateCheckOutcome.auth_required("1.5.4")
    assert outcome.state is UpdateCheckState.AUTH_REQUIRED
    assert outcome.as_dict()["update_available"] is False
    assert "sign in" in outcome.user_message.lower()


def test_failure_messages_differ_by_category() -> None:
    messages = {
        category: UpdateCheckOutcome.failed("1.5.4", category=category).user_message
        for category in ("timeout", "network", "invalid_response", "unknown")
    }
    assert len(set(messages.values())) == len(messages), "each category should read distinctly"
    for message in messages.values():
        assert message and message[0].isupper()


# --- classify_update_check_exception: sorted into safe, non-secret categories -----


def test_classifies_timeout_directly_and_wrapped() -> None:
    assert classify_update_check_exception(TimeoutError("timed out")) == "timeout"
    assert classify_update_check_exception(socket.timeout("timed out")) == "timeout"
    wrapped = urllib.error.URLError(socket.timeout("timed out"))
    assert classify_update_check_exception(wrapped) == "timeout"


def test_classifies_connection_refused_as_network() -> None:
    refused = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    assert classify_update_check_exception(refused) == "network"
    assert classify_update_check_exception(ConnectionRefusedError()) == "network"
    assert classify_update_check_exception(OSError("network down")) == "network"


def test_classifies_expired_or_invalid_credentials_as_auth() -> None:
    unauthorized = urllib.error.HTTPError("https://x/artifacts", 401, "Unauthorized", {}, None)
    assert classify_update_check_exception(unauthorized) == "auth"


def test_other_http_errors_are_network_not_auth() -> None:
    server_error = urllib.error.HTTPError("https://x/artifacts", 503, "Unavailable", {}, None)
    assert classify_update_check_exception(server_error) == "network"


def test_classifies_a_malformed_response_distinctly() -> None:
    assert classify_update_check_exception(ValueError("bad json")) == "invalid_response"
    assert classify_update_check_exception(KeyError("token")) == "invalid_response"


def test_unrecognised_exceptions_fall_back_to_unknown() -> None:
    assert classify_update_check_exception(RuntimeError("???")) == "unknown"
