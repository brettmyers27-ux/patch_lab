from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from core.update_check import (
    UpdatePreferences,
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
