from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from app.ui import MainWindow
from core.privacy import PrivacyStore


def _window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    distribution_mode: bool = True,
    linked: bool = True,
) -> tuple[MainWindow, SimpleNamespace]:
    QApplication.instance() or QApplication([])
    monkeypatch.setenv("PATCHLAB_DISTRIBUTION_MODE", "1" if distribution_mode else "0")
    monkeypatch.delenv("PATCHLAB_PRIVACY_SETTINGS", raising=False)
    privacy = PrivacyStore(tmp_path / "privacy.json")
    folder = tmp_path / "Linked Presets"
    folder.mkdir()
    privacy.save(True, linked_folder=folder if linked else None)
    window = MainWindow(privacy_store=privacy)
    env = SimpleNamespace(app_data_dir=tmp_path / "app-data")
    return window, env


def test_starts_a_quiet_scan_when_a_folder_is_already_linked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    with (
        patch("app.ui.storage_status") as status,
        patch.object(window.runner, "start") as start,
    ):
        status.return_value.available = True
        window.maybe_start_automatic_link_scan(env=env)
    start.assert_called_once()
    args, kwargs = start.call_args
    assert args[0] == Path(window.privacy_choice.linked_folder)
    assert kwargs.get("local_library") is True


def test_does_not_scan_twice_in_the_same_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    with (
        patch("app.ui.storage_status") as status,
        patch.object(window.runner, "start") as start,
    ):
        status.return_value.available = True
        window.maybe_start_automatic_link_scan(env=env)
        window.maybe_start_automatic_link_scan(env=env)
    start.assert_called_once()


def test_no_folder_linked_does_not_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch, linked=False)
    with (
        patch("app.ui.storage_status") as status,
        patch.object(window.runner, "start") as start,
    ):
        status.return_value.available = True
        window.maybe_start_automatic_link_scan(env=env)
    start.assert_not_called()


def test_storage_unavailable_does_not_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    with (
        patch("app.ui.storage_status") as status,
        patch.object(window.runner, "start") as start,
    ):
        status.return_value.available = False
        window.maybe_start_automatic_link_scan(env=env)
    start.assert_not_called()


def test_already_running_does_not_scan_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    with (
        patch("app.ui.storage_status") as status,
        patch.object(window.runner, "start") as start,
        patch.object(type(window.runner), "running", new=property(lambda self: True)),
    ):
        status.return_value.available = True
        window.maybe_start_automatic_link_scan(env=env)
    start.assert_not_called()


def test_dev_mode_never_auto_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch, distribution_mode=False)
    with (
        patch("app.ui.storage_status") as status,
        patch.object(window.runner, "start") as start,
    ):
        status.return_value.available = True
        window.maybe_start_automatic_link_scan(env=env)
    start.assert_not_called()
