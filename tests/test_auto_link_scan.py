from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from app.ui import MainWindow
from core.privacy import PrivacyStore
from core.storage import StoragePreferences


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
    assert kwargs.get("refresh_only") is True
    assert kwargs.get("workers") == 1
    assert window._automatic_link_scan_active is True
    assert "link" not in window._workflow_activities
    assert "Checking for new presets" not in window.statusBar().currentMessage()


def test_automatic_scan_does_not_repaint_progress_or_flood_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    with (
        patch("app.ui.storage_status") as status,
        patch.object(window.runner, "start"),
        patch.object(window, "append_log") as append_log,
    ):
        status.return_value.available = True
        window.maybe_start_automatic_link_scan(env=env)
        window._local_library_progress_changed(
            {"stage": "render", "current": 1200, "total": 5000}
        )
        window._local_library_log("processed preset 1 of 5000")

    assert "render" not in window._workflow_activities
    assert window._automatic_log_lines_suppressed == 1
    # Only the single high-level startup line is allowed through here.
    assert append_log.call_count == 1


def test_unlinked_user_can_start_custom_synthesis_with_factory_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No personal library must never force the retrieval-only fallback."""

    window, _env = _window(tmp_path, monkeypatch, linked=False)
    window.privacy_choice = window.privacy_store.save(False)
    window._match_audio_path = tmp_path / "one-shot.wav"
    window._model_asset_error = None
    window._workflow_match_error = ""
    with (
        patch("app.ui.synthesis_readiness") as readiness,
        patch.object(window.match_runner, "start") as start,
    ):
        readiness.return_value = SimpleNamespace(available=True, reason="")
        window.start_match()

    start.assert_called_once()
    _args, kwargs = start.call_args
    assert kwargs["factory_only"] is False
    assert kwargs["local_db"] is None
    assert kwargs["local_audio_root"] is None


def test_starting_match_pauses_an_automatic_check_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _env = _window(tmp_path, monkeypatch, linked=False)
    window._match_audio_path = tmp_path / "one-shot.wav"
    window._model_asset_error = None
    window._workflow_match_error = ""
    window._automatic_link_scan_active = True
    with (
        patch("app.ui.synthesis_readiness") as readiness,
        patch.object(window.match_runner, "start"),
        patch.object(window.runner, "cancel") as cancel,
        patch.object(type(window.runner), "running", new=property(lambda self: True)),
    ):
        readiness.return_value = SimpleNamespace(available=True, reason="")
        window.start_match()

    cancel.assert_called_once()


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


def test_distribution_render_button_uses_bounded_link_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _env = _window(tmp_path, monkeypatch)
    window.storage_preferences = StoragePreferences(compact_mode=True)
    with (
        patch("app.ui.storage_status") as status,
        patch("app.ui.preparation_queue_ids", return_value=[41, 42]),
        patch.object(window.runner, "start") as linked_start,
        patch.object(window.render_runner, "start") as direct_render_start,
    ):
        status.return_value.available = True
        status.return_value.reason = ""
        window.start_render()

    linked_start.assert_called_once_with(
        Path(window.privacy_choice.linked_folder),
        local_library=True,
        preset_ids=[41, 42],
    )
    direct_render_start.assert_not_called()
