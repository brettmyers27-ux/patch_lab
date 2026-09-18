from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from app.ui import MainWindow
from core.privacy import PrivacyStore
from core.update_check import (
    UpdatePreferences,
    load_update_preferences,
    save_update_preferences,
)


def _window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    distribution_mode: bool = True,
) -> tuple[MainWindow, SimpleNamespace]:
    QApplication.instance() or QApplication([])
    monkeypatch.setenv("PATCHLAB_DISTRIBUTION_MODE", "1" if distribution_mode else "0")
    monkeypatch.delenv("PATCHLAB_PRIVACY_SETTINGS", raising=False)
    privacy = PrivacyStore(tmp_path / "privacy.json")
    # Record a consent decision before construction. Leaving it undecided
    # queues a real QTimer.singleShot(0, self._show_consent_dialog) that
    # these tests never drain; it stays pending on the shared QApplication
    # queue and later fires a blocking, unmocked modal dialog.exec() the
    # moment any other test in the same run pumps the event loop (e.g.
    # test_worker_runtime.py's QEventLoop().exec()), hanging the suite.
    privacy.save(True)
    window = MainWindow(privacy_store=privacy)
    env = SimpleNamespace(app_data_dir=tmp_path / "app-data")
    return window, env


def test_checks_on_launch_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    save_update_preferences(UpdatePreferences(auto_check=True), env)
    with patch.object(window.update_check_runner, "start") as start:
        window.maybe_check_for_update(env=env)
    start.assert_called_once()


def test_does_not_check_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    save_update_preferences(UpdatePreferences(auto_check=False), env)
    with patch.object(window.update_check_runner, "start") as start:
        window.maybe_check_for_update(env=env)
    start.assert_not_called()


def test_dev_mode_never_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    window, env = _window(tmp_path, monkeypatch, distribution_mode=False)
    save_update_preferences(UpdatePreferences(auto_check=True), env)
    with patch.object(window.update_check_runner, "start") as start:
        window.maybe_check_for_update(env=env)
    start.assert_not_called()


def test_packaged_pkg_checks_private_release_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PKG checks its private release catalog, never a source checkout."""

    window, env = _window(tmp_path, monkeypatch)
    monkeypatch.setenv("PATCHLAB_PACKAGED_INSTALLER", "1")
    save_update_preferences(UpdatePreferences(auto_check=True), env)
    with patch.object(window.update_check_runner, "start") as start:
        window.maybe_check_for_update(env=env)
    start.assert_called_once()


def test_does_not_check_again_while_already_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    save_update_preferences(UpdatePreferences(auto_check=True), env)
    with (
        patch.object(window.update_check_runner, "start") as start,
        patch.object(
            type(window.update_check_runner),
            "running",
            new=property(lambda self: True),
        ),
    ):
        window.maybe_check_for_update(env=env)
    start.assert_not_called()


def test_no_prompt_when_already_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    with patch.object(window, "_prompt_update_available") as prompt:
        window._update_check_completed(
            {"current_version": "1.4.5", "remote_version": "1.4.5", "update_available": False},
            env=env,
        )
    prompt.assert_not_called()


def test_prompts_when_a_newer_version_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    with patch.object(window, "_prompt_update_available") as prompt:
        window._update_check_completed(
            {"current_version": "1.4.5", "remote_version": "1.4.6", "update_available": True},
            env=env,
        )
    prompt.assert_called_once_with("1.4.6", env=env)


def test_does_not_reprompt_a_skipped_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    save_update_preferences(UpdatePreferences(auto_check=True, skipped_version="1.4.6"), env)
    with patch.object(window, "_prompt_update_available") as prompt:
        window._update_check_completed(
            {"current_version": "1.4.5", "remote_version": "1.4.6", "update_available": True},
            env=env,
        )
    prompt.assert_not_called()


def test_choosing_skip_records_the_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, env = _window(tmp_path, monkeypatch)
    with patch("app.ui.load_update_preferences", return_value=UpdatePreferences()):
        with patch("app.ui.save_update_preferences") as save:
            with patch.object(QMessageBox, "exec", return_value=0):
                with patch.object(QMessageBox, "clickedButton") as clicked:
                    box_holder: list = []
                    original_init = QMessageBox.__init__

                    def capture_init(self, *args, **kwargs):
                        original_init(self, *args, **kwargs)
                        box_holder.append(self)

                    with patch.object(QMessageBox, "__init__", capture_init):
                        def fake_clicked_button():
                            box = box_holder[-1]
                            for button in box.buttons():
                                if button.text() == "Skip This Version":
                                    return button
                            return None

                        clicked.side_effect = fake_clicked_button
                        window._prompt_update_available("9.9.9")
    save.assert_called_once()
    saved_preferences = save.call_args[0][0]
    assert saved_preferences.skipped_version == "9.9.9"


def test_choosing_update_now_calls_apply_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _env = _window(tmp_path, monkeypatch)
    with patch.object(QMessageBox, "exec", return_value=0):
        with patch.object(QMessageBox, "clickedButton") as clicked:
            box_holder: list = []
            original_init = QMessageBox.__init__

            def capture_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                box_holder.append(self)

            with patch.object(QMessageBox, "__init__", capture_init):
                def fake_clicked_button():
                    box = box_holder[-1]
                    for button in box.buttons():
                        if button.text() == "Update Now":
                            return button
                    return None

                clicked.side_effect = fake_clicked_button
                with patch.object(window, "_apply_update") as apply_update:
                    window._prompt_update_available("9.9.9")
    apply_update.assert_called_once()


def test_apply_update_spawns_a_detached_process_and_quits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _env = _window(tmp_path, monkeypatch)
    # Patch the QApplication *reference inside app.ui*, not the real PySide6
    # class globally: QApplication.instance() is a SIP/C++-backed static
    # method, and patching it process-wide risks leaving Qt's global state
    # broken for later tests instead of cleanly restoring.
    with (
        patch("subprocess.Popen") as popen,
        patch("app.ui.QApplication") as fake_qapplication,
    ):
        fake_qapplication.instance.return_value = SimpleNamespace(quit=lambda: None)
        window._apply_update()
    popen.assert_called_once()
    args = popen.call_args[0][0]
    assert args[0] == "/bin/bash"
    assert args[1].endswith("apply_update.sh")
    assert popen.call_args[1]["start_new_session"] is True
