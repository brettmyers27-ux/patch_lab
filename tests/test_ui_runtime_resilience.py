from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from app.ui import MainWindow
from core.privacy import PrivacyStore


def _window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> MainWindow:
    QApplication.instance() or QApplication([])
    monkeypatch.setenv("PATCHLAB_DISTRIBUTION_MODE", "1")
    monkeypatch.delenv("PATCHLAB_PRIVACY_SETTINGS", raising=False)
    privacy = PrivacyStore(tmp_path / "privacy.json")
    privacy.save(True)
    return MainWindow(privacy_store=privacy)


def test_model_error_remains_visible_during_unrelated_library_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _window(tmp_path, monkeypatch)
    try:
        error = (
            "PatchLab model assets are unavailable at "
            f"{tmp_path / 'missing-cache'}"
        )
        window.report_model_asset_error(error)
        window._render_progress_changed(
            {
                "completed_note_pairs": 7,
                "total_note_pairs": 70,
                "renders_per_second": 1.0,
                "eta_seconds": 63,
            }
        )

        assert window.match_stats.text() == error
        assert "Model files need attention" in window.match_card_status.text()
        assert not window.match_start_button.isEnabled()
        assert error in window.log_pane.toPlainText()
    finally:
        window.close()


def test_large_library_job_requires_explicit_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _window(tmp_path, monkeypatch)
    start = Mock()
    window.runner.start = start  # type: ignore[method-assign]
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    prompts: list[str] = []

    def decline(_parent, _title, message, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        prompts.append(str(message))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", decline)
    try:
        window.choose_folder()

        start.assert_not_called()
        assert prompts
        assert "1–4 hours" in prompts[0]
        assert "matching may be noticeably slower" in prompts[0]
        assert "was not started" in window.log_pane.toPlainText()
    finally:
        window.close()


def test_linked_folder_card_refreshes_without_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _window(tmp_path, monkeypatch)
    window.runner.start = Mock()  # type: ignore[method-assign]
    linked = tmp_path / "Linked Presets"
    linked.mkdir()
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(linked),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    try:
        window.choose_folder()
        assert window.privacy_store.load().linked_folder == str(linked)
        assert window.scan_box.property("workflowState") == "in-progress"

        window._scan_completed(
            {
                "found": 0,
                "searchable_local": 0,
                "relay_uploaded": 0,
                "relay_already_present": 0,
                "relay_upload_failed": 0,
                "relay_disabled_after_failures": 0,
                "factory_skipped_upload": 0,
            }
        )
        assert window.scan_box.property("workflowState") == "complete"
        assert "Linked Presets" in window.scan_card_status.text()
    finally:
        window.close()
