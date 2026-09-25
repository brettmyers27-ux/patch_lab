"""The visible Phase 4 action presents one truthful preparation workflow."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication

from app.ui import MainWindow
from core.privacy import PrivacyStore


@pytest.fixture()
def window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MainWindow:
    QApplication.instance() or QApplication([])
    monkeypatch.setenv("PATCHLAB_DISTRIBUTION_MODE", "1")
    monkeypatch.delenv("PATCHLAB_PRIVACY_SETTINGS", raising=False)
    root = tmp_path / "presets"
    root.mkdir()
    privacy = PrivacyStore(tmp_path / "privacy.json")
    privacy.save(True, linked_folder=root)
    return MainWindow(privacy_store=privacy)


def test_one_prepare_action_replaces_the_old_two_cards(window: MainWindow) -> None:
    titles = [card.button.text() for card in window.hero_cards]

    assert titles == ["Link My Preset Folder", "Prepare Preset Library", "Match a Sound"]
    assert "Render Sound Library" not in titles
    assert "Analyze & Learn" not in titles


def test_prepare_progress_is_current_queue_terminal_count_and_never_regresses(
    window: MainWindow,
) -> None:
    window._prepare_active = True
    window._prepare_total = 100
    window._prepare_completed = 0
    window._prepare_started_at = time.monotonic() - 120
    progress = window._local_library_progress_changed

    progress({"stage": "prepare", "current": 1, "total": 100, "current_stage": "render", "preset_name": "Bass"})
    progress({"stage": "prepare", "current": 60, "total": 100, "current_stage": "commit", "preset_name": "Lead"})
    progress({"stage": "prepare", "current": 59, "total": 100, "current_stage": "cleanup", "preset_name": "Lead"})

    assert window.render_progress.value() == 60
    assert "About" in window.render_stats.text()
    assert "60 / 100 prepared" in window.render_stats.text()


def test_prepare_click_snapshots_the_backend_queue(window: MainWindow) -> None:
    with (
        patch("app.ui.preparation_queue_ids", return_value=[3, 8, 13]),
        patch.object(window.runner, "start") as start,
    ):
        window.start_render()

    start.assert_called_once_with(
        Path(window.privacy_choice.linked_folder),
        local_library=True,
        preset_ids=[3, 8, 13],
    )
    assert window._prepare_total == 3
