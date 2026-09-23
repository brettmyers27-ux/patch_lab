"""The real MainWindow's side of update-check state handling.

Manual and automatic checks share one worker and one completion signal; only
the UI's own bookkeeping (``_update_check_manual``) tells them apart. These
drive that signal exactly as the real worker would and assert what a user
does or does not see.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.update_check import UpdatePreferences, save_update_preferences
from tests.test_gui_release_flows import FakeMessageBox, Gui, gui  # noqa: F401


def _result(state: str, **extra) -> dict:
    payload = {
        "state": state, "current_version": "1.5.6", "remote_version": None,
        "update_available": state == "update_available",
    }
    payload.update(extra)
    return payload


@pytest.fixture
def update_gui(gui: Gui):  # noqa: F811
    window = gui.window
    window.update_check_runner.start = MagicMock(name="update_check_runner.start")
    FakeMessageBox.shown = []
    return gui


# --- A/B: automatic checks stay silent for CURRENT, normal for UPDATE_AVAILABLE ----


def test_A_automatic_current_check_shows_nothing(update_gui) -> None:
    window = update_gui.window
    window.maybe_check_for_update()
    window.update_check_runner.completed.emit(_result("current"))
    assert FakeMessageBox.shown == []


def test_B_automatic_update_available_shows_the_normal_prompt(update_gui) -> None:
    window = update_gui.window
    window.maybe_check_for_update()
    window.update_check_runner.completed.emit(
        _result("update_available", remote_version="1.5.7")
    )
    assert any("1.5.7" in call["text"] for call in FakeMessageBox.shown)


# --- F: automatic startup failures are recorded, never shown, never "current" -----


def test_F_automatic_check_failed_does_not_interrupt_startup(update_gui) -> None:
    window = update_gui.window
    window.maybe_check_for_update()
    window.update_check_runner.completed.emit(
        _result("check_failed", failure_category="timeout",
                user_message="PatchLab couldn't reach the update service in time.")
    )
    assert FakeMessageBox.shown == [], "no modal dialog on a quiet startup check"
    assert any("Update check failed" in line for line in window.log_pane.toPlainText().splitlines())


def test_F_automatic_auth_required_does_not_pop_a_signin_dialog(update_gui, monkeypatch) -> None:
    window = update_gui.window
    reconnect = MagicMock(return_value=True)
    monkeypatch.setattr(window, "_reconnect_support_service", reconnect)
    window.maybe_check_for_update()
    assert window.update_check_runner.start.call_count == 1, "the automatic check itself"
    window.update_check_runner.completed.emit(_result("auth_required"))
    reconnect.assert_not_called()
    assert window.update_check_runner.start.call_count == 1, "no retry launched"


def test_automatic_worker_crash_is_recorded_silently(update_gui) -> None:
    window = update_gui.window
    window.maybe_check_for_update()
    window.update_check_runner.failed.emit("Update check exited with code 1")
    assert FakeMessageBox.shown == []


# --- manual checks: always a visible result ---------------------------------


def test_manual_current_tells_the_user_they_are_up_to_date(update_gui) -> None:
    window = update_gui.window
    window._start_update_check(manual=True)
    window.update_check_runner.completed.emit(_result("current"))
    assert any("up to date" in call["text"].lower() for call in FakeMessageBox.shown)


def test_C_manual_check_failed_explains_in_plain_language(update_gui) -> None:
    window = update_gui.window
    window._start_update_check(manual=True)
    window.update_check_runner.completed.emit(
        _result("check_failed", failure_category="timeout",
                user_message="PatchLab couldn't reach the update service in time. "
                             "Check your internet connection and try again.")
    )
    assert len(FakeMessageBox.shown) == 1
    text = FakeMessageBox.shown[0]["text"]
    assert "internet connection" in text
    assert "TimeoutError" not in text and "Traceback" not in text


def test_manual_worker_crash_still_tells_the_user_something(update_gui) -> None:
    window = update_gui.window
    window._start_update_check(manual=True)
    window.update_check_runner.failed.emit("Update check exited with code 1")
    assert len(FakeMessageBox.shown) == 1
    assert "try again" in FakeMessageBox.shown[0]["text"].lower()


def test_manual_check_ignores_a_previously_skipped_version(update_gui) -> None:
    window = update_gui.window
    save_update_preferences(UpdatePreferences(auto_check=True, skipped_version="1.5.7"))
    window._start_update_check(manual=True)
    window.update_check_runner.completed.emit(
        _result("update_available", remote_version="1.5.7")
    )
    assert any("1.5.7" in call["text"] for call in FakeMessageBox.shown), (
        "a manual click must answer for real, even for a version the user skipped before"
    )


def test_automatic_check_still_respects_a_skipped_version(update_gui) -> None:
    window = update_gui.window
    save_update_preferences(UpdatePreferences(auto_check=True, skipped_version="1.5.7"))
    window.maybe_check_for_update()
    window.update_check_runner.completed.emit(
        _result("update_available", remote_version="1.5.7")
    )
    assert FakeMessageBox.shown == []


# --- E: manual auth-required signs in, then retries the SAME check once -----


def test_E_manual_auth_required_signs_in_and_retries_once(update_gui, monkeypatch) -> None:
    window = update_gui.window
    reconnect = MagicMock(return_value=True)
    monkeypatch.setattr(window, "_reconnect_support_service", reconnect)
    window._start_update_check(manual=True)
    assert window.update_check_runner.start.call_count == 1
    window.update_check_runner.completed.emit(_result("auth_required"))

    reconnect.assert_called_once()
    assert window.update_check_runner.start.call_count == 2, "signed in, retried once"
    assert window._update_check_manual is True

    # The retried check succeeds: no second sign-in prompt, no stuck state.
    window.update_check_runner.completed.emit(_result("current"))
    assert any("up to date" in call["text"].lower() for call in FakeMessageBox.shown)


def test_a_declined_signin_explains_without_looping(update_gui, monkeypatch) -> None:
    window = update_gui.window
    reconnect = MagicMock(return_value=False)
    monkeypatch.setattr(window, "_reconnect_support_service", reconnect)
    window._start_update_check(manual=True)
    assert window.update_check_runner.start.call_count == 1
    window.update_check_runner.completed.emit(_result("auth_required"))
    reconnect.assert_called_once()
    assert window.update_check_runner.start.call_count == 1, "declined sign-in: no retry launched"


def test_signin_is_only_attempted_once_per_manual_check(update_gui, monkeypatch) -> None:
    """If the retried check ALSO comes back auth_required, don't sign in again."""

    window = update_gui.window
    reconnect = MagicMock(return_value=True)
    monkeypatch.setattr(window, "_reconnect_support_service", reconnect)
    window._start_update_check(manual=True)
    window.update_check_runner.completed.emit(_result("auth_required"))
    reconnect.assert_called_once()
    assert window.update_check_runner.start.call_count == 2

    window.update_check_runner.completed.emit(_result("auth_required"))
    assert reconnect.call_count == 1, "still only once"
    assert window.update_check_runner.start.call_count == 2, "no further retry launched"
    assert len(FakeMessageBox.shown) == 1
    assert "sign in" in FakeMessageBox.shown[0]["text"].lower()
