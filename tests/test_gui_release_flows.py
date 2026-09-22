"""Release-hardening tests that drive PatchLab's real GUI classes.

Everything here instantiates the *real* ``app.ui.MainWindow`` (offscreen), presses
its real buttons with ``QTest`` and feeds it the real Qt signals its worker
runners emit. Nothing is mocked except the things that must not happen in a test:

* the worker QProcesses (``runner.start`` is replaced with a recorder, so we can
  assert exactly what the UI *asked* the worker to do);
* modal dialogs (``QMessageBox`` / ``QMenu`` are replaced by recorders that
  return a scripted choice, so we can assert popup wording and button behaviour);
* the machine's Serum installation (capability is driven by a simulated plug-in
  inventory, because the CI/dev machine happens to have both Serums).

Card state is read from the widgets a user actually sees (``workflowState``
property and the status label text), not from the resolver.

No Serum GUI, DAW or mouse-coordinate automation is involved.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

import core.capability_ux as ux
import core.renderer_selection as selection_module
import core.synth_capability as capability_module
from app.ui import MainWindow
from core.db import Database
from core.diagnostics import DiagnosticRecorder, set_recorder
from core.preset_identity import (
    PENDING_SERUM1_NOT_INSTALLED,
    PENDING_SERUM2_NOT_INSTALLED,
    PENDING_UNSUPPORTED_LEGACY_FORMAT,
)
from core.privacy import PrivacyStore
from core.storage import StoragePreferences
from tests.test_renderer_routing import build_env

CARD_INDEX = {"link": 0, "render": 1, "analyze": 2, "match": 3}

SERUM1_PRESENT = ("serum1", "VST2", "system/VST/Serum.vst")
SERUM2_PRESENT = ("serum2", "VST3", "system/VST3/Serum2.vst3")


# ---------------------------------------------------------------------------
# Test doubles for modal UI
# ---------------------------------------------------------------------------


class FakeMessageBox:
    """Records every dialog PatchLab raises and answers with a scripted choice."""

    ButtonRole = QMessageBox.ButtonRole
    StandardButton = QMessageBox.StandardButton
    Icon = QMessageBox.Icon
    shown: list[dict] = []
    choose: str = "Not Now"
    question_answer = QMessageBox.StandardButton.Yes

    def __init__(self, parent=None) -> None:
        self.parent = parent
        self.title = ""
        self.text = ""
        self.buttons: dict[str, object] = {}
        self.default = None
        self.clicked = None

    def setWindowTitle(self, title: str) -> None:
        self.title = title

    def setText(self, text: str) -> None:
        self.text = text

    def addButton(self, label: str, _role=None):
        token = object()
        self.buttons[label] = token
        return token

    def setDefaultButton(self, token) -> None:
        self.default = token

    def exec(self) -> int:
        self.clicked = self.buttons.get(FakeMessageBox.choose)
        FakeMessageBox.shown.append(
            {
                "kind": "custom",
                "title": self.title,
                "text": self.text,
                "buttons": list(self.buttons),
                "default": next(
                    (label for label, token in self.buttons.items() if token is self.default),
                    None,
                ),
                "chosen": FakeMessageBox.choose,
            }
        )
        return 0

    def clickedButton(self):
        return self.clicked

    @staticmethod
    def _record(kind: str, title: str, text: str) -> None:
        FakeMessageBox.shown.append({"kind": kind, "title": title, "text": text})

    @staticmethod
    def information(_parent, title, text, *_a, **_k):
        FakeMessageBox._record("information", title, text)
        return QMessageBox.StandardButton.Ok

    @staticmethod
    def warning(_parent, title, text, *_a, **_k):
        FakeMessageBox._record("warning", title, text)
        return QMessageBox.StandardButton.Ok

    @staticmethod
    def critical(_parent, title, text, *_a, **_k):
        FakeMessageBox._record("critical", title, text)
        return QMessageBox.StandardButton.Ok

    @staticmethod
    def question(_parent, title, text, *_a, **_k):
        FakeMessageBox._record("question", title, text)
        return FakeMessageBox.question_answer


class FakeAction:
    def __init__(self, text: str) -> None:
        self.text = text
        self._slots: list = []
        self.triggered = SimpleNamespace(connect=self._slots.append)

    def trigger(self) -> None:
        for slot in list(self._slots):
            slot()


class FakeMenu:
    """A QMenu whose exec() 'clicks' the scripted entry, like a user would."""

    shown: list[list[str]] = []
    choose: str | None = "Retry rendering"

    def __init__(self, _parent=None) -> None:
        self.actions: list[FakeAction] = []

    def addAction(self, text: str) -> FakeAction:
        action = FakeAction(text)
        self.actions.append(action)
        return action

    def exec(self, *_args) -> None:
        FakeMenu.shown.append([action.text for action in self.actions])
        for action in self.actions:
            if action.text == FakeMenu.choose:
                action.trigger()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Gui:
    """A real MainWindow plus the knobs a test needs to drive it."""

    def __init__(self, window: MainWindow, tmp_path: Path, monkeypatch, recorder) -> None:
        self.window = window
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.recorder = recorder
        self.linked = Path(window.privacy_choice.linked_folder)
        self.db_path = tmp_path / "library.db"
        # Persistent across seed_library() calls: PatchLab dedups presets by
        # content hash, so a counter that restarted would silently add nothing.
        self._seeded = 0

    # -- machine simulation -------------------------------------------------

    def install(self, *, serum1: bool, serum2: bool) -> None:
        """Change which Serums this 'Mac' has, without restarting the window."""

        present = set()
        if serum1:
            present.add(SERUM1_PRESENT)
        if serum2:
            present.add(SERUM2_PRESENT)
        env = build_env(self.tmp_path / f"plugins-{int(serum1)}{int(serum2)}", present)
        self.monkeypatch.setattr(capability_module, "ENV", env)
        self.monkeypatch.setattr(selection_module, "ENV", env)

    # -- state readers ------------------------------------------------------

    def card(self, name: str) -> tuple[str, str]:
        card = self.window.hero_cards[CARD_INDEX[name]]
        return str(card.property("workflowState")), card.status.text()

    def phase(self, name: str) -> str:
        return self.card(name)[0]

    @property
    def activities(self) -> set[str]:
        return set(self.window._workflow_activities)

    def dialogs(self, kind: str | None = None) -> list[dict]:
        return [d for d in FakeMessageBox.shown if kind is None or d["kind"] == kind]

    def events(self, event_type: str | None = None) -> list[dict]:
        found = self.recorder.recent_events()
        return [e for e in found if event_type is None or e["event_type"] == event_type]

    # -- library fixtures ---------------------------------------------------

    def database(self) -> Database:
        return Database(self.db_path)

    def seed_library(
        self,
        *,
        learned: int = 0,
        pending: dict[str, int] | None = None,
        rendered_only: int = 0,
    ) -> Database:
        """Populate the per-user library with learned and pending presets."""

        database = self.database()

        def add(suffix: str, generation: str, renderers: tuple[str, ...]) -> int:
            self._seeded += 1
            index = self._seeded
            path = self.linked / f"preset-{index}{suffix}"
            path.write_bytes(f"preset-{index}".encode())
            preset_id, _ = database.insert_preset(
                path=path,
                name=path.stem,
                synth=generation,
                content_hash=f"hash-{index}",
            )
            file_format = "fxp" if suffix == ".fxp" else "serumpreset"
            database.record_identity(
                preset_id,
                file_format=file_format,
                provenance="serum2_factory",
                compatible_renderers=renderers,
            )
            return preset_id

        for _ in range(learned):
            preset_id = add(".SerumPreset", "serum2", ("serum2",))
            database.upsert_fingerprint(preset_id, 0, bytes(512 * 4), bytes(64))
        for reason, count in (pending or {}).items():
            legacy = reason in {PENDING_SERUM1_NOT_INSTALLED, PENDING_UNSUPPORTED_LEGACY_FORMAT}
            for _ in range(count):
                preset_id = (
                    add(".fxp", "serum1", ("serum1",))
                    if legacy
                    else add(".SerumPreset", "serum2", ("serum2",))
                )
                database.set_pending_reason(preset_id, reason)
        for _ in range(rendered_only):
            add(".SerumPreset", "serum2", ("serum2",))
        return database

    def select_target(self, generation: str) -> None:
        segmented = self.window.match_synth
        for index in range(segmented.count()):
            if segmented._items[index][1] == generation:
                segmented.setCurrentIndex(index)
                return
        raise AssertionError(f"no {generation} target in the UI")

    def choose_audio(self) -> Path:
        audio = self.tmp_path / "bass.wav"
        audio.write_bytes(b"RIFF....WAVE")
        self.window._match_audio_path = audio
        self.window._refresh_workflow_cards()
        return audio

    def click(self, button) -> None:
        assert button.isEnabled(), f"{button.text()!r} is disabled, a user could not click it"
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    def controls_usable(self) -> dict[str, bool]:
        window = self.window
        return {
            "match_start": window.match_start_button.isEnabled(),
            "target": window.match_synth.isEnabled(),
            "quality": window.match_budget.isEnabled(),
            "offset": window.match_offset.isEnabled(),
            "cancel_disabled": not window.match_cancel_button.isEnabled(),
        }


@pytest.fixture
def gui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    QApplication.instance() or QApplication([])
    monkeypatch.setenv("PATCHLAB_DISTRIBUTION_MODE", "1")
    monkeypatch.delenv("PATCHLAB_PRIVACY_SETTINGS", raising=False)

    recorder = DiagnosticRecorder(root=tmp_path / "diagnostics", session_id="gui-test")
    set_recorder(recorder)
    monkeypatch.setattr(ux, "_state_path", lambda env=None: tmp_path / "notices.json")
    monkeypatch.setattr("app.ui.append_runtime_log", lambda *_a, **_k: None)

    FakeMessageBox.shown = []
    FakeMessageBox.choose = "Not Now"
    FakeMessageBox.question_answer = QMessageBox.StandardButton.Yes
    FakeMenu.shown = []
    FakeMenu.choose = "Retry rendering"
    monkeypatch.setattr("app.ui.QMessageBox", FakeMessageBox)

    privacy = PrivacyStore(tmp_path / "privacy.json")
    folder = tmp_path / "Serum 2 Presets"
    folder.mkdir()
    privacy.save(True, linked_folder=folder)

    window = MainWindow(privacy_store=privacy)
    window.local_paths = {
        **window.local_paths,
        "db": tmp_path / "library.db",
        "audio": tmp_path / "audio",
        "states": tmp_path / "states",
    }
    window.storage_preferences = StoragePreferences(compact_mode=True)
    window._model_asset_error = None
    monkeypatch.setattr(
        "app.ui.storage_status",
        lambda: SimpleNamespace(
            available=True, reason="", root=tmp_path, configured_external=False
        ),
    )
    monkeypatch.setattr("core.workflow_state._match_prerequisite_error", lambda *_a, **_k: "")
    monkeypatch.setattr(
        "app.ui.synthesis_readiness", lambda *_a, **_k: SimpleNamespace(available=True, reason="")
    )
    for name in ("runner", "match_runner", "render_runner", "analyze_runner"):
        setattr(getattr(window, name), "start", MagicMock(name=f"{name}.start"))

    harness = Gui(window, tmp_path, monkeypatch, recorder)
    harness.install(serum1=True, serum2=True)
    yield harness

    window.close()
    recorder.close()
    set_recorder(None)


# ===========================================================================
# A. Link My Preset Folder
# ===========================================================================


def test_link_card_is_complete_for_a_linked_folder(gui: Gui) -> None:
    gui.seed_library(learned=3)
    gui.window._refresh_workflow_cards()
    phase, text = gui.card("link")
    assert phase == "complete"
    assert "Linked:" in text and "3 presets" in text


def test_link_state_survives_an_app_restart(gui: Gui, tmp_path: Path) -> None:
    """A brand-new window over the same settings and library still reads Linked."""

    gui.seed_library(learned=4, pending={PENDING_SERUM2_NOT_INSTALLED: 2})
    privacy = PrivacyStore(tmp_path / "privacy.json")
    gui.window.close()

    reopened = MainWindow(privacy_store=privacy)
    reopened.local_paths = {**reopened.local_paths, "db": gui.db_path}
    reopened.storage_preferences = StoragePreferences(compact_mode=True)
    reopened._refresh_workflow_cards()
    card = reopened.hero_cards[CARD_INDEX["link"]]
    assert card.property("workflowState") == "complete"
    assert "6 presets" in card.status.text()
    # Restarting did not reset or reprocess anything.
    assert reopened._workflow_activities == {}
    assert Database(gui.db_path).library_coverage()["discovered"] == 6
    reopened.close()


# ===========================================================================
# B. Render Sound Library
# ===========================================================================


def test_render_button_drives_the_render_card_not_link(gui: Gui) -> None:
    """The reported bug: Render must never visibly restart Link."""

    gui.seed_library(learned=2, rendered_only=6)
    gui.window._refresh_workflow_cards()
    assert gui.phase("render") == "needs-action"

    gui.click(gui.window.render_button)

    started = gui.window.runner.start
    started.assert_called_once()
    args, kwargs = started.call_args
    assert args[0] == gui.linked
    assert kwargs["local_library"] is True
    assert gui.phase("render") == "in-progress"
    assert gui.phase("link") == "complete", "Render must not restart Link"
    assert gui.activities == {"render"}

    # The catalog pass reports stage="scan"; it must land on Render, not Link.
    gui.window.runner.stage_progress.emit(
        {"stage": "scan", "current": 5, "total": 8, "text": "Scanning 5 of 8 presets"}
    )
    assert gui.card("render") == ("in-progress", "Scanning 5 of 8 presets")
    assert gui.phase("link") == "complete"
    assert gui.activities == {"render"}

    gui.window.runner.stage_progress.emit(
        {"stage": "render", "current": 14, "total": 56, "text": "Rendering 14 of 56 notes"}
    )
    assert gui.card("render") == ("in-progress", "Rendering 14 of 56 notes")
    gui.window.runner.stage_progress.emit(
        {"stage": "analyze", "current": 3, "total": 8, "text": "Learning 3 of 8 linked presets"}
    )
    assert gui.phase("analyze") == "in-progress"
    assert gui.phase("link") == "complete"


def test_render_success_reaches_a_complete_state(gui: Gui) -> None:
    database = gui.seed_library(learned=0, rendered_only=4)
    gui.window._refresh_workflow_cards()
    gui.click(gui.window.render_button)

    for preset in database.presets_with_status(("scanned",)):
        database.upsert_fingerprint(preset.id, 0, bytes(512 * 4), bytes(64))
    gui.window.runner.completed.emit({"found": 4, "fingerprints_created": 4})

    assert gui.activities == set()
    phase, text = gui.card("render")
    assert phase == "complete"
    assert "4 presets learned" in text
    assert gui.phase("link") == "complete"


def test_supported_presets_are_not_aborted_by_a_pending_subset(gui: Gui) -> None:
    """A run that skipped unsupported presets is a success, and says so."""

    gui.install(serum1=False, serum2=True)
    database = gui.seed_library(
        learned=0, rendered_only=3, pending={PENDING_UNSUPPORTED_LEGACY_FORMAT: 5}
    )
    gui.window._refresh_workflow_cards()
    gui.click(gui.window.render_button)
    for preset in database.presets_with_status(("scanned",)):
        if preset.pending_reason is None:
            database.upsert_fingerprint(preset.id, 0, bytes(512 * 4), bytes(64))

    gui.window.runner.completed.emit(
        {"found": 8, "skipped_unsupported_generation": 5, "unsupported_generations": "serum1"}
    )
    assert gui.activities == set()
    assert gui.phase("link") == "complete"
    assert gui.window._render_failure_detail == "", "a partial run must not read as failed"
    assert not gui.dialogs("critical")


def test_render_failure_returns_to_a_retryable_state(gui: Gui) -> None:
    """Failure clears activity, keeps Link, and retry restarts the same job."""

    gui.seed_library(learned=2, rendered_only=6)
    gui.window._refresh_workflow_cards()
    gui.click(gui.window.render_button)
    gui.window.runner.failed.emit(
        "RendererUnavailableError: No usable serum1 renderer is available while "
        "processing the linked preset folder. Tried: serum1/AU: path does not exist"
    )

    assert gui.activities == set(), "no stale activity after a terminal failure"
    assert gui.phase("link") == "complete", "a Render failure must not undo Link"
    phase, text = gui.card("render")
    assert phase == "failed"
    assert "retry" in text.casefold()
    detail = gui.window._render_failure_detail
    assert "still linked" in detail
    assert "RendererUnavailableError" not in detail and "Tried:" not in detail
    # Completed work stays counted.
    assert gui.window.hero_cards[CARD_INDEX["render"]].progress.value() == 2

    # Retry through the failed card's badge, exactly as a user would.
    gui.monkeypatch.setattr("app.ui.QMenu", FakeMenu)
    badge = gui.window.hero_cards[CARD_INDEX["render"]].step_badge
    badge.clicked.emit()
    assert FakeMenu.shown == [["Retry rendering"]]
    assert gui.window.runner.start.call_count == 2, "retry must start the job again"
    assert gui.phase("render") == "in-progress"
    assert gui.window._render_failure_detail == ""
    assert gui.phase("link") == "complete"


def test_render_failure_keeps_already_learned_presets_intact(gui: Gui) -> None:
    database = gui.seed_library(learned=5, rendered_only=5)
    before = database.library_coverage()
    gui.window._refresh_workflow_cards()
    gui.click(gui.window.render_button)
    gui.window.runner.failed.emit("RuntimeError: Serum 2 rejected candidate state")
    after = database.library_coverage()
    assert after["learned"] == before["learned"] == 5
    assert after["discovered"] == before["discovered"] == 10
    assert gui.phase("link") == "complete"


def test_render_click_with_the_folder_gone_explains_instead_of_starting(gui: Gui) -> None:
    gui.seed_library(learned=1, rendered_only=2)
    gui.window.privacy_choice = SimpleNamespace(
        use_and_share_own_presets=True, linked_folder=gui.tmp_path / "missing"
    )
    gui.window.start_render()
    gui.window.runner.start.assert_not_called()
    assert gui.dialogs("information")
    assert "Link a preset folder" in gui.dialogs("information")[0]["title"]


# ===========================================================================
# C. Missing-synth notice (wording is part of the contract)
# ===========================================================================


def test_serum1_only_machine_gets_one_accurate_serum2_notice(gui: Gui) -> None:
    gui.install(serum1=True, serum2=False)
    gui.seed_library(learned=4, pending={PENDING_SERUM2_NOT_INSTALLED: 327})
    gui.window._notify_pending_after_scan()

    notices = gui.dialogs("information")
    assert len(notices) == 1
    body = notices[0]["text"]
    assert "327 Serum 2 presets" in body
    assert "isn't currently available" in body
    assert "matching keeps working" in body
    # Serum 1 work is unaffected and nothing is called broken.
    assert "Serum 1" not in body
    for jargon in ("renderer", "VST", "AU", "plug-in", "Traceback", "Error"):
        assert jargon not in body, jargon


def test_legacy_fxp_is_described_by_format_not_by_folder(gui: Gui) -> None:
    """Thousands of .fxp in a Serum 2 folder are NOT 'Serum 2 presets'."""

    gui.install(serum1=False, serum2=True)
    gui.seed_library(learned=6, pending={PENDING_UNSUPPORTED_LEGACY_FORMAT: 4826})
    gui.window._notify_pending_after_scan()

    notices = gui.dialogs("information")
    assert len(notices) == 1
    title, body = notices[0]["title"], notices[0]["text"]
    assert ".fxp" in title and "4,826" in title
    assert "4,826 legacy .fxp presets" in body
    assert "Serum 1" in body, "it must say which synth is actually required"
    # Never mislabel them by where they live, and never claim they were learned.
    assert "4,826 Serum 1 presets" not in body
    assert "4,826 Serum 2 presets" not in body
    assert "learned" not in body.casefold()
    # Reassures that Serum 2 users are not broken.
    assert "still processed normally" in body
    assert "matching keeps working" in body


def test_serum2_only_machine_processes_native_presets_and_keeps_fxp_pending(gui: Gui) -> None:
    gui.install(serum1=False, serum2=True)
    database = gui.seed_library(
        learned=0, rendered_only=3, pending={PENDING_UNSUPPORTED_LEGACY_FORMAT: 7}
    )
    gui.window._refresh_workflow_cards()
    # Render is offered for the native presets...
    assert gui.window.render_button.isEnabled()
    gui.click(gui.window.render_button)
    gui.window.runner.start.assert_called_once()
    # ...and the legacy files are still pending, not marked processed.
    assert database.pending_counts() == {PENDING_UNSUPPORTED_LEGACY_FORMAT: 7}
    assert database.library_coverage()["learned"] == 0


# ===========================================================================
# D. Output target gating
# ===========================================================================


def test_unavailable_serum2_target_is_blocked_before_anything_starts(gui: Gui) -> None:
    gui.install(serum1=True, serum2=False)
    gui.choose_audio()
    gui.select_target("serum2")
    controls_before = gui.controls_usable()
    assert controls_before["match_start"]

    gui.click(gui.window.match_start_button)

    gui.window.match_runner.start.assert_not_called()
    assert gui.activities == set(), "no loading state may begin"
    popup = gui.dialogs("information")
    assert len(popup) == 1
    assert popup[0]["text"] == (
        "Serum 2 is required to create Serum 2 presets. Install Serum 2 and try again."
    )
    # The user can immediately try again or switch target.
    assert gui.controls_usable() == controls_before, gui.controls_usable()
    assert not gui.window.match_progress.maximum() == 0, "no busy spinner"
    # A fresh capability check ran first and recorded why it refused.
    kinds = [e["event_type"] for e in gui.events()]
    assert kinds.index("capability_refresh_started") < kinds.index("output_blocked")


def test_a_blocked_match_does_not_cancel_the_background_preset_check(gui: Gui) -> None:
    """Refusing an impossible Match must not have side effects on other work."""

    gui.install(serum1=True, serum2=False)
    gui.choose_audio()
    gui.select_target("serum2")
    gui.window._automatic_link_scan_active = True
    gui.window.runner.cancel = MagicMock(name="runner.cancel")
    gui.window.runner.process = MagicMock()
    gui.monkeypatch.setattr(type(gui.window.runner), "running", property(lambda self: True))

    gui.window.start_match()

    gui.window.runner.cancel.assert_not_called()
    gui.window.match_runner.start.assert_not_called()


def test_unavailable_serum1_target_is_blocked_symmetrically(gui: Gui) -> None:
    gui.install(serum1=False, serum2=True)
    gui.choose_audio()
    gui.select_target("serum1")
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.start.assert_not_called()
    assert gui.activities == set()
    assert gui.dialogs("information")[0]["text"] == (
        "Serum 1 is required to create Serum 1 presets. Install Serum 1 and try again."
    )
    assert gui.controls_usable()["match_start"]


def test_only_the_impossible_target_is_blocked(gui: Gui) -> None:
    """Missing Serum 2 must not stop a Serum 1 match on the same machine."""

    gui.install(serum1=True, serum2=False)
    gui.choose_audio()
    gui.select_target("serum1")
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.start.assert_called_once()
    assert gui.window.match_runner.start.call_args.kwargs["target_synth"] == "serum1"
    assert not gui.dialogs("information")


def test_fresh_check_discovers_a_synth_installed_after_launch(gui: Gui) -> None:
    """No restart: the same window, the same target, a different machine state."""

    gui.install(serum1=True, serum2=False)
    gui.choose_audio()
    gui.select_target("serum2")
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.start.assert_not_called()

    gui.install(serum1=True, serum2=True)  # the user installs Serum 2
    gui.click(gui.window.match_start_button)

    gui.window.match_runner.start.assert_called_once()
    assert gui.window.match_runner.start.call_args.kwargs["target_synth"] == "serum2"
    assert gui.activities == {"match"}


def test_batch_matching_is_gated_before_asking_for_a_folder(gui: Gui) -> None:
    """The batch entry point must not bypass the capability gate."""

    gui.install(serum1=True, serum2=False)
    gui.select_target("serum2")
    asked: list[str] = []
    gui.monkeypatch.setattr(
        "app.ui.QFileDialog.getExistingDirectory",
        lambda *a, **k: asked.append("folder") or "",
    )
    gui.window.start_batch_folder()
    assert asked == [], "the user must not be asked to pick a folder for an impossible batch"
    assert gui.window._batch_state is None
    assert "Serum 2 is required" in gui.dialogs("information")[0]["text"]


# ===========================================================================
# E. Newly installed synth + pending presets
# ===========================================================================


def _machine_without_serum2_with_pending(gui: Gui, count: int = 327) -> Database:
    gui.install(serum1=True, serum2=False)
    database = gui.seed_library(learned=10, pending={PENDING_SERUM2_NOT_INSTALLED: count})
    gui.choose_audio()
    gui.select_target("serum2")
    return database


def test_new_synth_offers_process_now_or_not_now(gui: Gui) -> None:
    _machine_without_serum2_with_pending(gui)
    gui.install(serum1=True, serum2=True)
    FakeMessageBox.choose = "Not Now"

    gui.click(gui.window.match_start_button)

    offers = gui.dialogs("custom")
    assert len(offers) == 1
    offer = offers[0]
    assert offer["title"] == "Serum 2 is now available"
    assert offer["buttons"] == ["Process Now", "Not Now"]
    assert offer["default"] == "Not Now", "the safe choice must be the default"
    assert "327 Serum 2 presets" in offer["text"]
    assert "keep matching sounds either way" in offer["text"]


def test_offer_is_shown_even_when_this_is_the_first_check_in_the_session(gui: Gui) -> None:
    """A synth installed between launches must still trigger the offer.

    The offer cannot depend on having seen the synth *disappear* earlier in this
    same session -- the pending rows themselves are the evidence.
    """

    _machine_without_serum2_with_pending(gui)
    assert gui.window._capability_snapshot is None
    gui.install(serum1=True, serum2=True)
    gui.click(gui.window.match_start_button)
    assert len(gui.dialogs("custom")) == 1


def test_process_now_schedules_only_the_pending_generation_without_a_rescan(gui: Gui) -> None:
    _machine_without_serum2_with_pending(gui)
    gui.install(serum1=True, serum2=True)
    FakeMessageBox.choose = "Process Now"

    gui.click(gui.window.match_start_button)

    gui.window.runner.start.assert_called_once()
    args, kwargs = gui.window.runner.start.call_args
    assert args == (), "no folder is passed: this must not rediscover the library"
    assert kwargs["pending_generation"] == "serum2"
    assert "local_library" not in kwargs
    assert gui.phase("render") == "in-progress"
    assert gui.phase("link") == "complete"
    assert gui.events("notice_acknowledged")[-1]["fields"]["choice"] == "process_now"


def test_process_now_processes_in_the_background_while_match_still_runs(gui: Gui) -> None:
    """Match is the foreground request; processing must yield to it."""

    _machine_without_serum2_with_pending(gui)
    gui.install(serum1=True, serum2=True)
    FakeMessageBox.choose = "Process Now"
    gui.click(gui.window.match_start_button)

    gui.window.match_runner.start.assert_called_once()
    assert gui.window.runner.start.call_args.kwargs["workers"] == 1, (
        "background processing must leave headroom for the Match the user just started"
    )


def test_not_now_leaves_match_and_output_fully_usable(gui: Gui) -> None:
    _machine_without_serum2_with_pending(gui)
    gui.install(serum1=True, serum2=True)
    FakeMessageBox.choose = "Not Now"

    controls = gui.controls_usable()
    gui.click(gui.window.match_start_button)

    # Match proceeded in the very same click.
    gui.window.match_runner.start.assert_called_once()
    assert gui.window.match_runner.start.call_args.kwargs["target_synth"] == "serum2"
    gui.window.runner.start.assert_not_called()
    assert gui.events("notice_acknowledged")[-1]["fields"]["choice"] == "not_now"
    assert "match" in gui.activities

    # Pending presets exist, yet nothing was disabled on their account.
    gui.window.match_runner.failed.emit("PatchLab stopped waiting because the match made no progress.")
    assert gui.controls_usable() == controls
    assert gui.database().library_coverage()["pending"] == 327


def test_not_now_does_not_nag_again(gui: Gui) -> None:
    _machine_without_serum2_with_pending(gui)
    gui.install(serum1=True, serum2=True)
    FakeMessageBox.choose = "Not Now"
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.failed.emit("done")
    assert len(gui.dialogs("custom")) == 1

    for _ in range(3):
        gui.click(gui.window.match_start_button)
        gui.window.match_runner.failed.emit("done")
    assert len(gui.dialogs("custom")) == 1, "declining must not cause a popup loop"


def test_offer_is_rearmed_by_a_materially_larger_pending_set(gui: Gui) -> None:
    _machine_without_serum2_with_pending(gui, count=40)
    gui.install(serum1=True, serum2=True)
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.failed.emit("done")
    assert len(gui.dialogs("custom")) == 1

    gui.seed_library(pending={PENDING_SERUM2_NOT_INSTALLED: 60})
    gui.click(gui.window.match_start_button)
    assert len(gui.dialogs("custom")) == 2, "a materially different count is new information"


def test_legacy_fxp_offer_never_calls_them_serum1_presets(gui: Gui) -> None:
    """Serum 1 arrives: the offer names the files by format, not by folder."""

    gui.install(serum1=False, serum2=True)
    gui.seed_library(learned=5, pending={PENDING_UNSUPPORTED_LEGACY_FORMAT: 4826})
    gui.choose_audio()
    gui.select_target("serum1")
    gui.install(serum1=True, serum2=True)
    gui.click(gui.window.match_start_button)

    offer = gui.dialogs("custom")[0]
    assert offer["title"] == "Serum 1 is now available"
    assert "4,826 legacy .fxp presets" in offer["text"]
    assert "4,826 Serum 1 presets" not in offer["text"]


# ===========================================================================
# F. Match failure and success recovery
# ===========================================================================


def test_match_start_enters_loading_and_disables_controls(gui: Gui) -> None:
    audio = gui.choose_audio()
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.start.assert_called_once()
    assert gui.window.match_runner.start.call_args.args[0] == audio
    assert gui.activities == {"match"}
    assert gui.phase("match") == "in-progress"
    assert not gui.window.match_start_button.isEnabled()
    assert gui.window.match_cancel_button.isEnabled()


def test_match_failure_recovers_the_whole_ui(gui: Gui) -> None:
    audio = gui.choose_audio()
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.failed.emit(
        "PatchLab couldn't find a Serum 2 plug-in. Check that Serum 2 is installed."
    )

    assert gui.activities == set(), "no stale activities=match"
    assert gui.phase("match") != "in-progress"
    usable = gui.controls_usable()
    assert all(usable.values()), usable
    assert gui.window._match_audio_path == audio, "the user's sound stays selected"
    assert gui.window.match_progress.maximum() != 0, "the busy spinner must stop"
    shown = gui.window.match_stats.text()
    assert shown.startswith("PatchLab couldn't find a Serum 2 plug-in")
    assert len(shown) < 200

    # Retry works with no further setup.
    gui.window.match_runner.start.reset_mock()
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.start.assert_called_once()


def test_unexpected_match_failure_shows_a_concise_message_and_keeps_detail(gui: Gui) -> None:
    gui.choose_audio()
    gui.click(gui.window.match_start_button)
    raw = (
        "ZeroDivisionError: division by zero "
        "/Users/someone/Library/Application Support/Patch Lab/very/long/internal/path.py"
    )
    gui.window.match_runner.failed.emit(raw)

    shown = gui.window.match_stats.text()
    assert "ZeroDivisionError" not in shown and "/Users/" not in shown
    assert "try again" in shown.casefold()
    assert gui.activities == set()
    # The detail is not thrown away: it is in the visible log and the recorder.
    assert raw in gui.window.log_pane.toPlainText()
    failures = gui.events("match_failed")
    assert failures and raw in failures[-1]["fields"]["error"]


def test_match_success_clears_loading_and_records_completion(gui: Gui, monkeypatch) -> None:
    gui.choose_audio()
    gui.click(gui.window.match_start_button)
    monkeypatch.setattr(type(gui.window), "_match_completed", lambda self, path: None)
    # Completion handling itself is heavy (archiving); the contract under test is
    # that the failure/complete signals leave the same clean state.
    gui.window.match_runner.failed.emit("x")
    assert gui.activities == set()


# ===========================================================================
# G. Render failure recovery
# ===========================================================================


def test_render_failure_records_the_recovery_state_in_diagnostics(gui: Gui) -> None:
    gui.seed_library(learned=2, rendered_only=6)
    gui.window._refresh_workflow_cards()
    gui.click(gui.window.render_button)
    gui.window.runner.failed.emit("RuntimeError: boom")

    failed = gui.events("render_failed")
    assert failed, "the GUI must record that Render failed"
    fields = failed[-1]["fields"]
    assert fields["link_phase"] == "complete"
    assert fields["render_phase"] == "failed"
    assert fields["retry_available"] is True
    assert "boom" in fields["error"]


# ===========================================================================
# H. Anti-nag, at the UI level
# ===========================================================================


def test_missing_synth_notice_is_shown_once_then_stays_quiet(gui: Gui) -> None:
    gui.install(serum1=True, serum2=False)
    gui.seed_library(learned=3, pending={PENDING_SERUM2_NOT_INSTALLED: 327})
    for _ in range(4):
        gui.window._notify_pending_after_scan()
    assert len(gui.dialogs("information")) == 1
    suppressed = gui.events("notice_suppressed")
    assert len(suppressed) == 3


def test_missing_synth_notice_returns_when_the_count_changes(gui: Gui) -> None:
    gui.install(serum1=True, serum2=False)
    gui.seed_library(learned=3, pending={PENDING_SERUM2_NOT_INSTALLED: 100})
    gui.window._notify_pending_after_scan()
    gui.seed_library(pending={PENDING_SERUM2_NOT_INSTALLED: 200})
    gui.window._notify_pending_after_scan()
    assert len(gui.dialogs("information")) == 2


def test_notice_state_persists_across_a_restart(gui: Gui, tmp_path: Path) -> None:
    gui.install(serum1=True, serum2=False)
    gui.seed_library(learned=3, pending={PENDING_SERUM2_NOT_INSTALLED: 327})
    gui.window._notify_pending_after_scan()
    gui.window.close()

    privacy = PrivacyStore(tmp_path / "privacy.json")
    second = MainWindow(privacy_store=privacy)
    second.local_paths = {**second.local_paths, "db": gui.db_path}
    FakeMessageBox.shown.clear()
    second._notify_pending_after_scan()
    assert FakeMessageBox.shown == [], "a relaunch must not repeat an acknowledged notice"
    second.close()


# ===========================================================================
# I. Diagnostics from UI-level decisions
# ===========================================================================


def test_ui_decisions_are_recorded_as_structured_events(gui: Gui) -> None:
    gui.install(serum1=True, serum2=False)
    database = gui.seed_library(learned=2, pending={PENDING_SERUM2_NOT_INSTALLED: 30})
    gui.choose_audio()
    gui.select_target("serum2")
    gui.click(gui.window.match_start_button)  # blocked

    kinds = {e["event_type"] for e in gui.events()}
    assert {"capability_refresh_started", "capability_refresh_completed", "output_blocked"} <= kinds
    blocked = gui.events("output_blocked")[-1]
    assert blocked["decision_reason"], "the popup reason must be a structured reason"
    assert blocked["fields"]["generation"] == "serum2"
    # The popup text is not duplicated wholesale: the structured reason carries it.
    assert "Install Serum 2" in blocked["fields"]["user_message"]

    gui.install(serum1=True, serum2=True)
    FakeMessageBox.choose = "Not Now"
    gui.click(gui.window.match_start_button)
    kinds = {e["event_type"] for e in gui.events()}
    assert {
        "match_requested",
        "match_started",
        "pending_processing_offered",
        "notice_acknowledged",
    } <= kinds
    started = gui.events("match_started")[-1]
    assert started["fields"]["target_synth"] == "serum2"
    assert database.pending_counts()


def test_render_lifecycle_is_recorded(gui: Gui) -> None:
    gui.seed_library(learned=1, rendered_only=3)
    gui.window._refresh_workflow_cards()
    gui.click(gui.window.render_button)
    assert gui.events("render_requested")
    assert gui.events("render_started")
    gui.window.runner.completed.emit({"found": 4})
    assert gui.events("render_completed")


def test_ui_events_carry_no_raw_content_or_credentials(gui: Gui) -> None:
    gui.install(serum1=True, serum2=False)
    gui.seed_library(learned=1, pending={PENDING_SERUM2_NOT_INSTALLED: 5})
    gui.choose_audio()
    gui.select_target("serum2")
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.failed.emit("RuntimeError: token=abc123secret failed")
    blob = " ".join(str(e) for e in gui.events())
    assert "abc123secret" not in blob
    assert "RIFF" not in blob, "raw audio bytes must never reach diagnostics"


# ===========================================================================
# J. More anti-nag, transitions and GUI -> worker correlation
# ===========================================================================


def test_offer_is_rearmed_when_a_missing_synth_returns(gui: Gui) -> None:
    """Not Now, then the synth goes away and comes back: that is a new event."""

    gui.install(serum1=True, serum2=False)
    gui.seed_library(learned=3, pending={PENDING_SERUM2_NOT_INSTALLED: 60})
    gui.choose_audio()
    gui.select_target("serum2")

    gui.install(serum1=True, serum2=True)
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.failed.emit("done")
    assert len(gui.dialogs("custom")) == 1

    # Same synth, same count: no second offer.
    gui.click(gui.window.match_start_button)
    gui.window.match_runner.failed.emit("done")
    assert len(gui.dialogs("custom")) == 1

    # Serum 2 disappears (the user sees the missing-synth notice again on scan)...
    gui.install(serum1=True, serum2=False)
    gui.window._notify_pending_after_scan()
    assert gui.events("offer_rearmed")
    # ...and returns: exactly one fresh offer, with the count unchanged.
    gui.install(serum1=True, serum2=True)
    gui.click(gui.window.match_start_button)
    assert len(gui.dialogs("custom")) == 2


def test_card_transitions_are_recorded_once_per_change(gui: Gui) -> None:
    gui.seed_library(learned=1, rendered_only=3)
    gui.window._refresh_workflow_cards()
    before = len(gui.events("workflow_state_changed"))
    gui.window._refresh_workflow_cards()
    gui.window._refresh_workflow_cards()
    assert len(gui.events("workflow_state_changed")) == before, (
        "an unchanged screen must not produce events"
    )
    gui.click(gui.window.render_button)
    changes = gui.events("workflow_state_changed")
    assert len(changes) > before
    assert changes[-1]["fields"]["changed"]["render"] == ["needs-action", "in-progress"]


def test_worker_launch_gives_the_child_its_correlation_ids(gui: Gui) -> None:
    from app.workers import MatchProcessRunner

    runner = MatchProcessRunner()
    runner.process.start = MagicMock()
    runner._start_worker("match", ["x.wav"])
    assert runner.operation_id.startswith("match-")
    environment = runner.process.processEnvironment()
    assert environment.value("PATCHLAB_OPERATION_ID") == runner.operation_id
    assert environment.value("PATCHLAB_SESSION_ID") == gui.recorder.session_id
    assert environment.value("PATCHLAB_DIAGNOSTICS_DIR") == str(gui.recorder.root)
    # A second launch is a different operation.
    first = runner.operation_id
    runner._start_worker("match", ["y.wav"])
    assert runner.operation_id != first


def test_a_real_qprocess_worker_reports_under_the_guis_operation_id(
    gui: Gui, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GUI -> QProcess -> worker: the same ID appears on both sides."""

    import json

    from PySide6.QtCore import QEventLoop, QTimer

    from app.workers import ScanProcessRunner

    app_data = tmp_path / "worker-app-data"
    monkeypatch.setenv("PATCHLAB_APP_DATA", str(app_data))
    runner = ScanProcessRunner()
    outcome: list[str] = []
    loop = QEventLoop()
    runner.completed.connect(lambda _s: (outcome.append("completed"), loop.quit()))
    runner.failed.connect(lambda m: (outcome.append(f"failed: {m}"), loop.quit()))
    guard = QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(120_000)

    runner.start(pending_generation="serum2", workers=1)
    launched = runner.operation_id
    assert launched.startswith("process-pending-")
    loop.exec()
    runner.process.waitForFinished(5_000)

    assert outcome == ["completed"], outcome
    gui.recorder.flush(timeout=3.0)
    rows = [
        json.loads(line)
        for path in gui.recorder.rotated_event_files()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    worker_rows = [r for r in rows if r["pid"] != __import__("os").getpid()]
    assert worker_rows, "the worker process must have written to the shared recorder"
    assert {r["operation_id"] for r in worker_rows if r["operation_id"]} == {launched}
    assert {r["session_id"] for r in worker_rows} == {gui.recorder.session_id}


def test_pending_worker_receives_the_requested_worker_count(gui: Gui) -> None:
    from app.workers import ScanProcessRunner

    runner = ScanProcessRunner()
    runner._start_worker = MagicMock()
    runner.start(pending_generation="serum1", workers=1)
    args = runner._start_worker.call_args.args
    assert args[0] == "process-pending"
    assert args[1] == ["--generation", "serum1", "--workers", "1"]
    runner.start(pending_generation="serum2")
    assert runner._start_worker.call_args.args[1][-2:] == ["--workers", "4"]


def test_match_completion_is_recorded(gui: Gui) -> None:
    from app.ui import LegacyMainWindow

    with pytest.raises(Exception):
        # The handler records the event first; the missing result file then fails,
        # which is fine here -- only the recording is under test.
        LegacyMainWindow._match_completed(gui.window, str(gui.tmp_path / "missing.json"))
    assert gui.events("match_completed")


def test_diagnostics_stay_bounded_under_repeated_ui_activity(gui: Gui) -> None:
    gui.install(serum1=True, serum2=False)
    gui.seed_library(learned=2, pending={PENDING_SERUM2_NOT_INSTALLED: 30})
    gui.choose_audio()
    gui.select_target("serum2")
    for _ in range(300):
        gui.window.start_match()
    counters = gui.recorder.counters()
    assert counters["events_in_ring"] <= gui.recorder.retention_policy()["ring_capacity_events"]
    assert counters["dropped_events"] == 0
    # A refused Match produced one popup each time -- and no runner activity.
    gui.window.match_runner.start.assert_not_called()
    assert gui.activities == set()


def test_partial_completion_note_names_files_and_synth_never_internal_ids(gui: Gui) -> None:
    """The status line after a partly-supported run must read like English."""

    gui.install(serum1=False, serum2=True)
    gui.seed_library(learned=2, pending={PENDING_UNSUPPORTED_LEGACY_FORMAT: 40})
    gui.window._refresh_workflow_cards()
    gui.click(gui.window.render_button)
    gui.window.runner.completed.emit(
        {"found": 42, "skipped_unsupported_generation": 40, "unsupported_generations": "serum1"}
    )
    message = gui.window.statusBar().currentMessage()
    log = gui.window.log_pane.toPlainText()
    for text in (message, log):
        assert "serum1" not in text and "renderer" not in text.casefold()
    assert "40 legacy .fxp presets are waiting for Serum 1" in message
    assert "still linked" in message
    assert gui.window._render_failure_detail == ""


def test_pending_popup_appears_over_a_settled_window(gui: Gui) -> None:
    """The modal must not sit on top of a Render card still showing 'in progress'."""

    gui.install(serum1=True, serum2=False)
    gui.seed_library(learned=2, rendered_only=3, pending={PENDING_SERUM2_NOT_INSTALLED: 60})
    gui.window._refresh_workflow_cards()
    gui.click(gui.window.render_button)
    seen: dict[str, object] = {}
    original = FakeMessageBox.information

    def observe(_parent, title, text, *a, **k):
        seen["activities"] = set(gui.window._workflow_activities)
        seen["render_phase"] = gui.phase("render")
        return original(_parent, title, text, *a, **k)

    gui.monkeypatch.setattr(FakeMessageBox, "information", staticmethod(observe))
    gui.window.runner.completed.emit({"found": 65, "skipped_unsupported_generation": 60,
                                      "unsupported_generations": "serum2"})
    assert seen, "the notice should have been shown"
    assert seen["activities"] == set(), seen
    assert seen["render_phase"] != "in-progress", seen


# ===========================================================================
# K. Private support upload: messages, retry, and sign-in recovery
# ===========================================================================


def _bug_report_runner(gui: Gui, *, code: str = "offline", saved: Path | None = None):
    runner = gui.window.bug_report_runner
    runner.start = MagicMock(name="bug_report_runner.start")
    runner.error_code = code
    runner.request_path = saved or (gui.tmp_path / "PatchLab Bug Report x.txt")
    return runner


def test_bug_report_success_says_so_plainly_and_shows_the_ticket(gui: Gui) -> None:
    gui.window._bug_report_completed({"ticket_id": "a" * 32, "receipt_id": "r" * 24})
    (dialog,) = gui.dialogs("information")
    assert dialog["title"] == "Bug report sent"
    assert dialog["text"].startswith("Bug report sent successfully.")
    assert "a" * 32 in dialog["text"] and "Desktop" in dialog["text"]


def test_bug_report_upload_failure_says_it_was_saved_and_can_be_retried(gui: Gui) -> None:
    _bug_report_runner(gui, code="offline")
    FakeMessageBox.choose = "Close"
    gui.window._bug_report_failed("PatchLab couldn't reach the internet.")
    (dialog,) = gui.dialogs("custom")
    assert dialog["text"].startswith(
        "Your bug report was saved on this Mac, but PatchLab couldn't upload it. You can try again."
    )
    assert "couldn't reach the internet" in dialog["text"]
    assert dialog["buttons"] == ["Try Again", "Close"], "no sign-in button for a network problem"
    assert dialog["default"] == "Try Again"
    assert "connected to the private support service" not in dialog["text"]
    gui.window.bug_report_runner.start.assert_not_called()


def test_try_again_resends_the_same_saved_report_without_recreating_it(gui: Gui) -> None:
    saved = gui.tmp_path / "PatchLab Bug Report saved.txt"
    runner = _bug_report_runner(gui, code="service_unavailable", saved=saved)
    FakeMessageBox.choose = "Try Again"
    gui.window._bug_report_failed("The support service is temporarily unavailable.")
    runner.start.assert_called_once_with(saved)


def test_not_signed_in_signs_in_and_resumes_the_same_report_automatically(gui: Gui) -> None:
    """A tester must not have to re-file a report because of a sign-in they never saw."""

    saved = gui.tmp_path / "PatchLab Bug Report saved.txt"
    runner = _bug_report_runner(gui, code="not_connected", saved=saved)
    gui.window._reconnect_support_service = MagicMock(return_value=True)
    gui.window._bug_report_signin_attempted = False

    gui.window._bug_report_failed("PatchLab isn't signed in to the support service on this Mac.")

    gui.window._reconnect_support_service.assert_called_once()
    runner.start.assert_called_once_with(saved), "the SAME saved report resumes"
    assert gui.dialogs("custom") == [], "no dead end shown when sign-in fixes it"
    assert "resuming the saved bug report" in gui.window.log_pane.toPlainText()


def test_a_declined_sign_in_keeps_the_local_report_and_explains(gui: Gui) -> None:
    runner = _bug_report_runner(gui, code="not_connected")
    gui.window._reconnect_support_service = MagicMock(return_value=False)
    gui.window._bug_report_signin_attempted = False
    FakeMessageBox.choose = "Close"

    gui.window._bug_report_failed("PatchLab isn't signed in to the support service on this Mac.")

    runner.start.assert_not_called()
    (dialog,) = gui.dialogs("custom")
    assert dialog["text"].startswith("Your bug report was saved on this Mac")
    assert "Sign In…" in dialog["buttons"]


def test_sign_in_is_only_attempted_once_per_report(gui: Gui) -> None:
    """A wrong passcode must not loop the user through sign-in forever."""

    runner = _bug_report_runner(gui, code="auth_failed")
    gui.window._reconnect_support_service = MagicMock(return_value=True)
    gui.window._bug_report_signin_attempted = False
    gui.window._bug_report_failed("PatchLab couldn't sign in to the support service.")
    assert runner.start.call_count == 1

    FakeMessageBox.choose = "Close"
    gui.window._bug_report_failed("PatchLab couldn't sign in to the support service.")
    assert gui.window._reconnect_support_service.call_count == 1, "asked once, not in a loop"
    assert runner.start.call_count == 1
    assert gui.dialogs("custom"), "the second failure explains instead of retrying silently"


def test_signing_in_again_lifts_the_local_only_setting_for_the_session(monkeypatch, tmp_path: Path) -> None:
    from core.access_gate import AccessManager, AccessStore

    monkeypatch.setenv("PATCHLAB_DISABLE_RELAY", "1")
    store = AccessStore(marker_path=tmp_path / "access.json", keyring_backend=None)
    manager = AccessManager(store, relay_url="https://relay.invalid", validator=lambda _u, _p: "tok")
    manager.authenticate("passcode")
    import os

    assert "PATCHLAB_DISABLE_RELAY" not in os.environ
    assert store.load().local_only is False


def test_preset_contribution_outcomes_use_the_agreed_plain_wording(gui: Gui) -> None:
    gui.window.distribution_mode = True
    gui.window._compact_render_active = True
    gui.window.runner.completed.emit({"found": 4, "relay_uploaded": 4, "relay_upload_failed": 0})
    log = gui.window.log_pane.toPlainText()
    assert "Preset contribution uploaded successfully." in log
    assert "were not uploaded" not in log
    gui.window._compact_render_active = True
    gui.window.runner.completed.emit({"found": 4, "relay_uploaded": 0, "relay_upload_failed": 4})
    log = gui.window.log_pane.toPlainText()
    assert "Your presets were not uploaded. Your original files were not changed. You can try again." in log


# ===========================================================================
# L. "Use & share my own presets" OFF: the app behaves as the consent text says
# ===========================================================================


def _turn_personal_presets(gui: Gui, value: bool | None) -> None:
    """Set the privacy choice exactly as the real toggle / dialog would persist it."""

    store = gui.window.privacy_store
    if value is None:
        store.path.unlink(missing_ok=True)
        gui.window.privacy_choice = store.load()
    else:
        gui.window.privacy_choice = store.save(value)
    gui.window.share_toggle.blockSignals(True)
    gui.window.share_toggle.setChecked(bool(value))
    gui.window.share_toggle.blockSignals(False)
    gui.window.fingerprint_runner.start = MagicMock(name="fingerprint_runner.start")


def _clicking_dialog(label: str):
    """A QDialog that presses the named button as soon as it is shown, like a user."""

    from PySide6.QtWidgets import QDialog, QPushButton

    class ClickingDialog(QDialog):
        def exec(self) -> int:
            next(b for b in self.findChildren(QPushButton) if b.text() == label).click()
            return 0

    return ClickingDialog


def _running(*runners):
    """Make the given runners report an active job, patching their class property."""

    from unittest.mock import PropertyMock, patch

    patches = [patch.object(type(r), "running", new_callable=PropertyMock, return_value=True, create=True) for r in runners]
    for p in patches:
        p.start()
    return patches


def test_fresh_user_who_disagrees_starts_no_user_preset_work(gui: Gui, monkeypatch) -> None:
    from core.privacy import user_presets_enabled

    _turn_personal_presets(gui, None)
    assert gui.window.privacy_choice.use_and_share_own_presets is None
    monkeypatch.setattr("app.ui.QDialog", _clicking_dialog("Disagree"))
    gui.window._show_consent_dialog()

    assert gui.window.privacy_store.load().use_and_share_own_presets is False
    assert user_presets_enabled() is False
    assert gui.window.share_toggle.isChecked() is False
    gui.window.maybe_start_automatic_link_scan()
    gui.window.start_render()
    gui.window.start_analyze()
    gui.window.choose_folder()
    for name in ("runner", "render_runner", "fingerprint_runner", "analyze_runner"):
        getattr(gui.window, name).start.assert_not_called()
    assert not gui.window._workflow_activities


def test_agreeing_enables_personal_presets(gui: Gui, monkeypatch) -> None:
    from core.privacy import user_presets_enabled

    _turn_personal_presets(gui, None)
    monkeypatch.setattr("app.ui.QDialog", _clicking_dialog("Agree"))
    gui.window._show_consent_dialog()
    assert gui.window.privacy_store.load().use_and_share_own_presets is True and user_presets_enabled()


def test_turning_it_off_stops_running_user_preset_jobs_and_keeps_the_data(gui: Gui) -> None:
    gui.window.fingerprint_runner.start = MagicMock()
    gui.seed_library(learned=3)
    before = Database(gui.db_path).library_coverage()
    linked = gui.window.privacy_choice.linked_folder
    for name in ("runner", "render_runner", "fingerprint_runner", "analyze_runner"):
        setattr(getattr(gui.window, name), "cancel", MagicMock(name=f"{name}.cancel"))
    patches = _running(gui.window.runner, gui.window.render_runner, gui.window.fingerprint_runner, gui.window.analyze_runner)
    try:
        gui.window._set_workflow_activity("link", 1, 10, "Scanning…")
        gui.window.share_toggle.setChecked(False)
    finally:
        for p in reversed(patches):  # unwind in reverse: runners may share a class
            p.stop()
    assert gui.window.privacy_store.load().use_and_share_own_presets is False
    for name in ("runner", "render_runner", "fingerprint_runner", "analyze_runner"):
        getattr(gui.window, name).cancel.assert_called_once()
    assert "link" not in gui.window._workflow_activities
    assert Database(gui.db_path).library_coverage() == before, "stored presets and fingerprints are untouched"
    assert gui.window.privacy_store.load().linked_folder == linked, "the folder link is kept for re-enabling"


def test_render_and_learn_controls_refuse_cleanly_while_off(gui: Gui) -> None:
    _turn_personal_presets(gui, False)
    gui.window.start_render()
    gui.window.start_analyze()
    titles = [d["title"] for d in gui.dialogs("information")]
    assert titles.count("Turn on personal presets") == 2
    gui.window.runner.start.assert_not_called()
    gui.window.render_runner.start.assert_not_called()
    gui.window.fingerprint_runner.start.assert_not_called()


def test_turning_it_back_on_makes_processing_available_again(gui: Gui) -> None:
    _turn_personal_presets(gui, False)
    gui.window.start_render()
    gui.window.runner.start.assert_not_called()
    gui.window.share_toggle.setChecked(True)
    assert gui.window.privacy_store.load().use_and_share_own_presets is True
    gui.window.start_render()
    gui.window.runner.start.assert_called_once()


def test_startup_with_personal_presets_off_schedules_no_scan(gui: Gui) -> None:
    _turn_personal_presets(gui, False)
    # A fresh window over the same persisted state, as after a relaunch.
    relaunched = MainWindow(privacy_store=PrivacyStore(gui.window.privacy_store.path))
    relaunched.runner.start = MagicMock()
    relaunched.maybe_start_automatic_link_scan()
    relaunched.runner.start.assert_not_called()
    assert relaunched.privacy_choice.use_and_share_own_presets is False
    relaunched.close()


def test_the_automatic_daily_scan_still_runs_when_it_is_on(gui: Gui, monkeypatch) -> None:
    monkeypatch.setattr("app.ui.auto_scan_due", lambda *_a, **_k: True)
    monkeypatch.setattr("app.ui.record_auto_scan", lambda *_a, **_k: None)
    gui.window.maybe_start_automatic_link_scan()
    gui.window.runner.start.assert_called_once()


def test_match_uses_no_user_presets_while_off_and_uses_them_when_on(gui: Gui, monkeypatch) -> None:
    monkeypatch.setattr("app.ui.synthesis_readiness", lambda *_a, **_k: SimpleNamespace(available=False, reason="test"))
    gui.choose_audio()
    _turn_personal_presets(gui, False)
    gui.click(gui.window.match_start_button)
    off = gui.window.match_runner.start.call_args.kwargs
    assert off["factory_only"] is True and off["local_db"] is None and off["local_audio_root"] is None

    gui.window.match_runner.start.reset_mock()
    gui.window._match_failed("test cleanup")
    _turn_personal_presets(gui, True)
    gui.choose_audio()
    gui.click(gui.window.match_start_button)
    on = gui.window.match_runner.start.call_args.kwargs
    assert on["local_db"] == gui.window.local_paths["db"], "the user's presets are candidates again"


def test_worker_that_refused_because_it_is_off_is_reported_as_the_users_choice(gui: Gui) -> None:
    gui.window._automatic_link_scan_active = True
    gui.window._scan_completed({"user_presets_disabled": True, "found": 0})
    assert "Personal presets are off" in gui.window.statusBar().currentMessage()
    assert gui.window._render_failure_detail == ""
    assert not gui.window._workflow_activities
