#!/usr/bin/env python3
"""Exercise PatchLab's real GUI handlers for the release-critical flows.

Runs in a source checkout or -- through the packaged worker dispatcher
(``--patchlab-worker release-flow-gate``) -- inside the frozen ``PatchLab.app``,
so the exact handlers, widgets and signals a user reaches are what is tested.

It follows the repository's existing packaged-gate convention
(``verify_workflow_cards.py`` and friends): a real ``MainWindow`` is built
offscreen, so no window ever appears on screen. Worker QProcesses, modal dialogs
and the machine's Serum installation are replaced by recorders/simulations; every
other line is production code. No Serum GUI, DAW or mouse automation is involved.

Prints one ``RELEASE_FLOW_CHECK`` line per assertion and a final
``RELEASE_FLOW_GATE=`` JSON summary; exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    print(
        f"RELEASE_FLOW_CHECK {'PASS' if condition else 'FAIL'} {name}"
        + (f" [{detail}]" if detail else ""),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["PATCHLAB_DISTRIBUTION_MODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="patchlab-release-flow-") as temporary:
        profile = Path(temporary)
        os.environ["PATCHLAB_APP_DATA"] = str(profile / "app-data")
        os.environ["PATCHLAB_PRIVACY_SETTINGS"] = str(profile / "privacy.json")
        # An orchestrator may hand in a shared diagnostics directory so one support
        # bundle can span this GUI session and the worker processes it ran with.
        os.environ.setdefault("PATCHLAB_DIAGNOSTICS_DIR", str(profile / "diagnostics"))
        _run(profile, args.output)

    failed = [name for name, ok, _detail in CHECKS if not ok]
    print(
        "RELEASE_FLOW_GATE="
        + json.dumps(
            {"passed": len(CHECKS) - len(failed), "failed": failed}, sort_keys=True
        ),
        flush=True,
    )
    return 1 if failed else 0


def _run(profile: Path, output: Path | None) -> None:
    from PySide6.QtWidgets import QApplication, QMessageBox

    import app.ui as ui
    import core.capability_ux as capability_ux
    import core.renderer_selection as selection_module
    import core.synth_capability as capability_module
    from core.db import Database
    from core.diagnostics import recorder
    from core.platform_env import ENV, PluginCandidate
    from core.privacy import PrivacyStore
    from core.storage import StoragePreferences

    application = QApplication.instance() or QApplication([])
    shown: list[dict] = []

    class Calls:
        """Stands in for a worker's start(): records how the UI invoked it."""

        def __init__(self) -> None:
            self.calls: list[tuple[tuple, dict]] = []

        def __call__(self, *args, **kwargs) -> None:
            self.calls.append((args, kwargs))

        @property
        def call_count(self) -> int:
            return len(self.calls)

        def reset_mock(self) -> None:
            self.calls.clear()

        def assert_called_once(self) -> None:
            assert len(self.calls) == 1, f"expected one call, saw {len(self.calls)}"

    class Box:
        """Records dialogs; answers the offer with the scripted choice."""

        ButtonRole = QMessageBox.ButtonRole
        StandardButton = QMessageBox.StandardButton
        choice = "Not Now"

        def __init__(self, _parent=None) -> None:
            self.title = self.text = ""
            self.buttons: dict[str, object] = {}
            self.clicked = None

        def setWindowTitle(self, value):
            self.title = value

        def setText(self, value):
            self.text = value

        def addButton(self, label, _role=None):
            self.buttons[label] = object()
            return self.buttons[label]

        def setDefaultButton(self, _button):
            pass

        def exec(self):
            self.clicked = self.buttons.get(Box.choice)
            shown.append(
                {"kind": "offer", "title": self.title, "text": self.text,
                 "buttons": list(self.buttons)}
            )

        def clickedButton(self):
            return self.clicked

        @staticmethod
        def _note(kind, title, text):
            shown.append({"kind": kind, "title": title, "text": text})
            return QMessageBox.StandardButton.Ok

        information = staticmethod(lambda _p, t, x, *a, **k: Box._note("information", t, x))
        warning = staticmethod(lambda _p, t, x, *a, **k: Box._note("warning", t, x))
        critical = staticmethod(lambda _p, t, x, *a, **k: Box._note("critical", t, x))

    ui.QMessageBox = Box
    ui.append_runtime_log = lambda *_a, **_k: None
    ui.storage_status = lambda: SimpleNamespace(
        available=True, reason="", root=profile, configured_external=False
    )
    ui.synthesis_readiness = lambda *_a, **_k: SimpleNamespace(available=True, reason="")

    import core.workflow_state as workflow_state

    workflow_state._match_prerequisite_error = lambda *_a, **_k: ""

    def machine(serum1: bool, serum2: bool) -> None:
        """Simulate which Serums are installed, without touching the real ones."""

        root = profile / f"plugins-{int(serum1)}{int(serum2)}"
        candidates = []
        for candidate in ENV.plugin_candidates:
            wanted = (candidate.synth == "serum1" and serum1) or (
                candidate.synth == "serum2" and serum2
            )
            path = root / f"{candidate.synth}-{candidate.format}-{candidate.path.name}"
            if wanted and candidate.format in {"VST2", "VST3"} and candidate.path.suffix != ".component":
                path.mkdir(parents=True, exist_ok=True)
            candidates.append(PluginCandidate(candidate.synth, candidate.format, path, candidate.hostable))
        import dataclasses

        env = dataclasses.replace(ENV, plugin_candidates=tuple(candidates))
        capability_module.ENV = env
        selection_module.ENV = env

    capability_ux._state_path = lambda env=None: profile / "notices.json"

    privacy = PrivacyStore(profile / "privacy.json")
    linked = profile / "Serum 2 Presets"
    linked.mkdir()
    privacy.save(True, linked_folder=linked)
    window = ui.MainWindow(privacy_store=privacy)
    window.local_paths = {
        **window.local_paths,
        "db": profile / "library.db",
        "audio": profile / "audio",
        "states": profile / "states",
    }
    window.storage_preferences = StoragePreferences(compact_mode=True)
    window._model_asset_error = None
    for name in ("runner", "match_runner", "render_runner", "analyze_runner"):
        setattr(getattr(window, name), "start", Calls())

    def card(name: str) -> tuple[str, str]:
        widget = window.hero_cards[("link", "render", "analyze", "match").index(name)]
        return str(widget.property("workflowState")), widget.status.text()

    def controls_usable() -> bool:
        return all(
            (
                window.match_start_button.isEnabled(),
                window.match_synth.isEnabled(),
                window.match_budget.isEnabled(),
                window.match_offset.isEnabled(),
                not window.match_cancel_button.isEnabled(),
            )
        )

    def select_target(generation: str) -> None:
        segmented = window.match_synth
        for index in range(segmented.count()):
            if segmented._items[index][1] == generation:
                segmented.setCurrentIndex(index)

    database = Database(window.local_paths["db"])
    counter = [0]

    def seed(*, learned=0, unprocessed=0, pending: dict[str, int] | None = None) -> None:
        def add(suffix, synth, renderers):
            counter[0] += 1
            path = linked / f"p{counter[0]}{suffix}"
            path.write_bytes(b"x" + str(counter[0]).encode())
            preset_id, _ = database.insert_preset(
                path=path, name=path.stem, synth=synth, content_hash=f"gate-{counter[0]}"
            )
            database.record_identity(
                preset_id,
                file_format="fxp" if suffix == ".fxp" else "serumpreset",
                provenance="serum2_factory",
                compatible_renderers=renderers,
            )
            return preset_id

        for _ in range(learned):
            preset_id = add(".SerumPreset", "serum2", ("serum2",))
            database.upsert_fingerprint(preset_id, 0, bytes(2048), bytes(64))
        for _ in range(unprocessed):
            add(".SerumPreset", "serum2", ("serum2",))
        for reason, count in (pending or {}).items():
            legacy = reason in {"serum1_not_installed", "unsupported_legacy_format"}
            for _ in range(count):
                database.set_pending_reason(
                    add(".fxp", "serum1", ("serum1",)) if legacy else add(".SerumPreset", "serum2", ("serum2",)),
                    reason,
                )

    audio = profile / "sound.wav"
    audio.write_bytes(b"RIFF....WAVE")
    window._match_audio_path = audio

    # ---- Link / Render (bug B) -------------------------------------------
    machine(serum1=False, serum2=True)
    seed(learned=2, unprocessed=4, pending={"unsupported_legacy_format": 30})
    window._refresh_workflow_cards()
    check("link card complete for a linked folder", card("link")[0] == "complete", card("link")[1])
    window.render_button.click()
    window.runner.start.assert_called_once()
    check("render click activates the Render card", card("render")[0] == "in-progress")
    check("render click does NOT restart Link", card("link")[0] == "complete")
    window.runner.stage_progress.emit({"stage": "scan", "current": 3, "total": 6, "text": "Scanning 3 of 6 presets"})
    check("catalog progress lands on Render, not Link",
          card("render")[1] == "Scanning 3 of 6 presets" and card("link")[0] == "complete")
    window.runner.failed.emit("RendererUnavailableError: No usable serum1 renderer is available. Tried: x")
    check("render failure clears activity", not window._workflow_activities)
    check("render failure leaves Link complete", card("link")[0] == "complete")
    check("render failure is retryable", card("render")[0] == "failed" and "still linked" in window._render_failure_detail)
    check("render failure text has no internal jargon",
          not any(word in window._render_failure_detail.casefold() for word in ("renderer", "tried:", "error:")))
    check("learned presets survive the failure", database.library_coverage()["learned"] == 2)

    # ---- Missing-synth notice wording ------------------------------------
    shown.clear()
    window._notify_pending_after_scan()
    notices = [d for d in shown if d["kind"] == "information"]
    check("one missing-synth notice", len(notices) == 1)
    body = notices[0]["text"] if notices else ""
    check("legacy .fxp named by format", "30 legacy .fxp presets" in body and "Serum 1" in body, body[:90])
    check("legacy .fxp never called Serum 1/2 presets",
          "30 Serum 1 presets" not in body and "30 Serum 2 presets" not in body)
    window._notify_pending_after_scan()
    check("notice is not repeated", len([d for d in shown if d["kind"] == "information"]) == 1)

    # ---- Output gating ------------------------------------------------------
    window.runner.failed.emit("done")  # settle
    machine(serum1=True, serum2=False)
    select_target("serum2")
    shown.clear()
    window.match_start_button.click()
    check("blocked Match never starts the worker", window.match_runner.start.call_count == 0)
    check("blocked Match shows the concise popup",
          any(d["text"] == "Serum 2 is required to create Serum 2 presets. Install Serum 2 and try again." for d in shown))
    check("blocked Match starts no loading state", "match" not in window._workflow_activities)
    check("blocked Match leaves every control usable", controls_usable())
    select_target("serum1")
    window.match_start_button.click()
    check("other target still works", window.match_runner.start.call_count == 1)
    window.match_runner.failed.emit("PatchLab couldn't finish this match. Your sound is still selected — you can try again.")

    # ---- Newly installed synth + pending presets --------------------------
    window.match_runner.start.reset_mock()
    window.runner.start.reset_mock()  # the earlier Render click already used it
    seed(pending={"serum2_not_installed": 60})
    select_target("serum2")
    machine(serum1=True, serum2=True)
    shown.clear()
    Box.choice = "Not Now"
    window.match_start_button.click()
    offers = [d for d in shown if d["kind"] == "offer"]
    check("newly available synth offers processing", len(offers) == 1 and offers[0]["buttons"] == ["Process Now", "Not Now"])
    check("offer names native presets accurately", offers and "60 Serum 2 presets" in offers[0]["text"])
    check("Not Now still starts the Match in the same click", window.match_runner.start.call_count == 1)
    check("Not Now starts no processing", window.runner.start.call_count == 0)
    window.match_runner.failed.emit("x")
    check("controls usable after Not Now + failure", controls_usable())
    shown.clear()
    window.match_start_button.click()
    window.match_runner.failed.emit("x")
    check("Not Now is not nagged again", not [d for d in shown if d["kind"] == "offer"])

    # ---- Match failure recovery ------------------------------------------------
    window.match_start_button.click()
    window.match_runner.failed.emit("ZeroDivisionError: division by zero /Users/x/very/long/path.py")
    shown_text = window.match_stats.text()
    check("match failure clears activity", "match" not in window._workflow_activities)
    check("match failure recovers controls", controls_usable())
    check("match failure keeps the user's sound", window._match_audio_path == audio)
    check("match failure message is concise", "ZeroDivisionError" not in shown_text and "/Users/" not in shown_text, shown_text)
    check("match failure detail is retained in the log", "ZeroDivisionError" in window.log_pane.toPlainText())

    # A credential-shaped string in a worker's error must not survive into diagnostics.
    window.match_start_button.click()
    window.match_runner.failed.emit("RuntimeError: relay token=abc123secret was rejected")

    # ---- Diagnostics from the UI ------------------------------------------------
    recorder().flush(timeout=3.0)
    kinds = {event["event_type"] for event in recorder().recent_events()}
    wanted = {
        "render_requested", "render_started", "render_failed", "match_requested",
        "match_started", "match_failed", "output_blocked", "pending_processing_offered",
        "notice_acknowledged", "workflow_state_changed", "capability_refresh_completed",
    }
    check("UI decisions recorded as structured events", wanted <= kinds, f"missing {sorted(wanted - kinds)}")
    failed_event = [e for e in recorder().recent_events() if e["event_type"] == "render_failed"][-1]["fields"]
    check("render_failed records the recovery state",
          failed_event["link_phase"] == "complete" and failed_event["retry_available"] is True)

    # ---- A support bundle written from the same session -------------------------
    from core.support_bundle import create_support_bundle

    bundle = create_support_bundle(
        ticket_id="a" * 32,
        comments="release-flow gate representative failure",
        directory=(output or profile) / "release-flow-bundle",
        operation="render-sound-library",
        settings=window._diagnostic_settings(),
        archive=False,
    )
    names = sorted(path.name for path in bundle.files)
    check("support bundle written from the GUI session", len(names) == 6, ",".join(names))
    blob = " ".join(path.read_text(errors="replace") for path in bundle.files)
    check("credential-shaped text is redacted from every bundle file", "abc123secret" not in blob)
    settings = json.loads((bundle.directory / "reproduction.json").read_text())["settings"]
    check("bundle records UI recovery state", "ui_recovery" in settings and settings["ui_recovery"].get("link"))
    window.close()


if __name__ == "__main__":
    raise SystemExit(main())
