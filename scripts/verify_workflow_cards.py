#!/usr/bin/env python3
"""Capture and verify truthful workflow-card states in dev or a frozen app."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _card_snapshot(window) -> dict[str, dict[str, object]]:  # type: ignore[no-untyped-def]
    names = ("link", "render", "analyze", "match")
    return {
        name: {
            "phase": str(card.property("workflowState")),
            "text": card.status.text(),
            "current": card.progress.value(),
            "total": card.progress.maximum(),
        }
        for name, card in zip(names, window.hero_cards, strict=True)
    }


def _capture(window, application, output: Path, name: str) -> dict:  # type: ignore[no-untyped-def]
    application.processEvents()
    path = output / f"{name}.png"
    window.grab().save(str(path))
    return {"screenshot": str(path), "cards": _card_snapshot(window)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/workflow_cards"),
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["PATCHLAB_DISTRIBUTION_MODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="patchlab-workflow-profile-") as temporary:
        profile = Path(temporary)
        os.environ["PATCHLAB_APP_DATA"] = str(profile / "app-data")
        os.environ["PATCHLAB_PRIVACY_SETTINGS"] = str(profile / "privacy.json")

        from PySide6.QtWidgets import QApplication

        from app.ui import MainWindow
        from core.build_info import current_build_info
        from core.db import Database
        from core.factory_bundle import FactoryBundle
        from core.factory_verify import FactoryVerification
        from core.privacy import PrivacyStore
        from core.render import MIDI_NOTES

        application = QApplication.instance() or QApplication([])
        privacy = PrivacyStore(profile / "privacy.json")
        # Declined/disabled personal sharing is still a fresh, unlinked
        # library state, and avoids opening the first-run modal during an
        # unattended screenshot gate.
        privacy.save(False)
        known_factory = len(FactoryBundle().known_hashes())
        verification = FactoryVerification(
            bundle_available=True,
            factory_directories_found=0,
            local_files_found=0,
            known_bundle_hashes=known_factory,
            matched_hashes=0,
            missing_hashes=(),
            unknown_local_hashes=(),
            local_paths_by_hash={},
            elapsed_s=0.0,
        )
        window = MainWindow(
            privacy_store=privacy,
            factory_verification=verification,
        )
        window.resize(1440, 810)
        window.show()
        scenarios: dict[str, dict] = {}

        scenarios["fresh"] = _capture(
            window, application, output, "01-fresh-install"
        )

        linked = profile / "Linked Presets"
        linked.mkdir()
        window.privacy_choice = privacy.save(True, linked_folder=linked)
        database = Database(window.local_paths["db"])
        preset_ids: list[int] = []
        for index in range(3):
            preset_path = linked / f"Preset {index + 1}.fxp"
            preset_path.write_bytes(b"CcnK" + bytes([index + 1]) * 32)
            preset_id, _inserted = database.insert_preset(
                path=preset_path,
                name=preset_path.stem,
                synth="serum1",
                content_hash=f"workflow-fixture-{index}",
            )
            preset_ids.append(preset_id)
        with database.connect() as connection:
            connection.executemany(
                "INSERT INTO renders(preset_id,midi_note,wav_path,peak_dbfs,rms_dbfs,duration_s) "
                "VALUES (?,?,?,?,?,?)",
                [
                    (
                        preset_ids[0],
                        note,
                        str(profile / "audio" / str(preset_ids[0]) / f"{note}.wav"),
                        -1.0,
                        -12.0,
                        5.0,
                    )
                    for note in MIDI_NOTES
                ],
            )
        window._refresh_workflow_cards()
        scenarios["linked"] = _capture(
            window, application, output, "02-linked-with-work-remaining"
        )

        progress_checks: dict[str, bool] = {}
        window._local_library_progress_changed(
            {"stage": "scan", "current": 1, "total": 3, "text": "Scanning 1 of 3 presets"}
        )
        first = window.scan_progress.value()
        window._local_library_progress_changed(
            {"stage": "scan", "current": 2, "total": 3, "text": "Scanning 2 of 3 presets"}
        )
        progress_checks["link_advances"] = window.scan_progress.value() > first
        window._set_workflow_activity("render", 14, 21, "Rendering 14 of 21 notes")
        progress_checks["render_counts"] = (
            window.render_progress.value(),
            window.render_progress.maximum(),
        ) == (14, 21)
        window._set_workflow_activity("analyze", 2, 3, "Learning 2 of 3 linked presets")
        progress_checks["analyze_counts"] = (
            window.learn_progress.value(),
            window.learn_progress.maximum(),
        ) == (2, 3)
        window._set_workflow_activity("match", 24, 50, "Optimizing 24 of 50 evaluations")
        progress_checks["match_counts"] = (
            window.match_progress.value(),
            window.match_progress.maximum(),
        ) == (24, 50)
        scenarios["in_progress"] = _capture(
            window, application, output, "03-all-live-progress-states"
        )
        window._workflow_activities.clear()

        with database.connect() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO renders("
                "preset_id,midi_note,wav_path,peak_dbfs,rms_dbfs,duration_s"
                ") VALUES (?,?,?,?,?,?)",
                [
                    (
                        preset_id,
                        note,
                        str(profile / "audio" / str(preset_id) / f"{note}.wav"),
                        -1.0,
                        -12.0,
                        5.0,
                    )
                    for preset_id in preset_ids
                    for note in MIDI_NOTES
                ],
            )
        audio = profile / "query.wav"
        audio.write_bytes(b"workflow-card-fixture")
        window._match_audio_path = audio
        window._workflow_last_match_complete = True
        window._refresh_workflow_cards()
        scenarios["complete"] = _capture(
            window, application, output, "04-complete"
        )

        original_cache = os.environ.get("PATCHLAB_MODEL_CACHE")
        broken_cache = profile / "empty-model-cache"
        broken_cache.mkdir()
        os.environ["PATCHLAB_MODEL_CACHE"] = str(broken_cache)
        window._workflow_last_match_complete = False
        window._refresh_workflow_cards()
        scenarios["broken_cache"] = _capture(
            window, application, output, "05-broken-model-cache"
        )
        if original_cache is None:
            os.environ.pop("PATCHLAB_MODEL_CACHE", None)
        else:
            os.environ["PATCHLAB_MODEL_CACHE"] = original_cache

        window.close()

    expected = {
        "fresh": {
            "link": "needs-action",
            "render": "not-required",
            "analyze": "complete",
            "match": "needs-action",
        },
        "linked": {
            "link": "complete",
            "render": "needs-action",
            "analyze": "complete",
            "match": "needs-action",
        },
        "in_progress": {
            "link": "in-progress",
            "render": "in-progress",
            "analyze": "in-progress",
            "match": "in-progress",
        },
        "complete": {
            "link": "complete",
            "render": "complete",
            "analyze": "complete",
            "match": "complete",
        },
    }
    state_checks = {
        scenario: all(
            scenarios[scenario]["cards"][card]["phase"] == phase
            for card, phase in card_states.items()
        )
        for scenario, card_states in expected.items()
    }
    broken_text = scenarios["broken_cache"]["cards"]["match"]["text"]
    payload = {
        "runtime": "packaged" if getattr(sys, "frozen", False) else "development",
        "build": current_build_info().as_dict(),
        "profile": "clean temporary app-data while developer repository remains populated",
        "scenarios": scenarios,
        "state_checks": state_checks,
        "progress_checks": progress_checks,
        "broken_cache_specific": (
            scenarios["broken_cache"]["cards"]["match"]["phase"] == "needs-action"
            and "Tokenizer cache missing" in str(broken_text)
        ),
        "analyze_incremental_limitation": (
            "linked presets join search"
            in str(scenarios["linked"]["cards"]["analyze"]["text"])
        ),
        "developer_paths_used_for_card_state": False,
    }
    payload["gate_pass"] = all(
        (
            *state_checks.values(),
            *progress_checks.values(),
            payload["broken_cache_specific"],
            payload["analyze_incremental_limitation"],
        )
    )
    report = output / "report.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("WORKFLOW_CARD_GATE=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
