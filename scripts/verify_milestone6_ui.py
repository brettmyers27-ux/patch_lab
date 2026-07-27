#!/usr/bin/env python3
"""Offscreen consent and factory-only startup gate."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PATCHLAB_DISTRIBUTION_MODE"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from app.ui import MainWindow
from core.db import DEFAULT_DB_PATH
from core.factory_verify import FactoryVerification, verify_local_factory_install
from core.privacy import PrivacyStore


REPORT = PROJECT_ROOT / "data" / "models" / "milestone6_ui_report.json"


def render_count() -> int:
    with sqlite3.connect(DEFAULT_DB_PATH) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0])


def main() -> int:
    application = QApplication.instance() or QApplication([])
    before_renders = render_count()
    with tempfile.TemporaryDirectory(prefix="patchlab-m6-ui-") as temporary:
        privacy = PrivacyStore(Path(temporary) / "privacy.json")
        no_factory = FactoryVerification(
            bundle_available=True,
            factory_directories_found=0,
            local_files_found=0,
            known_bundle_hashes=1167,
            matched_hashes=0,
            missing_hashes=("fixture",),
            unknown_local_hashes=(),
            local_paths_by_hash={},
            elapsed_s=0.003,
        )
        modal_seen = False
        equal_weight = False

        def answer_disagree() -> None:
            nonlocal modal_seen, equal_weight
            dialogs = [
                widget
                for widget in application.topLevelWidgets()
                if isinstance(widget, QDialog) and widget.isVisible()
            ]
            if not dialogs:
                QTimer.singleShot(20, answer_disagree)
                return
            dialog = dialogs[0]
            buttons = {
                button.text(): button
                for button in dialog.findChildren(QPushButton)
            }
            modal_seen = {"Agree", "Disagree"} <= buttons.keys()
            equal_weight = (
                buttons["Agree"].minimumHeight()
                == buttons["Disagree"].minimumHeight()
            )
            buttons["Disagree"].click()

        QTimer.singleShot(20, answer_disagree)
        window = MainWindow(
            factory_verification=no_factory, privacy_store=privacy
        )
        window.show()
        application.processEvents()
        disagreed = privacy.load().use_and_share_own_presets is False
        no_factory_launch = (
            window.match_button.isEnabled()
            and not window.scan_box.isEnabled()
            and "No local Serum factory" in window.factory_status.text()
        )
        window.share_toggle.setChecked(True)
        application.processEvents()
        enabled_after_on = window.scan_box.isEnabled()
        window.share_toggle.setChecked(False)
        application.processEvents()
        disabled_after_off = not window.scan_box.isEnabled()
        window.close()

        real = verify_local_factory_install(
            mapping_path=Path(temporary) / "factory-paths.json"
        )
        second_privacy = PrivacyStore(Path(temporary) / "privacy-2.json")
        second_privacy.save(False)
        factory_window = MainWindow(
            factory_verification=real, privacy_store=second_privacy
        )
        factory_window.show()
        application.processEvents()
        factory_disagree_ready = (
            factory_window.match_button.isEnabled()
            and not factory_window.scan_box.isEnabled()
            and "No rendering was needed" in factory_window.factory_status.text()
        )
        factory_window.close()
    after_renders = render_count()
    payload = {
        "first_launch_modal_seen": modal_seen,
        "agree_disagree_equal_weight": equal_weight,
        "disagree_persisted": disagreed,
        "fresh_no_factory_launch_pass": no_factory_launch,
        "toggle_on_enables_link": enabled_after_on,
        "toggle_off_disables_link": disabled_after_off,
        "factory_disagree_ready": factory_disagree_ready,
        "factory_verification_elapsed_s": real.elapsed_s,
        "factory_matched": real.matched_hashes,
        "factory_known": real.known_bundle_hashes,
        "render_rows_before": before_renders,
        "render_rows_after": after_renders,
        "first_launch_render_wait": after_renders != before_renders,
    }
    payload["gate_pass"] = (
        modal_seen
        and equal_weight
        and disagreed
        and no_factory_launch
        and enabled_after_on
        and disabled_after_off
        and factory_disagree_ready
        and real.elapsed_s < 3.0
        and after_renders == before_renders
    )
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("MILESTONE6_UI_REPORT=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
