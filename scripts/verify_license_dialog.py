#!/usr/bin/env python3
"""Render and structurally verify the required license agreement dialog."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["PATCHLAB_DISTRIBUTION_MODE"] = "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication, QDialog, QPushButton  # noqa: E402

from app.license_dialog import (  # noqa: E402
    COPYRIGHT_HOLDER,
    LICENSE_AGREEMENT_TEXT,
    LicenseAgreementDialog,
)
from core.access_gate import AccessStore  # noqa: E402
from core.privacy import PrivacyStore  # noqa: E402


SCREENSHOT = (
    PROJECT_ROOT
    / "data"
    / "models"
    / "license_dialog"
    / "license-agreement.png"
)


class MemoryKeyring:
    def get_password(self, _service: str, _account: str) -> None:
        return None

    def set_password(self, _service: str, _account: str, _value: str) -> None:
        return None

    def delete_password(self, _service: str, _account: str) -> None:
        return None


def main() -> int:
    application = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="patchlab-license-dialog-") as directory:
        store = AccessStore(
            marker_path=Path(directory) / "access-state.json",
            keyring_backend=MemoryKeyring(),
        )
        dialog = LicenseAgreementDialog(store)
        dialog.resize(680, 560)
        dialog.show()
        application.processEvents()
        SCREENSHOT.parent.mkdir(parents=True, exist_ok=True)
        screenshot_saved = dialog.grab().save(str(SCREENSHOT))
        payload = {
            "copyright_holder": COPYRIGHT_HOLDER,
            "exact_agreement_text": (
                dialog.agreement_text.toPlainText() == LICENSE_AGREEMENT_TEXT
            ),
            "scrollable_read_only_area": dialog.agreement_text.isReadOnly(),
            "agree_button": dialog.agree_button.text() == "I Agree",
            "decline_button": dialog.decline_button.text() == "Decline",
            "not_accepted_before_click": store.needs_license_agreement(),
            "screenshot_saved": bool(screenshot_saved and SCREENSHOT.is_file()),
            "screenshot": str(SCREENSHOT),
        }
        dialog.agree_button.click()
        accepted = store.load()
        payload["accepted_after_click"] = (
            accepted.agreed_to_license
            and bool(accepted.license_accepted_at)
        )
        viewer = LicenseAgreementDialog(read_only_view=True)
        payload["settings_view_is_read_only"] = (
            viewer.read_only_view
            and viewer.agreement_text.isReadOnly()
            and not hasattr(viewer, "agree_button")
        )
        viewer.close()
        dialog.close()
        from app.ui import MainWindow

        privacy = PrivacyStore(Path(directory) / "privacy-settings.json")
        privacy.save(False)
        window = MainWindow(privacy_store=privacy)
        settings_buttons: list[str] = []
        original_exec = QDialog.exec

        def capture_settings(current: QDialog) -> int:
            settings_buttons.extend(
                button.text()
                for button in current.findChildren(QPushButton)
            )
            return int(QDialog.DialogCode.Accepted)

        QDialog.exec = capture_settings
        try:
            window.open_settings()
        finally:
            QDialog.exec = original_exec
            window.close()
        payload["settings_has_view_license"] = (
            "View License Agreement" in settings_buttons
        )
    payload["gate_pass"] = all(
        value
        for key, value in payload.items()
        if key
        not in {
            "copyright_holder",
            "screenshot",
        }
    )
    print("LICENSE_DIALOG_GATE=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
