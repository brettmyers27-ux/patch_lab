"""Required first-run PatchLab license agreement."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from app import theme
from core.access_gate import AccessStore


COPYRIGHT_HOLDER = "Brett Myers"
LICENSE_AGREEMENT_TEXT = f"""PatchLab License Agreement

PatchLab and its source code, models, and associated materials are the
proprietary property of {COPYRIGHT_HOLDER}. All rights are reserved.

By clicking "I Agree," you agree that you will not copy, modify, reverse-
engineer, decompile, redistribute, or use this software or its source
code to create a derivative or competing product, in whole or in part,
without prior written permission.

{COPYRIGHT_HOLDER} reserves the right to pursue all remedies
available under applicable law for any violation of this agreement,
including injunctive relief and monetary damages.

If you do not agree, do not use this software."""


class LicenseAgreementDialog(QDialog):
    def __init__(
        self,
        store: AccessStore | None = None,
        parent=None,  # type: ignore[no-untyped-def]
        *,
        read_only_view: bool = False,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.read_only_view = read_only_view
        self.setWindowTitle("PatchLab License Agreement")
        self.setModal(True)
        self.setMinimumSize(620, 500)
        self.setStyleSheet(theme.load_stylesheet())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        title = QLabel("PatchLab License Agreement")
        title.setStyleSheet("font-size: 21px; font-weight: 750;")
        layout.addWidget(title)

        self.agreement_text = QTextEdit()
        self.agreement_text.setObjectName("licenseAgreementText")
        self.agreement_text.setReadOnly(True)
        self.agreement_text.setPlainText(LICENSE_AGREEMENT_TEXT)
        self.agreement_text.setStyleSheet(
            "QTextEdit {"
            " background: #111827;"
            " color: #F8FAFC;"
            " border: 1px solid #273653;"
            " border-radius: 8px;"
            " padding: 12px;"
            " font-size: 14px;"
            "}"
        )
        layout.addWidget(self.agreement_text, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        if read_only_view:
            close = QPushButton("Close")
            close.setObjectName("primaryButton")
            close.clicked.connect(self.accept)
            buttons.addWidget(close)
        else:
            self.decline_button = QPushButton("Decline")
            self.decline_button.clicked.connect(self.reject)
            self.agree_button = QPushButton("I Agree")
            self.agree_button.setObjectName("primaryButton")
            self.agree_button.clicked.connect(self._agree)
            buttons.addWidget(self.decline_button)
            buttons.addWidget(self.agree_button)
        layout.addLayout(buttons)

    def _agree(self) -> None:
        if self.store is None:
            raise RuntimeError("A persistent access store is required for acceptance")
        self.store.accept_license()
        self.accept()
