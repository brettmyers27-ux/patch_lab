"""First-run passcode dialog for distribution builds."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from core.access_gate import AccessManager


class PasscodeDialog(QDialog):
    def __init__(self, manager: AccessManager, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Welcome to PatchLab")
        self.setModal(True)
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        title = QLabel("Private-group access")
        title.setStyleSheet("font-size: 20px; font-weight: 750;")
        detail = QLabel(
            "Enter the group passcode once. PatchLab validates it with the "
            "private sharing service and stores it in your system keychain."
        )
        detail.setWordWrap(True)
        self.passcode = QLineEdit()
        self.passcode.setEchoMode(QLineEdit.EchoMode.Password)
        self.passcode.setPlaceholderText("Group passcode")
        self.passcode.returnPressed.connect(self._unlock)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        buttons = QHBoxLayout()
        self.local_button = QPushButton("Continue without sharing")
        self.local_button.clicked.connect(self._continue_local)
        self.unlock_button = QPushButton("Unlock")
        self.unlock_button.setObjectName("primaryButton")
        self.unlock_button.clicked.connect(self._unlock)
        buttons.addWidget(self.local_button)
        buttons.addStretch(1)
        buttons.addWidget(self.unlock_button)
        layout.addWidget(title)
        layout.addWidget(detail)
        layout.addWidget(self.passcode)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

    def _unlock(self) -> None:
        value = self.passcode.text()
        self.passcode.clear()
        self.unlock_button.setEnabled(False)
        success, message, unreachable = self.manager.authenticate(value)
        self.unlock_button.setEnabled(True)
        self.status.setText(message)
        self.local_button.setVisible(unreachable)
        if success:
            self.accept()
        else:
            self.passcode.setFocus()

    def _continue_local(self) -> None:
        self.manager.continue_locally()
        self.accept()
