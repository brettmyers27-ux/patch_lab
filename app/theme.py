"""Central Patch Lab visual tokens and theme loader."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QWidget


BASE = "#0A0E17"
SURFACE = "#111827"
SURFACE_RAISED = "#151C2C"
BORDER = "#273653"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
TEAL = "#2DD4BF"
VIOLET = "#A855F7"
AMBER = "#F59E0B"
BLUE = "#3B82F6"
GREEN = "#22C55E"
RED = "#F43F5E"

APP_DIR = Path(__file__).resolve().parent
ICON_DIR = APP_DIR / "icons"
QSS_PATH = APP_DIR / "theme.qss"


def load_stylesheet() -> str:
    # Qt resolves url() in stylesheets against the process working directory,
    # which the app never controls, so icon references are written as {ICONS}
    # and expanded to an absolute path here. Forward slashes work on every
    # platform Qt parses stylesheets on, including Windows.
    sheet = QSS_PATH.read_text(encoding="utf-8")
    return sheet.replace("{ICONS}", ICON_DIR.as_posix())


def add_glow(
    widget: QWidget,
    color: str,
    *,
    blur: int = 28,
    alpha: int = 85,
    y_offset: int = 4,
) -> None:
    effect = QGraphicsDropShadowEffect(widget)
    glow = QColor(color)
    glow.setAlpha(alpha)
    effect.setColor(glow)
    effect.setBlurRadius(blur)
    effect.setOffset(0, y_offset)
    widget.setGraphicsEffect(effect)
