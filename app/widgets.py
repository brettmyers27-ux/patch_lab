"""Reusable code-painted widgets for the Patch Lab desktop interface."""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app import theme


ACCENTS = {
    "teal": theme.TEAL,
    "violet": theme.VIOLET,
    "amber": theme.AMBER,
    "blue": theme.BLUE,
    "green": theme.GREEN,
}

CATEGORY_KEYWORDS = (
    ("Bass", ("bass", " bs ", "bs_", "sub")),
    ("Lead", ("lead", "ld ", "ld_")),
    ("Pad", ("pad", "pads")),
    ("Pluck", ("pluck",)),
    ("Growl", ("growl", "grwl")),
    ("Keys", ("keys", "piano", "rhodes")),
    ("Arp", ("arp", "arpeggio")),
    ("FX", (" fx", "fx_", "riser", "impact")),
    ("Vocal", ("vox", "vocal")),
    ("Drum", ("drum", "kick", "snare", "perc")),
)


def derive_category(name: str) -> str:
    lowered = f" {name.casefold()} "
    for label, needles in CATEGORY_KEYWORDS:
        if any(needle in lowered for needle in needles):
            return label
    return "Uncategorized"


def derive_style_character(audio_path: Path | None) -> tuple[str, str]:
    """Best-effort Style/Character labels from simple spectral heuristics.

    Estimated, not guaranteed-accurate — a lightweight bucketed heuristic,
    not a trained classifier. Returns ("—", "—") if analysis is unavailable.
    """

    if audio_path is None or not Path(audio_path).is_file():
        return "—", "—"
    try:
        import numpy as np
        import soundfile as sf

        audio, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            return "—", "—"
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(audio.size, d=1.0 / rate)
        total = float(spectrum.sum())
        if total <= 0:
            return "—", "—"
        centroid = float((freqs * spectrum).sum() / total)
        zero_crossings = int(np.sum(np.abs(np.diff(np.sign(audio))) > 0))
        zcr = zero_crossings / max(audio.size, 1)
        character = "Bright" if centroid > 2500 else "Warm" if centroid > 700 else "Dark"
        style = "Digital" if zcr > 0.08 else "Analog"
        return style, character
    except Exception:
        return "—", "—"


def icon(name: str) -> QIcon:
    return QIcon(str(theme.ICON_DIR / f"{name}.svg"))


def escape_mnemonic(text: str) -> str:
    """Preserve literal ampersands on Qt controls that enable mnemonics."""

    return text.replace("&", "&&")


class SegmentedControl(QWidget):
    """Compact pill selector with the small QComboBox API used by the workers."""

    currentIndexChanged = Signal(int)
    # Fires on every click, including a click on the already-selected segment.
    # currentIndexChanged stays change-only because it drives state updates, so
    # anything that must re-trigger on a repeat click — replaying audio at the
    # same octave, for instance — has to listen to this instead.
    itemClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("segmentedControl")
        self._items: list[tuple[str, object]] = []
        self._buttons: list[QPushButton] = []
        self._current_index = -1
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(0)

    def addItem(self, text: str, user_data: object = None) -> None:
        index = len(self._items)
        self._items.append((text, user_data))
        button = QPushButton(text)
        button.setObjectName("segmentButton")
        button.setCheckable(True)
        button.clicked.connect(
            lambda _checked=False, item_index=index: self._button_clicked(item_index)
        )
        self._group.addButton(button, index)
        self._buttons.append(button)
        self._layout.addWidget(button, 1)
        if self._current_index < 0:
            self.setCurrentIndex(0)

    def _button_clicked(self, index: int) -> None:
        # An exclusive QButtonGroup keeps the active segment checked, so
        # re-clicking it re-emits clicked() with checked already True. Select
        # first (a no-op when unchanged), then always announce the click.
        self.setCurrentIndex(index)
        self.itemClicked.emit(index)

    def count(self) -> int:
        return len(self._items)

    def currentIndex(self) -> int:
        return self._current_index

    def currentData(self) -> object:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        return None

    def currentText(self) -> str:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][0]
        return ""

    def itemData(self, index: int) -> object:
        if 0 <= index < len(self._items):
            return self._items[index][1]
        return None

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._items):
            return
        changed = index != self._current_index
        self._current_index = index
        self._buttons[index].setChecked(True)
        if changed:
            self.currentIndexChanged.emit(index)

    def setItemText(self, index: int, text: str) -> None:
        if not 0 <= index < len(self._items):
            return
        _old_text, data = self._items[index]
        self._items[index] = (text, data)
        self._buttons[index].setText(text)


class HeroCard(QFrame):
    """Action card whose public button/progress/status preserve existing wiring."""

    def __init__(
        self,
        title: str,
        icon_name: str,
        accent: str,
        *,
        enabled: bool,
        step: int | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("heroCard")
        self.setProperty("accent", accent)
        self.setMinimumHeight(78)
        self.setMaximumHeight(84)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.card_layout = QHBoxLayout(self)
        self.card_layout.setContentsMargins(13, 9, 13, 9)
        self.card_layout.setSpacing(11)
        self.step_badge: QLabel | None = None
        if step is not None:
            step_badge = QLabel(str(step), self)
            step_badge.setObjectName("stepBadge")
            step_badge.setProperty("accent", accent)
            step_badge.setFixedSize(20, 20)
            step_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            step_badge.raise_()
            self.step_badge = step_badge
        self.badge = QLabel()
        self.badge.setObjectName("iconBadge")
        self.badge.setPixmap(icon(icon_name).pixmap(28, 28))
        self.badge.setFixedSize(47, 47)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.card_layout.addWidget(
            self.badge, 0, Qt.AlignmentFlag.AlignVCenter
        )
        self.body = QVBoxLayout()
        self.body.setSpacing(7)
        self.button = QPushButton(escape_mnemonic(title))
        self.button.setObjectName("heroAction")
        self.button.setEnabled(enabled)
        self.progress = QProgressBar()
        self.progress.setProperty("accent", accent)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.status = QLabel("Ready")
        self.status.setObjectName("cardStatus")
        self.status.setStyleSheet(f"color: {ACCENTS[accent]};")
        self.body.addWidget(self.button)
        self.body.addWidget(self.progress)
        self.body.addWidget(self.status)
        self.card_layout.addLayout(self.body, 1)
        # QGraphicsDropShadowEffect (theme.add_glow) is not used here: it
        # conflicts with hosting this widget tree inside a QGraphicsProxyWidget
        # (the app's scaled-canvas rendering) and causes paint/bounds glitches.
        # The QSS accent border on #heroCard already carries the color cue.
        self._position_step_badge()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._position_step_badge()

    def _position_step_badge(self) -> None:
        if self.step_badge is not None:
            self.step_badge.move(self.width() - self.step_badge.width() - 8, 8)

    def setCompact(self, compact: bool) -> None:
        if compact:
            self.setFixedHeight(64)
            self.card_layout.setContentsMargins(9, 6, 9, 6)
            self.card_layout.setSpacing(8)
            self.body.setSpacing(4)
            self.badge.setFixedSize(38, 38)
            self.badge.setPixmap(self.badge.pixmap().scaled(
                24,
                24,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
        else:
            self.setMinimumHeight(78)
            self.setMaximumHeight(84)
            self.card_layout.setContentsMargins(13, 9, 13, 9)
            self.card_layout.setSpacing(11)
            self.body.setSpacing(7)
            self.badge.setFixedSize(47, 47)


class ConfidenceRing(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self.setFixedSize(58, 58)
        self.setAccessibleName("Recommendation confidence")

    def setValue(self, value: float) -> None:  # Qt-style API
        self._value = float(max(0.0, min(100.0, value)))
        self.setAccessibleDescription(f"{self._value:.1f} percent")
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(6, 6, self.width() - 12, self.height() - 12)
        painter.setPen(QPen(QColor("#243047"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 90 * 16, -360 * 16)
        color = QColor(theme.GREEN if self._value >= 70 else theme.AMBER)
        painter.setPen(QPen(color, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 90 * 16, int(-360 * 16 * self._value / 100.0))
        painter.setPen(QColor(theme.TEXT))
        font = QFont(self.font())
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"{self._value:.0f}%")


class RotaryKnob(QWidget):
    """Read-only normalized parameter display rendered with QPainter."""

    def __init__(
        self,
        label: str = "",
        value: float = 0.0,
        display_value: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.label = label
        self.value = float(max(0.0, min(1.0, value)))
        self.display_value = display_value
        self.setMinimumSize(62, 54)
        self.setMaximumHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setParameter(self, label: str, value: float, display_value: str) -> None:
        self.label = label
        self.value = float(max(0.0, min(1.0, value)))
        self.display_value = display_value
        self.setToolTip(f"{label}: {display_value}")
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2.0, 20)
        radius = 12.0
        outer = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
        painter.setPen(
            QPen(
                QColor("#334155"),
                5,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawArc(outer, 225 * 16, -270 * 16)
        accent = QColor(theme.BLUE)
        painter.setPen(
            QPen(
                accent,
                5,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawArc(outer, 225 * 16, int(-270 * 16 * self.value))
        angle = math.radians(225 - 270 * self.value)
        pointer = QPointF(
            center.x() + math.cos(angle) * 7,
            center.y() - math.sin(angle) * 7,
        )
        painter.setPen(
            QPen(
                QColor("#E2E8F0"),
                2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.drawLine(center, pointer)
        painter.setPen(QColor(theme.MUTED))
        small = QFont(self.font())
        small.setPointSize(7)
        small.setBold(True)
        painter.setFont(small)
        painter.drawText(
            QRectF(2, 35, self.width() - 4, 9),
            Qt.AlignmentFlag.AlignCenter,
            self.label.upper()[:15],
        )
        painter.setPen(QColor(theme.TEXT))
        value_font = QFont(self.font())
        value_font.setPointSize(8)
        painter.setFont(value_font)
        painter.drawText(
            QRectF(2, 45, self.width() - 4, 9),
            Qt.AlignmentFlag.AlignCenter,
            self.display_value[:16],
        )


class PresetThumbnail(QWidget):
    """Small custom-painted abstract preview art for a recommended preset."""

    def __init__(self, accent: str = "blue", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._accent = accent
        self.setFixedSize(64, 64)

    def setAccent(self, accent: str) -> None:
        self._accent = accent
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        base = QColor(ACCENTS.get(self._accent, theme.BLUE))
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        top = QColor(base)
        top.setAlpha(210)
        bottom = QColor(theme.SURFACE_RAISED)
        gradient.setColorAt(0.0, top)
        gradient.setColorAt(1.0, bottom)
        path = QPainterPath()
        path.addRoundedRect(rect, 12, 12)
        painter.fillPath(path, gradient)
        painter.setPen(
            QPen(QColor("#FFFFFF").lighter(160), 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        heights = (10, 22, 34, 26, 16, 30, 12)
        count = len(heights)
        margin = 10.0
        span = self.width() - 2 * margin
        for index, height in enumerate(heights):
            x = margin + span * (index + 0.5) / count
            painter.drawLine(
                QPointF(x, self.height() / 2 - height / 2),
                QPointF(x, self.height() / 2 + height / 2),
            )


class WaveformMark(QWidget):
    """Small custom-painted cyan/violet wordmark symbol."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(52, 38)

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        heights = (7, 17, 29, 35, 23, 14, 8)
        colors = (theme.TEAL, theme.TEAL, "#22D3EE", "#38BDF8", "#6366F1", theme.VIOLET, theme.VIOLET)
        spacing = self.width() / (len(heights) + 1)
        for index, (height, color) in enumerate(zip(heights, colors, strict=True), start=1):
            x = spacing * index
            painter.setPen(
                QPen(
                    QColor(color),
                    3,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
            )
            painter.drawLine(QPointF(x, (self.height() - height) / 2), QPointF(x, (self.height() + height) / 2))
