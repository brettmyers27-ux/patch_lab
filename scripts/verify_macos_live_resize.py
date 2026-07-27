#!/usr/bin/env python3
"""Perform a real Cocoa corner drag and verify every sampled frame stays 16:9."""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.ui import MainWindow
from core.platform_env import ENV


OUTPUT = PROJECT_ROOT / "data" / "models" / "ui_redesign"
SCREENSHOT = OUTPUT / "milestone7-macos-mid-drag-16x9.png"
REPORT = OUTPUT / "milestone7-macos-live-resize-report.json"


class CGPoint(ctypes.Structure):
    _fields_ = (("x", ctypes.c_double), ("y", ctypes.c_double))


class NativeMouse:
    MOVED = 5
    LEFT_DOWN = 1
    LEFT_UP = 2
    LEFT_DRAGGED = 6
    LEFT_BUTTON = 0
    SESSION_EVENT_TAP = 1

    def __init__(self) -> None:
        core_graphics = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        core_foundation = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        self._create = core_graphics.CGEventCreateMouseEvent
        self._create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            CGPoint,
            ctypes.c_uint32,
        ]
        self._create.restype = ctypes.c_void_p
        self._post = core_graphics.CGEventPost
        self._post.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        self._release = core_foundation.CFRelease
        self._release.argtypes = [ctypes.c_void_p]

    def post(self, event_type: int, x: float, y: float) -> None:
        event = self._create(
            None,
            event_type,
            CGPoint(float(x), float(y)),
            self.LEFT_BUTTON,
        )
        if not event:
            raise RuntimeError("CGEventCreateMouseEvent returned NULL")
        try:
            self._post(self.SESSION_EVENT_TAP, event)
        finally:
            self._release(event)


def busy_result() -> dict:
    best: tuple[int, dict] | None = None
    for path in (PROJECT_ROOT / "data" / "matches").glob("*/result.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        recommendation = result.get("recommendation")
        if not isinstance(recommendation, dict):
            continue
        changed = sum(
            len(values.get("changed", []))
            for values in recommendation.get("settings", {}).values()
        )
        if best is None or changed > best[0]:
            best = changed, result
    if best is None:
        raise FileNotFoundError("No real match result available")
    return best[1]


def main() -> int:
    if ENV.branch != "macos":
        print("SKIP: native live-resize verification requires macOS")
        return 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._show_match_result(busy_result())
    window.resize(1024, 576)
    window.move(80, 70)
    window.show()
    window.raise_()
    window.activateWindow()
    if window.windowHandle() is not None:
        window.windowHandle().requestActivate()

    native_mouse = NativeMouse()
    samples: list[dict[str, float]] = []
    state: dict[str, float | int | bool] = {
        "step": 0,
        "mouse_down": False,
        "captured_while_down": False,
    }

    def finish() -> None:
        ratios = [sample["ratio"] for sample in samples]
        widths = [sample["width"] for sample in samples]
        payload = {
            "native_aspect_installed": window._native_aspect_installed,
            "sample_count": len(samples),
            "samples": samples,
            "max_ratio_deviation": max(
                (abs(ratio - 16 / 9) for ratio in ratios),
                default=999.0,
            ),
            "window_resized_during_drag": (
                bool(widths) and max(widths) - min(widths) >= 20
            ),
            "screenshot_captured_while_mouse_down": bool(
                state["captured_while_down"]
            ),
            "screenshot": str(SCREENSHOT),
        }
        payload["gate_pass"] = (
            payload["native_aspect_installed"]
            and payload["sample_count"] >= 20
            and payload["max_ratio_deviation"] < 0.01
            and payload["window_resized_during_drag"]
            and payload["screenshot_captured_while_mouse_down"]
            and SCREENSHOT.is_file()
        )
        REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("MACOS_LIVE_RESIZE_REPORT=" + json.dumps(payload, sort_keys=True))
        window.close()
        application.exit(0 if payload["gate_pass"] else 1)

    def drag_step() -> None:
        step = int(state["step"])
        x = float(state["start_x"]) + step * 4.0
        y = float(state["start_y"]) + step * 0.8
        native_mouse.post(NativeMouse.LEFT_DRAGGED, x, y)
        samples.append(
            {
                "step": float(step),
                "width": float(window.width()),
                "height": float(window.height()),
                "ratio": window.width() / max(window.height(), 1),
            }
        )
        if step == 15:
            state["captured_while_down"] = bool(state["mouse_down"])
            window.screen().grabWindow(int(window.winId())).save(
                str(SCREENSHOT)
            )
        if step >= 30:
            native_mouse.post(
                NativeMouse.LEFT_UP,
                float(state["start_x"]) + step * 4.0,
                float(state["start_y"]) + step * 0.8,
            )
            state["mouse_down"] = False
            QTimer.singleShot(300, finish)
            return
        state["step"] = step + 1
        QTimer.singleShot(45, drag_step)

    def press_corner() -> None:
        native_mouse.post(
            NativeMouse.LEFT_DOWN,
            float(state["start_x"]),
            float(state["start_y"]),
        )
        state["mouse_down"] = True
        QTimer.singleShot(120, drag_step)

    def move_to_corner() -> None:
        frame = window.frameGeometry()
        state["start_x"] = frame.x() + frame.width() - 8
        state["start_y"] = frame.y() + frame.height() - 8
        native_mouse.post(
            NativeMouse.MOVED,
            float(state["start_x"]),
            float(state["start_y"]),
        )
        QTimer.singleShot(180, press_corner)

    QTimer.singleShot(1100, move_to_corner)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
