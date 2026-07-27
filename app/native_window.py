"""Native window integration needed for continuous macOS aspect constraints."""

from __future__ import annotations

import ctypes

from PySide6.QtWidgets import QApplication, QWidget

from core.platform_env import ENV, PlatformEnv


class _NSSize(ctypes.Structure):
    _fields_ = (("width", ctypes.c_double), ("height", ctypes.c_double))


def enforce_native_aspect_ratio(
    widget: QWidget,
    width: float,
    height: float,
    *,
    env: PlatformEnv = ENV,
) -> bool:
    """Set NSWindow.contentAspectRatio so Cocoa constrains every live-resize frame."""

    application = QApplication.instance()
    if (
        env.branch != "macos"
        or application is None
        or application.platformName().casefold() != "cocoa"
    ):
        return False
    try:
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        send_id = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
        )
        send_size = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(
                None,
                ctypes.c_void_p,
                ctypes.c_void_p,
                _NSSize,
            ),
        )
        native_view = ctypes.c_void_p(int(widget.winId()))
        window_selector = objc.sel_registerName(b"window")
        native_window = send_id(native_view, window_selector)
        if not native_window:
            return False
        aspect_selector = objc.sel_registerName(b"setContentAspectRatio:")
        send_size(
            ctypes.c_void_p(native_window),
            aspect_selector,
            _NSSize(float(width), float(height)),
        )
        return True
    except (AttributeError, OSError, TypeError, ValueError):
        return False
