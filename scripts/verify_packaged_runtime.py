#!/usr/bin/env python3
"""Verify model assets, build identity, and drag/drop inside the frozen app."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.build_info import assert_packaged_commit, current_build_info
from core.model_assets import ModelAssetsError, validate_model_assets


def _drag_drop_gate(audio_path: Path) -> dict[str, object]:
    from PySide6.QtCore import QMimeData, QPointF, QUrl, Qt
    from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
    from PySide6.QtWidgets import QApplication, QGraphicsProxyWidget

    from app.ui import MainWindow
    from core.privacy import PrivacyStore

    application = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="patchlab-packaged-ui-") as temporary:
        privacy = PrivacyStore(Path(temporary) / "privacy.json")
        privacy.save(True)
        window = MainWindow(privacy_store=privacy)
        window.show()
        application.processEvents()

        root_proxies = [
            item
            for item in window._scene.items()
            if isinstance(item, QGraphicsProxyWidget)
        ]
        proxy_accepts_drop = bool(root_proxies and root_proxies[0].acceptDrops())
        drop_center = window.match_drop.mapTo(
            window._ui_root,
            window.match_drop.rect().center(),
        )
        viewport_position = window._graphics_view.mapFromScene(
            QPointF(drop_center)
        )
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(audio_path))])
        drag_enter = QDragEnterEvent(
            viewport_position,
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        drag_move = QDragMoveEvent(
            viewport_position,
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        drop = QDropEvent(
            QPointF(viewport_position),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(window._graphics_view.viewport(), drag_enter)
        QApplication.sendEvent(window._graphics_view.viewport(), drag_move)
        QApplication.sendEvent(window._graphics_view.viewport(), drop)
        application.processEvents()
        selected = window._match_audio_path
        result = {
            "root_proxy_accepts_drops": proxy_accepts_drop,
            "drag_enter_accepted": drag_enter.isAccepted(),
            "drop_accepted": drop.isAccepted(),
            "selected_path": str(selected) if selected else None,
            "drag_drop_pass": selected == audio_path.resolve(),
        }
        window.close()
        application.processEvents()
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit")
    parser.add_argument("--drag-audio", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expect-model-error", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {
        "build": current_build_info().as_dict(),
    }
    try:
        assets = validate_model_assets()
    except ModelAssetsError as exc:
        report["model_error"] = str(exc)
        report["model_assets_pass"] = False
        report["gate_pass"] = bool(args.expect_model_error)
        print(
            "PACKAGED_RUNTIME_GATE="
            + json.dumps(report, separators=(",", ":")),
            flush=True,
        )
        return 0 if args.expect_model_error else 1

    if args.expect_model_error:
        report["gate_pass"] = False
        report["model_error"] = "Expected model validation to fail, but it passed"
        print(
            "PACKAGED_RUNTIME_GATE="
            + json.dumps(report, separators=(",", ":")),
            flush=True,
        )
        return 1

    report["model_assets_pass"] = True
    report["model_cache"] = str(assets.cache_dir)
    report["checkpoint"] = str(assets.checkpoint)
    if args.expected_commit:
        assert_packaged_commit(args.expected_commit)
        report["build_commit_pass"] = True
    if args.drag_audio:
        report["drag_drop"] = _drag_drop_gate(args.drag_audio.resolve())
    report["gate_pass"] = bool(
        report["model_assets_pass"]
        and report.get("build_commit_pass", True)
        and (
            not args.drag_audio
            or bool(dict(report.get("drag_drop", {})).get("drag_drop_pass"))
        )
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(
        "PACKAGED_RUNTIME_GATE="
        + json.dumps(report, separators=(",", ":")),
        flush=True,
    )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
