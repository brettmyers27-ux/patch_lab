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
from core.synthesis_assets import resolve_synthesis_assets, synthesis_readiness


def _drag_drop_gate(audio_path: Path) -> dict[str, object]:
    from PySide6.QtCore import QMimeData, QPoint, QPointF, QUrl, Qt
    from PySide6.QtGui import (
        QDragEnterEvent,
        QDragMoveEvent,
        QDropEvent,
    )
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
        viewport = window._graphics_view.viewport()
        drop_wiring = {
            "drop_widget": window.match_drop.acceptDrops(),
            "root_proxy": proxy_accepts_drop,
            "graphics_view": window._graphics_view.acceptDrops(),
            "viewport": viewport.acceptDrops(),
        }

        def viewport_point(x_ratio: float, y_ratio: float) -> QPoint:
            local = QPoint(
                round(window.match_drop.width() * x_ratio),
                round(window.match_drop.height() * y_ratio),
            )
            canvas = window.match_drop.mapTo(window._ui_root, local)
            return window._graphics_view.mapFromScene(QPointF(canvas))

        positions = [
            viewport_point(0.12, 0.25),
            viewport_point(0.30, 0.50),
            viewport_point(0.50, 0.72),
            viewport_point(0.70, 0.50),
            viewport_point(0.88, 0.25),
        ]

        def exercise_drag(path: Path) -> tuple[
            QDragEnterEvent,
            list[QDragMoveEvent],
            QDropEvent,
        ]:
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(path))])
            drag_enter = QDragEnterEvent(
                positions[0],
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.sendEvent(viewport, drag_enter)
            drag_moves: list[QDragMoveEvent] = []
            for position in positions:
                drag_move = QDragMoveEvent(
                    position,
                    Qt.DropAction.CopyAction,
                    mime,
                    Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier,
                )
                QApplication.sendEvent(viewport, drag_move)
                drag_moves.append(drag_move)
            drop = QDropEvent(
                QPointF(positions[-1]),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.sendEvent(viewport, drop)
            return drag_enter, drag_moves, drop

        drag_enter, drag_moves, drop = exercise_drag(audio_path)
        application.processEvents()
        selected = window._match_audio_path
        move_acceptance = [event.isAccepted() for event in drag_moves]
        positive_pass = bool(
            all(drop_wiring.values())
            and drag_enter.isAccepted()
            and all(move_acceptance)
            and drop.isAccepted()
            and selected == audio_path.resolve()
        )

        # A rejected drag must neither set the highlight nor mutate the selected
        # query. Exercise this through the graphics viewport too, because direct
        # widget delivery bypasses the scene behavior under test.
        window._match_audio_path = None
        unsupported = Path(temporary) / "unsupported.txt"
        unsupported.write_text("not audio", encoding="utf-8")
        rejected_enter, rejected_moves, rejected_drop = exercise_drag(unsupported)
        application.processEvents()
        rejected_move_acceptance = [
            event.isAccepted() for event in rejected_moves
        ]
        negative_pass = bool(
            not rejected_enter.isAccepted()
            and not any(rejected_move_acceptance)
            and not rejected_drop.isAccepted()
            and window._match_audio_path is None
        )
        result = {
            "drop_wiring": drop_wiring,
            "root_proxy_accepts_drops": proxy_accepts_drop,
            "drag_enter_accepted": drag_enter.isAccepted(),
            "drag_move_accepted": all(move_acceptance),
            "drag_move_acceptance": move_acceptance,
            "drop_accepted": drop.isAccepted(),
            "selected_path": str(selected) if selected else None,
            "unsupported_enter_rejected": not rejected_enter.isAccepted(),
            "unsupported_moves_rejected": not any(rejected_move_acceptance),
            "unsupported_drop_rejected": not rejected_drop.isAccepted(),
            "unsupported_selected_path": (
                str(window._match_audio_path)
                if window._match_audio_path is not None
                else None
            ),
            "negative_drag_pass": negative_pass,
            "drag_drop_pass": positive_pass and negative_pass,
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
    synthesis_assets = resolve_synthesis_assets()
    serum2_readiness = synthesis_readiness("serum2")
    report["serum2_synthesis"] = {
        "available": serum2_readiness.available,
        "reason": serum2_readiness.reason,
        "missing": list(serum2_readiness.missing),
        "library_db": str(synthesis_assets.library_db),
        "feature_dir": str(synthesis_assets.feature_dir),
        "render_state_roots": [
            str(path) for path in synthesis_assets.render_state_roots
        ],
    }
    if args.expected_commit:
        assert_packaged_commit(args.expected_commit)
        report["build_commit_pass"] = True
    if args.drag_audio:
        report["drag_drop"] = _drag_drop_gate(args.drag_audio.resolve())
    report["gate_pass"] = bool(
        report["model_assets_pass"]
        and serum2_readiness.available
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
