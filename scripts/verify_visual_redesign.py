#!/usr/bin/env python3
"""Run a real match and capture the redesigned UI at three 16:9 sizes."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(
    os.environ.get(
        "PATCHLAB_GATE_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QPoint, QRect, QTimer
from PySide6.QtWidgets import QApplication, QAbstractScrollArea, QFrame, QPushButton

from app.ui import LibraryEntryRow, MainWindow
from core.db import Database
from core.match_library import resolved_record_paths


OUTPUT = PROJECT_ROOT / "data" / "models" / "ui_redesign"
REPORT = OUTPUT / "visual_redesign_report.json"
FIXTURE = PROJECT_ROOT / "data" / "audio" / "67" / "60.wav"


def region_rect(widget, root) -> QRect:
    top_left = widget.mapTo(root, QPoint(0, 0))
    return QRect(top_left, widget.size())


def layout_has_overlap(window: MainWindow) -> bool:
    root = window._ui_root
    cards = root.findChildren(QFrame, "heroCard")
    regions = [
        root.findChild(QFrame, "topBar"),
        *cards,
        window.match_panel,
        window.match_results,
        window.log_pane,
    ]
    rects = [region_rect(widget, root) for widget in regions if widget.isVisible()]
    for index, left in enumerate(rects):
        for right in rects[index + 1 :]:
            intersection = left.intersected(right)
            if intersection.width() > 1 and intersection.height() > 1:
                return True
    return False


def capture(window: MainWindow, name: str, width: int, height: int) -> dict:
    window.resize(width, height)
    QApplication.processEvents()
    actual = window.size()
    path = OUTPUT / f"{name}-{actual.width()}x{actual.height()}.png"
    window.grab().save(str(path))
    scroll_areas = window.findChildren(QAbstractScrollArea)
    return {
        "path": str(path),
        "width": actual.width(),
        "height": actual.height(),
        "ratio": actual.width() / actual.height(),
        "horizontal_scroll_max": max(
            (
                area.horizontalScrollBar().maximum()
                for area in scroll_areas
                if area.horizontalScrollBar().isVisible()
            ),
            default=0,
        ),
        "vertical_scroll_max": max(
            (
                area.verticalScrollBar().maximum()
                for area in scroll_areas
                if area.verticalScrollBar().isVisible()
            ),
            default=0,
        ),
        "overlap_detected": layout_has_overlap(window),
    }


def _window_point(window: MainWindow, root_point: QPoint) -> QPoint:
    """Map a point in root-widget (scene) coordinates to window pixel coordinates.

    root has no QWidget parent (QGraphicsScene.addWidget requires a top-level
    widget), so QWidget.mapTo can't bridge from a root descendant to the
    QMainWindow directly — it only walks native widget parent-child chains,
    and the scene/proxy embedding isn't one. Go through the view's own
    scene<->viewport mapping instead, then the view (a real child widget)
    can map to the window normally.
    """

    view = window._graphics_view
    view_point = view.mapFromScene(float(root_point.x()), float(root_point.y()))
    return view.mapTo(window, view_point)


def crop_row(
    window: MainWindow,
    widgets: list,
    path: Path,
    *,
    top_padding: int = 0,
) -> None:
    root = window._ui_root
    root_points = [widget.mapTo(root, QPoint(0, 0)) for widget in widgets]
    corners = [
        (
            _window_point(window, point),
            _window_point(window, point + QPoint(widget.width(), widget.height())),
        )
        for point, widget in zip(root_points, widgets, strict=True)
    ]
    left = min(top_left.x() for top_left, _ in corners)
    top = min(top_left.y() for top_left, _ in corners)
    right = max(bottom_right.x() for _, bottom_right in corners)
    bottom = max(bottom_right.y() for _, bottom_right in corners)
    padded_top = max(0, top - top_padding)
    window.grab(
        QRect(left, padded_top, right - left, bottom - padded_top)
    ).save(str(path))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    if mapping := os.environ.get("PATCHLAB_GATE_FACTORY_MAPPING"):
        window.factory_mapping_path = Path(mapping).expanduser().resolve()
    window._set_match_file(str(FIXTURE))
    window.match_budget.setCurrentIndex(0)
    window.match_synth.setCurrentIndex(1)
    outcome: dict[str, str] = {}

    def completed(path: str) -> None:
        outcome["result"] = path
        application.quit()

    def failed(error: str) -> None:
        outcome["error"] = error
        application.quit()

    # Capture completion first, then invoke the window's normal completion
    # handler explicitly. This prevents an exception in a UI slot from
    # swallowing the gate's second Qt signal subscriber and leaving the nested
    # event loop waiting forever.
    window.match_runner.completed.disconnect()
    window.match_runner.failed.disconnect()
    window.match_runner.completed.connect(completed)
    window.match_runner.failed.connect(failed)
    QTimer.singleShot(600_000, application.quit)
    # Drive the same packaged MatchProcessRunner directly. MainWindow's
    # product-level prerequisite guards intentionally depend on launch-time
    # verification supplied by app.main; this standalone frozen gate supplies
    # its verified factory mapping explicitly instead.
    print("VISUAL_GATE_STAGE=starting-match-worker", flush=True)
    window.match_runner.start(
        FIXTURE,
        target_synth=str(window.match_synth.currentData()),
        budget=str(window.match_budget.currentData()),
        offset=0.0,
        session_root=Path(window._match_session.name),
        factory_only=True,
        factory_mapping=window.factory_mapping_path,
    )
    print(
        f"VISUAL_GATE_STAGE=worker-{window.match_runner.process.state().name}",
        flush=True,
    )
    application.exec()
    if "result" not in outcome:
        raise RuntimeError(outcome.get("error", "Timed out waiting for live match"))
    window._match_completed(outcome["result"])
    window.show()
    application.processEvents()

    sizes = [
        capture(window, "minimum", 1440, 810),
        capture(window, "mid", 1600, 900),
        capture(window, "full", 1920, 1080),
    ]
    window.resize(1440, 810)
    application.processEvents()
    root = window._ui_root
    cards = root.findChildren(QFrame, "heroCard")
    hero_path = OUTPUT / "hero-cards.png"
    crop_row(window, cards, hero_path)
    match_path = OUTPUT / "match-panel.png"
    window.match_panel.grab().save(str(match_path))
    results_path = OUTPUT / "results-panel-real-data.png"
    crop_row(
        window,
        [window.match_results],
        results_path,
        top_padding=18,
    )
    records = Database(window._match_database_path()).list_match_library()
    latest = records[0] if records else None
    archived_source = archived_result = None
    if latest is not None:
        archived_source, archived_result = resolved_record_paths(
            latest, window._match_library_root()
        )
    window.nav_tabs.setCurrentIndex(1)
    application.processEvents()
    library_rows = window.library_container.findChildren(LibraryEntryRow)
    library_path = OUTPUT / "library-populated.png"
    window.grab().save(str(library_path))
    first_octave_count = (
        len(library_rows[0].findChildren(QPushButton, "rowOctaveButton"))
        if library_rows
        else 0
    )
    if latest is not None:
        window.open_library_match(latest.match_uid)
        application.processEvents()
    reopen_rows = window.existing_list_layout.count() - 1
    # The three control-row cards must present one pill geometry — identical
    # heights on a shared top edge — so no control crowds its section label.
    control_pills = [
        window.match_budget,
        window.match_synth,
        window.match_offset,
        window.match_start_button,
        window.match_cancel_button,
        window.render_pause_button,
        window.render_cancel_button,
        window.analyze_cancel_button,
    ]
    pill_heights = sorted({pill.height() for pill in control_pills})
    pill_tops = sorted({pill.mapTo(root, QPoint(0, 0)).y() for pill in control_pills})

    # The drop zone belongs at the bottom of the recommendation column, below
    # the octave card, absorbing the leftover height.
    octave_bottom = (
        window.octave_card.mapTo(root, QPoint(0, 0)).y() + window.octave_card.height()
    )
    drop_top = window.match_drop.mapTo(root, QPoint(0, 0)).y()

    log_text = window.log_pane.toPlainText()
    payload = {
        "live_result_path": outcome["result"],
        "control_pill_heights": pill_heights,
        "control_pill_tops": pill_tops,
        # These controls intentionally live in three separate bounded cards;
        # equal height is the cross-card invariant, not a shared global row.
        "control_pills_uniform": len(pill_heights) == 1,
        "drop_zone_below_octave_card": drop_top >= octave_bottom,
        "drop_zone_height": window.match_drop.height(),
        "uploaded_audio_play_button_ready": (
            window.match_drop.play_button.isVisible()
            and window.match_drop.play_button.isEnabled()
        ),
        "screenshots": sizes,
        "hero_cards_screenshot": str(hero_path),
        "match_panel_screenshot": str(match_path),
        "results_panel_screenshot": str(results_path),
        "library_screenshot": str(library_path),
        "navigation_tab_count": window.nav_tabs.count(),
        "library_row_count": len(library_rows),
        "library_first_row_octaves": first_octave_count,
        "archived_source_exists": bool(archived_source and archived_source.is_file()),
        "archived_result_exists": bool(archived_result and archived_result.is_file()),
        "library_reopen_existing_rows": reopen_rows,
        "closest_match_rows": window.existing_list_layout.count() - 1,  # minus trailing stretch
        "total_existing_matches": len(window._existing_matches),
        "confidence_ring_value": window.confidence_ring._value,
        "octave_selector_options": window.octave_selector.count(),
        "parameter_visuals_removed": (
            not window.parameter_strip.isVisible()
            and not window.settings_tree.isVisible()
            and not window.parameter_knobs
        ),
        "log_line_count": len(log_text.splitlines()),
        "log_contains_live_worker_output": (
            "MATCH_PROGRESS=" in log_text and "MATCH_RESULT=" in log_text
        ),
        "all_ratios_16_9": all(
            abs(item["ratio"] - 16 / 9) < 0.01 for item in sizes
        ),
        "no_horizontal_clipping": all(
            item["horizontal_scroll_max"] == 0 for item in sizes
        ),
        "no_panel_overlap": all(
            not item["overlap_detected"] for item in sizes
        ),
    }
    payload["gate_pass"] = (
        payload["closest_match_rows"] == payload["total_existing_matches"]
        and payload["total_existing_matches"] == 10
        and payload["confidence_ring_value"] > 0
        and payload["octave_selector_options"] == 7
        and payload["control_pills_uniform"]
        and payload["drop_zone_below_octave_card"]
        and payload["uploaded_audio_play_button_ready"]
        and payload["parameter_visuals_removed"]
        and payload["log_contains_live_worker_output"]
        and payload["all_ratios_16_9"]
        and payload["no_horizontal_clipping"]
        and payload["no_panel_overlap"]
        and payload["navigation_tab_count"] == 2
        and payload["library_row_count"] >= 1
        and payload["library_first_row_octaves"] == 7
        and payload["archived_source_exists"]
        and payload["archived_result_exists"]
        and payload["library_reopen_existing_rows"] == 10
    )
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("VISUAL_REDESIGN_REPORT=" + json.dumps(payload, sort_keys=True))
    window.close()
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
