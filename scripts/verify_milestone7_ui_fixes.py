#!/usr/bin/env python3
"""Verify the four post-redesign UI fixes with a busy real match result."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QApplication, QAbstractScrollArea, QFrame

from app.ui import MainWindow


OUTPUT = PROJECT_ROOT / "data" / "models" / "ui_redesign"
REPORT = OUTPUT / "milestone7_ui_fix_report.json"


def busiest_real_result() -> tuple[Path, dict, int]:
    choices: list[tuple[int, Path, dict]] = []
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
        if result.get("existing_matches"):
            choices.append((changed, path, result))
    if not choices:
        raise FileNotFoundError("No completed real match results")
    changed, path, result = max(choices, key=lambda item: item[0])
    return path, result, changed


def crop_widgets(window: MainWindow, widgets: list, path: Path) -> None:
    points = [widget.mapTo(window, QPoint(0, 0)) for widget in widgets]
    left = min(point.x() for point in points)
    top = min(point.y() for point in points)
    right = max(
        point.x() + widget.width()
        for point, widget in zip(points, widgets, strict=True)
    )
    bottom = max(
        point.y() + widget.height()
        for point, widget in zip(points, widgets, strict=True)
    )
    window.grab(QRect(left, top, right - left, bottom - top)).save(str(path))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    application.processEvents()
    result_path, result, changed_count = busiest_real_result()
    window._show_match_result(result)
    window.resize(1024, 576)
    application.processEvents()

    root = window.centralWidget()
    cards = root.findChildren(QFrame, "heroCard")
    control_cards = list(window.control_cards)
    scroll_areas = window.findChildren(QAbstractScrollArea)
    visible_scrollbars = [
        {
            "widget": area.metaObject().className(),
            "object_name": area.objectName(),
            "vertical": area.verticalScrollBar().isVisible(),
            "horizontal": area.horizontalScrollBar().isVisible(),
        }
        for area in scroll_areas
        if area.verticalScrollBar().isVisible()
        or area.horizontalScrollBar().isVisible()
    ]
    last_match_row = window.existing_table.rowCount() - 1
    all_match_rows_visible = (
        last_match_row >= 0
        and window.existing_table.rowViewportPosition(last_match_row)
        + window.existing_table.rowHeight(last_match_row)
        <= window.existing_table.viewport().height()
    )
    config = control_cards[0]
    action_rects = [
        QRect(
            widget.mapTo(config, QPoint(0, 0)),
            widget.size(),
        )
        for widget in (
            window.match_offset,
            window.match_start_button,
            window.match_cancel_button,
        )
    ]
    match_actions_do_not_overlap = all(
        not left.intersects(right)
        for index, left in enumerate(action_rects)
        for right in action_rects[index + 1 :]
    )
    play_buttons = [button for button, _detail in window._match_play_buttons]
    play_rects = [
        QRect(
            button.mapTo(window.existing_table.viewport(), QPoint(0, 0)),
            button.size(),
        )
        for button in play_buttons
    ]
    play_buttons_do_not_overlap = all(
        not left.intersects(right)
        for index, left in enumerate(play_rects)
        for right in play_rects[index + 1 :]
    )
    table_top_gap = window.existing_table.mapTo(
        window.closest_panel, QPoint(0, 0)
    ).y()
    temporary_root = Path(window._match_session.name)
    (temporary_root / "cleanup-fixture.wav").write_bytes(b"temporary")

    hero_path = OUTPUT / "milestone7-hero-cards-ampersand.png"
    crop_widgets(window, cards, hero_path)
    controls_path = OUTPUT / "milestone7-match-control-cards.png"
    crop_widgets(window, control_cards, controls_path)
    busy_path = OUTPUT / "milestone7-busy-result-no-scroll.png"
    window.grab().save(str(busy_path))
    octave_default_text = window.octave_selector.currentText()

    played_paths: list[Path] = []
    recommendation_requests: list[tuple[Path, int]] = []
    window._play_audio = lambda path: played_paths.append(Path(path))
    window._match_result = result
    window._match_result_path = result_path
    window.preview_runner.start_recommendation = (
        lambda path, note: recommendation_requests.append((Path(path), int(note)))
    )
    window.octave_selector.setCurrentIndex(5)  # C6 / MIDI 84
    window._play_existing_match(dict(result["existing_matches"][0]))
    window.play_winner()
    closest_match_uses_selected_octave = (
        bool(played_paths) and played_paths[-1].name == "84.wav"
    )
    recommendation_uses_selected_octave = (
        bool(recommendation_requests)
        and recommendation_requests[-1][1] == 84
    )

    analyze_text = window.learn_button.text().replace("&&", "&")
    payload = {
        "busy_result_path": str(result_path),
        "busy_base_name": result["recommendation"]["base_name"],
        "busy_changed_settings": changed_count,
        "analyze_button_text": analyze_text,
        "ampersand_preserved": analyze_text == "Analyze & Learn",
        "control_card_count": len(control_cards),
        "control_cards_bounded": len(control_cards) == 3,
        "match_actions_do_not_overlap": match_actions_do_not_overlap,
        "play_buttons_do_not_overlap": play_buttons_do_not_overlap,
        "closest_table_top_gap": table_top_gap,
        "closest_header_pulled_up": table_top_gap <= 45,
        "octave_selector_count": window.octave_selector.count(),
        "octave_selector_default": octave_default_text,
        "shared_octave_selector_ready": (
            window.octave_selector.count() == 7
            and window.octave_selector.itemData(3) == 60
        ),
        "closest_match_uses_selected_octave": (
            closest_match_uses_selected_octave
        ),
        "recommendation_uses_selected_octave": (
            recommendation_uses_selected_octave
        ),
        "parameter_visuals_removed": (
            not window.parameter_strip.isVisible()
            and not window.settings_tree.isVisible()
            and not window.parameter_knobs
        ),
        "visible_scrollbars": visible_scrollbars,
        "zero_default_scrollbars": not visible_scrollbars,
        "all_match_rows_visible": all_match_rows_visible,
        "window_size": [window.width(), window.height()],
        "window_ratio": window.width() / window.height(),
        "ratio_16_9": abs(window.width() / window.height() - 16 / 9) < 0.01,
        "native_aspect_installed": window._native_aspect_installed,
        "hero_screenshot": str(hero_path),
        "control_screenshot": str(controls_path),
        "busy_result_screenshot": str(busy_path),
    }
    window.close()
    application.processEvents()
    payload["temporary_match_audio_removed_on_close"] = (
        not temporary_root.exists()
    )
    payload["gate_pass"] = (
        payload["ampersand_preserved"]
        and payload["control_cards_bounded"]
        and payload["busy_changed_settings"] >= 100
        and payload["match_actions_do_not_overlap"]
        and payload["play_buttons_do_not_overlap"]
        and payload["closest_header_pulled_up"]
        and payload["shared_octave_selector_ready"]
        and payload["closest_match_uses_selected_octave"]
        and payload["recommendation_uses_selected_octave"]
        and payload["parameter_visuals_removed"]
        and payload["zero_default_scrollbars"]
        and payload["all_match_rows_visible"]
        and payload["ratio_16_9"]
        and payload["temporary_match_audio_removed_on_close"]
    )
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("MILESTONE7_UI_FIX_REPORT=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
