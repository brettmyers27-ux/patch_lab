"""PatchLab single-window interface."""

from __future__ import annotations

import html
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QStackedWidget,
    QTabBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import theme
from app.__version__ import __version__
from app.native_window import enforce_native_aspect_ratio
from app.widgets import (
    ConfidenceRing,
    HeroCard,
    PresetThumbnail,
    SegmentedControl,
    WaveformMark,
    derive_category,
    derive_style_character,
    escape_mnemonic,
    icon,
)
from app.workers import (
    AnalyzeProcessRunner,
    BugReportProcessRunner,
    ExportProcessRunner,
    FactoryVerificationProcessRunner,
    MatchProcessRunner,
    PreviewProcessRunner,
    RenderProcessRunner,
    ScanProcessRunner,
    StorageProcessRunner,
    UpdateCheckProcessRunner,
    UpdateDownloadProcessRunner,
)
from core.audio_input import SUPPORTED_AUDIO_SUFFIXES
from core.branding import display_match_name, generated_preset_name
from core.build_info import current_build_info
from core.db import DEFAULT_DB_PATH, Database
from core.factory_verify import FactoryVerification
from core.local_library import auto_scan_due, default_local_paths, record_auto_scan
from core.library_status import preparation_queue_ids, preset_library_status
from core.update_check import (
    UpdatePreferences,
    load_update_preferences,
    save_update_preferences,
    update_space_preflight,
)
from core.worker_runtime import worker_invocation
from core.match_batch import (
    discover_batch_audio,
    disambiguated_preset_path,
    resumable_batch_files,
    sanitize_folder_name,
)
from core.match_library import (
    DEFAULT_MATCH_LIBRARY_ROOT,
    archive_match,
    delete_archived_match,
    resolve_result_path,
    resolved_record_paths,
)
from core.platform_env import ENV, PlatformEnv
from core.preview_cache import (
    PREVIEW_NOTES,
    preview_cache_identity,
    preview_cache_path,
    recommendation_cache_key,
    touch_preview,
    unmodified_recommendation_basis_index,
)
from core.privacy import PrivacyStore, distribution_mode, set_active_store, user_presets_enabled
from core.runtime_log import append_runtime_log
from core.storage import (
    StoragePreferences,
    audio_root_size,
    clear_preview_cache,
    load_storage_preferences,
    prepare_audio_root,
    prune_preview_cache,
    preview_cache_usage,
    save_storage_preferences,
    storage_status,
)
from core.synthesis_assets import synthesis_readiness
from core.workflow_state import (
    WorkflowActivity,
    WorkflowCardState,
    resolve_workflow_state,
)


class AudioDropLabel(QFrame):
    file_dropped = Signal(str)
    browse_requested = Signal()
    play_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("dropZone")
        self.setProperty("dragActive", False)
        self.setAcceptDrops(True)
        # Expands to absorb whatever vertical space is left in its column
        # rather than sitting at a fixed height; the minimum keeps the orb,
        # title, detail and play row from ever colliding.
        self.setMinimumHeight(112)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        self.upload_button = QPushButton()
        self.upload_button.setObjectName("uploadOrb")
        self.upload_button.setIcon(icon("upload"))
        self.upload_button.setIconSize(self.upload_button.sizeHint())
        self.upload_button.clicked.connect(self.browse_requested)
        self.title = QLabel("Drag & drop an audio file here")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setWordWrap(True)
        self.title.setStyleSheet("font-size: 13px; font-weight: 700;")
        self.detail = QLabel(
            "WAV, AIFF, FLAC, MP3 or OGG · up to 10 seconds analyzed"
        )
        self.detail.setObjectName("muted")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("font-size: 9px;")
        self.play_button = QPushButton("  Play uploaded sound")
        self.play_button.setObjectName("compactActionButton")
        self.play_button.setIcon(icon("play"))
        self.play_button.setEnabled(False)
        self.play_button.setToolTip(
            "Play the audio you uploaded, to compare it against a match."
        )
        self.play_button.clicked.connect(self.play_requested)
        layout.addStretch(1)
        layout.addWidget(self.upload_button, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.title)
        layout.addWidget(self.detail)
        layout.addWidget(self.play_button, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)

    def setText(self, text: str) -> None:
        self.title.setText(text)

    def set_playable(self, playable: bool) -> None:
        self.play_button.setEnabled(playable)

    @staticmethod
    def _accepts_audio_drag(mime_data) -> bool:  # type: ignore[no-untyped-def]
        urls = mime_data.urls()
        return (
            len(urls) == 1
            and Path(urls[0].toLocalFile()).suffix.casefold()
            in SUPPORTED_AUDIO_SUFFIXES
        )

    def _set_drag_active(self, active: bool) -> None:
        self.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._accepts_audio_drag(event.mimeData()):
            self._set_drag_active(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._accepts_audio_drag(event.mimeData()):
            self._set_drag_active(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._set_drag_active(False)
        event.accept()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._set_drag_active(False)
        urls = event.mimeData().urls()
        if self._accepts_audio_drag(event.mimeData()):
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()
        else:
            event.ignore()


class LibraryEntryRow(QFrame):
    """Focusable row which opens on double-click or Enter."""

    activated = Signal(str)

    def __init__(self, match_uid: str) -> None:
        super().__init__()
        self.match_uid = match_uid
        self.setObjectName("matchRow")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.activated.emit(self.match_uid)
        event.accept()

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.activated.emit(self.match_uid)
            event.accept()
            return
        super().keyPressEvent(event)

class ScaledGraphicsView(QGraphicsView):
    """QGraphicsView that keeps its scene uniformly fit to the viewport.

    fitInView must be called from the view's own resizeEvent, not the outer
    window's — the view's viewport geometry is not guaranteed to be updated
    yet when the containing QMainWindow's resizeEvent fires, which produces
    a wrong (usually tiny) initial scale if called from there instead.
    """

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if AudioDropLabel._accepts_audio_drag(event.mimeData()):
            super().dragEnterEvent(event)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if AudioDropLabel._accepts_audio_drag(event.mimeData()):
            super().dragMoveEvent(event)
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if AudioDropLabel._accepts_audio_drag(event.mimeData()):
            super().dropEvent(event)
        else:
            event.ignore()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        scene = self.scene()
        if scene is not None:
            self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class LegacyMainWindow(QMainWindow):
    def __init__(
        self,
        *,
        factory_verification: FactoryVerification | None = None,
        privacy_store: PrivacyStore | None = None,
    ) -> None:
        super().__init__()
        self.distribution_mode = distribution_mode()
        self.factory_verification = factory_verification
        self.privacy_store = privacy_store or PrivacyStore()
        set_active_store(self.privacy_store)
        self.privacy_choice = self.privacy_store.load()
        self.factory_mapping_path = (
            ENV.app_data_dir / "factory-paths.json"
            if self.distribution_mode
            else Path(__file__).resolve().parents[1] / "data" / "local" / "factory_paths.json"
        )
        self.local_paths = default_local_paths()
        self._ensure_patchlab_export_folders()
        self.setWindowTitle("PatchLab")
        self.resize(1050, 900)
        self.runner = ScanProcessRunner(self)
        self.runner.log.connect(self.append_log)
        self.runner.progress.connect(self._progress)
        self.runner.stage_progress.connect(self._local_library_progress_changed)
        self.runner.completed.connect(self._scan_completed)
        self.runner.failed.connect(self._scan_failed)
        self.fingerprint_runner = ScanProcessRunner(self)
        self.fingerprint_runner.log.connect(self.append_log)
        self.fingerprint_runner.stage_progress.connect(self._local_library_progress_changed)
        self.fingerprint_runner.completed.connect(self._fingerprint_completed)
        self.fingerprint_runner.failed.connect(self._fingerprint_failed)
        self.update_check_runner = UpdateCheckProcessRunner(self)
        self.update_check_runner.log.connect(self.append_log)
        self.update_check_runner.completed.connect(self._update_check_completed)
        self.update_check_runner.failed.connect(self._update_check_failed)
        self._update_check_manual = False
        self._update_check_auth_retry_attempted = False
        self.update_download_runner = UpdateDownloadProcessRunner(self)
        self.update_download_runner.log.connect(self.append_log)
        self.update_download_runner.progress.connect(self._update_download_progress)
        self.update_download_runner.completed.connect(self._update_download_completed)
        self.update_download_runner.failed.connect(self._update_download_failed)
        self.render_runner = RenderProcessRunner(self)
        self.render_runner.log.connect(self.append_log)
        self.render_runner.progress.connect(self._render_progress_changed)
        self.render_runner.completed.connect(self._render_completed)
        self.render_runner.failed.connect(self._render_failed)
        self.render_runner.control_changed.connect(self._render_control_changed)
        self.analyze_runner = AnalyzeProcessRunner(self)
        self.analyze_runner.log.connect(self.append_log)
        self.analyze_runner.progress.connect(self._analyze_progress_changed)
        self.analyze_runner.completed.connect(self._analyze_completed)
        self.analyze_runner.failed.connect(self._analyze_failed)
        self.match_runner = MatchProcessRunner(self)
        self.match_runner.log.connect(self.append_log)
        self.match_runner.progress.connect(self._match_progress_changed)
        self.match_runner.completed.connect(self._match_completed)
        self.match_runner.failed.connect(self._match_failed)
        self.export_runner = ExportProcessRunner(self)
        self.export_runner.log.connect(self.append_log)
        self.export_runner.completed.connect(self._export_completed)
        self.export_runner.failed.connect(self._export_failed)
        self.preview_runner = PreviewProcessRunner(self)
        self.preview_runner.log.connect(self.append_log)
        self.preview_runner.request_completed.connect(self._preview_request_completed)
        self.preview_runner.request_failed.connect(self._preview_request_failed)
        self._render_paused = False
        self._render_failure_detail: str = ""
        # True while Render Sound Library is driving the shared scan runner,
        # so its progress and failures land on the Render card rather than Link.
        self._compact_render_active = False
        self._prepare_active = False
        self._prepare_total = 0
        self._prepare_completed = 0
        self._prepare_started_at = 0.0
        # Last capability snapshot, so a later check can tell that a synth
        # newly became available rather than merely being present.
        self._capability_snapshot = None
        self._model_asset_error: str | None = None
        self._match_audio_path: Path | None = None
        self._match_result_path: Path | None = None
        self._match_result: dict | None = None
        self._existing_matches: list[dict] = []
        self._existing_page = 0
        self._favorite_hashes: set[str] = set()
        self._current_match_uid: str | None = None
        self._preview_requests: dict[
            str, list[tuple[QPushButton | None, str, int]]
        ] = {}
        self._preview_inflight: dict[tuple[str, int], str] = {}
        self._preview_request_targets: dict[str, tuple[str, int]] = {}
        self._preview_silent_requests: set[str] = set()
        self._export_context_uid: str | None = None
        self._batch_state: dict | None = None
        self._workflow_activities: dict[str, WorkflowActivity] = {}
        self._workflow_last_match_complete = False
        self._match_session = tempfile.TemporaryDirectory(
            prefix="patchlab-match-app-"
        )

        central = QWidget(self)
        layout = QVBoxLayout(central)
        title = QLabel("PatchLab")
        title.setStyleSheet("font-size: 28px; font-weight: 650;")
        subtitle = QLabel(
            "Catalog, render, learn, and match your Serum sound library. "
            "Serum 1 uses native FXP loading; Serum 2 uses audio-verified reconstructed render state."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        self.factory_status = QLabel("")
        self.factory_status.setWordWrap(True)
        self.factory_status.setVisible(self.distribution_mode)
        layout.addWidget(self.factory_status)

        self.scan_button, self.scan_progress = self._section(
            layout,
            "1. Link My Preset Folder"
            if self.distribution_mode
            else "1. Select Preset Folder",
            (
                "Optionally link your own presets so they become searchable here and "
                "contribute the preset files and fingerprints—never audio—to the shared library."
                if self.distribution_mode
                else "Recursively scan FXP and SerumPreset files, deduplicate by content, and dump every exposed parameter."
            ),
            enabled=True,
        )
        self.scan_box = self.scan_button.parentWidget()
        self.scan_button.clicked.connect(self.choose_folder)
        self.privacy_settings = QGroupBox("Privacy")
        privacy_layout = QVBoxLayout(self.privacy_settings)
        self.share_toggle = QCheckBox("Use && share my own presets")
        self.share_toggle.setToolTip(
            "When on, linked presets are processed locally and preset files plus fingerprints "
            "may be contributed. Rendered audio is never uploaded."
        )
        self.share_toggle.setChecked(
            bool(self.privacy_choice.use_and_share_own_presets)
        )
        self.share_toggle.toggled.connect(self._privacy_toggled)
        privacy_layout.addWidget(self.share_toggle)
        self.privacy_settings.setVisible(self.distribution_mode)
        layout.addWidget(self.privacy_settings)
        render_ready = self._render_library_complete()
        self.render_button, self.render_progress = self._section(
            layout,
            "2. Render Sound Library",
            "Render seven sustained C notes per verified preset. Available after the scan gate passes.",
            enabled=render_ready,
        )
        self.render_button.clicked.connect(self.start_render)
        render_controls = QHBoxLayout()
        self.render_pause_button = QPushButton("Pause")
        self.render_pause_button.setEnabled(False)
        self.render_pause_button.clicked.connect(self.toggle_render_pause)
        self.render_cancel_button = QPushButton("Cancel")
        self.render_cancel_button.setEnabled(False)
        self.render_cancel_button.clicked.connect(self.cancel_render)
        self.render_stats = QLabel("Ready")
        render_controls.addWidget(self.render_pause_button)
        render_controls.addWidget(self.render_cancel_button)
        render_controls.addWidget(self.render_stats, 1)
        layout.addLayout(render_controls)
        self.learn_button, self.learn_progress = self._section(
            layout,
            "3. Analyze & Learn",
            "Compute CLAP/spectral features and train the parameter model.",
            enabled=self._render_library_complete(),
        )
        self.learn_button.clicked.connect(self.start_analyze)
        learn_options = QHBoxLayout()
        self.deep_training = QCheckBox("Deep training (recommended, slower)")
        self.deep_training.setChecked(True)
        self.analyze_cancel_button = QPushButton("Cancel")
        self.analyze_cancel_button.setEnabled(False)
        self.analyze_cancel_button.clicked.connect(self.analyze_runner.cancel)
        self.analyze_cancel_button.clicked.connect(self.fingerprint_runner.cancel)
        self.analyze_stats = QLabel("Ready")
        learn_options.addWidget(self.deep_training)
        learn_options.addWidget(self.analyze_cancel_button)
        learn_options.addWidget(self.analyze_stats, 1)
        layout.addLayout(learn_options)
        match_ready = self.distribution_mode or (
            (Path(__file__).resolve().parents[1] / "data" / "models" / "param_model.pt").is_file()
            and (Path(__file__).resolve().parents[1] / "data" / "features" / "preset_index.npy").is_file()
        )
        self.match_button, self.match_progress = self._section(
            layout,
            "4. Match a Sound",
            "Upload or drop audio, browse the ten closest owned presets, and generate a native Serum preset.",
            enabled=match_ready,
        )
        self.match_button.setText("Select Audio")
        self.match_button.clicked.connect(self.choose_match_file)
        self.match_drop = AudioDropLabel()
        self.match_drop.file_dropped.connect(self._set_match_file)
        layout.addWidget(self.match_drop)
        match_options = QHBoxLayout()
        self.match_offset = QDoubleSpinBox()
        self.match_offset.setRange(0.0, 86_400.0)
        self.match_offset.setDecimals(2)
        self.match_offset.setSuffix(" s offset")
        self.match_offset.setToolTip("For files over ten seconds, choose where analysis begins.")
        self.match_budget = QComboBox()
        self.match_budget.addItem("Quick · about 15 seconds", "quick")
        self.match_budget.addItem("Balanced · about 1 minute", "balanced")
        self.match_budget.addItem("Best Quality · 2–5 minutes", "best")
        self.match_budget.setCurrentIndex(1)
        self.match_synth = QComboBox()
        self.match_synth.addItem("Generate for Serum 2", "serum2")
        self.match_synth.addItem("Generate for Serum 1", "serum1")
        self.match_start_button = QPushButton("Run Match")
        self.match_start_button.setEnabled(False)
        self.match_start_button.clicked.connect(self.start_match)
        self.match_cancel_button = QPushButton("Cancel")
        self.match_cancel_button.setEnabled(False)
        self.match_cancel_button.clicked.connect(self.match_runner.cancel)
        match_options.addWidget(self.match_offset)
        match_options.addWidget(self.match_budget)
        match_options.addWidget(self.match_synth)
        match_options.addWidget(self.match_start_button)
        match_options.addWidget(self.match_cancel_button)
        layout.addLayout(match_options)
        self.match_stats = QLabel("Choose WAV, MP3, FLAC, OGG, or AIFF.")
        self.match_stats.setWordWrap(True)
        layout.addWidget(self.match_stats)

        self.match_results = QGroupBox("Match results")
        results_layout = QVBoxLayout(self.match_results)
        self.existing_heading = QLabel("Closest presets you own")
        self.existing_heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.existing_table = QTableWidget(0, 7)
        self.existing_table.setHorizontalHeaderLabels(
            ["Preset", "Synth", "Similarity", "Source path", "Audition", "Save Copy",
             "Open Preset File Location"]
        )
        self.existing_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.existing_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.existing_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.existing_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.existing_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.existing_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.existing_table.setMinimumHeight(285)
        self.recommendation_heading = QLabel("Recommended new preset")
        self.recommendation_heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.recommendation_confidence = QLabel("")
        self.recommendation_confidence.setWordWrap(True)
        recommendation_actions = QHBoxLayout()
        self.winner_play_button = QPushButton("▶ Audition recommendation")
        self.winner_play_button.clicked.connect(self.play_winner)
        self.save_preset_button = QPushButton("Save as preset…")
        self.save_preset_button.clicked.connect(self.save_match_preset)
        recommendation_actions.addWidget(self.winner_play_button)
        recommendation_actions.addWidget(self.save_preset_button)
        recommendation_actions.addStretch(1)
        self.settings_tree = QTreeWidget()
        self.settings_tree.setHeaderLabels(["Section / setting", "Value"])
        self.settings_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.settings_tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.settings_tree.setMinimumHeight(260)
        limitation = QLabel(
            "Wavetable note: recommendations retain the selected base preset’s wavetable "
            "references and settings, but cannot create missing third-party wavetable content."
        )
        limitation.setWordWrap(True)
        limitation.setStyleSheet("color: #666; font-size: 11px;")
        results_layout.addWidget(self.existing_heading)
        results_layout.addWidget(self.existing_table)
        results_layout.addWidget(self.recommendation_heading)
        results_layout.addWidget(self.recommendation_confidence)
        results_layout.addLayout(recommendation_actions)
        results_layout.addWidget(self.settings_tree)
        results_layout.addWidget(limitation)
        self.match_results.setVisible(False)
        layout.addWidget(self.match_results)

        log_label = QLabel("Activity log")
        log_label.setStyleSheet("font-weight: 600;")
        self.log_pane = QPlainTextEdit()
        self.log_pane.setReadOnly(True)
        self.log_pane.setMaximumBlockCount(10_000)
        self.log_pane.setMinimumHeight(160)
        layout.addWidget(log_label)
        layout.addWidget(self.log_pane)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)
        self.statusBar().showMessage(f"Ready — {ENV.branch}, compute: {ENV.compute_backend}")
        if self.distribution_mode:
            self._apply_factory_status()
            self._apply_privacy_choice()
            if self.privacy_choice.use_and_share_own_presets is None:
                QTimer.singleShot(0, self._show_consent_dialog)

    def _apply_factory_status(self) -> None:
        verification = self.factory_verification
        if verification is None:
            text = "Checking installed factory presets in the background…"
            color = "#64748B"
        elif not verification.bundle_available:
            text = (
                "Factory fingerprint bundle is unavailable. Reinstall PatchLab to restore "
                "instant factory matching."
            )
            color = "#9b2c2c"
        elif verification.no_factory_install:
            text = (
                "Factory fingerprints are ready. No local Serum factory preset folders were "
                "found; matching still works, while factory audition and preset export remain unavailable."
            )
            color = "#9a6700"
        else:
            text = (
                f"Factory library ready in {verification.elapsed_s:.2f}s: "
                f"{verification.matched_hashes:,}/{verification.known_bundle_hashes:,} "
                "factory presets matched locally. No rendering was needed."
            )
            if verification.missing_hashes:
                text += (
                    f" {len(verification.missing_hashes):,} bundled fingerprints do not "
                    "match a local file; matching remains available."
                )
            color = "#087443"
        self.factory_status.setText(text)
        self.factory_status.setStyleSheet(
            f"border: 1px solid {color}; border-radius: 6px; padding: 9px; color: {color};"
        )

    def _show_consent_dialog(self) -> None:
        if not self.distribution_mode or self.privacy_store.load().use_and_share_own_presets is not None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Your preset library is optional")
        dialog.setModal(True)
        dialog.setMinimumWidth(520)
        layout = QVBoxLayout(dialog)
        heading = QLabel("Would you like to link your preset library to PatchLab?")
        heading.setStyleSheet("font-size: 18px; font-weight: 650;")
        body = QLabel(
            "If enabled, you can designate a local folder for PatchLab to index. "
            "PatchLab will scan your linked files locally to enable local search "
            "functionality while synchronizing non-factory preset definitions, "
            "configuration metadata, and acoustic fingerprints with PatchLab’s "
            "global database. Rendered audio files remain exclusively on your "
            "device.\n\n"
            "If declined, PatchLab will not be able to analyze your presets or "
            "match uploaded audio to its nearest match in your presets. It will "
            "only match to closest factory presets. You can update this "
            "preference anytime in Privacy settings."
        )
        body.setWordWrap(True)
        buttons = QHBoxLayout()
        disagree = QPushButton("Disagree")
        agree = QPushButton("Agree")
        disagree.setMinimumHeight(38)
        agree.setMinimumHeight(38)
        buttons.addWidget(disagree, 1)
        buttons.addWidget(agree, 1)
        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addLayout(buttons)

        def choose(value: bool) -> None:
            self.privacy_choice = self.privacy_store.save(value)
            self.share_toggle.blockSignals(True)
            self.share_toggle.setChecked(value)
            self.share_toggle.blockSignals(False)
            if not value:
                self._stop_user_preset_work()
            self._apply_privacy_choice()
            dialog.accept()

        disagree.clicked.connect(lambda: choose(False))
        agree.clicked.connect(lambda: choose(True))
        dialog.exec()

    def _privacy_toggled(self, enabled: bool) -> None:
        if not self.distribution_mode:
            return
        self.privacy_choice = self.privacy_store.save(enabled)
        if not enabled:
            self._stop_user_preset_work()
        else:
            self.append_log(
                "Personal presets are on again. Anything already learned is usable "
                "again, and waiting presets can be processed."
            )
        self._apply_privacy_choice()

    def _stop_user_preset_work(self) -> None:
        """Stop every job that works on the user's own presets. Deletes nothing.

        The choice is already saved, and every processing boundary re-reads it,
        so this is what makes the stop prompt rather than what makes it safe.
        All of these jobs are resumable; stored presets, fingerprints and renders
        stay on disk but are inactive until the user turns the setting back on.
        """

        stopped = []
        for label, runner in (
            ("preset scan", getattr(self, "runner", None)),
            ("preset rendering", getattr(self, "render_runner", None)),
            ("learning", getattr(self, "fingerprint_runner", None)),
            ("analysis", getattr(self, "analyze_runner", None)),
        ):
            if runner is not None and getattr(runner, "running", False):
                runner.cancel()
                stopped.append(label)
        for card in ("link", "render", "analyze"):
            self._workflow_activities.pop(card, None)
        self._automatic_link_scan_active = False
        self._compact_render_active = False
        self.append_log(
            "Personal presets are off. PatchLab will match against factory presets "
            "only, and will not scan, render, analyze or share your presets."
            + (f" Stopped: {', '.join(stopped)}." if stopped else "")
        )

    def _explain_personal_presets_off(self) -> None:
        QMessageBox.information(
            self,
            "Turn on personal presets",
            "Turn on “Use & share my own presets” in Settings → Privacy first. "
            "While it is off, PatchLab matches against factory presets only and "
            "does not scan, render, analyze or share your own presets.",
        )

    def _apply_privacy_choice(self) -> None:
        enabled = bool(self.privacy_choice.use_and_share_own_presets)
        self.scan_box.setEnabled(True)
        self.scan_box.setToolTip(
            "" if enabled else "Turn on “Use & share my own presets” in Privacy to link a folder."
        )
        self._refresh_workflow_cards()

    def _set_workflow_activity(
        self,
        card: str,
        current: int,
        total: int,
        text: str,
    ) -> None:
        self._workflow_activities[card] = WorkflowActivity(
            current=max(int(current), 0),
            total=max(int(total), 0),
            text=text,
        )
        self._refresh_workflow_cards()

    def _clear_workflow_activity(self, *cards: str) -> None:
        for card in cards:
            self._workflow_activities.pop(card, None)
        self._refresh_workflow_cards()

    def _refresh_workflow_cards(self) -> None:
        """Refresh all four cards together from persisted and live state."""

        if not hasattr(self, "hero_cards"):
            return
        storage = storage_status()
        state = resolve_workflow_state(
            privacy=self.privacy_choice,
            local_database_path=self.local_paths["db"],
            audio_selected=self._match_audio_path is not None,
            match_completed=self._workflow_last_match_complete,
            activities=self._workflow_activities,
            match_prerequisite_error=self._model_asset_error or "",
            compact_mode=bool(
                getattr(
                    self,
                    "storage_preferences",
                    load_storage_preferences(),
                ).compact_mode
            ),
            audio_storage_error=storage.reason if not storage.available else "",
            render_failed_detail=(
                self._render_failure_detail
                if "render" not in self._workflow_activities
                else ""
            ),
        )
        resolved_cards = state.as_dict()
        phases = {name: card_state.phase for name, card_state in resolved_cards.items()}
        previous_phases = getattr(self, "_last_card_phases", None)
        if previous_phases is not None and phases != previous_phases:
            self._ui_event(
                "workflow_state_changed",
                "workflow card state changed",
                changed={
                    name: [previous_phases.get(name), phase]
                    for name, phase in phases.items()
                    if previous_phases.get(name) != phase
                },
                activities=sorted(self._workflow_activities),
            )
        self._last_card_phases = phases
        card_keys = (
            ("link", "render", "match")
            if len(self.hero_cards) == 3
            else ("link", "render", "analyze", "match")
        )
        for key, card in zip(card_keys, self.hero_cards, strict=True):
            resolved: WorkflowCardState = resolved_cards[key]
            card.setWorkflowState(
                resolved.phase,
                resolved.text,
                resolved.current,
                resolved.total,
                detail=resolved.detail,
            )

        self.scan_button.setEnabled("link" not in self._workflow_activities)
        self.render_button.setEnabled(
            "render" not in self._workflow_activities
            and state.render.phase == "needs-action"
        )
        # In distribution mode this is an informational, optional action. It
        # explains how linked presets join retrieval without replacing the
        # shipped training set.
        self.learn_button.setEnabled("analyze" not in self._workflow_activities)
        self.match_button.setEnabled(
            "match" not in self._workflow_activities and not state.match.detail
        )
        self._workflow_match_error = state.match.detail
        if hasattr(self, "match_start_button"):
            self.match_start_button.setEnabled(
                self._match_audio_path is not None
                and "match" not in self._workflow_activities
                and not self._workflow_match_error
                and self._model_asset_error is None
            )

    def _local_library_progress_changed(self, detail: dict) -> None:
        # Automatic maintenance must never repaint the workflow/status area or
        # make the freshly opened application look occupied. Manual jobs keep
        # their detailed progress exactly as before.
        if getattr(self, "_automatic_link_scan_active", False):
            return
        stage = str(detail.get("stage", "scan"))
        current = int(detail.get("current", 0))
        total = int(detail.get("total", 0))
        text = str(detail.get("text", stage.replace("-", " ").title()))
        if getattr(self, "_prepare_active", False) and stage in {
            "discovery", "prepare-queue", "ingest", "scan"
        }:
            # A Prepare click owns its incremental source check.  The worker
            # records every stat/hash decision but samples this signal at 8 Hz,
            # so Qt stays responsive while a large library is reconciled.
            if stage == "prepare-queue":
                self._prepare_total = total
                self.render_progress.setRange(0, max(total, 1))
                self.render_progress.setValue(0)
                self.render_stats.setText(text)
                self._set_workflow_activity(
                    "render", 0, total, "Preparing Preset Library · " + text
                )
                return
            elapsed = max(time.monotonic() - self._prepare_started_at, 0.0)
            eta_text = "Estimating time…"
            if current >= 2 and elapsed > 0:
                remaining = max(total - current, 0)
                seconds = int((elapsed / current) * remaining)
                if seconds < 90:
                    eta_text = "About 1 minute remaining"
                elif seconds < 3600:
                    eta_text = f"About {max(1, round(seconds / 60))} minutes remaining"
                else:
                    eta_text = f"About {max(1, round(seconds / 3600))} hours remaining"
            self.render_stats.setText(f"{eta_text} · {text}")
            self._workflow_activities.pop("link", None)
            self._workflow_activities.pop("analyze", None)
            self._set_workflow_activity("render", current, total, text)
            return
        if getattr(self, "_prepare_active", False) and stage == "prepare":
            # Phase 3 reports a terminal item only after durable PREPARED
            # commit (or a per-preset failure). The UI keeps the initial queue
            # denominator and clamps the numerator so progress never regresses.
            self._prepare_completed = max(self._prepare_completed, current)
            completed = min(self._prepare_completed, self._prepare_total)
            self.render_progress.setRange(0, max(self._prepare_total, 1))
            self.render_progress.setValue(completed)
            elapsed = max(time.monotonic() - self._prepare_started_at, 0.0)
            eta_text = "Estimating time…"
            if completed >= 2 and elapsed > 0:
                remaining = max(self._prepare_total - completed, 0)
                seconds = int((elapsed / completed) * remaining)
                if seconds < 90:
                    eta_text = "About 1 minute remaining"
                elif seconds < 3600:
                    eta_text = f"About {max(1, round(seconds / 60))} minutes remaining"
                else:
                    eta_text = f"About {max(1, round(seconds / 3600))} hours remaining"
            stage_label = str(detail.get("current_stage", "prepare")).replace("_", " ")
            name = str(detail.get("preset_name", ""))
            current_text = f"{stage_label.title()} “{name}”" if name else text
            self.render_stats.setText(
                f"{eta_text} · {completed} / {self._prepare_total} prepared · {current_text}"
            )
            self._set_workflow_activity(
                "render",
                completed,
                self._prepare_total,
                f"Preparing Preset Library · {completed} / {self._prepare_total} prepared",
            )
            return
        if stage == "render":
            self._workflow_activities.pop("link", None)
            self._workflow_activities.pop("analyze", None)
            self._set_workflow_activity("render", current, total, text)
        elif stage == "analyze":
            self._workflow_activities.pop("link", None)
            self._workflow_activities.pop("render", None)
            self._set_workflow_activity("analyze", current, total, text)
        elif getattr(self, "_compact_render_active", False):
            # Render Sound Library owns this job. Its catalog/classification
            # pass still reports stage="scan", but showing that on the Link card
            # made a Render click look like linking had restarted.
            self._workflow_activities.pop("link", None)
            self._workflow_activities.pop("analyze", None)
            self._set_workflow_activity("render", current, total, text)
        else:
            self._workflow_activities.pop("render", None)
            self._workflow_activities.pop("analyze", None)
            self._set_workflow_activity("link", current, total, text)

    def _render_library_complete(self) -> bool:
        database_path = self.local_paths["db"]
        if not database_path.is_file():
            return False
        import sqlite3

        connection = sqlite3.connect(database_path)
        presets = int(connection.execute("SELECT COUNT(*) FROM presets").fetchone()[0])
        rendered = int(
            connection.execute(
                "SELECT COUNT(*) FROM (SELECT preset_id FROM renders "
                "GROUP BY preset_id HAVING COUNT(DISTINCT midi_note)>=7)"
            ).fetchone()[0]
        )
        fingerprinted = int(
            connection.execute(
                "SELECT COUNT(*) FROM fingerprints WHERE midi_note=0"
            ).fetchone()[0]
        )
        connection.close()
        compact_mode = bool(
            getattr(
                self, "storage_preferences", load_storage_preferences()
            ).compact_mode
        )
        ready = fingerprinted if compact_mode else rendered
        return presets > 0 and ready >= presets

    def _section(
        self, layout: QVBoxLayout, title: str, description: str, *, enabled: bool
    ) -> tuple[QPushButton, QProgressBar]:
        box = QGroupBox(escape_mnemonic(title))
        row = QVBoxLayout(box)
        label = QLabel(description)
        label.setWordWrap(True)
        controls = QHBoxLayout()
        button = QPushButton(title.split(". ", 1)[-1])
        button.setEnabled(enabled)
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        controls.addWidget(button)
        controls.addWidget(progress, 1)
        row.addWidget(label)
        row.addLayout(controls)
        layout.addWidget(box)
        return button, progress

    def choose_folder(self) -> None:
        if self.distribution_mode and not user_presets_enabled():
            QMessageBox.information(
                self,
                "Turn on personal presets",
                "Turn on “Use & share my own presets” in Settings → Privacy, "
                "then click this card again to link a folder.",
            )
            return
        storage = storage_status()
        if not storage.available:
            QMessageBox.warning(
                self,
                "Audio storage is disconnected",
                f"{storage.reason}\n\nReconnect it or choose a new Audio Storage "
                "location in Settings. Matching remains available.",
            )
            self.append_log(f"Preset processing not started: {storage.reason}")
            return
        defaults = ENV.existing_preset_roots
        initial = str(defaults[0] if defaults else Path.home())
        selected = QFileDialog.getExistingDirectory(self, "Select Serum preset folder", initial)
        if not selected:
            return
        if self.distribution_mode:
            answer = QMessageBox.question(
                self,
                "Start local preset-library processing?",
                "PatchLab will check this folder and show presets that need "
                "preparation. It will not prepare them until you click Prepare "
                "Preset Library.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.append_log("Local preset-library processing was not started")
                return
        if self.distribution_mode:
            self.privacy_choice = self.privacy_store.save(
                True, linked_folder=Path(selected)
            )
        self._set_workflow_activity("link", 0, 0, "Checking preset library…")
        self.append_log(
            f"Checking linked preset library for {selected}"
            if self.distribution_mode
            else f"Starting isolated scan worker for {selected}"
        )
        self.statusBar().showMessage(
            "Checking your preset library…"
            if self.distribution_mode
            else "Scanning and dumping parameters…"
        )
        self.runner.start(
            Path(selected), refresh_only=self.distribution_mode
        )

    def maybe_start_automatic_link_scan(self, *, env: PlatformEnv = ENV) -> None:
        """Quietly catch up an already-linked folder on launch, at most once a day.

        This only reconciles the folder manifest after consent. Preparation
        remains an explicit action from the Prepare Preset Library button.
        """

        if not self.distribution_mode:
            return
        if not (
            self.privacy_choice.use_and_share_own_presets is True
            and user_presets_enabled()
            and self.privacy_choice.linked_folder
        ):
            return
        folder = Path(self.privacy_choice.linked_folder)
        if not folder.is_dir():
            return
        if not storage_status().available:
            return
        if self.runner.running or "link" in self._workflow_activities:
            return
        # Never add rendering load while the user is already matching. A later
        # launch will safely resume the incremental check instead.
        if getattr(self, "match_runner", None) is not None and self.match_runner.running:
            return
        if not auto_scan_due(env):
            return
        record_auto_scan(env)
        self._automatic_link_scan_active = True
        self._automatic_log_lines_suppressed = 0
        self.append_log("Background preset check started; matching remains ready.")
        # One worker deliberately leaves processor and disk headroom for an
        # immediate Match request. Manual library processing still uses four.
        self.runner.start(folder, refresh_only=True, workers=1)

    def maybe_verify_factory_install(self) -> None:
        """Build local factory-path mapping after the window is interactive."""

        if not self.distribution_mode or self.factory_verify_runner.running:
            return
        self.factory_verify_runner.start()

    def _factory_verify_completed(self, payload: dict) -> None:
        try:
            self.factory_verification = FactoryVerification(
                bundle_available=bool(payload["bundle_available"]),
                factory_directories_found=int(payload["factory_directories_found"]),
                local_files_found=int(payload["local_files_found"]),
                known_bundle_hashes=int(payload["known_bundle_hashes"]),
                matched_hashes=int(payload["matched_hashes"]),
                missing_hashes=tuple(str(value) for value in payload["missing_hashes"]),
                unknown_local_hashes=tuple(
                    str(value) for value in payload["unknown_local_hashes"]
                ),
                local_paths_by_hash={
                    str(key): str(value)
                    for key, value in dict(payload["local_paths_by_hash"]).items()
                },
                elapsed_s=float(payload["elapsed_s"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            self.append_log(f"Factory preset check returned invalid data: {exc}")
            return
        self._apply_factory_status()
        self.append_log("Installed factory preset check finished in the background.")

    def _factory_verify_failed(self, error: str) -> None:
        # The shipped fingerprint library is independent of local audition and
        # remains usable even if a local path check has a temporary problem.
        self.append_log(f"Installed factory preset check deferred: {error}")
        self.factory_status.setText(
            "Factory fingerprints remain ready. Local factory audition will retry "
            "on the next launch."
        )

    def _local_library_log(self, message: str) -> None:
        if not getattr(self, "_automatic_link_scan_active", False):
            self.append_log(message)
            return
        if "FAILED" in message.upper() or "ERROR" in message.upper():
            self.append_log(f"Background preset check: {message}")
        else:
            self._automatic_log_lines_suppressed += 1

    def maybe_check_for_update(self, *, env: PlatformEnv = ENV) -> None:
        """Quietly check the appropriate trusted PatchLab release channel."""

        if not self.distribution_mode:
            return
        if not load_update_preferences(env).auto_check:
            return
        if self.update_check_runner.running:
            return
        self._start_update_check(manual=False)

    def _start_update_check(self, *, manual: bool) -> None:
        """The one place that launches a check, so completion knows its origin.

        A manual click and a quiet startup check share the same worker and
        the same completion signal; only ``manual`` decides whether a
        CHECK_FAILED or AUTH_REQUIRED result gets a dialog. A manual check
        always resolves to something the user sees -- see
        ``_update_check_completed``/``_update_check_failed``.

        Does not touch ``_update_check_auth_retry_attempted``: this is also
        how the sign-in retry itself re-launches the check, and resetting
        the flag here would let that retry's own result trigger a second,
        unbounded sign-in attempt. Only a genuinely new manual request (the
        button) resets it.
        """

        self._update_check_manual = manual
        self.update_check_runner.start()

    def _update_check_completed(self, result: dict, *, env: PlatformEnv = ENV) -> None:
        manual = getattr(self, "_update_check_manual", False)
        state = str(
            result.get("state")
            or ("update_available" if result.get("update_available") else "current")
        )
        if state == "update_available":
            remote = result.get("remote_version")
            if not remote:
                return
            # A manual click means the user wants the real answer now, even
            # for a version they previously chose to skip.
            if not manual and load_update_preferences(env).skipped_version == remote:
                return
            self._update_check_auth_retry_attempted = False
            package = result.get("package")
            if isinstance(package, dict):
                self._prompt_update_available(str(remote), package=package, env=env)
            else:
                self._prompt_update_available(str(remote), env=env)
            return
        if state == "auth_required":
            self._handle_update_check_auth_required(manual=manual)
            return
        if state == "check_failed":
            message = str(
                result.get("user_message")
                or "PatchLab couldn't check for updates right now. Try again in a moment."
            )
            self.append_log(
                f"Update check failed ({result.get('failure_category', 'unknown')}): {message}"
            )
            if manual:
                QMessageBox.information(self, "Couldn't check for updates", message)
            return
        # CURRENT: automatic checks stay silent; a manual click still needs
        # an answer, or clicking the button would look like nothing happened.
        self._update_check_auth_retry_attempted = False
        if manual:
            QMessageBox.information(
                self, "You're up to date", f"PatchLab is up to date (v{__version__})."
            )

    def _update_check_failed(self, error: str) -> None:
        """The worker process itself did not complete (a crash, not CHECK_FAILED)."""

        self.append_log(error)
        if getattr(self, "_update_check_manual", False):
            QMessageBox.information(
                self,
                "Couldn't check for updates",
                "PatchLab couldn't check for updates right now. Try again in a moment.",
            )

    def _handle_update_check_auth_required(self, *, manual: bool) -> None:
        if not manual:
            # Startup checks must never pop a sign-in dialog unprompted --
            # core.access_gate.ensure_relay_token() already tried silently at
            # launch. Just record it; a manual check will surface this.
            self.append_log("Update check needs sign-in; try Check for Updates Now.")
            return
        if getattr(self, "_update_check_auth_retry_attempted", False):
            QMessageBox.information(
                self,
                "Sign-in needed",
                "PatchLab needs you to sign in to the support service to check for updates.",
            )
            return
        self._update_check_auth_retry_attempted = True
        self.append_log("Update check needs sign-in; asking now…")
        if self._reconnect_support_service() and not self.update_check_runner.running:
            self.append_log("Signed in; re-checking for updates…")
            self._start_update_check(manual=True)
        else:
            self.append_log("Sign-in was not completed; couldn't check for updates.")

    def _prompt_update_available(
        self,
        remote_version: str,
        *,
        package: dict[str, object] | None = None,
        env: PlatformEnv = ENV,
    ) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Update available")
        box.setText(
            f"PatchLab {remote_version} is available. You have {__version__}.\n\n"
            "Updating replaces PatchLab's app code only. Your linked presets, "
            "rendered audio, and everything the app has already learned live "
            "outside the app and are never touched by an update."
        )
        update_button = box.addButton("Update Now", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Remind Me Later", QMessageBox.ButtonRole.RejectRole)
        skip_button = box.addButton(
            "Skip This Version", QMessageBox.ButtonRole.DestructiveRole
        )
        box.exec()
        clicked = box.clickedButton()
        if clicked is update_button:
            if package is not None:
                self._download_package_update(package)
            else:
                self._apply_update()
        elif clicked is skip_button:
            preferences = load_update_preferences(env)
            save_update_preferences(
                UpdatePreferences(preferences.auto_check, remote_version), env
            )
        # "Remind Me Later" records nothing, so the next launch asks again.

    def _apply_update(self) -> None:
        import os
        import subprocess

        install_root = Path(__file__).resolve().parents[1]
        app_bundle = os.environ.get("PATCHLAB_APP_BUNDLE") or str(
            Path.home() / "Applications" / "PatchLab.app"
        )
        updater = install_root / "scripts" / "apply_update.sh"
        subprocess.Popen(
            ["/bin/bash", str(updater), str(os.getpid()), str(install_root), app_bundle],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.append_log(
            "Update starting; PatchLab will quit and relaunch automatically "
            "once it finishes."
        )
        QApplication.instance().quit()

    def _download_package_update(self, package: dict[str, object]) -> None:
        if self.update_download_runner.running:
            QMessageBox.information(self, "Update download", "PatchLab is already downloading this update.")
            return
        # Never start a multi-gigabyte download that cannot possibly finish.
        space = update_space_preflight(int(package.get("size") or 0))
        self._ui_event(
            "update_space_preflight",
            "checked free space before downloading an update",
            **space.as_dict(),
        )
        if not space.sufficient:
            self.append_log("Update not started: " + space.user_message())
            self.statusBar().showMessage("Not enough free space to install the update.")
            QMessageBox.warning(self, "Not enough free space", space.user_message())
            return
        try:
            self.update_download_runner.start(package)
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Update unavailable", str(exc))
            return
        version = str(package.get("version") or "the new version")
        self.statusBar().showMessage(f"Downloading PatchLab {version}…")
        self.append_log(f"Downloading checksum-verified PatchLab {version} installer…")

    def _update_download_progress(self, detail: dict) -> None:
        percent = int(detail.get("percent", 0))
        self.statusBar().showMessage(f"Downloading PatchLab update… {percent}%")

    def _update_download_failed(self, error: str) -> None:
        self.statusBar().showMessage("Update download did not finish.")
        self.append_log(f"PatchLab update download failed: {error}")
        QMessageBox.warning(
            self,
            "Update download did not finish",
            "The current PatchLab version is unchanged. Please try again later.\n\n"
            f"Details: {error}",
        )

    def _update_download_completed(self, result: dict) -> None:
        package = Path(str(result.get("path") or ""))
        version = str(result.get("version") or "the new version")
        if not package.is_file():
            self._update_download_failed("The downloaded installer could not be found.")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Update ready")
        box.setText(
            f"PatchLab {version} has been downloaded and verified.\n\n"
            "Choose Install Update to close PatchLab and open macOS Installer. "
            "Installer will replace the old PatchLab app while preserving your "
            "settings, library, and rendered audio."
        )
        box.setInformativeText(
            "macOS Installer opens in its own window and you must click through it "
            "to finish. PatchLab quits first, and you can reopen it once Installer "
            "reports success."
        )
        install = box.addButton("Install Update", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is not install:
            self.statusBar().showMessage("Update is downloaded and ready to install.")
            return
        program, arguments = worker_invocation(
            "open-downloaded-update",
            ["--parent-pid", str(os.getpid()), "--package", str(package)],
        )
        if not QProcess.startDetached(program, arguments):
            self._update_download_failed("Could not start macOS Installer.")
            return
        self.append_log(f"Opening macOS Installer for PatchLab {version} after exit.")
        QApplication.instance().quit()

    def append_log(self, message: str) -> None:
        self.log_pane.appendPlainText(message)
        bar = self.log_pane.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _progress(self, current: int, total: int) -> None:
        self._set_workflow_activity(
            "link", current, total, f"Scanning {current:,} of {total:,} presets"
        )

    def _scan_completed(self, summary: dict) -> None:
        automatic = bool(getattr(self, "_automatic_link_scan_active", False))
        compact_render = bool(getattr(self, "_compact_render_active", False))
        preparing = bool(getattr(self, "_prepare_active", False))
        self._automatic_link_scan_active = False
        self._compact_render_active = False
        self._prepare_active = False
        if preparing:
            self._workflow_activities.pop("render", None)
            self.render_cancel_button.setEnabled(False)
            status = preset_library_status(self.local_paths["db"])
            prepared = int(summary.get("fingerprints_created", 0) or 0)
            failed = int(summary.get("failed_load", 0) or 0)
            if status.needs_preparation == 0:
                text = f"{status.ready:,} presets ready · your preset library is up to date."
            else:
                text = (
                    f"{status.ready:,} presets ready · {status.needs_preparation:,} still need "
                    "preparation"
                )
                if failed:
                    text += f" · {failed} couldn't be processed"
            self.render_stats.setText(text)
            self.statusBar().showMessage(text)
            self.append_log(
                f"Preset Library preparation finished: {prepared:,} prepared, "
                f"{status.needs_preparation:,} remaining."
            )
            self._refresh_workflow_cards()
            return
        if summary.get("user_presets_disabled"):
            # The worker refused (or stopped) because personal presets are off.
            # That is the user's choice working, not a failure and not "0 found".
            for card in ("link", "render", "analyze"):
                self._workflow_activities.pop(card, None)
            self.scan_progress.setMaximum(100)
            self.scan_progress.setValue(0)
            self.append_log("Personal presets are off, so your presets were not processed.")
            self.statusBar().showMessage("Personal presets are off — matching uses factory presets only.")
            self._refresh_workflow_cards()
            return
        partial_note = ""
        if compact_render:
            # A partially supported library is a success, not a failure: the
            # supported subset was learned. Say what was skipped and why.
            skipped = int(summary.get("skipped_unsupported_generation", 0) or 0)
            if skipped:
                waiting = {
                    "serum1": ("legacy .fxp presets", "Serum 1"),
                    "serum2": ("Serum 2 presets", "Serum 2"),
                }
                kinds = [
                    waiting[name]
                    for name in str(summary.get("unsupported_generations", "")).split(",")
                    if name in waiting
                ]
                # Name what the files ARE and which synth they need -- never the
                # internal generation id, and never the folder they live in.
                subject = kinds[0][0] if len(kinds) == 1 else "presets"
                needs = " and ".join(dict.fromkeys(item[1] for item in kinds)) or "another Serum"
                note = (
                    f"{skipped:,} {subject} are waiting for {needs}. Your other "
                    "presets were processed, and your preset folder is still linked."
                )
                self.append_log(note)
                partial_note = note
        if self.distribution_mode:
            if int(summary.get("relay_uploaded", 0) or 0):
                self.append_log("Preset contribution uploaded successfully.")
            if int(summary.get("relay_upload_failed", 0) or 0):
                self.append_log(
                    "Your presets were not uploaded. Your original files were not "
                    "changed. You can try again."
                )
        self._workflow_activities.pop("link", None)
        self._workflow_activities.pop("render", None)
        self._workflow_activities.pop("analyze", None)
        self.scan_button.setEnabled(
            not self.distribution_mode
            or bool(self.privacy_choice.use_and_share_own_presets)
        )
        self.scan_progress.setMaximum(100)
        self.scan_progress.setValue(100)
        if self.distribution_mode:
            text = (
                f"Local library ready: found {summary.get('found', 0)}, "
                f"searchable {summary.get('searchable_local', 0)}, "
                f"uploaded {summary.get('relay_uploaded', 0)}, "
                f"already shared {summary.get('relay_already_present', 0)}, "
                f"relay failures {summary.get('relay_upload_failed', 0)}, "
                "relay stopped after failures "
                f"{summary.get('relay_disabled_after_failures', 0)}, "
                f"factory uploads skipped {summary.get('factory_skipped_upload', 0)}, "
                f"compact renders removed {summary.get('compacted_render_files', 0)} "
                f"({float(summary.get('compacted_render_bytes', 0)) / (1024 ** 3):.2f} GiB)"
            )
        else:
            text = (
                f"Scan complete: found {summary.get('found', 0)}, deduped {summary.get('deduped', 0)}, "
                f"params dumped {summary.get('params_dumped', 0)}, failed {summary.get('failed', 0)}, "
                f"Serum 2 unavailable {summary.get('serum2_disabled', 0)}"
            )
        if automatic:
            self.append_log(
                "Background preset check finished; PatchLab stayed ready for matching."
            )
        else:
            self.append_log(text)
            # The actionable "N presets are waiting" line beats the long generic
            # summary; the summary is still in the log.
            self.statusBar().showMessage(partial_note or text)
        if not self.distribution_mode:
            self.render_button.setEnabled(True)
        self._refresh_workflow_cards()
        if compact_render:
            phases = self._card_phases()
            self._ui_event(
                "render_completed",
                "Render Sound Library finished",
                operation_id=getattr(self.runner, "operation_id", ""),
                found=summary.get("found"),
                skipped_unsupported=summary.get("skipped_unsupported_generation", 0),
                fingerprints_created=summary.get("fingerprints_created"),
                link_phase=phases.get("link"),
                render_phase=phases.get("render"),
            )
            # The natural moment for the one informational popup: the user has
            # just seen PatchLab process their folder, so "N presets need Serum X"
            # is actionable now rather than abstract. It runs LAST -- after the
            # activities are cleared and the cards refreshed -- because it is
            # modal, and must not appear over a Render card still showing
            # "in progress". Shown once; it re-arms only if the count changes.
            self._notify_pending_after_scan()

    def _scan_failed(self, error: str) -> None:
        automatic = bool(getattr(self, "_automatic_link_scan_active", False))
        compact_render = bool(getattr(self, "_compact_render_active", False))
        preparing = bool(getattr(self, "_prepare_active", False))
        self._automatic_link_scan_active = False
        self._compact_render_active = False
        self._prepare_active = False
        self._workflow_activities.pop("link", None)
        self._workflow_activities.pop("render", None)
        self._workflow_activities.pop("analyze", None)
        if preparing:
            self._render_failure_detail = self._user_facing_error(error)
            self.render_cancel_button.setEnabled(False)
            self.render_stats.setText(self._render_failure_detail)
            self.statusBar().showMessage(self._render_failure_detail)
            self._refresh_workflow_cards()
            return
        if compact_render:
            # A Render failure must leave the Link card's success alone and put
            # the Render card into a retryable failed state. Nothing here
            # touches privacy_choice.linked_folder, so the folder stays linked.
            self._render_failure_detail = self._user_facing_error(error)
            self.render_button.setEnabled(True)
            self.render_stats.setText(self._render_failure_detail)
            self.append_log(f"Render Sound Library failed: {error}")
            self.statusBar().showMessage(self._render_failure_detail)
            self._refresh_workflow_cards()
            phases = self._card_phases()
            self._ui_event(
                "render_failed",
                "Render Sound Library failed; UI recovered",
                severity="warning",
                operation_id=getattr(self.runner, "operation_id", ""),
                error=error,
                link_phase=phases.get("link"),
                render_phase=phases.get("render"),
                retry_available=bool(
                    self._render_failure_detail
                    and "render" not in self._workflow_activities
                ),
                stale_activities=sorted(self._workflow_activities),
            )
            return
        self.append_log(
            f"{'Background preset check was deferred' if automatic else 'Scan failed'}: {error}"
        )
        if not automatic:
            self.statusBar().showMessage(error)
        self._refresh_workflow_cards()

    @staticmethod
    def _user_facing_error(error: str) -> str:
        """Turn an internal worker error into one concise, actionable sentence.

        Users should never be shown a traceback or an internal class name. The
        full technical detail stays in the flight recorder and the support
        bundle; this is only what appears on a card or in the status bar.
        """

        text = str(error)
        lowered = text.casefold()
        if "no usable serum2 renderer" in lowered or (
            "serum 2" in lowered and "renderer" in lowered
        ):
            return (
                "PatchLab couldn't start Serum 2. Your preset folder is still "
                "linked. Check your Serum 2 installation and try again."
            )
        if "no usable serum1 renderer" in lowered or (
            "serum 1" in lowered and ("renderer" in lowered or "unavailable" in lowered)
        ):
            return (
                "PatchLab couldn't start Serum 1, so legacy .fxp presets were "
                "skipped. Your preset folder is still linked."
            )
        if "renderer" in lowered and "unavailable" in lowered:
            return (
                "PatchLab couldn't start Serum. Your preset folder is still "
                "linked. Check your Serum installation and try again."
            )
        if "workers died" in lowered or "respawn" in lowered:
            return (
                "Serum kept stopping unexpectedly while PatchLab was working. Your "
                "preset folder is still linked. Check your Serum installation and "
                "try again."
            )
        if "stalled" in lowered or "no measurable progress" in lowered:
            return (
                "PatchLab stopped because the job had stopped making progress. "
                "Your earlier setup is unchanged, so you can try again."
            )
        if "disk" in lowered or "no space" in lowered:
            return "PatchLab ran out of disk space. Free some space and try again."
        if "permission" in lowered or "denied" in lowered:
            return (
                "PatchLab could not read a required file. Check the folder's "
                "permissions and try again."
            )
        if "filenotfound" in lowered or "no such file" in lowered:
            return (
                "PatchLab couldn't find a file it needed. If you moved or renamed "
                "it, choose it again and retry."
            )
        if "notadirectory" in lowered:
            return (
                "PatchLab couldn't open your preset folder. Re-link it and try again."
            )
        # Fall back to the raw message, trimmed: better a terse technical line
        # than a spinner that never stops.
        first = text.strip().splitlines()[0] if text.strip() else "Something went wrong."
        return first[:200]

    def start_render(self) -> None:
        if self.distribution_mode and not user_presets_enabled():
            self._ui_event("render_blocked", "personal presets are off")
            self._explain_personal_presets_off()
            return
        if self.distribution_mode:
            linked_folder = self.privacy_choice.linked_folder
            if not linked_folder or not Path(linked_folder).is_dir():
                QMessageBox.information(
                    self,
                    "Link a preset folder first",
                    "Link your preset folder first. PatchLab will then prepare "
                    "only new or changed presets for matching.",
                )
                return
            if self.runner.running:
                self.append_log("Preset Library preparation is already running")
                return
            preset_ids = preparation_queue_ids(self.local_paths["db"])
            if not preset_ids:
                self.render_stats.setText("Your preset library is up to date.")
                self.statusBar().showMessage("Your preset library is up to date.")
                self._refresh_workflow_cards()
                return
            self._prepare_active = True
            self._compact_render_active = False
            self._prepare_total = len(preset_ids)
            self._prepare_completed = 0
            self._prepare_started_at = time.monotonic()
            self._render_failure_detail = ""
            self.render_progress.setRange(0, self._prepare_total)
            self.render_progress.setValue(0)
            self.render_cancel_button.setEnabled(True)
            self.render_pause_button.setVisible(False)
            self.render_stats.setText(f"Estimating time… · 0 / {self._prepare_total} prepared")
            self._set_workflow_activity(
                "render",
                0,
                self._prepare_total,
                f"Preparing Preset Library · 0 / {self._prepare_total} prepared",
            )
            self.statusBar().showMessage("Preparing your preset library…")
            self.runner.start(
                Path(linked_folder),
                local_library=True,
                preset_ids=preset_ids,
            )
            return
        self._ui_event(
            "render_requested",
            "Render Sound Library selected",
            compact_mode=bool(self.storage_preferences.compact_mode),
            folder_linked=bool(self.privacy_choice.linked_folder),
            retry=bool(self._render_failure_detail),
        )
        storage = storage_status()
        if not storage.available:
            QMessageBox.warning(
                self,
                "Audio storage is disconnected",
                f"{storage.reason}\n\nReconnect it or choose a new location in Settings.",
            )
            self.append_log(f"Render not started: {storage.reason}")
            return
        # The distribution build's linked-folder pipeline is deliberately
        # render -> fingerprint -> compact in small batches.  Sending this
        # button straight to render_library() used to bypass that safeguard
        # and retain the complete WAV library until a separate Analyze click.
        # Reuse the incremental pipeline here so the visible Render card can
        # never turn compact storage into a whole-library temporary cache.
        if self.distribution_mode and self.storage_preferences.compact_mode:
            linked_folder = self.privacy_choice.linked_folder
            if not linked_folder or not Path(linked_folder).is_dir():
                QMessageBox.information(
                    self,
                    "Link a preset folder first",
                    "Link your Serum preset folder first. PatchLab will then "
                    "render, learn, and remove temporary WAVs in small batches.",
                )
                return
            if self.runner.running:
                self.append_log("Compact preset processing is already running")
                return
            folder = Path(linked_folder)
            # This is the Render Sound Library button, so its progress and any
            # failure belong to the Render card.
            #
            # Previously this drove the *link* card's activity, and the
            # linked-folder progress stream (stage="scan") drove it too, so
            # clicking Render visibly re-ran "Link My Preset Folder" for the
            # whole catalog pass and a later failure was reported by
            # _scan_failed -- which never sets _render_failure_detail. The
            # reported symptom ("did not render sound library, it just restarted
            # 'link my preset folder'") was exactly that mis-wiring.
            self._compact_render_active = True
            self._render_failure_detail = ""
            self._set_workflow_activity(
                "render", 0, 0, "Starting compact preset-library processing…"
            )
            self.append_log(
                "Starting compact linked-folder processing: renders are learned "
                "and removed in small batches."
            )
            self.statusBar().showMessage("Processing your presets locally…")
            self.runner.start(folder, local_library=True)
            self._ui_event(
                "render_started",
                "compact linked-folder processing started",
                operation_id=getattr(self.runner, "operation_id", ""),
                card_phases=self._card_phases(),
            )
            return
        self._set_workflow_activity("render", 0, 0, "Starting render workers…")
        self.render_pause_button.setEnabled(True)
        self.render_cancel_button.setEnabled(True)
        self.render_progress.setValue(0)
        self._render_paused = False
        self._render_failure_detail = ""
        self.render_pause_button.setText("Pause")
        self.render_stats.setText("Starting four render workers…")
        self.append_log("Starting resumable four-process library render")
        self.statusBar().showMessage("Rendering sound library…")
        if self.distribution_mode:
            self.render_runner.start(
                db_path=self.local_paths["db"],
                audio_root=self.local_paths["audio"],
                state_dir=self.local_paths["states"],
            )
        else:
            self.render_runner.start()

    def toggle_render_pause(self) -> None:
        if self._render_paused:
            self.render_runner.resume()
        else:
            self.render_runner.pause()

    def cancel_render(self) -> None:
        self.render_cancel_button.setEnabled(False)
        if getattr(self, "_prepare_active", False):
            self.render_stats.setText("Cancelling after the current preset…")
            self.runner.cancel()
            return
        self.render_stats.setText("Cancelling after current notes…")
        self.render_runner.cancel()

    def _render_progress_changed(self, detail: dict) -> None:
        current = int(detail.get("completed_note_pairs", 0))
        total = max(int(detail.get("total_note_pairs", 1)), 1)
        rate = float(detail.get("renders_per_second", 0.0))
        eta = detail.get("eta_seconds")
        self.render_progress.setMaximum(total)
        self.render_progress.setValue(current)
        eta_text = "calculating"
        if isinstance(eta, (int, float)):
            hours, remainder = divmod(max(int(eta), 0), 3600)
            minutes, seconds = divmod(remainder, 60)
            eta_text = f"{hours:d}:{minutes:02d}:{seconds:02d}"
        self.render_stats.setText(f"{rate:.2f} renders/s — ETA {eta_text}")
        self._set_workflow_activity(
            "render",
            current,
            total,
            f"Rendering {current:,} of {total:,} notes · ETA {eta_text}",
        )

    def _render_control_changed(self, state: str) -> None:
        if state == "paused":
            self._render_paused = True
            self.render_pause_button.setText("Resume")
            self.render_stats.setText("Paused")
        elif state == "resumed":
            self._render_paused = False
            self.render_pause_button.setText("Pause")

    def _render_completed(self, summary: dict) -> None:
        self._workflow_activities.pop("render", None)
        self.render_button.setEnabled(True)
        self.render_pause_button.setEnabled(False)
        self.render_cancel_button.setEnabled(False)
        cancelled = bool(summary.get("cancelled"))
        if not cancelled:
            self.render_progress.setValue(self.render_progress.maximum())
        text = (
            f"Render {'cancelled' if cancelled else 'complete'}: "
            f"new {summary.get('rendered_note_pairs', 0)}, "
            f"silent {summary.get('silent_note_pairs', 0)}, "
            f"clipped {summary.get('clipped_note_pairs', 0)}"
        )
        self.render_stats.setText(text)
        self.append_log(text)
        self.statusBar().showMessage(text)
        if not cancelled and self._render_library_complete():
            self.learn_button.setEnabled(True)
        self._refresh_workflow_cards()

    def _render_failed(self, error: str) -> None:
        self._workflow_activities.pop("render", None)
        self.render_button.setEnabled(True)
        self.render_pause_button.setEnabled(False)
        self.render_cancel_button.setEnabled(False)
        self._render_failure_detail = error
        self.render_stats.setText(error)
        self.append_log(error)
        self.statusBar().showMessage(error)
        self._refresh_workflow_cards()

    def _render_badge_clicked(self) -> None:
        """Offer a retry when the render card's status badge is showing failed.

        render_library() already skips notes it previously rendered, so
        re-running the exact same job resumes rather than starting over —
        this menu just makes that discoverable at the point of failure.
        """

        if not self._render_failure_detail or "render" in self._workflow_activities:
            return
        badge = self.sender()
        menu = QMenu(self)
        action = menu.addAction("Retry rendering")
        action.triggered.connect(self.start_render)
        if hasattr(badge, "mapToGlobal") and hasattr(badge, "rect"):
            menu.exec(badge.mapToGlobal(badge.rect().bottomLeft()))
        else:
            menu.exec()

    def start_analyze(self) -> None:
        if self.distribution_mode and not user_presets_enabled():
            self._explain_personal_presets_off()
            return
        if self.distribution_mode:
            pending = 0
            db_path = Path(self.local_paths["db"])
            if db_path.is_file():
                import sqlite3

                connection = sqlite3.connect(db_path)
                pending = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM presets WHERE status IN "
                        "('rendered','embedded') AND id NOT IN "
                        "(SELECT preset_id FROM fingerprints WHERE midi_note=0)"
                    ).fetchone()[0]
                )
                connection.close()
            if pending == 0:
                QMessageBox.information(
                    self,
                    "Personal learning is up to date",
                    "PatchLab already uses its shipped trained model and factory "
                    "fingerprints. Every rendered preset in your linked library "
                    "already has a fingerprint and is searchable.\n\n"
                    "Full parameter-model retraining is not incremental in this "
                    "release, so the installed app does not run a local-only "
                    "retrain that would replace prior learning.",
                )
                self.append_log(
                    "Analyze & Learn: shipped training retained; nothing rendered "
                    "is waiting to be fingerprinted."
                )
                self._refresh_workflow_cards()
                return
            self._set_workflow_activity(
                "analyze", 0, pending, f"Learning 0 of {pending:,} rendered presets…"
            )
            self.learn_button.setEnabled(False)
            self.analyze_cancel_button.setEnabled(True)
            self.learn_progress.setRange(0, pending)
            self.learn_progress.setValue(0)
            self.analyze_stats.setText(f"Learning 0 of {pending:,} rendered presets…")
            self.statusBar().showMessage("Fingerprinting rendered presets…")
            self.fingerprint_runner.start(fingerprint_only=True)
            return
        self._set_workflow_activity("analyze", 0, 0, "Starting analysis…")
        self.learn_button.setEnabled(False)
        self.analyze_cancel_button.setEnabled(True)
        self.learn_progress.setRange(0, 100)
        self.learn_progress.setValue(0)
        self.analyze_stats.setText("Starting target vectorization…")
        self.statusBar().showMessage("Analyzing and learning…")
        self.analyze_runner.start(self.deep_training.isChecked())

    def _fingerprint_completed(self, summary: dict) -> None:
        self._workflow_activities.pop("analyze", None)
        self.learn_button.setEnabled(True)
        self.analyze_cancel_button.setEnabled(False)
        created = int(summary.get("fingerprints_created", 0))
        text = f"Learned {created:,} rendered preset(s); now searchable."
        self.analyze_stats.setText(text)
        self.append_log(text)
        self.statusBar().showMessage(text)
        self._refresh_workflow_cards()

    def _fingerprint_failed(self, error: str) -> None:
        self._workflow_activities.pop("analyze", None)
        self.learn_button.setEnabled(True)
        self.analyze_cancel_button.setEnabled(False)
        self.analyze_stats.setText(error)
        self.append_log(error)
        self.statusBar().showMessage(error)
        self._refresh_workflow_cards()

    def _analyze_progress_changed(self, detail: dict) -> None:
        phase = str(detail.get("phase", "working"))
        if phase == "embeddings" and "completed_total" in detail:
            self.learn_progress.setMaximum(39_053)
            self.learn_progress.setValue(int(detail["completed_total"]))
            self.analyze_stats.setText(f"Embedding {detail['completed_total']:,}/39,053")
        elif phase == "synthetic-serum1" and "complete" in detail:
            self.learn_progress.setMaximum(20_000)
            self.learn_progress.setValue(int(detail["complete"]))
            self.analyze_stats.setText(f"Deep training patches {detail['complete']:,}/20,000")
        elif phase == "training" and "epoch" in detail:
            self.learn_progress.setMaximum(200)
            self.learn_progress.setValue(int(detail["epoch"]))
            self.analyze_stats.setText(
                f"Training epoch {detail['epoch']} — validation {detail['validation_loss']:.5f}"
            )
        else:
            self.learn_progress.setRange(0, 0)
            self.analyze_stats.setText(phase.replace("-", " ").title())
        current = int(
            detail.get(
                "completed_total",
                detail.get("complete", detail.get("epoch", 0)),
            )
        )
        total = (
            39_053
            if phase == "embeddings"
            else 20_000
            if phase == "synthetic-serum1"
            else 200
            if phase == "training"
            else 0
        )
        self._set_workflow_activity(
            "analyze",
            current,
            total,
            self.analyze_stats.text(),
        )

    def _analyze_completed(self, summary: dict) -> None:
        self._workflow_activities.pop("analyze", None)
        self.learn_progress.setRange(0, 100)
        self.learn_progress.setValue(100)
        self.learn_button.setEnabled(True)
        self.analyze_cancel_button.setEnabled(False)
        self.analyze_stats.setText("Analyze & Learn complete")
        self.append_log(f"Analyze & Learn complete: {summary}")
        self.statusBar().showMessage("Analyze & Learn complete")
        self._refresh_workflow_cards()

    def _analyze_failed(self, error: str) -> None:
        self._workflow_activities.pop("analyze", None)
        self.learn_progress.setRange(0, 100)
        self.learn_button.setEnabled(True)
        self.analyze_cancel_button.setEnabled(False)
        self.analyze_stats.setText(error)
        self.append_log(error)
        self.statusBar().showMessage(error)
        self._refresh_workflow_cards()

    def choose_match_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select a sound to match",
            str(Path.home()),
            "Audio files (*.wav *.mp3 *.flac *.ogg *.aif *.aiff)",
        )
        if selected:
            self._set_match_file(selected)

    def _set_match_file(self, selected: str) -> None:
        path = Path(selected).expanduser().resolve()
        if path.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES:
            QMessageBox.warning(
                self,
                "Unsupported audio",
                "Choose a WAV, MP3, FLAC, OGG, or AIFF file.",
            )
            return
        self._match_audio_path = path
        self._workflow_last_match_complete = False
        self._match_result = None
        self._match_result_path = None
        self.match_drop.setText(path.name)
        self.match_drop.set_playable(True)
        self.match_start_button.setEnabled(self._model_asset_error is None)
        self.match_stats.setText(
            self._model_asset_error
            or "Ready to match. Existing-preset results always search both synths."
        )
        # match_results now hosts the drop zone itself, so it stays visible;
        # only the per-result detail section is reset here.
        self.recommendation_details.setVisible(False)
        self.recommendation_placeholder.setVisible(True)
        self.save_preset_button.setEnabled(False)
        # A new sound is not the previous result: its save state must not linger.
        self._current_match_uid = None
        refresh_save_state = getattr(self, "_refresh_save_state", None)
        if refresh_save_state is not None:
            refresh_save_state()
        self._refresh_workflow_cards()

    def start_match(self) -> None:
        workflow_error = getattr(self, "_workflow_match_error", "")
        if self._model_asset_error or workflow_error:
            self.report_model_asset_error(
                self._model_asset_error or workflow_error, show_dialog=True
            )
            return
        if self._match_audio_path is None:
            return
        target_synth = str(self.match_synth.currentData())
        self._ui_event(
            "match_requested",
            "user asked to run a match",
            target_synth=target_synth,
            quality=str(self.match_budget.currentData()),
            audio_extension=self._match_audio_path.suffix.casefold(),
        )
        # CAPABILITY GATE (fresh, not cached startup state).
        #
        # Creating a Serum 2 preset needs Serum 2 installed. It does NOT need the
        # user's existing library to have been processed -- those are different
        # things, and conflating them would block Match for someone who simply
        # has pending presets. Only the synth requirement is checked here.
        #
        # It runs BEFORE anything else changes: a refused Match must not cancel
        # background work, disable the controls, or start a spinner. (It
        # previously ran after all three, which left a user who had just been
        # told to install Serum 2 staring at dead controls.)
        if not self._check_output_capability(target_synth, matching_now=True):
            return
        # Matching is always the foreground request. An automatic maintenance
        # scan is resumable, so stop it immediately rather than making a new
        # patch compete with a Serum host, CPU, or disk work from launch.
        if (
            getattr(self, "_automatic_link_scan_active", False)
            and self.runner.running
        ):
            self.append_log(
                "Pausing background preset check so Match can start immediately."
            )
            self.runner.cancel()
        self.match_start_button.setEnabled(False)
        self.match_button.setEnabled(False)
        self.match_cancel_button.setEnabled(True)
        self.match_synth.setEnabled(False)
        self.match_budget.setEnabled(False)
        self.match_offset.setEnabled(False)
        self.match_progress.setRange(0, 0)
        self._workflow_last_match_complete = False
        self._set_workflow_activity("match", 0, 0, "Loading audio and models…")
        self.match_stats.setText("Loading audio and models…")
        self.recommendation_details.setVisible(False)
        self.recommendation_placeholder.setVisible(True)
        budget = str(self.match_budget.currentData())
        # Analysis-by-synthesis produces a genuinely new patch; the factory
        # fingerprint path can only hand back the closest existing preset
        # unmodified. Prefer synthesis whenever its assets are present, and fall
        # back only when they are not — reporting why, so an install that
        # silently degrades to retrieval is visible instead of looking like a
        # quality problem.
        readiness = synthesis_readiness(target_synth)
        factory_only = self.distribution_mode and not readiness.available
        if factory_only:
            self.append_log(f"Synthesis unavailable — {readiness.reason}")
        self.append_log(
            f"Matching {self._match_audio_path.name} for {target_synth} ({budget}) "
            f"via {'factory fingerprints' if factory_only else 'analysis-by-synthesis'}"
        )
        self._ui_event(
            "match_started",
            "match worker launched",
            target_synth=target_synth,
            quality=budget,
            factory_only=bool(factory_only),
            card_phases=self._card_phases(),
        )
        self.match_runner.start(
            self._match_audio_path,
            target_synth=target_synth,
            budget=budget,
            offset=float(self.match_offset.value()),
            session_root=Path(self._match_session.name),
            factory_only=factory_only,
            factory_mapping=self.factory_mapping_path if factory_only else None,
            local_db=(
                self.local_paths["db"]
                if factory_only
                and bool(self.privacy_choice.use_and_share_own_presets)
                else None
            ),
            local_audio_root=(
                self.local_paths["audio"]
                if factory_only
                and bool(self.privacy_choice.use_and_share_own_presets)
                else None
            ),
        )

    def _match_progress_changed(self, detail: dict) -> None:
        self.match_stats.setStyleSheet("")
        phase = str(detail.get("phase", "working"))
        evaluations = int(detail.get("evaluations", 0))
        budget = int(detail.get("budget", 0))
        if budget:
            self.match_progress.setRange(0, budget)
            self.match_progress.setValue(min(evaluations, budget))
        else:
            self.match_progress.setRange(0, 0)
        best = detail.get("best_clap_cosine")
        if isinstance(best, (int, float)):
            self.match_stats.setText(
                f"{phase.replace('-', ' ').title()} — {evaluations}/{budget} evaluations, "
                f"best similarity {100.0 * float(best):.1f}%"
            )
        else:
            self.match_stats.setText(phase.replace("-", " ").title() + "…")
        self._set_workflow_activity(
            "match",
            evaluations,
            budget,
            (
                f"{phase.replace('-', ' ').title()} · "
                f"{evaluations:,} of {budget:,} evaluations"
                if budget
                else phase.replace("-", " ").title() + "…"
            ),
        )

    #: One plain sentence per playback failure kind. Never a PortAudio error
    #: code or an exception class -- those stay in diagnostics (see
    #: ``_report_playback_failure``).
    _PLAYBACK_MESSAGES = {
        "missing_file": "PatchLab can't find this audio file anymore.",
        "unsupported_audio": (
            "PatchLab can't play this audio file. It may be corrupted or in "
            "an unsupported format."
        ),
        "no_device": "PatchLab couldn't find an audio output device.",
        "device_error": (
            "PatchLab couldn't use your current audio output. Check your "
            "speakers or headphones and try again."
        ),
        "unknown": (
            "PatchLab couldn't play this audio. Check that your audio output "
            "device is connected, then try again."
        ),
    }

    @staticmethod
    def _reset_audio_devices() -> None:
        """Force PortAudio to re-enumerate host APIs and devices.

        The documented workaround (there is no public API for it) for a
        stream that fails after the OS default output device changed or
        disappeared mid-session -- exactly the shape of the tester's
        "paErrorCode -9986" report. Never touches the user's actual system
        audio settings; it only makes this process forget its stale view of
        them.
        """

        import sounddevice as sd

        for step in (sd._terminate, sd._initialize):
            try:
                step()
            except Exception:
                pass

    def _classify_playback_exception(self, exc: BaseException) -> str:
        """Ask PortAudio itself whether a device exists, rather than parsing text."""

        import sounddevice as sd

        if isinstance(exc, sd.PortAudioError):
            try:
                sd.query_devices(kind="output")
            except Exception:
                return "no_device"
            return "device_error"
        return "unknown"

    def _report_playback_failure(
        self, category: str, *, exc: BaseException | None = None, path: Path | None = None
    ) -> None:
        message = self._PLAYBACK_MESSAGES.get(category, self._PLAYBACK_MESSAGES["unknown"])
        self.append_log(f"Playback failed ({category}): {exc if exc is not None else message}")
        self.statusBar().showMessage(message)
        try:
            from core.diagnostics import recorder

            recorder().record(
                "playback", "playback_failed",
                f"audio playback failed: {category}",
                severity="warning",
                failure_category=category,
                path=str(path) if path is not None else "",
                exception_type=type(exc).__name__ if exc is not None else "",
            )
        except Exception:
            pass

    def _record_playback_recovered(self, path: Path) -> None:
        try:
            from core.diagnostics import recorder

            recorder().record(
                "playback", "playback_recovered",
                "audio device reset succeeded on retry", severity="info",
                path=str(path),
            )
        except Exception:
            pass

    def _play_audio(self, path: Path) -> bool:
        """Play one audio file. The one place every audition surface calls.

        Handles every failure kind this app's audition, preview, and Library
        playback share: a missing/unreadable file, unsupported or corrupt
        audio, no output device, or a PortAudio/device error -- for which
        exactly one bounded recovery attempt (reset PortAudio's device view,
        retry once) is made before giving up. Returns whether playback
        started; a caller may still show its own success message on True,
        but never needs to handle failure itself -- this already has.
        """

        import sounddevice as sd
        import soundfile as sf

        path = Path(path)
        if not path.is_file():
            self._report_playback_failure("missing_file", path=path)
            return False
        try:
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            self._report_playback_failure("unsupported_audio", exc=exc, path=path)
            return False

        last_exc: BaseException | None = None
        for attempt in (1, 2):
            try:
                sd.stop()
                sd.play(audio, rate, blocking=False)
                if attempt == 2:
                    self._record_playback_recovered(path)
                return True
            except Exception as exc:
                last_exc = exc
                if attempt == 1:
                    self._reset_audio_devices()
        category = self._classify_playback_exception(last_exc) if last_exc else "unknown"
        self._report_playback_failure(category, exc=last_exc, path=path)
        return False

    def _match_completed(self, result_path: str) -> None:
        self._ui_event(
            "match_completed",
            "match finished",
            operation_id=getattr(self.match_runner, "operation_id", ""),
        )
        self._workflow_activities.pop("match", None)
        self._workflow_last_match_complete = True
        import json

        self._match_result_path = Path(result_path)
        self._match_result = json.loads(
            self._match_result_path.read_text(encoding="utf-8")
        )
        self.match_progress.setRange(0, 100)
        self.match_progress.setValue(100)
        self.match_start_button.setEnabled(True)
        self.match_button.setEnabled(True)
        self.match_cancel_button.setEnabled(False)
        self.match_synth.setEnabled(True)
        self.match_budget.setEnabled(True)
        self.match_offset.setEnabled(True)
        self._show_match_result(self._match_result)
        self._refresh_workflow_cards()

    def _show_match_result(self, result: dict) -> None:
        self.match_results.setVisible(True)
        existing = list(result.get("existing_matches", []))
        self.existing_table.setRowCount(len(existing))
        for row_index, item in enumerate(existing):
            self.existing_table.setItem(
                row_index,
                0,
                QTableWidgetItem(
                    display_match_name(
                        item.get("name"),
                        row_index + 1,
                        source_path=item.get("source_path"),
                    )
                ),
            )
            self.existing_table.setItem(
                row_index,
                1,
                QTableWidgetItem("Serum 1" if item["synth"] == "serum1" else "Serum 2"),
            )
            self.existing_table.setItem(
                row_index,
                2,
                QTableWidgetItem(f"{float(item['similarity_percent']):.1f}%"),
            )
            self.existing_table.setItem(
                row_index, 3, QTableWidgetItem("PatchLab library")
            )
            play = QPushButton("▶")
            audition_path = item.get("audition_path")
            if audition_path:
                play.setToolTip(
                    f"Play MIDI note {item['audition_midi_note']} library render"
                )
                play.clicked.connect(
                    lambda _checked=False, path=Path(audition_path): self._play_audio(path)
                )
            elif item.get("preview_source_path"):
                play.setToolTip("Render and play this locally installed factory preset.")
                play.clicked.connect(
                    lambda _checked=False, detail=dict(item): self._render_preview(detail)
                )
            else:
                play.setEnabled(False)
                play.setToolTip("No rendered preview is shipped; link your own library for audition.")
            self.existing_table.setCellWidget(row_index, 4, play)

            # A closest match is an existing file PatchLab already found, not
            # a newly generated one: Save Copy makes an explicit copy
            # somewhere the user chooses, and Open Preset File Location
            # reveals the original in place -- never a copy made just so
            # Finder has something to show.
            reason = self._existing_match_export_blocker(item)
            save_copy = QPushButton("Save Copy")
            open_location = QPushButton("Open Preset File Location")
            for button in (save_copy, open_location):
                button.setObjectName("compactActionButton")
            if reason:
                for button in (save_copy, open_location):
                    button.setEnabled(False)
                    button.setToolTip(reason)
            else:
                save_copy.setToolTip("Save a copy of this preset to a folder you choose.")
                open_location.setToolTip("Reveal this preset's file in Finder.")
                save_copy.clicked.connect(
                    lambda _checked=False, index=row_index, detail=dict(item):
                    self.export_existing_match(index, detail)
                )
                open_location.clicked.connect(
                    lambda _checked=False, detail=dict(item):
                    self.open_existing_match_location(detail)
                )
            self.existing_table.setCellWidget(row_index, 5, save_copy)
            self.existing_table.setCellWidget(row_index, 6, open_location)

        recommendation = result.get("recommendation")
        self.settings_tree.clear()
        if not isinstance(recommendation, dict):
            self.recommendation_confidence.setText(str(result.get("message", "No confident match")))
            self.recommendation_confidence.setStyleSheet(
                "font-size: 17px; font-weight: 650; color: #b54708;"
            )
            self.winner_play_button.setEnabled(False)
            self.save_preset_button.setEnabled(False)
            self.match_stats.setText(str(result.get("message", "No confident match")))
            return

        similarity = float(recommendation["similarity_percent"])
        synth_name = "Serum 1" if recommendation["synth"] == "serum1" else "Serum 2"
        confidence = (
            f"{similarity:.1f}% CLAP similarity · {synth_name} · "
            "PatchLab generated preset · "
            f"{recommendation['evaluations']} evaluations in "
            f"{float(recommendation['elapsed_s']):.1f}s"
        )
        if result.get("no_confident_match"):
            confidence += "\nLow confidence — this sound may not be well-suited to Serum."
            self.recommendation_confidence.setStyleSheet(
                "font-size: 17px; font-weight: 650; color: #b54708;"
            )
        else:
            self.recommendation_confidence.setStyleSheet(
                "font-size: 17px; font-weight: 650; color: #087443;"
            )
        self.recommendation_confidence.setText(confidence)
        self.winner_play_button.setEnabled(
            bool(
                recommendation.get("winner_audio_path")
                or recommendation.get("preview_source_path")
            )
        )
        self.save_preset_button.setEnabled(
            bool(recommendation.get("export_available", True))
        )
        for section, values in recommendation.get("settings", {}).items():
            section_item = QTreeWidgetItem([str(section), ""])
            self.settings_tree.addTopLevelItem(section_item)
            for setting in values.get("changed", []):
                child = QTreeWidgetItem(
                    [str(setting["name"]), str(setting["value"])]
                )
                section_item.addChild(child)
            base_count = int(values.get("matches_base_count", 0))
            if base_count:
                base = QTreeWidgetItem(
                    [f"Matches base preset ({base_count} settings)", "collapsed"]
                )
                section_item.addChild(base)
            section_item.setExpanded(bool(values.get("changed")))
        self.match_stats.setText(str(result.get("message", "Match complete")))
        self.statusBar().showMessage("Match complete")

    @staticmethod
    def _match_failure_message(error: str) -> str:
        """One concise, plain-English line for a failed Match.

        A message that already starts "PatchLab " was written for the user (the
        worker sends those for known causes) and is shown as-is. Anything else is
        a raw exception -- a class name and often a filesystem path -- which is
        neither concise nor useful on screen, so it is summarised. The complete
        text is never lost: it is written to the application log and recorded in
        the flight recorder by the caller.
        """

        text = str(error).strip()
        if text.startswith("PatchLab "):
            return text.splitlines()[0][:200]
        lowered = text.casefold()
        if "filenotfound" in lowered or "no such file" in lowered:
            return (
                "PatchLab couldn't read that audio file. Check that it still "
                "exists, then try again."
            )
        if "decode" in lowered or "unsupported" in lowered or "soundfile" in lowered:
            return (
                "PatchLab couldn't read that audio file. Try a WAV, AIFF, FLAC or "
                "MP3 file."
            )
        return (
            "PatchLab couldn't finish this match. Your sound is still selected — "
            "you can try again."
        )

    def _match_failed(self, error: str) -> None:
        detail = str(error)
        error = self._match_failure_message(detail)
        self._workflow_activities.pop("match", None)
        self._workflow_last_match_complete = False
        self.match_progress.setRange(0, 100)
        self.match_progress.setValue(0)
        self.match_start_button.setEnabled(
            self._match_audio_path is not None and self._model_asset_error is None
        )
        self.match_button.setEnabled(self._model_asset_error is None)
        self.match_cancel_button.setEnabled(False)
        self.match_synth.setEnabled(True)
        self.match_budget.setEnabled(True)
        self.match_offset.setEnabled(True)
        self.match_stats.setText(error)
        self.match_stats.setStyleSheet("color: #ff5f67; font-weight: 700;")
        # The user sees the concise line; the full technical text stays here.
        self.append_log(f"Match failed: {detail}")
        self.statusBar().showMessage(error)
        if hasattr(self, "match_card_status"):
            self._refresh_workflow_cards()
        self._ui_event(
            "match_failed",
            "match failed; UI recovered",
            severity="warning",
            operation_id=getattr(self.match_runner, "operation_id", ""),
            error=detail,
            shown_to_user=error,
            controls_recovered=bool(
                self.match_synth.isEnabled() and not self.match_cancel_button.isEnabled()
            ),
            stale_activities=sorted(self._workflow_activities),
            source_audio_kept=self._match_audio_path is not None,
        )

    def report_model_asset_error(
        self,
        error: str,
        *,
        show_dialog: bool = False,
    ) -> None:
        """Keep model failures visible while unrelated jobs report progress."""

        self._model_asset_error = error
        self.match_start_button.setEnabled(False)
        self.match_button.setEnabled(False)
        self.match_stats.setText(error)
        self.match_stats.setStyleSheet("color: #ff5f67; font-weight: 700;")
        if hasattr(self, "match_card_status"):
            self._refresh_workflow_cards()
        self.append_log(f"Match unavailable: {error}")
        self.statusBar().showMessage("Matching unavailable — model files need attention")
        if show_dialog:
            QMessageBox.critical(self, "PatchLab model files are unavailable", error)

    def play_winner(
        self,
        _checked: bool = False,
        *,
        button: QPushButton | None = None,
    ) -> None:
        if not self._match_result:
            return
        recommendation = self._match_result.get("recommendation")
        if not isinstance(recommendation, dict):
            return
        note = (
            self._selected_preview_note()
            if hasattr(self, "_selected_preview_note")
            else 60
        )
        if self._match_result_path is None:
            return
        cache_key = recommendation_cache_key(
            self._match_result_path, recommendation
        )
        unmodified_source = (
            recommendation.get("preview_source_path")
            if not bool(recommendation.get("meaningfully_modified", False))
            else None
        )
        existing_audio: Path | None = None
        if not bool(recommendation.get("meaningfully_modified", False)):
            basis = next(
                (
                    item
                    for item in self._match_result.get("existing_matches", [])
                    if isinstance(item, dict)
                    and str(item.get("content_hash") or "") == cache_key
                ),
                None,
            )
            if basis and basis.get("audition_path"):
                existing_audio = (
                    Path(str(basis["audition_path"])).parent / f"{note}.wav"
                )
        self._resolve_octave_preview(
            cache_key=cache_key,
            note=note,
            synth=str(recommendation["synth"]),
            preview_source_path=unmodified_source,
            result_path=self._match_result_path,
            existing_audio_path=existing_audio,
            button=button,
        )

    def _preview_cache_root(self) -> Path:
        # Full library renders may live on a removable drive. Previews stay in
        # the small, capped app-data cache so matching and audition still work
        # while that drive is disconnected.
        return (
            Path(self.local_paths["preview_root"])
            if self.distribution_mode
            else DEFAULT_DB_PATH.parent
        )

    @staticmethod
    def _copy_preview_into_cache(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.stem}-{time.time_ns()}.tmp.wav"
        )
        shutil.copy2(source, temporary)
        temporary.replace(destination)

    def _resolve_octave_preview(
        self,
        *,
        cache_key: str,
        note: int,
        synth: str,
        preview_source_path: str | Path | None = None,
        result_path: Path | None = None,
        existing_audio_path: Path | None = None,
        button: QPushButton | None = None,
    ) -> None:
        """Single cache/resolve/render/play path for every octave control."""

        storage_key = preview_cache_identity(cache_key, synth)
        cached = preview_cache_path(self._preview_cache_root(), storage_key, note)
        if cached.is_file():
            touch_preview(cached)
            if self._play_audio(cached):
                self.statusBar().showMessage(
                    f"Playing cached C{1 + (note - 24) // 12} preview"
                )
            return

        # Migrate existing durable/local renders without asking Serum to render
        # the same (preset, octave) again.
        migration_sources: list[Path] = []
        if existing_audio_path is not None:
            migration_sources.append(Path(existing_audio_path))
        migration_sources.append(
            self._preview_cache_root()
            / "factory-previews"
            / cache_key
            / f"{note}.wav"
        )
        if result_path is not None:
            migration_sources.append(
                Path(result_path).resolve().parent / f"recommendation-{note}.wav"
            )
        for source in migration_sources:
            if source.is_file():
                self._copy_preview_into_cache(source, cached)
                if self._play_audio(cached):
                    self.statusBar().showMessage(
                        f"Playing cached C{1 + (note - 24) // 12} preview"
                    )
                return

        original_text = button.text() if button is not None else ""
        if button is not None:
            button.setText("Rendering…")
            button.setEnabled(False)
        target = (storage_key, note)
        existing_request = self._preview_inflight.get(target)
        if existing_request:
            self._preview_requests[existing_request].append(
                (button, original_text, note)
            )
            self.statusBar().showMessage(
                f"C{1 + (note - 24) // 12} preview is already rendering…"
            )
            return
        try:
            if preview_source_path:
                source = Path(preview_source_path).expanduser()
                if not source.is_absolute() and result_path is not None:
                    source = resolve_result_path(result_path, source)
                request_id = self.preview_runner.start(
                    source,
                    synth=synth,
                    midi_note=note,
                    content_hash=storage_key,
                    output_root=self._preview_cache_root(),
                )
            elif result_path is not None:
                request_id = self.preview_runner.start_recommendation(
                    result_path,
                    note,
                    output_root=self._preview_cache_root(),
                    cache_key=storage_key,
                )
            else:
                raise RuntimeError("No verified render source is available for this preview")
        except Exception as exc:
            if button is not None:
                button.setText(original_text)
                button.setEnabled(True)
            self._preview_failed(f"{type(exc).__name__}: {exc}")
            return
        self._preview_requests[request_id] = [(button, original_text, note)]
        self._preview_inflight[target] = request_id
        self._preview_request_targets[request_id] = target
        queued = self.preview_runner.pending_count
        suffix = f" ({queued} queued)" if queued else ""
        self.statusBar().showMessage(
            f"Rendering C{1 + (note - 24) // 12} preview{suffix}…"
        )

    def _render_preview(
        self, detail: dict, *, button: QPushButton | None = None
    ) -> None:
        self._resolve_octave_preview(
            cache_key=str(detail["content_hash"]),
            note=int(detail.get("audition_midi_note") or 60),
            synth=str(detail["synth"]),
            preview_source_path=detail.get("preview_source_path"),
            button=button,
        )

    def _preview_request_completed(self, request_id: str, path: str) -> None:
        requests = self._preview_requests.pop(
            request_id, [(None, "", 60)]
        )
        target = self._preview_request_targets.pop(request_id, None)
        if target is not None:
            self._preview_inflight.pop(target, None)
        silent = request_id in self._preview_silent_requests
        self._preview_silent_requests.discard(request_id)
        QTimer.singleShot(0, self._prune_preview_cache)
        note = requests[0][2]
        for button, original_text, _note in requests:
            if button is not None:
                button.setText(original_text)
                button.setEnabled(True)
        if silent and not any(button is not None for button, _text, _n in requests):
            # A pre-render nobody is waiting on: cache it and stay quiet.
            return
        self.statusBar().showMessage(
            f"C{1 + (note - 24) // 12} preview ready"
        )
        self._play_audio(Path(path))

    def _preview_request_failed(self, request_id: str, error: str) -> None:
        requests = self._preview_requests.pop(
            request_id, [(None, "", 60)]
        )
        target = self._preview_request_targets.pop(request_id, None)
        if target is not None:
            self._preview_inflight.pop(target, None)
        was_silent = request_id in self._preview_silent_requests
        self._preview_silent_requests.discard(request_id)
        for button, original_text, _note in requests:
            if button is not None:
                button.setText(original_text)
                button.setEnabled(True)
        if was_silent and not any(button is not None for button, _t, _n in requests):
            # A background pre-render failing must not hijack the status bar or
            # surface an error the user never asked for, especially mid-batch.
            self.append_log(f"Octave pre-render skipped: {error}")
            return
        self._preview_failed(error)

    def _preview_completed(self, path: str) -> None:
        self.statusBar().showMessage("Factory preview ready")
        self._play_audio(Path(path))

    def _preview_failed(self, error: str) -> None:
        self.append_log(f"Factory preview failed: {error}")
        self.statusBar().showMessage(error)

    # Xfer ships these system preset trees world-writable specifically so any
    # local user can save into them without administrator rights — Serum 1's
    # own "User" folder even contains a literal SaveYourPresetsHere.txt. These
    # are also the exact folders Serum's own preset browser scans, unlike an
    # arbitrary path under the user's home directory, so PatchLab output saved
    # here shows up inside Serum immediately rather than needing a manual
    # "Add Folder" step.
    MACOS_SERUM1_USER_PRESETS = Path(
        "/Library/Audio/Presets/Xfer Records/Serum Presets/Presets/User"
    )
    MACOS_SERUM2_USER_PRESETS = Path(
        "/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets/User"
    )

    def _ensure_patchlab_export_folders(self) -> None:
        """Create the configured preset-output folder(s) up front.

        A brand-new machine has never run an export, so waiting for that
        would leave the destination invisible in Serum's browser until the
        first save. Runs at construction time, before self.log_pane exists —
        a failure here must never crash or block startup, so it is tolerated
        completely silently; `_patchlab_export_folder` retries the same
        `mkdir` at export time and can report there if it still fails.
        """

        from core.preset_output import load_preset_output_preferences

        if load_preset_output_preferences().folder:
            # One configured folder serves both generations; creating it
            # twice via _patchlab_export_folder("serum1"/"serum2") is harmless.
            try:
                self._patchlab_export_folder("serum2")
            except OSError:
                pass
            return
        if ENV.branch != "macos":
            return
        for root in (self.MACOS_SERUM1_USER_PRESETS, self.MACOS_SERUM2_USER_PRESETS):
            try:
                (root / "PatchLab").mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    def _default_export_folder(self, synth: str) -> Path:
        """Return the Serum preset root generated presets belong under when
        Settings -> Generated Preset Folder has never been changed.

        See core.preset_output.default_preset_output_root for the shared
        implementation (also used by the Settings display).
        """

        from core.preset_output import default_preset_output_root

        return default_preset_output_root(synth, env=ENV)

    def _reveal_in_finder(self, path: Path) -> bool:
        """Select an exact, already-existing file in Finder. Never a copy,
        never an export, never Serum or DAW automation -- the standard macOS
        "Reveal in Finder" mechanism (equivalent to ``open -R``).

        Returns whether the file was found and revealed.
        """

        path = Path(path)
        if not path.is_file():
            self.append_log(f"Could not reveal preset: {path} no longer exists")
            self.statusBar().showMessage("PatchLab can't find this preset file anymore.")
            return False
        try:
            subprocess.run(["open", "-R", str(path)], check=False)
        except Exception as exc:
            self.append_log(f"Could not open Finder for {path}: {exc}")
            self.statusBar().showMessage("PatchLab couldn't open Finder for this preset.")
            return False
        self.statusBar().showMessage(f"Revealed in Finder: {path.name}")
        return True

    def _existing_match_export_blocker(self, item: dict) -> str:
        """Why this closest match cannot be exported, or "" when it can.

        A closest match is an existing preset file, so exporting it means copying
        the installed file. When PatchLab does not have that file, say exactly
        that instead of offering an action that cannot work.
        """

        if not item.get("local_source_available") or not item.get("source_path"):
            name = str(item.get("name") or "This preset")
            return (
                f"{name} is not installed on this Mac, so PatchLab has no file to "
                "copy. Install the pack it came from and run the match again."
            )
        synth = str(item.get("synth") or "")
        if synth not in ("serum1", "serum2"):
            return "PatchLab does not recognise this preset's Serum generation."
        try:
            from core.synth_capability import capability_for

            capability = capability_for(synth)
        except Exception:
            return ""
        if not capability.available:
            return capability.user_message()
        return ""

    def _existing_match_output(self, item: dict, *, folder: Path) -> Path:
        synth = str(item["synth"])
        extension = ".fxp" if synth == "serum1" else ".SerumPreset"
        base = generated_preset_name(synth)
        output = folder / f"{base}{extension}"
        counter = 2
        while output.exists():
            output = folder / f"{base} {counter}{extension}"
            counter += 1
        return output

    def export_existing_match(self, index: int, item: dict) -> None:
        """Save a COPY of one closest match to a folder the user chooses.

        A closest match is an existing preset PatchLab already found on this
        Mac, not a newly generated one -- so unlike the primary result, there
        is no auto-save and no Retry Saving Preset for it. This exists for
        the case a copy genuinely helps (organizing a find into the user's
        own collection); when the user just wants to see the file, Open
        Preset File Location reveals the original directly, with no copy.
        """

        if self._match_result_path is None:
            return
        reason = self._existing_match_export_blocker(item)
        if reason:
            QMessageBox.information(self, "This preset cannot be copied", reason)
            return
        suggested = self._existing_match_output(
            item, folder=self._patchlab_export_folder(str(item["synth"]))
        )
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save a copy of this closest match",
            str(suggested),
            "Serum preset (*.fxp *.SerumPreset)",
        )
        if not selected:
            return
        output = Path(selected)
        if output.suffix.casefold() != suggested.suffix.casefold():
            output = output.with_suffix(suggested.suffix)
        self._start_existing_match_export(index, output, label="Save Copy")

    def open_existing_match_location(self, item: dict) -> None:
        """Reveal a closest match's own existing file -- never a copy.

        Every closest match PatchLab can offer this for already has a real
        local file (the same precondition Save Copy requires), so this
        always reveals that original file directly rather than writing
        anything new just so Finder has something to show.
        """

        reason = self._existing_match_export_blocker(item)
        if reason:
            QMessageBox.information(self, "This preset cannot be located", reason)
            return
        self._reveal_in_finder(Path(str(item["source_path"])))

    def _start_existing_match_export(
        self, index: int, output: Path, *, label: str
    ) -> None:
        if self.export_runner.running:
            QMessageBox.information(
                self, "Export in progress", "Wait for the current export to finish."
            )
            return
        self._export_context_uid = None
        self.append_log(f"{label}: saving closest match #{index + 1} to {output}")
        self.statusBar().showMessage(f"{label}: writing {output.name}…")
        try:
            self.export_runner.start(
                self._match_result_path, output, existing_match=index
            )
        except Exception as exc:
            self.append_log(f"{label} could not start: {exc}")

    def _patchlab_export_folder(self, synth: str) -> Path:
        """Return (and create) where a generated preset for ``synth`` saves.

        The one shared resolver (core.preset_output) every generated-preset
        writer uses -- single Match, batch Match, Retry Saving Preset (via
        the incident's already-fixed target_path), Run Again, and the
        Settings display all agree on the same answer. With nothing
        configured this is still Serum's own per-synth "PatchLab" subfolder,
        exactly as before Settings -> Generated Preset Folder existed.
        """

        from core.preset_output import configured_preset_output_folder

        folder = configured_preset_output_folder(synth)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            # A read-only or missing parent must not block the export dialog;
            # fall back to the (unconfigured) default so a save can still
            # find somewhere to go.
            return self._default_export_folder(synth)
        return folder

    def _rename_saved_preset(self, match_uid: str, current_path: Path) -> None:
        """Prompt for a new name and rename an already-saved preset in place.

        A plain filesystem rename in the same folder — never a new export —
        so there is never a second, duplicate file left behind; the
        auto-named one simply becomes the new name.
        """

        if not current_path.is_file():
            QMessageBox.warning(
                self,
                "Preset not found",
                f"The saved preset is missing from disk:\n{current_path}",
            )
            return
        new_stem, accepted = QInputDialog.getText(
            self, "Rename preset", "Preset name:", text=current_path.stem
        )
        if not accepted:
            return
        sanitized = sanitize_folder_name(new_stem)
        if not sanitized:
            QMessageBox.warning(self, "Invalid name", "Enter a non-empty, filesystem-safe name.")
            return
        if sanitized == current_path.stem:
            return
        new_path = disambiguated_preset_path(
            current_path.parent, sanitized, current_path.suffix
        )
        try:
            current_path.rename(new_path)
        except OSError as exc:
            QMessageBox.warning(self, "Rename failed", f"Could not rename the preset:\n{exc}")
            return
        Database(self._match_database_path()).set_match_exported_path(match_uid, new_path)
        self.refresh_match_library()
        self.append_log(f"Renamed preset to: {new_path.name}")
        self.statusBar().showMessage(f"Preset renamed: {new_path.name}")

    def save_match_preset(self) -> None:
        """Rename the currently displayed match's already-saved preset.

        Every generated patch is now auto-saved the moment a match completes,
        so there is nothing left to export here — the button that used to
        write a brand-new file now just renames the one that already exists.
        """

        if self._batch_state is not None:
            QMessageBox.information(
                self,
                "Batch is running",
                "Renaming is reserved for the active batch. Try again after it finishes.",
            )
            return
        if self._current_match_uid is None:
            return
        record = Database(self._match_database_path()).get_match_library(
            self._current_match_uid
        )
        if record is None or record.exported_preset_path is None:
            if self.export_runner.running:
                QMessageBox.information(
                    self,
                    "Still saving",
                    "This preset is still being saved. Try renaming it again in a moment.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Nothing to rename",
                    "This match has no saved preset yet — no confident recommendation was generated.",
                )
            return
        self._rename_saved_preset(self._current_match_uid, Path(record.exported_preset_path))

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.runner.cancel()
        self.render_runner.cancel()
        self.analyze_runner.cancel()
        self.match_runner.cancel()
        self.export_runner.cancel()
        self.preview_runner.cancel()
        for runner in (self.match_runner, self.preview_runner):
            if (
                runner.process.state()
                != QProcess.ProcessState.NotRunning
                and not runner.process.waitForFinished(1500)
            ):
                runner.process.kill()
                runner.process.waitForFinished(1500)
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        try:
            self._match_session.cleanup()
        except Exception as exc:
            self.append_log(f"Temporary match audio cleanup deferred: {exc}")
        super().closeEvent(event)


class MainWindow(LegacyMainWindow):
    """Visual redesign that preserves the proven Milestone 0–6 behavior."""

    ASPECT_RATIO = 16.0 / 9.0
    DESIGN_WIDTH = 1920
    DESIGN_HEIGHT = 1080

    def __init__(
        self,
        *,
        factory_verification: FactoryVerification | None = None,
        privacy_store: PrivacyStore | None = None,
    ) -> None:
        self._aspect_guard = False
        self._last_resize_size = None
        self._native_aspect_installed = False
        self._settings_building = False
        self._settings_mode = "default"
        self._settings_sections: dict[str, dict] = {}
        QMainWindow.__init__(self)
        self.distribution_mode = distribution_mode()
        self.factory_verification = factory_verification
        self.privacy_store = privacy_store or PrivacyStore()
        set_active_store(self.privacy_store)
        self.privacy_choice = self.privacy_store.load()
        self.factory_mapping_path = (
            ENV.app_data_dir / "factory-paths.json"
            if self.distribution_mode
            else Path(__file__).resolve().parents[1]
            / "data"
            / "local"
            / "factory_paths.json"
        )
        self.storage_preferences = load_storage_preferences()
        self.local_paths = default_local_paths()
        self._ensure_patchlab_export_folders()
        self.runner = ScanProcessRunner(self)
        self.runner.log.connect(self._local_library_log)
        self.runner.progress.connect(self._progress)
        self.runner.stage_progress.connect(self._local_library_progress_changed)
        self.runner.completed.connect(self._scan_completed)
        self.runner.failed.connect(self._scan_failed)
        self.fingerprint_runner = ScanProcessRunner(self)
        self.fingerprint_runner.log.connect(self.append_log)
        self.fingerprint_runner.stage_progress.connect(self._local_library_progress_changed)
        self.fingerprint_runner.completed.connect(self._fingerprint_completed)
        self.fingerprint_runner.failed.connect(self._fingerprint_failed)
        self.update_check_runner = UpdateCheckProcessRunner(self)
        self.update_check_runner.log.connect(self.append_log)
        self.update_check_runner.completed.connect(self._update_check_completed)
        self.update_check_runner.failed.connect(self._update_check_failed)
        self._update_check_manual = False
        self._update_check_auth_retry_attempted = False
        self.update_download_runner = UpdateDownloadProcessRunner(self)
        self.update_download_runner.log.connect(self.append_log)
        self.update_download_runner.progress.connect(self._update_download_progress)
        self.update_download_runner.completed.connect(self._update_download_completed)
        self.update_download_runner.failed.connect(self._update_download_failed)
        self.factory_verify_runner = FactoryVerificationProcessRunner(self)
        self.factory_verify_runner.log.connect(self.append_log)
        self.factory_verify_runner.completed.connect(self._factory_verify_completed)
        self.factory_verify_runner.failed.connect(self._factory_verify_failed)
        self.bug_report_runner = BugReportProcessRunner(self)
        self.bug_report_runner.log.connect(self.append_log)
        self.bug_report_runner.completed.connect(self._bug_report_completed)
        self.bug_report_runner.failed.connect(self._bug_report_failed)
        # Automatic save-incident reports use their own runner so they can
        # never pop the manual report's sign-in or failure dialogs.
        self.incident_report_runner = BugReportProcessRunner(self)
        self.incident_report_runner.log.connect(self.append_log)
        self.incident_report_runner.completed.connect(self._incident_report_completed)
        self.incident_report_runner.failed.connect(self._incident_report_failed)
        self._incident_report_queue: list[tuple[str, str, Path]] = []
        self._incident_report_active: tuple[str, str, Path] | None = None
        self._reported_save_signatures: dict[tuple[str, ...], str] = {}
        #: The save incident the export worker is running for, if any.
        self._save_incident_context: dict | None = None
        self.render_runner = RenderProcessRunner(self)
        self.render_runner.log.connect(self.append_log)
        self.render_runner.progress.connect(self._render_progress_changed)
        self.render_runner.completed.connect(self._render_completed)
        self.render_runner.failed.connect(self._render_failed)
        self.render_runner.control_changed.connect(self._render_control_changed)
        self.analyze_runner = AnalyzeProcessRunner(self)
        self.analyze_runner.log.connect(self.append_log)
        self.analyze_runner.progress.connect(self._analyze_progress_changed)
        self.analyze_runner.completed.connect(self._analyze_completed)
        self.analyze_runner.failed.connect(self._analyze_failed)
        self.match_runner = MatchProcessRunner(self)
        self.match_runner.log.connect(self.append_log)
        self.match_runner.progress.connect(self._match_progress_changed)
        self.match_runner.completed.connect(self._match_completed)
        self.match_runner.failed.connect(self._match_failed)
        self.export_runner = ExportProcessRunner(self)
        self.export_runner.log.connect(self.append_log)
        self.export_runner.completed.connect(self._export_completed)
        self.export_runner.failed.connect(self._export_failed)
        self.preview_runner = PreviewProcessRunner(self)
        self.preview_runner.log.connect(self.append_log)
        self.preview_runner.request_completed.connect(self._preview_request_completed)
        self.preview_runner.request_failed.connect(self._preview_request_failed)
        self.storage_runner = StorageProcessRunner(self)
        self.storage_runner.log.connect(self.append_log)
        self.storage_runner.progress.connect(self._storage_progress_changed)
        self.storage_runner.completed.connect(self._storage_completed)
        self.storage_runner.failed.connect(self._storage_failed)
        self._storage_pending: str | None = None
        self._render_paused = False
        self._render_failure_detail: str = ""
        # True while Render Sound Library is driving the shared scan runner,
        # so its progress and failures land on the Render card rather than Link.
        self._compact_render_active = False
        self._prepare_active = False
        self._prepare_total = 0
        self._prepare_completed = 0
        self._prepare_started_at = 0.0
        # Last capability snapshot, so a later check can tell that a synth
        # newly became available rather than merely being present.
        self._capability_snapshot = None
        self._model_asset_error: str | None = None
        self._match_audio_path: Path | None = None
        self._match_result_path: Path | None = None
        self._match_result: dict | None = None
        self._existing_matches: list[dict] = []
        self._existing_page = 0
        self._favorite_hashes: set[str] = set()
        self._current_match_uid: str | None = None
        self._preview_requests: dict[
            str, list[tuple[QPushButton | None, str, int]]
        ] = {}
        self._preview_inflight: dict[tuple[str, int], str] = {}
        self._preview_request_targets: dict[str, tuple[str, int]] = {}
        self._preview_silent_requests: set[str] = set()
        self._export_context_uid: str | None = None
        self._batch_state: dict | None = None
        self._workflow_activities: dict[str, WorkflowActivity] = {}
        self._workflow_last_match_complete = False
        self._automatic_link_scan_active = False
        self._automatic_log_lines_suppressed = 0
        self._match_session = tempfile.TemporaryDirectory(
            prefix="patchlab-match-app-"
        )

        self.factory_status = QLabel("")
        self.factory_status.setWordWrap(True)
        self.factory_status.setVisible(self.distribution_mode)
        self.privacy_settings = QGroupBox("Privacy")
        privacy_layout = QVBoxLayout(self.privacy_settings)
        self.share_toggle = QCheckBox("Use && share my own presets")
        self.share_toggle.setToolTip(
            "When on, linked presets are processed locally and preset files plus "
            "fingerprints may be contributed. Rendered audio is never uploaded."
        )
        self.share_toggle.setChecked(
            bool(self.privacy_choice.use_and_share_own_presets)
        )
        self.share_toggle.toggled.connect(self._privacy_toggled)
        privacy_layout.addWidget(self.share_toggle)

        self.render_pause_button = QPushButton("Pause")
        self.render_pause_button.setEnabled(False)
        self.render_pause_button.clicked.connect(self.toggle_render_pause)
        self.render_pause_button.setVisible(False)
        self.render_cancel_button = QPushButton("Cancel")
        self.render_cancel_button.setEnabled(False)
        self.render_cancel_button.clicked.connect(self.cancel_render)
        self.render_stats = QLabel("Ready")
        self.deep_training = QCheckBox("Deep training")
        self.deep_training.setChecked(True)
        self.analyze_cancel_button = QPushButton("Cancel")
        self.analyze_cancel_button.setEnabled(False)
        self.analyze_cancel_button.clicked.connect(self.analyze_runner.cancel)
        self.analyze_cancel_button.clicked.connect(self.fingerprint_runner.cancel)
        self.analyze_stats = QLabel("Ready")

        self.match_offset = QDoubleSpinBox()
        self.match_offset.setRange(0.0, 86_400.0)
        self.match_offset.setDecimals(2)
        self.match_offset.setSuffix(" s offset")
        self.match_offset.setToolTip(
            "For files over ten seconds, choose where analysis begins."
        )
        self.match_budget = SegmentedControl()
        self.match_budget.addItem("Quick", "quick")
        self.match_budget.addItem("Balanced", "balanced")
        self.match_budget.addItem("Best Quality", "best")
        self.match_budget.setCurrentIndex(1)
        self.match_synth = SegmentedControl()
        self.match_synth.addItem("Serum 2", "serum2")
        self.match_synth.addItem("Serum 1", "serum1")
        self.match_start_button = QPushButton("Run Match")
        self.match_start_button.setEnabled(False)
        self.match_start_button.clicked.connect(self.start_match)
        self.match_cancel_button = QPushButton("Cancel")
        self.match_cancel_button.setEnabled(False)
        self.match_cancel_button.clicked.connect(self.match_runner.cancel)
        self.match_stats = QLabel("Choose WAV, MP3, FLAC, OGG, or AIFF.")
        self.match_stats.setWordWrap(True)

        self.setWindowTitle("PatchLab")
        self.setMinimumSize(1280, 720)
        self.setStyleSheet(theme.load_stylesheet())
        self._rebuild_visual_tree()
        self.resize(1440, 810)
        self.append_log(
            f"Interface ready · {ENV.branch} · compute {ENV.compute_backend} · "
            f"build {current_build_info().short_commit}"
        )
        self.statusBar().showMessage(
            f"Ready — {ENV.branch}, compute: {ENV.compute_backend}"
        )
        if self.distribution_mode:
            self._apply_factory_status()
            self._apply_privacy_choice()
            if self.privacy_choice.use_and_share_own_presets is None:
                QTimer.singleShot(0, self._show_consent_dialog)
        QTimer.singleShot(0, self._reconcile_save_incidents)

    def _rebuild_visual_tree(self) -> None:
        old_central = self.centralWidget()
        root = QWidget()
        root.setObjectName("appRoot")
        # root has no Qt-widget parent (QGraphicsScene.addWidget requires a
        # top-level widget), so it no longer inherits self's stylesheet via
        # normal parent-child cascading — apply it directly here too.
        root.setStyleSheet(theme.load_stylesheet())
        root.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 0, 12, 7)
        root_layout.setSpacing(16)
        root_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.root_layout = root_layout

        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(13, 6, 11, 6)
        top_layout.setSpacing(7)
        top_layout.addWidget(WaveformMark())
        wordmark = QLabel("PatchLab")
        wordmark.setObjectName("wordmark")
        version = QLabel(f"v{__version__}")
        version.setObjectName("versionTag")
        top_layout.addWidget(wordmark)
        top_layout.addWidget(version)
        top_layout.addStretch(1)
        self.settings_button = QPushButton("Settings")
        self.settings_button.setObjectName("navButton")
        self.settings_button.setIcon(icon("settings"))
        self.settings_button.clicked.connect(self.open_settings)
        self.help_button = QPushButton("Help")
        self.help_button.setObjectName("navButton")
        self.help_button.setIcon(icon("help"))
        self.help_button.clicked.connect(self.open_help)
        top_layout.addWidget(self.settings_button)
        top_layout.addWidget(self.help_button)
        top_bar.setMaximumHeight(48)
        self.top_bar = top_bar
        root_layout.addWidget(top_bar)

        self.factory_status.setParent(root)
        root_layout.addWidget(self.factory_status)
        self.privacy_settings.setParent(root)
        self.privacy_settings.setVisible(False)

        cards = (
            HeroCard(
                "Link My Preset Folder"
                if self.distribution_mode
                else "Select Preset Folder",
                "folder",
                "teal",
                enabled=True,
                step=1,
            ),
            HeroCard(
                "Prepare Preset Library",
                "waveform",
                "violet",
                enabled=True,
                step=2,
            ),
            HeroCard(
                "Match a Sound",
                "search-wave",
                "blue",
                enabled=True,
                step=3,
            ),
        )
        hero_layout = QHBoxLayout()
        hero_layout.setSpacing(11)
        for card in cards:
            hero_layout.addWidget(card, 1)
        root_layout.addLayout(hero_layout)
        scan_card, render_card, match_card = cards
        self.hero_cards = cards
        self.scan_button, self.scan_progress = scan_card.button, scan_card.progress
        self.render_button, self.render_progress = (
            render_card.button,
            render_card.progress,
        )
        # Deep training remains available in its settings panel, but personal
        # preset preparation is one operation and no longer has an Analyze card.
        self.learn_button, self.learn_progress = QPushButton(), QProgressBar()
        self.learn_button.setVisible(False)
        self.match_button, self.match_progress = match_card.button, match_card.progress
        self.scan_box = scan_card
        self.scan_card_status = scan_card.status
        self.render_card_status = render_card.status
        self.match_card_status = match_card.status
        self.scan_button.clicked.connect(self.choose_folder)
        self.render_button.clicked.connect(self.start_render)
        self.match_button.clicked.connect(self.choose_match_file)
        if render_card.step_badge is not None:
            render_card.step_badge.clicked.connect(self._render_badge_clicked)
        self._refresh_workflow_cards()

        control_row = QHBoxLayout()
        control_row.setSpacing(11)

        # One geometry for every pill in this row. QComboBox and QPushButton
        # derive different sizeHints from the same stylesheet padding (29 vs
        # 35 px), which previously left the Offset/Actions controls taller and
        # sitting closer to their labels than the Quality/Target Synth combos.
        # Pinning the height makes all three cards line up on one grid.
        pill_height = 31
        card_margins = (10, 7, 10, 8)
        card_spacing = 5
        group_spacing = 4

        config_card = QFrame()
        config_card.setObjectName("controlCard")
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(*card_margins)
        config_layout.setSpacing(card_spacing)
        config_title = QLabel("MATCH CONFIGURATION")
        config_title.setObjectName("controlTitle")
        self.match_stats.setObjectName("controlStat")
        self.match_stats.setWordWrap(False)
        self.match_stats.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(config_title)
        title_row.addStretch(1)
        self.batch_recursive = QCheckBox("Include subfolders")
        self.batch_recursive.setObjectName("batchRecursive")
        self.batch_button = QPushButton("Batch Folder…")
        self.batch_button.setObjectName("compactActionButton")
        self.batch_button.clicked.connect(self.start_batch_folder)
        title_row.addWidget(self.batch_recursive)
        title_row.addWidget(self.match_stats)
        config_layout.addLayout(title_row)
        options = QHBoxLayout()
        options.setSpacing(0)

        def _divider() -> QFrame:
            line = QFrame()
            line.setObjectName("groupDivider")
            line.setFrameShape(QFrame.Shape.VLine)
            return line

        quality_label = QLabel("QUALITY")
        quality_label.setObjectName("microLabel")
        synth_label = QLabel("TARGET SYNTH")
        synth_label.setObjectName("microLabel")
        offset_label = QLabel("OFFSET")
        offset_label.setObjectName("microLabel")
        actions_label = QLabel("ACTIONS")
        actions_label.setObjectName("microLabel")
        self.match_budget.setItemText(0, "Quick")
        self.match_budget.setItemText(1, "Balanced")
        self.match_budget.setItemText(2, "Best Quality")
        self.match_synth.setItemText(0, "Serum 2")
        self.match_synth.setItemText(1, "Serum 1")
        self.match_start_button.setObjectName("primaryButton")
        # Height is pinned via the pillRow property rather than
        # setFixedHeight: the stylesheet is applied to the tree after this
        # runs, and QSS min-height raises the widget minimum again during
        # polish, which would silently undo a Python-set height.
        for pill in (
            self.match_budget,
            self.match_synth,
            self.match_offset,
            self.match_start_button,
            self.match_cancel_button,
            self.render_pause_button,
            self.render_cancel_button,
            self.analyze_cancel_button,
        ):
            pill.setProperty("pillRow", True)
            pill.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
        quality_group = QVBoxLayout()
        quality_group.setSpacing(group_spacing)
        quality_group.addWidget(quality_label)
        quality_group.addWidget(self.match_budget)
        synth_group = QVBoxLayout()
        synth_group.setSpacing(group_spacing)
        synth_group.addWidget(synth_label)
        synth_group.addWidget(self.match_synth)
        offset_group = QVBoxLayout()
        offset_group.setSpacing(group_spacing)
        offset_group.addWidget(offset_label)
        self.match_offset.setFixedWidth(140)
        offset_group.addWidget(self.match_offset)
        actions_group = QVBoxLayout()
        actions_group.setSpacing(group_spacing)
        actions_group.addWidget(actions_label)
        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 7, 0, 0)
        actions_row.setSpacing(8)
        self.match_start_button.setMinimumWidth(100)
        self.match_cancel_button.setMinimumWidth(72)
        actions_row.addWidget(self.match_start_button)
        actions_row.addWidget(self.match_cancel_button)
        self.batch_button.setText("Batch…")
        self.batch_button.setMinimumWidth(72)
        self.batch_button.setProperty("pillRow", True)
        actions_row.addWidget(self.batch_button)
        actions_group.addLayout(actions_row)
        options.addLayout(quality_group, 4)
        options.addSpacing(10)
        options.addWidget(_divider())
        options.addSpacing(10)
        options.addLayout(synth_group, 2)
        options.addSpacing(10)
        options.addWidget(_divider())
        options.addSpacing(10)
        options.addLayout(offset_group, 1)
        options.addSpacing(10)
        options.addWidget(_divider())
        options.addSpacing(10)
        options.addLayout(actions_group, 3)
        config_layout.addLayout(options)

        # The library and training cards mirror the config card's rhythm —
        # title, micro label, then the pill row — so every pill across the
        # three cards lands on the same baseline.
        library_card = QFrame()
        library_card.setObjectName("controlCard")
        library_layout = QVBoxLayout(library_card)
        library_layout.setContentsMargins(*card_margins)
        library_layout.setSpacing(card_spacing)
        library_title = QLabel("LIBRARY BACKGROUND JOBS")
        library_title.setObjectName("controlTitle")
        library_layout.addWidget(library_title)
        library_group = QVBoxLayout()
        library_group.setSpacing(group_spacing)
        library_actions_label = QLabel("ACTIONS")
        library_actions_label.setObjectName("microLabel")
        library_group.addWidget(library_actions_label)
        library_controls = QHBoxLayout()
        library_controls.setSpacing(8)
        self.render_stats.setObjectName("controlStat")
        library_controls.addWidget(self.render_pause_button)
        library_controls.addWidget(self.render_cancel_button)
        library_controls.addWidget(self.render_stats, 1)
        library_group.addLayout(library_controls)
        library_layout.addLayout(library_group)

        training_card = QFrame()
        training_card.setObjectName("controlCard")
        training_layout = QVBoxLayout(training_card)
        training_layout.setContentsMargins(*card_margins)
        training_layout.setSpacing(card_spacing)
        training_title = QLabel(
            "PERSONAL LEARNING" if self.distribution_mode else "DEEP TRAINING"
        )
        training_title.setObjectName("controlTitle")
        training_layout.addWidget(training_title)
        training_group = QVBoxLayout()
        training_group.setSpacing(group_spacing)
        training_options_label = QLabel("OPTIONS")
        training_options_label.setObjectName("microLabel")
        training_group.addWidget(training_options_label)
        training_controls = QHBoxLayout()
        training_controls.setSpacing(18)
        if self.distribution_mode:
            self.deep_training.setText("Automatic with linked folder")
            self.deep_training.setChecked(True)
            self.deep_training.setEnabled(False)
            self.analyze_cancel_button.setVisible(False)
            self.analyze_stats.setText("Shipped model active")
        else:
            self.deep_training.setText("Deep training")
        self.analyze_stats.setObjectName("controlStat")
        training_controls.addWidget(
            self.deep_training,
            0,
            Qt.AlignmentFlag.AlignVCenter,
        )
        training_controls.addWidget(self.analyze_cancel_button)
        training_controls.addWidget(self.analyze_stats, 1)
        training_group.addLayout(training_controls)
        training_layout.addLayout(training_group)

        control_row.addWidget(config_card, 50)
        control_row.addWidget(library_card, 25)
        control_row.addWidget(training_card, 25)
        self.control_cards = (config_card, library_card, training_card)
        for card in self.control_cards:
            card.setMaximumHeight(92)
        # Positioned right after the hero row (index 3: top_bar, factory_status,
        # hero_layout precede it) so it reads as a second row of cards matching
        # the hero row's spacing, per the requested layout.
        root_layout.insertLayout(3, control_row)
        self.match_panel = config_card

        self.match_results = QWidget()
        results_layout = QHBoxLayout(self.match_results)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(9)
        closest = QGroupBox("✦  Closest Matches")
        closest.setObjectName("glassPanel")
        closest_layout = QVBoxLayout(closest)
        closest_layout.setContentsMargins(9, 2, 9, 7)
        closest_layout.setSpacing(0)
        closest_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.closest_panel = closest
        self.existing_heading = QLabel("Closest Matches")
        self.existing_heading.setVisible(False)
        column_header = QHBoxLayout()
        column_header.setContentsMargins(10, 2, 10, 2)
        column_header.setSpacing(8)
        header_rank = QLabel("#")
        header_rank.setObjectName("muted")
        header_rank.setFixedWidth(20)
        header_rank.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_name = self._muted_label("Preset")
        header_similarity = self._muted_label("Similarity")
        header_similarity.setFixedWidth(160)
        header_percent = self._muted_label("%")
        header_percent.setFixedWidth(52)
        header_percent.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_fav = self._muted_label("")
        header_fav.setFixedWidth(24)
        column_header.addWidget(header_rank)
        column_header.addWidget(header_name, 1)
        column_header.addWidget(header_similarity)
        column_header.addWidget(header_percent)
        column_header.addWidget(header_fav)
        closest_layout.addLayout(column_header)
        self.closest_placeholder = self._muted_label(
            "Run a match to see your closest owned presets here."
        )
        self.closest_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        closest_layout.addWidget(self.closest_placeholder)
        self.existing_list_container = QWidget()
        self.existing_list_layout = QVBoxLayout(self.existing_list_container)
        self.existing_list_layout.setContentsMargins(0, 4, 0, 0)
        self.existing_list_layout.setSpacing(4)
        closest_layout.addWidget(
            self.existing_list_container,
            0,
            Qt.AlignmentFlag.AlignTop,
        )
        self.existing_list_container.setVisible(False)
        self._existing_matches: list[dict] = []

        recommendation = QGroupBox("☆  Match a Sound")
        recommendation.setObjectName("glassPanel")
        recommendation_layout = QVBoxLayout(recommendation)
        recommendation_layout.setContentsMargins(10, 16, 10, 8)
        recommendation_layout.setSpacing(8)
        self.recommendation_heading = QLabel("Recommended Preset")
        self.recommendation_heading.setVisible(False)

        self.recommendation_placeholder = self._muted_label(
            "Run a match to see your recommended preset here."
        )
        self.recommendation_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        recommendation_layout.addWidget(self.recommendation_placeholder)

        self.recommendation_details = QWidget()
        self.recommendation_details.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        details_layout = QVBoxLayout(self.recommendation_details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(6)
        self.recommendation_details.setVisible(False)
        recommendation_layout.addWidget(self.recommendation_details)

        # The drop zone sits last, below the octave card, and is the only
        # expanding child — so the result content stays packed at the top of
        # the panel and the drop zone absorbs all remaining height instead of
        # leaving gaps between the rows above it.
        self.match_drop = AudioDropLabel()
        self.match_drop.file_dropped.connect(self._set_match_file)
        self.match_drop.browse_requested.connect(self.choose_match_file)
        self.match_drop.play_requested.connect(self.play_uploaded_audio)
        recommendation_layout.addWidget(self.match_drop, 1)

        header_row = QHBoxLayout()
        header_row.setSpacing(9)
        self.recommendation_thumbnail = PresetThumbnail("blue")
        header_row.addWidget(self.recommendation_thumbnail)
        info_column = QVBoxLayout()
        info_column.setSpacing(2)
        self.recommendation_badge = QLabel("BEST MATCH")
        self.recommendation_badge.setObjectName("cardStatus")
        self.recommendation_name = QLabel("")
        self.recommendation_name.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.recommendation_name.setWordWrap(True)
        self.recommendation_subtitle = QLabel("")
        self.recommendation_subtitle.setObjectName("muted")
        tags_row = QHBoxLayout()
        tags_row.setSpacing(4)
        self.recommendation_tags: list[QLabel] = []
        for _ in range(4):
            tag = QLabel("")
            tag.setObjectName("tagPill")
            tag.setVisible(False)
            tags_row.addWidget(tag)
            self.recommendation_tags.append(tag)
        tags_row.addStretch(1)
        info_column.addWidget(self.recommendation_badge)
        info_column.addWidget(self.recommendation_name)
        info_column.addWidget(self.recommendation_subtitle)
        info_column.addLayout(tags_row)
        header_row.addLayout(info_column, 1)
        self.confidence_ring = ConfidenceRing()
        header_row.addWidget(self.confidence_ring)
        details_layout.addLayout(header_row)

        self.recommendation_confidence = QLabel("")
        self.recommendation_confidence.setWordWrap(True)
        self.recommendation_confidence.setObjectName("muted")
        details_layout.addWidget(self.recommendation_confidence)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)
        self.save_preset_button = QPushButton("Rename Preset")
        self.save_preset_button.setObjectName("compactActionButton")
        self.save_preset_button.setIcon(icon("save"))
        self.save_preset_button.clicked.connect(self.save_match_preset)
        self.open_preset_location_button = QPushButton("Open Preset File Location")
        self.open_preset_location_button.setObjectName("compactActionButton")
        self.open_preset_location_button.clicked.connect(
            lambda _checked=False: self.open_preset_file_location()
        )
        self.save_preset_now_button = QPushButton("Save Preset")
        self.save_preset_now_button.setObjectName("compactActionButton")
        self.save_preset_now_button.setVisible(False)
        self.save_preset_now_button.clicked.connect(lambda _checked=False: self.save_preset_now())
        self.recommendation_more_button = QPushButton("…")
        self.recommendation_more_button.setObjectName("compactActionButton")
        self.recommendation_more_button.setFixedWidth(30)
        self.recommendation_more_button.clicked.connect(self._show_recommendation_more_menu)
        actions_row.addWidget(self.open_preset_location_button)
        actions_row.addWidget(self.save_preset_button)
        actions_row.addWidget(self.save_preset_now_button)
        actions_row.addWidget(self.recommendation_more_button)
        details_layout.addLayout(actions_row)
        # Shown only when the automatic save of this result has run out of
        # recovery; a normal save never shows any of this.
        self.save_failure_label = QLabel("")
        self.save_failure_label.setWordWrap(True)
        self.save_failure_label.setObjectName("muted")
        self.save_failure_label.setVisible(False)
        self.retry_save_button = QPushButton("Retry Saving Preset")
        self.retry_save_button.setObjectName("compactActionButton")
        self.retry_save_button.setVisible(False)
        self.retry_save_button.clicked.connect(lambda _checked=False: self.retry_saving_preset())
        self.run_again_button = QPushButton("Run Again")
        self.run_again_button.setObjectName("compactActionButton")
        self.run_again_button.setVisible(False)
        self.run_again_button.clicked.connect(lambda _checked=False: self.run_again_from_incident())
        save_state_row = QHBoxLayout()
        save_state_row.setSpacing(6)
        save_state_row.addWidget(self.retry_save_button)
        save_state_row.addWidget(self.run_again_button)
        save_state_row.addStretch(1)
        details_layout.addWidget(self.save_failure_label)
        details_layout.addLayout(save_state_row)
        # winner_play_button is retained (hidden) only so older code paths
        # that reference it for enable/disable state don't need special
        # casing; playback is now triggered directly by clicking an octave.
        self.winner_play_button = QPushButton()
        self.winner_play_button.setVisible(False)
        octave_card = QFrame()
        octave_card.setObjectName("controlCard")
        octave_layout = QVBoxLayout(octave_card)
        octave_layout.setContentsMargins(11, 8, 11, 8)
        octave_layout.setSpacing(5)
        octave_title_row = QHBoxLayout()
        octave_title = QLabel("AUDITION OCTAVE — click to play")
        octave_title.setObjectName("controlTitle")
        self.octave_status = QLabel("C4 · MIDI 60")
        self.octave_status.setObjectName("muted")
        octave_title_row.addWidget(octave_title)
        octave_title_row.addStretch(1)
        octave_title_row.addWidget(self.octave_status)
        octave_layout.addLayout(octave_title_row)
        self.octave_selector = SegmentedControl()
        for octave, note in enumerate((24, 36, 48, 60, 72, 84, 96), start=1):
            self.octave_selector.addItem(f"C{octave}", note)
        self.octave_selector.setCurrentIndex(3)
        # itemClicked drives playback so re-clicking the octave already
        # selected replays it; currentIndexChanged only refreshes the label,
        # and fires solely for programmatic changes or the first click of a
        # new octave, so a single click never plays twice.
        self.octave_selector.currentIndexChanged.connect(self._octave_changed)
        self.octave_selector.itemClicked.connect(self._octave_selected)
        octave_layout.addWidget(self.octave_selector)
        octave_help = QLabel(
            "Click an octave to play the recommendation at that note. "
            "Each closest match below has its own octave row."
        )
        octave_help.setObjectName("muted")
        octave_help.setWordWrap(True)
        octave_layout.addWidget(octave_help)
        self.octave_help = octave_help
        self.octave_card = octave_card
        details_layout.addWidget(octave_card)
        # settings_tree/parameter_strip back this panel's "…" full-settings
        # dialog (see _show_recommendation_more_menu) rather than being shown
        # inline — the redesigned recommendation surface stays compact by
        # default.
        self.parameter_knobs: list[QWidget] = []
        self.parameter_strip = QWidget(recommendation)
        self.parameter_strip.setVisible(False)
        self.settings_tree = QTreeWidget(recommendation)
        self.settings_tree.setVisible(False)
        self.settings_tree.itemExpanded.connect(self._settings_item_expanded)
        self.settings_tree.itemClicked.connect(self._settings_item_clicked)
        self.recommendation_panel = recommendation
        limitation = QLabel(
            "Custom wavetable content is retained from the base preset but "
            "cannot be generated or supplied by PatchLab."
        )
        limitation.setObjectName("muted")
        limitation.setWordWrap(True)
        limitation.setStyleSheet("font-size: 9px;")
        details_layout.addWidget(limitation)
        self.wavetable_limitation = limitation
        # Distribution adds the factory-health strip and three separated
        # control cards above the results. Keep the results bounded so its
        # bottom edge never runs underneath the log/status rows in the locked
        # 16:9 canvas.
        self.match_results.setFixedHeight(660 if self.distribution_mode else 740)
        results_layout.addWidget(closest, 43)
        results_layout.addWidget(recommendation, 57)
        self.match_results.setVisible(True)
        root_layout.addWidget(self.match_results)

        self.log_pane = QTextEdit()
        self.log_pane.setObjectName("logConsole")
        self.log_pane.setReadOnly(True)
        self.log_pane.document().setMaximumBlockCount(10_000)
        self.log_pane.setFixedHeight(34)
        self.log_pane.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.log_pane.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        root_layout.addWidget(self.log_pane)

        # Keep navigation inside the fixed design canvas.  The Match page owns
        # the existing dashboard unchanged; only the Library list scrolls.
        match_page = QWidget()
        match_page.setObjectName("matchPage")
        match_page_layout = QVBoxLayout(match_page)
        match_page_layout.setContentsMargins(0, 0, 0, 0)
        match_page_layout.setSpacing(16)
        while root_layout.count() > 1:
            item = root_layout.takeAt(1)
            if item.widget() is not None:
                match_page_layout.addWidget(item.widget())
            elif item.layout() is not None:
                match_page_layout.addLayout(item.layout())
            else:
                match_page_layout.addItem(item)
        self.library_page = self._create_library_page()
        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(match_page)
        self.page_stack.addWidget(self.library_page)
        root_layout.addWidget(self.page_stack, 1)
        self.nav_tabs = QTabBar()
        self.nav_tabs.setObjectName("mainTabs")
        self.nav_tabs.addTab("Match")
        self.nav_tabs.addTab("Library")
        self.nav_tabs.setExpanding(False)
        self.nav_tabs.currentChanged.connect(self._tab_changed)
        top_layout.insertWidget(max(0, top_layout.count() - 2), self.nav_tabs)

        self._ui_root = root
        root.setFixedSize(self.DESIGN_WIDTH, self.DESIGN_HEIGHT)
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(0, 0, self.DESIGN_WIDTH, self.DESIGN_HEIGHT)
        root_proxy = self._scene.addWidget(root)
        # QGraphicsItem.acceptDrops defaults to False and is entirely separate
        # from QWidget.setAcceptDrops (already set on AudioDropLabel and the
        # view) — without this, the scene silently discards every drag event
        # before it reaches the embedded widget tree, so drag-and-drop onto
        # AudioDropLabel never fires even though its handlers are correct.
        root_proxy.setAcceptDrops(True)
        self._scene.setBackgroundBrush(QColor(theme.BASE))
        self._graphics_view = ScaledGraphicsView(self._scene, self)
        self._graphics_view.setObjectName("scaledCanvas")
        self._graphics_view.setFrameShape(QFrame.Shape.NoFrame)
        self._graphics_view.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._graphics_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._graphics_view.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self._graphics_view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._graphics_view.setAcceptDrops(True)
        self._graphics_view.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter
        )
        self._graphics_view.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        self.setCentralWidget(self._graphics_view)
        self.status_health = QLabel("●  All Systems Operational")
        self.status_health.setStyleSheet(
            f"color: {theme.GREEN}; padding-right: 10px;"
        )
        self.statusBar().addPermanentWidget(self.status_health)
        if self.distribution_mode:
            self._apply_factory_status()
            self._apply_privacy_choice()
        # Keep the detached legacy container alive for this window lifetime.
        # Qt owns some moved controls through their original layouts until the
        # event loop settles; eager deletion here can invalidate them.
        self._detached_legacy_central = old_central
        self._compact_ui = False
        self.refresh_match_library()

    def _match_database_path(self) -> Path:
        return self.local_paths["db"] if self.distribution_mode else DEFAULT_DB_PATH

    def _match_library_root(self) -> Path:
        return (
            self.local_paths["matches"]
            if self.distribution_mode
            else DEFAULT_MATCH_LIBRARY_ROOT
        )

    def _create_library_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        header = QFrame()
        header.setObjectName("controlCard")
        header_layout = QHBoxLayout(header)
        title_column = QVBoxLayout()
        title = QLabel("MATCH LIBRARY")
        title.setObjectName("controlTitle")
        subtitle = QLabel(
            "Every completed match is saved here. Double-click or press Enter to reopen it."
        )
        subtitle.setObjectName("muted")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header_layout.addLayout(title_column, 1)
        self.library_batch_status = QLabel("No batch running")
        self.library_batch_status.setObjectName("controlStat")
        self.library_batch_cancel = QPushButton("Cancel Batch")
        self.library_batch_cancel.setObjectName("compactActionButton")
        self.library_batch_cancel.setEnabled(False)
        self.library_batch_cancel.clicked.connect(self.cancel_match_batch)
        header_layout.addWidget(self.library_batch_status)
        header_layout.addWidget(self.library_batch_cancel)
        layout.addWidget(header)

        self.library_scroll = QScrollArea()
        self.library_scroll.setObjectName("libraryScroll")
        self.library_scroll.setWidgetResizable(True)
        self.library_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.library_container = QWidget()
        self.library_list_layout = QVBoxLayout(self.library_container)
        self.library_list_layout.setContentsMargins(7, 7, 7, 7)
        self.library_list_layout.setSpacing(7)
        self.library_scroll.setWidget(self.library_container)
        layout.addWidget(self.library_scroll, 1)
        return page

    def _tab_changed(self, index: int) -> None:
        self.page_stack.setCurrentIndex(index)
        if index == 1:
            self.refresh_match_library()

    def _build_library_row(self, record) -> LibraryEntryRow:  # type: ignore[no-untyped-def]
        row = LibraryEntryRow(record.match_uid)
        row.activated.connect(self.open_library_match)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(10, 7, 10, 7)
        row_layout.setSpacing(8)
        source_button = QPushButton("▶")
        source_button.setObjectName("playButton")
        source_button.setToolTip("Play the archived source audio")
        source_button.clicked.connect(
            lambda _checked=False, uid=record.match_uid: self.play_library_source(uid)
        )
        row_layout.addWidget(source_button)
        text = QVBoxLayout()
        name = QLabel(record.source_name)
        name.setStyleSheet("font-weight: 700; font-size: 12px;")
        status = (
            "No confident match"
            if record.no_confident_match
            else f"{record.similarity_percent:.1f}% similarity"
        )
        synth = "Serum 1" if record.target_synth == "serum1" else "Serum 2"
        detail = QLabel(f"{record.created_at} · {synth} · {status}")
        detail.setObjectName("muted")
        text.addWidget(name)
        text.addWidget(detail)
        row_layout.addLayout(text, 1)
        for octave, note in enumerate((24, 36, 48, 60, 72, 84, 96), start=1):
            button = QPushButton(f"C{octave}")
            button.setObjectName("rowOctaveButton")
            button.setEnabled(not record.no_confident_match)
            button.clicked.connect(
                lambda _checked=False, uid=record.match_uid, midi=note, control=button:
                self.play_library_octave(uid, midi, control)
            )
            row_layout.addWidget(button)
        # Almost every entry already has an auto-saved preset by the time it
        # can be clicked: the normal case is Open Preset File Location +
        # Rename Preset. A save that ran out of automatic recovery offers
        # Retry Saving Preset / Run Again instead; a record from before
        # auto-save existed offers Save Preset. Exactly one of these states
        # applies, never a permanent Export Preset alongside the others.
        incident = getattr(self, "_library_incidents", {}).get(record.match_uid)
        incident_status = incident.status if incident is not None else ""
        saved_file = (
            record.exported_preset_path is not None
            and Path(record.exported_preset_path).is_file()
        )
        buttons: list[QPushButton] = []
        if saved_file:
            open_location = QPushButton("Open Preset File Location")
            open_location.setObjectName("compactActionButton")
            open_location.clicked.connect(
                lambda _checked=False, uid=record.match_uid: self.open_preset_file_location(uid)
            )
            rename = QPushButton("Rename Preset")
            rename.setObjectName("compactActionButton")
            rename.clicked.connect(
                lambda _checked=False, uid=record.match_uid: self.export_library_match(uid)
            )
            buttons = [open_location, rename]
        elif incident_status == "failed":
            retry = QPushButton("Retry Saving Preset")
            retry.setObjectName("compactActionButton")
            retry.clicked.connect(
                lambda _checked=False, uid=record.match_uid: self.retry_saving_preset(uid)
            )
            buttons = [retry]
        elif incident_status == "unrecoverable":
            again = QPushButton("Run Again")
            again.setObjectName("compactActionButton")
            again.clicked.connect(
                lambda _checked=False, uid=record.match_uid: self.run_again_from_incident(uid)
            )
            buttons = [again]
        else:
            save_now = QPushButton("Save Preset")
            save_now.setObjectName("compactActionButton")
            save_now.clicked.connect(
                lambda _checked=False, uid=record.match_uid: self.save_preset_now(uid)
            )
            buttons = [save_now]
        for button in buttons:
            button.setEnabled(not record.no_confident_match and self._batch_state is None)
            if self._batch_state is not None:
                button.setToolTip("Reserved for the active batch. Try again after it finishes.")
            row_layout.addWidget(button)
        delete = QPushButton("Delete")
        delete.setObjectName("compactActionButton")
        delete.clicked.connect(
            lambda _checked=False, uid=record.match_uid: self.delete_library_match(uid)
        )
        row_layout.addWidget(delete)
        return row

    def refresh_match_library(self) -> None:
        if not hasattr(self, "library_list_layout"):
            return
        layout = self.library_list_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        try:
            from core.preset_save import incidents_by_match

            self._library_incidents = incidents_by_match()
        except Exception:
            self._library_incidents = {}
        try:
            database = Database(self._match_database_path())
            records = database.list_match_library()
            batches = {batch.id: batch for batch in database.list_match_batches()}
        except Exception as exc:
            error = QLabel(f"The Match Library could not be opened: {exc}")
            error.setObjectName("muted")
            layout.addWidget(error)
            return
        if not records:
            if self._batch_state is not None:
                running = QFrame()
                running.setObjectName("controlCard")
                running_layout = QVBoxLayout(running)
                running_layout.addWidget(
                    QLabel(
                        f"▾  {self._batch_state['folder_name']} · "
                        f"{self.library_batch_status.text()}"
                    )
                )
                running_layout.addWidget(
                    self._muted_label(
                        "The first saved result will appear here when its verified export completes."
                    )
                )
                layout.addWidget(running)
            empty = QLabel(
                "Your completed matches will appear here automatically.\n"
                "Run Match a Sound or choose Batch Folder to begin."
            )
            empty.setObjectName("muted")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if self._batch_state is None:
                layout.addWidget(empty, 1)
            return
        rendered_batches: set[int] = set()
        for record in records:
            if record.batch_id is None:
                layout.addWidget(self._build_library_row(record))
                continue
            if record.batch_id in rendered_batches:
                continue
            rendered_batches.add(record.batch_id)
            batch = batches.get(record.batch_id)
            grouped = [item for item in records if item.batch_id == record.batch_id]
            group = QFrame()
            group.setObjectName("controlCard")
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(7, 7, 7, 7)
            group_layout.setSpacing(5)
            content = QWidget()
            content_layout = QVBoxLayout(content)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.setSpacing(5)
            for item in grouped:
                content_layout.addWidget(self._build_library_row(item))
            batch_name = batch.folder_name if batch else "Batch"
            batch_status = batch.status if batch else "saved"
            active_detail = ""
            if (
                self._batch_state is not None
                and int(self._batch_state["batch_id"]) == int(record.batch_id)
            ):
                active_detail = f" · {self.library_batch_status.text()}"
            header = QPushButton(
                f"▾  {batch_name} · {len(grouped)} files · {batch_status}{active_detail}"
            )
            header.setObjectName("compactActionButton")
            header.setCheckable(True)
            header.setChecked(True)
            header.toggled.connect(content.setVisible)
            group_layout.addWidget(header)
            group_layout.addWidget(content)
            layout.addWidget(group)
        layout.addStretch(1)

    def play_library_source(self, match_uid: str) -> None:
        record = Database(self._match_database_path()).get_match_library(match_uid)
        if record is None:
            return
        source, _result = resolved_record_paths(record, self._match_library_root())
        if self._play_audio(source):
            self.statusBar().showMessage(f"Playing archived source — {record.source_name}")

    def play_library_octave(
        self, match_uid: str, note: int, button: QPushButton
    ) -> None:
        record = Database(self._match_database_path()).get_match_library(match_uid)
        if record is None:
            return
        _source, result_path = resolved_record_paths(
            record, self._match_library_root()
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        recommendation = result.get("recommendation")
        if not isinstance(recommendation, dict):
            self.statusBar().showMessage("This Library entry has no preset preview")
            return
        cache_key = recommendation_cache_key(result_path, recommendation)
        unmodified_source = (
            recommendation.get("preview_source_path")
            if not bool(recommendation.get("meaningfully_modified", False))
            else None
        )
        existing_audio: Path | None = None
        if not bool(recommendation.get("meaningfully_modified", False)):
            basis = next(
                (
                    item
                    for item in result.get("existing_matches", [])
                    if isinstance(item, dict)
                    and str(item.get("content_hash") or "") == cache_key
                ),
                None,
            )
            if basis and basis.get("audition_path"):
                existing_audio = (
                    Path(str(basis["audition_path"])).parent / f"{note}.wav"
                )
        self._resolve_octave_preview(
            cache_key=cache_key,
            note=note,
            synth=str(recommendation["synth"]),
            preview_source_path=unmodified_source,
            result_path=result_path,
            existing_audio_path=existing_audio,
            button=button,
        )

    def open_library_match(self, match_uid: str) -> None:
        record = Database(self._match_database_path()).get_match_library(match_uid)
        if record is None:
            return
        source, result_path = resolved_record_paths(record, self._match_library_root())
        self._match_audio_path = source
        self._match_result_path = result_path
        self._match_result = json.loads(result_path.read_text(encoding="utf-8"))
        self._current_match_uid = match_uid
        self.match_drop.setText(record.source_name)
        self.match_drop.set_playable(True)
        self._show_match_result(self._match_result)
        self.nav_tabs.setCurrentIndex(0)
        self._refresh_save_state()

    def delete_library_match(self, match_uid: str) -> None:
        if QMessageBox.question(
            self,
            "Delete saved match?",
            "This permanently removes this Match Library entry and its private "
            "generated preview audio. Shared factory/user-preset previews are "
            "kept so future auditions remain instant.",
        ) != QMessageBox.StandardButton.Yes:
            return
        delete_archived_match(
            Database(self._match_database_path()),
            match_uid,
            library_root=self._match_library_root(),
            cache_root=self._preview_cache_root(),
        )
        if self._current_match_uid == match_uid:
            self._current_match_uid = None
        self.refresh_match_library()

    def export_library_match(self, match_uid: str) -> None:
        """Rename a library entry's already-saved preset in place.

        Every match auto-saves its preset the moment it completes, so this is
        only ever reached once a file genuinely exists (see
        _build_library_row); a record that never got one offers Save Preset
        instead, which goes through the save-incident machinery, not a
        dialog-based export.
        """

        if self._batch_state is not None:
            QMessageBox.information(
                self,
                "Batch is running",
                "Renaming is reserved for the active batch. Try again after it finishes.",
            )
            return
        record = Database(self._match_database_path()).get_match_library(match_uid)
        if record is None or record.exported_preset_path is None:
            return
        self._rename_saved_preset(match_uid, Path(record.exported_preset_path))

    def _archive_completed_result(
        self,
        result_path: Path,
        source_path: Path,
        *,
        batch_id: int | None = None,
    ):
        state = self._batch_state
        archived = archive_match(
            Database(self._match_database_path()),
            result_path=result_path,
            source_audio_path=source_path,
            target_synth=(
                str(state["target_synth"])
                if state is not None
                else str(self.match_synth.currentData())
            ),
            budget=(
                str(state["budget"])
                if state is not None
                else str(self.match_budget.currentData())
            ),
            library_root=self._match_library_root(),
            batch_id=batch_id,
        )
        self.append_log(f"Saved match to Library: {archived.record.source_name}")
        self.refresh_match_library()
        return archived

    def start_match(self) -> None:
        if self._batch_state is not None:
            QMessageBox.information(
                self,
                "Batch is running",
                "Single matches are paused while this batch runs. You can still browse and audition the Library.",
            )
            return
        super().start_match()

    def _match_completed(self, result_path: str) -> None:
        source = (
            self._batch_state["current_path"]
            if self._batch_state is not None
            else self._match_audio_path
        )
        LegacyMainWindow._match_completed(self, result_path)
        if source is None:
            self.append_log("Match completed, but its source path was unavailable for archiving")
            return
        try:
            archived = self._archive_completed_result(
                Path(result_path),
                Path(source),
                batch_id=(
                    int(self._batch_state["batch_id"])
                    if self._batch_state is not None
                    else None
                ),
            )
        except Exception as exc:
            self.append_log(f"Match completed but could not be archived: {exc}")
            if self._batch_state is not None:
                self._batch_file_failed(f"archive failed: {exc}")
            return
        self._match_result_path = archived.result_json_path
        self._match_audio_path = archived.source_audio_path
        self._match_result = json.loads(
            archived.result_json_path.read_text(encoding="utf-8")
        )
        self._current_match_uid = archived.record.match_uid
        # Audition previews are rendered only after an octave is clicked.
        # Prepared Match data never depends on this disposable cache.
        if self._batch_state is not None:
            self._batch_state["current_uid"] = archived.record.match_uid
        self._auto_save_generated_preset(archived, Path(source))
        self._refresh_save_state()

    def _auto_save_generated_preset(self, archived, source: Path) -> None:
        """Verified-export every generated patch straight into Serum's folder.

        Single matches and batch files share this path so the two never
        diverge. A batch's completion/failure bookkeeping (advancing to the
        next file) is driven from here via _batch_state["phase"]; a single
        match instead tags the export with _export_context_uid, which
        _export_completed/_export_failed already handle silently — no dialog,
        no new file, just the DB's exported_preset_path getting recorded.
        """

        recommendation = self._match_result.get("recommendation") if self._match_result else None
        if not isinstance(recommendation, dict):
            if self._batch_state is not None:
                self.append_log(
                    f"Batch retained no-confident match: {archived.record.source_name}; no preset was exported"
                )
                self._batch_file_completed()
            return
        extension = ".fxp" if recommendation["synth"] == "serum1" else ".SerumPreset"
        source_stem = sanitize_folder_name(source.stem) or "Sound"
        folder = (
            self._batch_state["export_folder"]
            if self._batch_state is not None
            else self._patchlab_export_folder(str(recommendation["synth"]))
        )
        output = disambiguated_preset_path(folder, f"PatchLab - {source_stem}", extension)
        if self._batch_state is not None:
            self._batch_state["phase"] = "export"
        else:
            self._export_context_uid = archived.record.match_uid
        incident = None
        try:
            from core.preset_save import SaveIncident

            source_meta = (self._match_result or {}).get("source") or {}
            incident = SaveIncident.create(
                match_uid=archived.record.match_uid,
                result_path=archived.result_json_path,
                target_path=output,
                synth=str(recommendation["synth"]),
                run_again={
                    "source_audio_path": str(archived.source_audio_path),
                    "target_synth": str(archived.record.target_synth),
                    "budget": str(archived.record.budget),
                    "start_offset_s": float(source_meta.get("start_offset_s", 0.0) or 0.0),
                },
            )
        except Exception as exc:
            # The incident is how a failed save stays recoverable; without one
            # the save still runs, just without automatic retries.
            self.append_log(f"Save incident could not be recorded: {exc}")
        try:
            if incident is not None:
                from core.preset_save import MAX_AUTOMATIC_ATTEMPTS

                self._save_incident_context = {
                    "incident_id": incident.incident_id,
                    "match_uid": archived.record.match_uid,
                    "trigger": "auto",
                }
                self.export_runner.start(
                    archived.result_json_path, output,
                    incident_id=incident.incident_id, trigger="auto",
                    attempts=MAX_AUTOMATIC_ATTEMPTS,
                )
            else:
                self.export_runner.start(archived.result_json_path, output)
        except RuntimeError as exc:
            self.append_log(f"Could not auto-save generated preset: {exc}")
            self._save_incident_context = None
            if incident is not None:
                from core.preset_save import record_worker_crash

                record_worker_crash(incident, f"the save could not start: {exc}", trigger="auto")
            if self._batch_state is not None:
                self._batch_file_failed(f"auto-save failed: {exc}")
            else:
                self._export_context_uid = None
                self._refresh_save_state()

    def _match_failed(self, error: str) -> None:
        if self._batch_state is not None:
            self._batch_file_failed(error)
            return
        super()._match_failed(error)

    def _export_completed(self, detail: dict) -> None:
        context = self._save_incident_context
        if context is not None and detail.get("incident_id") == context["incident_id"]:
            self._save_incident_finished(context, detail)
            return
        uid = self._export_context_uid
        if uid:
            Database(self._match_database_path()).set_match_exported_path(
                uid, Path(detail["path"])
            )
            self._export_context_uid = None
            self.append_log(f"Verified Library export saved: {detail['path']}")
            self.statusBar().showMessage(f"Preset saved: {detail['path']}")
            self.refresh_match_library()
            return
        if self._batch_state is not None and self._batch_state.get("phase") == "export":
            current_uid = self._batch_state.get("current_uid")
            if current_uid:
                Database(self._match_database_path()).set_match_exported_path(
                    str(current_uid), Path(detail["path"])
                )
            self.append_log(f"Batch verified preset saved: {detail['path']}")
            self._batch_file_completed()
            return
        # Remaining case: a closest match's "Save Copy" -- a plain, standalone
        # file, deliberately never tied to the primary result's own saved-
        # preset bookkeeping (_current_match_uid is a different result).
        message = f"Preset saved: {detail['path']}"
        self.append_log(message)
        warning = detail.get("verification_warning")
        if warning:
            self.append_log(f"Preset verification note: {warning}")
        self.statusBar().showMessage(message)
        if not warning:
            QMessageBox.information(self, "Preset saved", message)

    def _export_failed(self, error: str) -> None:
        context = self._save_incident_context
        if context is not None:
            self._save_incident_failed(context, error)
            return
        if self._export_context_uid:
            self._export_context_uid = None
            self.append_log(f"Library preset export failed: {error}")
            self.statusBar().showMessage(error)
            return
        if self._batch_state is not None and self._batch_state.get("phase") == "export":
            self._batch_file_failed(f"verified export failed: {error}")
            return
        self.append_log(f"Preset save failed: {error}")
        self.statusBar().showMessage(error)
        QMessageBox.critical(
            self,
            "Preset was not saved",
            "PatchLab could not write a valid preset to the selected location.\n\n"
            + error,
        )

    # ------------------------------------------------------------------
    # Generated-preset save lifecycle (core.preset_save)
    # ------------------------------------------------------------------

    def _save_incident_finished(self, context: dict, detail: dict) -> None:
        """A save incident reached a verified file: record it, then show it saved."""

        from core.preset_save import SaveIncident, finalize_saved_incident

        self._save_incident_context = None
        if self._export_context_uid == context["match_uid"]:
            self._export_context_uid = None
        final_path = Path(detail["path"])
        recorded = False
        try:
            incident = SaveIncident.load(context["incident_id"])
            recorded = finalize_saved_incident(
                incident, Database(self._match_database_path()), final_path
            )
        except Exception as exc:
            self.append_log(f"Could not finish recording the saved preset: {exc}")
        attempts = int(detail.get("attempts") or 1)
        if recorded:
            self.append_log(
                f"Verified preset saved: {final_path}"
                + (f" (after {attempts} attempts)" if attempts > 1 else "")
            )
            self.statusBar().showMessage(f"Preset saved: {final_path.name}")
        else:
            # The file is real and verified; only the Library entry is behind.
            # Reconciliation finishes it without writing a second preset.
            self.append_log(
                f"Preset saved at {final_path}, but the Library could not record it yet; "
                "PatchLab will finish this automatically."
            )
            QTimer.singleShot(1500, self._reconcile_save_incidents)
        warning = detail.get("verification_warning")
        if warning:
            self.append_log(f"Preset verification note: {warning}")
        if self._batch_state is not None and self._batch_state.get("phase") == "export":
            self.append_log(f"Batch verified preset saved: {final_path}")
            self._batch_file_completed()
            return
        self.refresh_match_library()
        self._refresh_save_state()

    def _save_incident_failed(self, context: dict, error: str) -> None:
        """Automatic recovery (or a manual retry) is exhausted for this result."""

        from core.preset_save import SaveIncident, record_worker_crash

        self._save_incident_context = None
        if self._export_context_uid == context["match_uid"]:
            self._export_context_uid = None
        failure = getattr(self.export_runner, "failure", None) or {}
        try:
            incident = SaveIncident.load(context["incident_id"])
            if not failure:
                # The worker ended without an outcome of its own (it crashed or
                # never started); record that as this incident's failure.
                record_worker_crash(incident, error, trigger=context["trigger"])
        except Exception as exc:
            self.append_log(f"Save incident could not be read: {exc}")
            incident = None
        self.append_log(
            "Preset save failed"
            + (f" after {failure.get('attempts')} attempt(s)" if failure.get("attempts") else "")
            + f" ({failure.get('kind', 'unknown')}/{failure.get('reason', 'unknown')}): {error}"
        )
        if incident is not None:
            self._submit_save_incident_report(
                incident, followup=context["trigger"] == "manual"
            )
        if self._batch_state is not None and self._batch_state.get("phase") == "export":
            self._batch_file_failed(f"preset could not be saved: {error}")
            return
        self.statusBar().showMessage(error)
        self.refresh_match_library()
        self._refresh_save_state()

    def _refresh_save_state(self) -> None:
        """The one place deciding what a generated result's save controls show.

        SAVE SUCCEEDED: Open Preset File Location (+ Rename Preset).
        AUTO SAVE STILL RUNNING: a disabled "Saving…" indicator, nothing else.
        AUTO SAVE FAILED (recovery exhausted) / a fixable destination issue
        after a failed manual retry: Retry Saving Preset, with the reason.
        UNRECOVERABLE: Run Again.
        A confident recommendation that predates Phase 1's save incidents and
        was never auto-saved: Save Preset (the same incident machinery, run
        by hand instead of automatically after a match).
        No confident match, or this recommendation cannot be saved at all:
        nothing.

        Never both a permanent Export/Save action and Open Preset File
        Location for the same result -- exactly one of the states above
        applies at a time.
        """

        open_button = getattr(self, "open_preset_location_button", None)
        if open_button is None:
            return
        rename, save_now = self.save_preset_button, self.save_preset_now_button
        label, retry, again = self.save_failure_label, self.retry_save_button, self.run_again_button

        def hide_everything() -> None:
            for widget in (open_button, rename, save_now, retry, again, label):
                widget.setVisible(False)

        # Whether a *fresh* recommendation can be saved at all lives on the
        # recommendation JSON, not the Library row -- it is applied directly
        # (setEnabled, not hidden) at the one call site that computes it
        # (_show_match_result), the same way these buttons' enabled state was
        # always set before this state machine existed. This method itself is
        # driven purely by _current_match_uid's own saved/incident state, so
        # it must stay reachable for retries, Run Again and Library rows even
        # when no recommendation is currently on screen at all.

        record = None
        if self._current_match_uid:
            try:
                record = Database(self._match_database_path()).get_match_library(
                    self._current_match_uid
                )
            except Exception:
                record = None
        saved_file = bool(
            record is not None
            and record.exported_preset_path is not None
            and Path(record.exported_preset_path).is_file()
        )
        incident = None
        if self._current_match_uid:
            try:
                from core.preset_save import incident_for_match

                incident = incident_for_match(self._current_match_uid)
            except Exception:
                incident = None
        saving = (
            self._save_incident_context is not None
            and self._save_incident_context.get("match_uid") == self._current_match_uid
        )
        status = incident.status if incident is not None else ""

        hide_everything()
        if self._batch_state is not None:
            # A batch drives its own progress UI; these controls describe
            # whatever single result is merely being displayed underneath it.
            return
        if saved_file:
            open_button.setVisible(True)
            open_button.setEnabled(True)
            rename.setText("Rename Preset")
            rename.setVisible(True)
            rename.setEnabled(True)
            return
        if saving:
            save_now.setText("Saving…")
            save_now.setEnabled(False)
            save_now.setVisible(self._save_incident_context.get("trigger") != "manual")
            retry.setText("Saving…")
            retry.setEnabled(False)
            retry.setVisible(self._save_incident_context.get("trigger") == "manual")
            return
        if status in ("failed", "unrecoverable"):
            from core.preset_save import incident_user_message

            label.setText(incident_user_message(incident))
            label.setVisible(True)
            retry.setText("Retry Saving Preset")
            retry.setEnabled(True)
            retry.setVisible(status == "failed")
            again.setEnabled(not self.match_runner.running)
            again.setVisible(status == "unrecoverable")
            return
        if record is not None and not record.no_confident_match:
            # A Library record from before auto-save existed: offer the same
            # save machinery, just triggered by hand instead of automatically.
            save_now.setText("Save Preset")
            save_now.setEnabled(True)
            save_now.setVisible(True)

    def open_preset_file_location(self, match_uid: str | None = None) -> None:
        """Reveal the exact file a generated preset was actually saved to.

        Always reads the Library record's own stored ``exported_preset_path``
        -- recorded once, at save time -- never recomputed from today's
        Settings -> Generated Preset Folder, so a later change to that
        setting never breaks revealing an older result.
        """

        match_uid = match_uid or self._current_match_uid
        if not match_uid:
            return
        record = Database(self._match_database_path()).get_match_library(match_uid)
        if record is None or record.exported_preset_path is None:
            QMessageBox.information(
                self, "No saved preset", "This result has no saved preset file yet."
            )
            return
        self._reveal_in_finder(Path(record.exported_preset_path))

    def save_preset_now(self, match_uid: str | None = None) -> None:
        """First save attempt for a Library result that never got one.

        For a record made before auto-save existed (or whose one archiving
        attempt failed before any incident was recorded): goes through the
        exact same save-incident machinery as auto-save -- build, verify,
        atomic commit, up to three automatic attempts -- just triggered by
        hand instead of running right after a fresh match.
        """

        from core.preset_save import MAX_AUTOMATIC_ATTEMPTS, SaveIncident

        match_uid = match_uid or self._current_match_uid
        if not match_uid:
            return
        if self._batch_state is not None:
            QMessageBox.information(
                self, "Batch is running", "Try saving this preset again after the batch finishes."
            )
            return
        if self.export_runner.running:
            QMessageBox.information(
                self, "Still saving", "PatchLab is saving another preset. Try again in a moment."
            )
            return
        record = Database(self._match_database_path()).get_match_library(match_uid)
        if record is None or record.no_confident_match:
            return
        source, result_path = resolved_record_paths(record, self._match_library_root())
        if not result_path.is_file():
            QMessageBox.information(
                self,
                "Result not found",
                "PatchLab can no longer find this result on your Mac, so it can't "
                "save a preset for it.",
            )
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        recommendation = result.get("recommendation")
        if not isinstance(recommendation, dict):
            return
        synth = str(recommendation["synth"])
        extension = ".fxp" if synth == "serum1" else ".SerumPreset"
        source_stem = sanitize_folder_name(Path(record.source_name).stem) or "Sound"
        output = disambiguated_preset_path(
            self._patchlab_export_folder(synth), f"PatchLab - {source_stem}", extension
        )
        source_meta = result.get("source") or {}
        incident = SaveIncident.create(
            match_uid=match_uid, result_path=result_path, target_path=output, synth=synth,
            run_again={
                "source_audio_path": str(source), "target_synth": str(record.target_synth),
                "budget": str(record.budget),
                "start_offset_s": float(source_meta.get("start_offset_s", 0.0) or 0.0),
            },
        )
        self._save_incident_context = {
            "incident_id": incident.incident_id, "match_uid": match_uid, "trigger": "auto",
        }
        self.append_log(f"Saving preset for {record.source_name} (incident {incident.incident_id})…")
        self.statusBar().showMessage("Saving your preset…")
        try:
            self.export_runner.start(
                result_path, output, incident_id=incident.incident_id, trigger="auto",
                attempts=MAX_AUTOMATIC_ATTEMPTS,
            )
        except Exception as exc:
            self._save_incident_context = None
            self.append_log(f"Could not start saving the preset: {exc}")
        self.refresh_match_library()
        self._refresh_save_state()

    def retry_saving_preset(self, match_uid: str | None = None) -> None:
        """Save the SAME generated result again; the Match is not re-run."""

        from core.preset_save import incident_for_match

        match_uid = match_uid or self._current_match_uid
        if not match_uid:
            return
        if self._batch_state is not None:
            QMessageBox.information(
                self, "Batch is running", "Try saving this preset again after the batch finishes."
            )
            return
        if self.export_runner.running:
            QMessageBox.information(
                self, "Still saving", "PatchLab is saving another preset. Try again in a moment."
            )
            return
        incident = incident_for_match(match_uid)
        if incident is None or incident.status != "failed":
            self._refresh_save_state()
            return
        # The incident recorded the exact archived result it was saving; that
        # is what is retried (the Match itself is never re-run here).
        result_path = Path(incident.result_path)
        if not result_path.is_file():
            QMessageBox.information(
                self,
                "Result not found",
                "PatchLab can no longer find this result on your Mac, so it can't "
                "save it again. Run this sound again to create a new preset.",
            )
            return
        self._save_incident_context = {
            "incident_id": incident.incident_id,
            "match_uid": match_uid,
            "trigger": "manual",
        }
        self.append_log(f"Retrying the save of this preset (incident {incident.incident_id})…")
        self.statusBar().showMessage("Saving your preset again…")
        try:
            self.export_runner.start(
                result_path, Path(incident.target_path),
                incident_id=incident.incident_id, trigger="manual", attempts=1,
            )
        except Exception as exc:
            self._save_incident_context = None
            self.append_log(f"Could not start saving the preset again: {exc}")
        self._refresh_save_state()
        self.refresh_match_library()

    def run_again_from_incident(self, match_uid: str | None = None) -> None:
        """Run the same sound again with the same settings, once, on request."""

        from core.preset_save import incident_for_match

        match_uid = match_uid or self._current_match_uid
        incident = incident_for_match(match_uid) if match_uid else None
        if incident is None or not incident.run_again:
            return
        if self.match_runner.running or self._batch_state is not None:
            QMessageBox.information(
                self, "PatchLab is busy", "Run this sound again once the current work finishes."
            )
            return
        config = incident.run_again
        source = Path(str(config.get("source_audio_path", "")))
        if not source.is_file():
            QMessageBox.information(
                self,
                "Original sound not found",
                "PatchLab no longer has the original sound for this result. "
                "Drop the sound in again to create a new preset.",
            )
            return
        self._set_match_file(str(source))
        self._select_segment(self.match_synth, str(config.get("target_synth", "")))
        self._select_segment(self.match_budget, str(config.get("budget", "")))
        try:
            self.match_offset.setValue(float(config.get("start_offset_s", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
        self.nav_tabs.setCurrentIndex(0)
        self.append_log(
            f"Running {source.name} again ({config.get('target_synth')}, {config.get('budget')}) "
            f"because its preset could not be saved (incident {incident.incident_id})."
        )
        self.start_match()

    @staticmethod
    def _select_segment(control, data: str) -> None:
        for index in range(control.count()):
            if str(control.itemData(index)) == data:
                control.setCurrentIndex(index)
                return

    def _reconcile_save_incidents(self) -> None:
        """At launch: finish interrupted saves and resend unsent incident reports."""

        try:
            from core.preset_save import prune_saved_incidents, reconcile_incidents, unfinished_incidents

            if self._save_incident_context is None and not self.export_runner.running:
                counts = reconcile_incidents(Database(self._match_database_path()))
                if counts["recorded"]:
                    self.append_log(
                        f"Recorded {counts['recorded']} saved preset(s) the Library had missed."
                    )
                if counts["marked_failed"]:
                    self.append_log(
                        f"{counts['marked_failed']} preset save(s) were interrupted; "
                        "they can be retried from the Library."
                    )
            prune_saved_incidents()
            for incident in unfinished_incidents():
                for kind in ("report", "followup"):
                    entry = getattr(incident, kind) or {}
                    request = entry.get("request_path")
                    if entry.get("status") == "saved_locally" and request and Path(request).is_file():
                        self._queue_incident_report(incident.incident_id, kind, Path(request))
        except Exception as exc:
            self.append_log(f"Save incidents could not be checked: {exc}")
        self.refresh_match_library()
        self._refresh_save_state()

    # ---- automatic diagnostic reports --------------------------------

    def _submit_save_incident_report(self, incident, *, followup: bool) -> None:
        """One automatic report per save incident, plus at most one follow-up.

        The first report reuses the incident id as its ticket id. A failed
        manual retry adds one linked follow-up ticket instead of touching the
        already-uploaded (immutable) first one. Identical failures within one
        session share a report rather than uploading duplicates.
        """

        kind = "followup" if followup else "report"
        if getattr(incident, kind):
            return
        last = incident.attempts[-1] if incident.attempts else {}
        signature = (
            kind,
            str(last.get("failure_kind", "")),
            str(last.get("failure_reason", "")),
            str(last.get("exception_type", "")),
            str(last.get("stage", "")),
        )
        existing = self._reported_save_signatures.get(signature)
        if existing and not followup:
            incident.report = {"ticket_id": existing, "status": "same_as_earlier_report"}
            incident.save_reports()
            self.append_log(
                f"Save incident {incident.incident_id} matches report {existing}; not sending a duplicate."
            )
            return
        try:
            import uuid as _uuid

            from core.bug_report import create_request
            from core.support_bundle import create_support_bundle

            ticket_id = _uuid.uuid4().hex if followup else incident.incident_id
            original = (incident.report or {}).get("ticket_id", "")
            if followup:
                comments = (
                    "AUTOMATIC FOLLOW-UP REPORT: manual recovery also failed.\n"
                    f"Original ticket: {original or 'not sent'}\n"
                    f"Save incident: {incident.incident_id}\n"
                    "The user pressed Retry Saving Preset and the save failed again."
                )
            else:
                comments = (
                    "AUTOMATIC REPORT: PatchLab could not save a generated preset after "
                    "its automatic retries.\n"
                    f"Save incident: {incident.incident_id}\n"
                    "No audio or preset content is included."
                )
            logs = (
                "=== SAVE INCIDENT ===\n"
                + json.dumps(incident.diagnostic_summary(), indent=2, sort_keys=True, default=str)
                + "\n\n"
                + self._diagnostic_log_text()
            )
            request_path = create_request(comments=comments, logs=logs, ticket_id=ticket_id)
            try:
                create_support_bundle(
                    ticket_id=ticket_id,
                    comments=comments,
                    operation="saving the generated preset",
                    settings=self._diagnostic_settings(),
                    ticket_path=request_path,
                )
            except Exception as exc:
                self.append_log(f"Save-incident diagnostic bundle could not be written: {exc}")
            entry = {
                "ticket_id": ticket_id,
                "status": "saved_locally",
                "request_path": str(request_path),
                "original_ticket_id": original if followup else "",
            }
            setattr(incident, kind, entry)
            incident.save_reports()
            if not followup:
                self._reported_save_signatures[signature] = ticket_id
            self.append_log(f"Saved an automatic report for this save failure: ticket {ticket_id}")
            self._queue_incident_report(incident.incident_id, kind, request_path)
        except Exception as exc:
            # Reporting must never interfere with the save or the UI.
            self.append_log(f"Automatic save-failure report could not be created: {exc}")

    def _queue_incident_report(self, incident_id: str, kind: str, request_path: Path) -> None:
        item = (incident_id, kind, Path(request_path))
        if item == self._incident_report_active or item in self._incident_report_queue:
            return
        self._incident_report_queue.append(item)
        self._start_next_incident_report()

    def _start_next_incident_report(self) -> None:
        if self._incident_report_active is not None or self.incident_report_runner.running:
            return
        while self._incident_report_queue:
            item = self._incident_report_queue.pop(0)
            try:
                self.incident_report_runner.start(item[2])
            except Exception as exc:
                self.append_log(f"Automatic report could not be sent now: {exc}")
                continue
            self._incident_report_active = item
            return

    def _update_incident_report(self, status: str, detail: dict) -> None:
        item = self._incident_report_active
        self._incident_report_active = None
        if item is None:
            return
        try:
            from core.preset_save import SaveIncident

            incident = SaveIncident.load(item[0])
            entry = dict(getattr(incident, item[1]) or {})
            entry.update({"status": status, **detail})
            setattr(incident, item[1], entry)
            incident.save_reports()
        except Exception as exc:
            self.append_log(f"Could not record the automatic report's status: {exc}")
        QTimer.singleShot(0, self._start_next_incident_report)

    def _incident_report_completed(self, result: dict) -> None:
        self.append_log(
            f"Automatic save-failure report sent: ticket {result.get('ticket_id', '')}, "
            f"receipt {result.get('receipt_id', '')}"
        )
        self._update_incident_report("uploaded", {"receipt_id": str(result.get("receipt_id", ""))})

    def _incident_report_failed(self, error: str) -> None:
        # Saved locally already; it is resent at the next launch. Never a dialog.
        code = getattr(self.incident_report_runner, "error_code", "") or "unknown"
        self.append_log(f"Automatic save-failure report kept on this Mac for now ({code}).")
        self._update_incident_report("saved_locally", {"last_upload_error": code})

    def _preview_completed(self, path: str) -> None:
        super()._preview_completed(path)

    def _preview_failed(self, error: str) -> None:
        super()._preview_failed(error)

    def start_batch_folder(self) -> None:
        if self._batch_state is not None:
            QMessageBox.information(self, "Batch already running", "Only one batch can run at a time.")
            return
        # Same fresh capability gate as a single Match, and before the user is
        # asked to choose a folder and a name for a batch that cannot be produced.
        if not self._check_output_capability(str(self.match_synth.currentData())):
            return
        selected = QFileDialog.getExistingDirectory(
            self, "Choose a folder of sounds", str(Path.home())
        )
        if not selected:
            return
        raw_name, accepted = QInputDialog.getText(
            self, "Name the preset folder", "Preset folder name:"
        )
        if not accepted:
            return
        folder_name = sanitize_folder_name(raw_name)
        if not folder_name:
            QMessageBox.warning(self, "Invalid folder name", "Enter a non-empty, filesystem-safe folder name.")
            return
        source_folder = Path(selected).resolve()
        discovery = discover_batch_audio(
            source_folder, recursive=self.batch_recursive.isChecked()
        )
        if not discovery.supported:
            QMessageBox.information(
                self, "No supported audio", f"No WAV, AIFF, FLAC, MP3, or OGG files were found. {discovery.unsupported_count} unsupported files were skipped."
            )
            return
        target_synth = str(self.match_synth.currentData())
        budget = str(self.match_budget.currentData())
        export_folder = self._patchlab_export_folder(target_synth) / folder_name
        if export_folder.exists():
            choice = QMessageBox.question(
                self,
                "Folder already exists",
                f"{export_folder} already exists. Add new presets without overwriting existing files?",
            )
            if choice != QMessageBox.StandardButton.Yes:
                return
        per_file_minutes = {"quick": 1.0, "balanced": 3.0, "best": 8.0}[budget]
        confirmation = QMessageBox.question(
            self,
            "Start batch?",
            f"Supported audio: {len(discovery.supported)}\n"
            f"Unsupported/skipped: {discovery.unsupported_count}\n"
            f"Quality: {budget.title()}\n"
            f"Destination: {export_folder}\n"
            f"Rough estimate: {len(discovery.supported) * per_file_minutes:.0f} minutes",
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return
        export_folder.mkdir(parents=True, exist_ok=True)
        database = Database(self._match_database_path())
        batch = database.find_match_batch(
            source_folder=source_folder,
            export_folder=export_folder,
            target_synth=target_synth,
            budget=budget,
        )
        if batch is None:
            batch_id = database.create_match_batch(
                folder_name=folder_name,
                source_folder=source_folder,
                export_folder=export_folder,
                target_synth=target_synth,
                budget=budget,
                total_files=len(discovery.supported),
            )
            completed_hashes: set[str] = set()
            prior_failed = 0
        else:
            batch_id = batch.id
            completed_hashes = database.batch_completed_hashes(batch.id)
            prior_failed = batch.failed_files
        pending, skipped = resumable_batch_files(
            list(discovery.supported), completed_hashes
        )
        self._batch_state = {
            "batch_id": batch_id,
            "folder_name": folder_name,
            "source_folder": source_folder,
            "export_folder": export_folder,
            "target_synth": target_synth,
            "budget": budget,
            "files": pending,
            "index": 0,
            "total": len(discovery.supported),
            "completed": len(completed_hashes),
            "failed": prior_failed,
            "skipped": skipped + discovery.unsupported_count,
            "unsupported": discovery.unsupported_count,
            "started": time.monotonic(),
            "cancel_requested": False,
            "phase": "idle",
        }
        database.update_match_batch(
            batch_id,
            completed_files=len(completed_hashes),
            failed_files=prior_failed,
            total_files=len(discovery.supported),
            status="running",
        )
        self.batch_button.setEnabled(False)
        self.save_preset_button.setEnabled(False)
        self.open_preset_location_button.setEnabled(False)
        self.library_batch_cancel.setEnabled(True)
        self.nav_tabs.setCurrentIndex(1)
        self.append_log(
            f"Batch started: {len(pending)} pending, {skipped} already completed by content hash, {discovery.unsupported_count} unsupported"
        )
        self._start_next_batch_file()

    def _start_next_batch_file(self) -> None:
        state = self._batch_state
        if state is None:
            return
        if state["cancel_requested"] or state["index"] >= len(state["files"]):
            self._finish_match_batch(
                "cancelled" if state["cancel_requested"] else "complete"
            )
            return
        path, digest = state["files"][state["index"]]
        state["index"] += 1
        state["current_path"] = path
        state["current_hash"] = digest
        state["current_uid"] = None
        state["phase"] = "match"
        current_number = state["completed"] + state["failed"] + 1
        elapsed = time.monotonic() - state["started"]
        done_this_run = max(state["index"] - 1, 0)
        remaining = (
            elapsed / done_this_run * (len(state["files"]) - done_this_run)
            if done_this_run else 0
        )
        self.library_batch_status.setText(
            f"{path.name} · {current_number}/{state['total']} · ETA {remaining / 60:.1f}m"
        )
        self.statusBar().showMessage(
            f"Batch matching {path.name} ({current_number}/{state['total']})"
        )
        self.append_log(f"Batch matching: {path.name}")
        # Batch runs take the same synthesis-or-fallback decision as a single
        # match so a folder is never processed at lower quality than the
        # one-off path would give it.
        batch_factory_only = self.distribution_mode and not synthesis_readiness(
            str(state["target_synth"])
        ).available
        self.match_runner.start(
            path,
            target_synth=state["target_synth"],
            budget=state["budget"],
            offset=0.0,
            session_root=Path(self._match_session.name),
            factory_only=batch_factory_only,
            factory_mapping=self.factory_mapping_path if batch_factory_only else None,
            local_db=(
                self.local_paths["db"]
                if batch_factory_only
                and bool(self.privacy_choice.use_and_share_own_presets)
                else None
            ),
            local_audio_root=(
                self.local_paths["audio"]
                if batch_factory_only
                and bool(self.privacy_choice.use_and_share_own_presets)
                else None
            ),
        )

    def _batch_file_completed(self) -> None:
        if self._batch_state is None:
            return
        self._batch_state["completed"] += 1
        self._batch_state["phase"] = "idle"
        self._persist_batch_progress("running")
        self.refresh_match_library()
        QTimer.singleShot(0, self._start_next_batch_file)

    def _batch_file_failed(self, error: str) -> None:
        if self._batch_state is None:
            return
        path = self._batch_state.get("current_path")
        self._batch_state["failed"] += 1
        self._batch_state["phase"] = "idle"
        self.append_log(f"Batch file failed; continuing: {Path(path).name if path else 'unknown'}: {error}")
        self._persist_batch_progress("running")
        QTimer.singleShot(0, self._start_next_batch_file)

    def _persist_batch_progress(self, status: str) -> None:
        state = self._batch_state
        if state is None:
            return
        Database(self._match_database_path()).update_match_batch(
            state["batch_id"],
            completed_files=state["completed"],
            failed_files=state["failed"],
            total_files=state["total"],
            status=status,
        )

    def cancel_match_batch(self) -> None:
        if self._batch_state is None:
            return
        self._batch_state["cancel_requested"] = True
        self.library_batch_cancel.setEnabled(False)
        self.library_batch_status.setText("Cancelling after the current file…")
        self.append_log("Batch cancellation requested; the in-flight file will finish safely")

    def _finish_match_batch(self, status: str) -> None:
        state = self._batch_state
        if state is None:
            return
        self._persist_batch_progress(status)
        elapsed = time.monotonic() - state["started"]
        summary = (
            f"Batch {status}: {state['completed']} completed, "
            f"{state['failed']} failed, {state['skipped']} skipped · {elapsed:.1f}s"
        )
        self.append_log(summary)
        self.statusBar().showMessage(summary)
        self.library_batch_status.setText(summary)
        self.library_batch_cancel.setEnabled(False)
        self.batch_button.setEnabled(True)
        if isinstance(self._match_result, dict) and isinstance(
            self._match_result.get("recommendation"), dict
        ):
            self.save_preset_button.setEnabled(True)
            self.open_preset_location_button.setEnabled(True)
        self._batch_state = None
        self.refresh_match_library()
        self.nav_tabs.setCurrentIndex(1)
        self._refresh_save_state()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.storage_runner.running:
            QMessageBox.information(
                self,
                "Storage move is still running",
                "Keep PatchLab open until the verified storage operation finishes. "
                "This prevents an interrupted move from confusing the current session.",
            )
            event.ignore()
            return
        if self._batch_state is not None:
            self._persist_batch_progress("cancelled")
            self.append_log("Batch marked cancelled because PatchLab is closing")
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self._native_aspect_installed:
            self._last_resize_size = event.size()
            return
        if (
            self._aspect_guard
            or self.isMaximized()
            or self.isFullScreen()
            or not self.isVisible()
        ):
            self._last_resize_size = event.size()
            return
        size = event.size()
        ratio = size.width() / max(size.height(), 1)
        if abs(ratio - self.ASPECT_RATIO) < 0.008:
            self._last_resize_size = size
            return
        previous = self._last_resize_size
        width_driven = (
            previous is None
            or abs(size.width() - previous.width())
            >= abs(size.height() - previous.height())
        )
        if width_driven:
            width = max(size.width(), self.minimumWidth())
            height = max(round(width / self.ASPECT_RATIO), self.minimumHeight())
        else:
            height = max(size.height(), self.minimumHeight())
            width = max(round(height * self.ASPECT_RATIO), self.minimumWidth())
        self._aspect_guard = True
        try:
            self.resize(width, height)
            self._last_resize_size = self.size()
        finally:
            self._aspect_guard = False


    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if ENV.branch == "macos" and not self._native_aspect_installed:
            QTimer.singleShot(0, self._install_native_aspect_ratio)

    def _install_native_aspect_ratio(self) -> None:
        self._native_aspect_installed = enforce_native_aspect_ratio(
            self,
            16.0,
            9.0,
        )
        if self._native_aspect_installed:
            self.append_log(
                "Native macOS live-resize constraint active · 16:9"
            )

    def open_settings(self) -> None:
        from app.license_dialog import LicenseAgreementDialog

        dialog = QDialog(self)
        dialog.setWindowTitle("PatchLab Settings")
        dialog.setMinimumWidth(430)
        layout = QVBoxLayout(dialog)
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 20px; font-weight: 750;")
        layout.addWidget(title)
        storage_card = QGroupBox("Audio Storage")
        storage_layout = QVBoxLayout(storage_card)
        current_storage = storage_status()
        location = QLabel(str(current_storage.root))
        location.setWordWrap(True)
        location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        location.setObjectName("muted")
        storage_layout.addWidget(QLabel("Full library render location"))
        storage_layout.addWidget(location)
        availability = QLabel(
            "Available"
            if current_storage.available
            else f"Disconnected · {current_storage.reason}"
        )
        availability.setObjectName("muted")
        storage_layout.addWidget(availability)
        if current_storage.available:
            try:
                free_bytes = shutil.disk_usage(current_storage.root).free
                used_bytes = audio_root_size(current_storage.root)
                usage = QLabel(
                    f"{used_bytes / 1e9:.1f} GB used here · "
                    f"{free_bytes / 1e9:.1f} GB free on this drive"
                )
            except OSError:
                usage = QLabel("Size unavailable")
            usage.setObjectName("muted")
            storage_layout.addWidget(usage)
        button_row = QHBoxLayout()
        choose_storage = QPushButton("Choose Audio Storage Folder…")
        choose_storage.setObjectName("compactActionButton")
        button_row.addWidget(choose_storage)
        open_folder = QPushButton("Open Audio Library Folder")
        open_folder.setObjectName("compactActionButton")
        open_folder.setEnabled(current_storage.available)
        open_folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(current_storage.root)))
        )
        button_row.addWidget(open_folder)
        storage_layout.addLayout(button_row)

        compact = QCheckBox("Compact storage after learning presets")
        compact.setChecked(self.storage_preferences.compact_mode)
        compact.setToolTip(
            "Keeps fingerprints and settings, then removes large renders that "
            "PatchLab can regenerate on demand."
        )
        storage_layout.addWidget(compact)
        cache_row = QHBoxLayout()
        cache_row.addWidget(QLabel("Preview cache limit"))
        cache_limit = QSpinBox()
        cache_limit.setRange(128, 4096)
        cache_limit.setSingleStep(128)
        cache_limit.setSuffix(" MB")
        cache_limit.setValue(self.storage_preferences.preview_cache_mb)
        cache_row.addWidget(cache_limit)
        cache_row.addStretch(1)
        storage_layout.addLayout(cache_row)
        cache_usage = QLabel(
            f"Preview cache: {preview_cache_usage(self._preview_cache_root()) / (1024 ** 2):.1f} MB"
        )
        cache_usage.setObjectName("muted")
        storage_layout.addWidget(cache_usage)
        clear_cache = QPushButton("Clear Preview Cache")
        clear_cache.setObjectName("compactActionButton")
        storage_layout.addWidget(clear_cache)
        free_space = QPushButton("Free Space Now")
        free_space.setObjectName("compactActionButton")
        storage_layout.addWidget(free_space)
        storage_note = QLabel(
            "Compact mode keeps the small database, fingerprints, settings, and "
            "a capped audition cache. Full seven-octave WAVs are removed only "
            "after learning finishes and can be regenerated later."
        )
        storage_note.setObjectName("muted")
        storage_note.setWordWrap(True)
        storage_layout.addWidget(storage_note)
        layout.addWidget(storage_card)

        from core.preset_output import (
            PresetOutputPreferences,
            configured_preset_output_folder,
            ensure_writable,
            save_preset_output_preferences,
        )

        preset_folder_card = QGroupBox("Generated Preset Folder")
        preset_folder_layout = QVBoxLayout(preset_folder_card)
        preset_folder_layout.addWidget(
            QLabel("Where PatchLab saves the presets it creates from a Match")
        )
        # A custom folder applies to both Serum generations; either resolves
        # to the same path once one is configured, so serum2 stands in for
        # "the configured folder" here.
        current_preset_folder = configured_preset_output_folder("serum2")
        preset_folder_location = QLabel(str(current_preset_folder))
        preset_folder_location.setWordWrap(True)
        preset_folder_location.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        preset_folder_location.setObjectName("muted")
        preset_folder_layout.addWidget(preset_folder_location)
        preset_folder_buttons = QHBoxLayout()
        change_preset_folder = QPushButton("Change Folder…")
        change_preset_folder.setObjectName("compactActionButton")
        preset_folder_buttons.addWidget(change_preset_folder)
        open_preset_folder = QPushButton("Open Folder")
        open_preset_folder.setObjectName("compactActionButton")
        open_preset_folder.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(configured_preset_output_folder("serum2")))
            )
        )
        preset_folder_buttons.addWidget(open_preset_folder)
        preset_folder_layout.addLayout(preset_folder_buttons)
        preset_folder_note = QLabel(
            "Changing this only affects presets PatchLab generates from now on — "
            "results already saved stay exactly where they are."
        )
        preset_folder_note.setObjectName("muted")
        preset_folder_note.setWordWrap(True)
        preset_folder_layout.addWidget(preset_folder_note)

        def change_generated_preset_folder() -> None:
            selected = QFileDialog.getExistingDirectory(
                dialog,
                "Choose Generated Preset Folder",
                str(current_preset_folder),
            )
            if not selected:
                return
            destination = Path(selected).expanduser().resolve()
            reason = ensure_writable(destination)
            if reason:
                QMessageBox.critical(
                    dialog,
                    "Folder is not writable",
                    "PatchLab can't save presets in this folder. Choose another folder.",
                )
                return
            save_preset_output_preferences(PresetOutputPreferences(folder=str(destination)))
            preset_folder_location.setText(str(destination))
            self.append_log(f"Generated preset folder set to {destination}")

        change_preset_folder.clicked.connect(change_generated_preset_folder)
        layout.addWidget(preset_folder_card)

        auto_update_toggle: QCheckBox | None = None
        if self.distribution_mode:
            update_card = QGroupBox("Updates")
            update_layout = QVBoxLayout(update_card)
            auto_update_toggle = QCheckBox("Automatically check for updates")
            auto_update_toggle.setChecked(load_update_preferences().auto_check)
            auto_update_toggle.setToolTip(
                "Checks GitHub for a newer PatchLab version each time the app "
                "opens. Updates replace app code only -- your linked presets, "
                "rendered audio, and everything already learned are never touched."
            )
            update_layout.addWidget(auto_update_toggle)

            def check_for_update_now() -> None:
                if self.update_check_runner.running:
                    QMessageBox.information(
                        dialog, "Already checking", "An update check is already running."
                    )
                    return
                self.append_log("Checking for a PatchLab update…")
                # A genuinely new request from the user: any earlier sign-in
                # attempt (this session, an earlier click) no longer applies.
                self._update_check_auth_retry_attempted = False
                self._start_update_check(manual=True)

            check_now = QPushButton("Check for Updates Now")
            check_now.setObjectName("compactActionButton")
            check_now.clicked.connect(check_for_update_now)
            update_layout.addWidget(check_now)
            update_version = QLabel(f"You have v{__version__}")
            update_version.setObjectName("muted")
            update_layout.addWidget(update_version)
            layout.addWidget(update_card)

        def persist_storage_controls() -> None:
            self.storage_preferences = StoragePreferences(
                self.storage_preferences.audio_root,
                compact.isChecked(),
                cache_limit.value(),
            )
            save_storage_preferences(self.storage_preferences)

        def choose_storage_location() -> None:
            if self.storage_runner.running:
                QMessageBox.information(
                    dialog,
                    "Storage job running",
                    "Wait for the current storage job to finish before changing folders.",
                )
                return
            selected = QFileDialog.getExistingDirectory(
                dialog,
                "Choose PatchLab Audio Storage",
                str(current_storage.root.parent),
            )
            if not selected:
                return
            destination = Path(selected).expanduser().resolve()
            try:
                prepare_audio_root(destination)
            except OSError as exc:
                QMessageBox.critical(dialog, "Folder is not writable", str(exc))
                return
            persist_storage_controls()
            source = Path(self.local_paths["audio"]).expanduser().resolve()
            if source == destination:
                return
            render_rows = 0
            if Path(self.local_paths["db"]).is_file():
                import sqlite3

                connection = sqlite3.connect(self.local_paths["db"])
                render_rows = int(
                    connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0]
                )
                connection.close()
            if render_rows:
                answer = QMessageBox.question(
                    dialog,
                    "Move existing renders?",
                    f"PatchLab found {render_rows:,} existing render records. Move "
                    "them to the selected folder now? Files are checksum-verified "
                    "before the originals are removed.",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Yes,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                self._storage_pending = "migrate"
                dialog.accept()
                self.storage_runner.start(
                    [
                        "migrate",
                        str(source),
                        str(destination),
                        "--db",
                        str(self.local_paths["db"]),
                    ]
                )
                self.append_log(
                    f"Moving full render storage to {destination}; originals are "
                    "kept until each copy is verified"
                )
                self.statusBar().showMessage("Moving PatchLab audio storage…")
                return
            self.storage_preferences = StoragePreferences(
                str(destination), compact.isChecked(), cache_limit.value()
            )
            save_storage_preferences(self.storage_preferences)
            self.local_paths = default_local_paths()
            location.setText(str(destination))
            availability.setText("Available")
            self.append_log(f"Audio storage location set to {destination}")
            self._refresh_workflow_cards()

        def free_space_now() -> None:
            if self.storage_runner.running:
                QMessageBox.information(
                    dialog,
                    "Storage job running",
                    "Wait for the current storage job to finish.",
                )
                return
            answer = QMessageBox.question(
                dialog,
                "Remove regenerable render files?",
                "This keeps every learned fingerprint and preset setting, but "
                "removes completed full-library WAV files. Auditions are rendered "
                "again only when needed.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            persist_storage_controls()
            self._storage_pending = "compact"
            dialog.accept()
            self.storage_runner.start(
                [
                    "compact",
                    "--db",
                    str(self.local_paths["db"]),
                    "--audio-root",
                    str(self.local_paths["audio"]),
                ]
            )
            self.statusBar().showMessage("Freeing PatchLab render space…")

        def clear_cache_now() -> None:
            summary = clear_preview_cache(self._preview_cache_root())
            cache_usage.setText("Preview cache: 0.0 MB")
            message = f"Cleared {summary.files:,} preview file(s)"
            self.append_log(message)
            self.statusBar().showMessage(message)

        choose_storage.clicked.connect(choose_storage_location)
        free_space.clicked.connect(free_space_now)
        clear_cache.clicked.connect(clear_cache_now)
        if self.distribution_mode:
            toggle = QCheckBox("Use && share my own presets")
            toggle.setChecked(self.share_toggle.isChecked())
            toggle.toggled.connect(self.share_toggle.setChecked)
            layout.addWidget(toggle)
            forget = QPushButton("Sign out / forget passcode")
            forget.setObjectName("compactActionButton")
            forget.clicked.connect(self._forget_passcode)
            layout.addWidget(forget)
            view_license = QPushButton("View License Agreement")
            view_license.setObjectName("compactActionButton")
            view_license.clicked.connect(
                lambda: LicenseAgreementDialog(
                    parent=dialog, read_only_view=True
                ).exec()
            )
            layout.addWidget(view_license)
            detail = QLabel(
                "When enabled, linked presets are processed locally and preset "
                "files plus fingerprints may be contributed. Audio is never uploaded."
            )
        else:
            detail = QLabel(
                "Developer mode is active. Distribution consent controls are not applied."
            )
        detail.setObjectName("muted")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        close = QPushButton("Done")
        close.setObjectName("primaryButton")
        def save_and_close() -> None:
            persist_storage_controls()
            if auto_update_toggle is not None:
                save_update_preferences(
                    UpdatePreferences(
                        auto_update_toggle.isChecked(),
                        load_update_preferences().skipped_version,
                    )
                )
            dialog.accept()

        close.clicked.connect(save_and_close)
        layout.addWidget(close)
        dialog.exec()

    def _storage_progress_changed(self, detail: dict) -> None:
        current = int(detail.get("current", 0))
        total = int(detail.get("total", 0))
        self.statusBar().showMessage(
            f"Moving audio storage: {current:,} of {total:,} files verified…"
        )

    def _storage_completed(self, summary: dict) -> None:
        operation = self._storage_pending or "storage"
        self._storage_pending = None
        self.storage_preferences = load_storage_preferences()
        self.local_paths = default_local_paths()
        files = int(summary.get("files", 0))
        bytes_freed = int(summary.get("bytes", 0))
        if operation == "migrate":
            message = (
                f"Audio storage move complete: {files:,} files, "
                f"{bytes_freed / (1024 ** 3):.2f} GiB verified"
            )
        else:
            message = (
                f"Compact cleanup complete: {files:,} render files removed, "
                f"{bytes_freed / (1024 ** 3):.2f} GiB freed"
            )
        self.append_log(message)
        self.statusBar().showMessage(message)
        self._prune_preview_cache()
        self._refresh_workflow_cards()

    def _storage_failed(self, error: str) -> None:
        operation = self._storage_pending or "storage operation"
        self._storage_pending = None
        self.append_log(f"{operation.title()} failed safely: {error}")
        self.statusBar().showMessage(f"Storage unchanged: {error}")
        QMessageBox.warning(
            self,
            "Storage was not changed",
            "PatchLab kept the existing location and original files. You can "
            f"retry after correcting this problem:\n\n{error}",
        )

    def _prune_preview_cache(self) -> None:
        try:
            summary = prune_preview_cache(
                self._preview_cache_root(),
                self.storage_preferences.preview_cache_mb,
            )
        except OSError as exc:
            self.append_log(f"Preview cache cleanup deferred: {exc}")
            return
        if summary.files:
            self.append_log(
                f"Preview cache limit removed {summary.files:,} old file(s) "
                f"({summary.bytes / (1024 ** 2):.1f} MB)"
            )

    def _forget_passcode(self) -> None:
        from core.access_gate import AccessStore

        AccessStore().clear()
        QMessageBox.information(
            self,
            "Signed out",
            "The saved PatchLab passcode was removed. Your license acceptance and "
            "preset-sharing choice were not changed. The passcode will be requested "
            "the next time PatchLab starts.",
        )

    def open_help(self) -> None:
        build = current_build_info()
        dialog = QDialog(self)
        dialog.setWindowTitle("PatchLab Help")
        dialog.setMinimumWidth(470)
        layout = QVBoxLayout(dialog)
        heading = QLabel("PatchLab Help")
        heading.setStyleSheet("font-size: 20px; font-weight: 750;")
        body = QLabel(
            "Match a sound immediately with PatchLab’s included factory library and "
            "trained synthesis model. Linking your own presets is optional and only "
            "adds them to your personal closest-match results.\n\n"
            f"Build: {build.short_commit} · {build.built_at_utc}\n\n"
            "If something goes wrong, send a report with the detail of what you "
            "were doing."
        )
        body.setWordWrap(True)
        report = QPushButton("Report a Problem…")
        report.setObjectName("compactActionButton")
        report.clicked.connect(lambda: (dialog.accept(), self.open_bug_report()))
        close = QPushButton("Done")
        close.setObjectName("primaryButton")
        close.clicked.connect(dialog.accept)
        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(report)
        layout.addWidget(close)
        dialog.exec()

    def _ui_event(self, event_type: str, message: str = "", **fields) -> None:
        """Record one GUI-level decision as a structured flight-recorder event.

        Structured fields carry the *reason*; the visible popup text is not
        duplicated here. Never raises: diagnostics must not affect the UI.
        """

        try:
            from core.diagnostics import record

            record("gui", event_type, message, **fields)
        except Exception:
            pass

    def _card_phases(self) -> dict[str, str]:
        """Current phase of the four workflow cards, for recovery diagnostics.

        Reads the phases the most recent refresh already computed. Re-resolving
        the whole workflow state here cost ~90 ms of GUI-thread work on every
        recorded click, for a value that is already in hand.
        """

        return dict(getattr(self, "_last_card_phases", None) or {})

    def _check_output_capability(
        self, generation: str, *, matching_now: bool = False
    ) -> bool:
        """Fresh capability gate for a requested output generation.

        Returns True when the operation may start. On failure the user gets one
        concise sentence and nothing is started -- an impossible operation must
        not be launched only to fail later.

        Deliberately cheap (path stats, no plug-in is opened), so a user who
        installs Serum while PatchLab is running is picked up without a restart.
        """

        from core.capability_ux import evaluate_output_target

        try:
            decision, snapshot = evaluate_output_target(
                generation,
                previous=getattr(self, "_capability_snapshot", None),
            )
        except Exception as exc:
            # A capability-check failure must never block the user: fall through
            # and let the operation report its own error.
            self.append_log(f"Capability check unavailable: {exc}")
            return True
        previous = getattr(self, "_capability_snapshot", None)
        self._capability_snapshot = snapshot
        name = "Serum 2" if generation == "serum2" else "Serum 1"
        if not decision.allowed:
            QMessageBox.information(self, f"{name} is required", decision.message)
            self.append_log(f"Match not started: {decision.message}")
            self.statusBar().showMessage(decision.message)
            return False
        if decision.became_available:
            self.append_log(f"{name} became available since the last check.")
        # Offer any pending work for this generation. This is decided by the
        # catalogued pending rows, not by having watched the synth disappear
        # earlier in this session: a synth installed between launches must be
        # offered exactly like one installed while the app was open.
        self._offer_pending_processing(generation, matching_now=matching_now)
        if previous is None:
            self._maybe_show_missing_synth_notice(snapshot)
        return True

    def _pending_counts(self) -> dict:
        try:
            from core.db import Database

            database_path = self.local_paths["db"]
            if not Path(database_path).is_file():
                return {}
            return Database(database_path).pending_counts()
        except Exception:
            return {}

    def _maybe_show_missing_synth_notice(self, snapshot) -> None:
        """One informational message about presets needing an absent synth."""

        from core.capability_ux import acknowledge, missing_synth_notice

        # Notices about waiting personal presets are meaningless (and would nag)
        # while personal presets are off.
        if not user_presets_enabled():
            return
        pending = self._pending_counts()
        if not pending:
            return
        try:
            notice = missing_synth_notice(snapshot=snapshot, pending_counts=pending)
        except Exception:
            return
        if notice is None:
            return
        QMessageBox.information(self, notice.title, notice.body)
        acknowledge(notice, choice="seen")
        self.append_log(notice.body)

    def _offer_pending_processing(
        self, generation: str, *, matching_now: bool = False
    ) -> None:
        """Offer to process presets that were waiting for ``generation``.

        Choosing Not Now must leave everything else working: Match uses whatever
        is already learned, and the pending presets simply do not participate in
        retrieval until processed.
        """

        from core.capability_ux import acknowledge, synth_available_notice
        from core.db import Database
        from core.preset_identity import blocked_reasons_for

        if not user_presets_enabled():
            return

        try:
            database_path = self.local_paths["db"]
            if not Path(database_path).is_file():
                return
            waiting = Database(database_path).presets_needing_generation(
                generation, reasons=sorted(blocked_reasons_for(generation))
            )
        except Exception:
            return
        if not waiting:
            return
        try:
            notice = synth_available_notice(
                generation=generation, pending_count=len(waiting)
            )
        except Exception:
            return
        if notice is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle(notice.title)
        box.setText(notice.body)
        process_now = box.addButton("Process Now", QMessageBox.ButtonRole.AcceptRole)
        not_now = box.addButton("Not Now", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(not_now)
        box.exec()
        chose_process = box.clickedButton() is process_now
        acknowledge(notice, choice="process_now" if chose_process else "not_now")
        if not chose_process:
            self.append_log(
                f"{len(waiting):,} waiting preset(s) left unprocessed for now. "
                "Matching continues with the presets already learned."
            )
            return
        # Process only the presets waiting on this generation -- no rescan.
        if self.runner.running:
            self.append_log("Preset processing is already running.")
            return
        self._compact_render_active = True
        self._render_failure_detail = ""
        self._set_workflow_activity(
            "render", 0, len(waiting), f"Processing {len(waiting):,} waiting presets…"
        )
        self.append_log(
            f"Processing {len(waiting):,} waiting {generation} preset(s) that were "
            "already catalogued; no rescan is needed."
        )
        self._ui_event(
            "process_pending_started",
            "processing waiting presets for a newly available synth",
            generation=generation,
            waiting=len(waiting),
            workers=1 if matching_now else 4,
        )
        # One worker when a Match is starting in the same click: Match is the
        # foreground request and processing is resumable, so it yields.
        self.runner.start(
            pending_generation=generation, workers=1 if matching_now else 4
        )

    def _notify_pending_after_scan(self) -> None:
        """Show the one 'these presets need Serum X' message after processing.

        Uses a fresh capability check so a synth installed during the scan is
        reflected, and defers the anti-nag decision to core.capability_ux.
        """

        try:
            from core.synth_capability import refresh_capabilities

            snapshot = refresh_capabilities(reason="linked-folder processing finished")
            self._capability_snapshot = snapshot
            self._maybe_show_missing_synth_notice(snapshot)
        except Exception as exc:
            self.append_log(f"Could not summarise pending presets: {exc}")

    def _last_operation_name(self) -> str:
        """Best guess at what the user was doing, for the bundle summary."""

        if "match" in self._workflow_activities:
            return "match"
        if "render" in self._workflow_activities or getattr(
            self, "_compact_render_active", False
        ):
            return "render-sound-library"
        if "link" in self._workflow_activities:
            return "link-preset-folder"
        if "analyze" in self._workflow_activities:
            return "analyze-and-learn"
        return ""

    def _diagnostic_settings(self) -> dict:
        """The UI-side switches that affect which code path ran."""

        try:
            return {
                "target_synth": str(self.match_synth.currentData()),
                "quality_mode": str(self.match_budget.currentData()),
                "start_offset_s": float(self.match_offset.value()),
                "distribution_mode": bool(self.distribution_mode),
                "compact_mode": bool(
                    getattr(self, "storage_preferences", None)
                    and self.storage_preferences.compact_mode
                ),
                "sharing_enabled": bool(
                    self.privacy_choice.use_and_share_own_presets
                ),
                "active_activities": sorted(self._workflow_activities),
                "render_failure_detail": self._render_failure_detail,
                "model_asset_error": self._model_asset_error or "",
                # UI recovery state: which cards a reader should expect to see,
                # so a report answers "was earlier setup preserved?".
                "ui_recovery": {
                    card: {
                        "phase": state.phase,
                        "text": state.text,
                    }
                    for card, state in self._current_card_states().items()
                },
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _current_card_states(self) -> dict:
        """Resolve the four workflow cards without touching the widgets."""

        try:
            storage = storage_status()
            return resolve_workflow_state(
                privacy=self.privacy_choice,
                local_database_path=self.local_paths["db"],
                audio_selected=self._match_audio_path is not None,
                match_completed=self._workflow_last_match_complete,
                activities=self._workflow_activities,
                match_prerequisite_error=self._model_asset_error or "",
                compact_mode=bool(
                    getattr(self, "storage_preferences", None)
                    and self.storage_preferences.compact_mode
                ),
                audio_storage_error=storage.reason if not storage.available else "",
                render_failed_detail=(
                    self._render_failure_detail
                    if "render" not in self._workflow_activities
                    else ""
                ),
            ).as_dict()
        except Exception:
            return {}

    def _diagnostic_log_text(self) -> str:
        """Capture an actionable, consented snapshot without touching audio."""

        build = current_build_info()
        visible_log = self.log_pane.toPlainText() if hasattr(self, "log_pane") else ""
        from core.model_assets import resolve_model_assets
        from core.synthesis_assets import synthesis_readiness

        assets = resolve_model_assets()
        synth_readiness = {
            synth: synthesis_readiness(synth) for synth in ("serum1", "serum2")
        }
        storage = storage_status()
        try:
            usage = shutil.disk_usage(storage.root)
            storage_space = f"{usage.free} bytes free / {usage.total} bytes total"
        except OSError:
            storage_space = "unavailable"
        database_path = self.local_paths["db"]
        database_summary = "not present"
        if database_path.is_file():
            try:
                with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
                    presets = int(connection.execute("SELECT COUNT(*) FROM presets").fetchone()[0])
                    renders = int(connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0])
                database_summary = f"{presets} presets, {renders} renders"
            except (OSError, sqlite3.Error) as exc:
                database_summary = f"unreadable: {type(exc).__name__}: {exc}"
        plugin_summary = ", ".join(
            f"{item.synth}/{item.format}={'present' if item.exists else 'missing'}"
            for item in ENV.plugin_candidates
        )
        # The old flat present/missing list could not answer "which renderer did
        # PatchLab actually pick, and why did it reject the others?", which is
        # the question both reported tickets turned on.
        renderer_summary = "unavailable"
        try:
            from core.diagnostics import DIAGNOSTIC_SCHEMA_VERSION, recorder
            from core.renderer_selection import select_renderer

            lines = []
            for synth in ("serum1", "serum2"):
                selection = select_renderer(synth, log_decision=False)
                lines.append(
                    f"  {synth}: selected={selection.renderer or 'NONE'} "
                    f"reason={selection.reason}"
                )
                for item in selection.rejections():
                    lines.append(
                        f"    rejected {item.label}: {item.rejection_reason}"
                    )
            renderer_summary = "\n" + "\n".join(lines)
            schema_line = (
                f"Diagnostics: schema={DIAGNOSTIC_SCHEMA_VERSION}; "
                f"session={recorder().session_id}; "
                f"retention={recorder().retention_policy()['max_total_event_bytes']} bytes max\n"
            )
        except Exception as exc:
            schema_line = f"Diagnostics: unavailable ({type(exc).__name__}: {exc})\n"
        return (
            "PatchLab diagnostic report\n"
            + schema_line
            + f"Renderer selection:{renderer_summary}\n"
            f"Build: {json.dumps(build.as_dict(), sort_keys=True)}\n"
            f"Runtime: Python {sys.version.split()[0]} · {ENV.system_name} · {ENV.machine}\n"
            f"Executable: {sys.executable}\n"
            f"Platform: {ENV.branch}; compute={ENV.compute_backend}; warning={ENV.compute_warning or 'none'}\n"
            f"Synthesis: serum1={'ready' if synth_readiness['serum1'].available else synth_readiness['serum1'].reason}; "
            f"serum2={'ready' if synth_readiness['serum2'].available else synth_readiness['serum2'].reason}\n"
            f"Model checkpoint: {assets.checkpoint} ({assets.checkpoint.stat().st_size if assets.checkpoint.is_file() else 0} bytes)\n"
            f"Storage: root={storage.root}; available={storage.available}; configured_external={storage.configured_external}; "
            f"reason={storage.reason or 'none'}; space={storage_space}\n"
            f"Local library: {database_path}; {database_summary}\n"
            f"Privacy/link state: sharing={self.privacy_choice.use_and_share_own_presets}; "
            f"linked_folder={self.privacy_choice.linked_folder or 'none'}\n"
            f"Plugins: {plugin_summary}\n"
            f"UI state: activities={','.join(sorted(self._workflow_activities)) or 'none'}; "
            f"match_file={self._match_audio_path or 'none'}; target={self.match_synth.currentData()}; "
            f"quality={self.match_budget.currentData()}; offset={self.match_offset.value()}\n"
            "\n--- Application log ---\n"
            + visible_log
            + "\n"
        )

    def open_bug_report(self) -> None:
        """Collect a required description, save it locally, then upload the saved bundle."""

        dialog = QDialog(self)
        dialog.setWindowTitle("Report a Problem")
        dialog.setMinimumSize(570, 390)
        layout = QVBoxLayout(dialog)
        heading = QLabel("Help us reproduce the problem")
        heading.setStyleSheet("font-size: 20px; font-weight: 750;")
        detail = QLabel(
            "Please describe exactly what happened, what you expected, what sound "
            "or action led to it, and any steps that reliably reproduce it. The more "
            "detail you provide, the faster we can fix it. A description is required.\n\n"
            "PatchLab first saves one combined report file on your Desktop, with a "
            "Ticket ID, your comments, and detailed diagnostics. It then tries to "
            "send a private support copy. It never sends your audio or preset files. "
            "The diagnostics can include app status and local file paths."
        )
        detail.setWordWrap(True)
        comments = QPlainTextEdit()
        comments.setPlaceholderText(
            "What were you doing? What did you expect? What happened instead? "
            "Please include enough detail to reproduce the problem."
        )
        comments.setMinimumHeight(150)
        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        send = QPushButton("Send Report")
        send.setObjectName("primaryButton")
        send.setEnabled(False)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(send)
        layout.addWidget(heading)
        layout.addWidget(detail)
        layout.addWidget(comments)
        layout.addLayout(buttons)

        comments.textChanged.connect(lambda: send.setEnabled(bool(comments.toPlainText().strip())))
        cancel.clicked.connect(dialog.reject)

        def submit() -> None:
            if self.bug_report_runner.running:
                QMessageBox.information(
                    dialog, "Report already sending", "Wait for the current report to finish."
                )
                return
            try:
                from core.bug_report import create_request, load_request

                request_path = create_request(
                    comments=comments.toPlainText(), logs=self._diagnostic_log_text()
                )
            except Exception as exc:
                QMessageBox.warning(dialog, "Report was not sent", str(exc))
                return
            self.append_log(f"Saved bug report locally: {request_path}")
            # The structured flight-recorder bundle is written next to the
            # readable ticket and BEFORE any upload is attempted, so a support
            # transport problem can never destroy the local evidence. It also
            # does not depend on the private support service being connected.
            bundle = None
            try:
                from core.support_bundle import create_support_bundle

                ticket_id = load_request(request_path).ticket_id
                bundle = create_support_bundle(
                    ticket_id=ticket_id,
                    comments=comments.toPlainText(),
                    operation=self._last_operation_name(),
                    settings=self._diagnostic_settings(),
                    ticket_path=request_path,
                )
                self.append_log(
                    f"Saved diagnostic bundle: {bundle.directory}"
                    + (f" (+{bundle.archive.name})" if bundle.archive else "")
                )
                if bundle.fingerprint:
                    self.append_log(f"Failure fingerprint: {bundle.fingerprint}")
            except Exception as exc:
                # A bundle problem must never block the readable ticket the user
                # already has, so it is reported and the flow continues.
                self.append_log(f"Diagnostic bundle could not be written: {exc}")
            try:
                self.bug_report_runner.start(request_path)
            except Exception as exc:
                QMessageBox.information(
                    dialog,
                    "Report saved locally",
                    "Your report was saved on your Desktop in “PatchLab Bug "
                    f"Reports”. It could not be sent right now: {exc}",
                )
                dialog.accept()
                return
            self._bug_report_signin_attempted = False
            self.append_log("Sending user-approved bug report to private support…")
            self.statusBar().showMessage("Bug report saved locally; sending private copy…")
            dialog.accept()

        send.clicked.connect(submit)
        dialog.exec()

    def _bug_report_completed(self, result: dict) -> None:
        self._bug_report_signin_attempted = False
        ticket_id = str(result.get("ticket_id", ""))
        receipt_id = str(result.get("receipt_id", ""))
        self.append_log(
            f"Bug report sent successfully: ticket {ticket_id}, receipt {receipt_id}"
        )
        self.statusBar().showMessage("Bug report sent successfully.")
        QMessageBox.information(
            self,
            "Bug report sent",
            "Bug report sent successfully.\n\n"
            f"Ticket ID: {ticket_id}\n\n"
            "Your saved copy is on your Desktop in “PatchLab Bug Reports”.",
        )

    def _bug_report_failed(self, error: str) -> None:
        code = getattr(self.bug_report_runner, "error_code", "")
        self.append_log(f"Bug report was saved locally but not uploaded ({code or 'unknown'}): {error}")
        needs_connection = code in {"not_connected", "auth_failed"}
        request_path = getattr(self.bug_report_runner, "request_path", None)

        # SIGN IN AND RESUME, WITHOUT MAKING THE USER FILE THE REPORT AGAIN.
        #
        # A tester submitted a report and only afterwards learned PatchLab was
        # not signed in to the support service. Sign-in is the whole remedy for
        # that, so ask for it immediately and resume the SAME saved ticket (same
        # ticket id, same bytes), which the service treats as one submission.
        if (
            needs_connection
            and request_path is not None
            and not getattr(self, "_bug_report_signin_attempted", False)
        ):
            self._bug_report_signin_attempted = True
            self.statusBar().showMessage("Signing in to the support service…")
            self.append_log(
                "The support service needs a sign-in before this report can upload; "
                "asking now and then resuming the same report."
            )
            if self._reconnect_support_service() and not self.bug_report_runner.running:
                try:
                    self.bug_report_runner.start(request_path)
                    self.append_log("Signed in; resuming the saved bug report upload…")
                    self.statusBar().showMessage("Sending your bug report…")
                    return
                except Exception as exc:
                    self.append_log(f"Could not resume the bug report upload: {exc}")

        self.statusBar().showMessage("Bug report saved on this Mac; upload didn't complete.")
        box = QMessageBox(self)
        box.setWindowTitle("Bug report not uploaded")
        box.setText(
            "Your bug report was saved on this Mac, but PatchLab couldn't upload it. "
            "You can try again.\n\n" + error
        )
        retry = box.addButton("Try Again", QMessageBox.ButtonRole.AcceptRole)
        connect = (
            box.addButton("Sign In…", QMessageBox.ButtonRole.ActionRole)
            if needs_connection
            else None
        )
        box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(retry)
        box.exec()
        clicked = box.clickedButton()
        if connect is not None and clicked is connect:
            if not self._reconnect_support_service():
                return
        elif clicked is not retry:
            return
        if request_path is None or self.bug_report_runner.running:
            return
        try:
            self.bug_report_runner.start(request_path)
            self.append_log("Retrying the saved bug report upload…")
            self.statusBar().showMessage("Sending the saved bug report again…")
        except Exception as exc:
            self.append_log(f"Could not restart the bug report upload: {exc}")

    def _reconnect_support_service(self) -> bool:
        """Let the user sign in again (for example after choosing local-only)."""

        try:
            from app.access_dialog import PasscodeDialog
            from core.access_gate import AccessManager

            dialog = PasscodeDialog(AccessManager(), self)
            return dialog.exec() == PasscodeDialog.DialogCode.Accepted
        except Exception as exc:
            self.append_log(f"Sign-in could not be opened: {exc}")
            return False

    def append_log(self, message: str) -> None:
        append_runtime_log(message)
        if not hasattr(self, "log_pane") or not isinstance(self.log_pane, QTextEdit):
            return
        upper = message.upper()
        if "MATCH" in upper or "PREVIEW" in upper:
            source, color = "MATCH", theme.BLUE
        elif "RENDER" in upper or "SILENT" in upper or "CLIPPING" in upper:
            source, color = "RENDER", theme.VIOLET
        elif "ANALY" in upper or "TRAIN" in upper or "EMBED" in upper:
            source, color = "ANALYZE", theme.AMBER
        elif "FAILED" in upper or "ERROR" in upper:
            source, color = "ERROR", theme.RED
        else:
            source, color = "INFO", theme.GREEN
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_pane.append(
            f'<span style="color:#64748B">[{stamp}]</span> '
            f'<span style="color:{color}; font-weight:700">{source}</span> '
            f'<span style="color:#CBD5E1">{html.escape(message)}</span>'
        )

    @staticmethod
    def _section_relevance(section: str, values: dict) -> tuple[int, int]:
        token = section.casefold()
        priority = 10
        for needle, score in (
            ("osc", 60),
            ("filter", 55),
            ("envelope", 50),
            ("fx", 45),
            ("lfo", 40),
            ("mod", 35),
            ("noise", 30),
            ("sub", 28),
            ("macro", 25),
            ("global", 20),
        ):
            if needle in token:
                priority = score
                break
        return priority, min(len(values.get("changed", [])), 25)

    def _populate_settings_default(self, settings: dict) -> list[dict]:
        self._settings_sections = {
            str(section): dict(values) for section, values in settings.items()
        }
        ranked = sorted(
            self._settings_sections,
            key=lambda section: self._section_relevance(
                section, self._settings_sections[section]
            ),
            reverse=True,
        )
        preview_ranked = [
            section
            for section in ranked
            if self._settings_sections[section].get("changed")
        ]
        preview_sections = set(preview_ranked[:2])
        featured: list[dict] = []
        for section in ranked:
            featured.extend(self._settings_sections[section].get("changed", []))

        self._settings_building = True
        self.settings_tree.blockSignals(True)
        try:
            self._settings_mode = "default"
            self.settings_tree.clear()
            self.settings_tree.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            for section, values in self._settings_sections.items():
                changed = list(values.get("changed", []))
                item = QTreeWidgetItem(
                    [
                        str(section).upper(),
                        f"{len(changed)} changed",
                    ]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, section)
                self.settings_tree.addTopLevelItem(item)
                if section in preview_sections:
                    preview = "  •  ".join(
                        f"{entry['name']}: {entry['value']}"
                        for entry in changed[:2]
                    )
                    child = QTreeWidgetItem([preview, ""])
                    item.addChild(child)
                    child.setHidden(
                        bool(getattr(self, "_result_compact_ui", False))
                    )
                    item.setExpanded(True)
                else:
                    base_count = int(values.get("matches_base_count", 0))
                    child = QTreeWidgetItem(
                        [
                            "Click to inspect",
                            f"{base_count} at base/default",
                        ]
                    )
                    item.addChild(child)
                    child.setHidden(
                        bool(getattr(self, "_result_compact_ui", False))
                    )
                    item.setExpanded(False)
        finally:
            self.settings_tree.blockSignals(False)
            self._settings_building = False
        return featured

    def _populate_settings_detail(self, section: str) -> None:
        values = self._settings_sections.get(section)
        if values is None:
            return
        self._settings_building = True
        self.settings_tree.blockSignals(True)
        try:
            self._settings_mode = "detail"
            self.settings_tree.clear()
            back = QTreeWidgetItem(["‹  ALL SECTIONS", ""])
            back.setData(0, Qt.ItemDataRole.UserRole, "__back__")
            self.settings_tree.addTopLevelItem(back)
            item = QTreeWidgetItem([section.upper(), ""])
            item.setData(0, Qt.ItemDataRole.UserRole, section)
            self.settings_tree.addTopLevelItem(item)
            for setting in values.get("changed", []):
                item.addChild(
                    QTreeWidgetItem(
                        [str(setting["name"]), str(setting["value"])]
                    )
                )
            base_count = int(values.get("matches_base_count", 0))
            if base_count:
                item.addChild(
                    QTreeWidgetItem(
                        [f"At base/default ({base_count} settings)", "collapsed"]
                    )
                )
            item.setExpanded(True)
            self.settings_tree.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
        finally:
            self.settings_tree.blockSignals(False)
            self._settings_building = False

    def _settings_item_expanded(self, item: QTreeWidgetItem) -> None:
        if self._settings_building or self._settings_mode != "default":
            return
        section = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(section, str) and section in self._settings_sections:
            self._populate_settings_detail(section)

    def _settings_item_clicked(
        self,
        item: QTreeWidgetItem,
        _column: int,
    ) -> None:
        if item.data(0, Qt.ItemDataRole.UserRole) == "__back__":
            self._populate_settings_default(self._settings_sections)

    def _show_recommendation_more_menu(self) -> None:
        if not self._match_result:
            return
        recommendation = self._match_result.get("recommendation")
        if not isinstance(recommendation, dict):
            return
        menu = QMenu(self)
        action = menu.addAction("View full settings breakdown")
        action.triggered.connect(lambda: self._open_settings_dialog(recommendation))
        menu.exec(
            self.recommendation_more_button.mapToGlobal(
                self.recommendation_more_button.rect().bottomLeft()
            )
        )

    def _open_settings_dialog(self, recommendation: dict) -> None:
        settings = recommendation.get("settings") or {}
        if not settings:
            self.statusBar().showMessage(
                "No detailed settings are available for this recommendation."
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("PatchLab Generated Preset — Full Settings")
        dialog.setMinimumSize(560, 520)
        layout = QVBoxLayout(dialog)
        self.settings_tree.setParent(dialog)
        self.settings_tree.setVisible(True)
        self.settings_tree.setMinimumHeight(430)
        layout.addWidget(self.settings_tree)
        self._populate_settings_default(settings)
        close = QPushButton("Close")
        close.setObjectName("primaryButton")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()
        self.settings_tree.setVisible(False)
        self.settings_tree.setParent(self.recommendation_panel)

    def _selected_preview_note(self) -> int:
        value = self.octave_selector.currentData()
        return int(value) if isinstance(value, int) else 60

    def _octave_changed(self, _index: int) -> None:
        note = self._selected_preview_note()
        octave = 1 + (note - 24) // 12
        self.octave_status.setText(f"C{octave} · MIDI {note}")
        for button, _detail in getattr(self, "_match_play_buttons", []):
            button.setToolTip(f"Play this preset at C{octave} (MIDI {note})")
        self.octave_selector.setToolTip(
            f"Playing the recommendation at C{octave} (MIDI {note})"
        )

    def _octave_selected(self, index: int) -> None:
        """User clicked an octave button — update state and play immediately."""

        self._octave_changed(index)
        self.play_winner(button=self.octave_selector.buttonAt(index))

    def play_uploaded_audio(self) -> None:
        """Audition the source file the user uploaded, for A/B against a match."""

        if self._match_audio_path is None:
            return
        if not self._match_audio_path.is_file():
            # Specific to this surface: the drop zone itself should stop
            # offering to replay a file that is provably gone, not just this
            # one click. _play_audio's own missing-file message covers any
            # other caller that hits the same path.
            self.match_drop.set_playable(False)
        if not self._play_audio(self._match_audio_path):
            return
        self.statusBar().showMessage(
            f"Playing uploaded audio — {self._match_audio_path.name}"
        )

    def _play_existing_match(
        self,
        detail: dict,
        note: int | None = None,
        *,
        button: QPushButton | None = None,
    ) -> None:
        if note is None:
            note = self._selected_preview_note()
        content_hash = str(detail.get("content_hash") or "")
        if not content_hash:
            self.statusBar().showMessage("This preset has no verified preview identity")
            return
        audition_path = detail.get("audition_path")
        selected: Path | None = None
        if audition_path:
            selected = Path(audition_path).parent / f"{note}.wav"
        self._resolve_octave_preview(
            cache_key=content_hash,
            note=note,
            synth=str(detail["synth"]),
            preview_source_path=detail.get("preview_source_path"),
            existing_audio_path=selected,
            button=button,
        )

    def _favorites_db_path(self) -> Path:
        if self.distribution_mode and bool(
            self.privacy_choice.use_and_share_own_presets
        ):
            return self.local_paths["db"]
        return DEFAULT_DB_PATH

    def _load_favorite_hashes(self) -> set[str]:
        try:
            return Database(self._favorites_db_path()).favorite_hashes()
        except Exception:
            return set()

    def _toggle_favorite(self, content_hash: str, button: QPushButton) -> None:
        favorited = button.isChecked()
        try:
            Database(self._favorites_db_path()).set_favorite(content_hash, favorited)
        except Exception as exc:
            self.append_log(f"Could not save favorite: {exc}")
            return
        if favorited:
            self._favorite_hashes.add(content_hash)
        else:
            self._favorite_hashes.discard(content_hash)

    OCTAVE_NOTES = (24, 36, 48, 60, 72, 84, 96)

    def _build_closest_match_row(
        self, item: dict, rank: int
    ) -> QFrame:
        row = QFrame()
        row.setObjectName("matchRow")
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(10, 6, 10, 6)
        row_layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(8)
        rank_label = QLabel(str(rank))
        rank_label.setObjectName("muted")
        rank_label.setFixedWidth(20)
        rank_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label = QLabel(
            display_match_name(
                item.get("name"),
                rank,
                source_path=item.get("source_path"),
            )
        )
        name_label.setTextFormat(Qt.TextFormat.PlainText)
        name_label.setStyleSheet("font-weight: 650;")
        name_label.setToolTip(
            "PatchLab library result · "
            f"{'Serum 1' if item['synth'] == 'serum1' else 'Serum 2'}"
        )
        relationship = str(item.get("recommendation_relationship") or "")
        similarity = float(item["similarity_percent"])
        bar = QProgressBar()
        bar.setObjectName("similarityBar")
        bar.setProperty("accent", "teal")
        bar.setRange(0, 1000)
        bar.setValue(round(similarity * 10))
        bar.setTextVisible(False)
        bar.setFixedWidth(160)
        percent_label = QLabel(f"{similarity:.1f}%")
        percent_label.setFixedWidth(52)
        percent_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        content_hash = item.get("content_hash")
        favorite = QPushButton("♥")
        favorite.setObjectName("favoriteButton")
        favorite.setCheckable(True)
        if content_hash:
            favorite.setChecked(content_hash in self._favorite_hashes)
            favorite.setToolTip("Favorite this preset")
            favorite.clicked.connect(
                lambda _checked=False, h=str(content_hash), b=favorite: self._toggle_favorite(h, b)
            )
        else:
            favorite.setEnabled(False)
            favorite.setToolTip("Favoriting is unavailable for this result.")
        header.addWidget(rank_label)
        header.addWidget(name_label, 1)
        if relationship:
            basis = QLabel("RECOMMENDED BASIS")
            basis.setObjectName("tagPill")
            basis.setToolTip(relationship)
            header.addWidget(basis)
        header.addWidget(bar)
        header.addWidget(percent_label)
        header.addWidget(favorite)
        row_layout.addLayout(header)

        octave_row = QHBoxLayout()
        octave_row.setSpacing(3)
        playable = bool(item.get("audition_path") or item.get("preview_source_path"))
        for octave, note in enumerate(self.OCTAVE_NOTES, start=1):
            note_button = QPushButton(f"C{octave}")
            note_button.setObjectName("rowOctaveButton")
            note_button.setEnabled(playable)
            if playable:
                note_button.setToolTip(f"Play at C{octave} (MIDI {note})")
                note_button.clicked.connect(
                    lambda _checked=False, detail=dict(item), n=note, b=note_button: self._play_existing_match(
                        detail, note=n, button=b
                    )
                )
            octave_row.addWidget(note_button)
        if not playable:
            octave_row.addWidget(self._muted_label("No local audio or factory preset is available."))
        row_layout.addLayout(octave_row)
        return row

    @staticmethod
    def _muted_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("muted")
        return label

    def _render_existing_matches(self) -> None:
        layout = self.existing_list_layout
        while layout.count():
            child = layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()
        for index, item in enumerate(self._existing_matches, start=1):
            layout.addWidget(
                self._build_closest_match_row(item, index)
            )
        layout.addStretch(1)
        has_matches = bool(self._existing_matches)
        self.closest_placeholder.setVisible(not has_matches)
        self.existing_list_container.setVisible(has_matches)

    def _show_match_result(self, result: dict) -> None:
        self.match_results.setVisible(True)
        existing = list(result.get("existing_matches", []))
        recommendation = result.get("recommendation")
        basis_index = unmodified_recommendation_basis_index(result)
        if basis_index is not None:
            item = dict(existing[basis_index])
            item["recommendation_relationship"] = (
                "This closest match is also the unchanged basis shown "
                "in the Recommended Preset panel."
            )
            existing[basis_index] = item
        self._existing_matches = existing
        self._favorite_hashes = self._load_favorite_hashes()
        self._render_existing_matches()

        self.settings_tree.clear()
        self.recommendation_placeholder.setVisible(False)
        self.recommendation_details.setVisible(True)
        if not isinstance(recommendation, dict):
            self.confidence_ring.setValue(0)
            self.recommendation_badge.setText("NO MATCH")
            self.recommendation_badge.setStyleSheet(f"color: {theme.RED};")
            self.recommendation_name.setText("No confident match")
            self.recommendation_subtitle.setText("")
            for tag in self.recommendation_tags:
                tag.setVisible(False)
            self.recommendation_confidence.setText(
                str(result.get("message", "No confident match"))
            )
            self.octave_selector.setEnabled(False)
            self.save_preset_button.setEnabled(False)
            self.open_preset_location_button.setEnabled(False)
            self.recommendation_more_button.setEnabled(False)
            self.match_stats.setText(
                str(result.get("message", "No confident match"))
            )
            self._refresh_save_state()
            return

        similarity = float(recommendation["similarity_percent"])
        self.confidence_ring.setValue(similarity)
        synth_key = str(recommendation["synth"])
        synth_name = "Serum 1" if synth_key == "serum1" else "Serum 2"
        confidence_label = (
            "High Match"
            if similarity >= 90
            else "Good Match"
            if similarity >= 80
            else "Fair Match"
            if similarity >= 65
            else "Low Match"
        )
        color = (
            theme.GREEN
            if similarity >= 90
            else theme.TEAL
            if similarity >= 80
            else theme.AMBER
            if similarity >= 65
            else theme.RED
        )
        meaningfully_modified = bool(
            recommendation.get("meaningfully_modified", False)
        )
        matching_basis = next(
            (
                item
                for item in existing
                if str(item.get("content_hash") or "")
                == str(recommendation.get("content_hash") or "")
            ),
            None,
        )
        base_name = (
            generated_preset_name(synth_key)
            if meaningfully_modified
            else display_match_name(
                matching_basis.get("name") if matching_basis else None,
                1,
                source_path=(
                    matching_basis.get("source_path")
                    if matching_basis
                    else recommendation.get("preview_source_path")
                ),
            )
        )
        self.recommendation_badge.setText(confidence_label.upper())
        self.recommendation_badge.setStyleSheet(f"color: {color};")
        self.recommendation_name.setText(base_name)
        category = "Generated" if meaningfully_modified else "Existing preset"
        self.recommendation_subtitle.setText(
            f"PatchLab generated · {synth_name}"
            if meaningfully_modified
            else f"Closest-match basis · {synth_name}"
        )
        self.recommendation_thumbnail.setAccent(
            "blue" if synth_key == "serum2" else "violet"
        )
        style_label, character_label = derive_style_character(
            (
                str(
                    resolve_result_path(
                        self._match_result_path,
                        recommendation["winner_audio_path"],
                    )
                )
                if self._match_result_path
                and recommendation.get("winner_audio_path")
                else recommendation.get("winner_audio_path")
            )
        )
        tag_values = (category, style_label, character_label, "PatchLab")
        for tag, value in zip(self.recommendation_tags, tag_values, strict=True):
            tag.setText(value)
            tag.setVisible(True)
        self.recommendation_confidence.setStyleSheet(
            "font-size: 11px;"
        )
        self.recommendation_confidence.setText(
            (
                "PatchLab generated preset · "
                f"{recommendation['evaluations']} evaluations · "
                f"{float(recommendation['elapsed_s']):.1f}s"
            )
            if meaningfully_modified
            else "Unmodified closest preset · the tagged result on the left "
            "is the same patch, shown as its recommendation basis"
        )
        self.octave_selector.setEnabled(
            bool(
                recommendation.get("winner_audio_path")
                or recommendation.get("preview_source_path")
            )
        )
        export_available = bool(recommendation.get("export_available", True))
        self._export_available_for_current_match = export_available
        self.save_preset_button.setEnabled(export_available)
        self.open_preset_location_button.setEnabled(export_available)
        self.recommendation_more_button.setEnabled(bool(recommendation.get("settings")))
        self.parameter_strip.setVisible(False)
        self.settings_tree.setVisible(False)
        self._octave_changed(self.octave_selector.currentIndex())
        self.match_stats.setText(str(result.get("message", "Match complete")))
        self.statusBar().showMessage("Match complete")
        self._refresh_workflow_cards()
        if self._batch_state is not None:
            self.save_preset_button.setEnabled(False)
            self.open_preset_location_button.setEnabled(False)
        self._refresh_save_state()

    def _scan_completed(self, summary: dict) -> None:
        super()._scan_completed(summary)

    def _render_progress_changed(self, detail: dict) -> None:
        super()._render_progress_changed(detail)

    def _render_completed(self, summary: dict) -> None:
        super()._render_completed(summary)

    def _analyze_progress_changed(self, detail: dict) -> None:
        super()._analyze_progress_changed(detail)

    def _analyze_completed(self, summary: dict) -> None:
        super()._analyze_completed(summary)

    def _match_progress_changed(self, detail: dict) -> None:
        super()._match_progress_changed(detail)
