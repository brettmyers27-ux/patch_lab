"""PatchLab single-window interface."""

from __future__ import annotations

import html
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    ExportProcessRunner,
    MatchProcessRunner,
    PreviewProcessRunner,
    RenderProcessRunner,
    ScanProcessRunner,
)
from core.audio_input import SUPPORTED_AUDIO_SUFFIXES
from core.branding import generated_preset_name, public_match_name
from core.db import DEFAULT_DB_PATH, Database
from core.factory_verify import FactoryVerification
from core.local_library import default_local_paths
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
from core.platform_env import ENV
from core.privacy import PrivacyStore, distribution_mode
from core.verify import library_scan_checks


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

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        urls = event.mimeData().urls()
        if len(urls) == 1 and Path(urls[0].toLocalFile()).suffix.casefold() in SUPPORTED_AUDIO_SUFFIXES:
            self.setProperty("dragActive", True)
            self.style().unpolish(self)
            self.style().polish(self)
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        urls = event.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()


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
        self.privacy_choice = self.privacy_store.load()
        self.factory_mapping_path = (
            ENV.app_data_dir / "factory-paths.json"
            if self.distribution_mode
            else Path(__file__).resolve().parents[1] / "data" / "local" / "factory_paths.json"
        )
        self.local_paths = default_local_paths()
        self.setWindowTitle("PatchLab")
        self.resize(1050, 900)
        self.runner = ScanProcessRunner(self)
        self.runner.log.connect(self.append_log)
        self.runner.progress.connect(self._progress)
        self.runner.completed.connect(self._scan_completed)
        self.runner.failed.connect(self._scan_failed)
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
        self.preview_runner.completed.connect(self._preview_completed)
        self.preview_runner.failed.connect(self._preview_failed)
        self._render_paused = False
        self._match_audio_path: Path | None = None
        self._match_result_path: Path | None = None
        self._match_result: dict | None = None
        self._existing_matches: list[dict] = []
        self._existing_page = 0
        self._favorite_hashes: set[str] = set()
        self._current_match_uid: str | None = None
        self._library_preview_button: QPushButton | None = None
        self._library_preview_uid: str | None = None
        self._export_context_uid: str | None = None
        self._batch_state: dict | None = None
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
        render_ready = not any(result.failed for result in library_scan_checks(DEFAULT_DB_PATH))
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
        self.existing_table = QTableWidget(0, 5)
        self.existing_table.setHorizontalHeaderLabels(
            ["Preset", "Synth", "Similarity", "Source path", "Audition"]
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
        if verification is None or not verification.bundle_available:
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
        heading = QLabel("Would you like to use and share your own presets?")
        heading.setStyleSheet("font-size: 18px; font-weight: 650;")
        body = QLabel(
            "If you agree, you can link a preset folder. PatchLab will process every linked "
            "preset locally so it is searchable on this computer. It will also contribute a "
            "copy of non-factory preset files and their fingerprints/settings to the developer’s "
            "shared library. Rendered audio is never uploaded.\n\n"
            "If you disagree, PatchLab remains fully usable with its built-in factory "
            "fingerprints. You can change this later with the single Privacy setting."
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
            # Local processing is resumable. Stopping the worker here ensures a
            # withdrawn choice also prevents its later upload phase.
            self.runner.cancel()
        self._apply_privacy_choice()

    def _apply_privacy_choice(self) -> None:
        enabled = bool(self.privacy_choice.use_and_share_own_presets)
        self.scan_box.setEnabled(enabled)
        self.scan_box.setToolTip(
            "" if enabled else "Turn on “Use & share my own presets” in Privacy to link a folder."
        )

    @staticmethod
    def _render_library_complete() -> bool:
        if not DEFAULT_DB_PATH.is_file():
            return False
        import sqlite3

        connection = sqlite3.connect(DEFAULT_DB_PATH)
        return int(connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0]) == 39_053

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
        if self.distribution_mode and not self.privacy_choice.use_and_share_own_presets:
            return
        defaults = ENV.existing_preset_roots
        initial = str(defaults[0] if defaults else Path.home())
        selected = QFileDialog.getExistingDirectory(self, "Select Serum preset folder", initial)
        if not selected:
            return
        self.scan_button.setEnabled(False)
        self.scan_progress.setValue(0)
        self.append_log(
            f"Starting local-first preset processing for {selected}"
            if self.distribution_mode
            else f"Starting isolated scan worker for {selected}"
        )
        self.statusBar().showMessage(
            "Processing your presets locally…"
            if self.distribution_mode
            else "Scanning and dumping parameters…"
        )
        self.runner.start(Path(selected), local_library=self.distribution_mode)
        if self.distribution_mode:
            self.privacy_choice = self.privacy_store.save(
                True, linked_folder=Path(selected)
            )

    def append_log(self, message: str) -> None:
        self.log_pane.appendPlainText(message)
        bar = self.log_pane.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _progress(self, current: int, total: int) -> None:
        self.scan_progress.setMaximum(max(total, 1))
        self.scan_progress.setValue(current)

    def _scan_completed(self, summary: dict) -> None:
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
                f"factory uploads skipped {summary.get('factory_skipped_upload', 0)}"
            )
        else:
            text = (
                f"Scan complete: found {summary.get('found', 0)}, deduped {summary.get('deduped', 0)}, "
                f"params dumped {summary.get('params_dumped', 0)}, failed {summary.get('failed', 0)}, "
                f"Serum 2 unavailable {summary.get('serum2_disabled', 0)}"
            )
        self.append_log(text)
        self.statusBar().showMessage(text)
        if not self.distribution_mode:
            self.render_button.setEnabled(True)

    def _scan_failed(self, error: str) -> None:
        self.scan_button.setEnabled(True)
        self.append_log(f"Scan failed: {error}")
        self.statusBar().showMessage(error)

    def start_render(self) -> None:
        self.render_button.setEnabled(False)
        self.render_pause_button.setEnabled(True)
        self.render_cancel_button.setEnabled(True)
        self.render_progress.setValue(0)
        self._render_paused = False
        self.render_pause_button.setText("Pause")
        self.render_stats.setText("Starting four render workers…")
        self.append_log("Starting resumable four-process library render")
        self.statusBar().showMessage("Rendering sound library…")
        self.render_runner.start()

    def toggle_render_pause(self) -> None:
        if self._render_paused:
            self.render_runner.resume()
        else:
            self.render_runner.pause()

    def cancel_render(self) -> None:
        self.render_cancel_button.setEnabled(False)
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

    def _render_control_changed(self, state: str) -> None:
        if state == "paused":
            self._render_paused = True
            self.render_pause_button.setText("Resume")
            self.render_stats.setText("Paused")
        elif state == "resumed":
            self._render_paused = False
            self.render_pause_button.setText("Pause")

    def _render_completed(self, summary: dict) -> None:
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

    def _render_failed(self, error: str) -> None:
        self.render_button.setEnabled(True)
        self.render_pause_button.setEnabled(False)
        self.render_cancel_button.setEnabled(False)
        self.render_stats.setText(error)
        self.append_log(error)
        self.statusBar().showMessage(error)

    def start_analyze(self) -> None:
        self.learn_button.setEnabled(False)
        self.analyze_cancel_button.setEnabled(True)
        self.learn_progress.setRange(0, 100)
        self.learn_progress.setValue(0)
        self.analyze_stats.setText("Starting target vectorization…")
        self.statusBar().showMessage("Analyzing and learning…")
        self.analyze_runner.start(self.deep_training.isChecked())

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

    def _analyze_completed(self, summary: dict) -> None:
        self.learn_progress.setRange(0, 100)
        self.learn_progress.setValue(100)
        self.learn_button.setEnabled(True)
        self.analyze_cancel_button.setEnabled(False)
        self.analyze_stats.setText("Analyze & Learn complete")
        self.append_log(f"Analyze & Learn complete: {summary}")
        self.statusBar().showMessage("Analyze & Learn complete")

    def _analyze_failed(self, error: str) -> None:
        self.learn_progress.setRange(0, 100)
        self.learn_button.setEnabled(True)
        self.analyze_cancel_button.setEnabled(False)
        self.analyze_stats.setText(error)
        self.append_log(error)
        self.statusBar().showMessage(error)

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
        self._match_result = None
        self._match_result_path = None
        self.match_drop.setText(path.name)
        self.match_drop.set_playable(True)
        self.match_start_button.setEnabled(True)
        self.match_stats.setText(
            "Ready to match. Existing-preset results always search both synths."
        )
        # match_results now hosts the drop zone itself, so it stays visible;
        # only the per-result detail section is reset here.
        self.recommendation_details.setVisible(False)
        self.recommendation_placeholder.setVisible(True)
        self.save_preset_button.setEnabled(False)

    def start_match(self) -> None:
        if self._match_audio_path is None:
            return
        self.match_start_button.setEnabled(False)
        self.match_button.setEnabled(False)
        self.match_cancel_button.setEnabled(True)
        self.match_synth.setEnabled(False)
        self.match_budget.setEnabled(False)
        self.match_offset.setEnabled(False)
        self.match_progress.setRange(0, 0)
        self.match_stats.setText("Loading audio and models…")
        self.recommendation_details.setVisible(False)
        self.recommendation_placeholder.setVisible(True)
        target_synth = str(self.match_synth.currentData())
        budget = str(self.match_budget.currentData())
        self.append_log(
            f"Matching {self._match_audio_path.name} for {target_synth} ({budget})"
        )
        self.match_runner.start(
            self._match_audio_path,
            target_synth=target_synth,
            budget=budget,
            offset=float(self.match_offset.value()),
            session_root=Path(self._match_session.name),
            factory_only=self.distribution_mode,
            factory_mapping=self.factory_mapping_path if self.distribution_mode else None,
            local_db=(
                self.local_paths["db"]
                if self.distribution_mode
                and bool(self.privacy_choice.use_and_share_own_presets)
                else None
            ),
        )

    def _match_progress_changed(self, detail: dict) -> None:
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

    @staticmethod
    def _play_audio(path: Path) -> None:
        import sounddevice as sd
        import soundfile as sf

        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        sd.stop()
        sd.play(audio, rate, blocking=False)

    def _match_completed(self, result_path: str) -> None:
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

    def _show_match_result(self, result: dict) -> None:
        self.match_results.setVisible(True)
        existing = list(result.get("existing_matches", []))
        self.existing_table.setRowCount(len(existing))
        for row_index, item in enumerate(existing):
            self.existing_table.setItem(
                row_index, 0, QTableWidgetItem(public_match_name(row_index + 1))
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

    def _match_failed(self, error: str) -> None:
        self.match_progress.setRange(0, 100)
        self.match_progress.setValue(0)
        self.match_start_button.setEnabled(self._match_audio_path is not None)
        self.match_button.setEnabled(True)
        self.match_cancel_button.setEnabled(False)
        self.match_synth.setEnabled(True)
        self.match_budget.setEnabled(True)
        self.match_offset.setEnabled(True)
        self.match_stats.setText(error)
        self.append_log(f"Match failed: {error}")
        self.statusBar().showMessage(error)

    def play_winner(self) -> None:
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
        if recommendation.get("preview_source_path"):
            self._render_preview(
                {
                    "preview_source_path": recommendation["preview_source_path"],
                    "synth": recommendation["synth"],
                    "audition_midi_note": note,
                    "content_hash": recommendation["content_hash"],
                }
            )
            return
        if self._match_result_path and recommendation.get("candidate_path"):
            cached = self._match_result_path.parent / f"recommendation-{note}.wav"
            if cached.is_file():
                self._play_audio(cached)
                return
            self.statusBar().showMessage(
                f"Rendering recommendation preview at MIDI {note}…"
            )
            try:
                self.preview_runner.start_recommendation(
                    self._match_result_path,
                    note,
                )
            except RuntimeError as exc:
                self.statusBar().showMessage(str(exc))
            return
        if recommendation.get("winner_audio_path"):
            self._play_audio(Path(recommendation["winner_audio_path"]))

    def _render_preview(self, detail: dict) -> None:
        self.statusBar().showMessage("Rendering factory preview locally…")
        self.preview_runner.start(
            Path(detail["preview_source_path"]),
            synth=str(detail["synth"]),
            midi_note=int(detail.get("audition_midi_note") or 60),
            content_hash=str(detail["content_hash"]),
            output_root=Path(self._match_session.name),
        )

    def _preview_completed(self, path: str) -> None:
        self.statusBar().showMessage("Factory preview ready")
        self._play_audio(Path(path))

    def _preview_failed(self, error: str) -> None:
        self.append_log(f"Factory preview failed: {error}")
        self.statusBar().showMessage(error)

    def _default_export_folder(self, synth: str) -> Path:
        token = "serum 2" if synth == "serum2" else "serum presets"
        matching = [
            path for path in ENV.existing_preset_roots if token in str(path).casefold()
        ]
        return matching[0] if matching else Path.home()

    def save_match_preset(self) -> None:
        if not self._match_result or self._match_result_path is None:
            return
        recommendation = self._match_result.get("recommendation")
        if not isinstance(recommendation, dict):
            return
        synth = str(recommendation["synth"])
        extension = ".fxp" if synth == "serum1" else ".SerumPreset"
        name = generated_preset_name(synth)
        suggested = self._default_export_folder(synth) / f"{name}{extension}"
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save generated Serum preset",
            str(suggested),
            "Serum 1 preset (*.fxp)"
            if synth == "serum1"
            else "Serum 2 preset (*.SerumPreset)",
        )
        if not selected:
            return
        output = Path(selected)
        if output.suffix.casefold() != extension.casefold():
            output = output.with_suffix(extension)
        self._start_preset_export(output, trigger=self.save_preset_button, label="Export Preset")

    def load_in_serum(self) -> None:
        """Write the recommendation straight into the local Serum presets folder.

        This never touches a live/running Serum instance — PatchLab has
        consistently avoided automating a real DAW/plugin session. This is
        the same verified export as "Export Preset", just written directly
        to the detected local install folder so Serum's own browser picks
        it up next time it refreshes, instead of prompting a save dialog.
        """

        if not self._match_result or self._match_result_path is None:
            return
        recommendation = self._match_result.get("recommendation")
        if not isinstance(recommendation, dict):
            return
        synth = str(recommendation["synth"])
        extension = ".fxp" if synth == "serum1" else ".SerumPreset"
        name = generated_preset_name(synth)
        folder = self._default_export_folder(synth)
        output = folder / f"{name}{extension}"
        counter = 2
        while output.exists():
            output = folder / f"{name} {counter}{extension}"
            counter += 1
        self._start_preset_export(
            output, trigger=self.load_in_serum_button, label="Load in Serum"
        )

    def _start_preset_export(
        self, output: Path, *, trigger: QPushButton, label: str
    ) -> None:
        self.save_preset_button.setEnabled(False)
        load_button = getattr(self, "load_in_serum_button", None)
        if load_button is not None:
            load_button.setEnabled(False)
        trigger.setText("Verifying…")
        self.statusBar().showMessage(
            "Generating the preset in temporary storage, then saving it to "
            f"{output.parent}…"
        )
        self.export_runner.start(self._match_result_path, output)

    def _export_completed(self, detail: dict) -> None:
        self.save_preset_button.setEnabled(True)
        self.save_preset_button.setText("Export Preset")
        load_button = getattr(self, "load_in_serum_button", None)
        if load_button is not None:
            load_button.setEnabled(True)
            load_button.setText("Load in Serum")
        warning = detail.get("verification_warning")
        message = f"Preset saved: {detail['path']}"
        self.append_log(message)
        if warning:
            self.append_log(f"Preset verification note: {warning}")
        self.statusBar().showMessage(message)
        if not warning:
            QMessageBox.information(self, "Preset ready", message)

    def _export_failed(self, error: str) -> None:
        self.save_preset_button.setEnabled(True)
        self.save_preset_button.setText("Export Preset")
        load_button = getattr(self, "load_in_serum_button", None)
        if load_button is not None:
            load_button.setEnabled(True)
            load_button.setText("Load in Serum")
        self.append_log(f"Preset export failed: {error}")
        self.statusBar().showMessage(error)
        QMessageBox.critical(
            self,
            "Preset was not saved",
            "PatchLab could not write a valid preset to the selected location.\n\n"
            + error,
        )

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
        self.privacy_choice = self.privacy_store.load()
        self.factory_mapping_path = (
            ENV.app_data_dir / "factory-paths.json"
            if self.distribution_mode
            else Path(__file__).resolve().parents[1]
            / "data"
            / "local"
            / "factory_paths.json"
        )
        self.local_paths = default_local_paths()
        self.runner = ScanProcessRunner(self)
        self.runner.log.connect(self.append_log)
        self.runner.progress.connect(self._progress)
        self.runner.completed.connect(self._scan_completed)
        self.runner.failed.connect(self._scan_failed)
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
        self.preview_runner.completed.connect(self._preview_completed)
        self.preview_runner.failed.connect(self._preview_failed)
        self._render_paused = False
        self._match_audio_path: Path | None = None
        self._match_result_path: Path | None = None
        self._match_result: dict | None = None
        self._existing_matches: list[dict] = []
        self._existing_page = 0
        self._favorite_hashes: set[str] = set()
        self._current_match_uid: str | None = None
        self._library_preview_button: QPushButton | None = None
        self._library_preview_uid: str | None = None
        self._export_context_uid: str | None = None
        self._batch_state: dict | None = None
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
        self.render_cancel_button = QPushButton("Cancel")
        self.render_cancel_button.setEnabled(False)
        self.render_cancel_button.clicked.connect(self.cancel_render)
        self.render_stats = QLabel("Ready")
        self.deep_training = QCheckBox("Deep training")
        self.deep_training.setChecked(True)
        self.analyze_cancel_button = QPushButton("Cancel")
        self.analyze_cancel_button.setEnabled(False)
        self.analyze_cancel_button.clicked.connect(self.analyze_runner.cancel)
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
            f"Interface ready · {ENV.branch} · compute {ENV.compute_backend}"
        )
        self.statusBar().showMessage(
            f"Ready — {ENV.branch}, compute: {ENV.compute_backend}"
        )
        if self.distribution_mode:
            self._apply_factory_status()
            self._apply_privacy_choice()
            if self.privacy_choice.use_and_share_own_presets is None:
                QTimer.singleShot(0, self._show_consent_dialog)

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

        scan_ready = not any(
            result.failed for result in library_scan_checks(DEFAULT_DB_PATH)
        )
        render_complete = self._render_library_complete()
        model_ready = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "models"
            / "param_model.pt"
        ).is_file() and (
            Path(__file__).resolve().parents[1]
            / "data"
            / "features"
            / "preset_index.npy"
        ).is_file()
        match_ready = self.distribution_mode or model_ready
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
                "Render Sound Library",
                "waveform",
                "violet",
                enabled=scan_ready,
                step=2,
            ),
            HeroCard(
                "Analyze & Learn",
                "brain",
                "amber",
                enabled=render_complete,
                step=3,
            ),
            HeroCard(
                "Match a Sound",
                "search-wave",
                "blue",
                enabled=match_ready,
                step=4,
            ),
        )
        hero_layout = QHBoxLayout()
        hero_layout.setSpacing(11)
        for card in cards:
            hero_layout.addWidget(card, 1)
        root_layout.addLayout(hero_layout)
        scan_card, render_card, learn_card, match_card = cards
        self.hero_cards = cards
        self.scan_button, self.scan_progress = scan_card.button, scan_card.progress
        self.render_button, self.render_progress = (
            render_card.button,
            render_card.progress,
        )
        self.learn_button, self.learn_progress = learn_card.button, learn_card.progress
        self.match_button, self.match_progress = match_card.button, match_card.progress
        self.scan_box = scan_card
        self.scan_card_status = scan_card.status
        self.render_card_status = render_card.status
        self.learn_card_status = learn_card.status
        self.match_card_status = match_card.status
        self.scan_button.clicked.connect(self.choose_folder)
        self.render_button.clicked.connect(self.start_render)
        self.learn_button.clicked.connect(self.start_analyze)
        self.match_button.clicked.connect(self.choose_match_file)
        if scan_ready:
            self.scan_progress.setValue(100)
            self.scan_card_status.setText("Library indexed ✓")
        if render_complete:
            self.render_progress.setValue(100)
            self.render_card_status.setText("Sound library ready ✓")
        if model_ready:
            self.learn_progress.setValue(100)
            self.learn_card_status.setText("Model trained ✓")
        if match_ready:
            self.match_progress.setValue(100)
            self.match_card_status.setText("Match ready ✓")

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
        training_title = QLabel("DEEP TRAINING")
        training_title.setObjectName("controlTitle")
        training_layout.addWidget(training_title)
        training_group = QVBoxLayout()
        training_group.setSpacing(group_spacing)
        training_options_label = QLabel("OPTIONS")
        training_options_label.setObjectName("microLabel")
        training_group.addWidget(training_options_label)
        training_controls = QHBoxLayout()
        training_controls.setSpacing(18)
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
        self.save_preset_button = QPushButton("Export Preset")
        self.save_preset_button.setObjectName("compactActionButton")
        self.save_preset_button.setIcon(icon("save"))
        self.save_preset_button.clicked.connect(self.save_match_preset)
        self.load_in_serum_button = QPushButton("Load in Serum")
        self.load_in_serum_button.setObjectName("compactActionButton")
        self.load_in_serum_button.clicked.connect(self.load_in_serum)
        self.recommendation_more_button = QPushButton("…")
        self.recommendation_more_button.setObjectName("compactActionButton")
        self.recommendation_more_button.setFixedWidth(30)
        self.recommendation_more_button.clicked.connect(self._show_recommendation_more_menu)
        actions_row.addWidget(self.save_preset_button)
        actions_row.addWidget(self.load_in_serum_button)
        actions_row.addWidget(self.recommendation_more_button)
        details_layout.addLayout(actions_row)
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
        self.match_results.setFixedHeight(740)
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
        self._scene.addWidget(root)
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
        export = QPushButton("Export Preset")
        export.setObjectName("compactActionButton")
        export.setEnabled(
            not record.no_confident_match and self._batch_state is None
        )
        if self._batch_state is not None:
            export.setToolTip("Exports resume after the active batch finishes.")
        export.clicked.connect(
            lambda _checked=False, uid=record.match_uid: self.export_library_match(uid)
        )
        delete = QPushButton("Delete")
        delete.setObjectName("compactActionButton")
        delete.clicked.connect(
            lambda _checked=False, uid=record.match_uid: self.delete_library_match(uid)
        )
        row_layout.addWidget(export)
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
        self._play_audio(source)
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
        cached = result_path.parent / f"recommendation-{note}.wav"
        if cached.is_file():
            self._play_audio(cached)
            return
        if self._batch_state is not None:
            self.statusBar().showMessage(
                "This uncached octave can be rendered after the active batch finishes"
            )
            return
        if self.preview_runner.process.state() != QProcess.ProcessState.NotRunning:
            self.statusBar().showMessage("Another preview is already rendering")
            return
        self._library_preview_button = button
        self._library_preview_uid = match_uid
        button.setText("Rendering…")
        button.setEnabled(False)
        self.preview_runner.start_recommendation(result_path, note)

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

    def delete_library_match(self, match_uid: str) -> None:
        if QMessageBox.question(
            self,
            "Delete saved match?",
            "This permanently removes the archived audio, generated files, and history entry.",
        ) != QMessageBox.StandardButton.Yes:
            return
        delete_archived_match(
            Database(self._match_database_path()),
            match_uid,
            library_root=self._match_library_root(),
        )
        if self._current_match_uid == match_uid:
            self._current_match_uid = None
        self.refresh_match_library()

    def export_library_match(self, match_uid: str) -> None:
        if self._batch_state is not None:
            QMessageBox.information(
                self,
                "Batch is running",
                "Verified exports are reserved for the active batch. Try again after it finishes.",
            )
            return
        record = Database(self._match_database_path()).get_match_library(match_uid)
        if record is None:
            return
        _source, result_path = resolved_record_paths(record, self._match_library_root())
        extension = ".fxp" if record.recommendation_synth == "serum1" else ".SerumPreset"
        suggested = (
            self._default_export_folder(record.recommendation_synth)
            / f"{generated_preset_name(record.recommendation_synth)}{extension}"
        )
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export verified preset", str(suggested),
            "Serum preset (*.fxp *.SerumPreset)",
        )
        if not selected:
            return
        output = Path(selected)
        if output.suffix.casefold() != extension.casefold():
            output = output.with_suffix(extension)
        self._export_context_uid = match_uid
        self.export_runner.start(result_path, output)

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

    def _start_preset_export(
        self, output: Path, *, trigger: QPushButton, label: str
    ) -> None:
        if self._batch_state is not None:
            QMessageBox.information(
                self,
                "Batch is running",
                "Verified exports are reserved for the active batch. Try again after it finishes.",
            )
            return
        super()._start_preset_export(output, trigger=trigger, label=label)

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
        if self._batch_state is None:
            return
        self._batch_state["current_uid"] = archived.record.match_uid
        recommendation = self._match_result.get("recommendation")
        if not isinstance(recommendation, dict):
            self.append_log(
                f"Batch retained no-confident match: {archived.record.source_name}; no preset was exported"
            )
            self._batch_file_completed()
            return
        extension = ".fxp" if recommendation["synth"] == "serum1" else ".SerumPreset"
        source_stem = sanitize_folder_name(Path(source).stem) or "Sound"
        output = disambiguated_preset_path(
            self._batch_state["export_folder"],
            f"PatchLab - {source_stem}",
            extension,
        )
        self._batch_state["phase"] = "export"
        self.export_runner.start(archived.result_json_path, output)

    def _match_failed(self, error: str) -> None:
        if self._batch_state is not None:
            self._batch_file_failed(error)
            return
        super()._match_failed(error)

    def _export_completed(self, detail: dict) -> None:
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
        super()._export_completed(detail)
        if self._current_match_uid:
            Database(self._match_database_path()).set_match_exported_path(
                self._current_match_uid, Path(detail["path"])
            )
            self.refresh_match_library()

    def _export_failed(self, error: str) -> None:
        if self._export_context_uid:
            self._export_context_uid = None
            self.append_log(f"Library preset export failed: {error}")
            self.statusBar().showMessage(error)
            return
        if self._batch_state is not None and self._batch_state.get("phase") == "export":
            self._batch_file_failed(f"verified export failed: {error}")
            return
        super()._export_failed(error)

    def _preview_completed(self, path: str) -> None:
        if self._library_preview_button is not None:
            button = self._library_preview_button
            cached = Path(path)
            octave = 1 + (int(cached.stem.split("-")[-1]) - 24) // 12
            button.setText(f"C{octave}")
            button.setEnabled(True)
            self._library_preview_button = None
            self._library_preview_uid = None
            self._play_audio(cached)
            self.statusBar().showMessage("Library octave preview ready")
            return
        super()._preview_completed(path)

    def _preview_failed(self, error: str) -> None:
        if self._library_preview_button is not None:
            self._library_preview_button.setText("Retry")
            self._library_preview_button.setEnabled(True)
            self._library_preview_button = None
            self._library_preview_uid = None
        super()._preview_failed(error)

    def start_batch_folder(self) -> None:
        if self._batch_state is not None:
            QMessageBox.information(self, "Batch already running", "Only one batch can run at a time.")
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
        export_folder = (
            self._default_export_folder(target_synth)
            / "PatchLab"
            / folder_name
        )
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
        self.load_in_serum_button.setEnabled(False)
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
        self.match_runner.start(
            path,
            target_synth=state["target_synth"],
            budget=state["budget"],
            offset=0.0,
            session_root=Path(self._match_session.name),
            factory_only=self.distribution_mode,
            factory_mapping=self.factory_mapping_path if self.distribution_mode else None,
            local_db=(
                self.local_paths["db"]
                if self.distribution_mode
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
            self.load_in_serum_button.setEnabled(True)
        self._batch_state = None
        self.refresh_match_library()
        self.nav_tabs.setCurrentIndex(1)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
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
        dialog = QDialog(self)
        dialog.setWindowTitle("PatchLab Settings")
        dialog.setMinimumWidth(430)
        layout = QVBoxLayout(dialog)
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 20px; font-weight: 750;")
        layout.addWidget(title)
        if self.distribution_mode:
            toggle = QCheckBox("Use && share my own presets")
            toggle.setChecked(self.share_toggle.isChecked())
            toggle.toggled.connect(self.share_toggle.setChecked)
            layout.addWidget(toggle)
            forget = QPushButton("Sign out / forget passcode")
            forget.setObjectName("compactActionButton")
            forget.clicked.connect(self._forget_passcode)
            layout.addWidget(forget)
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
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec()

    def _forget_passcode(self) -> None:
        from core.access_gate import AccessStore

        AccessStore().clear()
        QMessageBox.information(
            self,
            "Signed out",
            "The saved PatchLab passcode was removed. Your terms choice was not changed. "
            "The passcode will be requested the next time PatchLab starts.",
        )

    def open_help(self) -> None:
        QMessageBox.about(
            self,
            "About PatchLab",
            "PatchLab\n\n"
            "Select a library, render it, analyze it once, then match any sound. "
            "The complete workflow and troubleshooting guide are in README.md.\n\n"
            f"{Path(__file__).resolve().parents[1] / 'README.md'}",
        )

    def append_log(self, message: str) -> None:
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
        self.play_winner()

    def play_uploaded_audio(self) -> None:
        """Audition the source file the user uploaded, for A/B against a match."""

        if self._match_audio_path is None:
            return
        if not self._match_audio_path.is_file():
            message = "The uploaded audio file is no longer available."
            self.append_log(message)
            self.statusBar().showMessage(message)
            self.match_drop.set_playable(False)
            return
        try:
            self._play_audio(self._match_audio_path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user below
            message = f"Could not play the uploaded audio: {exc}"
            self.append_log(message)
            self.statusBar().showMessage(message)
            return
        self.statusBar().showMessage(
            f"Playing uploaded audio — {self._match_audio_path.name}"
        )

    def _play_existing_match(self, detail: dict, note: int | None = None) -> None:
        if note is None:
            note = self._selected_preview_note()
        audition_path = detail.get("audition_path")
        if audition_path:
            selected = Path(audition_path).parent / f"{note}.wav"
            if selected.is_file():
                self._play_audio(selected)
                return
        if detail.get("preview_source_path"):
            self._render_preview(
                {
                    **detail,
                    "audition_midi_note": note,
                }
            )
            return
        octave = 1 + (note - 24) // 12
        message = f"No C{octave} preview is available for this preset."
        self.append_log(message)
        self.statusBar().showMessage(message)

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
        name_label = QLabel(public_match_name(rank))
        name_label.setStyleSheet("font-weight: 650;")
        name_label.setToolTip(
            "PatchLab library result · "
            f"{'Serum 1' if item['synth'] == 'serum1' else 'Serum 2'}"
        )
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
                    lambda _checked=False, detail=dict(item), n=note: self._play_existing_match(
                        detail, note=n
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
        self._existing_matches = existing
        self._favorite_hashes = self._load_favorite_hashes()
        self._render_existing_matches()

        recommendation = result.get("recommendation")
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
            self.load_in_serum_button.setEnabled(False)
            self.recommendation_more_button.setEnabled(False)
            self.match_stats.setText(
                str(result.get("message", "No confident match"))
            )
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
        base_name = generated_preset_name(synth_key)
        self.recommendation_badge.setText(confidence_label.upper())
        self.recommendation_badge.setStyleSheet(f"color: {color};")
        self.recommendation_name.setText(base_name)
        category = "Generated"
        self.recommendation_subtitle.setText(f"PatchLab · {synth_name}")
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
            "PatchLab generated preset · "
            f"{recommendation['evaluations']} evaluations · "
            f"{float(recommendation['elapsed_s']):.1f}s"
        )
        self.octave_selector.setEnabled(
            bool(
                recommendation.get("winner_audio_path")
                or recommendation.get("preview_source_path")
            )
        )
        export_available = bool(recommendation.get("export_available", True))
        self.save_preset_button.setEnabled(export_available)
        self.load_in_serum_button.setEnabled(export_available)
        self.recommendation_more_button.setEnabled(bool(recommendation.get("settings")))
        self.parameter_strip.setVisible(False)
        self.settings_tree.setVisible(False)
        self._octave_changed(self.octave_selector.currentIndex())
        self.match_stats.setText(str(result.get("message", "Match complete")))
        self.statusBar().showMessage("Match complete")
        self.match_card_status.setText(f"{similarity:.1f}% · ready")
        if self._batch_state is not None:
            self.save_preset_button.setEnabled(False)
            self.load_in_serum_button.setEnabled(False)

    def _scan_completed(self, summary: dict) -> None:
        super()._scan_completed(summary)
        self.scan_card_status.setText(
            f"{summary.get('found', 0):,} found · {summary.get('failed', 0):,} failed"
        )

    def _render_progress_changed(self, detail: dict) -> None:
        super()._render_progress_changed(detail)
        current = int(detail.get("completed_note_pairs", 0))
        total = max(int(detail.get("total_note_pairs", 1)), 1)
        self.render_card_status.setText(f"{current:,} / {total:,} rendered")

    def _render_completed(self, summary: dict) -> None:
        super()._render_completed(summary)
        self.render_card_status.setText("Library rendered ✓")

    def _analyze_progress_changed(self, detail: dict) -> None:
        super()._analyze_progress_changed(detail)
        self.learn_card_status.setText(
            str(detail.get("phase", "analyzing")).replace("-", " ").title()
        )

    def _analyze_completed(self, summary: dict) -> None:
        super()._analyze_completed(summary)
        self.learn_card_status.setText("Model trained ✓")

    def _match_progress_changed(self, detail: dict) -> None:
        super()._match_progress_changed(detail)
        evaluations = int(detail.get("evaluations", 0))
        budget = int(detail.get("budget", 0))
        if budget:
            self.match_card_status.setText(f"{evaluations} / {budget} evaluations")
        else:
            self.match_card_status.setText(
                str(detail.get("phase", "matching")).replace("-", " ").title()
            )
