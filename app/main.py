#!/usr/bin/env python3
"""PatchLab application entry point."""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.ui import MainWindow  # noqa: E402
from core.audio_lifecycle import cleanup_stale_match_scratch  # noqa: E402
from core.factory_verify import verify_local_factory_install  # noqa: E402
from core.platform_env import ENV  # noqa: E402,F401
from core.privacy import distribution_mode  # noqa: E402


def main() -> int:
    mp.set_start_method("spawn", force=True)
    application = QApplication(sys.argv)
    application.setApplicationName("PatchLab")
    factory_verification = None
    if distribution_mode():
        cleanup_stale_match_scratch()
        factory_verification = verify_local_factory_install(
            mapping_path=ENV.app_data_dir / "factory-paths.json"
        )
    window = MainWindow(factory_verification=factory_verification)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
