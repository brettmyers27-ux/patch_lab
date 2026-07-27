#!/usr/bin/env python3
"""PatchLab application entry point."""

from __future__ import annotations

import multiprocessing as mp
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.access_dialog import PasscodeDialog  # noqa: E402
from app.ui import MainWindow  # noqa: E402
from core.access_gate import AccessManager  # noqa: E402
from core.audio_lifecycle import cleanup_stale_match_scratch  # noqa: E402
from core.factory_verify import verify_local_factory_install  # noqa: E402
from core.platform_env import ENV  # noqa: E402,F401
from core.privacy import distribution_mode  # noqa: E402


def _run_packaged_host_probe(output_path: str) -> int:
    """Exercise bundled DawDreamer against installed plugins for release QA."""
    from core.plugin_host import SILENCE_DBFS, verify_default_render

    report: dict[str, object] = {"distribution_mode": distribution_mode(), "synths": {}}
    passed = True
    for synth in ("serum1", "serum2"):
        attempts: list[dict[str, object]] = []
        synth_passed = False
        for candidate in ENV.plugins_for(synth):
            try:
                peak, rms, methods = verify_default_render(candidate)
                attempt = {
                    "format": candidate.format,
                    "path": str(candidate.path),
                    "peak_dbfs": peak,
                    "rms_dbfs": rms,
                    "state_methods": methods,
                    "non_silent": rms > SILENCE_DBFS,
                }
                attempts.append(attempt)
                if attempt["non_silent"]:
                    synth_passed = True
                    break
            except Exception as exc:
                attempts.append(
                    {
                        "format": candidate.format,
                        "path": str(candidate.path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        report["synths"][synth] = {"passed": synth_passed, "attempts": attempts}
        passed = passed and synth_passed
    report["gate_pass"] = passed
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if passed else 1


def main() -> int:
    mp.set_start_method("spawn", force=True)
    application = QApplication(sys.argv)
    application.setApplicationName("PatchLab")
    if probe_path := os.environ.get("PATCHLAB_PACKAGED_HOST_PROBE"):
        return _run_packaged_host_probe(probe_path)
    factory_verification = None
    if distribution_mode():
        access = AccessManager()
        if access.needs_prompt():
            dialog = PasscodeDialog(access)
            if dialog.exec() != dialog.DialogCode.Accepted:
                return 0
        cleanup_stale_match_scratch()
        factory_verification = verify_local_factory_install(
            mapping_path=ENV.app_data_dir / "factory-paths.json"
        )
    window = MainWindow(factory_verification=factory_verification)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
