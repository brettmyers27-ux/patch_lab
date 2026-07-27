#!/usr/bin/env python3
"""Milestone 0 environment and default-render gate."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.platform_env import ENV  # noqa: E402
from core.verify import (  # noqa: E402
    CheckResult,
    compute_check,
    format_table,
    import_checks,
    plugin_path_checks,
    preset_root_checks,
    python_check,
)


def main() -> int:
    env = ENV
    results = [
        CheckResult("platform branch", "PASS", f"{env.system_name}/{env.machine} -> {env.branch}"),
        python_check(),
    ]
    results.extend(import_checks())
    results.append(compute_check(env))

    try:
        import dawdreamer as daw

        engine = daw.RenderEngine(44_100, 512)
        results.append(CheckResult("DawDreamer engine", "PASS", repr(engine)))
    except Exception as exc:
        results.append(CheckResult("DawDreamer engine", "FAIL", repr(exc)))

    results.extend(plugin_path_checks(env))
    results.extend(preset_root_checks(env))

    try:
        from core.plugin_host import SILENCE_DBFS, verify_default_render

        for synth in ("serum1", "serum2"):
            errors: list[str] = []
            passed = False
            for candidate in env.plugins_for(synth):
                try:
                    peak, rms, methods = verify_default_render(candidate)
                    status = "PASS" if rms > SILENCE_DBFS else "FAIL"
                    detail = (
                        f"{candidate.format} {candidate.path}; peak={peak:.2f} dBFS, "
                        f"RMS={rms:.2f} dBFS; state API={methods}"
                    )
                    results.append(CheckResult(f"{synth} init render", status, detail))
                    passed = status == "PASS"
                    if passed:
                        break
                except Exception as exc:
                    errors.append(f"{candidate.format} {candidate.path}: {exc!r}")
            if not passed and not any(item.check == f"{synth} init render" for item in results):
                results.append(
                    CheckResult(f"{synth} init render", "FAIL", " | ".join(errors) or "No plugin found")
                )
    except Exception as exc:
        results.append(CheckResult("plugin render checks", "FAIL", repr(exc)))

    print("Patch Lab — Milestone 0 environment gate")
    print(format_table(results))
    print(f"\nDetected compute backend: {env.compute_backend}")
    if env.compute_warning:
        print(f"WARNING: {env.compute_warning}")
    print(f"Torch install command for this machine:\n{env.torch_install_command}")

    failed = [item for item in results if item.failed]
    print(f"\nGATE: {'FAIL' if failed else 'PASS'} ({len(failed)} failing checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
