#!/usr/bin/env python3
"""Milestone 0 empirical preset-state loading gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.platform_env import ENV, SynthVersion  # noqa: E402


def discover(roots: list[Path], extension: str) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file() and root.name.lower().endswith(extension.lower()):
            found.append(root.resolve())
        elif root.is_dir():
            found.extend(path.resolve() for path in root.rglob("*") if path.name.lower().endswith(extension.lower()))
    return sorted(set(found), key=lambda path: str(path).casefold())


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional preset files or roots. Defaults to existing platform preset roots.",
    )
    value.add_argument("--_child", nargs=2, metavar=("SYNTH", "PRESET"), help=argparse.SUPPRESS)
    return value


def main() -> int:
    try:
        from core.plugin_host import (
            PresetLoadError,
            inspect_preset_bytes,
            load_preset,
        )
    except ImportError as exc:
        print("Patch Lab — Milestone 0 preset-load spike")
        print(f"GATE: FAIL — missing runtime dependency: {exc}")
        print("Activate the Python 3.11 venv and install requirements.txt first.")
        return 1

    args = parser().parse_args()
    if args._child:
        synth, raw_path = args._child
        path = Path(raw_path)
        try:
            loaded = load_preset(ENV, synth, path)
            result = {
                "ok": True,
                "strategy": loaded.strategy,
                "plugin_format": loaded.plugin_format,
                "plugin_path": str(loaded.plugin_path),
                "rms_dbfs": loaded.rms_dbfs,
                "parameters": [
                    {"index": item.index, "name": item.name, "value": item.norm_value}
                    for item in loaded.parameters
                ],
                "attempts": [
                    {
                        "strategy": item.strategy,
                        "plugin_path": item.plugin_path,
                        "passed": item.passed,
                        "detail": item.detail,
                    }
                    for item in loaded.attempts
                ],
            }
        except PresetLoadError as exc:
            result = {
                "ok": False,
                "error": str(exc),
                "attempts": [
                    {
                        "strategy": item.strategy,
                        "plugin_path": item.plugin_path,
                        "passed": item.passed,
                        "detail": item.detail,
                    }
                    for item in exc.attempts
                ],
            }
        print("PATCHLAB_CHILD_RESULT=" + json.dumps(result), flush=True)
        return 0 if result["ok"] else 1

    roots = args.paths or list(ENV.existing_preset_roots)
    all_fxp = discover(roots, ".fxp")
    serum1_fxp = [path for path in all_fxp if "serum 2 presets" not in str(path).casefold()]
    samples = {
        "serum1": (serum1_fxp or all_fxp)[:3],
        "serum2": discover(roots, ".serumpreset")[:3],
    }

    print("Patch Lab — Milestone 0 preset-load spike")
    print(f"Search roots: {[str(path) for path in roots]}")
    print("\nSerum 2 byte inspection:")
    for path in samples["serum2"]:
        print(json.dumps(inspect_preset_bytes(path), indent=2))

    rows: list[tuple[str, str, str, str, str]] = []
    failures: dict[SynthVersion, list[str]] = {"serum1": [], "serum2": []}
    loaded_by_synth: dict[SynthVersion, list[dict[str, object]]] = {"serum1": [], "serum2": []}
    for synth in ("serum1", "serum2"):
        for path in samples[synth]:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--_child", synth, str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            marker = next(
                (
                    line.removeprefix("PATCHLAB_CHILD_RESULT=")
                    for line in completed.stdout.splitlines()
                    if line.startswith("PATCHLAB_CHILD_RESULT=")
                ),
                None,
            )
            if marker is not None:
                result = json.loads(marker)
            else:
                result = {
                    "ok": False,
                    "error": (
                        f"native host terminated child with exit code {completed.returncode}; "
                        f"stdout={completed.stdout.strip()!r}; stderr={completed.stderr.strip()!r}"
                    ),
                    "attempts": [],
                }

            if result["ok"]:
                loaded_by_synth[synth].append(result)
                parameters = result["parameters"]
                nonzero = sum(abs(float(item["value"])) > 1e-8 for item in parameters)
                rows.append(
                    (
                        synth,
                        path.name,
                        f"{result['plugin_format']}/{result['strategy']}",
                        str(nonzero),
                        f"{float(result['rms_dbfs']):.2f}",
                    )
                )
            else:
                detail = "; ".join(
                    f"{item['strategy']}@{item['plugin_path']}: {item['detail']}"
                    for item in result.get("attempts", [])
                )
                failures[synth].append(
                    f"{path}: {result.get('error', 'unknown error')} Tried: {detail or 'native strategy aborted'}"
                )
                rows.append((synth, path.name, "FAILED", "-", "-"))

    headers = ("SYNTH", "FILE", "VERIFIED FORMAT/STRATEGY", "NONZERO PARAMS", "RMS DBFS")
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i]) for i in range(5)]
    print("\n" + "  ".join(f"{headers[i]:<{widths[i]}}" for i in range(5)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(f"{row[i]:<{widths[i]}}" for i in range(5)))

    gate_failed = False
    serum2_supported = False
    for synth in ("serum1", "serum2"):
        loaded = loaded_by_synth[synth]
        if len(samples[synth]) < 3:
            print(f"\nFAIL {synth}: found {len(samples[synth])}/3 required real preset samples.")
            gate_failed = True
            continue
        def vector(result: dict[str, object]) -> dict[int, float]:
            return {int(item["index"]): float(item["value"]) for item in result["parameters"]}

        cross_diff = False
        if len(loaded) >= 2:
            first = vector(loaded[0])
            cross_diff = any(
                any(abs(first[index] - candidate[index]) > 1e-4 for index in first.keys() & candidate.keys())
                for candidate in (vector(other) for other in loaded[1:])
            )
        if synth == "serum1" and (len(loaded) != 3 or not cross_diff):
            print(f"\nFAIL serum1: {len(loaded)}/3 loaded; cross-preset vector difference={cross_diff}.")
            for failure in failures[synth]:
                print(f"  {failure}")
            gate_failed = True
        elif synth == "serum2" and (len(loaded) != 3 or not cross_diff):
            print("\nSERUM 2 STRATEGY REPORT: no strategy passed all mandatory checks.")
            print("Serum 1-only continuation is allowed after the Serum 1 gate passes.")
            for failure in failures[synth]:
                print(f"  {failure}")
        else:
            print(f"\nPASS {synth}: 3/3 loaded and preset vectors differ.")
            if synth == "serum2":
                serum2_supported = True

    if gate_failed:
        gate_label = "FAIL"
    elif serum2_supported:
        gate_label = "PASS (SERUM 1 + SERUM 2)"
    else:
        gate_label = "PASS (SERUM 1 ONLY; SERUM 2 DISABLED)"
    print(f"\nGATE: {gate_label}")
    capability_path = PROJECT_ROOT / "data" / "strategy_capabilities.json"
    capability_path.parent.mkdir(parents=True, exist_ok=True)
    supported: dict[str, dict[str, object]] = {}
    for synth, loaded in loaded_by_synth.items():
        if loaded:
            supported[synth] = {
                "strategy": loaded[0].get("strategy"),
                "plugin_format": loaded[0].get("plugin_format"),
                "plugin_path": loaded[0].get("plugin_path"),
                "verified_samples": len(loaded),
            }
    capability_path.write_text(
        json.dumps(
            {
                "gate": gate_label,
                "supported": supported,
                "disabled": [synth for synth in ("serum1", "serum2") if synth not in supported],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Capability manifest: {capability_path}")
    return 1 if gate_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
