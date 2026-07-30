#!/usr/bin/env python3
"""One-command, copy-pasteable Windows 11 installation and parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_launcher_config_path = PROJECT_ROOT / ".patchlab-launcher.json"
if _launcher_config_path.is_file():
    _launcher_config = json.loads(
        _launcher_config_path.read_text(encoding="utf-8-sig")
    )
    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    os.environ.setdefault(
        "PATCHLAB_RELAY_URL", str(_launcher_config["relay_url"])
    )
    os.environ.setdefault(
        "PATCHLAB_MODEL_CACHE", str(_launcher_config["model_cache"])
    )

from core.access_gate import AccessStore  # noqa: E402
from core.factory_bundle import DEFAULT_FACTORY_BUNDLE, FactoryBundle  # noqa: E402
from core.factory_match import run_factory_match_file  # noqa: E402
from core.factory_verify import verify_local_factory_install  # noqa: E402
from core.features import ClapEmbedder, load_audio_48k_mono  # noqa: E402
from core.platform_env import ENV  # noqa: E402
from core.plugin_host import (  # noqa: E402
    dump_dawdreamer_parameters,
    make_dawdreamer_processor,
)
from core.preset_scan import sha1_file  # noqa: E402


REFERENCE_PATH = PROJECT_ROOT / "core" / "reference" / "macos_plugin_parity.json"
Status = Literal["PASS", "FAIL", "WARN", "SKIP"]


@dataclass(frozen=True, slots=True)
class Result:
    check: str
    status: Status
    detail: str


def _signature(parameters: list[Any]) -> str:
    value = "\n".join(f"{item.index}\0{item.name}" for item in parameters)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compare_parameter_dump(
    synth: str,
    live: list[Any],
    reference: dict[str, Any],
) -> tuple[Result, dict[str, Any]]:
    expected = list(reference["parameters"])
    live_by_index = {int(item.index): item for item in live}
    name_mismatches: list[dict[str, Any]] = []
    value_mismatches: list[dict[str, Any]] = []
    tolerance = 1e-4
    for expected_item in expected:
        index = int(expected_item["index"])
        item = live_by_index.get(index)
        if item is None or item.name != expected_item["name"]:
            name_mismatches.append(
                {
                    "index": index,
                    "macos": expected_item["name"],
                    "windows": None if item is None else item.name,
                }
            )
            continue
        delta = abs(
            float(item.norm_value) - float(expected_item["normalized_value"])
        )
        if delta > tolerance:
            value_mismatches.append(
                {
                    "index": index,
                    "name": item.name,
                    "macos": expected_item["normalized_value"],
                    "windows": item.norm_value,
                    "delta": delta,
                }
            )
    count_equal = len(live) == int(reference["parameter_count"])
    signature_equal = _signature(live) == reference["index_name_sha256"]
    mismatch_fraction = len(value_mismatches) / max(1, len(expected))
    passed = (
        count_equal
        and signature_equal
        and not name_mismatches
        and mismatch_fraction <= 0.05
    )
    detail = (
        f"count macOS-reference={len(expected)} live-host={len(live)}; "
        f"index/name mismatches={len(name_mismatches)}; "
        f"current-value mismatches={len(value_mismatches)} "
        f"({100.0 * mismatch_fraction:.2f}%)"
    )
    return (
        Result(f"{synth} parameter index/name/value parity", "PASS" if passed else "FAIL", detail),
        {
            "count_equal": count_equal,
            "signature_equal": signature_equal,
            "name_mismatch_count": len(name_mismatches),
            "value_mismatch_count": len(value_mismatches),
            "value_mismatch_fraction": mismatch_fraction,
            "name_mismatches": name_mismatches[:50],
            "value_mismatches": value_mismatches[:50],
        },
    )


def _format_table(results: list[Result]) -> str:
    widths = (
        max(6, *(len(item.status) for item in results)),
        max(5, *(len(item.check) for item in results)),
    )
    lines = [
        f"{'STATUS':<{widths[0]}}  {'CHECK':<{widths[1]}}  DETAIL",
        f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * 60}",
    ]
    for item in results:
        lines.append(
            f"{item.status:<{widths[0]}}  {item.check:<{widths[1]}}  "
            f"{item.detail.replace(chr(10), ' | ')}"
        )
    return "\n".join(lines)


def _required_plugin(synth: str, required_format: str) -> Any:
    return next(
        item
        for item in ENV.plugins_for(synth)
        if item.format == required_format and item.hostable
    )


def _dump_parameters(
    reference: dict[str, Any], output_dir: Path
) -> tuple[list[Result], dict[str, Any]]:
    results: list[Result] = []
    reports: dict[str, Any] = {}
    for synth, required_format in (("serum1", "VST2"), ("serum2", "VST3")):
        try:
            candidate = _required_plugin(synth, required_format)
            _engine, processor = make_dawdreamer_processor(candidate)
            parameters = dump_dawdreamer_parameters(processor)
            result, report = compare_parameter_dump(
                synth, parameters, reference["plugins"][synth]
            )
            reports[synth] = {
                **report,
                "plugin_path": str(candidate.path),
                "parameter_count": len(parameters),
                "index_name_sha256": _signature(parameters),
                "parameters": [
                    {
                        "index": item.index,
                        "name": item.name,
                        "normalized_value": item.norm_value,
                        "display_value": item.display_value,
                    }
                    for item in parameters
                ],
            }
            results.append(result)
        except Exception as exc:
            results.append(
                Result(
                    f"{synth} parameter index/name/value parity",
                    "FAIL",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            reports[synth] = {"error": f"{type(exc).__name__}: {exc}"}
    (output_dir / "windows-parameter-dump.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results, reports


def _find_fixture_source(preset: Any) -> Path:
    suffix = ".fxp" if preset.synth == "serum1" else ".serumpreset"
    for root in ENV.factory_roots_for(preset.synth, existing_only=True):
        direct = root / Path(preset.relative_path)
        if direct.is_file() and sha1_file(direct) == preset.content_hash:
            return direct
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.casefold() == suffix
                and sha1_file(path) == preset.content_hash
            ):
                return path.resolve()
    raise FileNotFoundError(
        f"{preset.name} ({preset.content_hash}) was not found under the "
        f"resolved {preset.synth} factory roots"
    )


def _audio_parity(
    reference: dict[str, Any], output_dir: Path
) -> tuple[list[Result], dict[str, Any], Path | None, dict[str, str]]:
    from scripts.render_factory_preview import render_preview

    details: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    local_paths: dict[str, str] = {}
    bundle = FactoryBundle(DEFAULT_FACTORY_BUNDLE)
    midi_note = int(reference["factory_fixture_midi_note"])
    fixtures: list[tuple[Any, np.ndarray]] = []
    render_root = output_dir / "parity-current"
    try:
        for preset_id in reference["factory_fixture_ids"]:
            preset = bundle.preset_by_id(int(preset_id))
            fixtures.append(
                (preset, bundle.note_embedding(preset.id, midi_note))
            )
            source = _find_fixture_source(preset)
            local_paths[preset.content_hash] = str(source)
            cached = (
                render_root
                / "factory-previews"
                / preset.content_hash
                / f"{midi_note}.wav"
            )
            cached.unlink(missing_ok=True)
            preview_paths.append(
                render_preview(
                    source,
                    preset.synth,
                    midi_note,
                    preset.content_hash,
                    output_root=render_root,
                )
            )
        embedder = ClapEmbedder(ENV)
        actual = embedder.embed(
            [load_audio_48k_mono(path).waveform for path in preview_paths]
        )
        for (preset, expected), path, embedding in zip(
            fixtures, preview_paths, actual, strict=True
        ):
            cosine = float(np.dot(expected, embedding))
            details.append(
                {
                    "preset_id": preset.id,
                    "name": preset.name,
                    "synth": preset.synth,
                    "wav_path": str(path),
                    "clap_cosine": cosine,
                }
            )
        values = [float(item["clap_cosine"]) for item in details]
        minimum = min(values)
        mean = float(np.mean(values))
        thresholds = reference["thresholds"]
        passed = (
            minimum >= float(thresholds["minimum_fixture_clap_cosine"])
            and mean >= float(thresholds["minimum_mean_clap_cosine"])
        )
        result = Result(
            "factory render CLAP parity",
            "PASS" if passed else "FAIL",
            f"fixtures={len(values)}; min={minimum:.4f}; mean={mean:.4f}; "
            + ", ".join(
                f"{item['synth']}#{item['preset_id']}={item['clap_cosine']:.4f}"
                for item in details
            ),
        )
        return [result], {"fixtures": details, "minimum": minimum, "mean": mean}, preview_paths[0], local_paths
    except Exception as exc:
        return (
            [Result("factory render CLAP parity", "FAIL", f"{type(exc).__name__}: {exc}")],
            {"fixtures": details, "error": f"{type(exc).__name__}: {exc}"},
            preview_paths[0] if preview_paths else None,
            local_paths,
        )


def _credential_and_first_run() -> list[Result]:
    results: list[Result] = []
    store = AccessStore()
    state = store.load()
    passcode_saved = bool(store.passcode())
    results.append(
        Result(
            "Windows Credential Manager",
            "PASS" if state.authenticated_once and passcode_saved else "FAIL",
            f"authenticated marker={state.authenticated_once}; "
            f"passcode retrievable through keyring={passcode_saved}; secret not printed",
        )
    )
    process = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_first_run.py")],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    line = next(
        (value for value in process.stdout.splitlines() if value.startswith("FIRST_RUN_GATE=")),
        "",
    )
    passed = process.returncode == 0 and bool(line)
    results.append(
        Result(
            "passcode/consent once-only flow",
            "PASS" if passed else "FAIL",
            line[-500:] if line else (process.stderr.strip() or "no gate output"),
        )
    )
    return results


def _playback_probe(audio_path: Path | None) -> Result:
    try:
        import sounddevice as sd
        import soundfile as sf

        device = sd.query_devices(kind="output")
        if audio_path is None:
            raise FileNotFoundError("No rendered preview was available for playback")
        audio, sample_rate = sf.read(
            audio_path, dtype="float32", always_2d=True
        )
        probe = audio[: min(len(audio), max(1, int(sample_rate * 0.15)))]
        # A disconnected or sleeping Windows output device can leave
        # ``blocking=True`` waiting forever. Start playback asynchronously,
        # keep the probe alive for a bounded interval, then stop explicitly.
        sd.play(probe, samplerate=sample_rate, blocking=False)
        sd.sleep(max(200, int(1000 * len(probe) / sample_rate) + 50))
        sd.stop()
        return Result(
            "source/preview sounddevice playback",
            "PASS",
            f"played {audio_path.name} through: {device.get('name', 'unknown')}",
        )
    except Exception as exc:
        return Result(
            "source/preview sounddevice playback",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
        )


def _shortcut_probe() -> Result:
    # pywin32 is deliberately not an app dependency; use Windows' built-in COM
    # bridge through PowerShell to inspect both links.
    script = (
        "$s=New-Object -ComObject WScript.Shell;"
        "$d=[Environment]::GetFolderPath('Desktop');"
        "$p=[Environment]::GetFolderPath('Programs');"
        "$paths=@((Join-Path $d 'PatchLab.lnk'),"
        "(Join-Path $p 'PatchLab\\PatchLab.lnk'));"
        "$rows=@();foreach($x in $paths){"
        "if(Test-Path -LiteralPath $x){$l=$s.CreateShortcut($x);"
        "$rows+=@{path=$x;target=$l.TargetPath;args=$l.Arguments;icon=$l.IconLocation}}};"
        "$rows|ConvertTo-Json -Compress"
    )
    try:
        process = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        rows = json.loads(process.stdout or "[]")
        if isinstance(rows, dict):
            rows = [rows]
        passed = (
            process.returncode == 0
            and len(rows) == 2
            and all(str(row["target"]).casefold().endswith("pythonw.exe") for row in rows)
            and all("windows_launcher.pyw" in str(row["args"]) for row in rows)
            and all("patchlab.ico" in str(row["icon"]).casefold() for row in rows)
        )
        return Result(
            "Desktop/Start shortcuts",
            "PASS" if passed else "FAIL",
            json.dumps(rows, separators=(",", ":")),
        )
    except Exception as exc:
        return Result("Desktop/Start shortcuts", "FAIL", f"{type(exc).__name__}: {exc}")


def _ui_probe(output_dir: Path) -> Result:
    try:
        from PySide6.QtWidgets import QApplication

        from app.ui import MainWindow
        from core.privacy import PrivacyStore, distribution_mode

        application = QApplication.instance() or QApplication([])
        window = MainWindow(factory_verification=None)
        window.show()
        for _ in range(10):
            application.processEvents()
            time.sleep(0.05)
        screenshot = output_dir / "windows-ui.png"
        saved = window.grab().save(str(screenshot))
        ratio = window.width() / max(1, window.height())
        consent_persisted = (
            not distribution_mode()
            or PrivacyStore().load().use_and_share_own_presets is not None
        )
        window.close()
        passed = (
            saved
            and abs(ratio - (16 / 9)) < 0.02
            and consent_persisted
        )
        return Result(
            "Windows UI render",
            "PASS" if passed else "FAIL",
            f"screenshot={screenshot}; size={window.width()}x{window.height()}; "
            f"ratio={ratio:.5f}; consent choice persisted={consent_persisted}",
        )
    except Exception as exc:
        return Result("Windows UI render", "FAIL", f"{type(exc).__name__}: {exc}")


def _match_and_export_probe(
    target: Path | None, output_dir: Path
) -> tuple[Result, dict[str, Any]]:
    if target is None:
        return Result("real match + preset export", "FAIL", "No parity fixture rendered"), {}
    try:
        verification = verify_local_factory_install(
            mapping_path=output_dir / "factory-paths.json"
        )
        result_path = run_factory_match_file(
            target,
            target_synth="serum1",
            mapping_path=output_dir / "factory-paths.json",
            session_root=output_dir / "match",
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        closest = payload["existing_matches"][0]
        recommendation = payload["recommendation"]
        extension = ".fxp" if recommendation["synth"] == "serum1" else ".SerumPreset"
        export_path = output_dir / f"windows-diagnostic-export{extension}"
        process = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "export_match.py"),
                str(result_path),
                str(export_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        export_ok = process.returncode == 0 and export_path.is_file()
        expected_hash = target.parent.name
        own_preset_top = str(closest["content_hash"]) == expected_hash
        passed = (
            verification.matched_hashes > 0
            and own_preset_top
            and float(closest["similarity"]) >= 0.80
            and export_ok
        )
        report = {
            "result_path": str(result_path),
            "closest": closest,
            "recommendation": recommendation,
            "export_path": str(export_path),
            "export_output": process.stdout.strip(),
            "matched_factory_hashes": verification.matched_hashes,
            "expected_hash": expected_hash,
            "own_preset_top": own_preset_top,
        }
        return (
            Result(
                "real match + preset export",
                "PASS" if passed else "FAIL",
                f"top={closest['name']} score={float(closest['similarity']):.4f}; "
                f"own preset top={own_preset_top}; "
                f"factory hashes={verification.matched_hashes}; export={export_ok}",
            ),
            report,
        )
    except Exception as exc:
        return (
            Result("real match + preset export", "FAIL", f"{type(exc).__name__}: {exc}"),
            {"error": f"{type(exc).__name__}: {exc}"},
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--installer-gate",
        action="store_true",
        help="Skip checks that require shortcuts to have been created already.",
    )
    parser.add_argument(
        "--macos-reference-check",
        action="store_true",
        help="Maintainer-only calibration; never counts as Windows verification.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the diagnostic output directory (primarily for isolated tests).",
    )
    args = parser.parse_args()
    if not REFERENCE_PATH.is_file():
        print(f"FAIL: missing macOS parameter reference {REFERENCE_PATH}")
        return 1
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    output_dir = (
        args.output_dir
        or (
            PROJECT_ROOT / "data" / "models" / "windows-parity-calibration"
            if args.macos_reference_check
            else ENV.app_data_dir / "diagnostics"
        )
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    report: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "reference": str(REFERENCE_PATH),
        "installer_gate": args.installer_gate,
        "macos_reference_check": args.macos_reference_check,
    }

    actual_windows = platform.system() == "Windows"
    if not actual_windows and not args.macos_reference_check:
        results.append(
            Result(
                "real Windows 11 hardware",
                "FAIL",
                f"detected {platform.system()}/{platform.machine()}; this gate cannot be simulated",
            )
        )
    elif actual_windows:
        build = int(sys.getwindowsversion().build)
        passed = build >= 22000 and platform.machine().casefold() in {"amd64", "x86_64"}
        results.append(
            Result(
                "real Windows 11 x64 hardware",
                "PASS" if passed else "FAIL",
                f"build={build}; machine={platform.machine()}",
            )
        )
    else:
        results.append(
            Result(
                "macOS reference calibration only",
                "WARN",
                "This validates the macOS baseline, not Windows parity.",
            )
        )

    parameter_results, parameter_report = _dump_parameters(reference, output_dir)
    results.extend(parameter_results)
    report["parameters"] = parameter_report
    audio_results, audio_report, target, _local_paths = _audio_parity(reference, output_dir)
    results.extend(audio_results)
    report["audio_parity"] = audio_report
    results.extend(_credential_and_first_run() if actual_windows else [])
    match_result, match_report = _match_and_export_probe(target, output_dir)
    results.append(match_result)
    report["match_export"] = match_report
    if args.installer_gate:
        results.extend(
            (
                Result("Desktop/Start shortcuts", "SKIP", "created after this installer gate"),
                Result(
                    "source/preview sounddevice playback",
                    "SKIP",
                    "run the full diagnostic after install",
                ),
                Result("Windows UI render", "SKIP", "run the full diagnostic after install"),
            )
        )
    elif actual_windows:
        results.extend((_shortcut_probe(), _playback_probe(target), _ui_probe(output_dir)))

    report["results"] = [asdict(item) for item in results]
    report["gate_pass"] = not any(item.status == "FAIL" for item in results)
    report_path = output_dir / "windows-install-diagnostic.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("PatchLab — Windows 11 parity diagnostic")
    print(_format_table(results))
    print(f"\nFull machine-readable report: {report_path}")
    print(
        "\nWINDOWS_PARITY_RESULT="
        + json.dumps(
            {
                "gate_pass": report["gate_pass"],
                "report": str(report_path),
                "failed": [item.check for item in results if item.status == "FAIL"],
            },
            separators=(",", ":"),
        )
    )
    if not actual_windows:
        print(
            "\nIMPORTANT: This run did not occur on Windows. The port remains UNPROVEN "
            "until the same script passes on the licensed Windows 11 machine."
        )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
