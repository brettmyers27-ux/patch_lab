"""Reusable Milestone gate checks."""

from __future__ import annotations

import importlib
import json
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.platform_env import PlatformEnv


Status = Literal["PASS", "FAIL", "WARN"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    check: str
    status: Status
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


REQUIRED_IMPORTS = (
    "numpy",
    "soundfile",
    "librosa",
    "torch",
    "torchaudio",
    "dawdreamer",
    "pedalboard",
    "PySide6",
    "laion_clap",
    "sounddevice",
    "imageio_ffmpeg",
    "mido",
)


def python_check() -> CheckResult:
    version = sys.version_info
    status: Status = "PASS" if version[:2] == (3, 11) else "FAIL"
    return CheckResult(
        "Python 3.11",
        status,
        f"{version.major}.{version.minor}.{version.micro} at {sys.executable}",
    )


def import_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "version unavailable")
            results.append(CheckResult(f"import {module_name}", "PASS", str(version)))
        except Exception as exc:
            results.append(CheckResult(f"import {module_name}", "FAIL", repr(exc)))
    return results


def compute_check(env: PlatformEnv) -> CheckResult:
    try:
        import torch
    except ImportError as exc:
        return CheckResult("compute backend", "FAIL", repr(exc))

    if env.branch == "windows":
        cuda_selected = "/whl/cu128" in env.torch_install_command
        if cuda_selected:
            if not torch.cuda.is_available():
                return CheckResult(
                    "Windows CUDA",
                    "FAIL",
                    "NVIDIA hardware selected cu128, but torch.cuda.is_available() is false",
                )
            name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            return CheckResult(
                "Windows CUDA",
                "PASS",
                f"{name}; capability sm_{capability[0]}{capability[1]}",
            )
        return CheckResult(
            "Windows CPU compute",
            "PASS",
            "No NVIDIA adapter was selected; CPU-only PyTorch is active",
        )

    if torch.backends.mps.is_available():
        return CheckResult("Apple MPS", "PASS", "torch.backends.mps.is_available() is true")
    return CheckResult(
        "Apple MPS",
        "WARN",
        "MPS unavailable; CPU fallback selected with PYTORCH_ENABLE_MPS_FALLBACK=1",
    )


def plugin_path_checks(env: PlatformEnv) -> list[CheckResult]:
    results: list[CheckResult] = []
    for synth, required_format in (("serum1", "VST2"), ("serum2", "VST3")):
        searched = tuple(
            item
            for item in env.plugin_candidates
            if item.synth == synth and item.format == required_format and item.hostable
        )
        found = tuple(item for item in searched if item.exists)
        detail = ", ".join(f"{item.format}: {item.path}" for item in found)
        if not found:
            locations = "\n  - ".join(str(item.path) for item in searched)
            detail = (
                f"Required {required_format} binary was not found. Searched:\n  - {locations}"
                if locations
                else f"No {required_format} search locations were resolved"
            )
        results.append(
            CheckResult(
                f"{synth} {required_format} render binary",
                "PASS" if found else "FAIL",
                detail,
            )
        )
    return results


def preset_root_checks(env: PlatformEnv) -> list[CheckResult]:
    return [
        CheckResult(
            f"preset root {index}",
            "PASS" if path.exists() else "WARN",
            str(path),
        )
        for index, path in enumerate(env.preset_roots, start=1)
    ]


def library_scan_checks(db_path: Path) -> list[CheckResult]:
    """Audit the persisted Milestone 1 gate without loading a plugin."""

    path = Path(db_path).resolve()
    if not path.is_file():
        return [CheckResult("library database", "FAIL", f"Missing: {path}")]

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        status_rows = connection.execute(
            "SELECT synth,status,COUNT(*) AS count FROM presets "
            "GROUP BY synth,status ORDER BY synth,status"
        ).fetchall()
        counts = {
            (str(row["synth"]), str(row["status"])): int(row["count"])
            for row in status_rows
        }
        synth_totals = {
            str(row["synth"]): int(row["count"])
            for row in connection.execute(
                "SELECT synth,COUNT(*) AS count FROM presets GROUP BY synth"
            ).fetchall()
        }
        still_scanned = sum(
            count for (synth, status), count in counts.items() if status == "scanned"
        )

        cardinality_rows = connection.execute(
            "SELECT synth,COUNT(*) AS presets,MIN(n) AS minimum,MAX(n) AS maximum FROM "
            "(SELECT p.synth,p.id,COUNT(pa.param_index) AS n FROM presets p "
            "JOIN params pa ON pa.preset_id=p.id "
            "GROUP BY p.synth,p.id) GROUP BY synth ORDER BY synth"
        ).fetchall()
        cardinalities = {str(row["synth"]): row for row in cardinality_rows}
        passed = int(cardinalities.get("serum1")["presets"]) if cardinalities.get("serum1") else 0
        serum2_passed = (
            int(cardinalities.get("serum2")["presets"]) if cardinalities.get("serum2") else 0
        )
        total_supported = synth_totals.get("serum1", 0)
        failed = total_supported - passed
        failure_rate = failed / total_supported if total_supported else 1.0
        serum2_total = synth_totals.get("serum2", 0)
        serum2_failed = serum2_total - serum2_passed
        serum2_failure_rate = serum2_failed / serum2_total if serum2_total else 1.0
        strategy_rows = connection.execute(
            "SELECT load_strategy,COUNT(*) AS count FROM presets "
            "WHERE EXISTS (SELECT 1 FROM params WHERE params.preset_id=presets.id) "
            "GROUP BY load_strategy"
        ).fetchall()
        strategy_detail = ", ".join(
            f"{row['load_strategy']}: {row['count']}" for row in strategy_rows
        )
        full_settings = connection.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(json_valid(metadata_json) AND json_valid(settings_json)) AS valid "
            "FROM serum2_full_settings"
        ).fetchone()
        full_settings_total = int(full_settings["total"] or 0)
        full_settings_valid = int(full_settings["valid"] or 0)

        ids = [
            int(row[0])
            for row in connection.execute(
                "SELECT id FROM presets WHERE EXISTS "
                "(SELECT 1 FROM params WHERE params.preset_id=presets.id) ORDER BY id"
            ).fetchall()
        ]
        vector_detail = "Need at least two verified presets"
        vectors_differ = False
        if len(ids) >= 2:
            left_id, right_id = random.Random(1337).sample(ids, 2)

            def vector(preset_id: int) -> list[float]:
                return [
                    float(row[0])
                    for row in connection.execute(
                        "SELECT norm_value FROM params WHERE preset_id=? ORDER BY param_index",
                        (preset_id,),
                    ).fetchall()
                ]

            left, right = vector(left_id), vector(right_id)
            vectors_differ = len(left) != len(right) or any(
                abs(a - b) > 1e-4 for a, b in zip(left, right)
            )
            vector_detail = f"preset ids {left_id}/{right_id} differ={vectors_differ}"

        summary = json.dumps(
            {
                "serum1_verified": passed,
                "serum1_failed": failed,
                "serum2_verified": serum2_passed,
                "serum2_failed": serum2_failed,
            },
            sort_keys=True,
        )
        return [
            CheckResult("library database", "PASS", str(path)),
            CheckResult(
                "scan queue complete",
                "PASS" if still_scanned == 0 else "FAIL",
                f"unprocessed rows: {still_scanned}; {summary}",
            ),
            CheckResult(
                "Serum 1 failure rate",
                "PASS" if total_supported and failure_rate < 0.05 else "FAIL",
                f"{failed}/{total_supported} = {failure_rate:.2%}",
            ),
            CheckResult(
                "Serum 2 failure rate",
                "PASS" if serum2_total == 710 and serum2_failure_rate < 0.05 else "FAIL",
                f"{serum2_failed}/{serum2_total} = {serum2_failure_rate:.2%}",
            ),
            CheckResult(
                "parameter cardinality",
                "PASS"
                if cardinalities.get("serum1")
                and cardinalities["serum1"]["presets"] == passed
                and cardinalities["serum1"]["minimum"] == 316
                and cardinalities["serum1"]["maximum"] == 316
                and cardinalities.get("serum2")
                and cardinalities["serum2"]["presets"] == serum2_passed
                and cardinalities["serum2"]["minimum"] == 2623
                and cardinalities["serum2"]["maximum"] == 2623
                else "FAIL",
                "; ".join(
                    f"{synth}: {row['presets']} presets, {row['minimum']}/{row['maximum']} params"
                    for synth, row in cardinalities.items()
                ),
            ),
            CheckResult(
                "verified load strategy",
                "PASS" if len(strategy_rows) == 2 and all(row[0] for row in strategy_rows) else "FAIL",
                strategy_detail or "No successful strategy rows",
            ),
            CheckResult(
                "Serum 2 full settings",
                "PASS"
                if full_settings_total == serum2_passed == full_settings_valid == 710
                else "FAIL",
                f"{full_settings_valid}/{full_settings_total} valid JSON records; "
                f"mapped presets: {serum2_passed}",
            ),
            CheckResult(
                "parameter vector diversity",
                "PASS" if vectors_differ else "FAIL",
                vector_detail,
            ),
        ]
    finally:
        connection.close()


def format_table(results: list[CheckResult]) -> str:
    widths = (
        max(len("STATUS"), *(len(item.status) for item in results)),
        max(len("CHECK"), *(len(item.check) for item in results)),
    )
    lines = [f"{'STATUS':<{widths[0]}}  {'CHECK':<{widths[1]}}  DETAIL"]
    lines.append(f"{'-' * widths[0]}  {'-' * widths[1]}  {'-' * 60}")
    lines.extend(
        f"{item.status:<{widths[0]}}  {item.check:<{widths[1]}}  {item.detail}" for item in results
    )
    return "\n".join(lines)
