#!/usr/bin/env python3
"""Measure the real, headless Phase 3 preparation pipeline without changing it.

This utility intentionally calls :func:`core.local_library.process_linked_folder`
for every pass.  It copies real source presets into a temporary linked folder,
keeps its database and analysis work outside the app's normal data directory,
and removes that workspace unless ``--keep-work-dir`` is requested.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import Database
from core.local_library import process_linked_folder
from core.platform_env import ENV, PlatformEnv
from core.preparation import analysis_temp_root
from core.prepared_state import is_preset_prepared


def _tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.is_dir() else 0


def _source_files(root: Path, suffix: str) -> list[Path]:
    return sorted(path for path in root.rglob(f"*{suffix}") if path.is_file())


def _choose_sources(serum1_root: Path, serum2_root: Path, sample_size: int) -> list[Path]:
    serum1 = _source_files(serum1_root, ".fxp")
    serum2 = _source_files(serum2_root, ".SerumPreset")
    if not serum1 and not serum2:
        raise RuntimeError("no .fxp or .SerumPreset files were found in the supplied sources")
    desired_serum1 = min(len(serum1), max(1, sample_size // 10)) if serum1 else 0
    desired_serum2 = min(len(serum2), sample_size - desired_serum1)
    selected = serum1[:desired_serum1] + serum2[:desired_serum2]
    remaining = sample_size - len(selected)
    if remaining:
        selected.extend((serum2[desired_serum2:] + serum1[desired_serum1:])[:remaining])
    return selected


def _copy(paths: list[Path], destination: Path, *, start: int = 0) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(paths, start=start):
        # Prefixes keep same-named user presets distinct without modifying bytes.
        shutil.copy2(source, destination / f"{index:03d}-{source.name}")


@dataclass(slots=True)
class Timing:
    started: dict[int, dict[str, float]] = field(default_factory=dict)
    render_s: dict[int, float] = field(default_factory=dict)
    analyze_s: dict[int, float] = field(default_factory=dict)
    commit_s: dict[int, float] = field(default_factory=dict)
    cleanup_s: dict[int, float] = field(default_factory=dict)
    peak_work_bytes: int = 0
    temp_bytes: dict[int, int] = field(default_factory=dict)

    def hook(self, analysis_root: Path, database: Database) -> Callable[[str, int], None]:
        def record(stage: str, preset_id: int) -> None:
            now = time.monotonic()
            marks = self.started.setdefault(preset_id, {})
            if stage == "before_render":
                marks["render"] = now
            elif stage == "after_render":
                self.render_s[preset_id] = now - marks["render"]
                with database.connect() as connection:
                    paths = connection.execute(
                        "SELECT wav_path FROM renders WHERE preset_id=?", (preset_id,)
                    ).fetchall()
                self.temp_bytes[preset_id] = sum(
                    Path(row[0]).stat().st_size for row in paths if Path(row[0]).is_file()
                )
                self.peak_work_bytes = max(self.peak_work_bytes, _tree_size(analysis_root))
                marks["analyze"] = now
            elif stage == "after_features":
                self.analyze_s[preset_id] = now - marks["analyze"]
                marks["commit"] = now
            elif stage == "after_prepared_commit":
                self.commit_s[preset_id] = now - marks["commit"]
                marks["cleanup"] = now
            elif stage == "after_cleanup":
                self.cleanup_s[preset_id] = now - marks["cleanup"]
                self.peak_work_bytes = max(self.peak_work_bytes, _tree_size(analysis_root))

        return record


def _mean(values: dict[int, float]) -> float:
    return sum(values.values()) / len(values) if values else 0.0


def _pass(
    root: Path,
    *,
    db_path: Path,
    env: PlatformEnv,
    render_processes: int,
) -> tuple[dict[str, object], Timing]:
    database = Database(db_path)
    timing = Timing()
    started = time.monotonic()
    summary = process_linked_folder(
        root,
        db_path=db_path,
        audio_root=env.app_data_dir / "legacy-audio",
        state_dir=env.app_data_dir / "serum2-render-states",
        env=env,
        relay=None,
        render_processes=render_processes,
        log=lambda _message: None,
        preparation_stage_hook=timing.hook(analysis_temp_root(env), database),
    )
    elapsed = time.monotonic() - started
    prepared = int(summary.fingerprints_created)
    per_preset = list(timing.render_s.values())
    return (
        {
            "elapsed_s": round(elapsed, 3),
            "prepared": prepared,
            "prepared_per_min": round(prepared * 60 / elapsed, 3) if elapsed else 0.0,
            "summary": asdict(summary),
            "average_render_s": round(_mean(timing.render_s), 3),
            "average_analyze_s": round(_mean(timing.analyze_s), 3),
            "average_commit_s": round(_mean(timing.commit_s), 3),
            "average_cleanup_s": round(_mean(timing.cleanup_s), 3),
            "p50_preset_render_s": round(median(per_preset), 3) if per_preset else 0.0,
            "p95_preset_render_s": round(
                sorted(per_preset)[max(0, int(len(per_preset) * 0.95) - 1)], 3
            ) if per_preset else 0.0,
            "peak_working_storage_bytes": timing.peak_work_bytes,
            "average_temp_bytes_per_preset": round(
                sum(timing.temp_bytes.values()) / len(timing.temp_bytes)
            ) if timing.temp_bytes else 0,
            "max_temp_bytes_per_preset": max(timing.temp_bytes.values(), default=0),
        },
        timing,
    )


def _validate(database: Database, analysis_root: Path) -> dict[str, object]:
    with database.connect() as connection:
        rows = connection.execute("SELECT id FROM presets WHERE is_factory=0").fetchall()
        prepared = [int(row[0]) for row in rows if is_preset_prepared(connection, int(row[0]))]
        wavs = connection.execute("SELECT wav_path FROM renders").fetchall()
    remaining = [str(row[0]) for row in wavs if Path(row[0]).exists()]
    return {
        "prepared_contract_passed": len(prepared) == len(rows),
        "prepared_presets": len(prepared),
        "temporary_wavs_remaining": len(remaining),
        "analysis_work_bytes_after_cleanup": _tree_size(analysis_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serum1-source", type=Path, required=True)
    parser.add_argument("--serum2-source", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--render-processes", type=int, default=4)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    if not 1 <= args.sample_size <= 200:
        raise SystemExit("--sample-size must be between 1 and 200")

    sources = _choose_sources(args.serum1_source, args.serum2_source, args.sample_size + 11)
    if len(sources) < args.sample_size:
        raise SystemExit(f"only {len(sources)} usable real presets are available")
    base, reserve = sources[: args.sample_size], sources[args.sample_size :]
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir:
        workspace = args.work_dir.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="patchlab-preparation-benchmark-")
        workspace = Path(temporary.name)
    linked = workspace / "linked-presets"
    _copy(base, linked)
    env = replace(ENV, app_data_dir=workspace / "app-data")
    database = Database(env.app_data_dir / "library.db")

    baseline, _timing = _pass(
        linked, db_path=database.path, env=env, render_processes=args.render_processes
    )
    no_change, _ = _pass(linked, db_path=database.path, env=env, render_processes=args.render_processes)
    if reserve:
        _copy(reserve[:1], linked, start=len(base))
    one_new, _ = _pass(linked, db_path=database.path, env=env, render_processes=args.render_processes)
    if len(reserve) > 1:
        _copy(reserve[1:11], linked, start=len(base) + 1)
    ten_new, _ = _pass(linked, db_path=database.path, env=env, render_processes=args.render_processes)

    with database.connect() as connection:
        synth_counts = dict(
            connection.execute("SELECT synth, COUNT(*) FROM presets GROUP BY synth").fetchall()
        )
    result = {
        "baseline_config": {
            "phase3_active_presets": 1,
            "render_processes_argument": args.render_processes,
            "note": "Phase 3 deliberately renders one preset at a time; the compatibility argument is not used by prepare_work_queue.",
        },
        "sample": {"requested": args.sample_size, "selected": len(base), "by_synth": synth_counts},
        "baseline": baseline,
        "incremental": {"no_change": no_change, "one_new": one_new, "ten_new": ten_new},
        "correctness": _validate(database, analysis_temp_root(env)),
        "workspace": str(workspace) if args.keep_work_dir or args.work_dir else "removed",
    }
    if args.result_file:
        target = args.result_file.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BENCHMARK_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    if temporary is not None and not args.keep_work_dir:
        temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
