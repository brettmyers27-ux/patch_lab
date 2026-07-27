"""Recursive preset discovery, hash deduplication, and sequential ingestion."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from core.db import DEFAULT_DB_PATH, Database, PresetRecord
from core.platform_env import ENV, PlatformEnv
from core.plugin_host import (
    SILENCE_DBFS,
    ParameterValue,
    audio_levels,
    changed_parameter_count,
    dump_dawdreamer_parameters,
    make_dawdreamer_processor,
    render_dawdreamer_note,
)


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
CAPABILITY_PATH = Path(__file__).resolve().parents[1] / "data" / "strategy_capabilities.json"


@dataclass(slots=True)
class ScanSummary:
    found: int = 0
    inserted: int = 0
    deduped: int = 0
    params_dumped: int = 0
    failed: int = 0
    serum2_disabled: int = 0


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synth_for(path: Path) -> str | None:
    suffix = path.suffix.casefold()
    if suffix == ".fxp":
        return "serum1"
    if suffix == ".serumpreset":
        return "serum2"
    return None


def discover_presets(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and synth_for(path)),
        key=lambda path: str(path).casefold(),
    )


def load_capabilities() -> dict[str, dict[str, object]]:
    if not CAPABILITY_PATH.exists():
        return {}
    try:
        data = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    supported = data.get("supported", {})
    return supported if isinstance(supported, dict) else {}


class SequentialSerum1VST2:
    """One reusable engine/plugin instance for safe sequential FXP ingestion."""

    def __init__(self, env: PlatformEnv) -> None:
        candidates = [item for item in env.plugins_for("serum1") if item.format == "VST2"]
        if not candidates:
            raise RuntimeError("Verified Serum 1 VST2 binary is unavailable.")
        self.candidate = candidates[0]
        self.engine, self.processor = make_dawdreamer_processor(self.candidate)
        self.initial = dump_dawdreamer_parameters(self.processor)

    def ingest(self, path: Path) -> tuple[list[ParameterValue], float, str]:
        if self.processor.load_preset(str(path)) is False:
            raise RuntimeError("DawDreamer load_preset returned False")
        parameters = dump_dawdreamer_parameters(self.processor)
        changed = changed_parameter_count(self.initial, parameters)
        if changed < 5:
            raise RuntimeError(f"only {changed} parameters changed from init")
        _peak, rms = audio_levels(render_dawdreamer_note(self.engine, self.processor))
        if rms <= SILENCE_DBFS:
            raise SilentPresetError(f"C4 render is silent at {rms:.2f} dBFS")
        return parameters, rms, "VST2/S1-dawdreamer-vst2-fxp"


class SilentPresetError(RuntimeError):
    pass


def scan_and_ingest(
    root: Path,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    env: PlatformEnv = ENV,
    log: LogCallback = print,
    progress: ProgressCallback | None = None,
) -> ScanSummary:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    database = Database(db_path)
    summary = ScanSummary()
    paths = discover_presets(root)
    summary.found = len(paths)
    log(f"Found {summary.found} preset files under {root}")
    for index, path in enumerate(paths, start=1):
        synth = synth_for(path)
        assert synth is not None
        _preset_id, inserted = database.insert_preset(
            path=path, name=path.stem, synth=synth, content_hash=sha1_file(path)
        )
        if inserted:
            summary.inserted += 1
        else:
            summary.deduped += 1
        if progress:
            progress(index, max(len(paths), 1))
    log(f"Cataloged {summary.inserted}; content-hash duplicates {summary.deduped}")

    pending = database.presets_with_status(("scanned",))
    capabilities = load_capabilities()
    serum1_enabled = "serum1" in capabilities or bool(env.plugins_for("serum1"))
    needs_serum1 = any(preset.synth == "serum1" for preset in pending)
    ingestor = SequentialSerum1VST2(env) if serum1_enabled and needs_serum1 else None
    for index, preset in enumerate(pending, start=1):
        if preset.synth == "serum2" and "serum2" not in capabilities:
            database.mark_failed(
                preset.id,
                "failed_load",
                "Serum 2 disabled: Milestone 0 found no verified preset-loading strategy.",
            )
            summary.serum2_disabled += 1
            continue
        try:
            if preset.synth != "serum1" or ingestor is None:
                raise RuntimeError(f"No enabled sequential ingestor for {preset.synth}")
            parameters, rms, strategy = ingestor.ingest(preset.path)
            database.replace_params(preset.id, parameters, strategy)
            summary.params_dumped += 1
            log(
                f"[{index}/{len(pending)}] params_dumped id={preset.id} "
                f"{preset.name!r}: {len(parameters)} params, RMS {rms:.2f} dBFS"
            )
        except SilentPresetError as exc:
            database.mark_failed(preset.id, "failed_silent", str(exc))
            summary.failed += 1
            log(f"[{index}/{len(pending)}] failed_silent id={preset.id}: {exc}")
        except Exception as exc:
            database.mark_failed(preset.id, "failed_load", repr(exc))
            summary.failed += 1
            log(f"[{index}/{len(pending)}] failed_load id={preset.id}: {exc!r}")
        if progress:
            progress(index, max(len(pending), 1))

    dumped = database.presets_with_status(("params_dumped",))
    if len(dumped) >= 2:
        chosen = random.Random(1337).sample(dumped, 2)
        vectors = [database.param_vector(item.id) for item in chosen]
        different = len(vectors[0]) != len(vectors[1]) or any(
            abs(left - right) > 1e-4 for left, right in zip(vectors[0], vectors[1])
        )
        log(
            f"Spot check ids {chosen[0].id}/{chosen[1].id}: parameter vectors "
            f"{'DIFFER' if different else 'DO NOT DIFFER'}"
        )
    log("SCAN_SUMMARY=" + json.dumps(asdict(summary), sort_keys=True))
    return summary
