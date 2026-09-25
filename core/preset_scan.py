"""Recursive preset discovery, hash deduplication, and sequential ingestion."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

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


@dataclass(frozen=True, slots=True)
class PresetClassification:
    """Why one preset file was assigned to a Serum generation.

    The reason travels into diagnostics.  Without it, "5,634 presets need
    Serum 1" is an unexplained assertion; with it, a support bundle says the
    classification was made purely from the ``.fxp`` extension, which is what
    made a *Serum 2* preset folder demand a Serum 1 renderer.
    """

    path: Path
    generation: str | None
    reason: str
    evidence: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "generation": self.generation,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def classify_preset(path: Path) -> PresetClassification:
    """Classify a preset file's Serum generation, recording the reason.

    Classification is by file format, which is the only reliable signal: a
    ``.fxp`` **is** a Serum 1 patch file and a ``.serumpreset`` **is** a Serum 2
    patch file.

    Note where those files live, though.  Serum 2's own factory library ships
    thousands of legacy ``.fxp`` patches alongside its ``.serumpreset`` ones --
    on a representative install, 4,826 ``.fxp`` against 1,857
    ``.serumpreset`` under ``Serum 2 Presets/Presets``.  So "the user linked a
    Serum 2 folder" does not mean "every file needs Serum 2", and it certainly
    does not mean the Serum 2 subset should be blocked when Serum 1 is absent.
    That is why callers route per generation and process the supported subset
    rather than treating a mixed library as all-or-nothing.
    """

    suffix = path.suffix.casefold()
    if suffix == ".fxp":
        return PresetClassification(
            path=path,
            generation="serum1",
            reason=(
                "the .fxp container is Serum 1's patch format; PatchLab's verified "
                "loaders for it are the Serum 1 strategies (S1 VST2, S2 VST3 state, "
                "S5 AU). Serum 2 factory libraries also ship .fxp content, so a "
                "Serum 2 preset folder legitimately contains these."
            ),
            evidence=f"extension={suffix}",
        )
    if suffix == ".serumpreset":
        return PresetClassification(
            path=path,
            generation="serum2",
            reason="the .serumpreset container is Serum 2's patch format",
            evidence=f"extension={suffix}",
        )
    return PresetClassification(
        path=path,
        generation=None,
        reason="not a recognised Serum patch container",
        evidence=f"extension={suffix or '(none)'}",
    )


def synth_for(path: Path) -> str | None:
    """Return the Serum generation for ``path``, or None if it is not a preset."""

    return classify_preset(path).generation


def classify_generations(paths: Iterable[Path]) -> dict[str, int]:
    """Count presets per generation.

    Aggregated on purpose: a 5,000-file library must not produce 5,000
    classification log lines.  One summary carries the same diagnostic value.
    """

    counts: dict[str, int] = {}
    for path in paths:
        generation = synth_for(path)
        if generation is not None:
            counts[generation] = counts.get(generation, 0) + 1
    return counts


def log_classification_summary(
    paths: Iterable[Path],
    *,
    operation_id: str = "",
    linked_folder: Path | None = None,
) -> dict[str, int]:
    """Record one aggregated classification decision for a whole library."""

    from core.diagnostics import record

    materialised = list(paths)
    counts = classify_generations(materialised)
    examples = {}
    for generation in counts:
        example = next(
            (item for item in materialised if synth_for(item) == generation), None
        )
        if example is not None:
            examples[generation] = classify_preset(example).reason
    record(
        "preset-classification",
        "classification_summary",
        "classified "
        + ", ".join(f"{count} {generation}" for generation, count in sorted(counts.items())),
        operation_id=operation_id,
        phase="preset-classification",
        decision_reason=(
            "generation is decided by patch container format; a Serum 2 preset "
            "folder commonly contains both .serumpreset and legacy .fxp content"
        ),
        counts=counts,
        total_files=len(materialised),
        linked_folder=str(linked_folder) if linked_folder else None,
        reasons=examples,
    )
    return counts


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


class SequentialSerum1Ingestor:
    """One reusable engine/plugin instance for safe sequential FXP ingestion.

    Previously named ``SequentialSerum1VST2`` and hardcoded to
    ``item.format == "VST2"``, which is the defect behind the reported
    "Verified Serum 1 VST2 binary is unavailable." failure: constructing it at
    all demanded a Serum 1 VST2 binary, even on a machine whose Serum 1 was
    installed as VST3 or AU, and even when the pending work needed Serum 2.

    Renderer choice now goes through :mod:`core.renderer_selection`, which
    keeps VST2 as the highest preference (so nothing changes where VST2 exists)
    and falls back to the VST3 and AU strategies
    :func:`core.plugin_host.load_preset` already implements.
    """

    def __init__(
        self,
        env: PlatformEnv,
        *,
        operation_id: str = "",
    ) -> None:
        from core.renderer_selection import require_renderer

        selection = require_renderer(
            "serum1",
            env=env,
            operation_id=operation_id,
            phase="preset-classification",
            context="preparing sequential Serum 1 preset ingestion",
        )
        assert selection.selected is not None
        self.selection = selection
        # Rebuild the chosen candidate rather than searching for it again: the
        # second lookup could miss (and raise a bare StopIteration) whenever the
        # selector legitimately chose a candidate this filter did not match.
        from core.renderer_selection import renderer_candidate

        self.candidate = renderer_candidate(selection)
        self.engine, self.processor = make_dawdreamer_processor(self.candidate)
        self.initial = dump_dawdreamer_parameters(self.processor)

    @property
    def strategy_label(self) -> str:
        if self.candidate.format == "VST2":
            return "VST2/S1-dawdreamer-vst2-fxp"
        if self.candidate.format == "VST3":
            return "VST3/S2-dawdreamer-vst3-fxp-state"
        return "AU/S5-dawdreamer-au-direct-preset"

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
        return parameters, rms, self.strategy_label


#: Retained so existing callers and tests keep working.
SequentialSerum1VST2 = SequentialSerum1Ingestor


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
    # Local import avoids a module cycle: library_state uses this module's
    # cheap discovery, classification, and hash helpers.
    from core.library_state import reconcile_source_tree

    discovery = reconcile_source_tree(
        root, database, paths=paths, hash_file=sha1_file
    )
    summary.inserted = discovery.new_content
    summary.deduped = len(discovery.entries) - discovery.new_content
    for index, _entry in enumerate(discovery.entries, start=1):
        if progress:
            progress(index, max(len(paths), 1))
    log(
        f"Cataloged {summary.inserted}; content-hash duplicates {summary.deduped}; "
        f"computed {discovery.hashes_computed} hashes"
    )

    pending = database.presets_with_status(("scanned",))
    capabilities = load_capabilities()
    serum1_enabled = "serum1" in capabilities or bool(env.plugins_for("serum1"))
    needs_serum1 = any(preset.synth == "serum1" for preset in pending)
    ingestor = None
    if serum1_enabled and needs_serum1:
        from core.renderer_selection import RendererUnavailableError

        try:
            ingestor = SequentialSerum1Ingestor(env)
        except RendererUnavailableError as exc:
            # A missing Serum 1 renderer must not abort the whole scan: the
            # Serum 2 subset below is still processable.  Each Serum 1 preset is
            # marked failed with the real reason instead.
            ingestor = None
            log(f"Serum 1 ingestion unavailable: {exc}")
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
