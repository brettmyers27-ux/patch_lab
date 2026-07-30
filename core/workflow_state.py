"""One source of truth for the four workflow cards.

The resolver deliberately receives the running app's per-user paths.  It never
consults the repository development database or feature directories.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

from core.factory_bundle import DEFAULT_FACTORY_BUNDLE
from core.model_assets import ModelAssetsError, validate_model_assets
from core.privacy import PrivacyChoice
from core.render import MIDI_NOTES


WorkflowPhase = Literal["needs-action", "in-progress", "complete", "not-required"]


@dataclass(frozen=True, slots=True)
class WorkflowActivity:
    current: int
    total: int
    text: str


@dataclass(frozen=True, slots=True)
class WorkflowCardState:
    phase: WorkflowPhase
    text: str
    current: int
    total: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class WorkflowState:
    link: WorkflowCardState
    render: WorkflowCardState
    analyze: WorkflowCardState
    match: WorkflowCardState

    def as_dict(self) -> dict[str, WorkflowCardState]:
        return {
            "link": self.link,
            "render": self.render,
            "analyze": self.analyze,
            "match": self.match,
        }


@dataclass(frozen=True, slots=True)
class _LocalCounts:
    presets: int = 0
    rendered: int = 0
    error: str = ""


def _local_counts(database_path: Path, linked_folder: Path) -> _LocalCounts:
    """Read only the local app database and never create it as a side effect."""

    database_path = Path(database_path).expanduser().resolve()
    if not database_path.is_file():
        return _LocalCounts()
    try:
        connection = sqlite3.connect(
            f"file:{database_path}?mode=ro", uri=True, timeout=2.0
        )
        root = Path(linked_folder).expanduser().resolve()
        preset_ids: set[int] = set()
        for preset_id, raw_path in connection.execute(
            "SELECT id,path FROM presets"
        ).fetchall():
            try:
                Path(str(raw_path)).expanduser().resolve().relative_to(root)
            except (OSError, ValueError):
                continue
            preset_ids.add(int(preset_id))
        rendered_ids = {
            int(row[0])
            for row in connection.execute(
                "SELECT preset_id FROM renders GROUP BY preset_id "
                "HAVING COUNT(DISTINCT midi_note) >= ?",
                (len(MIDI_NOTES),),
            ).fetchall()
        }
        connection.close()
        return _LocalCounts(
            len(preset_ids),
            len(preset_ids.intersection(rendered_ids)),
        )
    except (OSError, sqlite3.Error) as exc:
        return _LocalCounts(error=f"{type(exc).__name__}: {exc}")


def _model_problem_label(detail: str) -> str:
    lowered = detail.casefold()
    if "pinned clap checkpoint is missing" in lowered:
        return "CLAP checkpoint missing · reinstall PatchLab"
    if "tokenizer" in lowered or "cache is incomplete" in lowered:
        return "Tokenizer cache missing · reinstall PatchLab"
    return "Model files need attention · reinstall PatchLab"


def _match_prerequisite_error(factory_bundle_path: Path) -> str:
    try:
        validate_model_assets()
    except ModelAssetsError as exc:
        return _model_problem_label(str(exc))

    bundle_path = Path(factory_bundle_path).expanduser().resolve()
    if not bundle_path.is_file():
        return "Factory fingerprints missing · reinstall PatchLab"
    try:
        connection = sqlite3.connect(
            f"file:{bundle_path}?mode=ro", uri=True, timeout=2.0
        )
        preset_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM presets WHERE searchable=1"
            ).fetchone()[0]
        )
        embedding_count = int(
            connection.execute("SELECT COUNT(*) FROM preset_embeddings").fetchone()[0]
        )
        sample = connection.execute(
            "SELECT embedding_f16 FROM preset_embeddings LIMIT 1"
        ).fetchone()
        connection.close()
        if (
            preset_count <= 0
            or embedding_count != preset_count
            or sample is None
            or len(bytes(sample[0])) != 512 * 2
        ):
            raise ValueError(
                f"{preset_count} searchable presets, {embedding_count} embeddings"
            )
    except (OSError, sqlite3.Error, ValueError) as exc:
        return f"Factory fingerprints unreadable · reinstall PatchLab ({exc})"
    return ""


def _activity_or(
    key: str,
    activities: Mapping[str, WorkflowActivity],
    fallback: WorkflowCardState,
) -> WorkflowCardState:
    activity = activities.get(key)
    if activity is None:
        return fallback
    return WorkflowCardState(
        "in-progress",
        activity.text,
        max(int(activity.current), 0),
        max(int(activity.total), 0),
    )


def resolve_workflow_state(
    *,
    privacy: PrivacyChoice,
    local_database_path: Path,
    audio_selected: bool,
    match_completed: bool = False,
    activities: Mapping[str, WorkflowActivity] | None = None,
    factory_bundle_path: Path = DEFAULT_FACTORY_BUNDLE,
    match_prerequisite_error: str = "",
) -> WorkflowState:
    """Resolve every card together from persisted machine state plus live jobs."""

    live = activities or {}
    linked = bool(privacy.use_and_share_own_presets and privacy.linked_folder)
    counts = (
        _local_counts(local_database_path, Path(str(privacy.linked_folder)))
        if linked
        else _LocalCounts()
    )
    folder_name = (
        Path(str(privacy.linked_folder)).expanduser().name or str(privacy.linked_folder)
        if linked
        else ""
    )

    if linked:
        link = WorkflowCardState(
            "complete",
            f"Linked: {folder_name} · {counts.presets:,} presets found",
            counts.presets,
            max(counts.presets, 1),
            counts.error,
        )
    else:
        link = WorkflowCardState(
            "needs-action", "Click to link your presets", 0, 1
        )
    link = _activity_or("link", live, link)

    if not linked:
        render = WorkflowCardState(
            "not-required",
            "Not required · factory library is ready",
            1,
            1,
        )
    elif counts.error:
        render = WorkflowCardState(
            "needs-action",
            "Local library database needs attention",
            0,
            1,
            counts.error,
        )
    elif counts.presets == 0:
        render = WorkflowCardState(
            "not-required", "No linked presets need rendering", 1, 1
        )
    elif counts.rendered >= counts.presets:
        render = WorkflowCardState(
            "complete",
            f"All {counts.presets:,} linked presets rendered",
            counts.presets,
            counts.presets,
        )
    else:
        remaining = counts.presets - counts.rendered
        render = WorkflowCardState(
            "needs-action",
            f"{remaining:,} of {counts.presets:,} presets still need rendering",
            counts.rendered,
            counts.presets,
        )
    render = _activity_or("render", live, render)

    analyze = WorkflowCardState(
        "complete",
        (
            "Using PatchLab’s trained model · linked presets join search"
            if linked
            else "Using PatchLab’s trained model"
        ),
        1,
        1,
        (
            "Personal presets are fingerprinted during the linked-folder job and "
            "added to retrieval. Full model retraining is not incremental in this "
            "release, so PatchLab never replaces shipped learning with local-only data."
        ),
    )
    analyze = _activity_or("analyze", live, analyze)

    prerequisite_error = (
        _model_problem_label(match_prerequisite_error)
        if match_prerequisite_error
        else _match_prerequisite_error(factory_bundle_path)
    )
    if prerequisite_error:
        match = WorkflowCardState(
            "needs-action", prerequisite_error, 0, 1, prerequisite_error
        )
    elif match_completed:
        match = WorkflowCardState(
            "complete", "Match complete · choose another sound", 1, 1
        )
    elif audio_selected:
        match = WorkflowCardState(
            "complete", "Audio selected · ready to match", 1, 1
        )
    else:
        match = WorkflowCardState(
            "needs-action", "Ready · choose an audio file", 0, 1
        )
    match = _activity_or("match", live, match)

    return WorkflowState(link=link, render=render, analyze=analyze, match=match)
