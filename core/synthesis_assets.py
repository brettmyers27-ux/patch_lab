"""Resolution and availability checks for the analysis-by-synthesis assets.

The distributed app ships factory fingerprints, which are enough to *retrieve*
the closest existing preset but not to *synthesize* a new one. Real synthesis
additionally needs the per-synth parameter targets, the Serum 2 render-state
templates, and a preset database that can resolve a candidate's base preset
back to a file on disk.

Those live under the source checkout during development and must be delivered
as gated artifacts for a distributed install, so every path here is resolved
through one env-overridable knob rather than hardcoded against the repository
layout. `synthesis_readiness()` is the single question the UI asks before
choosing the synthesis path over the factory-fingerprint fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from core.local_library import default_local_paths
from core.model_assets import runtime_root


SERUM2_TARGETS_NAME = "serum2_targets.npz"
SERUM2_SCHEMA_NAME = "serum2_target_schema.json"
PRESET_INDEX_NAME = "preset_index.npy"
NOTE_INDEX_NAME = "note_index.npy"

# Serum 2 candidates are rendered from a per-preset .vstpreset template, so a
# usable install needs the full set rather than a sample of it. The floor is
# deliberately low: it only has to distinguish "the artifact was delivered"
# from "the directory exists but is empty or half-extracted".
MIN_RENDER_STATE_FILES = 8


@dataclass(frozen=True, slots=True)
class SynthesisAssets:
    feature_dir: Path
    serum2_targets: Path
    serum2_schema: Path
    preset_index: Path
    note_index: Path
    render_states: Path
    library_db: Path
    # Factory Serum 2 render states ship with the install, while states for a
    # user's own linked presets are generated into the app data directory. Both
    # must be searchable or one of the two preset populations becomes
    # unrenderable, so keep every candidate root rather than a single path.
    render_state_roots: tuple[Path, ...] = ()

    def find_render_state(self, preset_id: int) -> Path | None:
        """Return the .vstpreset template for `preset_id`, or None if absent."""

        for root in self.render_state_roots or (self.render_states,):
            candidate = root / f"{preset_id}.vstpreset"
            if candidate.is_file():
                return candidate
        return None


@dataclass(frozen=True, slots=True)
class SynthesisReadiness:
    available: bool
    reason: str = ""
    missing: tuple[str, ...] = ()


def _distribution_mode() -> bool:
    return os.environ.get("PATCHLAB_DISTRIBUTION_MODE", "0").strip() == "1"


def resolve_synthesis_assets() -> SynthesisAssets:
    """Resolve every synthesis input from the same knobs on all platforms."""

    root = runtime_root()
    repo_features = root / "data" / "features"
    repo_models = root / "data" / "models"
    local = default_local_paths()

    feature_dir = Path(
        os.environ.get("PATCHLAB_FEATURE_DIR", str(repo_features))
    ).expanduser().resolve()
    schema = Path(
        os.environ.get(
            "PATCHLAB_SERUM2_SCHEMA", str(repo_models / SERUM2_SCHEMA_NAME)
        )
    ).expanduser().resolve()
    # Shipped factory states live beside the models at the same relative path in
    # both a checkout and an install; the app data directory holds states the
    # render worker generates for the user's own linked presets.
    shipped_states = repo_models / "serum2_render_states"
    override = os.environ.get("PATCHLAB_SERUM2_RENDER_STATES")
    if override:
        roots = (Path(override).expanduser().resolve(),)
    elif _distribution_mode():
        roots = (
            shipped_states.expanduser().resolve(),
            local["states"].expanduser().resolve(),
        )
    else:
        roots = (shipped_states.expanduser().resolve(),)
    render_states = roots[0]
    default_db = local["db"] if _distribution_mode() else root / "data" / "library.db"
    library_db = Path(
        os.environ.get("PATCHLAB_LIBRARY_DB", str(default_db))
    ).expanduser().resolve()
    return SynthesisAssets(
        feature_dir=feature_dir,
        serum2_targets=feature_dir / SERUM2_TARGETS_NAME,
        serum2_schema=schema,
        preset_index=feature_dir / PRESET_INDEX_NAME,
        note_index=feature_dir / NOTE_INDEX_NAME,
        render_states=render_states,
        library_db=library_db,
        render_state_roots=roots,
    )


def _render_state_count(roots: tuple[Path, ...]) -> int:
    total = 0
    for directory in roots:
        if not directory.is_dir():
            continue
        try:
            total += sum(1 for _ in directory.glob("*.vstpreset"))
        except OSError:
            continue
    return total


def synthesis_readiness(target_synth: str | None = None) -> SynthesisReadiness:
    """Report whether analysis-by-synthesis can run for `target_synth`.

    Serum 1 and Serum 2 have different requirements — Serum 1 reads its
    parameter targets straight out of the preset database, while Serum 2 needs
    the packaged target vectors, its schema, and the render-state templates. A
    machine can legitimately be able to synthesize one and not the other, so
    asking per-synth avoids disabling a path that would actually work.
    """

    assets = resolve_synthesis_assets()
    missing: list[str] = []

    for label, path in (
        (PRESET_INDEX_NAME, assets.preset_index),
        (NOTE_INDEX_NAME, assets.note_index),
    ):
        if not path.is_file():
            missing.append(label)
    if not assets.library_db.is_file():
        missing.append(assets.library_db.name)

    if target_synth != "serum1":
        if not assets.serum2_targets.is_file():
            missing.append(SERUM2_TARGETS_NAME)
        if not assets.serum2_schema.is_file():
            missing.append(SERUM2_SCHEMA_NAME)
        states = _render_state_count(assets.render_state_roots)
        if states < MIN_RENDER_STATE_FILES:
            missing.append(
                f"serum2_render_states/ "
                f"({states} of at least {MIN_RENDER_STATE_FILES} .vstpreset files)"
            )

    if missing:
        return SynthesisReadiness(
            available=False,
            reason=(
                "Synthesis assets are unavailable, so PatchLab returns the closest "
                "existing preset instead of generating a new patch. Missing: "
                + ", ".join(missing)
            ),
            missing=tuple(missing),
        )
    return SynthesisReadiness(available=True)
