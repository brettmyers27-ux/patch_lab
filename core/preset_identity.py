"""Preset identity: six separate facts that must never be collapsed into one.

The first round of this work treated ``.fxp == Serum 1 preset`` as a single
field, and that conflation is what made a *Serum 2* preset folder demand a
Serum 1 renderer. The concepts that were merged are:

1. **file format** -- the container on disk (``fxp`` / ``serumpreset``)
2. **origin generation** -- which Serum most likely authored it
3. **provenance** -- which library the file was found in
4. **compatible renderers** -- which installed Serum can *actually* open it
5. **selected renderer** -- which one PatchLab chose for this machine
6. **processed state** -- whether it has been rendered and learned yet

(4) is the one that bites, and it is not a property of the extension alone. It
is a property of what a given Serum build can ingest *through PatchLab's
headless mechanisms*. Serum 2 can import a legacy ``.fxp`` interactively in its
own browser, but see :data:`FXP_SERUM2_FINDING` -- every headless route was
tested against real presets and real Serum 2, and none of them work. So the
compatible-renderer set for a ``.fxp`` is ``("serum1",)``, recorded with that
reason rather than asserted from the extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from core.platform_env import ENV, PlatformEnv


FileFormat = Literal["fxp", "serumpreset"]
Generation = Literal["serum1", "serum2"]
Provenance = Literal[
    "serum1_factory", "serum2_factory", "user_folder", "unknown"
]

#: The measured conclusion of the headless legacy-.fxp investigation, quoted in
#: diagnostics so a support bundle explains the limitation rather than implying
#: PatchLab simply never tried.
#:
#: Tested on real Serum 2 (VST3 2.0.1 and AU) with five substantially different
#: real ``.fxp`` presets from a Serum 2 factory library:
#:
#: * ``load_preset(fxp)`` returns True but changes nothing -- every preset
#:   renders bit-identically to Serum 2's init patch (cosine 1.000).
#: * ``load_state`` with the FXP payload or the whole file: same silent no-op.
#: * ``load_vst3_preset`` with the payload wrapped in a .vstpreset: hard error.
#: * pedalboard ``raw_state`` (payload or whole file): silent no-op.
#: * pedalboard AU ClassInfo injection into ``Processor State`` or ``vstdata``:
#:   silent no-op.
#: * pedalboard ``preset_data``: hard error.
#: * ``parse_serum2_preset`` rejects the FXP container outright (bad magic).
#:
#: A native ``.serumpreset`` through the same harness *does* produce distinct
#: per-preset audio, which proves the harness detects real loading.
FXP_SERUM2_FINDING = (
    "Serum 2 cannot ingest a legacy .fxp through any headless mechanism "
    "available to PatchLab: load_preset/load_state silently keep the init "
    "patch, .vstpreset wrapping and preset_data error, and the FXP container is "
    "not a Serum 2 preset. Serum 2's .fxp import is a GUI browser feature with "
    "no automation surface, and PatchLab never drives a plug-in GUI."
)

#: Which installed generation can render each container, headlessly.
#:
#: Keep this as data with a reason attached: if a future Serum 2 build exposes a
#: headless legacy-import API, this table and its reason are the only things
#: that change.
COMPATIBLE_RENDERERS: Mapping[str, tuple[Generation, ...]] = {
    "fxp": ("serum1",),
    "serumpreset": ("serum2",),
}

COMPATIBILITY_REASONS: Mapping[str, str] = {
    "fxp": (
        "the FXP container is Serum 1's patch format and PatchLab's verified "
        "loaders for it are the Serum 1 strategies (VST2 direct, VST3 chunk "
        "state, AU). " + FXP_SERUM2_FINDING
    ),
    "serumpreset": (
        "the .serumpreset container is Serum 2's patch format; PatchLab loads it "
        "by reconstructing a .vstpreset against a live Serum 2 host template"
    ),
}

#: Stable, machine-readable reasons a discovered preset is not yet processed.
PENDING_SERUM1_NOT_INSTALLED = "serum1_not_installed"
PENDING_SERUM2_NOT_INSTALLED = "serum2_not_installed"
PENDING_SERUM1_RENDERER_INVALID = "serum1_renderer_invalid"
PENDING_SERUM2_RENDERER_INVALID = "serum2_renderer_invalid"
PENDING_UNSUPPORTED_LEGACY_FORMAT = "unsupported_legacy_format"
PENDING_RENDER_FAILED = "render_failed"
PENDING_CLASSIFICATION_UNKNOWN = "classification_unknown"
#: Nothing is missing -- this preset's renderer is available and it simply has
#: not been processed yet. Distinct from the "blocked" reasons above so the UI
#: can offer to process it and the counts can tell the two situations apart.
PENDING_AWAITING_PROCESSING = "awaiting_processing"

#: Reasons that mean "PatchLab cannot do this here", as opposed to
#: PENDING_AWAITING_PROCESSING which means "ready, just not done yet".
BLOCKED_PENDING_REASONS: frozenset[str] = frozenset(
    {
        PENDING_SERUM1_NOT_INSTALLED,
        PENDING_SERUM2_NOT_INSTALLED,
        PENDING_SERUM1_RENDERER_INVALID,
        PENDING_SERUM2_RENDERER_INVALID,
        PENDING_UNSUPPORTED_LEGACY_FORMAT,
        PENDING_CLASSIFICATION_UNKNOWN,
    }
)

def blocked_reasons_for(generation: str) -> frozenset[str]:
    """Pending reasons meaning "waiting on ``generation`` to become available".

    Excludes ``render_failed``: a preset that failed to render while its synth
    was perfectly available is not waiting for anything to be installed, so it
    must not trigger a "your synth is now available" offer.
    """

    if str(generation) == "serum1":
        return frozenset(
            {
                PENDING_SERUM1_NOT_INSTALLED,
                PENDING_SERUM1_RENDERER_INVALID,
                PENDING_UNSUPPORTED_LEGACY_FORMAT,
                PENDING_AWAITING_PROCESSING,
            }
        )
    return frozenset(
        {
            PENDING_SERUM2_NOT_INSTALLED,
            PENDING_SERUM2_RENDERER_INVALID,
            PENDING_AWAITING_PROCESSING,
        }
    )


PENDING_REASONS: tuple[str, ...] = (
    PENDING_AWAITING_PROCESSING,
    PENDING_SERUM1_NOT_INSTALLED,
    PENDING_SERUM2_NOT_INSTALLED,
    PENDING_SERUM1_RENDERER_INVALID,
    PENDING_SERUM2_RENDERER_INVALID,
    PENDING_UNSUPPORTED_LEGACY_FORMAT,
    PENDING_RENDER_FAILED,
    PENDING_CLASSIFICATION_UNKNOWN,
)

#: Human wording for each code. Kept beside the codes so the UI and the
#: diagnostics never drift apart.
PENDING_REASON_LABELS: Mapping[str, str] = {
    PENDING_SERUM1_NOT_INSTALLED: "Serum 1 is not installed",
    PENDING_SERUM2_NOT_INSTALLED: "Serum 2 is not installed",
    PENDING_SERUM1_RENDERER_INVALID: "the Serum 1 plug-in could not be started",
    PENDING_SERUM2_RENDERER_INVALID: "the Serum 2 plug-in could not be started",
    PENDING_UNSUPPORTED_LEGACY_FORMAT: (
        "this legacy preset needs Serum 1; Serum 2 cannot open it outside its own browser"
    ),
    PENDING_RENDER_FAILED: "rendering this preset failed",
    PENDING_AWAITING_PROCESSING: "it has not been processed yet",
    PENDING_CLASSIFICATION_UNKNOWN: "PatchLab could not identify this preset's format",
}

_EXTENSIONS: Mapping[str, FileFormat] = {
    ".fxp": "fxp",
    ".serumpreset": "serumpreset",
}


def file_format_for(path: Path) -> FileFormat | None:
    """The container format, or None if this is not a Serum preset file."""

    return _EXTENSIONS.get(Path(path).suffix.casefold())


#: Resolved factory roots per environment, cached because classifying a
#: 6,700-preset library once called ``path_is_factory`` -- which resolves every
#: factory root from scratch -- for every single file. Measured: that made
#: identification 3.5x the cost of the directory walk it piggybacks on.
_FACTORY_ROOT_CACHE: dict[int, tuple[tuple[str, Provenance], ...]] = {}


def _factory_prefixes(env: PlatformEnv) -> tuple[tuple[str, Provenance], ...]:
    key = id(env)
    cached = _FACTORY_ROOT_CACHE.get(key)
    if cached is not None:
        return cached
    prefixes: list[tuple[str, Provenance]] = []
    for generation, label in (("serum2", "serum2_factory"), ("serum1", "serum1_factory")):
        for root in env.factory_roots_for(generation):  # type: ignore[arg-type]
            try:
                resolved = str(Path(root).expanduser().resolve())
            except OSError:
                continue
            prefixes.append((resolved.rstrip("/\\"), label))  # type: ignore[arg-type]
    result = tuple(prefixes)
    _FACTORY_ROOT_CACHE[key] = result
    return result


def clear_provenance_cache() -> None:
    """Drop cached factory roots.  Tests and a changed environment use this."""

    _FACTORY_ROOT_CACHE.clear()


def provenance_for(path: Path, *, env: PlatformEnv | None = None) -> Provenance:
    """Which library this file came from.

    Provenance is separate from format on purpose: a ``.fxp`` inside a Serum 2
    factory library is still a Serum 1 patch, but knowing it shipped *with*
    Serum 2 is what lets PatchLab explain why a Serum 2 user has thousands of
    them, instead of looking like a misclassification.

    Serum 2 is checked first: a root nested inside another would otherwise be
    attributed to the outer library.
    """

    env = env or ENV
    try:
        value = str(Path(path).expanduser().resolve())
    except OSError:
        value = str(path)
    for prefix, label in _factory_prefixes(env):
        if value == prefix or value.startswith(prefix + "/") or value.startswith(
            prefix + "\\"
        ):
            return label
    return "user_folder"


@dataclass(frozen=True, slots=True)
class PresetIdentity:
    """All six facts about one preset file, kept separate."""

    path: Path
    file_format: FileFormat | None
    origin_generation: Generation | None
    provenance: Provenance
    compatible_renderers: tuple[str, ...]
    compatibility_reason: str

    @property
    def recognised(self) -> bool:
        return self.file_format is not None

    @property
    def is_legacy_in_serum2_library(self) -> bool:
        """A Serum 1 patch shipped inside a Serum 2 library.

        This is the exact population that made a Serum-2-only machine look
        broken: thousands of files the user reasonably thinks of as "my Serum 2
        presets" which only Serum 1 can actually render.
        """

        return self.file_format == "fxp" and self.provenance == "serum2_factory"

    def renderable_with(self, available: Sequence[str]) -> tuple[str, ...]:
        """Which of ``available`` generations can render this preset."""

        return tuple(
            name for name in self.compatible_renderers if name in set(available)
        )

    def pending_reason(self, capabilities: Mapping[str, object]) -> str | None:
        """Why this preset cannot be processed now, as a stable code.

        ``capabilities`` maps a generation to something with ``.available`` and
        ``.status`` (a :class:`core.synth_capability.SynthCapability`).
        Returns None when the preset is processable.
        """

        if not self.recognised:
            return PENDING_CLASSIFICATION_UNKNOWN
        statuses: list[str] = []
        for generation in self.compatible_renderers:
            capability = capabilities.get(generation)
            if capability is None:
                statuses.append("not_installed")
                continue
            if getattr(capability, "available", False):
                return None
            statuses.append(str(getattr(capability, "status", "not_installed")))
        # No compatible generation is usable. Report the most specific reason
        # for the *first* compatible generation, which is the one a user would
        # need to install.
        primary = self.compatible_renderers[0] if self.compatible_renderers else ""
        status = statuses[0] if statuses else "not_installed"
        if status in {"renderer_invalid", "not_hostable"}:
            return (
                PENDING_SERUM1_RENDERER_INVALID
                if primary == "serum1"
                else PENDING_SERUM2_RENDERER_INVALID
            )
        if primary == "serum1":
            # A .fxp found inside a Serum 2 library is the case worth naming
            # precisely, because "install Serum 1" is surprising advice to
            # someone who only bought Serum 2.
            if self.is_legacy_in_serum2_library:
                return PENDING_UNSUPPORTED_LEGACY_FORMAT
            return PENDING_SERUM1_NOT_INSTALLED
        return PENDING_SERUM2_NOT_INSTALLED

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "file_format": self.file_format,
            "origin_generation": self.origin_generation,
            "provenance": self.provenance,
            "compatible_renderers": list(self.compatible_renderers),
            "compatibility_reason": self.compatibility_reason,
            "legacy_in_serum2_library": self.is_legacy_in_serum2_library,
        }


def identify_preset(path: Path, *, env: PlatformEnv | None = None) -> PresetIdentity:
    """Resolve all six identity facts for one preset file."""

    path = Path(path)
    container = file_format_for(path)
    if container is None:
        return PresetIdentity(
            path=path,
            file_format=None,
            origin_generation=None,
            provenance=provenance_for(path, env=env),
            compatible_renderers=(),
            compatibility_reason="not a recognised Serum patch container",
        )
    origin: Generation = "serum1" if container == "fxp" else "serum2"
    return PresetIdentity(
        path=path,
        file_format=container,
        origin_generation=origin,
        provenance=provenance_for(path, env=env),
        compatible_renderers=COMPATIBLE_RENDERERS[container],
        compatibility_reason=COMPATIBILITY_REASONS[container],
    )


def summarise_identities(
    identities: Sequence[PresetIdentity],
    capabilities: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Counts a support bundle needs to answer "why only N of my M presets?".

    Deliberately aggregate: a 5,000-preset library must never produce 5,000
    log lines.
    """

    by_format: dict[str, int] = {}
    by_origin: dict[str, int] = {}
    by_provenance: dict[str, int] = {}
    legacy_in_s2 = 0
    unrecognised = 0
    pending_by_reason: dict[str, int] = {}
    processable = 0
    by_renderer: dict[str, int] = {}

    for identity in identities:
        if not identity.recognised:
            unrecognised += 1
            continue
        by_format[str(identity.file_format)] = by_format.get(str(identity.file_format), 0) + 1
        by_origin[str(identity.origin_generation)] = (
            by_origin.get(str(identity.origin_generation), 0) + 1
        )
        by_provenance[identity.provenance] = by_provenance.get(identity.provenance, 0) + 1
        if identity.is_legacy_in_serum2_library:
            legacy_in_s2 += 1
        if capabilities is not None:
            reason = identity.pending_reason(capabilities)
            if reason is None:
                processable += 1
                usable = identity.renderable_with(
                    [
                        name
                        for name, capability in capabilities.items()
                        if getattr(capability, "available", False)
                    ]
                )
                if usable:
                    by_renderer[usable[0]] = by_renderer.get(usable[0], 0) + 1
            else:
                pending_by_reason[reason] = pending_by_reason.get(reason, 0) + 1

    summary: dict[str, object] = {
        "discovered": len(identities),
        "unrecognised": unrecognised,
        "by_file_format": by_format,
        "by_origin_generation": by_origin,
        "by_provenance": by_provenance,
        "legacy_fxp_in_serum2_library": legacy_in_s2,
    }
    if capabilities is not None:
        summary.update(
            {
                "processable": processable,
                "processable_by_renderer": by_renderer,
                "pending": sum(pending_by_reason.values()),
                "pending_by_reason": pending_by_reason,
            }
        )
    return summary


__all__ = [
    "COMPATIBILITY_REASONS",
    "COMPATIBLE_RENDERERS",
    "FXP_SERUM2_FINDING",
    "BLOCKED_PENDING_REASONS",
    "blocked_reasons_for",
    "PENDING_AWAITING_PROCESSING",
    "PENDING_CLASSIFICATION_UNKNOWN",
    "PENDING_REASONS",
    "PENDING_REASON_LABELS",
    "PENDING_RENDER_FAILED",
    "PENDING_SERUM1_NOT_INSTALLED",
    "PENDING_SERUM1_RENDERER_INVALID",
    "PENDING_SERUM2_NOT_INSTALLED",
    "PENDING_SERUM2_RENDERER_INVALID",
    "PENDING_UNSUPPORTED_LEGACY_FORMAT",
    "FileFormat",
    "Generation",
    "PresetIdentity",
    "Provenance",
    "file_format_for",
    "identify_preset",
    "provenance_for",
    "summarise_identities",
]
