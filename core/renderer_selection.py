"""The one place PatchLab decides which Serum renderer to use, and why.

Before this module existed, three call sites each made that decision on their
own with a hardcoded plug-in format:

* ``core/matcher.py::_init_render_worker`` did
  ``next(item for item in ENV.plugins_for("serum1") if item.format == "VST2")``
  -- unconditionally, for *both* generations, whatever the user had asked to
  match.  On a Mac with Serum 2 but no Serum 1 that ``next()`` raised a bare
  ``StopIteration`` inside a multiprocessing pool initializer.
* ``core/preset_scan.py::SequentialSerum1VST2`` raised
  ``RuntimeError("Verified Serum 1 VST2 binary is unavailable.")`` whenever any
  ``.fxp`` preset existed, even though Serum 2's own factory library ships
  thousands of ``.fxp`` files and ``core/plugin_host.load_preset`` already
  implements verified Serum 1 strategies on VST3 and AU as well as VST2.
* ``core/local_library.py`` repeated the matcher's ``next()`` pattern for
  Serum 2.

Both reported production bugs were that duplication.  Renderer requirements
must depend on the work actually requested, selection must consider PatchLab's
real verified hierarchy rather than one hardcoded format, and every acceptance
and rejection must be explainable in a support bundle.

Format preference deliberately keeps each generation's historically-used format
first (Serum 1 -> VST2, Serum 2 -> VST3).  On any machine where the previous
code worked, this module picks exactly the same binary, so render output and
the frozen accuracy baselines are unchanged.  The additional formats are
fallbacks that only engage where the old code raised.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from core.diagnostics import record_decision, recorder
from core.platform_env import ENV, PlatformEnv, PluginCandidate


SynthVersion = Literal["serum1", "serum2"]

#: Verified render/load formats per generation, highest preference first.
#:
#: The first entry of each tuple is the format PatchLab has always used for
#: that generation, so selection is a no-op wherever it is installed.  The
#: remaining entries mirror the strategies ``core/plugin_host.load_preset``
#: already implements (``S2-dawdreamer-vst3-fxp-state`` for a Serum 1 preset in
#: a VST3 host, ``S5-dawdreamer-au-*`` for AU) and are only reached when the
#: preferred format is absent.
PREFERRED_FORMATS: Mapping[str, tuple[str, ...]] = {
    "serum1": ("VST2", "VST3", "AU"),
    "serum2": ("VST3", "AU"),
}

SUBSYSTEM = "renderer-selection"


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """Everything a remote diagnosis needs about one plug-in candidate."""

    synth: str
    format: str
    path: str
    exists: bool
    readable: bool
    hostable: bool
    accepted: bool
    preference_rank: int | None
    rejection_reason: str = ""
    size_bytes: int | None = None
    binary_sha1: str = ""
    plugin_version: str = ""
    architecture_compatible: bool | None = None

    @property
    def label(self) -> str:
        return f"{self.synth}/{self.format}"

    def as_dict(self) -> dict[str, object]:
        return {
            "renderer": self.label,
            "synth": self.synth,
            "format": self.format,
            "path": self.path,
            "exists": self.exists,
            "readable": self.readable,
            "hostable": self.hostable,
            "accepted": self.accepted,
            "preference_rank": self.preference_rank,
            "rejection_reason": self.rejection_reason,
            "size_bytes": self.size_bytes,
            "binary_sha1": self.binary_sha1,
            "plugin_version": self.plugin_version,
            "architecture_compatible": self.architecture_compatible,
        }


@dataclass(frozen=True, slots=True)
class RendererSelection:
    """The outcome of one renderer decision, with its full reasoning."""

    requested_synth: str
    selected: CandidateReport | None
    reason: str
    candidates: tuple[CandidateReport, ...]

    @property
    def available(self) -> bool:
        return self.selected is not None

    @property
    def renderer(self) -> str:
        return self.selected.label if self.selected is not None else ""

    @property
    def plugin_format(self) -> str:
        return self.selected.format if self.selected is not None else ""

    @property
    def path(self) -> Path | None:
        return Path(self.selected.path) if self.selected is not None else None

    def rejections(self) -> tuple[CandidateReport, ...]:
        return tuple(item for item in self.candidates if not item.accepted)

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_serum": self.requested_synth,
            "selected": self.renderer or None,
            "selected_path": self.selected.path if self.selected else None,
            "reason": self.reason,
            "available": self.available,
            "candidates": [item.as_dict() for item in self.candidates],
        }

    def user_message(self) -> str:
        """A concise, non-technical explanation for the UI."""

        generation = "Serum 2" if self.requested_synth == "serum2" else "Serum 1"
        if self.available:
            return f"{generation} renderer ready ({self.plugin_format})."
        if not self.candidates:
            return f"PatchLab has no known {generation} plug-in location on this system."
        if all(not item.exists for item in self.candidates):
            return (
                f"PatchLab couldn't find a {generation} plug-in. "
                f"Check that {generation} is installed."
            )
        unreadable = [item for item in self.candidates if item.exists and not item.readable]
        if unreadable:
            return (
                f"PatchLab found {generation} but could not read it. "
                "Check the plug-in's file permissions."
            )
        return f"PatchLab couldn't start {generation}."


class RendererUnavailableError(RuntimeError):
    """Raised instead of a bare ``StopIteration`` when no renderer qualifies.

    Carries the whole selection so the parent process, the support bundle and
    the user-facing message all describe the same decision.
    """

    def __init__(self, selection: RendererSelection, *, context: str = "") -> None:
        detail = "; ".join(
            f"{item.label}: {item.rejection_reason}" for item in selection.rejections()
        ) or "no candidate locations are known for this platform"
        where = f" while {context}" if context else ""
        super().__init__(
            f"No usable {selection.requested_synth} renderer is available{where}. "
            f"Tried: {detail}."
        )
        self.selection = selection
        self.requested_synth = selection.requested_synth
        self.context = context

    @property
    def user_message(self) -> str:
        return self.selection.user_message()


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _readable(path: Path) -> bool:
    try:
        if path.is_dir():
            # macOS AU/VST3 plug-ins are bundles; being able to list them is
            # the meaningful readability check.
            return os.access(path, os.R_OK | os.X_OK)
        return os.access(path, os.R_OK)
    except OSError:
        return False


def _bundle_size(path: Path) -> int | None:
    """Cheap size probe.  Bundles report their own entry, not a deep walk."""

    try:
        if path.is_dir():
            binary = path / "Contents" / "MacOS"
            if binary.is_dir():
                return sum(
                    item.stat().st_size for item in binary.iterdir() if item.is_file()
                )
            return None
        return path.stat().st_size
    except OSError:
        return None


def _plugin_version(path: Path) -> str:
    """Read a macOS bundle's short version string without parsing binaries."""

    try:
        plist = path / "Contents" / "Info.plist"
        if not plist.is_file():
            return ""
        import plistlib

        with plist.open("rb") as handle:
            data = plistlib.load(handle)
        for key in ("CFBundleShortVersionString", "CFBundleVersion"):
            value = data.get(key)
            if value:
                return str(value)
    except Exception:
        # Documented swallow: the plug-in version is a nice-to-have diagnostic
        # detail, not a selection input. A malformed Info.plist must never stop
        # PatchLab from using an otherwise valid plug-in. Empty string is
        # reported in the bundle as "version=?", which is honest.
        return ""
    return ""


def _architecture_compatible(path: Path, env: PlatformEnv) -> bool | None:
    """Whether a macOS bundle's binary includes this machine's architecture.

    ``None`` means "not determinable cheaply", which is an honest answer and
    distinguishable in the bundle from ``False``.
    """

    if env.branch != "macos":
        return None
    try:
        binary_dir = path / "Contents" / "MacOS"
        if not binary_dir.is_dir():
            return None
        binaries = [item for item in binary_dir.iterdir() if item.is_file()]
        if not binaries:
            return None
        import struct

        with binaries[0].open("rb") as handle:
            head = handle.read(4)
            if head == b"\xca\xfe\xba\xbe":
                # Universal binary: assume it carries this slice.
                return True
            if head in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
                cpu = struct.unpack("<i", handle.read(4))[0]
                arm64 = 0x0100000C
                x86_64 = 0x01000007
                machine = str(env.machine).casefold()
                if machine.startswith("arm"):
                    return cpu == arm64
                return cpu == x86_64
    except Exception:
        return None
    return None


def inspect_candidate(
    candidate: PluginCandidate,
    *,
    env: PlatformEnv | None = None,
    preference_rank: int | None = None,
    include_hash: bool = False,
) -> CandidateReport:
    """Describe one candidate without attempting to instantiate it.

    Deliberately cheap: stat, permission and Info.plist reads only.  Actually
    opening a plug-in costs seconds and is the render workers' job.
    """

    env = env or ENV
    path = Path(candidate.path)
    exists = False
    try:
        exists = path.exists()
    except OSError:
        exists = False
    readable = _readable(path) if exists else False
    reason = ""
    if not exists:
        reason = f"path does not exist: {path}"
    elif not readable:
        reason = f"path is not readable: {path}"
    elif not candidate.hostable:
        reason = f"{candidate.format} is not hostable by PatchLab's render engines"
    digest = ""
    if include_hash and exists and readable and not path.is_dir():
        digest = recorder().cached_file_digest(path)
    return CandidateReport(
        synth=str(candidate.synth),
        format=str(candidate.format),
        path=str(path),
        exists=exists,
        readable=readable,
        hostable=bool(candidate.hostable),
        accepted=False,
        preference_rank=preference_rank,
        rejection_reason=reason,
        size_bytes=_bundle_size(path) if exists else None,
        binary_sha1=digest,
        plugin_version=_plugin_version(path) if exists else "",
        architecture_compatible=_architecture_compatible(path, env) if exists else None,
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_renderer(
    synth: str,
    *,
    env: PlatformEnv | None = None,
    preferred_formats: Sequence[str] | None = None,
    operation_id: str = "",
    phase: str = "",
    log_decision: bool = True,
    include_hash: bool = False,
) -> RendererSelection:
    """Choose the renderer for ``synth`` and explain the choice.

    Never raises for an absent plug-in: the caller decides whether an
    unavailable renderer is fatal for the work it was asked to do.
    """

    env = env or ENV
    requested = str(synth)
    order = tuple(preferred_formats or PREFERRED_FORMATS.get(requested, ()))
    ranks = {value: index for index, value in enumerate(order)}

    reports: list[CandidateReport] = []
    for candidate in env.plugin_candidates:
        if str(candidate.synth) != requested:
            continue
        reports.append(
            inspect_candidate(
                candidate,
                env=env,
                preference_rank=ranks.get(str(candidate.format)),
                include_hash=include_hash,
            )
        )

    viable = [
        item
        for item in reports
        if item.exists and item.readable and item.hostable and item.preference_rank is not None
    ]
    # Sort by preference only. `viable` is built in ``plugin_candidates``
    # declaration order and Python's sort is stable, so ties keep that order --
    # which is exactly what the previous ``next(...)`` did. Adding the path as a
    # tie-breaker would have silently flipped a machine with Serum installed in
    # both ~/Library and /Library from the user copy to the system copy, and
    # those can be different plug-in versions.
    viable.sort(key=lambda item: item.preference_rank)

    selected: CandidateReport | None = None
    if viable:
        winner = viable[0]
        selected = CandidateReport(
            synth=winner.synth,
            format=winner.format,
            path=winner.path,
            exists=winner.exists,
            readable=winner.readable,
            hostable=winner.hostable,
            accepted=True,
            preference_rank=winner.preference_rank,
            rejection_reason="",
            size_bytes=winner.size_bytes,
            binary_sha1=winner.binary_sha1,
            plugin_version=winner.plugin_version,
            architecture_compatible=winner.architecture_compatible,
        )
        rank = winner.preference_rank or 0
        if rank == 0:
            reason = (
                f"{winner.format} is the highest-preference verified renderer for "
                f"{requested} and passed path, readability and hosting checks"
            )
        else:
            rejected_above = [
                item.format
                for item in reports
                if item.preference_rank is not None and item.preference_rank < rank
            ]
            reason = (
                f"{winner.format} accepted as fallback; preferred "
                f"{', '.join(sorted(set(rejected_above))) or 'format'} unavailable"
            )
    elif not reports:
        reason = f"no {requested} plug-in locations are known for platform {env.branch}"
    else:
        reason = (
            f"every known {requested} candidate was rejected: "
            + "; ".join(f"{item.label} ({item.rejection_reason})" for item in reports)
        )

    # Mark the chosen entry as accepted in the reported candidate list so the
    # bundle shows one coherent accept/reject table.
    final: list[CandidateReport] = []
    for item in reports:
        if (
            selected is not None
            and item.path == selected.path
            and item.format == selected.format
        ):
            final.append(selected)
            continue
        if item.preference_rank is None and not item.rejection_reason:
            item = CandidateReport(
                synth=item.synth,
                format=item.format,
                path=item.path,
                exists=item.exists,
                readable=item.readable,
                hostable=item.hostable,
                accepted=False,
                preference_rank=None,
                rejection_reason=(
                    f"{item.format} is not a verified render format for {requested}"
                ),
                size_bytes=item.size_bytes,
                binary_sha1=item.binary_sha1,
                plugin_version=item.plugin_version,
                architecture_compatible=item.architecture_compatible,
            )
        elif selected is not None and not item.rejection_reason:
            item = CandidateReport(
                synth=item.synth,
                format=item.format,
                path=item.path,
                exists=item.exists,
                readable=item.readable,
                hostable=item.hostable,
                accepted=False,
                preference_rank=item.preference_rank,
                rejection_reason=(
                    f"usable, but {selected.format} has higher preference"
                ),
                size_bytes=item.size_bytes,
                binary_sha1=item.binary_sha1,
                plugin_version=item.plugin_version,
                architecture_compatible=item.architecture_compatible,
            )
        final.append(item)

    selection = RendererSelection(
        requested_synth=requested,
        selected=selected,
        reason=reason,
        candidates=tuple(final),
    )
    if log_decision:
        record_decision(
            SUBSYSTEM,
            f"renderer_selection[{requested}]",
            outcome=selection.renderer or "unavailable",
            reason=reason,
            operation_id=operation_id,
            phase=phase,
            requested_serum=requested,
            selected=selection.renderer or None,
            selected_path=selected.path if selected else None,
            preference_order=list(order),
            candidates=[item.as_dict() for item in final],
        )
    return selection


def require_renderer(
    synth: str,
    *,
    env: PlatformEnv | None = None,
    operation_id: str = "",
    phase: str = "",
    context: str = "",
    include_hash: bool = False,
) -> RendererSelection:
    """Select a renderer, raising :class:`RendererUnavailableError` if none."""

    env = env or ENV
    selection = select_renderer(
        synth,
        env=env,
        operation_id=operation_id,
        phase=phase,
        include_hash=include_hash,
    )
    if not selection.available:
        raise RendererUnavailableError(selection, context=context)
    return selection


# ---------------------------------------------------------------------------
# Opening a renderer: the ONE way any code gets a live plug-in host
# ---------------------------------------------------------------------------


def renderer_candidate(selection: "RendererSelection"):
    """Rebuild the PluginCandidate a selection chose, without re-discovering it."""

    from core.platform_env import PluginCandidate

    assert selection.selected is not None
    return PluginCandidate(
        selection.requested_synth,  # type: ignore[arg-type]
        selection.selected.format,  # type: ignore[arg-type]
        Path(str(selection.selected.path)),
    )


def open_renderer(
    synth: str,
    *,
    env: PlatformEnv | None = None,
    operation_id: str = "",
    phase: str = "",
    context: str = "",
) -> tuple[Any, Any, "RendererSelection"]:
    """Return ``(engine, processor, selection)`` for ``synth`` on this machine.

    Every user-facing path that needs a live Serum host goes through here:
    preview, octave pre-render, factory preview, preset export, export
    verification, library rendering and Match's own render workers.  They used
    to each run their own ``next(item for item in ENV.plugins_for(...) if
    item.format == "VST2")``, which raised a bare ``StopIteration`` on a machine
    without that exact format -- surfacing to users as "StopIteration" with no
    indication that a Serum install was the problem, and to a multiprocessing
    initializer as an unexplained dead worker.

    A missing renderer now always raises :class:`RendererUnavailableError`,
    which carries the full selection report and a sentence a user can act on.
    """

    from core.plugin_host import make_dawdreamer_processor

    selection = require_renderer(
        synth, env=env, operation_id=operation_id, phase=phase, context=context
    )
    engine, processor = make_dawdreamer_processor(renderer_candidate(selection))
    return engine, processor, selection


# ---------------------------------------------------------------------------
# Preflight across a set of required generations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RendererPreflight:
    """Which of the requested generations this machine can actually render.

    A mixed library is not all-or-nothing.  ``supported`` proceeds;
    ``unsupported`` is reported clearly instead of aborting the whole job.
    """

    requested: tuple[str, ...]
    selections: Mapping[str, RendererSelection]

    @property
    def supported(self) -> tuple[str, ...]:
        return tuple(
            synth for synth in self.requested if self.selections[synth].available
        )

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(
            synth for synth in self.requested if not self.selections[synth].available
        )

    @property
    def fully_supported(self) -> bool:
        return not self.unsupported

    @property
    def any_supported(self) -> bool:
        return bool(self.supported)

    def selection_for(self, synth: str) -> RendererSelection:
        return self.selections[synth]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_generations": list(self.requested),
            "supported_generations": list(self.supported),
            "unsupported_generations": list(self.unsupported),
            "selections": {
                synth: selection.as_dict() for synth, selection in self.selections.items()
            },
        }

    def unsupported_reason(self) -> str:
        return "; ".join(
            f"{synth}: {self.selections[synth].reason}" for synth in self.unsupported
        )

    def user_message(self) -> str:
        if self.fully_supported:
            return "All required Serum renderers are ready."
        missing = ", ".join(
            "Serum 2" if synth == "serum2" else "Serum 1" for synth in self.unsupported
        )
        if self.any_supported:
            ready = ", ".join(
                "Serum 2" if synth == "serum2" else "Serum 1" for synth in self.supported
            )
            return (
                f"PatchLab couldn't start {missing}, so the presets that need it "
                f"were left waiting. Your {ready} presets were processed normally."
            )
        return self.selections[self.unsupported[0]].user_message()


def preflight_renderers(
    generations: Iterable[str],
    *,
    env: PlatformEnv | None = None,
    operation_id: str = "",
    phase: str = "renderer-preflight",
    include_hash: bool = False,
) -> RendererPreflight:
    """Resolve every generation the pending work needs, in one deterministic pass.

    This is intentionally cheap enough to run *before* an expensive scan or
    before creating a process pool, which is the whole point: a user should not
    wait 36 minutes to discover something knowable in milliseconds.
    """

    env = env or ENV
    requested = tuple(dict.fromkeys(str(value) for value in generations))
    selections = {
        synth: select_renderer(
            synth,
            env=env,
            operation_id=operation_id,
            phase=phase,
            include_hash=include_hash,
        )
        for synth in requested
    }
    preflight = RendererPreflight(requested=requested, selections=selections)
    record_decision(
        SUBSYSTEM,
        "renderer_preflight",
        outcome=(
            "all-available"
            if preflight.fully_supported
            else ("partial" if preflight.any_supported else "none-available")
        ),
        reason=(
            "every generation required by the pending work has a usable renderer"
            if preflight.fully_supported
            else preflight.unsupported_reason()
        ),
        operation_id=operation_id,
        phase=phase,
        requested_generations=list(requested),
        supported_generations=list(preflight.supported),
        unsupported_generations=list(preflight.unsupported),
    )
    return preflight


def renderer_inventory(
    *,
    env: PlatformEnv | None = None,
    include_hash: bool = False,
) -> list[dict[str, object]]:
    """Full accept/reject table for every known candidate, for the bundle."""

    env = env or ENV
    inventory: list[dict[str, object]] = []
    for synth in ("serum1", "serum2"):
        selection = select_renderer(
            synth,
            env=env,
            log_decision=False,
            include_hash=include_hash,
        )
        for item in selection.candidates:
            entry = item.as_dict()
            entry["selected_for_generation"] = item.accepted
            inventory.append(entry)
    return inventory


__all__ = [
    "PREFERRED_FORMATS",
    "CandidateReport",
    "RendererPreflight",
    "RendererSelection",
    "RendererUnavailableError",
    "inspect_candidate",
    "preflight_renderers",
    "renderer_inventory",
    "require_renderer",
    "select_renderer",
]
