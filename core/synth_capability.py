"""What Serum engines this computer can actually run, right now.

PatchLab previously answered that question once, at import time, from
``core.platform_env.ENV`` -- a module-level constant. That is wrong in two
directions: a user who installs Serum 2 after launch stays locked out until
they restart, and a user whose plug-in moved or broke is told it is fine.

This module keeps an explicit, refreshable capability model with a *tiered*
cost profile, because honestly answering "is Serum 2 available?" ranges from a
handful of ``stat`` calls to instantiating a synthesiser:

* **Tier 1 (cheap, milliseconds).** Existence, readability, hostable format and
  a (path, size, mtime) signature over every known candidate. This is enough to
  notice that a plug-in appeared, vanished or was replaced, and it is what the
  UI uses for a "quick fresh check" before refusing an operation.
* **Tier 2 (expensive, seconds).** Actually open the plug-in and render its
  init patch. Only run when tier 1's signature changed, or when a caller
  explicitly needs proof. Cached against the signature so it never repeats for
  an unchanged install.

Statuses distinguish the four cases that matter for diagnosis, rather than
collapsing to a boolean:

``not_installed``        no candidate path exists
``not_hostable``         a binary exists but no hostable format among them
``renderer_invalid``     deep validation opened it and it failed
``present_unvalidated``  tier 1 passed, tier 2 has not been run yet
``ready``                deep validated on this exact binary signature
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from core.diagnostics import DIAGNOSTIC_SCHEMA_VERSION, record, record_decision, recorder
from core.platform_env import ENV, PlatformEnv
from core.renderer_selection import RendererSelection, select_renderer


SUBSYSTEM = "synth-capability"

CapabilityStatus = Literal[
    "not_installed",
    "not_hostable",
    "renderer_invalid",
    "present_unvalidated",
    "ready",
]

#: Statuses from which PatchLab may start work that needs this generation.
USABLE_STATUSES: frozenset[str] = frozenset({"present_unvalidated", "ready"})

GENERATIONS: tuple[str, ...] = ("serum1", "serum2")

CACHE_FILENAME = "synth-capability.json"

#: A deep validation result is trusted for this long even if nothing changed,
#: so a plug-in that was broken by an OS update is eventually re-checked.
DEEP_RESULT_TTL_SECONDS = 24 * 60 * 60


def display_name(generation: str) -> str:
    return "Serum 2" if str(generation) == "serum2" else "Serum 1"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SynthCapability:
    """One generation's availability, with the reason behind it."""

    generation: str
    status: CapabilityStatus
    reason: str
    installed: bool
    validated: bool
    usable_renderers: tuple[str, ...]
    preferred_renderer: str
    preferred_path: str
    signature: str
    checked_at: str
    validation_detail: str = ""
    candidates: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        """Whether PatchLab may start work requiring this generation."""

        return self.status in USABLE_STATUSES

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "status": self.status,
            "reason": self.reason,
            "installed": self.installed,
            "validated": self.validated,
            "available": self.available,
            "usable_renderers": list(self.usable_renderers),
            "preferred_renderer": self.preferred_renderer,
            "preferred_path": self.preferred_path,
            "signature": self.signature,
            "checked_at": self.checked_at,
            "validation_detail": self.validation_detail,
            "candidates": [dict(item) for item in self.candidates],
        }

    def user_message(self) -> str:
        """One concise, non-technical sentence."""

        name = display_name(self.generation)
        if self.available:
            return f"{name} is available."
        if self.status == "not_installed":
            return f"{name} is required. Install {name} and try again."
        if self.status == "not_hostable":
            return (
                f"PatchLab found {name}, but not in a format it can use. "
                f"Install the {name} VST3 or AU version."
            )
        return (
            f"PatchLab found {name} but couldn't start it. "
            f"Check your {name} installation and try again."
        )


@dataclass(frozen=True, slots=True)
class CapabilitySnapshot:
    """Both generations, captured together."""

    capabilities: Mapping[str, SynthCapability]
    captured_at: str
    deep: bool
    elapsed_ms: float

    def for_generation(self, generation: str) -> SynthCapability:
        return self.capabilities[str(generation)]

    def available(self, generation: str) -> bool:
        return self.for_generation(generation).available

    @property
    def available_generations(self) -> tuple[str, ...]:
        return tuple(name for name in GENERATIONS if self.capabilities[name].available)

    @property
    def unavailable_generations(self) -> tuple[str, ...]:
        return tuple(
            name for name in GENERATIONS if not self.capabilities[name].available
        )

    @property
    def any_available(self) -> bool:
        return bool(self.available_generations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "deep": self.deep,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "available_generations": list(self.available_generations),
            "unavailable_generations": list(self.unavailable_generations),
            "capabilities": {
                name: capability.as_dict()
                for name, capability in self.capabilities.items()
            },
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        }

    def signature(self) -> str:
        return "|".join(
            f"{name}:{self.capabilities[name].signature}" for name in GENERATIONS
        )


# ---------------------------------------------------------------------------
# Tier 1: cheap signature
# ---------------------------------------------------------------------------


def candidate_signature(generation: str, *, env: PlatformEnv | None = None) -> str:
    """A cheap fingerprint of every candidate path's existence and identity.

    Pure ``stat`` calls -- no plug-in is opened. This changes when a plug-in is
    installed, removed, moved or updated, which is exactly the trigger for
    re-running the expensive validation and nothing else.
    """

    env = env or ENV
    parts: list[str] = []
    for candidate in env.plugin_candidates:
        if str(candidate.synth) != str(generation):
            continue
        path = Path(candidate.path)
        try:
            stat = path.stat()
            parts.append(
                f"{candidate.format}:{path}:1:{stat.st_size}:{int(stat.st_mtime)}"
            )
        except OSError:
            parts.append(f"{candidate.format}:{path}:0::")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Tier 2: deep validation
# ---------------------------------------------------------------------------


def _validate_renderer(selection: RendererSelection) -> tuple[bool, str]:
    """Open the selected plug-in and render its init patch.

    Returns ``(ok, detail)``. Never raises: a plug-in that crashes the host is
    a capability answer, not a PatchLab failure.
    """

    if selection.selected is None:
        return False, "no renderer was selected"
    try:
        from core.plugin_host import verify_default_render
        from core.platform_env import PluginCandidate

        candidate = PluginCandidate(
            selection.requested_synth,  # type: ignore[arg-type]
            selection.selected.format,  # type: ignore[arg-type]
            Path(selection.selected.path),
        )
        peak, rms, methods = verify_default_render(candidate)
        # An init patch that renders pure silence means the plug-in loaded but
        # is not usable for rendering, which is a different failure from "the
        # binary is missing" and must be reported as such.
        if rms <= -90.0:
            return False, (
                f"init patch rendered silence ({rms:.1f} dBFS); the plug-in "
                "loaded but produced no audio"
            )
        required = {"load_preset", "load_state"}
        missing = sorted(required - set(methods))
        if missing:
            return False, f"host is missing required methods: {', '.join(missing)}"
        return True, f"init patch rendered at {rms:.1f} dBFS (peak {peak:.1f} dBFS)"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_path() -> Path | None:
    root = recorder().root
    if root is None:
        return None
    return root / CACHE_FILENAME


def _load_cache() -> dict[str, Any]:
    path = _cache_path()
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(payload: Mapping[str, Any]) -> None:
    path = _cache_path()
    if path is None:
        return
    try:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


def _cached_deep_result(generation: str, signature: str) -> tuple[bool, str] | None:
    """A previous deep validation for this exact binary signature, if fresh."""

    entry = _load_cache().get(str(generation))
    if not isinstance(entry, dict):
        return None
    if str(entry.get("signature")) != signature:
        return None
    validated_at = float(entry.get("validated_at", 0.0))
    if time.time() - validated_at > DEEP_RESULT_TTL_SECONDS:
        return None
    if "ok" not in entry:
        return None
    return bool(entry["ok"]), str(entry.get("detail", ""))


def _store_deep_result(generation: str, signature: str, ok: bool, detail: str) -> None:
    cache = _load_cache()
    cache[str(generation)] = {
        "signature": signature,
        "ok": bool(ok),
        "detail": detail,
        "validated_at": time.time(),
    }
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Capability resolution
# ---------------------------------------------------------------------------


def capability_for(
    generation: str,
    *,
    env: PlatformEnv | None = None,
    deep: bool = False,
    operation_id: str = "",
    use_cache: bool = True,
) -> SynthCapability:
    """Resolve one generation's capability.

    ``deep=False`` is the cheap path and never opens a plug-in. ``deep=True``
    validates, reusing a cached result for an unchanged binary signature.
    """

    env = env or ENV
    generation = str(generation)
    signature = candidate_signature(generation, env=env)
    selection = select_renderer(
        generation,
        env=env,
        operation_id=operation_id,
        phase="capability-refresh",
        log_decision=False,
    )
    candidates = tuple(item.as_dict() for item in selection.candidates)
    now = datetime.now(timezone.utc).isoformat()
    usable = tuple(
        item.label for item in selection.candidates if item.exists and item.readable and item.hostable
    )
    any_exists = any(item.exists for item in selection.candidates)

    def build(status: CapabilityStatus, reason: str, *, validated: bool = False, detail: str = "") -> SynthCapability:
        return SynthCapability(
            generation=generation,
            status=status,
            reason=reason,
            installed=any_exists,
            validated=validated,
            usable_renderers=usable,
            preferred_renderer=selection.renderer,
            preferred_path=selection.selected.path if selection.selected else "",
            signature=signature,
            checked_at=now,
            validation_detail=detail,
            candidates=candidates,
        )

    if not selection.available:
        if not any_exists:
            return build(
                "not_installed",
                f"no {display_name(generation)} plug-in exists at any known location",
            )
        return build(
            "not_hostable",
            f"a {display_name(generation)} binary exists but none is in a hostable "
            f"verified format: {selection.reason}",
        )

    if not deep:
        cached = _cached_deep_result(generation, signature) if use_cache else None
        if cached is not None:
            ok, detail = cached
            if ok:
                return build("ready", f"{selection.reason} (validated earlier)", validated=True, detail=detail)
            return build("renderer_invalid", f"deep validation failed earlier: {detail}", detail=detail)
        return build(
            "present_unvalidated",
            f"{selection.reason}; not yet opened, so treated as present but unproven",
        )

    cached = _cached_deep_result(generation, signature) if use_cache else None
    if cached is None:
        ok, detail = _validate_renderer(selection)
        _store_deep_result(generation, signature, ok, detail)
    else:
        ok, detail = cached
    if ok:
        return build("ready", selection.reason, validated=True, detail=detail)
    return build("renderer_invalid", f"deep validation failed: {detail}", detail=detail)


def refresh_capabilities(
    *,
    env: PlatformEnv | None = None,
    deep: bool = False,
    operation_id: str = "",
    use_cache: bool = True,
    reason: str = "",
) -> CapabilitySnapshot:
    """Resolve both generations and record the decision.

    Cheap enough to call before refusing an operation. Never triggers a preset
    rescan and never touches the library database.
    """

    started = time.perf_counter()
    record(
        SUBSYSTEM,
        "capability_refresh_started",
        f"capability refresh ({'deep' if deep else 'quick'})",
        operation_id=operation_id,
        phase="capability-refresh",
        decision_reason=reason or "explicit capability refresh",
        deep=deep,
    )
    capabilities = {
        generation: capability_for(
            generation,
            env=env,
            deep=deep,
            operation_id=operation_id,
            use_cache=use_cache,
        )
        for generation in GENERATIONS
    }
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    snapshot = CapabilitySnapshot(
        capabilities=capabilities,
        captured_at=datetime.now(timezone.utc).isoformat(),
        deep=deep,
        elapsed_ms=elapsed_ms,
    )
    record_decision(
        SUBSYSTEM,
        "capability_refresh",
        outcome=", ".join(
            f"{name}={capabilities[name].status}" for name in GENERATIONS
        ),
        reason="; ".join(f"{name}: {capabilities[name].reason}" for name in GENERATIONS),
        operation_id=operation_id,
        phase="capability-refresh",
        deep=deep,
        elapsed_ms=round(elapsed_ms, 3),
        available_generations=list(snapshot.available_generations),
        serum1=capabilities["serum1"].as_dict(),
        serum2=capabilities["serum2"].as_dict(),
    )
    record(
        SUBSYSTEM,
        "capability_refresh_completed",
        f"capability refresh finished in {elapsed_ms:.1f} ms",
        operation_id=operation_id,
        phase="capability-refresh",
        deep=deep,
        elapsed_ms=round(elapsed_ms, 3),
    )
    return snapshot


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityChange:
    """What changed between two snapshots."""

    became_available: tuple[str, ...]
    became_unavailable: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.became_available or self.became_unavailable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "became_available": list(self.became_available),
            "became_unavailable": list(self.became_unavailable),
        }


def diff_capabilities(
    previous: CapabilitySnapshot | None, current: CapabilitySnapshot
) -> CapabilityChange:
    """Which generations newly became (un)available."""

    if previous is None:
        return CapabilityChange(became_available=(), became_unavailable=())
    gained: list[str] = []
    lost: list[str] = []
    for generation in GENERATIONS:
        was = previous.for_generation(generation).available
        now = current.for_generation(generation).available
        if now and not was:
            gained.append(generation)
        elif was and not now:
            lost.append(generation)
    change = CapabilityChange(
        became_available=tuple(gained), became_unavailable=tuple(lost)
    )
    if change.changed:
        record(
            SUBSYSTEM,
            "capability_changed",
            "synth availability changed: "
            + ", ".join(
                [f"+{name}" for name in gained] + [f"-{name}" for name in lost]
            ),
            severity="info",
            phase="capability-refresh",
            decision_reason=(
                "a plug-in appeared or disappeared since the last refresh, so "
                "pending work for that generation may now be processable"
            ),
            **change.as_dict(),
        )
    return change


__all__ = [
    "CACHE_FILENAME",
    "DEEP_RESULT_TTL_SECONDS",
    "GENERATIONS",
    "USABLE_STATUSES",
    "CapabilityChange",
    "CapabilitySnapshot",
    "CapabilityStatus",
    "SynthCapability",
    "candidate_signature",
    "capability_for",
    "diff_capabilities",
    "display_name",
    "refresh_capabilities",
]
