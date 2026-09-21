"""Decide what to tell the user about synth availability, and when.

Two rules shape everything here.

**Capability and library processing are different things.** Whether PatchLab can
*create* a Serum 2 preset depends only on Serum 2 being installed. Which
existing presets can participate in retrieval depends on how much of the library
has been learned. Conflating them is what would make "you have 327 unprocessed
presets" wrongly block a Match, so this module keeps the two decisions apart:
:func:`evaluate_output_target` gates output, and :func:`synth_available_notice` only ever
*offers* library work.

**Do not nag.** An informational message about a missing synth is worth showing
once, not on every launch. Acknowledgement state is persisted and keyed by the
facts that matter -- which generation, and how many presets -- so the message
returns when something materially changes and stays quiet when it does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from core.diagnostics import record, record_decision
from core.platform_env import ENV, PlatformEnv
from core.synth_capability import (
    CapabilitySnapshot,
    SynthCapability,
    display_name,
    refresh_capabilities,
)


SUBSYSTEM = "capability-ux"
STATE_FILENAME = "capability-notices.json"

#: A count change smaller than this does not re-show an acknowledged notice.
#: One or two new presets is not news; a materially different library is.
MATERIAL_COUNT_DELTA = 25


NoticeKind = Literal["missing_synth_presets", "synth_now_available"]


def _state_path(env: PlatformEnv | None = None) -> Path:
    env = env or ENV
    root = Path(env.app_data_dir) / "diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    return root / STATE_FILENAME


def load_notice_state(env: PlatformEnv | None = None) -> dict[str, Any]:
    path = _state_path(env)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def save_notice_state(state: Mapping[str, Any], env: PlatformEnv | None = None) -> None:
    try:
        path = _state_path(env)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(state), sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Output-target gating (PART 9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutputTargetDecision:
    """Whether PatchLab may create output for a requested generation."""

    generation: str
    allowed: bool
    capability: SynthCapability
    message: str
    rechecked: bool
    became_available: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "allowed": self.allowed,
            "message": self.message,
            "rechecked": self.rechecked,
            "became_available": self.became_available,
            "capability": self.capability.as_dict(),
        }


def evaluate_output_target(
    generation: str,
    *,
    env: PlatformEnv | None = None,
    previous: CapabilitySnapshot | None = None,
    operation_id: str = "",
) -> tuple[OutputTargetDecision, CapabilitySnapshot]:
    """Gate a requested output generation on a *fresh* capability check.

    Always re-checks rather than trusting startup state, so a user who installs
    Serum 2 while PatchLab is open is not told to restart. The check is the cheap
    tier -- a handful of ``stat`` calls -- so this is safe to call on every click.

    Note what this deliberately does **not** consider: how much of the user's
    library has been processed. Creating a Serum 2 preset needs Serum 2, not a
    finished library.
    """

    snapshot = refresh_capabilities(
        env=env,
        operation_id=operation_id,
        reason=f"user requested {generation} output",
    )
    capability = snapshot.for_generation(generation)
    was_available = (
        previous.for_generation(generation).available if previous is not None else None
    )
    became = bool(capability.available and was_available is False)
    name = display_name(generation)
    if capability.available:
        message = ""
    else:
        message = f"{name} is required to create {name} presets. " + (
            f"Install {name} and try again."
            if capability.status == "not_installed"
            else capability.user_message()
        )
    decision = OutputTargetDecision(
        generation=generation,
        allowed=capability.available,
        capability=capability,
        message=message,
        rechecked=True,
        became_available=became,
    )
    record_decision(
        SUBSYSTEM,
        "output_target_capability_check",
        outcome="allowed" if decision.allowed else "blocked",
        reason=capability.reason,
        operation_id=operation_id,
        phase="capability-refresh",
        generation=generation,
        status=capability.status,
        became_available=became,
        renderer=capability.preferred_renderer,
    )
    if not decision.allowed:
        record(
            SUBSYSTEM,
            "output_blocked",
            f"refused to start {generation} output: {capability.status}",
            severity="warning",
            operation_id=operation_id,
            phase="capability-refresh",
            decision_reason=(
                "starting an operation whose required engine is absent would "
                "produce no result; refusing up front is the honest outcome"
            ),
            generation=generation,
            user_message=message,
        )
    return decision, snapshot


# ---------------------------------------------------------------------------
# Notices (PART 6, PART 10, PART 12)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Notice:
    """One concise, actionable message for the user."""

    kind: NoticeKind
    generation: str
    title: str
    body: str
    count: int
    offers_processing: bool = False
    key: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "generation": self.generation,
            "title": self.title,
            "body": self.body,
            "count": self.count,
            "offers_processing": self.offers_processing,
            "key": self.key,
        }


def _notice_key(kind: str, generation: str) -> str:
    return f"{kind}:{generation}"


def _should_show(state: Mapping[str, Any], key: str, count: int) -> bool:
    """Anti-nag: show once, then only when the facts materially change."""

    entry = state.get(key)
    if not isinstance(entry, dict):
        return True
    if not entry.get("acknowledged"):
        return True
    previous = int(entry.get("count", 0) or 0)
    return abs(count - previous) >= MATERIAL_COUNT_DELTA


def missing_synth_notice(
    *,
    snapshot: CapabilitySnapshot,
    pending_counts: Mapping[str, int],
    env: PlatformEnv | None = None,
    operation_id: str = "",
) -> Notice | None:
    """Tell the user once that some presets need a synth they do not have.

    Deliberately reassuring about what still works: a Serum 1 user with Serum 2
    presets in their folder has not broken anything, and their Serum 1 presets
    are processed normally.
    """

    from core.preset_identity import (
        PENDING_SERUM1_NOT_INSTALLED,
        PENDING_SERUM2_NOT_INSTALLED,
        PENDING_UNSUPPORTED_LEGACY_FORMAT,
    )

    groups = {
        "serum1": int(pending_counts.get(PENDING_SERUM1_NOT_INSTALLED, 0))
        + int(pending_counts.get(PENDING_UNSUPPORTED_LEGACY_FORMAT, 0)),
        "serum2": int(pending_counts.get(PENDING_SERUM2_NOT_INSTALLED, 0)),
    }
    state = load_notice_state(env)
    for generation, count in groups.items():
        if count <= 0 or snapshot.for_generation(generation).available:
            continue
        # Seeing the synth *missing* while presets wait for it re-arms the
        # "it is now available" offer: if the user answered Not Now earlier and
        # the synth later disappears and comes back, that return is a genuinely
        # new event and deserves one fresh offer.
        offer_key = _notice_key("synth_now_available", generation)
        if offer_key in state:
            state = {k: v for k, v in state.items() if k != offer_key}
            save_notice_state(state, env)
            record(
                SUBSYSTEM,
                "offer_rearmed",
                f"{generation} is missing again; the availability offer is re-armed",
                operation_id=operation_id,
                decision_reason=(
                    "a synth that goes missing and later returns is a new event, so "
                    "any earlier answer to the availability offer no longer applies"
                ),
                generation=generation,
            )
        key = _notice_key("missing_synth_presets", generation)
        if not _should_show(state, key, count):
            record(
                SUBSYSTEM,
                "notice_suppressed",
                f"missing-{generation} notice suppressed as already acknowledged",
                operation_id=operation_id,
                decision_reason=(
                    "the user acknowledged this and the count has not materially "
                    f"changed (threshold {MATERIAL_COUNT_DELTA})"
                ),
                generation=generation,
                count=count,
            )
            continue
        name = display_name(generation)
        # Describe presets by what they ARE, not by the folder they sit in. A
        # Serum 2 factory folder holds thousands of legacy .fxp files; calling
        # them "Serum 1 presets" (or, worse, "Serum 2 presets") would be false.
        # Serum 2's own .serumpreset files really are Serum 2 presets.
        subject = "legacy .fxp presets" if generation == "serum1" else "Serum 2 presets"
        notice = Notice(
            kind="missing_synth_presets",
            generation=generation,
            title=f"{count:,} {subject} need {name}",
            body=(
                f"PatchLab found {count:,} {subject}"
                + (" that require " + name + " for processing" if generation == "serum1" else "")
                + f", but {name} isn't currently available. Your other presets are "
                "still processed normally, and matching keeps working. These can be "
                f"processed later if you install {name}."
            ),
            count=count,
            key=key,
        )
        record(
            SUBSYSTEM,
            "notice_shown",
            f"informing the user that {count} presets need {name}",
            operation_id=operation_id,
            decision_reason=(
                "an actionable state change worth one message; suppressed on "
                "subsequent launches unless the count changes materially"
            ),
            **notice.as_dict(),
        )
        return notice
    return None


def synth_available_notice(
    *,
    generation: str,
    pending_count: int,
    env: PlatformEnv | None = None,
    operation_id: str = "",
) -> Notice | None:
    """Offer to process pending presets now that their synth is available.

    An offer, never a requirement: declining must leave Match fully usable.
    """

    if pending_count <= 0:
        return None
    state = load_notice_state(env)
    key = _notice_key("synth_now_available", generation)
    if not _should_show(state, key, pending_count):
        return None
    name = display_name(generation)
    subject = "legacy .fxp presets" if generation == "serum1" else "Serum 2 presets"
    notice = Notice(
        kind="synth_now_available",
        generation=generation,
        title=f"{name} is now available",
        body=(
            f"{name} is now available. PatchLab has {pending_count:,} {subject} "
            "that haven't been processed yet. Process them now to include them in "
            "preset matching? You can keep matching sounds either way."
        ),
        count=pending_count,
        offers_processing=True,
        key=key,
    )
    record(
        SUBSYSTEM,
        "pending_processing_offered",
        f"offering to process {pending_count} pending {generation} preset(s)",
        operation_id=operation_id,
        decision_reason=(
            f"{name} became available and PatchLab already has these presets "
            "catalogued, so they can be processed without rediscovery"
        ),
        **notice.as_dict(),
    )
    return notice


def acknowledge(notice: Notice, *, env: PlatformEnv | None = None, choice: str = "") -> None:
    """Record that the user saw this notice, so it is not shown again."""

    state = dict(load_notice_state(env))
    state[notice.key or _notice_key(notice.kind, notice.generation)] = {
        "acknowledged": True,
        "count": notice.count,
        "choice": choice,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    save_notice_state(state, env)
    record(
        SUBSYSTEM,
        "notice_acknowledged",
        f"user acknowledged {notice.kind} for {notice.generation}"
        + (f" with choice {choice!r}" if choice else ""),
        decision_reason="anti-nag state recorded so the notice is not repeated",
        kind=notice.kind,
        generation=notice.generation,
        count=notice.count,
        choice=choice,
    )


def record_pending_exclusion(
    *,
    pending_counts: Mapping[str, int],
    learned: int,
    operation_id: str = "",
) -> None:
    """Record that pending presets are excluded from retrieval, and why.

    Match is never blocked by them; they simply do not participate until
    processed. Saying so explicitly in diagnostics stops a future reader from
    concluding Match was broken.
    """

    total_pending = sum(int(value) for value in pending_counts.values())
    if total_pending <= 0:
        return
    record(
        SUBSYSTEM,
        "pending_presets_excluded_from_retrieval",
        f"{total_pending:,} pending preset(s) are not in the retrieval index; "
        f"{learned:,} learned preset(s) are",
        operation_id=operation_id,
        decision_reason=(
            "library processing determines which presets can be retrieved; synth "
            "capability determines what PatchLab can create. Match proceeds on the "
            "learned subset and is never blocked by pending work."
        ),
        pending_by_reason=dict(pending_counts),
        pending_total=total_pending,
        learned=learned,
    )


__all__ = [
    "MATERIAL_COUNT_DELTA",
    "Notice",
    "NoticeKind",
    "OutputTargetDecision",
    "acknowledge",
    "evaluate_output_target",
    "load_notice_state",
    "missing_synth_notice",
    "record_pending_exclusion",
    "save_notice_state",
    "synth_available_notice",
]
