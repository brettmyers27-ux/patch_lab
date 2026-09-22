"""One way for a preview/export/render worker to report why it failed.

A beta tester's audition and export failures reached the log as bare
``KeyError: 'hosts'`` and ``StopIteration`` -- no code location, no indication
that an absent Serum install was the cause, and nothing a remote diagnosis could
use.  This records the full structured detail into the flight recorder (so a
support bundle carries it) and returns the one plain sentence the UI should show.

Never records preset bytes, audio, or credentials: only identities and locations.
"""

from __future__ import annotations

import traceback
from typing import Any


def _user_sentence(exc: BaseException, *, operation: str, synth: str) -> str:
    from core.renderer_selection import RendererUnavailableError

    explicit = getattr(exc, "user_message", "")
    if explicit:
        return str(explicit)
    if isinstance(exc, RendererUnavailableError):
        return exc.user_message
    if isinstance(exc, (FileNotFoundError, PermissionError, OSError)) and getattr(exc, "filename", None):
        return f"PatchLab couldn't use {exc.filename} while {operation}."
    label = {"serum1": "Serum 1", "serum2": "Serum 2"}.get(synth, "Serum")
    return (
        f"PatchLab couldn't finish {operation}. Check that {label} is installed "
        "and that PatchLab can open it."
    )


def report_worker_failure(
    exc: BaseException,
    *,
    subsystem: str,
    operation: str,
    synth: str = "",
    requested_renderer: str = "",
    selected_renderer: str = "",
    **fields: Any,
) -> dict[str, Any]:
    """Record a worker failure in full and return what the caller should print."""

    frames = traceback.extract_tb(exc.__traceback__)
    location = ""
    if frames:
        last = frames[-1]
        location = f"{last.filename}:{last.lineno} in {last.name}"
    detail: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "code_location": location,
        "operation": operation,
        "serum_generation": synth,
        "requested_renderer": requested_renderer,
        "selected_renderer": selected_renderer,
        "traceback": traceback.format_exc(),
        **fields,
    }
    selection = getattr(exc, "selection", None)
    if selection is not None and hasattr(selection, "as_dict"):
        try:
            detail["renderer_selection"] = selection.as_dict()
            detail.setdefault("selected_renderer", str(getattr(selection, "renderer", "")))
        except Exception:
            pass
    worker_detail = getattr(exc, "detail", None)
    if isinstance(worker_detail, dict):
        detail["worker_init_failure"] = {
            key: worker_detail.get(key)
            for key in ("message", "exception_type", "worker_id", "pid",
                        "required_synths", "requested_synth", "fingerprint")
            if key in worker_detail
        }
    try:
        from core.diagnostics import record_failure, worker_identity

        # record_failure owns phase/worker_id/serum_generation/message as named
        # parameters; passing them again inside **fields would be a duplicate.
        reserved = {"message", "operation_id", "phase", "worker_id", "renderer",
                    "serum_generation", "traceback", "exception_type"}
        marker = record_failure(
            subsystem,
            f"{subsystem.replace('-', '_')}_failed",
            exc,
            message=f"{operation} failed: {type(exc).__name__}: {exc}",
            phase=operation,
            worker_id=worker_identity(subsystem),
            renderer=selected_renderer or requested_renderer,
            serum_generation=synth,
            **{k: v for k, v in detail.items() if k not in reserved},
        )
        detail["failure_fingerprint"] = marker.digest
        detail["worker_id"] = worker_identity(subsystem)
    except Exception:
        pass
    detail["user_message"] = _user_sentence(exc, operation=operation, synth=synth)
    return detail
