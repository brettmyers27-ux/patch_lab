"""The save lifecycle for a generated preset: build once, commit safely, retry sanely.

A completed Match is expensive; saving its preset should almost never lose it.
This module owns everything between "the matcher produced a recommendation" and
"a verified file sits at a path the Library records":

* one **save incident** per generated result, persisted under the app data
  folder, which records every attempt (automatic or manual) with a failure
  classification and never any preset or audio bytes;
* a **staged artifact**: the verified preset is kept privately inside the
  incident until it is committed, so a filesystem failure retries *that same
  preset* instead of rebuilding it or re-running the Match;
* an **atomic, no-clobber commit** that never overwrites another file and never
  leaves a partial preset under the final name;
* a **bounded retry loop** whose behaviour depends on what kind of failure
  happened.

The UI decides what to show; this module decides what is true.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable


INCIDENTS_DIRNAME = "preset-save-incidents"
INCIDENT_FILENAME = "incident.json"
#: Report state lives in its own file: the UI updates it while a save worker
#: may be writing ``incident.json``, so the two writers never share a file.
REPORTS_FILENAME = "reports.json"
MAX_AUTOMATIC_ATTEMPTS = 3
#: Waits before automatic attempts 2 and 3.  Short, bounded, never a hot loop.
AUTOMATIC_BACKOFF_S: tuple[float, ...] = (0.5, 2.0)
_MESSAGE_LIMIT = 2_000
_TRACEBACK_LIMIT = 12_000
SAVED_INCIDENT_RETENTION_S = 14 * 24 * 60 * 60


class SaveFailureKind(str, Enum):
    TRANSIENT_IO = "transient_io"
    DESTINATION = "destination"
    CONSTRUCTION = "construction"
    VALIDATION = "validation"
    IDENTITY = "identity"


class IncidentStatus(str, Enum):
    SAVING = "saving"
    #: The file is verified at its final path; the Library has not recorded it yet.
    COMMITTED = "committed"
    SAVED = "saved"
    FAILED = "failed"
    #: This result cannot produce a valid preset; the user should run it again.
    UNRECOVERABLE = "unrecoverable"


class PresetValidationError(RuntimeError):
    """A preset was written but is not a file PatchLab may call saved."""

    user_message = (
        "PatchLab made the preset, but it didn't pass its final checks, so it was not saved."
    )

    def __init__(self, message: str, *, reason: str = "invalid_preset") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class SaveFailure:
    kind: SaveFailureKind
    reason: str

    @property
    def retry_automatically(self) -> bool:
        # A base that cannot be identified will not become identifiable by
        # asking again; every other kind may clear on its own.
        return self.kind is not SaveFailureKind.IDENTITY

    @property
    def discards_staged_artifact(self) -> bool:
        return self.kind in (
            SaveFailureKind.CONSTRUCTION,
            SaveFailureKind.VALIDATION,
            SaveFailureKind.IDENTITY,
        )

    @property
    def rebuild_can_help(self) -> bool:
        return self.kind in (SaveFailureKind.CONSTRUCTION, SaveFailureKind.VALIDATION)


_DISK_FULL = {errno.ENOSPC, errno.EDQUOT}
_PERMISSION = {errno.EACCES, errno.EPERM}
_READ_ONLY = {errno.EROFS}
_MISSING = {errno.ENOENT, errno.ENOTDIR}
_TRANSIENT = {
    errno.EBUSY, errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT, errno.EIO,
    errno.ESTALE, errno.ENOLCK, errno.EDEADLK, errno.EMFILE, errno.ENFILE,
}


def _os_error_in(exc: BaseException) -> OSError | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno is not None:
            return current
        current = current.__cause__ or current.__context__
    return None


def classify_save_failure(exc: BaseException, *, stage: str) -> SaveFailure:
    """Name what went wrong, because each kind recovers differently.

    ``stage`` is where it happened: ``construct`` (resolving the base and
    serialising), ``validate`` (decode + headless reload of the staged file) or
    ``commit`` (publishing to the destination and re-checking it there).
    """

    from core.base_preset_identity import BaseIdentityError

    if isinstance(exc, BaseIdentityError):
        return SaveFailure(SaveFailureKind.IDENTITY, "base_identity")
    if isinstance(exc, PresetValidationError):
        return SaveFailure(SaveFailureKind.VALIDATION, exc.reason)
    os_error = _os_error_in(exc)
    if os_error is not None:
        number = os_error.errno
        # Space and permission problems are the destination's fault wherever
        # they surface; the staging area can fill up too.
        if number in _DISK_FULL:
            return SaveFailure(SaveFailureKind.DESTINATION, "disk_full")
        if number in _READ_ONLY:
            return SaveFailure(SaveFailureKind.DESTINATION, "read_only")
        if stage == "commit":
            if number in _PERMISSION:
                return SaveFailure(SaveFailureKind.DESTINATION, "permission")
            if number in _MISSING:
                return SaveFailure(SaveFailureKind.DESTINATION, "folder_missing")
            return SaveFailure(SaveFailureKind.TRANSIENT_IO, "filesystem")
        if number in _TRANSIENT:
            return SaveFailure(SaveFailureKind.TRANSIENT_IO, "filesystem")
        if number in _PERMISSION:
            return SaveFailure(SaveFailureKind.DESTINATION, "permission")
    try:
        from core.renderer_selection import RendererUnavailableError

        if isinstance(exc, RendererUnavailableError):
            return SaveFailure(SaveFailureKind.VALIDATION, "renderer_unavailable")
    except Exception:  # pragma: no cover - import guard only
        pass
    if stage == "validate":
        return SaveFailure(SaveFailureKind.VALIDATION, "reload_failed")
    if stage == "commit":
        return SaveFailure(SaveFailureKind.VALIDATION, "committed_file_invalid")
    return SaveFailure(SaveFailureKind.CONSTRUCTION, "serialization")


# --------------------------------------------------------------------------
# Plain-language messages.  No jargon, no traces; the incident holds those.
# --------------------------------------------------------------------------

AUTOMATIC_FAILURE_MESSAGE = (
    "PatchLab created your preset, but couldn't save the file. Your result is "
    "still here. Try saving it again."
)
UNRECOVERABLE_MESSAGE = (
    "PatchLab couldn't create a valid preset file from this result. Run this "
    "sound again to create a new preset."
)
_MANUAL_MESSAGES = {
    "disk_full": "PatchLab needs more free disk space before it can save this preset.",
    "permission": "PatchLab still can't save this preset because this folder isn't writable.",
    "read_only": "PatchLab still can't save this preset because this folder isn't writable.",
    "folder_missing": (
        "PatchLab still can't save this preset because its folder is missing. "
        "Reconnect the drive or folder, then try again."
    ),
}


_AUTOMATIC_DESTINATION_MESSAGES = {
    "disk_full": (
        "PatchLab created your preset, but this disk is full, so it couldn't save "
        "the file. Your result is still here. Free up some space, then try saving it again."
    ),
    "permission": (
        "PatchLab created your preset, but this folder isn't writable, so it couldn't "
        "save the file. Your result is still here. Try saving it again."
    ),
    "read_only": (
        "PatchLab created your preset, but this folder isn't writable, so it couldn't "
        "save the file. Your result is still here. Try saving it again."
    ),
    "folder_missing": (
        "PatchLab created your preset, but its folder is missing, so it couldn't "
        "save the file. Your result is still here. Try saving it again."
    ),
}


def automatic_failure_message(failure: SaveFailure) -> str:
    if failure.kind is SaveFailureKind.DESTINATION:
        return _AUTOMATIC_DESTINATION_MESSAGES.get(failure.reason, AUTOMATIC_FAILURE_MESSAGE)
    return AUTOMATIC_FAILURE_MESSAGE


def manual_failure_message(failure: SaveFailure, *, unrecoverable: bool) -> str:
    if unrecoverable:
        return UNRECOVERABLE_MESSAGE
    if failure.kind is SaveFailureKind.DESTINATION:
        return _MANUAL_MESSAGES.get(
            failure.reason,
            "PatchLab still can't save this preset to this folder.",
        ) + " Your result is still here."
    return (
        "PatchLab still couldn't save this preset. Your result is still here; "
        "try again in a moment."
    )


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_preset_file(
    path: Path, *, synth: str, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Confirm a file on disk is a complete preset of the right generation.

    Raises ``PresetValidationError``; returns what was checked on success.
    """

    from core.fxp import parse_fxp
    from core.serum2_preset import parse_serum2_preset

    path = Path(path)
    if not path.is_file():
        raise PresetValidationError(f"{path} does not exist", reason="missing_file")
    size = path.stat().st_size
    if size <= 0:
        raise PresetValidationError(f"{path} is empty", reason="empty_file")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise PresetValidationError(
            f"{path} does not contain the preset that was verified", reason="content_changed"
        )
    try:
        if synth == "serum2":
            parsed = parse_serum2_preset(path)
            if not isinstance(parsed.data, dict) or not parsed.data:
                raise PresetValidationError(
                    f"{path} has no Serum 2 settings graph", reason="empty_graph"
                )
            detail: dict[str, Any] = {"payload_version": parsed.payload_version}
        elif synth == "serum1":
            parsed_fxp = parse_fxp(path)
            if not parsed_fxp.payload:
                raise PresetValidationError(f"{path} has no Serum 1 state", reason="empty_graph")
            detail = {"plugin_id": parsed_fxp.plugin_id.decode("latin-1", "replace")}
        else:
            raise PresetValidationError(f"unknown synth {synth!r}", reason="unknown_generation")
    except PresetValidationError:
        raise
    except Exception as exc:
        raise PresetValidationError(
            f"{path} is not a valid {synth} preset: {type(exc).__name__}: {exc}",
            reason="wrong_format",
        ) from exc
    return {"exists": True, "size": size, "sha256": digest, "synth": synth, **detail}


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _next_free(destination: Path) -> Path:
    from core.match_batch import disambiguated_preset_path

    return disambiguated_preset_path(destination.parent, destination.stem, destination.suffix)


def commit_verified_preset(
    staged: Path, destination: Path, *, expected_sha256: str
) -> Path:
    """Publish ``staged`` at ``destination`` atomically, never clobbering a file.

    * The bytes are first copied to a hidden sibling (same volume), flushed,
      then linked into place.  A crash leaves at most a hidden ``.tmp`` file,
      never a truncated preset under the final name.
    * If ``destination`` already holds exactly these bytes (an earlier attempt
      got that far), it is returned as-is: committing is idempotent.
    * If it holds anything else, a free "Name 2" is used instead.  Another
      preset is never overwritten.

    Returns the final path, which may differ from ``destination``.
    """

    staged = Path(staged)
    destination = Path(destination).expanduser()
    if sha256_file(staged) != expected_sha256:
        raise PresetValidationError(
            "the staged preset changed after it was verified", reason="staged_changed"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.parent.resolve() / destination.name
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return destination
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
    )
    sibling = Path(handle.name)
    try:
        with handle:
            with staged.open("rb") as source:
                shutil.copyfileobj(source, handle)
            handle.flush()
            os.fsync(handle.fileno())
        for _ in range(100):
            if destination.exists():
                destination = _next_free(destination)
            try:
                # link() is atomic and refuses to replace an existing name.
                os.link(sibling, destination)
                break
            except FileExistsError:
                continue
            except OSError as exc:
                if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.EXDEV, errno.EMLINK):
                    raise
                # A volume without hard links (some network/exFAT shares):
                # re-check immediately before the atomic rename.
                if destination.exists():
                    continue
                os.replace(sibling, destination)
                break
        else:  # pragma: no cover - 100 consecutive collisions
            raise FileExistsError(errno.EEXIST, "no free preset name", str(destination))
    finally:
        sibling.unlink(missing_ok=True)
    _fsync_directory(destination.parent)
    return destination


# --------------------------------------------------------------------------
# Incidents
# --------------------------------------------------------------------------


def incidents_root(app_data_dir: Path | None = None) -> Path:
    if app_data_dir is None:
        from core.platform_env import ENV

        app_data_dir = ENV.app_data_dir
    return Path(app_data_dir) / INCIDENTS_DIRNAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim(value: str, limit: int) -> str:
    from core.diagnostics import redact_text

    text = redact_text(str(value))
    return text if len(text) <= limit else text[: limit - 16] + " …[truncated]"


@dataclass
class SaveIncident:
    """Everything known about saving one generated result, persisted as JSON."""

    incident_id: str
    directory: Path
    match_uid: str = ""
    result_path: str = ""
    target_path: str = ""
    synth: str = ""
    status: str = IncidentStatus.SAVING.value
    base_identity: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    staged: dict[str, Any] = field(default_factory=dict)
    final_path: str = ""
    last_failure: dict[str, Any] = field(default_factory=dict)
    #: Everything needed to run the same sound again with the same settings.
    run_again: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    followup: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)

    _FIELDS = (
        "incident_id", "match_uid", "result_path", "target_path", "synth", "status",
        "base_identity", "attempts", "staged", "final_path", "last_failure",
        "run_again", "created_utc", "updated_utc",
    )
    _REPORT_FIELDS = ("report", "followup")

    # ---- persistence -----------------------------------------------------

    @property
    def path(self) -> Path:
        return self.directory / INCIDENT_FILENAME

    @classmethod
    def create(
        cls,
        *,
        match_uid: str,
        result_path: Path,
        target_path: Path,
        synth: str,
        run_again: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> "SaveIncident":
        incident_id = uuid.uuid4().hex
        incident = cls(
            incident_id=incident_id,
            directory=incidents_root() / incident_id if root is None else Path(root) / incident_id,
            match_uid=str(match_uid),
            result_path=str(result_path),
            target_path=str(target_path),
            synth=str(synth),
            run_again=dict(run_again or {}),
        )
        incident.save()
        return incident

    @classmethod
    def load(cls, incident_id: str, *, root: Path | None = None) -> "SaveIncident":
        directory = (incidents_root() if root is None else Path(root)) / incident_id
        raw = json.loads((directory / INCIDENT_FILENAME).read_text(encoding="utf-8"))
        values = {name: raw[name] for name in cls._FIELDS if name in raw}
        reports_path = directory / REPORTS_FILENAME
        if reports_path.is_file():
            try:
                reports = json.loads(reports_path.read_text(encoding="utf-8"))
                values.update({name: reports[name] for name in cls._REPORT_FIELDS if name in reports})
            except (OSError, ValueError):
                pass
        values["incident_id"] = incident_id
        return cls(directory=directory, **values)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        temporary.replace(path)

    def save(self) -> None:
        """Persist the save state (attempts, status, staged artifact)."""

        self.updated_utc = _utc_now()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write_json(self.path, {name: getattr(self, name) for name in self._FIELDS})

    def save_reports(self) -> None:
        """Persist only the automatic-report state (see ``REPORTS_FILENAME``)."""

        self.directory.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.directory / REPORTS_FILENAME,
            {name: getattr(self, name) for name in self._REPORT_FIELDS},
        )

    # ---- staged artifact -------------------------------------------------

    def staging_path(self, extension: str) -> Path:
        return self.directory / f"staged{extension}"

    def valid_staged_artifact(self) -> Path | None:
        """The verified preset kept from an earlier attempt, if it is intact."""

        if not self.staged.get("verified"):
            return None
        path = Path(str(self.staged.get("path", "")))
        try:
            if path.is_file() and sha256_file(path) == self.staged.get("sha256"):
                return path
        except OSError:
            return None
        return None

    def discard_staged_artifact(self) -> None:
        path = self.staged.get("path")
        if path:
            Path(str(path)).unlink(missing_ok=True)
        self.staged = {}

    # ---- attempts --------------------------------------------------------

    def record_attempt(
        self,
        *,
        trigger: str,
        stage: str,
        target_path: Path | str,
        ok: bool,
        started: float,
        exc: BaseException | None = None,
        failure: SaveFailure | None = None,
        validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attempt = {
            "incident_id": self.incident_id,
            "attempt": len(self.attempts) + 1,
            "trigger": trigger,
            "operation": "saving the generated preset",
            "stage": stage,
            "target_path": str(target_path),
            "synth": self.synth,
            "base_identity": dict(self.base_identity),
            "ok": bool(ok),
            "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "elapsed_s": round(time.time() - started, 3),
            "validation": dict(validation or {}),
        }
        if exc is not None:
            attempt.update(
                {
                    "exception_type": type(exc).__name__,
                    "message": _trim(str(exc), _MESSAGE_LIMIT),
                    "traceback": _trim(
                        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                        _TRACEBACK_LIMIT,
                    ),
                }
            )
        if failure is not None:
            attempt["failure_kind"] = failure.kind.value
            attempt["failure_reason"] = failure.reason
            self.last_failure = {"kind": failure.kind.value, "reason": failure.reason}
        self.attempts.append(attempt)
        _record_diagnostic(attempt, exc)
        return attempt

    def last_save_failure(self) -> SaveFailure | None:
        if not self.last_failure:
            return None
        return SaveFailure(SaveFailureKind(self.last_failure["kind"]), str(self.last_failure["reason"]))

    def diagnostic_summary(self) -> dict[str, Any]:
        """Everything support needs, and nothing the user did not agree to share."""

        from core.diagnostics import sanitize

        return sanitize(
            {
                "incident_id": self.incident_id,
                "match_uid": self.match_uid,
                "status": self.status,
                "synth": self.synth,
                "target_path": self.target_path,
                "target_folder_exists": Path(self.target_path).parent.is_dir() if self.target_path else False,
                "final_path": self.final_path,
                "base_identity": self.base_identity,
                "last_failure": self.last_failure,
                "staged_artifact": {
                    key: self.staged.get(key)
                    for key in ("sha256", "size", "verified", "mode", "validation")
                    if key in self.staged
                },
                "attempts": self.attempts,
                "report": self.report,
                "created_utc": self.created_utc,
            }
        )


def _record_diagnostic(attempt: dict[str, Any], exc: BaseException | None) -> None:
    try:
        from core.diagnostics import record_failure, recorder

        fields = {
            key: value
            for key, value in attempt.items()
            if key not in {
                "traceback", "message", "incident_id", "attempt", "stage", "synth",
                "exception_type",
            }
        }
        if exc is not None and not attempt["ok"]:
            record_failure(
                "preset-save",
                "preset_save_attempt_failed",
                exc,
                message=(
                    f"save attempt {attempt['attempt']} ({attempt['trigger']}) failed at "
                    f"{attempt['stage']}: {attempt.get('failure_kind')}/{attempt.get('failure_reason')}"
                ),
                phase=attempt["stage"],
                serum_generation=attempt["synth"],
                save_incident_id=attempt["incident_id"],
                save_attempt=attempt["attempt"],
                **fields,
            )
        else:
            recorder().record(
                "preset-save",
                "preset_save_attempt_succeeded" if attempt["ok"] else "preset_save_attempt_failed",
                f"save attempt {attempt['attempt']} ({attempt['trigger']}) "
                + ("succeeded" if attempt["ok"] else "failed"),
                phase=attempt["stage"],
                save_incident_id=attempt["incident_id"],
                save_attempt=attempt["attempt"],
                serum_generation=attempt["synth"],
                **fields,
            )
    except Exception:
        pass


def incident_for_match(match_uid: str, *, root: Path | None = None) -> SaveIncident | None:
    """The newest incident recorded for one Library entry, if any."""

    base = incidents_root() if root is None else Path(root)
    newest: SaveIncident | None = None
    if not base.is_dir():
        return None
    for candidate in base.glob(f"*/{INCIDENT_FILENAME}"):
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(raw.get("match_uid")) != str(match_uid):
            continue
        if newest is None or str(raw.get("created_utc", "")) > newest.created_utc:
            newest = SaveIncident.load(candidate.parent.name, root=base)
    return newest


def unfinished_incidents(*, root: Path | None = None) -> list[SaveIncident]:
    base = incidents_root() if root is None else Path(root)
    found: list[SaveIncident] = []
    if not base.is_dir():
        return found
    for candidate in base.glob(f"*/{INCIDENT_FILENAME}"):
        try:
            incident = SaveIncident.load(candidate.parent.name, root=base)
        except (OSError, ValueError, TypeError):
            continue
        if incident.status != IncidentStatus.SAVED.value:
            found.append(incident)
    return found


def prune_saved_incidents(*, root: Path | None = None, now: float | None = None) -> int:
    """Remove long-finished incidents; failed ones are kept for the user."""

    base = incidents_root() if root is None else Path(root)
    now = time.time() if now is None else now
    removed = 0
    if not base.is_dir():
        return removed
    for candidate in base.glob(f"*/{INCIDENT_FILENAME}"):
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            if raw.get("status") != IncidentStatus.SAVED.value:
                continue
            if now - candidate.stat().st_mtime < SAVED_INCIDENT_RETENTION_S:
                continue
            shutil.rmtree(candidate.parent, ignore_errors=True)
            removed += 1
        except (OSError, ValueError):
            continue
    return removed


# --------------------------------------------------------------------------
# The lifecycle
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StagedPreset:
    """A preset built and fully verified in private storage."""

    path: Path
    sha256: str
    mode: str
    base_identity: dict[str, Any]
    validation: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)


class StageTracker:
    """Lets the build callable say how far it got, for classification."""

    def __init__(self) -> None:
        self.stage = "construct"


BuildFn = Callable[[StageTracker, Path], StagedPreset]


@dataclass(frozen=True, slots=True)
class SaveOutcome:
    saved: bool
    incident: SaveIncident
    final_path: Path | None = None
    failure: SaveFailure | None = None
    unrecoverable: bool = False
    staged: StagedPreset | None = None

    @property
    def attempts(self) -> int:
        return len(self.incident.attempts)


def _attempt(
    incident: SaveIncident,
    *,
    build: BuildFn,
    destination: Path,
    trigger: str,
    extension: str,
    reuse_staged: bool,
) -> tuple[Path | None, SaveFailure | None, StagedPreset | None]:
    started = time.time()
    tracker = StageTracker()
    staged_path = incident.valid_staged_artifact() if reuse_staged else None
    staged: StagedPreset | None = None
    try:
        if staged_path is None:
            incident.discard_staged_artifact()
            staged = build(tracker, incident.staging_path(extension))
            incident.base_identity = dict(staged.base_identity or incident.base_identity)
            incident.staged = {
                "path": str(staged.path),
                "sha256": staged.sha256,
                "size": staged.path.stat().st_size,
                "verified": True,
                "mode": staged.mode,
                "validation": staged.validation,
                "payload": staged.payload,
            }
            incident.save()
            staged_path = staged.path
        else:
            tracker.stage = "commit"
            staged = StagedPreset(
                path=staged_path,
                sha256=str(incident.staged["sha256"]),
                mode=str(incident.staged.get("mode", "")),
                base_identity=dict(incident.base_identity),
                validation=dict(incident.staged.get("validation") or {}),
                payload=dict(incident.staged.get("payload") or {}),
            )
        tracker.stage = "commit"
        final = commit_verified_preset(staged_path, destination, expected_sha256=staged.sha256)
        try:
            checked = validate_preset_file(final, synth=incident.synth, expected_sha256=staged.sha256)
        except Exception:
            # A file that fails its final check is not a preset we may leave
            # under a real name.  It is ours (verified bytes, fresh name) or an
            # identical earlier commit, so removing it loses nothing.
            final.unlink(missing_ok=True)
            raise
        incident.record_attempt(
            trigger=trigger, stage="commit", target_path=final, ok=True,
            started=started, validation={**staged.validation, "final": checked},
        )
        return final, None, staged
    except Exception as exc:
        failure = classify_save_failure(exc, stage=tracker.stage)
        incident.record_attempt(
            trigger=trigger, stage=tracker.stage, target_path=destination, ok=False,
            started=started, exc=exc, failure=failure,
        )
        if failure.discards_staged_artifact:
            incident.discard_staged_artifact()
        incident.save()
        return None, failure, None


def run_save_lifecycle(
    incident: SaveIncident,
    *,
    build: BuildFn,
    destination: Path,
    extension: str,
    trigger: str = "auto",
    max_attempts: int = MAX_AUTOMATIC_ATTEMPTS,
    backoff: tuple[float, ...] = AUTOMATIC_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
) -> SaveOutcome:
    """Save one generated result.

    ``trigger="auto"``: up to ``max_attempts`` attempts with short backoff;
    an identity failure stops immediately (asking again cannot change it).

    ``trigger="manual"``: one attempt that reuses the verified staged preset if
    there is one.  If the preset itself is the problem (construction or
    validation) one bounded rebuild from the archived result follows; if that
    fails too, the result is unrecoverable and the user is offered Run Again.
    A destination or transient failure is never "fixed" by rebuilding.
    """

    incident.status = IncidentStatus.SAVING.value
    incident.save()
    failure: SaveFailure | None = None
    if trigger == "manual":
        plan = [("manual", True)]
    else:
        plan = [(trigger, True)] * max(1, int(max_attempts))
    index = 0
    while index < len(plan):
        attempt_trigger, reuse = plan[index]
        final, failure, staged = _attempt(
            incident, build=build, destination=destination, trigger=attempt_trigger,
            extension=extension, reuse_staged=reuse,
        )
        if final is not None:
            incident.status = IncidentStatus.COMMITTED.value
            incident.final_path = str(final)
            incident.last_failure = {}
            incident.save()
            return SaveOutcome(True, incident, final_path=final, staged=staged)
        assert failure is not None
        if trigger == "manual":
            if attempt_trigger == "manual" and failure.rebuild_can_help:
                plan.append(("rebuild", False))
        elif not failure.retry_automatically:
            break
        index += 1
        if trigger != "manual" and index < len(plan):
            sleep(float(backoff[min(index - 1, len(backoff) - 1)]) if backoff else 0.0)

    last_trigger = incident.attempts[-1].get("trigger") if incident.attempts else ""
    unrecoverable = trigger == "manual" and failure is not None and (
        failure.kind is SaveFailureKind.IDENTITY
        or (last_trigger == "rebuild" and failure.rebuild_can_help)
    )
    incident.status = (
        IncidentStatus.UNRECOVERABLE.value if unrecoverable else IncidentStatus.FAILED.value
    )
    incident.save()
    return SaveOutcome(False, incident, failure=failure, unrecoverable=unrecoverable)


# --------------------------------------------------------------------------
# Filesystem <-> Library agreement
# --------------------------------------------------------------------------


def finalize_saved_incident(incident: SaveIncident, database: Any, final_path: Path) -> bool:
    """Record a committed preset in the Library, only if the file is really there.

    Returns True when the Library now points at a verified file.  A database
    failure leaves the incident ``committed`` (file saved, Library pending) so
    reconciliation can finish the job later without writing a second preset.
    """

    final_path = Path(final_path)
    expected = str(incident.staged.get("sha256") or "") or None
    validate_preset_file(final_path, synth=incident.synth, expected_sha256=expected)
    incident.final_path = str(final_path)
    incident.status = IncidentStatus.COMMITTED.value
    incident.save()
    recorded = False
    if incident.match_uid:
        try:
            recorded = bool(database.set_match_exported_path(incident.match_uid, final_path))
        except Exception as exc:
            _record_diagnostic(
                {
                    "incident_id": incident.incident_id, "attempt": len(incident.attempts),
                    "trigger": "library", "stage": "record", "synth": incident.synth,
                    "ok": False, "target_path": str(final_path),
                },
                exc,
            )
            return False
    else:
        recorded = True
    if recorded:
        incident.status = IncidentStatus.SAVED.value
        incident.last_failure = {}
        incident.save()
    return recorded


def reconcile_incidents(database: Any, *, root: Path | None = None) -> dict[str, int]:
    """Bring interrupted saves back into agreement at startup.

    * ``committed``: the verified file exists, the Library write did not land
      -> record it (never regenerate).
    * ``saving``: the app stopped mid-save -> ``failed``, so the result offers
      Retry Saving Preset instead of silently vanishing.
    """

    counts = {"recorded": 0, "marked_failed": 0}
    for incident in unfinished_incidents(root=root):
        if incident.status == IncidentStatus.COMMITTED.value and incident.final_path:
            try:
                if finalize_saved_incident(incident, database, Path(incident.final_path)):
                    counts["recorded"] += 1
                    continue
            except PresetValidationError:
                pass
            incident.status = IncidentStatus.FAILED.value
            incident.last_failure = incident.last_failure or {
                "kind": SaveFailureKind.VALIDATION.value, "reason": "committed_file_invalid",
            }
            incident.final_path = ""
            incident.save()
            counts["marked_failed"] += 1
        elif incident.status == IncidentStatus.SAVING.value:
            incident.status = IncidentStatus.FAILED.value
            incident.last_failure = incident.last_failure or {
                "kind": SaveFailureKind.TRANSIENT_IO.value, "reason": "interrupted",
            }
            incident.save()
            counts["marked_failed"] += 1
    return counts


# --------------------------------------------------------------------------
# What the UI needs
# --------------------------------------------------------------------------


def incident_user_message(incident: SaveIncident) -> str:
    """The one plain sentence for an incident's current state."""

    if incident.status == IncidentStatus.UNRECOVERABLE.value:
        return UNRECOVERABLE_MESSAGE
    failure = incident.last_save_failure() or SaveFailure(SaveFailureKind.TRANSIENT_IO, "unknown")
    manual = any(attempt.get("trigger") in ("manual", "rebuild") for attempt in incident.attempts)
    if manual:
        return manual_failure_message(failure, unrecoverable=False)
    return automatic_failure_message(failure)


def incidents_by_match(*, root: Path | None = None) -> dict[str, SaveIncident]:
    """The newest incident per Library entry, read in one pass."""

    base = incidents_root() if root is None else Path(root)
    found: dict[str, SaveIncident] = {}
    if not base.is_dir():
        return found
    for candidate in base.glob(f"*/{INCIDENT_FILENAME}"):
        try:
            incident = SaveIncident.load(candidate.parent.name, root=base)
        except (OSError, ValueError, TypeError):
            continue
        if not incident.match_uid:
            continue
        known = found.get(incident.match_uid)
        if known is None or incident.created_utc > known.created_utc:
            found[incident.match_uid] = incident
    return found


def record_worker_crash(incident: SaveIncident, message: str, *, trigger: str) -> None:
    """The save worker ended without reporting an outcome; say so, as a failure."""

    if incident.status not in (IncidentStatus.SAVING.value,):
        return
    failure = SaveFailure(SaveFailureKind.TRANSIENT_IO, "worker_exited")
    incident.record_attempt(
        trigger=trigger, stage="worker", target_path=incident.target_path, ok=False,
        started=time.time(), exc=RuntimeError(message), failure=failure,
    )
    incident.status = IncidentStatus.FAILED.value
    incident.save()
