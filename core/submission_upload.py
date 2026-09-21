"""Upload one already-created bundle to the private support service.

The whole contract is: stream one ``.zip`` to ``POST /submissions``, get a
receipt back.  This module adds only what a desktop app needs around that: a
short bounded retry for genuinely transient failures, stable error codes,
verification of the receipt, and flight-recorder events (never credentials).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import time
import urllib.error
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.privacy import user_presets_enabled
from core.relay_client import MultipartFileBody, RelayClient

BUG_REPORT = "bug_report"
PRESET_CONTRIBUTION = "preset_contribution"

MAX_ATTEMPTS = 4
BASE_DELAY_S = 1.0
MAX_RETRY_AFTER_S = 30.0
#: Statuses that mean "try again shortly"; everything else is final.
TRANSIENT_STATUSES = frozenset({429, 502, 503, 504})

#: Server error codes -> is it worth trying again unchanged?
NON_RETRYABLE_CODES = frozenset(
    {
        "invalid_type", "invalid_id", "invalid_sha256", "invalid_version",
        "unsupported_file", "file_too_large", "submission_conflict",
        "auth_required", "auth_invalid",
    }
)

USER_MESSAGES = {
    "not_connected": "PatchLab isn't signed in to the support service on this Mac.",
    "auth_failed": "PatchLab couldn't sign in to the support service.",
    "offline": "PatchLab couldn't reach the internet.",
    "timeout": "The upload took too long.",
    "service_unavailable": "The support service is temporarily unavailable.",
    "sha256_mismatch": "The upload was damaged in transit.",
    "file_too_large": "The file is too large to upload.",
    "submission_conflict": "This was already uploaded earlier.",
    "rejected": "The support service didn't accept the file.",
    "receipt_mismatch": "The support service's receipt didn't match the file.",
    "file_missing": "The saved file could not be found.",
    "consent_off": "Personal presets are turned off, so nothing was uploaded.",
    "unknown": "The upload didn't complete.",
}


class UploadError(Exception):
    """A failed upload with a stable ``code`` (never a credential or a stack trace)."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool = False,
        http_status: int | None = None,
        attempts: int = 0,
        detail: str = "",
    ) -> None:
        super().__init__(detail or code)
        self.code = code
        self.retryable = retryable
        self.http_status = http_status
        self.attempts = attempts
        self.detail = detail

    @property
    def user_message(self) -> str:
        return USER_MESSAGES.get(self.code, USER_MESSAGES["unknown"])


@dataclass(frozen=True, slots=True)
class UploadResult:
    submission_id: str
    receipt_id: str
    duplicate: bool
    size: int
    sha256: str
    attempts: int
    elapsed_s: float
    stored_md5: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_relay() -> tuple[RelayClient | None, str]:
    """Return ``(client, "ok")`` or ``(None, reason)`` with the accurate reason."""

    if os.environ.get("PATCHLAB_DISABLE_RELAY", "").strip() == "1":
        return None, "local_only"
    url = os.environ.get("PATCHLAB_RELAY_URL", "").strip()
    if not url:
        return None, "no_url"
    password: str | Callable[[], str | None] = os.environ.get("PATCHLAB_RELAY_PASSWORD", "")
    token = None
    if not password:
        from core import access_gate

        token = access_gate.stored_token()
        if token:
            # The saved token is tried first; the keychain is consulted (with a
            # time limit) only if the service says the token has expired.
            password = access_gate.stored_passcode
        else:
            password = access_gate.stored_passcode() or ""
    if not password and not token:
        return None, "no_credentials"
    return RelayClient(url, password, token=token), "ok"


def relay_host(relay: RelayClient) -> str:
    return urllib.parse.urlsplit(str(getattr(relay, "base_url", ""))).hostname or ""


def _error_body(exc: urllib.error.HTTPError) -> dict[str, Any]:
    try:
        raw = exc.read(64 * 1024)
        value = json.loads(raw.decode("utf-8", errors="replace"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def classify(exc: BaseException) -> tuple[str, bool, int | None, str]:
    """Map a transport exception to ``(code, retryable, http_status, category)``."""

    if isinstance(exc, urllib.error.HTTPError):
        body = _error_body(exc)
        server_code = str(body.get("error_code") or "")
        status = int(exc.code)
        if status in TRANSIENT_STATUSES:
            return "service_unavailable", True, status, "http_transient"
        if status == 401:
            return "auth_failed", False, status, "http_auth"
        if server_code == "storage_unavailable":
            return "service_unavailable", True, status, "http_transient"
        if server_code == "sha256_mismatch":
            return "sha256_mismatch", True, status, "http_integrity"
        if server_code in {"file_too_large", "submission_conflict"} or status == 413:
            return ("file_too_large" if status == 413 else server_code), False, status, "http_rejected"
        return "rejected", False, status, "http_rejected"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout", True, None, "timeout"
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return "timeout", True, None, "timeout"
        return "offline", True, None, "network"
    if isinstance(exc, (ConnectionError, OSError)):
        return "offline", True, None, "network"
    return "unknown", False, None, "other"


def _retry_after(exc: BaseException) -> float | None:
    if isinstance(exc, urllib.error.HTTPError):
        value = exc.headers.get("Retry-After") if exc.headers else None
        try:
            return min(MAX_RETRY_AFTER_S, max(0.0, float(value)))
        except (TypeError, ValueError):
            return None
    return None


def _timeout_for(size: int) -> float:
    return float(min(300.0, 30.0 + size / (128 * 1024)))


def _record(**kwargs: Any) -> None:
    try:
        from core.diagnostics import recorder

        event_type = kwargs.pop("event_type")
        message = kwargs.pop("message")
        recorder().record("relay-upload", event_type, message, **kwargs)
    except Exception:
        pass


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def upload_file(
    relay: RelayClient,
    *,
    kind: str,
    submission_id: str,
    path: Path,
    version: str,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] | None = None,
    jitter: Callable[[], float] = random.random,
    progress: Callable[[str], None] | None = None,
) -> UploadResult:
    """Upload ``path`` once, retrying only transient failures, then give up.

    The file is uploaded as it is: never regenerated or re-compressed here, so a
    retry sends the identical bytes with the identical ``submission_id`` and the
    server treats it as the same submission.
    """

    path = Path(path)
    if kind == PRESET_CONTRIBUTION and not user_presets_enabled():
        # Last line of defence: whatever asked for this upload, a contribution is
        # a use of the user's own presets and must never leave while it is OFF.
        raise UploadError("consent_off", detail="personal presets are turned off")
    try:
        size = path.stat().st_size
        digest = sha256_file(path)
    except OSError as exc:
        raise UploadError("file_missing", detail=f"{type(exc).__name__}") from exc
    host = relay_host(relay)
    started = time.monotonic()
    _record(
        event_type="upload_started",
        message=f"uploading {kind} {submission_id}",
        submission_id=submission_id, submission_type=kind, artifact_bytes=size,
        artifact_sha256=digest, relay_host=host, max_attempts=max_attempts,
    )
    last: UploadError | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_started = time.monotonic()
        counter: dict[str, MultipartFileBody] = {}
        try:
            if progress:
                progress(f"Uploading (attempt {attempt} of {max_attempts})…")
            response = relay.post_submission(
                kind=kind, submission_id=submission_id, path=path, sha256=digest,
                version=version, timeout=_timeout_for(size),
                on_body=lambda body: counter.__setitem__("body", body),
            )
            if (
                not isinstance(response, dict)
                or response.get("ok") is not True
                or str(response.get("submission_id")) != submission_id
                or not response.get("receipt_id")
                or str(response.get("sha256")) != digest
                or int(response.get("size", -1)) != size
            ):
                raise UploadError("receipt_mismatch", detail="receipt did not match the uploaded file")
            result = UploadResult(
                submission_id=submission_id,
                receipt_id=str(response["receipt_id"]),
                duplicate=bool(response.get("duplicate", False)),
                size=size, sha256=digest, attempts=attempt,
                elapsed_s=round(time.monotonic() - started, 3),
                stored_md5=str(response.get("stored_md5", "")),
            )
            _record(
                event_type="upload_succeeded",
                message=f"{kind} {submission_id} uploaded",
                submission_id=submission_id, submission_type=kind, receipt_id=result.receipt_id,
                attempt=attempt, elapsed_s=result.elapsed_s, bytes_sent=size,
                duplicate=result.duplicate, relay_host=host,
            )
            return result
        except UploadError as exc:
            raised: BaseException = exc
            error = exc
            status, category = exc.http_status, "receipt"
        except Exception as exc:  # noqa: BLE001 - classified below
            raised = exc
            code, retryable, status, category = classify(exc)
            error = UploadError(code, retryable=retryable, http_status=status, detail=type(exc).__name__)
        error.attempts = attempt
        sent = counter["body"].sent if "body" in counter else 0
        _record(
            event_type="upload_attempt_failed",
            message=f"{kind} upload attempt {attempt} failed: {error.code}",
            severity="warning",
            submission_id=submission_id, submission_type=kind, attempt=attempt,
            elapsed_s=round(time.monotonic() - attempt_started, 3), http_status=status,
            bytes_sent=sent, failure_category=category, error_code=error.code,
            retryable=error.retryable, relay_host=host,
        )
        last = error
        if not error.retryable or attempt >= max_attempts:
            break
        retry_after = _retry_after(raised)
        delay = retry_after if retry_after is not None else BASE_DELAY_S * (2 ** (attempt - 1)) * (0.75 + 0.5 * jitter())
        _record(
            event_type="upload_retry_scheduled",
            message=f"retrying {kind} upload in {delay:.1f}s",
            submission_id=submission_id, submission_type=kind, attempt=attempt, delay_s=round(delay, 2),
        )
        if progress:
            progress(f"Couldn't upload yet — trying again in {int(round(delay))} seconds…")
        (sleep or _sleep)(delay)
    assert last is not None
    _record(
        event_type="upload_failed",
        message=f"{kind} {submission_id} was not uploaded: {last.code}",
        severity="error",
        submission_id=submission_id, submission_type=kind, error_code=last.code,
        attempts=last.attempts, elapsed_s=round(time.monotonic() - started, 3),
        http_status=last.http_status, relay_host=host, local_copy_kept=True,
    )
    raise last
