"""Narrow upload-only client for contributed preset files and fingerprints."""

from __future__ import annotations

import json
import mimetypes
import secrets
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class UploadReceipt:
    content_hash: str
    stored: bool
    relative_path: str


class RelayClient:
    """Private relay operations available to an authenticated PatchLab user."""

    def __init__(
        self,
        base_url: str,
        password: str,
        *,
        timeout: float = 30.0,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.password = password
        self.timeout = timeout
        self._token: str | None = token

    def _json(
        self, endpoint: str, payload: dict[str, Any], *, authenticated: bool = True
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.token()}"
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        for attempt in range(2):
            if authenticated:
                headers["Authorization"] = f"Bearer {self.token()}"
            request = urllib.request.Request(
                self.base_url + endpoint,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if (
                    exc.code != 401
                    or not authenticated
                    or not self.password
                    or attempt
                ):
                    raise
                self._token = None
        raise AssertionError("unreachable relay retry state")

    def artifact_manifest(self) -> list[dict[str, Any]]:
        """Read the authenticated private artifact catalog.

        This is deliberately separate from the GitHub source version check:
        a packaged update is only considered available when its actual PKG is
        present behind the trusted-group relay and can be checksum verified.
        """

        for attempt in range(2):
            request = urllib.request.Request(
                self.base_url + "/artifacts",
                headers={"Authorization": f"Bearer {self.token()}"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                rows = payload.get("artifacts")
                if not isinstance(rows, list):
                    raise ValueError("relay returned an invalid artifact catalog")
                return [row for row in rows if isinstance(row, dict)]
            except urllib.error.HTTPError as exc:
                if exc.code != 401 or not self.password or attempt:
                    raise
                self._token = None
        raise AssertionError("unreachable relay retry state")

    def download_artifact(
        self,
        *,
        name: str,
        destination: Path,
        size: int,
        sha256: str,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Resume, checksum-verify, and atomically publish one relay artifact."""

        if not name or "/" in name or "\\" in name:
            raise ValueError("invalid private artifact name")
        if size <= 0 or len(sha256) != 64:
            raise ValueError("invalid private artifact metadata")
        destination = Path(destination).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(f".{destination.name}.part")
        if destination.is_file() and destination.stat().st_size == size:
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == sha256:
                return destination
            destination.unlink(missing_ok=True)
        if partial.exists() and partial.stat().st_size > size:
            partial.unlink()

        for attempt in range(2):
            start = partial.stat().st_size if partial.exists() else 0
            headers = {"Authorization": f"Bearer {self.token()}"}
            if start:
                headers["Range"] = f"bytes={start}-"
            request = urllib.request.Request(
                self.base_url + "/artifacts/" + urllib.parse.quote(name, safe=""),
                headers=headers,
                method="GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=max(self.timeout, 300.0)) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    # A relay which ignored a resume request must never append a
                    # complete payload to a partial one.
                    if start and status != 206:
                        partial.unlink(missing_ok=True)
                        start = 0
                        mode = "wb"
                    else:
                        mode = "ab" if start else "wb"
                    received = start
                    last_reported = -1
                    with partial.open(mode) as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                            received += len(chunk)
                            if received > size:
                                raise ValueError("download exceeded its declared size")
                            percent = int(received * 100 / size)
                            if progress is not None and percent != last_reported:
                                progress(received, size)
                                last_reported = percent
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 401 or not self.password or attempt:
                    raise
                self._token = None
        if not partial.is_file() or partial.stat().st_size != size:
            actual = partial.stat().st_size if partial.exists() else 0
            raise ValueError(f"download is incomplete ({actual}/{size} bytes)")
        digest = hashlib.sha256()
        with partial.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            partial.unlink(missing_ok=True)
            raise ValueError("download checksum did not match the published release")
        with partial.open("rb") as stream:
            header = stream.read(4)
        if header != b"xar!":
            partial.unlink(missing_ok=True)
            raise ValueError("download is not a valid macOS installer package")
        partial.replace(destination)
        return destination

    def token(self) -> str:
        if self._token is None:
            result = self._json(
                "/auth", {"password": self.password}, authenticated=False
            )
            self._token = str(result["token"])
        return self._token

    def check_hash(self, content_hash: str) -> bool:
        result = self._json("/check-hash", {"content_hash": content_hash})
        return bool(result["exists"])

    def upload(
        self,
        *,
        preset_path: Path,
        relative_path: str,
        content_hash: str,
        fingerprint: dict[str, Any],
    ) -> UploadReceipt:
        preset_path = Path(preset_path).resolve()
        if preset_path.suffix.casefold() not in {".fxp", ".serumpreset"}:
            raise ValueError("Relay uploads are restricted to Serum preset files")
        boundary = "PatchLab" + secrets.token_hex(16)
        parts: list[bytes] = []

        def field(name: str, value: str) -> None:
            parts.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode("utf-8"),
                    b"\r\n",
                ]
            )

        field("content_hash", content_hash)
        field("relative_path", relative_path)
        field(
            "fingerprint_json",
            json.dumps(fingerprint, separators=(",", ":"), ensure_ascii=False),
        )
        content_type = mimetypes.guess_type(preset_path.name)[0] or "application/octet-stream"
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="preset"; '
                    f'filename="{preset_path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                preset_path.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        body = b"".join(parts)
        # A defensive invariant: WAV/AIFF container signatures may never enter
        # the relay request body. Preset internals are not interpreted here.
        if preset_path.suffix.casefold() not in {".fxp", ".serumpreset"}:
            raise AssertionError("non-preset upload payload")
        for attempt in range(2):
            request = urllib.request.Request(
                self.base_url + "/upload",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.token()}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 401 or not self.password or attempt:
                    raise
                self._token = None
        return UploadReceipt(
            content_hash=content_hash,
            stored=bool(result.get("stored", True)),
            relative_path=relative_path,
        )

    def submit_bug_report(
        self,
        *,
        ticket_id: str,
        comments: str,
        logs: str,
    ) -> dict[str, str]:
        """Send a user-approved text-only diagnostics report to private support."""

        if not ticket_id or not comments.strip():
            raise ValueError("A bug report needs an identifier and a description")
        boundary = "PatchLabBugReport" + secrets.token_hex(16)
        parts: list[bytes] = []

        def field(name: str, value: str) -> None:
            parts.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode("utf-8"),
                    b"\r\n",
                ]
            )

        field("ticket_id", ticket_id)
        field("comments", comments)
        field("logs", logs)
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        for attempt in range(2):
            request = urllib.request.Request(
                self.base_url + "/bug-reports",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.token()}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 401 or not self.password or attempt:
                    raise
                self._token = None
        return {"ticket_id": str(result["ticket_id"])}
