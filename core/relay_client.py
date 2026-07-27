"""Narrow upload-only client for contributed preset files and fingerprints."""

from __future__ import annotations

import json
import mimetypes
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class UploadReceipt:
    content_hash: str
    stored: bool
    relative_path: str


class RelayClient:
    """Only exposes auth, hash existence, and preset upload operations."""

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
