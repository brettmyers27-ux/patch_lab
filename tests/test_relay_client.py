from __future__ import annotations

import io
import json
import hashlib
import urllib.error
import urllib.request

from core.relay_client import RelayClient


class Response:
    def __init__(self, payload: dict) -> None:
        self.stream = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.stream.read()


def test_expired_stored_token_reauthenticates_once(monkeypatch) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, *, timeout: float):
        del timeout
        requests.append(request)
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 401, "expired", {}, None
            )
        if request.full_url.endswith("/auth"):
            return Response({"token": "fresh-token"})
        return Response({"exists": True})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = RelayClient(
        "https://relay.invalid",
        "stored-passcode",
        token="expired-token",
    )

    assert client.check_hash("a" * 40) is True
    assert len(requests) == 3
    assert requests[0].headers["Authorization"] == "Bearer expired-token"
    assert requests[1].full_url.endswith("/auth")
    assert "Authorization" not in requests[1].headers
    assert requests[2].headers["Authorization"] == "Bearer fresh-token"


def test_bug_report_posts_only_text_fields_to_private_endpoint(monkeypatch) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, *, timeout: float):
        del timeout
        requests.append(request)
        return Response({"ticket_id": "d" * 32})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = RelayClient(
        "https://relay.invalid", "unused-passcode", token="already-authorized"
    )

    receipt = client.submit_bug_report(
        ticket_id="d" * 32,
        comments="Run Match stayed disabled after choosing a sound.",
        logs="PatchLab diagnostic fixture",
    )

    assert receipt == {"ticket_id": "d" * 32}
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == "https://relay.invalid/bug-reports"
    assert request.headers["Authorization"] == "Bearer already-authorized"
    body = request.data.decode("utf-8")
    assert 'name="comments"' in body
    assert 'name="logs"' in body
    assert "preset" not in body.casefold()
    assert "audio" not in body.casefold()


def test_private_package_download_resumes_and_verifies_before_publishing(
    tmp_path, monkeypatch
) -> None:
    payload = b"xar!" + b"verified-installer" * 20
    target = tmp_path / "PatchLab-1.5.3-macOS.pkg"
    partial = target.with_name(f".{target.name}.part")
    partial.write_bytes(payload[:10])
    requests: list[urllib.request.Request] = []

    class BinaryResponse:
        status = 206

        def __init__(self, body: bytes) -> None:
            self.stream = io.BytesIO(body)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def getcode(self):
            return self.status

        def read(self, size: int = -1) -> bytes:
            return self.stream.read(size)

    def urlopen(request: urllib.request.Request, *, timeout: float):
        del timeout
        requests.append(request)
        return BinaryResponse(payload[10:])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = RelayClient("https://relay.invalid", "", token="authorized")
    result = client.download_artifact(
        name=target.name,
        destination=target,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert result == target
    assert target.read_bytes() == payload
    assert not partial.exists()
    assert requests[0].headers["Range"] == "bytes=10-"
