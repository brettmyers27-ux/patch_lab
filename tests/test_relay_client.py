from __future__ import annotations

import io
import json
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
