from __future__ import annotations

import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts import install_support


def test_download_resumes_existing_partial_and_verifies_sha256(tmp_path: Path) -> None:
    payload = (b"PatchLab resumable installer fixture\n" * 100_000) + b"end"
    seen_ranges: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            range_header = self.headers.get("Range")
            seen_ranges.append(range_header)
            start = int(range_header.removeprefix("bytes=").removesuffix("-")) if range_header else 0
            body = payload[start:]
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Length", str(len(body)))
            if range_header:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "artifact.bin"
    partial = tmp_path / "artifact.bin.part"
    partial.write_bytes(payload[:123_456])
    try:
        result = install_support._download(
            f"http://127.0.0.1:{server.server_port}/artifact.bin",
            destination,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    finally:
        server.shutdown()
        thread.join()

    assert result == "downloaded and verified"
    assert seen_ranges == ["bytes=123456-"]
    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_download_retries_transient_500_and_preserves_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"PatchLab transient retry fixture" * 10_000
    requests = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            nonlocal requests
            requests += 1
            if requests < 3:
                body = json.dumps({"detail": "temporary upstream failure"}).encode()
                self.send_response(500)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args) -> None:
            return

    monkeypatch.setattr(install_support.time, "sleep", lambda _seconds: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "retry.bin"
    try:
        result = install_support._download(
            f"http://127.0.0.1:{server.server_port}/retry.bin",
            destination,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    finally:
        server.shutdown()
        thread.join()

    assert result == "downloaded and verified"
    assert requests == 3
    assert destination.read_bytes() == payload


def test_artifact_preflight_names_unreachable_artifact_before_clap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            nonlocal requests
            requests += 1
            body = json.dumps(
                {
                    "detail": {
                        "error": "artifact is not retrievable from storage",
                        "artifact": "delta_param_model.pt",
                        "drive_status": 403,
                    }
                }
            ).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args) -> None:
            return

    monkeypatch.setattr(install_support.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        install_support,
        "_artifact_manifest",
        lambda _relay_url: (
            "token",
            [
                {
                    "name": "delta_param_model.pt",
                    "size": 92_150_023,
                    "sha256": "a" * 64,
                    "destination": "data/models/delta_param_model.pt",
                }
            ],
        ),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(install_support.InstallError) as captured:
            install_support._artifacts_preflight(
                argparse.Namespace(
                    relay_url=f"http://127.0.0.1:{server.server_port}"
                )
            )
    finally:
        server.shutdown()
        thread.join()

    message = str(captured.value)
    assert "delta_param_model.pt" in message
    assert "HTTP 502" in message
    assert "large public CLAP download has not started" in message
    assert "completed data is preserved" in message
    assert requests == install_support.MAX_NETWORK_ATTEMPTS
