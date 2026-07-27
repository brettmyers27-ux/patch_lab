from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts.install_support import _download


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
        result = _download(
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
