#!/usr/bin/env python3
"""Exercise PatchLab's real client against a locally running private relay."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELAY_ROOT = PROJECT_ROOT.parent / "patchlab-relay"
for path in (PROJECT_ROOT, RELAY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.relay_client import RelayClient
from relay.app import create_app
from relay.service import LocalTestStore, RelayService, hash_password


def main() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    with tempfile.TemporaryDirectory(prefix="patchlab-real-relay-") as directory:
        root = Path(directory)
        service = RelayService(
            LocalTestStore(root / "store"),
            hash_password("trusted"),
            b"local-test-token-secret",
        )
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(service),
                host="127.0.0.1",
                port=port,
                log_level="error",
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not server.started:
            raise RuntimeError("Local relay did not start")
        preset = root / "Fixture.fxp"
        preset.write_bytes(b"CcnK" + b"PatchLabLocalRelay" * 8)
        digest = hashlib.sha1(preset.read_bytes()).hexdigest()
        fingerprint = {"schema": 1, "content_hash": digest, "synth": "serum1"}
        client = RelayClient(f"http://127.0.0.1:{port}", "trusted")
        before = client.check_hash(digest)
        receipt = client.upload(
            preset_path=preset,
            relative_path="Gate/Fixture.fxp",
            content_hash=digest,
            fingerprint=fingerprint,
        )
        after = client.check_hash(digest)
        server.should_exit = True
        thread.join(timeout=10)
        stored = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]
    payload = {
        "real_http_server": True,
        "check_before": before,
        "upload_stored": receipt.stored,
        "check_after": after,
        "stored_files": stored,
        "server_stopped": not thread.is_alive(),
    }
    payload["gate_pass"] = (
        not before and receipt.stored and after and payload["server_stopped"]
    )
    print("LOCAL_RELAY_HTTP=" + json.dumps(payload, sort_keys=True))
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
