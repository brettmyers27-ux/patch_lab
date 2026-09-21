"""Real HTTP: PatchLab's client and bug-report worker against the real relay app.

The relay runs in-process on a local port with the filesystem test store, and
faults (503s, slow storage, dead port) are injected around it.  It needs the
relay's own dependencies, so run it with the relay virtualenv:

    PYTHONPATH=. ../patchlab-relay/.venv/bin/python -m pytest tests/test_relay_http_e2e.py

It is skipped (not failed) in an environment without them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

uvicorn = pytest.importorskip("uvicorn")
RELAY_ROOT = Path(__file__).resolve().parents[2] / "patchlab-relay"
if not (RELAY_ROOT / "relay").is_dir():
    pytest.skip("the relay checkout is not next to this repository", allow_module_level=True)
sys.path.insert(0, str(RELAY_ROOT))

from relay import submissions as relay_submissions  # noqa: E402
from relay.app import create_app  # noqa: E402
from relay.service import LocalTestStore, RelayService, hash_password  # noqa: E402

from core import submission_upload as up  # noqa: E402
from core.contribution_bundle import Candidate, build_bundles  # noqa: E402
from core.relay_client import RelayClient  # noqa: E402
from core.submission_upload import UploadError, upload_file  # noqa: E402

PASSCODE = "trusted-e2e-passcode"


class FaultStore(LocalTestStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.fail_puts = 0
        self.delay = 0.0

    def put_submission(self, **kwargs):
        if self.delay:
            time.sleep(self.delay)
        if self.fail_puts:
            self.fail_puts -= 1
            raise OSError("simulated storage outage")
        return super().put_submission(**kwargs)


class Flaky:
    """ASGI wrapper that answers the first N POSTs with a transient error."""

    def __init__(self, app) -> None:
        self.app = app
        self.status = 503
        self.remaining = 0
        self.seen = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/submissions" and self.remaining > 0:
            self.remaining -= 1
            self.seen += 1
            await send({"type": "http.response.start", "status": self.status,
                        "headers": [(b"content-type", b"text/html"), (b"retry-after", b"0")]})
            await send({"type": "http.response.body", "body": b"<html>upstream unavailable</html>"})
            return
        await self.app(scope, receive, send)


class Server:
    def __init__(self, root: Path) -> None:
        self.store = FaultStore(root / "store")
        self.service = RelayService(self.store, hash_password(PASSCODE), b"e2e-secret")
        self.flaky = Flaky(create_app(self.service))
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = int(probe.getsockname()[1])
        self.url = f"http://127.0.0.1:{self.port}"
        self.root = root / "store" / "PatchLab Uploads"
        self._server = uvicorn.Server(uvicorn.Config(self.flaky, host="127.0.0.1", port=self.port, log_level="error"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> "Server":
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        assert self._server.started
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    def stored(self, folder: str) -> list[Path]:
        base = self.root / folder
        return sorted(base.rglob("*.zip")) if base.exists() else []


@pytest.fixture()
def server(tmp_path: Path):
    instance = Server(tmp_path).start()
    yield instance
    instance.stop()


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(up, "_sleep", lambda _s: None)


def client(server: Server, **kwargs) -> RelayClient:
    return RelayClient(server.url, PASSCODE, **kwargs)


def make_bundle(tmp_path: Path, name: str = "bundle.zip", payload: bytes = b"diagnostic bytes") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("summary.txt", payload)
    return path


SID = "1" * 32


def test_bug_report_reaches_private_storage_with_identical_bytes(server: Server, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    result = upload_file(client(server), kind=up.BUG_REPORT, submission_id=SID, path=bundle, version="1.5.4")
    stored = server.stored("bug-reports")
    assert len(stored) == 1 and stored[0].parent.name == SID
    assert hashlib.sha256(stored[0].read_bytes()).hexdigest() == hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert result.receipt_id and result.attempts == 1 and not result.duplicate
    receipt = json.loads((stored[0].parent / "receipt.json").read_text())
    assert receipt["patchlab_version"] == "1.5.4" and receipt["type"] == "bug_report"


def test_preset_contribution_bundle_reaches_its_own_private_folder(server: Server, tmp_path: Path) -> None:
    preset = tmp_path / "Lead.fxp"
    preset.write_bytes(b"CcnK" + b"x" * 200)
    (bundle,) = list(
        build_bundles([Candidate("c" * 40, preset, "Pack/Lead.fxp", 1)],
                      fingerprint_for=lambda c: {"content_hash": c.content_hash}, work_dir=tmp_path / "w")
    )
    result = upload_file(client(server), kind=up.PRESET_CONTRIBUTION, submission_id=bundle.submission_id,
                         path=bundle.path, version="1.5.4")
    stored = server.stored("preset-contributions")
    assert len(stored) == 1 and server.stored("bug-reports") == []
    assert hashlib.sha256(stored[0].read_bytes()).hexdigest() == result.sha256
    assert preset.read_bytes() == b"CcnK" + b"x" * 200, "the user's preset is never modified"


@pytest.mark.parametrize("status", [503, 502, 504, 429])
def test_transient_gateway_errors_are_retried_to_success(server: Server, tmp_path: Path, status: int) -> None:
    server.flaky.status, server.flaky.remaining = status, 2
    result = upload_file(client(server), kind=up.BUG_REPORT, submission_id=SID, path=make_bundle(tmp_path), version="1.5.4")
    assert result.attempts == 3 and server.flaky.seen == 2
    assert len(server.stored("bug-reports")) == 1


def test_storage_outage_is_retried_and_never_leaves_a_partial_copy(server: Server, tmp_path: Path) -> None:
    server.store.fail_puts = 2
    result = upload_file(client(server), kind=up.BUG_REPORT, submission_id=SID, path=make_bundle(tmp_path), version="1.5.4")
    assert result.attempts == 3 and len(server.stored("bug-reports")) == 1
    assert not list(server.root.rglob("*.tmp"))


def test_persistent_storage_failure_gives_up_after_bounded_attempts_and_keeps_the_local_file(
    server: Server, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server.store.fail_puts = 99
    bundle = make_bundle(tmp_path)
    before = bundle.read_bytes()
    with caplog.at_level(logging.INFO), pytest.raises(UploadError) as raised:
        upload_file(client(server), kind=up.BUG_REPORT, submission_id=SID, path=bundle, version="1.5.4")
    assert raised.value.code == "service_unavailable" and raised.value.attempts == up.MAX_ATTEMPTS
    assert bundle.read_bytes() == before and server.stored("bug-reports") == []
    assert PASSCODE not in caplog.text and "Bearer" not in caplog.text


def test_client_timeout_then_retry_is_idempotent_even_though_the_server_finished(
    server: Server, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.store.delay = 1.5
    monkeypatch.setattr(up, "_timeout_for", lambda size: 0.4)
    calls = {"n": 0}
    original = server.store.put_submission

    def slow_first(**kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            server.store.delay = 0
        return original(**kwargs)

    server.store.put_submission = slow_first  # type: ignore[method-assign]
    result = upload_file(client(server), kind=up.BUG_REPORT, submission_id=SID, path=make_bundle(tmp_path), version="1.5.4")
    time.sleep(2.0)  # let the abandoned first request finish on the server
    assert len(server.stored("bug-reports")) == 1, "one stored artifact despite two requests"
    assert result.attempts >= 2


def test_retry_of_a_completed_submission_returns_the_same_receipt(server: Server, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    first = upload_file(client(server), kind=up.BUG_REPORT, submission_id=SID, path=bundle, version="1.5.4")
    second = upload_file(client(server), kind=up.BUG_REPORT, submission_id=SID, path=bundle, version="1.5.4")
    assert second.duplicate and second.receipt_id == first.receipt_id
    assert len(server.stored("bug-reports")) == 1


def test_wrong_credentials_fail_fast_without_retrying(server: Server, tmp_path: Path) -> None:
    bad = RelayClient(server.url, "wrong-passcode")
    with pytest.raises(UploadError) as raised:
        upload_file(bad, kind=up.BUG_REPORT, submission_id=SID, path=make_bundle(tmp_path), version="1.5.4")
    assert raised.value.code == "auth_failed" and raised.value.attempts == 1
    assert server.stored("bug-reports") == []


def test_an_expired_token_is_refreshed_once_with_the_saved_passcode(server: Server, tmp_path: Path) -> None:
    stale = client(server, token="1.deadbeef")
    result = upload_file(stale, kind=up.BUG_REPORT, submission_id=SID, path=make_bundle(tmp_path), version="1.5.4")
    assert result.attempts == 1 and len(server.stored("bug-reports")) == 1


def test_oversized_upload_is_refused_with_a_stable_code(server: Server, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(relay_submissions.MAX_BYTES, "bug_report", 1024)
    big = make_bundle(tmp_path, payload=bytes(range(256)) * 200)
    with pytest.raises(UploadError) as raised:
        upload_file(client(server), kind=up.BUG_REPORT, submission_id=SID, path=big, version="1.5.4")
    assert raised.value.code == "file_too_large" and raised.value.attempts == 1
    assert server.stored("bug-reports") == []


def test_raw_requests_cannot_choose_types_or_paths(server: Server, tmp_path: Path) -> None:
    """The contract itself, spoken by hand rather than through PatchLab's client."""

    token = client(server).token()
    bundle = make_bundle(tmp_path)

    def post(**fields):
        from core.relay_client import MultipartFileBody

        body = MultipartFileBody(
            {"type": "bug_report", "submission_id": SID, "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
             "patchlab_version": "1.5.4", **fields},
            field_name="file", path=bundle, filename="../../../../etc/cron.d/evil.zip",
        )
        request = urllib.request.Request(
            server.url + "/submissions", data=body, method="POST",
            headers={"Authorization": f"Bearer {token}", "Content-Type": body.content_type,
                     "Content-Length": str(body.length)},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    assert post(type="../../secrets")[1]["error_code"] == "invalid_type"
    assert post(type="audio_dump")[0] == 400
    assert post(submission_id="../../../etc/passwd")[1]["error_code"] == "invalid_id"
    status, body = post()
    assert status == 201 and body["ok"] is True
    paths = [str(p.relative_to(server.root)) for p in server.root.rglob("*") if p.is_file()]
    assert all(".." not in p and "etc" not in p for p in paths)
    assert not (tmp_path / "etc").exists()


def test_a_relay_that_is_down_fails_quickly_and_boundedly(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = int(probe.getsockname()[1])
    dead = RelayClient(f"http://127.0.0.1:{dead_port}", PASSCODE)
    started = time.monotonic()
    with pytest.raises(UploadError) as raised:
        upload_file(dead, kind=up.BUG_REPORT, submission_id=SID, path=make_bundle(tmp_path), version="1.5.4")
    assert raised.value.code == "offline" and raised.value.attempts == up.MAX_ATTEMPTS
    assert time.monotonic() - started < 10


# ---------------------------------------------------------------------------
# The bug-report worker (the process the frozen app actually launches)
# ---------------------------------------------------------------------------


def _run_worker(monkeypatch, tmp_path, server_url: str, capsys) -> tuple[int, str, Path, str]:
    import core.bug_report as bug_report_module
    import scripts.submit_bug_report as worker

    monkeypatch.setattr(bug_report_module, "reports_root", lambda: tmp_path)
    monkeypatch.setenv("PATCHLAB_RELAY_URL", server_url)
    monkeypatch.setenv("PATCHLAB_RELAY_PASSWORD", PASSCODE)
    monkeypatch.delenv("PATCHLAB_DISABLE_RELAY", raising=False)
    existing = sorted(tmp_path.glob("PatchLab Bug Report *.txt"))
    request = existing[0] if existing else bug_report_module.create_request(comments="e2e", logs="some log")
    monkeypatch.setattr(sys, "argv", ["x", "--request", str(request)])
    code = worker.main()
    ticket = bug_report_module.load_request(request).ticket_id
    return code, capsys.readouterr().out, request, ticket


def test_worker_uploads_the_saved_bundle_to_the_relay(server: Server, tmp_path: Path, monkeypatch, capsys) -> None:
    code, out, _request, ticket = _run_worker(monkeypatch, tmp_path, server.url, capsys)
    assert code == 0 and "BUG_REPORT_RESULT=" in out
    result = json.loads(out.split("BUG_REPORT_RESULT=")[1].splitlines()[0])
    archive = tmp_path / f"PatchLab Bug Report {ticket} diagnostics.zip"
    (stored,) = server.stored("bug-reports")
    assert stored.parent.name == ticket
    assert stored.read_bytes() == archive.read_bytes()
    assert result["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest() and result["receipt_id"]
    assert any(n.endswith("ticket.txt") for n in zipfile.ZipFile(stored).namelist())


def test_worker_with_a_dead_relay_keeps_the_local_report_and_a_later_retry_succeeds(
    server: Server, tmp_path: Path, monkeypatch, capsys
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{probe.getsockname()[1]}"
    code, out, request, ticket = _run_worker(monkeypatch, tmp_path, dead, capsys)
    assert code == 1 and "BUG_REPORT_ERROR_CODE=offline" in out
    directory = tmp_path / f"PatchLab Bug Report {ticket} diagnostics"
    archive = directory.with_suffix(".zip")
    assert request.is_file() and (directory / "summary.txt").is_file() and archive.is_file()
    snapshot = (archive.read_bytes(), (directory / "summary.txt").read_bytes())

    code, out, _r, _t = _run_worker(monkeypatch, tmp_path, server.url, capsys)  # the manual retry
    assert code == 0
    assert (archive.read_bytes(), (directory / "summary.txt").read_bytes()) == snapshot, "not regenerated"
    (stored,) = server.stored("bug-reports")
    assert stored.read_bytes() == snapshot[0]


def test_worker_output_never_contains_credentials(server: Server, tmp_path: Path, monkeypatch, capsys) -> None:
    code, out, _request, _ticket = _run_worker(monkeypatch, tmp_path, server.url, capsys)
    assert PASSCODE not in out and "Bearer" not in out


# ---------------------------------------------------------------------------
# Large-installer downloads through the frozen-1.5.3 client contract
# ---------------------------------------------------------------------------


def _artifact_server(tmp_path: Path, payload: bytes):
    """The real relay app + real ArtifactCatalog serving ``payload`` as a 'macOS PKG'."""

    from relay.artifacts import Artifact, ArtifactCatalog, LocalArtifactStore

    source = tmp_path / "PatchLab-9.9.9-macOS.pkg"
    source.write_bytes(payload)
    artifact = Artifact(name=source.name, version="9.9.9", size=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest(),
                        destination="releases/macos/" + source.name, drive_file_id="local", kind="macos-package")
    catalog = ArtifactCatalog([artifact], LocalArtifactStore({source.name: source}))
    server = Server(tmp_path)
    server.flaky.app = create_app(server.service, catalog)
    return server.start(), artifact


def test_installer_larger_than_the_buffered_response_limit_downloads_chunked_and_intact(tmp_path: Path) -> None:
    from relay.app import MAX_BUFFERED_RESPONSE_BYTES

    payload = b"xar!" + bytes(range(256)) * ((MAX_BUFFERED_RESPONSE_BYTES // 256) + 4096)
    assert len(payload) > MAX_BUFFERED_RESPONSE_BYTES
    server, artifact = _artifact_server(tmp_path, payload)
    try:
        api = client(server)
        request = urllib.request.Request(server.url + "/artifacts/" + artifact.name,
                                         headers={"Authorization": f"Bearer {api.token()}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            assert response.status == 200
            assert response.headers.get("Content-Length") is None, "large responses must be streamed"
            assert response.headers.get("Transfer-Encoding") == "chunked"
            assert response.headers.get("Accept-Ranges") == "bytes"
            body = response.read()
        assert hashlib.sha256(body).hexdigest() == artifact.sha256
    finally:
        server.stop()


def test_a_download_that_stops_part_way_is_resumed_by_the_1_5_3_client_from_its_partial_file(tmp_path: Path) -> None:
    """If a stream ever stalls, clicking Update again must continue, not restart."""

    payload = b"xar!" + bytes(range(256)) * 20000
    server, artifact = _artifact_server(tmp_path, payload)
    try:
        api = client(server)
        destination = tmp_path / "updates" / artifact.name
        destination.parent.mkdir(parents=True)
        cut = 1_234_567
        (destination.parent / f".{artifact.name}.part").write_bytes(payload[:cut])
        seen: list[int] = []
        result = api.download_artifact(name=artifact.name, destination=destination, size=artifact.size,
                                       sha256=artifact.sha256, progress=lambda got, _total: seen.append(got))
        assert result.read_bytes() == payload
        assert seen and seen[0] > cut, "the client continued from its partial file"
    finally:
        server.stop()
