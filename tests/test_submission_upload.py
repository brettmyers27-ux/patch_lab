"""Client-side behaviour of the single-file upload (no network)."""

from __future__ import annotations

import io
import json
import socket
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import submission_upload as up
from core.contribution_bundle import Candidate, ContributionLedger, build_bundles
from core.diagnostics import DiagnosticRecorder, set_recorder
from core.relay_client import MultipartFileBody, RelayClient
from core.submission_upload import UploadError, upload_file

SID = "a" * 32


def http_error(status: int, body: dict | None = None, headers: dict | None = None) -> urllib.error.HTTPError:
    payload = json.dumps(body or {}).encode()
    return urllib.error.HTTPError("https://relay.invalid/submissions", status, "x", headers or {}, io.BytesIO(payload))


class ScriptedRelay:
    base_url = "https://relay.example.invalid"

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def post_submission(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, dict):
            return outcome
        path = Path(kwargs["path"])
        return {
            "ok": True, "submission_id": kwargs["submission_id"], "receipt_id": "r" * 24,
            "sha256": kwargs["sha256"], "size": path.stat().st_size, "stored_md5": "m", "duplicate": False,
        }


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("summary.txt", "hello")
    return path


@pytest.fixture()
def events(tmp_path: Path):
    recorder = DiagnosticRecorder(root=tmp_path / "diag", session_id="upload-test")
    set_recorder(recorder)
    yield lambda: [SimpleNamespace(**e) for e in recorder.recent_events() if e["subsystem"] == "relay-upload"]
    recorder.close()
    set_recorder(None)


def run(relay, bundle, sleeps=None, **kwargs):
    sleeps = sleeps if sleeps is not None else []
    return upload_file(relay, kind=up.BUG_REPORT, submission_id=SID, path=bundle, version="1.5.4",
                       sleep=sleeps.append, jitter=lambda: 0.5, **kwargs)


def test_success_returns_a_verified_receipt(bundle: Path, events) -> None:
    result = run(ScriptedRelay(), bundle)
    assert result.receipt_id == "r" * 24 and result.attempts == 1 and result.sha256 == up.sha256_file(bundle)
    kinds = [e.event_type for e in events()]
    assert kinds == ["upload_started", "upload_succeeded"]


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_transient_http_statuses_are_retried_with_backoff_then_succeed(bundle: Path, status: int, events) -> None:
    sleeps: list[float] = []
    relay = ScriptedRelay(http_error(status), http_error(status), "ok")
    result = run(relay, bundle, sleeps)
    assert result.attempts == 3 and len(relay.calls) == 3
    assert sleeps == [1.0, 2.0], "exponential backoff (jitter pinned to neutral)"
    assert [e.event_type for e in events()].count("upload_retry_scheduled") == 2
    assert {c["submission_id"] for c in relay.calls} == {SID}, "a retry reuses the same submission id"


def test_retry_after_header_is_honoured_but_capped(bundle: Path) -> None:
    sleeps: list[float] = []
    run(ScriptedRelay(http_error(429, headers={"Retry-After": "3"}), "ok"), bundle, sleeps)
    assert sleeps == [3.0]
    sleeps = []
    run(ScriptedRelay(http_error(503, headers={"Retry-After": "9999"}), "ok"), bundle, sleeps)
    assert sleeps == [up.MAX_RETRY_AFTER_S]


@pytest.mark.parametrize(
    "status, body, code",
    [
        (401, {"error_code": "auth_invalid"}, "auth_failed"),
        (400, {"error_code": "invalid_type"}, "rejected"),
        (400, {"error_code": "unsupported_file"}, "rejected"),
        (413, {"error_code": "file_too_large"}, "file_too_large"),
        (409, {"error_code": "submission_conflict"}, "submission_conflict"),
        (500, {}, "rejected"),
    ],
)
def test_permanent_failures_are_not_retried(bundle: Path, status: int, body: dict, code: str) -> None:
    relay = ScriptedRelay(http_error(status, body))
    sleeps: list[float] = []
    with pytest.raises(UploadError) as raised:
        run(relay, bundle, sleeps)
    assert raised.value.code == code and raised.value.retryable is False
    assert len(relay.calls) == 1 and sleeps == []


def test_offline_is_retried_a_bounded_number_of_times_then_reported(bundle: Path, events) -> None:
    relay = ScriptedRelay(*[urllib.error.URLError(socket.gaierror("no internet")) for _ in range(10)])
    sleeps: list[float] = []
    with pytest.raises(UploadError) as raised:
        run(relay, bundle, sleeps)
    assert raised.value.code == "offline" and raised.value.attempts == up.MAX_ATTEMPTS
    assert len(relay.calls) == up.MAX_ATTEMPTS and len(sleeps) == up.MAX_ATTEMPTS - 1
    assert sum(sleeps) < 15, "bounded: the user is never left waiting long"
    assert events()[-1].event_type == "upload_failed"
    assert events()[-1].fields["local_copy_kept"] is True


def test_timeouts_are_classified_and_retried(bundle: Path) -> None:
    relay = ScriptedRelay(socket.timeout("timed out"), urllib.error.URLError(socket.timeout()), "ok")
    assert run(relay, bundle).attempts == 3


def test_a_receipt_that_does_not_match_the_file_is_never_trusted(bundle: Path) -> None:
    wrong = {"ok": True, "submission_id": SID, "receipt_id": "x", "sha256": "0" * 64, "size": 1}
    with pytest.raises(UploadError) as raised:
        run(ScriptedRelay(wrong), bundle)
    assert raised.value.code == "receipt_mismatch"


def test_missing_file_fails_cleanly_without_touching_the_network(tmp_path: Path) -> None:
    relay = ScriptedRelay()
    with pytest.raises(UploadError) as raised:
        run(relay, tmp_path / "gone.zip")
    assert raised.value.code == "file_missing" and relay.calls == []


def test_the_uploaded_file_is_never_modified(bundle: Path) -> None:
    before = bundle.read_bytes()
    with pytest.raises(UploadError):
        run(ScriptedRelay(*[http_error(503)] * 5), bundle)
    assert bundle.read_bytes() == before


def test_diagnostics_never_contain_credentials(bundle: Path, events) -> None:
    secret = "SUPER-SECRET-BEARER-abc123"
    relay = RelayClient("https://relay.invalid", "passcode-xyz-789", token=secret)
    relay.post_submission = lambda **_kw: (_ for _ in ()).throw(http_error(503))  # type: ignore[method-assign]
    with pytest.raises(UploadError):
        run(relay, bundle)
    blob = json.dumps([vars(e) for e in events()])
    assert secret not in blob and "passcode-xyz-789" not in blob and "Bearer" not in blob
    fields = events()[-1].fields
    assert fields["relay_host"] == "relay.invalid" and fields["submission_id"] == SID
    assert fields["error_code"] == "service_unavailable"


def test_upload_body_is_streamed_multipart_not_base64_json(bundle: Path) -> None:
    payload = bundle.read_bytes()
    body = MultipartFileBody(
        {"type": "bug_report", "submission_id": SID}, field_name="file", path=bundle, filename="../weird name.zip"
    )
    chunks = []
    while chunk := body.read(7):
        chunks.append(chunk)
    raw = b"".join(chunks)
    assert len(raw) == body.length and body.sent == body.length
    assert payload in raw, "raw bytes, not base64"
    assert b'filename="_weird name.zip"' in raw or b"weird name.zip" in raw
    assert b"../" not in raw.split(payload)[0]
    assert max(len(c) for c in chunks) <= 7, "reads are bounded by the requested size"


# ---------------------------------------------------------------------------
# Connection state and the bug-report worker
# ---------------------------------------------------------------------------


def test_connection_reasons_are_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    from core import access_gate

    monkeypatch.delenv("PATCHLAB_RELAY_PASSWORD", raising=False)
    monkeypatch.setenv("PATCHLAB_DISABLE_RELAY", "1")
    assert up.resolve_relay() == (None, "local_only")
    monkeypatch.delenv("PATCHLAB_DISABLE_RELAY")
    monkeypatch.setenv("PATCHLAB_RELAY_URL", "")
    assert up.resolve_relay() == (None, "no_url")
    monkeypatch.setenv("PATCHLAB_RELAY_URL", "https://relay.invalid")
    monkeypatch.setattr(access_gate, "stored_token", lambda: None)
    monkeypatch.setattr(access_gate, "stored_passcode", lambda *_a, **_k: None)
    assert up.resolve_relay() == (None, "no_credentials")
    monkeypatch.setattr(access_gate, "stored_passcode", lambda *_a, **_k: "pw")
    client, reason = up.resolve_relay()
    assert reason == "ok" and client is not None and client.base_url == "https://relay.invalid"


def test_a_saved_token_never_touches_the_keychain_unless_it_has_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reported hang: a background worker waiting forever on a keychain prompt."""

    from core import access_gate

    monkeypatch.delenv("PATCHLAB_RELAY_PASSWORD", raising=False)
    monkeypatch.delenv("PATCHLAB_DISABLE_RELAY", raising=False)
    monkeypatch.setenv("PATCHLAB_RELAY_URL", "https://relay.invalid")
    reads: list[str] = []
    monkeypatch.setattr(access_gate, "stored_token", lambda: "1.savedtoken")
    monkeypatch.setattr(access_gate, "stored_passcode", lambda *_a, **_k: reads.append("keychain") or "pw")
    client, reason = up.resolve_relay()
    assert reason == "ok" and reads == [], "resolving a connection must not read the keychain"

    import urllib.request

    seen: list[str] = []

    class Response:
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *_a): return False
        def read(self): return json.dumps(self.payload).encode()

    def urlopen(request, *, timeout):
        seen.append(request.full_url.rsplit("/", 1)[-1])
        if request.full_url.endswith("/auth"):
            return Response({"token": "fresh"})
        if len([s for s in seen if s == "submissions"]) == 1:
            raise http_error(401, {"error_code": "auth_invalid"})
        return Response({"ok": True})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    bundle = Path(__file__)
    client.post_submission(kind="bug_report", submission_id=SID, path=bundle, sha256="0" * 64, version="1", timeout=5)
    assert seen == ["submissions", "auth", "submissions"] and reads == ["keychain"], "read once, only after the 401"


def test_a_keychain_that_never_answers_is_abandoned_after_the_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    from core import access_gate

    release = threading.Event()
    monkeypatch.setattr(access_gate.AccessStore, "passcode", lambda self: release.wait(30) and "never")
    started = time.monotonic()
    assert access_gate.stored_passcode(timeout=0.3) is None
    assert time.monotonic() - started < 3
    release.set()


def test_a_keychain_that_answers_returns_the_passcode(monkeypatch: pytest.MonkeyPatch) -> None:
    from core import access_gate

    monkeypatch.setattr(access_gate.AccessStore, "passcode", lambda self: "group-passcode")
    assert access_gate.stored_passcode(timeout=2) == "group-passcode"


def test_worker_reports_not_connected_but_keeps_the_saved_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import sys

    import core.bug_report as bug_report_module
    import scripts.submit_bug_report as worker

    monkeypatch.setattr(bug_report_module, "reports_root", lambda: tmp_path)
    monkeypatch.setattr(up, "resolve_relay", lambda: (None, "local_only"))
    request = bug_report_module.create_request(comments="it broke", logs="log text")
    monkeypatch.setattr(sys, "argv", ["x", "--request", str(request)])
    assert worker.main() == 1
    out = capsys.readouterr().out
    assert "BUG_REPORT_ERROR_CODE=not_connected" in out and "BUG_REPORT_CONNECTION=local_only" in out
    assert request.is_file()
    ticket = bug_report_module.load_request(request).ticket_id
    assert (tmp_path / f"PatchLab Bug Report {ticket} diagnostics" / "summary.txt").is_file()
    assert (tmp_path / f"PatchLab Bug Report {ticket} diagnostics" / "ticket.txt").is_file()


def test_worker_retry_uploads_the_same_saved_archive_without_regenerating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import sys

    import core.bug_report as bug_report_module
    import scripts.submit_bug_report as worker

    monkeypatch.setattr(bug_report_module, "reports_root", lambda: tmp_path)
    request = bug_report_module.create_request(comments="it broke", logs="log text")
    ticket = bug_report_module.load_request(request).ticket_id
    relay = ScriptedRelay(urllib.error.URLError(socket.gaierror("offline")), "ok")
    monkeypatch.setattr(up, "resolve_relay", lambda: (relay, "ok"))
    monkeypatch.setattr(up, "_sleep", lambda _s: None)
    monkeypatch.setattr(sys, "argv", ["x", "--request", str(request)])

    assert worker.main() == 0  # attempt 1 fails, the automatic retry succeeds
    first = capsys.readouterr().out
    assert "BUG_REPORT_RESULT=" in first
    archive = tmp_path / f"PatchLab Bug Report {ticket} diagnostics.zip"
    assert archive.is_file()
    stamp = (archive.stat().st_mtime_ns, (archive.with_suffix("").joinpath("summary.txt")).stat().st_mtime_ns)
    sent = {Path(c["path"]) for c in relay.calls}
    assert sent == {archive}

    relay.outcomes = ["ok"]
    assert worker.main() == 0  # a manual retry
    assert (archive.stat().st_mtime_ns, archive.with_suffix("").joinpath("summary.txt").stat().st_mtime_ns) == stamp
    assert relay.calls[-1]["submission_id"] == ticket
    assert {c["sha256"] for c in relay.calls} == {up.sha256_file(archive)}, "identical bytes every attempt"
    assert "ticket.txt" in zipfile.ZipFile(archive).namelist() or any(
        n.endswith("/ticket.txt") for n in zipfile.ZipFile(archive).namelist()
    )


# ---------------------------------------------------------------------------
# Contribution bundles
# ---------------------------------------------------------------------------


def _candidates(tmp_path: Path, count: int) -> list[Candidate]:
    items = []
    for index in range(count):
        path = tmp_path / "presets" / f"P{index}.fxp"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"CcnK" + bytes([index]) * 64)
        items.append(Candidate(f"{index:040x}", path, f"Pack/P{index}.fxp", index))
    return items


def test_bundles_are_deterministic_so_retries_share_one_submission_id(tmp_path: Path) -> None:
    items = _candidates(tmp_path, 5)
    fingerprint = lambda item: {"content_hash": item.content_hash, "embeddings": {"0": [0.1, 0.2]}}
    first = [(b.submission_id, b.path.read_bytes()) for b in build_bundles(items, fingerprint_for=fingerprint, work_dir=tmp_path / "w1")]
    second = [(b.submission_id, b.path.read_bytes()) for b in build_bundles(list(reversed(items)), fingerprint_for=fingerprint, work_dir=tmp_path / "w2")]
    assert first == second and len(first) == 1


def test_a_large_set_is_split_into_bounded_bundles_and_manifests_are_complete(tmp_path: Path) -> None:
    items = _candidates(tmp_path, 12)
    bundles = list(build_bundles(items, fingerprint_for=lambda i: {"pad": "x" * 4000, "h": i.content_hash},
                                 work_dir=tmp_path / "w", max_bytes=2000))
    assert len(bundles) > 1
    assert sorted(h for b in bundles for h in b.content_hashes) == sorted(i.content_hash for i in items)
    for bundle in bundles:
        with zipfile.ZipFile(bundle.path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["count"] == len(bundle.content_hashes)
            assert not any(n.endswith((".wav", ".aif", ".aiff", ".flac")) for n in archive.namelist())
    assert len({b.submission_id for b in bundles}) == len(bundles)


def test_ledger_records_only_what_was_receipted_and_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = ContributionLedger(path)
    assert "a" * 40 not in ledger
    ledger.add_many(["a" * 40])
    assert "a" * 40 in ContributionLedger(path)
    path.write_text("not json")
    assert "a" * 40 not in ContributionLedger(path), "a damaged ledger just means re-offering, never a crash"
