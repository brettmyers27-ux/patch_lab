#!/usr/bin/env python3
"""Non-interactive primitives used by install.sh.

Secrets are accepted on stdin, never as command-line arguments. All failures
become one-line actionable messages rather than tracebacks.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.access_gate import AccessManager, AccessStore  # noqa: E402


CLAP_NAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/"
    f"{CLAP_NAME}?download=true"
)
CLAP_SIZE = 2_352_471_003
CLAP_SHA256 = "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd"
MAX_NETWORK_ATTEMPTS = 4
TRANSIENT_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


class InstallError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, *, size: int, sha256: str) -> bool:
    return path.is_file() and path.stat().st_size == size and _sha256(path) == sha256


def _backoff(attempt: int, label: str) -> None:
    delay = 2 ** attempt
    print(
        f"  {label}: transient network failure; retrying in {delay}s "
        f"({attempt + 1}/{MAX_NETWORK_ATTEMPTS - 1})",
        flush=True,
    )
    time.sleep(delay)


def _http_detail(body: bytes) -> str:
    if not body:
        return "the server returned no explanation"
    text = body.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(text)
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, dict):
            return json.dumps(detail, sort_keys=True)
        if detail:
            return str(detail)
    except json.JSONDecodeError:
        pass
    return text[:1000]


def _small_request(
    url: str,
    *,
    label: str,
    token: str | None = None,
    byte_range: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    for attempt in range(MAX_NETWORK_ATTEMPTS):
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if byte_range:
            headers["Range"] = byte_range
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return (
                    int(getattr(response, "status", response.getcode())),
                    {
                        key.casefold(): value
                        for key, value in response.headers.items()
                    },
                    response.read(),
                )
        except urllib.error.HTTPError as exc:
            body = exc.read()
            if (
                exc.code in TRANSIENT_HTTP_STATUS
                and attempt < MAX_NETWORK_ATTEMPTS - 1
            ):
                _backoff(attempt, label)
                continue
            return (
                exc.code,
                {key.casefold(): value for key, value in exc.headers.items()},
                body,
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ) as exc:
            if attempt < MAX_NETWORK_ATTEMPTS - 1:
                _backoff(attempt, label)
                continue
            reason = getattr(exc, "reason", exc)
            raise InstallError(
                f"{label} is unreachable after {MAX_NETWORK_ATTEMPTS} attempts: "
                f"{reason}. Completed downloads are preserved; rerun the installer "
                "to resume."
            ) from exc
    raise AssertionError("network retry loop ended unexpectedly")


def _download(
    url: str,
    destination: Path,
    *,
    size: int,
    sha256: str,
    token: str | None = None,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _verified(destination, size=size, sha256=sha256):
        return "already verified"
    if destination.exists():
        raise InstallError(
            f"{destination} exists but fails its checksum; move it aside and rerun"
        )
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > size:
        partial.unlink()

    attempts = 0
    while attempts < MAX_NETWORK_ATTEMPTS:
        if partial.is_file() and partial.stat().st_size == size:
            if _sha256(partial) == sha256:
                partial.replace(destination)
                return "resumed and verified"
            partial.unlink()
            raise InstallError(
                f"{destination.name} failed SHA-256 verification; partial file removed"
            )
        start = partial.stat().st_size if partial.exists() else 0
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise InstallError(
                    f"{destination.name}: artifact authorization expired or was "
                    "rejected. Completed downloads are preserved; rerun the "
                    "installer to refresh authorization and resume."
                ) from exc
            body = exc.read()
            if (
                exc.code in TRANSIENT_HTTP_STATUS
                and attempts < MAX_NETWORK_ATTEMPTS - 1
            ):
                _backoff(attempts, destination.name)
                attempts += 1
                continue
            raise InstallError(
                f"{destination.name}: artifact download failed with HTTP "
                f"{exc.code} after {attempts + 1} attempt(s): "
                f"{_http_detail(body)}. Completed downloads are preserved; "
                "rerun the installer to resume."
            ) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ) as exc:
            if attempts < MAX_NETWORK_ATTEMPTS - 1:
                _backoff(attempts, destination.name)
                attempts += 1
                continue
            reason = getattr(exc, "reason", exc)
            raise InstallError(
                f"{destination.name}: download service is unreachable after "
                f"{MAX_NETWORK_ATTEMPTS} attempts: {reason}. Completed downloads "
                "are preserved; rerun the installer to resume."
            ) from exc

        status = getattr(response, "status", response.getcode())
        if start and status != 206:
            response.close()
            partial.unlink(missing_ok=True)
            continue
        mode = "ab" if start else "wb"
        downloaded = start
        last_report = 0.0
        try:
            with response, partial.open(mode) as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 5:
                        percent = min(100.0, downloaded * 100 / size)
                        print(
                            f"  {destination.name}: {percent:.1f}% "
                            f"({downloaded}/{size} bytes)",
                            flush=True,
                        )
                        last_report = now
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ) as exc:
            if attempts < MAX_NETWORK_ATTEMPTS - 1:
                _backoff(attempts, destination.name)
                attempts += 1
                continue
            raise InstallError(
                f"{destination.name}: connection failed after "
                f"{MAX_NETWORK_ATTEMPTS} attempts. The partial download is "
                "preserved; rerun the installer to resume."
            ) from exc

        actual_size = partial.stat().st_size
        if actual_size == size:
            break
        if actual_size > size:
            partial.unlink()
            raise InstallError(
                f"{destination.name} exceeded its declared size; partial file removed"
            )
        if attempts < MAX_NETWORK_ATTEMPTS - 1:
            _backoff(attempts, destination.name)
            attempts += 1
            continue
        raise InstallError(
            f"{destination.name} stopped at {actual_size}/{size} bytes after "
            f"{MAX_NETWORK_ATTEMPTS} attempts. The partial download is preserved; "
            "rerun the installer to resume."
        )
    else:
        raise InstallError(
            f"{destination.name} could not be downloaded after "
            f"{MAX_NETWORK_ATTEMPTS} attempts. Completed data is preserved; "
            "rerun the installer to resume."
        )

    actual_size = partial.stat().st_size
    if actual_size != size:
        raise InstallError(
            f"{destination.name} stopped at {actual_size}/{size} bytes; rerun to resume"
        )
    actual_sha256 = _sha256(partial)
    if actual_sha256 != sha256:
        partial.unlink(missing_ok=True)
        raise InstallError(
            f"{destination.name} failed SHA-256 verification; partial file removed"
        )
    partial.replace(destination)
    return "downloaded and verified"


def _auth(args: argparse.Namespace) -> None:
    passcode = sys.stdin.read().rstrip("\r\n")
    if not passcode:
        raise InstallError("no group passcode was supplied")
    manager = AccessManager(relay_url=args.relay_url)
    ok, message, _offline = manager.authenticate(passcode)
    passcode = ""
    if not ok:
        raise InstallError(message)
    state = manager.store.load()
    if not state.authenticated_once or not state.token:
        raise InstallError("relay authentication did not produce a reusable token")
    print("AUTH_OK credential stored through PatchLab's access store")


def _auth_status(_args: argparse.Namespace) -> None:
    state = AccessStore().load()
    if not state.authenticated_once:
        raise InstallError("no prior successful group authentication")
    print("AUTH_STATUS authenticated_once=true")


def _artifact_token() -> str:
    state = AccessStore().load()
    if not state.authenticated_once or not state.token:
        raise InstallError("authenticate before downloading private artifacts")
    return state.token


def _artifact_manifest(relay_url: str) -> tuple[str, list[dict]]:
    token = _artifact_token()
    for attempt in range(2):
        status, _headers, body = _small_request(
            relay_url.rstrip("/") + "/artifacts",
            label="private artifact manifest",
            token=token,
        )
        if status == 200:
            manifest = json.loads(body)
            break
        if status == 401:
            passcode = AccessStore().passcode()
            if attempt or not passcode:
                raise InstallError(
                    "group authentication expired; rerun and enter the passcode"
                )
            manager = AccessManager(relay_url=relay_url)
            ok, message, _offline = manager.authenticate(passcode)
            passcode = ""
            if not ok:
                raise InstallError(message)
            token = _artifact_token()
            continue
        raise InstallError(
            f"private artifact manifest failed with HTTP {status}: "
            f"{_http_detail(body)}. Completed downloads are preserved; rerun "
            "the installer to resume."
        )

    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise InstallError("relay returned an empty or invalid artifact manifest")
    return token, rows


def _artifacts_preflight(args: argparse.Namespace) -> None:
    token, rows = _artifact_manifest(args.relay_url)
    for row in rows:
        name = str(row["name"])
        url = (
            args.relay_url.rstrip("/")
            + "/artifacts/"
            + urllib.parse.quote(name, safe="")
        )
        status, headers, body = _small_request(
            url,
            label=f"private artifact {name}",
            token=token,
            byte_range="bytes=0-0",
        )
        if (
            status != 206
            or len(body) != 1
            or not headers.get("content-range", "").startswith("bytes 0-0/")
        ):
            raise InstallError(
                f"{name} is not retrievable from private storage (HTTP "
                f"{status}: {_http_detail(body)}). The large public CLAP "
                "download has not started, and any completed data is preserved. "
                "Retry later or contact the PatchLab operator."
            )
        print(f"ARTIFACT_PREFLIGHT_OK name={name} status=206 bytes=1")
    print(f"ARTIFACT_PREFLIGHT_PASS count={len(rows)}")


def _extract_tar_gz(archive: Path, target: Path) -> int:
    """Extract `archive` into `target`, refusing anything that escapes it.

    The archive arrives over the network, so its member names are untrusted
    input: a member named ../../x or an absolute path would otherwise write
    outside the install root. Links are rejected outright rather than resolved,
    since a symlink can redirect a later member to an arbitrary location.
    """

    target.mkdir(parents=True, exist_ok=True)
    resolved_target = target.resolve()
    extracted = 0
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise InstallError(
                    f"{archive.name}: refusing archive containing link {member.name!r}"
                )
            if not (member.isfile() or member.isdir()):
                raise InstallError(
                    f"{archive.name}: refusing archive containing special file "
                    f"{member.name!r}"
                )
            candidate = (resolved_target / member.name).resolve()
            if candidate != resolved_target and resolved_target not in candidate.parents:
                raise InstallError(
                    f"{archive.name}: refusing member {member.name!r} that escapes "
                    f"{resolved_target}"
                )
            extracted += 1
        bundle.extractall(resolved_target)
    return extracted


def _artifacts(args: argparse.Namespace) -> None:
    token, rows = _artifact_manifest(args.relay_url)
    root = args.install_root.resolve()
    completed = 0
    total = 0
    for row in rows:
        name = str(row["name"])
        size = int(row["size"])
        sha256 = str(row["sha256"])
        unpack = str(row.get("unpack") or "")
        # Older relay manifests identify archive artifacts by their filename and
        # directory destination but omit the newer explicit ``unpack`` field.
        # Treat a .tar.gz artifact aimed at a directory as the legacy spelling
        # of extract-tar-gz.  Without this compatibility path the downloader
        # tries to checksum the existing destination directory as a file.
        if not unpack and name.casefold().endswith(".tar.gz"):
            unpack = "extract-tar-gz"
        relative = Path(str(row["destination"]))
        # An unpacked artifact names the directory it expands into; a plain one
        # names the file itself. Download archives beside their target so the
        # checksum is verified before anything is written into the tree.
        destination = (
            (root / relative / name) if unpack else (root / relative)
        ).resolve()
        if root not in destination.parents:
            raise InstallError(f"manifest destination escapes install root: {name}")
        print(f"Private artifact: {name} ({size} bytes)")
        result = _download(
            args.relay_url.rstrip("/")
            + "/artifacts/"
            + urllib.parse.quote(name, safe=""),
            destination,
            size=size,
            sha256=sha256,
            token=token,
        )
        print(f"  {result}")
        if unpack:
            if unpack != "extract-tar-gz":
                raise InstallError(f"{name}: unsupported unpack mode {unpack!r}")
            count = _extract_tar_gz(destination, (root / relative).resolve())
            destination.unlink()
            print(f"  extracted {count} entries into {relative}")
        completed += 1
        total += size
    print(f"ARTIFACTS_OK count={completed} bytes={total}")


def _clap(args: argparse.Namespace) -> None:
    url = CLAP_URL
    size = CLAP_SIZE
    sha256 = CLAP_SHA256
    if os.environ.get("PATCHLAB_INSTALL_TEST_MODE") == "1":
        url = os.environ.get("PATCHLAB_CLAP_URL", url)
        size = int(os.environ.get("PATCHLAB_CLAP_SIZE", size))
        sha256 = os.environ.get("PATCHLAB_CLAP_SHA256", sha256)

    destination = args.install_root / "data" / "models" / CLAP_NAME
    if _verified(destination, size=size, sha256=sha256):
        print(f"CLAP_OK already verified at {destination}")
        return
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if cache.is_dir():
        for candidate in cache.glob(f"models--lukewys--laion_clap/**/{CLAP_NAME}"):
            if _verified(candidate, size=size, sha256=sha256):
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(candidate, destination)
                except OSError:
                    shutil.copy2(candidate, destination)
                print(f"CLAP_OK reused verified Hugging Face cache file at {candidate}")
                return
    print(f"Public CLAP checkpoint: {size} bytes")
    result = _download(
        url,
        destination,
        size=size,
        sha256=sha256,
    )
    print(f"CLAP_OK {result}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    auth = subparsers.add_parser("auth")
    auth.add_argument("--relay-url", required=True)
    auth.set_defaults(handler=_auth)
    auth_status = subparsers.add_parser("auth-status")
    auth_status.set_defaults(handler=_auth_status)
    artifacts = subparsers.add_parser("artifacts")
    artifacts.add_argument("--relay-url", required=True)
    artifacts.add_argument("--install-root", type=Path, required=True)
    artifacts.set_defaults(handler=_artifacts)
    artifact_preflight = subparsers.add_parser("artifacts-preflight")
    artifact_preflight.add_argument("--relay-url", required=True)
    artifact_preflight.set_defaults(handler=_artifacts_preflight)
    clap = subparsers.add_parser("clap")
    clap.add_argument("--install-root", type=Path, required=True)
    clap.set_defaults(handler=_clap)
    args = parser.parse_args()
    try:
        args.handler(args)
        return 0
    except (InstallError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"PatchLab install error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
