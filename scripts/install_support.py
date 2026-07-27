#!/usr/bin/env python3
"""Non-interactive primitives used by install.sh.

Secrets are accepted on stdin, never as command-line arguments. All failures
become one-line actionable messages rather than tracebacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
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

    while True:
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
                    "artifact authorization expired or was rejected; rerun the installer"
                ) from exc
            raise InstallError(f"download failed with HTTP {exc.code}: {url}") from exc
        except urllib.error.URLError as exc:
            raise InstallError(f"download service is unreachable: {exc.reason}") from exc

        status = getattr(response, "status", response.getcode())
        if start and status != 206:
            response.close()
            partial.unlink(missing_ok=True)
            continue
        mode = "ab" if start else "wb"
        downloaded = start
        last_report = 0.0
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
        break

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
    passcode = sys.stdin.read()
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


def _artifacts(args: argparse.Namespace) -> None:
    token = _artifact_token()
    for attempt in range(2):
        request = urllib.request.Request(
            args.relay_url.rstrip("/") + "/artifacts",
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                manifest = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise InstallError(
                    f"artifact manifest failed with HTTP {exc.code}"
                ) from exc
            passcode = AccessStore().passcode()
            if attempt or not passcode:
                raise InstallError(
                    "group authentication expired; rerun and enter the passcode"
                ) from exc
            manager = AccessManager(relay_url=args.relay_url)
            ok, message, _offline = manager.authenticate(passcode)
            passcode = ""
            if not ok:
                raise InstallError(message)
            token = _artifact_token()
        except urllib.error.URLError as exc:
            raise InstallError(f"relay is unreachable: {exc.reason}") from exc

    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise InstallError("relay returned an empty or invalid artifact manifest")
    root = args.install_root.resolve()
    completed = 0
    total = 0
    for row in rows:
        name = str(row["name"])
        size = int(row["size"])
        sha256 = str(row["sha256"])
        destination = (root / str(row["destination"])).resolve()
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
