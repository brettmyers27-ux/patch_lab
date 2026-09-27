#!/usr/bin/env python3
"""Provision the reproducible Python build runtime used by macOS packaging."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from platform_compatibility import (
    MACOS_MINIMUM,
    PYTHON_RUNTIME_ARCHIVE,
    PYTHON_RUNTIME_SHA256,
    PYTHON_RUNTIME_URL,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provision(cache: Path) -> Path:
    """Download and verify the build-only Python runtime, returning its interpreter."""

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / PYTHON_RUNTIME_ARCHIVE
    runtime = cache / "python"
    if not archive.is_file() or _sha256(archive) != PYTHON_RUNTIME_SHA256:
        temporary = archive.with_suffix(".partial")
        temporary.unlink(missing_ok=True)
        urllib.request.urlretrieve(PYTHON_RUNTIME_URL, temporary)
        if _sha256(temporary) != PYTHON_RUNTIME_SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("downloaded Python runtime checksum did not match")
        temporary.replace(archive)
    python = runtime / "bin" / "python3.11"
    if not python.is_file():
        if runtime.exists():
            shutil.rmtree(runtime)
        with tarfile.open(archive, "r:gz") as package:
            package.extractall(cache, filter="data")
    if not python.is_file():
        raise RuntimeError("portable Python archive did not contain python3.11")
    return python


def build_python(cache: Path, requirements: Path) -> Path:
    """Return an isolated, pinned packaging interpreter built from the runtime."""

    runtime = provision(cache / "runtime")
    venv = cache / "venv"
    python = venv / "bin" / "python"
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = venv / ".patchlab-requirements-sha256"
    if not python.is_file() or not marker.is_file() or marker.read_text().strip() != fingerprint:
        if venv.exists():
            shutil.rmtree(venv)
        environment = dict(os.environ, MACOSX_DEPLOYMENT_TARGET=MACOS_MINIMUM)
        subprocess.run([str(runtime), "-m", "venv", str(venv)], check=True, env=environment)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            env=environment,
            stdout=sys.stderr,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(requirements)],
            check=True,
            env=environment,
            stdout=sys.stderr,
        )
        marker.write_text(fingerprint + "\n", encoding="utf-8")
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--requirements", type=Path)
    args = parser.parse_args()
    cache = args.cache.expanduser().resolve()
    print(build_python(cache, args.requirements.resolve()) if args.requirements else provision(cache))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
