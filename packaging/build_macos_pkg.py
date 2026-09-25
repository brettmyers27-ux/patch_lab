#!/usr/bin/env python3
"""Build PatchLab.app and wrap it in a standard macOS Installer package.

The release artifact is intentionally local and ignored by Git.  It contains
the licensed runtime artifacts already selected by packaging/patchlab.spec;
never attach it to a public source release.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "patchlab.spec"
IDENTIFIER = "com.patchlab.desktop"
INSTALL_LOCATION = "/Applications"
INSTALLER_SCRIPTS = ROOT / "packaging" / "macos-installer-scripts"


class PackageBuildError(RuntimeError):
    """Raised when a release artifact cannot be created safely."""


def read_version() -> str:
    namespace: dict[str, str] = {}
    exec((ROOT / "app" / "__version__.py").read_text(encoding="utf-8"), namespace)
    return str(namespace["__version__"])


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise PackageBuildError(
            f"{name} is required. Install Apple's Command Line Tools, then retry."
        )


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
    )
    if completed.returncode:
        raise PackageBuildError(
            f"{' '.join(command)} exited with status {completed.returncode}"
        )


def _validate_app(app: Path, version: str) -> None:
    info_path = app / "Contents" / "Info.plist"
    executable = app / "Contents" / "MacOS" / "PatchLab"
    if not app.is_dir() or not info_path.is_file() or not executable.is_file():
        raise PackageBuildError("PyInstaller did not produce a complete PatchLab.app")
    if not os.access(executable, os.X_OK):
        raise PackageBuildError("PatchLab.app executable is not marked executable")
    with info_path.open("rb") as handle:
        info = plistlib.load(handle)
    if info.get("CFBundleIdentifier") != IDENTIFIER:
        raise PackageBuildError("PatchLab.app has an unexpected bundle identifier")
    if str(info.get("CFBundleShortVersionString")) != version:
        raise PackageBuildError("PatchLab.app version does not match app/__version__.py")
    # Per-user state must remain in Application Support, never inside the app
    # payload that macOS Installer replaces during an upgrade.
    forbidden = (
        "library.db",
        "privacy-settings.json",
        "access-state.json",
        "match_library",
    )
    found = [name for name in forbidden if any(app.rglob(name))]
    if found:
        raise PackageBuildError(
            "The app bundle unexpectedly contains user data: " + ", ".join(found)
        )


def _component_plist(payload_root: Path, work_root: Path) -> Path:
    """Describe the payload's bundles, with bundle relocation switched off.

    pkgbuild marks an app bundle relocatable by default.  macOS Installer then
    asks LaunchServices where it last saw ``com.patchlab.desktop`` and shoves
    the payload *there* instead of ``install-location``.  A tester with a copy
    of PatchLab.app anywhere else -- ~/Applications, Downloads, a second
    checkout -- therefore gets the app installed somewhere unexpected, and the
    postinstall verification of /Applications/PatchLab.app correctly fails the
    whole install.  PatchLab has exactly one supported location, so relocation
    is never wanted.
    """

    plist_path = work_root / "component.plist"
    _run(["pkgbuild", "--analyze", "--root", str(payload_root), str(plist_path)])
    with plist_path.open("rb") as handle:
        components = plistlib.load(handle)
    if not isinstance(components, list) or not components:
        raise PackageBuildError("pkgbuild could not describe the PatchLab.app payload")
    for component in components:
        component["BundleIsRelocatable"] = False
    if not any(
        component.get("RootRelativeBundlePath") == "PatchLab.app"
        for component in components
    ):
        raise PackageBuildError(
            "The analyzed payload does not contain PatchLab.app as a bundle"
        )
    with plist_path.open("wb") as handle:
        plistlib.dump(components, handle)
    return plist_path


def _assert_component_contract(component: Path, work_root: Path, version: str) -> None:
    """Read back the built component and refuse to ship a wrong install plan."""

    extracted = work_root / "component-check"
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True)
    _run(["xar", "-x", "-C", str(extracted), "-f", str(component), "PackageInfo"])
    package_info = (extracted / "PackageInfo").read_text(encoding="utf-8")
    root = ET.fromstring(package_info)
    # A non-relocatable build still carries an empty <relocate/>; what must
    # never appear is a bundle listed inside it.
    relocate = root.find("relocate")
    if relocate is not None and len(relocate):
        raise PackageBuildError(
            "The component package still allows bundle relocation; macOS Installer "
            "could install PatchLab outside /Applications"
        )
    if root.attrib.get("install-location") != INSTALL_LOCATION:
        raise PackageBuildError(
            f"The component package does not install to {INSTALL_LOCATION}"
        )
    if root.attrib.get("identifier") != IDENTIFIER or root.attrib.get("version") != version:
        raise PackageBuildError("The component package identity does not match this build")


def build_pkg(
    app: Path,
    *,
    destination: Path,
    version: str,
    work_root: Path,
) -> Path:
    """Create a flat PKG that installs exactly one bundle in /Applications."""

    _require_tool("pkgbuild")
    _require_tool("productbuild")
    _require_tool("xar")
    if not (INSTALLER_SCRIPTS / "preinstall").is_file() or not (
        INSTALLER_SCRIPTS / "postinstall"
    ).is_file():
        raise PackageBuildError("The macOS replacement installer scripts are missing")
    _validate_app(app, version)
    work_root.mkdir(parents=True, exist_ok=True)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    component = work_root / "PatchLab-component.pkg"
    temporary = destination.with_name(f".{destination.name}.building")
    temporary.unlink(missing_ok=True)
    # Package a deliberately clean staging root. The resulting payload contains
    # PatchLab.app only (plus any Apple filesystem metadata generated by
    # pkgbuild itself), never build-machine or per-user PatchLab data.
    payload_root = work_root / "payload-root"
    payload_app = payload_root / "PatchLab.app"
    if payload_root.exists():
        shutil.rmtree(payload_root)
    shutil.copytree(app, payload_app, symlinks=True)
    package_environment = dict(os.environ, COPYFILE_DISABLE="1")
    _run(
        [
            "pkgbuild",
            "--root",
            str(payload_root),
            "--component-plist",
            str(_component_plist(payload_root, work_root)),
            "--install-location",
            INSTALL_LOCATION,
            "--identifier",
            IDENTIFIER,
            "--version",
            version,
            "--scripts",
            str(INSTALLER_SCRIPTS),
            str(component),
        ],
        environment=package_environment,
    )
    _assert_component_contract(component, work_root, version)
    _run(
        ["productbuild", "--package", str(component), str(temporary)],
        environment=package_environment,
    )
    temporary.replace(destination)
    return destination


def freeze_app(*, work_root: Path, allow_dirty: bool) -> Path:
    """Build an arm64 PyInstaller app in isolated temporary directories."""

    _require_tool("xcrun")
    environment = dict(os.environ)
    # PyInstaller imports optional Hugging Face modules while analyzing the
    # application. A release build must use the same bundled, offline model
    # contract as the frozen app and must never turn a local build into a
    # network retry because a tokenizer metadata probe is unavailable.
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    # PyInstaller otherwise cleans its shared per-user cache before a build.
    # A stale cache created by another macOS sandboxed process can be
    # undeletable, aborting a perfectly valid release build before it reaches
    # PatchLab. Keep every cache file in this disposable build directory.
    environment["PYINSTALLER_CONFIG_DIR"] = str(work_root / "pyinstaller-cache")
    if allow_dirty:
        environment["PATCHLAB_ALLOW_DIRTY_BUILD"] = "1"
    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            "--distpath",
            str(work_root / "dist"),
            "--workpath",
            str(work_root / "pyinstaller-work"),
            str(SPEC),
        ],
        environment=environment,
    )
    app = work_root / "dist" / "PatchLab.app"
    architecture = subprocess.check_output(
        ["lipo", "-archs", str(app / "Contents" / "MacOS" / "PatchLab")],
        text=True,
    ).strip()
    if "arm64" not in architecture.split():
        raise PackageBuildError(
            f"PatchLab.app is not an Apple Silicon build (architectures: {architecture})"
        )
    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build PatchLab-<version>-macOS.pkg without modifying user data."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "release", help="PKG destination"
    )
    parser.add_argument(
        "--app",
        type=Path,
        help="Package this prebuilt PatchLab.app instead of running PyInstaller",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Diagnostic only; release builds normally require committed source",
    )
    parser.add_argument(
        "--keep-app",
        action="store_true",
        help="Also copy the finished PatchLab.app beside the PKG",
    )
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise PackageBuildError("PatchLab PKGs can only be built on macOS")
    if __import__("platform").machine() != "arm64":
        raise PackageBuildError("Build this PKG on an Apple Silicon Mac")

    version = read_version()
    output = args.output_dir.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="patchlab-pkg-build-") as temporary:
        work_root = Path(temporary)
        app = (
            args.app.expanduser().resolve()
            if args.app is not None
            else freeze_app(work_root=work_root, allow_dirty=args.allow_dirty)
        )
        artifact = build_pkg(
            app,
            destination=output / f"PatchLab-{version}-macOS.pkg",
            version=version,
            work_root=work_root,
        )
        if args.keep_app:
            target_app = output / "PatchLab.app"
            if target_app.exists():
                shutil.rmtree(target_app)
            shutil.copytree(app, target_app, symlinks=True)
    print(f"PATCHLAB_PKG={artifact}")
    print("PATCHLAB_PKG_INSTALL_LOCATION=/Applications")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackageBuildError as exc:
        print(f"PATCHLAB_PKG_ERROR={exc}", file=sys.stderr)
        raise SystemExit(1)
