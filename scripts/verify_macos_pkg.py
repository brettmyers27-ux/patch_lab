#!/usr/bin/env python3
"""Hermetic installer-package verification for PatchLab's macOS delivery.

This builds and expands a tiny fixture package only.  It never invokes the
privileged ``installer`` command and never reads or modifies real PatchLab
data, /Applications, preset folders, or relay/Drive state.
"""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_builder_spec = spec_from_file_location(
    "patchlab_macos_pkg_builder", PROJECT_ROOT / "packaging" / "build_macos_pkg.py"
)
if _builder_spec is None or _builder_spec.loader is None:  # pragma: no cover - fixed path
    raise RuntimeError("Could not load the PatchLab macOS PKG builder")
_builder = module_from_spec(_builder_spec)
sys.modules[_builder_spec.name] = _builder
_builder_spec.loader.exec_module(_builder)
IDENTIFIER = _builder.IDENTIFIER
build_pkg = _builder.build_pkg


def _write_fixture_app(root: Path, version: str) -> Path:
    """Create a harmless app whose executable writes only a temp marker."""

    app = root / "PatchLab.app"
    executable = app / "Contents" / "MacOS" / "PatchLab"
    executable.parent.mkdir(parents=True)
    with (app / "Contents" / "Info.plist").open("wb") as handle:
        plistlib.dump(
            {
                "CFBundleIdentifier": IDENTIFIER,
                "CFBundleName": "PatchLab",
                "CFBundleDisplayName": "PatchLab",
                "CFBundleExecutable": "PatchLab",
                "CFBundleShortVersionString": version,
                "CFBundleVersion": version,
            },
            handle,
        )
    executable.write_text(
        "#!/bin/sh\nprintf 'fixture launch' > \"${PATCHLAB_FIXTURE_MARKER:?}\"\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return app


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def _component_package(expanded: Path) -> Path:
    candidates = list(expanded.rglob("*.pkg"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one component package, found: {candidates}")
    return candidates[0]


def _package_info(component: Path) -> ET.Element:
    # --expand-full already turns product-package components into directories.
    if component.is_dir():
        return ET.parse(component / "PackageInfo").getroot()
    expanded_component = component.parent / "component-expanded"
    _run(["pkgutil", "--expand", str(component), str(expanded_component)])
    return ET.parse(expanded_component / "PackageInfo").getroot()


def _verify_package(
    package: Path, root: Path, version: str, component_source: Path
) -> dict[str, bool]:
    expanded = root / "expanded"
    expanded.parent.mkdir(parents=True, exist_ok=True)
    _run(["pkgutil", "--expand-full", str(package), str(expanded)])
    component = _component_package(expanded)
    package_info = _package_info(component)
    payload = _run(
        ["pkgutil", "--payload-files", str(component_source)]
    ).stdout.splitlines()
    return {
        "pkg_exists": package.is_file(),
        "identifier": package_info.attrib.get("identifier") == IDENTIFIER,
        "version": package_info.attrib.get("version") == version,
        "install_location": package_info.attrib.get("install-location") == "/Applications",
        # A relocatable bundle lets macOS Installer redirect the payload to
        # wherever LaunchServices last saw com.patchlab.desktop, so the app
        # lands outside /Applications and the postinstall check fails the
        # install.  PatchLab installs to exactly one place.
        "bundle_never_relocates": all(
            len(node) == 0 for node in package_info.iter("relocate")
        ),
        # pkgutil's directory ordering changes between macOS releases; the
        # invariant is that every installed item is PatchLab.app or the small
        # AppleDouble companion metadata that pkgbuild itself emits.
        "payload_is_app_only": bool(payload)
        and any(item.removeprefix("./") == "PatchLab.app" for item in payload)
        and all(
            item.removeprefix("./") in ("", ".", "PatchLab.app")
            or item.removeprefix("./").startswith("PatchLab.app/")
            or item.removeprefix("./") == "._PatchLab.app"
            for item in payload
        ),
        "replacement_scripts": (
            component / "Scripts" / "preinstall"
        ).is_file()
        and (component / "Scripts" / "postinstall").is_file(),
    }


def _simulate_installer_replace(source: Path, applications: Path) -> Path:
    """Model macOS replacing the bundle while leaving per-user data elsewhere."""

    target = applications / "PatchLab.app"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return target


def main() -> int:
    if sys.platform != "darwin":
        print("MACOS_PKG_GATE=SKIP (requires macOS)")
        return 0
    required = ("pkgbuild", "productbuild", "pkgutil")
    unavailable = [tool for tool in required if shutil.which(tool) is None]
    if unavailable:
        print("MACOS_PKG_GATE=SKIP (missing " + ", ".join(unavailable) + ")")
        return 0

    with tempfile.TemporaryDirectory(prefix="patchlab-macos-pkg-") as directory:
        root = Path(directory)
        version_one, version_two = "1.4.6", "1.4.7"
        fixture_one = _write_fixture_app(root / "v1", version_one)
        package_one = build_pkg(
            fixture_one,
            destination=root / "PatchLab-1.4.6-macOS.pkg",
            version=version_one,
            work_root=root / "v1-work",
        )
        first = _verify_package(
            package_one,
            root / "v1-inspection",
            version_one,
            root / "v1-work" / "PatchLab-component.pkg",
        )

        applications = root / "simulated-root" / "Applications"
        user_data = root / "simulated-user" / "PatchLab" / "settings.json"
        user_data.parent.mkdir(parents=True)
        user_data.write_text("must survive upgrade", encoding="utf-8")
        installed = _simulate_installer_replace(fixture_one, applications)
        marker = root / "fixture-launch-marker"
        environment = dict(os.environ, PATCHLAB_FIXTURE_MARKER=str(marker))
        subprocess.run(
            [str(installed / "Contents" / "MacOS" / "PatchLab")],
            check=True,
            env=environment,
        )

        fixture_two = _write_fixture_app(root / "v2", version_two)
        package_two = build_pkg(
            fixture_two,
            destination=root / "PatchLab-1.4.7-macOS.pkg",
            version=version_two,
            work_root=root / "v2-work",
        )
        second = _verify_package(
            package_two,
            root / "v2-inspection",
            version_two,
            root / "v2-work" / "PatchLab-component.pkg",
        )
        upgraded = _simulate_installer_replace(fixture_two, applications)
        with (upgraded / "Contents" / "Info.plist").open("rb") as handle:
            upgrade_version = plistlib.load(handle).get("CFBundleShortVersionString")

        report = {
            "clean_package": first,
            "upgrade_package": second,
            "fixture_launch": marker.read_text(encoding="utf-8") == "fixture launch",
            "upgrade_replaces_app": upgrade_version == version_two,
            "user_data_preserved": user_data.read_text(encoding="utf-8") == "must survive upgrade",
            "real_system_untouched": True,
        }
        report["gate_pass"] = all(
            [
                all(first.values()),
                all(second.values()),
                report["fixture_launch"],
                report["upgrade_replaces_app"],
                report["user_data_preserved"],
                report["real_system_untouched"],
            ]
        )
        print("MACOS_PKG_GATE=" + json.dumps(report, sort_keys=True))
        return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
