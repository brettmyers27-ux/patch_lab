# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files


ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT))
VERSION = {}
exec((ROOT / "app" / "__version__.py").read_text(), VERSION)

datas = [
    (str(ROOT / "app" / "theme.qss"), "app"),
    (str(ROOT / "app" / "icons"), "app/icons"),
]
for source, destination in (
    (ROOT / "data" / "dist" / "factory_bundle.sqlite", "data/dist"),
    (
        ROOT / "data" / "models" / "music_audioset_epoch_15_esc_90.14.pt",
        "data/models",
    ),
):
    if source.exists():
        datas.append((str(source), destination))

binaries = []
hiddenimports = []
for package in ("dawdreamer", "laion_clap"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden
# Numba's cached librosa kernels require a real source-file locator. Keeping
# the .py files beside the frozen modules avoids "no locator available" at
# first match while preserving JIT performance.
datas += collect_data_files("librosa", include_py_files=True)
hiddenimports += [
    entry.module
    for entry in __import__(
        "core.worker_runtime", fromlist=["WORKER_ENTRY_POINTS"]
    ).WORKER_ENTRY_POINTS.values()
]

analysis = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(ROOT / "packaging" / "runtime_distribution.py")],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="PatchLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "app" / "icons" / "PatchLab.icns"),
)
collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="PatchLab",
)
app = BUNDLE(
    collection,
    name="PatchLab.app",
    icon=str(ROOT / "app" / "icons" / "PatchLab.icns"),
    bundle_identifier="com.patchlab.desktop",
    version=VERSION["__version__"],
    info_plist={
        "CFBundleDisplayName": "PatchLab",
        "CFBundleName": "PatchLab",
        "CFBundleShortVersionString": VERSION["__version__"],
        "CFBundleVersion": VERSION["__version__"],
        "NSHighResolutionCapable": True,
    },
)
