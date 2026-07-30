# -*- mode: python ; coding: utf-8 -*-
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files


ROOT = Path(SPECPATH).parent
sys.path.insert(0, str(ROOT))
VERSION = {}
exec((ROOT / "app" / "__version__.py").read_text(), VERSION)

SOURCE_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    text=True,
).strip()
TRACKED_CHANGES = subprocess.check_output(
    ["git", "status", "--porcelain", "--untracked-files=no"],
    cwd=ROOT,
    text=True,
).strip()
if TRACKED_CHANGES and os.environ.get("PATCHLAB_ALLOW_DIRTY_BUILD") != "1":
    raise RuntimeError(
        "Refusing to build PatchLab.app from uncommitted tracked source. "
        "Commit the changes first, or set PATCHLAB_ALLOW_DIRTY_BUILD=1 for "
        "a clearly marked non-release diagnostic build."
    )
build_info_path = ROOT / "build" / "patchlab-build-info.json"
build_info_path.parent.mkdir(parents=True, exist_ok=True)
build_info_path.write_text(
    json.dumps(
        {
            "source_commit": SOURCE_COMMIT,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_dirty": bool(TRACKED_CHANGES),
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

datas = [
    (str(ROOT / "app" / "theme.qss"), "app"),
    (str(ROOT / "app" / "icons"), "app/icons"),
    (str(build_info_path), "."),
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
hf_cache = ROOT / "data" / "models" / "huggingface"
if hf_cache.is_dir():
    # Bundle only the pinned runtime snapshots—not download locks or local
    # Hugging Face/Xet logs, which can contain machine-specific diagnostics.
    required_model_caches = __import__(
        "core.model_assets",
        fromlist=["TOKENIZER_REQUIREMENTS"],
    ).TOKENIZER_REQUIREMENTS
    for model_cache_name in required_model_caches:
        model_cache = hf_cache / "transformers" / model_cache_name
        if model_cache.is_dir():
            datas.append(
                (
                    str(model_cache),
                    f"data/models/huggingface/transformers/{model_cache_name}",
                )
            )
    transformers_version = hf_cache / "transformers" / "version.txt"
    if transformers_version.is_file():
        datas.append(
            (
                str(transformers_version),
                "data/models/huggingface/transformers",
            )
        )

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
        "PatchLabSourceCommit": SOURCE_COMMIT,
    },
)
