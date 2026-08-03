#!/usr/bin/env python3
"""Map catalog preset hashes to their local files on this machine.

The synthesis catalog stores the absolute preset path from the machine that
built it, so on any other computer those paths dangle. Stage 2's Phase 4 could
not rebuild the embedding world because 4,412 Serum 1 presets were unreachable
for exactly that reason.

Given the transferred preset package (a `presets/` folder of hash-named files
plus `manifest.json`), this writes a mapping in the same `local_paths_by_hash`
shape that `core/matcher._factory_paths_by_hash` and
`core/serum2_preset_writer._factory_path_for_hash` already consume — so no
resolution code has to change, and the factory mapping this machine already
generated is merged in rather than replaced.

Usage:
    python scripts/build_preset_path_map.py --package D:/PatchLab-Stage2-Presets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.platform_env import ENV  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        type=Path,
        required=True,
        help="Transferred preset package containing manifest.json and presets/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to the app data directory's preset-paths.json",
    )
    parser.add_argument(
        "--merge-factory",
        action="store_true",
        default=True,
        help="Also carry over this machine's existing factory-paths.json entries",
    )
    args = parser.parse_args()

    package = args.package.expanduser().resolve()
    manifest_path = package / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"No manifest.json in {package}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mapping: dict[str, str] = {}
    missing: list[str] = []
    for content_hash, entry in manifest.get("presets_by_hash", {}).items():
        candidate = package / str(entry["relative_path"])
        if candidate.is_file():
            mapping[str(content_hash)] = str(candidate)
        else:
            missing.append(str(content_hash))

    merged_from_factory = 0
    if args.merge_factory:
        factory = ENV.app_data_dir / "factory-paths.json"
        if factory.is_file():
            try:
                raw = json.loads(factory.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raw = {}
            for content_hash, value in raw.get("local_paths_by_hash", {}).items():
                # A locally installed factory preset is preferable to the
                # transferred copy: it is the file this machine's Serum
                # actually browses.
                if Path(str(value)).is_file():
                    mapping[str(content_hash)] = str(value)
                    merged_from_factory += 1

    output = args.output or (ENV.app_data_dir / "preset-paths.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_package": str(package),
                "local_paths_by_hash": mapping,
                "unresolved_hashes": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"PRESET_PATH_MAP={output}")
    print(f"resolved={len(mapping)} from_factory={merged_from_factory} missing={len(missing)}")
    if missing:
        print(f"first missing hashes: {missing[:5]}")
    print()
    print("Point the resolver at it for this session with:")
    print(f'  set PATCHLAB_FACTORY_MAPPING={output}')
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
