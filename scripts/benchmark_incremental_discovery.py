#!/usr/bin/env python3
"""Benchmark only filesystem discovery and report content-hash counts."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import Database
from core.library_state import reconcile_source_tree
from core.preset_scan import sha1_file


def _run(size: int) -> dict[str, dict[str, float | int]]:
    with tempfile.TemporaryDirectory(prefix=f"patchlab-discovery-{size}-") as temporary:
        base = Path(temporary)
        root = base / "presets"
        root.mkdir()
        for index in range(size):
            (root / f"p{index:05d}.fxp").write_bytes(f"preset-{index}".encode())
        database = Database(base / "library.db")

        def measure(label: str) -> tuple[str, dict[str, float | int]]:
            hashes = 0

            def counted(path: Path) -> str:
                nonlocal hashes
                hashes += 1
                return sha1_file(path)

            started = time.perf_counter()
            result = reconcile_source_tree(root, database, hash_file=counted)
            return label, {
                "seconds": round(time.perf_counter() - started, 6),
                "hashes": hashes,
                "files": len(result.entries),
            }

        results = dict([measure("first_scan"), measure("unchanged_second_scan")])
        (root / "added-one.fxp").write_bytes(b"added-one")
        results.update([measure("add_one")])
        for index in range(100):
            (root / f"added-{index:03d}.fxp").write_bytes(f"added-{index}".encode())
        results.update([measure("add_one_hundred")])
        changed = root / "p00000.fxp"
        changed.write_bytes(b"modified-content-with-different-size")
        results.update([measure("modify_one")])
        (root / "p00001.fxp").unlink()
        results.update([measure("remove_one")])
        (root / "p00002.fxp").rename(root / "renamed-p00002.fxp")
        results.update([measure("rename_one")])
        return results


def main() -> None:
    report = {str(size): _run(size) for size in (100, 1_000, 5_000)}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
