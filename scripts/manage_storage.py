#!/usr/bin/env python3
"""Background storage migration and compact-cleanup worker."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.platform_env import ENV
from core.storage import (
    adopt_legacy_renders,
    compact_render_library,
    configured_audio_root,
    migrate_audio_storage,
    prepare_audio_root,
    preview_cache_root,
    prune_preview_cache,
)


def main() -> int:
    defaults = {
        "db": ENV.app_data_dir / "library.db",
        "audio": configured_audio_root(),
        "preview_root": preview_cache_root(),
    }
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate = subparsers.add_parser("migrate")
    migrate.add_argument("source", type=Path)
    migrate.add_argument("destination", type=Path)
    migrate.add_argument("--db", type=Path, default=defaults["db"])
    compact = subparsers.add_parser("compact")
    compact.add_argument("--db", type=Path, default=defaults["db"])
    compact.add_argument("--audio-root", type=Path, default=defaults["audio"])
    prune = subparsers.add_parser("prune-previews")
    prune.add_argument("--root", type=Path, default=defaults["preview_root"])
    prune.add_argument("--limit-mb", type=int, required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("destination", type=Path)
    adopt = subparsers.add_parser(
        "adopt",
        help=(
            "Reconcile a legacy render folder the database has never tracked "
            "(e.g. an old manual backup) against the current library. "
            "Copy-only: never deletes or modifies legacy_root."
        ),
    )
    adopt.add_argument("legacy_root", type=Path)
    adopt.add_argument("--db", type=Path, default=defaults["db"])
    adopt.add_argument("--audio-root", type=Path, default=defaults["audio"])
    args = parser.parse_args()

    def progress(detail: dict[str, int | str]) -> None:
        print("STORAGE_PROGRESS=" + json.dumps(detail, sort_keys=True), flush=True)

    try:
        if args.command == "migrate":
            result = migrate_audio_storage(
                args.source,
                args.destination,
                database_path=args.db,
                log=lambda message: print(message, flush=True),
                progress=progress,
            )
        elif args.command == "compact":
            result = compact_render_library(
                args.db,
                args.audio_root,
                log=lambda message: print(message, flush=True),
            )
        elif args.command == "prune-previews":
            result = prune_preview_cache(args.root, args.limit_mb)
        elif args.command == "adopt":
            result = adopt_legacy_renders(
                args.legacy_root,
                database_path=args.db,
                audio_root=args.audio_root,
                log=lambda message: print(message, flush=True),
                progress=progress,
            )
        else:
            prepare_audio_root(args.destination)
            from core.storage import StorageOperationSummary

            result = StorageOperationSummary()
    except Exception as exc:
        print(f"STORAGE_ERROR={type(exc).__name__}: {exc}", flush=True)
        return 1
    print("STORAGE_RESULT=" + json.dumps(asdict(result), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
