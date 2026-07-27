#!/usr/bin/env python3
"""Run the same factory hash verification used at application launch."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.factory_verify import verify_local_factory_install


MAPPING = PROJECT_ROOT / "data" / "local" / "factory_paths.json"
REPORT = PROJECT_ROOT / "data" / "models" / "local_factory_verification_report.json"


def main() -> int:
    result = verify_local_factory_install(mapping_path=MAPPING)
    payload = asdict(result)
    payload["no_factory_install"] = result.no_factory_install
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    concise = {
        "bundle_available": result.bundle_available,
        "factory_directories_found": result.factory_directories_found,
        "local_files_found": result.local_files_found,
        "known_bundle_hashes": result.known_bundle_hashes,
        "matched_hashes": result.matched_hashes,
        "missing_hashes": len(result.missing_hashes),
        "unknown_local_hashes": len(result.unknown_local_hashes),
        "elapsed_s": result.elapsed_s,
        "mapping_path": str(MAPPING),
        "report_path": str(REPORT),
    }
    print("LOCAL_FACTORY_REPORT=" + json.dumps(concise, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
