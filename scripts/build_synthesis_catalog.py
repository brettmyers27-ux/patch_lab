#!/usr/bin/env python3
"""Build the sanitized preset catalog that analysis-by-synthesis consumes.

The developer `data/library.db` also carries renders, match history, full
Serum 2 settings, and other private runtime state, none of which synthesis
needs. This produces the smallest database that still lets a candidate resolve
back to a preset: stable identity plus Serum 1's automation targets. Serum 2
uses its separately packaged target matrix and render-state templates instead.

Both the PyInstaller spec and the gated-artifact packaging call this, so the
frozen build and a git-clone install can never drift apart on what "the
catalog" means.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_SQL = """
CREATE TABLE presets (
  id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  name TEXT NOT NULL,
  synth TEXT NOT NULL,
  content_hash TEXT NOT NULL
);
INSERT INTO presets
  SELECT id,path,name,synth,content_hash FROM source.presets;
CREATE TABLE params (
  preset_id INTEGER,
  param_index INTEGER,
  param_name TEXT,
  norm_value REAL,
  display_value TEXT,
  PRIMARY KEY (preset_id,param_index)
);
INSERT INTO params
  SELECT pa.preset_id,pa.param_index,pa.param_name,
         pa.norm_value,pa.display_value
  FROM source.params pa
  JOIN source.presets p ON p.id=pa.preset_id
  WHERE p.synth='serum1';
CREATE INDEX idx_presets_content_hash ON presets(content_hash);
"""


def build_synthesis_catalog(source_library: Path, destination: Path) -> Path:
    """Write the sanitized catalog for `source_library` to `destination`."""

    source_library = Path(source_library).resolve()
    if not source_library.is_file():
        raise RuntimeError(
            f"A synthesis catalog requires an existing preset database: "
            f"{source_library}"
        )
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    with sqlite3.connect(destination) as catalog:
        catalog.execute("ATTACH DATABASE ? AS source", (str(source_library),))
        catalog.executescript(CATALOG_SQL)
        catalog.commit()
        catalog.execute("VACUUM")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "library.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "build" / "patchlab-synthesis-catalog.sqlite",
    )
    args = parser.parse_args()
    written = build_synthesis_catalog(args.source, args.output)
    with sqlite3.connect(f"file:{written}?mode=ro", uri=True) as catalog:
        presets = catalog.execute("SELECT COUNT(*) FROM presets").fetchone()[0]
        params = catalog.execute("SELECT COUNT(*) FROM params").fetchone()[0]
    print(
        f"SYNTHESIS_CATALOG={written} presets={presets} params={params} "
        f"bytes={written.stat().st_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
