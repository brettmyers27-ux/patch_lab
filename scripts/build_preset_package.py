"""Package every catalog preset for the Windows machine.

Files are named by content hash so nothing collides across folders, and the
manifest lets the PC rebuild a hash -> local-path mapping in exactly the format
core/serum2_preset_writer._factory_path_for_hash already consumes.
"""
import json, shutil, sqlite3, sys
from pathlib import Path

DEST = Path(sys.argv[1])
DB = Path("data/library.db")

DEST.mkdir(parents=True, exist_ok=True)
files_dir = DEST / "presets"
files_dir.mkdir(exist_ok=True)

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row
rows = con.execute(
    "SELECT id,path,name,synth,content_hash FROM presets ORDER BY id"
).fetchall()

manifest, missing, copied, total = {}, [], 0, 0
for row in rows:
    src = Path(str(row["path"]))
    if not src.is_file():
        missing.append(str(row["id"]))
        continue
    rel = f"{row['content_hash']}{src.suffix}"
    dst = files_dir / rel
    if not dst.exists():
        shutil.copy2(src, dst)
        total += dst.stat().st_size
    copied += 1
    manifest[str(row["content_hash"])] = {
        "relative_path": f"presets/{rel}",
        "preset_id": int(row["id"]),
        "name": str(row["name"]),
        "synth": str(row["synth"]),
    }

(DEST / "manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "purpose": "Serum source presets for Stage 2 atomic index rebuild",
            "preset_count": copied,
            "missing_preset_ids": missing,
            "presets_by_hash": manifest,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"copied {copied}/{len(rows)} presets, {total/1e9:.2f} GB")
if missing:
    print(f"MISSING {len(missing)}: {missing[:5]}")
