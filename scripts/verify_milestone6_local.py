#!/usr/bin/env python3
"""Real plug-in gate for consented local processing and upload dedup."""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import Database
from core.factory_match import run_factory_match_file
from core.local_library import process_linked_folder


REPORT = PROJECT_ROOT / "data" / "models" / "milestone6_local_processing_report.json"
FACTORY_ID = 67
NONFACTORY_ID = 628


class RecordingRelay:
    def __init__(self) -> None:
        self.hashes: set[str] = set()
        self.checks: list[str] = []
        self.uploads: list[dict[str, Any]] = []

    def check_hash(self, content_hash: str) -> bool:
        self.checks.append(content_hash)
        return content_hash in self.hashes

    def upload(
        self,
        *,
        preset_path: Path,
        relative_path: str,
        content_hash: str,
        fingerprint: dict[str, Any],
    ) -> None:
        self.uploads.append(
            {
                "preset_path": str(preset_path),
                "preset_suffix": preset_path.suffix,
                "relative_path": relative_path,
                "content_hash": content_hash,
                "fingerprint": fingerprint,
            }
        )
        self.hashes.add(content_hash)


def contains_audio_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            any(token in str(key).casefold() for token in ("audio", "wav", "waveform"))
            or contains_audio_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(contains_audio_key(item) for item in value)
    return False


def source_rows() -> dict[int, dict[str, Any]]:
    with Database().connect() as connection:
        rows = connection.execute(
            "SELECT id,path,content_hash,is_factory FROM presets WHERE id IN (?,?)",
            (FACTORY_ID, NONFACTORY_ID),
        ).fetchall()
    return {int(row["id"]): dict(row) for row in rows}


def main() -> int:
    mp.set_start_method("spawn", force=True)
    sources = source_rows()
    relay = RecordingRelay()
    with tempfile.TemporaryDirectory(prefix="patchlab-m6-local-") as temporary:
        root = Path(temporary)
        linked = root / "linked"
        (linked / "Factory Copy").mkdir(parents=True)
        (linked / "My Pack").mkdir(parents=True)
        factory_source = Path(sources[FACTORY_ID]["path"])
        own_source = Path(sources[NONFACTORY_ID]["path"])
        factory_copy = linked / "Factory Copy" / factory_source.name
        own_copy = linked / "My Pack" / own_source.name
        shutil.copyfile(factory_source, factory_copy)
        shutil.copyfile(own_source, own_copy)
        db = root / "user-data" / "library.db"
        audio = root / "user-data" / "audio"
        states = root / "user-data" / "states"
        first = process_linked_folder(
            linked,
            db_path=db,
            audio_root=audio,
            state_dir=states,
            relay=relay,
            render_processes=1,
            log=lambda message: print(message, flush=True),
        )
        uploads_after_first = len(relay.uploads)
        second = process_linked_folder(
            linked,
            db_path=db,
            audio_root=audio,
            state_dir=states,
            relay=relay,
            render_processes=1,
            log=lambda message: print(message, flush=True),
        )
        uploads_after_second = len(relay.uploads)
        with Database(db).connect() as connection:
            local = connection.execute(
                "SELECT id,content_hash,is_factory FROM presets ORDER BY id"
            ).fetchall()
        own_row = next(
            row
            for row in local
            if str(row["content_hash"]) == str(sources[NONFACTORY_ID]["content_hash"])
        )
        query = audio / str(int(own_row["id"])) / "60.wav"
        result_path = run_factory_match_file(
            query,
            target_synth="serum1",
            mapping_path=PROJECT_ROOT / "data" / "local" / "factory_paths.json",
            local_db_path=db,
            session_root=root / "matches",
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result_hashes = {
            str(item["content_hash"]) for item in result["existing_matches"]
        }
        uploaded_audio = any(
            item["preset_suffix"].casefold()
            not in {".fxp", ".serumpreset"}
            or contains_audio_key(item["fingerprint"])
            for item in relay.uploads
        )
        wav_count = len(list(audio.rglob("*.wav")))
        payload = {
            "fixtures": {
                "factory_hash": sources[FACTORY_ID]["content_hash"],
                "nonfactory_hash": sources[NONFACTORY_ID]["content_hash"],
            },
            "first_run": asdict(first),
            "second_run": asdict(second),
            "uploads_after_first": uploads_after_first,
            "uploads_after_second": uploads_after_second,
            "relay_checks": relay.checks,
            "upload_relative_paths": [
                item["relative_path"] for item in relay.uploads
            ],
            "uploaded_audio": uploaded_audio,
            "persistent_local_wavs": wav_count,
            "nonfactory_present_in_top10": (
                sources[NONFACTORY_ID]["content_hash"] in result_hashes
            ),
            "top10_hashes": sorted(result_hashes),
        }
        payload["gate_pass"] = (
            first.searchable_local == 2
            and first.factory_skipped_upload == 1
            and first.relay_uploaded == 1
            and second.relay_uploaded == 0
            and second.relay_already_present == 1
            and uploads_after_first == uploads_after_second == 1
            and not uploaded_audio
            and wav_count == 14
            and payload["nonfactory_present_in_top10"]
        )
        REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("MILESTONE6_LOCAL_REPORT=" + json.dumps(payload, sort_keys=True))
        return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
