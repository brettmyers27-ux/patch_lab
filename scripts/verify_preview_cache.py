#!/usr/bin/env python3
"""Two-launch packaged gate for durable octave previews and reference cleanup."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import Database
from core.match_library import (
    archive_match,
    delete_archived_match,
    resolved_record_paths,
)
from core.preview_cache import (
    preview_cache_path,
    result_cache_keys,
    unmodified_recommendation_basis_index,
)
from scripts.render_factory_preview import render_preview


def _factory(mapping_path: Path) -> tuple[str, Path]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    for digest, value in mapping["local_paths_by_hash"].items():
        path = Path(value)
        if path.suffix.casefold() == ".fxp" and path.is_file():
            return str(digest), path
    raise RuntimeError("No verified local Serum 1 factory preset was available")


def _fixture(
    root: Path,
    name: str,
    *,
    digest: str,
    factory_source: Path,
    modified: bool,
) -> tuple[Path, Path]:
    session = root / "fixtures" / name
    session.mkdir(parents=True)
    source = session / "query.wav"
    winner = session / "winner.wav"
    candidate = session / "candidate.npz"
    sf.write(source, np.zeros(2048, dtype=np.float32), 48_000)
    sf.write(winner, np.ones(2048, dtype=np.float32) * 0.1, 48_000)
    np.savez(
        candidate,
        vector=np.asarray([0.1, 0.25, 0.75, 0.9], dtype=np.float32),
        mask=np.asarray([True, True, True, False], dtype=np.bool_),
    )
    payload = {
        "source": {"path": str(source)},
        "existing_matches": [
            {
                "name": factory_source.stem,
                "synth": "serum1",
                "content_hash": digest,
                "source_path": str(factory_source),
                "preview_source_path": str(factory_source),
                "similarity_percent": 91.0,
            }
        ],
        "recommendation": {
            "synth": "serum1",
            "base_name": factory_source.stem,
            "content_hash": digest,
            "meaningfully_modified": modified,
            "similarity_percent": 91.0,
            "winner_audio_path": str(winner),
            "candidate_path": str(candidate),
            "preview_source_path": str(factory_source) if not modified else None,
        },
    }
    result = session / "result.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result, source


def _archive_pair(
    database: Database,
    root: Path,
    *,
    prefix: str,
    digest: str,
    factory_source: Path,
    modified: bool,
) -> list[str]:
    uids: list[str] = []
    for index in range(2):
        result, source = _fixture(
            root,
            f"{prefix}-{index}",
            digest=digest,
            factory_source=factory_source,
            modified=modified,
        )
        archived = archive_match(
            database,
            result_path=result,
            source_audio_path=source,
            target_synth="serum1",
            budget="quick",
            library_root=root / "matches",
        )
        uids.append(archived.record.match_uid)
    return uids


def _initial(root: Path, mapping_path: Path) -> dict:
    started = time.monotonic()
    digest, factory_source = _factory(mapping_path)
    database = Database(root / "library.db")
    factory_uids = _archive_pair(
        database,
        root,
        prefix="factory",
        digest=digest,
        factory_source=factory_source,
        modified=False,
    )
    generated_uids = _archive_pair(
        database,
        root,
        prefix="generated",
        digest=digest,
        factory_source=factory_source,
        modified=True,
    )

    rendered: list[Path] = []
    render_times: dict[str, float] = {}
    for note in (48, 60):
        before = time.monotonic()
        path = render_preview(
            factory_source,
            "serum1",
            note,
            digest,
            output_root=root,
        )
        render_times[str(note)] = time.monotonic() - before
        rendered.append(path)

    generated_record = database.get_match_library(generated_uids[0])
    assert generated_record is not None
    _source, generated_result = resolved_record_paths(
        generated_record, root / "matches"
    )
    generated_key = next(
        key
        for key in result_cache_keys(generated_result)
        if key.startswith("generated-")
    )
    generated_wav = preview_cache_path(root, generated_key, 60)
    generated_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(generated_wav, np.ones(2048, dtype=np.float32) * 0.1, 48_000)
    delete_archived_match(
        database,
        generated_uids[0],
        library_root=root / "matches",
        cache_root=root,
    )
    generated_survived_first_delete = generated_wav.is_file()
    delete_archived_match(
        database,
        generated_uids[1],
        library_root=root / "matches",
        cache_root=root,
    )
    generated_removed_last_delete = not generated_wav.exists()

    manifest = {
        "digest": digest,
        "factory_source": str(factory_source),
        "factory_uids": factory_uids,
        "paths": [str(path) for path in rendered],
        "mtimes_ns": {str(path): path.stat().st_mtime_ns for path in rendered},
        "render_times_s": render_times,
        "generated_key": generated_key,
    }
    (root / "restart-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "phase": "initial",
        "root": str(root),
        "factory_source": str(factory_source),
        "cache_paths": [str(path) for path in rendered],
        "cache_file_count": len(list((root / "audio" / digest).glob("*.wav"))),
        "generated_survived_first_delete": generated_survived_first_delete,
        "generated_removed_last_delete": generated_removed_last_delete,
        "elapsed_s": time.monotonic() - started,
        "gate_pass": generated_survived_first_delete
        and generated_removed_last_delete
        and all(path.is_file() for path in rendered),
    }


def _restart(root: Path) -> dict:
    started = time.monotonic()
    manifest = json.loads((root / "restart-manifest.json").read_text())
    digest = str(manifest["digest"])
    factory_source = Path(manifest["factory_source"])
    before_mtimes = {str(key): int(value) for key, value in manifest["mtimes_ns"].items()}
    replay_times: dict[str, float] = {}
    after_mtimes: dict[str, int] = {}
    for note in (48, 60):
        before = time.monotonic()
        path = render_preview(
            factory_source, "serum1", note, digest, output_root=root
        )
        replay_times[str(note)] = time.monotonic() - before
        after_mtimes[str(path)] = path.stat().st_mtime_ns
    unchanged = before_mtimes == after_mtimes

    fresh_started = time.monotonic()
    fresh = render_preview(
        factory_source, "serum1", 72, digest, output_root=root
    )
    fresh_elapsed = time.monotonic() - fresh_started
    audio, _rate = sf.read(fresh, dtype="float32", always_2d=True)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    database = Database(root / "library.db")
    factory_uids = list(manifest["factory_uids"])
    delete_archived_match(
        database,
        factory_uids[0],
        library_root=root / "matches",
        cache_root=root,
    )
    survives_shared_delete = fresh.is_file()
    delete_archived_match(
        database,
        factory_uids[1],
        library_root=root / "matches",
        cache_root=root,
    )
    retained_by_preset_policy = fresh.is_file()

    base_result = {
        "existing_matches": [{"content_hash": digest}],
        "recommendation": {
            "content_hash": digest,
            "meaningfully_modified": False,
        },
    }
    modified_result = json.loads(json.dumps(base_result))
    modified_result["recommendation"]["meaningfully_modified"] = True
    duplicate_label_gate = (
        unmodified_recommendation_basis_index(base_result) == 0
        and unmodified_recommendation_basis_index(modified_result) is None
    )
    generated_orphans = [
        path.name
        for path in (root / "audio").glob("generated-*")
        if path.is_dir()
    ]
    factory_files = sorted((root / "audio" / digest).glob("*.wav"))
    payload = {
        "phase": "restart",
        "root": str(root),
        "cache_paths": [str(path) for path in factory_files],
        "cache_file_count": len(factory_files),
        "mtimes_before_ns": before_mtimes,
        "mtimes_after_ns": after_mtimes,
        "mtimes_unchanged": unchanged,
        "cached_replay_times_s": replay_times,
        "fresh_library_render_path": str(fresh),
        "fresh_library_render_s": fresh_elapsed,
        "fresh_library_rms": rms,
        "shared_survived_first_delete": survives_shared_delete,
        "factory_retained_after_last_history_delete": retained_by_preset_policy,
        "generated_orphan_directories": generated_orphans,
        "unmodified_basis_labeled_only": duplicate_label_gate,
        "remaining_match_rows": len(database.list_match_library()),
        "elapsed_s": time.monotonic() - started,
    }
    payload["gate_pass"] = (
        unchanged
        and rms > 1e-3
        and len(factory_files) == 3
        and survives_shared_delete
        and retained_by_preset_policy
        and not generated_orphans
        and duplicate_label_gate
        and payload["remaining_match_rows"] == 0
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("initial", "restart"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--factory-mapping", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "initial":
        if args.factory_mapping is None:
            raise RuntimeError("--factory-mapping is required for the initial phase")
        payload = _initial(root, args.factory_mapping.expanduser().resolve())
    else:
        payload = _restart(root)
    print("PREVIEW_CACHE_GATE=" + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
