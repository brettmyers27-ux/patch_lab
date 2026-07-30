from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from core.db import Database
from core.match_library import archive_match, delete_archived_match
from core.preview_cache import preview_cache_path, result_cache_keys


def _match_fixture(
    root: Path,
    name: str,
    *,
    modified: bool,
    content_hash: str = "a" * 40,
) -> tuple[Path, Path]:
    session = root / name
    session.mkdir()
    source = session / "source.wav"
    winner = session / "winner.wav"
    candidate = session / "candidate.npz"
    sf.write(source, np.zeros(512, dtype=np.float32), 48_000)
    sf.write(winner, np.ones(512, dtype=np.float32) * 0.1, 48_000)
    np.savez(
        candidate,
        vector=np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
        mask=np.asarray([True, True, False], dtype=np.bool_),
    )
    result = session / "result.json"
    result.write_text(
        json.dumps(
            {
                "source": {"path": str(source)},
                "existing_matches": [
                    {
                        "name": "Shared preset",
                        "synth": "serum2",
                        "content_hash": content_hash,
                    }
                ],
                "recommendation": {
                    "synth": "serum2",
                    "content_hash": content_hash,
                    "base_name": "Shared preset",
                    "similarity_percent": 90.0,
                    "meaningfully_modified": modified,
                    "winner_audio_path": str(winner),
                    "candidate_path": str(candidate),
                },
            }
        ),
        encoding="utf-8",
    )
    return result, source


def _archive(
    db: Database, root: Path, name: str, *, modified: bool
):
    result, source = _match_fixture(root, name, modified=modified)
    return archive_match(
        db,
        result_path=result,
        source_audio_path=source,
        target_synth="serum2",
        budget="balanced",
        library_root=root / "matches",
    )


def test_generated_preview_is_shared_then_removed_after_last_reference() -> None:
    with tempfile.TemporaryDirectory(prefix="patchlab-preview-ref-") as directory:
        root = Path(directory)
        db = Database(root / "library.db")
        first = _archive(db, root, "first", modified=True)
        second = _archive(db, root, "second", modified=True)
        first_keys = result_cache_keys(first.result_json_path)
        second_keys = result_cache_keys(second.result_json_path)
        generated = next(key for key in first_keys if key.startswith("generated-"))
        assert generated in second_keys
        cached = preview_cache_path(root, generated, 60)
        cached.parent.mkdir(parents=True)
        sf.write(cached, np.ones(512, dtype=np.float32) * 0.1, 48_000)

        assert delete_archived_match(
            db,
            first.record.match_uid,
            library_root=root / "matches",
            cache_root=root,
        )
        assert cached.is_file()

        assert delete_archived_match(
            db,
            second.record.match_uid,
            library_root=root / "matches",
            cache_root=root,
        )
        assert not cached.exists()
        assert not cached.parent.exists()


def test_preset_preview_policy_retains_shared_factory_cache() -> None:
    with tempfile.TemporaryDirectory(prefix="patchlab-preview-preset-") as directory:
        root = Path(directory)
        db = Database(root / "library.db")
        archived = _archive(db, root, "unmodified", modified=False)
        preset_key = "a" * 40
        cached = preview_cache_path(root, preset_key, 48)
        cached.parent.mkdir(parents=True)
        sf.write(cached, np.ones(512, dtype=np.float32) * 0.1, 48_000)

        assert delete_archived_match(
            db,
            archived.record.match_uid,
            library_root=root / "matches",
            cache_root=root,
        )
        assert cached.is_file()
