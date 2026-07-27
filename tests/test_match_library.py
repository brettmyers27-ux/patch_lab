from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.db import Database
from core.match_library import (
    archive_match,
    delete_archived_match,
    resolve_result_path,
    resolved_record_paths,
)


def _fixture(root: Path, *, recommendation: bool = True) -> tuple[Path, Path]:
    session = root / "session"
    session.mkdir()
    source = session / "source.aiff"
    winner = session / "winner.wav"
    sf.write(source, np.zeros(2400, dtype=np.float32), 48_000)
    sf.write(winner, np.ones(2400, dtype=np.float32) * 0.1, 48_000)
    candidate = session / "candidate.npz"
    np.savez(candidate, vector=np.zeros(3), mask=np.ones(3))
    payload: dict = {
        "source": {"path": str(source), "start_offset_s": 0.0},
        "message": "No confident match" if not recommendation else "Match complete",
        "no_confident_match": not recommendation,
        "existing_matches": [{"name": "kept"}],
    }
    if recommendation:
        payload["recommendation"] = {
            "synth": "serum2",
            "base_name": "PatchLab Generated Serum 2",
            "similarity_percent": 91.5,
            "clap_similarity": 0.915,
            "winner_audio_path": str(winner),
            "candidate_path": str(candidate),
        }
    result = session / "result.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result, source


class MatchLibraryTest(unittest.TestCase):
    def test_archive_survives_ephemeral_source_and_deletes_cleanly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="patchlab-library-") as directory:
            root = Path(directory)
            db = Database(root / "library.db")
            result, source = _fixture(root)
            archived = archive_match(
                db,
                result_path=result,
                source_audio_path=source,
                target_synth="serum2",
                budget="balanced",
                library_root=root / "matches",
            )
            self.assertFalse(archived.record.result_json_path.is_absolute())
            self.assertFalse(archived.record.source_audio_path.is_absolute())
            source.unlink()
            result.unlink()
            archived_source, archived_result = resolved_record_paths(
                archived.record, root / "matches"
            )
            stored = json.loads(archived_result.read_text())
            self.assertTrue(archived_source.is_file())
            self.assertTrue(stored["source"]["path"].startswith("source/"))
            self.assertTrue(
                resolve_result_path(
                    archived_result,
                    stored["recommendation"]["winner_audio_path"],
                ).is_file()
            )
            self.assertTrue(
                resolve_result_path(
                    archived_result,
                    stored["recommendation"]["candidate_path"],
                ).is_file()
            )
            self.assertEqual(stored["existing_matches"][0]["name"], "kept")
            self.assertTrue(
                delete_archived_match(
                    db,
                    archived.record.match_uid,
                    library_root=root / "matches",
                )
            )
            self.assertEqual(db.list_match_library(), [])
            self.assertFalse(archived.entry_root.exists())

    def test_no_confident_match_is_archived(self) -> None:
        with tempfile.TemporaryDirectory(prefix="patchlab-no-match-") as directory:
            root = Path(directory)
            db = Database(root / "library.db")
            result, source = _fixture(root, recommendation=False)
            archived = archive_match(
                db,
                result_path=result,
                source_audio_path=source,
                target_synth="serum1",
                budget="quick",
                library_root=root / "matches",
            )
            self.assertTrue(archived.record.no_confident_match)
            self.assertTrue(archived.result_json_path.is_file())


if __name__ == "__main__":
    unittest.main()
