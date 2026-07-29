from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.match_batch import (
    discover_batch_audio,
    disambiguated_preset_path,
    resumable_batch_files,
    sanitize_folder_name,
)
from core.match_library import file_sha1


class MatchBatchHelpersTest(unittest.TestCase):
    def test_discovery_resume_and_disambiguation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="patchlab-batch-") as directory:
            root = Path(directory)
            (root / "one.wav").write_bytes(b"one")
            (root / "two.aiff").write_bytes(b"two")
            (root / "ignore.txt").write_text("unsupported")
            nested = root / "nested"
            nested.mkdir()
            (nested / "three.flac").write_bytes(b"three")
            shallow = discover_batch_audio(root)
            self.assertEqual(len(shallow.supported), 2)
            self.assertEqual(shallow.unsupported_count, 1)
            deep = discover_batch_audio(root, recursive=True)
            self.assertEqual(len(deep.supported), 3)
            pending, skipped = resumable_batch_files(
                list(deep.supported), {file_sha1(deep.supported[0])}
            )
            self.assertEqual(skipped, 1)
            self.assertEqual(len(pending), 2)
            self.assertEqual(sanitize_folder_name('  My:/Batch*  '), "My-Batch")
            self.assertEqual(sanitize_folder_name("CON"), "PatchLab-CON")
            self.assertEqual(sanitize_folder_name("con.txt"), "PatchLab-con.txt")
            self.assertEqual(sanitize_folder_name("LPT1"), "PatchLab-LPT1")
            existing = root / "PatchLab Generated Serum 2.SerumPreset"
            existing.write_bytes(b"x")
            self.assertEqual(
                disambiguated_preset_path(
                    root, "PatchLab Generated Serum 2", ".SerumPreset"
                ).name,
                "PatchLab Generated Serum 2 2.SerumPreset",
            )


if __name__ == "__main__":
    unittest.main()
