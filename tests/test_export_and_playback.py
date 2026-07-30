"""Guards for closest-match playability and generated-preset export location."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ClosestMatchPlayabilityTest(unittest.TestCase):
    """Every retrieved preset must remain auditionable without a render library.

    The UI enables a row's octave buttons only when the result carries an
    `audition_path` or a `preview_source_path`. The synthesis path originally
    emitted just `audition_path`, which resolves to the pre-rendered library —
    something a packaged install does not ship. Every closest match therefore
    showed "No local audio or factory preset is available" with dead octave
    buttons. The preset file must be carried forward as a render-on-demand
    fallback, exactly as the factory-fingerprint path already does.
    """

    def test_synthesis_results_carry_a_preview_source_fallback(self) -> None:
        source = (PROJECT_ROOT / "core" / "match_workflow.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"preview_source_path": preview_source',
            source,
            "run_match_file must emit preview_source_path so closest matches "
            "stay playable on installs without the pre-rendered library",
        )

    def test_ui_treats_preview_source_as_playable(self) -> None:
        source = (PROJECT_ROOT / "app" / "ui.py").read_text(encoding="utf-8")
        self.assertIn(
            'item.get("audition_path") or item.get("preview_source_path")',
            source,
            "a row with only a preset file must still be playable",
        )


class GeneratedPresetExportLocationTest(unittest.TestCase):
    """Generated presets default into a PatchLab folder the user owns.

    Defaulting to the machine-wide preset tree points people outside their own
    folder at a location that usually needs administrator rights, so the save
    silently fails.
    """

    def test_export_sites_use_the_patchlab_folder(self) -> None:
        source = (PROJECT_ROOT / "app" / "ui.py").read_text(encoding="utf-8")
        # The save dialog, the direct write, the library row export, and batch
        # output must all agree on one destination.
        self.assertGreaterEqual(
            len(re.findall(r"_patchlab_export_folder\(", source)),
            5,
            "every generated-preset destination should resolve through "
            "_patchlab_export_folder",
        )

    def test_export_dialog_does_not_default_to_the_bare_preset_root(self) -> None:
        source = (PROJECT_ROOT / "app" / "ui.py").read_text(encoding="utf-8")
        self.assertNotIn(
            'suggested = self._default_export_folder(synth) / f"{name}{extension}"',
            source,
            "the Export Preset dialog must open on the PatchLab subfolder",
        )

    def test_patchlab_folder_prefers_a_user_owned_root(self) -> None:
        source = (PROJECT_ROOT / "app" / "ui.py").read_text(encoding="utf-8")
        self.assertIn("def _patchlab_export_folder", source)
        # Preferring ENV.preset_roots over existing_preset_roots is what allows
        # a user-owned location to win even before it has been created.
        self.assertIn("for candidate in matching:", source)
        self.assertIn("under_home", source)


if __name__ == "__main__":
    unittest.main()
