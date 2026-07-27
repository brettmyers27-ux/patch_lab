from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.fxp import build_fxp, parse_fxp
from core.preset_export import (
    PresetExportResult,
    PresetExportVerification,
    commit_temporary_export,
    write_and_verify_native_preset,
)
from core.preset_scan import sha1_file
from core.serum2_preset import parse_serum2_preset
from core.serum2_preset_writer import branded_serum2_metadata, encode_serum2_preset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Verifier:
    def __init__(self, *, decoded: bool, coverage: float) -> None:
        self.decoded = decoded
        self.coverage = coverage

    def verify(
        self,
        export: PresetExportResult,
        **_kwargs: object,
    ) -> PresetExportVerification:
        similarity = 0.7021
        expected = 0.9045
        structurally_valid = self.decoded and self.coverage >= 0.85
        return PresetExportVerification(
            export=export,
            decoded_graph_equal=self.decoded,
            max_parameter_delta=None,
            render_state_coverage=self.coverage,
            clap_similarity=similarity,
            expected_clap_similarity=expected,
            similarity_delta=similarity - expected,
            passed=structurally_valid and abs(similarity - expected) <= 0.15,
        )


def _fake_write(
    output_path: Path,
    **_kwargs: object,
) -> PresetExportResult:
    output = Path(output_path)
    output.write_bytes(b"valid native preset")
    return PresetExportResult(
        path=output,
        synth="serum2",
        mode="test",
        base_preset_id=1,
        )


class _Control:
    def __init__(self) -> None:
        self.enabled = False
        self.text = ""
        self.messages: list[str] = []

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def setText(self, text: str) -> None:
        self.text = text

    def showMessage(self, message: str) -> None:
        self.messages.append(message)


class _WindowSurface:
    def __init__(self) -> None:
        self.save_preset_button = _Control()
        self.load_in_serum_button = _Control()
        self._status = _Control()
        self.logs: list[str] = []

    def statusBar(self) -> _Control:
        return self._status

    def append_log(self, message: str) -> None:
        self.logs.append(message)


class PresetExportLifecycleTest(unittest.TestCase):
    def test_audio_mismatch_is_advisory_and_keeps_valid_preset(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="patchlab-export-soft-check-"
        ) as directory:
            output = Path(directory) / "generated.SerumPreset"
            with patch(
                "core.preset_export.write_native_preset",
                side_effect=_fake_write,
            ):
                verification = write_and_verify_native_preset(
                    output,
                    synth="serum2",
                    base_preset_id=1,
                    vector=np.zeros(1, dtype=np.float32),
                    mask=np.ones(1, dtype=np.bool_),
                    meaningfully_modified=True,
                    midi_note=60,
                    target_audio=np.zeros(48_000, dtype=np.float32),
                    expected_clap_similarity=0.9045,
                    verifier=_Verifier(decoded=True, coverage=0.987),
                )

            self.assertFalse(verification.passed)
            self.assertTrue(verification.structurally_valid)
            self.assertTrue(output.is_file())

    def test_structural_failure_removes_generated_preset(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="patchlab-export-hard-check-"
        ) as directory:
            output = Path(directory) / "generated.SerumPreset"
            with patch(
                "core.preset_export.write_native_preset",
                side_effect=_fake_write,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "structural verification failed",
                ):
                    write_and_verify_native_preset(
                        output,
                        synth="serum2",
                        base_preset_id=1,
                        vector=np.zeros(1, dtype=np.float32),
                        mask=np.ones(1, dtype=np.bool_),
                        meaningfully_modified=True,
                        midi_note=60,
                        target_audio=np.zeros(48_000, dtype=np.float32),
                        expected_clap_similarity=0.9045,
                        verifier=_Verifier(decoded=False, coverage=0.2),
                    )

            self.assertFalse(output.exists())

    def test_commit_moves_private_temporary_file_to_destination(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="patchlab-export-commit-"
        ) as directory:
            root = Path(directory)
            temporary = root / "private" / "generated.fxp"
            temporary.parent.mkdir()
            temporary.write_bytes(b"preset bytes")
            destination = root / "chosen" / "My Preset.fxp"

            committed = commit_temporary_export(temporary, destination)

            self.assertEqual(committed, destination.resolve())
            self.assertEqual(destination.read_bytes(), b"preset bytes")
            self.assertFalse(temporary.exists())

    def test_factory_export_uses_and_deletes_private_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="patchlab-export-script-"
        ) as directory:
            root = Path(directory)
            source = root / "source.fxp"
            source.write_bytes(
                build_fxp(
                    b"fixture state",
                    plugin_id=b"XfsX",
                    program_name="Fixture",
                )
            )
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "factory_only": True,
                        "recommendation": {
                            "factory_source_path": str(source),
                            "content_hash": sha1_file(source),
                            "synth": "serum1",
                            "clap_similarity": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "chosen" / "Saved.fxp"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "export_match.py"),
                    str(result_path),
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            payload_line = next(
                line
                for line in completed.stdout.splitlines()
                if line.startswith("EXPORT_RESULT=")
            )
            payload = json.loads(payload_line.split("=", 1)[1])
            self.assertTrue(output.is_file())
            self.assertTrue(payload["temporary_export_used"])
            self.assertTrue(payload["temporary_export_deleted"])
            self.assertEqual(Path(payload["path"]), output.resolve())
            self.assertEqual(parse_fxp(output).program_name, "PatchLab Serum 1 Match")

    def test_serum2_metadata_drops_inherited_pack_branding(self) -> None:
        metadata = branded_serum2_metadata(
            {
                "fileType": "SerumPreset",
                "product": "Serum2",
                "productVersion": "2.0.13",
                "version": 5,
                "presetAuthor": "WA Production",
                "presetDescription": "https://waproduction.com",
                "presetName": "WA Bass",
                "tags": ["WA", "SoundMatch"],
                "url": "https://waproduction.com",
                "vendor": "WA Production",
            },
            name="PatchLab Serum 2 Match",
        )

        self.assertEqual(metadata["presetName"], "PatchLab Serum 2 Match")
        self.assertEqual(metadata["presetAuthor"], "PatchLab")
        self.assertEqual(metadata["vendor"], "PatchLab")
        self.assertEqual(metadata["tags"], ["PatchLab", "Generated"])
        self.assertEqual(metadata["presetDescription"], "Generated by PatchLab.")
        self.assertNotIn("url", metadata)
        self.assertNotIn("WA", json.dumps(metadata))
        self.assertNotIn("SoundMatch", json.dumps(metadata))

    def test_factory_serum2_export_rebrands_metadata(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="patchlab-factory-serum2-"
        ) as directory:
            root = Path(directory)
            source = root / "source.SerumPreset"
            source.write_bytes(
                encode_serum2_preset(
                    {
                        "fileType": "SerumPreset",
                        "product": "Serum2",
                        "productVersion": "2.0.13",
                        "version": 5,
                        "presetName": "External Preset",
                        "presetAuthor": "External Author",
                        "vendor": "External Vendor",
                        "url": "https://example.invalid",
                    },
                    {"components": []},
                    1,
                )
            )
            result_path = root / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "factory_only": True,
                        "recommendation": {
                            "factory_source_path": str(source),
                            "content_hash": sha1_file(source),
                            "synth": "serum2",
                            "clap_similarity": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "chosen" / "Saved.SerumPreset"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "export_match.py"),
                    str(result_path),
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout)
            metadata = parse_serum2_preset(output).metadata
            self.assertEqual(metadata["presetName"], "PatchLab Serum 2 Match")
            self.assertEqual(metadata["presetAuthor"], "PatchLab")
            self.assertEqual(metadata["vendor"], "PatchLab")
            self.assertNotIn("url", metadata)

    def test_audio_warning_is_logged_without_modal_error(self) -> None:
        from app.ui import LegacyMainWindow, QMessageBox

        surface = _WindowSurface()
        detail = {
            "path": "/chosen/My Preset.SerumPreset",
            "verification_warning": (
                "The saved preset reloaded correctly, but its verification "
                "render differed from the preview."
            ),
        }

        with patch.object(QMessageBox, "information") as information:
            LegacyMainWindow._export_completed(surface, detail)

        information.assert_not_called()
        self.assertTrue(surface.save_preset_button.enabled)
        self.assertIn("Preset saved:", surface.logs[0])
        self.assertIn("verification note", surface.logs[1].casefold())


if __name__ == "__main__":
    unittest.main()
