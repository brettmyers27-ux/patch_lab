from __future__ import annotations

import unittest
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from scripts.verify_windows_install import compare_parameter_dump
from scripts.verify_windows_wheels import compatible
from scripts.render_factory_preview import _catalog_render_state


class WindowsParityHelpersTest(unittest.TestCase):
    def test_parameter_diff_detects_index_name_drift(self) -> None:
        reference = {
            "parameter_count": 2,
            "index_name_sha256": "not-the-live-signature",
            "parameters": [
                {"index": 0, "name": "A", "normalized_value": 0.5},
                {"index": 1, "name": "B", "normalized_value": 0.25},
            ],
        }
        live = [
            SimpleNamespace(index=0, name="A", norm_value=0.5),
            SimpleNamespace(index=1, name="Different", norm_value=0.25),
        ]
        result, report = compare_parameter_dump("serum1", live, reference)
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(report["name_mismatch_count"], 1)

    def test_parameter_diff_allows_small_default_value_noise(self) -> None:
        live = [
            SimpleNamespace(index=0, name="A", norm_value=0.50001),
        ]
        from scripts.verify_windows_install import _signature

        reference = {
            "parameter_count": 1,
            "index_name_sha256": _signature(live),
            "parameters": [
                {"index": 0, "name": "A", "normalized_value": 0.5},
            ],
        }
        result, _report = compare_parameter_dump("serum1", live, reference)
        self.assertEqual(result.status, "PASS")

    def test_wheel_compatibility_tags(self) -> None:
        self.assertTrue(compatible("package-1.0-py3-none-any.whl"))
        self.assertTrue(compatible("package-1.0-cp311-cp311-win_amd64.whl"))
        self.assertTrue(compatible("package-1.0-cp39-abi3-win_amd64.whl"))
        self.assertFalse(compatible("package-1.0-cp311-cp311-win32.whl"))
        self.assertFalse(compatible("package-1.0-cp311-cp311-macosx_12_0_arm64.whl"))

    def test_installer_and_launcher_keep_the_console_hidden(self) -> None:
        root = Path(__file__).resolve().parents[1]
        installer = (root / "install.ps1").read_text(encoding="utf-8")
        launcher = (root / "app" / "windows_launcher.pyw").read_text(encoding="utf-8")
        self.assertIn(r".venv\Scripts\pythonw.exe", installer)
        self.assertIn("CreateShortcut", installer)
        self.assertIn("PATCHLAB_DISTRIBUTION_MODE", launcher)
        self.assertIn("utf-8-sig", launcher)
        self.assertNotIn("shell=True", launcher)

    def test_serum2_factory_preview_resolves_state_by_catalog_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "catalog.sqlite"
            states = root / "states"
            states.mkdir()
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE presets (id INTEGER PRIMARY KEY, synth TEXT, content_hash TEXT)"
                )
                connection.execute(
                    "INSERT INTO presets VALUES (712, 'serum2', 'stable-hash')"
                )
                connection.commit()
            expected = states / "712.vstpreset"
            expected.write_bytes(b"state")
            assets = SimpleNamespace(
                library_db=database,
                find_render_state=lambda preset_id: (
                    states / f"{preset_id}.vstpreset"
                ),
            )

            actual = _catalog_render_state("stable-hash", assets)

            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
