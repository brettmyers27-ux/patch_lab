from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest.mock import patch

from core.preparation_diagnostic import (
    FAILURE_STATUSES,
    collect,
    discover_database,
    platform_data_candidates,
    write_output,
)
from core.preparation_error_diagnostic import collect as collect_error_details


def _database(path: Path, *, older_schema: bool = False, failures: list[tuple[str, str, str | None]] | None = None) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE presets (id INTEGER PRIMARY KEY,synth TEXT,status TEXT" + ("" if older_schema else ",error TEXT") + ")")
        if not older_schema:
            connection.execute("CREATE TABLE preparation_jobs (preset_id INTEGER,state TEXT,last_error TEXT,attempt_count INTEGER)")
            connection.execute("CREATE TABLE renders (preset_id INTEGER,midi_note INTEGER,peak_dbfs REAL,rms_dbfs REAL)")
        for index, (synth, status, error) in enumerate(failures or [], start=1):
            if older_schema:
                connection.execute("INSERT INTO presets VALUES (?,?,?)", (index, synth, status))
            else:
                connection.execute("INSERT INTO presets VALUES (?,?,?,?)", (index, synth, status, error))
                connection.execute("INSERT INTO preparation_jobs VALUES (?,?,?,?)", (index, "failed", error, index))
                if status == "failed_silent":
                    connection.execute("INSERT INTO renders VALUES (?,?,?,?)", (index, 36, -61.0, -72.0))


class PreparationDiagnosticTests(unittest.TestCase):
    def test_follow_up_collector_keeps_semantic_error_and_redacts_private_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "library.db"
            _database(database)
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "INSERT INTO presets VALUES (1,?,?,?)",
                    ("serum2", "failed_load", "RuntimeError: /Users/tester/Presets/private.fxp"),
                )
                connection.execute(
                    "INSERT INTO preparation_jobs VALUES (?,?,?,?)",
                    (1, "failed", "RuntimeError: renderer process exited before note 48", 2),
                )
            payload = collect_error_details(database)
        group = payload["failure_groups"][0]
        self.assertEqual(group["semantic_error"], "RuntimeError: renderer process exited before note 48")
        self.assertEqual(group["error_source"], "preparation_jobs.last_error")
        self.assertEqual(payload["relationship"]["rows_with_job_state"], 1)
        self.assertNotIn("tester", str(payload))
    def test_failure_family_fixtures_are_grouped_and_database_is_unchanged(self) -> None:
        failures = [
            ("serum2", "failed_load", "RuntimeError: rendering did not produce seven valid note files"),
            ("serum2", "failed_load", "FileNotFoundError: hidden path"),
            ("serum2", "failed_load", "only 0 parameters changed from init"),
            ("serum2", "failed_load", "BrokenPipeError: worker ended"),
            ("serum2", "failed_silent", "Silent rendered MIDI notes: 36"),
            ("serum2", "failed_load", "BrokenPipeError: worker ended"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "library.db"
            _database(database, failures=failures)
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            payload = collect(database)
            after = hashlib.sha256(database.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertFalse((Path(str(database) + "-wal")).exists())
            self.assertFalse((Path(str(database) + "-journal")).exists())
        families = {item["sanitized_error_family"]: item["count"] for item in payload["failure_groups"]}
        self.assertEqual(families["worker_pipe"], 2)
        self.assertEqual(families["seven_note_validation"], 1)
        self.assertEqual(families["render_state_missing"], 1)
        self.assertEqual(families["state_application"], 1)
        self.assertEqual(families["silent_notes:36"], 1)
        self.assertNotIn("hidden path", str(payload))
        self.assertTrue(all(item["representative_correlation_ids"] for item in payload["failure_groups"]))
        self.assertEqual(payload["patchlab_version"], "not_stored_by_1.6.6_library_database")

    def test_no_failures_and_older_schema_degrade_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean = Path(temporary) / "clean.db"
            _database(clean)
            self.assertEqual(collect(clean)["totals"]["failure_rows"], 0)
            older = Path(temporary) / "older.db"
            _database(older, older_schema=True, failures=[("serum2", "failed_load", None)])
            payload = collect(older)
        self.assertEqual(payload["totals"]["failure_rows"], 1)
        self.assertTrue(any("unavailable" in item for item in payload["unavailable_fields"]))

    def test_macos_discovery_is_home_relative_and_windows_paths_are_native(self) -> None:
        mac_home = Path("/private/tmp/diagnostic-home")
        mac = platform_data_candidates(system_name="Darwin", home=mac_home)
        self.assertEqual(mac, (mac_home / "Library" / "Application Support" / "Patch Lab" / "library.db",))
        windows = platform_data_candidates(
            system_name="Windows",
            home=PureWindowsPath(r"C:\\Users\\Tester"),
            environ={"LOCALAPPDATA": r"D:\\Local AppData"},
        )
        self.assertEqual(str(windows[0]), r"D:\Local AppData\Patch Lab\library.db")
        self.assertEqual(str(windows[1]), r"D:\Local AppData\PatchLab\library.db")

    def test_discovery_rejects_ambiguity_and_output_contains_no_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            standard = home / "Library" / "Application Support" / "Patch Lab"
            standard.mkdir(parents=True)
            database = standard / "library.db"
            _database(database, failures=[("serum1", "failed_load", "RuntimeError: private path")])
            with (
                patch("core.preparation_diagnostic.platform.system", return_value="Darwin"),
                patch("core.preparation_diagnostic.Path.home", return_value=home),
            ):
                result = discover_database()
            self.assertEqual(result.status, "found")
            output = home / "Desktop" / "patchlab-1.6.6-preparation-diagnostics.json"
            write_output(output, collect(database))
            self.assertTrue(output.is_file())
            self.assertNotIn(str(home), output.read_text(encoding="utf-8"))

    def test_status_contract_is_fixed(self) -> None:
        self.assertEqual(FAILURE_STATUSES, ("failed_load", "failed_silent", "incomplete", "stale"))


if __name__ == "__main__":
    unittest.main()
