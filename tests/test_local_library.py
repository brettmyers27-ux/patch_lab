from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core.db import Database
from core.library_state import mark_preset_prepared, reconcile_source_tree
from core.local_library import (
    auto_scan_due,
    fingerprint_pending_presets,
    process_linked_folder,
    record_auto_scan,
)
from core.plugin_host import ParameterValue
from core.preset_scan import sha1_file
from core.render import MIDI_NOTES, RenderSummary


class FailingRelay:
    """A support service that is unreachable: every upload attempt fails."""

    def __init__(self) -> None:
        self.attempts = 0

    def post_submission(self, **_kwargs):
        self.attempts += 1
        raise ConnectionError("relay unavailable")


class LocalRelayResilienceTest(unittest.TestCase):
    def test_upload_failures_do_not_abort_local_scan(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="patchlab-relay-resilience-"
        ) as temporary:
            root = Path(temporary)
            linked = root / "linked"
            linked.mkdir()
            db_path = root / "library.db"
            audio_root = root / "audio"
            state_dir = root / "states"
            database = Database(db_path)
            paths = []
            preset_ids = []
            for index in range(4):
                path = linked / f"Preset {index}.fxp"
                path.write_bytes(b"CcnK" + bytes([index]) * 32)
                paths.append(path)
                preset_id, _inserted = database.insert_preset(
                    path=path,
                    name=path.stem,
                    synth="serum1",
                    content_hash=sha1_file(path),
                )
                preset_ids.append(preset_id)
                database.replace_params(
                    preset_id,
                    [ParameterValue(0, "Master", 0.5, "50%")],
                    "test",
                )
            reconcile_source_tree(linked, database)
            for preset_id in preset_ids:
                for note in (0, *MIDI_NOTES):
                    database.upsert_fingerprint(
                        preset_id,
                        note,
                        np.zeros(512, dtype=np.float32).tobytes(),
                        np.zeros(9, dtype=np.float32).tobytes(),
                    )
                self.assertTrue(mark_preset_prepared(database, preset_id))

            relay = FailingRelay()
            messages: list[str] = []
            fake_render = RenderSummary(selected_presets=4)
            with (
                patch(
                    "core.local_library.FactoryBundle"
                ) as factory_bundle,
                patch(
                    "core.local_library.render_library",
                    return_value=fake_render,
                ),
            ):
                factory_bundle.return_value.known_hashes.return_value = set()
                summary = process_linked_folder(
                    linked,
                    db_path=db_path,
                    audio_root=audio_root,
                    state_dir=state_dir,
                    relay=relay,
                    render_processes=1,
                    log=messages.append,
                    upload_sleep=lambda _seconds: None,
                )

            self.assertEqual(summary.searchable_local, 4)
            # One bundle for all four presets, retried a bounded number of times.
            self.assertEqual(summary.relay_upload_failed, 4)
            self.assertEqual(summary.relay_disabled_after_failures, 1)
            self.assertEqual(summary.relay_uploaded, 0)
            self.assertEqual(relay.attempts, 4)
            for path in paths:
                self.assertTrue(path.is_file(), "the user's preset files are never touched")
            self.assertFalse((root / "contribution-ledger.json").exists())
            self.assertTrue(
                any(
                    "Preset sharing skipped (will retry next scan)" in message
                    for message in messages
                )
            )
            self.assertTrue(
                any(
                    "Sharing paused for the remainder of this scan" in message
                    for message in messages
                )
            )
            self.assertTrue(messages[-1].startswith("LOCAL_LIBRARY_SUMMARY="))

    def test_new_presets_are_shared_in_one_bundle_and_never_offered_twice(self) -> None:
        with tempfile.TemporaryDirectory(prefix="patchlab-contribution-") as temporary:
            root = Path(temporary)
            linked = root / "linked"
            linked.mkdir()
            db_path = root / "library.db"
            database = Database(db_path)
            paths = []
            preset_ids = []
            for index in range(4):
                path = linked / f"Preset {index}.fxp"
                path.write_bytes(b"CcnK" + bytes([index]) * 32)
                paths.append(path)
                preset_id, _ = database.insert_preset(
                    path=path, name=path.stem, synth="serum1", content_hash=sha1_file(path)
                )
                preset_ids.append(preset_id)
                database.replace_params(preset_id, [ParameterValue(0, "Master", 0.5, "50%")], "test")
            reconcile_source_tree(linked, database)
            for preset_id in preset_ids:
                for note in (0, *MIDI_NOTES):
                    database.upsert_fingerprint(
                        preset_id, note,
                        np.zeros(512, dtype=np.float32).tobytes(),
                        np.zeros(9, dtype=np.float32).tobytes(),
                    )
                self.assertTrue(mark_preset_prepared(database, preset_id))

            class GoodRelay:
                base_url = "https://relay.invalid"

                def __init__(self) -> None:
                    self.sent: list[dict] = []
                    self.archives: list[list[str]] = []

                def post_submission(self, **kwargs):
                    import zipfile

                    self.sent.append(kwargs)
                    with zipfile.ZipFile(kwargs["path"]) as archive:
                        self.archives.append(archive.namelist())
                    return {
                        "ok": True, "submission_id": kwargs["submission_id"], "receipt_id": "r" * 24,
                        "sha256": kwargs["sha256"], "size": Path(kwargs["path"]).stat().st_size,
                    }

            def scan(relay):
                with (
                    patch("core.local_library.FactoryBundle") as factory_bundle,
                    patch("core.local_library.render_library", return_value=RenderSummary(selected_presets=4)),
                ):
                    factory_bundle.return_value.known_hashes.return_value = set()
                    return process_linked_folder(
                        linked, db_path=db_path, audio_root=root / "audio", state_dir=root / "states",
                        relay=relay, render_processes=1, log=lambda _m: None,
                        upload_sleep=lambda _s: None,
                    )

            relay = GoodRelay()
            first = scan(relay)
            self.assertEqual(len(relay.sent), 1, "one request for the whole contribution, not one per preset")
            self.assertEqual(relay.sent[0]["kind"], "preset_contribution")
            self.assertEqual(first.relay_uploaded, 4)
            names = relay.archives[0]
            self.assertIn("manifest.json", names)
            self.assertEqual(sum(1 for n in names if n.startswith("fingerprints/")), 4)
            self.assertFalse(any(n.lower().endswith((".wav", ".aif", ".aiff", ".flac")) for n in names))
            for path in paths:
                self.assertTrue(path.is_file(), "the originals are only ever read")

            second = scan(relay)
            self.assertEqual(len(relay.sent), 1, "already-contributed presets are not sent again")
            self.assertEqual(second.relay_already_present, 4)
            self.assertEqual(second.relay_uploaded, 0)

    def test_compact_mode_renders_fingerprints_and_deletes_in_small_batches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="patchlab-compact-batches-") as temporary:
            root = Path(temporary)
            linked = root / "linked"
            linked.mkdir()
            database = Database(root / "library.db")
            for index in range(25):
                path = linked / f"Preset {index:02d}.fxp"
                path.write_bytes(b"CcnK" + bytes([index]) * 32)
                preset_id, _ = database.insert_preset(
                    path=path,
                    name=path.stem,
                    synth="serum1",
                    content_hash=sha1_file(path),
                )
                database.replace_params(
                    preset_id,
                    [ParameterValue(0, "Master", 0.5, "50%")],
                    "test",
                )

            batch_sizes: list[int] = []

            def fake_render_library(**kwargs):  # type: ignore[no-untyped-def]
                preset_ids = list(kwargs["preset_ids"])
                batch_sizes.append(len(preset_ids))
                audio_root = Path(kwargs["audio_root"])
                with database.connect() as connection:
                    for preset_id in preset_ids:
                        connection.execute(
                            "UPDATE presets SET status='rendered' WHERE id=?",
                            (preset_id,),
                        )
                        for note in (24, 36, 48, 60, 72, 84, 96):
                            wav = audio_root / str(preset_id) / f"{note}.wav"
                            wav.parent.mkdir(parents=True, exist_ok=True)
                            sf.write(
                                wav,
                                np.ones((128, 2), dtype=np.float32),
                                44_100,
                                subtype="FLOAT",
                                format="WAV",
                            )
                            connection.execute(
                                "INSERT INTO renders VALUES (?,?,?,?,?,?)",
                                (preset_id, note, str(wav), -1.0, -12.0, 5.0),
                            )
                return RenderSummary(
                    selected_presets=len(preset_ids),
                    queued_presets=len(preset_ids),
                    rendered_note_pairs=len(preset_ids) * 7,
                )

            class FakeEmbedder:
                def __init__(self, _env) -> None:  # type: ignore[no-untyped-def]
                    pass

                def embed(self, waveforms):  # type: ignore[no-untyped-def]
                    return np.ones((len(waveforms), 512), dtype=np.float32)

            with (
                patch("core.local_library.FactoryBundle") as factory_bundle,
                patch("core.local_library.render_library", side_effect=fake_render_library),
                patch("core.local_library.ClapEmbedder", FakeEmbedder),
            ):
                factory_bundle.return_value.known_hashes.return_value = set()
                summary = process_linked_folder(
                    linked,
                    db_path=database.path,
                    audio_root=root / "audio",
                    state_dir=root / "states",
                    relay=None,
                    render_processes=1,
                    compact_mode=True,
                    log=lambda _message: None,
                )

            self.assertEqual(batch_sizes, [1] * 25)
            self.assertEqual(summary.fingerprints_created, 25)
            self.assertEqual(summary.compacted_render_files, 25 * 7)
            with database.connect() as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM renders").fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM presets WHERE status='embedded'"
                    ).fetchone()[0],
                    25,
                )

    def test_failed_silent_preset_is_not_rendered_again_on_a_later_link_scan(self) -> None:
        with tempfile.TemporaryDirectory(prefix="patchlab-silent-no-retry-") as temporary:
            root = Path(temporary)
            linked = root / "linked"
            linked.mkdir()
            path = linked / "Silent.fxp"
            path.write_bytes(b"CcnK-silent")
            database = Database(root / "library.db")
            preset_id, _ = database.insert_preset(
                path=path,
                name=path.stem,
                synth="serum1",
                content_hash=sha1_file(path),
            )
            database.replace_params(
                preset_id,
                [ParameterValue(0, "Master", 0.5, "50%")],
                "test",
            )
            database.mark_failed(preset_id, "failed_silent", "silent at every note")

            with (
                patch("core.local_library.FactoryBundle") as factory_bundle,
                patch("core.local_library.render_library") as render_library,
            ):
                factory_bundle.return_value.known_hashes.return_value = set()
                process_linked_folder(
                    linked,
                    db_path=database.path,
                    audio_root=root / "audio",
                    state_dir=root / "states",
                    relay=None,
                    render_processes=1,
                    compact_mode=True,
                    log=lambda _message: None,
                )

            render_library.assert_not_called()


class FakeEmbedder:
    def __init__(self, _env) -> None:  # type: ignore[no-untyped-def]
        pass

    def embed(self, waveforms):  # type: ignore[no-untyped-def]
        return np.ones((len(waveforms), 512), dtype=np.float32)


class FingerprintCatchUpTest(unittest.TestCase):
    """Covers audio rendered outside process_linked_folder -- e.g. the
    standalone render-library job, which never fingerprints on its own."""

    def _rendered_preset_with_no_fingerprint(
        self, database: Database, audio_root: Path, name: str
    ) -> int:
        source_root = audio_root.parent / "linked"
        source_root.mkdir(exist_ok=True)
        source = source_root / f"{name}.fxp"
        source.write_bytes(f"CcnK-{name}".encode())
        preset_id, _ = database.insert_preset(
            path=source,
            name=name,
            synth="serum1",
            content_hash=sha1_file(source),
        )
        reconcile_source_tree(source_root, database)
        database.replace_params(
            preset_id,
            [ParameterValue(0, "Master", 0.5, "50%")],
            "test",
        )
        with database.connect() as connection:
            for note in MIDI_NOTES:
                wav = audio_root / str(preset_id) / f"{note}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    wav,
                    np.ones((128, 2), dtype=np.float32),
                    44_100,
                    subtype="FLOAT",
                    format="WAV",
                )
                connection.execute(
                    "INSERT INTO renders VALUES (?,?,?,?,?,?)",
                    (preset_id, note, str(wav), -1.0, -12.0, 5.0),
                )
            connection.execute(
                "UPDATE presets SET status='rendered' WHERE id=?", (preset_id,)
            )
        return preset_id

    def test_fingerprints_a_preset_rendered_outside_the_linked_folder_pipeline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="patchlab-fingerprint-catchup-") as temporary:
            root = Path(temporary)
            audio_root = root / "audio"
            database = Database(root / "library.db")
            preset_id = self._rendered_preset_with_no_fingerprint(
                database, audio_root, "Standalone Render"
            )

            with (
                patch("core.local_library.ClapEmbedder", FakeEmbedder),
            ):
                summary = fingerprint_pending_presets(
                    database.path, audio_root, log=lambda _m: None, compact_mode=False
                )

            self.assertEqual(summary.fingerprints_created, 1)
            with database.connect() as connection:
                fingerprinted = {
                    row[0]
                    for row in connection.execute(
                        "SELECT preset_id FROM fingerprints WHERE midi_note=0"
                    )
                }
            self.assertEqual(fingerprinted, {preset_id})

    def test_leaves_an_already_fingerprinted_preset_untouched(self) -> None:
        with tempfile.TemporaryDirectory(prefix="patchlab-fingerprint-catchup-") as temporary:
            root = Path(temporary)
            audio_root = root / "audio"
            database = Database(root / "library.db")
            preset_id = self._rendered_preset_with_no_fingerprint(
                database, audio_root, "Already Learned"
            )
            for note in (0, *MIDI_NOTES):
                database.upsert_fingerprint(
                    preset_id,
                    note,
                    np.zeros(512, dtype=np.float32).tobytes(),
                    np.zeros(9, dtype=np.float32).tobytes(),
                )
            self.assertTrue(mark_preset_prepared(database, preset_id))

            embed_calls: list[int] = []

            class CountingEmbedder(FakeEmbedder):
                def embed(self, waveforms):  # type: ignore[no-untyped-def]
                    embed_calls.append(len(waveforms))
                    return super().embed(waveforms)

            with patch("core.local_library.ClapEmbedder", CountingEmbedder):
                summary = fingerprint_pending_presets(
                    database.path, audio_root, log=lambda _m: None, compact_mode=False
                )

            self.assertEqual(summary.fingerprints_created, 0)
            self.assertEqual(embed_calls, [])

    def test_nothing_pending_is_a_cheap_no_op(self) -> None:
        with tempfile.TemporaryDirectory(prefix="patchlab-fingerprint-catchup-") as temporary:
            root = Path(temporary)
            database = Database(root / "library.db")

            with patch("core.local_library.ClapEmbedder") as embedder_cls:
                summary = fingerprint_pending_presets(
                    database.path, root / "audio", log=lambda _m: None
                )

            embedder_cls.assert_not_called()
            self.assertEqual(summary.fingerprints_created, 0)


class AutoScanThrottleTest(unittest.TestCase):
    def test_due_when_no_marker_exists_yet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = SimpleNamespace(app_data_dir=Path(temporary))
            self.assertTrue(auto_scan_due(env))

    def test_not_due_again_within_the_same_day(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = SimpleNamespace(app_data_dir=Path(temporary))
            record_auto_scan(env, at=datetime.now(timezone.utc))
            self.assertFalse(auto_scan_due(env))

    def test_due_again_after_the_interval_elapses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = SimpleNamespace(app_data_dir=Path(temporary))
            record_auto_scan(
                env, at=datetime.now(timezone.utc) - timedelta(hours=25)
            )
            self.assertTrue(auto_scan_due(env))

    def test_a_corrupt_marker_fails_open_to_due(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = SimpleNamespace(app_data_dir=Path(temporary))
            marker = Path(temporary) / "last-auto-link-scan.json"
            marker.write_text("not json", encoding="utf-8")
            self.assertTrue(auto_scan_due(env))


if __name__ == "__main__":
    unittest.main()
