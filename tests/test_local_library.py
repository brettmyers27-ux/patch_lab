from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.db import Database
from core.local_library import process_linked_folder
from core.plugin_host import ParameterValue
from core.preset_scan import sha1_file
from core.render import RenderSummary


class FailingRelay:
    def __init__(self) -> None:
        self.checks: list[str] = []
        self.upload_attempts: list[str] = []

    def check_hash(self, content_hash: str) -> bool:
        self.checks.append(content_hash)
        return False

    def upload(
        self,
        *,
        preset_path: Path,
        relative_path: str,
        content_hash: str,
        fingerprint: dict,
    ) -> None:
        del preset_path, relative_path, fingerprint
        self.upload_attempts.append(content_hash)
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
                database.replace_params(
                    preset_id,
                    [ParameterValue(0, "Master", 0.5, "50%")],
                    "test",
                )
                database.upsert_fingerprint(
                    preset_id,
                    0,
                    np.zeros(512, dtype=np.float32).tobytes(),
                    np.zeros(10, dtype=np.float32).tobytes(),
                )
                with database.connect() as connection:
                    connection.execute(
                        "UPDATE presets SET status='rendered' WHERE id=?",
                        (preset_id,),
                    )

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
                )

            self.assertEqual(summary.searchable_local, 4)
            self.assertEqual(summary.relay_upload_failed, 3)
            self.assertEqual(summary.relay_disabled_after_failures, 1)
            self.assertEqual(summary.relay_uploaded, 0)
            self.assertEqual(len(relay.upload_attempts), 3)
            self.assertTrue(
                any(
                    "Relay upload skipped (will retry next scan)" in message
                    for message in messages
                )
            )
            self.assertTrue(
                any(
                    "Relay disabled for the remainder of this scan" in message
                    for message in messages
                )
            )
            self.assertTrue(messages[-1].startswith("LOCAL_LIBRARY_SUMMARY="))


if __name__ == "__main__":
    unittest.main()
