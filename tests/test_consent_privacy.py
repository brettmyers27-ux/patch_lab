"""Turning "Use & share my own presets" OFF must make the app's behaviour match its consent text.

While OFF, PatchLab must not scan, analyse, render, match against or upload the
user's own presets (rows with ``is_factory=0``), yet keep every byte of their
stored data so switching it back ON resumes.  Factory presets never depend on it.

Matrix: A fresh decline · B existing user OFF · C re-enable · D restart ·
E match cache · F render queue · G contribution · H factory functionality ·
in-flight withdrawal.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core import local_library as library
from core.db import Database
from core.factory_match import _local_search_rows
from core.library_state import mark_preset_prepared, reconcile_source_tree
from core.plugin_host import ParameterValue
from core.prepared_state import REQUIRED_FINGERPRINT_NOTES
from core.preset_scan import sha1_file
from core.privacy import (
    PrivacyStore,
    UserPresetsDisabled,
    require_user_presets,
    set_active_store,
    user_presets_enabled,
)
from core.render import RenderSummary, _select_records
from core.submission_upload import BUG_REPORT, PRESET_CONTRIBUTION, UploadError, upload_file


@pytest.fixture()
def consent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A distribution-mode app whose privacy file is the only thing that changes."""

    settings = tmp_path / "privacy-settings.json"
    monkeypatch.setenv("PATCHLAB_DISTRIBUTION_MODE", "1")
    monkeypatch.setenv("PATCHLAB_PRIVACY_SETTINGS", str(settings))
    set_active_store(None)

    class Handle:
        path = settings

        @staticmethod
        def set(value, *, folder: Path | None = None) -> None:
            PrivacyStore().save(value, linked_folder=folder)

    yield Handle
    set_active_store(None)


def _add(database: Database, folder: Path, name: str, *, factory: bool, rendered: bool = True) -> int:
    path = folder / f"{name}.fxp"
    path.write_bytes(b"CcnK" + name.encode() * 4)
    preset_id, _ = database.insert_preset(path=path, name=name, synth="serum1", content_hash=sha1_file(path))
    reconcile_source_tree(folder, database)
    database.replace_params(preset_id, [ParameterValue(0, "Master", 0.5, "50%")], "test")
    database.set_factory_status(preset_id, factory)
    if rendered:
        for note in REQUIRED_FINGERPRINT_NOTES:
            database.upsert_fingerprint(
                preset_id,
                note,
                np.ones(512, dtype=np.float32).tobytes(),
                np.zeros(9, dtype=np.float32).tobytes(),
            )
        with database.connect() as connection:
            connection.execute("UPDATE presets SET status='rendered' WHERE id=?", (preset_id,))
        assert mark_preset_prepared(database, preset_id)
    return preset_id


@pytest.fixture()
def library_db(tmp_path: Path):
    folder = tmp_path / "linked"
    folder.mkdir()
    database = Database(tmp_path / "library.db")
    ids = {
        "user": _add(database, folder, "MyLead", factory=False),
        "factory": _add(database, folder, "FactoryPad", factory=True),
    }
    return database, folder, ids


def _rows(database: Database) -> list[tuple]:
    with sqlite3.connect(database.path) as connection:
        return connection.execute(
            "SELECT id,path,name,content_hash,status,is_factory FROM presets ORDER BY id"
        ).fetchall()


# --- the single authoritative state -----------------------------------------


def test_the_state_fails_closed(consent) -> None:
    assert user_presets_enabled() is False, "no answer yet is OFF"
    consent.set(True)
    assert user_presets_enabled() is True
    consent.set(False)
    assert user_presets_enabled() is False
    consent.path.write_text("{ not json")
    assert user_presets_enabled() is False, "a damaged file is OFF"
    consent.path.write_text(json.dumps({"use_and_share_own_presets": "yes"}))
    assert user_presets_enabled() is False, "only an explicit true is ON"
    with pytest.raises(UserPresetsDisabled):
        require_user_presets("anything")


def test_developer_builds_do_not_apply_distribution_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PATCHLAB_DISTRIBUTION_MODE", raising=False)
    assert user_presets_enabled() is True


# --- A. fresh decline --------------------------------------------------------


def test_A_fresh_decline_scans_nothing(consent, tmp_path: Path) -> None:
    consent.set(False)
    linked = tmp_path / "MyPresets"
    linked.mkdir()
    (linked / "Secret.fxp").write_bytes(b"CcnK-user")
    messages: list[str] = []
    with (
        patch.object(library, "discover_presets", side_effect=AssertionError("the folder must not be walked")),
        patch.object(library, "sha1_file", side_effect=AssertionError("no user preset may be read")),
        patch.object(library, "render_library", side_effect=AssertionError("no rendering")),
    ):
        summary = library.process_linked_folder(
            linked, db_path=tmp_path / "library.db", audio_root=tmp_path / "audio",
            state_dir=tmp_path / "states", relay=MagicMock(), log=messages.append,
        )
    assert summary.user_presets_disabled is True and summary.found == 0
    assert not (tmp_path / "library.db").exists(), "not even a database was created"
    assert any("turned off" in m for m in messages)


def test_A_declined_user_is_never_asked_about_pending_work(consent, library_db) -> None:
    database, _folder, _ids = library_db
    consent.set(False)
    with database.connect() as connection:
        connection.execute("UPDATE presets SET pending_reason='serum2_not_installed', compatible_renderers='serum2' WHERE is_factory=0")
    assert database.presets_needing_generation("serum2") == []
    assert database.pending_presets() == []


# --- B. existing user turns it OFF ------------------------------------------


def test_B_off_makes_stored_presets_inactive_but_deletes_nothing(consent, library_db) -> None:
    database, _folder, ids = library_db
    consent.set(True)
    before = _rows(database)
    assert {r.id for r in database.renderable_presets()} == {ids["user"], ids["factory"]}
    assert {r.id for r in database.presets_with_status(["rendered"])} == {ids["user"], ids["factory"]}
    consent.set(False)
    assert {r.id for r in database.renderable_presets()} == {ids["factory"]}
    assert {r.id for r in database.presets_with_status(["rendered"])} == {ids["factory"]}
    assert _rows(database) == before, "OFF never deletes or edits stored data"
    with sqlite3.connect(database.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0] == (
            2 * len(REQUIRED_FINGERPRINT_NOTES)
        )


def test_B_off_stops_scan_render_and_upload_entry_points(consent, library_db, tmp_path: Path) -> None:
    database, folder, _ids = library_db
    consent.set(False)
    relay = MagicMock()
    with patch.object(library, "render_library", side_effect=AssertionError("no rendering")):
        pending = library.process_pending_for_generation(
            "serum2", db_path=database.path, audio_root=tmp_path / "a", state_dir=tmp_path / "s", log=lambda _m: None
        )
        learned = library.fingerprint_pending_presets(database.path, tmp_path / "a", log=lambda _m: None)
        scanned = library.process_linked_folder(
            folder, db_path=database.path, audio_root=tmp_path / "a", state_dir=tmp_path / "s",
            relay=relay, log=lambda _m: None,
        )
    assert pending.user_presets_disabled and learned.user_presets_disabled and scanned.user_presets_disabled
    relay.post_submission.assert_not_called()


# --- C. re-enable ------------------------------------------------------------


def test_C_turning_it_back_on_makes_existing_data_usable_again(consent, library_db) -> None:
    database, _folder, ids = library_db
    consent.set(False)
    assert {r.id for r in database.renderable_presets()} == {ids["factory"]}
    assert _local_search_rows(database.path)[1] == []
    consent.set(True)
    assert {r.id for r in database.renderable_presets()} == {ids["user"], ids["factory"]}
    matrix, rows = _local_search_rows(database.path)
    assert [r["preset_id"] for r in rows] == [ids["user"]] and matrix is not None


def test_C_pending_presets_resume_after_re_enabling(consent, library_db) -> None:
    database, _folder, ids = library_db
    with database.connect() as connection:
        connection.execute(
            "UPDATE presets SET pending_reason='serum2_not_installed', compatible_renderers='serum2' WHERE id=?",
            (ids["user"],),
        )
    consent.set(False)
    assert database.presets_needing_generation("serum2") == []
    consent.set(True)
    assert [r.id for r in database.presets_needing_generation("serum2")] == [ids["user"]]


# --- D. restart ---------------------------------------------------------------


def test_D_the_choice_survives_a_restart_and_starts_nothing(consent, library_db) -> None:
    database, _folder, _ids = library_db
    consent.set(False)
    fresh_process_view = PrivacyStore()  # what a relaunch reads
    assert fresh_process_view.load().use_and_share_own_presets is False
    assert user_presets_enabled() is False
    assert _local_search_rows(database.path) == (None, [])


# --- E. match candidate pool / cache ----------------------------------------


def test_E_match_pool_cannot_contain_user_presets_after_off(consent, library_db) -> None:
    database, _folder, ids = library_db
    consent.set(True)
    _matrix, rows = _local_search_rows(database.path)
    assert [r["preset_id"] for r in rows] == [ids["user"]], "sanity: eligible while ON"
    consent.set(False)
    for _ in range(3):  # every later Match, however it was reached
        assert _local_search_rows(database.path) == (None, [])
        assert _local_search_rows(None) == (None, [])


def test_E_user_render_states_are_not_consulted_while_off(consent, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from core import synthesis_assets

    monkeypatch.delenv("PATCHLAB_SERUM2_RENDER_STATES", raising=False)
    consent.set(True)
    with_user = synthesis_assets.resolve_synthesis_assets().render_state_roots
    consent.set(False)
    without_user = synthesis_assets.resolve_synthesis_assets().render_state_roots
    assert len(with_user) == 2 and len(without_user) == 1 and without_user[0] == with_user[0]


# --- F. render queue ---------------------------------------------------------


def test_F_a_queued_user_preset_is_not_rendered_when_consent_is_off(consent, library_db) -> None:
    database, _folder, ids = library_db
    consent.set(False)
    chosen = _select_records(database, [ids["user"], ids["factory"]])
    assert [r.id for r in chosen] == [ids["factory"]], "the queued user preset is inactive, and not an error"
    consent.set(True)
    assert {r.id for r in _select_records(database, [ids["user"], ids["factory"]])} == {ids["user"], ids["factory"]}


def test_F_on_unknown_ids_are_still_an_error(consent, library_db) -> None:
    database, _folder, _ids = library_db
    consent.set(True)
    with pytest.raises(KeyError):
        _select_records(database, [99999])


# --- G. contribution ----------------------------------------------------------


class _Relay:
    base_url = "https://relay.invalid"

    def __init__(self) -> None:
        self.calls = 0

    def post_submission(self, **kwargs):
        self.calls += 1
        path = Path(kwargs["path"])
        return {"ok": True, "submission_id": kwargs["submission_id"], "receipt_id": "r" * 24,
                "sha256": kwargs["sha256"], "size": path.stat().st_size}


def test_G_a_preset_contribution_cannot_be_uploaded_while_off(consent, tmp_path: Path) -> None:
    consent.set(False)
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK\x03\x04" + b"x" * 64)
    relay = _Relay()
    with pytest.raises(UploadError) as raised:
        upload_file(relay, kind=PRESET_CONTRIBUTION, submission_id="a" * 32, path=bundle, version="1.5.4")
    assert raised.value.code == "consent_off" and relay.calls == 0


def test_G_bug_reports_are_unaffected_by_the_preset_choice(consent, tmp_path: Path) -> None:
    consent.set(False)
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"PK\x03\x04" + b"x" * 64)
    relay = _Relay()
    result = upload_file(relay, kind=BUG_REPORT, submission_id="b" * 32, path=bundle, version="1.5.4")
    assert relay.calls == 1 and result.receipt_id


def test_G_contributing_stops_when_consent_is_withdrawn_mid_run(consent, library_db, tmp_path: Path) -> None:
    database, folder, ids = library_db
    consent.set(True)
    relay = _Relay()
    summary = library.LocalLibrarySummary()
    rows = [dict(id=ids["user"], content_hash=sha1_file(folder / "MyLead.fxp"), is_factory=0)]

    def withdraw(item):
        consent.set(False)
        return {"content_hash": item.content_hash}

    with patch.object(library, "_fingerprint_payload", side_effect=lambda _db, _id: withdraw(MagicMock(content_hash="h"))):
        with pytest.raises(UserPresetsDisabled):
            library._contribute_presets(
                database=database, relay=relay, root=folder, rows=rows,
                id_to_path={ids["user"]: folder / "MyLead.fxp"}, ledger_path=tmp_path / "ledger.json",
                summary=summary, log=lambda _m: None, sleep=lambda _s: None,
            )
    assert relay.calls == 0


# --- H. factory functionality -------------------------------------------------


def test_H_factory_presets_are_unaffected_by_the_choice(consent, library_db) -> None:
    database, _folder, ids = library_db
    consent.set(False)
    assert ids["factory"] in {r.id for r in database.renderable_presets()}
    assert ids["factory"] in {r.id for r in database.presets_with_status(["rendered"])}
    assert database.renderable_presets("serum1")[0].id == ids["factory"]


# --- in-flight withdrawal -----------------------------------------------------


def test_withdrawing_consent_between_render_batches_stops_further_user_work(consent, tmp_path: Path) -> None:
    folder = tmp_path / "linked"
    folder.mkdir()
    database = Database(tmp_path / "library.db")
    for index in range(30):  # two compact batches of 24 and 6
        _add(database, folder, f"User{index:02d}", factory=False, rendered=False)
    consent.set(True, folder=folder)
    batches: list[list[int]] = []

    def fake_render(**kwargs):
        batches.append(list(kwargs["preset_ids"]))
        consent.set(False)  # the user flips the switch while batch 1 is running
        return RenderSummary(selected_presets=len(kwargs["preset_ids"]))

    with (
        patch.object(library, "FactoryBundle") as bundle,
        patch.object(library, "render_library", side_effect=fake_render),
        patch.object(library, "ClapEmbedder", return_value=MagicMock()),
    ):
        bundle.return_value.known_hashes.return_value = set()
        summary = library.process_linked_folder(
            folder, db_path=database.path, audio_root=tmp_path / "audio", state_dir=tmp_path / "states",
            relay=_Relay(), render_processes=1, compact_mode=True, log=lambda _m: None,
        )
    assert len(batches) == 1 and len(batches[0]) == 24, "the second batch never started"
    assert summary.user_presets_disabled is True
    assert len(_rows(database)) == 30, "nothing was deleted"
