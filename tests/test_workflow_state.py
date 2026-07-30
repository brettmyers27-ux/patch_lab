from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from core.db import Database
from core.model_assets import ModelAssetsError
from core.privacy import PrivacyChoice
from core.render import MIDI_NOTES
from core.workflow_state import WorkflowActivity, resolve_workflow_state


def _factory_bundle(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE presets (id INTEGER PRIMARY KEY, searchable INTEGER);
        CREATE TABLE preset_embeddings (
          preset_id INTEGER PRIMARY KEY,
          embedding_f16 BLOB NOT NULL
        );
        INSERT INTO presets(id,searchable) VALUES (1,1);
        """
    )
    connection.execute(
        "INSERT INTO preset_embeddings(preset_id,embedding_f16) VALUES (?,?)",
        (1, bytes(512 * 2)),
    )
    connection.commit()
    connection.close()
    return path


def _resolve(
    tmp_path: Path,
    privacy: PrivacyChoice,
    *,
    audio_selected: bool = False,
    activities: dict[str, WorkflowActivity] | None = None,
):
    with patch("core.workflow_state.validate_model_assets"):
        return resolve_workflow_state(
            privacy=privacy,
            local_database_path=tmp_path / "app-data" / "library.db",
            factory_bundle_path=_factory_bundle(tmp_path / "factory.sqlite"),
            audio_selected=audio_selected,
            activities=activities,
        )


def test_fresh_install_ignores_populated_developer_data(tmp_path: Path) -> None:
    # A populated lookalike repository database must have no influence because
    # only the explicit per-user app-data path is passed to the resolver.
    developer_db = Database(tmp_path / "repo" / "data" / "library.db")
    preset_id, _ = developer_db.insert_preset(
        path=tmp_path / "repo-preset.fxp",
        name="Repo preset",
        synth="serum1",
        content_hash="repo-only",
    )
    with developer_db.connect() as connection:
        connection.execute(
            "UPDATE presets SET status='rendered' WHERE id=?", (preset_id,)
        )

    state = _resolve(tmp_path, PrivacyChoice(None, None))

    assert state.link.phase == "needs-action"
    assert state.render.phase == "not-required"
    assert state.analyze.phase == "complete"
    assert state.match.phase == "needs-action"
    assert state.match.text == "Ready · choose an audio file"


def test_linked_folder_counts_real_remaining_renders(tmp_path: Path) -> None:
    linked = tmp_path / "My Presets"
    linked.mkdir()
    database = Database(tmp_path / "app-data" / "library.db")
    ids: list[int] = []
    for index in range(3):
        preset_id, _ = database.insert_preset(
            path=linked / f"Preset {index}.fxp",
            name=f"Preset {index}",
            synth="serum1",
            content_hash=f"linked-{index}",
        )
        ids.append(preset_id)
    with database.connect() as connection:
        connection.executemany(
            "INSERT INTO renders(preset_id,midi_note,wav_path,peak_dbfs,rms_dbfs,duration_s) "
            "VALUES (?,?,?,?,?,?)",
            [
                (ids[0], note, str(tmp_path / f"{note}.wav"), -1.0, -12.0, 5.0)
                for note in MIDI_NOTES
            ],
        )

    state = _resolve(tmp_path, PrivacyChoice(True, str(linked)))

    assert state.link.phase == "complete"
    assert "My Presets" in state.link.text
    assert "3 presets" in state.link.text
    assert state.render.phase == "needs-action"
    assert state.render.text == "2 of 3 presets still need rendering"
    assert (state.render.current, state.render.total) == (1, 3)


def test_live_activity_and_broken_cache_are_explicit(tmp_path: Path) -> None:
    activities = {
        "render": WorkflowActivity(14, 70, "Rendering 14 of 70 notes")
    }
    state = _resolve(
        tmp_path,
        PrivacyChoice(None, None),
        activities=activities,
    )
    assert state.render.phase == "in-progress"
    assert (state.render.current, state.render.total) == (14, 70)

    with patch(
        "core.workflow_state.validate_model_assets",
        side_effect=ModelAssetsError(
            "the required roberta tokenizer/model cache is incomplete"
        ),
    ):
        broken = resolve_workflow_state(
            privacy=PrivacyChoice(None, None),
            local_database_path=tmp_path / "app-data" / "library.db",
            factory_bundle_path=_factory_bundle(tmp_path / "factory-2.sqlite"),
            audio_selected=False,
        )
    assert broken.match.phase == "needs-action"
    assert broken.match.text == "Tokenizer cache missing · reinstall PatchLab"
