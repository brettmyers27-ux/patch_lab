"""Render Sound Library must resolve its data locations the same way in a
packaged app as ``scripts/process_local_library.py`` already does -- never a
dev-checkout path (``data/library.db``, ``data/audio``, the shipped-model
``data/models/serum2_render_states``) that does not exist inside a frozen
app's Resources.

``--db``/``--audio-root``/``--state-dir`` remain available for tests and gate
scripts that want a disposable location; the worker invoked *without* them,
in distribution mode, must land on ``core.local_library.default_local_paths``
-- the one place every other per-user-data worker already resolves from --
never on ``core.db.DEFAULT_DB_PATH`` et al.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import scripts.render_library as render_library_script
from core.db import DEFAULT_DB_PATH
from core.render import DEFAULT_AUDIO_ROOT, RenderSummary


def _args(db=None, audio_root=None, state_dir=None) -> argparse.Namespace:
    return argparse.Namespace(db=db, audio_root=audio_root, state_dir=state_dir)


# --- _resolve_local_paths: the actual decision under test -------------------


def test_distribution_mode_with_no_args_uses_the_packaged_per_user_paths(monkeypatch) -> None:
    packaged = {"db": Path("/appdata/library.db"), "audio": Path("/appdata/audio"),
                "states": Path("/appdata/serum2-render-states")}
    monkeypatch.setattr(render_library_script, "distribution_mode", lambda: True)
    monkeypatch.setattr("core.local_library.default_local_paths", lambda: packaged)

    db, audio, state = render_library_script._resolve_local_paths(_args())
    assert (db, audio, state) == (packaged["db"], packaged["audio"], packaged["states"])


def test_distribution_mode_never_falls_back_to_the_dev_checkout_paths(monkeypatch) -> None:
    packaged = {"db": Path("/appdata/library.db"), "audio": Path("/appdata/audio"),
                "states": Path("/appdata/serum2-render-states")}
    monkeypatch.setattr(render_library_script, "distribution_mode", lambda: True)
    monkeypatch.setattr("core.local_library.default_local_paths", lambda: packaged)

    db, audio, state = render_library_script._resolve_local_paths(_args())
    for path in (db, audio, state):
        assert "data" not in path.parts or path.parts != DEFAULT_DB_PATH.parts
    assert db != DEFAULT_DB_PATH
    assert audio != DEFAULT_AUDIO_ROOT


def test_explicit_args_win_even_in_distribution_mode(monkeypatch) -> None:
    monkeypatch.setattr(render_library_script, "distribution_mode", lambda: True)
    monkeypatch.setattr(
        "core.local_library.default_local_paths",
        lambda: {"db": Path("/appdata/library.db"), "audio": Path("/appdata/audio"),
                  "states": Path("/appdata/states")},
    )
    chosen = _args(db=Path("/custom/lib.db"), audio_root=Path("/custom/audio"),
                   state_dir=Path("/custom/states"))
    result = render_library_script._resolve_local_paths(chosen)
    assert result == (Path("/custom/lib.db"), Path("/custom/audio"), Path("/custom/states"))


def test_dev_checkout_keeps_its_existing_convenience_paths(monkeypatch) -> None:
    monkeypatch.setattr(render_library_script, "distribution_mode", lambda: False)
    db, audio, state = render_library_script._resolve_local_paths(_args())
    assert (db, audio, state) == (DEFAULT_DB_PATH, DEFAULT_AUDIO_ROOT, None)


def test_dev_checkout_explicit_args_still_win(monkeypatch) -> None:
    monkeypatch.setattr(render_library_script, "distribution_mode", lambda: False)
    chosen = _args(db=Path("/x/lib.db"), audio_root=Path("/x/audio"), state_dir=Path("/x/states"))
    assert render_library_script._resolve_local_paths(chosen) == (
        Path("/x/lib.db"), Path("/x/audio"), Path("/x/states"),
    )


# --- the worker's main(), invoked exactly as the GUI/CLI would --------------


@pytest.fixture
def fake_render(monkeypatch, tmp_path):
    """Replace the expensive real renderer; capture what paths it received."""

    captured = {}

    def fake(*, db_path, audio_root, state_dir, **_kwargs):
        captured.update(db_path=db_path, audio_root=audio_root, state_dir=state_dir)
        return RenderSummary()

    monkeypatch.setattr(render_library_script, "render_library", fake)
    db_path = tmp_path / "empty.db"
    from core.db import Database

    Database(db_path).migrate()
    return captured, db_path


def test_the_worker_invoked_without_state_dir_in_a_packaged_app_uses_packaged_paths(
    monkeypatch, fake_render, tmp_path
) -> None:
    """The exact scenario the diagnostic pass flagged: no --state-dir, frozen app."""

    captured, _db = fake_render
    packaged = {"db": tmp_path / "packaged-lib.db", "audio": tmp_path / "packaged-audio",
                "states": tmp_path / "packaged-states"}
    from core.db import Database

    Database(packaged["db"]).migrate()
    monkeypatch.setattr(render_library_script, "distribution_mode", lambda: True)
    monkeypatch.setattr("core.local_library.default_local_paths", lambda: packaged)
    monkeypatch.setattr(
        "sys.argv", ["render_library.py"]  # no --db, no --audio-root, no --state-dir
    )
    render_library_script.main()
    assert captured["db_path"] == packaged["db"]
    assert captured["audio_root"] == packaged["audio"]
    assert captured["state_dir"] == packaged["states"]
    assert captured["db_path"] != DEFAULT_DB_PATH
    assert captured["state_dir"] is not None, "never the silent None -> dev-checkout fallback"


def test_the_worker_with_an_explicit_state_dir_still_honors_it(monkeypatch, fake_render, tmp_path) -> None:
    captured, db = fake_render
    monkeypatch.setattr(render_library_script, "distribution_mode", lambda: True)
    custom_states = tmp_path / "custom-states"
    monkeypatch.setattr(
        "sys.argv",
        ["render_library.py", "--db", str(db), "--audio-root", str(tmp_path / "audio"),
         "--state-dir", str(custom_states)],
    )
    render_library_script.main()
    assert captured["state_dir"] == custom_states


def test_the_real_default_local_paths_resolver_is_what_gets_called(monkeypatch, fake_render, tmp_path) -> None:
    """Confirms main() defers to core.local_library.default_local_paths itself
    (not a private copy of its logic), using an explicit env so this does not
    depend on the process-wide ENV singleton bound at import time."""

    from types import SimpleNamespace

    from core.local_library import default_local_paths

    captured, _db = fake_render
    isolated = tmp_path / "isolated-app-data"
    isolated.mkdir()
    real_paths = default_local_paths(env=SimpleNamespace(app_data_dir=isolated))
    from core.db import Database

    Database(real_paths["db"]).migrate()

    monkeypatch.setattr(render_library_script, "distribution_mode", lambda: True)
    monkeypatch.setattr("core.local_library.default_local_paths", lambda: real_paths)
    monkeypatch.setattr("sys.argv", ["render_library.py"])
    render_library_script.main()

    for path in captured.values():
        assert str(path).startswith(str(isolated))
    assert "PatchLab/soundmatch/data" not in str(captured["db_path"])
