"""core.preset_output: the one resolver every generated-preset writer shares.

Covers test matrix C/D/O/P from the diagnostic pass: the setting survives a
restart, an unset preference reproduces 1.5.5's exact default, a custom
folder applies to both Serum generations, and an unwritable folder is
rejected before it is ever saved.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.preset_output import (
    MACOS_SERUM1_USER_PRESETS,
    MACOS_SERUM2_USER_PRESETS,
    PresetOutputPreferences,
    configured_preset_output_folder,
    default_preset_output_root,
    ensure_writable,
    load_preset_output_preferences,
    save_preset_output_preferences,
    settings_path,
)


def _env(tmp_path: Path, **overrides):
    base = dict(
        app_data_dir=tmp_path, branch="macos",
        preset_roots=(), existing_preset_roots=(),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- P. default/unset preserves exactly today's (1.5.5) behavior -----------


def test_P_unset_preference_matches_the_macos_default(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert default_preset_output_root("serum1", env=env) == MACOS_SERUM1_USER_PRESETS
    assert default_preset_output_root("serum2", env=env) == MACOS_SERUM2_USER_PRESETS
    assert configured_preset_output_folder("serum1", env=env) == MACOS_SERUM1_USER_PRESETS / "PatchLab"
    assert configured_preset_output_folder("serum2", env=env) == MACOS_SERUM2_USER_PRESETS / "PatchLab"


def test_P_missing_or_corrupt_settings_file_is_the_same_as_unset(tmp_path: Path) -> None:
    env = _env(tmp_path)
    assert load_preset_output_preferences(env) == PresetOutputPreferences()
    settings_path(env).parent.mkdir(parents=True, exist_ok=True)
    settings_path(env).write_text("not json")
    assert load_preset_output_preferences(env) == PresetOutputPreferences()
    assert configured_preset_output_folder("serum2", env=env) == MACOS_SERUM2_USER_PRESETS / "PatchLab"


def test_non_macos_default_prefers_an_existing_writable_root_under_home(tmp_path: Path) -> None:
    home_root = tmp_path / "home" / "Documents" / "Xfer" / "Serum 2 Presets"
    home_root.mkdir(parents=True)
    env = _env(
        tmp_path, branch="windows",
        preset_roots=(home_root,), existing_preset_roots=(home_root,),
    )
    import unittest.mock

    with unittest.mock.patch("pathlib.Path.home", return_value=tmp_path / "home"):
        assert default_preset_output_root("serum2", env=env) == home_root


# --- O. the setting survives a restart (a fresh load reads the same value) --


def test_O_the_setting_survives_a_restart(tmp_path: Path) -> None:
    env = _env(tmp_path)
    save_preset_output_preferences(PresetOutputPreferences(folder="/Users/x/Music/PatchLab Presets"), env)
    reloaded = load_preset_output_preferences(env)  # a fresh read, as at the next launch
    assert reloaded.folder == "/Users/x/Music/PatchLab Presets"


def test_the_write_is_atomic_no_partial_file_is_ever_visible(tmp_path: Path) -> None:
    env = _env(tmp_path)
    save_preset_output_preferences(PresetOutputPreferences(folder="/a/b"), env)
    assert list(tmp_path.glob(".*tmp")) == []
    assert settings_path(env).is_file()


# --- one folder for BOTH Serum generations, once configured ----------------


def test_a_custom_folder_applies_to_both_serum_generations(tmp_path: Path) -> None:
    env = _env(tmp_path)
    custom = tmp_path / "Music" / "PatchLab Presets"
    save_preset_output_preferences(PresetOutputPreferences(folder=str(custom)), env)
    assert configured_preset_output_folder("serum1", env=env) == custom
    assert configured_preset_output_folder("serum2", env=env) == custom


def test_a_custom_folder_is_used_exactly_as_chosen_no_extra_subfolder(tmp_path: Path) -> None:
    env = _env(tmp_path)
    custom = tmp_path / "Wherever I Want"
    save_preset_output_preferences(PresetOutputPreferences(folder=str(custom)), env)
    assert configured_preset_output_folder("serum2", env=env) == custom
    assert configured_preset_output_folder("serum2", env=env).name != "PatchLab"


def test_an_env_override_wins_over_the_saved_preference(tmp_path: Path, monkeypatch) -> None:
    env = _env(tmp_path)
    save_preset_output_preferences(PresetOutputPreferences(folder=str(tmp_path / "saved")), env)
    monkeypatch.setenv("PATCHLAB_PRESET_OUTPUT_FOLDER", str(tmp_path / "override"))
    assert configured_preset_output_folder("serum1", env=env) == tmp_path / "override"


# --- D. an unwritable folder is rejected, cleanly, before anything is saved -


def test_D_ensure_writable_accepts_a_real_folder_and_leaves_no_trace(tmp_path: Path) -> None:
    folder = tmp_path / "Presets"
    assert ensure_writable(folder) == ""
    assert folder.is_dir()
    assert list(folder.iterdir()) == [], "the write-check probe must not be left behind"


def test_D_ensure_writable_rejects_a_read_only_folder(tmp_path: Path) -> None:
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        reason = ensure_writable(folder)
    finally:
        folder.chmod(0o700)
    assert reason != ""


def test_D_ensure_writable_rejects_a_path_that_is_actually_a_file(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-folder"
    blocked.write_text("x")
    assert ensure_writable(blocked) != ""
