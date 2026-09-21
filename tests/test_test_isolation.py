"""The suite's own isolation guarantee."""

from __future__ import annotations

from pathlib import Path

from core.platform_env import ENV


def test_app_data_is_a_throwaway_directory() -> None:
    real = Path.home() / "Library" / "Application Support" / "Patch Lab"
    assert Path(ENV.app_data_dir).resolve() != real.resolve()
    assert "patchlab-test-appdata" in str(ENV.app_data_dir)


def test_default_paths_all_resolve_inside_the_sandbox() -> None:
    from core.local_library import default_local_paths

    sandbox = Path(ENV.app_data_dir).resolve()
    for name, path in default_local_paths().items():
        if name == "audio":
            continue  # may follow a user-configured external volume
        assert str(Path(path).resolve()).startswith(str(sandbox)), (name, path)
