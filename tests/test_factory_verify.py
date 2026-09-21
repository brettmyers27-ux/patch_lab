from pathlib import Path

from core.factory_verify import _factory_files


def test_factory_files_selects_only_requested_regular_presets(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    expected = nested / "fixture.SerumPreset"
    expected.write_bytes(b"fixture")
    (nested / "ignore.fxp").write_bytes(b"other")
    (nested / "directory.SerumPreset").mkdir()

    assert _factory_files(tmp_path, ".serumpreset") == {expected.resolve()}
