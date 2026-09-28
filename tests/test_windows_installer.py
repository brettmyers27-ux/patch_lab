from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_installer_embeds_a_frozen_payload_without_dev_prerequisites() -> None:
    setup = (ROOT / "packaging" / "windows" / "PatchLab.iss").read_text(encoding="utf-8")
    build = (ROOT / "packaging" / "build_windows_installer.ps1").read_text(encoding="utf-8")

    assert "AppPayload" in setup
    assert 'Source: "{#AppPayload}\\*"' in setup
    assert "PatchLab.exe" in setup
    assert "Python.Python" not in setup
    assert "Git.Git" not in setup
    assert "PatchLab-source.bundle" not in setup
    assert "-m PyInstaller" in build
    assert "packaged-runtime-gate" in build
    assert 'minimum_windows = "Windows 10 x64 build 19041"' in build
    assert "bundle create" not in build


def test_windows_release_build_refuses_tracked_secrets() -> None:
    build = (ROOT / "packaging" / "build_windows_installer.ps1").read_text(encoding="utf-8")

    assert "data|private|\\.venv|gcloud|relay-credentials" in build
    assert "Potential credential material" in build
