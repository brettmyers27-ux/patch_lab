from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_source_bundle_uses_canonical_installer() -> None:
    installer = (ROOT / "install.ps1").read_text(encoding="utf-8")
    bootstrap = (ROOT / "packaging" / "windows" / "bootstrap.ps1").read_text(
        encoding="utf-8"
    )
    setup = (ROOT / "packaging" / "windows" / "PatchLab.iss").read_text(
        encoding="utf-8"
    )

    assert "PATCHLAB_REPO_BUNDLE" in installer
    assert "git.exe clone --branch main $cloneSource $InstallRoot" in installer
    assert "git.exe -C $InstallRoot fetch $RepoBundle main" in installer
    assert "git.exe -C $InstallRoot merge --ff-only FETCH_HEAD" in installer
    assert "remote set-url origin $RepoUrl" in installer
    assert "PATCHLAB_REPO_BUNDLE" in bootstrap
    assert "-File $InstallScript" in bootstrap
    assert "install.ps1" in setup
    assert "PatchLab-source.bundle" in setup
    assert "Uninstallable=no" in setup


def test_windows_installer_is_a_thin_public_safe_bootstrapper() -> None:
    setup = (ROOT / "packaging" / "windows" / "PatchLab.iss").read_text(
        encoding="utf-8"
    ).casefold()
    build = (
        ROOT / "packaging" / "build_windows_installer.ps1"
    ).read_text(encoding="utf-8").casefold()

    assert "pyinstaller" not in setup
    assert "service-account" not in setup
    assert "gcloud" not in setup
    assert "runtime_family_id = \"v1-legacy-stock-clap\"" in build
    assert "bundle create $sourcebundle main" in build
    assert "data|private|\\.venv|gcloud|relay-credentials" in build
