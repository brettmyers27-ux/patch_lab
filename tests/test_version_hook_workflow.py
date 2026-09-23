"""The version hook against real git history: one release, many commits.

Builds a throwaway repo with its own ``main`` (playing the role of
origin/main) and a release-prep branch, then runs the actual
``scripts/verify_version_increment.py`` CLI -- exactly what the pre-commit
hook and CI invoke -- against it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_version_increment.py"
VERSION_FILE = "app/__version__.py"


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True,
    )


def _git(repo: Path, *args: str) -> None:
    result = _run(repo, *args)
    assert result.returncode == 0, f"git {args}: {result.stdout}{result.stderr}"


def _write_version(repo: Path, version: str) -> None:
    path = repo / VERSION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'"""Single source of truth for PatchLab\'s public version."""\n\n'
        f'__version__ = "{version}"\n'
    )


def _commit(repo: Path, version: str, message: str) -> str:
    _write_version(repo, version)
    _git(repo, "add", VERSION_FILE)
    # --allow-empty: most commits in a real release repeat the same version
    # (they change code, not the version file), so nothing else need differ.
    _git(repo, "commit", "--allow-empty", "-m", message)
    return _run(repo, "rev-parse", "HEAD").stdout.strip()


def _verify_staged(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT), "--staged"],
        cwd=repo, text=True, capture_output=True,
    )


def _verify_range(repo: Path, base: str, head: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT), "--range", base, head],
        cwd=repo, text=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo with 'main' at a released 1.5.5, playing the role of origin/main."""

    repo = tmp_path / "patchlab"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "scripts").mkdir(exist_ok=True)
    _commit(repo, "1.5.5", "released 1.5.5")
    return repo


# --- staged (pre-commit hook) behaviour, against a released main -----------


def test_1_leaving_the_released_baseline_is_allowed(repo: Path) -> None:
    _write_version(repo, "1.5.6")
    _git(repo, "add", VERSION_FILE)
    result = _verify_staged(repo)
    assert result.returncode == 0, result.stdout
    assert "1.5.5 -> 1.5.6" in result.stdout


def test_2_a_second_branch_commit_staying_at_1_5_6_is_allowed(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "v1.5.6-fixes")
    _commit(repo, "1.5.6", "Phase 1")
    _write_version(repo, "1.5.6")  # untouched by this commit's real changes
    _git(repo, "add", VERSION_FILE)
    result = _verify_staged(repo)
    assert result.returncode == 0, result.stdout
    assert "1.5.6 unchanged (same release)" in result.stdout


def test_3_a_version_regression_is_rejected(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "v1.5.6-fixes")
    _commit(repo, "1.5.6", "Phase 1")
    _write_version(repo, "1.5.5")
    _git(repo, "add", VERSION_FILE)
    result = _verify_staged(repo)
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_4_staying_at_the_released_baseline_is_rejected(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "v1.5.6-fixes")
    _write_version(repo, "1.5.5")  # never actually bumped
    _git(repo, "add", VERSION_FILE)
    result = _verify_staged(repo)
    assert result.returncode == 1, result.stdout
    assert "last released version" in result.stdout


def test_5_a_malformed_version_is_rejected(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "v1.5.6-fixes")
    (repo / VERSION_FILE).write_text('__version__ = "1.5.six"\n')
    _git(repo, "add", VERSION_FILE)
    result = _verify_staged(repo)
    assert result.returncode == 1
    assert "FAIL" in result.stdout


def test_a_skipped_step_off_the_baseline_is_rejected(repo: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "v1.5.6-fixes")
    _write_version(repo, "1.5.8")
    _git(repo, "add", VERSION_FILE)
    result = _verify_staged(repo)
    assert result.returncode == 1


def test_6_the_next_release_cycle_still_requires_a_bump_to_leave_its_baseline(repo: Path) -> None:
    """Once 1.5.6 ships (main advances), a fresh 1.5.7 branch works the same way."""

    _commit(repo, "1.5.6", "released 1.5.6")  # main now ships 1.5.6
    _git(repo, "checkout", "-q", "-b", "v1.5.7-fixes")
    _write_version(repo, "1.5.6")  # stayed at the (new) baseline
    _git(repo, "add", VERSION_FILE)
    stayed = _verify_staged(repo)
    assert stayed.returncode == 1, stayed.stdout

    _write_version(repo, "1.5.7")
    _git(repo, "add", VERSION_FILE)
    bumped = _verify_staged(repo)
    assert bumped.returncode == 0, bumped.stdout
    assert "1.5.6 -> 1.5.7" in bumped.stdout


# --- range checks (CI, on push/PR into main) --------------------------------


def test_range_accepts_one_bump_then_several_same_version_commits(repo: Path) -> None:
    base = _run(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "-b", "v1.5.6-fixes")
    _commit(repo, "1.5.6", "Phase 1")
    _commit(repo, "1.5.6", "Phase 2")
    head = _commit(repo, "1.5.6", "Phase 3")
    result = _verify_range(repo, base, head)
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("PASS") == 3


def test_4_range_rejects_a_branch_that_never_left_the_baseline(repo: Path) -> None:
    base = _run(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "-b", "v1.5.6-fixes")
    head = _commit(repo, "1.5.5", "no real version bump")
    result = _verify_range(repo, base, head)
    assert result.returncode == 1
    assert "last released version" in result.stdout


def test_range_rejects_a_regression_anywhere_in_the_branch(repo: Path) -> None:
    base = _run(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "-b", "v1.5.6-fixes")
    _commit(repo, "1.5.6", "Phase 1")
    head = _commit(repo, "1.5.5", "accidental downgrade")
    result = _verify_range(repo, base, head)
    assert result.returncode == 1


def test_range_with_no_commits_passes_trivially(repo: Path) -> None:
    head = _run(repo, "rev-parse", "HEAD").stdout.strip()
    result = _verify_range(repo, head, head)
    assert result.returncode == 0
    assert "no non-merge commits" in result.stdout
