"""The generated-preset save lifecycle: retries, classification, atomicity, agreement.

Failures are injected at the real seams (the build step and the filesystem
commit); everything else -- staging, incident persistence, the no-clobber
commit, validation, and the Library record -- is the production code.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

import core.preset_save as preset_save
from core.base_preset_identity import BaseIdentityError
from core.db import Database
from core.fxp import build_fxp
from core.preset_save import (
    IncidentStatus,
    PresetValidationError,
    SaveFailureKind,
    SaveIncident,
    StagedPreset,
    UNRECOVERABLE_MESSAGE,
    classify_save_failure,
    commit_verified_preset,
    finalize_saved_incident,
    incident_user_message,
    reconcile_incidents,
    run_save_lifecycle,
    sha256_file,
    validate_preset_file,
)


def _fxp(name: str = "Generated") -> bytes:
    return build_fxp(b"serum one state " + name.encode(), plugin_id=b"XfsX", program_name=name)


class Build:
    """A build step that writes a real preset, with scripted failures."""

    def __init__(self, failures: list[BaseException | None] | None = None, *, stage: str = "construct") -> None:
        self.failures = list(failures or [])
        self.calls = 0
        self.stage = stage

    def __call__(self, tracker, staging: Path) -> StagedPreset:
        self.calls += 1
        tracker.stage = self.stage
        failure = self.failures.pop(0) if self.failures else None
        if failure is not None:
            raise failure
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_bytes(_fxp())
        tracker.stage = "validate"
        checked = validate_preset_file(staging, synth="serum1")
        return StagedPreset(
            path=staging, sha256=sha256_file(staging), mode="test",
            base_identity={"synth": "serum1", "catalog_id": 12, "content_hash": "ab" * 20},
            validation=checked,
        )


class FlakyCommit:
    """Wrap the real commit, failing the first calls with scripted errors."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = list(errors)
        self.calls = 0
        self.real = preset_save.commit_verified_preset

    def __call__(self, staged, destination, *, expected_sha256):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.real(staged, destination, expected_sha256=expected_sha256)


@pytest.fixture
def incident(tmp_path: Path) -> SaveIncident:
    return SaveIncident.create(
        match_uid="match-1", result_path=tmp_path / "result.json",
        target_path=tmp_path / "Presets" / "PatchLab - Bass.fxp", synth="serum1",
        run_again={"source_audio_path": str(tmp_path / "bass.wav"), "target_synth": "serum1",
                   "budget": "balanced", "start_offset_s": 1.5},
        root=tmp_path / "incidents",
    )


def _run(incident, build, **kwargs):
    sleeps: list[float] = []
    outcome = run_save_lifecycle(
        incident, build=build, destination=Path(incident.target_path), extension=".fxp",
        sleep=sleeps.append, **kwargs,
    )
    return outcome, sleeps


def _eio() -> OSError:
    return OSError(errno.EIO, "Input/output error")


# --- classification ------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,stage,kind,reason",
    [
        (OSError(errno.EBUSY, "busy"), "commit", SaveFailureKind.TRANSIENT_IO, "filesystem"),
        (OSError(errno.ENOSPC, "full"), "commit", SaveFailureKind.DESTINATION, "disk_full"),
        (OSError(errno.ENOSPC, "full"), "construct", SaveFailureKind.DESTINATION, "disk_full"),
        (PermissionError(errno.EACCES, "denied"), "commit", SaveFailureKind.DESTINATION, "permission"),
        (OSError(errno.EROFS, "ro"), "commit", SaveFailureKind.DESTINATION, "read_only"),
        (FileNotFoundError(errno.ENOENT, "gone"), "commit", SaveFailureKind.DESTINATION, "folder_missing"),
        (ValueError("cbor"), "construct", SaveFailureKind.CONSTRUCTION, "serialization"),
        (RuntimeError("Serum 2 rejected"), "validate", SaveFailureKind.VALIDATION, "reload_failed"),
        (PresetValidationError("x", reason="empty_file"), "commit", SaveFailureKind.VALIDATION, "empty_file"),
        (BaseIdentityError("no base"), "construct", SaveFailureKind.IDENTITY, "base_identity"),
    ],
)
def test_failures_are_classified(exc, stage, kind, reason) -> None:
    failure = classify_save_failure(exc, stage=stage)
    assert (failure.kind, failure.reason) == (kind, reason)


def test_a_wrapped_os_error_is_still_recognised() -> None:
    try:
        try:
            raise OSError(errno.ENOSPC, "full")
        except OSError as inner:
            raise RuntimeError("write failed") from inner
    except RuntimeError as outer:
        assert classify_save_failure(outer, stage="commit").reason == "disk_full"


# --- D. the normal path is invisible and complete ------------------------------


def test_D_first_attempt_saves_validates_and_is_recorded(incident, tmp_path) -> None:
    build = Build()
    outcome, sleeps = _run(incident, build)
    assert outcome.saved and sleeps == [] and build.calls == 1
    assert outcome.final_path == Path(incident.target_path).resolve()
    assert outcome.final_path.read_bytes() == _fxp()
    assert incident.status == IncidentStatus.COMMITTED.value
    assert [a["ok"] for a in incident.attempts] == [True]
    assert incident.attempts[0]["validation"]["final"]["sha256"] == sha256_file(outcome.final_path)


# --- E. a transient failure recovers on its own ----------------------------------


def test_E_transient_commit_failure_retries_the_same_preset(incident, monkeypatch) -> None:
    flaky = FlakyCommit([OSError(errno.EBUSY, "resource busy")])
    monkeypatch.setattr(preset_save, "commit_verified_preset", flaky)
    build = Build()
    outcome, sleeps = _run(incident, build)
    assert outcome.saved
    assert build.calls == 1, "the same verified preset is committed again, not rebuilt"
    assert flaky.calls == 2
    assert sleeps == [0.5], "one short, bounded backoff"
    assert [a["ok"] for a in incident.attempts] == [False, True]
    assert incident.attempts[0]["failure_kind"] == "transient_io"


# --- F. three automatic failures ---------------------------------------------------


def test_F_three_failures_preserve_the_verified_preset(incident, monkeypatch) -> None:
    monkeypatch.setattr(preset_save, "commit_verified_preset", FlakyCommit([_eio(), _eio(), _eio()]))
    build = Build()
    outcome, sleeps = _run(incident, build)
    assert not outcome.saved and not outcome.unrecoverable
    assert len(incident.attempts) == 3 and build.calls == 1
    assert sleeps == [0.5, 2.0], "bounded backoff, never a hot loop"
    assert incident.status == IncidentStatus.FAILED.value
    staged = incident.valid_staged_artifact()
    assert staged is not None and staged.read_bytes() == _fxp()
    reloaded = SaveIncident.load(incident.incident_id, root=incident.directory.parent)
    assert [a["attempt"] for a in reloaded.attempts] == [1, 2, 3]
    for attempt in reloaded.attempts:
        for key in ("incident_id", "attempt", "operation", "target_path", "synth", "base_identity",
                    "exception_type", "message", "traceback", "failure_kind", "validation"):
            assert key in attempt


def test_F_an_identity_failure_is_not_retried_pointlessly(incident) -> None:
    build = Build([BaseIdentityError("catalog preset 777 is not in the bundle")])
    outcome, sleeps = _run(incident, build)
    assert not outcome.saved and len(incident.attempts) == 1 and sleeps == []
    assert incident.last_failure["kind"] == "identity"


# --- G. manual retry succeeds with the same result ----------------------------------


def test_G_manual_retry_commits_the_preserved_preset_without_rebuilding(incident, monkeypatch) -> None:
    monkeypatch.setattr(preset_save, "commit_verified_preset", FlakyCommit([_eio(), _eio(), _eio()]))
    build = Build()
    _run(incident, build)
    staged_sha = incident.staged["sha256"]
    monkeypatch.setattr(preset_save, "commit_verified_preset", FlakyCommit([]))
    outcome, _ = _run(incident, build, trigger="manual")
    assert outcome.saved and build.calls == 1, "no Match re-run and no rebuild"
    assert sha256_file(outcome.final_path) == staged_sha
    assert incident.attempts[-1]["trigger"] == "manual" and incident.attempts[-1]["ok"]


# --- H. manual retry fails: one attempt, a clear message, no loop --------------------


def test_H_failed_manual_retry_is_one_attempt_and_stays_retryable(incident, monkeypatch) -> None:
    monkeypatch.setattr(preset_save, "commit_verified_preset", FlakyCommit([_eio()] * 4))
    build = Build()
    _run(incident, build)
    outcome, sleeps = _run(incident, build, trigger="manual")
    assert not outcome.saved and not outcome.unrecoverable
    assert len(incident.attempts) == 4 and sleeps == []
    assert incident.status == IncidentStatus.FAILED.value
    assert "try again in a moment" in incident_user_message(incident)


# --- I. destination problems are explained, never "fixed" by regenerating ------------


@pytest.mark.parametrize(
    "error,phrase",
    [
        (OSError(errno.ENOSPC, "No space left on device"), "more free disk space"),
        (PermissionError(errno.EACCES, "Permission denied"), "isn't writable"),
        (OSError(errno.EROFS, "Read-only file system"), "isn't writable"),
    ],
)
def test_I_destination_failures_do_not_regenerate(incident, monkeypatch, error, phrase) -> None:
    monkeypatch.setattr(preset_save, "commit_verified_preset", FlakyCommit([error] * 4))
    build = Build()
    _run(incident, build)
    outcome, _ = _run(incident, build, trigger="manual")
    assert not outcome.saved and not outcome.unrecoverable
    assert build.calls == 1, "a full disk is not fixed by building the preset again"
    assert incident.valid_staged_artifact() is not None, "the result is kept for later"
    assert phrase in incident_user_message(incident)
    assert "Traceback" not in incident_user_message(incident)


# --- J. a preset that cannot be made valid: one rebuild, then Run Again -------------


def test_J_validation_failure_gets_one_bounded_rebuild_then_run_again(incident) -> None:
    invalid = PresetValidationError("reloaded graph differs", reason="reload_mismatch")
    build = Build([invalid] * 5, stage="validate")
    _run(incident, build)
    assert build.calls == 3, "each automatic attempt rebuilds a preset that failed validation"
    outcome, _ = _run(incident, build, trigger="manual")
    assert outcome.unrecoverable and build.calls == 5, "manual attempt + exactly one rebuild"
    assert [a["trigger"] for a in incident.attempts[-2:]] == ["manual", "rebuild"]
    assert incident.status == IncidentStatus.UNRECOVERABLE.value
    assert incident_user_message(incident) == UNRECOVERABLE_MESSAGE
    assert incident.run_again == {
        "source_audio_path": incident.run_again["source_audio_path"],
        "target_synth": "serum1", "budget": "balanced", "start_offset_s": 1.5,
    }


def test_J_a_rebuild_that_succeeds_saves_normally(incident) -> None:
    build = Build([ValueError("cbor")] * 4, stage="construct")
    _run(incident, build)
    build.failures = [ValueError("cbor")]
    outcome, _ = _run(incident, build, trigger="manual")
    assert outcome.saved and incident.attempts[-1]["trigger"] == "rebuild"


def test_J_identity_failure_goes_straight_to_run_again(incident) -> None:
    build = Build([BaseIdentityError("no base")] * 3)
    _run(incident, build)
    outcome, _ = _run(incident, build, trigger="manual")
    assert outcome.unrecoverable and build.calls == 2, "no rebuild can fix an unidentifiable base"


# --- K. atomic, no-clobber commit ------------------------------------------------------


def test_K_an_interrupted_write_leaves_no_preset_behind(tmp_path, monkeypatch) -> None:
    staged = tmp_path / "staged.fxp"
    staged.write_bytes(_fxp())
    destination = tmp_path / "Presets" / "PatchLab - Bass.fxp"

    def torn(source, target):
        target.write(source.read(10))
        raise OSError(errno.EIO, "device went away mid-write")

    monkeypatch.setattr(preset_save.shutil, "copyfileobj", torn)
    with pytest.raises(OSError):
        commit_verified_preset(staged, destination, expected_sha256=sha256_file(staged))
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == [], "not even the hidden staging file"


def test_K_another_preset_is_never_overwritten(tmp_path) -> None:
    staged = tmp_path / "staged.fxp"
    staged.write_bytes(_fxp("New"))
    destination = tmp_path / "PatchLab - Bass.fxp"
    destination.write_bytes(_fxp("Someone else's"))
    final = commit_verified_preset(staged, destination, expected_sha256=sha256_file(staged))
    assert final.name == "PatchLab - Bass 2.fxp"
    assert destination.read_bytes() == _fxp("Someone else's")
    again = commit_verified_preset(staged, destination, expected_sha256=sha256_file(staged))
    assert again.name == "PatchLab - Bass 3.fxp", "a different file still is never replaced"


def test_K_committing_the_same_bytes_twice_is_idempotent(tmp_path) -> None:
    staged = tmp_path / "staged.fxp"
    staged.write_bytes(_fxp())
    destination = tmp_path / "PatchLab - Bass.fxp"
    first = commit_verified_preset(staged, destination, expected_sha256=sha256_file(staged))
    second = commit_verified_preset(staged, destination, expected_sha256=sha256_file(staged))
    assert first == second == destination.resolve()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["PatchLab - Bass.fxp", "staged.fxp"]


def test_K_repeated_identically_named_results_each_get_their_own_file(tmp_path) -> None:
    destination = tmp_path / "PatchLab - Kick.fxp"
    finals = []
    for index in range(3):
        staged = tmp_path / f"staged-{index}.fxp"
        staged.write_bytes(_fxp(f"Take {index}"))
        finals.append(commit_verified_preset(staged, destination, expected_sha256=sha256_file(staged)))
    assert [f.name for f in finals] == ["PatchLab - Kick.fxp", "PatchLab - Kick 2.fxp", "PatchLab - Kick 3.fxp"]
    assert len({f.read_bytes() for f in finals}) == 3


def test_K_a_staged_preset_changed_after_verification_is_refused(tmp_path) -> None:
    staged = tmp_path / "staged.fxp"
    staged.write_bytes(_fxp())
    expected = sha256_file(staged)
    staged.write_bytes(_fxp("tampered"))
    with pytest.raises(PresetValidationError):
        commit_verified_preset(staged, tmp_path / "out.fxp", expected_sha256=expected)
    assert not (tmp_path / "out.fxp").exists()


def test_K_a_committed_file_that_fails_its_final_check_is_removed(incident, monkeypatch) -> None:
    def corrupting(staged, destination, *, expected_sha256):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"")
        return destination

    monkeypatch.setattr(preset_save, "commit_verified_preset", corrupting)
    outcome, _ = _run(incident, Build())
    assert not outcome.saved
    assert not Path(incident.target_path).exists(), "an empty file never looks like a saved preset"
    assert incident.attempts[0]["failure_kind"] == "validation"


@pytest.mark.parametrize("payload,reason", [(b"", "empty_file"), (b"not a preset", "wrong_format")])
def test_validation_rejects_empty_and_foreign_files(tmp_path, payload, reason) -> None:
    path = tmp_path / "x.fxp"
    path.write_bytes(payload)
    with pytest.raises(PresetValidationError) as caught:
        validate_preset_file(path, synth="serum1")
    assert caught.value.reason == reason


def test_validation_rejects_the_wrong_generation(tmp_path) -> None:
    path = tmp_path / "x.SerumPreset"
    path.write_bytes(_fxp())
    with pytest.raises(PresetValidationError):
        validate_preset_file(path, synth="serum2")


# --- L. the Library and the filesystem agree --------------------------------------------


@pytest.fixture
def library(tmp_path: Path) -> Database:
    database = Database(tmp_path / "library.db")
    database.migrate()
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO match_library(match_uid,source_name,source_audio_path,source_content_hash,"
            "result_json_path,target_synth,budget,similarity_percent,base_name,recommendation_synth,"
            "no_confident_match,created_at) VALUES ('match-1','Bass','a.wav','h','r.json','serum1',"
            "'balanced',90.0,'Base','serum1',0,'now')"
        )
    return database


class BrokenDatabase:
    def set_match_exported_path(self, *_args):
        raise OSError(errno.EIO, "database is locked")


def test_L_saved_means_file_and_library_agree(incident, library) -> None:
    outcome, _ = _run(incident, Build())
    assert finalize_saved_incident(incident, library, outcome.final_path)
    assert library.get_match_library("match-1").exported_preset_path == outcome.final_path
    assert incident.status == IncidentStatus.SAVED.value


def test_L_a_failed_library_write_is_reconciled_without_a_second_preset(incident, library) -> None:
    outcome, _ = _run(incident, Build())
    assert not finalize_saved_incident(incident, BrokenDatabase(), outcome.final_path)
    assert incident.status == IncidentStatus.COMMITTED.value
    assert library.get_match_library("match-1").exported_preset_path is None
    files_before = sorted(Path(incident.target_path).parent.iterdir())
    counts = reconcile_incidents(library, root=incident.directory.parent)
    assert counts["recorded"] == 1
    assert library.get_match_library("match-1").exported_preset_path == outcome.final_path
    assert sorted(Path(incident.target_path).parent.iterdir()) == files_before
    assert SaveIncident.load(incident.incident_id, root=incident.directory.parent).status == "saved"


def test_L_the_library_never_records_a_missing_file(incident, library) -> None:
    outcome, _ = _run(incident, Build())
    outcome.final_path.unlink()
    with pytest.raises(PresetValidationError):
        finalize_saved_incident(incident, library, outcome.final_path)
    assert library.get_match_library("match-1").exported_preset_path is None
    counts = reconcile_incidents(library, root=incident.directory.parent)
    assert counts["marked_failed"] == 1
    assert library.get_match_library("match-1").exported_preset_path is None


def test_L_an_interrupted_save_becomes_retryable_at_launch(incident, library) -> None:
    incident.status = IncidentStatus.SAVING.value
    incident.save()
    reconcile_incidents(library, root=incident.directory.parent)
    assert SaveIncident.load(incident.incident_id, root=incident.directory.parent).status == "failed"


def test_L_a_row_update_that_matches_nothing_is_not_success(library, tmp_path) -> None:
    assert not library.set_match_exported_path("no-such-match", tmp_path / "x.fxp")


# --- diagnostics carry detail but never content --------------------------------------


def test_incident_diagnostics_contain_no_preset_bytes(incident, monkeypatch) -> None:
    monkeypatch.setattr(preset_save, "commit_verified_preset", FlakyCommit([_eio()] * 3))
    _run(incident, Build())
    summary = json.dumps(incident.diagnostic_summary())
    assert "serum one state" not in summary
    assert _fxp().hex() not in summary
    assert '"attempts"' in summary and '"failure_kind": "transient_io"' in summary
