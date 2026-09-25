"""Centralized audio playback: one shared ``_play_audio``, every surface benefits.

A tester hit "Internal PortAudio error paErrorCode -9986" after likely
switching audio output devices; ``_play_audio`` had no error handling at all,
and only one of its seven callers wrapped it locally. These drive the real
``MainWindow._play_audio`` with a faked ``sounddevice`` (no real audio
hardware needed) and separately confirm every call site is wired to it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import sounddevice as sd

from tests.test_gui_release_flows import Gui, gui  # noqa: F401


class FakePortAudioError(Exception):
    """Stands in for sounddevice.PortAudioError without touching real audio."""


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "sound.wav"
    sf.write(path, np.zeros(4800, dtype=np.float32), 48000)
    return path


@pytest.fixture
def playback(gui: Gui, monkeypatch):  # noqa: F811
    """A real window with a faked sounddevice: no hardware, deterministic."""

    window = gui.window
    monkeypatch.setattr(sd, "PortAudioError", FakePortAudioError, raising=False)
    calls = {"play": [], "stop": 0, "reset": 0, "query": []}

    def fake_stop():
        calls["stop"] += 1

    monkeypatch.setattr(sd, "stop", fake_stop)
    monkeypatch.setattr(sd, "_terminate", lambda: calls.__setitem__("reset", calls["reset"] + 1))
    monkeypatch.setattr(sd, "_initialize", lambda: None)
    monkeypatch.setattr(sd, "query_devices", lambda kind=None: calls["query"].append(kind) or {"name": "x"})
    window._playback_calls = calls
    return gui


def _script_play(monkeypatch, *outcomes) -> list:
    """sd.play() raises/succeeds per ``outcomes``, one per call."""

    remaining = list(outcomes)
    played: list[tuple] = []

    def fake_play(audio, rate, blocking=False):
        outcome = remaining.pop(0)
        played.append((audio, rate))
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(sd, "play", fake_play)
    return played


# --- A: normal playback ------------------------------------------------------


def test_A_normal_playback_succeeds(playback, wav, monkeypatch) -> None:
    played = _script_play(monkeypatch, None)
    window = playback.window
    assert window._play_audio(wav) is True
    assert len(played) == 1
    assert window._playback_calls["stop"] == 1
    assert window._playback_calls["reset"] == 0, "no recovery needed on a clean success"


# --- B: first attempt fails, refresh succeeds, second attempt plays ---------


def test_B_recovers_after_one_reset_and_says_nothing_to_the_user(playback, wav, monkeypatch) -> None:
    window = playback.window
    played = _script_play(monkeypatch, FakePortAudioError("paErrorCode -9986"), None)
    messages = []
    monkeypatch.setattr(window.statusBar(), "showMessage", messages.append)

    assert window._play_audio(wav) is True
    assert len(played) == 2, "stopped, refreshed, and actually retried"
    assert window._playback_calls["reset"] == 1
    assert not any("-9986" in m or "Error" in m for m in messages), "no error shown on recovery"


def test_recovery_is_recorded_in_diagnostics_not_shown_to_the_user(playback, wav, monkeypatch) -> None:
    window = playback.window
    _script_play(monkeypatch, FakePortAudioError("device changed"), None)
    window._play_audio(wav)
    events = playback.recorder.recent_events()
    recovered = [e for e in events if e.get("event_type") == "playback_recovered"]
    assert len(recovered) == 1
    assert str(wav) in recovered[0].get("fields", {}).get("path", "")


# --- C: retry also fails -> exactly one clear message -----------------------


def test_C_two_failures_give_one_plain_message_never_the_error_code(playback, wav, monkeypatch) -> None:
    window = playback.window
    played = _script_play(
        monkeypatch,
        FakePortAudioError("paErrorCode -9986"),
        FakePortAudioError("paErrorCode -9986"),
    )
    messages = []
    monkeypatch.setattr(window.statusBar(), "showMessage", messages.append)

    assert window._play_audio(wav) is False
    assert len(played) == 2, "exactly one retry, never a loop"
    assert window._playback_calls["reset"] == 1
    assert len(messages) == 1
    assert "-9986" not in messages[0] and "PortAudio" not in messages[0]
    assert "speakers or headphones" in messages[0]


def test_no_output_device_gets_its_own_message(playback, wav, monkeypatch) -> None:
    window = playback.window
    _script_play(monkeypatch, FakePortAudioError("x"), FakePortAudioError("x"))
    monkeypatch.setattr(sd, "query_devices", lambda kind=None: (_ for _ in ()).throw(FakePortAudioError("none")))
    messages = []
    monkeypatch.setattr(window.statusBar(), "showMessage", messages.append)

    assert window._play_audio(wav) is False
    assert "couldn't find an audio output device" in messages[0]


def test_failure_diagnostics_carry_the_real_detail(playback, wav, monkeypatch) -> None:
    window = playback.window
    _script_play(monkeypatch, FakePortAudioError("boom -9986"), FakePortAudioError("boom -9986"))
    window._play_audio(wav)
    events = playback.recorder.recent_events()
    failed = [e for e in events if e.get("event_type") == "playback_failed"]
    assert len(failed) == 1
    assert failed[0]["fields"]["failure_category"] == "device_error"
    assert failed[0]["fields"]["exception_type"] == "FakePortAudioError"


# --- D: missing/unreadable source file --------------------------------------


def test_D_missing_file_gets_a_specific_message(playback, tmp_path, monkeypatch) -> None:
    window = playback.window
    messages = []
    monkeypatch.setattr(window.statusBar(), "showMessage", messages.append)
    assert window._play_audio(tmp_path / "gone.wav") is False
    assert messages == ["PatchLab can't find this audio file anymore."]


def test_corrupt_or_unsupported_audio_gets_a_distinct_message(playback, tmp_path, monkeypatch) -> None:
    window = playback.window
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not actually audio")
    messages = []
    monkeypatch.setattr(window.statusBar(), "showMessage", messages.append)
    assert window._play_audio(bad) is False
    assert "corrupted or in an unsupported format" in messages[0]


def test_missing_file_never_triggers_a_device_reset(playback, tmp_path) -> None:
    window = playback.window
    window._play_audio(tmp_path / "gone.wav")
    assert window._playback_calls["reset"] == 0, "a file problem is not a device problem"


# --- E-I: every audition surface routes through the shared function --------


def test_E_closest_match_audition_uses_the_shared_player(playback, monkeypatch) -> None:
    from core.preview_cache import preview_cache_identity, preview_cache_path

    window = playback.window
    called = MagicMock(return_value=True)
    monkeypatch.setattr(window, "_play_audio", called)
    cached = preview_cache_path(
        window._preview_cache_root(), preview_cache_identity("abc", "serum2"), 60
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"x")

    window._play_existing_match(
        {"content_hash": "abc", "synth": "serum2", "audition_path": None}, 60
    )
    called.assert_called_once_with(cached)


def test_F_generated_result_octave_preview_gates_its_message_on_success(playback, monkeypatch) -> None:
    window = playback.window
    window._preview_requests = {"req1": [(None, "", 60)]}
    window._preview_request_targets = {}
    window._preview_inflight = {}
    window._preview_silent_requests = set()
    messages = []
    monkeypatch.setattr(window.statusBar(), "showMessage", messages.append)
    monkeypatch.setattr(window, "_play_audio", lambda p: True)
    window._preview_request_completed("req1", "/tmp/generated.wav")
    assert any("preview ready" in m for m in messages)


def test_G_octave_audition_does_not_claim_success_on_failure(playback, monkeypatch) -> None:
    window = playback.window
    cached = window._preview_cache_root() / "cachekey" / "60.wav"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"x")
    monkeypatch.setattr(window, "_play_audio", lambda p: False)
    messages = []
    monkeypatch.setattr(window.statusBar(), "showMessage", messages.append)
    window._resolve_octave_preview(cache_key="cachekey", note=60, synth="serum2")
    assert not any("Playing cached" in m for m in messages), (
        "a failed play must not also claim it's playing"
    )


def test_H_library_source_audition_gates_on_success(playback, wav, monkeypatch) -> None:
    from core.db import Database, MatchLibraryRecord

    window = playback.window
    record = MatchLibraryRecord(
        id=1, match_uid="u1", source_name="Bass.wav", source_audio_path=wav,
        source_content_hash="h", result_json_path=Path("/tmp/r.json"),
        target_synth="serum2", budget="quick", similarity_percent=90.0,
        base_name="b", recommendation_synth="serum2", no_confident_match=False,
        batch_id=None, exported_preset_path=None, created_at="now",
    )
    monkeypatch.setattr(Database, "get_match_library", lambda self, uid: record)
    monkeypatch.setattr("app.ui.resolved_record_paths", lambda rec, root: (wav, Path("/tmp/r.json")))
    monkeypatch.setattr(window, "_play_audio", lambda p: False)
    messages = []
    monkeypatch.setattr(window.statusBar(), "showMessage", messages.append)
    window.play_library_source("u1")
    assert not any("Playing archived source" in m for m in messages)


def test_I_uploaded_target_audition_gates_on_success(playback, wav, monkeypatch) -> None:
    window = playback.window
    window._match_audio_path = wav
    monkeypatch.setattr(window, "_play_audio", lambda p: False)
    messages = []
    monkeypatch.setattr(window.statusBar(), "showMessage", messages.append)
    window.play_uploaded_audio()
    assert not any("Playing uploaded audio" in m for m in messages)


def test_uploaded_target_disables_the_drop_zone_when_the_file_is_gone(playback, tmp_path) -> None:
    window = playback.window
    window._match_audio_path = tmp_path / "gone.wav"
    playable = []
    window.match_drop.set_playable = playable.append
    window.play_uploaded_audio()
    assert playable == [False]
