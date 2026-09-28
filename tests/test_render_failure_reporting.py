"""Render-stage failures retain their original cause for preparation."""

from pathlib import Path

from core import render


class _NotSet:
    def is_set(self) -> bool:
        return False


def test_state_load_failure_names_stage_and_preserves_plugin_error(monkeypatch, tmp_path: Path) -> None:
    task = render.RenderTask(
        preset_id=7,
        source_path=tmp_path / "source.serumpreset",
        synth="serum2",
        midi_notes=render.MIDI_NOTES,
        audio_root=tmp_path / "audio",
        state_dir=tmp_path / "states",
    )
    monkeypatch.setattr(render, "_worker_host", lambda _synth: (object(), object()))

    def reject(_task, _processor):
        raise RuntimeError("PluginProcessor::loadVST3Preset: unknown error")

    monkeypatch.setattr(render, "_load_task_state", reject)
    result = render._process_task(task, _NotSet(), _NotSet())

    assert result.rows == []
    assert result.error is not None
    assert result.error.startswith("serum2 state load failed: RuntimeError:")
    assert "PluginProcessor::loadVST3Preset: unknown error" in result.error
    assert "source.serumpreset" not in result.error


def test_public_render_summary_excludes_private_failure_text() -> None:
    summary = render.RenderSummary(
        failure_errors={7: "load failed at /Users/private/preset.serumpreset"}
    )
    assert "failure_errors" not in render.summary_dict(summary)
