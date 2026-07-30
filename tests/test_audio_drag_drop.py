from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMimeData, QUrl

from app.ui import ScaledGraphicsView


class _DropEvent:
    def __init__(self, paths: list[Path]) -> None:
        self._mime = QMimeData()
        self._mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])

    def mimeData(self) -> QMimeData:
        return self._mime


def test_scaled_view_accepts_one_supported_local_audio_file(tmp_path: Path) -> None:
    audio = tmp_path / "query.WAV"
    audio.touch()

    assert ScaledGraphicsView._audio_path(_DropEvent([audio])) == str(audio)


def test_scaled_view_rejects_non_audio_and_multiple_files(tmp_path: Path) -> None:
    text = tmp_path / "notes.txt"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.flac"
    for path in (text, first, second):
        path.touch()

    assert ScaledGraphicsView._audio_path(_DropEvent([text])) is None
    assert ScaledGraphicsView._audio_path(_DropEvent([first, second])) is None
