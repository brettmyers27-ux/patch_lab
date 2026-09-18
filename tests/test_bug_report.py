from __future__ import annotations

from pathlib import Path

import pytest

from core import bug_report


def test_bug_report_requires_a_description_and_persists_text_only_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bug_report, "pending_reports_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="description"):
        bug_report.create_request(comments="   ", logs="diagnostic")

    request_path = bug_report.create_request(
        comments="  Run Match did not enable after choosing a sound.  ",
        logs="PatchLab log fixture",
    )
    request = bug_report.load_request(request_path)

    assert request_path.parent == tmp_path
    assert request_path.suffix == ".json"
    assert request.comments == "Run Match did not enable after choosing a sound."
    assert request.logs == "PatchLab log fixture"
    assert len(request.ticket_id) == 32
