from core.progress_eta import ProgressETA, format_eta


def test_progress_eta_warms_up_then_revises_upward_gradually() -> None:
    estimate = ProgressETA(started_at=0.0)

    assert estimate.update(1, 100, now=0.5) is None
    first = estimate.update(2, 100, now=1.0)
    assert first is not None
    revised = estimate.update(3, 100, now=10.0)
    assert revised is not None
    assert revised < first * 2


def test_eta_label_is_approximate_and_naturally_rounded() -> None:
    assert format_eta(None) == "Estimating time…"
    assert format_eta(8.2) == "About 8 seconds remaining"
    assert format_eta(181) == "About 3 minutes remaining"
