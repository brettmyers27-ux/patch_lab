"""Stable, reusable progress estimates for long-running user-visible work."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class ProgressETA:
    """Estimate remaining time from sampled completed-item counts."""

    started_at: float = field(default_factory=time.monotonic)
    _last_count: int = 0
    _last_at: float | None = None
    _rate: float | None = None
    _remaining: float | None = None

    def update(self, completed: int, total: int, *, now: float | None = None) -> float | None:
        """Return estimated remaining seconds, or ``None`` during warmup."""

        now = time.monotonic() if now is None else now
        total = max(int(total), 0)
        completed = max(0, min(int(completed), total))
        elapsed = max(now - self.started_at, 0.0)
        if total == 0 or completed >= total:
            self._last_count, self._last_at, self._remaining = completed, now, 0.0
            return 0.0
        if self._last_at is not None and completed > self._last_count:
            interval = now - self._last_at
            if interval > 0:
                sample = (completed - self._last_count) / interval
                average = completed / max(elapsed, 0.001)
                sample = min(max(sample, average * 0.25), average * 4.0)
                self._rate = sample if self._rate is None else 0.25 * sample + 0.75 * self._rate
        elif self._rate is None and completed > 0 and elapsed > 0:
            self._rate = completed / elapsed
        previous_at = self._last_at
        self._last_count, self._last_at = completed, now
        if completed < 2 or elapsed < 1.0 or not self._rate:
            return None
        candidate = (total - completed) / self._rate
        if self._remaining is not None and previous_at is not None:
            expected = max(0.0, self._remaining - max(now - previous_at, 0.0))
            candidate = min(candidate, expected) if candidate <= expected else expected + (candidate - expected) * 0.15
        self._remaining = max(candidate, 0.0)
        return self._remaining


def format_eta(remaining_seconds: float | None) -> str:
    """Return a naturally rounded, deliberately approximate ETA label."""

    if remaining_seconds is None:
        return "Estimating time…"
    seconds = max(0, int(round(remaining_seconds)))
    if seconds < 90:
        return f"About {max(1, seconds)} seconds remaining"
    if seconds < 3600:
        return f"About {max(1, round(seconds / 60))} minutes remaining"
    return f"About {max(1, round(seconds / 3600))} hours remaining"
