"""Match speed-ups: cached target spectra, streaming evaluation, worker count, warm cache.

The speed-ups change *scheduling*, never the objective a candidate is given, so
these tests pin the numbers: every candidate must score exactly what the
original per-candidate arithmetic produces.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import core.matcher as matcher_module
from core.matcher import (
    WORKER_INIT_FAILURE_PREFIX,
    AnalysisBySynthesisMatcher,
    Candidate,
    SearchConfig,
    TargetSpectra,
    _DeterministicRenderPool,
    embedding_comparison_audio,
    loudness_normalize,
    multi_resolution_stft_loss,
    objective_weights,
)
from core.platform_env import recommended_render_workers


RATE = 48_000
GIB = 1 << 30


def _tone(seed: int, seconds: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    wave = np.sin(2 * np.pi * (110 + 40 * seed) * t) * np.exp(-2 * t) + 0.05 * rng.standard_normal(len(t))
    return wave.astype(np.float32)


# --- change 1: the target's spectra are computed once, with identical results ----


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cached_target_spectra_give_bit_identical_loss(seed: int) -> None:
    target, candidate = _tone(seed), _tone(seed + 10)
    assert TargetSpectra(target).loss(candidate) == multi_resolution_stft_loss(target, candidate)


def test_target_spectra_handles_a_candidate_of_a_different_length() -> None:
    target, candidate = _tone(0), _tone(1, seconds=0.6)
    assert TargetSpectra(target).loss(candidate) == multi_resolution_stft_loss(target, candidate)


def test_target_spectra_handles_a_clip_too_short_for_the_stft() -> None:
    short = np.zeros(300, dtype=np.float32)
    assert TargetSpectra(short).loss(short) == multi_resolution_stft_loss(short, short) == 10.0


def test_the_target_is_transformed_once_not_once_per_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    target = _tone(0)
    spectra = TargetSpectra(target)
    calls: list[int] = []
    real = matcher_module.librosa.stft
    monkeypatch.setattr(matcher_module.librosa, "stft", lambda y, **k: (calls.append(len(y)), real(y, **k))[1])
    for seed in range(5):
        spectra.loss(_tone(seed + 20))
    assert len(calls) == 5 * 3, "three resolutions per candidate and none for the target"


# --- change 2: streaming evaluation -------------------------------------------------


class _FakeEmbedder:
    """Deterministic stand-in for CLAP: a unit vector that depends on the audio."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def embed(self, waveforms):
        self.batches.append(len(waveforms))
        out = []
        for waveform in waveforms:
            w = np.asarray(waveform, dtype=np.float64)
            vector = np.array([w[i * 37 % len(w)] + 0.001 * i for i in range(512)])
            out.append(vector / np.linalg.norm(vector))
        return np.asarray(out, dtype=np.float32)


class _FakePool:
    """imap-capable pool that returns pre-made renders, recording how far it got."""

    def __init__(self, renders: dict[int, np.ndarray | None], on_yield=None) -> None:
        self.renders = renders
        self.on_yield = on_yield
        self.map_calls = 0
        self.imap_calls = 0

    def _run(self, payloads):
        for payload in payloads:
            candidate = payload[0]
            waveform = self.renders[candidate.base_preset_id]
            yield (waveform, 1.0, None if waveform is not None else "RuntimeError: preset rejected")

    def imap(self, _function, payloads):
        self.imap_calls += 1
        for index, item in enumerate(self._run(payloads)):
            if self.on_yield is not None:
                self.on_yield(index)
            yield item

    def map(self, function, payloads):
        self.map_calls += 1
        return list(self._run(payloads))


def _matcher(pool: _FakePool, embedder: _FakeEmbedder) -> AnalysisBySynthesisMatcher:
    matcher = object.__new__(AnalysisBySynthesisMatcher)
    matcher.pool = pool
    matcher.embedder = embedder
    matcher.tracker = None
    matcher._rendered_batches = 0
    matcher.supervisor = SimpleNamespace(
        worker_states=lambda: [], check=lambda: None
    )
    matcher._fail_on_worker_init = lambda detail: (_ for _ in ()).throw(RuntimeError(detail["message"]))
    return matcher


def _candidates(count: int) -> list[Candidate]:
    return [
        Candidate("serum2", index, np.zeros(4, np.float32), np.ones(4, bool), "cma")
        for index in range(count)
    ]


def _expected_objective(waveform, target, target_embedding, duration, config, embedder) -> tuple[float, float, float]:
    """The original per-candidate arithmetic, written out independently."""

    normalized = loudness_normalize(np.asarray(waveform))[: len(target)]
    if len(normalized) < len(target):
        normalized = np.pad(normalized, (0, len(target) - len(normalized)))
    embedded = embedder.embed([embedding_comparison_audio(normalized, duration, adaptive=True)])[0]
    stft = multi_resolution_stft_loss(target, normalized)
    clap = float(np.clip(np.dot(target_embedding, embedded), -1.0, 1.0))
    stft_weight, clap_weight = objective_weights(duration, config)
    return stft, clap, stft_weight * stft + clap_weight * (1.0 - clap)


@pytest.mark.parametrize("count", [1, 7, 8, 9, 16, 20])
def test_streaming_evaluation_scores_every_candidate_exactly_as_before(count: int) -> None:
    config = SearchConfig()
    target = loudness_normalize(_tone(99))
    duration = len(target) / RATE
    embedder = _FakeEmbedder()
    target_embedding = embedder.embed([target])[0]
    renders = {index: _tone(index) * (0.5 + index / 10) for index in range(count)}
    matcher = _matcher(_FakePool(renders), embedder)
    candidates = _candidates(count)

    matcher._evaluate(candidates, 60, duration, target, target_embedding, config)

    reference = _FakeEmbedder()
    for index, candidate in enumerate(candidates):
        stft, clap, objective = _expected_objective(
            renders[index], target, target_embedding, duration, config, reference
        )
        assert candidate.stft_loss == pytest.approx(stft, abs=1e-12)
        assert candidate.clap_cosine == pytest.approx(clap, abs=1e-6)
        assert candidate.objective == pytest.approx(objective, abs=1e-6)
        assert candidate.waveform is not None


def test_embedding_overlaps_rendering() -> None:
    """Early results are embedded before the last render has been handed back."""

    count = 20
    target = loudness_normalize(_tone(99))
    embedder = _FakeEmbedder()
    seen_at_yield: list[int] = []
    pool = _FakePool(
        {index: _tone(index) for index in range(count)},
        on_yield=lambda index: seen_at_yield.append(len(embedder.batches)),
    )
    matcher = _matcher(pool, embedder)
    target_embedding = embedder.embed([target])[0]
    embedder.batches.clear()  # only the candidates' batches are of interest
    matcher._evaluate(_candidates(count), 60, len(target) / RATE, target, target_embedding, SearchConfig())
    assert pool.imap_calls == 1 and pool.map_calls == 0
    assert seen_at_yield[-1] >= 1, "nothing was embedded while renders were still arriving"
    assert embedder.batches[0] == AnalysisBySynthesisMatcher.STREAM_CHUNK
    assert sum(embedder.batches) == count, "every candidate is embedded exactly once"


def test_failed_renders_stay_unscored_and_do_not_block_the_rest() -> None:
    target = loudness_normalize(_tone(99))
    embedder = _FakeEmbedder()
    renders = {0: _tone(0), 1: None, 2: _tone(2)}
    matcher = _matcher(_FakePool(renders), embedder)
    candidates = _candidates(3)
    matcher._evaluate(candidates, 60, len(target) / RATE, target, embedder.embed([target])[0], SearchConfig())
    assert candidates[1].objective == float("inf") and candidates[1].waveform is None
    assert np.isfinite(candidates[0].objective) and np.isfinite(candidates[2].objective)


def test_a_worker_initialisation_failure_is_still_one_terminal_error() -> None:
    class _Pool(_FakePool):
        def _run(self, payloads):
            for _ in payloads:
                yield (None, 0.0, WORKER_INIT_FAILURE_PREFIX + '{"message": "Serum could not start"}')

    target = loudness_normalize(_tone(99))
    embedder = _FakeEmbedder()
    matcher = _matcher(_Pool({}), embedder)
    with pytest.raises(RuntimeError, match="Serum could not start"):
        matcher._evaluate(_candidates(4), 60, len(target) / RATE, target, embedder.embed([target])[0], SearchConfig())


def test_pools_without_imap_still_work_through_map() -> None:
    class _MapOnly:
        def __init__(self) -> None:
            self.inner = _FakePool({0: _tone(0), 1: _tone(1)})

        def map(self, function, payloads):
            return self.inner.map(function, payloads)

    target = loudness_normalize(_tone(99))
    embedder = _FakeEmbedder()
    matcher = _matcher(_MapOnly(), embedder)
    candidates = _candidates(2)
    matcher._evaluate(candidates, 60, len(target) / RATE, target, embedder.embed([target])[0], SearchConfig())
    assert all(np.isfinite(item.objective) for item in candidates)


def test_the_supervisor_and_tracker_see_each_batch_once() -> None:
    target = loudness_normalize(_tone(99))
    embedder = _FakeEmbedder()
    events: list[tuple[str, object]] = []
    matcher = _matcher(_FakePool({index: _tone(index) for index in range(10)}), embedder)
    matcher.supervisor = SimpleNamespace(
        worker_states=lambda: [{"alive": True, "pid": 1}], check=lambda: events.append(("check", None))
    )
    matcher.tracker = SimpleNamespace(
        note_heartbeat=lambda *a, **k: None,
        bump=lambda name, amount: events.append(("bump", amount)),
        mark_progress=lambda **k: events.append(("progress", k["rendered"])),
    )
    matcher._evaluate(_candidates(10), 60, len(target) / RATE, target, embedder.embed([target])[0], SearchConfig())
    assert events == [("check", None), ("bump", 10), ("progress", 10)]
    assert matcher._rendered_batches == 1


def test_the_target_spectra_are_cached_across_generations() -> None:
    matcher = _matcher(_FakePool({}), _FakeEmbedder())
    target = _tone(5)
    first = matcher._target_spectra(target)
    assert matcher._target_spectra(target) is first
    assert matcher._target_spectra(_tone(6)) is not first


# --- the deterministic benchmark pool gets the same streaming interface ---------


class _AsyncJob:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _OneWorkerPool:
    def __init__(self, index: int, log: list[tuple[int, int]]) -> None:
        self.index, self.log = index, log

    def apply_async(self, function, args):
        self.log.append((self.index, args[0]))
        return _AsyncJob(function(args[0]))


class _Context:
    def __init__(self) -> None:
        self.log: list[tuple[int, int]] = []
        self.count = 0

    def Pool(self, _processes, *, initializer, initargs):
        pool = _OneWorkerPool(self.count, self.log)
        self.count += 1
        return pool


def test_deterministic_pool_imap_keeps_position_order_and_host_pinning() -> None:
    context = _Context()
    pool = _DeterministicRenderPool(context, 3, lambda: None, ())
    results = list(pool.imap(lambda value: value * 10, list(range(8))))
    assert results == [value * 10 for value in range(8)]
    assert context.log == [(value % 3, value) for value in range(8)], "position i always goes to host i % 3"


# --- change 4: worker count ----------------------------------------------------------


def test_default_never_drops_below_the_long_standing_four() -> None:
    assert recommended_render_workers(cores=(2, 2), memory_bytes=64 * GIB) == 4
    assert recommended_render_workers(cores=None, logical_cpus=2, memory_bytes=64 * GIB) == 4


def test_performance_cores_plus_half_the_efficiency_cores_up_to_eight() -> None:
    assert recommended_render_workers(cores=(4, 6), memory_bytes=24 * GIB) == 7
    assert recommended_render_workers(cores=(4, 4), memory_bytes=16 * GIB) == 6
    assert recommended_render_workers(cores=(12, 4), memory_bytes=64 * GIB) == 8


def test_small_or_unknown_memory_keeps_four_workers() -> None:
    assert recommended_render_workers(cores=(8, 4), memory_bytes=8 * GIB) == 4
    assert recommended_render_workers(cores=(8, 4), memory_bytes=None) == 4


def test_windows_and_intel_use_half_the_logical_cpus() -> None:
    assert recommended_render_workers(cores=None, logical_cpus=12, memory_bytes=32 * GIB) == 6
    assert recommended_render_workers(cores=None, logical_cpus=32, memory_bytes=64 * GIB) == 8


def test_environment_override_wins_and_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATCHLAB_RENDER_WORKERS", "3")
    assert recommended_render_workers(cores=(12, 4), memory_bytes=64 * GIB) == 3
    monkeypatch.setenv("PATCHLAB_RENDER_WORKERS", "999")
    assert recommended_render_workers() == 32
    monkeypatch.setenv("PATCHLAB_RENDER_WORKERS", "nonsense")
    assert recommended_render_workers(cores=(4, 6), memory_bytes=24 * GIB) == 7


def test_match_workflow_asks_for_the_recommended_count_when_none_is_given() -> None:
    import inspect

    from core.match_workflow import run_match_file

    default = inspect.signature(run_match_file).parameters["matcher_processes"].default
    assert default is None


# --- Match must work against the bundled catalog, which has no coverage table ----


def _catalog(path, *, with_coverage: bool, mask: int | None = None):
    import sqlite3

    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE presets(id INTEGER PRIMARY KEY, name TEXT, synth TEXT, path TEXT, content_hash TEXT)"
        )
        connection.execute("INSERT INTO presets VALUES (7, 'Warm Pad', 'serum2', '/p/pad.SerumPreset', 'abc')")
        if with_coverage:
            connection.execute("CREATE TABLE fingerprint_note_coverage(preset_id INTEGER PRIMARY KEY, note_mask INTEGER)")
            connection.execute("INSERT INTO fingerprint_note_coverage VALUES (7, ?)", (mask,))
    return path


def test_closest_match_details_work_against_the_bundled_catalog_without_coverage(tmp_path) -> None:
    """The packaged app's catalog has only presets/params; every Match used to die here."""

    from core.match_workflow import _preset_details
    from core.prepared_state import RENDER_MIDI_NOTES

    details = _preset_details([7], _catalog(tmp_path / "catalog.sqlite", with_coverage=False))
    assert details[7]["name"] == "Warm Pad"
    assert details[7]["audible_midi_notes"] == list(RENDER_MIDI_NOTES), "full note range, as before"


def test_closest_match_details_use_recorded_coverage_when_the_library_has_it(tmp_path) -> None:
    from core.match_workflow import _preset_details
    from core.prepared_state import RENDER_MIDI_NOTES

    details = _preset_details([7], _catalog(tmp_path / "library.db", with_coverage=True, mask=0b101))
    assert details[7]["audible_midi_notes"] == [RENDER_MIDI_NOTES[0], RENDER_MIDI_NOTES[2]]


def test_a_library_row_without_a_recorded_mask_offers_the_full_range(tmp_path) -> None:
    from core.match_workflow import _preset_details
    from core.prepared_state import RENDER_MIDI_NOTES

    details = _preset_details([7], _catalog(tmp_path / "library.db", with_coverage=True, mask=None))
    assert details[7]["audible_midi_notes"] == list(RENDER_MIDI_NOTES)
