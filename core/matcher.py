"""Pitch-aware analysis-by-synthesis matching with persistent render workers."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import librosa
import numpy as np
import soundfile as sf

from core.dataset import _serum1_targets, _serum2_targets
from core.delta_model import load_delta_model, predict_delta
from core.features import CLAP_SAMPLE_RATE, ClapEmbedder, handcrafted_features
from core.local_library import default_local_paths
from core.match import cosine_topk, l2_normalize
from core.perturbation import perturb_serum1, perturb_serum2
from core.platform_env import ENV
from core.synthesis_assets import resolve_synthesis_assets
from core.train import load_parameter_model, predict_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 44_100
_RENDER: dict[str, Any] = {}


def _audio_root() -> Path:
    """Return the render cache root for the current install shape.

    A distributed install renders into the per-user app data directory; the
    source checkout keeps renders under data/audio. Resolving this lazily (and
    not at import time) keeps a frozen build from binding the checkout layout.
    """

    if os.environ.get("PATCHLAB_DISTRIBUTION_MODE", "0").strip() == "1":
        return default_local_paths()["audio"]
    return PROJECT_ROOT / "data" / "audio"


@dataclass(slots=True)
class SearchConfig:
    population: int = 16
    max_evaluations: int = 300
    max_seconds: float = 300.0
    stall_generations: int = 5
    seed_mutations_per_preset: int = 8
    stft_weight: float = 0.35
    clap_weight: float = 0.65
    random_seed: int = 2026
    adaptive_preprocessing: bool = True


@dataclass(slots=True)
class Candidate:
    synth: str
    base_preset_id: int
    vector: np.ndarray
    mask: np.ndarray
    origin: str
    exact_base: bool = False
    midi_note: int | None = None
    objective: float = math.inf
    stft_loss: float = math.inf
    clap_cosine: float = -1.0
    waveform: np.ndarray | None = field(default=None, repr=False)


@dataclass(slots=True)
class MatchResult:
    midi_note: int
    acoustic_midi_note: int
    detected_hz: float | None
    pitch_confidence: float
    sub_bass_fraction: float
    unpitched_fallback: bool
    note_hypotheses: tuple[int, ...]
    comparison_duration_s: float
    stft_weight: float
    clap_weight: float
    best: Candidate
    runner_up: Candidate
    best_excluding_preset: Candidate | None
    retrieved_preset_ids: list[int]
    objective_trace: list[dict[str, Any]]
    evaluations: int
    elapsed_s: float
    evaluations_per_second: float


@dataclass(frozen=True, slots=True)
class PitchEstimate:
    midi_note: int
    frequency_hz: float | None
    confidence: float
    unpitched_fallback: bool


def analyze_pitch(audio: np.ndarray, sample_rate: int) -> PitchEstimate:
    values = np.asarray(audio, dtype=np.float32)
    if len(values) < 2048 or float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) < 1e-4:
        return PitchEstimate(60, None, 0.0, True)
    f0, voiced, probability = librosa.pyin(
        values,
        fmin=float(librosa.note_to_hz("C1")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sample_rate,
        frame_length=2048,
    )
    valid = f0[np.isfinite(f0) & voiced]
    if valid.size < 3:
        usable_probability = probability[np.isfinite(probability)]
        confidence = (
            float(np.median(usable_probability))
            if usable_probability.size
            else 0.0
        )
        return PitchEstimate(60, None, confidence, True)
    hz = float(np.median(valid))
    midi = int(np.clip(round(float(librosa.hz_to_midi(hz))), 24, 96))
    valid_probability = probability[np.isfinite(f0) & voiced]
    confidence = (
        float(np.median(valid_probability))
        if valid_probability.size
        else 0.0
    )
    return PitchEstimate(midi, hz, confidence, False)


def detect_midi_note(audio: np.ndarray, sample_rate: int) -> tuple[int, float | None, bool]:
    """Compatibility wrapper for callers that predate pitch confidence."""

    estimate = analyze_pitch(audio, sample_rate)
    return (
        estimate.midi_note,
        estimate.frequency_hz,
        estimate.unpitched_fallback,
    )


def sub_bass_energy_fraction(audio: np.ndarray, sample_rate: int) -> float:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if len(values) < 2:
        return 0.0
    values = values - float(np.mean(values))
    spectrum = np.fft.rfft(values)
    energy = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(len(values), 1.0 / sample_rate)
    total = float(np.sum(energy[frequencies >= 20.0]))
    if total <= 1e-20:
        return 0.0
    return float(
        np.sum(energy[(frequencies >= 20.0) & (frequencies < 100.0)])
        / total
    )


def prepare_query_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    adaptive: bool = True,
) -> tuple[np.ndarray, float]:
    """Return the aligned 48 kHz comparison signal and render duration."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    values = values[: int(round(min(4.0, len(values) / sample_rate) * sample_rate))]
    if adaptive and len(values):
        trimmed, _index = librosa.effects.trim(
            values,
            top_db=55.0,
            frame_length=min(1024, max(64, 2 ** int(math.floor(math.log2(len(values)))))),
            hop_length=128,
        )
        if len(trimmed):
            values = trimmed
    duration = max(0.25 if adaptive else 0.0, len(values) / sample_rate)
    target_samples = max(1, int(round(duration * sample_rate)))
    if len(values) < target_samples:
        values = np.pad(values, (0, target_samples - len(values)))
    else:
        values = values[:target_samples]
    if sample_rate != CLAP_SAMPLE_RATE:
        values = librosa.resample(
            values,
            orig_sr=sample_rate,
            target_sr=CLAP_SAMPLE_RATE,
            res_type="soxr_hq",
        ).astype(np.float32)
    return loudness_normalize(values), duration


def embedding_comparison_audio(
    audio: np.ndarray,
    duration_s: float,
    *,
    adaptive: bool,
) -> np.ndarray:
    """Apply identical explicit short-clip padding before CLAP."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if adaptive and duration_s < 1.0:
        samples = CLAP_SAMPLE_RATE
        values = values[:samples]
        if len(values) < samples:
            values = np.pad(values, (0, samples - len(values)))
    return np.ascontiguousarray(values, dtype=np.float32)


def objective_weights(
    duration_s: float,
    config: SearchConfig,
) -> tuple[float, float]:
    """Blend 65/35 STFT/CLAP at <=0.5 s into 35/65 at >=1.5 s."""

    if not config.adaptive_preprocessing:
        return config.stft_weight, config.clap_weight
    position = float(np.clip((duration_s - 0.5) / 1.0, 0.0, 1.0))
    clap_weight = 0.35 + 0.30 * position
    return 1.0 - clap_weight, clap_weight


def loudness_normalize(audio: np.ndarray, target_dbfs: float = -18.0) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    if rms <= 1e-8:
        return values.copy()
    gain = (10.0 ** (target_dbfs / 20.0)) / rms
    return np.ascontiguousarray(values * gain, dtype=np.float32)


def multi_resolution_stft_loss(target: np.ndarray, candidate: np.ndarray) -> float:
    length = min(len(target), len(candidate))
    if length < 512:
        return 10.0
    left, right = target[:length], candidate[:length]
    losses = []
    for fft_size in (512, 1024, 2048):
        target_mag = np.abs(librosa.stft(left, n_fft=fft_size, hop_length=fft_size // 4))
        candidate_mag = np.abs(librosa.stft(right, n_fft=fft_size, hop_length=fft_size // 4))
        losses.append(float(np.mean(np.abs(np.log1p(target_mag) - np.log1p(candidate_mag)))))
    return float(np.mean(losses))


def _init_render_worker(scratch_root: str) -> None:
    from core.plugin_host import make_dawdreamer_processor

    hosts = {}
    for synth, required in (("serum1", "VST2"), ("serum2", "VST3")):
        candidate = next(
            item for item in ENV.plugins_for(synth) if item.format == required and item.hostable
        )
        hosts[synth] = make_dawdreamer_processor(candidate)
    assets = resolve_synthesis_assets()
    _RENDER.update(
        hosts=hosts,
        s1=_serum1_targets(assets.library_db),
        s2=_serum2_targets(assets.serum2_targets, assets.serum2_schema),
        schema=json.loads(assets.serum2_schema.read_text(encoding="utf-8")),
        assets=assets,
        library_db=assets.library_db,
        state_path=Path(scratch_root) / f"worker-{os.getpid()}.vstpreset",
    )



def _state_file(assets: Any, preset_id: int) -> Path:
    """Locate a Serum 2 render-state template across every candidate root."""

    found = assets.find_render_state(preset_id)
    if found is None:
        raise RuntimeError(
            f"No Serum 2 render state for preset {preset_id} in "
            + ", ".join(str(root) for root in assets.render_state_roots)
        )
    return found


def _state_root_for(assets: Any, preset_id: int) -> Path:
    return _state_file(assets, preset_id).parent


def _render_candidate_unsafe(payload: tuple[Candidate, int, float]) -> tuple[np.ndarray, float]:
    candidate, midi_note, duration = payload
    engine, processor = _RENDER["hosts"][candidate.synth]
    assets = _RENDER["assets"]
    if candidate.synth == "serum1":
        with sqlite3.connect(_RENDER["library_db"]) as connection:
            row = connection.execute(
                "SELECT path FROM presets WHERE id=?", (candidate.base_preset_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError(
                f"Serum 1 preset {candidate.base_preset_id} is not in "
                f"{_RENDER['library_db']}; the synthesis database and the "
                "retrieval index disagree about the library contents"
            )
        path = row[0]
        if processor.load_preset(str(path)) is False:
            raise RuntimeError(f"Serum 1 rejected preset {candidate.base_preset_id}")
        if not candidate.exact_base:
            for index, value in enumerate(candidate.vector):
                processor.set_parameter(index, float(value))
        coverage = 1.0
    elif candidate.exact_base:
        from core.serum2_state_reconstruct import load_render_state

        load_render_state(processor, candidate.base_preset_id, _state_root_for(assets, candidate.base_preset_id))
        coverage = 1.0
    else:
        from core.serum2_preset import Serum2Preset
        from core.serum2_state_reconstruct import (
            decode_host_template,
            reconstruct_partial_vstpreset,
        )
        from core.serum2_targets import decode_vector

        template = decode_host_template(
            _state_file(assets, candidate.base_preset_id).read_bytes()
        )
        graph = decode_vector(candidate.vector, _RENDER["schema"], candidate.mask)
        decoded = Serum2Preset(
            path=_state_file(assets, candidate.base_preset_id),
            metadata={"presetName": "PatchLab candidate"},
            data=graph,
            metadata_length=0,
            cbor_length=0,
            payload_version=0,
            compressed_length=0,
        )
        blob, partition = reconstruct_partial_vstpreset(
            decoded, template, merge_matching_lists=True
        )
        state_path: Path = _RENDER["state_path"]
        state_path.write_bytes(blob)
        if processor.load_vst3_preset(str(state_path)) is False:
            raise RuntimeError("Serum 2 rejected candidate state")
        coverage = partition.coverage
    if hasattr(processor, "clear_midi"):
        processor.clear_midi()
    processor.add_midi_note(midi_note, 100, 0.0, duration)
    engine.render(duration)
    audio = np.asarray(engine.get_audio(), dtype=np.float32)
    if audio.shape[0] != 2 and audio.shape[1] == 2:
        audio = audio.T
    mono = np.ascontiguousarray(np.mean(audio, axis=0), dtype=np.float32)
    waveform = librosa.resample(
        mono, orig_sr=SAMPLE_RATE, target_sr=CLAP_SAMPLE_RATE, res_type="soxr_hq"
    ).astype(np.float32, copy=False)
    return waveform, coverage


def _render_candidate(
    payload: tuple[Candidate, int, float]
) -> tuple[np.ndarray | None, float, str | None]:
    try:
        waveform, coverage = _render_candidate_unsafe(payload)
        return waveform, coverage, None
    except Exception as exc:
        return None, 0.0, f"{type(exc).__name__}: {exc}"


class AnalysisBySynthesisMatcher:
    def __init__(self, processes: int = 4) -> None:
        context = mp.get_context("spawn")
        self._scratch = tempfile.TemporaryDirectory(
            prefix="patchlab-match-session-"
        )
        self.pool = context.Pool(
            processes,
            initializer=_init_render_worker,
            initargs=(self._scratch.name,),
        )
        self.embedder = ClapEmbedder(ENV)
        assets = resolve_synthesis_assets()
        self.stores = {
            1: _serum1_targets(assets.library_db),
            2: _serum2_targets(assets.serum2_targets, assets.serum2_schema),
        }
        self.preset_index = np.load(assets.preset_index, mmap_mode="r")
        self.note_index = np.load(assets.note_index, mmap_mode="r")
        manifest = np.load(assets.feature_dir / "similarity_manifest.npz")
        self.preset_ids = manifest["preset_ids"].astype(np.int64)
        self.preset_synths = manifest["preset_synths"].astype(np.uint8)
        self.note_midi_notes = manifest["note_midi_notes"].astype(np.int16)
        self.note_synths = manifest["note_synths"].astype(np.uint8)
        self.preset_rows = {int(value): index for index, value in enumerate(self.preset_ids)}
        self.absolute_model, self.absolute_checkpoint = load_parameter_model(device=ENV.compute_backend)
        self.delta_model, self.delta_checkpoint = load_delta_model(device=ENV.compute_backend)
        self.continuous_masks = {
            1: np.asarray(
                [not bool(field.get("stepped")) for field in self.stores[1].mapping], dtype=np.bool_
            ),
            2: np.asarray(
                [field.get("encoding") != "one_hot" for field in self.stores[2].mapping],
                dtype=np.bool_,
            ),
        }

    def close(self) -> None:
        self.pool.close()
        self.pool.join()
        self._scratch.cleanup()

    def _retrieve(self, embedding: np.ndarray, synth: str | None, count: int = 5) -> list[int]:
        if synth is None:
            rows = np.arange(len(self.preset_ids))
        else:
            code = 1 if synth == "serum1" else 2
            rows = np.flatnonzero(self.preset_synths == code)
        _scores, positions = cosine_topk(
            embedding[None, :], np.asarray(self.preset_index[rows]), k=count, normalized=True
        )
        return [int(self.preset_ids[rows[position]]) for position in positions[0]]

    def query_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        adaptive_preprocessing: bool = True,
    ) -> np.ndarray:
        """Return the retrieval embedding for the same <=4 s comparison clip."""

        mono, duration = prepare_query_audio(
            audio,
            sample_rate,
            adaptive=adaptive_preprocessing,
        )
        embedded = embedding_comparison_audio(
            mono,
            duration,
            adaptive=adaptive_preprocessing,
        )
        return self.embedder.embed([embedded])[0]

    def retrieve_existing(
        self, embedding: np.ndarray, count: int = 10
    ) -> list[tuple[int, float]]:
        """Return cross-synth owned presets and cosine scores."""

        scores, positions = cosine_topk(
            l2_normalize(np.asarray(embedding, dtype=np.float32))[None, :],
            np.asarray(self.preset_index),
            k=count,
            normalized=True,
        )
        return [
            (int(self.preset_ids[position]), float(score))
            for score, position in zip(scores[0], positions[0], strict=True)
        ]

    def _library_conditioned_note(
        self, embedding: np.ndarray, synth: str | None, acoustic_note: int
    ) -> tuple[int, float]:
        if synth is None:
            rows = np.arange(len(self.note_midi_notes))
        else:
            code = 1 if synth == "serum1" else 2
            rows = np.flatnonzero(self.note_synths == code)
        scores, positions = cosine_topk(
            embedding[None, :], np.asarray(self.note_index[rows]), k=1, normalized=True
        )
        score = float(scores[0, 0])
        # Serum patches may transpose or spectrally suppress the played
        # fundamental. A strong note-level library match identifies the MIDI
        # input that actually produced the target more reliably than pYIN.
        if score >= 0.50:
            return int(self.note_midi_notes[rows[int(positions[0, 0])]]), score
        return acoustic_note, score

    def _seeds(
        self,
        embedding: np.ndarray,
        features: np.ndarray,
        retrieved: Sequence[int],
        config: SearchConfig,
    ) -> list[Candidate]:
        seeds: list[Candidate] = []
        for rank, preset_id in enumerate(retrieved):
            code = int(self.preset_synths[self.preset_rows[preset_id]])
            synth = "serum1" if code == 1 else "serum2"
            store = self.stores[code]
            row = store.preset_row[preset_id]
            base = np.asarray(store.vectors[row], dtype=np.float32)
            mask = np.asarray(store.masks[row], dtype=np.bool_)
            seeds.append(Candidate(synth, preset_id, base.copy(), mask.copy(), f"retrieved-{rank+1}", True))
            neighbor_embedding = np.asarray(self.preset_index[self.preset_rows[preset_id]])
            delta_vector = predict_delta(
                self.delta_model,
                embedding,
                neighbor_embedding,
                base,
                synth,
                self.continuous_masks[code],
            )
            seeds.append(Candidate(synth, preset_id, delta_vector, mask.copy(), f"delta-{rank+1}"))
            for mutation in range(config.seed_mutations_per_preset):
                rng = np.random.default_rng(
                    np.random.SeedSequence([config.random_seed, preset_id, mutation])
                )
                if code == 1:
                    vector, _ = perturb_serum1(base, mask, store.mapping, rng)
                else:
                    vector, _ = perturb_serum2(
                        base, mask, self.absolute_checkpoint["serum2_schema"], rng
                    )
                seeds.append(Candidate(synth, preset_id, vector, mask.copy(), f"mutation-{rank+1}-{mutation+1}"))
        top_id = int(retrieved[0])
        top_code = int(self.preset_synths[self.preset_rows[top_id]])
        top_synth = "serum1" if top_code == 1 else "serum2"
        absolute = predict_parameters(
            self.absolute_model,
            self.absolute_checkpoint,
            embedding,
            features,
            top_synth,
        )
        top_store = self.stores[top_code]
        top_mask = np.asarray(top_store.masks[top_store.preset_row[top_id]], dtype=np.bool_)
        seeds.append(Candidate(top_synth, top_id, absolute, top_mask, "absolute-model"))
        return seeds

    @staticmethod
    def _note_hypotheses(
        primary: int,
        pitch: PitchEstimate,
        sub_bass_fraction: float,
        *,
        adaptive: bool,
    ) -> tuple[int, ...]:
        if not adaptive:
            return (int(primary),)
        uncertain = pitch.confidence < 0.85 or pitch.unpitched_fallback
        bass_classified = sub_bass_fraction >= 0.45
        if not uncertain and not bass_classified:
            return (int(primary),)

        if pitch.unpitched_fallback:
            proposed = [60, 48, 72, 36]
        else:
            octave_family = {
                int(np.clip(pitch.midi_note + offset, 24, 96))
                for offset in (0, -12, 12)
            }
            genre_prior = 60
            if bass_classified:
                genre_prior = min(
                    (note for note in (24, 36) if note not in octave_family),
                    key=lambda note: abs(note - pitch.midi_note),
                    default=36,
                )
            proposed = [
                pitch.midi_note,
                pitch.midi_note - 12,
                pitch.midi_note + 12,
                genre_prior,
            ]
        result: list[int] = []
        for note in proposed:
            clamped = int(np.clip(note, 24, 96))
            if clamped not in result:
                result.append(clamped)
        return tuple(result[:4])

    def _evaluate(
        self,
        candidates: list[Candidate],
        midi_note: int,
        duration: float,
        target: np.ndarray,
        target_embedding: np.ndarray,
        config: SearchConfig,
    ) -> None:
        waveforms: list[np.ndarray | None] = [None] * len(candidates)
        live_positions = []
        live_payloads = []
        for position, candidate in enumerate(candidates):
            candidate_note = candidate.midi_note or midi_note
            cached = _audio_root() / str(candidate.base_preset_id) / f"{candidate_note}.wav"
            if candidate.exact_base and cached.is_file():
                audio, rate = sf.read(cached, dtype="float32", always_2d=True)
                mono = np.mean(audio, axis=1, dtype=np.float32)[: int(round(duration * rate))]
                if rate != CLAP_SAMPLE_RATE:
                    mono = librosa.resample(
                        mono, orig_sr=rate, target_sr=CLAP_SAMPLE_RATE, res_type="soxr_hq"
                    ).astype(np.float32)
                waveforms[position] = mono
            else:
                live_positions.append(position)
                live_payloads.append((candidate, candidate_note, duration))
        if live_payloads:
            rendered = self.pool.map(_render_candidate, live_payloads)
            for position, (waveform, _coverage, _error) in zip(
                live_positions, rendered, strict=True
            ):
                waveforms[position] = waveform
        successful = [position for position, waveform in enumerate(waveforms) if waveform is not None]
        target_samples = len(target)
        normalized = []
        for position in successful:
            waveform = loudness_normalize(np.asarray(waveforms[position]))
            waveform = waveform[:target_samples]
            if len(waveform) < target_samples:
                waveform = np.pad(
                    waveform,
                    (0, target_samples - len(waveform)),
                )
            normalized.append(np.asarray(waveform, dtype=np.float32))
        embedding_audio = [
            embedding_comparison_audio(
                waveform,
                duration,
                adaptive=config.adaptive_preprocessing,
            )
            for waveform in normalized
        ]
        embeddings = self.embedder.embed(embedding_audio) if normalized else np.empty((0, 512), dtype=np.float32)
        stft_weight, clap_weight = objective_weights(duration, config)
        for position, waveform, embedded in zip(successful, normalized, embeddings, strict=True):
            candidate = candidates[position]
            candidate.waveform = waveform
            candidate.stft_loss = multi_resolution_stft_loss(target, waveform)
            candidate.clap_cosine = float(
                np.clip(np.dot(target_embedding, embedded), -1.0, 1.0)
            )
            candidate.objective = (
                stft_weight * candidate.stft_loss
                + clap_weight * (1.0 - candidate.clap_cosine)
            )

    def match(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        synth_hint: str | None = None,
        config: SearchConfig | None = None,
        target_embedding: np.ndarray | None = None,
        exclude_preset_id: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> MatchResult:
        import cma

        config = config or SearchConfig()
        started = time.monotonic()
        full_mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        pitch = analyze_pitch(full_mono, sample_rate)
        sub_bass_fraction = sub_bass_energy_fraction(full_mono, sample_rate)
        target, duration = prepare_query_audio(
            full_mono,
            sample_rate,
            adaptive=config.adaptive_preprocessing,
        )
        objective_audio = embedding_comparison_audio(
            target,
            duration,
            adaptive=config.adaptive_preprocessing,
        )
        objective_embedding = self.embedder.embed([objective_audio])[0]
        retrieval_embedding = (
            l2_normalize(np.asarray(target_embedding, dtype=np.float32))
            if target_embedding is not None
            else objective_embedding
        )
        features = handcrafted_features(target)
        midi_note, library_note_score = self._library_conditioned_note(
            retrieval_embedding,
            synth_hint,
            pitch.midi_note,
        )
        hypotheses = (
            (midi_note,)
            if config.adaptive_preprocessing and library_note_score >= 0.90
            else self._note_hypotheses(
                midi_note,
                pitch,
                sub_bass_fraction,
                adaptive=config.adaptive_preprocessing,
            )
        )
        retrieved = self._retrieve(retrieval_embedding, synth_hint, 5)
        candidates = self._seeds(retrieval_embedding, features, retrieved, config)
        for index, candidate in enumerate(candidates):
            candidate.midi_note = hypotheses[index % len(hypotheses)]
        self._evaluate(candidates, midi_note, duration, target, objective_embedding, config)
        evaluations = len(candidates)
        candidates.sort(key=lambda item: item.objective)
        trace = [
            {
                "generation": 0,
                "evaluations": evaluations,
                "best_objective": candidates[0].objective,
                "stft_loss": candidates[0].stft_loss,
                "clap_cosine": candidates[0].clap_cosine,
                "midi_note": candidates[0].midi_note,
                "note_hypotheses": list(hypotheses),
            }
        ]
        if progress_callback is not None:
            progress_callback(
                {
                    "evaluations": evaluations,
                    "budget": config.max_evaluations,
                    "best_clap_cosine": candidates[0].clap_cosine,
                    "best_objective": candidates[0].objective,
                    "generation": 0,
                }
            )
        # Exact in-library retrieval needs no numerical refinement.
        if candidates[0].clap_cosine < 0.999 or candidates[0].stft_loss > 1e-4:
            for seed_rank, seed in enumerate(candidates[:3]):
                code = 1 if seed.synth == "serum1" else 2
                active = np.flatnonzero(self.continuous_masks[code] & seed.mask)
                if active.size == 0:
                    continue
                remaining = config.max_evaluations - evaluations
                if remaining < config.population:
                    break
                strategy = cma.CMAEvolutionStrategy(
                    seed.vector[active],
                    0.08,
                    {
                        "bounds": [0.0, 1.0],
                        "popsize": config.population,
                        "seed": config.random_seed + seed_rank,
                        "CMA_diagonal": True,
                        "verbose": -9,
                        "verb_log": 0,
                        "verb_disp": 0,
                    },
                )
                best_value = seed.objective
                stale = 0
                generation = 0
                while (
                    evaluations + config.population <= config.max_evaluations
                    and time.monotonic() - started < config.max_seconds
                    and stale < config.stall_generations
                ):
                    generation += 1
                    proposed = strategy.ask()
                    generation_candidates = []
                    for values in proposed:
                        vector = seed.vector.copy()
                        vector[active] = np.asarray(values, dtype=np.float32)
                        generation_candidates.append(
                            Candidate(
                                seed.synth,
                                seed.base_preset_id,
                                vector,
                                seed.mask,
                                "cma",
                                midi_note=seed.midi_note,
                            )
                        )
                    self._evaluate(
                        generation_candidates,
                        midi_note,
                        duration,
                        target,
                        objective_embedding,
                        config,
                    )
                    objectives = [item.objective for item in generation_candidates]
                    strategy.tell(proposed, objectives)
                    evaluations += len(generation_candidates)
                    candidates.extend(generation_candidates)
                    current = min(objectives)
                    if current < best_value - 1e-4:
                        best_value = current
                        stale = 0
                    else:
                        stale += 1
                    winner = min(generation_candidates, key=lambda item: item.objective)
                    trace.append(
                        {
                            "seed_rank": seed_rank + 1,
                            "generation": generation,
                            "evaluations": evaluations,
                            "best_objective": winner.objective,
                            "stft_loss": winner.stft_loss,
                            "clap_cosine": winner.clap_cosine,
                            "midi_note": winner.midi_note,
                        }
                    )
                    if progress_callback is not None:
                        overall = min(candidates, key=lambda item: item.objective)
                        progress_callback(
                            {
                                "evaluations": evaluations,
                                "budget": config.max_evaluations,
                                "best_clap_cosine": overall.clap_cosine,
                                "best_objective": overall.objective,
                                "generation": generation,
                            }
                        )
        candidates.sort(key=lambda item: item.objective)
        excluded_id = retrieved[0] if exclude_preset_id is None else exclude_preset_id
        excluding = next((item for item in candidates if item.base_preset_id != excluded_id), None)
        elapsed = time.monotonic() - started
        stft_weight, clap_weight = objective_weights(duration, config)
        return MatchResult(
            midi_note=int(candidates[0].midi_note or midi_note),
            acoustic_midi_note=pitch.midi_note,
            detected_hz=pitch.frequency_hz,
            pitch_confidence=pitch.confidence,
            sub_bass_fraction=sub_bass_fraction,
            unpitched_fallback=pitch.unpitched_fallback,
            note_hypotheses=hypotheses,
            comparison_duration_s=duration,
            stft_weight=stft_weight,
            clap_weight=clap_weight,
            best=candidates[0],
            runner_up=candidates[1],
            best_excluding_preset=excluding,
            retrieved_preset_ids=retrieved,
            objective_trace=trace,
            evaluations=evaluations,
            elapsed_s=elapsed,
            evaluations_per_second=evaluations / max(elapsed, 1e-6),
        )
