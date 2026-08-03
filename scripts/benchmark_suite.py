#!/usr/bin/env python3
"""Deterministic, resumable Stage 2 benchmark suite.

The suite is the single scorer for the Stage 2 A/B comparison.  It keeps large
per-sample details under ``data/stage2`` and writes only the compact summary to
``docs/benchmarks``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import tempfile
import time
import traceback
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.audio_input import SUPPORTED_AUDIO_SUFFIXES
from core.factory_bundle import DEFAULT_FACTORY_BUNDLE, FactoryBundle, FactoryPreset
from core.factory_verify import verify_local_factory_install
from core.features import CLAP_SAMPLE_RATE, ClapEmbedder
from core.match import cosine_topk, l2_normalize
from core.match_workflow import run_match_file
from core.matcher import embedding_comparison_audio, prepare_query_audio
from core.model_assets import configure_model_environment
from core.platform_env import ENV
from core.plugin_host import make_dawdreamer_processor
from core.render import SAMPLE_RATE, _render_audio, _trim_tail
from core.serum2_state_reconstruct import load_render_state
from core.synthesis_assets import resolve_synthesis_assets


DEFAULT_BAM_DIR = Path.home() / "Documents" / "PatchLab" / "benchmarks" / "BAM"
DEFAULT_DETAIL_DIR = PROJECT_ROOT / "data" / "stage2" / "baseline"
DEFAULT_SUMMARY = PROJECT_ROOT / "docs" / "benchmarks" / "stage2-baseline.json"
DEFAULT_SEED = 20260802
APPLEDOUBLE_PREFIX = "._"
BENCHMARK_AUDIO_SUFFIXES = SUPPORTED_AUDIO_SUFFIXES - {".mp3", ".flac", ".ogg"}


@dataclass(frozen=True, slots=True)
class StackConfiguration:
    name: str
    clap_checkpoint: str
    feature_dir: str
    library_db: str
    factory_bundle: str
    parameter_model: str
    delta_model: str


@dataclass(frozen=True, slots=True)
class BenchmarkFactoryPreset:
    """A factory-bundle row joined to its synthesis-catalog identity."""

    bundle: FactoryPreset
    catalog_id: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_audio_files(directory: Path) -> list[Path]:
    """Return real root-level benchmark audio, excluding Mac sidecar files."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and not path.name.startswith(APPLEDOUBLE_PREFIX)
            and path.suffix.casefold() in BENCHMARK_AUDIO_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def target_synth_for_name(name: str) -> str:
    folded = name.casefold().replace("-", " ").replace("_", " ")
    serum1_markers = ("serum 1", "serum1", "[s1]", " s1 ")
    serum2_markers = ("serum 2", "serum2", "[s2]", " s2 ")
    if any(marker in folded for marker in serum1_markers):
        return "serum1"
    if any(marker in folded for marker in serum2_markers):
        return "serum2"
    return "serum2"


def _detail_name(path: Path) -> str:
    suffix = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    safe = "".join(character if character.isalnum() else "-" for character in path.stem)
    safe = "-".join(piece for piece in safe.split("-") if piece)[:80] or "sample"
    return f"{safe}-{suffix}.json"


def _stack_configuration(name: str, factory_bundle: Path) -> StackConfiguration:
    assets = resolve_synthesis_assets()
    model = configure_model_environment()
    return StackConfiguration(
        name=name,
        clap_checkpoint=str(model.checkpoint.resolve()),
        feature_dir=str(assets.feature_dir.resolve()),
        library_db=str(assets.library_db.resolve()),
        factory_bundle=str(factory_bundle.resolve()),
        parameter_model=str(
            Path(
                os.environ.get(
                    "PATCHLAB_PARAM_MODEL",
                    PROJECT_ROOT / "data" / "models" / "param_model.pt",
                )
            ).expanduser().resolve()
        ),
        delta_model=str(
            Path(
                os.environ.get(
                    "PATCHLAB_DELTA_MODEL",
                    PROJECT_ROOT / "data" / "models" / "delta_param_model.pt",
                )
            ).expanduser().resolve()
        ),
    )


def _load_resumable(path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if all(payload.get(key) == value for key, value in expected.items()):
        return payload
    return None


def run_bam_suite(
    *,
    bam_dir: Path,
    detail_dir: Path,
    stack: StackConfiguration,
    budget: str,
    limit: int | None,
) -> dict[str, Any]:
    files = benchmark_audio_files(bam_dir)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No benchmark audio found in {bam_dir}")
    output_dir = detail_dir / "bam"
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dir = detail_dir / "match-sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, source in enumerate(files, start=1):
        source_hash = _sha256(source)
        target_synth = target_synth_for_name(source.name)
        expected = {
            "source_sha256": source_hash,
            "budget": budget,
            "target_synth": target_synth,
            "stack": asdict(stack),
        }
        detail_path = output_dir / _detail_name(source)
        existing = _load_resumable(detail_path, expected)
        if existing is not None and existing.get("status") == "complete":
            row = existing
            disposition = "resumed"
        else:
            sample_started = time.monotonic()
            print(
                f"BAM_START={index}/{len(files)} file={source.name} synth={target_synth}",
                flush=True,
            )
            try:
                result_path = run_match_file(
                    source,
                    target_synth=target_synth,
                    budget=budget,
                    session_root=session_dir,
                    progress_callback=lambda value, i=index, total=len(files): print(
                        "BAM_PROGRESS="
                        + json.dumps({"sample": i, "total": total, **value}, sort_keys=True),
                        flush=True,
                    ),
                )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                recommendation = result.get("recommendation") or {}
                row = {
                    **expected,
                    "status": result.get("status"),
                    "source": str(source.resolve()),
                    "source_name": source.name,
                    "clap_similarity": recommendation.get("clap_similarity"),
                    "meaningfully_modified": recommendation.get("meaningfully_modified"),
                    "evaluations": recommendation.get("evaluations", 0),
                    "match_elapsed_s": recommendation.get("elapsed_s", 0.0),
                    "wall_clock_s": time.monotonic() - sample_started,
                    "result_path": str(result_path.resolve()),
                    "error": None,
                }
            except Exception as exc:
                row = {
                    **expected,
                    "status": "error",
                    "source": str(source.resolve()),
                    "source_name": source.name,
                    "clap_similarity": None,
                    "meaningfully_modified": None,
                    "evaluations": 0,
                    "match_elapsed_s": 0.0,
                    "wall_clock_s": time.monotonic() - sample_started,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=12),
                }
            _atomic_json(detail_path, row)
            disposition = row["status"]
        rows.append(row)
        print(
            f"BAM_RESULT={index}/{len(files)} status={disposition} "
            f"similarity={row.get('clap_similarity')}",
            flush=True,
        )
    scores = [float(row["clap_similarity"]) for row in rows if row.get("clap_similarity") is not None]
    errors = [row for row in rows if row.get("status") != "complete"]
    return {
        "requested": len(files),
        "completed": len(scores),
        "failed": len(errors),
        "mean_clap_similarity": mean(scores) if scores else None,
        "median_clap_similarity": median(scores) if scores else None,
        "min_clap_similarity": min(scores) if scores else None,
        "meaningfully_modified": sum(bool(row.get("meaningfully_modified")) for row in rows),
        "evaluations": sum(int(row.get("evaluations") or 0) for row in rows),
        "elapsed_s": time.monotonic() - started,
        "detail_dir": str(output_dir.resolve()),
        "errors": [
            {"source_name": row["source_name"], "error": row.get("error")} for row in errors
        ],
    }


class FactoryRenderer:
    """Cache C3 renders while reusing PatchLab's verified headless host path."""

    def __init__(self, cache_root: Path, local_paths: dict[str, str]) -> None:
        self.cache_root = Path(cache_root).resolve()
        self.local_paths = local_paths
        self.hosts: dict[str, tuple[Any, Any]] = {}

    def _host(self, synth: str) -> tuple[Any, Any]:
        if synth not in self.hosts:
            required = "VST2" if synth == "serum1" else "VST3"
            candidate = next(
                item
                for item in ENV.plugins_for(synth)
                if item.format == required and item.hostable
            )
            self.hosts[synth] = make_dawdreamer_processor(candidate)
        return self.hosts[synth]

    def render(self, selected: BenchmarkFactoryPreset, midi_note: int = 60) -> Path:
        preset = selected.bundle
        output = self.cache_root / str(selected.catalog_id) / f"{midi_note}.wav"
        if output.is_file():
            return output
        engine, processor = self._host(preset.synth)
        if preset.synth == "serum1":
            source = self.local_paths.get(preset.content_hash)
            if not source:
                raise FileNotFoundError(
                    f"No locally verified Serum 1 source for {preset.id}:{preset.name}"
                )
            if processor.load_preset(str(Path(source).resolve())) is False:
                raise RuntimeError(f"Serum 1 rejected preset {preset.id}:{preset.name}")
        else:
            load_render_state(processor, selected.catalog_id)
        audio = _trim_tail(_render_audio(engine, processor, midi_note))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.wav")
        sf.write(temporary, audio.T, SAMPLE_RATE, subtype="FLOAT", format="WAV")
        temporary.replace(output)
        return output


def _read_comparison_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    prepared, duration = prepare_query_audio(mono, int(sample_rate), adaptive=True)
    return embedding_comparison_audio(prepared, duration, adaptive=True)


def _select_factory_presets(
    *,
    bundle: FactoryBundle,
    count: int,
    seed: int,
    local_paths: dict[str, str],
    indexed_ids: set[int],
) -> list[BenchmarkFactoryPreset]:
    assets = resolve_synthesis_assets()
    with closing(sqlite3.connect(assets.library_db)) as connection:
        catalog_id_by_hash = {
            str(content_hash): int(preset_id)
            for preset_id, content_hash in connection.execute(
                "SELECT id,content_hash FROM presets"
            )
        }
    candidates = [
        BenchmarkFactoryPreset(preset, catalog_id_by_hash[preset.content_hash])
        for preset in bundle.presets(searchable_only=True)
        if preset.content_hash in catalog_id_by_hash
        and catalog_id_by_hash[preset.content_hash] in indexed_ids
        and (
            (preset.synth == "serum1" and preset.content_hash in local_paths)
            or (
                preset.synth == "serum2"
                and assets.find_render_state(catalog_id_by_hash[preset.content_hash])
                is not None
            )
        )
    ]
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} renderable indexed factory presets; need {count}"
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def _embed_in_batches(embedder: ClapEmbedder, audio: Sequence[np.ndarray], batch_size: int = 8) -> np.ndarray:
    rows = []
    for start in range(0, len(audio), batch_size):
        stop = min(start + batch_size, len(audio))
        rows.append(embedder.embed(audio[start:stop]))
        print(f"EMBED_PROGRESS={stop}/{len(audio)}", flush=True)
    return np.concatenate(rows, axis=0)


def _retrieval_rows(
    embeddings: np.ndarray,
    expected_ids: Sequence[int],
    index: np.ndarray,
    index_ids: np.ndarray,
) -> list[dict[str, Any]]:
    scores, positions = cosine_topk(
        l2_normalize(np.asarray(embeddings, dtype=np.float32)),
        np.asarray(index),
        k=5,
        normalized=True,
    )
    rows = []
    for expected, row_scores, row_positions in zip(
        expected_ids, scores, positions, strict=True
    ):
        found = [int(index_ids[position]) for position in row_positions]
        rows.append(
            {
                "preset_id": int(expected),
                "retrieved_ids": found,
                "scores": [float(value) for value in row_scores],
                "top1": bool(found[0] == expected),
                "top5": bool(expected in found),
            }
        )
    return rows


def _musical_gate(audio: np.ndarray, *, bpm: float, division: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    samples_per_beat = CLAP_SAMPLE_RATE * 60.0 / bpm
    period = max(2, int(round(samples_per_beat * 4.0 / division)))
    phase = np.arange(len(values), dtype=np.int64) % period
    envelope = (phase < int(round(period * 0.52))).astype(np.float32)
    ramp = max(1, int(round(CLAP_SAMPLE_RATE * 0.004)))
    kernel = np.ones(ramp, dtype=np.float32) / ramp
    envelope = np.convolve(envelope, kernel, mode="same").astype(np.float32)
    return np.ascontiguousarray(values * envelope, dtype=np.float32)


def _codec_roundtrip(audio: np.ndarray, temporary: Path) -> np.ndarray:
    import imageio_ffmpeg

    source = temporary / "source.wav"
    encoded = temporary / "encoded.mp3"
    decoded = temporary / "decoded.wav"
    sf.write(source, audio, CLAP_SAMPLE_RATE, subtype="PCM_16", format="WAV")
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for command in (
        [executable, "-v", "error", "-y", "-i", str(source), "-b:a", "96k", str(encoded)],
        [
            executable,
            "-v",
            "error",
            "-y",
            "-i",
            str(encoded),
            "-ac",
            "1",
            "-ar",
            str(CLAP_SAMPLE_RATE),
            str(decoded),
        ],
    ):
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, creationflags=flags
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip()[-1000:])
    result, _rate = sf.read(decoded, dtype="float32", always_2d=False)
    return np.ascontiguousarray(result, dtype=np.float32)


def _invariance_variants(audio: np.ndarray, *, seed: int) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    rng = random.Random(seed)
    bpm = float(rng.randint(140, 174))
    variants: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for division in (4, 8, 16):
        variants.append(
            (
                f"gate_1_{division}",
                _musical_gate(audio, bpm=bpm, division=division),
                {"bpm": bpm, "division": f"1/{division}"},
            )
        )
    semitones = rng.choice((-12, -7, -2, 2, 7, 12))
    pitched = librosa.effects.pitch_shift(
        audio, sr=CLAP_SAMPLE_RATE, n_steps=float(semitones), res_type="soxr_hq"
    ).astype(np.float32)
    variants.append(("pitch", pitched, {"semitones": semitones}))
    gain_db = rng.choice((-6.0, 6.0))
    variants.append(
        ("loudness", (audio * (10.0 ** (gain_db / 20.0))).astype(np.float32), {"gain_db": gain_db})
    )
    with tempfile.TemporaryDirectory(prefix="patchlab-stage2-codec-") as directory:
        codec = _codec_roundtrip(audio, Path(directory))
    variants.append(("codec_mp3_96k", codec, {"bitrate": "96k"}))
    return variants


def run_retrieval_suites(
    *,
    detail_dir: Path,
    stack: StackConfiguration,
    seed: int,
    factory_count: int,
    invariance_count: int,
    factory_bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assets = resolve_synthesis_assets()
    manifest = np.load(assets.feature_dir / "similarity_manifest.npz")
    index_ids = manifest["preset_ids"].astype(np.int64)
    index = l2_normalize(np.load(assets.preset_index, mmap_mode="r"))
    verification_path = detail_dir / "factory-verification.json"
    verification = verify_local_factory_install(
        bundle_path=factory_bundle, mapping_path=verification_path
    )
    bundle = FactoryBundle(factory_bundle)
    selected = _select_factory_presets(
        bundle=bundle,
        count=max(factory_count, invariance_count),
        seed=seed,
        local_paths=verification.local_paths_by_hash,
        indexed_ids={int(value) for value in index_ids},
    )
    renderer = FactoryRenderer(
        detail_dir / "factory-renders-by-catalog-id-v2",
        verification.local_paths_by_hash,
    )
    render_rows = []
    comparison_audio = []
    for position, selected_preset in enumerate(selected, start=1):
        preset = selected_preset.bundle
        started = time.monotonic()
        try:
            path = renderer.render(selected_preset, 60)
            comparison_audio.append(_read_comparison_audio(path))
            render_rows.append(
                {
                    "preset_id": selected_preset.catalog_id,
                    "factory_bundle_preset_id": preset.id,
                    "name": preset.name,
                    "synth": preset.synth,
                    "render_path": str(path.resolve()),
                    "elapsed_s": time.monotonic() - started,
                    "error": None,
                }
            )
        except Exception as exc:
            render_rows.append(
                {
                    "preset_id": selected_preset.catalog_id,
                    "factory_bundle_preset_id": preset.id,
                    "name": preset.name,
                    "synth": preset.synth,
                    "render_path": None,
                    "elapsed_s": time.monotonic() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            comparison_audio.append(np.zeros(CLAP_SAMPLE_RATE * 4, dtype=np.float32))
        print(
            f"FACTORY_RENDER={position}/{len(selected)} preset={selected_preset.catalog_id} "
            f"status={'ok' if render_rows[-1]['error'] is None else 'error'}",
            flush=True,
        )
    _atomic_json(detail_dir / "factory-render-details.json", render_rows)
    successful_positions = [
        index for index, row in enumerate(render_rows) if row["error"] is None
    ]
    if len(successful_positions) < max(factory_count, invariance_count):
        raise RuntimeError(
            f"Only {len(successful_positions)} factory renders succeeded; need "
            f"{max(factory_count, invariance_count)}"
        )
    embedder = ClapEmbedder(ENV)
    retrieval_positions = successful_positions[:factory_count]
    clean_embeddings = _embed_in_batches(
        embedder, [comparison_audio[position] for position in retrieval_positions]
    )
    clean_expected = [selected[position].catalog_id for position in retrieval_positions]
    clean_rows = _retrieval_rows(clean_embeddings, clean_expected, index, index_ids)
    for row, position in zip(clean_rows, retrieval_positions, strict=True):
        row.update(
            {
                "name": selected[position].bundle.name,
                "synth": selected[position].bundle.synth,
                "factory_bundle_preset_id": selected[position].bundle.id,
                "render_path": render_rows[position]["render_path"],
            }
        )
    _atomic_json(detail_dir / "library-retrieval-details.json", clean_rows)
    retrieval = {
        "requested": factory_count,
        "completed": len(clean_rows),
        "retrieval_at_1": mean(row["top1"] for row in clean_rows),
        "retrieval_at_5": mean(row["top5"] for row in clean_rows),
        "seed": seed,
    }

    invariant_positions = successful_positions[:invariance_count]
    variant_audio: list[np.ndarray] = []
    variant_meta: list[dict[str, Any]] = []
    for position in invariant_positions:
        selected_preset = selected[position]
        preset = selected_preset.bundle
        for variant_name, values, parameters in _invariance_variants(
            comparison_audio[position], seed=seed + selected_preset.catalog_id
        ):
            variant_audio.append(values)
            variant_meta.append(
                {
                    "preset_id": selected_preset.catalog_id,
                    "factory_bundle_preset_id": preset.id,
                    "name": preset.name,
                    "synth": preset.synth,
                    "variant": variant_name,
                    "parameters": parameters,
                }
            )
    variant_embeddings = _embed_in_batches(embedder, variant_audio)
    invariant_rows = _retrieval_rows(
        variant_embeddings,
        [row["preset_id"] for row in variant_meta],
        index,
        index_ids,
    )
    for row, metadata in zip(invariant_rows, variant_meta, strict=True):
        row.update(metadata)
    _atomic_json(detail_dir / "rhythm-invariance-details.json", invariant_rows)
    by_variant: dict[str, dict[str, Any]] = {}
    for variant in sorted({row["variant"] for row in invariant_rows}):
        subset = [row for row in invariant_rows if row["variant"] == variant]
        by_variant[variant] = {
            "count": len(subset),
            "retrieval_at_1": mean(row["top1"] for row in subset),
            "retrieval_at_5": mean(row["top5"] for row in subset),
        }
    invariance = {
        "base_presets": invariance_count,
        "variants": len(invariant_rows),
        "retrieval_at_1": mean(row["top1"] for row in invariant_rows),
        "retrieval_at_5": mean(row["top5"] for row in invariant_rows),
        "by_variant": by_variant,
        "seed": seed,
    }
    return retrieval, invariance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-name", default="baseline")
    parser.add_argument("--bam-dir", type=Path, default=DEFAULT_BAM_DIR)
    parser.add_argument("--detail-dir", type=Path, default=DEFAULT_DETAIL_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--factory-bundle", type=Path, default=DEFAULT_FACTORY_BUNDLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--budget", choices=("quick", "balanced", "best"), default="balanced")
    parser.add_argument("--factory-count", type=int, default=200)
    parser.add_argument("--invariance-count", type=int, default=50)
    parser.add_argument("--bam-limit", type=int)
    parser.add_argument(
        "--suite",
        choices=("all", "bam", "retrieval"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.factory_count < 200:
        raise ValueError("--factory-count must be at least 200")
    if args.invariance_count < 50:
        raise ValueError("--invariance-count must be at least 50")
    if args.bam_limit is not None and args.bam_limit <= 0:
        raise ValueError("--bam-limit must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    configure_model_environment()
    detail_dir = args.detail_dir.expanduser().resolve()
    detail_dir.mkdir(parents=True, exist_ok=True)
    stack = _stack_configuration(args.stack_name, args.factory_bundle)
    started = time.monotonic()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": 2,
        "stack": asdict(stack),
        "seed": args.seed,
        "started_at": _utc_now(),
        "benchmark_population": {
            "policy": "root-level real audio; AppleDouble, Ableton .asd, and metadata excluded",
            "user_authorized_mixed_formats": True,
        },
    }
    try:
        if args.suite in {"all", "bam"}:
            summary["bam"] = run_bam_suite(
                bam_dir=args.bam_dir,
                detail_dir=detail_dir,
                stack=stack,
                budget=args.budget,
                limit=args.bam_limit,
            )
        if args.suite in {"all", "retrieval"}:
            retrieval, invariance = run_retrieval_suites(
                detail_dir=detail_dir,
                stack=stack,
                seed=args.seed,
                factory_count=args.factory_count,
                invariance_count=args.invariance_count,
                factory_bundle=args.factory_bundle,
            )
            summary["library_retrieval"] = retrieval
            summary["rhythm_invariance"] = invariance
        summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc(limit=20)
    summary["completed_at"] = _utc_now()
    summary["elapsed_s"] = time.monotonic() - started
    _atomic_json(args.summary.expanduser().resolve(), summary)
    _atomic_json(detail_dir / "summary.json", summary)
    print("STAGE2_BENCHMARK_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
