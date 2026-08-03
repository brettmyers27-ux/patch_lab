#!/usr/bin/env python3
"""Generate the disk-bounded, resumable Stage 2 synth training corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import _serum1_targets, _serum2_targets
from core.features import CLAP_DIMENSIONS, HANDCRAFTED_NAMES, ClapEmbedder, handcrafted_features
from core.factory_verify import verify_local_factory_install
from core.matcher import Candidate, _init_render_worker, _render_candidate
from core.model_assets import configure_model_environment
from core.perturbation import perturb_serum1, perturb_serum2
from core.platform_env import ENV
from core.synthesis_assets import resolve_synthesis_assets


DEFAULT_ROOT = PROJECT_ROOT / "data" / "stage2" / "training"
DEFAULT_BASE_CLIPS = 30_000
DEFAULT_VARIANTS_PER_CLIP = 13
DEFAULT_SEED = 20260802
MIDI_NOTES = (24, 36, 48, 60, 72, 84, 96)
NOTES_PER_PATCH = 3
TARGET_RATE = 48_000
RENDER_SECONDS = 4.0
SERUM1_WEIGHT = 0.70


@dataclass(frozen=True, slots=True)
class BaseDefinition:
    base_index: int
    provenance_key: str
    synth: str
    synth_code: int
    preset_id: int
    content_hash: str
    midi_note: int
    perturb_seed: int
    vector: np.ndarray
    mask: np.ndarray
    change: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def provenance_key(
    *, content_hash: str, synth: str, perturb_seed: int, midi_note: int
) -> str:
    raw = f"{content_hash}|{synth}|{perturb_seed}|{midi_note}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _smooth_gate(audio: np.ndarray, *, bpm: float, division: int) -> np.ndarray:
    samples_per_beat = TARGET_RATE * 60.0 / bpm
    period = max(2, int(round(samples_per_beat * 4.0 / division)))
    phase = np.arange(len(audio), dtype=np.int64) % period
    envelope = (phase < int(round(period * 0.52))).astype(np.float32)
    ramp = max(1, int(round(TARGET_RATE * 0.004)))
    envelope = np.convolve(
        envelope, np.ones(ramp, dtype=np.float32) / ramp, mode="same"
    ).astype(np.float32)
    return np.ascontiguousarray(audio * envelope, dtype=np.float32)


def training_variants(
    waveform: np.ndarray, *, seed: int
) -> list[tuple[str, np.ndarray]]:
    """Return exactly 13 deterministic in-memory variants for one render."""

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(seed)
    variants: list[tuple[str, np.ndarray]] = [("clean", audio.copy())]
    for division, bpm in ((4, 148.0), (8, 160.0), (16, 172.0)):
        variants.append(
            (f"gate_1_{division}", _smooth_gate(audio, bpm=bpm, division=division))
        )
    for semitones in (-12, -7, -2, 2, 7, 12):
        shifted = librosa.effects.pitch_shift(
            audio,
            sr=TARGET_RATE,
            n_steps=float(semitones),
            res_type="soxr_hq",
        ).astype(np.float32)
        variants.append((f"pitch_{semitones:+d}", shifted))
    for gain_db in (-6.0, 6.0):
        variants.append(
            (
                f"gain_{gain_db:+.0f}db",
                np.ascontiguousarray(
                    audio * (10.0 ** (gain_db / 20.0)), dtype=np.float32
                ),
            )
        )
    rms = max(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))), 1e-6)
    noise = rng.normal(0.0, rms * 0.01, len(audio)).astype(np.float32)
    variants.append(("noise_40db", np.ascontiguousarray(audio + noise, dtype=np.float32)))
    if len(variants) != DEFAULT_VARIANTS_PER_CLIP:
        raise AssertionError(f"Expected 13 variants, produced {len(variants)}")
    return variants


def _manifest_connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS base_clips (
          base_index INTEGER PRIMARY KEY,
          provenance_key TEXT NOT NULL UNIQUE,
          synth TEXT NOT NULL,
          preset_id INTEGER NOT NULL,
          midi_note INTEGER NOT NULL,
          perturb_seed INTEGER NOT NULL,
          raw_audio_path TEXT NOT NULL,
          shard_path TEXT NOT NULL,
          pair_count INTEGER NOT NULL,
          rms_dbfs REAL NOT NULL,
          render_coverage REAL NOT NULL,
          completed_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    return connection


def _validate_metadata(
    connection: sqlite3.Connection,
    *,
    base_clips: int,
    variants_per_clip: int,
    seed: int,
) -> None:
    expected = {
        "schema_version": "2",
        "base_clips": str(base_clips),
        "variants_per_clip": str(variants_per_clip),
        "seed": str(seed),
        "sample_rate": str(TARGET_RATE),
        "render_seconds": str(RENDER_SECONDS),
        "notes_per_patch": str(NOTES_PER_PATCH),
    }
    existing = dict(connection.execute("SELECT key,value FROM metadata"))
    if existing and existing != expected:
        raise RuntimeError(
            "Existing Stage 2 training manifest uses different settings: "
            f"existing={existing}, requested={expected}"
        )
    connection.executemany(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", expected.items()
    )
    connection.commit()


def _catalog_hashes(database: Path) -> dict[int, str]:
    with closing(sqlite3.connect(database)) as connection:
        return {
            int(preset_id): str(content_hash)
            for preset_id, content_hash in connection.execute(
                "SELECT id,content_hash FROM presets"
            )
        }


def _base_definition(
    index: int,
    *,
    attempt: int,
    seed: int,
    serum1: Any,
    serum2: Any,
    serum2_schema: dict[str, Any],
    hashes: dict[int, str],
    eligible_ids: dict[int, np.ndarray],
    forced_synth_code: int | None = None,
) -> BaseDefinition:
    patch_index = index // NOTES_PER_PATCH
    note_slot = index % NOTES_PER_PATCH
    perturb_seed = int(
        np.random.SeedSequence([seed, patch_index, attempt]).generate_state(1)[0]
    )
    rng = np.random.default_rng(perturb_seed)
    synth_code = (
        forced_synth_code
        if forced_synth_code is not None
        else 1
        if float(rng.random()) < SERUM1_WEIGHT
        else 2
    )
    synth = "serum1" if synth_code == 1 else "serum2"
    store = serum1 if synth_code == 1 else serum2
    preset_ids = eligible_ids[synth_code]
    preset_id = int(rng.choice(preset_ids))
    row = store.preset_row[preset_id]
    base = np.asarray(store.vectors[row], dtype=np.float32)
    mask = np.asarray(store.masks[row], dtype=np.bool_)
    if synth_code == 1:
        vector, change = perturb_serum1(base, mask, store.mapping, rng)
    else:
        vector, change = perturb_serum2(base, mask, serum2_schema, rng)
    patch_notes = rng.choice(
        np.asarray(MIDI_NOTES), size=NOTES_PER_PATCH, replace=False
    )
    midi_note = int(patch_notes[note_slot])
    content_hash = hashes[preset_id]
    return BaseDefinition(
        base_index=index,
        provenance_key=provenance_key(
            content_hash=content_hash,
            synth=synth,
            perturb_seed=perturb_seed,
            midi_note=midi_note,
        ),
        synth=synth,
        synth_code=synth_code,
        preset_id=preset_id,
        content_hash=content_hash,
        midi_note=midi_note,
        perturb_seed=perturb_seed,
        vector=vector,
        mask=mask,
        change=change,
    )


def _render_batch(
    pool: Any,
    definitions: Sequence[BaseDefinition],
) -> list[tuple[np.ndarray | None, float, str | None]]:
    payloads = [
        (
            Candidate(
                definition.synth,
                definition.preset_id,
                definition.vector,
                definition.mask,
                "stage2-training",
                midi_note=definition.midi_note,
            ),
            definition.midi_note,
            RENDER_SECONDS,
        )
        for definition in definitions
    ]
    return pool.map(_render_candidate, payloads, chunksize=1)


def _features_for_variants(
    embedder: ClapEmbedder,
    variants: Sequence[tuple[str, np.ndarray]],
    *,
    embedding_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    audio = [values for _name, values in variants]
    embedded = []
    for start in range(0, len(audio), embedding_batch_size):
        embedded.append(embedder.embed(audio[start : start + embedding_batch_size]))
    with ThreadPoolExecutor(max_workers=min(8, len(audio))) as executor:
        features = np.stack(
            list(executor.map(handcrafted_features, audio)), axis=0
        ).astype(np.float32)
    return np.concatenate(embedded, axis=0), features


def _write_shard(
    path: Path,
    *,
    definitions: Sequence[BaseDefinition],
    embeddings: Sequence[np.ndarray],
    features: Sequence[np.ndarray],
    augmentation_names: Sequence[Sequence[str]],
) -> int:
    base_count = len(definitions)
    pair_count = sum(len(values) for values in embeddings)
    maximum_vector_length = max(len(item.vector) for item in definitions)
    vectors = np.zeros((base_count, maximum_vector_length), dtype=np.float16)
    packed_masks = np.zeros(
        (base_count, (maximum_vector_length + 7) // 8), dtype=np.uint8
    )
    lengths = np.zeros(base_count, dtype=np.uint16)
    for row, definition in enumerate(definitions):
        length = len(definition.vector)
        vectors[row, :length] = definition.vector.astype(np.float16)
        packed = np.packbits(definition.mask, bitorder="little")
        packed_masks[row, : len(packed)] = packed
        lengths[row] = length
    pair_base_rows = np.concatenate(
        [np.full(len(values), row, dtype=np.uint16) for row, values in enumerate(embeddings)]
    )
    payload = {
        "base_indices": np.asarray([item.base_index for item in definitions], dtype=np.int32),
        "provenance_keys": np.asarray([item.provenance_key for item in definitions], dtype="U64"),
        "synth_codes": np.asarray([item.synth_code for item in definitions], dtype=np.uint8),
        "preset_ids": np.asarray([item.preset_id for item in definitions], dtype=np.int32),
        "midi_notes": np.asarray([item.midi_note for item in definitions], dtype=np.int16),
        "perturb_seeds": np.asarray([item.perturb_seed for item in definitions], dtype=np.uint32),
        "vector_lengths": lengths,
        "parameter_vectors": vectors,
        "parameter_masks_packed": packed_masks,
        "pair_base_rows": pair_base_rows,
        "augmentation_names": np.asarray(
            [name for names in augmentation_names for name in names], dtype="U24"
        ),
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float16),
        "handcrafted": np.concatenate(features, axis=0).astype(np.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    temporary.replace(path)
    return pair_count


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base-clips", type=int, default=DEFAULT_BASE_CLIPS)
    parser.add_argument("--variants-per-clip", type=int, default=DEFAULT_VARIANTS_PER_CLIP)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--batch-bases", type=int, default=30)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--variant-base-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit-bases", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.base_clips <= 0
        or args.processes <= 0
        or args.batch_bases <= 0
        or args.variant_base_group <= 0
    ):
        raise ValueError("Counts and process settings must be positive")
    if args.variants_per_clip != DEFAULT_VARIANTS_PER_CLIP:
        raise ValueError("Stage 2's fixed augmentation recipe has exactly 13 variants")
    if args.base_clips % NOTES_PER_PATCH or args.batch_bases % NOTES_PER_PATCH:
        raise ValueError("Base and batch clip counts must preserve complete three-note patch groups")
    if args.limit_bases is not None and args.limit_bases % NOTES_PER_PATCH:
        raise ValueError("--limit-bases must preserve complete three-note patch groups")
    if args.base_clips * args.variants_per_clip < 390_000 and args.limit_bases is None:
        raise ValueError("Full Stage 2 generation must produce at least 390,000 pairs")
    os.environ.setdefault("PATCHLAB_DISTRIBUTION_MODE", "1")
    configure_model_environment()
    assets = resolve_synthesis_assets()
    output_root = args.output_root.expanduser().resolve()
    raw_root = output_root / "raw-audio"
    shard_root = output_root / "shards"
    report_path = output_root / "generation-report.json"
    raw_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    connection = _manifest_connection(output_root / "manifest.sqlite")
    try:
        _validate_metadata(
            connection,
            base_clips=args.base_clips,
            variants_per_clip=args.variants_per_clip,
            seed=args.seed,
        )
        complete = {
            int(row[0]) for row in connection.execute("SELECT base_index FROM base_clips")
        }
        pending = [index for index in range(args.base_clips) if index not in complete]
        if args.limit_bases is not None:
            pending = pending[: args.limit_bases]
        serum1 = _serum1_targets(assets.library_db)
        serum2 = _serum2_targets(assets.serum2_targets, assets.serum2_schema)
        serum2_schema = json.loads(assets.serum2_schema.read_text(encoding="utf-8"))
        hashes = _catalog_hashes(assets.library_db)
        verification = verify_local_factory_install(
            mapping_path=assets.factory_mapping
        )
        locally_verified_hashes = set(verification.local_paths_by_hash)
        eligible_ids = {
            1: np.asarray(
                sorted(
                    preset_id
                    for preset_id in serum1.preset_row
                    if hashes.get(preset_id) in locally_verified_hashes
                ),
                dtype=np.int64,
            ),
            2: np.asarray(
                sorted(
                    preset_id
                    for preset_id in serum2.preset_row
                    if assets.find_render_state(preset_id) is not None
                ),
                dtype=np.int64,
            ),
        }
        if any(len(values) == 0 for values in eligible_ids.values()):
            raise RuntimeError(
                "No locally renderable training bases for one or both synths: "
                f"serum1={len(eligible_ids[1])}, serum2={len(eligible_ids[2])}"
            )
        embedder = ClapEmbedder(ENV)
        context = mp.get_context("spawn")
        started = time.monotonic()
        processed = 0
        retries = 0
        scratch = tempfile.TemporaryDirectory(prefix="patchlab-stage2-generate-")
        try:
            with context.Pool(
                args.processes,
                initializer=_init_render_worker,
                initargs=(scratch.name, assets),
            ) as pool:
                for offset in range(0, len(pending), args.batch_bases):
                    indices = pending[offset : offset + args.batch_bases]
                    definitions = [
                        _base_definition(
                            index,
                            attempt=0,
                            seed=args.seed,
                            serum1=serum1,
                            serum2=serum2,
                            serum2_schema=serum2_schema,
                            hashes=hashes,
                            eligible_ids=eligible_ids,
                        )
                        for index in indices
                    ]
                    rendered: list[tuple[np.ndarray, float]] = []
                    for group_start in range(0, len(definitions), NOTES_PER_PATCH):
                        group = definitions[group_start : group_start + NOTES_PER_PATCH]
                        forced_synth = group[0].synth_code
                        last_errors: list[str] = []
                        for attempt in range(8):
                            if attempt:
                                retries += NOTES_PER_PATCH
                                group = [
                                    _base_definition(
                                        definition.base_index,
                                        attempt=attempt,
                                        seed=args.seed,
                                        serum1=serum1,
                                        serum2=serum2,
                                        serum2_schema=serum2_schema,
                                        hashes=hashes,
                                        eligible_ids=eligible_ids,
                                        forced_synth_code=forced_synth,
                                    )
                                    for definition in group
                                ]
                            results = _render_batch(pool, group)
                            accepted: list[tuple[np.ndarray, float]] = []
                            last_errors = []
                            for waveform, coverage, error in results:
                                values = (
                                    np.asarray(waveform, dtype=np.float32)
                                    if waveform is not None
                                    else np.asarray([], dtype=np.float32)
                                )
                                rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))) if len(values) else 0.0
                                if waveform is None or not np.all(np.isfinite(values)) or rms <= 1e-4:
                                    last_errors.append(error or f"rms={rms}")
                                else:
                                    accepted.append((values, float(coverage)))
                            if not last_errors:
                                definitions[group_start : group_start + NOTES_PER_PATCH] = group
                                rendered.extend(accepted)
                                break
                            if attempt == 7:
                                raise RuntimeError(
                                    f"Render failed for patch group {group[0].base_index // NOTES_PER_PATCH} "
                                    f"after 8 attempts: {last_errors[0]}"
                                )
                    embedding_rows: list[np.ndarray] = []
                    feature_rows: list[np.ndarray] = []
                    augmentation_rows: list[list[str]] = []
                    rms_rows: list[float] = []
                    all_variants: list[list[tuple[str, np.ndarray]]] = []
                    for definition, (waveform, _coverage) in zip(definitions, rendered, strict=True):
                        raw_path = raw_root / f"{definition.base_index:06d}.wav"
                        temporary_raw = raw_path.with_suffix(".tmp.wav")
                        sf.write(
                            temporary_raw,
                            waveform,
                            TARGET_RATE,
                            subtype="FLOAT",
                            format="WAV",
                        )
                        temporary_raw.replace(raw_path)
                        all_variants.append(
                            training_variants(
                                waveform, seed=definition.perturb_seed
                            )
                        )
                        rms = float(
                            np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))
                        )
                        rms_rows.append(20.0 * np.log10(max(rms, 1e-12)))
                    for group_start in range(0, len(definitions), args.variant_base_group):
                        group = all_variants[
                            group_start : group_start + args.variant_base_group
                        ]
                        flattened = [variant for variants in group for variant in variants]
                        embedded, handcrafted = _features_for_variants(
                            embedder,
                            flattened,
                            embedding_batch_size=args.embedding_batch_size,
                        )
                        cursor = 0
                        for variants in group:
                            stop = cursor + len(variants)
                            embedding_rows.append(embedded[cursor:stop])
                            feature_rows.append(handcrafted[cursor:stop])
                            augmentation_rows.append(
                                [name for name, _values in variants]
                            )
                            cursor = stop
                    shard_path = shard_root / (
                        f"shard-{indices[0]:06d}-{indices[-1]:06d}.npz"
                    )
                    pair_count = _write_shard(
                        shard_path,
                        definitions=definitions,
                        embeddings=embedding_rows,
                        features=feature_rows,
                        augmentation_names=augmentation_rows,
                    )
                    completed_at = _utc_now()
                    connection.executemany(
                        """
                        INSERT INTO base_clips(
                          base_index,provenance_key,synth,preset_id,midi_note,
                          perturb_seed,raw_audio_path,shard_path,pair_count,
                          rms_dbfs,render_coverage,completed_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        [
                            (
                                definition.base_index,
                                definition.provenance_key,
                                definition.synth,
                                definition.preset_id,
                                definition.midi_note,
                                definition.perturb_seed,
                                str((raw_root / f"{definition.base_index:06d}.wav").resolve()),
                                str(shard_path.resolve()),
                                len(embedding_rows[row]),
                                rms_rows[row],
                                rendered[row][1],
                                completed_at,
                            )
                            for row, definition in enumerate(definitions)
                        ],
                    )
                    connection.commit()
                    processed += len(definitions)
                    total_complete = int(
                        connection.execute("SELECT COUNT(*) FROM base_clips").fetchone()[0]
                    )
                    total_pairs = int(
                        connection.execute(
                            "SELECT COALESCE(SUM(pair_count),0) FROM base_clips"
                        ).fetchone()[0]
                    )
                    coverage_min, coverage_mean, coverage_max = connection.execute(
                        "SELECT MIN(render_coverage),AVG(render_coverage),MAX(render_coverage) FROM base_clips"
                    ).fetchone()
                    distinct_presets = int(
                        connection.execute(
                            "SELECT COUNT(DISTINCT synth || ':' || preset_id) FROM base_clips"
                        ).fetchone()[0]
                    )
                    elapsed = max(time.monotonic() - started, 1e-6)
                    rate = processed / elapsed
                    report = {
                        "status": (
                            "complete"
                            if total_complete == args.base_clips
                            else "partial"
                        ),
                        "base_clips": total_complete,
                        "target_base_clips": args.base_clips,
                        "training_pairs": total_pairs,
                        "target_training_pairs": args.base_clips
                        * args.variants_per_clip,
                        "exact_patch_groups": total_complete // NOTES_PER_PATCH,
                        "notes_per_patch": NOTES_PER_PATCH,
                        "distinct_source_presets": distinct_presets,
                        "new_base_clips_this_run": processed,
                        "render_retries_this_run": retries,
                        "serum1_base_clips": int(
                            connection.execute(
                                "SELECT COUNT(*) FROM base_clips WHERE synth='serum1'"
                            ).fetchone()[0]
                        ),
                        "serum2_base_clips": int(
                            connection.execute(
                                "SELECT COUNT(*) FROM base_clips WHERE synth='serum2'"
                            ).fetchone()[0]
                        ),
                        "render_coverage": {
                            "minimum": float(coverage_min),
                            "mean": float(coverage_mean),
                            "maximum": float(coverage_max),
                        },
                        "raw_audio_bytes": _directory_bytes(raw_root),
                        "shard_bytes": _directory_bytes(shard_root),
                        "rate_base_clips_per_second": rate,
                        "eta_seconds": (
                            (len(pending) - processed) / rate if rate else None
                        ),
                        "processes": args.processes,
                        "seed": args.seed,
                        "variants_per_clip": args.variants_per_clip,
                        "serum2_automation_coverage_limit": "approximately 40%; structural",
                        "updated_at": completed_at,
                    }
                    _atomic_json(report_path, report)
                    print(
                        "STAGE2_GENERATION_PROGRESS="
                        + json.dumps(report, sort_keys=True),
                        flush=True,
                    )
        finally:
            scratch.cleanup()
        final = json.loads(report_path.read_text(encoding="utf-8"))
        if args.limit_bases is None and final["status"] != "complete":
            raise RuntimeError("Stage 2 generator stopped before completing its target")
        print("STAGE2_GENERATION_SUMMARY=" + json.dumps(final, sort_keys=True))
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
