#!/usr/bin/env python3
"""Partially fine-tune CLAP's audio tower on synth-domain invariances."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sqlite3
import subprocess
import tempfile
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.features import CLAP_DIMENSIONS, CLAP_SAMPLE_RATE, ClapEmbedder
from core.model_assets import configure_model_environment
from core.platform_env import ENV


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "stage2" / "training"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "models" / "patchlab_clap_ft_v1.pt"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage2" / "clap-finetune-report.json"
DEFAULT_SEED = 20260802
TARGET_SAMPLES = CLAP_SAMPLE_RATE * 4


@dataclass(frozen=True, slots=True)
class ClipRecord:
    base_index: int
    preset_id: int
    midi_note: int
    perturb_seed: int
    path: Path


@dataclass(frozen=True, slots=True)
class AttemptConfiguration:
    name: str
    unfreeze_stages: int
    learning_rate: float
    augmentation_strength: float


ATTEMPTS = (
    AttemptConfiguration("last-stage", 1, 1.0e-5, 1.0),
    AttemptConfiguration("projection-only", 0, 3.0e-5, 0.75),
    AttemptConfiguration("last-stage-low-lr", 1, 3.0e-6, 0.65),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_preset_ids(
    preset_ids: Sequence[int], *, seed: int, validation_fraction: float
) -> tuple[list[int], list[int]]:
    values = np.asarray(sorted(set(map(int, preset_ids))), dtype=np.int64)
    if len(values) < 2:
        raise ValueError("At least two presets are required for a held-out split")
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    validation_count = max(1, int(round(len(values) * validation_fraction)))
    return (
        sorted(map(int, values[validation_count:])),
        sorted(map(int, values[:validation_count])),
    )


def symmetric_info_nce(
    first: torch.Tensor, second: torch.Tensor, temperature: torch.Tensor
) -> torch.Tensor:
    first = nn.functional.normalize(first, dim=-1)
    second = nn.functional.normalize(second, dim=-1)
    logits = first @ second.T / temperature.clamp(0.01, 0.20)
    labels = torch.arange(len(first), device=first.device)
    return 0.5 * (
        nn.functional.cross_entropy(logits, labels)
        + nn.functional.cross_entropy(logits.T, labels)
    )


def _records(corpus: Path) -> list[ClipRecord]:
    database = corpus / "manifest.sqlite"
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(sqlite3.connect(database)) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        expected = int(metadata["base_clips"])
        rows = connection.execute(
            "SELECT base_index,preset_id,midi_note,perturb_seed,raw_audio_path "
            "FROM base_clips ORDER BY base_index"
        ).fetchall()
    if len(rows) != expected:
        raise RuntimeError(
            f"Raw corpus is incomplete: {len(rows)} of {expected} base clips"
        )
    result = [
        ClipRecord(int(index), int(preset), int(note), int(perturb_seed), Path(path).resolve())
        for index, preset, note, perturb_seed, path in rows
    ]
    missing = [str(row.path) for row in result if not row.path.is_file()]
    if missing:
        raise RuntimeError(f"Raw corpus has {len(missing)} missing files; first={missing[0]}")
    patch_notes: dict[tuple[int, int], set[int]] = {}
    patch_counts: dict[tuple[int, int], int] = {}
    for row in result:
        key = (row.preset_id, row.perturb_seed)
        patch_notes.setdefault(key, set()).add(row.midi_note)
        patch_counts[key] = patch_counts.get(key, 0) + 1
    invalid = [
        key
        for key, count in patch_counts.items()
        if count != 3 or len(patch_notes[key]) != 3
    ]
    if invalid:
        raise RuntimeError(
            "Contrastive corpus must contain three distinct notes per exact patch; "
            f"invalid groups={len(invalid)}, first={invalid[0]}"
        )
    return result


def _load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if sample_rate != CLAP_SAMPLE_RATE:
        mono = librosa.resample(
            mono,
            orig_sr=int(sample_rate),
            target_sr=CLAP_SAMPLE_RATE,
            res_type="soxr_hq",
        ).astype(np.float32)
    values = mono[:TARGET_SAMPLES]
    if len(values) < TARGET_SAMPLES:
        values = np.pad(values, (0, TARGET_SAMPLES - len(values)))
    return np.ascontiguousarray(values, dtype=np.float32)


def _gate(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bpm = float(rng.integers(140, 175))
    division = int(rng.choice(np.asarray((4, 8, 16))))
    period = max(2, int(round(CLAP_SAMPLE_RATE * 60.0 / bpm * 4.0 / division)))
    phase = np.arange(len(audio), dtype=np.int64) % period
    envelope = (phase < int(round(period * 0.52))).astype(np.float32)
    ramp = max(1, int(round(CLAP_SAMPLE_RATE * 0.004)))
    envelope = np.convolve(
        envelope, np.ones(ramp, dtype=np.float32) / ramp, mode="same"
    ).astype(np.float32)
    return np.ascontiguousarray(audio * envelope, dtype=np.float32)


def _pump(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bpm = float(rng.integers(140, 175))
    period = CLAP_SAMPLE_RATE * 60.0 / bpm
    phase = (np.arange(len(audio)) % period) / period
    envelope = 0.25 + 0.75 * np.minimum(phase / 0.35, 1.0)
    return np.ascontiguousarray(audio * envelope.astype(np.float32), dtype=np.float32)


def _tail(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    delay = int(round(CLAP_SAMPLE_RATE * float(rng.uniform(0.07, 0.24))))
    wet = audio.copy()
    for repeat, gain in ((1, 0.28), (2, 0.14), (3, 0.07)):
        offset = delay * repeat
        if offset < len(wet):
            wet[offset:] += audio[:-offset] * gain
    return np.ascontiguousarray(wet, dtype=np.float32)


def _eq_tilt(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    amount = float(rng.uniform(-0.35, 0.35))
    delayed = np.pad(audio[:-1], (1, 0))
    return np.ascontiguousarray(audio + amount * (audio - delayed), dtype=np.float32)


def _width_proxy(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    delay = int(rng.integers(8, 48))
    shifted = np.pad(audio[:-delay], (delay, 0))
    return np.ascontiguousarray(0.75 * audio + 0.25 * shifted, dtype=np.float32)


def _codec_roundtrip(audio: np.ndarray, *, codec: str) -> np.ndarray:
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory(prefix="patchlab-stage2-ft-codec-") as directory:
        root = Path(directory)
        suffix = ".mp3" if codec == "mp3" else ".ogg"
        source, encoded, decoded = root / "in.wav", root / f"x{suffix}", root / "out.wav"
        sf.write(source, audio, CLAP_SAMPLE_RATE, subtype="PCM_16", format="WAV")
        executable = imageio_ffmpeg.get_ffmpeg_exe()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        encode = (
            [executable, "-v", "error", "-y", "-i", str(source), "-b:a", "96k", str(encoded)]
            if codec == "mp3"
            else [executable, "-v", "error", "-y", "-i", str(source), "-c:a", "libvorbis", "-q:a", "4", str(encoded)]
        )
        for command in (
            encode,
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
                command,
                check=False,
                capture_output=True,
                text=True,
                creationflags=flags,
            )
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip()[-1000:])
        result, _sample_rate = sf.read(decoded, dtype="float32", always_2d=False)
    return np.ascontiguousarray(result[:TARGET_SAMPLES], dtype=np.float32)


def augment_waveform(
    audio: np.ndarray,
    *,
    seed: int,
    strength: float = 1.0,
    allow_codec: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(audio, dtype=np.float32).copy()
    recipes = ["gate", "pump", "pitch", "gain", "tail", "eq", "noise", "width", "offset"]
    recipe = str(rng.choice(recipes))
    if recipe == "gate":
        values = _gate(values, rng)
    elif recipe == "pump":
        values = _pump(values, rng)
    elif recipe == "pitch":
        semitones = int(rng.choice(np.asarray((-12, -7, -2, 2, 7, 12))))
        values = librosa.effects.pitch_shift(
            values,
            sr=CLAP_SAMPLE_RATE,
            n_steps=float(semitones) * strength,
            res_type="soxr_hq",
        ).astype(np.float32)
    elif recipe == "gain":
        gain_db = float(rng.uniform(-6.0, 6.0)) * strength
        values *= 10.0 ** (gain_db / 20.0)
    elif recipe == "tail":
        values = _tail(values, rng)
    elif recipe == "eq":
        values = _eq_tilt(values, rng)
    elif recipe == "noise":
        rms = max(float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))), 1e-6)
        values += rng.normal(0.0, rms * 0.015 * strength, len(values)).astype(np.float32)
    elif recipe == "width":
        values = _width_proxy(values, rng)
    else:
        offset = int(rng.integers(0, max(1, int(CLAP_SAMPLE_RATE * 0.15 * strength))))
        values = np.pad(values[offset:], (0, offset))
    if allow_codec and float(rng.random()) < 0.05 * strength:
        values = _codec_roundtrip(values, codec=str(rng.choice(("mp3", "ogg"))))
    return np.ascontiguousarray(values[:TARGET_SAMPLES], dtype=np.float32)


def _trainable_parameters(
    clap: ClapEmbedder, *, unfreeze_stages: int
) -> tuple[Any, list[nn.Parameter], dict[str, int]]:
    model = clap.model.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    audio_branch = model.audio_branch
    selected_layers = list(audio_branch.layers[-unfreeze_stages:]) if unfreeze_stages else []
    modules: list[nn.Module] = [model.audio_projection]
    modules.extend(selected_layers)
    if unfreeze_stages:
        modules.extend((audio_branch.norm, audio_branch.tscam_conv))
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    counts = {
        "trainable": sum(parameter.numel() for parameter in parameters),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, parameters, counts


def _differentiable_embeddings(model: Any, waveforms: Sequence[np.ndarray]) -> torch.Tensor:
    from laion_clap.training.data import get_audio_features

    features = []
    for waveform in waveforms:
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        features.append(
            get_audio_features(
                {},
                tensor,
                480_000,
                data_truncating="rand_trunc",
                data_filling="repeatpad",
                audio_cfg=model.audio_cfg,
                require_grad=False,
            )
        )
    return model.get_audio_embedding(features)


def _embed_batches(clap: ClapEmbedder, audio: Sequence[np.ndarray], batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(audio), batch_size):
        rows.append(clap.embed(audio[start : start + batch_size]))
    return np.concatenate(rows, axis=0)


def _retrieval_rate(query: np.ndarray, gallery: np.ndarray) -> float:
    scores = np.asarray(query, dtype=np.float32) @ np.asarray(gallery, dtype=np.float32).T
    return float(np.mean(np.argmax(scores, axis=1) == np.arange(len(query))))


def _validation_audio(
    records_by_patch: dict[tuple[int, int], list[ClipRecord]],
    validation_ids: Sequence[int],
    *,
    seed: int,
    limit: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    rng = random.Random(seed)
    ids = list(validation_ids)
    rng.shuffle(ids)
    ids = ids[:limit]
    gallery, clean_queries, augmented_queries = [], [], []
    for position, preset_id in enumerate(ids):
        patch_keys = sorted(key for key in records_by_patch if key[0] == preset_id)
        records = records_by_patch[patch_keys[0]]
        first = records[0]
        different_notes = [item for item in records if item.midi_note != first.midi_note]
        second = different_notes[0] if different_notes else (records[1] if len(records) > 1 else first)
        gallery_audio = _load_audio(first.path)
        query_audio = _load_audio(second.path)
        gallery.append(gallery_audio)
        clean_queries.append(query_audio)
        augmented_queries.append(
            augment_waveform(
                query_audio,
                seed=seed + position * 104729,
                strength=1.0,
                allow_codec=True,
            )
        )
    return gallery, clean_queries, augmented_queries


def _evaluate(
    clap: ClapEmbedder,
    validation_audio: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
    *,
    batch_size: int,
) -> dict[str, float]:
    gallery, clean_queries, augmented_queries = validation_audio
    gallery_embeddings = _embed_batches(clap, gallery, batch_size)
    clean_embeddings = _embed_batches(clap, clean_queries, batch_size)
    augmented_embeddings = _embed_batches(clap, augmented_queries, batch_size)
    return {
        "clean_retrieval_at_1": _retrieval_rate(clean_embeddings, gallery_embeddings),
        "augmented_retrieval_at_1": _retrieval_rate(
            augmented_embeddings, gallery_embeddings
        ),
        "validation_presets": len(gallery),
    }


def _save_checkpoint(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state, "patchlab_metadata": metadata}, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--validation-presets", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.10 <= args.validation_fraction < 0.5:
        raise ValueError("Validation fraction must be at least 10% and below 50%")
    if args.batch_size < 2 or args.steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("Training counts must be positive and batch size at least two")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assets = configure_model_environment()
    records = _records(args.corpus.expanduser().resolve())
    records_by_preset: dict[int, list[ClipRecord]] = {}
    records_by_patch: dict[tuple[int, int], list[ClipRecord]] = {}
    for record in records:
        records_by_preset.setdefault(record.preset_id, []).append(record)
        records_by_patch.setdefault((record.preset_id, record.perturb_seed), []).append(record)
    patch_keys_by_preset: dict[int, list[tuple[int, int]]] = {}
    for key in records_by_patch:
        patch_keys_by_preset.setdefault(key[0], []).append(key)
    for keys in patch_keys_by_preset.values():
        keys.sort()
    train_ids, validation_ids = split_preset_ids(
        list(records_by_preset),
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    validation = _validation_audio(
        records_by_patch,
        validation_ids,
        seed=args.seed,
        limit=args.validation_presets,
    )
    report: dict[str, Any] = {
        "seed": args.seed,
        "corpus": str(args.corpus.expanduser().resolve()),
        "base_checkpoint": str(assets.checkpoint),
        "base_checkpoint_sha256": _sha256(assets.checkpoint),
        "train_preset_count": len(train_ids),
        "validation_preset_count": len(validation_ids),
        "validation_fraction": len(validation_ids) / len(records_by_preset),
        "attempts": [],
    }
    pinned = ClapEmbedder(ENV, checkpoint=assets.checkpoint)
    pinned_metrics = _evaluate(pinned, validation, batch_size=args.batch_size)
    report["pinned"] = pinned_metrics
    del pinned
    gc.collect()
    torch.cuda.empty_cache()
    adopted = False
    rng = np.random.default_rng(args.seed)
    for attempt_index, configuration in enumerate(ATTEMPTS[: args.attempts], start=1):
        started = time.monotonic()
        clap = ClapEmbedder(ENV, checkpoint=assets.checkpoint)
        model, parameters, counts = _trainable_parameters(
            clap, unfreeze_stages=configuration.unfreeze_stages
        )
        model.train()
        temperature = nn.Parameter(torch.tensor(0.07, device=ENV.compute_backend))
        optimizer = torch.optim.AdamW(
            [
                {"params": parameters, "lr": configuration.learning_rate},
                {"params": [temperature], "lr": configuration.learning_rate * 5.0},
            ],
            weight_decay=0.01,
        )
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step in range(1, args.steps + 1):
            selected_ids = rng.choice(
                np.asarray(train_ids), size=args.batch_size, replace=False
            )
            first_audio, second_audio = [], []
            for batch_position, preset_id_value in enumerate(selected_ids):
                preset_id = int(preset_id_value)
                patch_keys = patch_keys_by_preset[preset_id]
                choices = records_by_patch[patch_keys[int(rng.integers(len(patch_keys)))]]
                first_record = choices[int(rng.integers(len(choices)))]
                different_notes = [
                    item for item in choices if item.midi_note != first_record.midi_note
                ]
                second_pool = different_notes or choices
                second_record = second_pool[int(rng.integers(len(second_pool)))]
                first = _load_audio(first_record.path)
                second = _load_audio(second_record.path)
                # Keep a clean anchor in half the pairs so the learned
                # invariances cannot drift the original embedding world away.
                first_audio.append(
                    first
                    if float(rng.random()) < 0.5
                    else augment_waveform(
                        first,
                        seed=args.seed + step * 1_000_003 + batch_position,
                        strength=configuration.augmentation_strength * 0.5,
                        allow_codec=True,
                    )
                )
                second_audio.append(
                    augment_waveform(
                        second,
                        seed=args.seed + step * 2_000_003 + batch_position,
                        strength=configuration.augmentation_strength,
                        allow_codec=True,
                    )
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                combined = _differentiable_embeddings(
                    model, [*first_audio, *second_audio]
                )
                first_embeddings = combined[: args.batch_size]
                second_embeddings = combined[args.batch_size :]
                loss = symmetric_info_nce(
                    first_embeddings, second_embeddings, temperature
                ) / args.gradient_accumulation
            loss.backward()
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)
            if step % args.gradient_accumulation == 0 or step == args.steps:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 25 == 0 or step == args.steps:
                print(
                    "STAGE2_CLAP_TRAIN_PROGRESS="
                    + json.dumps(
                        {
                            "attempt": attempt_index,
                            "name": configuration.name,
                            "step": step,
                            "steps": args.steps,
                            "mean_recent_loss": float(np.mean(losses[-25:])),
                            "temperature": float(temperature.detach().cpu()),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        model.eval()
        metrics = _evaluate(clap, validation, batch_size=args.batch_size)
        passed = (
            metrics["augmented_retrieval_at_1"]
            > pinned_metrics["augmented_retrieval_at_1"]
            and metrics["clean_retrieval_at_1"]
            >= pinned_metrics["clean_retrieval_at_1"] - 0.01
        )
        attempt_report = {
            "configuration": asdict(configuration),
            "trainable_parameters": counts,
            "steps": args.steps,
            "mean_final_100_loss": float(np.mean(losses[-100:])),
            "elapsed_s": time.monotonic() - started,
            "metrics": metrics,
            "gate_pass": passed,
        }
        report["attempts"].append(attempt_report)
        if passed:
            _save_checkpoint(
                args.output.expanduser().resolve(),
                model,
                {
                    "format": "patchlab_clap_ft_v1",
                    "attempt": attempt_report,
                    "pinned": pinned_metrics,
                    "base_checkpoint_sha256": report["base_checkpoint_sha256"],
                },
            )
            adopted = True
            report["adopted_attempt"] = attempt_index
            report["checkpoint"] = str(args.output.expanduser().resolve())
            report["checkpoint_sha256"] = _sha256(args.output.expanduser().resolve())
            report["checkpoint_bytes"] = args.output.expanduser().resolve().stat().st_size
            del clap, model
            gc.collect()
            torch.cuda.empty_cache()
            break
        del clap, model
        gc.collect()
        torch.cuda.empty_cache()
    report["adopted"] = adopted
    report["decision"] = (
        "fine-tuned encoder passed held-out invariance and clean gates"
        if adopted
        else "keep pinned encoder; all honest fine-tuning attempts failed the gate"
    )
    args.report.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.expanduser().resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("STAGE2_CLAP_FINETUNE_SUMMARY=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
