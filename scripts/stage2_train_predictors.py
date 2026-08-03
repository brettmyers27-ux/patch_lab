#!/usr/bin/env python3
"""Train versioned absolute and delta predictors from Stage 2 NPZ shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import TargetStore, _serum1_targets, _serum2_targets
from core.delta_model import DeltaInferenceMLP, load_delta_model
from core.features import HANDCRAFTED_NAMES
from core.model_assets import configure_model_environment
from core.platform_env import ENV
from core.synthesis_assets import resolve_synthesis_assets
from core.train import ParameterInferenceMLP, load_parameter_model


DEFAULT_CORPUS = PROJECT_ROOT / "data" / "stage2" / "training"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "data" / "stage2" / "artifacts-v2"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage2" / "predictor-training-report.json"
DEFAULT_SEED = 1337


@dataclass(frozen=True, slots=True)
class Split:
    train: dict[int, set[int]]
    validation: dict[int, set[int]]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_counts(corpus: Path) -> tuple[int, int, int]:
    database = corpus / "manifest.sqlite"
    if not database.is_file():
        raise FileNotFoundError(database)
    with closing(sqlite3.connect(database)) as connection:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        complete, pairs = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(pair_count),0) FROM base_clips"
        ).fetchone()
    return int(metadata["base_clips"]), int(complete), int(pairs)


def _shards(corpus: Path, maximum: int | None = None) -> list[Path]:
    paths = sorted((corpus / "shards").glob("shard-*.npz"))
    return paths[:maximum] if maximum is not None else paths


def _preset_split(paths: Sequence[Path], *, seed: int, fraction: float) -> Split:
    by_synth: dict[int, set[int]] = {1: set(), 2: set()}
    for path in paths:
        with np.load(path) as shard:
            for code in (1, 2):
                selected = np.asarray(shard["synth_codes"]) == code
                by_synth[code].update(map(int, np.asarray(shard["preset_ids"])[selected]))
    rng = np.random.default_rng(seed)
    train: dict[int, set[int]] = {}
    validation: dict[int, set[int]] = {}
    for code in (1, 2):
        values = np.asarray(sorted(by_synth[code]), dtype=np.int64)
        if len(values) < 2:
            raise RuntimeError(f"Stage 2 corpus has fewer than two synth-{code} presets")
        rng.shuffle(values)
        count = max(1, int(round(len(values) * fraction)))
        validation[code] = set(map(int, values[:count]))
        train[code] = set(map(int, values[count:]))
    return Split(train, validation)


def _unpack_base(shard: Any) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(shard["parameter_vectors"], dtype=np.float32)
    lengths = np.asarray(shard["vector_lengths"], dtype=np.int64)
    packed = np.asarray(shard["parameter_masks_packed"], dtype=np.uint8)
    masks = np.unpackbits(packed, axis=1, bitorder="little")[:, : vectors.shape[1]].astype(bool)
    for row, length in enumerate(lengths):
        masks[row, int(length) :] = False
    return vectors, masks


def _pair_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as shard:
        base_rows = np.asarray(shard["pair_base_rows"], dtype=np.int64)
        vectors, masks = _unpack_base(shard)
        return {
            "embeddings": np.asarray(shard["embeddings"], dtype=np.float32),
            "handcrafted": np.asarray(shard["handcrafted"], dtype=np.float32),
            "synths": np.asarray(shard["synth_codes"], dtype=np.int64)[base_rows],
            "presets": np.asarray(shard["preset_ids"], dtype=np.int64)[base_rows],
            "base_rows": base_rows,
            "targets": vectors[base_rows],
            "masks": masks[base_rows],
        }


def _selected_rows(data: dict[str, np.ndarray], split: Split, validation: bool) -> np.ndarray:
    pools = split.validation if validation else split.train
    keep = np.zeros(len(data["synths"]), dtype=bool)
    for code in (1, 2):
        selected = data["synths"] == code
        keep[selected] = np.isin(data["presets"][selected], np.asarray(sorted(pools[code])))
    return np.flatnonzero(keep)


def _standardizer(paths: Sequence[Path], split: Split) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(len(HANDCRAFTED_NAMES), dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for path in paths:
        data = _pair_arrays(path)
        rows = _selected_rows(data, split, False)
        values = data["handcrafted"][rows].astype(np.float64)
        total += values.sum(axis=0)
        square += np.square(values).sum(axis=0)
        count += len(values)
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _baseline_means(paths: Sequence[Path], split: Split, dimensions: dict[int, int]) -> dict[int, np.ndarray]:
    sums = {code: np.zeros(dimensions[code], dtype=np.float64) for code in (1, 2)}
    counts = {code: np.zeros(dimensions[code], dtype=np.float64) for code in (1, 2)}
    for path in paths:
        data = _pair_arrays(path)
        rows = _selected_rows(data, split, False)
        # One target per base is enough; pair weighting would repeat each target 13 times.
        for row in rows[np.unique(data["base_rows"][rows], return_index=True)[1]]:
            code = int(data["synths"][row])
            width = dimensions[code]
            mask = data["masks"][row, :width]
            sums[code] += data["targets"][row, :width] * mask
            counts[code] += mask
    return {
        code: np.divide(sums[code], np.maximum(counts[code], 1)).astype(np.float32)
        for code in (1, 2)
    }


def _batches(
    paths: Sequence[Path],
    split: Split,
    *,
    validation: bool,
    batch_size: int,
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
    sample_limit: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    ordered = list(paths)
    if not validation:
        rng.shuffle(ordered)
    yielded = 0
    for path in ordered:
        data = _pair_arrays(path)
        rows = _selected_rows(data, split, validation)
        if not validation:
            rng.shuffle(rows)
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            if sample_limit is not None:
                selected = selected[: max(0, sample_limit - yielded)]
            if not len(selected):
                return
            features = np.concatenate(
                (
                    data["embeddings"][selected],
                    (data["handcrafted"][selected] - mean) / std,
                ),
                axis=1,
            ).astype(np.float32)
            yield {key: values[selected] for key, values in data.items()} | {"features": features}
            yielded += len(selected)
            if sample_limit is not None and yielded >= sample_limit:
                return


def _absolute_metrics(
    model: ParameterInferenceMLP,
    batches: Iterator[dict[str, np.ndarray]],
    device: torch.device,
    dimensions: dict[int, int],
    baselines: dict[int, np.ndarray],
) -> dict[str, dict[str, float]]:
    totals = {code: [0.0, 0.0, 0.0] for code in (1, 2)}
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            inputs = torch.from_numpy(batch["features"]).to(device)
            hidden = model.encode(inputs)
            synths = batch["synths"]
            for code, name in ((1, "serum1"), (2, "serum2")):
                rows = np.flatnonzero(synths == code)
                if not len(rows):
                    continue
                width = dimensions[code]
                prediction = model.heads[name](hidden[torch.as_tensor(rows, device=device)]).float().cpu().numpy()
                target = batch["targets"][rows, :width]
                mask = batch["masks"][rows, :width]
                totals[code][0] += float((np.abs(prediction - target) * mask).sum())
                totals[code][1] += float((np.abs(baselines[code][None, :] - target) * mask).sum())
                totals[code][2] += float(mask.sum())
    return {
        name: {
            "mae": totals[code][0] / totals[code][2],
            "mean_baseline_mae": totals[code][1] / totals[code][2],
            "observations": totals[code][2],
        }
        for code, name in ((1, "serum1"), (2, "serum2"))
    }


def _masked_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1)


def train_absolute(
    paths: Sequence[Path], split: Split, stores: dict[int, TargetStore], *,
    output: Path, epochs: int, patience: int, batch_size: int,
    samples_per_epoch: int | None, seed: int, checkpoint_name: str,
) -> dict[str, Any]:
    device = torch.device(ENV.compute_backend)
    dimensions = {code: stores[code].dimension for code in (1, 2)}
    mean, std = _standardizer(paths, split)
    baselines = _baseline_means(paths, split, dimensions)
    model = ParameterInferenceMLP(512 + len(HANDCRAFTED_NAMES), dimensions[1], dimensions[2]).to(device)
    with torch.no_grad():
        for code, name in ((1, "serum1"), (2, "serum2")):
            layer = model.heads[name][0]
            layer.weight.zero_()
            layer.bias.copy_(torch.logit(torch.from_numpy(baselines[code]).to(device).clamp(1e-6, 1 - 1e-6)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_score, best_state, best_metrics = math.inf, None, None
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in _batches(paths, split, validation=False, batch_size=batch_size, seed=seed + epoch, mean=mean, std=std, sample_limit=samples_per_epoch):
            inputs = torch.from_numpy(batch["features"]).to(device)
            targets = torch.from_numpy(batch["targets"]).to(device)
            masks = torch.from_numpy(batch["masks"]).to(device)
            synths = torch.from_numpy(batch["synths"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            per_synth = []
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model.encode(inputs)
                for code, name in ((1, "serum1"), (2, "serum2")):
                    selected = synths == code
                    if torch.any(selected):
                        width = dimensions[code]
                        per_synth.append(_masked_loss(model.heads[name](hidden[selected]), targets[selected, :width], masks[selected, :width]))
                loss = torch.stack(per_synth).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = _absolute_metrics(
            model,
            _batches(paths, split, validation=True, batch_size=batch_size, seed=seed, mean=mean, std=std),
            device, dimensions, baselines,
        )
        score = float(np.mean([metrics[name]["mae"] for name in ("serum1", "serum2")]))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_score": score, "by_synth": metrics}
        history.append(entry)
        print("STAGE2_PARAM_PROGRESS=" + json.dumps(entry, sort_keys=True), flush=True)
        if score < best_score - 1e-7:
            best_score = score
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("Absolute predictor produced no checkpoint")
    old_model, old_checkpoint = load_parameter_model(device=str(device))
    old_standardizer = old_checkpoint["feature_standardizer"]
    old_metrics = _absolute_metrics(
        old_model,
        _batches(
            paths, split, validation=True, batch_size=batch_size, seed=seed,
            mean=np.asarray(old_standardizer["mean"], dtype=np.float32),
            std=np.asarray(old_standardizer["std"], dtype=np.float32),
        ),
        device, dimensions, baselines,
    )
    checkpoint = {
        "model_state": best_state,
        "model_config": {"input_dimension": 512 + len(HANDCRAFTED_NAMES), "serum1_dimension": dimensions[1], "serum2_dimension": dimensions[2]},
        "feature_standardizer": {"mean": mean, "std": std, "names": list(HANDCRAFTED_NAMES)},
        "clap_checkpoint": checkpoint_name,
        "serum1_mapping": stores[1].mapping,
        "serum2_schema": json.loads(resolve_synthesis_assets().serum2_schema.read_text(encoding="utf-8")),
        "split": {"seed": seed, "train_preset_ids": {"serum1": sorted(split.train[1]), "serum2": sorted(split.train[2])}, "validation_preset_ids": {"serum1": sorted(split.validation[1]), "serum2": sorted(split.validation[2])}},
        "deep_training": True,
        "best_validation": best_metrics,
        "stage2_scaled_corpus": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "checkpoint": str(output.resolve()), "checkpoint_bytes": output.stat().st_size,
        "checkpoint_sha256": _sha256(output), "epochs_completed": len(history),
        "elapsed_s": time.monotonic() - started, "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "old_validation": old_metrics, "new_validation": best_metrics, "history": history,
    }


def _index_lookup(feature_dir: Path) -> tuple[np.ndarray, dict[int, int]]:
    manifest = np.load(feature_dir / "similarity_manifest.npz")
    ids = np.asarray(manifest["preset_ids"], dtype=np.int64)
    embeddings = np.asarray(np.load(feature_dir / "preset_index.npy"), dtype=np.float32)
    return embeddings, {int(value): row for row, value in enumerate(ids)}


def _continuous_masks(stores: dict[int, TargetStore]) -> dict[int, np.ndarray]:
    return {
        1: np.asarray([not bool(field.get("stepped")) for field in stores[1].mapping], dtype=bool),
        2: np.asarray([field.get("encoding") != "one_hot" for field in stores[2].mapping], dtype=bool),
    }


def _delta_metrics(
    model: DeltaInferenceMLP, batches: Iterator[dict[str, np.ndarray]], device: torch.device,
    stores: dict[int, TargetStore], preset_embeddings: np.ndarray, preset_rows: dict[int, int],
    continuous: dict[int, np.ndarray],
) -> dict[str, dict[str, float]]:
    totals = {code: [0.0, 0.0, 0.0] for code in (1, 2)}
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            for code, name in ((1, "serum1"), (2, "serum2")):
                rows = np.flatnonzero(batch["synths"] == code)
                if not len(rows):
                    continue
                width = stores[code].dimension
                ids = batch["presets"][rows]
                neighbor = np.stack([stores[code].vectors[stores[code].preset_row[int(value)]] for value in ids]).astype(np.float32)
                neighbor_emb = np.stack([preset_embeddings[preset_rows[int(value)]] for value in ids]).astype(np.float32)
                predicted = model.forward_synth(
                    torch.from_numpy(batch["embeddings"][rows]).to(device),
                    torch.from_numpy(neighbor_emb).to(device),
                    torch.from_numpy(neighbor).to(device), name,
                ).float().cpu().numpy()
                absolute = np.clip(neighbor + predicted * continuous[code][None, :], 0, 1)
                target = batch["targets"][rows, :width]
                mask = batch["masks"][rows, :width]
                totals[code][0] += float((np.abs(absolute - target) * mask).sum())
                totals[code][1] += float((np.abs(neighbor - target) * mask).sum())
                totals[code][2] += float(mask.sum())
    return {
        name: {"mae": totals[code][0] / totals[code][2], "neighbor_mae": totals[code][1] / totals[code][2], "observations": totals[code][2]}
        for code, name in ((1, "serum1"), (2, "serum2"))
    }


def train_delta(
    paths: Sequence[Path], split: Split, stores: dict[int, TargetStore], *, feature_dir: Path,
    output: Path, epochs: int, patience: int, batch_size: int, samples_per_epoch: int | None,
    seed: int,
) -> dict[str, Any]:
    device = torch.device(ENV.compute_backend)
    mean, std = _standardizer(paths, split)
    dimensions = {code: stores[code].dimension for code in (1, 2)}
    continuous = _continuous_masks(stores)
    preset_embeddings, preset_rows = _index_lookup(feature_dir)
    missing = sorted({int(value) for path in paths for value in _pair_arrays(path)["presets"] if int(value) not in preset_rows})
    if missing:
        raise RuntimeError(f"Preset index is missing {len(missing)} Stage 2 base presets; first={missing[0]}")
    model = DeltaInferenceMLP(dimensions[1], dimensions[2]).to(device)
    for head in model.heads.values():
        head[0].weight.data.zero_()
        head[0].bias.data.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_score, best_state, best_metrics = math.inf, None, None
    history: list[dict[str, Any]] = []
    stale = 0
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in _batches(paths, split, validation=False, batch_size=batch_size, seed=seed + epoch, mean=mean, std=std, sample_limit=samples_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            per_synth = []
            for code, name in ((1, "serum1"), (2, "serum2")):
                rows = np.flatnonzero(batch["synths"] == code)
                if not len(rows):
                    continue
                ids = batch["presets"][rows]
                neighbor = np.stack([stores[code].vectors[stores[code].preset_row[int(value)]] for value in ids]).astype(np.float32)
                neighbor_emb = np.stack([preset_embeddings[preset_rows[int(value)]] for value in ids]).astype(np.float32)
                width = dimensions[code]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prediction = model.forward_synth(
                        torch.from_numpy(batch["embeddings"][rows]).to(device),
                        torch.from_numpy(neighbor_emb).to(device),
                        torch.from_numpy(neighbor).to(device), name,
                    )
                    target_delta = torch.from_numpy(batch["targets"][rows, :width] - neighbor).to(device)
                    mask = torch.from_numpy(batch["masks"][rows, :width] & continuous[code][None, :]).to(device)
                    per_synth.append(_masked_loss(prediction, target_delta, mask))
            loss = torch.stack(per_synth).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = _delta_metrics(
            model, _batches(paths, split, validation=True, batch_size=batch_size, seed=seed, mean=mean, std=std),
            device, stores, preset_embeddings, preset_rows, continuous,
        )
        score = float(np.mean([metrics[name]["mae"] for name in ("serum1", "serum2")]))
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_score": score, "by_synth": metrics}
        history.append(entry)
        print("STAGE2_DELTA_PROGRESS=" + json.dumps(entry, sort_keys=True), flush=True)
        if score < best_score - 1e-7:
            best_score, best_metrics, stale = score, metrics, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("Delta predictor produced no checkpoint")
    old_model, _old_checkpoint = load_delta_model(device=str(device))
    old_metrics = _delta_metrics(
        old_model, _batches(paths, split, validation=True, batch_size=batch_size, seed=seed, mean=mean, std=std),
        device, stores, preset_embeddings, preset_rows, continuous,
    )
    checkpoint = {
        "model_state": best_state,
        "model_config": {"serum1": dimensions[1], "serum2": dimensions[2]},
        "serum1_mapping": stores[1].mapping,
        "serum2_schema": json.loads(resolve_synthesis_assets().serum2_schema.read_text(encoding="utf-8")),
        "split": {"seed": seed, "train_preset_ids": {"serum1": sorted(split.train[1]), "serum2": sorted(split.train[2])}, "validation_preset_ids": {"serum1": sorted(split.validation[1]), "serum2": sorted(split.validation[2])}},
        "neighbor_manifest": str((feature_dir / "similarity_manifest.npz").resolve()),
        "stage2_scaled_corpus": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "checkpoint": str(output.resolve()), "checkpoint_bytes": output.stat().st_size,
        "checkpoint_sha256": _sha256(output), "epochs_completed": len(history),
        "elapsed_s": time.monotonic() - started, "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "old_validation": old_metrics, "new_validation": best_metrics, "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-epoch", type=int, default=120_000)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-delta", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.10 <= args.validation_fraction < 0.5:
        raise ValueError("Validation fraction must be at least 10% and below 50%")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    configure_model_environment()
    assets = resolve_synthesis_assets()
    corpus = args.corpus.expanduser().resolve()
    expected, complete, pairs = _manifest_counts(corpus)
    if complete != expected and not args.allow_partial:
        raise RuntimeError(f"Stage 2 corpus incomplete: {complete} of {expected} bases")
    paths = _shards(corpus, args.max_shards)
    if not paths:
        raise RuntimeError(f"No Stage 2 shards under {corpus}")
    split = _preset_split(paths, seed=args.seed, fraction=args.validation_fraction)
    stores = {1: _serum1_targets(assets.library_db), 2: _serum2_targets(assets.serum2_targets, assets.serum2_schema)}
    artifacts = args.artifacts.expanduser().resolve()
    feature_dir = (args.feature_dir or assets.feature_dir).expanduser().resolve()
    started = time.monotonic()
    report: dict[str, Any] = {
        "corpus": str(corpus), "expected_base_clips": expected, "completed_base_clips": complete,
        "training_pairs": pairs, "shards_used": len(paths), "seed": args.seed,
        "split": {"train": {str(code): len(split.train[code]) for code in (1, 2)}, "validation": {str(code): len(split.validation[code]) for code in (1, 2)}},
    }
    report["parameter_model"] = train_absolute(
        paths, split, stores, output=artifacts / "param_model_stage2.pt", epochs=args.epochs,
        patience=args.patience, batch_size=args.batch_size, samples_per_epoch=args.samples_per_epoch,
        seed=args.seed, checkpoint_name=Path(os.environ["PATCHLAB_CLAP_CHECKPOINT"]).name,
    )
    if not args.skip_delta:
        report["delta_model"] = train_delta(
            paths, split, stores, feature_dir=feature_dir, output=artifacts / "delta_param_model_stage2.pt",
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch, seed=args.seed,
        )
    report["elapsed_s"] = time.monotonic() - started
    report["status"] = "complete"
    _atomic_json(args.report.expanduser().resolve(), report)
    print("STAGE2_PREDICTOR_SUMMARY=" + json.dumps({key: value for key, value in report.items() if key not in {"parameter_model", "delta_model"}}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
