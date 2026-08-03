"""Nearest-neighbor-conditioned delta parameter model."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from core.dataset import FEATURE_DIR, TrainingBundle, load_training_bundle
from core.platform_env import PlatformEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "models" / "delta_param_model.pt"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "models" / "delta_training_report.json"


class DeltaInferenceMLP(nn.Module):
    def __init__(self, serum1_dimension: int, serum2_dimension: int) -> None:
        super().__init__()
        self.parameter_encoders = nn.ModuleDict(
            {
                "serum1": nn.Sequential(
                    nn.Linear(serum1_dimension, 512), nn.SiLU(), nn.LayerNorm(512)
                ),
                "serum2": nn.Sequential(
                    nn.Linear(serum2_dimension, 512), nn.SiLU(), nn.LayerNorm(512)
                ),
            }
        )
        self.trunk = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.SiLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.SiLU(),
            nn.LayerNorm(768),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleDict(
            {
                "serum1": nn.Sequential(nn.Linear(768, serum1_dimension), nn.Tanh()),
                "serum2": nn.Sequential(nn.Linear(768, serum2_dimension), nn.Tanh()),
            }
        )

    def forward_synth(
        self,
        target_embedding: torch.Tensor,
        neighbor_embedding: torch.Tensor,
        neighbor_parameters: torch.Tensor,
        synth: str,
    ) -> torch.Tensor:
        encoded = self.parameter_encoders[synth](neighbor_parameters)
        hidden = self.trunk(torch.cat((target_embedding, neighbor_embedding, encoded), dim=1))
        return self.heads[synth](hidden)


@dataclass(slots=True)
class DeltaData:
    bundle: TrainingBundle
    preset_embeddings: np.ndarray
    preset_embedding_rows: dict[int, int]
    neighbor_by_preset: dict[int, int]
    real_train_indices: np.ndarray
    validation_indices: np.ndarray
    perturb_embeddings: dict[int, np.ndarray]
    perturb_targets: dict[int, np.ndarray]
    perturb_base_ids: dict[int, np.ndarray]
    perturb_indices: dict[int, np.ndarray]
    continuous_masks: dict[int, np.ndarray]


def load_delta_data() -> DeltaData:
    bundle = load_training_bundle(seed=1337)
    preset_manifest = np.load(FEATURE_DIR / "preset_manifest.npz")
    preset_ids = preset_manifest["preset_ids"].astype(np.int64)
    neighbor_manifest = np.load(FEATURE_DIR / "delta_neighbors.npz")
    neighbor_by_preset = {
        int(preset_id): int(neighbor_id)
        for preset_id, neighbor_id in zip(
            neighbor_manifest["preset_ids"], neighbor_manifest["neighbor_preset_ids"], strict=True
        )
    }
    perturb_embeddings: dict[int, np.ndarray] = {}
    perturb_targets: dict[int, np.ndarray] = {}
    perturb_base_ids: dict[int, np.ndarray] = {}
    perturb_indices: dict[int, np.ndarray] = {}
    for code, tag, synth in ((1, "s1", "serum1"), (2, "s2", "serum2")):
        complete = np.load(FEATURE_DIR / f"perturb_{tag}_complete.npy", mmap_mode="r")
        if not np.all(complete):
            raise RuntimeError(f"Perturbation dataset incomplete for {synth}")
        bases = np.load(FEATURE_DIR / f"perturb_{tag}_base_ids.npy", mmap_mode="r")
        allowed = np.asarray(bundle.train_preset_ids[synth], dtype=np.int64)
        perturb_indices[code] = np.flatnonzero(np.isin(bases, allowed)).astype(np.int64)
        perturb_embeddings[code] = np.load(
            FEATURE_DIR / f"perturb_{tag}_embeddings.npy", mmap_mode="r"
        )
        perturb_targets[code] = np.load(
            FEATURE_DIR / f"perturb_{tag}_targets.npy", mmap_mode="r"
        )
        perturb_base_ids[code] = bases
    continuous_masks = {
        1: np.asarray(
            [not bool(field.get("stepped")) for field in bundle.targets[1].mapping],
            dtype=np.bool_,
        ),
        2: np.asarray(
            [field.get("encoding") != "one_hot" for field in bundle.targets[2].mapping],
            dtype=np.bool_,
        ),
    }
    return DeltaData(
        bundle=bundle,
        preset_embeddings=np.load(FEATURE_DIR / "preset_embeddings.npy", mmap_mode="r"),
        preset_embedding_rows={int(value): index for index, value in enumerate(preset_ids)},
        neighbor_by_preset=neighbor_by_preset,
        real_train_indices=bundle.train_indices,
        validation_indices=bundle.validation_indices,
        perturb_embeddings=perturb_embeddings,
        perturb_targets=perturb_targets,
        perturb_base_ids=perturb_base_ids,
        perturb_indices=perturb_indices,
        continuous_masks=continuous_masks,
    )


class DeltaDataset(Dataset[tuple[np.ndarray, ...]]):
    def __init__(self, data: DeltaData, *, validation: bool = False) -> None:
        self.data = data
        if validation:
            self.rows = [("real", int(index)) for index in data.validation_indices]
        else:
            self.rows = [("real", int(index)) for index in data.real_train_indices]
            for code in (1, 2):
                self.rows.extend((f"perturb{code}", int(index)) for index in data.perturb_indices[code])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> tuple[np.ndarray, ...]:
        kind, index = self.rows[item]
        bundle = self.data.bundle
        if kind == "real":
            preset_id = int(bundle.note_preset_ids[index])
            code = int(bundle.note_synths[index])
            target_embedding = np.asarray(bundle.embeddings[index], dtype=np.float32)
            neighbor_id = self.data.neighbor_by_preset[preset_id]
            store = bundle.targets[code]
            target_row = store.preset_row[preset_id]
            target = np.asarray(store.vectors[target_row], dtype=np.float32)
            mask = np.asarray(store.masks[target_row], dtype=np.bool_)
        else:
            code = int(kind[-1])
            neighbor_id = int(self.data.perturb_base_ids[code][index])
            target_embedding = np.asarray(self.data.perturb_embeddings[code][index], dtype=np.float32)
            store = bundle.targets[code]
            target = np.asarray(self.data.perturb_targets[code][index], dtype=np.float32)
            mask = np.asarray(store.masks[store.preset_row[neighbor_id]], dtype=np.bool_)
        neighbor = np.asarray(store.vectors[store.preset_row[neighbor_id]], dtype=np.float32)
        neighbor_embedding = np.asarray(
            self.data.preset_embeddings[self.data.preset_embedding_rows[neighbor_id]], dtype=np.float32
        )
        delta = target - neighbor
        training_mask = mask & self.data.continuous_masks[code]
        return (
            target_embedding,
            neighbor_embedding,
            neighbor,
            delta,
            training_mask,
            mask,
            np.asarray(code),
            target,
        )


def delta_collate(rows: list[tuple[np.ndarray, ...]]) -> tuple[torch.Tensor, ...]:
    maximum = max(len(row[2]) for row in rows)
    count = len(rows)
    neighbor = np.zeros((count, maximum), dtype=np.float32)
    delta = np.zeros_like(neighbor)
    training_masks = np.zeros_like(neighbor, dtype=np.bool_)
    evaluation_masks = np.zeros_like(neighbor, dtype=np.bool_)
    target = np.zeros_like(neighbor)
    for index, row in enumerate(rows):
        width = len(row[2])
        neighbor[index, :width] = row[2]
        delta[index, :width] = row[3]
        training_masks[index, :width] = row[4]
        evaluation_masks[index, :width] = row[5]
        target[index, :width] = row[7]
    return (
        torch.from_numpy(np.stack([row[0] for row in rows])),
        torch.from_numpy(np.stack([row[1] for row in rows])),
        torch.from_numpy(neighbor),
        torch.from_numpy(delta),
        torch.from_numpy(training_masks),
        torch.from_numpy(evaluation_masks),
        torch.as_tensor([int(row[6]) for row in rows], dtype=torch.uint8),
        torch.from_numpy(target),
    )


def _loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_delta_model(
    environment: PlatformEnv,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    patience: int = 15,
    seed: int = 1337,
) -> dict[str, Any]:
    _seed(seed)
    data = load_delta_data()
    train_set, validation_set = DeltaDataset(data), DeltaDataset(data, validation=True)
    source_groups = np.asarray(
        [
            int(data.bundle.note_synths[index])
            if kind == "real"
            else 2 + int(kind[-1])
            for kind, index in train_set.rows
        ],
        dtype=np.uint8,
    )
    group_counts = {code: int(np.sum(source_groups == code)) for code in (1, 2, 3, 4)}
    weights = torch.as_tensor(
        [1.0 / group_counts[int(group)] for group in source_groups], dtype=torch.double
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=4 * max(group_counts.values()),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, collate_fn=delta_collate, num_workers=0
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=False, collate_fn=delta_collate, num_workers=0
    )
    dimensions = {name: data.bundle.targets[code].dimension for code, name in ((1, "serum1"), (2, "serum2"))}
    device = torch.device(environment.compute_backend)
    model = DeltaInferenceMLP(dimensions["serum1"], dimensions["serum2"]).to(device)
    for head in model.heads.values():
        nn.init.zeros_(head[0].weight)
        nn.init.zeros_(head[0].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    absolute_report = json.loads(
        (PROJECT_ROOT / "data" / "models" / "milestone3_training_report.json").read_text()
    )
    started = time.monotonic()
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] = {}
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for (
            target_emb,
            neighbor_emb,
            neighbor,
            delta,
            training_masks,
            _evaluation_masks,
            synths,
            _target,
        ) in train_loader:
            target_emb, neighbor_emb = target_emb.to(device), neighbor_emb.to(device)
            neighbor, delta, training_masks, synths = (
                neighbor.to(device),
                delta.to(device),
                training_masks.to(device),
                synths.to(device),
            )
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for code, synth in ((1, "serum1"), (2, "serum2")):
                selected = synths == code
                if not torch.any(selected):
                    continue
                width = dimensions[synth]
                prediction = model.forward_synth(
                    target_emb[selected], neighbor_emb[selected], neighbor[selected, :width], synth
                )
                losses.append(
                    _loss(
                        prediction,
                        delta[selected, :width],
                        training_masks[selected, :width],
                    )
                )
            torch.stack(losses).mean().backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        totals = {name: [0.0, 0.0, 0.0] for name in dimensions}
        with torch.inference_mode():
            for (
                target_emb,
                neighbor_emb,
                neighbor,
                _delta,
                _training_masks,
                evaluation_masks,
                synths,
                target,
            ) in validation_loader:
                target_emb, neighbor_emb = target_emb.to(device), neighbor_emb.to(device)
                neighbor, evaluation_masks, synths, target = (
                    neighbor.to(device),
                    evaluation_masks.to(device),
                    synths.to(device),
                    target.to(device),
                )
                for code, synth in ((1, "serum1"), (2, "serum2")):
                    selected = synths == code
                    if not torch.any(selected):
                        continue
                    width = dimensions[synth]
                    predicted_delta = model.forward_synth(
                        target_emb[selected], neighbor_emb[selected], neighbor[selected, :width], synth
                    )
                    continuous = torch.from_numpy(data.continuous_masks[code]).to(device)
                    predicted_delta = predicted_delta * continuous.unsqueeze(0)
                    absolute = torch.clamp(neighbor[selected, :width] + predicted_delta, 0.0, 1.0)
                    mask = evaluation_masks[selected, :width]
                    error = (absolute - target[selected, :width]).abs() * mask
                    neighbor_error = (neighbor[selected, :width] - target[selected, :width]).abs() * mask
                    totals[synth][0] += float(error.sum().cpu())
                    totals[synth][1] += float(neighbor_error.sum().cpu())
                    totals[synth][2] += float(mask.sum().cpu())
        metrics = {}
        for synth in dimensions:
            mae = totals[synth][0] / totals[synth][2]
            neighbor_mae = totals[synth][1] / totals[synth][2]
            absolute_mae = float(absolute_report["by_synth"][synth]["mae"])
            metrics[synth] = {
                "mae": mae,
                "nearest_neighbor_mae": neighbor_mae,
                "milestone3_absolute_mae": absolute_mae,
                "improvement_vs_milestone3": 1.0 - mae / absolute_mae,
            }
        score = float(np.mean([metrics[synth]["mae"] for synth in dimensions]))
        print(
            "DELTA_TRAIN_PROGRESS="
            + json.dumps({"epoch": epoch, "score": score, "by_synth": metrics}, sort_keys=True),
            flush=True,
        )
        if score < best_score - 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Delta training produced no checkpoint")
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state": best_state,
        "model_config": dimensions,
        "serum1_mapping": data.bundle.targets[1].mapping,
        "serum2_schema": json.loads(
            (PROJECT_ROOT / "data" / "models" / "serum2_target_schema.json").read_text()
        ),
        "split": {
            "seed": seed,
            "train_preset_ids": data.bundle.train_preset_ids,
            "validation_preset_ids": data.bundle.validation_preset_ids,
        },
        "neighbor_manifest": str((FEATURE_DIR / "delta_neighbors.npz").resolve()),
    }
    torch.save(checkpoint, DEFAULT_CHECKPOINT)
    report = {
        "checkpoint": str(DEFAULT_CHECKPOINT.resolve()),
        "epochs_completed": epoch,
        "elapsed_s": time.monotonic() - started,
        "training_rows_by_source": {
            "real_serum1": group_counts[1],
            "real_serum2": group_counts[2],
            "perturb_serum1": group_counts[3],
            "perturb_serum2": group_counts[4],
        },
        "validation": best_metrics,
        "gate_pass": all(np.isfinite(row["mae"]) for row in best_metrics.values()),
    }
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True))
    print("DELTA_TRAIN_SUMMARY=" + json.dumps(report, sort_keys=True))
    return report


def load_delta_model(
    path: Path | None = None, *, device: str = "cpu"
) -> tuple[DeltaInferenceMLP, dict[str, Any]]:
    path = (
        Path(os.environ.get("PATCHLAB_DELTA_MODEL", str(DEFAULT_CHECKPOINT)))
        if path is None
        else path
    ).expanduser().resolve()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = DeltaInferenceMLP(config["serum1"], config["serum2"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_delta(
    model: DeltaInferenceMLP,
    target_embedding: np.ndarray,
    neighbor_embedding: np.ndarray,
    neighbor_parameters: np.ndarray,
    synth: str,
    continuous_mask: np.ndarray | None = None,
) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.inference_mode():
        delta = model.forward_synth(
            torch.from_numpy(np.array(target_embedding, dtype=np.float32, copy=True))
            .unsqueeze(0)
            .to(device),
            torch.from_numpy(np.array(neighbor_embedding, dtype=np.float32, copy=True))
            .unsqueeze(0)
            .to(device),
            torch.from_numpy(np.array(neighbor_parameters, dtype=np.float32, copy=True))
            .unsqueeze(0)
            .to(device),
            synth,
        )[0].cpu().numpy()
    if continuous_mask is not None:
        delta *= np.asarray(continuous_mask, dtype=np.float32)
    return np.clip(np.asarray(neighbor_parameters) + delta, 0.0, 1.0).astype(np.float32)
