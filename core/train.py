"""Two-head Serum parameter inference model and deterministic trainer."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.dataset import ParameterDataset, TrainingBundle, parameter_collate
from core.features import CLAP_CHECKPOINT_NAME, HANDCRAFTED_NAMES
from core.platform_env import PlatformEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "models" / "param_model.pt"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "models" / "milestone3_training_report.json"


class ParameterInferenceMLP(nn.Module):
    def __init__(self, input_dimension: int, serum1_dimension: int, serum2_dimension: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dimension, 1024),
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
                "serum1": nn.Sequential(nn.Linear(768, serum1_dimension), nn.Sigmoid()),
                "serum2": nn.Sequential(nn.Linear(768, serum2_dimension), nn.Sigmoid()),
            }
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.trunk(inputs)


def load_parameter_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    *,
    device: str = "cpu",
) -> tuple[ParameterInferenceMLP, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = ParameterInferenceMLP(
        int(config["input_dimension"]),
        int(config["serum1_dimension"]),
        int(config["serum2_dimension"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_parameters(
    model: ParameterInferenceMLP,
    checkpoint: dict[str, Any],
    embedding: np.ndarray,
    handcrafted: np.ndarray,
    synth: str,
) -> np.ndarray:
    standardizer = checkpoint["feature_standardizer"]
    standardized = (
        np.asarray(handcrafted, dtype=np.float32) - np.asarray(standardizer["mean"], dtype=np.float32)
    ) / np.asarray(standardizer["std"], dtype=np.float32)
    features = np.concatenate((np.asarray(embedding, dtype=np.float32), standardized))
    device = next(model.parameters()).device
    with torch.inference_mode():
        hidden = model.encode(torch.from_numpy(features).unsqueeze(0).to(device))
        output = model.heads[synth](hidden)[0].cpu().numpy()
    if synth == "serum1":
        for field in checkpoint["serum1_mapping"]:
            if not field.get("stepped") or not field.get("step_values"):
                continue
            index = int(field["index"])
            values = np.asarray(field["step_values"], dtype=np.float32)
            output[index] = values[np.argmin(np.abs(values - output[index]))]
    return output.astype(np.float32, copy=False)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _masked_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1)


def _batch_loss(
    model: ParameterInferenceMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    synths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hidden = model.encode(inputs)
    losses: dict[str, torch.Tensor] = {}
    for code, name in ((1, "serum1"), (2, "serum2")):
        selected = synths == code
        if not torch.any(selected):
            continue
        prediction = model.heads[name](hidden[selected])
        dimension = prediction.shape[1]
        losses[name] = _masked_loss(
            prediction,
            targets[selected, :dimension],
            masks[selected, :dimension].to(prediction.dtype),
        )
    return torch.stack(list(losses.values())).mean(), losses


def _baseline_means(bundle: TrainingBundle) -> dict[str, np.ndarray]:
    result = {}
    for code, name in ((1, "serum1"), (2, "serum2")):
        store = bundle.targets[code]
        rows = [store.preset_row[preset_id] for preset_id in bundle.train_preset_ids[name]]
        values = store.vectors[rows]
        masks = store.masks[rows]
        sums = np.sum(values * masks, axis=0, dtype=np.float64)
        counts = np.sum(masks, axis=0)
        result[name] = (sums / np.maximum(counts, 1)).astype(np.float32)
    return result


@dataclass(slots=True)
class ValidationResult:
    loss: float
    by_synth: dict[str, dict[str, Any]]


def validate(
    model: ParameterInferenceMLP,
    loader: DataLoader,
    device: torch.device,
    baseline: dict[str, np.ndarray],
    mappings: dict[str, list[dict[str, Any]]],
) -> ValidationResult:
    model.eval()
    absolute_sums: dict[str, np.ndarray] = {}
    baseline_sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for inputs, targets, masks, synths, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            synths = synths.to(device)
            hidden = model.encode(inputs)
            for code, name in ((1, "serum1"), (2, "serum2")):
                selected = synths == code
                if not torch.any(selected):
                    continue
                prediction = model.heads[name](hidden[selected])
                dimension = prediction.shape[1]
                target = targets[selected, :dimension]
                mask = masks[selected, :dimension].to(prediction.dtype)
                error = (prediction - target).abs() * mask
                base = torch.from_numpy(baseline[name]).to(device)
                base_error = (base.unsqueeze(0) - target).abs() * mask
                absolute_sums.setdefault(name, np.zeros(dimension, dtype=np.float64))
                baseline_sums.setdefault(name, np.zeros(dimension, dtype=np.float64))
                counts.setdefault(name, np.zeros(dimension, dtype=np.float64))
                absolute_sums[name] += error.sum(dim=0).cpu().numpy()
                baseline_sums[name] += base_error.sum(dim=0).cpu().numpy()
                counts[name] += mask.sum(dim=0).cpu().numpy()
    results = {}
    for name in ("serum1", "serum2"):
        valid = counts[name] > 0
        per_field = np.divide(
            absolute_sums[name], counts[name], out=np.zeros_like(absolute_sums[name]), where=valid
        )
        baseline_field = np.divide(
            baseline_sums[name], counts[name], out=np.zeros_like(baseline_sums[name]), where=valid
        )
        model_mae = float(absolute_sums[name].sum() / counts[name].sum())
        baseline_mae = float(baseline_sums[name].sum() / counts[name].sum())
        worst = np.argsort(np.where(valid, per_field, -1.0))[-20:][::-1]
        results[name] = {
            "mae": model_mae,
            "baseline_mae": baseline_mae,
            "improvement": 1.0 - model_mae / baseline_mae if baseline_mae else 0.0,
            "valid_fields": int(np.sum(valid)),
            "worst_20": [
                {
                    "index": int(index),
                    "name": mappings[name][int(index)]["name"],
                    "mae": float(per_field[index]),
                    "baseline_mae": float(baseline_field[index]),
                }
                for index in worst
                if valid[index]
            ],
        }
    # Monitor the acceptance metric directly and weight the two synths equally;
    # the training objective itself remains masked SmoothL1.
    return ValidationResult(
        loss=float(np.mean([results[name]["mae"] for name in ("serum1", "serum2")])),
        by_synth=results,
    )


def train_model(
    bundle: TrainingBundle,
    platform_env: PlatformEnv,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    patience: int = 20,
    seed: int = 1337,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    _seed_everything(seed)
    device = torch.device(platform_env.compute_backend)
    train_dataset = ParameterDataset(bundle, bundle.train_indices)
    validation_dataset = ParameterDataset(bundle, bundle.validation_indices)
    train_synths = np.asarray(bundle.note_synths[bundle.train_indices], dtype=np.uint8)
    synth_counts = {code: int(np.sum(train_synths == code)) for code in (1, 2)}
    if any(count == 0 for count in synth_counts.values()):
        raise RuntimeError(f"Training split is missing a synth: {synth_counts}")
    sample_weights = torch.as_tensor(
        [1.0 / synth_counts[int(code)] for code in train_synths], dtype=torch.double
    )
    sampler_generator = torch.Generator().manual_seed(seed)
    balanced_samples = 2 * max(synth_counts.values())
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=balanced_samples,
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=parameter_collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=parameter_collate,
    )
    baseline = _baseline_means(bundle)
    model = ParameterInferenceMLP(
        input_dimension=bundle.embeddings.shape[1] + bundle.handcrafted.shape[1],
        serum1_dimension=bundle.targets[1].dimension,
        serum2_dimension=bundle.targets[2].dimension,
    ).to(device)
    # Sparse Serum 2 fields make the training-set mean an unusually strong
    # baseline. Start each sigmoid head exactly there (within numerical bounds)
    # so rare fields do not inherit arbitrary ~0.5 predictions; optimization
    # then learns only audio-correlated departures from that baseline.
    with torch.no_grad():
        for name in ("serum1", "serum2"):
            output_layer = model.heads[name][0]
            assert isinstance(output_layer, nn.Linear)
            output_layer.weight.zero_()
            means = torch.from_numpy(baseline[name]).to(device=device, dtype=output_layer.bias.dtype)
            bounded = means.clamp(1e-6, 1.0 - 1e-6)
            output_layer.bias.copy_(torch.logit(bounded))
    decay_parameters = []
    no_decay_parameters = []
    for parameter in model.parameters():
        (decay_parameters if parameter.ndim >= 2 else no_decay_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": 0.01},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=3e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    mappings = {
        "serum1": bundle.targets[1].mapping,
        "serum2": bundle.targets[2].mapping,
    }
    best_loss = math.inf
    best_state = None
    best_result = None
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        training_losses = []
        for inputs, targets, masks, synths, _ in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            synths = synths.to(device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss, _ = _batch_loss(model, inputs, targets, masks, synths)
            else:
                loss, _ = _batch_loss(model, inputs, targets, masks, synths)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        result = validate(model, validation_loader, device, baseline, mappings)
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(training_losses)),
            "validation_loss": result.loss,
            "serum1_mae": result.by_synth["serum1"]["mae"],
            "serum2_mae": result.by_synth["serum2"]["mae"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(entry)
        print("TRAIN_PROGRESS=" + json.dumps(entry, sort_keys=True), flush=True)
        if result.loss < best_loss - 1e-7:
            best_loss = result.loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_result = result
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_result is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    serum2_schema = json.loads(
        (PROJECT_ROOT / "data" / "models" / "serum2_target_schema.json").read_text()
    )
    checkpoint = {
        "model_state": best_state,
        "model_config": {
            "input_dimension": bundle.embeddings.shape[1] + bundle.handcrafted.shape[1],
            "serum1_dimension": bundle.targets[1].dimension,
            "serum2_dimension": bundle.targets[2].dimension,
        },
        "feature_standardizer": {
            "mean": bundle.standardizer_mean,
            "std": bundle.standardizer_std,
            "names": list(HANDCRAFTED_NAMES),
        },
        "clap_checkpoint": CLAP_CHECKPOINT_NAME,
        "serum1_mapping": mappings["serum1"],
        "serum2_schema": serum2_schema,
        "split": {
            "train_preset_ids": bundle.train_preset_ids,
            "validation_preset_ids": bundle.validation_preset_ids,
            "seed": seed,
        },
        "deep_training": any(preset_id < 0 for preset_id in bundle.train_preset_ids["serum1"]),
        "best_validation": best_result.by_synth,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    report = {
        "device": str(device),
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "elapsed_s": time.monotonic() - started,
        "by_synth": best_result.by_synth,
        "required_improvement": 0.20,
        "deep_training": checkpoint["deep_training"],
        "balanced_sampler": {
            "source_rows_by_synth": {
                "serum1": synth_counts[1],
                "serum2": synth_counts[2],
            },
            "samples_per_epoch": balanced_samples,
            "expected_fraction_per_synth": 0.5,
        },
        "gate_pass": all(
            best_result.by_synth[name]["improvement"] >= 0.20
            for name in ("serum1", "serum2")
        ),
        "history": history,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("TRAIN_SUMMARY=" + json.dumps({key: value for key, value in report.items() if key != "history"}, sort_keys=True))
    return report
