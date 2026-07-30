"""Training data assembly over the feature store and SQLite parameter labels."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from core.db import DEFAULT_DB_PATH
from core.serum2_targets import expanded_output_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"


@dataclass(slots=True)
class TargetStore:
    vectors: np.ndarray
    masks: np.ndarray
    preset_row: dict[int, int]
    dimension: int
    mapping: list[dict[str, Any]]


@dataclass(slots=True)
class TrainingBundle:
    embeddings: np.ndarray
    handcrafted: np.ndarray
    note_preset_ids: np.ndarray
    note_synths: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    standardizer_mean: np.ndarray
    standardizer_std: np.ndarray
    targets: dict[int, TargetStore]
    train_preset_ids: dict[str, list[int]]
    validation_preset_ids: dict[str, list[int]]


def _serum1_targets(db_path: Path) -> TargetStore:
    connection = sqlite3.connect(db_path)
    preset_ids = [
        int(row[0])
        for row in connection.execute(
            "SELECT id FROM presets WHERE synth='serum1' "
            "AND EXISTS (SELECT 1 FROM params WHERE preset_id=presets.id) ORDER BY id"
        )
    ]
    preset_row = {preset_id: index for index, preset_id in enumerate(preset_ids)}
    dimension = int(
        connection.execute(
            "SELECT MAX(param_index)+1 FROM params pa JOIN presets p ON p.id=pa.preset_id "
            "WHERE p.synth='serum1'"
        ).fetchone()[0]
    )
    vectors = np.zeros((len(preset_ids), dimension), dtype=np.float32)
    masks = np.zeros_like(vectors, dtype=np.bool_)
    names = [Counter() for _ in range(dimension)]
    displays = [set() for _ in range(dimension)]
    normalized_values = [set() for _ in range(dimension)]
    for preset_id, index, name, value, display in connection.execute(
        "SELECT pa.preset_id,pa.param_index,pa.param_name,pa.norm_value,pa.display_value "
        "FROM params pa JOIN presets p ON p.id=pa.preset_id "
        "WHERE p.synth='serum1' ORDER BY pa.preset_id,pa.param_index"
    ):
        row = preset_row[int(preset_id)]
        parameter_index = int(index)
        if value is not None:
            vectors[row, parameter_index] = float(value)
            masks[row, parameter_index] = True
        names[int(index)][str(name)] += 1
        if len(displays[int(index)]) <= 32:
            displays[int(index)].add(str(display))
        if value is not None and len(normalized_values[int(index)]) <= 64:
            normalized_values[int(index)].add(float(value))
    mapping = []
    for index in range(dimension):
        aliases = [name for name, _ in names[index].most_common()]
        mapping.append(
            {
                "index": index,
                "name": aliases[0],
                "aliases": aliases,
                "stepped": len(displays[index]) <= 32,
                "observed_display_values": len(displays[index]),
                "step_values": (
                    sorted(normalized_values[index]) if len(displays[index]) <= 32 else []
                ),
            }
        )
    return TargetStore(
        vectors=vectors,
        masks=masks,
        preset_row=preset_row,
        dimension=dimension,
        mapping=mapping,
    )


def _serum2_targets(
    targets_path: Path | None = None,
    schema_path: Path | None = None,
) -> TargetStore:
    """Load the Serum 2 parameter targets.

    Both paths are injectable so a distributed install can point at delivered
    artifacts instead of the source checkout; omitting them keeps the existing
    repository-relative behavior for training and offline scripts.
    """

    stored = np.load(targets_path or FEATURE_DIR / "serum2_targets.npz")
    vectors = stored["vectors"].astype(np.float32, copy=False)
    masks = stored["masks"].astype(np.bool_, copy=False)
    preset_ids = stored["preset_ids"].astype(np.int64, copy=False)
    schema = json.loads(
        (
            schema_path
            or PROJECT_ROOT / "data" / "models" / "serum2_target_schema.json"
        ).read_text(encoding="utf-8")
    )
    return TargetStore(
        vectors=vectors,
        masks=masks,
        preset_row={int(preset_id): index for index, preset_id in enumerate(preset_ids)},
        dimension=int(schema["vector_length"]),
        mapping=expanded_output_mapping(schema),
    )


def _split_presets(
    db_path: Path, seed: int, validation_fraction: float
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    connection = sqlite3.connect(db_path)
    rng = np.random.default_rng(seed)
    train: dict[str, list[int]] = {}
    validation: dict[str, list[int]] = {}
    for synth in ("serum1", "serum2"):
        ids = np.asarray(
            [
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM presets WHERE synth=? AND status='rendered' "
                    "AND EXISTS (SELECT 1 FROM params WHERE preset_id=presets.id) ORDER BY id",
                    (synth,),
                )
            ],
            dtype=np.int64,
        )
        rng.shuffle(ids)
        count = max(1, int(round(len(ids) * validation_fraction)))
        validation[synth] = sorted(map(int, ids[:count]))
        train[synth] = sorted(map(int, ids[count:]))
    return train, validation


def load_training_bundle(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    seed: int = 1337,
    validation_fraction: float = 0.1,
) -> TrainingBundle:
    complete = np.load(FEATURE_DIR / "note_complete.npy", mmap_mode="r")
    if not bool(np.all(complete)):
        raise RuntimeError("Feature extraction is incomplete")
    embeddings = np.load(FEATURE_DIR / "note_embeddings.npy", mmap_mode="r")
    handcrafted = np.load(FEATURE_DIR / "note_handcrafted.npy", mmap_mode="r")
    manifest = np.load(FEATURE_DIR / "note_manifest.npz")
    preset_ids = manifest["preset_ids"].astype(np.int64, copy=False)
    synths = manifest["synths"].astype(np.uint8, copy=False)
    train_presets, validation_presets = _split_presets(db_path, seed, validation_fraction)
    train_ids = set(train_presets["serum1"]) | set(train_presets["serum2"])
    validation_ids = set(validation_presets["serum1"]) | set(validation_presets["serum2"])
    train_indices = np.asarray(
        [index for index, preset_id in enumerate(preset_ids) if int(preset_id) in train_ids],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [index for index, preset_id in enumerate(preset_ids) if int(preset_id) in validation_ids],
        dtype=np.int64,
    )
    mean = np.mean(handcrafted[train_indices], axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(handcrafted[train_indices], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return TrainingBundle(
        embeddings=embeddings,
        handcrafted=handcrafted,
        note_preset_ids=preset_ids,
        note_synths=synths,
        train_indices=train_indices,
        validation_indices=validation_indices,
        standardizer_mean=mean,
        standardizer_std=std,
        targets={1: _serum1_targets(db_path), 2: _serum2_targets()},
        train_preset_ids=train_presets,
        validation_preset_ids=validation_presets,
    )


def add_serum1_synthetic(bundle: TrainingBundle) -> TrainingBundle:
    complete = np.load(FEATURE_DIR / "synthetic_s1_complete.npy", mmap_mode="r")
    if len(complete) != 20_000 or not bool(np.all(complete)):
        raise RuntimeError("The 20,000-patch Serum 1 augmentation store is incomplete")
    synthetic_embeddings = np.load(FEATURE_DIR / "synthetic_s1_embeddings.npy", mmap_mode="r")
    synthetic_features = np.load(FEATURE_DIR / "synthetic_s1_handcrafted.npy", mmap_mode="r")
    synthetic_targets = np.load(FEATURE_DIR / "synthetic_s1_targets.npy", mmap_mode="r")
    original_count = len(bundle.note_preset_ids)
    synthetic_ids = -np.arange(1, len(complete) + 1, dtype=np.int64)
    embeddings = np.concatenate((bundle.embeddings, synthetic_embeddings), axis=0)
    handcrafted = np.concatenate((bundle.handcrafted, synthetic_features), axis=0)
    preset_ids = np.concatenate((bundle.note_preset_ids, synthetic_ids))
    synths = np.concatenate((bundle.note_synths, np.ones(len(complete), dtype=np.uint8)))
    train_indices = np.concatenate(
        (bundle.train_indices, np.arange(original_count, original_count + len(complete), dtype=np.int64))
    )
    store = bundle.targets[1]
    serum1_vectors = np.concatenate((store.vectors, synthetic_targets), axis=0)
    serum1_masks = np.concatenate(
        (store.masks, np.ones_like(synthetic_targets, dtype=np.bool_)), axis=0
    )
    serum1_rows = dict(store.preset_row)
    start = len(store.vectors)
    serum1_rows.update({int(preset_id): start + index for index, preset_id in enumerate(synthetic_ids)})
    targets = dict(bundle.targets)
    targets[1] = TargetStore(
        vectors=serum1_vectors,
        masks=serum1_masks,
        preset_row=serum1_rows,
        dimension=store.dimension,
        mapping=store.mapping,
    )
    train_preset_ids = {name: list(ids) for name, ids in bundle.train_preset_ids.items()}
    train_preset_ids["serum1"].extend(map(int, synthetic_ids))
    mean = np.mean(handcrafted[train_indices], axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(
        np.std(handcrafted[train_indices], axis=0, dtype=np.float64).astype(np.float32),
        1e-6,
    )
    return TrainingBundle(
        embeddings=embeddings,
        handcrafted=handcrafted,
        note_preset_ids=preset_ids,
        note_synths=synths,
        train_indices=train_indices,
        validation_indices=bundle.validation_indices,
        standardizer_mean=mean,
        standardizer_std=std,
        targets=targets,
        train_preset_ids=train_preset_ids,
        validation_preset_ids=bundle.validation_preset_ids,
    )


class ParameterDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]]):
    def __init__(self, bundle: TrainingBundle, indices: Sequence[int]) -> None:
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        note_index = int(self.indices[item])
        synth = int(self.bundle.note_synths[note_index])
        preset_id = int(self.bundle.note_preset_ids[note_index])
        feature = np.concatenate(
            (
                np.asarray(self.bundle.embeddings[note_index], dtype=np.float32),
                (
                    np.asarray(self.bundle.handcrafted[note_index], dtype=np.float32)
                    - self.bundle.standardizer_mean
                )
                / self.bundle.standardizer_std,
            )
        )
        store = self.bundle.targets[synth]
        row = store.preset_row[preset_id]
        return (
            torch.from_numpy(feature),
            torch.from_numpy(np.asarray(store.vectors[row], dtype=np.float32)),
            torch.from_numpy(np.asarray(store.masks[row], dtype=np.bool_)),
            synth,
            preset_id,
        )


def parameter_collate(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(item[1].shape[0] for item in batch)
    targets = torch.zeros((len(batch), maximum), dtype=torch.float32)
    masks = torch.zeros((len(batch), maximum), dtype=torch.bool)
    for index, (_, target, mask, _, _) in enumerate(batch):
        targets[index, : target.shape[0]] = target
        masks[index, : mask.shape[0]] = mask
    return (
        torch.stack([item[0] for item in batch]),
        targets,
        masks,
        torch.tensor([item[3] for item in batch], dtype=torch.int64),
        torch.tensor([item[4] for item in batch], dtype=torch.int64),
    )
