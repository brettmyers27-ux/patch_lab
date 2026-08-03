from __future__ import annotations

import torch

from scripts.stage2_finetune_clap import split_preset_ids, symmetric_info_nce


def test_preset_split_is_deterministic_and_disjoint() -> None:
    train_a, validation_a = split_preset_ids(
        list(range(100)), seed=20260802, validation_fraction=0.10
    )
    train_b, validation_b = split_preset_ids(
        list(range(100)), seed=20260802, validation_fraction=0.10
    )

    assert (train_a, validation_a) == (train_b, validation_b)
    assert len(validation_a) == 10
    assert set(train_a).isdisjoint(validation_a)


def test_info_nce_rewards_aligned_positive_pairs() -> None:
    aligned = torch.eye(4)
    reversed_rows = torch.flip(aligned, dims=(0,))
    temperature = torch.tensor(0.07)

    aligned_loss = symmetric_info_nce(aligned, aligned, temperature)
    mismatched_loss = symmetric_info_nce(aligned, reversed_rows, temperature)

    assert aligned_loss < mismatched_loss
