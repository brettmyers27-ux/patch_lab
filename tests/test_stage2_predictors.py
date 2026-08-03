from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.stage2_train_predictors import Split, _pair_arrays, _selected_rows


def test_stage2_shard_unpack_and_preset_split(tmp_path: Path) -> None:
    path = tmp_path / "shard-000.npz"
    masks = np.asarray([[True, False, True], [True, True, False]], dtype=bool)
    np.savez(
        path,
        pair_base_rows=np.asarray([0, 0, 1, 1], dtype=np.uint16),
        synth_codes=np.asarray([1, 2], dtype=np.uint8),
        preset_ids=np.asarray([10, 20], dtype=np.int32),
        vector_lengths=np.asarray([3, 3], dtype=np.uint16),
        parameter_vectors=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float16),
        parameter_masks_packed=np.packbits(masks, axis=1, bitorder="little"),
        embeddings=np.zeros((4, 512), dtype=np.float16),
        handcrafted=np.zeros((4, 9), dtype=np.float32),
    )
    data = _pair_arrays(path)
    assert data["targets"].shape == (4, 3)
    assert data["masks"].tolist() == [masks[0].tolist(), masks[0].tolist(), masks[1].tolist(), masks[1].tolist()]
    split = Split(train={1: {10}, 2: set()}, validation={1: set(), 2: {20}})
    assert _selected_rows(data, split, False).tolist() == [0, 1]
    assert _selected_rows(data, split, True).tolist() == [2, 3]
