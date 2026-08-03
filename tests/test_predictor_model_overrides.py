from __future__ import annotations

from pathlib import Path

import torch

from core.delta_model import DeltaInferenceMLP, load_delta_model
from core.train import ParameterInferenceMLP, load_parameter_model


def test_parameter_model_environment_override(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "parameter.pt"
    model = ParameterInferenceMLP(4, 3, 5)
    torch.save(
        {
            "model_config": {
                "input_dimension": 4,
                "serum1_dimension": 3,
                "serum2_dimension": 5,
            },
            "model_state": model.state_dict(),
        },
        path,
    )
    monkeypatch.setenv("PATCHLAB_PARAM_MODEL", str(path))
    loaded, _checkpoint = load_parameter_model()
    assert loaded.heads["serum1"][0].out_features == 3


def test_delta_model_environment_override(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "delta.pt"
    model = DeltaInferenceMLP(3, 5)
    torch.save(
        {
            "model_config": {"serum1": 3, "serum2": 5},
            "model_state": model.state_dict(),
        },
        path,
    )
    monkeypatch.setenv("PATCHLAB_DELTA_MODEL", str(path))
    loaded, _checkpoint = load_delta_model()
    assert loaded.parameter_encoders["serum2"][0].in_features == 5
