from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS package verification")
def test_macos_pkg_fixture_gate() -> None:
    """The PKG targets /Applications and upgrades without touching user data."""

    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_macos_pkg.py")],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"gate_pass": true' in completed.stdout.lower()


def test_frozen_tokenizer_cache_is_bundled_where_the_runtime_looks() -> None:
    """Regression: the spec bundled the cache under data/models/huggingface while
    core.model_assets resolves it under the runtime family, so a frozen app
    reported an incomplete tokenizer cache."""

    spec = (PROJECT_ROOT / "packaging" / "patchlab.spec").read_text(encoding="utf-8")
    assert '"data/models/huggingface"' not in spec
    assert 'f"data/runtime/{RUNTIME_FAMILY_ID}/models/huggingface"' in spec

    from core.runtime_compatibility import RUNTIME_FAMILY_ID, runtime_data_root

    assert runtime_data_root().as_posix().endswith(f"data/runtime/{RUNTIME_FAMILY_ID}")


def test_frozen_app_still_bundles_the_parameter_predictors() -> None:
    """Regression: a spec change dropped these while core.train / core.delta_model
    still load them from data/models, so a frozen Match failed with FileNotFoundError."""

    spec = (PROJECT_ROOT / "packaging" / "patchlab.spec").read_text(encoding="utf-8")
    for name in ("param_model.pt", "delta_param_model.pt"):
        assert f'(ROOT / "data" / "models" / "{name}", "data/models")' in spec
