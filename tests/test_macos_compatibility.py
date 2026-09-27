from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "packaging"))
import macos_compatibility as compatibility  # noqa: E402
from platform_compatibility import MACOS_MINIMUM, MACOS_REQUIRED_ARCHITECTURE  # noqa: E402


def _row(path: Path, *, minimum: str = "12.3", architectures: list[str] | None = None) -> dict[str, object]:
    return {
        "path": str(path),
        "owner": "test dependency",
        "architectures": architectures or ["arm64"],
        "minimum_macos": minimum,
        "sdk": "15.0",
        "developer_path": False,
        "native_cpu_flag": False,
    }


def test_compatibility_contract_is_central_and_uses_the_evidence_based_floor() -> None:
    assert MACOS_MINIMUM == "12.3"
    assert MACOS_REQUIRED_ARCHITECTURE == "arm64"


def test_release_gate_accepts_target_floor_binary(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "PatchLab.app" / "Contents" / "MacOS" / "PatchLab"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test")
    monkeypatch.setattr(compatibility, "_macho", lambda _path: True)
    monkeypatch.setattr(compatibility, "_inspect", lambda path: _row(path))
    report = compatibility.audit(tmp_path / "PatchLab.app")
    assert report["passed"] is True
    assert report["macho_count"] == 1


def test_release_gate_rejects_high_floor_missing_arm_and_machine_specific_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "PatchLab.app" / "Contents" / "Frameworks" / "bad.dylib"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test")
    row = _row(binary, minimum="26.0", architectures=["x86_64"])
    row["developer_path"] = True
    row["native_cpu_flag"] = True
    monkeypatch.setattr(compatibility, "_macho", lambda _path: True)
    monkeypatch.setattr(compatibility, "_inspect", lambda _path: row)
    report = compatibility.audit(tmp_path / "PatchLab.app")
    assert report["passed"] is False
    assert any("exceeds 12.3" in item for item in report["failures"])
    assert any("missing arm64" in item for item in report["failures"])
    assert any("developer absolute path" in item for item in report["failures"])
    assert any("machine-native CPU flag" in item for item in report["failures"])
