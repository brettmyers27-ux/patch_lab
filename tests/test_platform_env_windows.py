from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.platform_env import detect_platform_env, windows_vst2_roots


class _Key:
    def __init__(self, value: str) -> None:
        self.value = value

    def __enter__(self) -> "_Key":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeWinreg:
    HKEY_LOCAL_MACHINE = "HKLM"
    HKEY_CURRENT_USER = "HKCU"

    def __init__(self) -> None:
        self.values = {
            ("HKLM", r"SOFTWARE\VST"): r"D:\Audio\VST64",
            ("HKCU", r"SOFTWARE\WOW6432Node\VST"): r"E:\Plug-ins",
        }

    def OpenKey(self, hive: str, path: str) -> _Key:
        try:
            return _Key(self.values[(hive, path)])
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    @staticmethod
    def QueryValueEx(key: _Key, _name: str) -> tuple[str, int]:
        return key.value, 1


class WindowsPlatformTests(unittest.TestCase):
    def test_vst2_registry_roots_precede_common_fallbacks(self) -> None:
        roots = windows_vst2_roots(
            environ={"ProgramFiles": r"C:\Program Files"},
            winreg_module=_FakeWinreg(),
        )
        rendered = [str(path).replace("/", "\\") for path in roots]
        self.assertEqual(rendered[0], r"D:\Audio\VST64")
        self.assertEqual(rendered[1], r"E:\Plug-ins")
        self.assertIn(r"C:\Program Files\Common Files\VST2", rendered)
        self.assertIn(r"C:\Program Files\VSTPlugins", rendered)
        self.assertIn(r"C:\Program Files\Steinberg\VSTPlugins", rendered)

    def test_windows_branch_uses_cpu_wheels_without_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env_values = {
                "USERPROFILE": r"C:\Users\PatchLabTest",
                "LOCALAPPDATA": str(Path(temporary) / "LocalAppData"),
                "ProgramFiles": r"C:\Program Files",
                "PATCHLAB_APP_DATA": str(Path(temporary) / "app-data"),
                "PATCHLAB_MODEL_CACHE": str(Path(temporary) / "model-cache"),
                "PATCHLAB_WINDOWS_TORCH": "cpu",
            }
            with (
                patch("core.platform_env.platform.system", return_value="Windows"),
                patch("core.platform_env.platform.machine", return_value="AMD64"),
                patch.dict("core.platform_env.os.environ", env_values, clear=False),
            ):
                environment = detect_platform_env()

        self.assertEqual(environment.branch, "windows")
        self.assertIn("/whl/cpu", environment.torch_install_command)
        self.assertIn("override", environment.torch_install_reason)
        candidates = [
            str(item.path).replace("/", "\\")
            for item in environment.plugin_candidates
            if item.synth == "serum1" and item.format == "VST2"
        ]
        self.assertIn(
            r"C:\Program Files\Common Files\VST2\Serum_x64.dll",
            candidates,
        )
        serum1_root = environment.factory_roots_for("serum1")[0]
        mixed_case_child = Path(str(serum1_root).upper()) / "Bass" / "Preset.fxp"
        self.assertTrue(environment.path_is_factory("serum1", mixed_case_child))
        self.assertFalse(environment.path_is_factory("serum2", mixed_case_child))

    def test_windows_branch_uses_cuda_128_when_nvidia_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch("core.platform_env.platform.system", return_value="Windows"),
                patch("core.platform_env.platform.machine", return_value="AMD64"),
                patch.dict(
                    "core.platform_env.os.environ",
                    {
                        "PATCHLAB_APP_DATA": str(Path(temporary) / "app-data"),
                        "PATCHLAB_MODEL_CACHE": str(Path(temporary) / "model-cache"),
                        "PATCHLAB_WINDOWS_TORCH": "cuda",
                    },
                    clear=False,
                ),
            ):
                environment = detect_platform_env()

        self.assertIn("/whl/cu128", environment.torch_install_command)


if __name__ == "__main__":
    unittest.main()
