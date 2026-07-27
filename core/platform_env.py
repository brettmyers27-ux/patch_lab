"""The only module allowed to branch on the host operating system."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PluginFormat = Literal["VST2", "VST3", "AU", "CLAP"]
SynthVersion = Literal["serum1", "serum2"]


@dataclass(frozen=True, slots=True)
class PluginCandidate:
    synth: SynthVersion
    format: PluginFormat
    path: Path
    hostable: bool = True

    @property
    def exists(self) -> bool:
        return self.path.exists()


@dataclass(frozen=True, slots=True)
class FactoryPresetRoot:
    synth: SynthVersion
    path: Path


@dataclass(frozen=True, slots=True)
class PlatformEnv:
    branch: Literal["windows", "macos"]
    system_name: str
    machine: str
    compute_backend: Literal["cuda", "mps", "cpu"]
    compute_warning: str | None
    torch_install_command: str
    plugin_candidates: tuple[PluginCandidate, ...]
    preset_roots: tuple[Path, ...]
    factory_preset_roots: tuple[FactoryPresetRoot, ...]
    app_data_dir: Path
    ascii_safe_paths: bool
    legacy_max_path: int | None

    def plugins_for(
        self, synth: SynthVersion, *, existing_only: bool = True, hostable_only: bool = True
    ) -> tuple[PluginCandidate, ...]:
        return tuple(
            item
            for item in self.plugin_candidates
            if item.synth == synth
            and (not existing_only or item.exists)
            and (not hostable_only or item.hostable)
        )

    @property
    def existing_preset_roots(self) -> tuple[Path, ...]:
        return tuple(path for path in self.preset_roots if path.exists())

    def factory_roots_for(
        self, synth: SynthVersion, *, existing_only: bool = False
    ) -> tuple[Path, ...]:
        return tuple(
            item.path
            for item in self.factory_preset_roots
            if item.synth == synth and (not existing_only or item.path.exists())
        )

    def path_is_factory(self, synth: SynthVersion, path: Path) -> bool:
        """Classify only against the correct synth's platform-resolved roots."""

        candidate = Path(path).expanduser().resolve()
        for root in self.factory_roots_for(synth):
            resolved = root.expanduser().resolve()
            if self.branch == "windows":
                prefix = str(resolved).rstrip("\\/").casefold()
                value = str(candidate).casefold()
                if value == prefix or value.startswith(prefix + "\\") or value.startswith(prefix + "/"):
                    return True
            else:
                try:
                    candidate.relative_to(resolved)
                    return True
                except ValueError:
                    pass
        return False


def _torch_backend(branch: str) -> tuple[str, str | None]:
    try:
        import torch
    except ImportError:
        return "cpu", "PyTorch is not installed; compute validation cannot run."

    if branch == "windows":
        if torch.cuda.is_available():
            return "cuda", None
        return "cpu", "CUDA is unavailable; the RTX 5070 requires the cu128 wheel."

    if torch.backends.mps.is_available():
        return "mps", None
    return "cpu", "MPS is unavailable; training will fall back to CPU."


def detect_platform_env() -> PlatformEnv:
    """Resolve all OS-dependent facts once for the rest of the application."""
    system_name = platform.system()
    machine = platform.machine()

    if system_name == "Windows":
        branch = "windows"
        user_home = Path(os.environ.get("USERPROFILE", str(Path.home())))
        common = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Common Files"
        candidates = (
            PluginCandidate("serum1", "VST3", common / "VST3" / "Serum.vst3"),
            PluginCandidate("serum2", "VST3", common / "VST3" / "Serum2.vst3"),
            PluginCandidate(
                "serum1", "VST2", Path(r"C:\Program Files\Steinberg\VstPlugins\Serum_x64.dll")
            ),
            PluginCandidate("serum1", "VST2", Path(r"C:\Program Files\VstPlugins\Serum_x64.dll")),
            PluginCandidate("serum2", "CLAP", common / "CLAP" / "Serum2.clap", hostable=False),
        )
        preset_roots = (
            user_home / "Documents" / "Xfer" / "Serum Presets" / "Presets",
            user_home / "Documents" / "Xfer" / "Serum 2 Presets",
        )
        factory_roots = (
            FactoryPresetRoot("serum1", preset_roots[0]),
            FactoryPresetRoot("serum2", preset_roots[1]),
        )
        app_data_dir = Path(
            os.environ.get(
                "PATCHLAB_APP_DATA",
                str(
                    Path(
                        os.environ.get(
                            "LOCALAPPDATA", str(user_home / "AppData" / "Local")
                        )
                    )
                    / "Patch Lab"
                ),
            )
        )
        torch_command = (
            "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128"
        )
        ascii_safe_paths = True
        legacy_max_path = 260
    elif system_name == "Darwin":
        branch = "macos"
        user_home = Path.home()
        user_plugins = user_home / "Library" / "Audio" / "Plug-Ins"
        system_plugins = Path("/Library/Audio/Plug-Ins")
        # AU is listed first by design. The strategy engine may select VST3 for
        # state injection when AU does not expose a compatible state setter.
        candidates = (
            PluginCandidate("serum1", "AU", user_plugins / "Components" / "Serum.component"),
            PluginCandidate("serum2", "AU", user_plugins / "Components" / "Serum2.component"),
            PluginCandidate("serum1", "AU", system_plugins / "Components" / "Serum.component"),
            PluginCandidate("serum2", "AU", system_plugins / "Components" / "Serum2.component"),
            PluginCandidate("serum1", "VST2", user_plugins / "VST" / "Serum.vst"),
            PluginCandidate("serum1", "VST2", system_plugins / "VST" / "Serum.vst"),
            PluginCandidate("serum1", "VST3", user_plugins / "VST3" / "Serum.vst3"),
            PluginCandidate("serum2", "VST3", user_plugins / "VST3" / "Serum2.vst3"),
            PluginCandidate("serum1", "VST3", system_plugins / "VST3" / "Serum.vst3"),
            PluginCandidate("serum2", "VST3", system_plugins / "VST3" / "Serum2.vst3"),
        )
        preset_roots = (
            user_home / "Music" / "Xfer" / "Serum Presets" / "Presets",
            user_home / "Documents" / "Xfer" / "Serum Presets" / "Presets",
            Path("/Library/Audio/Presets/Xfer Records/Serum Presets/Presets"),
            user_home / "Music" / "Xfer" / "Serum 2 Presets",
            user_home / "Documents" / "Xfer" / "Serum 2 Presets",
            Path("/Library/Audio/Presets/Xfer Records/Serum 2 Presets"),
        )
        factory_roots = (
            FactoryPresetRoot("serum1", preset_roots[0]),
            FactoryPresetRoot("serum1", preset_roots[1]),
            FactoryPresetRoot("serum1", preset_roots[2]),
            FactoryPresetRoot("serum2", preset_roots[3]),
            FactoryPresetRoot("serum2", preset_roots[4]),
            FactoryPresetRoot("serum2", preset_roots[5]),
        )
        app_data_dir = Path(
            os.environ.get(
                "PATCHLAB_APP_DATA",
                str(user_home / "Library" / "Application Support" / "Patch Lab"),
            )
        )
        torch_command = "pip install torch torchaudio"
        ascii_safe_paths = True
        legacy_max_path = None
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    else:
        raise RuntimeError(
            f"Unsupported operating system {system_name!r}; Patch Lab supports Windows and macOS."
        )

    default_model_cache = (
        app_data_dir / "models" / "huggingface"
        if os.environ.get("PATCHLAB_DISTRIBUTION_MODE", "0").strip() == "1"
        else Path(__file__).resolve().parents[1] / "data" / "models" / "huggingface"
    )
    model_cache = Path(
        os.environ.get("PATCHLAB_MODEL_CACHE", str(default_model_cache))
    ).expanduser().resolve()
    model_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(model_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(model_cache / "transformers"))
    backend, warning = _torch_backend(branch)
    return PlatformEnv(
        branch=branch,
        system_name=system_name,
        machine=machine,
        compute_backend=backend,  # type: ignore[arg-type]
        compute_warning=warning,
        torch_install_command=torch_command,
        plugin_candidates=candidates,
        preset_roots=preset_roots,
        factory_preset_roots=factory_roots,
        app_data_dir=app_data_dir,
        ascii_safe_paths=ascii_safe_paths,
        legacy_max_path=legacy_max_path,
    )


ENV = detect_platform_env()
