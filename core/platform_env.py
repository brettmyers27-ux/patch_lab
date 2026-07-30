"""The only module allowed to branch on the host operating system."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.model_assets import configure_model_environment


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
    torch_install_reason: str
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


def _dedupe_paths(paths: list[Path]) -> tuple[Path, ...]:
    """Preserve probe order while treating Windows paths case-insensitively."""

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).replace("/", "\\").casefold()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return tuple(unique)


def windows_vst2_roots(
    *,
    environ: dict[str, str] | os._Environ[str] | None = None,
    winreg_module: Any | None = None,
) -> tuple[Path, ...]:
    """Resolve 64-bit VST2 roots from the registry and common conventions.

    ``winreg_module`` is injectable so the complete lookup can be exercised on
    macOS without pretending that Windows itself was tested.
    """

    values = os.environ if environ is None else environ
    roots: list[Path] = []
    if winreg_module is None:
        try:
            import winreg as winreg_module  # type: ignore[no-redef]
        except ImportError:
            winreg_module = None

    if winreg_module is not None:
        registry_keys = (
            (winreg_module.HKEY_LOCAL_MACHINE, r"SOFTWARE\VST"),
            (winreg_module.HKEY_CURRENT_USER, r"SOFTWARE\VST"),
            (winreg_module.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\VST"),
            (winreg_module.HKEY_CURRENT_USER, r"SOFTWARE\WOW6432Node\VST"),
        )
        for hive, key_name in registry_keys:
            try:
                with winreg_module.OpenKey(hive, key_name) as key:
                    value, _kind = winreg_module.QueryValueEx(key, "VSTPluginsPath")
            except (FileNotFoundError, OSError):
                continue
            if isinstance(value, str) and value.strip():
                roots.append(Path(os.path.expandvars(value.strip().strip('"'))))

    program_files = Path(
        values.get("ProgramW6432") or values.get("ProgramFiles") or r"C:\Program Files"
    )
    roots.extend(
        (
            program_files / "Common Files" / "VST2",
            program_files / "VSTPlugins",
            program_files / "Steinberg" / "VSTPlugins",
        )
    )
    return _dedupe_paths(roots)


def windows_nvidia_hardware_present(
    *, environ: dict[str, str] | os._Environ[str] | None = None
) -> tuple[bool, str]:
    """Detect physical NVIDIA hardware without depending on an installed torch."""

    values = os.environ if environ is None else environ
    forced = values.get("PATCHLAB_WINDOWS_TORCH", "").strip().casefold()
    if forced in {"cuda", "cpu"}:
        return forced == "cuda", f"PATCHLAB_WINDOWS_TORCH={forced} override"

    commands: list[list[str]] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        commands.append([nvidia_smi, "--query-gpu=name", "--format=csv,noheader"])
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell:
        commands.append(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join \"`n\"",
            ]
        )
    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode == 0 and "nvidia" in output.casefold():
            first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
            return True, f"NVIDIA adapter detected: {first_line}"
    return False, "No NVIDIA display adapter was detected; using CPU-only PyTorch wheels"


def _torch_backend(branch: str) -> tuple[str, str | None]:
    try:
        import torch
    except ImportError:
        return "cpu", "PyTorch is not installed; compute validation cannot run."

    if branch == "windows":
        if torch.cuda.is_available():
            return "cuda", None
        return "cpu", "CUDA is unavailable; compute will use the CPU."

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
        program_files = Path(
            os.environ.get("ProgramW6432")
            or os.environ.get("ProgramFiles", r"C:\Program Files")
        )
        common = program_files / "Common Files"
        vst2_paths: list[Path] = []
        for root in windows_vst2_roots():
            vst2_paths.extend((root / "Serum_x64.dll", root / "Serum.dll"))
        candidates = (
            PluginCandidate("serum1", "VST3", common / "VST3" / "Serum.vst3"),
            PluginCandidate("serum2", "VST3", common / "VST3" / "Serum2.vst3"),
            *(PluginCandidate("serum1", "VST2", path) for path in _dedupe_paths(vst2_paths)),
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
        has_nvidia, torch_reason = windows_nvidia_hardware_present()
        torch_index = "cu128" if has_nvidia else "cpu"
        torch_command = (
            "pip install torch torchaudio "
            f"--index-url https://download.pytorch.org/whl/{torch_index}"
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
        torch_reason = "Standard PyPI wheels include Apple MPS support"
        ascii_safe_paths = True
        legacy_max_path = None
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    else:
        raise RuntimeError(
            f"Unsupported operating system {system_name!r}; Patch Lab supports Windows and macOS."
        )

    # Model assets use one environment-driven resolver on both platforms.
    # In frozen builds the runtime hook points this at bundled assets; installed
    # source launchers point it at the checkout's populated cache.
    configure_model_environment()
    backend, warning = _torch_backend(branch)
    return PlatformEnv(
        branch=branch,
        system_name=system_name,
        machine=machine,
        compute_backend=backend,  # type: ignore[arg-type]
        compute_warning=warning,
        torch_install_command=torch_command,
        torch_install_reason=torch_reason,
        plugin_candidates=candidates,
        preset_roots=preset_roots,
        factory_preset_roots=factory_roots,
        app_data_dir=app_data_dir,
        ascii_safe_paths=ascii_safe_paths,
        legacy_max_path=legacy_max_path,
    )


ENV = detect_platform_env()
