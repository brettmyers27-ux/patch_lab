"""Where PatchLab saves the presets it generates: one setting, every writer.

Single Match auto-save, batch Match auto-save, Retry Saving Preset, Run
Again, and the Settings UI itself all resolve the destination folder through
``configured_preset_output_folder`` -- never a value computed independently
and handed down separately, so there is exactly one answer to "where does a
newly generated preset go right now."

Changing the setting only ever affects presets generated *after* the
change: a Library record's ``exported_preset_path`` is an absolute path
recorded at save time, and this module never moves, renames, or otherwise
touches an already-saved file.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from core.platform_env import ENV, PlatformEnv


SETTINGS_FILENAME = "preset-output-settings.json"
SCHEMA_VERSION = 1
#: Test/automation escape hatch, matching PATCHLAB_AUDIO_STORAGE's convention.
OVERRIDE_ENV = "PATCHLAB_PRESET_OUTPUT_FOLDER"

# Xfer ships these system preset trees world-writable specifically so any
# local user can save into them without administrator rights -- Serum 1's own
# "User" folder even contains a literal SaveYourPresetsHere.txt. These are
# also the exact folders Serum's own preset browser scans, unlike an
# arbitrary path under the user's home directory, so PatchLab output saved
# here shows up inside Serum immediately rather than needing a manual
# "Add Folder" step. This is the *default* only; a configured custom folder
# replaces it entirely.
MACOS_SERUM1_USER_PRESETS = Path(
    "/Library/Audio/Presets/Xfer Records/Serum Presets/Presets/User"
)
MACOS_SERUM2_USER_PRESETS = Path(
    "/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets/User"
)


@dataclass(frozen=True, slots=True)
class PresetOutputPreferences:
    """``folder`` is None until a user explicitly picks one (the default)."""

    folder: str | None = None
    schema: int = SCHEMA_VERSION


def settings_path(env: PlatformEnv = ENV) -> Path:
    return Path(env.app_data_dir) / SETTINGS_FILENAME


def load_preset_output_preferences(env: PlatformEnv = ENV) -> PresetOutputPreferences:
    path = settings_path(env)
    if not path.is_file():
        return PresetOutputPreferences()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        folder = str(raw.get("folder") or "").strip() or None
        return PresetOutputPreferences(folder=folder)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # A damaged preference must not silently redirect saves elsewhere.
        return PresetOutputPreferences()


def save_preset_output_preferences(
    preferences: PresetOutputPreferences, env: PlatformEnv = ENV
) -> Path:
    path = settings_path(env)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(asdict(preferences), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return path


def default_preset_output_root(synth: str, *, env: PlatformEnv = ENV) -> Path:
    """Today's per-synth Serum location, used only when nothing is configured."""

    if env.branch == "macos":
        return (
            MACOS_SERUM2_USER_PRESETS if synth == "serum2" else MACOS_SERUM1_USER_PRESETS
        )

    token = "serum 2" if synth == "serum2" else "serum presets"
    home = Path.home().resolve()

    def under_home(path: Path) -> bool:
        try:
            path.resolve().relative_to(home)
        except (OSError, ValueError):
            return False
        return True

    matching = [path for path in env.preset_roots if token in str(path).casefold()]
    for candidate in matching:
        if under_home(candidate) and candidate.is_dir():
            return candidate
    for candidate in matching:
        if under_home(candidate):
            return candidate
    existing = [path for path in env.existing_preset_roots if token in str(path).casefold()]
    return existing[0] if existing else home


def configured_preset_output_folder(synth: str, *, env: PlatformEnv = ENV) -> Path:
    """The one destination every generated-preset writer resolves to.

    A configured custom folder is used exactly as chosen, for BOTH Serum
    generations -- one setting, not two confusing per-synth pickers. With
    nothing configured, this preserves 1.5.5's exact default: Serum's own
    per-synth User preset folder, with a "PatchLab" subfolder so a user's own
    library is never mixed with generated output.
    """

    override = os.environ.get(OVERRIDE_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    configured = load_preset_output_preferences(env).folder
    if configured:
        return Path(configured).expanduser().resolve()
    return (default_preset_output_root(synth, env=env) / "PatchLab").expanduser().resolve()


def ensure_writable(folder: Path) -> str:
    """Verify PatchLab can actually save into ``folder``. "" means it can.

    Creates ``folder`` (and parents) if missing, writes a small temporary
    probe file, then removes it -- never leaves anything behind either way.
    """

    folder = Path(folder).expanduser()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / f".patchlab-write-check-{os.getpid()}-{time.time_ns()}.tmp"
        probe.write_bytes(b"ok")
        probe.unlink()
        return ""
    except OSError as exc:
        return str(exc)
