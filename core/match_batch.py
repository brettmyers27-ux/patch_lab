"""Pure helpers shared by the Match Library batch UI and its tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.audio_input import SUPPORTED_AUDIO_SUFFIXES
from core.match_library import file_sha1


@dataclass(frozen=True, slots=True)
class BatchDiscovery:
    supported: tuple[Path, ...]
    unsupported_count: int


def sanitize_folder_name(value: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f<>:\"/\\\\|?*]+", "-", value.strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
    return cleaned[:80]


def discover_batch_audio(folder: Path, *, recursive: bool = False) -> BatchDiscovery:
    root = Path(folder).expanduser().resolve()
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = sorted((path for path in iterator if path.is_file()), key=lambda p: str(p).casefold())
    supported = tuple(
        path for path in files if path.suffix.casefold() in SUPPORTED_AUDIO_SUFFIXES
    )
    return BatchDiscovery(supported, len(files) - len(supported))


def resumable_batch_files(
    files: tuple[Path, ...] | list[Path],
    completed_hashes: set[str],
) -> tuple[list[tuple[Path, str]], int]:
    pending: list[tuple[Path, str]] = []
    skipped = 0
    for path in files:
        digest = file_sha1(path)
        if digest in completed_hashes:
            skipped += 1
        else:
            pending.append((path, digest))
    return pending, skipped


def disambiguated_preset_path(folder: Path, stem: str, extension: str) -> Path:
    output = Path(folder) / f"{stem}{extension}"
    suffix = 2
    while output.exists():
        output = Path(folder) / f"{stem} {suffix}{extension}"
        suffix += 1
    return output
