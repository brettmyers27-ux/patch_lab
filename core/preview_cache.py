"""Content-addressed, durable octave-preview cache helpers.

Factory/local presets use their source-file SHA-1 directly. Genuinely modified
recommendations use ``generated-<sha256>`` where the digest covers the synth,
the exact float32 parameter vector, and its boolean mask. This keeps generated
audio separate from the preset that seeded it while deduplicating byte-identical
recommendations across Match Library entries.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.db import Database
from core.match_library import resolve_entry_path, resolve_result_path


PREVIEW_NOTES = (24, 36, 48, 60, 72, 84, 96)
GENERATED_PREFIX = "generated-"


@dataclass(frozen=True, slots=True)
class PreviewCleanup:
    deleted_keys: tuple[str, ...] = ()
    retained_shared_keys: tuple[str, ...] = ()
    retained_preset_keys: tuple[str, ...] = ()
    deleted_wavs: int = 0


def preview_cache_path(cache_root: Path, cache_key: str, midi_note: int) -> Path:
    """Return the canonical path and reject unsafe/non-octave inputs."""

    key = str(cache_key).strip()
    if not key or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in key):
        raise ValueError(f"Unsafe preview cache key: {cache_key!r}")
    if int(midi_note) not in PREVIEW_NOTES:
        raise ValueError(f"Unsupported preview note {midi_note}")
    return Path(cache_root).expanduser().resolve() / "audio" / key / f"{int(midi_note)}.wav"


def _generated_key_from_candidate(
    result_path: Path,
    recommendation: dict[str, Any],
) -> str | None:
    candidate_value = recommendation.get("candidate_path")
    if not candidate_value:
        return None
    candidate_path = resolve_result_path(result_path, str(candidate_value))
    if not candidate_path.is_file():
        return None
    with np.load(candidate_path) as stored:
        vector = np.ascontiguousarray(stored["vector"], dtype=np.float32)
        mask = np.ascontiguousarray(stored["mask"], dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(str(recommendation.get("synth", "")).encode("ascii", errors="ignore"))
    digest.update(vector.tobytes())
    digest.update(mask.tobytes())
    return GENERATED_PREFIX + digest.hexdigest()


def recommendation_cache_key(
    result_path: Path,
    recommendation: dict[str, Any],
) -> str:
    """Resolve a stable identity for a recommendation.

    An unmodified recommendation is the underlying preset and intentionally
    shares its content hash. A modified recommendation is keyed by its actual
    parameter data, never by the seed preset.
    """

    if not bool(recommendation.get("meaningfully_modified", False)):
        content_hash = str(recommendation.get("content_hash") or "").strip()
        if content_hash:
            return content_hash
    stored_key = str(recommendation.get("preview_cache_key") or "").strip()
    if stored_key:
        return stored_key
    generated = _generated_key_from_candidate(Path(result_path), recommendation)
    if generated:
        return generated
    # Old archives can lack their candidate file. A match UID/directory is a
    # stable final fallback and avoids colliding with the seed preset.
    digest = hashlib.sha256(
        f"{Path(result_path).resolve().parent.name}:{recommendation.get('synth', '')}".encode()
    ).hexdigest()
    return GENERATED_PREFIX + digest


def annotate_recommendation_cache_key(
    result: dict[str, Any],
    result_path: Path,
) -> str | None:
    recommendation = result.get("recommendation")
    if not isinstance(recommendation, dict):
        return None
    key = recommendation_cache_key(result_path, recommendation)
    recommendation["preview_cache_key"] = key
    return key


def unmodified_recommendation_basis_index(result: dict[str, Any]) -> int | None:
    """Return the row that is honestly the unchanged recommendation basis."""

    recommendation = result.get("recommendation")
    if not isinstance(recommendation, dict) or bool(
        recommendation.get("meaningfully_modified", False)
    ):
        return None
    content_hash = str(recommendation.get("content_hash") or "")
    for index, item in enumerate(result.get("existing_matches", [])):
        if isinstance(item, dict) and str(item.get("content_hash") or "") == content_hash:
            return index
    return None


def result_cache_keys(result_path: Path) -> set[str]:
    """Read every cache identity referenced by one archived result."""

    result_path = Path(result_path).expanduser().resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    keys = {
        str(item["content_hash"])
        for item in result.get("existing_matches", [])
        if isinstance(item, dict) and item.get("content_hash")
    }
    recommendation = result.get("recommendation")
    if isinstance(recommendation, dict):
        keys.add(recommendation_cache_key(result_path, recommendation))
    return keys


def match_library_reference_counts(
    db: Database,
    *,
    library_root: Path,
) -> Counter[str]:
    """Count cache references from authoritative Match Library rows/results."""

    counts: Counter[str] = Counter()
    for record in db.list_match_library():
        result_path = resolve_entry_path(library_root, record.result_json_path)
        if not result_path.is_file():
            continue
        counts.update(result_cache_keys(result_path))
    return counts


def cleanup_deleted_entry_previews(
    deleted_keys: set[str],
    *,
    remaining_references: Counter[str],
    cache_root: Path,
) -> PreviewCleanup:
    """Remove only unreferenced generated audio.

    Factory and user-preset hashes are intentionally a permanent shared cache:
    they remain useful after Match Library history is deleted and are expensive
    to recreate. Generated recommendations have no owner outside Match Library,
    so their last reference removes the complete hash directory.
    """

    deleted: list[str] = []
    shared: list[str] = []
    presets: list[str] = []
    deleted_wavs = 0
    for key in sorted(deleted_keys):
        if remaining_references.get(key, 0):
            shared.append(key)
            continue
        if not key.startswith(GENERATED_PREFIX):
            presets.append(key)
            continue
        directory = preview_cache_path(cache_root, key, PREVIEW_NOTES[0]).parent
        if directory.is_dir():
            deleted_wavs += sum(1 for path in directory.glob("*.wav") if path.is_file())
            shutil.rmtree(directory)
        deleted.append(key)
    audio_root = Path(cache_root).expanduser().resolve() / "audio"
    if audio_root.is_dir() and not any(audio_root.iterdir()):
        audio_root.rmdir()
    return PreviewCleanup(
        deleted_keys=tuple(deleted),
        retained_shared_keys=tuple(shared),
        retained_preset_keys=tuple(presets),
        deleted_wavs=deleted_wavs,
    )
