"""Durable storage helpers for completed Match a Sound runs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.db import Database, MatchLibraryRecord


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCH_LIBRARY_ROOT = PROJECT_ROOT / "data" / "match_library"


@dataclass(frozen=True, slots=True)
class ArchivedMatch:
    record: MatchLibraryRecord
    entry_root: Path
    source_audio_path: Path
    result_json_path: Path


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_entry_path(library_root: Path, stored_path: Path | str) -> Path:
    """Resolve a DB path while refusing traversal outside the library root."""

    root = Path(library_root).expanduser().resolve()
    candidate = Path(stored_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Match library path escapes its root: {stored_path}")
    return resolved


def resolve_result_path(result_path: Path, value: str | Path) -> Path:
    """Resolve a path embedded in result.json (new archives use relative paths)."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(result_path).expanduser().resolve().parent / candidate).resolve()


def archive_match(
    db: Database,
    *,
    result_path: Path,
    source_audio_path: Path,
    target_synth: str,
    budget: str,
    library_root: Path = DEFAULT_MATCH_LIBRARY_ROOT,
    batch_id: int | None = None,
    exported_preset_path: Path | None = None,
) -> ArchivedMatch:
    """Copy an ephemeral match into durable storage and insert its DB record."""

    result_path = Path(result_path).expanduser().resolve()
    source_audio_path = Path(source_audio_path).expanduser().resolve()
    root = Path(library_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    recommendation = result.get("recommendation")
    recommendation_dict = recommendation if isinstance(recommendation, dict) else {}

    match_uid = uuid.uuid4().hex
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{match_uid}-", dir=root))
    final_root = root / match_uid
    inserted = False
    try:
        source_folder = temporary_root / "source"
        source_folder.mkdir()
        archived_source = source_folder / source_audio_path.name
        shutil.copy2(source_audio_path, archived_source)
        result.setdefault("source", {})["path"] = archived_source.relative_to(temporary_root).as_posix()

        for key, archive_name in (
            ("winner_audio_path", "winner.wav"),
            ("candidate_path", "candidate.npz"),
        ):
            value = recommendation_dict.get(key)
            if not value:
                continue
            original = resolve_result_path(result_path, str(value))
            if original.is_file():
                shutil.copy2(original, temporary_root / archive_name)
                recommendation_dict[key] = archive_name

        archived_result = temporary_root / "result.json"
        archived_result.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_root.rename(final_root)

        relative_source = Path(match_uid) / "source" / source_audio_path.name
        relative_result = Path(match_uid) / "result.json"
        similarity = float(
            recommendation_dict.get(
                "clap_similarity",
                result.get("top_similarity", 0.0),
            )
            or 0.0
        )
        if similarity <= 1.0:
            similarity *= 100.0
        no_confident = bool(
            result.get("no_confident_match")
            or result.get("no_confident")
            or not recommendation_dict
        )
        recommendation_synth = str(
            recommendation_dict.get("synth") or target_synth
        )
        record_id = db.insert_match_library(
            match_uid=match_uid,
            source_name=source_audio_path.name,
            source_audio_path=relative_source,
            source_content_hash=file_sha1(source_audio_path),
            result_json_path=relative_result,
            target_synth=target_synth,
            budget=budget,
            similarity_percent=similarity,
            base_name=str(
                recommendation_dict.get("base_name")
                or recommendation_dict.get("preset_name")
                or source_audio_path.stem
            ),
            recommendation_synth=recommendation_synth,
            no_confident_match=no_confident,
            batch_id=batch_id,
            exported_preset_path=exported_preset_path,
        )
        inserted = True
        record = db.get_match_library(match_uid)
        if record is None or record.id != record_id:
            raise RuntimeError("Archived match was not readable after insertion")
        return ArchivedMatch(
            record=record,
            entry_root=final_root,
            source_audio_path=resolve_entry_path(root, relative_source),
            result_json_path=resolve_entry_path(root, relative_result),
        )
    except Exception:
        if inserted:
            db.delete_match_library(match_uid)
        shutil.rmtree(final_root if final_root.exists() else temporary_root, ignore_errors=True)
        raise


def delete_archived_match(
    db: Database,
    match_uid: str,
    *,
    library_root: Path = DEFAULT_MATCH_LIBRARY_ROOT,
) -> bool:
    record = db.get_match_library(match_uid)
    if record is None:
        return False
    entry_root = resolve_entry_path(library_root, Path(record.result_json_path).parent)
    staged = entry_root.parent / f".delete-{match_uid}"
    if entry_root.exists():
        entry_root.rename(staged)
    try:
        db.delete_match_library(match_uid)
    except Exception:
        if staged.exists():
            staged.rename(entry_root)
        raise
    if staged.exists():
        shutil.rmtree(staged)
    return True


def resolved_record_paths(
    record: MatchLibraryRecord,
    library_root: Path = DEFAULT_MATCH_LIBRARY_ROOT,
) -> tuple[Path, Path]:
    return (
        resolve_entry_path(library_root, record.source_audio_path),
        resolve_entry_path(library_root, record.result_json_path),
    )
