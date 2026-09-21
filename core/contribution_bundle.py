"""Package contributed presets into a few deterministic bundles for upload.

Each bundle is one ``.zip`` holding the preset files, their derived fingerprint
JSON (never audio) and a manifest.  Bundles are byte-for-byte deterministic for
the same inputs, so their content-derived submission id is stable: a retry of an
unfinished contribution sends the identical file under the identical id.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

LEDGER_NAME = "contribution-ledger.json"
#: Stay well under the relay's 32 MiB request ceiling.
MAX_BUNDLE_BYTES = 20 * 1024 * 1024
#: Bound the work done in one scan; the next scan continues where this stopped.
MAX_BUNDLES_PER_SCAN = 5
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class Candidate:
    content_hash: str
    source: Path
    relative_path: str
    preset_id: int


@dataclass(frozen=True, slots=True)
class Bundle:
    path: Path
    submission_id: str
    content_hashes: tuple[str, ...]
    size: int

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


class ContributionLedger:
    """Which preset content hashes this Mac has already contributed.

    Only written after the service has returned a receipt for the bundle.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._hashes = {str(value) for value in raw.get("uploaded", [])}
        except (OSError, ValueError, AttributeError):
            self._hashes = set()

    def __contains__(self, content_hash: str) -> bool:
        return content_hash in self._hashes

    def add_many(self, hashes: Iterable[str]) -> None:
        self._hashes.update(hashes)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(
            json.dumps({"schema": 1, "uploaded": sorted(self._hashes)}), encoding="utf-8"
        )
        os.replace(temporary, self.path)


def _add(handle: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    handle.writestr(info, data)


def build_bundles(
    candidates: Iterable[Candidate],
    *,
    fingerprint_for: Callable[[Candidate], dict[str, Any]],
    work_dir: Path | None = None,
    max_bytes: int = MAX_BUNDLE_BYTES,
) -> Iterator[Bundle]:
    """Yield finished bundles one at a time (each is written to disk, not RAM)."""

    ordered = sorted(candidates, key=lambda item: item.content_hash)
    directory = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="patchlab-contribution-"))
    directory.mkdir(parents=True, exist_ok=True)
    index = 0
    position = 0
    while position < len(ordered):
        index += 1
        partial = directory / f"contribution-{index}.partial.zip"
        entries: list[dict[str, str]] = []
        hashes: list[str] = []
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            while position < len(ordered):
                item = ordered[position]
                payload = json.dumps(
                    fingerprint_for(item), sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                file_name = f"presets/{item.content_hash}/{Path(item.source).name}"
                _add(handle, file_name, Path(item.source).read_bytes())
                _add(handle, f"fingerprints/{item.content_hash}.json", payload)
                entries.append(
                    {
                        "content_hash": item.content_hash,
                        "relative_path": item.relative_path,
                        "file": file_name,
                        "fingerprint": f"fingerprints/{item.content_hash}.json",
                    }
                )
                hashes.append(item.content_hash)
                position += 1
                if handle.fp is not None and handle.fp.tell() >= max_bytes:
                    break
            manifest = json.dumps(
                {"schema": 1, "count": len(entries), "presets": entries},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8")
            _add(handle, "manifest.json", manifest)
        digest = hashlib.sha256(partial.read_bytes()).hexdigest()
        submission_id = digest[:32]
        final = directory / f"patchlab-contribution-{submission_id}.zip"
        os.replace(partial, final)
        yield Bundle(final, submission_id, tuple(hashes), final.stat().st_size)
