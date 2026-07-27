"""Fast launch-time matching of shipped factory hashes to local preset files."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from core.factory_bundle import DEFAULT_FACTORY_BUNDLE, FactoryBundle
from core.platform_env import ENV, PlatformEnv
from core.preset_scan import sha1_file


@dataclass(frozen=True, slots=True)
class FactoryVerification:
    bundle_available: bool
    factory_directories_found: int
    local_files_found: int
    known_bundle_hashes: int
    matched_hashes: int
    missing_hashes: tuple[str, ...]
    unknown_local_hashes: tuple[str, ...]
    local_paths_by_hash: dict[str, str]
    elapsed_s: float

    @property
    def no_factory_install(self) -> bool:
        return self.factory_directories_found == 0


def verify_local_factory_install(
    *,
    bundle_path: Path = DEFAULT_FACTORY_BUNDLE,
    env: PlatformEnv = ENV,
    mapping_path: Path | None = None,
) -> FactoryVerification:
    started = time.monotonic()
    if not Path(bundle_path).is_file():
        return FactoryVerification(
            bundle_available=False,
            factory_directories_found=0,
            local_files_found=0,
            known_bundle_hashes=0,
            matched_hashes=0,
            missing_hashes=(),
            unknown_local_hashes=(),
            local_paths_by_hash={},
            elapsed_s=time.monotonic() - started,
        )
    known = FactoryBundle(bundle_path).known_hashes()
    paths: set[Path] = set()
    directory_count = 0
    for synth, suffix in (("serum1", ".fxp"), ("serum2", ".serumpreset")):
        for root in env.factory_roots_for(synth, existing_only=True):
            directory_count += 1
            paths.update(
                path.resolve()
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() == suffix
            )
    local: dict[str, str] = {}
    unknown: set[str] = set()
    for path in sorted(paths, key=lambda value: str(value).casefold()):
        digest = sha1_file(path)
        if digest in known:
            local.setdefault(digest, str(path))
        else:
            unknown.add(digest)
    missing = known - local.keys()
    result = FactoryVerification(
        bundle_available=True,
        factory_directories_found=directory_count,
        local_files_found=len(paths),
        known_bundle_hashes=len(known),
        matched_hashes=len(local),
        missing_hashes=tuple(sorted(missing)),
        unknown_local_hashes=tuple(sorted(unknown)),
        local_paths_by_hash=local,
        elapsed_s=time.monotonic() - started,
    )
    if mapping_path is not None:
        mapping_path = Path(mapping_path).expanduser().resolve()
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_path.write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8"
        )
    return result
