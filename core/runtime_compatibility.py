"""Fail-closed selection for the proven PatchLab V1 runtime family.

The CLAP checkpoint, retrieval indexes, factory bundle, and synthesis catalog
are one embedding world.  They must never be selected independently: a valid
file from a different generation is still an invalid runtime.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path


RUNTIME_FAMILY_ID = "v1-legacy-stock-clap"
CLAP_CHECKPOINT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_CHECKPOINT_BYTES = 2_352_471_003
CLAP_CHECKPOINT_SHA256 = (
    "fbaf3a305704890450ec4604fe9a1ab806515b68827ec44a73f500c06c042baf"
)


@dataclass(frozen=True, slots=True)
class RuntimeArtifact:
    relative_path: str
    sha256: str


# These are identities, not permissive minimum sizes.  Any replacement, even
# a newer-looking private artifact, must be deliberately adopted as a new
# family with its own review and manifest.
REQUIRED_ARTIFACTS = (
    RuntimeArtifact(
        "dist/factory_bundle.sqlite",
        "668b8613681e4ac503f5508e8f45a43e3895165e048312c5d57cd867c8c5624e",
    ),
    RuntimeArtifact(
        "features/preset_index.npy",
        "5036e5cabbb6bcc2a743b89f9c984d5c90ac4864587e6fe21cf9d9150611483d",
    ),
    RuntimeArtifact(
        "features/note_index.npy",
        "b6c489276ce56668611620efe191f0491502575b56d3b94d0f74802811fb5342",
    ),
    RuntimeArtifact(
        "features/similarity_manifest.npz",
        "3e1b38bf2c99987955c49f829352f4a0f12ec30f243c1e7131907293a1fbf7e0",
    ),
    RuntimeArtifact(
        "features/delta_neighbors.npz",
        "f4596571bc14923f3aebfc87f3484d6d44bb792a33758c2df2831acddbd0a0cc",
    ),
    RuntimeArtifact(
        "features/serum2_targets.npz",
        "3d97d80fb6c37e2791f97e759e83a52a4659e39c7cf84eeb955ffcaf22a6aa6d",
    ),
    RuntimeArtifact(
        "models/patchlab-synthesis-catalog.sqlite",
        "661d5387e17c3650b7ed4f30bbac2cac9056ccf17e93320df4e3421f6f52ef16",
    ),
    RuntimeArtifact(
        "models/serum2_target_schema.json",
        "024ea1f736b5b4899434ff7fb6df0d74f0923458f005d3590482910677ab5e06",
    ),
    RuntimeArtifact(
        "models/serum2_render_state_manifest.json",
        "29f426b9185b8949d83919e3c52cb3728163b344d5f25dcbbdd5144f3b2ed471",
    ),
    RuntimeArtifact(
        f"models/{CLAP_CHECKPOINT_NAME}",
        "fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd",
    ),
)
MIN_RENDER_STATE_FILES = 710
STATE_CHECKSUMS_NAME = "runtime-sha256s.txt"
STATE_CHECKSUMS_SHA256 = "7045ecc8203a551aa064773f859b38d3e40357291eed7fcbf1fabcdb78777b0c"
_VALIDATED_SIGNATURES: dict[Path, tuple[tuple[str, int, int], ...]] = {}


class RuntimeCompatibilityError(RuntimeError):
    """Raised when no complete, approved runtime family is present."""


def runtime_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(str(frozen_root)).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def runtime_data_root() -> Path:
    """The isolated root for the only V1 runtime family currently approved."""

    return runtime_root() / "data" / "runtime" / RUNTIME_FAMILY_ID


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _state_checksums(path: Path) -> dict[str, str]:
    """Read the transferred, hash-pinned identities of all render states."""

    checksums: dict[str, str] = {}
    pattern = re.compile(
        r"^([0-9a-f]{64})\s+\./artifacts/data/models/serum2_render_states/(.+\.vstpreset)$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            checksums[match.group(2)] = match.group(1)
    return checksums


def validate_runtime_family() -> Path:
    """Return the approved root, or explain why matching is unavailable.

    This intentionally has no environment override.  Overrides are useful for
    experiments, but allowing one in a production resolver would reintroduce
    the cross-family mixing this boundary exists to prevent.
    """

    root = runtime_data_root()
    problems: list[str] = []
    signature: list[tuple[str, int, int]] = []
    for artifact in REQUIRED_ARTIFACTS:
        path = root / artifact.relative_path
        if not path.is_file():
            problems.append(f"missing {artifact.relative_path}")
            continue
        stat = path.stat()
        signature.append((artifact.relative_path, stat.st_size, stat.st_mtime_ns))
        if (
            artifact.relative_path.endswith(CLAP_CHECKPOINT_NAME)
            and stat.st_size != CLAP_CHECKPOINT_BYTES
        ):
            problems.append(f"wrong size for {artifact.relative_path}")
            continue
    states = root / "models" / "serum2_render_states"
    state_files = sorted(states.glob("*.vstpreset")) if states.is_dir() else []
    count = len(state_files)
    if states.is_dir():
        signature.extend(
            (
                f"models/serum2_render_states/{item.name}",
                item.stat().st_size,
                item.stat().st_mtime_ns,
            )
            for item in state_files
        )
    if count != MIN_RENDER_STATE_FILES:
        problems.append(
            "wrong Serum 2 render-state count "
            f"({count}; expected {MIN_RENDER_STATE_FILES})"
        )
    checksum_file = root / STATE_CHECKSUMS_NAME
    if not checksum_file.is_file() or _sha256(checksum_file) != STATE_CHECKSUMS_SHA256:
        problems.append(f"missing or invalid {STATE_CHECKSUMS_NAME}")
    else:
        checksums = _state_checksums(checksum_file)
        if len(checksums) != MIN_RENDER_STATE_FILES:
            problems.append(
                f"invalid render-state checksum manifest ({len(checksums)} entries)"
            )
    signature.append(
        (STATE_CHECKSUMS_NAME, *(
            (checksum_file.stat().st_size, checksum_file.stat().st_mtime_ns)
            if checksum_file.is_file() else (0, 0)
        ))
    )
    if problems:
        raise RuntimeCompatibilityError(
            "PatchLab's approved V1 runtime family "
            f"{RUNTIME_FAMILY_ID!r} is unavailable or mixed:\n• "
            + "\n• ".join(problems)
            + "\nRun the supported PatchLab installer after the operator publishes "
            "the matching private runtime family."
        )
    resolved_signature = tuple(signature)
    if _VALIDATED_SIGNATURES.get(root) == resolved_signature:
        return root
    # Hashing all members is intentionally delayed until after the cheap
    # signature check.  A process may open the factory bundle repeatedly, but
    # a changed path, byte size, timestamp, or state count always invalidates
    # the cache and forces the exact-hash verification below.
    for artifact in REQUIRED_ARTIFACTS:
        path = root / artifact.relative_path
        if _sha256(path) != artifact.sha256:
            raise RuntimeCompatibilityError(
                "PatchLab's approved V1 runtime family "
                f"{RUNTIME_FAMILY_ID!r} changed during validation: "
                f"hash mismatch for {artifact.relative_path}"
            )
    checksums = _state_checksums(checksum_file)
    for state in state_files:
        if checksums.get(state.name) != _sha256(state):
            raise RuntimeCompatibilityError(
                "PatchLab's approved V1 runtime family "
                f"{RUNTIME_FAMILY_ID!r} changed during validation: "
                f"hash mismatch for models/serum2_render_states/{state.name}"
            )
    _VALIDATED_SIGNATURES[root] = resolved_signature
    return root
