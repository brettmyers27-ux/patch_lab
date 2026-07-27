"""Strict parser for Serum 2's documented XferJson preset container."""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cbor2
import zstandard


MAGIC = b"XferJson\x00"
CONTAINER_HEADER_SIZE = len(MAGIC) + 8
PAYLOAD_HEADER_SIZE = 8


class Serum2PresetError(ValueError):
    """Raised when a Serum 2 preset container is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class Serum2Preset:
    path: Path
    metadata: Any
    data: Any
    metadata_length: int
    cbor_length: int
    payload_version: int
    compressed_length: int


def parse_serum2_preset(path: Path) -> Serum2Preset:
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    minimum = CONTAINER_HEADER_SIZE + PAYLOAD_HEADER_SIZE
    if len(raw) < minimum:
        raise Serum2PresetError(f"File is too short: {len(raw)} bytes")
    if raw[: len(MAGIC)] != MAGIC:
        raise Serum2PresetError(f"Bad magic: {raw[:len(MAGIC)]!r}")

    offset = len(MAGIC)
    metadata_length = struct.unpack_from("<Q", raw, offset)[0]
    offset += 8
    metadata_end = offset + metadata_length
    if metadata_end + PAYLOAD_HEADER_SIZE > len(raw):
        raise Serum2PresetError(
            f"Metadata length {metadata_length} exceeds the {len(raw)}-byte container"
        )
    try:
        metadata = json.loads(raw[offset:metadata_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Serum2PresetError(f"Invalid metadata JSON: {exc}") from exc

    offset = metadata_end
    cbor_length, payload_version = struct.unpack_from("<II", raw, offset)
    offset += PAYLOAD_HEADER_SIZE
    compressed = raw[offset:]
    if not compressed:
        raise Serum2PresetError("Missing compressed CBOR payload")
    try:
        decompressed = zstandard.ZstdDecompressor().decompress(
            compressed, max_output_size=cbor_length
        )
    except zstandard.ZstdError as exc:
        raise Serum2PresetError(f"Zstandard decompression failed: {exc}") from exc
    if len(decompressed) != cbor_length:
        raise Serum2PresetError(
            f"CBOR length mismatch: header={cbor_length}, decoded={len(decompressed)}"
        )

    stream = io.BytesIO(decompressed)
    try:
        data = cbor2.CBORDecoder(stream).decode()
    except cbor2.CBORDecodeError as exc:
        raise Serum2PresetError(f"CBOR decode failed: {exc}") from exc
    trailing = stream.read()
    if trailing:
        raise Serum2PresetError(f"CBOR payload has {len(trailing)} trailing bytes")

    return Serum2Preset(
        path=path,
        metadata=metadata,
        data=data,
        metadata_length=metadata_length,
        cbor_length=cbor_length,
        payload_version=payload_version,
        compressed_length=len(compressed),
    )
