"""Minimal FXP container parsing and writing.

Serum owns the opaque state payload. This module deliberately understands only
the standard big-endian FXP container header.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


FXP_HEADER_SIZE = 60


class FxpError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FxpChunk:
    byte_size: int
    version: int
    plugin_id: bytes
    plugin_version: int
    num_programs: int
    program_name: str
    payload: bytes


def parse_fxp_bytes(data: bytes) -> FxpChunk:
    if len(data) < FXP_HEADER_SIZE:
        raise FxpError(f"FXP is only {len(data)} bytes; expected at least {FXP_HEADER_SIZE}.")
    if data[:4] != b"CcnK":
        raise FxpError(f"Bad FXP magic {data[:4]!r}; expected b'CcnK'.")
    if data[8:12] != b"FPCh":
        raise FxpError(f"Unsupported FXP type {data[8:12]!r}; expected chunk preset b'FPCh'.")

    byte_size, version = struct.unpack_from(">II", data, 4)[0], struct.unpack_from(">I", data, 12)[0]
    plugin_id = data[16:20]
    plugin_version, num_programs = struct.unpack_from(">II", data, 20)
    program_name = data[28:56].split(b"\0", 1)[0].decode("latin-1", errors="replace")
    (chunk_size,) = struct.unpack_from(">I", data, 56)
    payload_end = FXP_HEADER_SIZE + chunk_size
    if payload_end > len(data):
        raise FxpError(f"FXP declares {chunk_size} payload bytes but only {len(data) - 60} remain.")
    return FxpChunk(
        byte_size=byte_size,
        version=version,
        plugin_id=plugin_id,
        plugin_version=plugin_version,
        num_programs=num_programs,
        program_name=program_name,
        payload=data[FXP_HEADER_SIZE:payload_end],
    )


def parse_fxp(path: Path) -> FxpChunk:
    return parse_fxp_bytes(path.read_bytes())


def build_fxp(
    payload: bytes,
    *,
    plugin_id: bytes,
    plugin_version: int = 1,
    program_name: str = "PatchLab",
) -> bytes:
    if len(plugin_id) != 4:
        raise FxpError("A VST2 plugin ID must contain exactly four bytes.")
    name = program_name.encode("ascii", errors="replace")[:27].ljust(28, b"\0")
    body = b"FPCh" + struct.pack(">I4sII", 1, plugin_id, plugin_version, 1) + name
    body += struct.pack(">I", len(payload)) + payload
    return b"CcnK" + struct.pack(">I", len(body)) + body
