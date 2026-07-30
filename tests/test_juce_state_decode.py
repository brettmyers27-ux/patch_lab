from __future__ import annotations

from scripts.render_factory_preview import _decode_juce_memory_block


_ALPHABET = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"


def _encode_juce_memory_block(payload: bytes) -> str:
    characters: list[str] = []
    for bit_offset in range(0, len(payload) * 8, 6):
        value = 0
        for index in range(6):
            absolute = bit_offset + index
            if absolute < len(payload) * 8:
                value |= ((payload[absolute // 8] >> (absolute % 8)) & 1) << index
        characters.append(_ALPHABET[value])
    return f"{len(payload)}." + "".join(characters)


def test_decodes_juce_vst3_memory_block_encoding() -> None:
    payload = b"XferJson\\0component-state\\x00\\xff"

    assert _decode_juce_memory_block(_encode_juce_memory_block(payload)) == payload
