from __future__ import annotations

import pytest

from scripts.version_policy import (
    format_version,
    next_version,
    parse_version_text,
    replace_version,
)


def test_patch_version_advances_one_digit_at_a_time() -> None:
    assert next_version((1, 0, 0)) == (1, 0, 1)
    assert next_version((1, 0, 8)) == (1, 0, 9)


def test_patch_rolls_into_minor_without_two_digit_components() -> None:
    assert next_version((1, 0, 9)) == (1, 1, 0)
    assert next_version((1, 8, 9)) == (1, 9, 0)


def test_major_two_cannot_be_reached_automatically() -> None:
    with pytest.raises(ValueError, match="explicit approval"):
        next_version((1, 9, 9))
    with pytest.raises(ValueError, match="explicit approval"):
        next_version((2, 0, 0))


def test_version_file_parser_rejects_multi_digit_components() -> None:
    with pytest.raises(ValueError, match="one digit"):
        parse_version_text('__version__ = "1.0.10"\n')


def test_version_replacement_preserves_the_module() -> None:
    original = (
        '"""Single source of truth."""\n\n'
        '__version__ = "1.0.0"\n'
    )
    updated = replace_version(original, (1, 0, 1))
    assert format_version(parse_version_text(updated)) == "1.0.1"
    assert updated.startswith('"""Single source of truth."""')
