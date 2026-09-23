from __future__ import annotations

import pytest

from scripts.version_policy import (
    format_version,
    is_valid_transition,
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


# --- is_valid_transition: one release, many commits ------------------------


def test_leaving_the_released_baseline_requires_a_bump() -> None:
    # 1: main/base 1.5.5 -> branch 1.5.6 = allowed
    assert is_valid_transition((1, 5, 5), (1, 5, 6), base=(1, 5, 5))


def test_a_later_commit_may_stay_at_the_same_unreleased_version() -> None:
    # 2: another branch commit staying 1.5.6 = allowed
    assert is_valid_transition((1, 5, 6), (1, 5, 6), base=(1, 5, 5))


def test_a_version_may_never_move_backward() -> None:
    # 3: 1.5.6 -> 1.5.5 = rejected
    assert not is_valid_transition((1, 5, 6), (1, 5, 5), base=(1, 5, 5))


def test_staying_at_the_baseline_itself_is_rejected() -> None:
    # 4: branch never bumped above released 1.5.5 = rejected when preparing a release
    assert not is_valid_transition((1, 5, 5), (1, 5, 5), base=(1, 5, 5))


def test_a_skipped_step_is_rejected() -> None:
    assert not is_valid_transition((1, 5, 6), (1, 5, 8), base=(1, 5, 5))


def test_a_second_bump_before_shipping_is_still_allowed() -> None:
    # 6: future next release can move 1.5.6 -> 1.5.7 appropriately
    assert is_valid_transition((1, 5, 6), (1, 5, 7), base=(1, 5, 6))
    # ...including mid-branch, before that next baseline ever ships.
    assert is_valid_transition((1, 5, 6), (1, 5, 7), base=(1, 5, 5))


def test_an_unknown_baseline_falls_back_to_the_strict_rule() -> None:
    """No reachable released ref -> never silently accept "no change"."""

    assert not is_valid_transition((1, 5, 6), (1, 5, 6), base=None)
    assert is_valid_transition((1, 5, 6), (1, 5, 7), base=None)


def test_2_0_0_still_requires_explicit_approval_regardless_of_base() -> None:
    with pytest.raises(ValueError, match="explicit approval"):
        next_version((1, 9, 9))
    assert not is_valid_transition((1, 9, 9), (2, 0, 0), base=(1, 9, 8))
