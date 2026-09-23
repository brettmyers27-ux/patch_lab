"""PatchLab's deliberately bounded public-version sequence."""

from __future__ import annotations

import re
from pathlib import Path


VERSION_FILE = Path("app/__version__.py")
VERSION_PATTERN = re.compile(
    r'^__version__\s*=\s*"(?P<version>\d+\.\d+\.\d+)"\s*$',
    re.MULTILINE,
)


def parse_version_text(text: str) -> tuple[int, int, int]:
    """Return the single-digit public version stored in ``text``."""

    match = VERSION_PATTERN.search(text)
    if match is None:
        raise ValueError("app/__version__.py does not contain a valid __version__")
    parts = tuple(int(part) for part in match.group("version").split("."))
    if len(parts) != 3 or any(part < 0 or part > 9 for part in parts):
        raise ValueError("PatchLab version components must each be one digit (0-9)")
    return parts  # type: ignore[return-value]


def format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def next_version(version: tuple[int, int, int]) -> tuple[int, int, int]:
    """Advance one position without ever crossing into 2.0.0 automatically."""

    major, minor, patch = version
    if major != 1:
        raise ValueError(
            "Automatic versioning is locked to PatchLab 1.x; "
            "2.0.0 requires Brett Myers' explicit approval"
        )
    if patch < 9:
        return major, minor, patch + 1
    if minor < 9:
        return major, minor + 1, 0
    raise ValueError(
        "The next version would be 2.0.0, which requires "
        "Brett Myers' explicit approval"
    )


def is_valid_transition(
    old: tuple[int, int, int],
    new: tuple[int, int, int],
    *,
    base: tuple[int, int, int] | None = None,
) -> bool:
    """Whether ``old -> new`` is an acceptable version change for one commit.

    A single release (say 1.5.6) is normally built across many commits, so
    once a branch has actually moved past the last *released* version
    (``base``, typically what origin/main ships), staying at that same
    version is fine -- most commits touch no version-worthy behavior change,
    or are additional work belonging to the same still-unpublished release.

    Advancing by exactly one step (:func:`next_version`) is always allowed,
    whether that is the commit that first leaves ``base`` or a later commit
    that decides the release needs to be renumbered again before it ships.

    Going backward, skipping a step, or staying at ``base`` itself (which
    would mean the branch never actually started preparing a new release) is
    never allowed. When ``base`` is unknown (no released reference could be
    found), the strict historical rule applies: every commit must advance --
    the safe default rather than silently accepting "no change".
    """

    if new == old:
        return base is not None and old != base
    try:
        return new == next_version(old)
    except ValueError:
        return False


def replace_version(text: str, version: tuple[int, int, int]) -> str:
    if VERSION_PATTERN.search(text) is None:
        raise ValueError("app/__version__.py does not contain a valid __version__")
    return VERSION_PATTERN.sub(
        f'__version__ = "{format_version(version)}"',
        text,
        count=1,
    )
