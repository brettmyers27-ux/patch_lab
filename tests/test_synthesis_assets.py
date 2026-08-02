"""Guards for the synthesis-vs-retrieval routing decision.

The distributed app spent several releases silently returning the closest
existing preset instead of generating a patch, because `factory_only` was wired
to `distribution_mode` rather than to whether synthesis was actually possible.
These tests pin the corrected behavior so that regression cannot return
unnoticed.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from core.synthesis_assets import (
    MIN_RENDER_STATE_FILES,
    _distribution_mode,
    resolve_synthesis_assets,
    synthesis_readiness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EnvironmentGuard(unittest.TestCase):
    """Restore every environment knob the resolver reads."""

    KEYS = (
        "PATCHLAB_DISTRIBUTION_MODE",
        "PATCHLAB_FEATURE_DIR",
        "PATCHLAB_SERUM2_SCHEMA",
        "PATCHLAB_SERUM2_RENDER_STATES",
        "PATCHLAB_LIBRARY_DB",
    )

    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in self.KEYS}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class SynthesisReadinessTest(EnvironmentGuard):
    def test_missing_render_states_disable_serum2_but_not_serum1(self) -> None:
        """Serum 1 needs no render-state templates, so it must stay available.

        Reporting one unavailable synth as both being unavailable would push a
        working Serum 1 install onto the retrieval fallback for no reason.
        """

        empty = Path(self.enterContext_tempdir())
        os.environ["PATCHLAB_SERUM2_RENDER_STATES"] = str(empty)
        self.assertFalse(synthesis_readiness("serum2").available)
        self.assertTrue(synthesis_readiness("serum1").available)

    def test_unavailable_readiness_names_what_is_missing(self) -> None:
        empty = Path(self.enterContext_tempdir())
        os.environ["PATCHLAB_SERUM2_RENDER_STATES"] = str(empty)
        readiness = synthesis_readiness("serum2")
        self.assertTrue(readiness.missing)
        # An operator has to be able to act on the message without reading code.
        self.assertIn("serum2_render_states", readiness.reason)

    def test_repository_checkout_can_synthesize_both_synths(self) -> None:
        for synth in ("serum1", "serum2"):
            self.assertTrue(
                synthesis_readiness(synth).available,
                f"{synth} synthesis should be available in a full checkout",
            )

    def test_distribution_mode_searches_shipped_and_user_state_roots(self) -> None:
        """Factory states ship with the install; user states are generated.

        Both populations must be renderable, so more than one root has to be
        searched in distribution mode.
        """

        os.environ["PATCHLAB_DISTRIBUTION_MODE"] = "1"
        os.environ.pop("PATCHLAB_SERUM2_RENDER_STATES", None)
        assets = resolve_synthesis_assets()
        self.assertGreater(len(assets.render_state_roots), 1)

    def test_find_render_state_resolves_a_known_factory_preset(self) -> None:
        assets = resolve_synthesis_assets()
        available = sorted(assets.render_states.glob("*.vstpreset"))
        if len(available) < MIN_RENDER_STATE_FILES:
            self.skipTest("checkout has no Serum 2 render states")
        preset_id = int(available[0].stem)
        self.assertEqual(assets.find_render_state(preset_id), available[0])

    def test_find_render_state_returns_none_when_absent(self) -> None:
        self.assertIsNone(resolve_synthesis_assets().find_render_state(-1))

    def test_frozen_bootloader_implies_distribution_mode(self) -> None:
        os.environ.pop("PATCHLAB_DISTRIBUTION_MODE", None)
        with patch.object(sys, "frozen", True, create=True):
            self.assertTrue(_distribution_mode())

    def test_empty_library_database_is_not_synthesis_ready(self) -> None:
        empty = Path(self.enterContext_tempdir())
        database = empty / "library.db"
        database.touch()
        os.environ["PATCHLAB_LIBRARY_DB"] = str(database)
        readiness = synthesis_readiness("serum2")
        self.assertFalse(readiness.available)
        self.assertIn("missing preset catalog", readiness.reason)

    def test_library_database_requires_at_least_one_preset(self) -> None:
        empty = Path(self.enterContext_tempdir())
        database = empty / "library.db"
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("CREATE TABLE presets (id INTEGER PRIMARY KEY)")
            connection.commit()
        os.environ["PATCHLAB_LIBRARY_DB"] = str(database)
        self.assertFalse(synthesis_readiness("serum2").available)

    def enterContext_tempdir(self) -> str:
        import tempfile

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return directory.name


class NoHardcodedRepositoryPathsTest(unittest.TestCase):
    """The matcher must not bind the source layout.

    A frozen build and a git-clone install resolve these locations
    differently. Reintroducing a repository-relative constant here is what made
    synthesis impossible outside a developer checkout, so fail loudly on it.
    """

    FORBIDDEN = (
        re.compile(r"PROJECT_ROOT\s*/\s*[\"']data[\"']\s*/\s*[\"']library\.db[\"']"),
        re.compile(r"FEATURE_DIR\s*/"),
        re.compile(r"\bSTATE_ROOT\b"),
        re.compile(r"\bDEFAULT_DB_PATH\b"),
    )

    def test_matcher_resolves_paths_through_synthesis_assets(self) -> None:
        source = (PROJECT_ROOT / "core" / "matcher.py").read_text(encoding="utf-8")
        for pattern in self.FORBIDDEN:
            self.assertIsNone(
                pattern.search(source),
                f"core/matcher.py must resolve paths via resolve_synthesis_assets(); "
                f"found a hardcoded {pattern.pattern!r}",
            )

    def test_parent_passes_one_resolved_asset_set_to_spawned_workers(self) -> None:
        source = (PROJECT_ROOT / "core" / "matcher.py").read_text(encoding="utf-8")
        self.assertIn("assets = resolve_synthesis_assets()", source)
        self.assertIn("initargs=(self._scratch.name, assets)", source)


class UiRoutesOnReadinessTest(unittest.TestCase):
    """`factory_only` must follow asset availability, not the mode flag."""

    def test_start_match_does_not_gate_on_distribution_mode_alone(self) -> None:
        source = (PROJECT_ROOT / "app" / "ui.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "factory_only=self.distribution_mode",
            source,
            "start_match must choose the synthesis path whenever synthesis "
            "assets are present; gating on distribution_mode alone silently "
            "returns the closest existing preset instead of a generated patch",
        )
        self.assertIn("synthesis_readiness", source)


class FrozenLibrosaPackagingTest(unittest.TestCase):
    """Frozen synthesis must load Librosa through a real source locator."""

    def test_spec_collects_librosa_as_source_modules(self) -> None:
        spec = (PROJECT_ROOT / "packaging" / "patchlab.spec").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'module_collection_mode={"librosa": "py"}',
            spec,
            "Loading librosa from PyInstaller's PYZ makes Numba cache=True "
            "kernels fail with 'no locator available' before synthesis starts",
        )

    def test_spec_builds_a_minimal_synthesis_catalog(self) -> None:
        spec = (PROJECT_ROOT / "packaging" / "patchlab.spec").read_text(
            encoding="utf-8"
        )
        self.assertIn("patchlab-synthesis-catalog.sqlite", spec)
        self.assertNotIn(
            '(str(ROOT / "data" / "library.db"),',
            spec,
            "The private developer database must not be copied wholesale",
        )


if __name__ == "__main__":
    unittest.main()
