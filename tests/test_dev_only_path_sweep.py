"""Bounded static sweep: no GUI-reachable worker may fall back to a dev-only
checkout-relative path (``core.db.DEFAULT_DB_PATH``, ``core.render.DEFAULT_AUDIO_ROOT``,
``core.serum2_state_reconstruct.DEFAULT_RENDER_STATE_DIR``) once running in
distribution mode.

Phase 1 found this exact bug class in generated-preset saving; Phase 2 found
it again in Render Sound Library. This is not a full audit -- it targets the
one remaining call site the sweep found still capable of reaching a
dev-only default from a real GUI action, and pins down that it stays gated.
"""

from __future__ import annotations

from pathlib import Path

from core.db import DEFAULT_DB_PATH
from core.render import DEFAULT_AUDIO_ROOT


def _source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_the_scan_worker_uses_dev_defaults_only_outside_distribution_mode() -> None:
    """app/ui.py's one remaining bare-scan call site: guarded by distribution_mode."""

    source = _source("app/ui.py")
    call = source.split("self.runner.start(Path(selected), local_library=")[1][:40]
    assert call.startswith("self.distribution_mode)"), (
        "the plain 'scan' worker (which falls back to core.db.DEFAULT_DB_PATH, "
        "a checkout-relative path) must only run when NOT in distribution mode -- "
        "a packaged app has no such checkout to fall back to"
    )


def test_render_library_worker_never_uses_the_dev_only_constants_directly() -> None:
    """Phase 2's fix: scripts/render_library.py resolves its own paths, not argparse defaults."""

    source = _source("scripts/render_library.py")
    assert "default=DEFAULT_DB_PATH" not in source
    assert "default=DEFAULT_AUDIO_ROOT" not in source
    assert "_resolve_local_paths" in source


def test_render_library_core_function_is_reached_with_dev_defaults_only_from_one_place() -> None:
    """core.render.render_library()'s own DEFAULT_DB_PATH/DEFAULT_AUDIO_ROOT
    defaults must only ever be reachable via the already-guarded CLI wrapper,
    never called bare from anywhere else GUI-reachable."""

    local_source = _source("core/local_library.py")
    assert "render_function=render_library" in local_source
    assert "def _process_linked_folder(" in local_source
    signature = local_source.split("def _process_linked_folder(")[1].split(") -> ")[0]
    assert "db_path: Path,\n" in signature or "db_path: Path," in signature
    assert "= DEFAULT_DB_PATH" not in signature
    assert "= DEFAULT_AUDIO_ROOT" not in signature

    # Phase 3 routes rendering through the crash-safe lifecycle. The injected
    # renderer still receives every runtime path explicitly.
    preparation_source = _source("core/preparation.py")
    render_call = preparation_source.split("rendered = render_function(", 1)[1].split(
        ")", 1
    )[0]
    assert "db_path=database.path" in render_call
    assert "audio_root=jobs_root" in render_call
    assert "state_dir=state_dir" in render_call


def test_factory_bundle_default_resolves_from_the_runtime_family_not_a_checkout_path() -> None:
    """core.factory_bundle.DEFAULT_FACTORY_BUNDLE: the constant every other
    GUI-reachable default (core/factory_match.py, core/factory_verify.py,
    core/workflow_state.py, core/local_library.py) inherits its safety from."""

    source = _source("core/factory_bundle.py")
    assert "DEFAULT_FACTORY_BUNDLE = runtime_data_root()" in source
