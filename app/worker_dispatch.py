"""Early worker dispatch used by both frozen and development subprocesses."""

from __future__ import annotations

import importlib
import os
import sys
import time
import traceback

from core.worker_runtime import (
    WORKER_ENTRY_POINTS,
    WORKER_FLAG,
    WORKER_READY_PREFIX,
)


def run_worker(worker_name: str, arguments: list[str]) -> int:
    try:
        entry_point = WORKER_ENTRY_POINTS[worker_name]
    except KeyError:
        print(f"PATCHLAB_WORKER_ERROR=unknown worker: {worker_name}", flush=True)
        return 2

    # Test-only fault injection lets the packaged integration gate prove that
    # the parent kills a worker which never reaches the handshake.
    if delay := os.environ.get("PATCHLAB_WORKER_TEST_READY_DELAY_SECONDS"):
        time.sleep(max(0.0, float(delay)))

    # This must be emitted before importing the potentially heavy worker
    # module. The GUI's bounded startup timer waits for this exact sentinel.
    print(f"{WORKER_READY_PREFIX}{worker_name}", flush=True)
    try:
        module = importlib.import_module(entry_point.module)
        entry = getattr(module, entry_point.callable_name)
        sys.argv = [entry_point.module, *arguments]
        result = entry()
        return int(result or 0)
    except SystemExit as exc:
        return int(exc.code or 0)
    except BaseException as exc:
        print(
            f"PATCHLAB_WORKER_ERROR={type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        return 1


def frozen_worker_request(arguments: list[str]) -> tuple[str, list[str]] | None:
    """Parse the flag before QApplication and the main-window imports exist."""

    if not arguments or arguments[0] != WORKER_FLAG:
        return None
    if len(arguments) < 2:
        return "", []
    return arguments[1], arguments[2:]


def main() -> int:
    if len(sys.argv) < 2:
        print("PATCHLAB_WORKER_ERROR=worker name is required", flush=True)
        return 2
    return run_worker(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
