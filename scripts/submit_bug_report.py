#!/usr/bin/env python3
"""QProcess entry point for a user-approved PatchLab bug report.

Two independent stages:

A. the local bundle (always written first, never depends on the network), and
B. one upload of that already-saved bundle to the private support service.

Stage B failing can never affect stage A; a retry re-sends the same saved file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.bug_report import load_request


def _ensure_bundle(request) -> Path | None:
    """Guarantee the structured bundle exists on disk before any upload.

    The GUI normally writes it when the report is created, but this must not
    depend on that: a failed or unreachable support service can never be allowed
    to leave the user without local evidence. ``create_support_bundle`` writes to
    a deterministic per-ticket directory, so calling it again is idempotent
    rather than duplicating anything.
    """

    try:
        from core.support_bundle import SUMMARY_FILENAME, create_support_bundle

        from core.bug_report import reports_root
        from core.support_bundle import BUNDLE_DIRNAME_SUFFIX

        directory = (
            reports_root()
            / f"PatchLab Bug Report {request.ticket_id}{BUNDLE_DIRNAME_SUFFIX}"
        )
        if (directory / SUMMARY_FILENAME).is_file():
            print(f"BUG_REPORT_BUNDLE={directory}", flush=True)
            return directory
        bundle = create_support_bundle(
            ticket_id=request.ticket_id,
            comments=request.comments,
            directory=directory,
            ticket_path=request.report_path,
        )
        print(f"BUG_REPORT_BUNDLE={bundle.directory}", flush=True)
        if bundle.fingerprint:
            print(f"BUG_REPORT_FINGERPRINT={bundle.fingerprint}", flush=True)
        return bundle.directory
    except Exception as exc:
        # Never fail the report over the bundle; the readable ticket already
        # exists and the upload attempt should still proceed.
        print(f"BUG_REPORT_BUNDLE_ERROR={type(exc).__name__}: {exc}", flush=True)
        return None


def _archive_for_upload(request, directory: Path | None) -> Path:
    """The one file to upload: the bundle archive saved earlier, reused as-is."""

    from core.support_bundle import archive_bundle_directory

    if directory is not None:
        existing = directory.with_suffix(".zip")
        if existing.is_file():
            return existing
        rebuilt = archive_bundle_directory(directory)
        if rebuilt is not None:
            return rebuilt
    raise FileNotFoundError("the saved diagnostic bundle could not be found")


def _fail(code: str, message: str, *, attempts: int = 0) -> int:
    print(f"BUG_REPORT_ERROR_CODE={code}", flush=True)
    print(f"BUG_REPORT_ATTEMPTS={attempts}", flush=True)
    print(f"BUG_REPORT_ERROR={message}", flush=True)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument(
        "--bundle-only",
        action="store_true",
        help="Write the local diagnostic bundle and skip the upload attempt.",
    )
    args = parser.parse_args()
    try:
        request = load_request(args.request)
    except Exception as exc:
        return _fail("unreadable_report", f"The saved report could not be read ({type(exc).__name__}).")
    # Stage A: local evidence first, unconditionally, before the network is touched.
    directory = _ensure_bundle(request)
    if args.bundle_only:
        print("BUG_REPORT_RESULT=" + json.dumps({"uploaded": False}), flush=True)
        return 0

    # Stage B: upload the saved bundle.
    from app.__version__ import __version__
    from core.submission_upload import BUG_REPORT, UploadError, resolve_relay, upload_file

    relay, reason = resolve_relay()
    if relay is None:
        code = "not_connected"
        print(f"BUG_REPORT_CONNECTION={reason}", flush=True)
        return _fail(code, "PatchLab isn't signed in to the support service on this Mac.")
    try:
        artifact = _archive_for_upload(request, directory)
        result = upload_file(
            relay,
            kind=BUG_REPORT,
            submission_id=request.ticket_id,
            path=artifact,
            version=__version__,
            progress=lambda text: print(text, flush=True),
        )
    except UploadError as exc:
        return _fail(exc.code, exc.user_message, attempts=exc.attempts)
    except FileNotFoundError:
        return _fail("file_missing", "The saved diagnostic file could not be found.")
    except Exception as exc:
        return _fail("unknown", f"The upload didn't complete ({type(exc).__name__}).")
    payload = {"ticket_id": request.ticket_id, **result.as_dict()}
    print(
        "BUG_REPORT_RESULT=" + json.dumps(payload, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
