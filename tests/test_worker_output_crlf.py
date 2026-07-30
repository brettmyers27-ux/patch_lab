"""Guards for Windows CRLF worker output — every background job was broken.

Windows' text-mode stdout translates every printed '\n' to '\r\n'. Every
_read_output() in app/workers.py split only on bare '\n', leaving a trailing
'\r' attached to the popped line. That corrupted the exact-string worker-ready
handshake, producing "<worker> worker returned the wrong startup handshake
(<worker>)" — the two names print identically because the extra '\r' is
invisible, and this affected scan, render, analyze, match, export, and preview
uniformly since they all shared the same vulnerable pattern. It never
reproduced on macOS, where line endings are bare '\n'.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.workers import MatchProcessRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WorkerCrlfHandshakeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_crlf_handshake_is_accepted(self) -> None:
        runner = MatchProcessRunner()
        runner._worker_name = "match"
        failures: list[str] = []
        runner.failed.connect(failures.append)

        runner._buffer = "PATCHLAB_WORKER_READY=match\r\n"
        while (line := runner._pop_line()) is not None:
            runner._handle_worker_line(line)

        self.assertEqual(failures, [])
        self.assertTrue(runner._worker_ready)

    def test_crlf_progress_payload_has_no_stray_carriage_return(self) -> None:
        """A trailing \\r leaking into a captured value (a result path, an
        error string, a JSON payload) is a second-order symptom of the same
        bug — verify the popped line itself is clean, not just the handshake.
        """

        runner = MatchProcessRunner()
        runner._buffer = 'MATCH_RESULT=/tmp/result.json\r\n'
        line = runner._pop_line()
        self.assertEqual(line, "MATCH_RESULT=/tmp/result.json")
        self.assertNotIn("\r", line)

    def test_pop_line_is_a_noop_on_unix_line_endings(self) -> None:
        runner = MatchProcessRunner()
        runner._buffer = "PATCHLAB_WORKER_READY=match\n"
        self.assertEqual(runner._pop_line(), "PATCHLAB_WORKER_READY=match")

    def test_all_read_output_implementations_use_the_shared_helper(self) -> None:
        source = (PROJECT_ROOT / "app" / "workers.py").read_text(encoding="utf-8")
        # Exactly one occurrence is expected: inside _pop_line() itself, the
        # single place allowed to split the raw buffer. Any _read_output that
        # splits again directly has reintroduced the CRLF bug for that worker.
        self.assertEqual(
            source.count('self._buffer.split("\\n", 1)'),
            1,
            "a _read_output implementation is splitting the buffer directly "
            "again instead of going through _pop_line(), which reintroduces "
            "the CRLF handshake bug for that one worker type",
        )
        self.assertGreaterEqual(
            source.count("while (line := self._pop_line())"),
            6,
            "expected all six worker output readers (scan, render, analyze, "
            "match, export, preview) to share the CRLF-safe line reader",
        )


if __name__ == "__main__":
    unittest.main()
