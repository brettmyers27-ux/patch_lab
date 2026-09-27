#!/usr/bin/env python3
"""Frozen launcher for the follow-up preparation-error diagnostic."""
from __future__ import annotations

import subprocess
import sys

from core.preparation_error_diagnostic import run_collector


def _show_message(title: str, message: str, *, error: bool = False) -> None:
    if sys.platform == "darwin":
        command = f'display alert "{title.replace(chr(34), chr(92) + chr(34))}" message "{message.replace(chr(34), chr(92) + chr(34))}"'
        subprocess.run(["/usr/bin/osascript", "-e", command], check=False)
        return
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    (messagebox.showerror if error else messagebox.showinfo)(title, message, parent=root)


def main() -> int:
    output, error = run_collector()
    if output is None:
        _show_message("PatchLab Preparation Error Diagnostic", error, error=True)
        return 1
    _show_message("PatchLab Preparation Error Diagnostic", "Diagnostic complete. Send this file to PatchLab support:\n\n" + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
