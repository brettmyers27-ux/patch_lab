#!/usr/bin/env python3
"""Minimal frozen-app launcher for the portable preparation diagnostic."""
from __future__ import annotations

import argparse
import subprocess
import sys

from core.preparation_diagnostic import run_collector


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--noninteractive", action="store_true")
    args, _unknown = parser.parse_known_args()
    output, error = run_collector()
    if args.noninteractive:
        return 0 if output is not None else 1
    if output is None:
        _show_message("PatchLab Preparation Diagnostic", error, error=True)
        return 1
    _show_message(
        "PatchLab Preparation Diagnostic",
        "Diagnostic complete. Send this file to PatchLab support:\n\n" + str(output),
    )
    return 0


def _show_message(title: str, message: str, *, error: bool = False) -> None:
    """Use the operating system's native, dependency-free message surface."""
    if sys.platform == "darwin":
        escaped_title = title.replace('"', '\\"')
        escaped_message = message.replace('"', '\\"')
        command = f'display alert "{escaped_title}" message "{escaped_message}"'
        subprocess.run(["/usr/bin/osascript", "-e", command], check=False)
        return
    # Windows PyInstaller distributions carry Tcl/Tk from their build runtime;
    # it is deliberately imported only on Windows so macOS stays stdlib-only.
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    (messagebox.showerror if error else messagebox.showinfo)(title, message, parent=root)


if __name__ == "__main__":
    raise SystemExit(main())
