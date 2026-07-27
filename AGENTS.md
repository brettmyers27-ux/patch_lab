# PatchLab agent instructions

PatchLab is a Python 3.11/PySide6 desktop matcher for licensed Serum and Serum
2 installations. It hosts the user's locally installed plug-ins through
DawDreamer. Serum itself is never distributed by this repository.

For a trusted group member, the supported installation path is the repository
root `install.sh`. Inspect it, then run it. Do not replace it with manual venv
instructions, a PyInstaller build, or an attempt to reconstruct missing model
files. The installer checks the machine, creates or updates the checkout and
venv, authenticates the member, downloads verified artifacts, and creates the
Finder launcher.

Installation requires:

- Apple Silicon macOS 12.3 or newer;
- Python 3.11, git, at least 8 GB free, and a licensed Serum or Serum 2 plug-in;
- the trusted-group passcode, entered interactively and never logged.

The trained models, similarity index, and factory fingerprint bundle are
licensing-controlled artifacts. They are available only through the
passcode-authenticated private relay. Never commit, re-host, attach to a GitHub
Release, upload to another service, or replace their authenticated URLs with
public ones.

Everything below `data/` is private runtime or user-derived state. It may
contain model weights, factory settings capable of reconstructing presets,
personal preset files, renders, match history, and fingerprints. Never add any
`data/` content to Git, even when troubleshooting an installation.

Keep the public repository free of passcodes, tokens, OAuth credentials,
private relay secrets, personal paths, preset files, rendered audio, databases,
and model or index artifacts. The relay URL is public configuration; its
secrets are not.
