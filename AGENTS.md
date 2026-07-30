# PatchLab agent instructions

PatchLab is a Python 3.11/PySide6 desktop matcher for licensed Serum and Serum
2 installations. It hosts the user's locally installed plug-ins through
DawDreamer. Serum itself is never distributed by this repository.

PatchLab is proprietary software distributed to a private trusted group. All
rights are reserved by Brett Myers. Do not copy, redistribute, publish,
sublicense, or create derivative or competing products from this source except
as expressly authorized by the repository license and first-launch agreement.

For a trusted group member, the supported installation path is the repository
root `install.sh` on macOS or `install.ps1` on Windows. Inspect it, then run it.
Do not replace it with manual venv instructions, a PyInstaller build, or an
attempt to reconstruct missing model files. The installer checks the machine,
creates or updates the checkout and venv, authenticates the member, downloads
verified artifacts, and creates the native launcher.

Installation requires:

- Apple Silicon macOS 12.3 or newer, or 64-bit Windows 11;
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

Keep the source repository free of passcodes, tokens, OAuth credentials,
private relay secrets, personal paths, preset files, rendered audio, databases,
and model or index artifacts. The relay URL is non-secret configuration; its
credentials are not.

Every source commit must advance `app/__version__.py` by exactly one PatchLab
version. Version components remain single digits: `1.0.8` advances to `1.0.9`,
then to `1.1.0`. Run `python scripts/bump_version.py` once before each commit.
Never set or automatically roll into `2.0.0`; only Brett Myers may explicitly
authorize that major version.
