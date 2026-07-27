#!/bin/bash
# PatchLab trusted-group installer for Apple Silicon macOS.
#
# Safe to rerun: source updates are fast-forward only, dependencies use a
# checksum marker, and every large download resumes through a .part file.

set -eu

REPO_URL="${PATCHLAB_REPO_URL:-https://github.com/brettmyers27-ux/patch_lab.git}"
RELAY_URL="${PATCHLAB_RELAY_URL:-https://patchlab-relay-482507024870.us-central1.run.app}"
INSTALL_ROOT="${PATCHLAB_INSTALL_ROOT:-$HOME/Documents/PatchLab/soundmatch}"
USER_APPLICATIONS="${PATCHLAB_USER_APPLICATIONS:-$HOME/Applications}"
APP_PATH="$USER_APPLICATIONS/PatchLab.app"
PYTHON_BIN="${PATCHLAB_PYTHON:-python3.11}"

say() {
    printf '%s\n' "$*"
}

fail() {
    printf 'PatchLab install error: %s\n' "$*" >&2
    exit 1
}

case "$INSTALL_ROOT" in
    ""|"/") fail "The install root must be a specific non-root directory." ;;
esac
case "$USER_APPLICATIONS" in
    ""|"/") fail "The Applications destination must be a specific non-root directory." ;;
esac

command -v uname >/dev/null 2>&1 || fail "uname is required."
[ "$(uname -s)" = "Darwin" ] || fail "The supported installer currently requires macOS. Serum has no native Linux plug-in build."
[ "$(uname -m)" = "arm64" ] || fail "PatchLab currently requires an Apple Silicon Mac (arm64)."

OS_VERSION="$(sw_vers -productVersion 2>/dev/null || true)"
[ -n "$OS_VERSION" ] || fail "Could not determine the macOS version."
OS_MAJOR="$(printf '%s' "$OS_VERSION" | cut -d. -f1)"
OS_MINOR="$(printf '%s' "$OS_VERSION" | cut -d. -f2)"
if [ "$OS_MAJOR" -lt 12 ] || { [ "$OS_MAJOR" -eq 12 ] && [ "$OS_MINOR" -lt 3 ]; }; then
    fail "PatchLab requires macOS 12.3 or newer; this Mac reports $OS_VERSION."
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python 3.11 is missing. Install Python 3.11, then rerun this installer."
PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
PYTHON_SERIES="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "$PYTHON_SERIES" = "3.11" ] || fail "The virtual environment must use Python 3.11.x; $PYTHON_BIN is $PYTHON_VERSION."
command -v git >/dev/null 2>&1 || fail "git is missing. Install Apple's Command Line Tools with: xcode-select --install"

FREE_KB="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
REQUIRED_KB=$((8 * 1024 * 1024))
[ "${FREE_KB:-0}" -ge "$REQUIRED_KB" ] || fail "At least 8 GB free is required; only $((FREE_KB / 1024 / 1024)) GB is available."

SERUM_FOUND=0
for plugin in \
    "/Library/Audio/Plug-Ins/VST/Serum.vst" \
    "/Library/Audio/Plug-Ins/VST3/Serum.vst3" \
    "/Library/Audio/Plug-Ins/VST3/Serum2.vst3" \
    "/Library/Audio/Plug-Ins/Components/Serum.component" \
    "/Library/Audio/Plug-Ins/Components/Serum2.component" \
    "$HOME/Library/Audio/Plug-Ins/VST3/Serum.vst3" \
    "$HOME/Library/Audio/Plug-Ins/VST3/Serum2.vst3" \
    "$HOME/Library/Audio/Plug-Ins/Components/Serum.component" \
    "$HOME/Library/Audio/Plug-Ins/Components/Serum2.component"
do
    if [ -e "$plugin" ]; then
        SERUM_FOUND=1
        break
    fi
done
[ "$SERUM_FOUND" -eq 1 ] || fail "A licensed Serum or Serum 2 installation was not found in the system or user Audio/Plug-Ins folders."

say "PatchLab preflight passed: macOS $OS_VERSION, arm64, Python $PYTHON_VERSION, at least 8 GB free, Serum found."

if [ -e "$INSTALL_ROOT" ]; then
    [ -d "$INSTALL_ROOT/.git" ] || fail "$INSTALL_ROOT exists but is not a Git checkout. It was left untouched."
    say "Updating existing PatchLab checkout (fast-forward only)..."
    git -C "$INSTALL_ROOT" pull --ff-only || fail "The checkout has local/divergent changes. Resolve them before rerunning."
else
    say "Cloning the public PatchLab source..."
    mkdir -p "$(dirname "$INSTALL_ROOT")"
    git clone "$REPO_URL" "$INSTALL_ROOT" || fail "Could not clone the public repository."
fi

cd "$INSTALL_ROOT"
if [ ! -x ".venv/bin/python" ]; then
    say "Creating the Python 3.11 environment..."
    "$PYTHON_BIN" -m venv .venv || fail "Could not create the virtual environment."
fi

DEPENDENCY_HASH="$(
    {
        shasum -a 256 requirements.txt
        printf '%s\n' "macos-torch-torchaudio-v1"
    } | shasum -a 256 | awk '{print $1}'
)"
DEPENDENCY_MARKER=".venv/.patchlab-dependencies-$DEPENDENCY_HASH"
if [ ! -f "$DEPENDENCY_MARKER" ]; then
    say "Installing PatchLab dependencies. This is the longest setup step..."
    .venv/bin/python -m pip install --upgrade pip || fail "pip could not update."
    .venv/bin/python -m pip install torch torchaudio || fail "PyTorch installation failed."
    .venv/bin/python -m pip install -r requirements.txt || fail "PatchLab dependency installation failed."
    rm -f .venv/.patchlab-dependencies-*
    : > "$DEPENDENCY_MARKER"
else
    say "Dependencies already match this checkout; skipping installation."
fi

export PATCHLAB_RELAY_URL="$RELAY_URL"
if .venv/bin/python scripts/install_support.py auth-status >/dev/null 2>&1; then
    say "Existing group authentication found; the app will not prompt again."
else
    if [ "${PATCHLAB_INSTALL_TEST_MODE:-0}" = "1" ] && [ -n "${PATCHLAB_PASSCODE_FILE:-}" ]; then
        [ -r "$PATCHLAB_PASSCODE_FILE" ] || fail "The test passcode file is not readable."
        .venv/bin/python scripts/install_support.py auth --relay-url "$RELAY_URL" < "$PATCHLAB_PASSCODE_FILE" || exit 1
    else
        [ -r /dev/tty ] || fail "A terminal is required to enter the private-group passcode."
        printf 'Private-group passcode (input hidden): ' > /dev/tty
        IFS= read -r -s GROUP_PASSCODE < /dev/tty
        printf '\n' > /dev/tty
        printf '%s' "$GROUP_PASSCODE" | .venv/bin/python scripts/install_support.py auth --relay-url "$RELAY_URL" || exit 1
        GROUP_PASSCODE=""
        unset GROUP_PASSCODE
    fi
fi

.venv/bin/python scripts/install_support.py clap --install-root "$INSTALL_ROOT" || exit 1
CLAP_RUNTIME_MARKER="$INSTALL_ROOT/data/models/huggingface/.patchlab-clap-runtime-v1"
if [ ! -f "$CLAP_RUNTIME_MARKER" ]; then
    say "Preparing CLAP runtime files for offline first use..."
    .venv/bin/python scripts/cache_clap.py || fail "CLAP runtime preparation failed. Check the network connection and rerun; completed downloads will be reused."
    : > "$CLAP_RUNTIME_MARKER"
else
    say "CLAP runtime files already prepared; skipping model initialization."
fi
.venv/bin/python scripts/install_support.py artifacts --relay-url "$RELAY_URL" --install-root "$INSTALL_ROOT" || exit 1

say "Installing the Finder launcher..."
LAUNCHER_TMP="$USER_APPLICATIONS/.PatchLab.app.installing"
rm -rf "$LAUNCHER_TMP"
mkdir -p "$LAUNCHER_TMP/Contents/MacOS" "$LAUNCHER_TMP/Contents/Resources"
cp app/icons/PatchLab.icns "$LAUNCHER_TMP/Contents/Resources/PatchLab.icns"
VERSION="$(.venv/bin/python -c 'from app.__version__ import __version__; print(__version__)')"
cat > "$LAUNCHER_TMP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>PatchLab</string>
  <key>CFBundleIdentifier</key><string>com.patchlab.desktop</string>
  <key>CFBundleName</key><string>PatchLab</string>
  <key>CFBundleDisplayName</key><string>PatchLab</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleIconFile</key><string>PatchLab.icns</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOF
printf -v RELAY_URL_Q '%q' "$RELAY_URL"
printf -v INSTALL_ROOT_Q '%q' "$INSTALL_ROOT"
printf -v PYTHON_PATH_Q '%q' "$INSTALL_ROOT/.venv/bin/python"
printf -v APP_MAIN_Q '%q' "$INSTALL_ROOT/app/main.py"
printf -v MODEL_CACHE_Q '%q' "$INSTALL_ROOT/data/models/huggingface"
cat > "$LAUNCHER_TMP/Contents/MacOS/PatchLab" <<EOF
#!/bin/bash
export PATCHLAB_DISTRIBUTION_MODE=1
export PATCHLAB_RELAY_URL=$RELAY_URL_Q
export PATCHLAB_MODEL_CACHE=$MODEL_CACHE_Q
cd $INSTALL_ROOT_Q
exec $PYTHON_PATH_Q $APP_MAIN_Q
EOF
chmod 755 "$LAUNCHER_TMP/Contents/MacOS/PatchLab"
rm -rf "$APP_PATH"
mv "$LAUNCHER_TMP" "$APP_PATH"

if [ "${PATCHLAB_INSTALL_APPLICATIONS:-ask}" = "copy" ]; then
    if [ -w /Applications ]; then
        rm -rf /Applications/PatchLab.app
        cp -R "$APP_PATH" /Applications/PatchLab.app
        say "Copied launcher to /Applications/PatchLab.app"
    else
        say "The installer cannot write to /Applications without administrator approval; drag $APP_PATH there in Finder."
    fi
elif [ "${PATCHLAB_INSTALL_APPLICATIONS:-ask}" = "ask" ] && [ -r /dev/tty ]; then
    printf 'Also copy PatchLab.app to /Applications? [y/N] ' > /dev/tty
    IFS= read -r COPY_REPLY < /dev/tty
    case "$COPY_REPLY" in
        y|Y|yes|YES)
            if [ -w /Applications ]; then
                rm -rf /Applications/PatchLab.app
                cp -R "$APP_PATH" /Applications/PatchLab.app
                say "Copied launcher to /Applications/PatchLab.app"
            else
                say "Administrator access is required; drag $APP_PATH to /Applications in Finder."
            fi
            ;;
    esac
fi

DISK_KB="$(du -sk "$INSTALL_ROOT" | awk '{print $1}')"
say ""
say "PatchLab installation complete."
say "  Source and models: $INSTALL_ROOT"
say "  Finder launcher:   $APP_PATH"
say "  Disk used:         $DISK_KB KiB"
say "Open PatchLab from Finder. If macOS warns that it is from an unidentified developer, right-click PatchLab, choose Open, and confirm once."
