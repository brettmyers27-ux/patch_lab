#!/bin/bash
# PatchLab self-update helper.
#
# Spawned detached by the running app when the user chooses "Update Now" in
# the update-available dialog. Not meant to be run by hand under normal use.
#
# Safety property this depends on: install.sh only ever writes inside the
# git checkout (source code) and refreshes shipped model artifacts through
# scripts/install_support.py -- it never touches the per-user Application
# Support directory where library.db, rendered audio, and learned
# fingerprints live. An update therefore cannot invalidate a library someone
# has already linked and analyzed; only PatchLab's own code changes.

set -eu

PARENT_PID="${1:?parent PID is required}"
INSTALL_ROOT="${2:?install root is required}"
APP_PATH="${3:?app bundle path is required}"

LOG="$(dirname "$INSTALL_ROOT")/patchlab-update.log"

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG"
}

log "Update starting; waiting for PatchLab (pid $PARENT_PID) to exit..."
WAITED=0
while kill -0 "$PARENT_PID" 2>/dev/null; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [ "$WAITED" -ge 30 ]; then
        log "PatchLab did not exit within 30s; proceeding anyway."
        break
    fi
done

log "Running the installer against $INSTALL_ROOT..."
if PATCHLAB_INSTALL_ROOT="$INSTALL_ROOT" PATCHLAB_INSTALL_APPLICATIONS=skip \
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/brettmyers27-ux/patch_lab/main/install.sh)" \
    >> "$LOG" 2>&1
then
    log "Update finished successfully."
else
    log "Update failed; PatchLab was left at its previous version."
fi

log "Relaunching PatchLab..."
open "$APP_PATH" >> "$LOG" 2>&1 || log "Could not relaunch automatically; open PatchLab from Finder."
