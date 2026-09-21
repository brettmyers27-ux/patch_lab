# PatchLab deployment architecture

Where every piece of PatchLab actually lives, and the exact commands that move a
release into production. No secret values appear in this document.

## Source of truth

| What | Where |
|---|---|
| Application code | GitHub `brettmyers27-ux/patch_lab` (public) |
| Production stable branch | `main` — always the source a released installer was built from |
| V2 development branch | `v2-development` |
| Relay service code | GitHub `brettmyers27-ux/patchlab-relay` (**private**) |
| Website | hosted on the mini PC; it deploys from Git like any other machine |
| Release installers | Workspace Drive → **PatchLab Releases**, folder `1L39zAZ6rrJQ8ehC0XiBDc8OCuRhh3bag` |
| Bug reports, preset contributions, runtime/model artifacts | Workspace Drive → **PatchLab Contributions**, folder `13FIK87lMbZAvzdWIpbjghaKL_x6Ex7rl` |
| Secrets | Google Secret Manager, project `patchlab-relay` |
| Relay runtime | Cloud Run service `patchlab-relay`, region `us-central1` |

No physical machine is a source of truth. The Mac laptop (V1/release), the
Windows PC (V2 research, Windows builds) and the mini PC (website) all clone
from GitHub. Nothing important should sit unpushed on any of them.

## Runtime path

```
PatchLab desktop app
    │  HTTPS, group passcode → short-lived bearer token
    ▼
Cloud Run relay  (authenticates to Google server-side)
    │  Drive API as the Workspace identity
    ▼
Google Workspace Drive
```

Users never see Google. They never sign in to Google, never need Drive Desktop,
and no Google credential is ever shipped inside the app. The desktop app knows
only the relay URL and the artifact metadata the relay returns.

### Storage identity

Drive writes are performed by a delegated **user** OAuth grant for the Workspace
account that owns both folders. The client id, client secret and refresh token
live in Secret Manager (`patchlab-oauth-client-id`, `patchlab-oauth-client-secret`,
`patchlab-oauth-refresh-token`) and are injected into Cloud Run as environment
variables.

This matters: in Drive, uploaded files are owned by *the uploading account*, not
by the owner of the destination folder. When the grant belonged to a personal
Google account, every upload into these Workspace folders consumed that personal
account's 15 GB quota and eventually failed with
`The user's Drive storage quota has been exceeded`. Production must always
authenticate as the Workspace account.

Re-authorize (one-time, interactive, operator only):

```bash
cd patchlab-relay && python scripts/authorize_workspace_drive.py
```

It refuses any identity other than the expected Workspace account, proves it can
write to both folders, and writes the refresh token straight into Secret Manager
without printing it. Redeploy the relay afterwards.

## HTTP surface

| Route | Used by |
|---|---|
| `POST /auth` | every client, exchanges the group passcode for a token |
| `POST /submissions` | current clients: one `.zip`, `type=bug_report` or `preset_contribution` |
| `GET /artifacts` | the update check and the installer's artifact preflight |
| `GET /artifacts/{name}` | installer/runtime and release downloads, supports HTTP Range resume |
| `POST /upload`, `POST /check-hash`, `POST /bug-reports` | legacy routes kept for already-installed 1.5.3 clients |

Never remove a route an installed client still calls. Downloads stream in 1 MiB
chunks; responses over 32 MiB deliberately omit `Content-Length` so Cloud Run
uses chunked transfer. The service request timeout is 3600s so a multi-gigabyte
installer can stream in one request; a client that still times out resumes with
a Range request.

## Release flow (macOS)

1. Build and gate on the Mac, from a clean tree:
   `.venv/bin/python packaging/build_macos_pkg.py --output-dir <dir> --keep-app`
2. Run the release gates (unit + real-Serum, frozen app, PKG, relay).
3. Verify a real installation: `sudo bash release-artifacts/verify_real_install.sh`.
4. Push the exact source commit to `main`.
5. Publish, one command:

```bash
cd patchlab-relay
PATCHLAB_RELAY_PASSCODE=... python scripts/publish_release.py \
    --platform macos --version 1.5.4 \
    --artifact ~/Documents/PatchLab/release-artifacts/PatchLab-1.5.4-macOS.pkg \
    --sha256 <sha256 the gates verified>
```

That verifies the local artifact, uploads it into **PatchLab Releases**, checks
the size and MD5 Drive stored, updates `artifacts.json`, deploys Cloud Run,
verifies production serves and streams it, and only then deletes the superseded
installer *for that platform*. Releases holds at most one current installer per
platform; runtime/model artifacts stay in Contributions and are never touched.

6. Verify the live updater:
   `.venv/bin/python release-artifacts/verify_production_update.py`

## Release flow (Windows)

Identical, from the Windows machine, once the EXE has passed its own gates:

```bash
python scripts/publish_release.py --platform windows --version <version> \
    --artifact <path>\PatchLab-<version>-Windows.exe --sha256 <sha256>
```

macOS clients ignore `windows-package` catalog entries, and vice versa, so the
two platforms release independently.

## Recovery

| Situation | Action |
|---|---|
| Bad relay revision | `gcloud run services update-traffic patchlab-relay --region us-central1 --to-revisions <previous>=100` |
| Bad release published | re-run `publish_release.py` with the previous verified installer and SHA-256; it becomes current again |
| Drive quota exceeded | check the authenticated identity first — it is almost always the wrong Google account, not real exhaustion |
| Refresh token revoked/expired | re-run `scripts/authorize_workspace_drive.py`, then redeploy |
| Group passcode rotation | `patchlab-relay/scripts/rotate_passcode.sh` |
| Catalog wrong | `artifacts.json` in the relay repo is the source; fix, commit, deploy |

## Rules

- Never publish an installer whose source is not the exact commit on `main`.
- Never grant public/anonymous write access to a Drive folder; the relay is the
  only writer.
- Never put a Google credential, service-account key or passcode in either
  repository. Folder IDs and the public relay URL are not secrets.
- Never delete bug reports, preset contributions or runtime artifacts as part of
  a release.
