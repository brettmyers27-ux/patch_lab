# PatchLab

## Quick Start — trusted group install

Prerequisites: either an Apple Silicon Mac running macOS 12.3 or newer or a
64-bit Windows 11 PC, Python 3.11, git, at least 8 GB free, licensed Serum 1
VST2 and Serum 2 VST3 installations, and the private-group passcode.

### macOS

The recommended installation is to download and inspect the installer before
running it:

```bash
curl -O https://raw.githubusercontent.com/brettmyers27-ux/patch_lab/main/install.sh
less install.sh
bash install.sh
```

For a trusted member who wants the one-command form:

```bash
curl -fsSL https://raw.githubusercontent.com/brettmyers27-ux/patch_lab/main/install.sh | bash
```

### Windows 11

The recommended Windows form also downloads the installer for inspection
before it runs:

```powershell
irm https://raw.githubusercontent.com/brettmyers27-ux/patch_lab/main/install.ps1 -OutFile install.ps1
notepad .\install.ps1
.\install.ps1
```

For a trusted member who wants the one-command form:

```powershell
irm https://raw.githubusercontent.com/brettmyers27-ux/patch_lab/main/install.ps1 | iex
```

If local execution policy blocks the inspected script, change policy for this
PowerShell process only, then rerun it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install.ps1
```

The Windows installer creates a console-free Desktop shortcut and a Start Menu
shortcut. Both target the venv's `pythonw.exe` through the small
`app\windows_launcher.pyw` bootstrap, which sets distribution mode, the relay
URL, and the model cache without setting a permanent system environment
variable.

The first install downloads roughly 2–3 GB of Python packages from PyPI, the
2.35 GB public LAION-CLAP music checkpoint from Hugging Face, and 240 MB of
passcode-gated PatchLab models, index, and factory fingerprints. Expect roughly
5 GB of network transfer and 15–45 minutes on typical broadband. Downloads are
checksum-verified and resumable. The installer also caches CLAP's small runtime
metadata/tokenizer files so the first match does not need another network
request. The result is a small `PatchLab.app` launcher in `~/Applications`;
subsequent installer runs update safely and skip completed work. Windows uses
the same resumable files and installs to
`%USERPROFILE%\Documents\PatchLab\soundmatch`.

Immediately after authentication, the installer probes one byte from every
private artifact. An unavailable model or factory bundle therefore stops the
install before the 2.35 GB CLAP download begins. Transient relay/network
failures are retried with bounded exponential backoff; completed and partial
downloads are preserved so rerunning resumes rather than restarts.

Patch Lab is a cross-platform desktop application for cataloging, rendering,
learning, and matching Serum presets. Development is deliberately gate-driven:
the plugin host and real preset-state round trip must be proven on the target
machine before library ingestion is enabled.

It runs Serum headlessly through DawDreamer—never by automating a DAW—and
provides a PySide6 desktop workflow for scanning presets, rendering an audition
library, learning audio similarity, matching samples, and exporting verified
native presets. Serum and Serum 2 are commercial, user-supplied plugins and are
not part of this project.

## Quick start from source

PatchLab requires Python 3.11, a licensed local Serum installation, and enough
disk space to render your own preset library.

```bash
git clone https://github.com/brettmyers27-ux/patch_lab.git
cd patch_lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchaudio
pip install -r requirements.txt
python scripts/verify_env.py
python app/main.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Use CUDA 12.8 wheels
when an NVIDIA adapter is present:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

On a Windows PC without an NVIDIA adapter, use the CPU wheels instead:

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

`install.ps1` makes this selection automatically and prints the detected
adapter and reason before downloading anything.

The detailed macOS and Windows setup, plugin locations, and verification gates
appear below.

## What this repository does not include

This proprietary source repository intentionally excludes all runtime and
user-derived data. It does **not** distribute:

- the rendered audio library or match-history audio;
- `library.db` or any scanned preset database;
- trained parameter models, the similarity index, or the CLAP checkpoint cache;
- Serum presets, third-party commercial preset-pack content, wavetables, or
  samples;
- private relay credentials or server configuration.

A source checkout must build its own database, renders, features, and models
from presets the user owns. The packaged-app distribution path may later ship
separately licensed, non-source artifacts as release downloads, but none are
committed to Git.

## License

Copyright © 2026 Brett Myers. All rights reserved. PatchLab is proprietary
software distributed for limited personal use by authorized private-group
members, subject to the repository `LICENSE` and the agreement presented at
first launch. Third-party components retain their respective licenses.

Copies obtained while earlier versions were offered under the MIT License
remain governed by the rights granted with those copies. The current
proprietary terms apply going forward and do not purport to withdraw previously
granted rights.

## Current build status

Milestone 0 (environment and preset-loading spike) is implemented. On the
current Apple Silicon test machine, `verify_env.py` passes (with the permitted
CPU-fallback warning). Serum 1 passes through its VST2 build and DawDreamer's
native FXP loader. Serum 2 parameter dumping uses its decoded CBOR engine graph
and the live VST3 parameter API. Serum 2 rendering uses reconstructed two-chunk
VST3 state: although the host automation projection stays frozen at init, a
four-second audio gate proved all five fixtures distinct from init and from one
another. This strategy is therefore render-only and is not used for parameter
dumping.

Milestone 1 is also implemented and has passed against the real library on the
current machine. The scan found 6,790 files, deduplicated 1,190 copies by SHA-1,
and verified 4,869 unique Serum 1 presets with 316 exposed parameters each.
There were 21 individually recorded Serum 1 failures (0.43%): three load/state
failures and 18 silent C4 probes. All 710 unique Serum 2 presets are now
parameter-dumped and enabled for the normal pipeline.

Serum 2 Step 2a parameter calibration is complete. The live VST3 exposes 2,623
parameters. Patch Lab caches 535,993 normalized-value/display-text observations
in `data/models/serum2_param_calibration.json`, including refined boundaries for
stepped and enum controls. Subsequent runs validate and reuse this cache unless
`scripts/calibrate_serum2.py --force` is requested.

Serum 2 Step 2b component mapping is complete for the five accepted real-preset
fixtures. It maps 181 of 485 scalar `plainParams` values overall (37.32%), and
180 of 248 ordinary continuous/enum synth values (72.58%). All accepted matches
are explicit aliases; no fuzzy match is counted as coverage. The remaining
structural gaps are reported rather than discarded: the public parameter list
has 64 modulation amount/output pairs but no source/destination controls, has
generic FX value banks but no FX topology controls, and exposes neither
wavetable/sample selection nor embedded custom-wavetable data hooks. The full
machine-readable report is `data/models/serum2_mapping_report.json`.

Application order is dependency-aware. Patch Lab applies enable/mode/type/sync
selectors before dependent values for oscillator engines and warp, filters,
LFO timing, modulation routes, FX topology, and clip/arp components. Continuous
numeric inverse lookup can optionally refine the cached 201-point result against
the live plugin with bounded bisection to a normalized interval of `1e-4`.

Serum 2 Step 2c passes §6.1 on all five accepted fixtures. Step 3 has completed
across the full 710-preset catalog: 710 succeeded, zero failed, and every row
stores all 2,623 exposed live parameters. Mean per-preset CBOR application
coverage is 42.37%; weighted ordinary synth-control coverage is 37,431/47,881
(78.17%). The resumable run report is
`data/models/serum2_catalog_report.json`.

All 710 Serum 2 presets also have their complete decoded CBOR graph stored as
valid JSON in `serum2_full_settings`. These 710 records coexist with all
1,862,330 mapped parameter rows and are the authoritative Serum 2 settings and
Milestone 3 label source. The render-state cache contains 710 id-only
`.vstpreset` files built from those records.

Milestone 2's renderer is implemented with four explicit `spawn` workers,
worker-local plug-in instances, id-only float32 WAV paths, parent-only SQLite
writes, pause/cancel controls, and note-pair resume. Its mixed 50-preset gate
passed: 350/350 unique rows, durations from 5.0 to 8.0 seconds, zero load
failures, one correctly flagged silent Serum 1 octave, and three distinct Serum
2 sanity renders. A SIGTERM test resumed from 67 rows without duplicates.

The full Milestone 2 run is complete: 5,579 presets and all 39,053 required
note pairs are present (34,083 Serum 1 and 4,970 Serum 2). Serum 1 finished with
4,827 fully non-silent presets, 42 presets containing at least one silent
octave, and zero load failures. Serum 2 finished with 688 fully non-silent
presets, 22 presets containing at least one silent octave, and zero load
failures. The exhaustive audit found no missing/duplicate note sets, malformed
audio, path violations, out-of-range durations, unexpected silent rows, or
temporary files. The production resume pass took 4,213.18 seconds (1:10:13),
excluding the already completed 350-row gate. Its machine-readable report is
`data/models/milestone2_full_render_report.json`; rerun the audit with
`python scripts/verify_render_library.py`.

Milestone 3 is implemented as a resumable pipeline. Serum 2's authoritative
decoded settings become 4,792 conceptual fields with an invertible one-hot and
numeric 14,726-output target vector, a parallel presence mask, and a JSON
schema. The current 710-preset schema contains 379
oscillator, 40 filter, 44 envelope, 147 LFO, 1,088 modulation-matrix, 2,666 FX,
8 macro, and 420 global/other fields. Categorical values use stable integer
category tables expanded as one-hot slices; continuous values use documented
corpus min/max scaling. An encode/decode audit verifies every populated value
after each schema build.

Audio analysis stores normalized 512-dimensional LAION-CLAP embeddings and
nine standardized spectral/time-domain features at both note and preset level.
The pinned music checkpoint is
`data/models/music_audioset_epoch_15_esc_90.14.pt`. Both Serum versions share
one note index and one preset index; search is brute-force numpy cosine, with no
FAISS dependency. The parameter model is the specified 521 → 1024 → 1024 → 768
MLP with separate 316-value Serum 1 and one-hot-expanded Serum 2 heads. Its checkpoint
contains both mappings, the Serum 2 decoding schema, the feature standardizer,
the preset-wise split, and the CLAP checkpoint identity.

The completed gates are: 100.000% tie-aware self-retrieval (99.839% strict),
96.899% tie-aware octave generalization, Serum 1 validation MAE 0.078696 versus
0.175624 baseline (55.19% better), and Serum 2 validation MAE 0.013478 versus
0.017836 baseline (24.43% better). The 20-preset-per-synth audio round trip had
zero silent renders and mean CLAP cosine similarity 0.468671 for Serum 1 and
0.193193 for Serum 2.

Serum 2 prediction has an important fidelity limitation: decoded model output
contains setting leaves but not every private runtime/editor subtree required
by Serum's state loader. Predicted leaves are therefore overlaid onto a valid
live init state before rendering. Unsupported variable topology and embedded
content remain at init; the measured mean structural overlay coverage in the
round-trip sample is 1.35%. This is why the Serum 2 audio round-trip score is
reported separately and is substantially lower than its parameter-label MAE
would suggest. Full decoded settings from real presets remain authoritative
labels and the existing 710 render-state files remain the high-fidelity loading
path for owned presets.

Run the complete workflow from the GUI or from the project root:

```bash
python scripts/run_milestone3.py --deep-training
python scripts/verify_milestone3.py --require-deep
```

The Deep training option creates 20,000 deterministic Serum 1 C4 patches. It
keeps modulation routing at init, enforces audible source priors, retries rather
than storing silent examples, and never adds synthetic examples to validation.

### Analysis-by-synthesis upgrade

The matcher now keeps Milestone 3's excellent retrieval index and adds a
render-in-the-loop optimizer. It first estimates pitch and loudness, retrieves
five same-synth presets, and builds 51 deterministic starting candidates from
the exact presets, an auxiliary delta model, and local perturbations. Exact
owned-preset candidates reuse the canonical library WAV; changed candidates are
rendered by a four-process pool whose workers retain their own Serum 1 and Serum
2 instances. Search minimizes a 35% multi-resolution STFT / 65% CLAP objective
with diagonal CMA-ES (population 16, at most 300 evaluations). It adjusts only
continuous controls; stepped categories and Serum 2 topology stay attached to
the selected real preset. Every run returns the best candidate, runner-up, best
candidate excluding a known target, and a generation-by-generation objective
trace.

On-manifold training augmentation is complete for both synths: 20,000 accepted
perturbations per synth. Serum 1 discarded 0.304% of attempts and Serum 2
discarded 0.897%, both well below the 15% gate. Their RMS and spectral-centroid
distributions passed the plausibility comparison with the real library. The
Serum 2 perturbations retain a mean 90.47% structural overlay coverage.

The neighbor-relative delta model was evaluated honestly and did **not** beat
the accepted Milestone 3 absolute predictor. Serum 1 validation MAE was
0.083894 versus 0.078696 (6.61% worse); Serum 2 was 0.017278 versus 0.013478
(28.20% worse). It remains only an auxiliary candidate generator. The absolute
model and exact retrieved-preset seeds remain active, so this negative result
does not regress matching quality.

The final 20-preset-per-synth held-out gate passed:

| Synth | One-shot CLAP | Analysis-by-synthesis CLAP | Own preset wins | Mean best excluding target | Median time |
|---|---:|---:|---:|---:|---:|
| Serum 1 | 0.468671 | 0.992021 | 19/20 | 0.824097 | 10.99 s |
| Serum 2 | 0.193193 | 1.000000 | 20/20 | 0.791091 | 13.33 s |

The separate ten-file pitch/EQ/drive/chorus transformation set is
informational rather than a hard gate; all ten completed with mean CLAP cosine
0.873661. Full per-query traces and timings are stored in
`data/models/analysis_by_synthesis_gate_report.json`.

Generate or verify the upgrade from the project root with:

```bash
python scripts/generate_perturbations.py
python scripts/build_delta_neighbors.py
python scripts/train_delta_model.py
python scripts/run_matcher_gates.py
python scripts/verify_analysis_by_synthesis.py
```

### Milestone 4 — Match a Sound

The fourth section is now enabled after analysis artifacts exist. It accepts
WAV, MP3, FLAC, OGG, and AIFF through a file picker or drag-and-drop. Audio is
decoded with SoundFile or torchaudio; MP3 falls back to the bundled
`imageio-ffmpeg` executable when needed. Matching is intentionally scoped to
single-preset clips no longer than four seconds. Longer jams and layered
recordings are not treated as single-patch targets.

Query preprocessing is adaptive for short one-shots. Leading and trailing
silence are trimmed, with a 0.25-second comparison floor, and every candidate
is rendered with the same note-on duration and aligned to the same comparison
length as the query. Below one second, query and candidate are identically
zero-padded to one second before CLAP embedding. The objective is 65% STFT /
35% CLAP at 0.5 seconds and shorter, changes linearly, and reaches 35% STFT /
65% CLAP at 1.5 seconds and longer. This keeps transient and spectral-envelope
matching dominant for stabs while retaining CLAP's semantic strength for
longer clips.

Pitch seeding also adapts to ambiguous material. If pYIN confidence is below
0.85, pitch detection falls back, or at least 45% of spectral energy is below
100 Hz, the initial search is distributed over the detected note, neighboring
octaves, and a bass/unpitched genre prior. The measured objective—not the pitch
detector—chooses the winning note.

Matching runs in an isolated UI worker so the window stays responsive. The UI
shows completed evaluations and the current best similarity, and offers Quick
(51 evaluations), Balanced (300), and Best Quality (600) search budgets.
Recommendation generation defaults to Serum 2 and can be switched to Serum 1.
The ten closest owned presets always come from the combined cross-synth index,
independent of that selection. Each result includes its source path, similarity,
and an audition button. One shared C1–C7 octave selector controls every
closest-match play button and the generated recommendation audition, so
comparisons always use the same note.

The recommended preset shows its measured CLAP similarity, base preset,
elapsed time, the shared octave selector, and the native preset export action.
Detailed parameter counts are intentionally omitted from the compact result
surface.
Confidence labels are calibrated to the real-sample benchmark and the accepted
round-trip results: 0.90 and above is **High Match**, 0.80–0.90 is **Good
Match**, 0.65–0.80 is **Fair Match**, and below 0.65 is **Low Match / no
confident match**. Five seconds of silence short-circuits safely to “No
confident match.”

“Save as preset” produces native files, not JSON recipes:

- Serum 1 exact winners copy their source FXP. Optimized winners use
  DawDreamer’s verified `save_state(filepath)` VST2 state, then losslessly wrap
  its native `FBCh` chunk as a standard `FPCh` program preset. Reloaded
  parameters must match within `1e-4`.
- Serum 2 exact winners copy the original `.SerumPreset`. Optimized winners
  overlay schema-covered changes onto the authoritative complete base graph,
  retain private state and asset references, encode CBOR, zstd-compress it, and
  recompute Serum’s MD5-of-compressed-payload metadata hash.

Every generated export is first written into private temporary storage. It is
decoded/reloaded and reconstructed where applicable before it can be saved.
Structural failures remain fatal, but the render-to-preview CLAP comparison is
advisory because plug-in phase, note duration, release state, and referenced
assets can change that score without making the native preset invalid. After
the structural check passes, Patch Lab atomically publishes the preset to the
location chosen by the user and deletes the private temporary copy.

The accepted ten-file export gate covered two copied and three optimized files per synth:
mean reload/render CLAP was 0.957729 for Serum 1 and 0.969010 for Serum 2. All
Serum 2 fixtures retained valid base wavetable/sample references; reconstructed
state coverage was 97.10%–99.89%.

The real UI worker path also passed its upload gates: a raw render ranked its
own preset #1, a 96 kbps MP3 ranked it #1 (required top five), and silence
showed the low-confidence state without loading the matcher. Re-run the
persisted audit with:

```bash
python scripts/verify_milestone4.py
```

### Match Library and batch folders

Every completed match—including an honest no-confident-match result—is
automatically archived in the **Library** tab. The durable entry lives under
`data/match_library/<match_uid>/` in a developer build, or the platform
application-data folder in a distribution build. It contains a copied source
file, the generated winner/candidate data, a portable `result.json`, and C1–C7
preview WAVs cached on first audition. Database paths are relative to the
library root, so moving the PatchLab data directory does not invalidate saved
history. Nothing is pruned automatically; Delete in the Library is the only
removal path.

Library rows can replay the archived source, render or replay any generated
octave, export through the same mandatory round-trip verifier, and reopen the
complete result with Enter or a double-click. Batch results are grouped under
their batch name.

**Batch Folder…** processes supported files sequentially with the currently
selected quality and target synth. Subfolders are opt-in. Presets are written
to `<Serum preset folder>/PatchLab/<chosen name>/`, and existing files are
never overwritten. Cancellation waits for the in-flight match and verified
export to finish. Re-running the same source/destination skips completed audio
by SHA-1 content hash; per-file failures are logged and do not stop the batch.
To protect the single headless plugin pipeline, a second batch, a one-off
match, and unrelated exports are refused while a batch is active. Browsing and
cached audition playback remain available.

### Milestone 6 — private-group distribution

The distribution build starts from a preset-file-free factory fingerprint
bundle at `data/dist/factory_bundle.sqlite`. It contains 1,167 complete factory
presets, 8,169 per-note embeddings, preset embeddings, and full parameter/
settings labels. It is 75.39 MiB. It contains no rendered audio, original
preset file, or developer-owned preset. One classified factory preset was
excluded because its source row was silent and had no complete fingerprint.

On every distribution launch, Patch Lab hashes the locally installed factory
preset folders and builds a hash-to-path map. This check takes under a second
on the accepted macOS catalog and never renders or embeds. A missing or
different local factory file only disables that preset's audition/export; the
shipped fingerprint remains searchable. With no Serum factory folders at all,
the app still launches and clearly reports that local loading is unavailable.

The first distribution launch keeps three distinct decisions in order:

1. the required proprietary License Agreement, which must be accepted before
   the application can be used;
2. private-group authentication, when the installer has not already stored a
   valid credential;
3. a separate, optional preset-sharing consent choice after the main window
   opens.

License acceptance is stored with a UTC timestamp in PatchLab's existing access
state. It appears before authentication because the terms govern whether the
software may be used at all. Declining exits without reaching authentication or
the main window. Settings includes **View License Agreement** for later
read-only review; reviewing it does not reset or replace the acceptance record.

Agreeing to the separate preset-sharing choice enables “Link My Preset Folder”
and means both:

- every linked preset is processed locally and added to the user's searchable
  index; retained local WAVs support audition;
- non-factory preset files plus fingerprints/settings may be contributed to
  the private developer library. Audio is never contributed.

Disagreeing leaves instant factory matching enabled and visibly disables the
folder section. “Use & share my own presets” is the single persisted On/Off
setting for changing this later. Turning it off does not delete the user's
local data, but immediately excludes it from use and disables linking.

Local work always runs before the storage-side relay hash check. Shared-pool
dedup therefore never skips this user's scan, parameter extraction, rendering,
or embedding. The accepted two-preset gate processed one factory Serum 2 and
one non-factory Serum 1 preset into 14 local WAVs, uploaded only the
non-factory preset, found it in combined search, and uploaded nothing on the
second run.

Set `PATCHLAB_DISTRIBUTION_MODE=1` only in packaged trusted-group builds.
The normal developer checkout retains the existing four-step development
workflow. The packaged macOS app and Windows shortcuts enable distribution mode
automatically.

After license acceptance, a clean distribution profile asks for the
private-group passcode before showing the main window. It validates that
passcode through the relay's existing `/auth` endpoint and stores it in macOS
Keychain or Windows Credential Manager through `keyring`. If no system keychain
is available, PatchLab stores only a non-secret success marker and the
short-lived relay token—never the passcode in a JSON file. A previously
authenticated user can always launch offline. A first-time user whose relay is
unavailable can explicitly continue locally without sharing. Settings includes
**Sign out / forget passcode**; signing out preserves both the license
acceptance record and the separate preset-sharing choice.

The relay URL is supplied to the packaged app as `PATCHLAB_RELAY_URL`.
Developer and automated environments may still supply
`PATCHLAB_RELAY_PASSWORD`, but a normal packaged user enters it once in the
first-run dialog and the app retrieves it from the OS keychain thereafter.

The backend lives in the separate private `patchlab-relay` repository next to
this project. Contribution routes are `/auth`, `/check-hash`, and `/upload`;
its delegated OAuth credentials stay server-side. The same bearer
authentication protects `/artifacts` and ranged `/artifacts/{name}` downloads
for the licensing-controlled installer files. No artifact metadata or bytes
are available without a valid group token, and audio is never uploaded.

Match candidate waveforms remain in memory while optimization runs. Once a
match completes, its source, winning render, candidate state, result metadata,
and on-demand octave previews are copied into the durable Match Library.
Unselected process scratch remains temporary and is deleted after the pool
closes. Distribution startup also removes interrupted PatchLab match scratch
older than one hour.

Re-run the Milestone 6 gates with:

```bash
python scripts/verify_local_factory.py
python scripts/verify_milestone6_local.py
python scripts/audit_audio_lifecycle.py --run-cma
cd ../patchlab-relay
PYTHONPATH=. ../soundmatch/.venv/bin/python tests/test_relay.py
```

### Building the macOS app

The supported trusted-group delivery is now `install.sh` plus its lightweight
Finder launcher. The monolithic PyInstaller bundle is not distributed; its
spec remains available for a future signed and notarized release.

Packaging tools are intentionally separate from runtime dependencies:

```bash
pip install -r requirements-dev.txt
pyinstaller --clean --noconfirm packaging/patchlab.spec
```

This creates the one-folder bundle `dist/PatchLab.app`. The runtime hook
enables distribution mode without requiring a terminal or environment toggle.
The build embeds the factory fingerprint bundle and pinned CLAP checkpoint when
those private/local artifacts are present at their documented `data/` paths,
but neither artifact is committed to Git.

The build remains useful for package engineering, but it is currently 3.3 GB
and is not the tester delivery path. `install.sh` creates a few-kilobyte
launcher around the source venv instead. Neither launcher is Apple-notarized.
On first launch, macOS may show an “unidentified developer” warning;
right-click the app, choose **Open**, and confirm once.

### Fresh clone versus packaged app

A fresh source clone is deliberately small and is not a ready-to-use learned
matcher. In addition to Python dependencies and licensed local Serum plugins,
development use needs:

- `data/models/param_model.pt` and the generated parameter/schema mappings;
- the per-note and per-preset numpy similarity indexes and manifests;
- the pinned LAION-CLAP checkpoint/cache;
- either a newly scanned and rendered local preset library, or the separately
  distributed `data/dist/factory_bundle.sqlite` fingerprint bundle.

The shortest non-technical path is `install.sh` on macOS or `install.ps1` on
Windows. Each gets public dependencies from PyPI and Hugging Face, downloads
the four private artifacts through the authenticated relay, and creates the
native launcher/shortcuts. Cloning the repository without running the installer
remains the developer path and requires rebuilding or separately supplying all
artifacts above. The factory bundle and trained models are never public Release
assets or source-controlled files.

## Installer troubleshooting

| Problem | What to do |
|---|---|
| Wrong passcode | Rerun and enter the trusted-group passcode again. Input is hidden and is never logged. |
| Relay unreachable | Check the internet connection and rerun. Installation stops at authentication before large downloads or launcher creation. |
| Private artifact unavailable | The installer names the failing artifact before downloading CLAP. Retry later or contact the PatchLab operator; existing `.part` files and verified downloads are preserved. |
| Insufficient disk | Free enough space for the 8 GB preflight requirement, then rerun; verified and partial downloads are preserved. |
| Python 3.11 missing | Install Python 3.11 so `python3.11` is available. Other minor versions are intentionally refused. |
| Microsoft Store Python opens instead of Python | Install 64-bit Python 3.11 from python.org with “Add python.exe to PATH,” then disable the `python.exe`/`python3.exe` App execution aliases in Windows Settings. |
| Serum not found on macOS | Install licensed Serum 1 VST2 and Serum 2 VST3 builds in a standard system or user `Audio/Plug-Ins` folder. |
| Serum 1 VST2 not found on Windows | Install `Serum_x64.dll`, or correct `VSTPluginsPath` under `HKLM`/`HKCU\SOFTWARE\VST`. The installer prints every registry and common-folder location searched. |
| PowerShell blocks `install.ps1` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then rerun `.\install.ps1`. This does not change machine-wide policy. |
| Gatekeeper warning | Right-click `PatchLab.app`, choose **Open**, and confirm once. The lightweight launcher is unsigned. |
| Windows SmartScreen warning | Choose **More info**, confirm the script/source is PatchLab, then choose **Run anyway**. |

## Requirements

- 64-bit Windows 11 (CPU or supported NVIDIA GPU), or Apple Silicon macOS 12.3+
- Python 3.11 in a `.venv` at this project root
- Licensed local Serum plugin installations
- Three real Serum 1 presets and three real Serum 2 presets for the spike

Ableton Live is neither required nor used. Patch Lab hosts plugins headlessly in
its own process.

## Setup — macOS / Apple Silicon

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchaudio
pip install -r requirements.txt
python scripts/verify_env.py
python scripts/spike_preset_load.py /path/to/preset/root
```

The standard PyPI torch wheel includes MPS support. Patch Lab sets
`PYTORCH_ENABLE_MPS_FALLBACK=1`; if MPS is unavailable it warns and uses CPU.

## Setup — Windows 11

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts\verify_env.py
python scripts\spike_preset_load.py C:\path\to\preset\root
```

The CUDA 12.8 index is mandatory for an RTX 5070 (Blackwell, sm_120). On a PC
without an NVIDIA adapter, replace `cu128` with `cpu`. `platform_env.py` and
`install.ps1` detect this rather than assuming every Windows member owns an
NVIDIA GPU.

### Windows parity gate

The Windows installer runs the critical plug-in/audio parity subset before it
creates shortcuts. After installation, rerun the complete copy-pasteable
diagnostic with:

```powershell
& "$env:USERPROFILE\Documents\PatchLab\soundmatch\.venv\Scripts\python.exe" "$env:USERPROFILE\Documents\PatchLab\soundmatch\scripts\verify_windows_install.py"
```

It checks real Windows 11/x64 identity, Serum 1 VST2 and Serum 2 VST3 parameter
count/index/name/current-value equivalence against the committed macOS
reference, four known factory C4 renders against macOS CLAP embeddings,
Credential Manager persistence, once-only passcode/consent state, a real
factory match and native preset export, source/preview playback, both
`pythonw.exe` shortcuts, and a real Windows UI render. It writes
`windows-ui.png`, the full live parameter dumps, and a JSON report under the
Windows Patch Lab application-data `diagnostics` folder.

The source and macOS baseline are ready for this test, but Windows parity is
**unproven until this diagnostic passes on a licensed Windows 11 machine**.
Do not distribute the Windows installer to the group based only on macOS/static
checks: a plug-in parameter-index mismatch can produce valid-looking presets
that sound wrong.

Maintainers can recheck every direct Python pin against official PyPI metadata
with `python scripts/verify_windows_wheels.py`. The current gate includes
CPython 3.11 Windows x64 or platform-independent wheels for every pin; in
particular, the deliberate `dawdreamer==0.8.3` and `pedalboard==0.9.24` pins
both have `cp311-win_amd64` wheels.

## Milestone 0 gates

`verify_env.py` reports the selected platform branch, Python and dependency
versions, compute backend, every plugin and preset probe path, DawDreamer's live
plugin API surface, and non-silent init renders for both synth versions. It also
prints the exact torch installation command for the detected OS.

`spike_preset_load.py` takes exactly three samples of each format, prints Serum 2
container-byte findings, attempts S1–S5 in order, and accepts a load only when at
least five parameters change and a one-second C4 render exceeds -60 dBFS. The
script isolates native host probes so a crashing plugin is reported rather than
terminating the gate runner. It also verifies that different preset files
produce different vectors. A
fully documented Serum 2 strategy failure permits a Serum 1-only continuation;
Serum 1 remains mandatory.

### Current macOS strategy status

The verified system paths are:

- `/Library/Audio/Plug-Ins/VST/Serum.vst`
- `/Library/Audio/Plug-Ins/Components/Serum.component`
- `/Library/Audio/Plug-Ins/Components/Serum2.component`

S1 (native FXP loading through Serum 1 VST2) is attempted before VST3 state
injection and passes all §6.1 checks. S5 can host both AU instruments through
DawDreamer and Pedalboard when macOS Audio Component Registrar access is
available, but neither host can apply the preset files through its AU state API.
Serum 2 is instead enabled through the verified CBOR-to-live-parameter
reconstruction strategy, recorded as
`VST3/S2-cbor-parameter-reconstruction-v1`.

The partitioned-state experiment is reproducible with
`scripts/spike_serum2_partitioned_state.py`; its state and audio findings are in
`data/models/serum2_partitioned_state_report.json` and
`data/models/serum2_partitioned_audio_report.json`. It is accepted only as the
Serum 2 render source. The plugin exposes no usable VST3 program-list API in
either host; its lone `Bank` parameter selects only anonymous `Prog 1` through
`Prog 128` entries and does not load preset-folder contents.

## Four-step workflow

1. Select Preset Folder
2. Render Sound Library
3. Analyze & Learn
4. Match a Sound

Launch the Milestone 1 desktop shell from the project root with:

```bash
python app/main.py
```

The folder scan runs in an isolated process, reports progress and activity in
the window, and commits each preset independently. The persisted gate can be
rechecked without loading Serum:

```bash
python scripts/verify_scan.py
```

## Serum 2 — known limitations

Captured settings include exposed oscillator, filter, envelope, visible LFO,
macro, global, routing-level, and limited arp/clip automation controls. Values
are applied in dependency order and continuous numeric controls use calibrated
inverse lookup with optional live bisection.

Serum 2 exposes only three unlabeled FX value banks (`FX Main`, `FX Bus 1`, and
`FX Bus 2`, 16 controls each). It exposes no FX type/topology selector, and a
live sweep of all 104 enum-like controls produced no dynamic FX relabeling.
FX modules therefore cannot be reconstructed safely through the host-automation
fallback and are excluded there rather than guessed. They are retained in the
full Serum 2 settings target and reconstructed-state path.

Modulation source/destination assignment is not recoverable through standard
host automation. Serum exposes only `Mod N Amount` and `Mod N Out` for 64 slots;
those values can affect a route already present in the current state but cannot
create the preset's missing route. The host-parameter fallback and Serum 1
synthetic patches therefore exclude those assignments. Serum 2's authoritative
`serum2_full_settings` labels and reconstructed-state path do include all 64
source/destination pairs, so learned Serum 2 recipes are not limited by that
automation-surface gap.

Reconstructing native `Comp` and `Cont` state chunks recovers internal FX and
modulation state for rendering, but neither DawDreamer nor Pedalboard
synchronizes those changes back to the host automation vector. Render workers
therefore load the id-only reconstructed state, while settings display and
training labels use the full, invertible `serum2_full_settings` target vector;
the mapped live-parameter representation remains only a fallback.
Across all 710 cached render states, mean structural match is 98.48%. The main
remaining state gaps are custom tuning arrays, MIDI/arp editor metadata, and
LFO-point modulation assignments; only two unmatched FX leaves were observed,
and no `ModSlot` route was structurally unmatched.

Some engine-specific granular, multisample, spectral, arp/clip, and LFO mode
data likewise lacks a semantically named host control. The catalog report marks
these fields unsupported or dependency-blocked.

## Known wavetable limitation

Custom wavetable *content* is not a VST parameter. Recommendations can cover
wavetable position, warp, and all other exposed settings, but cannot identify
which third-party wavetable the user should import. Referenced custom wavetables
and samples must be installed on the machine doing the rendering.
