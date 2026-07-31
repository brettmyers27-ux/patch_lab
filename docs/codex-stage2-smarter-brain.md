# Codex Prompt — Stage 2: Smarter Brain (run ON the Windows 11 / RTX 5070 PC)

Copy everything below this line into Codex **on the Windows PC**, from the repo root
(`%USERPROFILE%\Documents\PatchLab\soundmatch`), after `git pull`.

---

Read `docs/ACCURACY_ROADMAP.md` in full before doing anything. This prompt implements
**Stage 2** of that roadmap and follows its §5 conventions exactly: numeric before/after
gates on the *same* benchmark sets, no silent scope cuts, stop at the final reporting gate,
never automate a live DAW or plugin session, reuse existing verified pipelines.

## Context recap

Historical accepted ladder (measured on macOS, before recent changes): ~0.99 retrieval when
the answer is in the library; ~0.87 lightly-disguised; ~0.82 forced rebuild; **~0.68 mean on
the user's real BAM samples** — the number that matters.

**Those numbers are stale and must be re-measured before anything else.** Since they were
recorded: analysis-by-synthesis was actually wired into distribution mode (it previously
always returned the closest existing preset), the matcher was made asset-relative, adaptive
preprocessing/pitch hypotheses landed (Stage 1), Serum 2 export was fixed, and this Windows
machine became a first-class install. Stage 2's improvement claims are only meaningful
against a fresh baseline on this machine, on the same input audio.

**What Stage 2 changes** (roadmap §4): (a) fine-tune the CLAP audio encoder on synth-domain
audio with augmentations that *teach* the invariances — a gated/pitched/FX'd copy of a patch
is a positive for its original; (b) scale synthetic training data ~10× and retrain the
parameter-prediction and delta models on it. Taxonomy items this implements or directly
serves: **A1, A2, A4, A7, A8** (as augmentation transforms), **B9** (rhythm/timbre
decomposition, as the positive-pair definition), **D21** (invariant retrieval — the
fine-tuned encoder becomes the carrier of the invariances), **E23–E25** (gates below).

## Ground truth about THIS machine — verified, do not re-derive

- This is a **git-clone install**. `data/` is untracked, so the 71 GiB macOS render library
  and the full `data/library.db` are **not here and never will be**. What IS here, delivered
  as gated artifacts into the repo tree: `data/models/patchlab-synthesis-catalog.sqlite`
  (preset identity + full Serum 1 automation params, 5,600 presets),
  `data/models/serum2_render_states/` (710 `.vstpreset` templates),
  `data/features/{preset_index.npy, note_index.npy, similarity_manifest.npz,
  serum2_targets.npz}`, `data/models/{serum2_target_schema.json, param_model.pt,
  delta_param_model.pt, music_audioset_epoch_15_esc_90.14.pt}`,
  `data/dist/factory_bundle.sqlite` (1,167 factory presets incl. 512-d CLAP embeddings),
  and the HF tokenizer cache under `data/models/huggingface`.
- **Never assume `data/library.db` exists.** Resolution helpers already handle this:
  `core/synthesis_assets.resolve_synthesis_assets()` honors `PATCHLAB_DISTRIBUTION_MODE=1`
  (which selects the synthesis catalog) and per-path env knobs (`PATCHLAB_LIBRARY_DB`,
  `PATCHLAB_FEATURE_DIR`, …). `core/model_assets.configure_model_environment()` sets the
  HF cache/offline env. Reuse both; do not invent parallel resolution.
- Rendering works here: DawDreamer hosts Serum 1 (VST2 `Serum_x64.dll`) and Serum 2 (VST3),
  4-process render pool, spawn context. The CRLF worker-output bug is fixed — worker lines
  are popped through `_ProcessRunnerBase._pop_line`. Keep it that way in any new worker.
- Version policy: every commit must advance `app/__version__.py` by exactly one patch step,
  each component a single digit (roll `x.y.9` → `x.(y+1).0`). The enforcing hook may not be
  installed on this machine — follow the policy manually anyway; macOS will reject pulls
  that violate it.
- **This PC is the sole writer for the duration of Stage 2.** Start with
  `git pull` on a clean tree; commit and push at each phase boundary. The Mac will not
  write until Stage 2 lands.
- Benchmark inputs: the user is copying the BAM sample folder from the Mac to
  `%USERPROFILE%\Documents\PatchLab\benchmarks\BAM` (~49 `.aif` files; ignore `.asd`).
  **These exact files are the before/after benchmark. If the folder is missing, stop and
  report — Stage 2 cannot be measured without it, and an unmeasured Stage 2 is a scope cut.**

## Phase 0 — Environment, parity, and inventory (hard gate)

1. `git pull`; confirm clean tree at or after commit `99bd028`, version ≥ 1.1.1.
2. **CUDA reality check.** The RTX 5070 is Blackwell (`sm_120`); a mismatched torch wheel
   imports fine and then dies at kernel launch. Run an actual bf16 matmul on the GPU and
   report `torch.__version__`, `torch.version.cuda`, device name, capability, and the
   matmul result device. If it fails, fix the torch install per
   `ENV.torch_install_command` (cu128 index) before proceeding.
3. **Plugin parameter parity — the silent-wrongness gate.** Run
   `scripts/verify_windows_install.py` and report the parameter-parity section verbatim:
   Serum 1 index→name signature vs `core/reference/macos_plugin_parity.json` must be
   **exactly equal** (316 params, matching sha), and the audio-parity CLAP cosines
   reported. Training data rendered here feeds models used on both platforms; a parameter
   mapping mismatch would poison everything downstream while sounding "fine". **Hard stop
   on any index/name mismatch — report, do not work around.**
4. Inventory and report: free disk (need ≥ 200 GiB; hard-stop below 120 GiB), delivered
   artifact presence (list above), benchmark folder present with file count, Serum 1/2
   versions.
5. Baseline the test suite on this machine (`pytest tests/ -q` with
   `PATCHLAB_DISTRIBUTION_MODE=1` so asset resolution matches the delivered layout).
   Record which tests, if any, fail *before* your changes — your final suite must have **no
   new failures**; pre-existing environment failures are reported, not fixed silently. The
   two UI screenshot gates (`verify_visual_redesign.py`, `verify_milestone4_ui.py`) need
   macOS fixture audio not present here; note them as macOS-only, to be run there on pull.

## Phase 1 — Fresh baselines (same inputs, current code)

Create `scripts/benchmark_suite.py` — deterministic, resumable, and the single scorer used
for every before/after claim in Stage 2:

1. **BAM real-sample suite**: for each `.aif` in the benchmark folder, run the existing
   match pipeline (`core/match_workflow.run_match_file`, budget `balanced`, serum2 unless
   the filename says otherwise) and record per-sample `clap_similarity`,
   `meaningfully_modified`, `evaluations`, elapsed. Report mean/median/min.
2. **Library retrieval gate**: for ≥ 200 randomly-seeded factory presets, embed their
   renders (render at C3 via the existing preview path if no audio exists) and confirm
   retrieval@1 of the preset itself from `preset_index`. Report the rate (~0.99 expected).
3. **Rhythm-invariance gate (E23)**: take ≥ 50 factory renders, apply an *external*
   amplitude gate (musical rates: 1/4, 1/8, 1/16 at 140–174 BPM) and pitch/loudness/codec
   variants, and measure retrieval@1 and @5 of the ungated original. This is the number
   Stage 2 most directly exists to move. Expect it to be mediocre at baseline — record it
   honestly.
4. Write machine-readable summaries to `docs/benchmarks/stage2-baseline.json` (tracked —
   `data/` is gitignored) plus per-sample detail under `data/stage2/` (untracked). Commit.

## Phase 2 — Synthetic training-data scale-up (streaming, disk-bounded)

Goal: ~**10×** the Milestone-3 corpus (~39k render-derived pairs → target ~390k pairs;
floor 5× if wall-clock demands, reported explicitly as a scope adjustment, never silently).

1. New `scripts/stage2_generate_training_data.py`. Sources of presets:
   - Serum 1: catalog presets + perturbations via `core/perturbation.perturb_serum1` and
     `scripts/generate_synthetic_serum1.py` patterns (full automation coverage — weight
     Serum 1 heavily);
   - Serum 2: base render-states + `perturb_serum2` within the ~40% mapped coverage
     (roadmap: the automation cap is structural — state it, don't fight it).
2. **Streaming pipeline — render → embed (current pinned CLAP) → handcrafted features →
   store (param vector, mask, embedding, features, synth, provenance) → delete the WAV.**
   Store as sharded npz/SQLite under `data/stage2/`. Audio for the *prediction* models is
   never kept; that is what makes 10× fit on disk.
3. **Keep a bounded raw-audio corpus for Phase 3**: ~25–35k clips, 3–4 s, 48 kHz, spread
   across presets and notes (C1–C7), ≤ 60 GiB. These are the contrastive-learning bases.
4. Resumable by provenance key (preset hash + perturbation seed + note); progress logging
   with rate and ETA; a crashed run must continue, not restart. Respect the 4-worker render
   pool; never automate a live DAW.
5. Report real counts, rates, disk used.

## Phase 3 — Fine-tune the CLAP audio encoder (the invariance teacher)

New `scripts/stage2_finetune_clap.py`:

1. **Positive-pair definition is the whole point — it encodes the founding insight**: for a
   given preset, positives are (a) its renders at *different notes* (A1), and (b) on-the-fly
   augmented copies of a render: external amplitude gating at musical rates and sidechain
   pump (B9/A7/A8), pitch shift ±2–12 semitones (A1), loudness jitter ±6 dB (A2), MP3/OGG
   round-trip (A4), reverb/delay tails (A7), EQ tilt, mild noise, stereo-width change,
   start-offset jitter. Negatives: other presets in-batch. Augment in the dataloader —
   never store augmented WAVs.
2. Architecture: load the pinned checkpoint via the existing laion_clap path
   (`core/features.ClapEmbedder` shows the loading conventions;
   `configure_model_environment()` first). Freeze the text tower entirely. Fine-tune the
   HTSAT audio tower **partially** — last N transformer blocks + projection (or LoRA) — to
   fit 12 GB VRAM with bf16 autocast, gradient accumulation as needed. Output must remain
   **512-d** and drop-in compatible with `cosine_topk` retrieval.
3. Loss: symmetric InfoNCE over audio-audio positive pairs; temperature learnable or ~0.07.
   Hold out ≥ 10% of presets entirely (not just clips) for validation.
4. **Mid-phase gate before anything downstream**: on the held-out presets, the fine-tuned
   encoder must (a) improve gated/augmented-copy retrieval@1 vs the pinned encoder, and
   (b) not degrade clean same-preset retrieval by more than 1 point. If it fails, iterate
   (fewer unfrozen blocks, lower LR, milder augmentation mix) and report each attempt; if it
   still fails after honest attempts, **keep the pinned encoder, say so, and proceed to
   Phase 5 with it** — a worse encoder must not ship because the phase "should" produce one.
5. Save as `data/models/patchlab_clap_ft_v1.pt`. **Never overwrite or rename the pinned
   `music_audioset_epoch_15_esc_90.14.pt`** — `model_assets.py` validates it and every
   existing index was built with it. Selection goes through the existing
   `PATCHLAB_CLAP_CHECKPOINT` env knob for A/B.

## Phase 4 — Rebuild the embedding world, atomically

An index built with one encoder is meaningless to another. If (and only if) Phase 3's gate
passed, rebuild **every** embedding artifact with the fine-tuned encoder, as one consistent
set under `data/stage2/artifacts-v2/`: `preset_index.npy`, `note_index.npy`,
`similarity_manifest.npz`, the factory bundle's `preset_embeddings`/`note_embeddings`
(reuse `scripts/build_similarity_index.py` and `scripts/build_factory_bundle.py` — extend
them to accept an encoder/checkpoint argument rather than duplicating them), and delta
neighbors (`scripts/build_delta_neighbors.py`). Embedding sources that the Mac's 71 GiB
render library previously provided must be re-rendered here as needed (stream, don't
hoard). **Mixing old and new embedding artifacts is forbidden — the A/B harness must load
each stack whole.**

## Phase 5 — Retrain the prediction models on the scaled corpus

1. Retrain `param_model` and `delta_param_model` on the Phase 2 corpus (embeddings from
   whichever encoder survived Phase 3's gate), reusing `core/train.py` /
   `core/delta_model.py` / `scripts/run_milestone3.py` conventions — GPU, bf16, same
   validation-split discipline (`_split_presets` seed conventions in `core/dataset.py`).
   Preset-level holdout, never clip-level.
2. Report old-vs-new validation metrics per synth, and training wall-clock/VRAM.
3. Save as versioned files under `data/stage2/artifacts-v2/`; do not overwrite the shipped
   models in place.

## Phase 6 — End-to-end A/B on the SAME benchmarks (the verdict)

Run `scripts/benchmark_suite.py` twice, changing nothing but the stack:

- **A**: pinned encoder + shipped indexes + shipped models (= Phase 1 numbers, re-used).
- **B**: Stage 2 stack (fine-tuned encoder if adopted, v2 indexes, retrained models),
  selected via env knobs only — no code forks.

Report per-sample BAM table (A vs B side by side), the three gate numbers, and:

**Adopt B only if**: BAM mean improves; library retrieval@1 does not drop below A;
rhythm-invariance retrieval@1 improves materially; no new test failures. Partial adoption
is allowed and must be stated (e.g., retrained predictor with pinned encoder). If B loses,
Stage 2's deliverable is the honest report and the reusable pipeline — say so plainly.

## Phase 7 — Handoff

1. Commit all code, `docs/benchmarks/*.json` summaries, and an updated
   `docs/ACCURACY_ROADMAP.md` (§1 facts + Stage 2 status with the new measured ladder).
   Push. Version policy on every commit.
2. **Do NOT commit model/index artifacts** (`data/` is ignored regardless) and do not
   touch the relay. Instead write `docs/benchmarks/stage2-artifact-manifest.json`: every
   adopted artifact's filename, byte size, SHA-256, and intended install destination. The
   Mac holds the relay credentials and will publish them from that manifest.
3. Final report: every phase's real numbers, everything not implemented as a named
   limitation with its reason, wall-clock per phase, and what Stage 3 should assume. Stop
   at this reporting gate — do not begin Stage 3 work.
