# PatchLab — Accuracy Roadmap (Master Reference)

This is the authoritative reference for all remaining accuracy work. It exists so that
**any capable model (Opus/Sonnet) can author the next stage's Codex prompt from this file
alone, without loss of intent.** Read the whole file before writing any stage prompt.

---

## 1. Project facts (verified as of 2026-08-03)

- Repo: `~/Documents/PatchLab/soundmatch`, venv at `.venv`, run app via `.venv/bin/python app/main.py`.
- Rendering: headless VST hosting via DawDreamer (+ Pedalboard for state work). **Never**
  automates a live DAW/plugin session. This constraint is permanent.
- Library: 5,579 presets (Serum 1: 4,869; Serum 2: 710). Milestone 2 rendered a 7-note set
  (C1–C7, 5–8 s, 44.1 kHz stereo float32) for 5,515 presets; 64 marked `failed_silent`.
  39,053 render rows, ~71 GiB, report at `data/models/milestone2_full_render_report.json`.
- Matching: CLAP embeddings + a trained parameter-prediction model (Milestone 3), then
  search/refinement. Serum 1 params fully mapped via the VST automation surface.
  Serum 2: ~40% weighted parameter coverage via mapped automation
  (`core/serum2_mapping.py`); FX topology and mod-matrix source/dest are **structurally
  unreachable** through automation (confirmed by exhaustive surface investigation,
  `data/models/serum2_surface_investigation.json`). Serum 2 rendering uses reconstructed
  two-chunk `.vstpreset` states (`core/serum2_state_reconstruct.py`,
  `data/models/serum2_render_states/`), which the DSP engine honors even though host
  automation readback stays at init.
- Serum 2 preset files are fully *decoded* (XferJson/zstd/CBOR, `core/serum2_preset.py`) —
  we can read every setting including the mod matrix; we just can't push all of it through
  the automation surface.
- Current Windows Stage 2B ladder (99 user-authorized BAM files): the adopted fine-tuned
  encoder + complete v2 indexes + shipped predictors scores BAM mean **0.784226**, median
  0.793119, minimum 0.541550; factory retrieval@1 **0.785** and @5 0.895;
  rhythm/codec/pitch/loudness invariance retrieval@1 **0.403333** and @5 0.596667.
  The previous pinned production stack scored BAM mean 0.770395, retrieval@1 0.560, and
  invariance@1 0.256667. The fine-tuned + retrained-predictor alternative scored BAM mean
  0.780465 and retrieval@1 0.775, so the shipped predictors remain production.
  Historical macOS figures (~0.99 in-library, ~0.87 lightly disguised, ~0.82 forced
  rebuild, ~0.68 BAM) remain historical rather than current acceptance claims.
- UI: PySide6, 16:9 proportionally-scaled canvas (QGraphicsScene at 1920×1080 design size).
  UI gates: `scripts/verify_visual_redesign.py` and `scripts/verify_milestone4_ui.py` must
  keep passing after any change. (`scripts/verify_milestone7_ui_fixes.py` is stale/superseded.)

## 2. The master principle

**Transcribe, don't search.** Any property measurable directly from the target audio must
be measured and *written onto the preset as parameter values*; search budget is reserved
only for what cannot be measured. The user's founding insight: a one-shot quarter-note
bass stab and a sustained patch with a quarter-note LFO gate are the *same sound* — rhythm
is a parameter we control, not a timbre difference to be penalized.

## 3. Complete taxonomy of output-influencing factors

Every item below either ships in a stage or is explicitly documented as a limitation.
Nothing may be silently dropped.

### A. Input analysis (what the system hears)
1. **Pitch** — targets arrive at arbitrary notes; library renders are C1–C7 only;
   key-tracked filters make timbre note-dependent. Requires fundamental detection
   (incl. glide trajectory), pitch-normalized comparison, verification renders at the
   detected note.
2. **Loudness** — LUFS-normalize both sides before any embedding/feature computation.
3. **Silence/fades/DC** — trim to the usable region before analysis.
4. **Codec robustness** — targets are often MP3/rips; library is pristine WAV.
5. **Polyphony** — chord targets vs. mono renders: detect, reduce-to-root or flag honestly.
6. **Stereo width** — mid/side ratio and correlation are measurable and map to
   unison/detune/width parameters (transcribable).
7. **Baked-in production FX** — reverb tails, delay throws, and sidechain pumping are not
   patch properties. Pump is amplitude modulation (same family as the LFO insight).
   Separate "body" from "tail"; treat pump as rhythm.
8. **Tempo** — detect BPM so measured modulation rates become synced divisions
   (e.g. 2.67 Hz → 1/8 @ 160 BPM) written to synced LFOs.

### B. Comparison objective (how sounds are scored)
9. **Timbre/rhythm decomposition** — the LFO insight. Amplitude envelope, gate rate, and
   ADSR are extracted data, not similarity penalties.
10. **Transient vs. sustain** — separate windows for attack character and body.
11. **Harmonic vs. noise** — HPSS split; noise-component matched on its own axis.
12. **Evolving timbre** — wavetable sweeps / filter envelopes are timbre *trajectories*;
    compare frame-wise; transcribe as env→cutoff / env→WT-position routings.
13. **Pitch trajectory** — glides/808 slides → portamento/pitch-env settings.
14. **Composite scoring** — visible sub-scores (timbre, envelope, noisiness, brightness,
    width), never one opaque cosine.
15. **Best-window alignment** — compare against the best-matching render segment, never
    wall-to-wall.
16. **Perceptual weighting** — equal-loudness/log-frequency weighting of spectral features.

### C. Search structure (how the budget is spent)
17. **Wavetable timbre index** — pre-render and fingerprint every factory wavetable on a
    neutral patch; direct lookup replaces "inherit from seed."
18. **Staged optimization** — source/wavetable → filter/FX → *written* transcriptions →
    small joint polish. Never one flat search.
19. **Parameter sensitivity** — concentrate budget on audibly-impactful parameters.
20. **Dry-first FX** — match dry timbre before FX parameters enter.

### D. Retrieval (upstream of everything)
21. **Invariant retrieval** — nearest-neighbor must share every invariance from A/B
    (rhythm, pitch, loudness), else the true match is filtered out before optimization
    and nothing downstream can recover it.
22. **Velocity/dynamics** — single-velocity renders are a documented limitation.

### E. Truth & measurement
23. **Regression gates encoding the insights** — e.g. externally-gated known presets must
    retrieve their ungated originals and yield correct transcribed gate rates.
24. **Personal BAM benchmark + blind listening** — ears are the verdict; CLAP supports.
25. **Calibrated confidence** — sub-scores surfaced in results so a low score names its
    cause.

## 4. Stage plan

- **Stage 1 — Smarter ears** (Mac only, no new hardware): implements A1–A8, B9–B16,
  C17–C20, D21, E23–E25. Prompt: `docs/codex-stage1-smarter-ears.md`. Expected: the
  largest single jump on the real-sample benchmark; makes every later stage measurable.
- **Stage 2 — Smarter brain / Stage 2B completion** (completed on the user's RTX 5070 PC,
  2026-08-03): built 30,000 base clips and 390,000 prediction pairs, fine-tuned CLAP with
  synth-invariance positives, and retrained both predictors. Stage 2B supplied all missing
  Serum 1 presets, rendered all 5,579 presets / 39,053 note rows, and rebuilt the embedding
  world atomically. On the identical final gate, the fine-tuned encoder + v2 indexes +
  shipped predictors improved BAM mean 0.770395 → **0.784226**, factory retrieval@1
  0.560 → **0.785**, and invariance@1 0.256667 → **0.403333**, with no benchmark errors.
  That encoder/index stack is adopted; shipped predictors remain adopted, while both
  Stage 2 retrained predictors remain experimental and are not relay candidates.
- **Stage 3 — Our own algorithm**: (a) neural Serum surrogate (params → predicted audio
  embedding) enabling millions of virtual experiments per minute with real-synth
  verification of winners (research-frontier: cf. Sound2Synth/DiffMoog); (b) **layer
  decomposition** — match dominant layer, subtract, match residual, output a preset
  *stack*; this is what closes the gap on dense production sounds.
- **Throughout**: vocal-chop targets always asymptote lower — always attempt, always
  output, score honestly. Serum 2's automation cap is structural; state it in reports
  rather than implying more data will fix it.

## 5. Conventions for authoring stage prompts (follow exactly)

Every Codex stage prompt must:
1. Open with a context recap: current accepted metrics and what this stage changes.
2. Reference this roadmap file and name the taxonomy items it implements.
3. Give numbered, concrete deliverables — name target modules/files.
4. Define verification gates with numeric before/after comparisons on the *same*
   benchmark sets used previously (never new-benchmark-only claims).
5. Include the honesty clause: no silent scope cuts; unimplementable items are reported
   as limitations with evidence, not dropped.
6. Require the existing UI gates and milestone verifiers to keep passing.
7. End with a stop-at-gate instruction: report all results before considering the work
   done, and stop at the final reporting gate rather than starting the next stage.
8. Never violate the no-live-DAW-automation constraint.
9. Prefer reusing existing verified pipelines (render library, reconstruction path,
   benchmark suites) over building parallel ones.
