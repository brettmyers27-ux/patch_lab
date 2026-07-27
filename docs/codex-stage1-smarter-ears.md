# Codex Prompt — Stage 1: Smarter Ears (Perceptual Decomposition & Transcription-First Matching)

Copy everything below this line into Codex, in the `soundmatch` repo.

---

Read `docs/ACCURACY_ROADMAP.md` first — it is the authoritative reference for this work.
This stage implements taxonomy items A1–A8, B9–B16, C17–C20, D21, and E23–E25.

## Context

Current accepted metrics: ~0.99 in-library retrieval, ~0.87 disguised, ~0.82 forced
rebuild, **~0.68 average on the real BAM production samples** — that last number is the
one this stage exists to move. The core defect: the objective compares audio
holistically, so a preset with the exact right timbre but a different rhythm (LFO gate,
one-shot vs. sustain, sidechain pump) is punished for a difference that is actually a
*parameter we control*. The governing principle for everything below:

**Transcribe, don't search.** Any property measurable directly from the target audio is
measured and written onto the preset as parameter values. Search budget is reserved for
what cannot be measured.

Constraints that remain absolute: no live DAW/plugin automation (headless DawDreamer
rendering only); reuse the existing verified render library, Serum 2 reconstruction path,
and benchmark suites rather than building parallel pipelines; the existing UI gates
(`scripts/verify_visual_redesign.py`, `scripts/verify_milestone4_ui.py`) must still pass
when you finish.

## Part A — Target-audio analysis front-end

New module `core/audio_analysis.py` producing a single structured analysis dict per
target, persisted alongside `result.json` and logged. For every incoming target:

1. **Usable region**: trim leading/trailing silence, remove DC offset, detect fade-outs;
   record the analysis window actually used.
2. **Loudness**: integrated LUFS; define one normalization level (e.g. −18 LUFS) and
   apply it to *both* target audio and library/candidate renders before any embedding or
   feature computation, everywhere in the pipeline.
3. **Pitch**: fundamental detection (pYIN or equivalent — pick a well-maintained
   dependency, pin it) with confidence; full pitch trajectory so glides/slides are
   captured (start pitch, end pitch, glide time); mono vs. polyphonic classification via
   chroma/multi-f0 evidence. For chords: reduce to the detected root for matching and
   record `polyphonic: true` so the UI/result can state the limitation honestly.
4. **Tempo & gating**: onset-strength + amplitude-envelope autocorrelation → detect
   periodic amplitude modulation (gate/pump) rate in Hz; estimate BPM when a plausible
   grid exists; when BPM confidence is high express the modulation rate as the nearest
   sync division (1/4, 1/8, 1/16, dotted, triplet — record the quantization error), else
   keep Hz. Sidechain-style pump (smooth periodic ducking) and hard gating (square-ish
   chopping) should be distinguished by the duty-cycle/shape of the extracted envelope.
5. **Amplitude envelope**: fit ADSR (attack, decay, sustain level, release) to the
   macro-envelope; classify the target as `one_shot` / `sustained` / `gated_rhythmic` /
   `evolving`.
6. **Harmonic/noise split**: HPSS (librosa or equivalent); record harmonic-vs-noise
   energy ratio and keep both component signals available for scoring.
7. **Transient/sustain windows**: detect the attack transient boundary; define separate
   transient and body analysis windows.
8. **Stereo image**: mid/side energy ratio, inter-channel correlation, estimated width;
   flag unison-style detune spread (chorused sidebands around the fundamental).
9. **Reverb/delay tail**: decay slope after the final onset; separate the `body` window
   (used for timbre matching) from the `tail` (excluded from timbre matching, its decay
   time recorded).
10. **Timbre trajectory**: frame-wise spectral centroid + MFCC trajectory over the body;
    classify static vs. evolving timbre; for evolving targets record the trajectory
    direction/extent (e.g. brightening sweep) for Part D's transcription.

Codec robustness (A4): when the input is lossy (mp3/aac), note it in the analysis dict;
in scoring, compute features on a bandwidth-limited version of *both* sides (e.g. low-pass
at the codec's effective cutoff) so pristine-render vs. lossy-target asymmetry doesn't
penalize matches.

## Part B — Comparison objective rework

In `core/matcher.py` (and wherever similarity is computed):

1. **Pitch-matched comparison**: compare the target against the library render at the
   nearest available C (C1–C7 exist for every preset), compensating the remaining
   within-octave offset by pitch-shifting the target's comparison copy to that C (never
   modify the stored target). Final candidate verification renders must be generated at
   the *detected* MIDI note instead of a fixed C.
2. **Best-window alignment**: score the target's body window against a sliding window of
   the candidate render and take the best alignment — never wall-to-wall. For gated
   targets, compare per sounding-segment (each gate slice) so a gated target and a
   sustained render compare on what is actually sounding.
3. **Composite score** replacing the single cosine: weighted combination of
   (a) CLAP cosine on loudness/pitch-normalized best-window audio,
   (b) MFCC + spectral centroid/flatness/contrast distances with equal-loudness
   (A-weighting or ISO 226-based) frequency weighting,
   (c) envelope-class agreement (one_shot/sustained/gated/evolving),
   (d) harmonic-vs-noise ratio agreement,
   (e) stereo-width agreement.
   Every sub-score is stored individually in `result.json` and surfaced in logs — never
   only the blended number. Choose initial weights by validating against the existing
   disguised-sound benchmark (the blend must rank the true preset first at least as often
   as the old score does — measure and report).
4. **Old score kept side-by-side**: compute and record the legacy score for every
   benchmark run in this stage so all before/after tables are directly comparable.

## Part C — Wavetable timbre index

1. Enumerate the factory wavetable catalog (Serum 1 automation surface exposes wavetable
   selection). On a neutral init patch (single oscillator, filter/FX off, full sustain),
   render each wavetable at C4 at WT positions 0%, 50%, 100% (3 renders per table).
2. Fingerprint each render (CLAP embedding + the Part B DSP feature set); store the index
   in the existing database or a parquet file under `data/models/`.
3. At query time: the target's harmonic body component (from A6) queries the index →
   top-K candidate wavetables that constrain/seed Part D's source stage.
4. **Gate**: for 20 known library presets (varied character), the preset's true wavetable
   must appear in the top-10 lookup for a measured, honestly-reported fraction — report
   the number, do not tune the test to pass. Also report catalog size, render failures,
   and any wavetables that couldn't be enumerated.
5. Serum 2: wavetable *selection* is not exposed on its automation surface — the index is
   Serum-1-scoped for search; for Serum 2, wavetable identity comes from the decoded
   preset CBOR (already available) and is used at retrieval/reporting level only. State
   this limitation in the report.

## Part D — Staged optimization (restructure the search)

Replace the flat search with four stages; log each stage's budget and outcome:

- **D1 Source**: candidate sources = top-K wavetables from Part C + top retrieval seeds.
  Coarse timbre-only scoring (envelope frozen at sustained), small per-candidate budget.
- **D2 Timbre refinement**: on surviving candidates, optimize filter/EQ/distortion/unison
  against the timbre sub-scores only. Dry-first: match with reverb/delay off, then set
  reverb/delay from the measured tail (A9) rather than searching for them.
- **D3 Transcription pass** (written, not searched): measured ADSR → amp envelope;
  measured gate/pump rate → LFO routed to amp/volume (nearest sync division when BPM
  confidence is high, Hz otherwise; pump vs. gate selects LFO shape); pitch glide →
  portamento/pitch-envelope; stereo width → unison count/detune/width; evolving-timbre
  trajectory → envelope- or LFO-to-cutoff/WT-position routing with depth chosen from the
  measured trajectory extent. Use the existing verified parameter mappings; for Serum 2,
  transcribe onto whatever of these targets its mapped surface reaches and report
  per-synth transcription coverage.
- **D4 Joint polish**: small final budget over the most sensitive parameters
  (sensitivity = measured audible impact, reuse or build a one-off sensitivity table),
  scored with the full composite objective, verification-rendered at the detected note.

## Part E — Retrieval invariance

The nearest-neighbor retrieval step must embody the same invariances, or the true match
is filtered out before optimization can ever see it:

1. Retrieval queries use the loudness-normalized, pitch-compensated, body-window,
   per-segment representation from Parts A/B — not the raw whole-file embedding.
2. Measure retrieval recall before/after on the disguised-sound benchmark **plus** the
   new gated-variant gate (Part F.2). If recall drops on any existing benchmark, that is
   a failure to fix, not a tradeoff to accept silently.

## Part F — Gates & honest measurement

1. **Full before/after table** on all existing benchmark suites — in-library (40),
   disguised, forced-rebuild, and the real BAM sample set — reporting legacy score, new
   composite, and every sub-score. Preserve before/after audition WAV pairs in the same
   `data/evaluations/` structure as previous rounds.
2. **Rhythm-invariance gate (the LFO insight as a regression test)**: take 10 known
   library presets; externally process their renders into disguised targets (quarter- and
   eighth-note gating, sidechain-style pump, ±3 semitone pitch shifts, −6 dB level drops,
   mp3 96k re-encode). The system must (a) retrieve the ungated original in the top 3 for
   a reported fraction of cases, and (b) transcribe the gate rate to the correct sync
   division (or within 10% in Hz). Report per-transformation results.
3. **Transcription accuracy gate**: synthesize targets with known ground truth (renders
   with known ADSR settings, known LFO rates, known glide times, known unison width);
   measured values must land within stated tolerances (envelope times within 20% or
   10 ms, whichever is larger; LFO rate within 5%; width qualitatively correct).
   Report every measurement.
4. **No regressions**: `scripts/verify_visual_redesign.py` and
   `scripts/verify_milestone4_ui.py` still pass; existing milestone verifiers untouched
   and passing; the render library and databases are not modified destructively.
5. **UI surfacing (minimal)**: the result panel's confidence display should expose the
   sub-scores (a compact breakdown — e.g. in the existing "…" details view), and when the
   target was polyphonic or vocal-like, say so plainly next to the score. No broader UI
   redesign in this stage.

## Reporting

Report all results — every gate, every before/after number, every limitation encountered
(with evidence, not hand-waving) — before considering this stage done. No silent scope
cuts: anything from this prompt you could not implement must appear in the report as a
named limitation with the reason. Stop at this reporting gate; do not begin Stage 2 work
(CLAP fine-tuning, synthetic scale-up) — those are designed for different hardware and a
separate prompt.
