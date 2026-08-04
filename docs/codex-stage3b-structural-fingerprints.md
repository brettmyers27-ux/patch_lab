# Codex Prompt — Stage 3B: controlled fingerprints for structural guidance (run ON the Windows 11 / RTX 5070 PC)

Copy everything below into Codex **on the Windows PC**, from
`%USERPROFILE%\Documents\PatchLab\soundmatch`, after `git pull`.

---

Read `docs/ACCURACY_ROADMAP.md` §2 and §3 (taxonomy item **C17** especially), and
`docs/benchmarks/stage3a-report.md` in full, before starting. §5 conventions apply unchanged:
numeric before/after on the *same* benchmarks, no silent scope cuts, stop at the reporting gate,
never automate a live DAW.

## Context: what Stage 3A proved and what it didn't

Stage 3A proved the write side completely: 50/50 structural mutations loaded and rendered
audibly-different audio, reachability moved from 40.32% (automation-only) to 100%
(write/reconstruct), and a structural render only costs 3.3041× an automation render — cheap
enough for direct search. **None of that is in question. It is adopted infrastructure.**

What failed was guidance. All four structural estimators (FX type, wavetable, mod route, noise
sample) were trained as audio-similarity nearest-exemplar heads against real library preset
renders, and every one lost to its own held-out most-common-value baseline:

| Category | Estimator top-1 | Most-common-value baseline |
|---|---:|---:|
| FX type | 0.179 | 0.207 |
| Wavetable | 0.134 | 0.353 |
| Mod route | 0.079 | 0.114 |
| Noise sample | 0.409 | 0.600 |

Stage 3A's own diagnosis, stated plainly in its report: *"Whole-patch descriptors confound
topology with all other patch settings."* A real preset's render mixes the effect, the filter,
the envelope, the oscillator, and everything else into one sound — asking a model to identify one
structural choice from that mixture is a much harder problem than the choice itself warrants.
Because guidance was noise, wiring it into search made BAM mean regress (0.784226 → 0.780600),
so Stage 3A correctly shipped structural search **disabled by default**. That decision stands
until this stage clears the same baselines Stage 3A used to reject its estimators.

## What this stage does differently

Taxonomy item **C17** already specifies the fix for exactly one of these four categories: *"a
wavetable timbre index — pre-render and fingerprint every factory wavetable on a neutral patch;
direct lookup replaces inherit-from-seed."* Stage 3A did not build this — it used the same
whole-preset nearest-exemplar approach for wavetable as for the other three, which is why
wavetable's baseline gap was the widest (0.134 vs 0.353).

**This stage extends C17's method to all four categories.** Instead of learning from confounded
real-preset audio, isolate each structural choice by rendering it on an identical, otherwise-fixed
neutral base patch, so the only thing that varies between renders is the one field being
fingerprinted:

- **FX type** — one neutral dry patch, each of the 16 FX types inserted at a fixed slot and
  setting, rendered and fingerprinted.
- **Wavetable** — the 330-wavetable union from Stage 3A's vocabulary (`data/models/`, see its
  report), each loaded into the same neutral oscillator patch, rendered and fingerprinted. This
  is C17 exactly as specified.
- **Mod route** — trickier, because a route is a source→destination pair, not a single value.
  Render each of the (source, destination) pairs actually observed in the library (Stage 3A
  found 4,906 observed full routes — use that as the enumeration, not the full 271×139 cross
  product) wired at a fixed representative depth on the neutral patch, and fingerprint the
  *modulation signature* the route produces (periodic movement in the relevant measured
  trajectory — amplitude, spectral centroid, pitch, or filter cutoff depending on destination),
  not raw spectral similarity. This connects directly to taxonomy **B9/A8**: the target's tempo
  and modulation-rate detection already exists and should drive the match, the same way the
  user's founding LFO insight does for continuous rhythm parameters.
- **Noise sample** — the 230-noise-sample union, each loaded into the same neutral noise
  oscillator patch, rendered and fingerprinted against the target's HPSS noise component
  (taxonomy **B11**), matching the existing noise-matching axis rather than a new one.

## Phase 1 — Build the four controlled fingerprint indices

1. For each category, render the full enumerated set (from Stage 3A's vocabulary — do not
   re-derive it) on a shared neutral base patch per category. Document the exact neutral base
   patch used for each (same oscillator/filter/envelope settings throughout that category's
   renders) so confounding is provably absent.
2. Persist as `data/models/serum2_{category}_fingerprints.npz` (or one combined file if that's
   cleaner — your call, document it) with stable IDs matching Stage 3A's structural space
   (`data/models/serum2_structural_space.json`).
3. Report wall clock and count rendered per category.

## Phase 2 — Rebuild the four estimators against controlled fingerprints

1. Replace the whole-preset nearest-exemplar heads with direct matching against the Phase 1
   fingerprints: measure the target audio's relevant signature (spectral fingerprint for FX/
   wavetable/noise, modulation trajectory for mod route) and rank candidates by distance to the
   controlled fingerprints, not by similarity to real preset renders.
2. Evaluate top-1 and top-5 on the **same held-out set Stage 3A used**, so the comparison is
   apples-to-apples.
3. **Gate, unchanged from Stage 3A**: each estimator must beat its own most-common-value
   baseline (reuse Stage 3A's baseline numbers above — do not recompute them differently). An
   estimator that does not clear its baseline is dropped, exactly as Stage 3A dropped its four.
   Report all four, pass or fail, honestly — a partial win (e.g. wavetable and FX clear the bar,
   route and noise still don't) is a legitimate, expected, reportable outcome.

## Phase 3 — Fix the benchmark's synth classification (small, but blocks clean measurement)

Stage 3A's report flagged that the benchmark's filename-based Serum 1/Serum 2 classification
placed all 99 BAM files into the Serum 2 subset, making its "Serum 2 subset" indistinguishable
from the whole set. Before re-running the A/B:

1. Identify why the filename policy misclassifies — inspect `scripts/benchmark_suite.py`'s
   classification logic against the actual 99 filenames.
2. Fix it so the subset reflects which synth each target was actually matched/synthesized
   against, and report the corrected Serum 1/Serum 2 split size.
3. This is a measurement fix, not a scope change — it must not alter which files are in the
   benchmark or how they're scored, only how the subset is reported.

## Phase 4 — Re-run the same A/B, correctly subset

Run `scripts/benchmark_suite.py` on the identical 99 BAM files and seed `20260802`:

- **A** — Stage 2B adopted stack (structural search off). Reuse Stage 2B's recorded baseline.
- **B** — Stage 2B stack + structural search, now guided by whichever Phase 2 estimators cleared
  their baseline. If zero estimators cleared their baseline, do not run this arm — report that
  outcome and stop; regenerating Stage 3A's failed result would waste the render budget.

Report BAM mean/median/min, retrieval@1/@5, invariance@1/@5, and the corrected Serum 2 subset
from Phase 3, on both the whole set and the subset.

**Adopt only if**: Serum 2 subset BAM mean improves materially over Stage 2B; whole-set BAM mean
does not regress; retrieval@1 ≥ 0.785; invariance@1 ≥ 0.403333; no new test failures; both UI
verification gates still pass. If it clears these, structural search moves from opt-in to
default-on for Serum 2 targets. If it doesn't, say so plainly — a second honest "not yet" is a
real finding about how hard this problem is, not a failure to hide.

## Phase 5 — Handoff

1. Commit code, `docs/benchmarks/*.json`, and an updated roadmap §1/§4 reflecting the measured
   ladder. Push. Version policy: one patch step per commit, single-digit components.
2. **Do not commit model or index artifacts, and do not touch the relay.** If any estimator or
   fingerprint index is adopted, write `docs/benchmarks/stage3b-artifact-manifest.json` with each
   adopted artifact's filename, byte size, SHA-256, and install destination. The Mac holds the
   relay credentials and publishes from that manifest.
3. Final report: every phase's real numbers including all four estimators' pass/fail against
   their baselines (not just the ones that pass), the corrected Serum 1/Serum 2 subset sizes,
   named limitations with reasons, and wall clock per phase. Stop at this gate.
