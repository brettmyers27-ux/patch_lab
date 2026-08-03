# Stage 3A — Serum 2 structural-search report

Stage 3A proved that PatchLab can write, load, render, score, and export Serum 2's five
non-automatable structural categories. It did **not** pass the same-corpus sound-quality
adoption gate. Structural search therefore remains opt-in and Stage 2B remains production.

## Verdict

| Gate | Stage 2B A | Stage 3A B | Change | Result |
|---|---:|---:|---:|---|
| BAM mean | 0.784226 | 0.780600 | -0.003626 | Fail |
| BAM median | 0.793119 | 0.791120 | -0.001999 | Fail |
| BAM minimum | 0.541550 | 0.538664 | -0.002886 | Fail |
| Factory retrieval@1 | 0.785 | 0.785 | 0 | Pass |
| Factory retrieval@5 | 0.895 | 0.895 | 0 | Pass |
| Invariance@1 | 0.403333 | 0.403333 | 0 | Pass |
| Invariance@5 | 0.596667 | 0.596667 | 0 | Pass |
| BAM failures | 0 | 0 | 0 | Pass |

The benchmark's existing filename policy classified all 99 BAM files as Serum 2, so its
Serum 2 subset is 99/99 and has the same 0.780600 mean. This is not an independently
hand-labeled synth subset and should be treated as directional. Fifteen emitted winners
carried structural changes; the mean was 0.727 changed structural leaves per result.

Optimizer reachability coverage moved from the measured automation-only weighted coverage
of 0.403241 to 1.0: every existing structural leaf can be supplied to reconstruction and
native export. This is a reachability measurement, not a claim that every combination was
enumerated or correctly inferred. The failed estimator gate explains why higher coverage
did not improve BAM: PatchLab can now write structure, but cannot yet reliably transcribe
which structure the target needs.

## Phase results

1. **Cost gate.** Fifty trials per path measured 18.021 ms median for automation and
   59.542 ms for reconstruct/write/load/render, a 3.3041× ratio. Repeated state loads reused
   one live instance. The existing four-process pool sustained 2,856.90 evaluations/minute
   with no failures. Decision: direct search is viable only over tight shortlists; a neural
   surrogate is not a cost prerequisite for *evaluation*, but better learned guidance is a
   prerequisite for accuracy.
2. **Vocabulary.** All 710 Serum 2 presets plus the installed factory folders produced a
   330-wavetable union, 230-noise-sample union, 16 FX types, 271 sources, 139 named
   destinations, 4,906 observed full routes, and 55 embedded-WT payload hashes. Installed
   files and preset observations retain separate provenance. FX/mod host value strings were
   not readable because these choices are absent from the automation surface.
3. **Mutation hard gate.** All 50/50 one-field states loaded and all 50/50 rendered
   measurably different audio. Directional counts were 9/10 wavetable, 4/10 embedded WT,
   4/10 noise, 10/10 route, and 10/10 FX. The lower custom/noise directional counts are
   limitations, not relabeled successes.
4. **Estimators.** Every audio-only nearest-exemplar head failed its held-out most-common
   baseline and was dropped. FX top-1 was 0.179 vs 0.207; wavetable 0.134 vs 0.353; route
   0.079 vs 0.114; noise 0.409 vs 0.600. No estimator artifact was adopted.
5. **Matcher.** The existing matcher gained staged per-field proposals, persistent-worker
   reconstruction, continuous optimization on the winning structure, and native export of
   structural overrides. `K=2` follows the 3.3041× cost result. Every active WT/noise/FX/
   mod-route field in the seed is exposed; arbitrary overrides remain possible. Embedded
   custom payloads remain reachable through the API but are not automatically proposed
   because hashes alone cannot reconstruct licensed payload data. A second
   structure↔continuous round did not fit the fixed 300-evaluation budget.
6. **Same benchmark.** B completed 99/99 BAM files in 3,041.704 seconds including retrieval
   and invariance. It preserved upstream gates but regressed BAM, so structural search is
   disabled by default and no private artifact is a relay candidate.

## Timing and limitations

Measured execution wall clocks were approximately 13 s for the Phase 0 command, 13 s for
the vocabulary build/test command, 34 s for mutation verification, 90 s for estimator
render/evaluation, 26 s for the balanced end-to-end smoke, and 3,041.704 s for the final
benchmark. Engineering/documentation time is not reconstructed from estimates.

The key limitation is guidance, not state injection or render cost. Whole-patch descriptors
confound topology with all other patch settings. The next accuracy attempt should treat a
neural surrogate or controlled neutral-patch structural fingerprints as a **guidance
prerequisite**, validate them against the same common-value baselines, and only then revisit
opt-in structural search. This report stops at the Stage 3A gate.
