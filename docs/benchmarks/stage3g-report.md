# Stage 3G — modulation-route search and final structural adjudication

Stage 3G was run on Windows 11 / RTX 5070 at seed `20260802` after
`git pull --ff-only` confirmed that commit `5e85f9e` was current. The existing
in-context path is owned by `core/matcher.py`, with proposal and narrowing
helpers in `core/structural_search.py`; this stage extended those files rather
than creating another renderer, reconstruction path, or search engine.

## Phase 0: documentation and frozen retrieval gate

The supplied dirty Stage 3F handoff contained the duplicated/corrupted block
described in the prompt. It was removed and the final
`docs/benchmarks/stage3f-report.md` matches the already-canonical repository
text, so there is no net Stage 3F file diff. Its measured facts remain unchanged:
200 renders, 0.775@1 (155/200), 0.890@5 (178/200), host sensitivity for presets
190 and 216, and no Stage 3F structural-search adoption.

Hard Gate 0 passed before route work. Structural off and structural on both read
the same manifest-validated 200 frozen WAVs and returned 0.775@1 and 0.890@5.
Their complete 200-row files are byte-identical (SHA-256
`4a1f7ef7ede398f3e9e2a4f01dea8527266a58890104a76cfe46653422bd43d1`).
All frozen WAV hashes and modification times were unchanged and no WAV was
regenerated. The final Arm C read repeated the same result. Exact gate details
are in `stage3g-gates.json`.

## Real candidate workload

The prompt's simple category sum did not represent the live workload: every
route identity must be evaluated in every existing modulation slot. Across all
52 Serum 2 targets, the exact category ranges were:

| Item | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Wavetable evaluations | 330 | 990 | 990 |
| FX evaluations | 16 | 104 | 400 |
| Noise evaluations | 220 | 220 | 220 |
| Existing route fields | 0 | 21 | 32 |
| Routes before motion narrowing | 2,485 | 2,485 | 2,485 |
| Routes after motion narrowing | 605 | 1,606 | 1,847 |
| Complete structural evaluations | 896 | 38,262 | 54,523 |

Only 1/52 complete workloads fit 4,096 evaluations and only 2/52 fit 8,192.
The 4,096 budget was therefore not enough. The implementation raises the
per-target allowance only as needed, never above 8,192, and retains a strict
prefix of highest-ranked complete destination groups when necessary. Fifty
targets required this evaluation-count fallback. Fitted totals ranged from 896
to 8,192 (median 8,130.5). The exact sanitized 52-row before/after table is
`stage3g-candidate-audit.json`.

## Throughput and time fitting

Bounded 512-route probes used the real reconstructed-state render and trusted
full-mixture scoring path:

| Route-count band | Surviving routes | Probe route seconds | Evaluations/min | Initially projected complete structural seconds |
|---|---:|---:|---:|---:|
| Low | 605 | 23.578 | 1,302.91 | 211.82 |
| Median | 1,606 | 106.312 | 288.96 | 7,303.43 (1,700.25 after evaluation fitting) |
| High | 1,847 | 20.453 | 1,501.98 | 707.64 |

Peak VRAM is unavailable because the existing benchmark diagnostics do not
sample it; no estimate is presented as a measurement. The final Arm C passes
confirmed current-implementation route rates of 1,739.13, 358.97, and 2,131.58
evaluations/min for those same low/median/high bands. Their measured total
structural times were 161.719, 703.782, and 224.719 seconds.

The live route stage derives a conservative rate from the preceding non-route
categories, reserves 60 seconds for the final batch, and retains only complete
destination groups that project inside 900 seconds. Twelve of 52 targets needed
this additional time fallback. Across Arm C, structural time was 457.716 seconds
mean, 410.547 median, and 823.219 maximum. Route time was 380.648 seconds mean,
350.203 median, and 767.500 maximum. Every selected set completed, every target
stayed at or below 8,192 evaluations and 900 structural seconds, and no target
failed. Exact live per-target count/time fitting appears in
`stage3g-bam-comparison.json`; probe details appear in
`stage3g-throughput.json`.

The 605–1,847 route identities were therefore tractable on the RTX 5070 only
with destination-first fitting: 50 targets needed evaluation fitting and 12 of
those needed measured-time fitting. Routes were actually searched on 51/52
targets. One working patch exposed zero existing modulation slots, so there was
no proven field through which to write a route; the stage did not invent a new
slot outside the verified reconstruction path.

## Fixed three-arm BAM

All arms used the identical corrected corpus: 47 Serum 1 AIFF, 52 Serum 2 WAV,
99 total. The 47 Stage 2B Serum 1 rows were reused in every arm because every
Stage 3G code path is gated to Serum 2; all 52 Serum 2 rows were rerun in every
arm. Every arm completed 99/99 with zero errors.

| Metric | Arm A production | Arm B WT/FX/noise | Arm C WT/FX/noise/routes |
|---|---:|---:|---:|
| Whole-set mean | 0.780430 | 0.784945 | 0.782747 |
| Serum 2 mean | 0.787298 | 0.795894 | 0.791708 |
| Serum 2 median | 0.795389 | 0.801323 | 0.806117 |
| Serum 2 minimum | 0.646823 | 0.641258 | 0.618317 |
| Improved vs Arm A | — | 32 | 25 |
| Regressed vs Arm A | — | 19 | 26 |
| Unchanged vs Arm A | — | 1 | 1 |
| Paired mean delta vs Arm A | — | +0.008597 | +0.004411 |

The live Arm A rerun is lower than its historical reference (0.784226 whole,
0.794525 Serum 2), despite identical source hashes and the structural branch
being disabled. This is reported as observed cross-run synthesis/optimization
drift; historical values were not substituted. Arm B likewise did not reproduce
the historical Stage 3D result: it measured 0.784945 whole and 0.795894 Serum 2
versus 0.785492 and 0.796935 historically.

Direct Arm C versus Arm B was a regression:

* paired mean delta: **-0.004186**;
* improved: **24/52**;
* regressed: **26/52**;
* unchanged: **2/52**;
* whole-set delta: **-0.002199**.

## Retrieval, invariance, and verification

All three arms returned the exact same frozen retrieval and invariance rows:

| Gate | Arm A | Arm B | Arm C | Required |
|---|---:|---:|---:|---:|
| Frozen retrieval@1 | 0.775 | 0.775 | 0.775 | >= 0.775 |
| Frozen retrieval@5 | 0.890 | 0.890 | 0.890 | >= 0.890 |
| Invariance@1 | 0.403333 | 0.403333 | 0.403333 | >= 0.403333 |
| Invariance@5 | 0.596667 | 0.596667 | 0.596667 | >= 0.596667 |

Verification results:

* full automated suite: **126 passed**, one upstream deprecation warning;
* `scripts/verify_visual_redesign.py`: **pass** (`gate_pass=true`);
* `scripts/verify_milestone4_ui.py`: **pass**
  (`MILESTONE4_UI_GATE_PASS=true`);
* `git diff --check`: **pass**.

## Mechanical adoption decisions

Modulation routes do **not** earn inclusion. Arm C Serum 2 mean is lower than
Arm B (0.791708 < 0.795894) and Arm C whole-set mean regresses below Arm B
(0.782747 < 0.784945). Routes remain disabled even though retrieval,
invariance, compute, test, and UI gates pass.

Arm B is the preferred opt-in structural configuration, but deep structural
search does **not** become production-default. It fails both required absolute
quality gates: 0.795894 < 0.796935 on Serum 2 and 0.784945 < 0.785492 on the
whole set. The remaining retrieval, invariance, completion, test, and UI gates
pass. Production therefore remains the adopted Stage 2B stack, while the
FX/wavetable/noise structural pass remains opt-in.

This closes the structural-search arc. Direct route search is computationally
possible with hierarchical fitting, but it does not improve BAM over
FX/wavetable/noise alone. The measured unresolved limitations are the negative
paired route result, one patch with no existing writable route slot, and
cross-run synthesis/optimization drift in the current BAM reruns. No further
structural architecture is started here. The next priority is layer
decomposition: dominant layer, subtract, residual, preset stack.

## Version, files, and handoff

The single legal version transition is **1.3.5 -> 1.3.6**.

Changed files:

* `app/__version__.py`
* `core/match_workflow.py`
* `core/matcher.py`
* `core/structural_search.py`
* `docs/ACCURACY_ROADMAP.md`
* `docs/benchmarks/stage3g-bam-comparison.json`
* `docs/benchmarks/stage3g-candidate-audit.json`
* `docs/benchmarks/stage3g-gates.json`
* `docs/benchmarks/stage3g-report.md`
* `docs/benchmarks/stage3g-throughput.json`
* `scripts/benchmark_suite.py`
* `scripts/stage3g_candidate_audit.py`
* `scripts/stage3g_compare_arms.py`
* `tests/test_structural_search.py`

No model, index, private BAM audio, render cache, preset, or generated search
artifact is committed. No new non-code artifact is required, so no Stage 3G
artifact manifest is needed. The Mac does not need to install or publish
anything, and the private relay was not touched.
