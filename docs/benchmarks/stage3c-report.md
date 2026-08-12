# Stage 3C — in-context structural-search report

Stage 3C stopped at its mandatory Phase 0 integrity gate. The Stage 3B controlled
fingerprint index does not satisfy rank-1 self-retrieval for modulation routes or
noise samples, so no in-context candidate search, deep-mode timing, or 99-file A/B
was run. Stage 2B remains production and no artifact is adopted.

## Phase 0 verdict

The check loads the private Stage 3B NPZ, applies the same row and query
normalization and stable-ID tie break as `ControlledFingerprintIndex`, and asks each
fingerprint to retrieve itself. It first tests a deterministic sample of 20 per
category (all 16 for FX), then exhaustively checks every row.

| Category | Candidates | Sample rank-1 | Exhaustive rank-1 | Zero descriptors | Gate |
|---|---:|---:|---:|---:|---|
| FX type | 16 | 16/16 | 16/16 | 0 | Pass |
| Wavetable | 330 | 20/20 | 330/330 | 0 | Pass |
| Mod route | 4,906 | 16/20 | 3,682/4,906 | 131 | **Fail** |
| Noise sample | 230 | 19/20 | 220/230 | 0 | **Fail** |

This is an index identity-integrity failure, but the test alone cannot prove that
the ID array is shifted. The distinctness evidence directly shows a stronger local
cause: many route rows collapse to the same descriptor, and 131 route rows contain
no descriptor signal at all. A stable-ID tie break necessarily returns one member of
each collapsed group rather than every member's own ID. Noise has the same failure on
a smaller scale. Therefore Stage 3B's ID-specific estimator results for routes and
noise cannot support the domain-gap diagnosis recorded there. Its render counts and
raw benchmark observations remain historical measurements, but the failed lookup
conclusions are not a sound basis for Stage 3C search.

## Fingerprint distinctness

Clusters are connected components under cosine distance. The exact check uses
`1e-7`; `1e-6` and `1e-4` show near-duplicate sensitivity.

| Category | Exact clusters / members | At 1e-6 | At 1e-4 | Largest at 1e-4 |
|---|---:|---:|---:|---:|
| FX type | 0 / 0 | 0 / 0 | 0 / 0 | 1 |
| Wavetable | 0 / 0 | 0 / 0 | 5 / 12 | 4 |
| Mod route | 127 / 1,232 | 174 / 1,411 | 238 / 2,528 | 289 |
| Noise sample | 9 / 18 | 10 / 20 | 10 / 20 | 2 |

Wavetable and FX fingerprints are sufficiently distinct on this PC. Noise is not
largely indistinguishable—20 of 230 rows participate in near-duplicate clusters—but
it still fails the explicit self-retrieval rule. Routes are materially collapsed:
2,528 of 4,906 rows participate in clusters at `1e-4`, in addition to their 131
zero-information rows.

## Search and A/B gate

The Stage 3C prompt says to report a self-retrieval failure immediately rather than
proceeding. Candidate counts actually searched are therefore zero for FX,
wavetable, routes, and noise. Deep structural mode was not implemented and has no
per-target wall-clock measurement. Arm B was not produced; the adopted Stage 2B
metrics remain the unchanged reference:

- whole-set BAM mean 0.784226;
- corrected 52-file Serum 2 mean 0.794525, median 0.804774, minimum 0.647449;
- retrieval@1/@5 0.785/0.895;
- invariance@1/@5 0.403333/0.596667.

No new scoring claim is possible. In particular, Stage 3C did not reach the test
that could distinguish search/guidance limitations from a scoring-objective limit.
The immediate blocker is fingerprint integrity and structural-resource
distinguishability, especially for routes. Repair or regenerate those controlled
measurements, preserve one-to-one candidate provenance, and rerun Phase 0 before
implementing in-context search.

The detailed public measurements are in `stage3c-phase0.json`. Private stable-ID
failure examples are written by `scripts/verify_structural_fingerprints.py` under
ignored `data/stage3c/`; neither the index nor diagnostics are committed.
