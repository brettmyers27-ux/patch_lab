# Stage 3D — fingerprint repair and in-context structural-search report

Stage 3D repaired the noise and modulation-route identity sets and completed the
fixed 99-file A/B. In-context search improved the corrected Serum 2 BAM mean from
0.794525 to 0.796935 and the whole-set mean from 0.784226 to 0.785492. It is not
adopted: the current-machine retrieval rerun measured 0.780@1, below the explicit
0.785 adoption gate. Stage 2B remains production and deep structural search stays
opt-in.

## Noise diagnosis and repair

The ten Stage 3C noise failures had zero overlap with missing source files. All
twenty source files in the ten pairs exist on this PC. Eight pairs were the same
resource written with and without a leading slash. The remaining two were different
factory aliases whose decoded audio was sample-equivalent within one 16-bit least
significant bit; all ten pairs were audio-equivalent by that test.

Each two-member duplicate cluster was collapsed to its more frequently observed
representative. The repaired set contains 220 candidates. It passes exhaustive
self-retrieval 220/220 and has zero duplicate clusters through cosine distance
`1e-4`. This was duplicate identity/provenance, not resource unavailability or a
descriptor bug.

## Mod-route diagnosis and repair

Eight deterministic samples of the 131 zero-descriptor routes were rendered for
four seconds with the modulation disabled, at depth 32, and at depth 100. Every
sample produced exactly zero direct stereo, mono, and side audio difference at both
active depths. The sampled failures are therefore genuinely inactive in their
controlled carriers, not movement hidden from the mono trajectory descriptor.

Eight deterministic duplicate clusters were also inspected against their loaded
carrier states. Although the samples contained multiple source IDs and destination
parameters, all eight had identical loaded LFO and macro settings across their
members. Together with the audio collapse, this supports genuine indistinguishable
controlled behavior rather than row/ID misalignment.

The 131 zero rows were excluded. Connected descriptor clusters at `1e-4` were
collapsed to one representative, preferring higher observed frequency and then a
stable ID. The fixed-point repair collapsed 239 clusters covering 2,529 members;
the largest had 289 members. The route set shrank from 4,906 to **2,485** and then
passed exhaustive self-retrieval 2,485/2,485 with zero clusters through `1e-4`.

Target-motion narrowing ran on every Serum 2 BAM target. It retained 605–1,847
routes (median 1,606), never reaching the required limit of 300. Routes were
therefore documented and skipped for arm B rather than consuming an unbounded
budget.

## Deep in-context search

Arm B searched repaired FX, wavetable, and noise candidates in the live working
patch, then handed the winner to the unchanged continuous optimizer. The deep
structural allowance is separate: 4,096 evaluations and 900 seconds, while the
balanced continuous allowance remains 300 evaluations and 120 seconds. CLAP scoring
uses batches of 32 to keep VRAM bounded.

The number searched depends on how many fields the base patch exposes:

| Category | Targets | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| Wavetable | 52 | 330 | 990 | 990 |
| FX type | 52 | 16 | 104 | 400 |
| Noise sample | 52 | 220 | 220 | 220 |
| Mod route | 0 | — | — | — |

A two-target smoke showed that pass two changed zero winners while almost doubling
cost, so the final gate used one coordinate pass. Across the 52 Serum 2 targets,
wall clock averaged 103.81 seconds, median 78.54, and maximum 328.13.

## Fixed A/B and decision

The seed was `20260802`. All 52 corrected Serum 2 WAV targets ran arm B; the 47
Serum 1 AIFF rows were reused from adopted Stage 2B because this feature cannot
affect them. All 99 rows completed with zero errors.

| Metric | Arm A | Arm B | Gate |
|---|---:|---:|---|
| Whole-set BAM mean | 0.784226 | **0.785492** | Pass |
| Serum 2 BAM mean | 0.794525 | **0.796935** | Pass |
| Serum 2 median | 0.804774 | **0.809769** | Informational |
| Serum 2 minimum | 0.647449 | 0.636777 | Informational regression |
| Retrieval@1 / @5 | 0.785 / 0.895 | **0.780 / 0.890** | **Fail @1** |
| Invariance@1 / @5 | 0.403333 / 0.596667 | 0.403333 / 0.596667 | Pass |

Twenty-eight Serum 2 targets improved and 24 regressed; the paired mean delta was
+0.002410. Forty-five of 52 final Serum 2 winners carried at least one structural
override. The full test suite passed (123 tests), as did both UI gates.

The BAM improvement proves that exhaustive in-context structural search can add
value once fingerprint identities are clean; this does not support declaring the
scoring objective the current limit. Guidance remains the blocker for routes because
measured motion did not narrow them enough. The immediate adoption blocker is the
retrieval gate, and the mixed 28/24 paired result plus lower minimum argues for
further gating/calibration before enabling deep search by default. No artifact is
adopted and the relay is untouched.
