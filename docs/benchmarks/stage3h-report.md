# Stage 3H — BAM reproducibility audit

Stage 3H audited the unexpected Stage 3G BAM drift on Windows 11 / RTX
5070. The drift is real and is not a reporting arithmetic error. The current
shared multiprocessing pool preserved result order but allowed candidate chunks
to land on different persistent Serum hosts. Because Serum 2 retains host-local
state and sometimes has fresh-host audio modes, that changed the optimizer's
objective landscape even with fixed random seeds.

The benchmark now pins candidate position to one of four dedicated persistent
render hosts for every evaluation batch. This preserves four-way throughput and
still renders every genuinely different candidate state; it does not freeze or
reuse candidate audio. The interactive product remains on its existing default
dispatcher. Under the corrected benchmark procedure, two full Arm A runs differ
by 0.000281476 on the Serum 2 mean and 0.000147846 on the whole-set mean, both
inside the required 0.0005 threshold. Exact global agreement is not honestly
achievable: one target changed winner, and a targeted one-host test made that
target less stable rather than fixing it.

Corrected FX/wavetable/noise Arm B regressed corrected Arm A. Structural search
therefore remains opt-in, routes remain excluded, and production remains the
Stage 2B encoder/index/predictor stack with a new reproducible forward BAM
measurement procedure. Layer decomposition was not started.

## Phase 0 — Stage 3G arithmetic

All values were recalculated from the preserved per-target rows rather than
copied from the report:

| Metric | Arm A | Arm B | Arm C |
|---|---:|---:|---:|
| Serum 1 targets | 47 | 47 | 47 |
| Serum 2 targets | 52 | 52 | 52 |
| Total targets | 99 | 99 | 99 |
| Serum 2 mean | 0.787297587 | 0.795894421 | 0.791708471 |
| Whole-set mean | 0.780429775 | 0.784945284 | 0.782746603 |

Arm C versus Arm B recalculates to mean delta **-0.004185951**, 24
improved, 26 regressed, and 2 unchanged. The Stage 3G arithmetic is correct, so
the route decision is not revisited: routes remain excluded.

## Historical row comparisons

The complete sanitized target-by-target tables, including scores, deltas, base
preset IDs, candidate-state hashes, and decoded-audio hashes where available,
are in `stage3h-historical-comparison.json`.

### Stage 2B Arm A versus Stage 3G Arm A

Across all 99 rows, 47 were exactly unchanged and 52 changed. The 47 unchanged
rows are the unaffected Serum 1 rows reused from Stage 2B. Mean absolute delta
was 0.009334574 across all 99 and 0.017771593 across the 52 Serum 2 targets.
The largest positive delta was +0.032733202; the largest negative was
-0.083181918. Among Serum 2 rows, 44/52 retained the same base preset, 0/52
retained the same final candidate state, and 0/52 retained the same final
decoded audio.

The ten largest absolute changes were:

| Target | Delta |
|---|---:|
| Jaenga 1.wav | -0.083181918 |
| LSDREAM 5.wav | -0.051689744 |
| LSDREAM 1.wav | -0.049544156 |
| jkyl 4.wav | -0.046391845 |
| Space Laces 7.wav | -0.045297742 |
| Space Laces 6.wav | -0.043459594 |
| jkyl 1.wav | -0.035884559 |
| Zomboy 8.wav | -0.035488844 |
| YDG 1.wav | -0.035158813 |
| AC Free 1.wav | +0.032733202 |

### Stage 3D Arm B versus Stage 3G Arm B

Across all 99 rows, 52 were exactly unchanged and 47 changed. Mean absolute
delta was 0.008577812 across all 99 and 0.016330834 across the 52 Serum 2
targets. The largest positive delta was +0.065028548; the largest negative was
-0.054766834. Among Serum 2 rows, 48/52 retained the same base preset, 2/52
retained the same candidate state, and 5/52 retained the same decoded audio.

The ten largest absolute changes were:

| Target | Delta |
|---|---:|
| Sudley 1.wav | +0.065028548 |
| jkyl 1.wav | -0.054766834 |
| Space Laces 6.wav | +0.048843622 |
| AC Free 2.wav | +0.044593155 |
| Grabbitz 1.wav | -0.041314662 |
| LSDREAM 4.wav | -0.039506495 |
| LSDREAM 6.wav | +0.035135090 |
| Zomboy 7.wav | -0.034665763 |
| Jaenga 4.wav | -0.032992125 |
| Zomboy 1.wav | -0.032247067 |

This is a broad Serum 2 shift, not one or two outliers. Different final states
are the dominant direct cause; equivalent-state audio variation also exists.

## Benchmark/config trace

The source targets and all adopted assets match. Current and Stage 2B copies of
`preset_index.npy`, `note_index.npy`, `similarity_manifest.npz`, and
`factory_bundle.sqlite` are byte-identical. Relevant SHA-256 values are recorded
in `stage3h-results.json`; the encoder begins `fbaf3a30`, the parameter model
`5d0195e1`, and the delta model `25713aea`.

| Setting | Historical/current comparison | Classification |
|---|---|---|
| Encoder/checkpoint | Same fine-tuned checkpoint and SHA-256 | Same |
| Index/factory bundle | Different historical path, byte-identical files | Irrelevant path difference |
| Predictors | Same shipped parameter and delta predictors | Same |
| Corpus | Same source hashes; 47 AIFF Serum 1 and 52 WAV Serum 2 | Same |
| Serum classification | Stage 2B detail rows predate the format-provenance correction; final aggregation uses the preserved 47/52 split | Intentional reporting correction, membership/scores unchanged |
| Structural flag, Arm A | Off | Same |
| Structural categories, Arm B | FX + wavetable + noise; routes off | Same categories |
| Suite seed | 20260802 | Same |
| Search seed | 2026 plus deterministic preset/mutation/seed-rank derivations | Same |
| Initial candidates | 5 retrieved × (base + delta + 8 mutations) + 1 absolute = 51 | Same |
| Continuous budget | 300 evaluations, population 16, 120 seconds, stall 5 | Same |
| Render note | Library-conditioned/detected hypotheses per target | Same |
| Render/scoring rate | Serum render 44.1 kHz, resampled to 48 kHz CLAP comparison | Same |
| Target preprocessing | Adaptive trim/normalization, maximum 4 seconds | Same |
| Objective | Duration-dependent STFT/CLAP weighting and best prepared comparison signal | Same |
| Stage 2B candidate embedding | One CLAP call for each evaluation list | Historical behavior |
| Current candidate embedding | Batches of 32, introduced in Stage 3D for bounded VRAM | Intentional production implementation change; output-influencing |
| Stage 3D structural evaluation | One category list passed to `_evaluate` | Historical behavior |
| Stage 3G structural evaluation | Fixed proposal batches of 128, introduced for route-scale work | Intentional implementation change; output-influencing even with routes off |
| Legacy render dispatch | Shared `multiprocessing.Pool`; result order fixed, worker ownership free | Reproducibility defect |
| Corrected BAM dispatch | Four dedicated one-worker pools; candidate position pinned modulo four | Benchmark correction |
| Cache behavior | Independent match session and host population for every target/run | Same corrected procedure |

The Stage 2B absolute number cannot be reproduced as a trustworthy control.
Its raw render/cache/host assignment was not frozen, current code includes two
intentional batching changes, and the nominally identical current shared-pool
condition is demonstrably unstable. Production synthesis was not changed to
force the historical score to return.

## Five-repeat diagnostic subset

The required subset used the five largest Stage 2B-to-Stage 3G Arm A changes
(Jaenga 1, LSDREAM 5, LSDREAM 1, jkyl 4, Space Laces 7) plus the three smallest
Serum 2 changes (Zomboy 6, Zomboy 5, jkyl 2). Every repetition ran in a fresh
Python process with fresh Serum hosts.

### Legacy shared pool

| Target | Score span | Unique states | Unique decoded audios |
|---|---:|---:|---:|
| Jaenga 1 | 0.033132374 | 5 | 5 |
| LSDREAM 5 | 0.120634794 | 5 | 5 |
| LSDREAM 1 | 0.070016861 | 4 | 4 |
| jkyl 4 | 0.018316329 | 5 | 5 |
| Space Laces 7 | 0.049199700 | 5 | 5 |
| Zomboy 6 | 0.023958087 | 2 | 4 |
| Zomboy 5 | 0.001413763 | 2 | 3 |
| jkyl 2 | 0.020487726 | 5 | 5 |

All 8/8 targets changed optimizer/final winner. Initial seed-render modes varied
on 5/8. Zomboy 5 and Zomboy 6 produced more decoded-audio hashes than state
hashes, proving equivalent-state audio variation on 2/8. The instability affects
seed renders, optimization renders/objective selection, and final output.

Fixed state plus fixed decoded audio always produced the same BAM score (for
example the three identical-state AC Free 1 legacy preflight runs); there is no
measured scoring instability. WAV file hashes vary because libsndfile writes
mutable container metadata, so the verifier also hashes sample rate, shape, and
decoded float32 samples. Exact per-run state/audio hashes are in
`stage3h-results.json`.

### Corrected pinned pool

All eight targets produced **zero score span, one candidate-state hash, and one
decoded-audio hash** across five fresh processes. This isolates render dispatch
as the broad fix. It does not freeze candidate audio: every state continues to
be synthesized and scored.

## Full corrected verification

| Metric | Arm A run 1 | Arm A run 2 | Span |
|---|---:|---:|---:|
| Serum 2 mean | 0.792934913 | 0.792653437 | **0.000281476** |
| Whole-set mean | 0.783390795 | 0.783242949 | **0.000147846** |
| Serum 2 median | 0.812225014 | 0.810699791 | — |
| Serum 2 minimum | 0.605371356 | 0.605371356 | 0 |

Both required aggregate spans are below 0.0005. Fifty-one of 52 Serum 2 rows
were exact. `jkyl 1.wav` changed CMA state on the same base preset and changed
score 0.823500991 to 0.808864236. A targeted five-run single-host test did not
remove it: it produced five states, five decoded audios, and a 0.029998243 score
span. The remaining variance is therefore honestly recorded rather than hidden.
The forward baseline is Arm A run 1, with run 2 as its stability bound.

Corrected preferred Arm B (FX + wavetable + noise, routes excluded) measured:

| Metric | Corrected Arm A run 1 | Corrected Arm B |
|---|---:|---:|
| Serum 2 mean | **0.792934913** | 0.790872711 |
| Serum 2 median | **0.812225014** | 0.797310799 |
| Serum 2 minimum | **0.605371356** | 0.599511087 |
| Whole-set mean | **0.783390795** | 0.782307618 |

Paired Arm B versus Arm A over the 52 Serum 2 targets: mean delta
**-0.002062201**, 24 improved, 28 regressed, and 0 unchanged. Because Arm B is
already lower on both required aggregate metrics, a second Arm B run cannot
establish the required positive improvement and was not run.

Frozen retrieval remained **0.775@1 / 0.890@5**. Invariance remained
**0.403333@1 / 0.596667@5**.

## Regression coverage and gates

`scripts/stage3h_bam_audit.py` records canonical decoded-audio and candidate
state hashes, executes fresh-process target audits, verifies repeat stability,
and generates historical comparisons. Benchmark row resume keys now include
the matcher process count and deterministic-dispatch version, preventing legacy
rows from being silently reused as corrected measurements.

Tests cover canonical decoded-audio hashing, instability summaries, and fixed
position-to-worker mapping. Verification results:

* full automated suite: **129 passed**, one upstream deprecation warning;
* `scripts/verify_visual_redesign.py`: **pass** (`gate_pass=true`);
* `scripts/verify_milestone4_ui.py`: **pass**
  (`MILESTONE4_UI_GATE_PASS=true`).

## Decision and handoff

The trustworthy forward BAM reference replaces the unreproducible historical
absolute score. This is evidence-based, not a lowered gate: the old control has
no reproducible host/cache assignment, the current legacy condition changed all
52 Serum 2 Arm A rows historically and all eight diagnostic targets across
fresh repeats, and even one-host execution cannot make every target exact.

FX/wavetable/noise structural search is **not adopted** because corrected Arm B
is lower than corrected Arm A on Serum 2 and the whole set. It remains opt-in.
Routes remain excluded because the arithmetically valid Stage 3G Arm C result is
worse than Arm B. The relay is untouched.

No Mac action is required. There is no new distributable artifact or manifest.
The Mac may pull the source/documentation normally; it does not need to publish,
install, or regenerate model/index files. Stage 3H stops here and does not start
layer decomposition.
