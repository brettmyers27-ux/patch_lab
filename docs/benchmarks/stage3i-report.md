# Stage 3I — two-layer decomposition feasibility

Stage 3I implemented a reusable, experimental one-or-two-preset stack and tested
phase-robust residual matching on the fixed Windows 11 BAM corpus. The fresh
Stage 3H control reproduced exactly, but the 16-target continuation pilot failed
one mandatory condition: residual matching selected Layer 2 on 3/4 low-residual
negative controls, above the permitted 2/4. The full 99-file two-layer A/B was
therefore not run.

Two-layer decomposition is **not validated** and is not enabled in the product.
The measured blocker is selection calibration, not an absence of signal in the
high-residual hypothesis: on the 12 high-residual targets, residual Arm C beat
both the one-layer control and compute-matched duplicate-stacking Arm B.

## Exact pre-change control and Hard Gate 0

The control used seed `20260802`, structural search off, the production Stage 2B
encoder/index/predictors, 47 preserved unaffected AIFF rows, 52 freshly rendered
Serum 2 WAV rows, and Stage 3H's four dedicated one-worker pools with candidate
position pinned modulo four. All 99 rows are in `stage3i-control.json`.

| Metric | Stage 3H reference | Fresh Stage 3I control | Absolute delta |
|---|---:|---:|---:|
| Whole-set mean | 0.783390794740 | 0.783390794740 | 0.000000000000 |
| Serum 2 mean | 0.792934912902 | 0.792934912902 | 0.000000000000 |
| Serum 2 median | 0.812225013971 | 0.812225013971 | 0 |
| Serum 2 minimum | 0.605371356010 | 0.605371356010 | 0 |
| Serum 1 source-group mean | — | 0.772831344858 | — |

Both required aggregate deltas are below `0.0005`; Hard Gate 0 passed.
The run completed 99/99 without error in 1,862.407 seconds. Including the 47
instant preserved rows, per-target wall time was mean 18.803 s, p50 20.579 s,
and p95 61.497 s. Across the 52 fresh rows it was mean 35.797 s, p50 28.250 s,
and p95 68.279 s.

## Implementation and files

`core/layer_decomposition.py` owns reusable aligned STFT-magnitude subtraction.
`core/preset_stack.py` owns the versioned one/two-layer representation, exact
null candidate, timing shift, mixing, and deterministic stack-level polish.
`scripts/stage3i_two_layer.py` orchestrates the existing matcher, renderer,
preprocessing, Stage 3H dispatcher, private resumable artifacts, pilot gate, and
commit-safe diagnostics. `tests/test_layer_decomposition.py` covers the new DSP,
serialization, null, and deterministic-polish contracts. Production matcher,
render/export code, UI, structural-search defaults, and relay were not changed.

The machine-readable committed diagnostics are:

* `stage3i-control.json` — all 99 control scores and source hashes;
* `stage3i-residuals.json` — all 99 subtraction diagnostics and pilot labels;
* `stage3i-pilot.json` — complete 16-target scores and layer diagnostics;
* `stage3i-results.json` — gates and final decision.

All generated presets, candidate states, residual WAVs, mixed WAVs, match
sessions, and private paths remain ignored under `data/`.

## Residual extraction

The system uses the same adaptive `prepare_query_audio` signal as production:
silence is trimmed, comparison is capped at four seconds, short clips retain the
same explicit padding rules, and both sides are loudness normalized. Layer 1 is
the exact preserved production winner; it is not re-optimized.

For each target, Layer 1 is magnitude-aligned over deterministic offsets from
`-100` to `+100` ms in 5 ms steps. A Hann STFT uses FFT 2048 and hop 512. For
each offset, the bounded scalar

`a = clip(<T_mag,L1_mag> / <L1_mag,L1_mag>, 0, 2)`

is fitted and the smallest residual spectral-energy ratio wins. The residual is
`max(T_mag - a * L1_mag, 0)` and is inverted with the target STFT phase. RMS and
spectral energy relative to target, duration, centroid, spectral flatness,
harmonic/percussive ratio, and final-quarter energy are recorded before residual
normalization. The matcher only receives the normalized reconstruction.

Across all 99 targets, residual spectral-energy ratio was mean 0.597880, p50
0.610996, p95 0.919246, minimum 0.125974, and maximum 0.962877. Residual RMS
ratio was mean 0.743167, p50 0.769287, and p95 0.957198. Fitted Layer 1 scale
ranged 0.137408–1.312955 (mean 0.549769, p50 0.548496). Alignment ranged
`-100` to `+100` ms with p50 `-10` ms.

## Stack representation and mix search

`PresetStack` records stack version, target synth, layer count, state reference,
base preset, candidate-state and decoded-audio hashes, gain, timing, role,
per-layer match score, combined score, residual energy, selection, and diagnostic
fields. It can serialize either one layer or two and rejects inconsistent data.

Only Layer 1 gain, Layer 2 gain, and Layer 2 timing are searched. Initial gains
come from fitted magnitude scales. The fixed schedule evaluates nine coarse gain
pairs, all eleven `-25` to `+25` ms offsets for the best pair, then at most 25
local gain pairs: at most 34 distinct gain pairs and 45 mixture evaluations,
below the 100-combination limit. The exact unchanged Layer 1 score/audio is an
explicit null candidate and wins unless a mixture is strictly higher.

## Fixed 16-target pilot

The pilot was selected only after residuals existed for all 99 targets. The 12
largest energy ratios are hypothesis targets; the four smallest non-overlapping
ratios are negative controls. Arm B made the same second matcher call against the
full target. Arm C matched the phase-robust residual. Both arms received the same
mix search.

| Selection | Target | Residual energy | A | B | C | B selected | C selected |
|---|---|---:|---:|---:|---:|:---:|:---:|
| High | VR vox 1.aif | 0.962877 | 0.541550 | 0.541550 | 0.541550 | no | no |
| High | Sully 1.wav | 0.956313 | 0.727330 | 0.739890 | 0.727330 | yes | no |
| High | VR vox 2.aif | 0.956152 | 0.595179 | 0.596379 | 0.641434 | yes | yes |
| High | Sully 2.wav | 0.936026 | 0.769619 | 0.789300 | 0.769619 | yes | no |
| High | Dill Lead 1.aif | 0.933088 | 0.746486 | 0.789221 | 0.821610 | yes | yes |
| High | LSDream Lead 5.aif | 0.917708 | 0.583191 | 0.634645 | 0.717245 | yes | yes |
| High | Mazare Bass 2.aif | 0.910940 | 0.762484 | 0.802512 | 0.829772 | yes | yes |
| High | VR Bass 2.aif | 0.903760 | 0.733277 | 0.829536 | 0.787502 | yes | yes |
| High | jkyl 5.wav | 0.884117 | 0.825036 | 0.875761 | 0.829937 | yes | yes |
| High | Grabbitz 1.wav | 0.867170 | 0.797735 | 0.834060 | 0.812431 | yes | yes |
| High | LSDream Bass 3.aif | 0.861326 | 0.721504 | 0.821955 | 0.768375 | yes | yes |
| High | Grabbitz 2.wav | 0.854363 | 0.717662 | 0.759250 | 0.800090 | yes | yes |
| Low | Mazare Bass 1.aif | 0.125974 | 0.761841 | 0.791825 | 0.761841 | yes | no |
| Low | YDG 2.wav | 0.141154 | 0.859931 | 0.875564 | 0.871985 | yes | yes |
| Low | Nasty VR Bass 2.aif | 0.177986 | 0.850390 | 0.871575 | 0.860024 | yes | yes |
| Low | LSDREAM 4.wav | 0.196199 | 0.759774 | 0.803893 | 0.772202 | yes | yes |

### Pilot aggregates and gate

| Subset | Arm A | Arm B | Arm C | C−A | C−B | B selected | C selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| 12 high residual | 0.710087642 | 0.751171534 | 0.753907944 | +0.043820302 | +0.002736410 | 11/12 | 9/12 |
| 4 low controls | 0.807983771 | 0.835714161 | 0.816513062 | +0.008529291 | -0.019201100 | 4/4 | 3/4 |

Four of five pilot conditions passed:

1. C > A on high residual: pass.
2. C > B on high residual: pass.
3. At least 6/12 high targets select Layer 2: pass (9/12).
4. At most 2/4 low controls select Layer 2: **fail (3/4)**.
5. No infrastructure failure: pass.

The failure is localized to the selection gate. Residual extraction and residual
retrieval show measurable value on the high set, including +0.134054 on LSDream
Lead 5 and +0.082429 on Grabbitz 2, and beat compute-matched duplicate stacking
in aggregate. However, a strict positive-CLAP rule also accepts nontrivial gains
on three low-energy controls. The current selector therefore cannot distinguish
useful decomposition from added-patch score polishing reliably enough.

## Layer 2 usefulness diagnostics

Arm C selected 12/16 overall. Among those, mean improvement was +0.046663,
median +0.046563, range +0.004901 to +0.134054, and 11/12 (91.7%) improved by
at least +0.005. Five selected stacks reused the same base preset identity, but
zero had identical final candidate-state hashes. No selected C layer had a
relative gain at or below -30 dB, improvement below +0.001, or the measured
noise/tail flag. Layer 2 gain ranged -17.752 to -4.541 dB (p50 -11.081 dB),
relative gain ranged -16.712 to +13.534 dB (p50 -3.950 dB), and offsets covered
the full allowed -25 to +25 ms range (p50 0 ms).

Arm B selected 15/16; 12/15 used the same base preset and 7/15 produced the same
candidate-state hash as Layer 1. This confirms that naive duplicate stacking is
a strong control and often duplicates Layer 1. Arm C reduced exact duplication,
but did not calibrate selection on low residuals.

Pilot wall time including match plus audio-only polish was Arm B mean 40.987 s,
p50 38.609 s, p95 70.245 s; Arm C mean 33.405 s, p50 33.328 s, p95 45.567 s.
Mix-only polish averaged 0.246 s for B and 0.256 s for C; synthesis/matching is
the dominant cost.

## Full A/B, adoption, and feasibility decision

The full 99-file A/B was not run. The prompt explicitly prohibits it when any
pilot condition fails. Therefore no full-suite Serum 1/Serum 2 decomposition
means, selection rate, benefit correlation, or adoption thresholds are claimed.
Historical or pilot-only gains were not substituted for a full result.

Two-layer decomposition is **not validated**. The ten-condition full feasibility
gate is not evaluated because its required full Arm B does not exist. The
experimental modules remain opt-in benchmark infrastructure only; normal match
output stays one preset and structural search remains off by default.

## Regression safety

* frozen retrieval: **0.775@1 / 0.890@5**, pass;
* frozen corpus: all 200 manifest hashes unchanged, pass;
* invariance: **0.403333@1 / 0.596667@5**, pass;
* full automated suite: **138 passed**, one upstream deprecation warning;
* `scripts/verify_visual_redesign.py`: **pass** (`gate_pass=true`);
* `scripts/verify_milestone4_ui.py`: **pass**
  (`MILESTONE4_UI_GATE_PASS=true`).

## Limitations and next priority

This is a CLAP/BAM feasibility measurement, not blind listening. The fixed
Stage 3H source-group split reports 47 Serum 1-provenance AIFF and 52 Serum 2 WAV
targets, but all preserved Layer 1 recommendations are actually Serum 2; because
Layer 2 must follow Layer 1's synth family, this pilot measures Serum 2 stacks
only. Existing production preprocessing trims silence and caps the comparison
body but has no explicit learned reverb/delay-tail classifier. Stage 3I records
tail/noise diagnostics rather than claiming perfect separation.

The exact next priority is a narrowly scoped selection-calibration stage: derive
a residual-significance/complexity threshold from the measured high/low pilot,
require benefit beyond ordinary duplicate stacking or a calibrated minimum
delta, and rerun the same frozen 16 targets before any full benchmark. Do not
expand layer count, jointly optimize Serum parameters, start a neural surrogate,
or integrate stacks into the UI until that negative-control gate passes.

No Mac action is required. There is no distributable artifact, model, index,
preset, or relay change, so no artifact manifest is needed.

The single legal implementation version transition is **1.3.7 -> 1.3.8**.
