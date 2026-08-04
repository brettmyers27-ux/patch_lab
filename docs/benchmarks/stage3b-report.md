# Stage 3B — controlled structural-fingerprint report

Stage 3B rendered every enumerated controlled structural choice successfully, but none of
the four direct estimators beat its unchanged Stage 3A most-common-value top-1 baseline.
The gated 99-file B arm was therefore not run. Structural search remains opt-in and disabled
by default; Stage 2B remains production.

## Verdict

| Estimator | Held out | Top-1 | Fixed common top-1 | Top-5 | Fixed common top-5 | Decision |
|---|---:|---:|---:|---:|---:|---|
| FX type | 140 | 0.207143 | 0.207143 | 0.728571 | 0.728571 | Drop (tie is not a win) |
| Wavetable | 119 | 0.008403 | 0.352941 | 0.016807 | 0.512605 | Drop |
| Mod route | 140 | 0.000000 | 0.114286 | 0.007143 | 0.257143 | Drop |
| Noise sample | 115 | 0.000000 | 0.600000 | 0.000000 | 0.686957 | Drop |

No estimator artifact is adopted. The private combined NPZ contains 16 FX, 330 wavetable,
4,906 observed full-route, and 230 noise fingerprints. All 5,482 states loaded and rendered;
there were zero render failures. Its byte size and SHA-256 are recorded in
`stage3b-artifact-manifest.json`, but it must not be published.

## Controlled bases and timing

- Wavetable uses preset 699 as a dry C4 carrier: Osc0 at unity, Osc1/2/noise silent,
  filter off, FX empty, and modulation depths zero. Only the Osc0 table path varies.
- Noise uses the same preset with Osc0/1/2 silent and noise at unity. Only the sample path
  varies.
- FX uses the small, validated one-slot topology in preset 4423. Only slot 0's integer type
  changes; all effect setting leaves remain fixed.
- Mod routes use ModSlot0 at depth 32 and category-specific amplitude/brightness trajectory
  fingerprints. Serum rejects a destination that is absent from the loaded topology, so one
  universal state cannot encode mutually exclusive FX modules. The complete observed set is
  covered by 333 deterministic carriers keyed by destination module type/ID. Those carriers
  retain their fixed validated module topology (and therefore can retain fixed FX coloration),
  because rewriting the topology list makes Serum reject the state.

Phase 1 category wall clocks were 1.802 s (FX), 12.058 s (wavetable), 548.135 s (routes),
and 15.987 s (noise), totaling 577.982 s. Phase 2 rendered all 710 exact Stage 3A presets in
126.418 s and evaluated the identical preset-ID-modulo-5 holdout. The full Phase 1+2 command
took 749.164 s; the remaining measured time was setup, fingerprint persistence, descriptor
calculation, and scoring.

Some factory states referenced original-machine external sample paths that are absent on this
PC. Serum logged those missing paths while still loading and rendering every state; this is a
named limitation because those renders may use Serum's fallback content. It did not create a
benchmark failure or justify relabeling a result.

## Classification correction and gate

The old helper recognized explicit synth markers, then assigned every unmarked name to Serum 2.
The fixed 99-file corpus preserves its source provenance in format: 47 original Serum 1 `.aif`
bounces and 52 later Serum 2 `.wav` bounces. Explicit markers remain authoritative. No file was
added, removed, or rescored.

The adopted Stage 2B A result remains 0.784226 mean / 0.793119 median / 0.541550 minimum on all
99 files. Its corrected 52-file Serum 2 subset is 0.794525 mean / 0.804774 median / 0.647449
minimum. Whole-set retrieval is 0.785/0.895 at 1/5 and invariance is 0.403333/0.596667. The
corresponding factory Serum 2 subsets are 0.812500/0.898438 (128 retrieval cases) and
0.460784/0.612745 (204 invariance variants).

Phase 3's classification audit/calculation took about one second. Phase 4 took zero render time:
the prompt requires arm B to be skipped when no estimator passes, preventing a knowingly unguided
repeat of Stage 3A's regression.

## Limitations and decision

Controlled fingerprints remove training-example confounding, but a fingerprint of one isolated
choice is still domain-mismatched against a complete target preset. Wavetable and noise identity
are especially masked by oscillator position, envelopes, filters, modulation, and other layers.
Route identity also cannot be reduced reliably to the short amplitude/centroid/flux trajectory
used here, and destination topology requires multiple carriers. These are measured limitations,
not silent scope cuts.

Because zero estimators passed, there is no Stage 3B B score and no adoption case to evaluate.
The production default is unchanged. This report stops at the Stage 3B gate.
