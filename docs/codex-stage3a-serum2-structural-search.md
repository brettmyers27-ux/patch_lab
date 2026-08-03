# Codex Prompt — Stage 3A: complete the Serum 2 search space (run ON the Windows 11 / RTX 5070 PC)

Copy everything below into Codex **on the Windows PC**, from
`%USERPROFILE%\Documents\PatchLab\soundmatch`, after `git pull`.

---

Read `docs/ACCURACY_ROADMAP.md` in full before starting — §2 (the master principle), §3 items
B9, B12, C17–C20, and §5 (prompt conventions) govern this work. §5 applies unchanged: numeric
before/after on the *same* benchmarks, no silent scope cuts, stop at the reporting gate, never
automate a live DAW.

## Context: where accuracy actually stands

Stage 2B is adopted and is the baseline this stage must beat. On the 99 user-authorized BAM
files, seed `20260802`:

- BAM mean **0.784226**, median 0.793119, minimum 0.541550
- factory retrieval@1 **0.785**, @5 0.895
- invariance@1 **0.403333**, @5 0.596667

Stage 2 taught an expensive lesson worth restating: it fine-tuned the encoder, rebuilt the
entire embedding world, and moved BAM mean only +0.0138 while moving retrieval@1 +0.225. The
*ears* improved dramatically; the *recreation* barely moved. That asymmetry is the whole
motivation for Stage 3 — the bottleneck is no longer perception, it is the ability of the
optimizer to actually construct the patch it is looking for.

## The specific gap this stage closes

Serum 2 exposes 2,623 automatable parameters, and `core/serum2_mapping.py` maps roughly 40% of
a preset's weighted content through them. The unmapped remainder is not a long tail of
knobs — it is exactly **five categories, all of them discrete "choose one from a list"
decisions that Serum 2 never exposed to host automation at all**
(`core/serum2_mapping.py:469-527`, evidence in `data/models/serum2_surface_investigation.json`):

| Category | Preset field | Why automation cannot reach it |
|---|---|---|
| Wavetable selection | `relativePathToWT` | No file/name/index selection parameter |
| Custom wavetable data | `embeddedWTData` | No custom-data loading hook in either host |
| Noise sample selection | `relativePathToNoiseSample` | No sample selection parameter |
| Mod matrix routing | `ModSlot*.source`, `.destModuleID`, `.destModuleParamID`, `.destModuleParamName`, `.destModuleTypeString` | No live source/destination parameter |
| FX slot topology | `FXRack*.type` | No live effect-type parameter |

Continuous settings *within* a chosen structure (filter cutoff, effect wet, mod depth via
`Mod N Amount`) are already automatable and already searched. What cannot currently be searched
is *which* effect sits in a slot, *which* wavetable an oscillator plays, and *what a mod slot is
wired to*.

**The enabling mechanism already exists and is already proven.** `reconstruct_partial_vstpreset`
(`core/serum2_state_reconstruct.py:290`) overlays arbitrary preset fields — including all five
categories above — onto a live plug-in state template and produces a loadable VST3 state that
Serum 2's DSP engine honors. Every one of the 710 Serum 2 presets in the render library was
rendered through exactly this path. The mechanism is not in question; what is missing is a
search loop that *proposes* structural values and evaluates them.

## Scope discipline: what "100%" means here, precisely

The user's goal for this stage is stated as: *the search process genuinely considers every
possible Serum 2 configuration.* Enumerating that space is not merely slow, it is impossible,
and the prompt must not pretend otherwise. Measured across the 710-preset Serum 2 library:

- 64 mod slots per preset; **271** distinct sources and **139** distinct destinations observed
  → on the order of 37,000 possible routings *per slot*
- 3 FX racks holding ~7 effect slots total; **16** distinct effect types observed
- **241** distinct wavetables and **102** distinct noise samples referenced

The mod matrix alone is ~37,000^64 combinations. Exhaustive search is off the table forever.

**Therefore "100% coverage" in this stage means reachability, not enumeration:**

1. **Completeness** — no structural field is excluded from what the optimizer may write. Any
   configuration Serum 2 can hold, PatchLab can propose, render, score, and emit.
2. **Guidance** — structural choices are *measured from the target audio* wherever the roadmap's
   master principle allows (§2, "transcribe, don't search"), so the search starts near the
   answer rather than wandering.
3. **Locality** — discrete search is spent only on the shortlist that measurement could not
   resolve.

Report against that definition. Claiming exhaustive coverage would be false; claiming that no
configuration is structurally locked out is the real, checkable win.

## Phase 0 — Measure the cost (gate; everything downstream depends on this)

Nothing about the design can be chosen before this number exists.

1. Benchmark, on this machine, the median wall-clock of a single Serum 2 evaluation under both
   paths, ≥ 50 trials each, same preset, same note, same duration:
   - **(a) automation path** — set N automatable parameters on an already-loaded plug-in, render.
   - **(b) structural path** — `reconstruct_partial_vstpreset` → write state → load state into
     the plug-in → render.
2. Report the ratio (b)/(a) explicitly, plus whether state loading can be amortized (does the
   plug-in instance survive repeated `setState` calls, or is re-instantiation required?).
3. Measure how many structural evaluations per minute this machine sustains with the existing
   four-process render pool.

**This number decides Phase 4's design.** State the decision rule in the report:
if a structural evaluation costs < ~5× an automation evaluation, direct discrete search over a
measurement-narrowed shortlist is viable. If it costs ≫ 20×, say so plainly and state that a
learned surrogate (roadmap §4 Stage 3a) becomes a prerequisite rather than an optional
follow-on — do not quietly build a search that cannot finish.

## Phase 1 — Enumerate the structural vocabulary

Completeness cannot be claimed without knowing the full legal value set for each category. Build
`core/serum2_structural_space.py` exposing, for each of the five categories, the enumerated legal
values with stable IDs.

1. Harvest observed values from all 710 library presets (the counts above are the floor, not the
   ceiling — they are what *factory presets happen to use*).
2. Extend to what Serum 2 actually *offers*: enumerate the installed wavetable and noise-sample
   folders on disk, and enumerate FX types and mod source/destination IDs from the plug-in's own
   value strings where readable.
3. Where the true legal set cannot be established from either source, say so and record the
   observed set as a documented lower bound with its size — that is an honest limitation, not a
   failure.
4. Persist as `data/models/serum2_structural_space.json` with provenance per entry
   (`observed_in_presets` vs `enumerated_from_install`).

Report the final size of each category's value set and which source established it.

## Phase 2 — Prove structural mutation actually changes the sound (hard gate)

This is Stage 2B's "prove 25 presets render" lesson applied to structure. Do not proceed to
expensive work on an unproven mechanism.

For each of the five categories, take ≥ 10 real library presets, mutate **only** that one
structural field to a different legal value, reconstruct, render, and verify:

1. the state loads without rejection,
2. the rendered audio **differs measurably** from the unmutated render (report the distance), and
3. the change is *directionally sensible* — e.g. swapping an FX slot to a reverb type lengthens
   the measured decay tail; repointing a mod slot to `kParamFreq` produces filter movement.

Report per category: attempted, loaded, audibly-changed, and directionally-correct counts.

**If a category fails to produce audible change, stop and report it.** A category that silently
does nothing is a structural finding about Serum 2 state handling that outranks the rest of this
stage — `embeddedWTData` and `relativePathToNoiseSample` are the two most likely to disappoint,
since they may require file-system resources the reconstructed state only references by path.

## Phase 3 — Transcribe structure from audio (the leverage)

This is where the roadmap's master principle does the heavy lifting, and it implements taxonomy
items B9, B12, C17, and C20. Build `core/structural_estimators.py`. For each category, produce a
*ranked shortlist* with calibrated confidence, not a single guess:

1. **FX type** — effects have listenable fingerprints. Reverb/delay show as decay tails separable
   from the dry body (taxonomy A7); chorus/flanger/phaser show as characteristic comb or notch
   motion; distortion shows as harmonic generation. Train on renders you can generate freely:
   take library presets, apply each of the 16 FX types at varied settings, learn type from audio.
2. **Wavetable** — build the wavetable timbre index the roadmap already specifies (**C17**):
   render every enumerated wavetable on one neutral patch, fingerprint it, and match the target's
   harmonic content directly. This replaces "inherit the seed's wavetable" with a lookup.
3. **Mod routing** — a mod slot's *audible* signature is periodic movement of a specific
   property. Measure the target's amplitude, spectral-centroid, and pitch trajectories; a
   periodic wiggle in centroid at a tempo-synced rate implies an LFO→filter-frequency routing
   (this is the user's founding insight, §2, applied to routing rather than scoring). Use the
   detected BPM (taxonomy A8) so the proposed rate is a synced division.
4. **Noise sample** — match against fingerprints of the enumerated noise samples using the HPSS
   noise component (taxonomy B11).

Report held-out top-1 and top-5 accuracy per estimator against ground truth taken from library
presets whose true structure is known. **An estimator that does not beat picking the most common
value is worthless — report that comparison explicitly and drop any estimator that fails it.**

## Phase 4 — The structural search loop

Extend the existing matcher rather than building a parallel one. Staged, per taxonomy **C18**:

1. **Propose** — estimators supply a ranked shortlist per structural field (top-K, K justified by
   Phase 0's cost measurement, not chosen arbitrarily).
2. **Search structure** — discrete search over the shortlist combinations, evaluated through the
   Phase 2 reconstruct→render→score path. Search one category at a time in a fixed order (source
   and wavetable first, then FX topology, then mod routing) rather than jointly — **C18** again.
3. **Search continuous** — hand the winning structure to the existing continuous optimizer for
   the automatable parameters within it. This is the current CMA-ES path, unchanged.
4. **Iterate** — one further structure↔continuous round if the budget from Phase 0 allows.

Structural fields the estimators cannot rank must still be *reachable*: leave them exposed to the
search at low priority rather than pinning them to the seed's value. Document any field that ends
up effectively pinned, and why.

## Phase 5 — Verdict on the same benchmarks

Run `scripts/benchmark_suite.py` on the identical 99 BAM files and identical seed `20260802`,
changing only the matcher stack:

- **A** — Stage 2B adopted stack. Reuse the recorded baseline; do not re-derive it.
- **B** — Stage 2B stack + structural search enabled for Serum 2.

Report BAM mean/median/min, factory retrieval@1/@5, invariance@1/@5, and **a Serum 2-only BAM
subset**, since this stage cannot affect Serum 1 targets and a whole-set mean will dilute the
effect into invisibility. Report the Serum 2 subset size — if it is small, say so, and treat the
result as directional rather than conclusive.

Also report, for the Serum 2 subset: mean weighted parameter coverage of emitted presets, before
and after. That number is this stage's headline claim and must move from ~40% toward ~100%.

**Adopt only if**: Serum 2-subset BAM mean improves materially; whole-set BAM mean does not
regress; factory retrieval@1 ≥ 0.785; invariance@1 ≥ 0.403333; no new test failures; UI gates
`scripts/verify_visual_redesign.py` and `scripts/verify_milestone4_ui.py` still pass.

If structural search improves coverage but *not* BAM score, that is a real and publishable
finding — it would mean the emitted presets are structurally richer without sounding closer,
which points at the scoring objective rather than the search. Report it as such rather than
burying it.

## Phase 6 — Handoff

1. Commit code, `docs/benchmarks/*.json`, and an updated roadmap §1/§4 reflecting the measured
   ladder. Push. Version policy: one patch step per commit, single-digit components.
2. **Do not commit model or index artifacts, and do not touch the relay.** If this stage produces
   any new shipped artifact (a wavetable fingerprint index, estimator weights), write
   `docs/benchmarks/stage3a-artifact-manifest.json` with each adopted artifact's filename, byte
   size, SHA-256, and install destination. The Mac holds the relay credentials and publishes from
   that manifest.
3. Final report: every phase's real numbers, Phase 0's cost ratio and the design decision it
   forced, named limitations with reasons, wall clock per phase, and an explicit recommendation on
   whether the neural surrogate (roadmap §4 Stage 3a) is now a prerequisite for further progress.
   Stop at this gate.
