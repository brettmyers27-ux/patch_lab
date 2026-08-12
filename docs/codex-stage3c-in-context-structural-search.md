# Codex Prompt — Stage 3C: in-context structural search (run ON the Windows 11 / RTX 5070 PC)

Copy everything below into Codex **on the Windows PC**, from
`%USERPROFILE%\Documents\PatchLab\soundmatch`, after `git pull`.

---

Read `docs/ACCURACY_ROADMAP.md` §2 and §3, plus `docs/benchmarks/stage3a-report.md` and
`docs/benchmarks/stage3b-report.md` in full, before starting. §5 conventions apply unchanged:
numeric before/after on the *same* benchmarks, no silent scope cuts, stop at the reporting gate,
never automate a live DAW.

## Context: two failures with one shared cause

Stage 3A trained structural estimators on **whole-preset renders**, where the structural choice
is confounded with every other setting. All four lost to their most-common-value baselines.

Stage 3B removed the confound by fingerprinting each choice on a **controlled neutral patch**.
All four lost again, two of them scoring exactly 0.000000. Stage 3B's own diagnosis names the
reason: *"a fingerprint of one isolated choice is still domain-mismatched against a complete
target preset."*

Both stages made the same structural mistake in opposite directions: **they compared an isolated
component against a complete mixture.** A neutral render of one wavetable does not resemble a
finished preset that uses that wavetable, because filter, envelope, FX, and layering dominate the
descriptor. No amount of estimator engineering fixes a comparison between two different kinds of
thing.

**The fix is to stop estimating and start evaluating in context.** Write the candidate structural
value into the actual working patch, render the whole patch, and score the full mixture against
the full target — the identical comparison the existing matcher already trusts for continuous
parameters. There is then no domain gap at all, because both sides are complete presets.

## Why this is now affordable (from your own measurements)

Stage 3A measured a structural evaluation at 3.3041× an automation evaluation and confirmed a
persistent worker instance survives repeated state loads. Stage 3B then rendered every enumerated
candidate and recorded per-category wall clocks (`docs/benchmarks/stage3b-fingerprints.json`):

| Category | Candidates | Stage 3B wall clock |
|---|---:|---:|
| FX type | 16 | 1.80 s |
| Wavetable | 330 | 12.06 s |
| Noise sample | 230 | 15.99 s |
| **Subtotal** | **576** | **~30 s** |
| Mod route | 4,906 | 548.13 s |

**An exhaustive in-context sweep of all three cheap categories costs roughly 30 seconds per
target.** The estimators in Stage 3A and 3B were built to avoid a cost that turns out to be
nearly free. Mod routes at ~9 minutes are the only category that genuinely needs narrowing.

## Phase 0 — Two decisive checks before building anything (hard gate)

Stage 3B's zeros may not be purely methodological. Both checks below are cheap and each one
separates "measured limitation" from "silently corrupted input." Run both and report both.

1. **Self-retrieval.** Query the Stage 3B fingerprint index with a fingerprint's *own* render, for
   a sample of ≥ 20 per category. Rank-1 must return itself. If it does not, the index has an
   ID-alignment defect and every Stage 3B conclusion built on it is void — report that
   immediately rather than proceeding.
2. **Fingerprint distinctness.** Stage 3B reported: *"Some factory states referenced
   original-machine external sample paths that are absent on this PC. Serum logged those missing
   paths while still loading and rendering every state... those renders may use Serum's fallback
   content."* If missing samples silently fell back to default content, many fingerprints in a
   category are near-identical and the category's estimator was measuring nothing — which would
   explain a 0.000000 far better than domain gap alone. Measure pairwise descriptor distances
   within each category and report the count of near-duplicate clusters, especially for **noise
   sample** (230 candidates, baseline 0.600, scored 0.000000) and **wavetable**.

Report both results plainly. If a category's candidates turn out to be largely indistinguishable
on this machine, say so — it is a resource-availability finding about this PC, and that category
must be excluded from Phase 1 rather than searched over meaningless options.

## Phase 1 — In-context coordinate search

Replace `core/structural_estimators.py`'s role in the search path. Do not delete the module or
its tests; it stays available and its Stage 3A/3B numbers stay on the record.

1. For each structural field, hold every other field at the current best candidate, then sweep
   **every** legal value that survived Phase 0 (not a top-K shortlist), writing each into the
   working patch via the proven Stage 3A reconstruction path, rendering, and scoring against the
   target with the existing scoring function.
2. Coordinate-descent order: **wavetable → FX type → noise**, then one repeat pass to catch
   interactions between fields. Report whether the second pass changed any field — if it never
   does, drop it and say so.
3. Hand the winning structure to the existing continuous optimizer, unchanged. This ordering is
   taxonomy **C18** (staged optimization) and matches Stage 3A's existing structure↔continuous
   handoff.
4. The current fixed 300-evaluation budget cannot absorb a 576-candidate sweep. Add an explicit
   **deep structural mode** with its own budget rather than starving the continuous stage; report
   the budget chosen and the measured per-target wall clock. Default-off remains correct until
   Phase 3 says otherwise.

## Phase 2 — Mod routes, narrowed by measurement rather than similarity

Routes are the one category too expensive to sweep exhaustively (4,906 ≈ 9 minutes per target),
and Stage 3B showed route identity does not reduce to a short trajectory descriptor compared
across domains.

Narrow by **measuring what the target actually does**, per taxonomy **B9/A8** and the project's
founding LFO insight:

1. Measure which properties of the target move periodically — amplitude, spectral centroid, pitch
   — and at what rate, using the existing tempo/modulation detection rather than a new analyzer.
2. Keep only routes whose destination could plausibly produce the movement actually observed. A
   target with no periodic centroid movement does not need any route to a filter-frequency
   destination; that alone should eliminate most of the 4,906.
3. Evaluate the surviving set in context, exactly as Phase 1 does.
4. **If narrowing does not get the candidate set under ~300, report the achieved count and skip
   routes for this stage** rather than consuming the whole budget on one field. A documented skip
   beats a blown budget.

## Phase 3 — A/B on the corrected subset

Run `scripts/benchmark_suite.py` on the identical 99 BAM files and seed `20260802`. Use Stage 3B's
**corrected** classification (47 Serum 1 / 52 Serum 2) — this stage cannot affect Serum 1 targets,
so the Serum 2 subset is the meaningful measurement.

- **A** — Stage 2B adopted stack, structural search off. Reuse the recorded baselines: whole-set
  BAM mean **0.784226**; corrected 52-file Serum 2 subset mean **0.794525**, median 0.804774,
  minimum 0.647449; retrieval@1 0.785 / @5 0.895; invariance@1 0.403333 / @5 0.596667.
- **B** — Stage 2B stack + in-context structural search enabled.

**Adopt only if**: Serum 2 subset BAM mean improves materially over 0.794525; whole-set BAM mean
does not regress below 0.784226; retrieval@1 ≥ 0.785; invariance@1 ≥ 0.403333; no new test
failures; both UI verification gates pass. On adoption, structural search becomes default-on for
Serum 2 targets in deep mode.

If in-context search *still* does not improve BAM despite having no domain gap and no estimator
in the loop, that is the most informative negative result this project has produced: it would mean
the limit is the **scoring objective**, not the search or the guidance, and Stage 4 should target
the objective (taxonomy B14's composite sub-scores) rather than more search. Report it that way.

## Phase 4 — Handoff

1. Commit code, `docs/benchmarks/*.json`, and an updated roadmap §1/§4 reflecting the measured
   ladder. Push. Version policy: one patch step per commit, single-digit components.
2. **Do not commit model or index artifacts, and do not touch the relay.** If anything is adopted,
   write `docs/benchmarks/stage3c-artifact-manifest.json` with each adopted artifact's filename,
   byte size, SHA-256, and install destination. The Mac holds the relay credentials and publishes
   from that manifest.
3. Final report: Phase 0's two check results, per-category candidate counts actually searched,
   per-target wall clock in deep mode, the full A/B on both whole set and corrected Serum 2
   subset, named limitations with reasons, and an explicit recommendation on whether the limit is
   now search, guidance, or the scoring objective. Stop at this gate.
