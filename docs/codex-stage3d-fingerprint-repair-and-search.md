# Codex Prompt — Stage 3D: repair mod-route/noise fingerprints, then resume in-context search
(run ON the Windows 11 / RTX 5070 PC)

Copy everything below into Codex **on the Windows PC**, from
`%USERPROFILE%\Documents\PatchLab\soundmatch`, after `git pull`.

---

Read `docs/ACCURACY_ROADMAP.md` §2 and §3, plus `docs/benchmarks/stage3b-report.md` and
`docs/benchmarks/stage3c-report.md` in full, before starting. §5 conventions apply unchanged:
numeric before/after on the *same* benchmarks, no silent scope cuts, stop at the reporting gate,
never automate a live DAW.

## Context and a correction to the previous prompt

Stage 3C's Phase 0 gate ran exactly as intended and found a real problem — but it stopped more
than it needed to, because the prior prompt's wording was ambiguous. The actual per-category
results (`docs/benchmarks/stage3c-phase0.json`):

| Category | Exhaustive self-retrieval | Zero-descriptor rows | Gate |
|---|---:|---:|---|
| FX type | 16/16 | 0 | **Pass** |
| Wavetable | 330/330 | 0 | **Pass** |
| Mod route | 3,682/4,906 | 131 | Fail |
| Noise sample | 220/230 | 0 | Fail |

**FX and wavetable are clean.** Zero integrity problems, zero near-duplicates below `1e-4`. Their
Stage 3B domain-gap conclusions (FX tied its baseline at 0.207; wavetable lost badly, 0.008 vs
0.353) are trustworthy, real findings. There is no reason to re-verify them and no reason to defer
in-context search on them any further — proceed with Phase 1 below for these two categories
immediately, independent of everything else in this prompt.

Only **mod route** and **noise sample** are blocked, and only they need repair.

## What the numbers already show about the two failures

Don't re-derive these from scratch — they're visible in `stage3c-phase0.json` and worth reading
before writing any code:

**Mod route** is the one with a real, structural problem: 131 rows have literally no descriptor
signal, and at a loose `1e-4` distance threshold, 2,528 of 4,906 rows — over half — collapse into
238 duplicate clusters (largest cluster: 289 members). Stage 3B's own methodology notes why this
is plausible: routes are covered by only 333 deterministic carrier patches (many source/destination
pairs share a carrier), and the descriptor is a short amplitude/centroid/flux trajectory at one
fixed depth. Two concrete hypotheses to check, not assume:

1. Many distinct "source" IDs may default to the same underlying modulator shape/rate when nothing
   else about them is set — meaning many *labeled* routes may be genuinely *audibly identical*, not
   a bug. If true, that's not something to fix, it's something to collapse.
2. Some destinations may not respond to depth-32 modulation in a way the amplitude/centroid/flux
   descriptor captures (e.g. a pan or stereo-width destination, or a destination requiring a larger
   depth to move audibly) — a descriptor blind spot, which *is* a bug in what's being measured.

**Noise sample** is a small, localized problem: 220/230 pass cleanly, only 10 fail, and the largest
duplicate cluster is 2 members. Stage 3B's own report already named a specific, concrete suspect:
*"Some factory states referenced original-machine external sample paths that are absent on this
PC... those renders may use Serum's fallback content."* Check whether the 10 failing/duplicate rows
are exactly the ones with missing external sample paths before looking anywhere else.

## Phase 1 — Resume in-context search for FX and wavetable (no gate, no repair needed)

Implement Stage 3C's original Phase 1 design for these two categories only:

1. Hold every other structural field fixed; sweep **every** legal FX type (16) and every legal
   wavetable (330) by writing each into the working patch via the proven reconstruction path,
   rendering, and scoring in-context against the target — full mixture vs. full mixture, the same
   comparison the matcher already trusts for continuous parameters.
2. Hand the winning structure to the existing continuous optimizer, unchanged.
3. Add the deep structural mode with its own evaluation budget (the fixed 300-evaluation budget
   cannot absorb this). Report the budget chosen and measured per-target wall clock.

## Phase 2 — Diagnose and repair noise-sample fingerprints

1. Pull the 10 failing/duplicate rows from `data/stage3c/` diagnostics and cross-reference against
   the missing-external-sample-path log Stage 3B produced. Report the overlap explicitly: how many
   of the 10 are exactly the samples with missing source files.
2. **If the overlap is total or near-total**: this is resource unavailability on this machine, not
   a fingerprint bug. Exclude those specific rows from the noise candidate set (a small, documented
   exclusion — do not touch the other 220), regenerate the index without them, and move on.
3. **If the overlap is partial**: investigate the remainder on its own terms (silent/near-silent
   source samples, duplicate source files under different names, or a genuine descriptor gap) and
   report what you find.
4. Re-run the Phase 0 self-retrieval and distinctness check for noise only. It must pass before
   noise enters Phase 4.

## Phase 3 — Diagnose and repair mod-route fingerprints

1. **Zero-descriptor rows (131).** Sample several. Render each route's carrier with and without
   the modulation active and diff the audio directly — is there truly no audible change, or does
   the change exist but fall outside what the amplitude/centroid/flux descriptor measures? If the
   latter, extend the descriptor to capture what's actually moving (e.g. stereo width/pan, or a
   longer-window measure for slow destinations) rather than discarding the rows.
2. **Duplicate clusters (238 at `1e-4`, 2,528 member rows).** Sample several clusters. For each,
   compare the underlying source modulators' actual settings (shape, rate, phase) as loaded. If
   cluster members genuinely share identical modulator behavior at the fixed depth used, that is a
   real finding, not a bug: **collapse each cluster to one representative candidate** rather than
   treating its members as separately searchable. Report the resulting deduplicated candidate count
   — this is likely to shrink the ~4,906-route search space substantially, which directly reduces
   how much narrowing Phase 5 needs to do.
3. Regenerate the route fingerprint index reflecting whatever combination of descriptor fixes and
   deduplication the diagnosis supports. Re-run the Phase 0 self-retrieval and distinctness check
   for routes only. It must pass before routes enter Phase 5.
4. If some portion of the 131 or the clustered rows resist repair (a destination that is genuinely
   unmeasurable with reasonable effort), exclude exactly those and report the count — a partial,
   documented category is legitimate; do not block the whole category on an irreducible remainder.

## Phase 4 — Noise in-context search (only if Phase 2 passed its gate)

Same method as Phase 1, over the repaired noise candidate set.

## Phase 5 — Mod-route narrowing and search (only if Phase 3 passed its gate)

1. Start from the deduplicated candidate count from Phase 3, not the original 4,906.
2. If that count is still too large for direct in-context sweep at reasonable cost, narrow further
   by measuring the target's own periodic movement (amplitude, spectral centroid, pitch, using the
   existing tempo/modulation detection) and keeping only routes whose destination could plausibly
   produce what's actually observed — taxonomy **B9/A8**, the same founding-insight approach used
   elsewhere in this project. A target with no periodic centroid movement doesn't need any route to
   a filter-frequency destination.
3. If the surviving set is still impractically large, report the achieved count and skip routes for
   this stage rather than consuming the whole budget on one field — a documented skip beats a blown
   budget.
4. Evaluate the surviving set in-context, same method as Phase 1.

## Phase 6 — A/B on whatever categories reached search

Run `scripts/benchmark_suite.py` on the identical 99 BAM files and seed `20260802`, using Stage
3B's corrected classification (47 Serum 1 / 52 Serum 2):

- **A** — Stage 2B adopted stack, structural search off. Reuse recorded baselines: whole-set BAM
  mean **0.784226**; corrected 52-file Serum 2 subset mean **0.794525**, median 0.804774, minimum
  0.647449; retrieval@1/@5 0.785/0.895; invariance@1/@5 0.403333/0.596667.
- **B** — Stage 2B stack + in-context structural search enabled for whichever categories reached
  Phase 4/5 successfully. State plainly which categories are included in arm B and which were
  excluded and why (e.g. "FX and wavetable only; noise excluded pending further repair").

**Adopt only if**: Serum 2 subset BAM mean improves materially over 0.794525; whole-set BAM mean
does not regress below 0.784226; retrieval@1 ≥ 0.785; invariance@1 ≥ 0.403333; no new test
failures; both UI verification gates pass. Partial adoption (e.g. FX+wavetable search on, routes
still off) is a legitimate outcome — state it plainly, same as every prior stage's convention.

If arm B still does not improve BAM despite clean fingerprints and no domain gap, that is the
signal from the last two prompts finally landing cleanly: the limit is the **scoring objective**,
not search or guidance, and Stage 4 should target taxonomy **B14**'s composite sub-scores rather
than more search. Report it that way if it happens.

## Phase 7 — Handoff

1. Commit code, `docs/benchmarks/*.json`, and an updated roadmap §1/§4 reflecting the measured
   ladder. Push. Version policy: one patch step per commit, single-digit components.
2. **Do not commit model or index artifacts, and do not touch the relay.** If anything is adopted,
   write `docs/benchmarks/stage3d-artifact-manifest.json` with each adopted artifact's filename,
   byte size, SHA-256, and install destination. The Mac holds the relay credentials and publishes
   from that manifest.
3. Final report: the noise/route repair diagnosis and what it found (bug vs. genuine
   indistinguishability), the deduplicated route candidate count, which categories reached search
   and which didn't and why, per-target wall clock in deep mode, the full A/B on both whole set and
   corrected Serum 2 subset, and an explicit recommendation on whether the limit is now search,
   guidance, or the scoring objective. Stop at this gate.
