# Codex Prompt — Stage 3F: freeze the retrieval corpus, then re-adjudicate Stage 3D
(run ON the Windows 11 / RTX 5070 PC)

Copy everything below into Codex **on the Windows PC**, from
`%USERPROFILE%\Documents\PatchLab\soundmatch`, after `git pull`.

---

Read `docs/benchmarks/stage3d-report.md` and `docs/benchmarks/stage3e-report.md` in full before
starting. §5 conventions from `docs/ACCURACY_ROADMAP.md` apply: numeric before/after, no silent
scope cuts, stop at the gate.

## What Stage 3E actually found

Stage 3E ruled out GPU/embedding nondeterminism and structural-search coupling completely — both
arms produced bit-identical retrieval results across five repeats. But it found the real cause,
and it's a measurement bug, not a search bug: `FactoryRenderer` (`scripts/benchmark_suite.py:376`)
caches each factory preset's render under `detail_dir / "factory-renders-by-catalog-id-v2"`, and
every stage invocation uses a fresh `detail_dir`. Two presets (196, 216) render *differently*
across fresh Serum 2 host sessions — Stage 3E's own words: *"ID 216 produced two distinct
CLAP/retrieval modes across five fresh hosts (four misses and one correct result)."* Preset 190
similarly flipped. This is Serum 2 host-initialization/state-order sensitivity, unrelated to
anything PatchLab's code does.

The consequence: Stage 3D's "regression" (0.785 arm A vs 0.780 arm B) was two different render-
cache generations being compared as if they were the same measurement — arm A reused Stage 2B's
old cached number, arm B was rendered fresh this session. It was never a fair same-conditions
comparison, and it never will be as long as every stage re-renders its own factory corpus from
scratch. This has been silently possible in every retrieval measurement since Stage 2B; it just
happened to land on the adoption boundary this time.

## The fix: stop re-rendering the factory corpus every stage

The renderer already caches correctly *within* a run. The bug is that nothing persists that cache
*across* runs. Point it at one permanent location instead of a fresh one per stage.

## Phase 1 — Freeze a canonical factory-render corpus

1. Render the full factory-preset selection used by `_select_factory_presets` (seed `20260802`,
   the existing `factory_count`/`invariance_count` union) exactly once, into a **persistent,
   version-controlled-by-manifest-only location** — e.g. `data/benchmark-cache/
   factory-renders-by-catalog-id-v2/`. This directory holds real audio and must stay `.gitignore`d,
   same as every other render cache in this project; only a small manifest (preset IDs, render
   count, seed, git commit this was frozen at) is committed.
2. Change `run_retrieval_suites` (or add a CLI flag, your call on the cleanest approach) so it
   accepts this frozen cache location instead of always deriving the render path from `detail_dir`.
   Every future benchmark invocation — this stage and every stage after it — should default to
   reading this frozen corpus rather than rendering its own.
3. Verify the fix does what it claims: run retrieval-only twice against the frozen corpus and
   confirm bit-identical results (Stage 3E already proved fixed-WAV embeddings are stable; this
   just confirms the plumbing reuses the frozen files rather than re-rendering).
4. Specifically re-render presets 190 and 216 a second time from a *fresh, separate* host session
   and diff against what's now frozen, to directly confirm the frozen copies are exactly what get
   reused going forward rather than silently re-rendered.

## Phase 2 — Produce the one canonical retrieval number

1. Run retrieval-only against the frozen corpus once. Since Stage 3E proved retrieval scoring has
   no dependency on the structural-search flag, this single number is valid for both arms — do not
   run it twice under different flag settings.
2. Report retrieval@1/@5 and compare against both historical reference points: the old reused
   0.785/0.895, and Stage 3D/3E's fresh 0.780/0.895. State which one the frozen corpus reproduces
   and why that resolves the discrepancy.

## Phase 3 — Re-adjudicate Stage 3D's adoption decision

Stage 3D's BAM and invariance results were measured on the synthesis side (deep structural search
producing and scoring emitted presets), not the retrieval side, and Stage 3E already proved
structural search doesn't touch retrieval scoring. **Those numbers do not need to be re-measured.**
Combine them with Phase 2's corrected, trustworthy retrieval number:

- Whole-set BAM mean: 0.784226 → 0.785492 (Stage 3D, unaffected by this bug)
- Serum 2 subset BAM mean: 0.794525 → 0.796935 (Stage 3D, unaffected by this bug)
- Invariance@1/@5: unchanged at 0.403333/0.596667 (Stage 3D, unaffected by this bug)
- Retrieval@1/@5: **Phase 2's frozen-corpus number** (replaces the disputed 0.780)

**Adopt deep structural search for FX, wavetable, and noise categories** (not mod routes — Stage
3D's target-motion narrowing never got the route candidate count under budget, so routes remain
excluded regardless of this stage's outcome) **if**: Phase 2's retrieval@1 ≥ 0.785; the already-
established BAM and invariance numbers still clear their gates (they do, per above); no new test
failures; both UI verification gates pass.

If Phase 2's frozen-corpus retrieval@1 still comes in below 0.785, that is now a trustworthy,
reproducible "no" rather than a measurement artifact — report it as such and leave structural
search opt-in.

On adoption: flip the relevant default (e.g. `PATCHLAB_SERUM2_STRUCTURAL_SEARCH` or its equivalent
default in the shipped code path) so Serum 2 synthesis uses deep in-context search by default,
scoped to FX/wavetable/noise only. This is a code-default change, not a new shippable artifact —
no relay action is needed either way, since Stage 3D's structural search reuses assets already
published in Stage 2B.

## Phase 4 — Handoff

1. Commit code, the frozen-corpus manifest (not the render audio itself), `docs/benchmarks/*.json`,
   and an updated roadmap §1/§4. Push. Version policy: one patch step per commit, single-digit
   components.
2. Final report: the frozen-corpus verification results, Phase 2's canonical retrieval number and
   which historical figure it matches, the re-adjudicated adoption decision with full numeric
   justification, and confirmation of whether the default was flipped. Stop at this gate.
