# Codex Prompt — Stage 3E: verify the Stage 3D retrieval regression before touching search again
(run ON the Windows 11 / RTX 5070 PC)

Copy everything below into Codex **on the Windows PC**, from
`%USERPROFILE%\Documents\PatchLab\soundmatch`, after `git pull`.

---

Read `docs/benchmarks/stage3d-report.md` before starting. This is a narrow, cheap verification —
not a new search design. Do not re-run deep structural search or the BAM suite. §5 conventions
from `docs/ACCURACY_ROADMAP.md` still apply: numeric before/after, no silent scope cuts, stop at
the gate.

## Why this check comes before any further design work

Stage 3D found a real, positive result — Serum 2 BAM mean improved 0.794525 → 0.796935 with clean
fingerprints and no scoring-objective concerns — but withheld adoption because arm B's
retrieval@1 measured 0.780 against a 0.785 gate. Before spending more engineering on search
design, that specific number deserves a second look, for two concrete reasons:

1. **The gap is exactly one preset.** `--factory-count` defaults to 200 and is enforced at a
   200 minimum (`scripts/benchmark_suite.py:753`). 0.785×200 = 157 correct; 0.780×200 = 156. The
   entire "regression" is one factory preset moving from correctly-retrieved to not.
2. **Retrieval scoring has no code path through structural search.** `run_retrieval_suites`
   (`scripts/benchmark_suite.py:576`) takes `stack`, `seed`, `factory_count`,
   `invariance_count`, and `factory_bundle` — it never reads `PATCHLAB_SERUM2_STRUCTURAL_SEARCH`
   or anything the deep structural search writes. Both arms embed the same 200 factory presets
   with the same checkpoint against the same fixed `preset_index`/`similarity_manifest.npz`. There
   is no logical mechanism visible in this code by which enabling structural search should move
   this number at all — which makes a real, causal effect unlikely and a near-tied
   floating-point/GPU-nondeterminism flip on one borderline case the more likely explanation.

This stage's only job is to find out which of those two it actually is.

## Phase 1 — Repeat the retrieval-only measurement, both arms, several times

`scripts/benchmark_suite.py` already supports `--suite retrieval`, which skips BAM entirely and
should run in well under a minute — do not run `--suite all` or `--suite bam` in this stage.

1. Run retrieval-only **five times** with `--structural-search` off (arm A conditions) and **five
   times** with it on (arm B conditions), same seed `20260802` each time, same
   `--factory-count 200`.
2. For every run, record the exact top1 rate and the full list of 200 `(preset_id, top1 bool)`
   results, not just the aggregate.
3. Report, per arm: whether all five runs agree exactly, and if not, which preset ID(s) flip and
   how often.

## Phase 2 — Diagnose whichever pattern shows up

**If arm A's five runs don't all agree with each other** (any variance under fixed settings): this
proves the retrieval measurement itself is noisy on this machine, independent of structural
search. Report the source if identifiable (GPU nondeterminism in the CLAP embedding batch step is
the leading candidate — check `_embed_in_batches`) and report how many of the 200 presets sit
within the same margin as the flipping one — i.e. how many cosine top-1 vs top-2 gaps are smaller
than the run-to-run embedding variance you just measured. That count is the real "at-risk" set,
not one preset.

**If arm A is perfectly stable across all five runs but arm B consistently reproduces 0.780** (not
noise, a real and repeatable 0.785→0.780 shift every time structural search is enabled): that is a
genuine, reproducible coupling this stage did not expect, and it needs a real explanation, not a
retry. Trace the one flipping preset ID specifically: confirm whether anything reachable from the
structural-search code path (env var, cache, temp file, shared in-process state) touches the
factory embedding or index used by `run_retrieval_suites` when both suites run in the same
process. Report exactly what you find, even if the answer is "no coupling found and this remains
unexplained" — do not paper over an unexplained reproducible effect.

## Phase 3 — Report and recommend, do not re-adjudicate the gate yourself

1. State plainly which of the two Phase 2 outcomes occurred, with the numbers to support it.
2. If it's measurement noise: recommend whether the adoption gate should be evaluated on an
   averaged/repeated retrieval measurement rather than a single run, and what averaged arm B
   retrieval@1 actually is across the five runs. Do not unilaterally flip the Stage 3D adoption
   decision — report the corrected number and let the next stage or the user decide.
3. If it's a real, reproducible coupling: recommend it be fixed before any further adoption
   consideration, since an unexplained cross-talk between structural search and retrieval scoring
   is a correctness concern independent of this specific gate.
4. Do not touch production defaults, do not enable structural search by default, do not modify the
   relay. This stage produces a diagnosis and a recommendation, nothing else.

## Phase 4 — Handoff

1. Commit code (if any diagnostic scripts were added) and `docs/benchmarks/stage3e-*.json`. Push.
   Version policy: one patch step per commit, single-digit components.
2. Final report: the five-run results for both arms, the diagnosis from Phase 2, and the
   recommendation from Phase 3. Stop at this gate — do not proceed into new search design or
   re-run the full A/B in this stage.
