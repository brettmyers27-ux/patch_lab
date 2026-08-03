# Codex Prompt — Stage 2b: complete the atomic rebuild (run ON the Windows 11 / RTX 5070 PC)

Copy everything below into Codex **on the Windows PC**, from
`%USERPROFILE%\Documents\PatchLab\soundmatch`, after `git pull`.

---

Read `docs/ACCURACY_ROADMAP.md` §1 and `docs/codex-stage2-smarter-brain.md` before starting.
This finishes the one thing Stage 2 could not do, then re-runs its final gate. §5 conventions
apply unchanged: numeric before/after on the *same* benchmarks, no silent scope cuts, stop at
the reporting gate, never automate a live DAW.

## What Stage 2 established, and the single blocker

Stage 2 completed honestly and adopted nothing. Its own numbers:

- Baseline (pinned production stack, 99 BAM files): mean **0.770395**, factory retrieval@1
  **0.560** / @5 **0.770**, invariance@1 **0.256667** / @5 **0.503333**.
- The fine-tuned encoder `patchlab_clap_ft_v1.pt` **passed** its 115-preset held-out gate:
  clean retrieval@1 0.530435 → **0.634783**, augmented-copy retrieval@1 0.486957 →
  **0.608696**. That is the strongest signal Stage 2 produced.
- Phase 4 then required 5,579 presets / 39,053 note rows and could render only **1,167**:
  Serum 1 needed 4,869, had 457, **missing 4,412**. A partial index would shrink retrieval
  scope or mix encoder worlds, so no v2 index was built and the fine-tuned encoder was never
  carried into an end-to-end A/B.
- The only stack that *could* be tested — pinned encoder plus retrained predictors — failed:
  BAM +0.000835 (immaterial), retrieval@1 0.560 → **0.555**, invariance unchanged.

**So the encoder that showed a ~12-point invariance gain has never been measured end to end.**
That is the whole point of this prompt. Everything else about Stage 2 stands.

## What has changed: the missing presets are now available

The Mac transferred a preset package — every one of the 5,600 catalog presets, 1.84 GB. It
was **not** necessary to move the 71 GiB render library: this machine has Serum and a 5070
and can re-render far more cheaply than that transfer costs.

Package layout:

```
PatchLab-Stage2-Presets/
  manifest.json      # content_hash -> {relative_path, preset_id, name, synth}
  presets/           # 5,600 files, each named <content_hash><ext>
```

Files are hash-named because preset names collide across folders; the manifest carries the
original name and catalog `preset_id`.

**Why the catalog alone was not enough:** it stores the absolute macOS path of every preset,
which dangles on this machine. Resolution therefore goes through content hash, reusing the
mechanism already in `core/matcher._factory_paths_by_hash` and
`core/serum2_preset_writer._factory_path_for_hash` — no resolution code needs changing.

## Phase A — Wire up the presets (gate before anything expensive)

1. Copy the package off the USB drive onto **internal disk** (rendering 30k+ clips against
   USB is needlessly slow). Report the destination and free space; Phase B needs headroom
   for ~39,000 WAVs at roughly the Phase 2 rate.
2. Build the mapping:
   ```
   .venv\Scripts\python.exe scripts\build_preset_path_map.py --package <copied package path>
   ```
   It merges this machine's existing `factory-paths.json` (a locally installed factory preset
   beats a transferred copy) and writes `preset-paths.json` into the app data directory.
   **Report `resolved=` and `missing=`. `resolved` must be 5,600 and `missing` 0 — if not,
   stop and report which hashes are unresolved rather than proceeding with a partial world.**
3. Set `PATCHLAB_FACTORY_MAPPING` to that file for every subsequent step in this prompt.
4. **Prove resolution actually works before rendering anything.** For a random sample of ≥ 25
   Serum 1 presets that were previously missing, resolve catalog `preset_id → content_hash →
   local file`, load each into Serum through the existing render path, and confirm a
   non-silent render. Report the sample size and any failures. This is the gate: if presets
   still cannot be loaded, Phase B is 10 wasted hours.

## Phase B — Render the complete note set

1. Re-run the Phase 4 preflight from `docs/codex-stage2-smarter-brain.md`. It must now report
   **5,579 presets / 39,053 note rows renderable, 0 missing.** Report the exact numbers. If
   anything is still missing, stop — do not build a partial index.
2. Render the full C1–C7 note set for every preset not already rendered, reusing the existing
   render pipeline and its four-process pool. Resumable by (preset_id, note); a crashed run
   must continue, not restart. Report rate, wall clock, and count.
3. Renders are keyed by catalog `preset_id`, matching the shipped index's ID space — do not
   introduce a second ID namespace.

## Phase C — The atomic v2 embedding world

Rebuild **every** embedding artifact with `patchlab_clap_ft_v1.pt`, as one consistent set
under `data/stage2/artifacts-v2/`: `preset_index.npy`, `note_index.npy`,
`similarity_manifest.npz`, the factory bundle's `preset_embeddings`/`note_embeddings`, and
delta neighbors. Extend the existing build scripts to accept a checkpoint argument rather
than duplicating them. **Mixing v1 and v2 embedding artifacts is forbidden** — the A/B
harness must load each stack whole. Report artifact sizes and SHA-256s.

## Phase D — The A/B that Stage 2 never got to run

Run `scripts/benchmark_suite.py` on the identical 99 BAM files and the identical seeds
(`20260802`), changing nothing but the stack:

- **A** — pinned encoder + shipped indexes + shipped predictors. Reuse Stage 2's recorded
  baseline; do not re-derive it.
- **B** — fine-tuned encoder + v2 indexes + Stage 2's retrained predictors.
- **C** — fine-tuned encoder + v2 indexes + **shipped** predictors.

C matters: Stage 2 showed the retrained predictors slightly *hurt* factory retrieval, so the
encoder's gain must be measurable independently of them. Report all three side by side, plus
the per-sample BAM table for whichever of B/C wins.

**Adopt only if**: BAM mean improves materially; factory retrieval@1 ≥ 0.560; invariance@1
improves materially above 0.256667; no new test failures. Partial adoption (encoder only,
shipped predictors) is a legitimate and expected outcome — state it plainly. If the encoder
still does not deliver end to end despite its held-out gain, say so: that is a real finding
about the gap between held-out retrieval and full-scope retrieval, not a failure to hide.

## Phase E — Handoff

1. Commit code, `docs/benchmarks/*.json`, and an updated roadmap §1/§4 reflecting the new
   measured ladder. Push. Version policy: one patch step per commit, single-digit components.
2. **Do not commit model or index artifacts, and do not touch the relay.** Write
   `docs/benchmarks/stage2b-artifact-manifest.json` with each adopted artifact's filename,
   byte size, SHA-256, and install destination. The Mac holds the relay credentials and
   publishes from that manifest.
3. Final report: every phase's real numbers, named limitations with reasons, wall clock per
   phase, and what Stage 3 should now assume. Stop at this gate.
