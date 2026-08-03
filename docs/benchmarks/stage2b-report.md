# Stage 2B final report — Windows 11 / RTX 5070

Completed 2026-08-03. Stage 2B stops at this reporting gate; Stage 3 was not started.

## Verdict

Adopt stack C: the fine-tuned CLAP encoder, the complete v2 embedding/index world, and the
existing shipped parameter and delta predictors. C passed every adoption condition on the
same 99-file BAM corpus and deterministic seeds used for Stage 2. The Stage 2 retrained
predictors are not adopted: B passed the gates, but C was stronger on BAM mean and factory
retrieval while avoiding the known predictor regressions.

The Windows checkout now defaults to `patchlab_clap_ft_v1.pt`; its local production feature
and factory-bundle paths were atomically replaced with the checksum-verified v2 artifacts.
The relay was not touched. The Mac must publish the six files listed in
`stage2b-artifact-manifest.json` before other installations receive this stack.

## Phase A — preset resolution

- Copied the 5,600-preset package to internal disk at `E:/PatchLab-Stage2-Presets`:
  5,601 files, 1,839,882,180 bytes, and no missing manifest IDs.
- Built `preset-paths.json`: 5,600 resolved, 0 missing; 1,083 installed factory entries
  were preferred over transferred copies.
- Rendered a deterministic sample of 25 previously missing Serum 1 presets through the
  four-worker headless DawDreamer path at MIDI note 60 for four seconds: 25 non-silent,
  0 failures.
- Complete-world preflight: 5,579/5,579 presets available (Serum 1 4,869; Serum 2 710),
  39,053 required note rows, 0 missing.
- Aggregate Phase A wall clock was not captured. Package copy, mapping, and proof each
  completed successfully; absence of one aggregate timer is a named reporting limitation.

## Phases B/C — full render and atomic v2 world

- Rendered and embedded all 39,053 C1–C7 note rows for 5,579 presets with one four-process
  pool. The initial full rebuild took 7,567.484 seconds (2 h 6 m 7.484 s); the streaming
  pipeline retained feature arrays rather than roughly 71 GiB of WAV files.
- The resumable completion rerun took 14.953 seconds and recomputed no completed note row.
- Similarity gates passed: note self-retrieval 39,053/39,053 (1.000000; strict argmax
  0.999129) and octave generalization 0.952142, above the 0.70 target.
- The v2 factory bundle passed with 1,167 searchable presets, 8,169 note embeddings, and a
  1,167 × 512 search index. Delta-neighbor generation also passed with no self-neighbors.
- Phase B rendering and Phase C indexing were timed together by the rebuild coordinator;
  separate Phase C wall time is therefore not available.

## Phase D — identical A/B/C gate

| Gate | A: shipped | B: fine-tuned + retrained | C: fine-tuned + shipped | C vs A |
|---|---:|---:|---:|---:|
| BAM mean | 0.770395 | 0.780465 | **0.784226** | **+0.013830** |
| BAM median | 0.774904 | 0.796812 | **0.793119** | +0.018215 |
| BAM minimum | 0.460627 | 0.567934 | **0.541550** | +0.080923 |
| Factory retrieval@1 | 0.560 | 0.775 | **0.785** | **+0.225** |
| Factory retrieval@5 | 0.770 | 0.890 | **0.895** | +0.125 |
| Invariance retrieval@1 | 0.256667 | 0.403333 | **0.403333** | **+0.146667** |
| Invariance retrieval@5 | 0.503333 | 0.596667 | **0.596667** | +0.093334 |
| BAM failures | 0 | 0 | **0** | Pass |

C improved 58 BAM samples and regressed 41 against A. Against B, C improved 52, tied 2,
and regressed 45. C completed 98 meaningfully modified recommendations; one result was not
classified as meaningfully modified, but all 99 matches completed without error. The full
99-row A/B/C table is in `stage2b-bam-table.md`.

Invariance @1 by variant, A → C: codec 0.420 → 0.500; gate 1/4 0.140 → 0.320;
gate 1/8 0.160 → 0.280; gate 1/16 0.000 → 0.080; loudness 0.500 → 0.760;
pitch 0.320 → 0.480.

- B wall clock: 2,957.687 seconds (49 m 17.687 s).
- C wall clock: 2,798.703 seconds (46 m 38.703 s).
- Same corpus, balanced budget, factory count 200, invariance count 50, and seed 20260802
  were used for A, B, and C.

## Adoption and artifact handoff

Adopted: `patchlab_clap_ft_v1.pt`, `preset_index.npy`, `note_index.npy`,
`similarity_manifest.npz`, `delta_neighbors.npz`, and `factory_bundle.sqlite`. Exact sizes,
SHA-256 values, and install destinations are recorded in `stage2b-artifact-manifest.json`.
The shipped `param_model.pt` and `delta_param_model.pt` remain production. The experimental
`param_model_stage2.pt` and `delta_param_model_stage2.pt` remain preserved locally but are
not relay candidates.

## Named limitations and Stage 3 assumptions

- Serum 2 remains structurally constrained by the mapped automation surface, especially
  FX topology and modulation-matrix routes. More source presets do not remove this limit.
- Some transferred presets contain absolute macOS references to external samples. Serum
  emitted nonfatal missing-sample warnings, although all 39,053 required rows completed.
- C improves the aggregate BAM mean materially but does not improve every sample; 41 of 99
  scores were lower than A, and dense vocal/production targets remain difficult.
- The macOS screenshot gates remain for the Mac after pull; Stage 2B changed no UI code.
- Stage 3 should assume the fine-tuned encoder plus complete v2 indexes/factory bundle and
  shipped predictors are production once the Mac publishes the manifest. It may reuse the
  Stage 2 corpus and pipelines, but must not assume the rejected predictors were installed.
