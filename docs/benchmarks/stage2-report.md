# Stage 2 final report — Windows 11 / RTX 5070

Completed 2026-08-03. Stage 2 stops at this reporting gate; Stage 3 was not started.

## Verdict

No Stage 2 model or index artifact is adopted. The fine-tuned CLAP encoder passed its
held-out mid-phase gate, but this PC lacks 4,412 Serum 1 source presets required to rebuild
the complete embedding world atomically. The permitted partial B stack—pinned encoder and
indexes plus retrained predictors—raised the 99-file BAM mean only from 0.770395 to
0.771230, lowered factory retrieval@1 from 0.560 to 0.555, and left rhythm invariance@1
unchanged at 0.256667. It therefore fails two of the four final adoption conditions.

The accepted deliverables are the deterministic benchmark harness and the reusable data
generation, CLAP fine-tuning, complete-world preflight, and predictor-training pipelines.

## Phase results

### Phase 0 — environment and parity

- Pulled current `main`; the starting tree was at/after `99bd028` and clean.
- PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 5070, capability 12.0. A real bf16
  GPU matmul completed on CUDA.
- Serum 1 exposed exactly 316 parameters with an exact index/name signature match to the
  macOS reference. Serum 2 exposed 2,623 parameters. Both audio-parity gates passed.
- All required delivered artifacts were present, both synths were licensed, and all 99
  authorized BAM audio files decoded (47 AIF, 52 WAV; metadata excluded).
- The initial test baseline was 88 passing tests with no failure. The macOS screenshot
  gates were not runnable because their macOS-only fixture audio is absent on this PC.
- Wall clock: not captured as one phase timer because installation/parity repairs preceded
  the Stage 2 benchmark timer. This is a reporting limitation, not an estimated value.

### Phase 1 — fresh baseline

- BAM: 99/99, mean 0.770395, median 0.774904, minimum 0.460627.
- Factory retrieval: @1 0.560, @5 0.770 over 200 presets.
- Invariance: @1 0.256667, @5 0.503333 over 50 presets / 300 variants.
- Variant @1: codec 0.420; gate 1/4 0.140; gate 1/8 0.160; gate 1/16 0.000;
  loudness 0.500; pitch 0.320.
- Wall clock: 172.640 seconds.

### Phase 2 — scaled synthetic corpus

- 30,000 base clips, 10,000 exact three-note patch groups, 390,000 prediction pairs,
  1,000 NPZ shards, and 1,152 distinct source presets.
- Serum 1: 21,042 clips. Serum 2: 8,958 clips. Serum 2 remains limited to the mapped
  automation surface; additional data does not remove that structural cap.
- Raw audio: 30,000 mono 48 kHz float WAVs, 23,042,520,000 bytes. Shards:
  1,370,758,580 bytes. Total: 24,413,278,580 bytes.
- Generation rate 0.851306 clips/second; 540 group-level retry clips. Full audit read every
  WAV header and NPZ array and found no invalid shape, non-finite value, mismatched group,
  incomplete shard, or temporary file.
- Wall clock: 35,239.964 seconds (9 h 47 m 19.964 s).

### Phase 3 — CLAP fine-tuning

- Attempt 1 trained the last HTSAT stage plus audio projection for 2,000 steps, with the
  text tower frozen, bf16, symmetric InfoNCE, preset-level holdout, and on-the-fly gate,
  pump, pitch, loudness, codec, tail, EQ, noise, width, and offset augmentation.
- Held-out presets: 115. Clean retrieval@1 improved 0.530435 → 0.634783. Augmented-copy
  retrieval@1 improved 0.486957 → 0.608696. The mid-phase gate passed.
- Checkpoint: `patchlab_clap_ft_v1.pt`, 800,729,534 bytes, SHA-256
  `fbaf3a305704890450ec4604fe9a1ab806515b68827ec44a73f500c06c042baf`.
- Real app reload produced one finite, unit-normalized 512-dimensional embedding.
- Wall clock: 322.063 seconds.

### Phase 4 — atomic embedding world

- Required: 5,579 presets / 39,053 note rows. Renderable here: 1,167 presets.
- Serum 1 required 4,869, available 457, missing 4,412. Serum 2 required and available 710.
- A partial v2 index would shrink retrieval scope or mix encoder worlds, both forbidden.
  No v2 index was created and the fine-tuned encoder was not advanced to end-to-end B.
- Measured full preflight wall clock: 3.2 seconds. The earlier 1,167-preset factory-bundle
  probe was successful but was not separately timed, which is a named timing limitation.

### Phase 5 — predictor retraining

- Used all 1,000 shards / 390,000 pairs with preset-level splits: Serum 1 411 train / 46
  validation; Serum 2 625 train / 70 validation.
- Absolute model, old → new MAE: Serum 1 0.143991 → 0.091997; Serum 2 0.015366 →
  0.018287. Serum 1 improved materially; Serum 2 regressed.
- Delta model, old → new MAE: Serum 1 0.017086 → 0.010453; Serum 2 0.000857 →
  0.000537. Both improved over the shipped delta model but remained worse than the
  no-adjustment neighbor baselines (0.009357 and 0.000423).
- Absolute peak VRAM 378,922,496 bytes; delta peak VRAM 592,154,624 bytes.
- Wall clock: 867.250 seconds total (374.906 absolute; 457.672 delta; remaining time is
  corpus/split setup and report work).

### Phase 6 — same-input A/B

| Gate | A: shipped stack | B: retrained predictors | Result |
|---|---:|---:|---|
| BAM mean | 0.770395 | 0.771230 | Pass numerically (+0.000835), not material |
| Factory retrieval@1 | 0.560 | 0.555 | Fail |
| Factory retrieval@5 | 0.770 | 0.760 | Regressed |
| Invariance retrieval@1 | 0.256667 | 0.256667 | Fail: no material improvement |
| Invariance retrieval@5 | 0.503333 | 0.503333 | Unchanged |
| Test failures | 0 | 0 | Pass |

B improved 47 samples, tied 2, and regressed 50. Its median was 0.779999 and minimum
0.535637. All 99 runs completed, all recommendations were meaningfully modified, and no
sample errored. The 30-minute coordinator cap required one resumable restart; no completed
sample was recomputed. Observed coordinator wall clock was 2,791.6 seconds across both
executions (1,804.0 seconds before timeout plus 987.562 seconds after resume).

## Per-sample BAM comparison

| Sample | A | B | Delta |
|---|---:|---:|---:|
| AC  Free 1.wav | 0.726852 | 0.729744 | +0.002892 |
| AC  Free 2.wav | 0.774323 | 0.764833 | -0.009490 |
| Dill Bass 1.aif | 0.813834 | 0.807345 | -0.006489 |
| Dill Lead 1.aif | 0.810373 | 0.820817 | +0.010445 |
| Dill Lead 2.aif | 0.896428 | 0.890676 | -0.005753 |
| Do That 1.wav | 0.855714 | 0.883887 | +0.028174 |
| Do That 2.wav | 0.814285 | 0.815925 | +0.001641 |
| Do That 3.wav | 0.746422 | 0.779999 | +0.033577 |
| Do That 4.wav | 0.819210 | 0.862423 | +0.043213 |
| Do That 5.wav | 0.852629 | 0.845576 | -0.007053 |
| Excision Bass 1.aif | 0.724804 | 0.752346 | +0.027541 |
| Excision Lead 1.aif | 0.669822 | 0.719109 | +0.049288 |
| Grabbitz 1.wav | 0.816580 | 0.819744 | +0.003164 |
| Grabbitz 2.wav | 0.812936 | 0.712709 | -0.100227 |
| Grabbitz 3.wav | 0.882828 | 0.866527 | -0.016301 |
| Jaenga 1.wav | 0.743031 | 0.691412 | -0.051619 |
| Jaenga 2.wav | 0.805678 | 0.782236 | -0.023443 |
| Jaenga 3.wav | 0.710124 | 0.732323 | +0.022198 |
| Jaenga 4.wav | 0.729917 | 0.820564 | +0.090646 |
| jkyl 1.wav | 0.826242 | 0.811082 | -0.015160 |
| jkyl 2.wav | 0.608455 | 0.598619 | -0.009836 |
| jkyl 3.wav | 0.809370 | 0.835603 | +0.026233 |
| jkyl 4.wav | 0.774904 | 0.815709 | +0.040805 |
| jkyl 5.wav | 0.847149 | 0.859003 | +0.011854 |
| LSDREAM 1.wav | 0.794792 | 0.791486 | -0.003306 |
| LSDREAM 2.wav | 0.754287 | 0.745236 | -0.009051 |
| LSDREAM 3.wav | 0.754341 | 0.757137 | +0.002795 |
| LSDREAM 4.wav | 0.708450 | 0.739040 | +0.030591 |
| LSDREAM 5.wav | 0.857534 | 0.887067 | +0.029533 |
| LSDREAM 6.wav | 0.900105 | 0.884073 | -0.016031 |
| LSDream Bass 1.aif | 0.853393 | 0.853393 | +0.000000 |
| LSDream Bass 2.aif | 0.648038 | 0.726489 | +0.078451 |
| LSDream Bass 3.aif | 0.729963 | 0.767661 | +0.037698 |
| LSDream Bass 4.aif | 0.720465 | 0.741825 | +0.021360 |
| LSDream Bass 5.aif | 0.787042 | 0.765762 | -0.021280 |
| LSDream Lead 1.aif | 0.800868 | 0.818043 | +0.017175 |
| LSDream Lead 2.aif | 0.663375 | 0.668731 | +0.005357 |
| LSDream Lead 3.aif | 0.668863 | 0.619723 | -0.049140 |
| LSDream Lead 4.aif | 0.655020 | 0.612953 | -0.042067 |
| LSDream Lead 5.aif | 0.736475 | 0.690172 | -0.046303 |
| LSDream Lead 6.aif | 0.767113 | 0.722390 | -0.044724 |
| Mazare Bass 1.aif | 0.774173 | 0.813643 | +0.039470 |
| Mazare Bass 2.aif | 0.852920 | 0.791987 | -0.060934 |
| Nasty VR Bass 1.aif | 0.695764 | 0.782092 | +0.086328 |
| Nasty VR Bass 2.aif | 0.849284 | 0.772891 | -0.076393 |
| Nasty VR Bass 3.aif | 0.838380 | 0.811741 | -0.026638 |
| Nasty VR Bass 4.aif | 0.731003 | 0.707916 | -0.023087 |
| Nasty VR Downbeat.aif | 0.787086 | 0.750401 | -0.036685 |
| Skrill Bass 1.aif | 0.850529 | 0.849752 | -0.000777 |
| Skrill Bass 2.aif | 0.771838 | 0.821873 | +0.050035 |
| Skrill Bass 3.aif | 0.837202 | 0.861946 | +0.024744 |
| Skrill Bass 4.aif | 0.794671 | 0.796618 | +0.001947 |
| Skrill Bass 5.aif | 0.845638 | 0.816278 | -0.029360 |
| Skrill lead 1.aif | 0.930308 | 0.894832 | -0.035475 |
| Skrill lead 2.aif | 0.821294 | 0.815885 | -0.005409 |
| Space Laces 1.wav | 0.757031 | 0.731004 | -0.026026 |
| Space Laces 2.wav | 0.772861 | 0.739818 | -0.033043 |
| Space Laces 3.wav | 0.726833 | 0.722661 | -0.004172 |
| Space Laces 4.wav | 0.788463 | 0.826436 | +0.037973 |
| Space Laces 5.wav | 0.790735 | 0.770649 | -0.020086 |
| Space Laces 6.wav | 0.731629 | 0.774319 | +0.042691 |
| Space Laces 7.wav | 0.751299 | 0.786994 | +0.035696 |
| Space Laces 8.wav | 0.748292 | 0.707658 | -0.040634 |
| Sudley 1.wav | 0.744383 | 0.814329 | +0.069946 |
| Sully 1.wav | 0.749465 | 0.741782 | -0.007684 |
| Sully 2.wav | 0.848133 | 0.797765 | -0.050368 |
| Sully 3.wav | 0.817437 | 0.852002 | +0.034564 |
| Sully 4.wav | 0.740197 | 0.691916 | -0.048282 |
| Sully 5.wav | 0.856541 | 0.861344 | +0.004803 |
| VR Bass 1.aif | 0.768780 | 0.806146 | +0.037366 |
| VR Bass 2.aif | 0.706320 | 0.654119 | -0.052201 |
| VR Bass 3.aif | 0.798497 | 0.782149 | -0.016348 |
| VR Bass 4.aif | 0.869577 | 0.821683 | -0.047894 |
| VR Bass 5.aif | 0.753454 | 0.786165 | +0.032712 |
| VR Bass 6.aif | 0.839206 | 0.808209 | -0.030997 |
| VR Bass 7.aif | 0.807466 | 0.794486 | -0.012980 |
| VR Bass 8.aif | 0.885049 | 0.885284 | +0.000235 |
| VR Lead 1.aif | 0.697317 | 0.754889 | +0.057571 |
| VR Lead 2.aif | 0.870681 | 0.862050 | -0.008631 |
| VR vox 1.aif | 0.460627 | 0.535637 | +0.075010 |
| VR vox 2.aif | 0.539181 | 0.571289 | +0.032108 |
| VR vox 3.aif | 0.632181 | 0.538769 | -0.093412 |
| X Bass 1.aif | 0.746290 | 0.765179 | +0.018889 |
| X Bass 2.aif | 0.831503 | 0.846394 | +0.014891 |
| X Bass 3.aif | 0.827926 | 0.776536 | -0.051390 |
| YDG 1.wav | 0.717082 | 0.711001 | -0.006081 |
| YDG 2.wav | 0.780253 | 0.761717 | -0.018536 |
| YDG 3.wav | 0.701343 | 0.743131 | +0.041788 |
| YDG 4.wav | 0.820280 | 0.766899 | -0.053381 |
| ZD Bass 1.aif | 0.784343 | 0.826039 | +0.041697 |
| Zomboy 1.wav | 0.864575 | 0.850999 | -0.013577 |
| Zomboy 2.wav | 0.739665 | 0.797898 | +0.058233 |
| Zomboy 3.wav | 0.649846 | 0.680665 | +0.030819 |
| Zomboy 4.wav | 0.744796 | 0.710246 | -0.034549 |
| Zomboy 5.wav | 0.804357 | 0.761387 | -0.042970 |
| Zomboy 6.wav | 0.639118 | 0.652748 | +0.013630 |
| Zomboy 7.wav | 0.745855 | 0.745855 | +0.000000 |
| Zomboy 8.wav | 0.592134 | 0.681954 | +0.089819 |
| Zomboy 9.wav | 0.776007 | 0.756303 | -0.019705 |

## Named limitations and Stage 3 assumptions

- The complete fine-tuned embedding world cannot be built until the 4,412 missing Serum 1
  source presets (or their original 71 GiB render library on the Mac) are available.
- Serum 2 remains structurally constrained by its mapped automation coverage, especially
  FX topology and modulation routes.
- The macOS visual screenshot gates remain for the Mac to run after pull; no UI code was
  changed here.
- The initial parity command's exact audio-cosine values were not preserved in a compact
  tracked report, although its pass/fail gates and exact parameter counts were verified.
- Stage 3 should assume the shipped pinned encoder, shipped indexes, and shipped predictor
  models remain production. It may reuse the 30,000-clip/390,000-pair corpus and Stage 2
  scripts, but must not assume any experimental artifact was relayed or installed.

