# Stage 3E — retrieval-regression verification

Stage 3E reproduced retrieval five times per arm at seed `20260802` and 200
factory presets. Both arms returned exactly the same result on every run:
`0.780@1` (156/200) and `0.895@5` (179/200). Every top-1 boolean, top-5
boolean, retrieved-ID ranking, and score was identical across all ten runs.
The averaged arm B retrieval@1 is therefore **0.780**. No structural-search
coupling exists in the retrieval-only path.

## Five-run result

| Arm | Structural search | Retrieval@1, runs 1–5 | Retrieval@5, runs 1–5 | Exact 200-row agreement |
|---|---:|---|---|---:|
| A | Off | 0.780, 0.780, 0.780, 0.780, 0.780 | 0.895, 0.895, 0.895, 0.895, 0.895 | Yes |
| B | On | 0.780, 0.780, 0.780, 0.780, 0.780 | 0.895, 0.895, 0.895, 0.895, 0.895 | Yes |

The complete ten sets of 200 per-preset rows are preserved privately under
`data/stage3e`; they are excluded from Git under the repository's data policy.
There were no flipping IDs within either arm or between arms in these fixed-cache
runs. Fixed-WAV CLAP embeddings and retrieval scores were bit-for-bit stable, so
the observed embedding-batch variance was zero.

## Diagnosis

The Stage 3D comparison combined Stage 2B's existing arm-A render cache
(`0.785@1`) with a newly generated Stage 3D arm-B cache (`0.780@1`). Re-embedding
the retained Stage 2B, Stage 3D, and Stage 3E WAVs showed nonzero drift for exactly
two of the same 200 catalog IDs:

| Preset ID | Maximum cross-cache cosine distance | Current top-1/top-2 gap | Behavior |
|---:|---:|---:|---|
| 190 | 0.225236 | 0.155333 | Correct in Stage 2B/3A/3E; incorrect in Stage 3D |
| 216 | 0.195770 | 0.027798 | Correct in Stage 2B/3A/3D; incorrect in Stage 3E |

Those two presets are the complete measured at-risk set: the other 198 retained
embeddings were equal within numerical precision across the three independent
render caches. The net Stage 3E count remains 156 because ID 190 recovered while
ID 216 moved out of top 1.

Targeted fresh-host renders confirmed that this is Serum 2 render generation or
host-initialization/state-order sensitivity, not GPU nondeterminism in
`_embed_in_batches`. ID 216 produced two distinct CLAP/retrieval modes across five
fresh hosts (four misses and one correct result). ID 190's fresh-host mode matched
its Stage 3D miss, while sequential full-suite caches also produced its correct
mode. `FactoryRenderer` caches the first WAV for each ID, so repetitions against
one cache are stable but separate cache generations need not be equivalent.

The structural-search environment flag is consumed only by the BAM match path.
`run_retrieval_suites` renders factory presets, embeds their audio, and compares
against the fixed index without calling matcher or structural-search code. The
zero cross-arm difference confirms that static trace experimentally.

## Recommendation and gate

Evaluate future retrieval gates either against a frozen, canonical rendered
corpus or by averaging genuinely independent fresh render-cache populations.
Repeating embeddings against one cache is useful for measuring CLAP stability but
is not an independent test of Serum 2 rendering. The five requested arm-B runs
average **0.780@1** and do not justify changing the Stage 3D decision.

Per the prompt, Stage 3D adoption is not re-adjudicated here. Structural search
remains opt-in, production defaults are unchanged, the relay is untouched, and no
new search design or full BAM run was performed.
