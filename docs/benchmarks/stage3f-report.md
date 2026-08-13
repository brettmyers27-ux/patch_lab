# Stage 3F — frozen retrieval corpus and Stage 3D re-adjudication

Stage 3F froze the exact 200-preset factory selection at seed `20260802` into
`data/benchmark-cache/factory-renders-by-catalog-id-v2`. The WAVs remain private
and ignored; the committed manifest records the ordered catalog IDs and SHA-256
of every render. Once that manifest exists, the benchmark validates every member
and refuses to regenerate a missing or changed WAV.

## Frozen-corpus verification

The bootstrap created 200/200 renders at MIDI note 60. A second retrieval-only
invocation reused them: all 200 hashes and modification timestamps remained
unchanged, and the two complete per-preset result lists were identical.

| Measurement | Frozen run 1 | Frozen run 2 |
|---|---:|---:|
| Retrieval@1 | 0.775 (155/200) | 0.775 (155/200) |
| Retrieval@5 | 0.890 (178/200) | 0.890 (178/200) |
| Exact 200-row agreement | — | Yes |

Fresh, isolated Serum 2 hosts then rendered presets 190 and 216 outside the
frozen location. Neither WAV was byte-identical to its canonical copy. Their
CLAP cosines to the frozen copies were 0.769619 and 0.890912 respectively. This
directly confirms that the canonical files—not silently regenerated equivalents—
are reused by future retrieval gates.

## Canonical number and historical discrepancy

The canonical result is **0.775@1 / 0.890@5**. It reproduces neither historical
pair in full: the old Stage 2B cache was 0.785/0.895, Stage 3D was 0.780/0.890,
and Stage 3E was 0.780/0.895. Relative to Stage 2B, frozen presets 190 and 216
both lose top 1; preset 190 also loses top 5. Relative to Stage 3D, only preset
216 loses top 1. Relative to Stage 3E, only preset 190 loses top 1 and top 5.
The varying cache generations therefore fully explain the disagreement; the
frozen figure is now the forward reference.

## Stage 3D re-adjudication

The unaffected Stage 3D evidence remains positive: whole-set BAM improved
0.784226 → 0.785492, Serum 2 BAM improved 0.794525 → 0.796935, and invariance
remained 0.403333@1 / 0.596667@5. However, canonical retrieval@1 is 0.775,
below the required 0.785 gate.

Deep in-context structural search is therefore **not adopted**. FX, wavetable,
and noise search remain opt-in; modulation routes remain excluded because Stage
3D narrowing did not bring them under budget. The shipped default was not
flipped, no relay change was made, and the full BAM was not rerun. The full test
suite passed (124 tests), as did the visual-redesign and Milestone 4 UI gates.
