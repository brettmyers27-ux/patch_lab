# Match speed: streaming evaluation, worker count and the warm engine

Measured on an Apple M4 (4 performance + 6 efficiency cores, 24 GB), source tree,
warm caches, with unrelated apps loading the machine (so absolute numbers are
indicative; ratios are what to trust).

## Where a Match spent its time

| Stage | Before | Notes |
| --- | --- | --- |
| Start the match worker | ~9 s | Python/ML imports, CLAP, parameter + delta models, render-worker pool |
| Search (Quick, 51 renders) | ~13 s | render 40 %, CLAP 25-40 %, STFT 10 % |
| Search (Balanced, 291 renders) | ~77 s | render 69 %, CLAP 20 %, STFT 9 % |
| Save + verify the preset | ~8 s | a second process re-loaded CLAP and a Serum host |

A batch paid the first and last rows again for every file.

## What changed

1. **Cached target spectra** (`core.matcher.TargetSpectra`). The STFT loss used to
   recompute the target's three spectrograms for every candidate. Results are
   bit-identical; the loss is ~3.3x cheaper.
2. **Streaming evaluation** (`AnalysisBySynthesisMatcher._evaluate`). Candidates are
   dispatched with `imap` and scored in chunks of `STREAM_CHUNK` (8) as renders
   arrive, so the GPU embeds while the CPU workers render. Each candidate's
   objective is computed by exactly the same arithmetic as before
   (`tests/test_match_speedups.py` pins the numbers). With realistic render
   rates a batch is 1.35x (16 candidates) to 1.5x (51) faster.
3. **Worker count** (`core.platform_env.recommended_render_workers`). Performance
   cores + half the efficiency cores, between the old default of 4 and 8, only on
   machines with >= 16 GB (each worker is ~600 MB). `PATCHLAB_RENDER_WORKERS`
   overrides it.
4. **Warm engine** (`scripts/engine_server.py`, `app.workers.EngineConnection`).
   One long-lived worker keeps CLAP, the models and the Serum render workers
   loaded and serves Match and export jobs one at a time. The save-and-verify step
   shares the matcher's CLAP model. Choosing a sound starts loading the engine
   while the user picks a quality; it exits after `PATCHLAB_ENGINE_IDLE_SECONDS`
   (default 600) without work.

   Measured end to end through a real window: Quick Match + auto-save went from
   ~31 s to ~11 s for the first file and ~7.5 s for each file after it.

## Safety properties

* The engine is an optimisation, never a dependency. If it cannot start, dies, or
  `PATCHLAB_WARM_ENGINE=0`, `MatchProcessRunner` / `ExportProcessRunner` run the
  original one-process-per-job workers. Two unexpected engine deaths disable it
  for the session. Factory-fingerprint matches never use it.
* Any failed or stalled Match, a different target generation or worker count, or a
  dead render worker discards the warm matcher; the next Match builds a fresh one.
* Cancelling a running job stops the engine process (the same effect as
  terminating a one-shot worker); it is restarted on demand.
* An engine that exits on its idle timeout at the instant a job arrives is retried
  transparently on a new engine.
* Each job carries its own `PATCHLAB_OPERATION_ID`, so support bundles still tell
  one Match from the next.
* Tests set `PATCHLAB_WARM_ENGINE=0` (tests/conftest.py); the engine tests opt in.
