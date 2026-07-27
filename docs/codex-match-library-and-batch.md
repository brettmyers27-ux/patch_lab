# Codex Prompt — Match Library + Batch Folder Processing

Copy everything below this line into Codex, in the `soundmatch` repo.

---

Two related features: a persistent **Match Library** of every match ever run, and a
**batch folder** mode that turns a whole folder of audio files into presets and then shows
the results in that library.

Read `docs/ACCURACY_ROADMAP.md` §1 for verified project facts before starting. Constraints
that remain absolute: **never automate a live DAW or plugin session** (headless DawDreamer
rendering only); reuse the existing verified pipelines rather than building parallel ones;
the UI gates `scripts/verify_visual_redesign.py` and `scripts/verify_milestone4_ui.py` must
still pass when you finish.

## Critical existing behavior you must account for

Read these before designing anything — the first one will silently break the feature if
missed:

1. **Match results are currently ephemeral.** `MainWindow._match_session` is a
   `tempfile.TemporaryDirectory` (`app/ui.py`, created ~line 225, `.cleanup()` ~line 1151).
   Every `result.json` and every cached `recommendation-{note}.wav` lives inside it and is
   destroyed when the app closes. A library that merely stores the current path will be
   broken on next launch.
2. **`result.json` contains absolute paths into that temp directory** — at minimum
   `recommendation.winner_audio_path` and `recommendation.candidate_path`. Archiving must
   copy those files into durable storage **and rewrite the stored paths**, not just copy
   the JSON. Paths that already point at durable locations (`audition_path` and
   `preview_source_path` for existing matches point into `data/audio/…` and the scanned
   preset library) must be left alone.
3. **Lazy octave rendering already exists — reuse it, do not reinvent it.**
   `MainWindow.play_winner()` (`app/ui.py` ~line 965) looks for
   `result_path.parent / f"recommendation-{note}.wav"` and, when absent, calls
   `PreviewProcessRunner.start_recommendation(result_path, midi_note)`. Point that same
   mechanism at each library entry's archived folder and octave playback works with a
   cache-on-first-click behavior for free.
4. **Export is already verified.** `ExportProcessRunner.start(result_path, output_path)`
   round-trip-verifies (decode back, re-render, confirm similarity) before a file is
   presented as ready. Every preset written by the library or by batch mode must go
   through it. Do not add an unverified write path.
5. **Serum preset folder detection** is `MainWindow._default_export_folder(synth)` over
   `ENV.existing_preset_roots` (`core/platform_env.py`).
6. **The UI is a fixed 1920×1080 design canvas** inside a `QGraphicsScene`, uniformly
   scaled by `ScaledGraphicsView`. Anything you add must live inside that canvas and hold
   16:9 with no scrolling and no overlap at 1440×810, 1600×900 and 1920×1080.

## Part A — Durable storage

1. **Library root**: a durable directory outside the temp session — `data/match_library/`
   in the normal build. In distribution mode respect the existing `self.local_paths`
   convention rather than writing into the app bundle. One subdirectory per match, named
   by a stable generated `match_uid`, containing:
   - the copied source audio (see A2),
   - `result.json` with temp-session paths rewritten to point inside this folder,
   - any archived render referenced by the rewritten JSON,
   - `recommendation-{note}.wav` octave caches as they get rendered on demand.
2. **Copy the uploaded audio in.** The user's original file is copied into the entry's
   folder at archive time so entries keep working if the original is moved, renamed or
   deleted. Preserve the original filename for display; store its content hash.
3. **Nothing is auto-deleted.** No size cap, no pruning, no expiry. Deletion is an
   explicit user action (see C6) and must remove both the DB row and the entry's folder.
4. **Schema** — add to `SCHEMA_SQL` in `core/db.py` (it runs through the existing
   `migrate()` / `schema_migrations` path; do not create a second database):
   - `match_library`: `id`, `match_uid` (unique), `source_name`, `source_audio_path`,
     `source_content_hash`, `result_json_path`, `target_synth`, `budget`,
     `similarity_percent`, `base_name`, `recommendation_synth`, `no_confident_match`,
     `batch_id` (nullable FK), `exported_preset_path` (nullable), `created_at`.
   - `match_batches`: `id`, `folder_name`, `source_folder`, `export_folder`,
     `target_synth`, `budget`, `total_files`, `completed_files`, `failed_files`,
     `status` (`running` / `cancelled` / `complete`), `created_at`.
   - Store paths **relative to the library root**, not absolute, so the library survives
     the app directory being moved. Resolve to absolute at read time.
   - Add `Database` methods for insert / list (newest first) / get-by-uid / delete /
     batch create / batch progress update. Follow the style of the existing
     `set_favorite` / `favorite_hashes` methods.
5. **Archive every completed match automatically**, including no-confident-match results
   (they are still history the user may want to revisit) — record
   `no_confident_match = 1` and let the row render honestly rather than hiding it.
   Archiving happens in `_match_completed`, after `_show_match_result` succeeds.
6. **Migration safety**: existing databases must upgrade cleanly with no data loss and no
   destructive rewrite of `presets`, `renders`, `params`, `fingerprints`,
   `serum2_full_settings` or `favorites`. Verify against a copy of the real database.

## Part B — Tabbed navigation

1. Add a tab bar inside the scaled design canvas with two tabs: **Match** (everything
   that is on the dashboard today, unchanged) and **Library**. Match is the default tab on
   launch. Style it consistently with the existing control-card/pill visual language in
   `app/theme.qss` — reuse the existing tokens, do not introduce a new palette.
2. The tab bar must not break proportional scaling. It lives inside the 1920×1080 canvas
   and scales with everything else.

## Part C — The Library tab

A list of every saved match, newest first, each row showing:

1. **Source audio play button** — plays the archived copy of the uploaded file.
2. **Entry name** — the original filename, plus a muted secondary line with the match date,
   target synth, and similarity percent (or "No confident match" where applicable).
3. **C1–C7 octave buttons for the generated Serum patch**, reusing the existing
   `#rowOctaveButton` styling from the Closest Matches rows. Clicking one plays the
   generated preset at that note: play the cached `recommendation-{note}.wav` if present,
   otherwise render it via `PreviewProcessRunner.start_recommendation` against that
   entry's archived `result.json`, cache it in the entry folder, and play it when ready.
   Show a brief "Rendering…" state on the clicked button rather than freezing the UI, and
   re-clicking the same octave must always replay (the `SegmentedControl.itemClicked`
   change made for this already exists — match that behavior for these plain buttons,
   which fire `clicked` on every press anyway).
4. **Export Preset button per row** — writes that entry's preset through
   `ExportProcessRunner`, identical verification to the main panel.
5. **Double-click (and Enter on a focused row) opens the full match** — switch to the
   Match tab and repopulate it from the archived `result.json` so the user sees the same
   closest-matches list and generated-preset panel they saw when it first ran. Restore
   `_match_result`, `_match_result_path` (pointing at the archived copy) and
   `_match_audio_path` (pointing at the archived source copy) so that Export Preset,
   Load in Serum, the octave selector and the source-audio play button all work on the
   reopened entry exactly as they do on a fresh match.
6. **Delete** — a per-row action with a confirmation, removing the DB row and the entry
   folder. This is the only way entries are removed.
7. **Grouping by batch**: entries produced by a batch run are grouped under a collapsible
   header showing the batch's folder name, file count and status. Single matches appear
   ungrouped. Batch groups sort by recency alongside single entries.
8. **Empty state**: a muted message explaining that matches will appear here once run —
   never an empty void or a crash on a fresh install.
9. The list must handle a realistic library (hundreds of entries) without breaking the
   no-scroll-at-canvas-level rule — scroll *within* the library list is expected and fine;
   the window itself must still not scroll or clip.

## Part D — Batch folder processing

On the Match tab, add a control to process a whole folder:

1. **Pick a source folder**, then prompt for a **name for the output preset folder**.
   Presets are written to `<detected Serum presets folder>/PatchLab/<given name>/`,
   creating intermediate directories as needed. Sanitize the given name for filesystem
   safety and reject empty/invalid names with a clear message. If the folder already
   exists, ask whether to add to it or choose another name — **never silently overwrite an
   existing preset file**; disambiguate with a numeric suffix exactly as `load_in_serum`
   already does.
2. **Before starting**, show a confirmation summarizing: how many supported audio files
   were found, the chosen quality tier, the destination folder, and a rough time estimate
   derived from the tier. The user picks the quality tier (Quick / Balanced / Best
   Quality — reuse the existing budget values) and the target synth for this run;
   default both to the current Match tab settings.
3. **File discovery**: only `SUPPORTED_AUDIO_SUFFIXES`, non-recursive by default with a
   clear checkbox to include subfolders. Report the count of skipped/unsupported files
   rather than silently ignoring them.
4. **Execution**: process files sequentially through the existing `MatchProcessRunner`,
   archiving each result into the library as it completes (Part A) and exporting each
   preset through `ExportProcessRunner`. Do not spawn parallel plugin hosts — the render
   path is not designed for concurrent plugin instances.
5. **The app stays usable during a batch.** Progress (current file, N of M, elapsed,
   estimated remaining) shows in the Library tab's batch group and in the existing status
   bar / log pane. The user can browse the library, play entries, and open past matches
   while it runs. Starting a *second* batch while one is running is refused with a clear
   message; single one-off matches while a batch runs may either be queued or refused —
   pick one, implement it consistently, and state which you chose in your report.
6. **Cancel and resume**: a cancel control stops after the in-flight file finishes
   (never mid-render, which would leave a partial preset). Re-running the same source
   folder into the same batch skips files already completed — key on the source audio
   content hash, not the filename, and report how many were skipped.
7. **Per-file failure is not fatal**: log it, mark that file failed in the batch row,
   continue to the next. A batch finishes with an honest completed/failed/skipped tally.
8. **On completion, switch to the Library tab** with that batch's group expanded and
   visible, showing every processed sound.

## Part E — Verification gates

Report actual numbers and outcomes for each of these; do not assert success without
having run them:

1. **Persistence across restart**: run a match, close the app, relaunch — the entry is in
   the library, the source audio plays, an octave button renders and plays the generated
   patch, double-click reopens the full match with its closest-matches list intact, and
   Export Preset still produces a verified file. This directly proves the temp-session
   path-rewriting in A1/A2 is correct; it is the single most important gate here.
2. **Batch end-to-end**: assemble a small folder (5–8 files, include at least one
   unsupported file and one silent/no-confident-match file), run a batch, and confirm:
   presets land in the named folder, no pre-existing file was overwritten, every written
   preset passed round-trip verification, the library groups them under the batch name,
   the unsupported file was reported as skipped rather than silently dropped, and the
   no-confident-match file is present and honestly labeled rather than missing.
3. **Cancel/resume**: cancel a batch partway, restart it against the same folder, confirm
   completed files are skipped by content hash and the tally is correct.
4. **Migration**: upgrade a copy of the real existing database, confirm all pre-existing
   tables and row counts are unchanged and the new tables exist.
5. **UI gates**: `scripts/verify_visual_redesign.py` and `scripts/verify_milestone4_ui.py`
   both still pass. Add library/batch coverage to the visual gate in the same style as the
   existing assertions (tab switching works, library rows render with their octave
   buttons, no overlap/clipping at all three window sizes with a populated library).
6. **No regression** to the existing single-match flow, export round-trip verification,
   or the render library.

## Reporting

Report every gate result, with real numbers, before considering this done. No silent scope
cuts — anything you could not implement must appear in the report as a named limitation
with its reason, not be quietly dropped. Stop at this reporting gate rather than
continuing into further work.
