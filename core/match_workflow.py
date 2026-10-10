"""File-to-result workflow shared by the Match UI and integration gates."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import soundfile as sf

from core.audio_input import DecodedAudio, decode_audio_file
from core.branding import display_match_name, generated_preset_name
from core.matcher import AnalysisBySynthesisMatcher, Candidate, SearchConfig
from core.prepared_state import audible_notes_from_mask
from core.serum2_preset_writer import vector_was_modified


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = PROJECT_ROOT / "data" / "matches"
BUDGETS: dict[str, SearchConfig] = {
    "quick": SearchConfig(max_evaluations=51, max_seconds=30.0),
    "balanced": SearchConfig(max_evaluations=300, max_seconds=120.0),
    "best": SearchConfig(max_evaluations=600, max_seconds=300.0),
}
SECTION_LABELS = {
    "oscillators": "Oscillators",
    "filters": "Filter",
    "envelopes": "Envelopes",
    "lfos": "LFOs",
    "fx": "FX",
    "mod_matrix": "Mod Matrix",
    "macros": "Macros",
    "global_other": "Global",
}


def _emit(
    callback: Callable[[dict[str, Any]], None] | None, detail: dict[str, Any]
) -> None:
    if callback is not None:
        callback(detail)


def _section_for_serum1(name: str) -> str:
    value = name.casefold()
    if any(token in value for token in ("osc a", "a osc", "oscillator a")):
        return "Osc A"
    if any(token in value for token in ("osc b", "b osc", "oscillator b")):
        return "Osc B"
    if "sub" in value:
        return "Sub"
    if "noise" in value:
        return "Noise"
    if "filter" in value:
        return "Filter"
    if "env" in value or "envelope" in value:
        return "Envelopes"
    if "lfo" in value:
        return "LFOs"
    if any(
        token in value
        for token in (
            "fx",
            "hyper",
            "dist",
            "flang",
            "chorus",
            "delay",
            "reverb",
            "compress",
            "eq ",
        )
    ):
        return "FX"
    if "mod " in value or "matrix" in value:
        return "Mod Matrix"
    return "Global"


def _preset_details(
    preset_ids: list[int],
    database_path: Path,
) -> dict[int, dict[str, Any]]:
    placeholders = ",".join("?" for _ in preset_ids)
    with sqlite3.connect(database_path) as connection:
        # A packaged install reads the bundled synthesis catalog here, which
        # carries only ``presets`` and ``params``: per-note coverage is something
        # a *prepared local library* records. Joining a table the catalog does not
        # have failed every Match in the packaged app, so the join is used only
        # where the table exists; without it a preset simply offers the full
        # note range, as it always did.
        #
        # The fallback is for the bundled catalog ONLY. A database that is
        # recognisably a PatchLab library (it has the preparation bookkeeping
        # tables) but lacks the coverage table is damaged or half-migrated, and
        # quietly offering every note there would audition and export silent
        # notes -- so that case fails loudly instead.
        tables = {
            str(name)
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "fingerprint_note_coverage" in tables:
            query = (
                f"SELECT p.id,p.name,p.synth,p.path,p.content_hash,nc.note_mask "
                f"FROM presets p LEFT JOIN fingerprint_note_coverage nc "
                f"ON nc.preset_id=p.id WHERE p.id IN ({placeholders})"
            )
        elif tables & {"prepared_presets", "schema_migrations", "fingerprints"}:
            raise RuntimeError(
                f"The preset library database {database_path} is missing its "
                "playable-note coverage table (fingerprint_note_coverage); it "
                "looks damaged or only partly upgraded. Reopen the library in "
                "PatchLab to repair it, or reprepare it."
            )
        else:
            query = (
                f"SELECT p.id,p.name,p.synth,p.path,p.content_hash,NULL "
                f"FROM presets p WHERE p.id IN ({placeholders})"
            )
        rows = connection.execute(query, tuple(preset_ids)).fetchall()
    return {
        int(row[0]): {
            "name": str(row[1]),
            "synth": str(row[2]),
            "source_path": str(row[3]),
            "content_hash": str(row[4]),
            "audible_midi_notes": list(audible_notes_from_mask(row[5])),
        }
        for row in rows
    }


def _nearest_render(
    preset_id: int,
    midi_note: int,
    audio_root: Path,
    audible_notes: tuple[int, ...] | None = None,
) -> tuple[int, Path | None]:
    notes = audible_notes or (24, 36, 48, 60, 72, 84, 96)
    ordered = sorted(notes, key=lambda value: abs(value - midi_note))
    for note in ordered:
        path = audio_root / str(preset_id) / f"{note}.wav"
        if path.is_file():
            return note, path
    return ordered[0], None


def _serum1_settings(
    candidate: Candidate,
    base_vector: np.ndarray,
    database_path: Path,
) -> dict[str, dict[str, Any]]:
    from core.plugin_host import dump_dawdreamer_parameters
    from core.renderer_selection import open_renderer

    _engine, processor, _selection = open_renderer(
        "serum1", context="summarising Serum 1 candidate settings"
    )
    with sqlite3.connect(database_path) as connection:
        path = connection.execute(
            "SELECT path FROM presets WHERE id=?", (candidate.base_preset_id,)
        ).fetchone()[0]
    if processor.load_preset(str(path)) is False:
        raise RuntimeError("Could not format Serum 1 candidate settings")
    for index, value in enumerate(candidate.vector):
        processor.set_parameter(index, float(value))
    dumped = dump_dawdreamer_parameters(processor)
    groups: dict[str, dict[str, Any]] = {}
    for item in dumped:
        group = groups.setdefault(
            _section_for_serum1(item.name), {"changed": [], "matches_base_count": 0}
        )
        changed = abs(float(candidate.vector[item.index]) - float(base_vector[item.index])) >= 0.02
        if changed:
            group["changed"].append(
                {
                    "index": item.index,
                    "name": item.name,
                    "value": item.display_value,
                    "normalized": float(candidate.vector[item.index]),
                }
            )
        else:
            group["matches_base_count"] += 1
    return dict(sorted(groups.items()))


def _schema_value(field: Mapping[str, Any], vector: np.ndarray) -> Any:
    index = int(field["index"])
    if field["encoding"] == "one_hot":
        width = int(field["width"])
        return field["categories"][int(np.argmax(vector[index : index + width]))]
    minimum, maximum = float(field["minimum"]), float(field["maximum"])
    return minimum + float(np.clip(vector[index], 0.0, 1.0)) * (maximum - minimum)


def _display_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return "None"
    return str(value)


def _serum2_settings(
    candidate: Candidate, base_vector: np.ndarray, schema: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for field in schema["fields"]:
        index = int(field["index"])
        width = int(field.get("width", 1))
        if not bool(candidate.mask[index]):
            continue
        group_name = SECTION_LABELS.get(
            str(field.get("category", "global_other")), "Global"
        )
        group = groups.setdefault(
            group_name, {"changed": [], "matches_base_count": 0}
        )
        if field["encoding"] == "one_hot":
            changed = int(
                np.argmax(candidate.vector[index : index + width])
            ) != int(np.argmax(base_vector[index : index + width]))
        else:
            changed = (
                abs(float(candidate.vector[index]) - float(base_vector[index]))
                >= 0.02
            )
        if changed:
            value = _schema_value(field, candidate.vector)
            group["changed"].append(
                {
                    "index": index,
                    "name": str(field["name"]),
                    "value": _display_value(value),
                    "normalized": float(candidate.vector[index]),
                }
            )
        else:
            group["matches_base_count"] += 1
    return dict(sorted(groups.items()))


def _candidate_payload(
    matcher: AnalysisBySynthesisMatcher, candidate: Candidate
) -> tuple[dict[str, Any], np.ndarray]:
    code = 1 if candidate.synth == "serum1" else 2
    store = matcher.stores[code]
    row = store.preset_row[candidate.base_preset_id]
    base = np.asarray(store.vectors[row], dtype=np.float32)
    if candidate.synth == "serum1":
        settings = _serum1_settings(
            candidate,
            base,
            matcher.assets.library_db,
        )
    else:
        settings = _serum2_settings(
            candidate, base, matcher.absolute_checkpoint["serum2_schema"]
        )
    return settings, base


def _silence_result(decoded: DecodedAudio, session: Path, target_synth: str) -> Path:
    payload = {
        "status": "no_confident_match",
        "no_confident_match": True,
        "message": "No confident match — the selected audio is silent.",
        "source": {
            "path": str(decoded.path),
            "decoder": decoded.decoder,
            "source_duration_s": decoded.source_duration_s,
            "start_offset_s": decoded.start_offset_s,
            "used_duration_s": decoded.used_duration_s,
            "rms_dbfs": decoded.rms_dbfs,
        },
        "target_synth": target_synth,
        "existing_matches": [],
        "recommendation": None,
    }
    result_path = session / "result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result_path


def _retire_matcher(
    matcher: AnalysisBySynthesisMatcher,
    cache: "WarmMatcherCache | None",
    *,
    healthy: bool,
) -> None:
    """End of a Match: keep a healthy warm matcher, otherwise close it."""

    matcher.tracker = None
    if cache is not None:
        cache.release(matcher, healthy=healthy)
    else:
        matcher.close()


class WarmMatcherCache:
    """One fully loaded matcher kept alive between Matches (the warm engine).

    A matcher owns everything slow to build: the CLAP model, the parameter and
    delta models, the similarity indexes and the pool of Serum render workers.
    None of it depends on the sound being matched, so a long-lived process can
    build it once. Anything unexpected (a failed Match, a stalled or respawning
    pool, a different target generation or worker count) discards it, and the
    next Match builds a fresh one -- the same state a cold start would have.
    """

    def __init__(self, embedder: Any = None) -> None:
        self.matcher: AnalysisBySynthesisMatcher | None = None
        self._key: tuple[Any, ...] | None = None
        self._embedder = embedder
        self.reuse_count = 0

    @property
    def embedder(self) -> Any:
        """The CLAP model shared by the matcher and the export verifier.

        Normally it comes from the first matcher built (so the matcher's render
        workers start up while CLAP loads, as in a one-shot Match); an export
        that arrives before any Match loads it itself and the matcher reuses it.
        """

        if self._embedder is None:
            from core.features import ClapEmbedder
            from core.platform_env import ENV

            self._embedder = ClapEmbedder(ENV)
        return self._embedder

    def is_warm_for(self, required_synths: tuple[str, ...], processes: int) -> bool:
        return self.matcher is not None and self._key == (
            tuple(required_synths), int(processes), False
        )

    def acquire(
        self,
        *,
        processes: int,
        deterministic_render_dispatch: bool,
        required_synths: tuple[str, ...],
        operation_id: str,
    ) -> AnalysisBySynthesisMatcher:
        key = (tuple(required_synths), int(processes), bool(deterministic_render_dispatch))
        if self.matcher is not None and self._key == key and self.matcher.is_usable():
            self.reuse_count += 1
            self.matcher.begin_operation(operation_id)
            return self.matcher
        self.discard()
        matcher = AnalysisBySynthesisMatcher(
            processes=processes,
            deterministic_render_dispatch=deterministic_render_dispatch,
            required_synths=required_synths,
            operation_id=operation_id,
            embedder=self._embedder,
        )
        if self._embedder is None:
            self._embedder = getattr(matcher, "embedder", None)
        self.matcher, self._key = matcher, key
        return matcher

    def release(self, matcher: AnalysisBySynthesisMatcher, *, healthy: bool) -> None:
        if matcher is not self.matcher:
            matcher.close()
            return
        if not healthy or not matcher.is_usable():
            self.discard()

    def discard(self) -> None:
        matcher, self.matcher, self._key = self.matcher, None, None
        if matcher is not None:
            try:
                matcher.close()
            except Exception:
                pass

    def close(self) -> None:
        self.discard()


def run_match_file(
    input_path: Path,
    *,
    target_synth: str = "serum2",
    budget: str = "balanced",
    start_offset_s: float = 0.0,
    session_root: Path = DEFAULT_SESSION_ROOT,
    matcher_processes: int | None = None,
    deterministic_render_dispatch: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    matcher_cache: "WarmMatcherCache | None" = None,
) -> Path:
    """Decode a user file and persist the complete UI result artifact.

    ``matcher_cache`` is supplied only by the warm engine: it keeps one fully
    loaded matcher (CLAP, models, Serum render workers) alive between files so
    a second Match or a batch's next file skips the whole start-up cost. Without
    one, behaviour is exactly the original: build a matcher, use it, close it.
    """

    if matcher_processes is None:
        from core.platform_env import recommended_render_workers

        matcher_processes = recommended_render_workers()

    from core.diagnostic_env import capture_environment, write_environment
    from core.diagnostics import inherited_operation_id, new_operation_id, record
    from core.local_library import default_local_paths
    from core.operation_state import (
        MATCH_PHASES,
        StallWatchdog,
        capture_postmortem,
        start_operation,
        write_postmortem,
    )

    if target_synth not in {"serum1", "serum2"}:
        raise ValueError("target_synth must be serum1 or serum2")
    if budget not in BUDGETS:
        raise ValueError(f"Unknown search budget {budget!r}")

    operation_id = inherited_operation_id() or new_operation_id("match")
    tracker = start_operation(
        "match",
        subsystem="match",
        phases=MATCH_PHASES,
        operation_id=operation_id,
        target_synth=target_synth,
        quality_mode=budget,
        matcher_processes=matcher_processes,
    )

    def _phase(name: str, reason: str = "", **detail: Any) -> None:
        """Advance the state machine and the UI progress stream together."""

        tracker.enter_phase(name, reason=reason)
        _emit(progress_callback, {"phase": name, "evaluations": 0, **detail})

    session = Path(session_root) / uuid.uuid4().hex
    session.mkdir(parents=True, exist_ok=False)

    # One environment snapshot per operation, captured before the expensive
    # work, so a failure report always describes the machine it happened on.
    try:
        paths = default_local_paths()
        environment = capture_environment(
            operation="match",
            operation_id=operation_id,
            db_path=paths["db"],
            target_synth=target_synth,
            quality_mode=budget,
            settings={
                "matcher_processes": matcher_processes,
                "deterministic_render_dispatch": deterministic_render_dispatch,
                "start_offset_s": start_offset_s,
            },
        )
        write_environment(environment)
    except Exception as exc:  # diagnostics must never break Match
        environment = None
        record(
            "match",
            "environment_snapshot_failed",
            f"environment capture failed: {type(exc).__name__}: {exc}",
            severity="warning",
            operation_id=operation_id,
            exception=exc,
        )

    def _postmortem(trigger: str, exception: BaseException | None = None) -> dict[str, Any]:
        supervisor = getattr(matcher, "supervisor", None)
        return capture_postmortem(
            tracker,
            exception=exception,
            trigger=trigger,
            workers=supervisor.worker_states() if supervisor is not None else (),
            disk_paths=[session],
            extra={"session": str(session)},
        )

    matcher = None
    watchdog: StallWatchdog | None = None
    try:
        _phase("decoding", "decoding the user's audio file")
        decoded = decode_audio_file(input_path, start_offset_s=start_offset_s)
        tracker.mark_progress(text="audio decoded", samples=int(len(decoded.mono)))
        if decoded.silent:
            tracker.complete("input audio is silent; returning the silence result")
            _emit(
                progress_callback,
                {"phase": "complete", "evaluations": 0, "best_clap_cosine": 0.0},
            )
            return _silence_result(decoded, session, target_synth)

        if matcher_processes <= 0:
            raise ValueError("matcher_processes must be positive")

        # Renderer discovery and validation happen inside the matcher's
        # constructor, in the parent process, before any worker is spawned.
        _phase(
            "renderer-discovery",
            f"resolving the renderer required for a {target_synth} match",
        )
        _phase("loading-models", "loading CLAP, parameter and delta models")
        if matcher_cache is not None:
            matcher = matcher_cache.acquire(
                processes=matcher_processes,
                deterministic_render_dispatch=deterministic_render_dispatch,
                required_synths=(target_synth,),
                operation_id=operation_id,
            )
        else:
            matcher = AnalysisBySynthesisMatcher(
                processes=matcher_processes,
                deterministic_render_dispatch=deterministic_render_dispatch,
                required_synths=(target_synth,),
                operation_id=operation_id,
            )
        matcher.tracker = tracker
        tracker.annotate(
            renderer=matcher.renderer_preflight.selection_for(target_synth).renderer,
            pool_id=matcher.pool_id,
        )
        tracker.mark_progress(text="render pool ready", pool_id=matcher.pool_id)

        # Liveness: a phase-aware stall detector that captures a postmortem
        # BEFORE it terminates anything, because recovery destroys the worker
        # and stack state that explains a hang.
        def _on_stall(verdict: Any, postmortem: dict[str, Any]) -> None:
            write_postmortem(postmortem)
            if matcher is not None:
                matcher.supervisor.shutdown()

        watchdog = StallWatchdog(
            tracker,
            on_stall=_on_stall,
            postmortem_factory=lambda verdict: _postmortem("stall"),
        )
        watchdog.start()
    except BaseException as exc:
        write_postmortem(_postmortem("failure", exc))
        tracker.fail(exc)
        if matcher is not None:
            _retire_matcher(matcher, matcher_cache, healthy=False)
        raise
    healthy = False
    try:
        _phase("candidate-preparation", "embedding the target and retrieving neighbours")
        embedding = matcher.query_embedding(decoded.mono, decoded.sample_rate)
        retrieval = matcher.retrieve_existing(embedding, 10)
        tracker.set_count("retrieved_presets", len(retrieval))
        tracker.mark_progress(text="candidates retrieved", retrieved=len(retrieval))
        # Record that pending presets are simply absent from retrieval -- Match
        # is never blocked by them. Stated explicitly so a future reader of a
        # support bundle does not mistake a partly-processed library for a
        # broken Match.
        try:
            from core.capability_ux import record_pending_exclusion
            from core.db import Database

            # Resolved locally rather than reusing the snapshot block's `paths`,
            # which is only bound if that block succeeded.
            database_path = Path(default_local_paths()["db"])
            if database_path.is_file():
                database = Database(database_path)
                coverage = database.library_coverage()
                record_pending_exclusion(
                    pending_counts=coverage.get("pending_by_reason", {}),
                    learned=int(coverage.get("learned", 0)),
                    operation_id=operation_id,
                )
        except Exception:
            pass
        detail = _preset_details(
            [preset_id for preset_id, _score in retrieval],
            matcher.assets.library_db,
        )

        def search_progress(value: dict[str, Any]) -> None:
            # Every reported search generation is real forward progress, which
            # is what keeps the stall detector quiet during a legitimately long
            # search and noisy when one genuinely wedges.
            tracker.mark_progress(
                text="search generation complete",
                evaluations=value.get("evaluations"),
                generation=value.get("generation"),
            )
            _emit(progress_callback, {"phase": "searching", **value})

        tracker.enter_phase("evaluation", reason="running analysis-by-synthesis search")

        result = matcher.match(
            decoded.mono,
            decoded.sample_rate,
            synth_hint=target_synth,
            config=replace(
                BUDGETS[budget],
                structural_search=(
                    os.environ.get("PATCHLAB_SERUM2_STRUCTURAL_SEARCH", "0").strip()
                    == "1"
                ),
                structural_routes=(
                    os.environ.get("PATCHLAB_SERUM2_STRUCTURAL_ROUTES", "0").strip()
                    == "1"
                ),
                structural_route_probe_evaluations=(
                    int(os.environ["PATCHLAB_SERUM2_ROUTE_PROBE_EVALUATIONS"])
                    if os.environ.get("PATCHLAB_SERUM2_ROUTE_PROBE_EVALUATIONS")
                    else None
                ),
            ),
            target_embedding=embedding,
            progress_callback=search_progress,
        )
        existing = []
        for preset_id, score in retrieval:
            row = detail[preset_id]
            note, wav_path = _nearest_render(
                preset_id,
                result.midi_note,
                matcher.audio_root,
                tuple(row["audible_midi_notes"]),
            )
            # Pre-rendered auditions only exist where the full render library
            # was built locally; an install that never rendered has none. Carry
            # the preset file forward so the row can still be rendered on
            # demand, exactly as the factory-fingerprint path does. Without
            # this every closest match reports "No local audio or factory
            # preset is available" and its octave buttons stay dead.
            source_path = str(row.get("source_path") or "")
            preview_source = (
                source_path
                if source_path and not wav_path and Path(source_path).is_file()
                else None
            )
            existing.append(
                {
                    "preset_id": preset_id,
                    **row,
                    "similarity": score,
                    "similarity_percent": 100.0 * score,
                    "audition_midi_note": note,
                    "audition_path": str(wav_path) if wav_path else None,
                    "preview_source_path": preview_source,
                    "local_source_available": bool(wav_path or preview_source),
                }
            )
        settings, base_vector = _candidate_payload(matcher, result.best)
        modified = (
            bool(result.best.structural_overrides)
            or (
                not result.best.exact_base
                and vector_was_modified(result.best.vector, base_vector, result.best.mask)
            )
        )
        winner_path = session / "winner.wav"
        if result.best.waveform is None:
            raise RuntimeError("Matcher returned no winner audition waveform")
        sf.write(
            winner_path,
            result.best.waveform,
            48_000,
            subtype="FLOAT",
            format="WAV",
        )
        candidate_path = session / "candidate.npz"
        np.savez_compressed(
            candidate_path,
            vector=np.asarray(result.best.vector, dtype=np.float32),
            mask=np.asarray(result.best.mask, dtype=np.bool_),
            base_vector=base_vector,
            structural_overrides_json=np.asarray(
                json.dumps(result.best.structural_overrides, sort_keys=True)
            ),
        )
        no_confident = result.best.clap_cosine < 0.65
        payload = {
            "status": "complete",
            "no_confident_match": no_confident,
            "message": (
                "Low confidence — this sound may not be well-suited to Serum."
                if no_confident
                else "Match complete"
            ),
            "source": {
                "path": str(decoded.path),
                "decoder": decoded.decoder,
                "source_duration_s": decoded.source_duration_s,
                "start_offset_s": decoded.start_offset_s,
                "used_duration_s": decoded.used_duration_s,
                "rms_dbfs": decoded.rms_dbfs,
            },
            "budget": budget,
            "target_synth": target_synth,
            "detected": {
                "midi_note": result.midi_note,
                "acoustic_midi_note": result.acoustic_midi_note,
                "frequency_hz": result.detected_hz,
                "pyin_confidence": result.pitch_confidence,
                "sub_bass_fraction": result.sub_bass_fraction,
                "unpitched_fallback": result.unpitched_fallback,
                "note_hypotheses": list(result.note_hypotheses),
                "comparison_duration_s": result.comparison_duration_s,
            },
            "existing_matches": [
                {
                    **item,
                    "name": display_match_name(item.get("name"), index),
                }
                for index, item in enumerate(existing, start=1)
            ],
            "recommendation": {
                "synth": result.best.synth,
                "base_preset_id": result.best.base_preset_id,
                "base_name": generated_preset_name(result.best.synth),
                "origin": result.best.origin,
                "meaningfully_modified": modified,
                "clap_similarity": result.best.clap_cosine,
                "similarity_percent": 100.0 * result.best.clap_cosine,
                "stft_loss": result.best.stft_loss,
                "objective": result.best.objective,
                "stft_weight": result.stft_weight,
                "clap_weight": result.clap_weight,
                "evaluations": result.evaluations,
                "elapsed_s": result.elapsed_s,
                "winner_audio_path": str(winner_path),
                "candidate_path": str(candidate_path),
                "settings": settings,
                "structural_overrides": result.best.structural_overrides,
                "objective_trace": result.objective_trace,
            },
        }
        result_path = session / "result.json"
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _emit(
            progress_callback,
            {
                "phase": "complete",
                "evaluations": result.evaluations,
                "budget": BUDGETS[budget].max_evaluations
                + (
                    (
                        BUDGETS[budget].structural_max_evaluations_ceiling
                        if os.environ.get(
                            "PATCHLAB_SERUM2_STRUCTURAL_ROUTES", "0"
                        ).strip()
                        == "1"
                        else BUDGETS[budget].structural_max_evaluations
                    )
                    if os.environ.get(
                        "PATCHLAB_SERUM2_STRUCTURAL_SEARCH", "0"
                    ).strip()
                    == "1"
                    else 0
                ),
                "best_clap_cosine": result.best.clap_cosine,
            },
        )
        tracker.set_count("evaluations", int(result.evaluations))
        tracker.complete(
            "match produced a result artifact",
            evaluations=int(result.evaluations),
        )
        healthy = True
        return result_path
    except BaseException as exc:
        # Nothing is swallowed: the exception, its cause chain, the failing
        # phase and the worker/process state are all preserved before cleanup.
        write_postmortem(_postmortem("failure", exc))
        tracker.fail(exc)
        raise
    finally:
        if watchdog is not None:
            watchdog.stop()
        _retire_matcher(matcher, matcher_cache, healthy=healthy)
