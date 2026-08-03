"""Cross-synth native preset export used by the Milestone 4 UI."""

from __future__ import annotations

import os
import shutil
import sqlite3
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from core.db import DEFAULT_DB_PATH
from core.branding import generated_preset_name
from core.features import CLAP_SAMPLE_RATE, ClapEmbedder
from core.fxp import build_fxp, parse_fxp
from core.matcher import loudness_normalize
from core.platform_env import ENV
from core.plugin_host import dump_dawdreamer_parameters, make_dawdreamer_processor
from core.serum2_preset import parse_serum2_preset
from core.serum2_state_reconstruct import (
    DEFAULT_RENDER_STATE_DIR,
    decode_host_template,
    reconstruct_vstpreset,
)
from core.serum2_preset_writer import Serum2WriteResult, write_serum2_preset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class PresetExportResult:
    path: Path
    synth: str
    mode: str
    base_preset_id: int
    applied_fields: int = 0
    skipped_fields: tuple[str, ...] = ()
    asset_references: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PresetExportVerification:
    export: PresetExportResult
    decoded_graph_equal: bool
    max_parameter_delta: float | None
    render_state_coverage: float
    clap_similarity: float
    expected_clap_similarity: float
    similarity_delta: float
    passed: bool

    @property
    def structurally_valid(self) -> bool:
        """Whether the native preset decoded and reconstructed safely."""

        return self.decoded_graph_equal and self.render_state_coverage >= 0.85


def _preset_row(preset_id: int, synth: str, db_path: Path) -> tuple[Path, str]:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT path,name FROM presets WHERE id=? AND synth=?", (preset_id, synth)
        ).fetchone()
    if row is None:
        raise KeyError(f"Unknown {synth} preset id {preset_id}")
    return Path(row[0]).resolve(), str(row[1])


def _extract_native_vst2_chunk(bank: bytes) -> tuple[bytes, bytes, int]:
    # DawDreamer 0.8.3 save_state emits a standard VST2 FBCh bank chunk:
    # 28-byte header, 128-byte future area, uint32 chunk size, opaque state.
    if len(bank) < 160 or bank[:4] != b"CcnK" or bank[8:12] != b"FBCh":
        raise RuntimeError("DawDreamer did not return a VST2 FBCh chunk bank")
    plugin_id = bank[16:20]
    plugin_version = struct.unpack_from(">I", bank, 20)[0]
    chunk_size = struct.unpack_from(">I", bank, 156)[0]
    payload = bank[160 : 160 + chunk_size]
    if len(payload) != chunk_size:
        raise RuntimeError("DawDreamer VST2 state chunk is truncated")
    return payload, plugin_id, plugin_version


def write_serum1_preset(
    output_path: Path,
    *,
    base_preset_id: int,
    vector: np.ndarray,
    meaningfully_modified: bool,
    name: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> PresetExportResult:
    base_path, _base_name = _preset_row(base_preset_id, "serum1", db_path)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not meaningfully_modified:
        source = parse_fxp(base_path)
        blob = build_fxp(
            source.payload,
            plugin_id=source.plugin_id,
            plugin_version=source.plugin_version,
            program_name=name or generated_preset_name("serum1"),
        )
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.replace(output_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        parse_fxp(output_path)
        return PresetExportResult(
            output_path, "serum1", "copied-native-branded", base_preset_id
        )

    candidate = next(
        item
        for item in ENV.plugins_for("serum1")
        if item.format == "VST2" and item.hostable
    )
    _engine, processor = make_dawdreamer_processor(candidate)
    if processor.load_preset(str(base_path)) is False:
        raise RuntimeError(f"Serum 1 rejected base preset {base_path}")
    for index, value in enumerate(np.asarray(vector, dtype=np.float32)):
        processor.set_parameter(index, float(value))
    with tempfile.NamedTemporaryFile(suffix=".state", delete=False) as handle:
        state_path = Path(handle.name)
    try:
        processor.save_state(str(state_path))
        payload, plugin_id, plugin_version = _extract_native_vst2_chunk(
            state_path.read_bytes()
        )
    finally:
        state_path.unlink(missing_ok=True)
    blob = build_fxp(
        payload,
        plugin_id=plugin_id,
        plugin_version=plugin_version,
        program_name=name or generated_preset_name("serum1"),
    )
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    parse_fxp(output_path)
    return PresetExportResult(
        output_path, "serum1", "optimized-native-state", base_preset_id
    )


def write_native_preset(
    output_path: Path,
    *,
    synth: str,
    base_preset_id: int,
    vector: np.ndarray,
    mask: np.ndarray,
    meaningfully_modified: bool,
    name: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    structural_overrides: dict[str, Any] | None = None,
) -> PresetExportResult:
    if synth == "serum1":
        return write_serum1_preset(
            output_path,
            base_preset_id=base_preset_id,
            vector=vector,
            meaningfully_modified=meaningfully_modified,
            name=name,
            db_path=db_path,
        )
    if synth != "serum2":
        raise ValueError(f"Unknown synth {synth!r}")
    result: Serum2WriteResult = write_serum2_preset(
        output_path,
        base_preset_id=base_preset_id,
        vector=vector,
        mask=mask,
        meaningfully_modified=meaningfully_modified,
        name=name,
        db_path=db_path,
        structural_overrides=structural_overrides,
    )
    return PresetExportResult(
        path=result.path,
        synth="serum2",
        mode=result.mode,
        base_preset_id=result.base_preset_id,
        applied_fields=result.applied_fields,
        skipped_fields=result.skipped_fields,
        asset_references=result.asset_references,
    )


class PresetExportVerifier:
    """Persistent plug-in hosts for mandatory decoded-state and audio checks."""

    def __init__(self) -> None:
        self.hosts: dict[str, tuple[Any, Any]] = {}
        for synth, required in (("serum1", "VST2"), ("serum2", "VST3")):
            plugin = next(
                item
                for item in ENV.plugins_for(synth)
                if item.format == required and item.hostable
            )
            self.hosts[synth] = make_dawdreamer_processor(plugin)
        self.embedder = ClapEmbedder(ENV)
        self._temporary = tempfile.TemporaryDirectory(prefix="patchlab-export-verify-")

    def close(self) -> None:
        self._temporary.cleanup()

    def verify(
        self,
        export: PresetExportResult,
        *,
        intended_vector: np.ndarray,
        midi_note: int,
        target_audio: np.ndarray,
        expected_clap_similarity: float,
        duration_s: float | None = None,
        similarity_tolerance: float = 0.15,
    ) -> PresetExportVerification:
        target = np.asarray(target_audio, dtype=np.float32).reshape(-1)
        duration = duration_s or min(4.0, len(target) / CLAP_SAMPLE_RATE)
        engine, processor = self.hosts[export.synth]
        decoded_equal = True
        max_parameter_delta: float | None = None
        coverage = 1.0

        if export.synth == "serum1":
            if processor.load_preset(str(export.path)) is False:
                raise RuntimeError(f"Serum 1 rejected exported preset {export.path}")
            actual = np.asarray(
                [item.norm_value for item in dump_dawdreamer_parameters(processor)],
                dtype=np.float32,
            )
            intended = np.asarray(intended_vector, dtype=np.float32)
            if actual.shape != intended.shape:
                raise RuntimeError(
                    f"Serum 1 export exposed {len(actual)} parameters; expected {len(intended)}"
                )
            max_parameter_delta = float(np.max(np.abs(actual - intended)))
            decoded_equal = max_parameter_delta <= 1e-4
        else:
            parsed = parse_serum2_preset(export.path)
            template = decode_host_template(
                (DEFAULT_RENDER_STATE_DIR / f"{export.base_preset_id}.vstpreset").read_bytes()
            )
            vstpreset, partition = reconstruct_vstpreset(parsed, template)
            coverage = partition.coverage
            state_path = (
                Path(self._temporary.name)
                / f"{export.base_preset_id}-{export.path.stem}.vstpreset"
            )
            state_path.write_bytes(vstpreset)
            if processor.load_vst3_preset(str(state_path)) is False:
                raise RuntimeError(f"Serum 2 rejected reconstructed export {export.path}")

        if hasattr(processor, "clear_midi"):
            processor.clear_midi()
        processor.add_midi_note(int(midi_note), 100, 0.0, duration)
        engine.render(duration)
        audio = np.asarray(engine.get_audio(), dtype=np.float32)
        if audio.shape[0] != 2 and audio.shape[1] == 2:
            audio = audio.T
        mono = np.mean(audio, axis=0, dtype=np.float32)
        rendered = librosa.resample(
            mono,
            orig_sr=44_100,
            target_sr=CLAP_SAMPLE_RATE,
            res_type="soxr_hq",
        ).astype(np.float32)
        normalized_target = loudness_normalize(target[: len(rendered)])
        normalized_render = loudness_normalize(rendered[: len(normalized_target)])
        embeddings = self.embedder.embed([normalized_target, normalized_render])
        similarity = float(
            np.clip(np.dot(embeddings[0], embeddings[1]), -1.0, 1.0)
        )
        delta = similarity - float(expected_clap_similarity)
        passed = (
            decoded_equal
            and coverage >= 0.85
            and abs(delta) <= similarity_tolerance
        )
        return PresetExportVerification(
            export=export,
            decoded_graph_equal=decoded_equal,
            max_parameter_delta=max_parameter_delta,
            render_state_coverage=coverage,
            clap_similarity=similarity,
            expected_clap_similarity=float(expected_clap_similarity),
            similarity_delta=delta,
            passed=passed,
        )


def write_and_verify_native_preset(
    output_path: Path,
    *,
    synth: str,
    base_preset_id: int,
    vector: np.ndarray,
    mask: np.ndarray,
    meaningfully_modified: bool,
    midi_note: int,
    target_audio: np.ndarray,
    expected_clap_similarity: float,
    verifier: PresetExportVerifier | None = None,
    name: str | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    similarity_tolerance: float = 0.15,
    structural_overrides: dict[str, Any] | None = None,
) -> PresetExportVerification:
    """Write and reload a native preset, rejecting only structural failures.

    CLAP comparison is deliberately advisory. Plug-in rendering can vary with
    note duration, release state, random phase, or assets even when the native
    preset is valid. A similarity mismatch is returned to the caller as a
    warning instead of deleting a preset that decoded and reloaded correctly.
    """

    owns_verifier = verifier is None
    verifier = verifier or PresetExportVerifier()
    try:
        export = write_native_preset(
            output_path,
            synth=synth,
            base_preset_id=base_preset_id,
            vector=vector,
            mask=mask,
            meaningfully_modified=meaningfully_modified,
            name=name,
            db_path=db_path,
            structural_overrides=structural_overrides,
        )
        verification = verifier.verify(
            export,
            intended_vector=vector,
            midi_note=midi_note,
            target_audio=target_audio,
            expected_clap_similarity=expected_clap_similarity,
            similarity_tolerance=similarity_tolerance,
        )
        if not verification.structurally_valid:
            export.path.unlink(missing_ok=True)
            raise RuntimeError(
                "Preset export structural verification failed: "
                f"decoded={verification.decoded_graph_equal}, "
                f"coverage={verification.render_state_coverage:.3f}"
            )
        return verification
    finally:
        if owns_verifier:
            verifier.close()


def commit_temporary_export(temporary_path: Path, output_path: Path) -> Path:
    """Atomically publish a temporary preset and remove its private source.

    The generated preset may live on a different volume from the destination,
    so it is copied into a staging file beside the requested output, flushed,
    and atomically replaced there. The private temporary preset is unlinked
    only after the destination is complete.
    """

    source = Path(temporary_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Temporary preset is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        staging = Path(handle.name)
        with source.open("rb") as source_handle:
            shutil.copyfileobj(source_handle, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        staging.replace(destination)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    source.unlink()
    return destination
