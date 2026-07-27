"""Headless Serum hosting and verified preset-loading strategies."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.fxp import FxpError, parse_fxp
from core.platform_env import PlatformEnv, PluginCandidate, SynthVersion


SAMPLE_RATE = 44_100
BLOCK_SIZE = 512
SILENCE_DBFS = -60.0


@dataclass(frozen=True, slots=True)
class ParameterValue:
    index: int
    name: str
    norm_value: float
    display_value: str


@dataclass(frozen=True, slots=True)
class StrategyAttempt:
    strategy: str
    plugin_path: str
    passed: bool
    detail: str


@dataclass(slots=True)
class LoadedPreset:
    strategy: str
    plugin_format: str
    plugin_path: Path
    host: str
    parameters: list[ParameterValue]
    changed_from_init: int
    rms_dbfs: float
    peak_dbfs: float
    processor: Any = field(repr=False)
    engine: Any = field(default=None, repr=False)
    attempts: list[StrategyAttempt] = field(default_factory=list)


class PresetLoadError(RuntimeError):
    def __init__(self, message: str, attempts: Iterable[StrategyAttempt] = ()) -> None:
        super().__init__(message)
        self.attempts = list(attempts)


def audio_levels(audio: np.ndarray) -> tuple[float, float]:
    values = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    floor = np.finfo(np.float64).tiny
    return 20.0 * math.log10(max(peak, floor)), 20.0 * math.log10(max(rms, floor))


def _display_text(processor: Any, index: int, value: float) -> str:
    for method_name in ("get_parameter_text", "get_parameter_display"):
        method = getattr(processor, method_name, None)
        if method is None:
            continue
        for args in ((index,), (index, value)):
            try:
                return str(method(*args))
            except (TypeError, RuntimeError):
                pass
    return f"{value:.6g}"


def dawdreamer_parameter_display(processor: Any, index: int, value: float | None = None) -> str:
    """Return the plug-in's own formatted text for one normalized parameter."""

    current = float(processor.get_parameter(index)) if value is None else float(value)
    return _display_text(processor, index, current)


def dump_dawdreamer_parameters(processor: Any) -> list[ParameterValue]:
    descriptions = processor.get_parameters_description()
    result: list[ParameterValue] = []
    for fallback_index, description in enumerate(descriptions):
        if isinstance(description, dict):
            index = int(description.get("index", fallback_index))
            name = str(description.get("name", description.get("label", f"Parameter {index}")))
        else:
            index = fallback_index
            name = str(description)
        value = float(processor.get_parameter(index))
        result.append(ParameterValue(index, name, value, _display_text(processor, index, value)))
    return result


def dump_pedalboard_parameters(processor: Any) -> list[ParameterValue]:
    result: list[ParameterValue] = []
    for index, (name, parameter) in enumerate(processor.parameters.items()):
        raw = getattr(parameter, "raw_value", None)
        if raw is None:
            raw = getattr(parameter, "value", 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        display = str(getattr(parameter, "value", raw))
        result.append(ParameterValue(index, str(name), value, display))
    return result


def changed_parameter_count(
    initial: list[ParameterValue], loaded: list[ParameterValue], threshold: float = 1e-4
) -> int:
    initial_by_index = {item.index: item.norm_value for item in initial}
    return sum(
        1
        for item in loaded
        if item.index in initial_by_index
        and abs(item.norm_value - initial_by_index[item.index]) > threshold
    )


def parameter_vectors_differ(
    first: list[ParameterValue], second: list[ParameterValue], threshold: float = 1e-4
) -> bool:
    left = {item.index: item.norm_value for item in first}
    right = {item.index: item.norm_value for item in second}
    common = left.keys() & right.keys()
    return any(abs(left[index] - right[index]) > threshold for index in common)


def make_dawdreamer_processor(candidate: PluginCandidate) -> tuple[Any, Any]:
    import dawdreamer as daw

    engine = daw.RenderEngine(SAMPLE_RATE, BLOCK_SIZE)
    processor = engine.make_plugin_processor("serum", str(candidate.path))
    engine.load_graph([(processor, [])])
    return engine, processor


def render_dawdreamer_note(
    engine: Any, processor: Any, *, midi_note: int = 60, duration: float = 1.0
) -> np.ndarray:
    if hasattr(processor, "clear_midi"):
        processor.clear_midi()
    note_duration = max(0.05, duration - 0.1)
    processor.add_midi_note(midi_note, 100, 0.0, note_duration)
    engine.render(duration)
    return np.asarray(engine.get_audio(), dtype=np.float32)


def render_pedalboard_note(
    processor: Any, *, midi_note: int = 60, duration: float = 1.0
) -> np.ndarray:
    from mido import Message

    return np.asarray(
        processor(
            [
                Message("note_on", note=midi_note, velocity=100),
                Message("note_off", note=midi_note, velocity=0, time=max(0.05, duration - 0.1)),
            ],
            duration=duration,
            sample_rate=SAMPLE_RATE,
            num_channels=2,
        ),
        dtype=np.float32,
    )


def _load_state_bytes(processor: Any, payload: bytes, suffix: str = ".state") -> None:
    method = getattr(processor, "set_state", None)
    if method is not None:
        try:
            method(payload)
            return
        except TypeError:
            pass
    method = getattr(processor, "load_state", None)
    if method is None:
        raise AttributeError("Processor exposes neither set_state(bytes) nor load_state(path).")
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(payload)
        handle.flush()
        method(handle.name)


def _serum2_payloads(data: bytes) -> list[tuple[str, bytes]]:
    payloads: list[tuple[str, bytes]] = [("raw", data)]
    signatures = (
        (b"PK\x03\x04", 0),
        (b"\x28\xb5\x2f\xfd", 4),
        (b"\x78\x01", 2),
        (b"\x78\x9c", 2),
        (b"\x78\xda", 2),
    )
    for signature, offset in signatures:
        if data.startswith(signature) and offset:
            payloads.append((f"after-{signature.hex()}", data[offset:]))
    for marker in (b"{", b"<", b"CcnK", b"VST3"):
        offset = data.find(marker, 1, min(len(data), 4096))
        if offset > 0:
            payloads.append((f"from-{marker!r}@{offset}", data[offset:]))
    unique: list[tuple[str, bytes]] = []
    seen: set[bytes] = set()
    for label, payload in payloads:
        fingerprint = payload[:64] + len(payload).to_bytes(8, "big")
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append((label, payload))
    return unique


def inspect_preset_bytes(path: Path) -> dict[str, Any]:
    import struct

    data = path.read_bytes()
    findings: dict[str, Any] = {
        "path": str(path),
        "size": len(data),
        "first_32_hex": data[:32].hex(" "),
        "magic_ascii": "".join(chr(value) if 32 <= value < 127 else "." for value in data[:16]),
        "json_visible": b"{" in data[:4096],
        "xml_visible": b"<?xml" in data[:4096] or b"<" in data[:256],
        "zlib_header": data[:2] in (b"\x78\x01", b"\x78\x9c", b"\x78\xda"),
        "zstd_header": data.startswith(b"\x28\xb5\x2f\xfd"),
    }
    if data.startswith(b"XferJson\0") and len(data) >= 17:
        metadata_size = struct.unpack_from("<Q", data, 9)[0]
        payload_offset = 17 + metadata_size
        findings["container"] = "XferJson"
        findings["metadata_size"] = metadata_size
        findings["payload_offset"] = payload_offset
        findings["payload_first_32_hex"] = data[payload_offset : payload_offset + 32].hex(" ")
        findings["payload_has_zstd_after_8_bytes"] = data[
            payload_offset + 8 : payload_offset + 12
        ] == b"\x28\xb5\x2f\xfd"
    return findings


def _vst3_class_ids(bundle: Path, plugin_name: str) -> tuple[str | None, str | None]:
    """Return the component/controller CIDs for one named plug-in class."""
    module_infos = list(bundle.rglob("moduleinfo.json")) if bundle.is_dir() else []
    for path in module_infos:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        classes = data.get("Classes", []) if isinstance(data, dict) else []
        component_id: str | None = None
        controller_id: str | None = None
        for item in classes:
            if not isinstance(item, dict) or item.get("Name") != plugin_name:
                continue
            raw_id = item.get("CID")
            if not isinstance(raw_id, str):
                continue
            compact = raw_id.replace("-", "").strip()
            if len(compact) != 32:
                continue
            if item.get("Category") == "Audio Module Class":
                component_id = compact
            elif item.get("Category") == "Component Controller Class":
                controller_id = compact
        if component_id:
            return component_id, controller_id
    return None, None


def _vst3_class_id(bundle: Path, plugin_name: str = "Serum 2") -> str | None:
    """Compatibility wrapper returning the audio-module class, never an FX/controller CID."""
    return _vst3_class_ids(bundle, plugin_name)[0]


def serum2_component_state(preset_bytes: bytes) -> bytes:
    """Convert an XferJson SerumPreset container into VST3 IComponent state."""
    import struct

    if not preset_bytes.startswith(b"XferJson\0") or len(preset_bytes) < 17:
        raise ValueError("Serum 2 preset does not start with the XferJson container header.")
    (metadata_size,) = struct.unpack_from("<Q", preset_bytes, 9)
    metadata_end = 17 + metadata_size
    if metadata_end > len(preset_bytes):
        raise ValueError("Serum 2 preset metadata length extends past end of file.")
    metadata = json.loads(preset_bytes[17:metadata_end].decode("utf-8"))
    required = ("hash", "product", "productVersion", "url", "vendor", "version")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Serum 2 preset metadata is missing {missing}.")
    component_metadata = {"component": "processor"}
    component_metadata.update({key: metadata[key] for key in required})
    encoded = json.dumps(component_metadata, separators=(",", ":")).encode("utf-8")
    return b"XferJson\0" + struct.pack("<Q", len(encoded)) + encoded + preset_bytes[metadata_end:]


def build_vstpreset(
    component_state: bytes, class_id: str, *, controller_state: bytes | None = None
) -> bytes:
    """Build a Steinberg VST3 preset with validated Comp/optional Cont chunks."""
    import struct

    compact_id = class_id.replace("-", "").strip()
    if len(compact_id) != 32 or any(value not in "0123456789abcdefABCDEF" for value in compact_id):
        raise ValueError("VST3 class ID must be exactly 32 ASCII hexadecimal characters.")
    class_bytes = compact_id.encode("ascii")
    header_size = 48
    chunks: list[tuple[bytes, bytes]] = [(b"Comp", component_state)]
    if controller_state is not None:
        chunks.append((b"Cont", controller_state))
    data_area = b"".join(data for _, data in chunks)
    chunk_list_offset = header_size + len(data_area)
    header = b"VST3" + struct.pack("<I", 1) + class_bytes + struct.pack("<Q", chunk_list_offset)
    entries: list[bytes] = []
    offset = header_size
    for chunk_id, data in chunks:
        entries.append(chunk_id + struct.pack("<QQ", offset, len(data)))
        offset += len(data)
    chunk_list = b"List" + struct.pack("<I", len(entries)) + b"".join(entries)
    result = header + data_area + chunk_list
    inspect_vstpreset_bytes(result)
    return result


def inspect_vstpreset_bytes(data: bytes) -> dict[str, Any]:
    """Parse and bounds-check the public Steinberg .vstpreset layout."""
    import struct

    if len(data) < 56 or data[:4] != b"VST3":
        raise ValueError("Not a VST3 preset header.")
    version = struct.unpack_from("<I", data, 4)[0]
    class_id = data[8:40].decode("ascii")
    chunk_list_offset = struct.unpack_from("<Q", data, 40)[0]
    if chunk_list_offset + 8 > len(data) or data[chunk_list_offset : chunk_list_offset + 4] != b"List":
        raise ValueError("VST3 preset chunk-list offset is outside the file or lacks List magic.")
    entry_count = struct.unpack_from("<I", data, chunk_list_offset + 4)[0]
    expected_end = chunk_list_offset + 8 + entry_count * 20
    if expected_end != len(data):
        raise ValueError(f"VST3 chunk table ends at {expected_end}, file ends at {len(data)}.")
    entries: list[dict[str, Any]] = []
    for index in range(entry_count):
        base = chunk_list_offset + 8 + index * 20
        chunk_id = data[base : base + 4].decode("ascii")
        offset, size = struct.unpack_from("<QQ", data, base + 4)
        if offset < 48 or offset + size > chunk_list_offset:
            raise ValueError(f"Chunk {chunk_id} points outside the data area.")
        entries.append({"id": chunk_id, "offset": offset, "size": size})
    if not any(item["id"] == "Comp" for item in entries):
        raise ValueError("VST3 preset has no Comp component-state chunk.")
    return {
        "version": version,
        "class_id": class_id,
        "chunk_list_offset": chunk_list_offset,
        "entries": entries,
        "file_size": len(data),
    }


def _verify_daw_loaded(
    strategy: str,
    candidate: PluginCandidate,
    engine: Any,
    processor: Any,
    initial: list[ParameterValue],
    attempts: list[StrategyAttempt],
) -> LoadedPreset:
    loaded = dump_dawdreamer_parameters(processor)
    changed = changed_parameter_count(initial, loaded)
    audio = render_dawdreamer_note(engine, processor)
    peak, rms = audio_levels(audio)
    if changed < 5:
        raise RuntimeError(f"only {changed} parameters changed from init; at least 5 required")
    if rms <= SILENCE_DBFS:
        raise RuntimeError(f"C4 render is silent ({rms:.2f} dBFS; must exceed -60 dBFS)")
    return LoadedPreset(
        strategy=strategy,
        plugin_format=candidate.format,
        plugin_path=candidate.path,
        host="dawdreamer",
        parameters=loaded,
        changed_from_init=changed,
        rms_dbfs=rms,
        peak_dbfs=peak,
        processor=processor,
        engine=engine,
        attempts=attempts,
    )


def _try_dawdreamer(
    candidate: PluginCandidate,
    preset_path: Path,
    synth: SynthVersion,
    attempts: list[StrategyAttempt],
) -> LoadedPreset | None:
    strategies: list[tuple[str, Any]] = []
    data = preset_path.read_bytes()
    if synth == "serum1" and candidate.format == "VST2":
        strategies.append(("S1-dawdreamer-vst2-fxp", lambda p: p.load_preset(str(preset_path))))
    elif synth == "serum1" and candidate.format == "VST3":
        try:
            payload = parse_fxp(preset_path).payload
            strategies.append(("S2-dawdreamer-vst3-fxp-state", lambda p: _load_state_bytes(p, payload)))
        except FxpError as exc:
            attempts.append(StrategyAttempt("S2-fxp-parse", str(candidate.path), False, str(exc)))
    elif synth == "serum2" and candidate.format == "VST3":
        class_id, _controller_id = _vst3_class_ids(candidate.path, "Serum 2")
        if class_id:
            component_state = serum2_component_state(data)
            wrapped_variants = (
                ("S3-dawdreamer-vstpreset-component", build_vstpreset(component_state, class_id)),
                ("S3-dawdreamer-vstpreset-raw-file", build_vstpreset(data, class_id)),
            )

            for strategy_name, wrapped in wrapped_variants:
                def load_wrapped(processor: Any, value: bytes = wrapped) -> None:
                    method = getattr(processor, "load_vst3_preset", None)
                    if method is None:
                        raise AttributeError("load_vst3_preset is unavailable")
                    with tempfile.NamedTemporaryFile(suffix=".vstpreset") as handle:
                        handle.write(value)
                        handle.flush()
                        if method(handle.name) is False:
                            raise RuntimeError("load_vst3_preset returned False")

                strategies.append((strategy_name, load_wrapped))
        for label, payload in _serum2_payloads(data):
            strategies.append(
                (f"S3-dawdreamer-vst3-state-{label}", lambda p, value=payload: _load_state_bytes(p, value))
            )
    elif candidate.format == "AU":
        strategies.append(("S5-dawdreamer-au-direct-preset", lambda p: p.load_preset(str(preset_path))))
        strategies.append(("S5-dawdreamer-au-state", lambda p: _load_state_bytes(p, data)))

    for strategy, loader in strategies:
        try:
            print(f"PATCHLAB_STRATEGY_ATTEMPT={strategy}@{candidate.path}", flush=True)
            engine, processor = make_dawdreamer_processor(candidate)
            initial = dump_dawdreamer_parameters(processor)
            loader(processor)
            loaded = _verify_daw_loaded(strategy, candidate, engine, processor, initial, attempts)
            attempts.append(StrategyAttempt(strategy, str(candidate.path), True, "verified"))
            loaded.attempts = list(attempts)
            return loaded
        except Exception as exc:  # plugin APIs intentionally vary by build
            attempts.append(StrategyAttempt(strategy, str(candidate.path), False, repr(exc)))
    return None


def _try_pedalboard(
    candidate: PluginCandidate,
    preset_path: Path,
    synth: SynthVersion,
    attempts: list[StrategyAttempt],
) -> LoadedPreset | None:
    if candidate.format not in {"VST3", "AU"}:
        return None
    try:
        from pedalboard import load_plugin
    except ImportError as exc:
        attempts.append(StrategyAttempt("S4-pedalboard-import", str(candidate.path), False, repr(exc)))
        return None

    data = preset_path.read_bytes()
    payloads: list[tuple[str, bytes]]
    if candidate.format == "AU":
        payloads = [("classinfo", data)]
    elif synth == "serum1":
        try:
            payloads = [("fxp-state", parse_fxp(preset_path).payload)]
        except FxpError as exc:
            attempts.append(StrategyAttempt("S4-pedalboard-fxp-parse", str(candidate.path), False, str(exc)))
            return None
    else:
        payloads = _serum2_payloads(data)

    for label, payload in payloads:
        strategy = f"S4-pedalboard-{candidate.format.lower()}-{label}"
        try:
            print(f"PATCHLAB_STRATEGY_ATTEMPT={strategy}@{candidate.path}", flush=True)
            preferred_name = "Serum" if synth == "serum1" else "Serum 2"
            try:
                processor = load_plugin(str(candidate.path), plugin_name=preferred_name)
            except (ImportError, RuntimeError, ValueError):
                processor = load_plugin(str(candidate.path))
            initial = dump_pedalboard_parameters(processor)
            if candidate.format == "AU":
                import plistlib

                class_info = plistlib.loads(processor.raw_state)
                if synth == "serum1":
                    if "vstdata" not in class_info:
                        raise RuntimeError("Serum 1 AU ClassInfo has no vstdata field")
                    class_info["vstdata"] = payload
                    class_info["name"] = preset_path.stem
                else:
                    if "Processor State" not in class_info:
                        raise RuntimeError("Serum 2 AU ClassInfo has no Processor State field")
                    class_info["Processor State"] = serum2_component_state(payload)
                    class_info["name"] = preset_path.stem
                processor.raw_state = plistlib.dumps(
                    class_info, fmt=plistlib.FMT_BINARY, sort_keys=False
                )
            else:
                processor.raw_state = payload
            loaded_params = dump_pedalboard_parameters(processor)
            changed = changed_parameter_count(initial, loaded_params)
            audio = render_pedalboard_note(processor)
            peak, rms = audio_levels(audio)
            if changed < 5:
                raise RuntimeError(f"only {changed} parameters changed from init; at least 5 required")
            if rms <= SILENCE_DBFS:
                raise RuntimeError(f"C4 render is silent ({rms:.2f} dBFS)")
            attempts.append(StrategyAttempt(strategy, str(candidate.path), True, "verified"))
            return LoadedPreset(
                strategy=strategy,
                plugin_format=candidate.format,
                plugin_path=candidate.path,
                host="pedalboard",
                parameters=loaded_params,
                changed_from_init=changed,
                rms_dbfs=rms,
                peak_dbfs=peak,
                processor=processor,
                attempts=list(attempts),
            )
        except Exception as exc:
            attempts.append(StrategyAttempt(strategy, str(candidate.path), False, repr(exc)))
    return None


def load_preset(env: PlatformEnv, synth: SynthVersion, preset_path: Path) -> LoadedPreset:
    """Try S1-S5 and return only a preset that passes per-file verification."""
    attempts: list[StrategyAttempt] = []
    candidates = env.plugins_for(synth)
    if not candidates:
        raise PresetLoadError(f"No hostable {synth} plugin binary was found.")

    # S1-S3: DawDreamer with VST formats. S4: Pedalboard fallback.
    vst_candidates = tuple(item for item in candidates if item.format in {"VST2", "VST3"})
    for candidate in vst_candidates:
        loaded = _try_dawdreamer(candidate, preset_path, synth, attempts)
        if loaded is not None:
            return loaded
    for candidate in vst_candidates:
        loaded = _try_pedalboard(candidate, preset_path, synth, attempts)
        if loaded is not None:
            return loaded

    # S5: AU after VST state paths have been exhausted.
    for candidate in (item for item in candidates if item.format == "AU"):
        loaded = _try_dawdreamer(candidate, preset_path, synth, attempts)
        if loaded is not None:
            return loaded
        loaded = _try_pedalboard(candidate, preset_path, synth, attempts)
        if loaded is not None:
            return loaded
    raise PresetLoadError(f"No verified preset strategy succeeded for {preset_path}.", attempts)


def verify_default_render(candidate: PluginCandidate) -> tuple[float, float, list[str]]:
    """Load a plugin, report its API surface, and render init-state C4."""
    engine, processor = make_dawdreamer_processor(candidate)
    relevant_methods = sorted(
        name
        for name in dir(processor)
        if any(token in name.lower() for token in ("state", "preset", "parameter"))
    )
    audio = render_dawdreamer_note(engine, processor)
    peak, rms = audio_levels(audio)
    return peak, rms, relevant_methods
