"""Reconstruct Serum 2 VST3 processor/controller state from a merged preset graph."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cbor2
import zstandard

from core.plugin_host import build_vstpreset, inspect_vstpreset_bytes
from core.serum2_preset import Serum2Preset


XFER_MAGIC = b"XferJson\0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENDER_STATE_DIR = PROJECT_ROOT / "data" / "models" / "serum2_render_states"


def render_state_path(preset_id: int, state_dir: Path = DEFAULT_RENDER_STATE_DIR) -> Path:
    """Return the id-only reconstructed state path used by Serum 2 render workers."""

    if preset_id <= 0:
        raise ValueError("preset_id must be positive")
    return Path(state_dir) / f"{preset_id}.vstpreset"


def load_render_state(processor: Any, preset_id: int, state_dir: Path = DEFAULT_RENDER_STATE_DIR) -> Path:
    """Load an audio-verified reconstructed Serum 2 state into a worker-local plugin."""

    path = render_state_path(preset_id, state_dir).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    method = getattr(processor, "load_vst3_preset", None)
    if method is None:
        raise AttributeError("Serum 2 processor does not expose load_vst3_preset(filepath)")
    if method(str(path)) is False:
        raise RuntimeError(f"Serum 2 render state was rejected: {path}")
    return path


@dataclass(frozen=True, slots=True)
class XferState:
    metadata: dict[str, Any]
    version_marker: int
    data: Any


@dataclass(frozen=True, slots=True)
class HostStateTemplate:
    class_id: str
    component: XferState
    controller: XferState


@dataclass(frozen=True, slots=True)
class PartitionResult:
    component: Any
    controller: Any
    matched_component_paths: tuple[str, ...]
    matched_controller_paths: tuple[str, ...]
    matched_both_paths: tuple[str, ...]
    unmatched_paths: tuple[str, ...]
    total_leaves: int
    matched_leaves: int

    @property
    def coverage(self) -> float:
        return self.matched_leaves / self.total_leaves if self.total_leaves else 0.0


def decode_xfer_state(blob: bytes) -> XferState:
    if not blob.startswith(XFER_MAGIC) or len(blob) < 25:
        raise ValueError("State chunk is not an XferJson container")
    metadata_size = struct.unpack_from("<Q", blob, 9)[0]
    metadata_end = 17 + metadata_size
    metadata = json.loads(blob[17:metadata_end].decode("utf-8"))
    raw_size, version = struct.unpack_from("<II", blob, metadata_end)
    raw = zstandard.ZstdDecompressor().decompress(blob[metadata_end + 8 :])
    if len(raw) != raw_size:
        raise ValueError(f"CBOR length mismatch: header {raw_size}, decoded {len(raw)}")
    return XferState(metadata=metadata, version_marker=version, data=cbor2.loads(raw))


def encode_xfer_state(state: XferState) -> bytes:
    raw = cbor2.dumps(state.data)
    compressed = zstandard.ZstdCompressor().compress(raw)
    metadata = dict(state.metadata)
    metadata["hash"] = hashlib.md5(compressed).hexdigest()
    encoded_metadata = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return (
        XFER_MAGIC
        + struct.pack("<Q", len(encoded_metadata))
        + encoded_metadata
        + struct.pack("<II", len(raw), state.version_marker)
        + compressed
    )


def decode_host_template(vstpreset: bytes) -> HostStateTemplate:
    info = inspect_vstpreset_bytes(vstpreset)
    chunks: dict[str, bytes] = {}
    for entry in info["entries"]:
        chunks[entry["id"]] = vstpreset[entry["offset"] : entry["offset"] + entry["size"]]
    if "Comp" not in chunks or "Cont" not in chunks:
        raise ValueError("Live Serum 2 preset_data must contain both Comp and Cont chunks")
    return HostStateTemplate(
        class_id=str(info["class_id"]),
        component=decode_xfer_state(chunks["Comp"]),
        controller=decode_xfer_state(chunks["Cont"]),
    )


def _kind(value: Any) -> type[Any]:
    return type(value)


def _leaf_paths(value: Any, path: str) -> list[str]:
    if isinstance(value, dict):
        if not value:
            return [path]
        result: list[str] = []
        for key, child in value.items():
            result.extend(_leaf_paths(child, f"{path}.{key}" if path else str(key)))
        return result
    if isinstance(value, list):
        if not value:
            return [path]
        result = []
        for index, child in enumerate(value):
            result.extend(_leaf_paths(child, f"{path}.[{index}]"))
        return result
    return [path]


def structural_paths(value: Any, path: str = "") -> set[str]:
    result = {path or "$"}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            result.update(structural_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(structural_paths(child, f"{path}.[{index}]"))
    return result


def _assign_subtree(target: Any, key: Any, value: Any) -> None:
    target[key] = copy.deepcopy(value)


def partition_merged_graph(
    merged: Any, component_template: Any, controller_template: Any
) -> PartitionResult:
    """Partition by exact structural membership, recursing only through overlaps."""

    component = copy.deepcopy(component_template)
    controller = copy.deepcopy(controller_template)
    comp_paths: list[str] = []
    cont_paths: list[str] = []
    both_paths: list[str] = []
    unmatched: list[str] = []

    def record(paths: list[str], value: Any, path: str) -> None:
        paths.extend(_leaf_paths(value, path))

    def walk(source: Any, comp: Any, cont: Any, path: str) -> None:
        if isinstance(source, dict):
            if not isinstance(comp, dict) and not isinstance(cont, dict):
                record(unmatched, source, path)
                return
            for key, value in source.items():
                child_path = f"{path}.{key}" if path else str(key)
                in_comp = isinstance(comp, dict) and key in comp
                in_cont = isinstance(cont, dict) and key in cont
                if in_comp and not in_cont:
                    _assign_subtree(comp, key, value)
                    record(comp_paths, value, child_path)
                elif in_cont and not in_comp:
                    _assign_subtree(cont, key, value)
                    record(cont_paths, value, child_path)
                elif not in_comp and not in_cont:
                    record(unmatched, value, child_path)
                else:
                    comp_value, cont_value = comp[key], cont[key]
                    source_kind = _kind(value)
                    comp_match = _kind(comp_value) is source_kind
                    cont_match = _kind(cont_value) is source_kind
                    if comp_match and not cont_match:
                        _assign_subtree(comp, key, value)
                        record(comp_paths, value, child_path)
                    elif cont_match and not comp_match:
                        _assign_subtree(cont, key, value)
                        record(cont_paths, value, child_path)
                    elif isinstance(value, (dict, list)):
                        walk(value, comp_value, cont_value, child_path)
                    else:
                        _assign_subtree(comp, key, value)
                        _assign_subtree(cont, key, value)
                        record(both_paths, value, child_path)
            return
        if isinstance(source, list):
            if not isinstance(comp, list) and not isinstance(cont, list):
                record(unmatched, source, path)
                return
            for index, value in enumerate(source):
                child_path = f"{path}.[{index}]"
                in_comp = isinstance(comp, list) and index < len(comp)
                in_cont = isinstance(cont, list) and index < len(cont)
                if in_comp and not in_cont:
                    comp[index] = copy.deepcopy(value)
                    record(comp_paths, value, child_path)
                elif in_cont and not in_comp:
                    cont[index] = copy.deepcopy(value)
                    record(cont_paths, value, child_path)
                elif not in_comp and not in_cont:
                    record(unmatched, value, child_path)
                else:
                    comp_value, cont_value = comp[index], cont[index]
                    source_kind = _kind(value)
                    comp_match = _kind(comp_value) is source_kind
                    cont_match = _kind(cont_value) is source_kind
                    if comp_match and not cont_match:
                        comp[index] = copy.deepcopy(value)
                        record(comp_paths, value, child_path)
                    elif cont_match and not comp_match:
                        cont[index] = copy.deepcopy(value)
                        record(cont_paths, value, child_path)
                    elif isinstance(value, (dict, list)):
                        walk(value, comp_value, cont_value, child_path)
                    else:
                        comp[index] = copy.deepcopy(value)
                        cont[index] = copy.deepcopy(value)
                        record(both_paths, value, child_path)
            return
        record(unmatched, source, path)

    walk(merged, component, controller, "")
    total_paths = _leaf_paths(merged, "")
    matched = len(comp_paths) + len(cont_paths) + len(both_paths)
    return PartitionResult(
        component=component,
        controller=controller,
        matched_component_paths=tuple(comp_paths),
        matched_controller_paths=tuple(cont_paths),
        matched_both_paths=tuple(both_paths),
        unmatched_paths=tuple(unmatched),
        total_leaves=len(total_paths),
        matched_leaves=matched,
    )


def reconstruct_vstpreset(
    preset: Serum2Preset, template: HostStateTemplate
) -> tuple[bytes, PartitionResult]:
    partition = partition_merged_graph(
        preset.data, template.component.data, template.controller.data
    )
    component = XferState(
        metadata=template.component.metadata,
        version_marker=template.component.version_marker,
        data=partition.component,
    )
    controller_metadata = dict(template.controller.metadata)
    for source_key, target_key in (
        ("presetAuthor", "presetAuthor"),
        ("presetDescription", "presetDescription"),
        ("presetName", "presetName"),
    ):
        if source_key in preset.metadata:
            controller_metadata[target_key] = preset.metadata[source_key]
    controller = XferState(
        metadata=controller_metadata,
        version_marker=template.controller.version_marker,
        data=partition.controller,
    )
    return (
        build_vstpreset(
            encode_xfer_state(component),
            template.class_id,
            controller_state=encode_xfer_state(controller),
        ),
        partition,
    )


def reconstruct_partial_vstpreset(
    preset: Serum2Preset, template: HostStateTemplate, *, merge_matching_lists: bool = False
) -> tuple[bytes, PartitionResult]:
    """Overlay a sparse predicted recipe without replacing required state subtrees.

    ``reconstruct_vstpreset`` consumes the complete decoded graph from a real
    SerumPreset.  Model inference is deliberately sparse: it contains learned
    setting leaves but not Serum's editor/runtime bookkeeping.  Replacing a
    complete component with that sparse subtree can make an otherwise valid
    VST3 state unloadable.  This variant recursively overlays compatible leaves
    onto the live plug-in state and leaves omitted structure at its valid init
    value.
    """

    component = copy.deepcopy(template.component.data)
    controller = copy.deepcopy(template.controller.data)
    component_paths: list[str] = []
    controller_paths: list[str] = []
    both_paths: list[str] = []
    unmatched_paths: list[str] = []

    def compatible_scalar(source: Any, target: Any) -> bool:
        return type(source) is type(target) or (
            isinstance(source, (int, float))
            and not isinstance(source, bool)
            and isinstance(target, (int, float))
            and not isinstance(target, bool)
        )

    def overlay(source: Any, target: Any, path: str, matched: list[str]) -> None:
        if not isinstance(source, dict) or not isinstance(target, dict):
            unmatched_paths.extend(_leaf_paths(source, path))
            return
        for key, value in source.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key not in target:
                unmatched_paths.extend(_leaf_paths(value, child_path))
                continue
            current = target[key]
            if isinstance(value, dict) and isinstance(current, dict):
                overlay(value, current, child_path, matched)
            elif isinstance(value, list) and isinstance(current, list):
                if merge_matching_lists:
                    for index, item in enumerate(value):
                        item_path = f"{child_path}.[{index}]"
                        if index >= len(current):
                            unmatched_paths.extend(_leaf_paths(item, item_path))
                        elif isinstance(item, dict) and isinstance(current[index], dict):
                            overlay(item, current[index], item_path, matched)
                        elif compatible_scalar(item, current[index]):
                            current[index] = copy.deepcopy(item)
                            matched.append(item_path)
                        else:
                            unmatched_paths.extend(_leaf_paths(item, item_path))
                else:
                    # Variable arrays encode topology/content, not an
                    # individual setting leaf. Keep the template topology for
                    # unconstrained model predictions.
                    unmatched_paths.extend(_leaf_paths(value, child_path))
            elif isinstance(value, dict) and current == "default":
                target[key] = copy.deepcopy(value)
                matched.extend(_leaf_paths(value, child_path))
            elif compatible_scalar(value, current):
                target[key] = copy.deepcopy(value)
                matched.append(child_path)
            else:
                unmatched_paths.extend(_leaf_paths(value, child_path))

    if not isinstance(preset.data, dict):
        raise TypeError("Predicted Serum 2 settings must be a mapping")
    for key, value in preset.data.items():
        in_component = isinstance(component, dict) and key in component
        in_controller = isinstance(controller, dict) and key in controller
        if in_component:
            overlay({key: value}, component, "", component_paths)
        if in_controller:
            overlay({key: value}, controller, "", controller_paths)
        if in_component and in_controller:
            shared = set(component_paths).intersection(controller_paths)
            if shared:
                both_paths.extend(sorted(shared))
        if not in_component and not in_controller:
            unmatched_paths.extend(_leaf_paths(value, key))

    total_paths = _leaf_paths(preset.data, "")
    # A path present in both chunks represents one matched source leaf.
    matched_unique = set(component_paths).union(controller_paths)
    partition = PartitionResult(
        component=component,
        controller=controller,
        matched_component_paths=tuple(component_paths),
        matched_controller_paths=tuple(controller_paths),
        matched_both_paths=tuple(sorted(set(both_paths))),
        unmatched_paths=tuple(unmatched_paths),
        total_leaves=len(total_paths),
        matched_leaves=len(matched_unique),
    )
    component_state = XferState(
        metadata=template.component.metadata,
        version_marker=template.component.version_marker,
        data=component,
    )
    controller_metadata = dict(template.controller.metadata)
    controller_metadata.update(
        {
            target_key: preset.metadata[source_key]
            for source_key, target_key in (
                ("presetAuthor", "presetAuthor"),
                ("presetDescription", "presetDescription"),
                ("presetName", "presetName"),
            )
            if source_key in preset.metadata
        }
    )
    controller_state = XferState(
        metadata=controller_metadata,
        version_marker=template.controller.version_marker,
        data=controller,
    )
    return (
        build_vstpreset(
            encode_xfer_state(component_state),
            template.class_id,
            controller_state=encode_xfer_state(controller_state),
        ),
        partition,
    )
