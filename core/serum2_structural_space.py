"""Serum 2 structural choices that are not exposed to host automation.

The vocabulary is deliberately evidence based.  Values seen in presets and
files found in an installed Serum 2 library are recorded with separate
provenance.  Callers may still supply an arbitrary structural override: the
observed vocabulary is a search prior, not an allow-list.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.serum2_preset import Serum2Preset


AUDIO_EXTENSIONS = frozenset({".aif", ".aiff", ".flac", ".wav"})
STRUCTURAL_CATEGORIES = (
    "wavetable",
    "embedded_wavetable",
    "noise_sample",
    "mod_source",
    "mod_destination",
    "mod_route",
    "fx_type",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def stable_id(category: str, value: Any) -> str:
    """Return a content-addressed ID stable across machines and scan order."""

    digest = hashlib.sha256(_canonical(value)).hexdigest()[:20]
    return f"s2-{category}-{digest}"


@dataclass(slots=True)
class StructuralEntry:
    id: str
    value: Any
    provenance: set[str] = field(default_factory=set)
    observed_count: int = 0

    def to_json(self, *, include_value: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "provenance": sorted(self.provenance),
            "observed_count": self.observed_count,
        }
        if include_value:
            payload["value"] = self.value
        return payload


class StructuralSpace:
    """Mutable builder and immutable-style lookup for structural choices."""

    def __init__(self) -> None:
        self.categories: dict[str, dict[str, StructuralEntry]] = {
            category: {} for category in STRUCTURAL_CATEGORIES
        }
        self.preset_count = 0
        self.install_roots: list[str] = []

    def add(
        self,
        category: str,
        value: Any,
        provenance: str,
        *,
        observed_count: int = 1,
    ) -> StructuralEntry:
        if category not in self.categories:
            raise KeyError(category)
        identifier = stable_id(category, value)
        entry = self.categories[category].get(identifier)
        if entry is None:
            entry = StructuralEntry(identifier, value)
            self.categories[category][identifier] = entry
        entry.provenance.add(provenance)
        if provenance == "observed_in_presets":
            entry.observed_count += observed_count
        return entry

    def entries(self, category: str) -> tuple[StructuralEntry, ...]:
        return tuple(sorted(self.categories[category].values(), key=lambda item: item.id))

    def summary(self) -> dict[str, int]:
        return {name: len(values) for name, values in self.categories.items()}

    def to_json(self) -> dict[str, Any]:
        categories: dict[str, Any] = {}
        for name in STRUCTURAL_CATEGORIES:
            # Embedded payloads can contain hundreds of thousands of samples.
            # Their hashes establish stable identities; arbitrary payloads stay
            # reachable through overrides without copying licensed preset data.
            include_value = name != "embedded_wavetable"
            categories[name] = {
                "count": len(self.categories[name]),
                "completeness": (
                    "open_payload_domain; observed hashes are a lower bound"
                    if name == "embedded_wavetable"
                    else "observed/enumerated lower bound; arbitrary overrides remain reachable"
                ),
                "entries": [
                    item.to_json(include_value=include_value) for item in self.entries(name)
                ],
            }
        return {
            "schema_version": 1,
            "definition_of_complete": "reachability_not_enumeration",
            "preset_count": self.preset_count,
            "install_roots": sorted(set(self.install_roots)),
            "categories": categories,
        }


def _walk_graph(node: Any) -> Iterable[tuple[tuple[str, ...], Mapping[str, Any]]]:
    if isinstance(node, Mapping):
        yield (), node
        for key, value in node.items():
            for suffix, child in _walk_graph(value):
                yield (str(key),) + suffix, child
    elif isinstance(node, list):
        for index, value in enumerate(node):
            for suffix, child in _walk_graph(value):
                yield (str(index),) + suffix, child


def harvest_preset(space: StructuralSpace, preset: Serum2Preset) -> None:
    """Harvest all five structural categories from one parsed preset."""

    space.preset_count += 1
    for path, mapping in _walk_graph(preset.data):
        for key, category in (
            ("relativePathToWT", "wavetable"),
            ("relativePathToNoiseSample", "noise_sample"),
        ):
            value = mapping.get(key)
            if isinstance(value, str) and value:
                space.add(category, value.replace("\\", "/"), "observed_in_presets")
        if "embeddedWTData" in mapping:
            payload = mapping["embeddedWTData"]
            entry = space.add("embedded_wavetable", payload, "observed_in_presets")
            # Retain only the digest after registering it.  This prevents a
            # vocabulary report from becoming a preset-content archive.
            entry.value = {"sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
        if path and path[-1].startswith("ModSlot") and "source" in mapping:
            source = mapping.get("source")
            if isinstance(source, list) and len(source) == 2:
                space.add("mod_source", source, "observed_in_presets")
            destination = {
                key: mapping.get(key)
                for key in (
                    "destModuleID",
                    "destModuleParamID",
                    "destModuleParamName",
                    "destModuleTypeString",
                )
            }
            if destination["destModuleParamName"] is not None:
                # The named destination is the semantic 139-value vocabulary.
                space.add(
                    "mod_destination",
                    destination["destModuleParamName"],
                    "observed_in_presets",
                )
                space.add(
                    "mod_route",
                    {"source": source, "destination": destination},
                    "observed_in_presets",
                )
        if "type" in mapping and any(part.startswith("FXRack") for part in path):
            value = mapping.get("type")
            if isinstance(value, int):
                space.add("fx_type", value, "observed_in_presets")


def _add_installed_files(space: StructuralSpace, root: Path) -> None:
    tables = root / "Tables"
    if tables.is_dir():
        for path in sorted(tables.rglob("*")):
            if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS:
                space.add(
                    "wavetable",
                    path.relative_to(tables).as_posix(),
                    "enumerated_from_install",
                    observed_count=0,
                )
    noise_roots = (
        root / "Samples" / "Factory Non-Tonal" / "Noises",
        root / "Noises",
    )
    for noises in noise_roots:
        if not noises.is_dir():
            continue
        for path in sorted(noises.rglob("*")):
            if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS:
                space.add(
                    "noise_sample",
                    path.relative_to(noises).as_posix(),
                    "enumerated_from_install",
                    observed_count=0,
                )


def build_structural_space(
    presets: Iterable[Serum2Preset], install_roots: Iterable[Path] = ()
) -> StructuralSpace:
    space = StructuralSpace()
    for preset in presets:
        harvest_preset(space, preset)
    for root in install_roots:
        resolved = Path(root).expanduser().resolve()
        if resolved.is_dir():
            space.install_roots.append(str(resolved))
            _add_installed_files(space, resolved)
    return space


def apply_structural_overrides(graph: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Apply dotted/list-index paths without imposing a vocabulary allow-list.

    This is the checkable reachability guarantee: every existing Serum 2
    structural leaf, including a never-before-seen custom WT payload, can be
    supplied by a caller and passed to reconstruction.
    """

    for dotted_path, value in overrides.items():
        parts = dotted_path.split(".")
        cursor: Any = graph
        for part in parts[:-1]:
            cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
        final = parts[-1]
        if isinstance(cursor, list):
            cursor[int(final)] = value
        else:
            cursor[final] = value


def apply_existing_structural_overrides(
    graphs: Iterable[dict[str, Any]], overrides: Mapping[str, Any]
) -> dict[str, str]:
    """Apply each override to every state graph where its path exists.

    Serum duplicates some state between the VST3 component and controller.
    Updating both copies when present keeps reconstructed state coherent.
    """

    applied: dict[str, str] = {}
    for dotted_path, value in overrides.items():
        parts = dotted_path.split(".")
        hits = 0
        for graph in graphs:
            cursor: Any = graph
            try:
                for part in parts[:-1]:
                    cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
                final = parts[-1]
                if isinstance(cursor, list):
                    cursor[int(final)] = value
                elif final in cursor:
                    cursor[final] = value
                else:
                    continue
                hits += 1
            except (KeyError, IndexError, TypeError, ValueError):
                continue
        if hits:
            applied[dotted_path] = f"{hits} state graph(s)"
        else:
            raise KeyError(f"Structural path does not exist in the base state: {dotted_path}")
    return applied
