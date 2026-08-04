"""Staged Serum 2 structural proposals for the main matcher."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SEARCH_ORDER = ("noise_sample", "wavetable", "fx_type", "mod_route")


@dataclass(frozen=True, slots=True)
class StructuralProposal:
    category: str
    stable_id: str
    overrides: dict[str, Any]
    provenance: tuple[str, ...]
    priority: int


def load_vocabulary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_structural_fields(graph: Mapping[str, Any]) -> dict[str, list[str]]:
    """Enumerate every searchable structural field in one live base state."""

    found = {category: [] for category in SEARCH_ORDER}
    for oscillator in range(3):
        path = f"Oscillator{oscillator}.WTOsc{oscillator}.relativePathToWT"
        wt = graph.get(f"Oscillator{oscillator}", {}).get(f"WTOsc{oscillator}", {})
        if isinstance(wt, Mapping) and "relativePathToWT" in wt:
            found["wavetable"].append(path)
    noise = graph.get("Oscillator3", {}).get("NoiseOsc3", {})
    if isinstance(noise, Mapping) and "relativePathToNoiseSample" in noise:
        found["noise_sample"].append("Oscillator3.NoiseOsc3.relativePathToNoiseSample")
    for rack_index in range(3):
        effects = graph.get(f"FXRack{rack_index}", {}).get("FX", [])
        if isinstance(effects, list):
            for effect_index, effect in enumerate(effects):
                if isinstance(effect, Mapping) and "type" in effect:
                    found["fx_type"].append(f"FXRack{rack_index}.FX.{effect_index}.type")
    for slot_index in range(64):
        slot = graph.get(f"ModSlot{slot_index}", {})
        if isinstance(slot, Mapping) and "source" in slot:
            found["mod_route"].append(f"ModSlot{slot_index}")
    return found


def _overrides(category: str, value: Any, field_path: str) -> dict[str, Any]:
    if category == "noise_sample":
        return {field_path: value}
    if category == "wavetable":
        return {field_path: value}
    if category == "fx_type":
        return {field_path: value}
    if category == "mod_route":
        route = value
        destination = route["destination"]
        result = {f"{field_path}.source": route["source"]}
        for key in (
            "destModuleID",
            "destModuleParamID",
            "destModuleParamName",
            "destModuleTypeString",
        ):
            result[f"{field_path}.{key}"] = destination[key]
        return result
    raise KeyError(category)


def staged_proposals(
    vocabulary: Mapping[str, Any],
    *,
    top_k: int = 2,
    fields: Mapping[str, list[str]] | None = None,
    ranked_ids: Mapping[str, list[str]] | None = None,
    enabled_categories: set[str] | frozenset[str] | None = None,
) -> dict[str, list[StructuralProposal]]:
    """Return measured-prior proposals while preserving full API reachability."""

    categories = vocabulary.get("categories", {})
    result: dict[str, list[StructuralProposal]] = {}
    for category in SEARCH_ORDER:
        if enabled_categories is not None and category not in enabled_categories:
            result[category] = []
            continue
        entries = list(categories.get(category, {}).get("entries", []))
        rank = {
            identifier: index
            for index, identifier in enumerate((ranked_ids or {}).get(category, ()))
        }
        entries.sort(
            key=lambda item: (
                rank.get(str(item.get("id", "")), len(rank)),
                -int(item.get("observed_count", 0)),
                str(item.get("id", "")),
            )
        )
        proposals: list[StructuralProposal] = []
        category_fields = list((fields or {}).get(category, [])) or [
            {
                "noise_sample": "Oscillator3.NoiseOsc3.relativePathToNoiseSample",
                "wavetable": "Oscillator0.WTOsc0.relativePathToWT",
                "fx_type": "FXRack0.FX.0.type",
                "mod_route": "ModSlot0",
            }[category]
        ]
        priority = 0
        for field_path in category_fields:
            for entry in entries[:top_k]:
                value = entry.get("value")
                if value is None:
                    continue
                proposals.append(
                    StructuralProposal(
                        category,
                        str(entry["id"]),
                        _overrides(category, value, field_path),
                        tuple(entry.get("provenance", [])),
                        priority,
                    )
                )
                priority += 1
        result[category] = proposals
    return result
