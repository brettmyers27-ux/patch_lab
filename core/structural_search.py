"""Staged Serum 2 structural proposals for the main matcher."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import librosa
import numpy as np


SEARCH_ORDER = ("wavetable", "fx_type", "noise_sample", "mod_route")
CLEAN_STAGE3C_CATEGORIES = frozenset({"wavetable", "fx_type"})
PERIODIC_ROUTE_LIMIT = 300


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


def load_search_policy(path: Path) -> dict[str, Any]:
    """Load a private repaired-candidate policy, if this machine has one."""

    if not path.is_file():
        return {
            "enabled_categories": sorted(CLEAN_STAGE3C_CATEGORIES),
            "allowed_ids": {},
            "source": "Stage 3C clean-category fallback",
        }
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
    top_k: int | None = 2,
    fields: Mapping[str, list[str]] | None = None,
    ranked_ids: Mapping[str, list[str]] | None = None,
    enabled_categories: set[str] | frozenset[str] | None = None,
    allowed_ids: Mapping[str, set[str] | frozenset[str]] | None = None,
) -> dict[str, list[StructuralProposal]]:
    """Return measured-prior proposals while preserving full API reachability."""

    categories = vocabulary.get("categories", {})
    result: dict[str, list[StructuralProposal]] = {}
    for category in SEARCH_ORDER:
        if enabled_categories is not None and category not in enabled_categories:
            result[category] = []
            continue
        entries = list(categories.get(category, {}).get("entries", []))
        permitted = (allowed_ids or {}).get(category)
        if permitted is not None:
            entries = [
                entry for entry in entries if str(entry.get("id", "")) in permitted
            ]
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
            for entry in (entries if top_k is None else entries[:top_k]):
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


def measure_periodic_movement(
    audio: np.ndarray, sample_rate: int
) -> dict[str, float | bool]:
    """Measure periodic amplitude, brightness, and pitch movement in a target."""

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(values) < 2048:
        return {
            "amplitude": False,
            "brightness": False,
            "pitch": False,
            "amplitude_strength": 0.0,
            "brightness_strength": 0.0,
            "pitch_strength": 0.0,
        }
    hop = 256
    spectrum = np.abs(librosa.stft(values, n_fft=1024, hop_length=hop))
    trajectories = {
        "amplitude": librosa.feature.rms(
            S=spectrum, frame_length=1024, hop_length=hop
        )[0],
        "brightness": librosa.feature.spectral_centroid(
            S=spectrum, sr=sample_rate
        )[0],
    }
    pitches, magnitudes = librosa.piptrack(
        S=spectrum, sr=sample_rate, fmin=30.0, fmax=min(5000.0, sample_rate / 2)
    )
    pitch_track = np.zeros(pitches.shape[1], dtype=np.float32)
    for frame in range(pitches.shape[1]):
        column = magnitudes[:, frame]
        if np.any(column > 0):
            pitch_track[frame] = pitches[int(np.argmax(column)), frame]
    trajectories["pitch"] = pitch_track

    result: dict[str, float | bool] = {}
    frame_rate = sample_rate / hop
    for name, trajectory in trajectories.items():
        current = np.asarray(trajectory, dtype=np.float64)
        if name == "pitch":
            current = current[current > 0]
        if len(current) < 8:
            strength = 0.0
        else:
            current = current - float(np.mean(current))
            scale = float(np.std(current))
            if scale <= 1e-8:
                strength = 0.0
            else:
                windowed = current * np.hanning(len(current))
                power = np.square(np.abs(np.fft.rfft(windowed)))
                frequencies = np.fft.rfftfreq(len(current), d=1.0 / frame_rate)
                periodic = power[(frequencies >= 0.25) & (frequencies <= 20.0)]
                strength = (
                    float(np.max(periodic) / max(float(np.sum(power[1:])), 1e-12))
                    if len(periodic)
                    else 0.0
                )
        result[name] = strength >= 0.12
        result[f"{name}_strength"] = strength
    return result


def route_destination_axis(value: Mapping[str, Any]) -> frozenset[str]:
    """Map a route destination to target movements it can plausibly create."""

    destination = value.get("destination", {})
    text = " ".join(
        str(destination.get(key, ""))
        for key in ("destModuleParamName", "destModuleTypeString")
    ).casefold()
    axes: set[str] = set()
    if any(token in text for token in ("volume", "gain", "level", "amp", "wet", "mix")):
        axes.add("amplitude")
    if any(
        token in text
        for token in (
            "freq",
            "cutoff",
            "reso",
            "drive",
            "table",
            "warp",
            "shift",
            "blur",
            "tone",
            "damp",
        )
    ):
        axes.add("brightness")
    if any(token in text for token in ("pitch", "tune", "coarse", "fine", "semi")):
        axes.add("pitch")
    return frozenset(axes)


def narrow_mod_route_ids(
    entries: list[Mapping[str, Any]],
    movement: Mapping[str, float | bool],
) -> tuple[frozenset[str], dict[str, Any]]:
    """Keep only routes whose destination matches measured target movement."""

    active_axes = {
        axis for axis in ("amplitude", "brightness", "pitch") if movement.get(axis)
    }
    retained = frozenset(
        str(entry.get("id", ""))
        for entry in entries
        if route_destination_axis(entry.get("value", {})) & active_axes
    )
    return retained, {
        "active_axes": sorted(active_axes),
        "input_candidates": len(entries),
        "surviving_candidates": len(retained),
        "practical_limit": PERIODIC_ROUTE_LIMIT,
        "searchable": 0 < len(retained) <= PERIODIC_ROUTE_LIMIT,
        "movement": dict(movement),
    }
