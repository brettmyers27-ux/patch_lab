"""Map decoded Serum 2 engine parameters onto the live automation surface."""

from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from core.param_calibration import CalibrationTable, canonical_display
from core.serum2_preset import Serum2Preset


SCALAR_TYPES = (str, int, float, bool)
SELECTOR_TOKENS = ("enable", "mode", "type", "menu", "beatsync", "triplet", "dotted")

UNEXPOSED_PLAIN_CONTROLS = {
    "Global0": {
        "kParamLimitSameNotePolyphony",
        "kParamPolyCount",
        "kParamS1Compatibility",
    },
    "ClipPlayer0": {"kParamSpanKeyboardClip"},
    "MidiClip0": {"kParamLaunchRetrig", "kParamSpanKeyboardMode"},
}
UNEXPOSED_LFO_CONTROLS = {
    "kParamBeatSync",
    "kParamDefaultMode",
    "kParamDotted",
    "kParamMode",
    "kParamRate10x",
    "kParamTriplets",
    "kParamType",
}


GLOBAL_ALIASES = {
    "kParamMasterVolume": "Main Vol",
    "kParamMonoToggle": "Mono Toggle",
    "kParamLegato": "Legato",
    "kParamPortamentoTime": "Porta Time",
    "kParamPortamentoCurve": "Porta Curve",
    "kParamBendRangeUp": "Bend Up",
    "kParamBendRangeDn": "Bend Down",
    "kParamFXBus1Vol": "Bus 1 Vol",
    "kParamFXBus2Vol": "Bus 2 Vol",
    "kParamGlobalTuning": "Main Tuning",
}
OSC_ALIASES = {
    "kParamEnable": "Enable",
    "kParamVolume": "Level",
    "kParamPan": "Pan",
    "kParamOctave": "Octave",
    "kParamPitch": "Semi",
    "kParamFine": "Fine",
    "kParamPitchRatio": "Ratio",
    "kParamHzOffset": "Hz Offset",
    "kParamCoarsePit": "Coarse Pitch",
    "kParamCoarsePitch": "Coarse Pitch",
    "kParamPitchTrack": "Pitch Track",
    "kParamStart": "Start",
    "kParamEnd": "End",
    "kParamReverse": "Reverse",
    "kParamScanRate": "Scan Rate",
    "kParamScanBPMRate": "Scan BPM Rate",
    "kParamScanKeyTrack": "Scan Key Track",
    "kParamPosition": "Position",
    "kParamLoopStart": "Loop Start",
    "kParamLoopEnd": "Loop End",
    "kParamLoopCrossfade": "Loop X-Fade",
    "kParamRelativeLoop": "Relative Loop",
    "kParamSingleSlice": "Single Slice",
    "kParamDetune": "Uni Detune",
    "kParamDetuneWid": "Uni Width",
    "kParamUnisonStereo": "Uni Width",
    "kParamUnison": "Unison",
    "kParamUnisonStack": "Uni Stack",
    "kParamUnisonRange": "Uni Span",
    "kParamUnisonSpan": "Uni Span",
    "kParamRandomStart": "Uni Rand Start",
    "kParamUnisonWarp": "Uni Warp",
    "kParamUnisonWarp2": "Uni Warp 2",
    "kParamUnisonWTPos": "Uni WT Pos",
    "kParamRandomPhase": "Rand Phase",
    "kParamPhase": "Phase",
    "kParamInitialPhase": "Phase",
    "kParamTablePos": "WT Pos",
    "kParamWarpMenu": "Warp Mode",
    "kParamWarp": "Warp Var",
    "kParamWarpMenu2": "Warp 2 Mode",
    "kParamWarp2": "Warp 2 Var",
}
FILTER_ALIASES = {
    "kParamLevelOut": "Level",
    "kParamEnable": "On",
    "kParamType": "Type",
    "kParamFreq": "Freq",
    "kParamReso": "Res",
    "kParamDrive": "Drive",
    "kParamVar": "Var",
    "kParamWet": "Wet",
    "kParamStereo": "Stereo",
    "kParamX": "X",
    "kParamY": "Y",
}
ENV_ALIASES = {
    "kParamAttack": "Attack",
    "kParamHold": "Hold",
    "kParamDecay": "Decay",
    "kParamSustain": "Sustain",
    "kParamRelease": "Release",
    "kParamCurve1": "Atk Curve",
    "kParamCurve2": "Dec Curve",
    "kParamCurve3": "Rel Curve",
    "kParamStart": "Start",
    "kParamEnd": "End",
}
LFO_ALIASES = {
    "kParamRate": "Rate",
    "kParamSmooth": "Smooth",
    "kParamRise": "Rise",
    "kParamDelay": "Delay",
    "kParamPhase": "Phase",
}
CLIP_ALIASES = {
    "kParamEnabled": "Enable",
    "kParamTranspose": "Transpose",
    "kParamRate": "Rate",
    "kParamOffset": "Offset",
}
ARP_ALIASES = {
    "kParamEnabled": "Enable",
    "kParamRate": "Rate",
    "kParamShift": "Shift",
    "kParamRange": "Range",
    "kParamOffset": "Offset",
    "kParamRepeats": "Repeats",
    "kParamGate": "Gate",
    "kParamChance": "Chance",
    "kParamRetrigRate": "Retrig Rate",
    "kParamVeloDecay": "Velo Decay",
    "kParamVeloTarget": "Velo Target",
    "kParamTranspose": "Transpose",
}


@dataclass(frozen=True, slots=True)
class MappingItem:
    cbor_path: str
    component_path: str
    component_type: str
    internal_name: str
    value: Any
    category: str
    status: str
    reason: str
    live_name: str | None = None
    live_index: int | None = None
    method: str | None = None
    confidence: float | None = None
    dependency_priority: int = 1
    dependency_chain: str | None = None


def _component_type(path: str) -> str:
    parts = [part for part in path.split(".") if part and not part.startswith("[")]
    return parts[-1] if parts else "$"


def _camel_words(value: str) -> list[str]:
    value = value.removeprefix("kParam").removeprefix("kUIParam")
    return [part.casefold() for part in re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", value)]


def _normalized_tokens(value: str) -> set[str]:
    aliases = {
        "volume": "level",
        "reso": "res",
        "random": "rand",
        "frequency": "freq",
        "attack": "atk",
        "release": "rel",
        "width": "width",
    }
    words = _camel_words(value) if value.startswith(("kParam", "kUIParam")) else re.findall(
        r"[a-z]+|\d+", value.casefold()
    )
    return {aliases.get(word, word) for word in words if word not in {"param", "plain"}}


def _fuzzy_score(candidate: str, live_name: str) -> float:
    left, right = _normalized_tokens(candidate), _normalized_tokens(live_name)
    if not left or not right:
        return 0.0
    overlap = len(left & right) / len(left | right)
    sequence = difflib.SequenceMatcher(
        None, " ".join(sorted(left)), " ".join(sorted(right))
    ).ratio()
    return 0.65 * overlap + 0.35 * sequence


def _prefix_and_alias(component: str, internal: str) -> tuple[str | None, str | None]:
    if component == "Global0":
        return "", GLOBAL_ALIASES.get(internal)
    match = re.match(r"Oscillator([0-4])$", component)
    if match:
        number = int(match.group(1))
        prefix = {0: "A", 1: "B", 2: "C", 3: "Noise", 4: "Sub"}[number]
        if number == 3:
            alias = {"kParamEnable": "Enable", "kParamVolume": "Level"}.get(internal)
        elif number == 4:
            alias = {
                "kParamEnable": "Enable",
                "kParamVolume": "Level",
                "kParamOctave": "Octave",
                "kParamShape": "Shape",
                "kParamPhase": "Phase",
            }.get(internal)
        else:
            alias = OSC_ALIASES.get(internal)
        return prefix, alias
    match = re.match(r"(?:WT|Sample|MultiSample|Granular|Spectral)Osc([0-2])$", component)
    if match:
        return {0: "A", 1: "B", 2: "C"}[int(match.group(1))], OSC_ALIASES.get(internal)
    if component == "NoiseOsc3":
        return "Noise", {
            "kParamColor": "Pitch",
            "kParamPhase": "Phase",
            "kParamInitialPhase": "Phase",
        }.get(internal)
    if component == "SubOsc4":
        return "Sub", {
            "kParamShape": "Shape",
            "kParamPhase": "Phase",
            "kParamInitialPhase": "Phase",
            "kParamContinuousPhase": "Cont. Phase",
            "kParamContiguousPhase": "Cont. Phase",
        }.get(internal)
    match = re.match(r"VoiceFilter([01])$", component)
    if match:
        return f"Filter {int(match.group(1)) + 1}", FILTER_ALIASES.get(internal)
    match = re.match(r"Env([0-3])$", component)
    if match:
        return f"Env {int(match.group(1)) + 1}", ENV_ALIASES.get(internal)
    match = re.match(r"LFO([0-9])$", component)
    if match:
        return f"LFO {int(match.group(1)) + 1}", LFO_ALIASES.get(internal)
    match = re.match(r"Macro([0-7])$", component)
    if match and internal == "kParamValue":
        return "", f"Macro {int(match.group(1)) + 1}"
    if component == "ClipPlayer0":
        return "Clip Player", CLIP_ALIASES.get(internal)
    if component == "Arp0":
        return "Arp", ARP_ALIASES.get(internal)
    return None, None


def _category(component_path: str) -> str:
    if component_path.startswith("ModSlot"):
        return "mod_matrix"
    if component_path.startswith("FXRack"):
        return "fx"
    if component_path.startswith("RoutingSlot"):
        return "routing"
    if component_path.startswith(("Arp", "MidiClip", "ClipPlayer")):
        return "arp_clip"
    if component_path.startswith("SerumGUI"):
        return "gui"
    return "continuous_synth"


def _priority(internal: str) -> int:
    lowered = internal.casefold()
    return 0 if any(token in lowered for token in SELECTOR_TOKENS) else 1


def _dependency(component: str, internal: str) -> str | None:
    lowered = internal.casefold()
    if "warp" in lowered and not any(token in lowered for token in ("menu", "mode")):
        return f"{component}: warp mode/type -> warp amount"
    if component.startswith("VoiceFilter") and not any(
        token in lowered for token in ("enable", "type")
    ):
        return f"{component}: enable -> filter type -> dependent controls"
    if component.startswith("LFO") and internal in {"kParamRate", "kParamRise", "kParamDelay"}:
        return f"{component}: mode/sync/triplet/dotted -> timing controls"
    if component.startswith("Oscillator") and internal not in {"kParamEnable", "kParamType"}:
        return f"{component}: enable/type -> engine-specific controls"
    return None


class Serum2Mapper:
    def __init__(self, calibration: CalibrationTable) -> None:
        self.calibration = calibration
        self.live_entries = [
            entry
            for group in calibration.payload["parameters"].values()
            for entry in group
            if int(entry["index"]) <= 540
        ]
        self.live_by_name = {entry["name"]: entry for entry in self.live_entries}

    def _map_plain(
        self, component_path: str, internal: str, value: Any
    ) -> MappingItem:
        component = _component_type(component_path)
        category = _category(component_path)
        path = f"{component_path}.plainParams.{internal}"
        priority = _priority(internal)
        dependency = _dependency(component, internal)
        if not isinstance(value, SCALAR_TYPES):
            return MappingItem(
                path,
                component_path,
                component,
                internal,
                value,
                category,
                "unsupported",
                "Structured plainParams value is not settable through one automation parameter",
                dependency_priority=priority,
                dependency_chain=dependency,
            )
        if category == "fx":
            return MappingItem(
                path,
                component_path,
                component,
                internal,
                value,
                category,
                "unsupported",
                "FX topology/type is not exposed; generic FX Param controls cannot identify this module",
                dependency_priority=priority,
                dependency_chain=f"{component_path}: FX type/topology -> FX controls",
            )
        if category == "mod_matrix":
            slot = int(re.search(r"\d+", component_path).group()) + 1  # type: ignore[union-attr]
            live_name = f"Mod {slot} Amount" if internal == "kParamAmount" else None
            entry = self.live_by_name.get(live_name) if live_name else None
            return MappingItem(
                path,
                component_path,
                component,
                internal,
                value,
                category,
                "blocked_dependency",
                "Amount is exposed but source/destination route creation is not",
                live_name=live_name if entry else None,
                live_index=int(entry["index"]) if entry else None,
                method="explicit-blocked" if entry else None,
                confidence=1.0 if entry else None,
                dependency_priority=1,
                dependency_chain=f"{component_path}: source + destination -> amount",
            )
        if category in {"routing", "gui"}:
            return MappingItem(
                path,
                component_path,
                component,
                internal,
                value,
                category,
                "unsupported",
                "No corresponding named live automation control",
                dependency_priority=priority,
                dependency_chain=dependency,
            )

        if internal in UNEXPOSED_PLAIN_CONTROLS.get(component, set()):
            return MappingItem(
                path,
                component_path,
                component,
                internal,
                value,
                category,
                "unsupported",
                "The live automation surface has no corresponding compatibility/clip control",
                dependency_priority=priority,
                dependency_chain=dependency,
            )
        if component.startswith("LFO") and internal in UNEXPOSED_LFO_CONTROLS:
            return MappingItem(
                path,
                component_path,
                component,
                internal,
                value,
                category,
                "blocked_dependency",
                "LFO mode/sync/type is not exposed as a live automation parameter",
                dependency_priority=priority,
                dependency_chain=f"{component}: mode/sync/triplet/dotted -> timing controls",
            )

        prefix, alias = _prefix_and_alias(component, internal)
        expected = " ".join(part for part in (prefix, alias) if part) if alias else None
        if expected and expected in self.live_by_name:
            entry = self.live_by_name[expected]
            return MappingItem(
                path,
                component_path,
                component,
                internal,
                value,
                category,
                "mapped",
                "Explicit component/parameter alias",
                expected,
                int(entry["index"]),
                "explicit",
                1.0,
                priority,
                dependency,
            )

        # Restricted fuzzy matching: candidate prefix must agree with the module.
        if prefix is not None:
            candidate = " ".join(part for part in (prefix, " ".join(_camel_words(internal))) if part)
            pool = [
                entry
                for entry in self.live_entries
                if not prefix or str(entry["name"]).casefold().startswith(prefix.casefold() + " ")
            ]
            scored = sorted(
                ((_fuzzy_score(candidate, str(entry["name"])), entry) for entry in pool),
                key=lambda item: item[0],
                reverse=True,
            )
            if scored and scored[0][0] >= 0.90 and (
                len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.08
            ):
                score, entry = scored[0]
                return MappingItem(
                    path,
                    component_path,
                    component,
                    internal,
                    value,
                    category,
                    "mapped",
                    "Unique normalized-name match",
                    str(entry["name"]),
                    int(entry["index"]),
                    "fuzzy",
                    score,
                    priority,
                    dependency,
                )
        return MappingItem(
            path,
            component_path,
            component,
            internal,
            value,
            category,
            "unmapped",
            "No unambiguous live automation name",
            dependency_priority=priority,
            dependency_chain=dependency,
        )

    def map_preset(self, preset: Serum2Preset) -> dict[str, Any]:
        items: list[MappingItem] = []
        structural: list[dict[str, Any]] = []

        def walk(value: Any, path: str = "") -> None:
            if isinstance(value, dict):
                plain = value.get("plainParams")
                if isinstance(plain, dict):
                    for internal, target in plain.items():
                        items.append(self._map_plain(path, str(internal), target))
                for key, child in value.items():
                    child_path = f"{path}.{key}" if path else str(key)
                    if key == "relativePathToWT":
                        structural.append(
                            {
                                "path": child_path,
                                "category": "wavetable_selection",
                                "status": "unsupported",
                                "value": child,
                                "reason": "No live wavetable file/name/index selection parameter",
                            }
                        )
                    elif key == "embeddedWTData":
                        structural.append(
                            {
                                "path": child_path,
                                "category": "embedded_wavetable",
                                "status": "unsupported",
                                "length": len(child) if isinstance(child, list) else None,
                                "reason": "No custom-data/wavetable loading hook in either host",
                            }
                        )
                    elif key == "relativePathToNoiseSample":
                        structural.append(
                            {
                                "path": child_path,
                                "category": "sample_selection",
                                "status": "unsupported",
                                "value": child,
                                "reason": "No live noise/sample file selection parameter",
                            }
                        )
                    elif path.startswith("ModSlot") and key in {
                        "source",
                        "destModuleID",
                        "destModuleParamID",
                        "destModuleParamName",
                        "destModuleTypeString",
                    }:
                        structural.append(
                            {
                                "path": child_path,
                                "category": "mod_matrix_structure",
                                "status": "unsupported",
                                "value": child,
                                "reason": "No live modulation source/destination parameter",
                            }
                        )
                    elif path.startswith("FXRack") and key == "type":
                        structural.append(
                            {
                                "path": child_path,
                                "category": "fx_topology",
                                "status": "unsupported",
                                "value": child,
                                "reason": "No live effect type/topology parameter",
                            }
                        )
                    walk(child, child_path)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, f"{path}.[{index}]")

        walk(preset.data)
        status_counts = Counter(item.status for item in items)
        category_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for item in items:
            category_counts[item.category]["total"] += 1
            category_counts[item.category][item.status] += 1
        dependency_counts = Counter(
            item.dependency_chain for item in items if item.dependency_chain
        )
        total = len(items)
        mapped = status_counts["mapped"]
        metadata = preset.metadata if isinstance(preset.metadata, dict) else {}
        return {
            "file": str(preset.path),
            "preset_name": metadata.get("presetName", preset.path.stem),
            "plain_param_total": total,
            "mapped": mapped,
            "mapping_coverage": mapped / total if total else 0.0,
            "status_counts": dict(status_counts),
            "category_counts": {key: dict(value) for key, value in category_counts.items()},
            "dependency_chains": dict(dependency_counts),
            "items": [asdict(item) for item in items],
            "structural_gaps": structural,
        }


def aggregate_reports(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    reports = list(reports)
    status = Counter()
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    unmapped_by_component: dict[str, Counter[str]] = defaultdict(Counter)
    dependencies = Counter()
    structural = Counter()
    for report in reports:
        status.update(report["status_counts"])
        for category, counts in report["category_counts"].items():
            categories[category].update(counts)
        dependencies.update(report["dependency_chains"])
        for item in report["items"]:
            if item["status"] != "mapped":
                unmapped_by_component[item["component_type"]][item["internal_name"]] += 1
        for gap in report["structural_gaps"]:
            structural[gap["category"]] += 1
    total = sum(status.values())
    return {
        "presets": len(reports),
        "plain_param_total": total,
        "mapped": status["mapped"],
        "mapping_coverage": status["mapped"] / total if total else 0.0,
        "status_counts": dict(status),
        "category_counts": {key: dict(value) for key, value in categories.items()},
        "unmapped_by_component": {
            component: dict(values) for component, values in sorted(unmapped_by_component.items())
        },
        "dependency_chains": dict(dependencies),
        "structural_gap_counts": dict(structural),
    }
