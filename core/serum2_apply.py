"""Apply mapped Serum 2 CBOR values through the live automation surface."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from core.param_calibration import CalibrationMatch, CalibrationTable
from core.plugin_host import (
    SILENCE_DBFS,
    ParameterValue,
    audio_levels,
    changed_parameter_count,
    dawdreamer_parameter_display,
    dump_dawdreamer_parameters,
    make_dawdreamer_processor,
    render_dawdreamer_note,
)
from core.platform_env import PluginCandidate
from core.serum2_mapping import Serum2Mapper
from core.serum2_preset import Serum2Preset


FILTER_TYPE_ALIASES = {
    "HP12": "High 12",
    "B12": "Band 12",
    "FlangeN": "Flg -",
    "FlangeP": "Flg +",
    "MgL18": "MG Low 18",
}
WARP_MODE_ALIASES = {
    "kSync": "Sync",
    "kPD_OSC": "PD (Self)",
}
UNISON_STACK_ALIASES = {
    "kOctave1": "12 (1x)",
    "kOctave2": "12 (2x)",
    "kOctave3": "12 (3x)",
    "kOctaveFifth1": "12+7(1x)",
    "kOctaveFifth2": "12+7(2x)",
    "kOctaveFifth3": "12+7(3x)",
    "kCenter12": "Center-12",
    "kCenter24": "Center-24",
}


@dataclass(frozen=True, slots=True)
class AppliedValue:
    cbor_path: str
    category: str
    live_index: int
    live_name: str
    source_value: Any
    translated_target: Any
    normalized: float
    display: str
    method: str
    score: float


def _seconds_display(value: float) -> str:
    if abs(value) < 1.0:
        return f"{value * 1000.0:.12g} ms"
    return f"{value:.12g} s"


def _warp_mode(value: str, live_name: str) -> str:
    if value == "kFM_OSC":
        return "FM (B)" if live_name.startswith("A ") else "FM (A)"
    return WARP_MODE_ALIASES.get(value, value.removeprefix("k"))


def translated_target(item: dict[str, Any]) -> tuple[Any, str | None]:
    """Translate CBOR plain units/codes into the plug-in's displayed units.

    The optional second result is a direct mode. ``normalized`` is used only
    where the CBOR engine stores the same normalized coordinate exposed by VST.
    """

    internal = str(item["internal_name"])
    component = str(item["component_type"])
    live_name = str(item["live_name"])
    value = item["value"]
    numeric = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    if internal in {"kParamEnable", "kParamEnabled", "kParamMonoToggle", "kParamLegato"}:
        return ("On" if numeric and numeric >= 0.5 else "Off"), None
    if internal == "kParamType" and component.startswith("VoiceFilter"):
        return FILTER_TYPE_ALIASES.get(str(value), value), None
    if internal in {"kParamWarpMenu", "kParamWarpMenu2"}:
        return _warp_mode(str(value), live_name), None
    if internal == "kParamUnisonStack":
        return UNISON_STACK_ALIASES.get(str(value), value), None
    if internal in {"kParamPortamentoTime"}:
        return _seconds_display(float(value)), None
    if internal in {"kParamAttack", "kParamHold", "kParamDecay", "kParamRelease"} and component.startswith("Env"):
        return _seconds_display(float(value)), None
    if internal == "kParamSustain" and component.startswith("Env"):
        if numeric is None or numeric <= 0.0:
            return "-∞ dB", None
        return f"{20.0 * math.log10(numeric):.12g} dB", None
    if internal in {
        "kParamVolume",
        "kParamMasterVolume",
        "kParamFXBus1Vol",
        "kParamFXBus2Vol",
        "kParamLevelOut",
    }:
        return f"{float(value):.12g} dB", None
    if internal == "kParamColor" and component == "NoiseOsc3":
        return float(value) * 100.0, None
    if internal == "kParamFreq" and component.startswith("VoiceFilter"):
        return min(1.0, max(0.0, float(value))), "normalized"
    return value, None


def _resolve(
    calibration: CalibrationTable,
    processor: Any,
    item: dict[str, Any],
    *,
    refine: bool,
) -> tuple[CalibrationMatch, Any]:
    target, direct = translated_target(item)
    index = int(item["live_index"])
    if direct == "normalized":
        normalized = float(target)
        return CalibrationMatch(normalized, "", "normalized-direct", 1.0), target
    if refine:
        match = calibration.inverse_live(
            processor, str(item["live_name"]), target, index=index
        )
    else:
        match = calibration.inverse(str(item["live_name"]), target, index=index)
    return match, target


def apply_preset(
    preset: Serum2Preset,
    calibration: CalibrationTable,
    candidate: PluginCandidate,
    *,
    refine: bool = True,
) -> dict[str, Any]:
    engine, processor = make_dawdreamer_processor(candidate)
    initial = dump_dawdreamer_parameters(processor)
    report, _parameters = apply_preset_on_processor(
        preset,
        calibration,
        engine,
        processor,
        initial,
        refine=refine,
    )
    return report


def apply_preset_on_processor(
    preset: Serum2Preset,
    calibration: CalibrationTable,
    engine: Any,
    processor: Any,
    initial: list[ParameterValue],
    *,
    refine: bool = True,
) -> tuple[dict[str, Any], list[ParameterValue]]:
    """Apply one preset using a reusable processor and restore init first."""

    for parameter in initial[:541]:
        processor.set_parameter(parameter.index, parameter.norm_value)
    mapper = Serum2Mapper(calibration)
    mapping = mapper.map_preset(preset)
    applied: list[AppliedValue] = []
    errors: list[dict[str, Any]] = []
    mapped_items = sorted(
        (item for item in mapping["items"] if item["status"] == "mapped"),
        key=lambda item: (
            int(item["dependency_priority"]),
            str(item["component_path"]),
            str(item["internal_name"]),
        ),
    )
    for item in mapped_items:
        try:
            match, target = _resolve(calibration, processor, item, refine=refine)
            index = int(item["live_index"])
            processor.set_parameter(index, match.normalized)
            actual = float(processor.get_parameter(index))
            display = dawdreamer_parameter_display(processor, index, actual)
            applied.append(
                AppliedValue(
                    cbor_path=str(item["cbor_path"]),
                    category=str(item["category"]),
                    live_index=index,
                    live_name=str(item["live_name"]),
                    source_value=item["value"],
                    translated_target=target,
                    normalized=actual,
                    display=display,
                    method=match.method,
                    score=match.score,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "cbor_path": item["cbor_path"],
                    "live_name": item["live_name"],
                    "value": item["value"],
                    "error": repr(exc),
                }
            )

    final = dump_dawdreamer_parameters(processor)
    changed = changed_parameter_count(initial, final)
    audio = render_dawdreamer_note(engine, processor, midi_note=60, duration=1.0)
    peak_dbfs, rms_dbfs = audio_levels(audio)
    category_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for item in mapping["items"]:
        category_totals[str(item["category"])]["total"] += 1
        category_totals[str(item["category"])][str(item["status"])] += 1
    for item in applied:
        category_totals[item.category]["applied"] += 1
    total = int(mapping["plain_param_total"])
    report = {
        "file": str(preset.path),
        "preset_name": mapping["preset_name"],
        "plain_param_total": total,
        "mapped": len(mapped_items),
        "applied": len(applied),
        "application_coverage": len(applied) / total if total else 0.0,
        "category_counts": {key: dict(value) for key, value in category_totals.items()},
        "changed_from_init": changed,
        "non_silent": rms_dbfs > SILENCE_DBFS,
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "errors": errors,
        "applied_values": [asdict(item) for item in applied],
        "parameter_vector": [item.norm_value for item in final[:541]],
        "section_6_1_init_pass": changed >= 5,
        "section_6_1_silence_pass": rms_dbfs > SILENCE_DBFS,
    }
    return report, final
