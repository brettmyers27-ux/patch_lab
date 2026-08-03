from pathlib import Path

from core.serum2_preset import Serum2Preset
from core.serum2_structural_space import (
    apply_structural_overrides,
    build_structural_space,
    stable_id,
)


def _preset(data):
    return Serum2Preset(Path("fixture.SerumPreset"), {}, data, 0, 0, 2, 0)


def test_stable_ids_are_order_independent() -> None:
    left = {"b": 2, "a": [1, 3]}
    right = {"a": [1, 3], "b": 2}
    assert stable_id("route", left) == stable_id("route", right)


def test_harvests_all_structural_categories_and_provenance(tmp_path: Path) -> None:
    table = tmp_path / "Tables" / "Factory" / "Extra.wav"
    noise = tmp_path / "Samples" / "Factory Non-Tonal" / "Noises" / "Air.flac"
    table.parent.mkdir(parents=True)
    noise.parent.mkdir(parents=True)
    table.write_bytes(b"")
    noise.write_bytes(b"")
    graph = {
        "Oscillator0": {"WTOsc0": {"relativePathToWT": "Factory/Base.wav", "embeddedWTData": [1, 2]}},
        "Noise": {"relativePathToNoiseSample": "Air.flac"},
        "ModSlot0": {
            "source": [3, 0],
            "destModuleID": 0,
            "destModuleParamID": 3,
            "destModuleParamName": "kParamFreq",
            "destModuleTypeString": "VoiceFilter",
        },
        "FXRack0": {"FX": [{"type": 4}]},
    }
    space = build_structural_space([_preset(graph)], [tmp_path])
    assert space.summary() == {
        "wavetable": 2,
        "embedded_wavetable": 1,
        "noise_sample": 1,
        "mod_source": 1,
        "mod_destination": 1,
        "mod_route": 1,
        "fx_type": 1,
    }
    table_entry = next(item for item in space.entries("wavetable") if item.value == "Factory/Extra.wav")
    assert table_entry.provenance == {"enumerated_from_install"}


def test_structural_overrides_are_not_restricted_to_observed_values() -> None:
    graph = {"Oscillator0": {"WTOsc0": {"relativePathToWT": "Old.wav"}}, "FXRack0": {"FX": [{"type": 0}]}}
    apply_structural_overrides(
        graph,
        {
            "Oscillator0.WTOsc0.relativePathToWT": "Never/Seen.wav",
            "FXRack0.FX.0.type": 15,
        },
    )
    assert graph["Oscillator0"]["WTOsc0"]["relativePathToWT"] == "Never/Seen.wav"
    assert graph["FXRack0"]["FX"][0]["type"] == 15
