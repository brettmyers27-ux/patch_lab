import numpy as np

from core.structural_search import (
    SEARCH_ORDER,
    discover_structural_fields,
    measure_periodic_movement,
    narrow_mod_route_ids,
    staged_proposals,
)


def test_staged_proposals_use_fixed_order_and_observed_frequency() -> None:
    vocabulary = {
        "categories": {
            name: {"entries": []} for name in SEARCH_ORDER
        }
    }
    vocabulary["categories"]["wavetable"]["entries"] = [
        {"id": "rare", "value": "Rare.wav", "observed_count": 1, "provenance": ["observed_in_presets"]},
        {"id": "common", "value": "Common.wav", "observed_count": 10, "provenance": ["observed_in_presets"]},
    ]
    result = staged_proposals(vocabulary, top_k=1)
    assert tuple(result) == SEARCH_ORDER
    assert result["wavetable"][0].stable_id == "common"
    assert result["wavetable"][0].overrides == {
        "Oscillator0.WTOsc0.relativePathToWT": "Common.wav"
    }


def test_mod_route_proposal_writes_every_route_leaf() -> None:
    route = {
        "source": [26, 0],
        "destination": {
            "destModuleID": 0,
            "destModuleParamID": 3,
            "destModuleParamName": "kParamFreq",
            "destModuleTypeString": "VoiceFilter",
        },
    }
    vocabulary = {"categories": {name: {"entries": []} for name in SEARCH_ORDER}}
    vocabulary["categories"]["mod_route"]["entries"] = [
        {"id": "route", "value": route, "observed_count": 3, "provenance": ["observed_in_presets"]}
    ]
    proposal = staged_proposals(vocabulary, top_k=1)["mod_route"][0]
    assert len(proposal.overrides) == 5
    assert proposal.overrides["ModSlot0.destModuleParamName"] == "kParamFreq"


def test_discovers_every_slot_in_a_base_graph() -> None:
    graph = {
        "Oscillator0": {"WTOsc0": {"relativePathToWT": "A.wav"}},
        "FXRack0": {"FX": [{"type": 0}, {"type": 6}]},
        "ModSlot0": {"source": [1, 0]},
        "ModSlot1": {"source": [2, 0]},
    }
    fields = discover_structural_fields(graph)
    assert fields["wavetable"] == ["Oscillator0.WTOsc0.relativePathToWT"]
    assert fields["fx_type"] == ["FXRack0.FX.0.type", "FXRack0.FX.1.type"]
    assert fields["mod_route"] == ["ModSlot0", "ModSlot1"]


def test_controlled_ranking_reorders_and_gates_categories() -> None:
    vocabulary = {"categories": {name: {"entries": []} for name in SEARCH_ORDER}}
    vocabulary["categories"]["wavetable"]["entries"] = [
        {"id": "common", "value": "Common.wav", "observed_count": 10},
        {"id": "measured", "value": "Measured.wav", "observed_count": 1},
    ]
    result = staged_proposals(
        vocabulary,
        top_k=1,
        ranked_ids={"wavetable": ["measured"]},
        enabled_categories=frozenset({"wavetable"}),
    )
    assert result["wavetable"][0].stable_id == "measured"
    assert result["fx_type"] == []


def test_exhaustive_proposals_respect_repaired_allow_list() -> None:
    vocabulary = {"categories": {name: {"entries": []} for name in SEARCH_ORDER}}
    vocabulary["categories"]["fx_type"]["entries"] = [
        {"id": "zero", "value": 0},
        {"id": "one", "value": 1},
        {"id": "two", "value": 2},
    ]
    result = staged_proposals(
        vocabulary,
        top_k=None,
        enabled_categories=frozenset({"fx_type"}),
        allowed_ids={"fx_type": frozenset({"zero", "two"})},
    )
    assert [proposal.stable_id for proposal in result["fx_type"]] == ["two", "zero"]


def test_periodic_route_narrowing_uses_measured_axis() -> None:
    sample_rate = 48_000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    carrier = np.sin(2.0 * np.pi * 220.0 * time)
    audio = carrier * (0.6 + 0.4 * np.sin(2.0 * np.pi * 4.0 * time))
    movement = measure_periodic_movement(audio, sample_rate)
    assert movement["amplitude"] is True
    entries = [
        {
            "id": "gain",
            "value": {
                "destination": {
                    "destModuleParamName": "kParamGain",
                    "destModuleTypeString": "FXComp",
                }
            },
        },
        {
            "id": "pitch",
            "value": {
                "destination": {
                    "destModuleParamName": "kParamPitch",
                    "destModuleTypeString": "WTOsc",
                }
            },
        },
    ]
    retained, report = narrow_mod_route_ids(entries, movement)
    assert "gain" in retained
    assert report["input_candidates"] == 2
