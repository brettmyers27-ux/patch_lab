import numpy as np

from core.structural_search import (
    SEARCH_ORDER,
    discover_structural_fields,
    fit_mod_route_ids,
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


def test_route_budget_uses_full_set_when_it_fits() -> None:
    entries = [
        {
            "id": f"route-{index}",
            "observed_count": 1,
            "value": {
                "destination": {
                    "destModuleID": 0,
                    "destModuleParamID": 1,
                    "destModuleParamName": "kParamGain",
                    "destModuleTypeString": "FXComp",
                }
            },
        }
        for index in range(3)
    ]
    retained, report = fit_mod_route_ids(
        entries,
        {"amplitude": True, "amplitude_strength": 0.8},
        field_count=2,
        non_route_evaluations=100,
    )
    assert set(retained) == {"route-0", "route-1", "route-2"}
    assert report["selected_structural_evaluations"] == 106
    assert report["structural_budget"] == 4096
    assert report["hierarchical_fallback"] is False


def test_route_budget_preserves_complete_destination_groups() -> None:
    entries = []
    for destination, count, observed in (
        ("gain", 3, 10),
        ("level", 3, 5),
        ("mix", 1, 1),
    ):
        for index in range(count):
            entries.append(
                {
                    "id": f"{destination}-{index}",
                    "observed_count": observed,
                    "value": {
                        "destination": {
                            "destModuleID": 0,
                            "destModuleParamID": {
                                "gain": 1,
                                "level": 2,
                                "mix": 3,
                            }[destination],
                            "destModuleParamName": f"kParam{destination.title()}",
                            "destModuleTypeString": "FXComp",
                        }
                    },
                }
            )
    retained, report = fit_mod_route_ids(
        entries,
        {"amplitude": True, "amplitude_strength": 0.8},
        field_count=2,
        non_route_evaluations=4,
        standard_limit=8,
        maximum_limit=10,
    )
    assert set(retained) == {"gain-0", "gain-1", "gain-2"}
    assert report["hierarchical_fallback"] is True
    assert report["destination_groups_before"] == 3
    assert report["destination_groups_after"] == 1
    assert report["selected_structural_evaluations"] == 10
