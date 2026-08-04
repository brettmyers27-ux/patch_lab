from core.structural_search import SEARCH_ORDER, discover_structural_fields, staged_proposals


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
