from scripts.benchmark_serum2_structural_cost import design_decision


def test_structural_cost_decision_thresholds() -> None:
    assert design_decision(4.999) == "direct-shortlist-search-viable"
    assert design_decision(5.0) == "direct-search-only-with-tight-shortlists"
    assert design_decision(20.0) == "direct-search-only-with-tight-shortlists"
    assert design_decision(20.001) == "neural-surrogate-prerequisite"
