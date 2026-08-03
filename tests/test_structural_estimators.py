import numpy as np

from core.structural_estimators import (
    NearestStructuralEstimator,
    deterministic_split,
    evaluate_estimator,
)


def test_estimator_ranks_nearest_class_and_calibrates_confidence() -> None:
    features = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    estimator = NearestStructuralEstimator().fit(features, ["warm", "warm", "bright"])
    choices = estimator.rank_descriptor(np.asarray([0.0, 1.0], dtype=np.float32), 2)
    assert choices[0].value == "bright"
    assert abs(sum(item.confidence for item in choices) - 1.0) < 1e-6


def test_failed_estimator_is_disabled_against_common_baseline() -> None:
    train = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    estimator = NearestStructuralEstimator().fit(train, ["common", "rare"])
    test = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    metrics = evaluate_estimator(estimator, test, ["common", "common"])
    assert metrics.adopted is False
    assert estimator.enabled is False


def test_split_is_stable_by_preset_id() -> None:
    train, test = deterministic_split([1, 2, 5, 6, 10])
    assert train.tolist() == [0, 1, 3]
    assert test.tolist() == [2, 4]
