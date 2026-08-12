import numpy as np

from core.structural_fingerprint_validation import (
    deterministic_sample_indices,
    distinctness_components,
    distinctness_clusters,
    self_retrieval,
)


def test_self_retrieval_exposes_collapsed_descriptor_ties() -> None:
    features = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = self_retrieval(features, ["a", "b", "c"])
    assert result.queried == 3
    assert result.failed == 1
    assert result.failures[0][:2] == ("b", "a")


def test_distinctness_clusters_connected_near_duplicates() -> None:
    features = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.999999, 0.001], [0.0, 1.0]],
        dtype=np.float32,
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    result = distinctness_clusters(features, threshold=1e-6, block_size=2)
    assert result.cluster_count == 1
    assert result.clustered_members == 3
    assert result.largest_cluster == 3
    assert result.singleton_count == 1


def test_deterministic_sample_covers_short_and_long_sets() -> None:
    assert deterministic_sample_indices(3, 20).tolist() == [0, 1, 2]
    selected = deterministic_sample_indices(100, 20)
    assert len(selected) == 20
    assert selected[0] == 0
    assert selected[-1] == 99


def test_distinctness_components_returns_only_duplicate_groups() -> None:
    features = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    assert distinctness_components(features, threshold=1e-7) == ((0, 1), (2, 3))
