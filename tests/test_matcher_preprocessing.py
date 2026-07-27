from __future__ import annotations

import unittest

import numpy as np

from core.features import CLAP_SAMPLE_RATE
from core.matcher import (
    SearchConfig,
    embedding_comparison_audio,
    objective_weights,
    prepare_query_audio,
)


class MatcherPreprocessingTest(unittest.TestCase):
    def test_adaptive_objective_weights(self) -> None:
        fixtures = (
            (0.25, 0.65, 0.35),
            (0.50, 0.65, 0.35),
            (1.00, 0.50, 0.50),
            (1.50, 0.35, 0.65),
            (4.00, 0.35, 0.65),
        )
        for duration_s, expected_stft, expected_clap in fixtures:
            with self.subTest(duration_s=duration_s):
                stft, clap = objective_weights(duration_s, SearchConfig())
                self.assertAlmostEqual(stft, expected_stft)
                self.assertAlmostEqual(clap, expected_clap)

    def test_short_clap_padding_is_exact_and_shared(self) -> None:
        query = np.ones(12_000, dtype=np.float32)
        candidate = np.ones(20_000, dtype=np.float32)

        prepared_query = embedding_comparison_audio(query, 0.5, adaptive=True)
        prepared_candidate = embedding_comparison_audio(candidate, 0.5, adaptive=True)

        self.assertEqual(len(prepared_query), CLAP_SAMPLE_RATE)
        self.assertEqual(len(prepared_candidate), CLAP_SAMPLE_RATE)
        self.assertTrue(np.all(prepared_query[len(query) :] == 0))
        self.assertTrue(np.all(prepared_candidate[len(candidate) :] == 0))

    def test_query_preparation_is_bounded_to_four_seconds(self) -> None:
        sample_rate = 44_100
        source = np.full(sample_rate * 6, 0.1, dtype=np.float32)

        prepared, duration_s = prepare_query_audio(
            source,
            sample_rate,
            adaptive=False,
        )

        self.assertAlmostEqual(duration_s, 4.0)
        self.assertLessEqual(abs(len(prepared) - CLAP_SAMPLE_RATE * 4), 1)
