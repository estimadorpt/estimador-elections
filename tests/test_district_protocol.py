"""Scores and the sign-flip test of the district factor protocol."""

import numpy as np
from scipy.stats import norm

from src.evaluation.district_protocol import crps_cells, rps_cells, sign_flip_p


def test_crps_matches_closed_form_for_a_normal():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.0, 1.0, size=(200_000, 1))
    y = np.array([0.7])
    exact = 0.7 * (2 * norm.cdf(0.7) - 1) + 2 * norm.pdf(0.7) - 1 / np.sqrt(np.pi)
    assert abs(crps_cells(draws, y)[0] - exact) < 0.01


def test_crps_of_a_point_forecast_is_absolute_error():
    draws = np.full((50, 3), 2.0)
    y = np.array([2.0, 3.5, 1.0])
    np.testing.assert_allclose(crps_cells(draws, y), [0.0, 1.5, 1.0], atol=1e-12)


def test_rps_counts_each_wrong_step():
    draws = np.full((10, 2), 3)
    y = np.array([3, 5])
    np.testing.assert_allclose(rps_cells(draws, y), [0.0, 2.0])


def test_sign_flip_p():
    assert sign_flip_p(np.full(12, 0.1)) < 0.001
    assert sign_flip_p(np.full(12, -0.1)) == 1.0
    assert 0.2 < sign_flip_p(np.array([0.1, -0.1] * 6)) < 0.8
