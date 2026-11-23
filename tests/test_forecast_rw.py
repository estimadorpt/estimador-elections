"""The random-walk forecast path's polling-error schedule."""

import pytest

from src.forecast_rw import RW_NOBIAS_TAU, tau_for_horizon


def test_tau_is_flat_outside_the_calibrated_horizons_and_linear_between():
    assert tau_for_horizon(0) == RW_NOBIAS_TAU[7]  # nowcast: the shortest calibrated horizon
    assert tau_for_horizon(400) == RW_NOBIAS_TAU[56]
    assert tau_for_horizon(28) == RW_NOBIAS_TAU[28]
    mid = tau_for_horizon(42)
    assert mid == pytest.approx((RW_NOBIAS_TAU[28] + RW_NOBIAS_TAU[56]) / 2)


def test_tau_never_shrinks_with_horizon():
    taus = [tau_for_horizon(d) for d in range(0, 120)]
    assert all(b >= a for a, b in zip(taus, taus[1:]))
