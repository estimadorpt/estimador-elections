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


def test_full_diagnostics_names_the_worst_value_and_skips_absent_cells():
    import arviz as az
    import numpy as np

    from src.forecast_rw import full_diagnostics

    rng = np.random.default_rng(0)
    T, P = 5, 2
    eta = rng.normal(size=(4, 500, T, P))
    eta[:, :, 0, 1] = 0.0  # party 1 does not exist at t=0: constant, and masked
    eta[0, :, 3, 0] += 5.0  # one chain stuck elsewhere at (t=3, party 0)
    idata = az.from_dict(
        posterior={"eta": eta, "sigma": np.abs(rng.normal(size=(4, 500)))},
        sample_stats={"diverging": np.zeros((4, 500), bool), "step_size": np.full((4, 500), 0.1)},
        coords={"time": list("abcde"), "parties": ["X", "Y"]},
        dims={"eta": ["time", "parties"]},
    )
    exists = np.ones((T, P), bool)
    exists[0, 1] = False
    diag, table = full_diagnostics(idata, exists)
    assert diag["max_rhat_at"] == "eta[d, X]"
    assert not diag["passes_gate"]
    assert "eta[a, Y]" not in table.index  # absent cell dropped
    assert table.index[0] == "eta[d, X]"  # worst first
    assert diag["n_values"] == T * P - 1 + 1


def test_light_posterior_keeps_recent_weeks_and_drops_innovations():
    from types import SimpleNamespace

    import arviz as az
    import numpy as np

    from src.forecast_rw import _light_posterior

    rng = np.random.default_rng(0)
    T = 200
    idata = az.from_dict(
        posterior={"eta": rng.normal(size=(2, 10, T, 3)), "z_rw": rng.normal(size=(2, 10, T, 3)), "s": rng.normal(size=(2, 10))},
        coords={"time": np.arange(T), "parties": list("abc")},
        dims={"eta": ["time", "parties"], "z_rw": ["time", "parties"]},
    )
    light = _light_posterior(idata, SimpleNamespace(result_t=np.array([10])))
    kept = light["eta"]["time"].values
    assert "z_rw" not in light and "s" in light
    assert set(range(T - 52, T)) <= set(kept) and 10 in kept and len(kept) < T
    assert not np.isnan(light["eta"].values).any() and light["eta"].dtype == np.float32
