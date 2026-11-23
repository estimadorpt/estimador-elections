import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest

from src.models.poll_rw_model import RWConfig, RWData, build_rw_model, short_term_process


def _toy_data(T=30, P=3, n_polls=40, seed=0):
    rng = np.random.default_rng(seed)
    grid = pd.date_range("2020-01-05", periods=T, freq="7D")
    exists = np.ones((T, P), bool)
    exists[:10, 2] = False  # a party that enters later
    poll_t = rng.integers(10, T, n_polls)
    counts = rng.multinomial(800, [0.5, 0.3, 0.2], size=n_polls)
    return RWData(
        parties=[f"p{i}" for i in range(P)],
        grid=grid,
        exists=exists,
        poll_t=poll_t,
        poll_pollster=rng.integers(0, 2, n_polls),
        poll_cycle=np.zeros(n_polls, int),
        poll_counts=counts,
        pollsters=["a", "b"],
        cycles=[str(grid[-1].date())],
        result_t=np.array([], int),
        result_cycle=np.array([], int),
        result_counts=np.zeros((0, P), int),
        target_t=T - 1,
    )


def test_rw_dynamics_adds_no_variables():
    names = {v.name for v in build_rw_model(_toy_data(), RWConfig()).free_RVs}
    assert names == {"level0", "sigma_rw", "z_rw", "sigma_house", "house_raw", "sigma_bias", "bias_raw", "concentration_polls"}


@pytest.mark.parametrize("dynamics", ["rw_ou", "rw_m32"])
def test_short_term_models_build_with_finite_logp(dynamics):
    model = build_rw_model(_toy_data(), RWConfig(dynamics=dynamics))
    names = {v.name for v in model.free_RVs}
    assert {"short_timescale", "sigma_short", "z_short"} <= names
    assert np.isfinite(model.compile_logp()(model.initial_point()))


@pytest.mark.parametrize(
    "dynamics,kernel",
    [
        ("rw_ou", lambda k, t: np.exp(-k / t)),
        ("rw_m32", lambda k, t: (1 + np.sqrt(3) * k / t) * np.exp(-np.sqrt(3) * k / t)),
    ],
)
def test_short_term_process_has_the_kernel_autocorrelation(dynamics, kernel):
    # the property the GP model's calendar-wide HSGP lacks: short lags decorrelate as the kernel says
    rng = np.random.default_rng(1)
    shape = (200, 3000) if dynamics == "rw_ou" else (200, 2, 3000)
    e = pt.tensor("e", shape=(None,) * len(shape))
    for tau in (2.0, 6.0):
        x = pytensor.function([e], short_term_process(e, tau, dynamics))(rng.standard_normal(shape))
        assert abs(x.var() - 1.0) < 0.05
        for k in (1, 2, 4, 8):
            acf = (x[:-k] * x[k:]).mean() / x.var()
            assert abs(acf - kernel(k, tau)) < 0.03


@pytest.mark.parametrize("dynamics", ["rw_ou", "rw_m32"])
def test_short_term_models_sample_with_numpyro(dynamics):
    # the backtest samples with numpyro (JAX); scan needs static shapes there
    import pymc as pm

    model = build_rw_model(_toy_data(), RWConfig(dynamics=dynamics))
    with model:
        idata = pm.sample(draws=20, tune=20, chains=1, nuts_sampler="numpyro", progressbar=False, random_seed=1)
    assert np.isfinite(idata.posterior["eta"].values).all()
