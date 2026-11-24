"""The ExpQuad spectral density must keep a finite gradient when it underflows."""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from src.models.dynamic_gp_election_model import hsgp_sqrt_psd


def _sqrt_psd(ls_value: float, half_range: float, m: int = 35):
    X = np.linspace(-half_range, half_range, 200)[:, None]
    ls = pt.dscalar("ls")
    amp = pt.dscalar("amp")
    gp = pm.gp.HSGP(cov_func=amp**2 * pm.gp.cov.ExpQuad(1, ls=ls), m=[m], c=1.5)
    _, pymc_value = gp.prior_linearized(X=X)
    ours = hsgp_sqrt_psd(gp, "ExpQuad", amp, ls, pymc_value)
    return ls, amp, pymc_value, ours


def test_matches_pymc_where_finite():
    ls, amp, pymc_value, ours = _sqrt_psd(1460.0, half_range=5000.0)
    f = pytensor.function([ls, amp], [pymc_value, ours])
    a, b = f(1460.0, 0.2)
    np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-300)


def test_gradient_is_finite_when_density_underflows():
    # a 2015 backtest: ~6 years of data, a 4-year lengthscale, exponent ~ -1,200
    ls, amp, pymc_value, ours = _sqrt_psd(1460.0, half_range=1100.0)
    loss = pt.sum(ours)
    grads = pytensor.function([ls, amp], pytensor.grad(loss, [ls, amp]))(1460.0, 0.2)
    assert all(np.isfinite(g) for g in grads)
    pymc_grads = pytensor.function([ls, amp], pytensor.grad(pt.sum(pymc_value), [ls, amp]))(1460.0, 0.2)
    assert not all(np.isfinite(g) for g in pymc_grads)  # the bug this guards against


def test_other_kernels_keep_pymc_value():
    sentinel = pt.ones(3)
    assert hsgp_sqrt_psd(None, "Matern52", None, None, sentinel) is sentinel
