import numpy as np

from src.processing.polling_error import add_polling_error


def _draws(S=4000):
    base = np.array([0.30, 0.20, 0.05, 0.0, 0.45])
    return np.tile(base, (S, 1))


def test_zero_tau_is_identity():
    x = _draws(10)
    np.testing.assert_array_equal(add_polling_error(x, 0.0), x)


def test_rows_stay_on_the_simplex_and_absent_parties_stay_absent():
    out = add_polling_error(_draws(), 0.2, np.random.default_rng(0))
    np.testing.assert_allclose(out.sum(1), 1.0)
    assert (out >= 0).all()
    assert out[:, 3].max() == 0.0
    np.testing.assert_allclose(out.mean(0), _draws(1)[0], atol=0.01)


def test_spread_grows_with_tau_and_with_share():
    sd1 = add_polling_error(_draws(), 0.1, np.random.default_rng(1)).std(0)
    sd2 = add_polling_error(_draws(), 0.2, np.random.default_rng(1)).std(0)
    alive = [0, 1, 2, 4]
    np.testing.assert_allclose(sd2[alive] / sd1[alive], 2.0, rtol=0.1)
    assert sd2[4] > sd2[0] > sd2[1] > sd2[2] > 0
