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


def test_bloc_swing_and_heavy_tails_keep_tau_as_the_sd_and_correlate_blocs():
    import numpy as np

    from src.processing.polling_error import add_polling_error, bloc_signs

    parties = ["PS", "CH", "IL", "BE", "CDU", "PAN", "L", "AD"]
    base = np.tile(np.array([0.3, 0.25, 0.06, 0.03, 0.03, 0.02, 0.04, 0.27]), (40000, 1))
    signs = bloc_signs(parties)
    d_iid = np.log(add_polling_error(base, 0.2, np.random.default_rng(0))) - np.log(base)
    d_bloc = np.log(add_polling_error(base, 0.2, np.random.default_rng(0), bloc_rho=0.5, signs=signs)) - np.log(base)
    d_t = np.log(add_polling_error(base, 0.2, np.random.default_rng(0), df=5.0)) - np.log(base)
    ps, ch, ad = 0, 1, 7
    corr = lambda d, i, j: np.corrcoef(d[:, i], d[:, j])[0, 1]
    # renormalising already pulls big parties apart; the swing pulls blocs apart further ...
    assert corr(d_bloc, ps, ad) < corr(d_iid, ps, ad) - 0.2
    # ... and moves parties of one bloc together
    assert corr(d_bloc, ch, ad) > corr(d_iid, ch, ad) + 0.3
    assert abs(d_t[:, ad].std() / d_iid[:, ad].std() - 1) < 0.05  # same sd ...
    kurt = lambda x: ((x - x.mean()) ** 4).mean() / x.var() ** 2
    assert kurt(d_t[:, ad]) > kurt(d_iid[:, ad]) + 1  # ... heavier tails
