import pytest
import numpy as np
import pandas as pd

from src.processing.district_swing import (
    add_district_noise,
    blend_swing,
    calibrate_noise,
    dhondt_draws,
    seats_from_district_shares,
)
from src.processing.electoral_systems import calculate_dhondt


def test_dhondt_textbook_example():
    shares = np.array([[100, 80, 30, 20]], dtype=float) / 230
    assert dhondt_draws(shares, 8).tolist() == [[4, 3, 1, 0]]


def test_dhondt_matches_reference_implementation():
    rng = np.random.default_rng(1)
    shares = rng.dirichlet(np.ones(6), size=200)
    for seats in (2, 3, 9, 19, 48):
        fast = dhondt_draws(shares, seats)
        assert (fast.sum(1) == seats).all()
        for s, row in zip(shares[:20], fast[:20]):
            ref = calculate_dhondt({str(i): int(round(v * 1e7)) for i, v in enumerate(s)}, seats)
            assert [ref[str(i)] for i in range(6)] == row.tolist()


def test_blend_swing_reproduces_last_election_when_national_is_unchanged():
    last_nat = np.array([0.4, 0.35, 0.25])
    last_dist = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
    out = blend_swing(last_nat[None, :], last_nat, last_dist)
    np.testing.assert_allclose(out[0], last_dist, atol=1e-12)


def test_blend_swing_moves_with_national_and_stays_on_simplex():
    last_nat = np.array([0.4, 0.35, 0.25])
    last_dist = np.array([[0.5, 0.3, 0.2], [0.3, 0.4, 0.3]])
    new = np.array([[0.3, 0.35, 0.35]])
    out = blend_swing(new, last_nat, last_dist)
    np.testing.assert_allclose(out.sum(-1), 1.0)
    assert (out[0, :, 0] < last_dist[:, 0]).all()  # the party that fell nationally falls everywhere
    assert (out[0, :, 2] > last_dist[:, 2]).all()


def test_new_party_gets_national_share_under_proportional_rule():
    last_nat = np.array([0.6, 0.4, 0.0])
    last_dist = np.array([[0.7, 0.3, 0.0], [0.5, 0.5, 0.0]])
    out = blend_swing(np.array([[0.5, 0.3, 0.2]]), last_nat, last_dist, weight_proportional=1.0)
    assert out[0, :, 2].min() > 0.15


def test_noise_keeps_simplex_and_zeroes():
    shares = np.tile(np.array([[0.5, 0.5, 0.0]]), (100, 3, 1))
    out = add_district_noise(shares, np.array([0.2, 0.2, 0.2]), np.random.default_rng(0))
    np.testing.assert_allclose(out.sum(-1), 1.0)
    assert (out[..., 2] == 0).all()
    assert out[..., 0].std() > 0.01


def test_calibrate_noise_uses_only_earlier_elections():
    parties = ["A", "B"]
    d0, d1, d2 = (pd.Timestamp(x) for x in ("2019-01-01", "2022-01-01", "2024-01-01"))
    nat = {d: pd.Series([0.5, 0.5], index=parties) for d in (d0, d1, d2)}
    dist = {
        d0: pd.DataFrame([[0.6, 0.4], [0.4, 0.6]], columns=parties),
        d1: pd.DataFrame([[0.6, 0.4], [0.4, 0.6]], columns=parties),
        d2: pd.DataFrame([[0.9, 0.1], [0.1, 0.9]], columns=parties),  # a huge shift, after the cutoff
    }
    sd = calibrate_noise(dist, nat, d2, parties, floor=0.0)
    assert np.allclose(sd, 0.0)  # d0 -> d1 had no district change


def test_calibrate_noise_skips_districts_where_a_party_had_no_list():
    # C stands everywhere in both elections except district 0 in the second: that zero is a missing
    # list, not swing noise, and must not inflate C's sd (PAN in Viseu, 2025)
    parties = ["A", "B", "C"]
    d0, d1, d2 = (pd.Timestamp(x) for x in ("2019-01-01", "2022-01-01", "2024-01-01"))
    rows = [[0.5, 0.45, 0.05], [0.45, 0.5, 0.05], [0.5, 0.45, 0.05]]
    dist = {d0: pd.DataFrame(rows, columns=parties)}
    dist[d1] = pd.DataFrame([[0.52, 0.48, 0.0]] + rows[1:], columns=parties)
    nat = {d: pd.Series([0.48, 0.47, 0.05], index=parties) for d in (d0, d1)}
    sd = calibrate_noise(dist, nat, d2, parties, floor=0.0)
    assert sd[2] < 0.1


def test_seats_from_district_shares_shapes():
    shares = np.random.default_rng(0).dirichlet(np.ones(4), size=(10, 3))
    seats = seats_from_district_shares(shares, [2, 5, 9])
    assert seats.shape == (10, 3, 4)
    assert (seats.sum(-1) == np.array([2, 5, 9])[None, :]).all()


def test_district_posterior_reproduces_last_result_when_nothing_moves():
    xr = pytest.importorskip("xarray")
    from src.evaluation.backtest import load_truth
    from src.processing.district_swing import district_posterior_from_national

    try:
        district, national, _ = load_truth()
    except Exception as e:  # results files not available
        pytest.skip(f"no results data: {e}")
    last = max(national)
    parties = ["AD", "PS", "CH", "IL", "L", "CDU", "BE", "PAN"]  # a different order on purpose
    nat = xr.DataArray(
        np.broadcast_to(national[last][parties].values, (2, 3, 8)).copy(),
        dims=("chain", "draw", "parties_complete"),
        coords={"chain": [0, 1], "draw": [0, 1, 2], "parties_complete": parties},
    )
    out = district_posterior_from_national(nat, "2029-10-07", noise=False)
    assert out.dims == ("chain", "draw", "districts", "parties_complete")
    np.testing.assert_allclose(
        out.isel(chain=1, draw=2).values, district[last][parties].values, atol=1e-9
    )
    noisy = district_posterior_from_national(nat, "2029-10-07", noise=True)
    np.testing.assert_allclose(noisy.sum("parties_complete"), 1.0)
    assert float(noisy.std(["chain", "draw"]).max()) > 0
