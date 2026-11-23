import numpy as np
import pandas as pd
import pytest

from src.models.poll_rw_model import RWConfig, build_rw_model, prepare_rw_data

PARTIES = ["PS", "CH", "IL", "BE", "CDU", "PAN", "L", "AD"]


def _toy_polls():
    rows = []
    for i, d in enumerate(pd.date_range("2019-01-01", "2019-12-31", freq="21D")):
        ch = 0 if d < pd.Timestamp("2019-06-01") else 20
        rows.append(dict(date=d, pollster="A" if i % 2 else "B", PS=350, CH=ch, IL=0, BE=80, CDU=60, PAN=30, L=0, AD=300))
    return pd.DataFrame(rows)


def _toy_results():
    return pd.DataFrame(
        [dict(election_date=pd.Timestamp("2019-10-06"), PS=1_900_000, CH=67_000, IL=0, BE=500_000,
              CDU=330_000, PAN=170_000, L=0, AD=1_650_000)]
    )


def test_prepare_rw_data_grid_and_existence():
    data = prepare_rw_data(_toy_polls(), _toy_results(), "2020-03-01", ["2019-10-06"], PARTIES, RWConfig())
    assert data.grid[-1] == pd.Timestamp("2020-03-01")
    assert (np.diff(data.grid.values).astype("timedelta64[D]").astype(int) == 7).all()
    ch = PARTIES.index("CH")
    il = PARTIES.index("IL")
    assert not data.exists[0, ch] and data.exists[-1, ch]  # Chega appears mid-2019
    assert not data.exists[:, il].any()  # IL never seen
    assert data.cycles == ["2019-10-06", "2020-03-01"]
    assert set(np.unique(data.poll_cycle)) == {0, 1}
    assert data.result_t.tolist() == [int(np.argmin(np.abs(data.grid - pd.Timestamp("2019-10-06"))))]


def test_prepare_rw_data_ignores_future_results_and_polls():
    polls = _toy_polls()
    later = pd.DataFrame([dict(date=pd.Timestamp("2021-01-01"), pollster="A", PS=1, CH=1, IL=1, BE=1, CDU=1, PAN=1, L=1, AD=1)])
    data = prepare_rw_data(pd.concat([polls, later]), _toy_results(), "2019-10-06", ["2019-10-06"], PARTIES, RWConfig())
    assert len(data.result_t) == 0  # the target's own result is not used
    assert len(data.poll_t) == len(polls[polls.date <= "2019-10-06"])


def test_build_rw_model_has_finite_logp():
    data = prepare_rw_data(_toy_polls(), _toy_results(), "2020-03-01", ["2019-10-06"], PARTIES, RWConfig())
    for persistent in (False, True):
        model = build_rw_model(data, RWConfig(persistent_bias=persistent))
        logp = model.compile_logp()(model.initial_point())
        assert np.isfinite(logp)


def test_rw_architecture_options_build_and_centre():
    import pymc as pm

    data = prepare_rw_data(_toy_polls(), _toy_results(), "2020-03-01", ["2019-10-06"], PARTIES, RWConfig())
    cfg = RWConfig(pollster_concentration=True, house_drift_sd_prior=0.05)
    model = build_rw_model(data, cfg)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert {"pollster_concentration", "house_drift", "sigma_house_drift"} <= set(model.named_vars)
    drift = pm.draw(model["house_drift"], random_seed=1)  # (cycles, pollsters, parties)
    np.testing.assert_allclose(drift.sum(axis=2), 0.0, atol=1e-9)  # zero-sum over parties
    np.testing.assert_allclose(drift.sum(axis=1), 0.0, atol=1e-9)  # and over the active pollsters
    active = np.zeros(drift.shape[:2], bool)
    active[data.poll_cycle, data.poll_pollster] = True
    assert (drift[~active] == 0).all() and np.abs(drift[active]).max() > 0
    assert pm.draw(model["pollster_concentration"], random_seed=1).shape == (len(data.pollsters),)


@pytest.mark.optional
def test_backtest_dataset_has_no_future_information():
    from src.evaluation.backtest import make_dataset

    ds = make_dataset("2024-03-10", pd.Timestamp("2024-02-11"))
    assert pd.to_datetime(ds.polls_train["date"]).max() <= pd.Timestamp("2024-02-11")
    dates = pd.to_datetime(ds.results_national["election_date"])
    assert dates.max() < pd.Timestamp("2024-03-10")
    assert pd.to_datetime(ds.results_mult_district["election_date"]).max() < pd.Timestamp("2024-03-10")


def test_rescore_leaves_out_parties_not_on_the_ballot():
    from src.evaluation.backtest import PARTIES, rescore_ballot_parties

    rows = []
    for p in PARTIES:
        absent = p in ("CH", "IL")
        rows.append(dict(model="m", election="2015-10-04", horizon=56, party=p,
                         true=0.0 if absent else 10.0, err=0.0 if absent else 2.0,
                         crps=0.0 if absent else 1.0, in50=not absent, in90=not absent,
                         seats_in90=True, seat_err=0.0 if absent else 4.0, seat_logscore=1.0))
    parties = pd.DataFrame(rows)
    df = pd.DataFrame([dict(model="m", election="2015-10-04", horizon=56, nat_mae=1.5, dist_mae=0.75,
                            nat_cov50=0.75, nat_cov90=0.75, nat_mae_big=4 / 3)])
    out = rescore_ballot_parties(df, parties).iloc[0]
    assert out["nat_mae"] == 2.0
    assert out["nat_mae_big"] == 2.0  # PS and AD only, not Chega's zero
    assert out["nat_cov90"] == 1.0
    assert out["seat_mae"] == 4.0
    assert out["dist_mae"] == 0.75 * 8 / 6
    # mixed with a newer run's rows, which carry `ran`, the older run is still rescaled
    newer = parties.assign(model="new", ran=parties["true"] > 0)
    both = pd.concat([parties, newer], ignore_index=True)
    df2 = pd.concat([df, df.assign(model="new")], ignore_index=True)
    out2 = rescore_ballot_parties(df2, both).set_index("model")
    assert out2.loc["m", "dist_mae"] == 0.75 * 8 / 6
    assert out2.loc["new", "dist_mae"] == 0.75


def test_loo_tau_rules_and_calibration_gate():
    import pandas as pd

    from src.evaluation.backtest import calibration_verdict, loo_tau

    rows = []
    for E in ["e1", "e2", "e3"]:
        for tau, crps, cov in [(0.0, 1.0, 0.6), (0.1, 0.9, 0.8), (0.2, 0.95, 0.9), (0.3, 1.1, 0.95)]:
            rows.append(dict(election=E, tau=tau, nat_crps=crps, nat_cov90=cov, nat_cov90_polled=cov, seat_cov90=0.9))
    d = pd.DataFrame(rows)
    assert set(loo_tau(d, "crps").tau_chosen) == {0.1}
    assert set(loo_tau(d, "coverage").tau_chosen) == {0.2}  # smallest tau reaching 0.85
    v = calibration_verdict(loo_tau(d, "coverage"), d[d.tau == 0])
    assert v["pass"] and v["crps_ratio"] == 0.95
    # coverage 0.8 sits on the gate's lower edge, so the CRPS pick passes too; one notch less fails
    d.loc[d.tau == 0.1, "nat_cov90_polled"] = 0.79
    assert not calibration_verdict(loo_tau(d, "crps"), d[d.tau == 0])["pass"]
