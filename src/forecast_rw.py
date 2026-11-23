"""Forecast from the random-walk national model, with blend-swing districts and a calibrated
polling error.

This is the path the November 2026 backtest points to (`src/evaluation/backtest.py`,
`--calibrate`): at 7, 28 and 56 days before past elections, the random walk with no industry
bias plus a predict-time polling error passed the calibration gate (90% intervals that cover)
with the lowest national CRPS, while the production GP needed a wider error and scored worse.
It does not replace `python -m src.main --mode train/predict`; it gives a second reading of the
same polls.

    # if the election were held on the day of the last poll
    python -m src.forecast_rw --out outputs/WIP_rw_nowcast
    # a given election day
    python -m src.forecast_rw --target 2029-10-07 --out outputs/WIP_rw_2029

Writes to --out: national_summary.csv (median and 50%/90% intervals, in %), national_draws.npy
(draws x parties, polling error included), the seat files of `simulate_seat_allocation`, and
run.json (dates, horizon, polling error, sampler diagnostics).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd

# Election-level polling error (sd of a zero-sum log shift) for the random walk without an
# industry bias, by days from the last poll to election day. Chosen leave one election out so
# that the 90% interval of the other elections covers at least 85% of parties (WIP_grid_v1 on
# the Mac, 2015-2025, rw_nobias; mean of the five held-out choices). Recalibrate with
# `python -m src.evaluation.backtest --calibrate DIR --models rw_nobias` after any model change.
# Below 7 days the 7-day value is kept: the backtest has no shorter horizon, and polls stop
# (blackout) before polling day.
RW_NOBIAS_TAU = {7: 0.13, 28: 0.16, 56: 0.20}

# The regular election falls in autumn 2029 (see CLAUDE.md); used only to build the dataset when
# no target date is given.
PLACEHOLDER_ELECTION = "2029-10-07"

BIAS_FORMS = {
    "none": {"bias_sd_prior": 1e-3},
    "per_cycle": {},
    "persistent": {"persistent_bias": True},
}


def tau_for_horizon(days: float, table: Optional[dict] = None) -> float:
    """Polling error for a forecast `days` before election day: linear between the calibrated
    horizons, flat outside them (error stops growing beyond about five weeks in the backtest)."""
    table = RW_NOBIAS_TAU if table is None else table
    hs = sorted(table)
    return float(np.interp(days, hs, [table[h] for h in hs]))


def run(
    out: str,
    target: str = "last_poll",
    as_of: Optional[str] = None,
    bias: str = "none",
    polling_error_sd: Optional[float] = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 20261123,
    seat_samples: int = 2000,
) -> dict:
    import xarray as xr

    from src.evaluation.backtest import PARTIES, _diagnostics, _sample, make_dataset
    from src.models.poll_rw_model import RWConfig, build_rw_model, national_share_draws, prepare_rw_data
    from src.processing.district_swing import district_posterior_from_national
    from src.processing.polling_error import add_polling_error
    from src.processing.seat_prediction import simulate_seat_allocation

    os.makedirs(out, exist_ok=True)
    as_of_ts = pd.Timestamp(as_of).normalize() if as_of else pd.Timestamp.today().normalize()
    election = PLACEHOLDER_ELECTION if target == "last_poll" else target
    ds = make_dataset(election, as_of_ts)
    polls = ds.polls_train
    last_poll = pd.to_datetime(polls["date"]).max().normalize()
    target_ts = last_poll if target == "last_poll" else pd.Timestamp(target).normalize()
    if target_ts < last_poll:
        raise ValueError(f"target {target_ts.date()} is before the last poll ({last_poll.date()})")
    horizon = int((target_ts - last_poll).days)
    tau = tau_for_horizon(horizon) if polling_error_sd is None else float(polling_error_sd)

    cfg = RWConfig(**BIAS_FORMS[bias])
    data = prepare_rw_data(polls, ds.results_national, target_ts, ds.historical_election_dates, PARTIES, cfg)
    idata = _sample(build_rw_model(data, cfg), draws, tune, chains, seed)
    diag = _diagnostics(idata, var_names=["sigma_rw", "sigma_house", "concentration_polls", "house"])

    rng = np.random.default_rng(seed)
    nat = add_polling_error(national_share_draws(idata, data), tau, rng)  # (chains*draws, P)
    np.save(os.path.join(out, "national_draws.npy"), nat)
    q = np.quantile(nat, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0) * 100
    summary = pd.DataFrame(
        {"median": q[2], "mean": nat.mean(0) * 100, "lo90": q[0], "hi90": q[4], "lo50": q[1], "hi50": q[3]},
        index=pd.Index(PARTIES, name="party"),
    ).round(2)
    summary.sort_values("median", ascending=False).to_csv(os.path.join(out, "national_summary.csv"))

    national = xr.DataArray(
        nat.reshape(chains, draws, len(PARTIES)),
        dims=("chain", "draw", "parties_complete"),
        coords={"chain": np.arange(chains), "draw": np.arange(draws), "parties_complete": PARTIES},
    )
    districts = district_posterior_from_national(national, target_ts if target != "last_poll" else election)
    mode = "last_poll" if target == "last_poll" else "election_day"
    _, seat_summary = simulate_seat_allocation(
        district_vote_share_posterior=districts,
        dataset=ds,
        num_samples_for_seats=min(seat_samples, chains * draws),
        pred_dir=out,
        prediction_date_mode=mode,
        hdi_prob=0.90,
    )

    info = {
        "model": "random walk (src/models/poll_rw_model.py)",
        "bias_form": bias,
        "as_of": str(as_of_ts.date()),
        "last_poll": str(last_poll.date()),
        "target": str(target_ts.date()),
        "horizon_days": horizon,
        "polling_error_sd": tau,
        "polling_error_source": "RW_NOBIAS_TAU" if polling_error_sd is None else "--polling-error-sd",
        "n_polls": int(len(polls)),
        "n_polls_since_last_election": int((pd.to_datetime(polls["date"]) > pd.Timestamp(max(ds.historical_election_dates))).sum()),
        "districts": "blend swing (50/50 proportional/uniform) from the last election, calibrated noise",
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "seed": seed,
        "cfg": asdict(cfg),
        "diagnostics": diag,
    }
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(summary.sort_values("median", ascending=False).to_string())
    if seat_summary is not None:
        print(seat_summary.to_string())
    print(json.dumps({k: info[k] for k in ("last_poll", "target", "horizon_days", "polling_error_sd", "diagnostics")}, default=str))
    return info


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (name it WIP_... until diagnostics pass)")
    ap.add_argument("--target", default="last_poll", help="'last_poll' (default) or an election date YYYY-MM-DD")
    ap.add_argument("--as-of", default=None, help="use polls published up to this date (default: today)")
    ap.add_argument("--bias", choices=sorted(BIAS_FORMS), default="none", help="industry bias form (default: none)")
    ap.add_argument("--polling-error-sd", type=float, default=None, help="override the calibrated polling error")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261123)
    a = ap.parse_args(argv)
    run(a.out, a.target, a.as_of, a.bias, a.polling_error_sd, a.draws, a.tune, a.chains, a.seed)


if __name__ == "__main__":
    main()
