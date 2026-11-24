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
(draws x parties, polling error included), the seat files of `simulate_seat_allocation`,
run.json (dates, horizon, polling error, sampler diagnostics and the acceptance gate),
diagnostics.csv (r-hat, bulk and tail ESS of every sampled and derived value, worst first) and
trace.zarr (the posterior; see `--full-trace`).

The gate (CLAUDE.md: 0 divergences, r-hat < 1.01, ESS > 400) is checked on every value the
forecast depends on, the latent weekly path included: cells of a party that does not exist yet
(and the unused first step) are left out, since no share depends on them.
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
# that the 90% interval of the other elections covers at least 85% of parties (six elections,
# 2011-2025, rw_nobias, calibrated at a179d7e; mean of the six held-out choices: 0.142, 0.150,
# 0.192). The static form calibrates to the same values within 0.01. Recalibrate with
# `python -m src.evaluation.backtest --calibrate DIR --models rw_nobias` after any model change.
# Below 7 days the 7-day value is kept: the backtest has no shorter horizon, and polls stop
# (blackout) before polling day.
RW_NOBIAS_TAU = {7: 0.14, 28: 0.15, 56: 0.19}

# The regular election falls in autumn 2029 (see CLAUDE.md); used only to build the dataset when
# no target date is given.
PLACEHOLDER_ELECTION = "2029-10-07"

BIAS_FORMS = {
    "none": {"bias_sd_prior": 1e-3},
    "per_cycle": {},
    "persistent": {"persistent_bias": True},
    # one per-party bias shared by every cycle (backtest fit rw_static)
    "static": {"persistent_bias": True, "persistent_bias_sd": 0.3, "bias_sd_prior": 1e-3},
}


def tau_for_horizon(days: float, table: Optional[dict] = None) -> float:
    """Polling error for a forecast `days` before election day: linear between the calibrated
    horizons, flat outside them (error stops growing beyond about five weeks in the backtest)."""
    table = RW_NOBIAS_TAU if table is None else table
    hs = sorted(table)
    return float(np.interp(days, hs, [table[h] for h in hs]))


GATE = {"max_rhat": 1.01, "min_ess": 400}


def full_diagnostics(idata, exists: Optional[np.ndarray] = None) -> tuple[dict, pd.DataFrame]:
    """Convergence of every posterior value: (summary dict, per-cell table sorted worst first).

    exists: (time, parties) mask of the random walk; cells of `eta` and `z_rw` where a party does
    not exist yet, and the first step (unused), are dropped: nothing depends on them. Constant
    cells (r-hat undefined) are counted, not scored.
    """
    import arviz as az

    post = idata.posterior
    rhat = az.rhat(post, method="rank")
    bulk = az.ess(post, method="bulk")
    tail = az.ess(post, method="tail")
    rows = []
    for name in post.data_vars:
        cols = {"r_hat": rhat[name], "ess_bulk": bulk[name], "ess_tail": tail[name]}
        if rhat[name].ndim == 0:
            tab = pd.DataFrame({k: [float(v)] for k, v in cols.items()}, index=[name])
        else:
            tab = pd.concat([v.to_series().rename(k) for k, v in cols.items()], axis=1)
            if name in ("eta", "z_rw") and exists is not None and rhat[name].dims == ("time", "parties"):
                keep = np.asarray(exists, bool).copy()
                if name == "z_rw":
                    keep[0] = False  # the first step is masked out of the walk
                tab = tab[keep.reshape(-1)]
            tab.index = [f"{name}[{', '.join(map(str, i if isinstance(i, tuple) else (i,)))}]" for i in tab.index]
        tab.insert(0, "variable", name)
        rows.append(tab)
    table = pd.concat(rows)
    scored = table.dropna(subset=["r_hat"])
    st = idata.sample_stats
    out = {
        "divergences": int(st["diverging"].sum()),
        "mean_step_size": float(st["step_size"].mean()) if "step_size" in st else None,
        "mean_tree_depth": float(st["tree_depth"].mean()) if "tree_depth" in st else None,
        "n_values": int(len(table)),
        "n_constant": int(len(table) - len(scored)),
        "max_rhat": float(scored["r_hat"].max()),
        "max_rhat_at": str(scored["r_hat"].idxmax()),
        "min_ess_bulk": float(scored["ess_bulk"].min()),
        "min_ess_bulk_at": str(scored["ess_bulk"].idxmin()),
        "min_ess_tail": float(scored["ess_tail"].min()),
        "min_ess_tail_at": str(scored["ess_tail"].idxmin()),
        "n_rhat_over_1.01": int((scored["r_hat"] >= GATE["max_rhat"]).sum()),
        "n_ess_under_400": int(((scored["ess_bulk"] <= GATE["min_ess"]) | (scored["ess_tail"] <= GATE["min_ess"])).sum()),
        "by_variable": {
            v: {"max_rhat": round(float(g["r_hat"].max()), 4), "min_ess_bulk": round(float(g["ess_bulk"].min())),
                "min_ess_tail": round(float(g["ess_tail"].min()))}
            for v, g in scored.groupby("variable")
        },
    }
    out["passes_gate"] = bool(
        out["divergences"] == 0
        and out["max_rhat"] < GATE["max_rhat"]
        and out["min_ess_bulk"] > GATE["min_ess"]
        and out["min_ess_tail"] > GATE["min_ess"]
    )
    order = scored.assign(_k=-scored["r_hat"]).sort_values(["_k", "ess_bulk"]).index
    table = pd.concat([table.loc[order], table.drop(index=order)]).round(4)
    return out, table


def _light_posterior(idata, data, keep_weeks: int = 52, every: int = 4):
    """The posterior without the weekly innovations, and the latent path only at the last
    `keep_weeks` weeks, every `every`-th week before them and the election dates (float32)."""
    post = idata.posterior
    T = post.sizes["time"]
    weeks = set(range(max(T - keep_weeks, 0), T)) | set(range(T - 1, -1, -every)) | set(int(t) for t in data.result_t)
    light = post.drop_vars([v for v in ("z_rw", "eta") if v in post])
    if "time" in light.coords and not any("time" in light[v].dims for v in light.data_vars):
        light = light.drop_vars("time")  # else the kept weeks would be re-aligned to every week
    if "eta" in post:
        light = light.assign(eta=post["eta"].isel(time=sorted(weeks)).astype("float32"))
    return light


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
    full_trace: bool = False,
    kalman: bool = False,
) -> dict:
    import xarray as xr

    from src.evaluation.backtest import PARTIES, _sample, make_dataset
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
    if kalman:  # the weekly path integrated out (src/models/poll_rw_kalman.py): no eta or z_rw
        from src.models.poll_rw_kalman import fit_rw_kalman

        idata, nat_model, _ = fit_rw_kalman(data, cfg, draws, tune, chains, seed)
    else:
        idata = _sample(build_rw_model(data, cfg), draws, tune, chains, seed)
        nat_model = national_share_draws(idata, data)
    diag, diag_table = full_diagnostics(idata, data.exists)
    diag_table.to_csv(os.path.join(out, "diagnostics.csv"), index_label="value")
    if full_trace or kalman:
        idata.to_zarr(os.path.join(out, "trace.zarr"))
    else:
        import arviz as az

        az.InferenceData(posterior=_light_posterior(idata, data), sample_stats=idata.sample_stats).to_zarr(
            os.path.join(out, "trace.zarr")
        )

    rng = np.random.default_rng(seed)
    nat = add_polling_error(nat_model, tau, rng)  # (chains*draws, P)
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
        "model": "random walk, Kalman filter (src/models/poll_rw_kalman.py)" if kalman else "random walk (src/models/poll_rw_model.py)",
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
        "gate": GATE,
        "trace": "full" if (full_trace or kalman) else "light: no weekly innovations; latent path for the last 52 weeks, every 4th before and election dates",
    }
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(summary.sort_values("median", ascending=False).to_string())
    if seat_summary is not None:
        print(seat_summary.to_string())
    brief = {k: v for k, v in diag.items() if k != "by_variable"}
    print(json.dumps({**{k: info[k] for k in ("last_poll", "target", "horizon_days", "polling_error_sd")}, "diagnostics": brief}, default=str))
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
    ap.add_argument("--full-trace", action="store_true", help="save every variable in trace.zarr (about 1 GB at 4 x 2000)")
    ap.add_argument("--kalman", action="store_true", help="integrate the weekly path out with a Kalman filter (Gaussian polls)")
    a = ap.parse_args(argv)
    run(a.out, a.target, a.as_of, a.bias, a.polling_error_sd, a.draws, a.tune, a.chains, a.seed, full_trace=a.full_trace, kalman=a.kalman)


if __name__ == "__main__":
    main()
