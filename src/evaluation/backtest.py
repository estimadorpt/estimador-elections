"""Out-of-sample backtest of the legislative model on past Portuguese elections.

For a target election E and a horizon h (days), the fit sees only what was known on E - h:
polls published up to that day and the results of elections before E. It then forecasts E's
national shares, district shares and district seats, and is scored against E's result.

This replaces `--cutoff-date` / `--mode cross-validate` for retrodiction: both load every
historical result, so the target election's own result is in the fit.

Seats are scored on the 20 domestic districts with the 2024 seat map (226 seats) for every
election, against D'Hondt applied to E's actual district votes with the same map. That isolates
the error of the vote model from changes in the seat map; emigration seats are left out.

Usage
-----
    python -m src.evaluation.backtest --models rw gp_current poll_average last_result \
        --elections 2015-10-04 2019-10-06 2022-01-30 2024-03-10 2025-05-18 \
        --horizons 56 28 7 --draws 500 --tune 500 --out outputs/backtest/WIP_run
    # 2011-06-05 is also a valid target (2009 is its last result); it is not in the default list,
    # so runs stay comparable with the five-election grids
    python -m src.evaluation.backtest --district-oracle   # swing rules given the true national result
    python -m src.evaluation.backtest --rescore-districts DIR   # rescore blend variants from saved draws
    python -m src.evaluation.backtest --summarise outputs/backtest/WIP_run
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.processing.district_swing import (
    add_district_noise,
    blend_swing,
    calibrate_noise,
    dhondt_draws,
    seats_from_district_shares,
)

PARTIES = ["PS", "CH", "IL", "BE", "CDU", "PAN", "L", "AD"]
ALL_ELECTIONS = ["2009-09-27", "2011-06-05", "2015-10-04", "2019-10-06", "2022-01-30", "2024-03-10", "2025-05-18"]
BACKTEST_ELECTIONS = ["2015-10-04", "2019-10-06", "2022-01-30", "2024-03-10", "2025-05-18"]


# ----------------------------------------------------------------------------------------
# Truth
# ----------------------------------------------------------------------------------------

_TRUTH_CACHE: Dict = {}


def load_truth() -> Tuple[Dict[pd.Timestamp, pd.DataFrame], Dict[pd.Timestamp, pd.Series], pd.Series]:
    """District shares (D x P), national shares (P) per election, and the 2024 seat map."""
    if _TRUTH_CACHE:
        return _TRUTH_CACHE["district"], _TRUTH_CACHE["national"], _TRUTH_CACHE["seats"]
    from src.data.flexible_loaders import load_election_results_flexible

    with contextlib.redirect_stdout(io.StringIO()):
        d = load_election_results_flexible(ALL_ELECTIONS, PARTIES, aggregation_level="district")
    d["election_date"] = pd.to_datetime(d["election_date"])
    seats = pd.read_csv(os.path.join(DATA_DIR, "legislativas_2024_circulos_deputados.csv"), encoding="utf-8-sig")
    seats = seats.set_index("Circulo")["Deputados"].sort_index()
    district, national = {}, {}
    for t, g in d.groupby("election_date"):
        x = g.set_index("Circulo")[PARTIES].astype(float).reindex(seats.index)
        national[t] = x.sum(0) / x.values.sum()
        district[t] = x.div(x.sum(1), axis=0)
    _TRUTH_CACHE.update(district=district, national=national, seats=seats)
    return district, national, seats


# ----------------------------------------------------------------------------------------
# Information set as of E - h
# ----------------------------------------------------------------------------------------


def make_dataset(target: str, as_of: pd.Timestamp):
    """ElectionDataset that knows only what was public on `as_of`."""
    from src.data.dataset import ElectionDataset

    # 2009 is not a production history date, but its results let 2011 be a backtest target
    history = sorted(set(ElectionDataset.historical_election_dates) | {"2009-09-27"}, reverse=True)
    past = [d for d in history if pd.Timestamp(d) < pd.Timestamp(target)]

    class AsOfDataset(ElectionDataset):
        historical_election_dates = sorted(past, reverse=True)

        def _load_polls(self):
            polls = super()._load_polls()
            return polls[pd.to_datetime(polls["date"]) <= as_of].reset_index(drop=True)

    with contextlib.redirect_stdout(io.StringIO()):
        ds = AsOfDataset(election_date=target, baseline_timescales=[365], election_timescales=[14, 60])
    return ds


# ----------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------


def _sample(model, draws, tune, chains, seed, target_accept=0.9, var_names=None):
    import pymc as pm

    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler="numpyro",
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            var_names=var_names,
        )


def _diagnostics(idata, var_names=None) -> Dict:
    import arviz as az

    st = idata.sample_stats
    out = {"divergences": int(st["diverging"].sum())}
    for key, name in (("step_size", "step_size"), ("tree_depth", "tree_depth"), ("n_steps", "n_steps")):
        if key in st:
            out[f"mean_{name}"] = float(st[key].mean())
    try:
        s = az.summary(idata, var_names=var_names, kind="diagnostics")
        out["max_rhat"] = float(s["r_hat"].max())
        out["min_ess_bulk"] = float(s["ess_bulk"].min())
    except Exception as e:  # pragma: no cover
        out["diag_error"] = str(e)
    return out


def run_gp_current(
    ds, target, draws, tune, chains, seed, mask_unreported_parties=False, mask_absent_parties=False, **model_kwargs
) -> Dict:
    """The production DynamicGPElectionModel (both masks off reproduces it as of 22 Nov 2026).

    `model_kwargs` switch on the experimental options (pollster_concentration, undecided_variance).
    """
    from scipy.special import softmax
    from src.models.dynamic_gp_election_model import DynamicGPElectionModel

    with contextlib.redirect_stdout(io.StringIO()):
        m = DynamicGPElectionModel(
            ds,
            mask_unreported_parties=mask_unreported_parties,
            mask_absent_parties=mask_absent_parties,
            **model_kwargs,
        )
        model = m.build_model()
        exists_log = m._existence_log_mask() if mask_absent_parties else None
    free = [v.name for v in model.free_RVs]
    keep = free + ["national_trend_pt", "static_district_offset", "house_effects", "poll_bias"]
    idata = _sample(model, draws, tune, chains, seed, target_accept=0.95, var_names=keep)
    cal = pd.to_datetime(m.coords["calendar_time"])
    t = int(np.argmin(np.abs(cal - pd.Timestamp(target))))
    trend = idata.posterior["national_trend_pt"].isel(calendar_time=t).stack(s=("chain", "draw")).transpose("s", "parties_complete")
    parties = list(trend["parties_complete"].values)
    trend = trend.values
    if exists_log is not None:
        trend = trend + exists_log[t][None, :]
    order = [parties.index(p) for p in PARTIES]
    nat = softmax(trend, axis=1)[:, order]
    off = idata.posterior["static_district_offset"].stack(s=("chain", "draw")).transpose("s", "districts", "parties_complete")
    districts = list(off["districts"].values)
    dist = softmax(trend[:, None, :] + off.values, axis=2)[:, :, order]
    diag = _diagnostics(idata, var_names=[v for v in free if "raw" not in v])
    pb = idata.posterior["poll_bias"].mean(("chain", "draw")).to_series().reindex(PARTIES).round(3).to_dict()
    extra = {"poll_bias": pb}
    post = idata.posterior
    for name in (
        "concentration_polls", "sigma_pollster_concentration", "undecided_allocation_concentration", "sigma_poll_bias_cycle"
    ):
        if name in post:
            extra[name] = round(float(post[name].mean()), 3)
    if "pollster_concentration_raw" in post:
        conc = post["concentration_polls"] * np.exp(post["sigma_pollster_concentration"] * post["pollster_concentration_raw"])
        extra["pollster_concentration"] = conc.median(("chain", "draw")).to_series().round(1).to_dict()
    return {"national": nat, "district": dist, "districts": districts, "diag": diag, "extra": extra}


def run_rw(
    ds, target, draws, tune, chains, seed, cfg_overrides: Optional[Dict] = None, future_polls: Optional[pd.DataFrame] = None
) -> Dict:
    from src.models.poll_rw_model import RWConfig, build_rw_model, national_share_draws, prepare_rw_data

    cfg = RWConfig(**(cfg_overrides or {}))
    data = prepare_rw_data(
        ds.polls_train, ds.results_national, target, ds.historical_election_dates, PARTIES, cfg
    )
    model = build_rw_model(data, cfg)
    idata = _sample(model, draws, tune, chains, seed, target_accept=0.9)
    nat = national_share_draws(idata, data)
    checked = ["sigma_rw", "sigma_house", "sigma_bias", "concentration_polls", "house", "bias"]
    if cfg.dynamics != "rw":
        checked += ["short_timescale", "sigma_short"]
    diag = _diagnostics(idata, var_names=checked)
    bias = idata.posterior["bias"].mean(("chain", "draw")).to_pandas()
    extra = {
        "sigma_rw": idata.posterior["sigma_rw"].mean(("chain", "draw")).to_series().round(4).to_dict(),
        "sigma_bias": float(idata.posterior["sigma_bias"].mean()),
        "concentration_polls": float(idata.posterior["concentration_polls"].mean()),
        "bias_by_cycle": {c: bias.loc[c].round(3).to_dict() for c in bias.index},
        "cfg": asdict(cfg),
    }
    summary_extra = {}
    if cfg.dynamics != "rw":
        tau = idata.posterior["short_timescale"].values.ravel()
        q = np.quantile(tau, [0.05, 0.5, 0.95])
        prior = np.exp(np.log(cfg.short_timescale_weeks) + cfg.short_timescale_logsd * np.array([-1.645, 0.0, 1.645]))
        extra["short_timescale_weeks"] = {"posterior_q05_q50_q95": q.round(2).tolist(), "prior_q05_q50_q95": prior.round(2).tolist()}
        extra["sigma_short"] = idata.posterior["sigma_short"].mean(("chain", "draw")).to_series().round(4).to_dict()
        summary_extra.update(short_timescale_q50=float(q[1]), short_timescale_q05=float(q[0]), short_timescale_q95=float(q[2]),
                             sigma_short_mean=float(idata.posterior["sigma_short"].mean()))
    summary_extra.update(future_poll_scores(idata, data, future_polls, seed))
    return {"national": nat, "district": None, "diag": diag, "extra": extra, "summary_extra": summary_extra}


def load_future_polls(target: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Polls published after the cutoff and before election day (exit polls excluded)."""
    ds_full = make_dataset(target, pd.Timestamp(target))
    polls = ds_full.polls_train
    d = pd.to_datetime(polls["date"])
    return polls[(d > as_of) & (d < pd.Timestamp(target))].reset_index(drop=True)


def future_poll_scores(idata, data, future: Optional[pd.DataFrame], seed: int = 0, n_draws: int = 400) -> Dict:
    """Score a fit on the polls published after its cutoff: a test of the dynamics.

    For each such poll, the predictive is the latent opinion at the poll's week plus the
    pollster's house effect (zero for a pollster the fit has not seen) and the cycle bias,
    with Dirichlet-multinomial poll noise at the fitted concentration and the poll's size,
    over the parties the poll reported. post_poll_crps (pp) is a proper score; post_poll_mae
    (pp) compares the latent mean (no noise) with the poll.
    """
    from src.models.poll_rw_model import _to_grid_index

    if future is None or len(future) == 0:
        return {"post_n_polls": 0}
    rng = np.random.default_rng(seed)
    post = idata.posterior.stack(s=("chain", "draw"))
    S = post.sizes["s"]
    take = rng.choice(S, size=min(n_draws, S), replace=False)
    eta = post["eta"].transpose("s", "time", "parties").values[take]
    house = post["house"].transpose("s", "pollsters", "parties").values[take]
    bias = post["bias"].isel(cycles=-1).transpose("s", "parties").values[take]
    conc = post["concentration_polls"].values[take]
    t_idx = _to_grid_index(future["date"], data.grid)
    counts = future[list(data.parties)].to_numpy(float)
    crps, mae = [], []
    for i in range(len(future)):
        rep = (counts[i] > 0) & data.exists[t_idx[i]]
        if rep.sum() < 2:
            continue
        logit = eta[:, t_idx[i], :] + bias
        if future["pollster"].iloc[i] in data.pollsters:
            logit = logit + house[:, data.pollsters.index(future["pollster"].iloc[i]), :]
        logit = np.where(rep[None, :], logit, -np.inf)
        p = np.exp(logit - logit.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        n = int(counts[i, rep].sum())
        g = rng.gamma(np.clip(conc[:, None] * p[:, rep], 1e-8, None))
        sim = rng.multinomial(n, g / g.sum(1, keepdims=True)) / n
        obs = counts[i, rep] / n
        for j in range(rep.sum()):
            crps.append(crps_sample(sim[:, j] * 100, obs[j] * 100))
        mae.extend(np.abs(p[:, rep].mean(0) - obs) * 100)
    return {"post_n_polls": int(len(future)), "post_poll_crps": float(np.mean(crps)), "post_poll_mae": float(np.mean(mae))}


def run_poll_average(ds, target, window_days=21, min_polls=4) -> Dict:
    """Mean of the polls in the last `window_days` before the cutoff (at least `min_polls`)."""
    polls = ds.polls_train.sort_values("date")
    last = pd.to_datetime(polls["date"]).max()
    recent = polls[pd.to_datetime(polls["date"]) > last - pd.Timedelta(days=window_days)]
    if len(recent) < min_polls:
        recent = polls.tail(min_polls)
    shares = recent[PARTIES].div(recent[PARTIES].sum(1), axis=0).mean(0).values
    return {"national": shares[None, :], "district": None, "diag": {}, "extra": {"n_polls": int(len(recent))}}


def run_latest_per_firm(ds, target, window_days=45, half_life_days=10.0) -> Dict:
    """A poll-of-polls baseline: each firm's latest poll in the last `window_days`, weighted by
    sqrt(sample size) and by recency (weight halves every `half_life_days`), so a firm that polls
    every week counts once."""
    polls = ds.polls_train.copy()
    polls["date"] = pd.to_datetime(polls["date"])
    last = polls["date"].max()
    recent = polls[polls["date"] > last - pd.Timedelta(days=window_days)]
    latest = recent.sort_values("date").groupby("pollster").tail(1)
    age = (last - latest["date"]).dt.days.to_numpy(float)
    w = np.sqrt(latest["sample_size"].to_numpy(float)) * 0.5 ** (age / half_life_days)
    shares = latest[PARTIES].div(latest[PARTIES].sum(1), axis=0).to_numpy()
    nat = (w[:, None] * shares).sum(0) / w.sum()
    return {"national": nat[None, :], "district": None, "diag": {}, "extra": {"n_firms": int(len(latest))}}


def run_last_result(ds, target) -> Dict:
    district, national, _ = load_truth()
    last = max(t for t in national if t < pd.Timestamp(target))
    return {"national": national[last].values[None, :], "district": district[last].values[None, :, :], "diag": {}, "extra": {}}


# ----------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------


def crps_sample(x: np.ndarray, y: float) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 1:
        return float(abs(x[0] - y))
    term1 = np.mean(np.abs(x - y))
    # E|X - X'| via sorted-sample identity
    i = np.arange(1, n + 1)
    term2 = 2.0 * np.sum((2 * i - n - 1) * x) / (n * n)
    return float(term1 - 0.5 * term2)


def score(pred_nat: np.ndarray, pred_dist: np.ndarray, target: str, seats_map: pd.Series) -> Tuple[Dict, pd.DataFrame]:
    district, national, _ = load_truth()
    E = pd.Timestamp(target)
    y = national[E].values
    yd = district[E].values
    S = pred_nat.shape[0]
    mean = pred_nat.mean(0)
    # Parties that were not on the ballot (Chega and IL in 2015) are left out of every metric:
    # a forecast of ~0 for them is right by construction and would only dilute the averages.
    ran = y > 0
    rows = []
    for j, p in enumerate(PARTIES):
        lo50, hi50 = np.quantile(pred_nat[:, j], [0.25, 0.75])
        lo90, hi90 = np.quantile(pred_nat[:, j], [0.05, 0.95])
        rows.append(
            dict(
                party=p,
                ran=bool(ran[j]),
                true=y[j] * 100,
                mean=mean[j] * 100,
                err=(mean[j] - y[j]) * 100,
                lo90=lo90 * 100,
                hi90=hi90 * 100,
                in50=bool(lo50 <= y[j] <= hi50) if S > 1 else np.nan,
                in90=bool(lo90 <= y[j] <= hi90) if S > 1 else np.nan,
                # a party on the ballot that no poll had reported by the cutoff is forecast at ~0
                # with a [0, 0] interval; no polling-error size can cover it
                polled=bool(np.median(pred_nat[:, j]) >= 1e-3),
                crps=crps_sample(pred_nat[:, j] * 100, y[j] * 100),
            )
        )
    party = pd.DataFrame(rows)
    true_seats_d = seats_from_district_shares(yd[None], seats_map.values)[0]  # (D,P)
    seat_draws_d = seats_from_district_shares(pred_dist, seats_map.values)  # (S,D,P)
    tot = seat_draws_d.sum(1)
    true_tot = true_seats_d.sum(0)
    party["true_seats"] = true_tot
    party["mean_seats"] = tot.mean(0)
    party["seat_err"] = tot.mean(0) - true_tot
    lo, hi = np.quantile(tot, [0.05, 0.95], axis=0)
    party["seats_lo90"], party["seats_hi90"] = lo, hi
    party["seats_in90"] = (lo <= true_tot) & (true_tot <= hi) if S > 1 else np.nan
    p_exact = (tot == true_tot[None, :]).mean(0)
    party["seat_logscore"] = -np.log(np.clip(p_exact, 1 / (2 * S), None))
    # ranked probability score of each party's seat total: sum over k of (P(seats <= k) - 1[true <= k])^2
    ks = np.arange(int(max(tot.max(), true_tot.max())) + 1)
    cdf = (tot[:, None, :] <= ks[None, :, None]).mean(0)  # (K, P)
    obs = (true_tot[None, :] <= ks[:, None]).astype(float)
    party["seat_rps"] = ((cdf - obs) ** 2).sum(0)
    winner_true = int(np.argmax(true_tot))
    winners = np.argmax(tot + 1e-3 * pred_dist.sum(1), axis=1)  # tie-break by votes
    vote_winner_true = int(np.argmax(y))
    r = party["ran"].values
    err = np.abs(mean - y)[ran]
    big = [PARTIES.index(p) for p in ["PS", "AD", "CH"] if ran[PARTIES.index(p)]]
    summary = dict(
        nat_mae=float(err.mean() * 100),
        nat_mae_big=float(np.abs(mean - y)[big].mean() * 100),
        nat_rmse=float(np.sqrt((err**2).mean()) * 100),
        nat_max_err=float(err.max() * 100),
        nat_crps=float(party.loc[r, "crps"].mean()),
        nat_cov50=float(party.loc[r, "in50"].mean()) if S > 1 else np.nan,
        nat_cov90=float(party.loc[r, "in90"].mean()) if S > 1 else np.nan,
        nat_cov90_polled=float(party.loc[r & party["polled"].values, "in90"].mean()) if S > 1 else np.nan,
        dist_mae=float(np.abs(pred_dist.mean(0) - yd)[:, ran].mean() * 100),
        seat_mae=float(np.abs(tot.mean(0) - true_tot)[ran].mean()),
        seat_abs_err_total=float(np.abs(tot.mean(0) - true_tot).sum() / 2),
        dist_seats_misallocated=float(np.abs(seat_draws_d - true_seats_d[None]).sum((1, 2)).mean() / 2),
        seat_cov90=float(party.loc[r, "seats_in90"].mean()) if S > 1 else np.nan,
        seat_logscore=float(party.loc[r, "seat_logscore"].mean()),
        p_seat_winner=float((winners == winner_true).mean()),
        p_vote_winner=float((np.argmax(pred_nat, 1) == vote_winner_true).mean()),
        seat_rps=float(party.loc[r, "seat_rps"].mean()),
        winner_logscore=float(-np.log(max((winners == winner_true).mean(), 1 / (2 * S)))),
    )
    return summary, party


# ----------------------------------------------------------------------------------------
# Districts for models that only forecast national shares
# ----------------------------------------------------------------------------------------


def districts_from_national(nat: np.ndarray, target: str, noise: bool, seed: int = 0) -> np.ndarray:
    district, national, _ = load_truth()
    E = pd.Timestamp(target)
    last = max(t for t in national if t < E)
    shares = blend_swing(nat, national[last].values, district[last].values)
    if noise and nat.shape[0] > 1:
        sd = calibrate_noise(district, national, E, PARTIES)
        shares = add_district_noise(shares, sd, np.random.default_rng(seed))
    return shares


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

# fit name -> (runner, runner kwargs, {scored variant name: district rule})
# district rules: "model" = the model's own district shares; "blend" = blend swing from the last
# election without noise; "blend_noise" = blend swing plus calibrated district noise.
FITS = {
    "gp_current": ("gp_current", {}, {"gp_current": "model", "gp_current_blend": "blend_noise"}),
    "gp_masked": (
        "gp_current",
        {"mask_unreported_parties": True, "mask_absent_parties": True},
        {"gp_masked": "model", "gp_masked_blend": "blend_noise"},
    ),
    # experiments from the review (Nov 2026), each the masked GP plus one option
    "gp_pconc": (
        "gp_current",
        {"mask_unreported_parties": True, "mask_absent_parties": True, "pollster_concentration": True},
        {"gp_pconc": "model", "gp_pconc_blend": "blend_noise"},
    ),
    "gp_undecided": (
        "gp_current",
        {"mask_unreported_parties": True, "mask_absent_parties": True, "undecided_variance": True},
        {"gp_undecided": "model", "gp_undecided_blend": "blend_noise"},
    ),
    # industry bias = tight mean (sd 0.05) + per-cycle deviation with learned sd (rw_hybrid's form)
    "gp_bias_cycle": (
        "gp_current",
        {"mask_unreported_parties": True, "mask_absent_parties": True, "poll_bias_per_cycle": True},
        {"gp_bias_cycle": "model", "gp_bias_cycle_blend": "blend_noise"},
    ),
    # no industry bias at all (house effects only), like rw_nobias
    "gp_nobias": (
        "gp_current",
        {"mask_unreported_parties": True, "mask_absent_parties": True, "use_poll_bias": False},
        {"gp_nobias": "model", "gp_nobias_blend": "blend_noise"},
    ),
    # per-cycle deviation only, nothing carried over: no party-specific correction
    "gp_bias_cycle0": (
        "gp_current",
        {"mask_unreported_parties": True, "mask_absent_parties": True, "poll_bias_per_cycle": True, "poll_bias_mean_sd": 0.0},
        {"gp_bias_cycle0": "model", "gp_bias_cycle0_blend": "blend_noise"},
    ),
    "rw": ("rw", {}, {"rw": "blend_noise", "rw_no_district_noise": "blend"}),
    "rw_persistent": ("rw", {"persistent_bias": True}, {"rw_persistent": "blend_noise"}),
    "rw_nobias": ("rw", {"bias_sd_prior": 1e-3}, {"rw_nobias": "blend_noise"}),
    # one industry bias for every cycle, like the GP model's poll_bias (per-cycle spread ~0)
    "rw_static": (
        "rw",
        {"persistent_bias": True, "persistent_bias_sd": 0.3, "bias_sd_prior": 1e-3},
        {"rw_static": "blend_noise"},
    ),
    # a learned mean bias that carries forward, plus a per-cycle deviation around it
    "rw_hybrid": ("rw", {"persistent_bias": True, "persistent_bias_sd": 0.3}, {"rw_hybrid": "blend_noise"}),
    # opinion dynamics only (Nov 2026): bias form "none" (the live forecast's), the random walk
    # alone or plus a short mean-reverting term; dyn_rw is rw_nobias rerun alongside them
    "dyn_rw": ("rw", {"bias_sd_prior": 1e-3}, {"dyn_rw": "blend_noise"}),
    "dyn_ou": ("rw", {"bias_sd_prior": 1e-3, "dynamics": "rw_ou"}, {"dyn_ou": "blend_noise"}),
    "dyn_m32": ("rw", {"bias_sd_prior": 1e-3, "dynamics": "rw_m32"}, {"dyn_m32": "blend_noise"}),
    "poll_average": ("poll_average", {}, {"poll_average": "blend"}),
    "last_result": ("last_result", {}, {"last_result": "model"}),
    "latest_per_firm": ("latest_per_firm", {}, {"latest_per_firm": "blend"}),
}


def run_one(fit_name: str, target: str, horizon: int, draws: int, tune: int, chains: int, seed: int, out_dir: str):
    runner, kwargs, variants = FITS[fit_name]
    as_of = pd.Timestamp(target) - pd.Timedelta(days=horizon)
    tags = {v: f"{v}__{target}__h{horizon:03d}" for v in variants}
    if all(os.path.exists(os.path.join(out_dir, t + ".json")) for t in tags.values()):
        print(f"skip {fit_name} {target} h{horizon} (done)")
        return
    t0 = time.time()
    ds = make_dataset(target, as_of)
    if runner == "gp_current":
        res = run_gp_current(ds, target, draws, tune, chains, seed, **kwargs)
    elif runner == "rw":
        res = run_rw(ds, target, draws, tune, chains, seed, kwargs, future_polls=load_future_polls(target, as_of))
    elif runner == "poll_average":
        res = run_poll_average(ds, target)
    elif runner == "last_result":
        res = run_last_result(ds, target)
    elif runner == "latest_per_firm":
        res = run_latest_per_firm(ds, target)
    else:
        raise ValueError(runner)
    fit_seconds = round(time.time() - t0, 1)
    _, _, seats_map = load_truth()
    for variant, rule in variants.items():
        tag = tags[variant]
        if rule == "model" and res["district"] is not None:
            dist = res["district"]
            if "districts" in res:  # reorder model districts to the seat map
                idx = [res["districts"].index(d) for d in seats_map.index]
                dist = dist[:, idx, :]
        else:
            dist = districts_from_national(res["national"], target, noise=(rule == "blend_noise"), seed=seed)
        summary, party = score(res["national"], dist, target, seats_map)
        summary.update(
            model=variant,
            fit=fit_name,
            district_rule=rule,
            election=target,
            horizon=horizon,
            as_of=str(as_of.date()),
            n_polls=int(len(ds.polls_train)),
            last_poll=str(pd.to_datetime(ds.polls_train["date"]).max().date()),
            seconds=fit_seconds,
            **{f"diag_{k}": v for k, v in res["diag"].items()},
            **res.get("summary_extra", {}),
        )
        party.insert(0, "model", variant)
        party.insert(1, "election", target)
        party.insert(2, "horizon", horizon)
        party.to_csv(os.path.join(out_dir, tag + "__parties.csv"), index=False)
        with open(os.path.join(out_dir, tag + ".json"), "w") as f:
            json.dump({"summary": summary, "extra": res["extra"]}, f, indent=1, default=str)
        print(f"done {tag}: nat_mae={summary['nat_mae']:.2f} seat_mae={summary['seat_mae']:.2f} ({fit_seconds}s)")
    np.save(os.path.join(out_dir, f"{fit_name}__{target}__h{horizon:03d}__national.npy"), res["national"].astype(np.float32))
    if res["district"] is not None and res["district"].shape[0] > 1:  # the model's own districts, for rescoring
        dist = res["district"]
        if "districts" in res:
            dist = dist[:, [res["districts"].index(d) for d in seats_map.index], :]
        np.save(os.path.join(out_dir, f"{fit_name}__{target}__h{horizon:03d}__district.npy"), dist.astype(np.float32))


def rescore_ballot_parties(df: pd.DataFrame, parties: pd.DataFrame) -> pd.DataFrame:
    """Recompute party-level metrics over the parties on the ballot only (true share > 0).

    Runs scored before this rule averaged over all 8 parties, which in 2015 included Chega and
    IL (not on the ballot); the PS/AD/CH error then also counted Chega's zero. The per-party rows hold everything needed; district MAE is rescaled
    by 8 / parties on the ballot, exact when the absent parties were forecast at ~0.
    """
    out = df.copy()
    keys = ["model", "election", "horizon"]
    for i, row in out.iterrows():
        sel = parties
        for k in keys:
            sel = sel[sel[k].astype(str) == str(row[k])]
        sel = sel[sel["true"] > 0]
        if sel.empty:
            continue
        n_all = len(PARTIES)
        err = sel["err"].abs()
        out.loc[i, "nat_mae"] = err.mean()
        big = sel[sel["party"].isin(["PS", "AD", "CH"])]
        if not big.empty:
            out.loc[i, "nat_mae_big"] = big["err"].abs().mean()
        out.loc[i, "nat_rmse"] = float(np.sqrt((sel["err"] ** 2).mean()))
        out.loc[i, "nat_max_err"] = err.max()
        out.loc[i, "nat_crps"] = sel["crps"].mean()
        for c_out, c_in in [("nat_cov50", "in50"), ("nat_cov90", "in90"), ("seat_cov90", "seats_in90")]:
            if c_in in sel and sel[c_in].notna().any():
                out.loc[i, c_out] = sel[c_in].astype(float).mean()
        out.loc[i, "seat_mae"] = sel["seat_err"].abs().mean()
        out.loc[i, "seat_logscore"] = sel["seat_logscore"].mean()
        scored_all_parties = "ran" not in sel.columns or sel["ran"].isna().all()  # an older run
        if scored_all_parties and len(sel) < n_all:
            out.loc[i, "dist_mae"] = row["dist_mae"] * n_all / len(sel)
    return out


def calibrate_polling_error(
    out_dir: str,
    models: Sequence[str] = ("gp_masked", "rw"),
    horizon: int = 56,
    taus: Sequence[float] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6),
    seed: int = 1,
) -> pd.DataFrame:
    """Score saved national draws with an added election-level polling error of each size.

    Reads `<model>__<election>__h<horizon>__national.npy`, adds `add_polling_error(tau)`, derives
    districts by blend swing with noise and rescores. Prints the mean over elections per tau and
    a leave-one-election-out choice (tau picked by CRPS on the other elections), with a pass or
    fail against `CALIBRATION_GATE`.
    """
    from src.processing.polling_error import add_polling_error

    district, national, seats_map = load_truth()
    rng = np.random.default_rng(seed)
    rows = []
    for model in models:
        for f in sorted(os.listdir(out_dir)):
            if not (f.startswith(model + "__") and f.endswith(f"__h{horizon:03d}__national.npy")):
                continue
            E = f.split("__")[1]
            nat0 = np.load(os.path.join(out_dir, f))
            if nat0.shape[0] == 1:  # a point forecast (poll average, last result): give it draws
                nat0 = np.repeat(nat0, 2000, axis=0)
            for tau in taus:
                nat = add_polling_error(nat0, tau, rng)
                summ, _ = score(nat, districts_from_national(nat, E, noise=True), E, seats_map)
                rows.append(dict(model=model, election=E, horizon=horizon, tau=tau, **summ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(os.path.join(out_dir, f"polling_error_h{horizon:03d}.csv"), index=False)
    cols = ["nat_crps", "nat_cov50", "nat_cov90", "nat_cov90_polled", "seat_cov90", "seat_logscore", "seat_rps"]
    print(df.groupby(["model", "tau"])[cols].mean().round(3).to_string())
    for model, d in df.groupby("model"):
        for rule in ("crps", "coverage"):
            o = loo_tau(d, rule)
            verdict = calibration_verdict(o, d[d.tau == 0])
            print(
                f"{model} h{horizon} tau by {rule}, left out one election at a time: {list(o.tau_chosen)}; "
                + ", ".join(f"{k} {v}" for k, v in verdict.items())
            )
    return df


def loo_tau(d: pd.DataFrame, rule: str = "crps", target_cov: float = 0.85) -> pd.DataFrame:
    """For each election, the tau chosen on the other elections, and that election's row at it.

    `crps`: the tau with the lowest mean national CRPS. `coverage`: the smallest tau whose mean 90%
    coverage (polled parties) reaches `target_cov`, else the widest tau.
    """
    picks = []
    for E in d.election.unique():
        rest = d[d.election != E].groupby("tau")
        if rule == "crps":
            t = rest.nat_crps.mean().idxmin()
        else:
            cov = rest.nat_cov90_polled.mean()
            ok = cov[cov >= target_cov]
            t = ok.index.min() if len(ok) else cov.index.max()
        picks.append(d[(d.election == E) & (d.tau == t)].assign(tau_chosen=t))
    return pd.concat(picks)


# Pass marks for a polling-error calibration, chosen leave-one-election-out: the 90% national
# interval should hold the result 80-97% of the time, the 90% seat interval at least 85%, and the
# added error should cost at most 3% of CRPS against no added error.
# National coverage is counted over polled parties (see `score`): an unpolled entrant is a
# separate defect that no error size fixes. Plain `nat_cov90` is printed beside it.
CALIBRATION_GATE = {"nat_cov90_polled": (0.80, 0.97), "seat_cov90": (0.85, 1.0), "crps_ratio": (0.0, 1.03)}


def calibration_verdict(chosen: pd.DataFrame, untouched: pd.DataFrame) -> Dict[str, object]:
    """Gate check: `chosen` holds one row per election at its leave-one-out tau, `untouched` the
    same elections at tau = 0."""
    got = {
        "nat_cov90": float(chosen["nat_cov90"].mean()),
        "nat_cov90_polled": float(chosen["nat_cov90_polled"].mean()),
        "seat_cov90": float(chosen["seat_cov90"].mean()),
        "crps_ratio": float(chosen["nat_crps"].mean() / untouched["nat_crps"].mean()),
    }
    out: Dict[str, object] = {k: round(v, 3) for k, v in got.items()}
    out["pass"] = bool(all(lo <= got[k] <= hi for k, (lo, hi) in CALIBRATION_GATE.items()))
    # for ranking models at their calibrated tau (crps_ratio only compares a model with itself, and
    # a point forecast's ratio is against zero spread)
    out["nat_crps"] = round(float(chosen["nat_crps"].mean()), 3)
    if "seat_rps" in chosen:
        out["seat_rps"] = round(float(chosen["seat_rps"].mean()), 3)
    return out


def rescore_districts(out_dir: str, seed: int = 20261123) -> int:
    """Re-derive and rescore every blend-swing variant (and model-district variant whose district
    draws were saved) from the saved draws.

    For each `<fit>__<election>__h<horizon>__national.npy`, the variants of that fit whose district
    rule is "blend" or "blend_noise" are scored again with the current district code, keeping the
    rest of their summary (fit diagnostics, poll counts). Use after a change to the district rule
    or its noise calibration; no refit needed. Returns the number of variants rescored.
    """
    _, _, seats_map = load_truth()
    n = 0
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith("__national.npy"):
            continue
        fit_name, target, h = f[: -len("__national.npy")].split("__")
        if fit_name not in FITS:
            continue
        nat = np.load(os.path.join(out_dir, f)).astype(float)
        nat = nat / nat.sum(-1, keepdims=True)
        own = os.path.join(out_dir, f"{fit_name}__{target}__{h}__district.npy")
        for variant, rule in FITS[fit_name][2].items():
            if rule not in ("blend", "blend_noise") and not (rule == "model" and os.path.exists(own)):
                continue
            tag = f"{variant}__{target}__{h}"
            path = os.path.join(out_dir, tag + ".json")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                saved = json.load(fh)
            if rule == "model":
                dist = np.load(own).astype(float)
                dist = dist / dist.sum(-1, keepdims=True)
            else:
                dist = districts_from_national(nat, target, noise=(rule == "blend_noise"), seed=seed)
            summary, party = score(nat, dist, target, seats_map)
            saved["summary"].update(summary)
            party.insert(0, "model", variant)
            party.insert(1, "election", target)
            party.insert(2, "horizon", int(h[1:]))
            party.to_csv(os.path.join(out_dir, tag + "__parties.csv"), index=False)
            with open(path, "w") as fh:
                json.dump(saved, fh, indent=1, default=str)
            n += 1
    return n


PAIRED_METRICS = ["nat_crps", "nat_mae", "seat_rps", "seat_logscore", "winner_logscore"]


def compare(out_dir: str, a: str, b: str, metrics: Sequence[str] = PAIRED_METRICS) -> pd.DataFrame:
    """Per-election differences a - b (negative: a is better) for each horizon and metric.

    With five elections a mean hides everything; the table shows each election and how many
    elections `a` won, which is what the backtest can actually tell apart.
    """
    path = os.path.join(out_dir, "scores.csv")
    if not os.path.exists(path):
        summarise(out_dir)
    df = pd.read_csv(path)
    rows = []
    for h, g in df.groupby("horizon"):
        A = g[g.model == a].set_index("election")
        B = g[g.model == b].set_index("election")
        common = A.index.intersection(B.index)
        for m in metrics:
            if m not in A or m not in B:
                continue
            d = (A.loc[common, m] - B.loc[common, m]).dropna()
            row = {"horizon": h, "metric": m, "a_better": int((d < 0).sum()), "n": int(len(d)), "mean_diff": d.mean()}
            row.update({str(e)[:4]: v for e, v in d.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(out_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".json") and "__" in f:
            with open(os.path.join(out_dir, f)) as fh:
                rows.append(json.load(fh)["summary"])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    parties = pd.concat(
        [pd.read_csv(os.path.join(out_dir, f)) for f in os.listdir(out_dir) if f.endswith("__parties.csv")]
    )
    df = rescore_ballot_parties(df, parties)
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    cols = ["nat_mae", "nat_mae_big", "nat_crps", "nat_cov50", "nat_cov90", "dist_mae", "seat_mae",
            "dist_seats_misallocated", "seat_cov90", "seat_logscore", "seat_rps", "p_seat_winner"]
    cols = [c for c in cols if c in df]
    agg = df.groupby(["model", "horizon"])[cols].mean().round(3)
    agg.to_csv(os.path.join(out_dir, "scores_by_model_horizon.csv"))
    parties.to_csv(os.path.join(out_dir, "parties.csv"), index=False)
    return agg


def district_oracle() -> pd.DataFrame:
    """Swing rules for districts given the true national result of each election."""
    district, national, seats_map = load_truth()
    dates = sorted(national)
    rows = []
    for k, E in enumerate(dates):
        if k < 2:
            continue
        prev = dates[k - 1]
        y = national[E].values[None, :]
        rules = {
            "proportional": blend_swing(y, national[prev].values, district[prev].values, 1.0)[0],
            "uniform": blend_swing(y, national[prev].values, district[prev].values, 0.0)[0],
            "blend": blend_swing(y, national[prev].values, district[prev].values, 0.5)[0],
        }
        # static average log-ratio offsets over every past election (what the GP model learns)
        offs = []
        for t in dates[:k]:
            o = np.log(np.clip(district[t].values, 1e-6, None)) - np.log(np.clip(national[t].values, 1e-6, None))[None, :]
            o[:, national[t].values < 0.003] = np.nan
            offs.append(o)
        o = np.nan_to_num(np.nanmean(np.stack(offs), axis=0))
        lat = np.log(np.clip(y, 1e-6, None)) + o
        st = np.exp(lat - lat.max(1, keepdims=True))
        rules["static_offsets"] = st / st.sum(1, keepdims=True)
        true_d = district[E].values
        true_seats = seats_from_district_shares(true_d[None], seats_map.values)[0]
        for name, pred in rules.items():
            ps = seats_from_district_shares(pred[None], seats_map.values)[0]
            rows.append(
                dict(
                    election=str(E.date()),
                    rule=name,
                    district_mae_pp=float(np.abs(pred - true_d).mean() * 100),
                    district_seats_misallocated=float(np.abs(ps - true_seats).sum() / 2),
                    national_seat_error=float(np.abs(ps.sum(0) - true_seats.sum(0)).sum() / 2),
                )
            )
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["rw", "poll_average", "last_result"], help=f"fits to run: {list(FITS)}")
    ap.add_argument("--elections", nargs="+", default=BACKTEST_ELECTIONS)
    ap.add_argument("--horizons", nargs="+", type=int, default=[56, 28, 7])
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--out", default="outputs/backtest/WIP_backtest")
    ap.add_argument("--summarise", metavar="DIR")
    ap.add_argument("--calibrate", metavar="DIR", help="score saved draws with added polling error")
    ap.add_argument("--district-oracle", action="store_true")
    ap.add_argument("--rescore-districts", metavar="DIR", help="rescore blend-swing variants from saved draws")
    ap.add_argument("--compare", nargs=3, metavar=("DIR", "A", "B"), help="per-election differences A - B")
    args = ap.parse_args(argv)
    if args.compare:
        pd.set_option("display.width", 250)
        print(compare(*args.compare).round(3).to_string(index=False))
        return
    if args.rescore_districts:
        print(f"rescored {rescore_districts(args.rescore_districts, args.seed)} variants")
        pd.set_option("display.width", 250)
        print(summarise(args.rescore_districts).to_string())
        return
    if args.summarise:
        pd.set_option("display.width", 250)
        print(summarise(args.summarise).to_string())
        return
    if args.calibrate:
        pd.set_option("display.width", 250)
        for h in args.horizons:
            calibrate_polling_error(args.calibrate, models=args.models, horizon=h)
        return
    if args.district_oracle:
        df = district_oracle()
        pd.set_option("display.width", 200)
        print(df.round(2).to_string(index=False))
        print(df.groupby("rule")[["district_mae_pp", "district_seats_misallocated", "national_seat_error"]].mean().round(2))
        return
    os.makedirs(args.out, exist_ok=True)
    for E in args.elections:
        for h in args.horizons:
            for m in args.models:
                try:
                    run_one(m, E, h, args.draws, args.tune, args.chains, args.seed, args.out)
                except Exception as e:  # keep the grid going
                    import traceback

                    traceback.print_exc()
                    print(f"FAILED {m} {E} h{h}: {e}", file=sys.stderr)
    print(summarise(args.out).to_string())


if __name__ == "__main__":
    main()
