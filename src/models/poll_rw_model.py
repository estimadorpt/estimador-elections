"""National poll aggregation as a weekly random walk (the "rw" model).

Why this exists
---------------
`DynamicGPElectionModel` writes national opinion as three Hilbert-space GPs over the whole
calendar (2009 to the target election). With L = c * half-range ~ 5,500 days, its 40 basis
functions cannot draw anything with a period shorter than 4L/m ~ 550 days, so the "14-day" GP
moves on a ~1.5-year scale and the model cannot follow a campaign. Its pollster bias is one
static vector for 16 years, and its forecast has no term for the industry-wide polling error
of the election being forecast.

This model is the standard dynamic poll aggregator of the literature (Jackman 2005; Linzer
2013; Stoetzer et al. 2019), adapted to a multiparty system:

* opinion: per-party log-ratio random walk on a weekly grid, with party-specific volatility,
  started at each party's first appearance (Chega, IL, Livre did not exist in 2009); the
  softmax runs over the parties that exist at each date;
* polls: Dirichlet-multinomial around softmax(opinion + house effect of the pollster +
  industry bias of the election cycle);
* house effects: pollster x party, zero-sum across pollsters (relative to the average
  pollster) and across parties;
* industry bias: one zero-sum vector per election cycle, drawn around a persistent mean
  (``persistent_bias=True``) or around zero. For past cycles the election result pins it; for
  the cycle being forecast it is only weakly informed (by how the cycle's first polls compare
  with the previous result), so its prior spread enters the forecast. This is the term that
  makes intervals honest about polling misses like Chega 2025 or PS 2022;
* results: national results enter as a Dirichlet-multinomial with a high concentration, so
  the random walk passes (almost) through each result.

District shares and seats are not in this model: they are derived from the national draws by
`src.processing.district_swing` (swing from the last election, with calibrated noise), which
backtests better than static district offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

PARTIES_DEFAULT = ["PS", "CH", "IL", "BE", "CDU", "PAN", "L", "AD"]
MASK_LOG = -30.0  # log-weight given to a party that does not exist yet


@dataclass
class RWConfig:
    step_days: int = 7
    innovation_sd_prior: float = 0.04  # weekly sd of a party's log-ratio (HalfNormal scale)
    house_sd_prior: float = 0.10
    bias_sd_prior: float = 0.10  # spread of the cycle industry bias around its mean
    persistent_bias: bool = False  # learn a mean bias that carries into the next cycle
    persistent_bias_sd: float = 0.05
    poll_concentration_prior: tuple = (2.0, 0.01)  # Gamma(alpha, beta) for the DM concentration
    result_concentration: float = 20000.0  # national results: the walk passes through them
    entry_lead_days: int = 30  # a party "exists" this many days before its first non-zero share
    # Opinion dynamics. "rw": the random walk alone. "rw_ou": plus a short Ornstein-Uhlenbeck
    # term (Matern-1/2 GP: bumps that fade). "rw_m32": plus a short Matern-3/2 GP term (smooth
    # bumps with momentum). The short term's timescale is learned; its prior keeps it short
    # (median 5 weeks, 90% in about 2-11 weeks) so it cannot turn into a second random walk.
    dynamics: str = "rw"
    short_timescale_weeks: float = 5.0
    short_timescale_logsd: float = 0.5
    short_sd_prior: float = 0.08  # HalfNormal scale of the short term's stationary sd (log-ratio)
    # Fix the timescale (weeks) instead of learning it. Learned, it sank to about one grid step in
    # the pilot (2025, h56) and the chains did not mix (OU rhat 1.09, Matern rhat 1.24), so the
    # backtest compares fixed timescales: 4 weeks primary, 2 and 8 as sensitivity.
    short_timescale_fixed: Optional[float] = None


def short_term_process(eps, tau, dynamics: str, dt: float = 1.0):
    """Unit-variance stationary short-term process on a regular grid, exact discretisation.

    eps: standard normal innovations, (T, P) for "rw_ou" or (T, 2, P) for "rw_m32";
    tau: timescale in grid units of weeks (OU: correlation exp(-lag/tau); Matern-3/2 with
    lengthscale tau: (1 + sqrt(3) lag/tau) exp(-sqrt(3) lag/tau)); dt: grid step in weeks.
    Returns (T, P). Written with scan so the cost is linear in T (the HSGP approximation of
    the GP model cannot resolve short timescales over a 16-year calendar).
    """
    import pytensor

    eps = pt.as_tensor_variable(eps)
    tau = pt.as_tensor_variable(tau)
    if dynamics == "rw_ou":
        phi = pt.exp(-dt / tau)
        innov = pt.sqrt(1.0 - phi**2)
        x0 = eps[0]
        xs, _ = pytensor.scan(
            lambda e, prev, phi, innov: phi * prev + innov * e,
            sequences=eps[1:],
            outputs_info=x0,
            non_sequences=[phi, innov],
        )
        return pt.concatenate([x0[None, :], xs], axis=0)
    if dynamics == "rw_m32":
        lam = np.sqrt(3.0) / tau
        a = lam * dt
        e = pt.exp(-a)
        a11, a12 = e * (1 + a), e * dt
        a21, a22 = -e * lam**2 * dt, e * (1 - a)
        # stationary covariance diag(1, lam^2); Q = Pinf - A Pinf A'
        q11 = 1.0 - (a11**2 + lam**2 * a12**2)
        q12 = -(a11 * a21 + lam**2 * a12 * a22)
        q22 = lam**2 - (a21**2 + lam**2 * a22**2)
        l11 = pt.sqrt(pt.maximum(q11, 1e-12))
        l21 = q12 / l11
        l22 = pt.sqrt(pt.maximum(q22 - l21**2, 1e-12))
        f0 = eps[0, 0]
        d0 = lam * eps[0, 1]

        def step(e_t, f_prev, d_prev, a11, a12, a21, a22, l11, l21, l22):
            f = a11 * f_prev + a12 * d_prev + l11 * e_t[0]
            d = a21 * f_prev + a22 * d_prev + l21 * e_t[0] + l22 * e_t[1]
            return f, d

        (fs, _), _ = pytensor.scan(
            step,
            sequences=eps[1:],
            outputs_info=[f0, d0],
            non_sequences=[a11, a12, a21, a22, l11, l21, l22],
        )
        return pt.concatenate([f0[None, :], fs], axis=0)
    raise ValueError(f"unknown short-term dynamics {dynamics!r}")


@dataclass
class RWData:
    parties: List[str]
    grid: pd.DatetimeIndex  # weekly grid, ends exactly on the target date
    exists: np.ndarray  # (T, P) bool
    poll_t: np.ndarray
    poll_pollster: np.ndarray
    poll_cycle: np.ndarray
    poll_counts: np.ndarray
    pollsters: List[str]
    cycles: List[str]  # election dates (str) of each cycle; the last one is the target
    result_t: np.ndarray
    result_cycle: np.ndarray
    result_counts: np.ndarray
    target_t: int
    extra: Dict = field(default_factory=dict)
    poll_reported: Optional[np.ndarray] = None  # (N, P) bool; None = every existing party reported


def _to_grid_index(dates: pd.Series | pd.DatetimeIndex, grid: pd.DatetimeIndex) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates)).values.astype("datetime64[D]").astype(np.int64)
    g = grid.values.astype("datetime64[D]").astype(np.int64)
    idx = np.searchsorted(g, d)
    idx = np.clip(idx, 0, len(g) - 1)
    prev = np.clip(idx - 1, 0, len(g) - 1)
    choose_prev = np.abs(d - g[prev]) <= np.abs(g[idx] - d)
    return np.where(choose_prev, prev, idx).astype(int)


def prepare_rw_data(
    polls: pd.DataFrame,
    results_national: pd.DataFrame,
    target_date: str | pd.Timestamp,
    historical_dates: Sequence[str],
    parties: Sequence[str] = PARTIES_DEFAULT,
    cfg: RWConfig = RWConfig(),
) -> RWData:
    """Build the arrays the model needs.

    polls: one row per poll with columns date, pollster, and integer counts per party
        (as `ElectionDataset.polls_train`).
    results_national: one row per past election with election_date and counts per party.
    historical_dates: past election dates (any order); only those before target_date are used.
    """
    parties = list(parties)
    target = pd.Timestamp(target_date).normalize()
    past = sorted(pd.Timestamp(d) for d in historical_dates if pd.Timestamp(d) < target)
    polls = polls[pd.to_datetime(polls["date"]) <= target].copy()
    start = min(pd.to_datetime(polls["date"]).min(), past[0] if past else target)
    n_steps = int(np.ceil((target - start).days / cfg.step_days)) + 1
    grid = pd.DatetimeIndex([target - pd.Timedelta(days=cfg.step_days * k) for k in range(n_steps)][::-1])

    cycles_end = past + [target]
    cycle_names = [d.strftime("%Y-%m-%d") for d in cycles_end]

    def cycle_of(dates):
        d = pd.to_datetime(pd.Series(dates)).values
        ends = np.array(cycles_end, dtype="datetime64[ns]")
        # a poll belongs to the first election on or after it (strictly after for exit polls,
        # which ElectionDataset already drops)
        return np.searchsorted(ends, d, side="left").clip(0, len(ends) - 1)

    res = results_national.reset_index(drop=True).copy()
    res["election_date"] = pd.to_datetime(res["election_date"])
    res = res[res["election_date"].isin(past)].sort_values("election_date")

    # party existence: first date with a non-zero share in any poll or result
    first_seen = {}
    for p in parties:
        dates = []
        nz = polls.loc[polls[p] > 0, "date"]
        if len(nz):
            dates.append(pd.to_datetime(nz).min())
        nzr = res.loc[res[p] > 0, "election_date"]
        if len(nzr):
            dates.append(nzr.min())
        first_seen[p] = min(dates) - pd.Timedelta(days=cfg.entry_lead_days) if dates else target + pd.Timedelta(days=1)
    exists = np.stack([(grid >= first_seen[p]) for p in parties], axis=1)

    pollster_codes, pollsters = pd.factorize(polls["pollster"], sort=True)
    poll_t = _to_grid_index(polls["date"], grid)
    # a poll must not use a party that the grid says does not exist yet
    poll_counts = polls[parties].to_numpy().astype(int)
    poll_counts = np.where(exists[poll_t], poll_counts, 0)
    # A party with no count in a poll was not reported separately (PAN in 249 polls of
    # 2011-2018, Livre in 196 of 2015-2022 sit in "others"): it is left out of that poll's
    # likelihood instead of being read as zero support.
    poll_reported = poll_counts > 0

    result_counts = res[parties].to_numpy().astype(np.int64)
    result_t = _to_grid_index(res["election_date"], grid)
    result_counts = np.where(exists[result_t], result_counts, 0)

    return RWData(
        parties=parties,
        grid=grid,
        exists=exists,
        poll_t=poll_t,
        poll_pollster=pollster_codes.astype(int),
        poll_cycle=cycle_of(polls["date"]).astype(int),
        poll_counts=poll_counts,
        pollsters=list(pollsters),
        cycles=cycle_names,
        result_t=result_t,
        result_cycle=np.arange(len(res)),
        result_counts=result_counts,
        target_t=len(grid) - 1,
        extra={"first_seen": first_seen},
        poll_reported=poll_reported,
    )


def build_rw_model(data: RWData, cfg: RWConfig = RWConfig()) -> pm.Model:
    T, P = data.exists.shape
    J = len(data.pollsters)
    C = len(data.cycles)
    coords = {
        "time": data.grid.strftime("%Y-%m-%d"),
        "parties": data.parties,
        "pollsters": data.pollsters,
        "cycles": data.cycles,
        "polls": np.arange(len(data.poll_t)),
        "results": np.arange(len(data.result_t)),
    }
    # innovations only count once a party exists (before that its level is irrelevant)
    step_mask = data.exists.astype(float).copy()
    step_mask[0] = 0.0
    log_mask = np.where(data.exists, 0.0, MASK_LOG)
    with pm.Model(coords=coords) as model:
        level0 = pm.ZeroSumNormal("level0", sigma=1.5, dims="parties")
        sigma_rw = pm.HalfNormal("sigma_rw", sigma=cfg.innovation_sd_prior, dims="parties")
        z = pm.Normal("z_rw", 0.0, 1.0, dims=("time", "parties"))
        steps = z * sigma_rw[None, :] * step_mask * np.sqrt(cfg.step_days / 7.0)
        eta_rw = level0[None, :] + pt.cumsum(steps, axis=0)
        if cfg.dynamics == "rw":
            eta = pm.Deterministic("eta", eta_rw, dims=("time", "parties"))
        else:
            if cfg.short_timescale_fixed is not None:
                tau = pt.constant(float(cfg.short_timescale_fixed))
            else:
                tau = pm.LogNormal("short_timescale", mu=np.log(cfg.short_timescale_weeks), sigma=cfg.short_timescale_logsd)
            sigma_short = pm.HalfNormal("sigma_short", sigma=cfg.short_sd_prior, dims="parties")
            if cfg.dynamics == "rw_ou":
                # static shape: JAX's scan needs a concrete length (dims alone are symbolic)
                eps = pm.Normal("z_short", 0.0, 1.0, shape=(T, P), dims=("time", "parties"))
            elif cfg.dynamics == "rw_m32":
                model.add_coord("state", ["level", "slope"])
                eps = pm.Normal("z_short", 0.0, 1.0, shape=(T, 2, P), dims=("time", "state", "parties"))
            else:
                raise ValueError(f"unknown dynamics {cfg.dynamics!r}")
            short = short_term_process(eps, tau, cfg.dynamics, dt=cfg.step_days / 7.0) * sigma_short[None, :]
            eta = pm.Deterministic("eta", eta_rw + short, dims=("time", "parties"))
        log_mask_t = pt.as_tensor_variable(log_mask)

        # house effects: zero-sum over pollsters and over parties
        sigma_house = pm.HalfNormal("sigma_house", sigma=cfg.house_sd_prior, dims="parties")
        house_raw = pm.ZeroSumNormal("house_raw", sigma=1.0, dims=("pollsters", "parties"), n_zerosum_axes=2)
        house = pm.Deterministic("house", house_raw * sigma_house[None, :], dims=("pollsters", "parties"))

        # industry bias per cycle
        sigma_bias = pm.HalfNormal("sigma_bias", sigma=cfg.bias_sd_prior)
        bias_raw = pm.ZeroSumNormal("bias_raw", sigma=1.0, dims=("cycles", "parties"))
        if cfg.persistent_bias:
            bias_mean = pm.ZeroSumNormal("bias_mean", sigma=cfg.persistent_bias_sd, dims="parties")
            bias = pm.Deterministic("bias", bias_mean[None, :] + sigma_bias * bias_raw, dims=("cycles", "parties"))
        else:
            bias = pm.Deterministic("bias", sigma_bias * bias_raw, dims=("cycles", "parties"))

        if data.poll_reported is not None:
            poll_mask = np.where(data.poll_reported, 0.0, MASK_LOG)
        else:
            poll_mask = log_mask[data.poll_t]
        logits_polls = eta[data.poll_t] + house[data.poll_pollster] + bias[data.poll_cycle] + poll_mask
        p_polls = pt.special.softmax(logits_polls, axis=1)
        conc = pm.Gamma("concentration_polls", alpha=cfg.poll_concentration_prior[0], beta=cfg.poll_concentration_prior[1])
        n_polls = data.poll_counts.sum(1)
        pm.DirichletMultinomial(
            "poll_obs", n=n_polls, a=conc * p_polls + 1e-10, observed=data.poll_counts, dims=("polls", "parties")
        )

        if len(data.result_t):
            logits_res = eta[data.result_t] + log_mask_t[data.result_t]
            p_res = pt.special.softmax(logits_res, axis=1)
            pm.DirichletMultinomial(
                "result_obs",
                n=data.result_counts.sum(1),
                a=cfg.result_concentration * p_res + 1e-10,
                observed=data.result_counts,
                dims=("results", "parties"),
            )

        pm.Deterministic(
            "share_target",
            pt.special.softmax(eta[data.target_t] + log_mask_t[data.target_t], axis=0),
            dims="parties",
        )
    return model


def national_share_draws(idata, data: RWData, t_index: Optional[int] = None) -> np.ndarray:
    """(S, P) draws of national shares at grid point t_index (default: the target)."""
    if t_index is None or t_index == data.target_t:
        x = idata.posterior["share_target"].stack(s=("chain", "draw")).transpose("s", "parties").values
        return np.asarray(x)
    eta = idata.posterior["eta"].isel(time=t_index).stack(s=("chain", "draw")).transpose("s", "parties").values
    logits = eta + np.where(data.exists[t_index], 0.0, MASK_LOG)[None, :]
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
