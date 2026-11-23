"""District vote shares and seats from national vote-share draws.

The rule is a 50/50 blend of two swings from the most recent election:

* proportional (log-ratio) swing: each party's district share moves by the same factor as
  its national share, then shares are renormalised;
* uniform swing: each party's district share moves by the same number of points as its
  national share (clipped at zero), then renormalised.

With the true national result given, the blend misallocated 7.4 district seats per election
over 2015-2025, against 12.0 for the static average offsets the GP model uses, 8.0 for
proportional swing alone and 7.6 for uniform swing alone (`src/evaluation/backtest.py
--district-oracle`). Noise is added on the log-share scale, one independent draw per district
and party, with a party-specific scale measured on past elections' blend residuals (pooled
toward the all-party value), so seat probabilities carry district-level surprise.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

EPS = 1e-6
STOOD = 1e-4  # a district share below this means the party had no list there


def blend_swing(
    national: np.ndarray,
    last_national: np.ndarray,
    last_district: np.ndarray,
    weight_proportional: float = 0.5,
) -> np.ndarray:
    """District shares from national shares.

    national: (S, P) national share draws (rows sum to 1).
    last_national: (P,) national shares at the last election.
    last_district: (D, P) district shares at the last election.
    Returns (S, D, P).

    A party absent at the last election (share ~0) gets its national share in every district
    under the proportional rule (no geography known yet) and the uniform swing from zero.
    """
    national = np.asarray(national, float)
    last_national = np.asarray(last_national, float)
    last_district = np.asarray(last_district, float)
    absent = last_national < 1e-3
    ratio = np.where(absent[None, :], 1.0, last_district / np.clip(last_national, EPS, None)[None, :])  # (D,P)
    prop = national[:, None, :] * ratio[None, :, :]
    prop = prop / prop.sum(-1, keepdims=True)
    unif = np.clip(last_district[None, :, :] + (national - last_national[None, :])[:, None, :], 0.0, None)
    unif = unif / np.clip(unif.sum(-1, keepdims=True), EPS, None)
    w = weight_proportional
    return w * prop + (1 - w) * unif


def add_district_noise(
    shares: np.ndarray,
    party_sd: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Multiply each district-party share by exp(N(0, sd_p)) and renormalise. shares (S,D,P)."""
    rng = rng or np.random.default_rng(0)
    noise = rng.normal(size=shares.shape) * np.asarray(party_sd)[None, None, :]
    x = np.clip(shares, EPS, None) * np.exp(noise)
    x = np.where(shares <= EPS, 0.0, x)
    return x / x.sum(-1, keepdims=True)


def calibrate_noise(
    district_shares: Dict[pd.Timestamp, pd.DataFrame],
    national_shares: Dict[pd.Timestamp, pd.Series],
    before: pd.Timestamp,
    parties: Sequence[str],
    pool: float = 2.0,
    floor: float = 0.05,
) -> np.ndarray:
    """Party sd of log(true/blend) district residuals over elections before `before`.

    Each election after the first known one is predicted from the previous with its own true
    national shares. Districts where the party had no list in either election are left out. A party's sd is shrunk toward the all-party sd with `pool` pseudo
    elections; parties never measured get the pooled value.
    """
    dates = sorted(d for d in district_shares if d < before)
    errs = {p: [] for p in parties}
    for prev, cur in zip(dates[:-1], dates[1:]):
        pred = blend_swing(
            national_shares[cur].values[None, :],
            national_shares[prev].values,
            district_shares[prev].values,
        )[0]
        true = district_shares[cur].values
        before_d = district_shares[prev].values
        for j, p in enumerate(parties):
            if national_shares[cur][p] > 0.005 and national_shares[prev][p] > 0.005:
                # a district where the party did not stand in either election says nothing about
                # swing noise (PAN had no list in Viseu in 2025: log(0) would set its sd to 2)
                stood = (true[:, j] > STOOD) & (before_d[:, j] > STOOD)
                if stood.sum() < 2:
                    continue
                e = np.log(true[stood, j]) - np.log(np.clip(pred[stood, j], EPS, None))
                errs[p].append(np.mean(e**2))
    allv = [v for p in parties for v in errs[p]]
    pooled = np.sqrt(np.mean(allv)) if allv else 0.25
    out = []
    for p in parties:
        n = len(errs[p])
        ms = (np.sum(errs[p]) + pool * pooled**2) / (n + pool)
        out.append(max(np.sqrt(ms), floor))
    return np.array(out)


def dhondt_draws(shares: np.ndarray, seats: int) -> np.ndarray:
    """Vectorised D'Hondt. shares (S, P) -> seats (S, P)."""
    S, P = shares.shape
    div = np.arange(1, seats + 1, dtype=float)
    q = (shares[:, :, None] / div[None, None, :]).reshape(S, P * seats)
    # tie-break by a tiny party-order term so results are deterministic
    q = q - np.repeat(np.arange(P), seats)[None, :] * 1e-12
    top = np.argpartition(-q, seats - 1, axis=1)[:, :seats]
    party_of = top // seats
    out = np.zeros((S, P), dtype=int)
    rows = np.repeat(np.arange(S), seats)
    np.add.at(out, (rows, party_of.ravel()), 1)
    return out


def seats_from_district_shares(shares: np.ndarray, district_seats: Sequence[int]) -> np.ndarray:
    """shares (S, D, P), seats per district (D,) -> (S, D, P) seats."""
    return np.stack([dhondt_draws(shares[:, d, :], int(k)) for d, k in enumerate(district_seats)], axis=1)


def district_posterior_from_national(
    national,
    target_date,
    noise: bool = True,
    weight_proportional: float = 0.5,
    seed: int = 0,
):
    """District shares (chain, draw, districts, parties_complete) by blend swing, for `predict`.

    national: xarray DataArray (chain, draw, parties_complete) of national shares at the date
    being forecast (polling error, if any, already added). Districts swing from the last
    election before `target_date`; with `noise`, each district-party share gets the
    calibrated noise of `calibrate_noise` measured on every earlier election.
    """
    import xarray as xr

    from src.evaluation.backtest import PARTIES, load_truth

    district, national_hist, _ = load_truth()
    target = pd.Timestamp(target_date)
    last = max(t for t in national_hist if t < target)
    parties = [str(p) for p in national.coords["parties_complete"].values]
    missing = [p for p in parties if p not in PARTIES]
    if missing:
        raise ValueError(f"No district results for parties {missing}")
    nat = national.transpose("chain", "draw", "parties_complete").values
    n_chain, n_draw, n_party = nat.shape
    flat = nat.reshape(-1, n_party)
    last_d = district[last][parties]
    shares = blend_swing(flat, national_hist[last][parties].values, last_d.values, weight_proportional)
    if noise:
        sd_all = calibrate_noise(district, national_hist, target, PARTIES)
        sd = np.array([sd_all[PARTIES.index(p)] for p in parties])
        shares = add_district_noise(shares, sd, np.random.default_rng(seed))
    return xr.DataArray(
        shares.reshape(n_chain, n_draw, len(last_d.index), n_party),
        dims=("chain", "draw", "districts", "parties_complete"),
        coords={
            "chain": national.coords["chain"].values,
            "draw": national.coords["draw"].values,
            "districts": list(last_d.index),
            "parties_complete": parties,
        },
    )
