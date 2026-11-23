"""Election-level polling error for national share draws.

The backtest (src/evaluation/backtest.py) shows that both national models put the result of
the election being forecast inside their 90% intervals only about half the time: the polls of
one election miss together (Portugal 2022 and 2025), and neither model's posterior carries
that shared miss into the forecast. This module adds it after the fit: every draw gets one
shift per party on the log scale, Normal(0, tau), centred across the parties that exist, so
the draw stays on the simplex and keeps its ranking on average.

tau is calibrated on the backtest by leave-one-election-out (`calibrate_polling_error` in the
backtest harness); 0.15-0.2 brought the random-walk model's 90% coverage from 0.54 to 0.75-0.83
in the 8-weeks-out forecasts of 2015-2025. With 2025's result as the centre, tau = 0.17 gives
an sd of about 4.4 points for AD (34.5%), 3.6 for PS and CH (about 24%), 1.1 for IL (5.8%)
and 0.3 for PAN (1.4%). At 8 weeks out that error includes the campaign's movement, so the
value should be calibrated per horizon.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

ABSENT_SHARE = 1e-4  # a party below this mean share is treated as not running

# Left (-1) and right (+1) blocs for a correlated error: a shared left-right swing in the miss.
LEFT_RIGHT = {"PS": -1, "BE": -1, "CDU": -1, "L": -1, "PAN": -1, "AD": 1, "IL": 1, "CH": 1}


def bloc_signs(parties) -> np.ndarray:
    return np.array([LEFT_RIGHT.get(str(p), 0) for p in parties], dtype=float)


def add_polling_error(
    shares: np.ndarray,
    tau: float,
    rng: Optional[np.random.Generator] = None,
    absent_share: float = ABSENT_SHARE,
    bloc_rho: float = 0.0,
    signs: Optional[np.ndarray] = None,
    df: Optional[float] = None,
) -> np.ndarray:
    """Return (S, P) draws with an election-level polling error added.

    shares: (S, P) national vote shares, rows summing to 1.
    tau: sd of each party's shift on the log scale (0 returns the input unchanged).
    bloc_rho: share of each party's shift variance that is a left-right swing common to its
        bloc (`signs`, +1 right / -1 left / 0 neither, e.g. `bloc_signs(parties)`); 0 = independent.
    df: Student-t degrees of freedom for the whole draw's error (one scale per draw, so big
        misses hit many parties at once), standardised so tau stays the sd; None = normal.
    """
    shares = np.asarray(shares, dtype=float)
    if tau <= 0:
        return shares.copy()
    rng = np.random.default_rng() if rng is None else rng
    alive = shares.mean(0) > absent_share
    z = rng.normal(0.0, 1.0, shares.shape)
    if bloc_rho > 0:
        if signs is None:
            raise ValueError("bloc_rho needs signs (bloc_signs(parties))")
        swing = rng.normal(0.0, 1.0, (shares.shape[0], 1)) * np.asarray(signs, float)[None, :]
        z = np.sqrt(1 - bloc_rho) * z + np.sqrt(bloc_rho) * swing
    if df is not None:
        scale = np.sqrt(df / rng.chisquare(df, (shares.shape[0], 1))) * np.sqrt((df - 2) / df)
        z = z * scale
    eps = tau * z * alive[None, :]
    if alive.any():
        eps[:, alive] -= eps[:, alive].mean(1, keepdims=True)
    out = shares.copy()
    if not alive.any():
        return out
    sub = shares[:, alive]
    logits = np.log(np.clip(sub, 1e-300, None)) + eps[:, alive]
    e = np.exp(logits - logits.max(1, keepdims=True))
    # absent parties keep their (near-zero) share; the others share the rest
    out[:, alive] = e / e.sum(1, keepdims=True) * sub.sum(1, keepdims=True)
    return out
