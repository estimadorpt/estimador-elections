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


def add_polling_error(
    shares: np.ndarray,
    tau: float,
    rng: Optional[np.random.Generator] = None,
    absent_share: float = ABSENT_SHARE,
) -> np.ndarray:
    """Return (S, P) draws with an election-level polling error added.

    shares: (S, P) national vote shares, rows summing to 1.
    tau: sd of each party's shift on the log scale (0 returns the input unchanged).
    """
    shares = np.asarray(shares, dtype=float)
    if tau <= 0:
        return shares.copy()
    rng = np.random.default_rng() if rng is None else rng
    alive = shares.mean(0) > absent_share
    eps = rng.normal(0.0, tau, shares.shape) * alive[None, :]
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
