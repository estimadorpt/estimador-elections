"""Fixed protocol for testing a change to the district step (a new factor or a new noise model).

Agreed 23 Nov 2026 (doc "District factor test protocol"). The district step turns national
vote-share draws into district shares and seats. A candidate is compared with the baseline under
the same rules:

1. Joint tuning. Each variant has a small grid of hyperparameters (the blend weight w, the noise
   scale, the regional share of the noise, the factor's own settings). For every target election
   they are chosen by leave-one-election-out on district-share CRPS with the true national result:
   the target election never chooses its own settings. Baseline and candidate get the same search.
2. Stage A, true national result. Scores per target election (2011-2025): seat-weighted
   district-share CRPS (primary), district-share 90% coverage (gate 0.80-0.97), district-seat RPS,
   seat-total RPS, and for continuity district MAE and seats misallocated.
3. Stage B, forecast national. Saved random-walk national draws at 7, 28 and 56 days before each
   election (2015-2025), with the calibrated election-level polling error added, go through the
   same district step with the settings chosen in stage A. Primary: district-seat RPS. Secondary:
   seat-total RPS and its 90% coverage (gate >= 0.85).
4. Decision (candidate vs baseline): stage A better in at least 4 elections with a mean gain of
   at least 3% and a one-sided sign-flip p < 0.05 over region x election clusters; stage B better
   at 2 or more horizons in at least 3 of 5 elections; seat-total RPS no more than 1% worse at any
   horizon; both coverage gates hold.

    python -m src.evaluation.district_protocol --stage A
    python -m src.evaluation.district_protocol --stage B --national-dir DIR   # <fit>__<date>__hNNN__national.npy
    python -m src.evaluation.district_protocol --stage AB --national-dir DIR --out OUT
"""

from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.backtest import PARTIES, load_truth
from src.processing.district_swing import blend_swing, calibrate_noise, seats_from_district_shares
from src.processing.polling_error import add_polling_error

# NUTS-II regions of the 20 domestic circles (1 Norte, 2 Centro, 3 Lisboa/AML + Santarem + Setubal,
# 4 Alentejo, 5 Algarve, 6 Madeira, 7 Acores). Clusters for the sign-flip test.
REGION = {"Aveiro": 2, "Beja": 4, "Braga": 1, "Bragança": 1, "Castelo Branco": 2, "Coimbra": 2,
          "Évora": 4, "Faro": 5, "Guarda": 2, "Leiria": 2, "Lisboa": 3, "Portalegre": 4, "Porto": 1,
          "Santarém": 3, "Setúbal": 3, "Viana do Castelo": 1, "Vila Real": 1, "Viseu": 2,
          "Madeira": 6, "Açores": 7}

STAGE_A_TARGETS = ["2011-06-05", "2015-10-04", "2019-10-06", "2022-01-30", "2024-03-10", "2025-05-18"]
STAGE_B_TARGETS = ["2015-10-04", "2019-10-06", "2022-01-30", "2024-03-10", "2025-05-18"]
HORIZONS = (7, 28, 56)
POLLING_TAU = {7: 0.13, 28: 0.16, 56: 0.20}  # rw_nobias calibrated tau (src/forecast_rw.py RW_NOBIAS_TAU)
EPS = 1e-6


# ----------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------


@dataclass
class Variant:
    """A district step: mean (blend at weight w, plus any factor) and noise, with a tuning grid.

    sample(national (S,P), target, params, z (S,D,P), zr (S,R,P)) -> district shares (S,D,P)
    """

    name: str
    grid: Dict[str, Sequence]
    sample: Callable
    params_list: List[Dict] = field(init=False)

    def __post_init__(self):
        keys = list(self.grid)
        self.params_list = [dict(zip(keys, v)) for v in itertools.product(*(self.grid[k] for k in keys))]


def _last(target):
    _, national, _ = load_truth()
    E = pd.Timestamp(target)
    return max(t for t in national if t < E)


def _blend(national, target, w):
    district, nat_hist, _ = load_truth()
    last = _last(target)
    return blend_swing(national, nat_hist[last].values, district[last].values, w)


def _region_index(districts) -> np.ndarray:
    return np.array([REGION[d] for d in districts]) - 1


def _mix(z, zr, reg, phi):
    if phi <= 0:
        return z
    return np.sqrt(1 - phi) * z + np.sqrt(phi) * zr[:, reg, :]


def current_noise(national, target, p, z, zr):
    """Baseline: party log-sd pooled toward the all-party value (calibrate_noise), times a tuned k."""
    district, nat_hist, seats = load_truth()
    base = _blend(national, target, p["w"])
    sd = calibrate_noise(district, nat_hist, pd.Timestamp(target), PARTIES) * p["k"]
    e = _mix(z, zr, _region_index(seats.index), p.get("phi", 0.0)) * sd[None, None, :]
    x = np.clip(base, EPS, None) * np.exp(e)
    x = np.where(base <= EPS, 0.0, x)
    return x / x.sum(-1, keepdims=True)


def sqrt_noise(national, target, p, z, zr):
    """Share-scale noise: s = mu + c * sqrt(mu) * z, clipped at 0 and renormalised (one parameter c)."""
    _, _, seats = load_truth()
    base = _blend(national, target, p["w"])
    e = _mix(z, zr, _region_index(seats.index), p.get("phi", 0.0))
    x = np.clip(base + p["c"] * np.sqrt(np.clip(base, 0, None)) * e, 0.0, None)
    x = np.where(base <= EPS, 0.0, x)
    return x / np.clip(x.sum(-1, keepdims=True), EPS, None)


VARIANTS = {
    "current": Variant(
        "current", {"w": (0.25, 0.5, 0.75), "k": (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2)}, current_noise
    ),
    "sqrt": Variant(
        "sqrt", {"w": (0.25, 0.5, 0.75), "c": (0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06)}, sqrt_noise
    ),
    "sqrt_region": Variant(
        "sqrt_region",
        {"w": (0.25, 0.5, 0.75), "c": (0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06), "phi": (0.3,)},
        sqrt_noise,
    ),
}
# the variant every candidate is compared against
BASELINE = "current"


# ----------------------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------------------


def crps_cells(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample CRPS per cell. draws (S, ...), y (...). Same units as the input."""
    x = np.sort(draws, 0)
    n = x.shape[0]
    t1 = np.abs(x - y[None]).mean(0)
    i = np.arange(1, n + 1).reshape((n,) + (1,) * (x.ndim - 1))
    t2 = 2.0 * ((2 * i - n - 1) * x).sum(0) / (n * n)
    return t1 - 0.5 * t2


def rps_cells(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score per cell for integer counts. draws (S, ...), y (...)."""
    K = int(max(draws.max(), y.max())) + 1
    out = np.zeros(y.shape)
    for k in range(K):
        out += ((draws <= k).mean(0) - (y <= k)) ** 2
    return out


def share_scores(dist: np.ndarray, target: str) -> Dict[str, object]:
    """District-share scores, plus per-region CRPS for the cluster test. dist (S, D, P)."""
    district, national, seats = load_truth()
    E = pd.Timestamp(target)
    y = district[E].values
    ran = national[E].values > 0
    c = crps_cells(dist * 100, y * 100)[:, ran]  # (D, P_ran), pp
    w = seats.values / seats.values.sum()
    lo, hi = np.quantile(dist, [0.05, 0.95], axis=0)
    inn = ((lo <= y) & (y <= hi))[:, ran]
    reg = _region_index(seats.index)
    by_region = {r: float((c[reg == r].mean(1) * w[reg == r]).sum() / w[reg == r].sum()) for r in np.unique(reg)}
    return dict(
        share_crps=float((c.mean(1) * w).sum()),
        share_cov90=float(inn.mean()),
        dist_mae=float(np.abs(dist.mean(0) - y)[:, ran].mean() * 100),
        region_crps=by_region,
        region_weight={r: float(w[reg == r].sum()) for r in np.unique(reg)},
    )


def seat_scores(dist: np.ndarray, target: str) -> Dict[str, float]:
    district, national, seats = load_truth()
    E = pd.Timestamp(target)
    y = district[E].values
    ran = national[E].values > 0
    ts = seats_from_district_shares(y[None], seats.values)[0]  # (D, P)
    sd = seats_from_district_shares(dist, seats.values)  # (S, D, P)
    tot, ttot = sd.sum(1), ts.sum(0)
    lo, hi = np.quantile(tot, [0.05, 0.95], axis=0)
    return dict(
        dist_seat_rps=float(rps_cells(sd, ts)[:, ran].sum()),
        seat_total_rps=float(rps_cells(tot, ttot)[ran].mean()),
        seat_total_cov90=float(((lo <= ttot) & (ttot <= hi))[ran].mean()),
        seats_misallocated=float(np.abs(sd - ts[None]).sum((1, 2)).mean() / 2),
    )


# ----------------------------------------------------------------------------------------
# Stage A: true national result, leave-one-election-out tuning
# ----------------------------------------------------------------------------------------


def _shocks(S: int, seed: int):
    """Common random numbers: the same z for every variant and setting of one election."""
    _, _, seats = load_truth()
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(S, len(seats), len(PARTIES)))
    zr = rng.normal(size=(S, int(_region_index(seats.index).max()) + 1, len(PARTIES)))
    return z, zr


def tuning_table(variant: Variant, targets: Sequence[str], S: int = 2000, seed: int = 11) -> pd.DataFrame:
    """Share CRPS for every (target, setting) with the true national result."""
    _, national, _ = load_truth()
    rows = []
    for k, t in enumerate(targets):
        z, zr = _shocks(S, seed + k)
        nat = national[pd.Timestamp(t)].values[None, :].repeat(S, 0)
        for i, p in enumerate(variant.params_list):
            sc = share_scores(variant.sample(nat, t, p, z, zr), t)
            rows.append(dict(target=t, setting=i, share_crps=sc["share_crps"]))
    return pd.DataFrame(rows)


def loo_settings(table: pd.DataFrame, variant: Variant) -> Dict[str, Dict]:
    """For each target, the setting with the lowest mean share CRPS over the other targets."""
    piv = table.pivot(index="setting", columns="target", values="share_crps")
    out = {}
    for t in piv.columns:
        best = int(piv.drop(columns=t).mean(1).idxmin())
        out[t] = variant.params_list[best]
        for key, values in variant.grid.items():
            if len(values) > 1 and out[t][key] in (min(values), max(values)):
                # a choice on the edge of the grid means the search was cut short: widen the grid
                print(f"WARNING {variant.name}: {key}={out[t][key]} for {t} is on the edge of its grid {values}")
    return out


def stage_a(names: Sequence[str], targets: Sequence[str] = STAGE_A_TARGETS, S: int = 2000, seed: int = 11):
    _, national, _ = load_truth()
    rows, chosen = [], {}
    for name in names:
        v = VARIANTS[name]
        chosen[name] = loo_settings(tuning_table(v, targets, S, seed), v)
        for k, t in enumerate(targets):
            z, zr = _shocks(S, seed + k)
            nat = national[pd.Timestamp(t)].values[None, :].repeat(S, 0)
            dist = v.sample(nat, t, chosen[name][t], z, zr)
            sc = share_scores(dist, t)
            rows.append(dict(stage="A", variant=name, target=t, horizon=0, params=chosen[name][t],
                             **{k_: v_ for k_, v_ in sc.items()}, **seat_scores(dist, t)))
    return pd.DataFrame(rows), chosen


# ----------------------------------------------------------------------------------------
# Stage B: forecast national draws
# ----------------------------------------------------------------------------------------


def stage_b(names: Sequence[str], chosen: Dict[str, Dict], national_dir: str, fit: str = "rw_nobias",
            targets: Sequence[str] = STAGE_B_TARGETS, horizons: Sequence[int] = HORIZONS, seed: int = 23):
    rows = []
    for k, t in enumerate(targets):
        for h in horizons:
            path = os.path.join(national_dir, f"{fit}__{t}__h{h:03d}__national.npy")
            if not os.path.exists(path):
                print(f"missing {path}")
                continue
            nat = np.load(path).astype(float)
            nat = nat / nat.sum(-1, keepdims=True)
            nat = add_polling_error(nat, POLLING_TAU[h], np.random.default_rng(seed + 100 * k + h))
            z, zr = _shocks(nat.shape[0], seed + 1000 + 100 * k + h)
            for name in names:
                v = VARIANTS[name]
                dist = v.sample(nat, t, chosen[name][t], z, zr)
                sc = share_scores(dist, t)
                rows.append(dict(stage="B", variant=name, target=t, horizon=h, params=chosen[name][t],
                                 **sc, **seat_scores(dist, t)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------------------------


def sign_flip_p(diffs: np.ndarray, n_perm: int = 20000, seed: int = 5) -> float:
    """One-sided p that the mean of diffs (baseline minus candidate; positive = candidate better) is
    as large as observed if each cluster's sign were a coin flip."""
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return float(((flips * d[None]).mean(1) >= obs).mean())


def decide(a: pd.DataFrame, b: pd.DataFrame, cand: str, base: str = BASELINE,
           targets: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """The keep rule. `targets` restricts stage A to a subset (for a sensitivity line only)."""
    A = a.set_index(["variant", "target"])
    targets = sorted(targets if targets is not None else a.target.unique())
    da = np.array([A.loc[(base, t), "share_crps"] - A.loc[(cand, t), "share_crps"] for t in targets])
    gain = da / np.array([A.loc[(base, t), "share_crps"] for t in targets])
    clusters = []
    for t in targets:
        rb, rc = A.loc[(base, t), "region_crps"], A.loc[(cand, t), "region_crps"]
        clusters += [rb[r] - rc[r] for r in rb]
    p = sign_flip_p(np.array(clusters))
    cov_a = A.loc[cand, "share_cov90"].loc[targets]
    out = dict(
        candidate=cand,
        A_better_elections=int((da > 0).sum()),
        A_n_elections=len(targets),
        A_mean_gain_pct=round(float(gain.mean() * 100), 2),
        A_sign_flip_p=round(p, 4),
        A_cov90_mean=round(float(cov_a.mean()), 3),
        A_baseline_cov90_mean=round(float(A.loc[base, "share_cov90"].loc[targets].mean()), 3),
    )
    out["A_pass"] = bool(out["A_better_elections"] >= 4 and out["A_mean_gain_pct"] >= 3 and p < 0.05
                         and cov_a.mean() >= 0.80 and cov_a.mean() <= 0.97)
    if b is not None and len(b):
        B = b.set_index(["variant", "target", "horizon"])
        wins = {}
        worst_total = -np.inf
        cov_b = []
        for t in sorted(b.target.unique()):
            hs = [h for h in sorted(b.horizon.unique()) if (cand, t, h) in B.index and (base, t, h) in B.index]
            wins[t] = sum(B.loc[(cand, t, h), "dist_seat_rps"] < B.loc[(base, t, h), "dist_seat_rps"] for h in hs)
        for h in sorted(b.horizon.unique()):
            sb = b[(b.variant == base) & (b.horizon == h)].seat_total_rps.mean()
            sc = b[(b.variant == cand) & (b.horizon == h)].seat_total_rps.mean()
            worst_total = max(worst_total, (sc - sb) / sb * 100)
            cov_b.append(b[(b.variant == cand) & (b.horizon == h)].seat_total_cov90.mean())
        out.update(
            B_elections_better_at_2plus_horizons=int(sum(v >= 2 for v in wins.values())),
            B_n_elections=len(wins),
            B_worst_seat_total_rps_change_pct=round(float(worst_total), 2),
            B_seat_total_cov90_min=round(float(min(cov_b)), 3),
        )
        out["B_pass"] = bool(out["B_elections_better_at_2plus_horizons"] >= 3 and worst_total <= 1.0
                             and min(cov_b) >= 0.85)
        out["keep"] = bool(out["A_pass"] and out["B_pass"])
    return out


SCORE_COLS = ["share_crps", "share_cov90", "dist_mae", "dist_seat_rps", "seat_total_rps", "seat_total_cov90",
              "seats_misallocated"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="AB", choices=["A", "B", "AB"])
    ap.add_argument("--variants", nargs="+", default=["current", "sqrt", "sqrt_region"])
    ap.add_argument("--national-dir", default=None)
    ap.add_argument("--fit", default="rw_nobias")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    names = list(dict.fromkeys([BASELINE] + args.variants))
    a, chosen = stage_a(names, S=args.draws)
    print("\n== Stage A (true national), settings chosen leave-one-election-out")
    print(a.pivot(index="variant", columns="target", values="share_crps").round(4).to_string())
    print(a.groupby("variant", sort=False)[SCORE_COLS].mean().round(4).to_string())
    for n in names:
        print(n, {t[:4]: p for t, p in chosen[n].items()})
    b = None
    if "B" in args.stage and args.national_dir:
        b = stage_b(names, chosen, args.national_dir, args.fit)
        print("\n== Stage B (forecast national, %s + polling error)" % args.fit)
        print(b.pivot_table(index=["variant"], columns=["horizon", "target"], values="dist_seat_rps").round(2).to_string())
        print(b.groupby(["variant", "horizon"], sort=False)[SCORE_COLS].mean().round(4).to_string())
    print("\n== Stage A per election: district-seat RPS")
    print(a.pivot(index="variant", columns="target", values="dist_seat_rps").round(2).to_string())
    print("\n== Decision against", BASELINE)
    for n in names:
        if n != BASELINE:
            print(decide(a, b, n))
    # 2011 has one earlier election in the data (2009), so a noise calibrated on past elections has
    # nothing to learn from there: the same rule on 2015-2025 shows how much 2011 carries
    print("\n== Sensitivity: stage A on 2015-2025 only (not deciding)")
    for n in names:
        if n != BASELINE:
            d = decide(a, None, n, targets=[t for t in a.target.unique() if t >= "2015"])
            print({k: v for k, v in d.items() if k.startswith("A_") or k == "candidate"})
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        a.drop(columns=["region_crps", "region_weight"]).to_csv(os.path.join(args.out, "stage_a.csv"), index=False)
        if b is not None:
            b.drop(columns=["region_crps", "region_weight"]).to_csv(os.path.join(args.out, "stage_b.csv"), index=False)


if __name__ == "__main__":
    main()
