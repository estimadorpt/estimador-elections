"""Can INE census demographics improve blend swing's district shares? (ecological checks)

Every check takes the true national result as given (like `backtest --district-oracle`), so it
measures only the district step. Scores are the district oracle's: mean absolute error over
the 20 domestic districts x 8 parties (pp), district seats given to the wrong party, and the
error in national seat totals. 2015-2025 share one parish map (3,092 parishes, CAOP 2013),
which is also the map of the Censos 2021 tables, so every check works on parishes and
aggregates to districts with the last election's votes.

1. `gate`: in-sample, how much of blend's parish residual (true minus blend, pp) census
   covariates explain, against district dummies alone.
2. `learned`: the real forecast. A correction X_i * gamma_p fitted on earlier election pairs
   (ridge, parish level) is added to the target election's blend. Forms: "static" (same
   residual pattern again) and "scaled" (pattern times the party's national log swing).
3. `oracle`: the ceiling. Group swings (what blend missed, per sex x age x education cell)
   fitted on the target election's own parishes, leaving each district out, applied through
   that district's census composition. Ridge shrinks the swings toward a plausible width.
4. `exitpoll`: the Católica 2025 exit poll's individual vote rates by sex x age and by
   education (ERC 2025062), applied to census composition: how much district geography do
   real group differences imply?

    python -m src.evaluation.ecological --covariates parish_covariates.parquet --ine-dir INE_DIR
    (build the covariates with `python -m src.evaluation.ecological_covariates`)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.backtest import PARTIES, load_truth
from src.evaluation.ecological_covariates import AGE_ORDER, shares
from src.processing.district_swing import blend_swing, seats_from_district_shares

PARISH_ELECTIONS = ["2015-10-04", "2019-10-06", "2022-01-30", "2024-03-10", "2025-05-18"]


class Data:
    def __init__(self, covariates_path: str, ine_dir: str):
        from src.data.multi_level_loaders import load_parish_level_results

        with contextlib.redirect_stdout(io.StringIO()):
            pr = load_parish_level_results(PARISH_ELECTIONS, PARTIES)
        pr["freguesia"] = pr["territory_code"].str.replace("LOCAL-", "")
        self.cov = pd.read_parquet(covariates_path)
        geo = pr[pr.election_date == PARISH_ELECTIONS[-1]].set_index("freguesia")["district_name"]
        self.cov["district_name"] = geo.reindex(self.cov.index)
        self.X = shares(self.cov)
        self.votes = {
            pd.Timestamp(t): g.set_index("freguesia")[PARTIES].reindex(self.cov.index).fillna(0).astype(float)
            for t, g in pr.groupby("election_date")
        }
        self.dates = sorted(self.votes)
        self.district, self.national, self.seats = load_truth()
        names = self.cov["district_name"].to_numpy(dtype=object)
        self.D = (names[:, None] == self.seats.index.to_numpy(dtype=object)[None, :]).astype(float)
        self.dist_idx = self.D.argmax(1)
        self.ine_dir = ine_dir

    def parish(self, t):
        v = self.votes[t]
        return v.div(v.sum(axis=1).clip(lower=1), axis=0).values, v.sum(axis=1).values

    def prev(self, E):
        return self.dates[self.dates.index(E) - 1]

    def group_composition(self, keys: List[str]) -> np.ndarray:
        """Parish shares of the 20+ population in sex / age4 / edu3 cells (Censos 2021)."""
        ed = pd.read_parquet(os.path.join(self.ine_dir, "person_sex_age15_educ_2021.parquet"))
        ed["ai"] = ed["age_15grp"].map({a: i for i, a in enumerate(AGE_ORDER)})
        ed = ed[ed.ai >= 1]
        ed["age4"] = pd.cut(ed.ai, [0, 3, 6, 9, 13], labels=["20-34", "35-49", "50-64", "65+"])
        ed["edu3"] = ed.education_level_code.astype(str).map(
            lambda c: "tertiary" if c.startswith("5") else ("secondary" if c in ("3", "4") else "basic")
        )
        g = ed.groupby(["freguesia"] + keys, observed=True).value.sum().unstack(keys, fill_value=0)
        g = g.reindex(self.cov.index).fillna(0)
        return g.div(g.sum(axis=1).clip(lower=1), axis=0).values


def score(d: Data, E, pred: np.ndarray) -> Dict[str, float]:
    true = d.district[E].values
    ts = seats_from_district_shares(true[None], d.seats.values)[0]
    ps = seats_from_district_shares(pred[None], d.seats.values)[0]
    return dict(
        district_mae_pp=float(np.abs(pred - true).mean() * 100),
        district_seats_misallocated=float(np.abs(ps - ts).sum() / 2),
        national_seat_error=float(np.abs(ps.sum(0) - ts.sum(0)).sum() / 2),
    )


def blend_district(d: Data, E) -> np.ndarray:
    p = d.prev(E)
    return blend_swing(d.national[E].values[None], d.national[p].values, d.district[p].values, 0.5)[0]


def corrected(d: Data, E, corr_parish: Optional[np.ndarray]) -> np.ndarray:
    """District blend plus the last-election-vote-weighted mean of a parish correction (shares)."""
    base = blend_district(d, E)
    if corr_parish is None:
        return base
    _, wp = d.parish(d.prev(E))
    W = d.D * wp[:, None]
    c = W.T @ corr_parish / W.sum(0)[:, None]
    pred = np.where(base <= 1e-6, 0.0, np.clip(base + c, 1e-6, None))
    return pred / pred.sum(axis=1, keepdims=True)


def parish_residual(d: Data, a, b) -> np.ndarray:
    """true minus parish-level blend, pp, (parishes, parties)."""
    sa, _ = d.parish(a)
    sb, _ = d.parish(b)
    return (sb - blend_swing(d.national[b].values[None], d.national[a].values, sa, 0.5)[0]) * 100


def _std(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    mu = np.average(Z, axis=0, weights=w)
    sd = np.sqrt(np.average((Z - mu) ** 2, axis=0, weights=w))
    return (Z - mu) / np.where(sd > 0, sd, 1)


def _ridge(X, y, w, lam):
    sw = np.sqrt(w / w.mean())
    A = X * sw[:, None]
    return np.linalg.solve(A.T @ A + lam * len(y) * np.eye(X.shape[1]), A.T @ (y * sw))


def _wls_r2(M, y, w):
    sw = np.sqrt(w)
    A = np.column_stack([np.ones(len(y)), M]) * sw[:, None]
    b, *_ = np.linalg.lstsq(A, y * sw, rcond=None)
    ybar = np.average(y, weights=w)
    return 1 - ((y * sw - A @ b) ** 2).sum() / (w * (y - ybar) ** 2).sum()


# ----------------------------------------------------------------------------------------


def gate(d: Data) -> pd.DataFrame:
    rows = []
    Xd = pd.get_dummies(d.cov["district_name"]).astype(float).values[:, 1:]
    for a, b in zip(d.dates[:-1], d.dates[1:]):
        res = parish_residual(d, a, b)
        _, w = d.parish(b)
        X = _std(d.X.values, w)
        for j, p in enumerate(PARTIES):
            if d.national[b][p] < 0.005:
                continue
            y = res[:, j]
            rows.append(
                dict(
                    pair=f"{a.year}->{b.year}",
                    party=p,
                    residual_sd_pp=np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w)),
                    r2_district_dummies=_wls_r2(Xd, y, w),
                    r2_census=_wls_r2(X, y, w),
                )
            )
    return pd.DataFrame(rows)


def learned(d: Data, lams=(1.0, 10.0, 30.0)) -> pd.DataFrame:
    rows = []
    cols = list(d.X.columns)
    for E in d.dates[2:]:
        k = d.dates.index(E)
        p0 = d.prev(E)
        _, wp = d.parish(p0)
        Xs = _std(d.X[cols].values, wp)
        rows.append(dict(target=E.year, model="blend", **score(d, E, blend_district(d, E))))
        train = [(d.dates[i - 1], d.dates[i]) for i in range(1, k)]
        for form in ("static", "scaled_log"):
            for lam in lams:
                corr = np.zeros((len(Xs), len(PARTIES)))
                for j, p in enumerate(PARTIES):
                    Ys, Zs, Ws = [], [], []
                    for a, b in train:
                        if d.national[b][p] < 0.005 or (form != "static" and d.national[a][p] < 0.005):
                            continue
                        u = 1.0 if form == "static" else np.log(d.national[b][p] / d.national[a][p])
                        _, wa = d.parish(a)
                        _, wb = d.parish(b)
                        Ys.append(parish_residual(d, a, b)[:, j])
                        Zs.append(_std(d.X[cols].values, wa) * u)
                        Ws.append(wb)
                    if not Ys or d.national[E][p] < 0.005 or (form != "static" and d.national[p0][p] < 0.005):
                        continue
                    g = _ridge(np.vstack(Zs), np.concatenate(Ys), np.concatenate(Ws), lam)
                    u = 1.0 if form == "static" else np.log(d.national[E][p] / d.national[p0][p])
                    corr[:, j] = Xs @ g * u / 100
                rows.append(dict(target=E.year, model=f"learned_{form}_ridge{lam:g}", **score(d, E, corrected(d, E, corr))))
    return pd.DataFrame(rows)


def oracle(d: Data, groupings=(("age4",), ("sex", "age4"), ("age4", "edu3"), ("sex", "age4", "edu3")),
           lams=(1e-9, 1e-4, 1e-3, 1e-2)) -> pd.DataFrame:
    """Leave-one-district-out group swings fitted on the target election (an upper bound)."""
    rows = []
    comps = {"x".join(g): d.group_composition(list(g)) for g in groupings}
    for E in d.dates[1:]:
        p0 = d.prev(E)
        sp, wp = d.parish(p0)
        se, we = d.parish(E)
        pb = blend_swing(d.national[E].values[None], d.national[p0].values, sp, 0.5)[0]
        rows.append(dict(target=E.year, groups="-", ridge=0, model="blend", **score(d, E, blend_district(d, E))))
        for name, G in comps.items():
            gbar = np.average(G, axis=0, weights=wp)
            Gc = G - gbar
            for lam in lams:
                corr = np.zeros((len(G), len(PARTIES)))
                widths = []
                for j in range(len(PARTIES)):
                    y = (se[:, j] - pb[:, j]) * 100
                    for dd in range(len(d.seats)):
                        o = d.dist_idx != dd
                        Go = G[o] - np.average(G[o], axis=0, weights=we[o])
                        yo = y[o] - np.average(y[o], weights=we[o])
                        b = _ridge(Go, yo, we[o], lam)
                        corr[~o, j] = Gc[~o] @ b / 100
                    ball = _ridge(G - np.average(G, axis=0, weights=we), y - np.average(y, weights=we), we, lam)
                    widths.append(np.sqrt(np.average(ball**2, weights=gbar)))
                rows.append(
                    dict(target=E.year, groups=name, ridge=lam, model="oracle_lodo",
                         swing_width_pp_PS=widths[0], swing_width_pp_CH=widths[1], swing_width_pp_AD=widths[7],
                         **score(d, E, corrected(d, E, corr)))
                )
    return pd.DataFrame(rows)


# Católica / CESOP exit poll, 18 May 2025 (ERC deposit 2025062), vote % among all voters.
EXIT_2025_SEX_AGE = {  # women "M", men "H"; ages 18-24, 25-34, 35-44, 45-54, 55-64, 65+
    "M": {"AD": [29, 30, 27, 35, 41, 37], "PS": [14, 11, 19, 23, 27, 42], "CH": [19, 27, 29, 21, 16, 8],
          "IL": [7, 10, 7, 4, 2, 1], "L": [12, 9, 5, 3, 2, 1], "CDU": [2, 2, 2, 3, 3, 3],
          "BE": [7, 4, 3, 2, 1, 1], "PAN": [3, 3, 3, 2, 1, 0.4]},
    "H": {"AD": [28, 26, 24, 32, 36, 36], "PS": [9, 8, 13, 20, 28, 39], "CH": [31, 36, 39, 28, 22, 12],
          "IL": [14, 14, 9, 6, 3, 1], "L": [6, 6, 5, 3, 3, 1], "CDU": [1, 2, 2, 3, 3, 6],
          "BE": [2, 2, 2, 1, 1, 1], "PAN": [1, 1, 1, 1, 0.4, 0.4]},
}
EXIT_2025_EDU = {  # not completed secondary, secondary, tertiary
    "AD": [31, 28, 39], "PS": [32, 21, 19], "CH": [24, 31, 14], "IL": [1, 5, 9],
    "L": [1, 4, 7], "CDU": [3, 3, 3], "BE": [1, 2, 3], "PAN": [1, 1, 2],
}


def exitpoll(d: Data) -> pd.DataFrame:
    """District deviations implied by individual-level 2025 rates x census composition."""
    E = pd.Timestamp("2025-05-18")
    _, we = d.parish(E)
    ed = pd.read_parquet(os.path.join(d.ine_dir, "person_sex_age15_educ_2021.parquet"))
    ed["ai"] = ed["age_15grp"].map({a: i for i, a in enumerate(AGE_ORDER)})
    ed["v"] = ed.value * np.where(ed.ai == 0, 0.4, 1.0)  # 18-19 = 2/5 of 15-19
    ed["a6"] = pd.cut(ed.ai, [-1, 1, 3, 5, 7, 9, 13], labels=False)
    cells = [(s, a) for s in ("M", "H") for a in range(6)]
    comp = ed.groupby(["freguesia", "sex", "a6"]).v.sum().unstack(["sex", "a6"], fill_value=0)[cells]
    comp = comp.reindex(d.cov.index).fillna(0)
    rate = np.array([[EXIT_2025_SEX_AGE[s][p][a] for p in PARTIES] for s, a in cells], float)
    edu = d.cov[["edu_basic", "edu_secondary", "edu_tertiary"]].values
    rate_e = np.array([[EXIT_2025_EDU[p][k] for p in PARTIES] for k in range(3)], float)
    W = d.D * we[:, None]
    true = d.district[E].values
    rows = []
    for name, C, R in (("sex x age", comp.values, rate), ("education", edu, rate_e)):
        C = C / C.sum(axis=1, keepdims=True)
        R = R / R.sum(axis=1, keepdims=True)
        pred = (W.T @ C / W.sum(0)[:, None]) @ R
        pred = pred / pred.sum(axis=1, keepdims=True)
        for j, p in enumerate(PARTIES):
            dt = (true[:, j] - d.national[E].values[j]) * 100
            dc = (pred[:, j] - pred[:, j].mean()) * 100
            rows.append(dict(composition=name, party=p, true_district_sd_pp=dt.std(), implied_district_sd_pp=dc.std(),
                             correlation=np.corrcoef(dt, dc)[0, 1]))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--ine-dir", required=True)
    ap.add_argument("--out", default=None, help="directory for the CSV tables")
    ap.add_argument("--checks", nargs="+", default=["gate", "learned", "oracle", "exitpoll"])
    args = ap.parse_args(argv)
    d = Data(args.covariates, args.ine_dir)
    pd.set_option("display.width", 200)
    for name in args.checks:
        df = {"gate": gate, "learned": learned, "oracle": oracle, "exitpoll": exitpoll}[name](d)
        print(f"\n== {name}")
        if name in ("learned", "oracle"):
            keys = ["model"] if name == "learned" else ["model", "groups", "ridge"]
            num = [c for c in df.columns if c not in keys + ["target"]]
            print(df.groupby(keys, sort=False)[num].mean().round(3).to_string())
            print(df.pivot_table(index=keys, columns="target", values="district_mae_pp", sort=False).round(3).to_string())
        else:
            print(df.round(2).to_string(index=False))
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            df.to_csv(os.path.join(args.out, f"ecological_{name}.csv"), index=False)


if __name__ == "__main__":
    main()
