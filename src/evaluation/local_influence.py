"""Does local social influence add anything to blend swing? (pre-registered ABM test)

The specification and decision rule are in `docs/ABM_LOCAL_INFLUENCE_TEST.md`, committed before
this file was written. In short: the voter-model family of agent-based models predicts that a
party's share in a parish converges toward its neighbours' share (M1) and that gains spread to
the parishes next to where the party just gained (M2). Each signature is a correction on the
log share after parish-level blend swing; its coefficient is learned on earlier elections only
and scored, with the true national result given, on district error and seats.

    python -m src.evaluation.local_influence --all
    python -m src.evaluation.local_influence --checks learned oracle chega

Neighbours come from the parish polygons the website serves; set ESTIMADOR_WEB to the
estimador-web checkout (default ../estimador-web).
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.backtest import PARTIES, load_truth
from src.processing.district_swing import blend_swing, seats_from_district_shares

PARISH_ELECTIONS = ["2015-10-04", "2019-10-06", "2022-01-30", "2024-03-10", "2025-05-18"]
SMOOTH = 0.5  # pseudo-votes per party, so an empty cell has a log
BUFFER_DEG = 0.0003  # ~30 m: absorbs the web simplification of shared borders


# ----------------------------------------------------------------------------------------
# Neighbours
# ----------------------------------------------------------------------------------------


def parish_neighbours(web_dir: str, codes: List[str]) -> List[np.ndarray]:
    """Queen-contiguity neighbour indices for each parish code (buffered polygons)."""
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    geoms: Dict[str, object] = {}
    for f in glob.glob(os.path.join(web_dir, "public/data/population-geography/parishes/*.json")):
        with open(f) as fh:
            for feat in json.load(fh)["features"]:
                geoms[str(feat["properties"]["code"])] = shape(feat["geometry"]).buffer(0)
    missing = [c for c in codes if c not in geoms]
    if missing:
        raise ValueError(f"{len(missing)} parishes have no polygon, e.g. {missing[:5]}")
    polys = [geoms[c].buffer(BUFFER_DEG) for c in codes]
    tree = STRtree(polys)
    out = []
    for i, p in enumerate(polys):
        cand = tree.query(p, predicate="intersects")
        out.append(np.array(sorted(int(j) for j in cand if j != i), dtype=int))
    return out


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------


class Data:
    def __init__(self, web_dir: str):
        from src.data.multi_level_loaders import load_parish_level_results

        with contextlib.redirect_stdout(io.StringIO()):
            pr = load_parish_level_results(PARISH_ELECTIONS, PARTIES)
        pr["freguesia"] = pr["territory_code"].str.replace("LOCAL-", "")
        last = pr[pr.election_date == PARISH_ELECTIONS[-1]].set_index("freguesia")
        self.codes = sorted(last.index)
        self.district_name = last["district_name"].reindex(self.codes).to_numpy(dtype=object)
        self.municipality = last["full_municipality_id"].reindex(self.codes).to_numpy(dtype=object)
        self.votes = {
            pd.Timestamp(t): g.set_index("freguesia")[PARTIES].reindex(self.codes).fillna(0).astype(float).values
            for t, g in pr.groupby("election_date")
        }
        self.dates = sorted(self.votes)
        self.district, self.national, self.seats = load_truth()
        self.D = (self.district_name[:, None] == self.seats.index.to_numpy(dtype=object)[None, :]).astype(float)
        if (self.D.sum(1) != 1).any():
            raise ValueError("parish without a domestic district")
        self.nb = parish_neighbours(web_dir, self.codes)
        # sparse row-normalised adjacency as an index list, used for vote-pooled neighbour shares
        self.nb_rows = np.repeat(np.arange(len(self.codes)), [len(n) for n in self.nb])
        self.nb_cols = np.concatenate([n for n in self.nb]) if self.nb else np.array([], int)

    def prev(self, t, k=1):
        i = self.dates.index(t) - k
        return self.dates[i] if i >= 0 else None

    def tot(self, t):
        return self.votes[t].sum(1)

    def share(self, t, smooth=True):
        v = self.votes[t]
        if smooth:
            return (v + SMOOTH) / (v.sum(1, keepdims=True) + SMOOTH * v.shape[1])
        return v / np.clip(v.sum(1, keepdims=True), 1, None)

    def neighbour_share(self, t) -> np.ndarray:
        """Vote-pooled share of each parish's neighbours (own share where it has none)."""
        v = self.votes[t]
        pooled = np.zeros_like(v)
        np.add.at(pooled, self.nb_rows, v[self.nb_cols])
        has = pooled.sum(1) > 0
        s = (pooled + SMOOTH) / (pooled.sum(1, keepdims=True) + SMOOTH * v.shape[1])
        return np.where(has[:, None], s, self.share(t))


# ----------------------------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------------------------


def parish_blend(d: Data, t, last_parish: np.ndarray) -> np.ndarray:
    p = d.prev(t)
    return blend_swing(d.national[t].values[None], d.national[p].values, last_parish, 0.5)[0]


def denoised_last(d: Data, t) -> np.ndarray:
    """M0: t-1 shares shrunk toward the neighbour share by binomial sampling noise only."""
    p = d.prev(t)
    s = d.share(p, smooth=False)
    sn = d.neighbour_share(p)
    n = np.clip(d.tot(p), 1, None)[:, None]
    v = np.clip(sn * (1 - sn), 1e-6, None) / n  # sampling variance of the parish share
    w = n[:, 0]
    out = s.copy()
    for j in range(s.shape[1]):
        gap2 = (s[:, j] - sn[:, j]) ** 2
        tau2 = max(np.average(gap2 - v[:, j], weights=w), 1e-8)
        k = tau2 / (tau2 + v[:, j])
        out[:, j] = k * s[:, j] + (1 - k) * sn[:, j]
    out = np.where(d.share(p, smooth=False) <= 0, np.where(sn > 0.002, out, 0), out)
    return out / np.clip(out.sum(1, keepdims=True), 1e-12, None)


def features(d: Data, t) -> Dict[str, np.ndarray]:
    """(parishes, parties) predictors for target t; NaN where a party has no history."""
    p = d.prev(t)
    q = d.prev(t, 2)
    nat_p = d.national[p].values
    f = {}
    present_p = (nat_p > 1e-3)[None, :]
    f["gap"] = np.where(present_p, np.log(d.neighbour_share(p)) - np.log(d.share(p)), np.nan)
    if q is not None and q in d.votes:
        present_q = present_p & (d.national[q].values > 1e-3)[None, :]
        f["nb_change"] = np.where(present_q, np.log(d.neighbour_share(p)) - np.log(d.neighbour_share(q)), np.nan)
        f["own_change"] = np.where(present_q, np.log(d.share(p)) - np.log(d.share(q)), np.nan)
    return f


def residual(d: Data, t) -> np.ndarray:
    """log(true / parish blend) at t, (parishes, parties)."""
    b1 = parish_blend(d, t, d.share(d.prev(t), smooth=False))
    true = d.share(t)
    return np.log(true) - np.log(np.clip(b1, 1e-6, None))


MECHS = {"M1": ["gap"], "M2": ["nb_change", "own_change"], "M1+M2": ["gap", "nb_change", "own_change"]}


def stack(d: Data, targets: List, keys: List[str], weights: str = "parish"):
    """Rows (parish, party) of every target that has all `keys`; targets without them (no t-2
    election on the same map) are skipped. weights: "parish" = the parish's votes at t (the
    pre-registered rule); "party" = the party's expected votes there (votes x blend share),
    the inverse sampling variance of a log share (post hoc)."""
    X, y, w, cl, pj = [], [], [], [], []
    for t in targets:
        f = features(d, t)
        if any(k not in f for k in keys):
            continue
        r = residual(d, t)
        ok = np.isfinite(r)
        for k in keys:
            ok &= np.isfinite(f[k])
        ok &= (d.national[t].values > 1e-3)[None, :]
        if weights == "parish":
            wt = np.broadcast_to(d.tot(t)[:, None], r.shape)
        else:
            b1 = parish_blend(d, t, d.share(d.prev(t), smooth=False))
            wt = d.tot(t)[:, None] * np.clip(b1, 1e-4, None)
        X.append(np.column_stack([f[k][ok] for k in keys]))
        y.append(r[ok])
        w.append(wt[ok])
        cl.append(np.broadcast_to(d.municipality[:, None], r.shape)[ok])
        pj.append(np.broadcast_to(np.arange(r.shape[1])[None, :], r.shape)[ok])
    if not X:
        return None
    return np.vstack(X), np.concatenate(y), np.concatenate(w), np.concatenate(cl), np.concatenate(pj)


def fit(data, pooled: bool, n_parties: int):
    """Pooled: (keys,) coefficients. Party-specific: (parties, keys), a party never seen gets 0."""
    X, y, w, cl, pj = data
    if pooled:
        return wls(X, y, w, cl)
    B = np.zeros((n_parties, X.shape[1]))
    S = np.full((n_parties, X.shape[1]), np.nan)
    for j in range(n_parties):
        m = pj == j
        if m.sum() > X.shape[1] * 20:
            B[j], S[j] = wls(X[m], y[m], w[m], cl[m])
    return B, S


def wls(X, y, w, clusters) -> Tuple[np.ndarray, np.ndarray]:
    """Coefficients (no intercept: the national result already fixes the mean) and
    municipality-clustered standard errors."""
    sw = np.sqrt(w / w.mean())
    A = X * sw[:, None]
    b = y * sw
    XtX_inv = np.linalg.pinv(A.T @ A)
    beta = XtX_inv @ A.T @ b
    e = b - A @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(clusters):
        m = clusters == g
        s = A[m].T @ e[m]
        meat += np.outer(s, s)
    G = len(np.unique(clusters))
    cov = XtX_inv @ meat @ XtX_inv * G / max(G - 1, 1)
    return beta, np.sqrt(np.diag(cov))


def predict_parish(d: Data, t, mech: Optional[str], beta: Optional[np.ndarray], base: str = "raw") -> np.ndarray:
    last = d.share(d.prev(t), smooth=False) if base == "raw" else denoised_last(d, t)
    b1 = parish_blend(d, t, last)
    if mech is None:
        return b1
    f = features(d, t)
    corr = np.zeros_like(b1)
    beta = np.asarray(beta)
    for i, k in enumerate(MECHS[mech]):
        bk = beta[i] if beta.ndim == 1 else beta[:, i][None, :]
        corr += bk * np.nan_to_num(f[k], nan=0.0)
    x = np.where(b1 <= 1e-9, 0.0, np.clip(b1, 1e-9, None) * np.exp(corr))
    return x / np.clip(x.sum(1, keepdims=True), 1e-12, None)


def to_district(d: Data, t, parish_shares: np.ndarray) -> np.ndarray:
    W = d.D * d.tot(d.prev(t))[:, None]  # t-1 votes
    num = W.T @ parish_shares
    out = num / W.sum(0)[:, None]
    return out / out.sum(1, keepdims=True)


def score(d: Data, t, district_pred: np.ndarray, parish_pred: Optional[np.ndarray]) -> Dict[str, float]:
    true = d.district[t].values
    ts = seats_from_district_shares(true[None], d.seats.values)[0]
    ps = seats_from_district_shares(district_pred[None], d.seats.values)[0]
    out = dict(
        district_mae_pp=float(np.abs(district_pred - true).mean() * 100),
        seats_misallocated=float(np.abs(ps - ts).sum() / 2),
        national_seat_error=float(np.abs(ps.sum(0) - ts.sum(0)).sum() / 2),
    )
    if parish_pred is not None:
        tp = d.share(t, smooth=False)
        w = d.tot(t)
        out["parish_mae_pp"] = float(np.mean([np.average(np.abs(parish_pred[:, j] - tp[:, j]), weights=w) for j in range(tp.shape[1])]) * 100)
    return out


def b0(d: Data, t) -> np.ndarray:
    p = d.prev(t)
    return blend_swing(d.national[t].values[None], d.national[p].values, d.district[p].values, 0.5)[0]


# ----------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------


VARIANTS = {
    "pooled": dict(pooled=True, weights="parish"),  # pre-registered primary
    "party": dict(pooled=False, weights="parish"),  # pre-registered secondary
    "pooled, party-vote weights": dict(pooled=True, weights="party"),  # post hoc
    "party, party-vote weights": dict(pooled=False, weights="party"),  # post hoc
}


def run_rules(d: Data, oracle: bool, variants: List[str]) -> pd.DataFrame:
    rows = []
    targets = [pd.Timestamp(x) for x in PARISH_ELECTIONS[1:]]
    P = len(PARTIES)
    for t in targets:
        p1 = predict_parish(d, t, None, None)
        rows.append(dict(target=t.year, rule="B0 district blend", **score(d, t, b0(d, t), None)))
        rows.append(dict(target=t.year, rule="B1 parish blend", **score(d, t, to_district(d, t, p1), p1)))
        p0 = predict_parish(d, t, None, None, base="denoised")
        rows.append(dict(target=t.year, rule="M0 denoised", **score(d, t, to_district(d, t, p0), p0)))
        for v in variants:
            cfg = VARIANTS[v]
            for mech, keys in MECHS.items():
                fit_on = [t] if oracle else [s for s in targets if s < t]
                data = stack(d, fit_on, keys, cfg["weights"]) if fit_on else None
                if data is None or len(data[1]) == 0:
                    continue
                beta, se = fit(data, cfg["pooled"], P)
                pm = predict_parish(d, t, mech, beta)
                coef = (
                    ", ".join(f"{k} {b:+.3f} ({s:.3f})" for k, b, s in zip(keys, beta, se))
                    if cfg["pooled"]
                    else "; ".join(
                        f"{PARTIES[j]} " + ",".join(f"{b:+.2f}" for b in beta[j]) for j in range(P) if np.isfinite(se[j]).all()
                    )
                )
                used = sorted({s.year for s in fit_on if all(k in features(d, s) for k in keys)})
                rows.append(
                    dict(
                        target=t.year,
                        rule=f"{mech} [{v}]",
                        coef=coef,
                        fit_on="+".join(map(str, used)),
                        **score(d, t, to_district(d, t, pm), pm),
                    )
                )
    return pd.DataFrame(rows)


def chega_check(d: Data, min_voters: int = 1000) -> pd.DataFrame:
    """Chega's change in pp against its own and its neighbours' earlier share (parishes with at
    least 1,000 voters, district fixed effects, municipality-clustered SE). Descriptive only."""
    j = PARTIES.index("CH")
    rows = []
    for a, b in [("2019-10-06", "2022-01-30"), ("2022-01-30", "2024-03-10"), ("2024-03-10", "2025-05-18")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        big = d.tot(a) >= min_voters
        s_a = d.share(a, smooth=False)[:, j] * 100
        n_a = d.neighbour_share(a)[:, j] * 100
        dy = d.share(b, smooth=False)[:, j] * 100 - s_a
        dist = d.district_name
        X = np.column_stack([s_a, n_a])
        # within-district: demean by district (vote-weighted)
        w = d.tot(a)
        Xd, yd = X.copy(), dy.copy()
        for g in np.unique(dist[big]):
            m = big & (dist == g)
            Xd[m] -= np.average(X[m], axis=0, weights=w[m])
            yd[m] -= np.average(dy[m], weights=w[m])
        beta, se = wls(Xd[big], yd[big], w[big], d.municipality[big])
        rows.append(
            dict(
                change=f"{a.year}->{b.year}",
                min_voters=min_voters,
                parishes=int(big.sum()),
                own_share=f"{beta[0]:+.3f} ({se[0]:.3f})",
                neighbour_share=f"{beta[1]:+.3f} ({se[1]:.3f})",
            )
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Addendum: strategic desertion in small districts (district level, 2009-2025)
# ----------------------------------------------------------------------------------------


def viability(share: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    T = 0.75 / (magnitude + 1.0)
    return 1.0 / (1.0 + np.exp(-(share - T) / (0.2 * T)))


def strategic_rows(district, national, seats, t_prev, t):
    M = seats.values.astype(float)[:, None]
    last = district[t_prev].values
    pred = blend_swing(national[t].values[None], national[t_prev].values, last, 0.5)[0]
    dv = viability(pred, M) - viability(last, M)
    true = district[t].values
    y = np.log(np.clip(true, 1e-6, None)) - np.log(np.clip(pred, 1e-6, None))
    ok = (true > 1e-4) & (pred > 1e-4) & (national[t_prev].values > 1e-3)[None, :]
    return pred, dv, y, ok, np.broadcast_to(M, y.shape)


def strategic_check() -> pd.DataFrame:
    district, national, seats = load_truth()
    dates = sorted(district)
    pairs = list(zip(dates[:-1], dates[1:]))
    rows = []
    for k, (tp, t) in enumerate(pairs):
        if t.year < 2015:
            continue
        pred, dv, _, _, _ = strategic_rows(district, national, seats, tp, t)
        base = score_district(district, seats, t, pred)
        rows.append(dict(target=t.year, rule="B0 district blend", **base))
        for mode, fit_pairs in (("learned", pairs[:k]), ("oracle", [(tp, t)])):
            X, Y, W = [], [], []
            for a, b in fit_pairs:
                _, dva, ya, oka, Ma = strategic_rows(district, national, seats, a, b)
                X.append(dva[oka]); Y.append(ya[oka]); W.append(Ma[oka])
            X, Y, W = map(np.concatenate, (X, Y, W))
            theta = float((W * X * Y).sum() / max((W * X * X).sum(), 1e-12))
            resid = Y - theta * X
            se = float(np.sqrt((W * resid**2).sum() / W.sum() / max((W * X * X).sum() / W.sum(), 1e-12) / len(Y)))
            x = np.where(pred <= 1e-9, 0.0, pred * np.exp(theta * dv))
            x = x / x.sum(1, keepdims=True)
            rows.append(dict(target=t.year, rule=f"S {mode}", theta=f"{theta:+.3f} ({se:.3f})",
                             moved=int((np.abs(dv) > 0.1).sum()), **score_district(district, seats, t, x)))
    return pd.DataFrame(rows)


def score_district(district, seats, t, pred) -> Dict[str, float]:
    true = district[t].values
    ts = seats_from_district_shares(true[None], seats.values)[0]
    ps = seats_from_district_shares(pred[None], seats.values)[0]
    return dict(
        district_mae_pp=float(np.abs(pred - true).mean() * 100),
        seats_misallocated=float(np.abs(ps - ts).sum() / 2),
        national_seat_error=float(np.abs(ps.sum(0) - ts.sum(0)).sum() / 2),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checks", nargs="+", default=["learned", "oracle", "chega"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--web-dir", default=os.environ.get("ESTIMADOR_WEB", os.path.join("..", "estimador-web")))
    ap.add_argument("--out", default=None, help="write the tables as CSV into this folder")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    a = ap.parse_args(argv)
    checks = ["learned", "oracle", "chega", "strategic"] if a.all else a.checks
    out = {}
    if checks == ["strategic"]:
        df = strategic_check()
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(df.groupby("rule")[["district_mae_pp", "seats_misallocated", "national_seat_error"]].mean().round(3).to_string())
        return {"strategic": df}
    d = Data(a.web_dir)
    iso = sum(len(n) == 0 for n in d.nb)
    print(f"{len(d.codes)} parishes, mean {np.mean([len(n) for n in d.nb]):.1f} neighbours, {iso} with none")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)
    for c in checks:
        if c in ("learned", "oracle"):
            df = run_rules(d, oracle=(c == "oracle"), variants=a.variants)
            print(f"\n== {c} ==")
            print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            # means over the targets every rule has (M2 needs t-2, so learned M2 starts in 2024)
            for label, yrs in (("2022-2025", [2022, 2024, 2025]), ("2024-2025", [2024, 2025])):
                sub = df[df.target.isin(yrs)]
                n = sub.groupby("rule").target.nunique()
                summ = sub[sub.rule.isin(n[n == len(yrs)].index)].groupby("rule", sort=False)[
                    ["district_mae_pp", "seats_misallocated", "national_seat_error", "parish_mae_pp"]
                ].mean()
                print(f"-- mean over {label}")
                print(summ.round(3).to_string())
            out[c] = df
        elif c == "strategic":
            df = strategic_check()
            print("\n== strategic desertion (district level) ==")
            print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print(df.groupby("rule")[["district_mae_pp", "seats_misallocated", "national_seat_error"]].mean().round(3).to_string())
            out[c] = df
        elif c == "chega":
            df = pd.concat([chega_check(d, m) for m in (1000, 5000)], ignore_index=True)
            print("\n== chega (pp change on pp share, within district) ==")
            print(df.to_string(index=False))
            out[c] = df
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for k, df in out.items():
            df.to_csv(os.path.join(a.out, f"local_influence_{k}.csv"), index=False)
    return out


if __name__ == "__main__":
    main()
