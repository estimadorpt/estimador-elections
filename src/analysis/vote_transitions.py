"""Vote transitions between consecutive legislative elections from parish results.

Ecological inference of RxC transfer matrices (who in election a did what in election b) from
the 3,092 parishes, domestic only, with every registered voter in a category: the parties
grouped as CH, PSD_CDS (PSD, CDS and PPM in any combination), IL, PS, LEFT (BE, CDU, L), PAN,
OTHER, BLANK_NULL and ABST. Roll changes enter as a NEW row (net additions in the parish) and an
EXIT column (net removals), so both margins sum to the same total in every parish.

Estimators:
- ``lphom``: one matrix for the whole country, chosen to minimise the absolute error of the
  parishes' margins (a linear program, after Romero, Pavía and colleagues' lphom).
- ``unit_refine``: then, in each parish, the transport plan that reproduces that parish's two
  margins exactly and stays closest (L1, in votes) to the current matrix; the country matrix is
  the sum of the parish plans; repeated a few times (in the spirit of their nslphom). The result
  always lies inside the deterministic bounds.
- ``bounds``: Duncan-Davis bounds per parish, summed: what the margins alone prove.

Run: python -m src.analysis.vote_transitions --grouping parties --pairs 2024-2025 --out DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

CATS = ["CH", "PSD_CDS", "IL", "PS", "LEFT", "PAN", "OTHER", "BLANK_NULL", "ABST"]
ROWS = CATS + ["NEW"]
COLS = CATS + ["EXIT"]
PAIRS = [(2015, 2019), (2019, 2022), (2022, 2024), (2024, 2025)]
YEARS = [2015, 2019, 2022, 2024, 2025]

# Result columns of data/legislativas_{year}.parquet grouped into categories. Whatever is not
# listed goes to OTHER; blank and null votes to BLANK_NULL; registered minus voters to ABST.
PSD_CDS = ["PPD/PSD", "CDS-PP", "PPM", "CDS-PP.PPM", "PPD/PSD.CDS-PP", "PPD/PSD.CDS-PP.PPM"]
GROUPINGS = {
    "blocs": {"CH": ["CH"], "PSD_CDS": PSD_CDS, "IL": ["IL"], "PS": ["PS"],
              "LEFT": ["B.E.", "PCP-PEV", "L", "L/TDA"], "PAN": ["PAN"]},
    "parties": {"CH": ["CH"], "AD": PSD_CDS, "IL": ["IL"], "PS": ["PS"], "L": ["L", "L/TDA"],
                "CDU": ["PCP-PEV"], "BE": ["B.E."], "PAN": ["PAN"], "JPP": ["JPP"]},
}
# The survey thread's regions (EEP Q56_1, by district): 1 Norte, 2 Centro, 3 Lisboa e Vale do
# Tejo, 4 Alentejo, 5 Algarve, 6 Madeira, 7 Açores.
EEP_REGION = {"Aveiro": 2, "Beja": 4, "Braga": 1, "Bragança": 1, "Castelo Branco": 2, "Coimbra": 2,
              "Évora": 4, "Faro": 5, "Guarda": 2, "Leiria": 2, "Lisboa": 3, "Portalegre": 4,
              "Porto": 1, "Santarém": 3, "Setúbal": 3, "Viana do Castelo": 1, "Vila Real": 1,
              "Viseu": 2, "Madeira": 6, "Açores": 7}
_META = ["territoryCode", "territory_code", "territoryName", "territory_name", "number_voters",
         "percentage_voters", "subscribed_voters", "null_votes", "blank_votes"]


def build_blocs(data_dir: str | Path, grouping: str = "blocs", years=YEARS) -> pd.DataFrame:
    """Long table, one row per parish and election: code, year, district, municipality,
    eep_region, registered and the categories (which sum to registered). Domestic parishes only;
    the district and municipality come from the 2025 parish file's territory codes."""
    groups = GROUPINGS[grouping]
    cats = list(groups) + ["OTHER", "BLANK_NULL", "ABST"]
    out = []
    for y in years:
        d = pd.read_parquet(Path(data_dir) / f"legislativas_{y}.parquet")
        d = d.rename(columns={"territoryCode": "territory_code", "territoryName": "territory_name"})
        parties = [c for c in d.columns if c not in _META]
        r = pd.DataFrame({"code": d.territory_code.str.replace("LOCAL-", "", regex=False), "year": y,
                          "registered": d.subscribed_voters})
        used = []
        for g, cols in groups.items():
            cols = [c for c in cols if c in d.columns]
            used += cols
            r[g] = d[cols].fillna(0).sum(axis=1)
        r["OTHER"] = d[[c for c in parties if c not in used]].fillna(0).sum(axis=1)
        r["BLANK_NULL"] = d.null_votes + d.blank_votes
        r["ABST"] = d.subscribed_voters - d.number_voters
        assert (r[cats].sum(axis=1) - r.registered).abs().max() < 1, y
        out.append(r)
    df = pd.concat(out, ignore_index=True)
    for c in cats + ["registered"]:
        df[c] = df[c].round().astype(int)
    terr = pd.read_csv(Path(data_dir) / "territory_parishes_20250918.csv", dtype=str)
    terr = terr.assign(code=terr.territory_code_full.str.replace("LOCAL-", "", regex=False)).set_index("code")
    df["district"] = df.code.map(terr.district_name)
    df["municipality"] = df.code.map(terr.municipality_code.str.replace("LOCAL-", "", regex=False))
    df["municipality_name"] = df.code.map(terr.municipality_name)
    df["eep_region"] = df.district.map(EEP_REGION)
    assert df[["district", "municipality", "eep_region"]].notna().all().all()
    return df[["code", "year", "eep_region", "district", "municipality", "municipality_name", "registered"] + cats], cats


def pair_margins(blocs: pd.DataFrame, a: int, b: int, cats=CATS):
    """Parish margins for a pair of elections: X (n x J, election a + NEW), Y (n x K, election b +
    EXIT) and the parish metadata, in the same parish order."""
    A = blocs[blocs.year == a].set_index("code").sort_index()
    B = blocs[blocs.year == b].set_index("code").loc[A.index]
    diff = (B.registered - A.registered).to_numpy(float)
    X = np.c_[A[cats].to_numpy(float), np.clip(diff, 0, None)]
    Y = np.c_[B[cats].to_numpy(float), np.clip(-diff, 0, None)]
    assert np.allclose(X.sum(1), Y.sum(1))
    return X, Y, A.drop(columns=cats + ["year"])


def new_exit_zero(J: int, K: int) -> np.ndarray:
    """Structural zeros: a net addition to the roll (last row) cannot also be a net removal
    (last column)."""
    z = np.zeros((J, K), bool)
    z[J - 1, K - 1] = True
    return z


def _zero(zero, J, K):
    return np.zeros((J, K), bool) if zero is None else zero


def lphom(X: np.ndarray, Y: np.ndarray, zero: np.ndarray | None = None) -> np.ndarray:
    """One row-stochastic matrix P minimising sum_i,k |sum_j X_ij P_jk - Y_ik|."""
    n, J = X.shape
    K = Y.shape[1]
    nP = J * K
    # columns: P (J*K, row-major), e+ (n*K), e- (n*K)
    rows, cols, vals = [], [], []
    for k in range(K):
        for j in range(J):
            rows.append(np.arange(n) * K + k)
            cols.append(np.full(n, j * K + k))
            vals.append(X[:, j])
    r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
    eye = np.arange(n * K)
    A_fit = sp.csr_matrix(
        (np.concatenate([v, np.ones(n * K), -np.ones(n * K)]),
         (np.concatenate([r, eye, eye]), np.concatenate([c, nP + eye, nP + n * K + eye]))),
        shape=(n * K, nP + 2 * n * K))
    A_row = sp.csr_matrix((np.ones(nP), (np.repeat(np.arange(J), K), np.arange(nP))),
                          shape=(J, nP + 2 * n * K))
    A_eq = sp.vstack([A_fit, A_row]).tocsr()
    b_eq = np.concatenate([Y.reshape(-1), np.ones(J)])
    cost = np.concatenate([np.zeros(nP), np.ones(2 * n * K)])
    zero = _zero(zero, J, K).reshape(-1)
    bounds = [(0, 0) if z else (0, 1) for z in zero] + [(0, None)] * (2 * n * K)
    res = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[:nP].reshape(J, K)


def _unit_plan(x: np.ndarray, y: np.ndarray, P: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """Flows u (J x K) with row sums x and column sums y, closest in L1 to x_j P_jk."""
    J, K = P.shape
    target = (x[:, None] * P).reshape(-1)
    nU = J * K
    # columns: u, d+, d-  with u - d+ + d- = target
    A_dev = sp.hstack([sp.identity(nU), -sp.identity(nU), sp.identity(nU)])
    A_row = sp.hstack([sp.csr_matrix((np.ones(nU), (np.repeat(np.arange(J), K), np.arange(nU))), shape=(J, nU)),
                       sp.csr_matrix((J, 2 * nU))])
    A_col = sp.hstack([sp.csr_matrix((np.ones(nU), (np.tile(np.arange(K), J), np.arange(nU))), shape=(K, nU)),
                       sp.csr_matrix((K, 2 * nU))])
    A_eq = sp.vstack([A_dev, A_row, A_col]).tocsr()
    b_eq = np.concatenate([target, x, y])
    cost = np.concatenate([np.zeros(nU), np.ones(2 * nU)])
    ub = np.where(zero.reshape(-1), 0, np.inf)
    bounds = [(0, u) for u in ub] + [(0, None)] * (2 * nU)
    res = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:  # numerical trouble: fall back to the proportional plan
        return np.outer(x, y) / max(x.sum(), 1)
    return res.x[:nU].reshape(J, K)


def unit_refine(X: np.ndarray, Y: np.ndarray, P: np.ndarray, iters: int = 5, tol: float = 1e-4,
                zero: np.ndarray | None = None):
    """Parish plans that reproduce each parish's margins exactly, pulled towards the country
    matrix, which is re-estimated from their sum. Returns (P, flows n x J x K)."""
    zero = _zero(zero, *P.shape)
    for _ in range(iters):
        U = np.stack([_unit_plan(X[i], Y[i], P, zero) for i in range(len(X))])
        P_new = U.sum(0) / np.maximum(X.sum(0)[:, None], 1e-9)
        done = np.abs(P_new - P).max() < tol
        P = P_new
        if done:
            break
    return P, U


def ipf_plans(X: np.ndarray, Y: np.ndarray, P: np.ndarray, sweeps: int = 200,
              zero: np.ndarray | None = None) -> np.ndarray:
    """For every parish at once, the plan with margins x_i and y_i closest in KL divergence to
    the seed x_ij P_jk (iterative proportional fitting). Returns flows n x J x K."""
    U = X[:, :, None] * P[None]
    U = np.where(_zero(zero, *P.shape)[None], 0.0, U)
    for _ in range(sweeps):
        U *= (Y / np.maximum(U.sum(1), 1e-12))[:, None, :]
        U *= (X / np.maximum(U.sum(2), 1e-12))[:, :, None]
    return U


def ipf_em(X: np.ndarray, Y: np.ndarray, P: np.ndarray | None = None, iters: int = 200,
           tol: float = 1e-6, groups: np.ndarray | None = None, zero: np.ndarray | None = None,
           sweeps: int = 200):
    """Fixed point of: parish plans by IPF towards the current matrix, matrix re-estimated from
    the plans (an approximate EM for a homogeneous transition matrix; the entropy approach of
    Johnston and Hay). With ``groups`` (one label per parish) each group gets its own matrix.
    Returns (P for the country, flows, dict of group matrices)."""
    J, K = X.shape[1], Y.shape[1]
    if P is None:
        P = np.tile(Y.sum(0) / Y.sum(), (J, 1))
    labels = np.zeros(len(X), int) if groups is None else pd.factorize(groups)[0]
    G = labels.max() + 1
    PG = np.repeat(P[None], G, 0)
    for _ in range(iters):
        U = np.concatenate([ipf_plans(X[labels == g], Y[labels == g], PG[g], sweeps=sweeps, zero=zero) for g in range(G)])
        order = np.concatenate([np.where(labels == g)[0] for g in range(G)])
        U = U[np.argsort(order)]
        new = np.stack([U[labels == g].sum(0) / np.maximum(X[labels == g].sum(0)[:, None], 1e-9) for g in range(G)])
        done = np.abs(new - PG).max() < tol
        PG = new
        if done:
            break
    P = U.sum(0) / np.maximum(X.sum(0)[:, None], 1e-9)
    return P, U, PG


def bounds(X: np.ndarray, Y: np.ndarray):
    """Deterministic (Duncan-Davis) bounds on each country cell, as shares of the row."""
    N = X.sum(1)[:, None, None]
    lo = np.clip(X[:, :, None] + Y[:, None, :] - N, 0, None).sum(0)
    hi = np.minimum(X[:, :, None], Y[:, None, :]).sum(0)
    tot = np.maximum(X.sum(0)[:, None], 1e-9)
    return lo / tot, hi / tot


def flow_table(U: np.ndarray, rows, cols, pair: str, source: str, level: str, labels=None,
               lo=None, hi=None) -> pd.DataFrame:
    """Long table of flows summed over parishes (by ``labels`` if given): pair, source, level,
    area, from, to, votes, n_from, share_of_from, lo, hi (the format shared with the survey
    thread, plus area and votes)."""
    labels = np.full(len(U), "Portugal") if labels is None else np.asarray(labels)
    out = []
    for area in pd.unique(labels):
        F = U[labels == area].sum(0)
        n = F.sum(1)
        for j, r in enumerate(rows):
            for k, c in enumerate(cols):
                out.append({"pair": pair, "source": source, "level": level, "area": area, "from": r,
                            "to": c, "votes": F[j, k], "n_from": n[j],
                            "share_of_from": F[j, k] / n[j] if n[j] > 0 else np.nan,
                            "lo": np.nan if lo is None else lo[j, k],
                            "hi": np.nan if hi is None else hi[j, k]})
    return pd.DataFrame(out)


def run_pair(blocs: pd.DataFrame, cats, a: int, b: int, method: str = "ipf_region"):
    """Parish flows (n x J x K) for one pair with one estimator, plus the parish metadata and
    the country bounds."""
    X, Y, meta = pair_margins(blocs, a, b, cats)
    z = new_exit_zero(X.shape[1], Y.shape[1])
    if method == "ipf_region":
        _, U, _ = ipf_em(X, Y, iters=300, groups=meta.eep_region.to_numpy(), zero=z)
    elif method == "ipf_national":
        _, U, _ = ipf_em(X, Y, iters=300, zero=z)
    elif method == "lp_unit":
        _, U = unit_refine(X, Y, lphom(X, Y, z), iters=5, zero=z)
    else:
        raise ValueError(method)
    lo, hi = bounds(X, Y)
    return U, meta, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--grouping", default="blocs", choices=list(GROUPINGS))
    ap.add_argument("--pairs", default=",".join(f"{a}-{b}" for a, b in PAIRS))
    ap.add_argument("--methods", default="ipf_region,ipf_national")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    blocs, cats = build_blocs(args.data_dir, args.grouping)
    rows, cols = cats + ["NEW"], cats + ["EXIT"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tables = []
    for pair in args.pairs.split(","):
        a, b = map(int, pair.split("-"))
        for method in args.methods.split(","):
            U, meta, lo, hi = run_pair(blocs, cats, a, b, method)
            src = f"ei_{method}"
            tables += [flow_table(U, rows, cols, pair, src, "country", lo=lo, hi=hi),
                       flow_table(U, rows, cols, pair, src, "district", meta.district.to_numpy()),
                       flow_table(U, rows, cols, pair, src, "eep_region", meta.eep_region.astype(str).to_numpy())]
            np.save(out / f"flows_{args.grouping}_{pair}_{method}.npy", U.astype(np.float32))
            meta.reset_index().to_parquet(out / f"flows_{args.grouping}_{pair}_parishes.parquet")
            print(pair, method, "done", flush=True)
    pd.concat(tables).to_csv(out / f"transitions_{args.grouping}.csv", index=False)


if __name__ == "__main__":
    main()
