"""Survey odds ratios fitted to every parish's own margins (Johnston-Pattie entropy method), summed nationally.

Seed: the survey matrix (EEP raked, NEW row; or the Católica exit poll completed and raked), with an EXIT
column in which leaving the roll does not depend on the previous vote. Each parish's matrix is the IPF of
the seed to that parish's row (status at a, + NEW) and column (status at b, + EXIT) counts, so every parish
result is matched exactly and the survey supplies only the odds ratios. 90% intervals: survey bootstrap.
"""
import numpy as np, pandas as pd
from .survey import CATS, respondents, margins, estimate, PARISH
from .bounds import parish_pair, ROWS, COLS
from . import exitpoll as EPm

DEFF = 1.25


def ipf_parishes(seed, X, Y, iters=400):
    """seed R x C, X P x R, Y P x C -> P x R x C."""
    T = np.broadcast_to(seed, (X.shape[0],) + seed.shape).copy()
    T *= (X[:, :, None] > 0) & (Y[:, None, :] > 0)
    for _ in range(iters):
        rs = T.sum(2)
        T *= np.divide(X, rs, out=np.zeros_like(X), where=rs > 0)[:, :, None]
        cs = T.sum(1)
        T *= np.divide(Y, cs, out=np.zeros_like(Y), where=cs > 0)[:, None, :]
    return T


def seed_from(P, rows_p, cols_p):
    """Survey joint matrix (rows_p x cols_p, cols without EXIT) -> ROWS x COLS seed."""
    S = pd.DataFrame(0.0, ROWS, COLS)
    S.loc[rows_p, cols_p] = P
    if 'NEW' not in rows_p:  # exit poll: young voters behave like the recalled abstainers
        S.loc['NEW', cols_p] = S.loc['ABST', cols_p].values
    S['EXIT'] = S[CATS].sum(axis=1) * 0.02
    S.loc['NEW', 'EXIT'] = 0
    return S.values + 1e-9 * (S.values > -1)  # tiny floor keeps every allowed cell reachable


def national(T):
    F = T.sum(0)
    return F / F.sum(1, keepdims=True)


def eep_seeds(a, b, B, rng):
    df = respondents(a, b)
    u = df[df.usable].reset_index(drop=True)
    sa, sb, s_new = margins(a, b)
    cols = [k for k in CATS if sb[k] > 0]
    rows = CATS + ['NEW']
    rt = np.r_[sa.values * (1 - s_new), s_new]
    keep = [i for i in range(len(rows)) if rt[i] > 0]
    rows_k = [rows[i] for i in keep]
    r = rt[keep] / rt[keep].sum()
    c = sb[cols].values / sb[cols].sum()
    P0 = estimate(u, rows_k, cols, r, c)
    clustered = u.concelho.notna().all()
    groups = u.groupby('concelho').indices if clustered else None
    keys = list(groups) if clustered else None
    out = []
    for _ in range(B):
        if clustered:
            idx = np.concatenate([groups[keys[k]] for k in rng.choice(len(keys), len(keys))])
        else:
            idx = rng.integers(0, len(u), int(round(len(u) / DEFF)))
        out.append(estimate(u.iloc[idx], rows_k, cols, r, c))
    return P0, out, rows_k, cols, u.prev.value_counts()


def exitpoll_seeds(B, rng):
    C0, P0, _, _ = EPm.run()
    P0 = P0.values
    out = []
    comp = EPm.composition()
    _, sb, _ = margins(2024, 2025)
    voters = [k for k in CATS if k != 'ABST']
    vs = sb[voters] / sb[voters].sum()
    n = (vs * EPm_N / EP_DEFF).values
    orig = EPm.composition
    for _ in range(B):
        draw = comp.copy()
        for j, v in enumerate(voters):
            draw[v] = rng.dirichlet(comp[v].values * n[j] + 1e-3)
        EPm.composition = lambda d=draw: d
        try:
            _, Pb, _, _ = EPm.run()
        finally:
            EPm.composition = orig
        out.append(Pb.values)
    return P0, out, CATS, CATS


EPm_N, EP_DEFF = 35022, 10.0


REGION_NAME = {1: 'Norte', 2: 'Centro', 3: 'Lisboa e Vale do Tejo', 4: 'Alentejo', 5: 'Algarve'}


def by_area(T, labels):
    """T P x R x C, labels P -> dict area -> R x C share_of_from."""
    out = {}
    for lab in pd.unique(labels):
        F = T[labels == lab].sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[lab] = F / F.sum(1, keepdims=True), F.sum(1)
    return out


def run(B=200, seed=0):
    rng = np.random.default_rng(seed)
    res, areas = [], []
    p = pd.read_parquet(PARISH)
    for a, b in [(2015, 2019), (2019, 2022), (2022, 2024), (2024, 2025)]:
        X, Y, reg = parish_pair(a, b)
        dist = p[p.year == b].set_index('code').district.reindex(X.index).values
        regn = reg.map(REGION_NAME).values
        X, Y = X[ROWS].values.astype(float), Y[COLS].values.astype(float)
        if (a, b) == (2024, 2025):
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                P0, Pb, rows_p, cols_p = exitpoll_seeds(B, rng)
            src, nfrom = 'jp_exitpoll', None
        else:
            P0, Pb, rows_p, cols_p, nfrom = eep_seeds(a, b, B, rng)
            src = 'jp_eep'
        T0 = ipf_parishes(seed_from(P0, rows_p, cols_p), X, Y)
        est = national(T0)
        a0 = {**{('region', k): v for k, v in by_area(T0, regn).items()},
              **{('district', k): v for k, v in by_area(T0, dist).items()}}
        boots, aboots = [], {k: [] for k in a0}
        for pb in Pb:
            Tb = ipf_parishes(seed_from(pb, rows_p, cols_p), X, Y)
            boots.append(national(Tb))
            ab = {**{('region', k): v for k, v in by_area(Tb, regn).items()},
                  **{('district', k): v for k, v in by_area(Tb, dist).items()}}
            for k in aboots:
                aboots[k].append(ab[k][0])
        lo, hi = np.nanpercentile(np.stack(boots), [5, 95], axis=0)
        rowtot = X.sum(0)
        for i, fr in enumerate(ROWS):
            if rowtot[i] == 0:
                continue
            for j, to in enumerate(COLS):
                res.append(dict(pair=f'{a}-{b}', source=src, **{'from': fr}, to=to, share_of_from=est[i, j],
                                n_from=(int(nfrom.get(fr, 0)) if nfrom is not None else np.nan),
                                lo=lo[i, j], hi=hi[i, j]))
        for (lvl, area), (sh, rt) in a0.items():
            alo, ahi = np.nanpercentile(np.stack(aboots[(lvl, area)]), [5, 95], axis=0)
            for i, fr in enumerate(ROWS):
                if rt[i] == 0:
                    continue
                for j, to in enumerate(COLS):
                    areas.append(dict(pair=f'{a}-{b}', source=src, level=lvl, area=area, **{'from': fr}, to=to,
                                      share_of_from=sh[i, j], from_count=rt[i], lo=alo[i, j], hi=ahi[i, j]))
        print(a, b, 'done', flush=True)
    return pd.DataFrame(res), pd.DataFrame(areas)


def exitpoll_table(B=500, seed=7):
    """The Católica exit poll completed with the official margins and raked, with Dirichlet-bootstrap 90% intervals."""
    import contextlib, io
    rng = np.random.default_rng(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        P0, Pb, _, _ = exitpoll_seeds(B, rng)
    sh = lambda P: P / P.sum(1, keepdims=True)
    est = sh(P0)
    lo, hi = np.percentile(np.stack([sh(p) for p in Pb]), [5, 95], axis=0)
    return pd.DataFrame([dict(pair='2024-2025', source='exitpoll_catolica_raked', **{'from': fr}, to=to,
                              share_of_from=est[i, j], n_from=np.nan, lo=lo[i, j], hi=hi[i, j])
                         for i, fr in enumerate(CATS) for j, to in enumerate(CATS)])
