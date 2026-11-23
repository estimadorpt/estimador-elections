"""Recall mismatch: two readings of the gap between recalled and official previous vote.

Undo winner recall (superseded by detilt_matrix below, kept for reference): the excess recall of the previous winner W is taken as respondents from the other
rows (not NEW), in proportion to their official size, who kept their own row's current-vote pattern but
reported W. Remove that mixture from row W, give it back to the rows, then rake to both margins."""
import numpy as np, pandas as pd
from .survey import CATS, WINNER, PARISH, respondents, margins, table, ipf, ipf_factors, PAIRS


def detilt_matrix(M, rows, r, c, W=None):
    """Misreport hypothesis. M: weighted counts (rows x cols, prior added). Rake columns only, so the current
    vote is right and the recalled vote is as reported. Every row reported above its official share (R > r)
    is taken to hold false claims, made by respondents from the rows reported below it (in proportion to the
    shortfall), who keep their own row's current-vote pattern. Move them back, then rake to both margins.
    (Plain raking is the selection hypothesis: the survey over-sampled that row's real voters.)"""
    cs = M.sum(0)
    Mc = M * np.divide(c, cs, out=np.zeros_like(c), where=cs > 0)[None, :]
    Mc = Mc / Mc.sum()
    R = Mc.sum(1)
    elig = np.array([k != 'NEW' and rr > 0 for k, rr in zip(rows, r)])
    over = np.where(elig & (R > r))[0]
    under = np.where(elig & (R < r))[0]
    if len(over) and len(under):
        q = Mc[under] / Mc[under].sum(1, keepdims=True)
        deficit = (r - R)[under]
        wt = deficit / deficit.sum()
        for w in over:
            F = (R[w] - r[w]) * (wt[:, None] * q)
            Fw = F.sum(0)
            F *= np.minimum(1.0, Mc[w] * 0.999 / np.maximum(Fw, 1e-15))[None, :]  # per column, never below 0
            Mc[w] -= F.sum(0)
            Mc[under] += F
    return ipf(Mc, r, c)


def estimate_detilt(df, rows, cols, r, c, W, prior=1.0):
    M = table(df, rows, cols) * (len(df) / df.w.sum())
    M = M + prior * c[None, :] / c.sum() * (r[:, None] > 0)
    M[:, c == 0] = 0
    M[r == 0, :] = 0
    return detilt_matrix(M, rows, r, c, W)


RESIDENTS = 7_845_912  # mainland resident Portuguese citizens 18+, Censos 2021


def ghost_share(b):
    p = pd.read_parquet(PARISH)
    reg = p[p.eep_region.between(1, 5) & (p.year == b)].registered.sum()
    return 1 - RESIDENTS / reg


def run(B=400, deff=1.25, seed=3, pairs=((2015, 2019), (2019, 2022), (2022, 2024))):
    """Resident universe: names on the roll beyond the census count of resident citizens never answer a survey,
    so they are held out as abstainers at both elections, the survey is raked (selection) or de-tilted then
    raked (misreport) among residents, and the held-out names are added back to abstainer->abstainer."""
    rng = np.random.default_rng(seed)
    out = []
    for a, b in pairs:
        u = respondents(a, b); u = u[u.usable].reset_index(drop=True)
        sa, sb, s_new = margins(a, b)
        g = ghost_share(b)
        cols = [k for k in CATS if sb[k] > 0]
        rows = CATS + ['NEW']
        rt = np.r_[sa.values * (1 - s_new), s_new]
        ia, ja = rows.index('ABST'), cols.index('ABST')
        keep = [i for i in range(len(rows)) if rt[i] > 0]
        rows_k = [rows[i] for i in keep]
        r = rt[keep] / rt[keep].sum(); c = sb[cols].values / sb[cols].sum()
        ia = rows_k.index('ABST')
        r_res = r.copy(); r_res[ia] -= g; r_res /= r_res.sum()
        c_res = c.copy(); c_res[ja] -= g; c_res /= c_res.sum()

        def back(P):
            P = P / P.sum() * (1 - g)
            P[ia, ja] += g
            return P

        def raw(d):
            M = table(d, rows_k, cols) * (len(d) / d.w.sum())
            M = M + 1.0 * c_res[None, :] / c_res.sum() * (r_res[:, None] > 0)
            M[:, c_res == 0] = 0; M[r_res == 0, :] = 0
            return M
        ests = {'eep_raked_res': lambda d: back(ipf(raw(d), r_res, c_res)),
                'eep_detilt_res': lambda d: back(detilt_matrix(raw(d), rows_k, r_res, c_res))}
        clustered = u.concelho.notna().all()
        groups = u.groupby('concelho').indices if clustered else None
        keys = list(groups) if clustered else None
        idxs = []
        for _ in range(B):
            if clustered:
                idxs.append(np.concatenate([groups[keys[k]] for k in rng.choice(len(keys), len(keys))]))
            else:
                idxs.append(rng.integers(0, len(u), int(round(len(u) / deff))))
        n_from = u.prev.value_counts().reindex(rows_k).fillna(0).astype(int).values
        for name, est in ests.items():
            P = est(u); sh = P / P.sum(1, keepdims=True)
            bs = np.stack([(lambda p: p / p.sum(1, keepdims=True))(est(u.iloc[ix])) for ix in idxs])
            lo, hi = np.nanpercentile(bs, [5, 95], axis=0)
            for i, fr in enumerate(rows_k):
                for j, to in enumerate(cols):
                    out.append(dict(pair=f'{a}-{b}', source=name, **{'from': fr}, to=to, share_of_from=sh[i, j],
                                    n_from=n_from[i], lo=lo[i, j], hi=hi[i, j], share_of_electorate=P[i, j]))
        print(a, b, 'ghost share', round(g, 3), flush=True)
    return pd.DataFrame(out)
