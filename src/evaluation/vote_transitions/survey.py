"""Recalled-vote transition matrices from the EEP, raked to official mainland margins.

Categories shared with the vote-moves thread: CH, PSD_CDS, IL, PS, LEFT (BE+CDU+L), PAN, OTHER,
BLANK_NULL, ABST, plus NEW (came of age between the two elections, identified by age).
Universe: mainland registered voters at the later election (b). Rows: status at a. Columns: vote at b.
"""
import os, sys
from functools import lru_cache
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA = os.path.join(ROOT, 'data')
EEP_DTA = os.environ.get(
    'EEP_DTA', os.path.join(ROOT, '..', 'estimador-data', 'data', 'surveys', 'archive', 'apis', 'EEP_2002-2024.dta'))
# parish results in the 9 shared categories, built by the vote-moves thread (one row per parish and election)
PARISH = os.environ.get('VOTE_MOVES_PARISH', '/mnt/project-files/vote-moves/parish_blocs_2015_2025.parquet')
EEP_COLS = ['ano', 'Q21_1', 'Q21_2', 'Q21_6', 'Q21_7', 'Q4_1', 'Q42_1', 'Q43', 'Q44_2', 'Q56_1', 'Q5_1']


@lru_cache(maxsize=1)
def eep_frame():
    """EEP 2002-2024 (ICS, non-commercial research licence): the columns these analyses read, as codes."""
    import pyreadstat
    df, _ = pyreadstat.read_dta(EEP_DTA, usecols=EEP_COLS)
    return df
CATS = ['CH', 'PSD_CDS', 'IL', 'PS', 'LEFT', 'PAN', 'OTHER', 'BLANK_NULL', 'ABST']
DATES = {2011: '2011-06-05', 2015: '2015-10-04', 2019: '2019-10-06', 2022: '2022-01-30', 2024: '2024-03-10',
         2025: '2025-05-18'}
PAIRS = [(2011, 2015), (2015, 2019), (2019, 2022), (2022, 2024)]
COHORT = 97_000  # Portuguese nationals reaching 18 per year, mainland (Censos 2021, ages 15-24 / 10)
PARTY = {13: 'CH', 2: 'PSD_CDS', 6: 'PSD_CDS', 11: 'PSD_CDS', 29: 'PSD_CDS', 32: 'PSD_CDS', 33: 'PSD_CDS',
         35: 'PSD_CDS', 36: 'PSD_CDS', 16: 'IL', 7: 'PS', 1: 'LEFT', 3: 'LEFT', 4: 'LEFT', 5: 'LEFT', 15: 'LEFT',
         14: 'PAN', 91: 'BLANK_NULL', 92: 'ABST'}
OTHER = {8, 9, 10, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 34, 37, 38, 90}


def gap_years(a, b):
    return (pd.Timestamp(DATES[b]) - pd.Timestamp(DATES[a])).days / 365.25


def code(v):
    if pd.isna(v):
        return None
    v = int(v)
    if v in PARTY:
        return PARTY[v]
    if v in OTHER:
        return 'OTHER'
    return None  # 93 none, 97 n/a, 98 DK, 99 NR


def respondents(a, b):
    """One row per EEP-b respondent with prev (status at a) and cur (vote at b); unusable rows dropped."""
    d = eep_frame()
    d = d[d.ano.astype(int) == b]
    d = d[d.Q56_1.between(1, 5)]  # mainland
    cur = np.where(d.Q21_1 == 5, d.Q21_2.map(code), np.where(d.Q21_1.isin([98, 99]) | d.Q21_1.isna(), None, 'ABST'))
    prev = np.where(d.Q21_6.isin([1, 2]), d.Q21_7.map(code),
                    np.where(d.Q21_6.isin([3, 4]), 'ABST', None))
    age = d.Q42_1.where(d.Q42_1.between(18, 110))
    new = (age + 0.35 - gap_years(a, b)) < 18
    prev = np.where(new, 'NEW', prev)
    out = pd.DataFrame({'prev': prev, 'cur': cur, 'w': d.Q4_1.values, 'female': d.Q43.map({1: 0, 2: 1}).values,
                        'age': age.values, 'edu': d.Q44_2.values, 'region': d.Q56_1.values,
                        'concelho': d.Q5_1.values})
    out['usable'] = out.prev.notna() & out.cur.notna()
    return out


def old_margins(y):
    """Mainland shares of registered voters for elections before the parish file (2009, 2011)."""
    d = pd.read_parquet(os.path.join(DATA, f'legislativas_{y}.parquet'))
    d = d[d.territoryCode.str.replace('LOCAL-', '').str[:2].astype(int) < 30].fillna(0)
    reg = d.subscribed_voters.sum()
    psd = sum(d[c].sum() for c in ['PPD/PSD', 'CDS-PP'] if c in d)
    left = sum(d[c].sum() for c in ['B.E.', 'PCP-PEV'] if c in d)
    pan = d['PAN'].sum() if 'PAN' in d else 0
    ps = d['PS'].sum()
    bn = d.null_votes.sum() + d.blank_votes.sum()
    voters = d.number_voters.sum()
    parties = [c for c in d.columns if c not in ['territoryCode', 'territoryName', 'number_voters',
                                                  'percentage_voters', 'subscribed_voters', 'null_votes',
                                                  'blank_votes']]
    other = sum(d[c].sum() for c in parties) - psd - left - pan - ps
    s = pd.Series({'CH': 0, 'PSD_CDS': psd, 'IL': 0, 'PS': ps, 'LEFT': left, 'PAN': pan, 'OTHER': other,
                   'BLANK_NULL': bn, 'ABST': reg - voters}) / reg
    return s, reg


def margins(a, b):
    p = pd.read_parquet(PARISH)
    p = p[p.eep_region.between(1, 5)]
    tot = p.groupby('year')[CATS + ['registered']].sum()
    def one(y):
        if y in tot.index:
            return tot.loc[y, CATS] / tot.loc[y, 'registered'], tot.loc[y, 'registered']
        return old_margins(y)
    sa, _ = one(a)
    sb, regb = one(b)
    s_new = COHORT * gap_years(a, b) / regb
    return sa.astype(float), sb.astype(float), s_new


def ipf(M, r, c, iters=500, tol=1e-10):
    M = M.copy()
    for _ in range(iters):
        rs = M.sum(1)
        M *= np.divide(r, rs, out=np.zeros_like(r), where=rs > 0)[:, None]
        cs = M.sum(0)
        M *= np.divide(c, cs, out=np.zeros_like(c), where=cs > 0)[None, :]
        if np.abs(M.sum(1) - r).max() < tol:
            break
    return M


def ipf_factors(M, r, c, iters=2000, tol=1e-12):
    """IPF returning the row and column factors: P = a[:, None] * M * b[None, :]."""
    a = np.ones(len(r)); b = np.ones(len(c))
    for _ in range(iters):
        rs = (M * b[None, :]).sum(1)
        a = np.divide(r, rs, out=np.zeros_like(r), where=rs > 0)
        cs = (a[:, None] * M).sum(0)
        b = np.divide(c, cs, out=np.zeros_like(c), where=cs > 0)
        if np.abs((a[:, None] * M * b[None, :]).sum(1) - r).max() < tol:
            break
    return a, b


def table(df, rows, cols):
    t = df.pivot_table(index='prev', columns='cur', values='w', aggfunc='sum').reindex(index=rows, columns=cols)
    return t.fillna(0).values


def estimate(df, rows, cols, r, c, prior=1.0):
    """Weighted counts + one pseudo-respondent per row spread by the column targets, raked to (r, c)."""
    M = table(df, rows, cols)
    M = M * (len(df) / df.w.sum())  # weights to respondent scale
    M = M + prior * c[None, :] / c.sum() * (r[:, None] > 0)
    M[:, c == 0] = 0
    M[r == 0, :] = 0
    return ipf(M, r, c)


WINNER = {2011: 'PSD_CDS', 2015: 'PSD_CDS', 2019: 'PS', 2022: 'PS', 2024: 'PSD_CDS'}


def estimate_winnerfix(df, rows, cols, r, c, W, prior=1.0):
    """Treat the excess recall of the previous winner W as false claims made by a random slice of every
    current-vote column: remove that slice from row W and give it back to the other rows (not NEW) in
    proportion to their official shares, then rake as usual."""
    M = table(df, rows, cols) * (len(df) / df.w.sum())
    M = M + prior * c[None, :] / c.sum() * (r[:, None] > 0)
    M[:, c == 0] = 0
    M[r == 0, :] = 0
    _, bf = ipf_factors(M, r, c)
    Mc = M * bf[None, :]  # column-raked: current vote right, recalled vote as reported
    Mc = Mc / Mc.sum()
    w = rows.index(W)
    excess = Mc[w].sum() - r[w]
    if excess > 0:
        colshare = Mc.sum(0) / Mc.sum()
        take = np.minimum(excess * colshare, Mc[w] * 0.999)
        Mc[w] -= take
        others = [i for i, k in enumerate(rows) if i != w and k != 'NEW' and r[i] > 0]
        wt = r[others] / r[others].sum()
        Mc[others] += wt[:, None] * take[None, :]
    return ipf(Mc, r, c)


def run_pair(a, b, B=1000, deff=None, seed=0):
    df = respondents(a, b)
    u = df[df.usable].reset_index(drop=True)
    sa, sb, s_new = margins(a, b)
    cols = [k for k in CATS if sb[k] > 0]
    variants = {
        'eep_raked': (CATS, lambda d: d.assign(prev=d.prev.replace('NEW', 'ABST')),
                      sa.values.copy()),
        'eep_raked_new': (CATS + ['NEW'], lambda d: d, np.r_[sa.values * (1 - s_new), s_new]),
        'eep_raked_new_winnerfix': (CATS + ['NEW'], lambda d: d, np.r_[sa.values * (1 - s_new), s_new]),
    }
    rng = np.random.default_rng(seed)
    clustered = u.concelho.notna().all()
    groups = u.groupby('concelho').indices if clustered else None
    m_boot = len(u) if (clustered or deff is None) else int(round(len(u) / deff))
    rows_out = []
    for name, (rows, f, rt) in variants.items():
        keep = [i for i, k in enumerate(rows) if rt[i] > 0]
        rows_k = [rows[i] for i in keep]
        r = rt[keep] / rt[keep].sum()
        c = sb[cols].values / sb[cols].sum()
        d0 = f(u)
        est = (lambda d: estimate_winnerfix(d, rows_k, cols, r, c, WINNER[a])) if name.endswith('winnerfix') \
            else (lambda d: estimate(d, rows_k, cols, r, c))
        P = est(d0)
        share = P / P.sum(1, keepdims=True)
        n_from = d0.prev.value_counts().reindex(rows_k).fillna(0).astype(int).values
        boots = np.empty((B,) + share.shape)
        for bi in range(B):
            if clustered:
                keys = list(groups)
                pick = rng.choice(len(keys), len(keys), replace=True)
                idx = np.concatenate([groups[keys[k]] for k in pick])
            else:
                idx = rng.integers(0, len(u), m_boot)
            db = d0.iloc[idx]
            Pb = est(db)
            boots[bi] = Pb / Pb.sum(1, keepdims=True)
        lo, hi = np.nanpercentile(boots, [5, 95], axis=0)
        for i, fr in enumerate(rows_k):
            for j, to in enumerate(cols):
                rows_out.append(dict(pair=f'{a}-{b}', source=name, **{'from': fr}, to=to,
                                     share_of_from=share[i, j], n_from=n_from[i], lo=lo[i, j], hi=hi[i, j],
                                     share_of_electorate=P[i, j]))
        # survey weights only (no raking), NEW kept, for transparency
        if name == 'eep_raked_new':
            M = table(d0, rows_k, cols)
            W = M / M.sum(1, keepdims=True)
            bw = np.empty((B,) + W.shape)
            for bi in range(B):
                if clustered:
                    keys = list(groups)
                    pick = rng.choice(len(keys), len(keys), replace=True)
                    idx = np.concatenate([groups[keys[k]] for k in pick])
                else:
                    idx = rng.integers(0, len(u), m_boot)
                Mb = table(d0.iloc[idx], rows_k, cols)
                with np.errstate(invalid='ignore', divide='ignore'):
                    bw[bi] = Mb / Mb.sum(1, keepdims=True)
            lo, hi = np.nanpercentile(bw, [5, 95], axis=0)
            for i, fr in enumerate(rows_k):
                for j, to in enumerate(cols):
                    rows_out.append(dict(pair=f'{a}-{b}', source='eep_weighted', **{'from': fr}, to=to,
                                         share_of_from=W[i, j], n_from=n_from[i], lo=lo[i, j], hi=hi[i, j],
                                         share_of_electorate=np.nan))
    return pd.DataFrame(rows_out), dict(n_all=len(df), n_usable=len(u), s_new=s_new, clustered=clustered)


def deff_2024(B=600, seed=1):
    """Design effect of the transition cells, EEP 2024: concelho-cluster vs simple bootstrap variance."""
    u = respondents(2022, 2024)
    u = u[u.usable].reset_index(drop=True)
    sa, sb, s_new = margins(2022, 2024)
    rows = CATS
    cols = CATS
    d0 = u.assign(prev=u.prev.replace('NEW', 'ABST'))
    r = sa.values / sa.values.sum(); c = sb.values / sb.values.sum()
    rng = np.random.default_rng(seed)
    groups = u.groupby('concelho').indices
    keys = list(groups)
    def boot(clustered):
        out = []
        for _ in range(B):
            if clustered:
                pick = rng.choice(len(keys), len(keys), replace=True)
                idx = np.concatenate([groups[keys[k]] for k in pick])
            else:
                idx = rng.integers(0, len(u), len(u))
            P = estimate(d0.iloc[idx], rows, cols, r, c)
            out.append(P / P.sum(1, keepdims=True))
        return np.array(out)
    vc, vs = boot(True).var(0), boot(False).var(0)
    big = vs > 1e-4
    ratio = vc[big] / vs[big]
    return np.median(ratio), np.percentile(ratio, [25, 75]), big.sum()
