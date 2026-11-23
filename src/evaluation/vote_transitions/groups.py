"""Vote by sex, age, education and region across EEP waves, raked to the official mainland result.

Each wave's respondents (mainland, vote or abstention known) get their sociodemographic weight times a
factor that makes the weighted 9-category distribution equal the official mainland shares of registered
voters. Group shares are then weighted means. 90% intervals: bootstrap (concelho clusters in 2024,
n/deff draws elsewhere), re-raking each draw.
"""
import numpy as np, pandas as pd
from .survey import CATS, code, eep_frame, old_margins as _old_margins, PARISH

WAVES = [2009, 2011, 2015, 2019, 2022, 2024]
AGE_EDGES = [18, 25, 35, 45, 55, 65, 75, 200]
AGE_LABELS = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
REGION = {1: 'Norte', 2: 'Centro', 3: 'Lisboa e Vale do Tejo', 4: 'Alentejo', 5: 'Algarve'}
EDU = {1: 'basic', 2: 'secondary', 3: 'tertiary'}
SHOW = ['CH', 'PSD_CDS', 'PS', 'IL', 'LEFT', 'ABST']
DEFF = 1.25


def old_margins(y):
    return _old_margins(y)[0]


def margins_all():
    p = pd.read_parquet(PARISH)
    p = p[p.eep_region.between(1, 5)]
    tot = p.groupby('year')[CATS + ['registered']].sum()
    out = {y: tot.loc[y, CATS] / tot.loc[y, 'registered'] for y in tot.index}
    for y in [2009, 2011]:
        out[y] = old_margins(y)
    return out


def wave(y):
    d = eep_frame()
    d = d[(d.ano.astype(int) == y) & d.Q56_1.between(1, 5)]
    cur = np.where(d.Q21_1 == 5, d.Q21_2.map(code), np.where(d.Q21_1.isin([98, 99]) | d.Q21_1.isna(), None, 'ABST'))
    age = d.Q42_1.where(d.Q42_1.between(18, 110))
    out = pd.DataFrame({'cur': cur, 'w': d.Q4_1.values,
                        'sex': d.Q43.map({1: 'men', 2: 'women'}).values,
                        'age': pd.cut(age, AGE_EDGES, right=False, labels=AGE_LABELS).astype(str).values,
                        'edu': d.Q44_2.map(EDU).values, 'region': d.Q56_1.map(REGION).values,
                        'concelho': d.Q5_1.values})
    return out[out.cur.notna()].reset_index(drop=True)


def rake(df, target):
    ws = df.groupby('cur').w.sum()
    ws = ws / ws.sum()
    f = (target / ws).reindex(df.cur).values
    return df.w.values * np.nan_to_num(f)


DIMS = [('sex',), ('age',), ('edu',), ('region',), ('age', 'edu'), ('sex', 'age')]


def tabulate(df, w):
    rows = {}
    for dims in DIMS:
        g = df.assign(w=w).groupby(list(dims) + ['cur']).w.sum().unstack('cur').reindex(columns=CATS).fillna(0)
        g = g.div(g.sum(1), axis=0)
        for key, r in g.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            if any(k in ('nan', None) or (isinstance(k, float) and np.isnan(k)) for k in key):
                continue
            rows[('×'.join(dims), ' · '.join(map(str, key)))] = r
    return pd.DataFrame(rows).T


def run(B=500, seed=0):
    M = margins_all()
    rng = np.random.default_rng(seed)
    out = []
    for y in WAVES:
        df = wave(y)
        target = M[y]
        T = tabulate(df, rake(df, target))
        clustered = df.concelho.notna().all()
        groups = df.groupby('concelho').indices if clustered else None
        keys = list(groups) if clustered else None
        boots = []
        for _ in range(B):
            if clustered:
                idx = np.concatenate([groups[keys[k]] for k in rng.choice(len(keys), len(keys))])
            else:
                idx = rng.integers(0, len(df), int(round(len(df) / DEFF)))
            db = df.iloc[idx].reset_index(drop=True)
            boots.append(tabulate(db, rake(db, target)).reindex(index=T.index, columns=T.columns))
        arr = np.stack([b.values for b in boots])
        lo, hi = np.nanpercentile(arr, [5, 95], axis=0)
        n = {}
        for dims in DIMS:
            c = df.groupby(list(dims)).size()
            for key, v in c.items():
                key = key if isinstance(key, tuple) else (key,)
                n[('×'.join(dims), ' · '.join(map(str, key)))] = v
        for i, (dim, grp) in enumerate(T.index):
            for j, cat in enumerate(T.columns):
                out.append(dict(wave=y, dim=dim, group=grp, category=cat, share=T.iloc[i, j],
                                n_group=n.get((dim, grp), 0), lo=lo[i, j], hi=hi[i, j]))
        print(y, len(df), 'clustered' if clustered else f'deff {DEFF}')
    return pd.DataFrame(out)
