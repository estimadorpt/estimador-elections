"""Vote and abstention rates by sex x age x education, per district, calibrated to the 2025 result.

1. Census 2021 cells: Portuguese nationals 18+ resident in each district, sex x 7 age bands x 3 education
   levels (census_cells.parquet from prep.py).
2. Group gaps: additive multinomial logit (sex, age band, education) on EEP 2024, mainland, survey weight
   times a factor matching the official mainland 2024 result among residents; ridge prior sd 1 on slopes.
3. Calibration: in each district, one shift per party on the logit scale so that the cells add up to the
   district's 2025 result: votes per list, blank+null, and ABS = census residents - voters. Rates are
   therefore of resident citizens, not of registered voters (the roll is ~1 million longer than the census).
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from .survey import eep_frame, DATA

KEYS = ['AD', 'PS', 'CH', 'IL', 'L', 'CDU', 'BE', 'PAN', 'JPP', 'OUT', 'BLANK_NULL', 'ABS']
EEP_KEY = {13: 'CH', 2: 'AD', 6: 'AD', 29: 'AD', 32: 'AD', 33: 'AD', 35: 'AD', 36: 'AD', 7: 'PS', 16: 'IL',
           15: 'L', 3: 'CDU', 4: 'CDU', 5: 'CDU', 1: 'BE', 14: 'PAN', 91: 'BLANK_NULL', 92: 'ABS'}
FIT_KEYS = ['AD', 'PS', 'CH', 'IL', 'L', 'CDU', 'BE', 'PAN', 'OUT', 'BLANK_NULL', 'ABS']  # JPP uses OUT
AGE_EDGES = [18, 25, 35, 45, 55, 65, 75, 200]
AGES = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
EDUS = ['basic', 'secondary', 'tertiary']
DIST = {1: 'Aveiro', 2: 'Beja', 3: 'Braga', 4: 'Bragança', 5: 'Castelo Branco', 6: 'Coimbra', 7: 'Évora',
        8: 'Faro', 9: 'Guarda', 10: 'Leiria', 11: 'Lisboa', 12: 'Portalegre', 13: 'Porto', 14: 'Santarém',
        15: 'Setúbal', 16: 'Viana do Castelo', 17: 'Vila Real', 18: 'Viseu'}


def X_of(female, age, edu):
    cols = [np.ones(len(female)), np.asarray(female, float)]
    cols += [(np.asarray(age) == a).astype(float) for a in AGES if a != '35-44']
    cols += [(np.asarray(edu) == e).astype(float) for e in EDUS if e != 'basic']
    return np.column_stack(cols)


TERMS = ['intercept', 'female'] + [f'age {a}' for a in AGES if a != '35-44'] + \
        [f'edu {e}' for e in EDUS if e != 'basic']


def results_2025():
    d = pd.read_parquet(os.path.join(DATA, 'legislativas_2025.parquet')).fillna(0)
    code = d.territory_code.str.replace('LOCAL-', '').str[:2].astype(int)
    d = d[code < 50].copy()
    code = code[code < 50]
    d['district'] = code.map(DIST)
    d.loc[code.between(30, 39), 'district'] = 'Madeira'
    d.loc[code.between(40, 49), 'district'] = 'Açores'
    out = pd.DataFrame({'district': d.district,
                        'AD': d['PPD/PSD.CDS-PP.PPM'] + d['PPD/PSD.CDS-PP'], 'PS': d['PS'], 'CH': d['CH'],
                        'IL': d['IL'], 'L': d['L'], 'CDU': d['PCP-PEV'], 'BE': d['B.E.'], 'PAN': d['PAN'],
                        'JPP': d['JPP'], 'BLANK_NULL': d.blank_votes + d.null_votes,
                        'voters': d.number_voters, 'registered': d.subscribed_voters})
    named = ['AD', 'PS', 'CH', 'IL', 'L', 'CDU', 'BE', 'PAN', 'JPP', 'BLANK_NULL']
    out['OUT'] = out.voters - out[named].sum(1)
    return out.groupby('district').sum()


def eep_2024():
    d = eep_frame()
    d = d[(d.ano.astype(int) == 2024) & d.Q56_1.between(1, 5)]
    other = d.Q21_2.between(8, 90) & ~d.Q21_2.isin(list(EEP_KEY))
    key = np.where(d.Q21_1 == 5, d.Q21_2.map(EEP_KEY).where(~other, 'OUT'),
                   np.where(d.Q21_1.isin([98, 99]) | d.Q21_1.isna(), None, 'ABS'))
    age = pd.cut(d.Q42_1.where(d.Q42_1.between(18, 110)), AGE_EDGES, right=False, labels=AGES).astype(str)
    df = pd.DataFrame({'key': key, 'w': d.Q4_1.values, 'female': d.Q43.map({1: 0, 2: 1}).values,
                       'age': age.values, 'edu': d.Q44_2.map({1: 'basic', 2: 'secondary', 3: 'tertiary'}).values})
    return df.dropna().query("age != 'nan'").reset_index(drop=True)


def mainland_2024_targets(cells):
    d = pd.read_parquet(os.path.join(DATA, 'legislativas_2024.parquet')).fillna(0)
    tc = 'territory_code' if 'territory_code' in d else 'territoryCode'
    d = d[d[tc].str.replace('LOCAL-', '').str[:2].astype(int) < 30]
    cols = d.columns
    def s(*names):
        return sum(d[n].sum() for n in names if n in cols)
    t = {'AD': s('PPD/PSD.CDS-PP.PPM', 'PPD/PSD.CDS-PP'), 'PS': s('PS'), 'CH': s('CH'), 'IL': s('IL'),
         'L': s('L'), 'CDU': s('PCP-PEV'), 'BE': s('B.E.'), 'PAN': s('PAN'),
         'BLANK_NULL': s('blank_votes', 'null_votes')}
    voters = d.number_voters.sum()
    t['OUT'] = voters - sum(t.values())
    pop = cells[~cells.district.isin(['Madeira', 'Açores'])].n.sum()
    t['ABS'] = pop - voters
    return pd.Series(t) / pop


def fit_mnl(df, target, sd=1.0):
    ws = df.groupby('key').w.sum() / df.w.sum()
    w = df.w * (target / ws).reindex(df.key).values
    w = (w / w.mean()).values
    X = X_of(df.female.values, df.age.values, df.edu.values)
    K = len(FIT_KEYS)
    Y = np.zeros((len(df), K))
    Y[np.arange(len(df)), [FIT_KEYS.index(k) for k in df.key]] = 1
    p = X.shape[1]
    pen = np.r_[0.0, np.ones(p - 1)] / sd ** 2  # no penalty on intercepts

    def f(theta):
        B = np.c_[np.zeros(p), theta.reshape(p, K - 1)]
        eta = X @ B
        eta -= eta.max(1, keepdims=True)
        P = np.exp(eta); P /= P.sum(1, keepdims=True)
        nll = -(w[:, None] * Y * np.log(P + 1e-300)).sum() + 0.5 * (pen[:, None] * B[:, 1:] ** 2).sum()
        G = X.T @ (w[:, None] * (P - Y)) + pen[:, None] * B
        return nll, G[:, 1:].ravel()
    r = minimize(f, np.zeros(p * (K - 1)), jac=True, method='L-BFGS-B', options={'maxiter': 5000})
    B = np.c_[np.zeros(p), r.x.reshape(p, K - 1)]
    return pd.DataFrame(B, index=TERMS, columns=FIT_KEYS)


def calibrate(eta, n, T, iters=2000):
    """eta cells x K logits, n cell sizes, T target totals (K). Shift per party until totals match."""
    alpha = np.zeros(eta.shape[1])
    live = T > 0
    for _ in range(iters):
        e = eta + alpha
        e[:, ~live] = -np.inf
        e -= e.max(1, keepdims=True)
        P = np.exp(e); P /= P.sum(1, keepdims=True)
        tot = n @ P
        step = np.log(np.where(live, T, 1) / np.where(live, tot, 1))
        alpha += step
        if np.abs(step[live]).max() < 1e-10:
            break
    return P


def run():
    from ..survey_districts import census_cells
    cells = census_cells()
    cells['age'] = cells.age.astype(str)
    df = eep_2024()
    target = mainland_2024_targets(cells)
    B = fit_mnl(df, target)
    B.loc[B.index != 'intercept', 'OUT'] = 0.0  # one EEP respondent: no group pattern for other lists
    res = results_2025()
    out = []
    for dist, g in cells.groupby('district'):
        g = g.reset_index(drop=True)
        eta = X_of(g.female.values, g.age.values, g.edu.values) @ B.values
        eta = pd.DataFrame(eta, columns=FIT_KEYS)
        eta['JPP'] = eta['OUT']
        eta = eta[KEYS].values
        r = res.loc[dist]
        T = r[KEYS[:-1]].astype(float).values
        T = np.r_[T, max(g.n.sum() - r.voters, 0.03 * g.n.sum())]
        P = calibrate(eta, g.n.values, T)
        for i, row in g.iterrows():
            for k, key in enumerate(KEYS):
                out.append(dict(district=dist, sex='women' if row.female else 'men', age=row.age, edu=row.edu,
                                residents_2021=row.n, party=key, rate=P[i, k], count=row.n * P[i, k]))
    out = pd.DataFrame(out)
    return out, B, res, df, target


def main(out_dir):
    out, B, res, df, target = run()
    out.to_csv(os.path.join(out_dir, 'group_rates_2025.csv'), index=False)
    B.to_csv(os.path.join(out_dir, 'eep2024_mnl_coefficients.csv'))
    return out


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
