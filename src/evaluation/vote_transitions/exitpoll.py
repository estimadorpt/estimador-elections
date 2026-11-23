"""Católica 2025 exit poll: 2024->2025 transitions completed with the official margins and raked."""
import numpy as np, pandas as pd
import os
from .survey import CATS, ipf, gap_years, COHORT, DATA, PARISH

# Católica/CESOP 2025 exit poll, recalled 2024 vote (ERC deposit 2025062, pp. 14-15), transcribed
EP = os.environ.get('VOTE_MOVES_EXITPOLL', '/mnt/project-files/vote-moves/exitpoll_catolica_2025_transfers.csv')
R24 = {'AD': 'PSD_CDS', 'Abstenção': 'ABST', 'B/N': 'BLANK_NULL', 'BE': 'LEFT', 'CDU': 'LEFT', 'L': 'LEFT',
       'CH': 'CH', 'IL': 'IL', 'PAN': 'PAN', 'PS': 'PS', 'O': 'OTHER'}
C25 = {'AD': 'PSD_CDS', 'B/N': 'BLANK_NULL', 'CH': 'CH', 'IL': 'IL', 'PAN': 'PAN', 'PS': 'PS', 'Outros': 'OTHER'}


def left_weights():
    d = pd.read_parquet(os.path.join(DATA, 'legislativas_2025.parquet'))
    d = d[d.territory_code.str.replace('LOCAL-', '').str[:2].astype(int) < 30]
    w = {'BE': d['B.E.'].sum(), 'CDU': d['PCP-PEV'].sum(), 'L': d['L'].sum()}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def composition():
    """share_of_to: for each 2025 category, where its voters came from (2024 recalled, 'Ns' dropped)."""
    e = pd.read_csv(EP)
    e = e[e.table == 'row_pct_where_2025_voters_came_from']
    e = e[e.vote_2024_recalled != 'Ns']
    e['from'] = e.vote_2024_recalled.map(R24)
    lw = left_weights()
    rows = []
    for v25, g in e.groupby('vote_2025'):
        comp = g.groupby('from').pct.sum()
        comp = comp / comp.sum()
        if v25 in C25:
            rows.append(comp.rename(C25[v25]))
        else:
            rows.append((comp * lw[v25]).rename('LEFT_' + v25))
    C = pd.concat(rows, axis=1).fillna(0)
    left = C[[c for c in C if c.startswith('LEFT_')]].sum(1).rename('LEFT')
    C = pd.concat([C[[c for c in C if not c.startswith('LEFT_')]], left], axis=1)
    return C.reindex(index=CATS, fill_value=0)


def run():
    p = pd.read_parquet(PARISH)
    p = p[p.eep_region.between(1, 5)]
    tot = p.groupby('year')[CATS + ['registered']].sum()
    s24 = tot.loc[2024, CATS] / tot.loc[2024, 'registered']
    s25 = tot.loc[2025, CATS] / tot.loc[2025, 'registered']
    C = composition()
    voters = [k for k in CATS if k != 'ABST']
    F = C[voters] * s25[voters].values[None, :]  # flows into each 2025 voter category, share of electorate
    resid = s24 - F.sum(1)  # what 2024 row mass is left for 2025 abstention
    print('2024 row margins vs exit-poll flows into 2025 voting (pp of electorate):')
    print(pd.DataFrame({'official_2024': s24 * 100, 'to_2025_voting': F.sum(1) * 100, 'left_for_ABST': resid * 100}).round(2))
    out = []
    # (1) composition as reported (share_of_to), no raking needed: one margin only
    for to in voters:
        for fr in CATS:
            out.append(dict(pair='2024-2025', source='exitpoll_catolica_composition', **{'from': fr}, to=to,
                            share_of_to=C.loc[fr, to]))
    # (2) completed and raked matrix (NEW folded into ABST)
    M = F.copy()
    M['ABST'] = resid.clip(lower=0.002)
    M = M[CATS].values
    P = ipf(M, s24.values, s25.values)
    share = P / P.sum(1, keepdims=True)
    return C, pd.DataFrame(P, index=CATS, columns=CATS), share, out
