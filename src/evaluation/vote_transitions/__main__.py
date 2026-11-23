"""python -m src.evaluation.vote_transitions OUT_DIR [B]

Writes transitions_survey.csv (pair, source, from, to, share_of_from, n_from, lo, hi; 90% intervals),
transitions_survey_areas.csv (the parish-fitted survey matrices by NUTS-II region and district),
parish_bounds.csv, eep_vote_by_group.csv and group_rates_2025.csv. Env: EEP_DTA, INE_CONSTRAINTS,
VOTE_MOVES_PARISH, VOTE_MOVES_EXITPOLL. EEP outputs are for non-commercial research (ICS licence).
"""
import os, sys
import pandas as pd
from . import survey, bounds, jp, groups, group_rates

FORMAT = ['pair', 'source', 'from', 'to', 'share_of_from', 'n_from', 'lo', 'hi']


def main(out, B=200):
    os.makedirs(out, exist_ok=True)
    parts = []
    for a, b in survey.PAIRS:
        t, _ = survey.run_pair(a, b, B=B * 2, deff=1.25)
        parts.append(t)
    parts.append(jp.exitpoll_table(B=B * 2))
    nat, areas = jp.run(B)
    parts.append(nat)
    pd.concat(parts)[FORMAT].to_csv(os.path.join(out, 'transitions_survey.csv'), index=False)
    areas.to_csv(os.path.join(out, 'transitions_survey_areas.csv'), index=False)
    rows = []
    for a, b in bounds.PAIRS:
        lo, hi = bounds.bounds(a, b)
        for i in bounds.ROWS:
            for j in bounds.COLS:
                rows.append(dict(pair=f'{a}-{b}', **{'from': i}, to=j, lb=lo.loc[i, j], ub=hi.loc[i, j]))
    pd.DataFrame(rows).to_csv(os.path.join(out, 'parish_bounds.csv'), index=False)
    groups.run(B=B).to_csv(os.path.join(out, 'eep_vote_by_group.csv'), index=False)
    group_rates.main(out)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200)
