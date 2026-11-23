"""Duncan-Davis bounds summed over mainland parishes, and a check of the raked survey matrices."""
import numpy as np, pandas as pd
from .survey import CATS, PARISH

PAIRS = [(2015, 2019), (2019, 2022), (2022, 2024), (2024, 2025)]
ROWS = CATS + ['NEW']
COLS = CATS + ['EXIT']


def parish_pair(a, b):
    p = pd.read_parquet(PARISH)
    p = p[p.eep_region.between(1, 5)]
    A = p[p.year == a].set_index('code')
    Bt = p[p.year == b].set_index('code')
    A, Bt = A.align(Bt, join='inner', axis=0)
    X = A[CATS].astype(float).copy()
    Y = Bt[CATS].astype(float).copy()
    X['NEW'] = (Bt.registered - A.registered).clip(lower=0)
    Y['EXIT'] = (A.registered - Bt.registered).clip(lower=0)
    assert np.allclose(X.sum(1), Y.sum(1))
    return X[ROWS], Y[COLS], A.eep_region.astype(int)


def bounds(a, b):
    X, Y, reg = parish_pair(a, b)
    x, y = X.values[:, :, None], Y.values[:, None, :]
    N = X.values.sum(1)[:, None, None]
    lo = np.maximum(0, x + y - N).sum(0)
    hi = np.minimum(x, y).sum(0)
    tot = X.values.sum(0)[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        return pd.DataFrame(lo / tot, ROWS, COLS), pd.DataFrame(hi / tot, ROWS, COLS)
