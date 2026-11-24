import numpy as np
import pandas as pd
import pytest

from src.models.poll_rw_model import RWConfig, prepare_rw_data

PARTIES = ["PS", "CH", "IL", "BE", "CDU", "PAN", "L", "AD"]


def _toy():
    rows = []
    for i, d in enumerate(pd.date_range("2019-01-01", "2019-12-31", freq="21D")):
        ch = 0 if d < pd.Timestamp("2019-06-01") else 20
        pan = 30 if i % 3 else 0  # PAN left unreported in some polls
        rows.append(dict(date=d, pollster="A" if i % 2 else "B", PS=350, CH=ch, IL=0, BE=80, CDU=60, PAN=pan, L=0, AD=300 + 5 * i))
    polls = pd.DataFrame(rows)
    results = pd.DataFrame(
        [dict(election_date=pd.Timestamp("2019-10-06"), PS=1_900_000, CH=67_000, IL=0, BE=500_000,
              CDU=330_000, PAN=170_000, L=0, AD=1_650_000)]
    )
    return prepare_rw_data(polls, results, "2020-03-01", ["2019-10-06"], PARTIES, RWConfig())


def test_kalman_filter_matches_the_dense_gaussian():
    """The filter's log-likelihood and final state equal the joint Gaussian computed directly."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from src.models.poll_rw_kalman import build_events, kalman_filter

    data = _toy()
    cfg = RWConfig()
    ev = build_events(data, cfg)
    T, P = data.exists.shape
    K = len(ev["others"])
    J = len(data.pollsters)
    rng = np.random.default_rng(3)
    sigma = np.abs(rng.normal(0.05, 0.02, P))
    house = rng.normal(0, 0.05, (J, P))
    house -= house.mean(0, keepdims=True)
    conc = 300.0
    ll, x, Pm = kalman_filter(ev, jnp.asarray(sigma), jnp.asarray(house), conc)

    # dense version: cumulative step covariance up to each event's week
    A = ev["A"]
    cumsteps = np.cumsum(ev["steps"], axis=0)  # weeks of steps per party up to each event
    cov_x = [ev["P0"] + A @ np.diag(sigma**2 * cs) @ A.T for cs in cumsteps]
    others, ref = ev["others"], ev["ref"]
    h_alr = house[:, others] - house[:, ref][:, None]
    idx, mean, blocks = [], [], []
    for e in range(len(ev["kind"])):
        m = ev["m"][e].astype(bool)
        if not m.any():
            continue
        kind = ev["kind"][e]
        phi = (ev["n"][e] + conc) / (1 + conc) if kind == 0 else 1.0
        W = ev["W"][e]
        Vb = np.linalg.inv(W) @ np.linalg.inv(W).T
        off = h_alr[ev["pollster"][e]] if kind == 0 else np.zeros(K)
        idx.append((e, np.where(m)[0]))
        mean.append(ev["y"][e][m] - off[m])
        blocks.append(phi * Vb[np.ix_(m, m)])
    n = sum(len(i) for _, i in idx)
    C = np.zeros((n, n))
    pos = np.cumsum([0] + [len(i) for _, i in idx])
    for a, (ea, ia) in enumerate(idx):
        for b, (eb, ib) in enumerate(idx):
            C[pos[a]:pos[a + 1], pos[b]:pos[b + 1]] = cov_x[min(ea, eb)][np.ix_(ia, ib)]
        C[pos[a]:pos[a + 1], pos[a]:pos[a + 1]] += blocks[a]
    y = np.concatenate(mean)
    sign, logdet = np.linalg.slogdet(C)
    dense_ll = -0.5 * (logdet + y @ np.linalg.solve(C, y) + n * np.log(2 * np.pi))
    assert float(ll) - 0.5 * n * np.log(2 * np.pi) == pytest.approx(dense_ll, rel=1e-7, abs=1e-6)

    # final state: conditional of x_T given every observation
    xT = cov_x[-1]
    cross = np.concatenate([cov_x[min(len(cov_x) - 1, e)][:, i] for e, i in idx], axis=1)
    np.testing.assert_allclose(np.asarray(x), cross @ np.linalg.solve(C, y), atol=1e-8)
    np.testing.assert_allclose(np.asarray(Pm), xT - cross @ np.linalg.solve(C, cross.T), atol=1e-8)


def test_events_mask_unreported_parties_and_end_at_the_target():
    from src.models.poll_rw_kalman import build_events

    data = _toy()
    ev = build_events(data)
    pan = list(ev["others"]).index(PARTIES.index("PAN"))
    polls = ev["kind"] == 0
    assert polls.sum() == len(data.poll_t)
    assert (ev["m"][polls][:, pan] == 0).any() and (ev["m"][polls][:, pan] == 1).any()
    assert ev["kind"][-1] == 2 and not ev["m"][-1].any()  # a step-only event closes at the target
    # every random-walk week is counted once
    np.testing.assert_allclose(ev["steps"].sum(0), data.exists[1:].sum(0))
