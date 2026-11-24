"""The random-walk poll model with the weekly path integrated out by a Kalman filter ("rw_kf").

Same model as `poll_rw_model` (a weekly random walk per party, house effects zero-sum over
pollsters and parties, an optional industry bias per election cycle, results that pin the walk),
with one approximation: each poll and each result enters as a Gaussian observation of its
log-ratios against AD instead of a Dirichlet-multinomial.

* State: x_t = eta_t[p] - eta_t[AD] for the seven other parties. A step adds
  A diag(sigma_rw^2 * exists) A' to its covariance (A maps eta to log-ratios), so a party that
  does not exist yet keeps the prior spread of its starting level (level0, ZeroSumNormal 1.5).
* Observation of a poll with counts c over its reported parties R (AD always reported):
  y_p = log(c_p / c_AD), p in R, with covariance phi * V, V = diag(1/c_p) + 1/c_AD (the
  multinomial delta method) and phi = (n + conc) / (1 + conc), the Dirichlet-multinomial
  inflation at the model's poll concentration. Mean: x_t + house[j] + bias[cycle] (log-ratios).
* Results: the same with phi = n / result_concentration (the walk passes through them).

NUTS then samples only sigma_rw, the house effects, the concentration and the bias terms (about
160 values instead of ~7,000), and the tight weekly-step geometry that forces tree depth 10 on
2025 disappears. On 2025 at 56 days (24 Nov 2026 prototype) this gave 5.6x the effective draws
per minute with national shares within 0.2 pp of the exact model.

The momentum bias and the drifting house effects of `poll_rw_model` are not implemented here.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.models.poll_rw_model import RWConfig, RWData

REF_PARTY = "AD"


def _alr_matrix(parties, ref):
    P = len(parties)
    others = [i for i in range(P) if i != ref]
    A = np.zeros((len(others), P))
    for k, p in enumerate(others):
        A[k, p] = 1.0
        A[k, ref] = -1.0
    return A, others


def build_events(data: RWData, cfg: RWConfig = RWConfig()) -> Dict[str, np.ndarray]:
    """One event per poll or result in time order, plus a final step-only event at the target.

    Each event carries the random-walk weeks per party since the previous event, the
    observation in log-ratios with its mask, and the whitening matrix W of its base covariance
    (W V W' = I on the observed slots, identity elsewhere).
    """
    T, P = data.exists.shape
    ref = data.parties.index(REF_PARTY)
    A, others = _alr_matrix(data.parties, ref)
    K = len(others)
    step_mask = data.exists.astype(float).copy()
    step_mask[0] = 0.0
    reported = data.poll_reported if data.poll_reported is not None else data.poll_counts > 0

    def obs_row(c, r):
        m = np.array([r[p] for p in others], float)
        cc = np.where(r, c, 1.0)
        y = np.array([np.log(cc[p] / cc[ref]) for p in others]) * m
        V = np.diag([1.0 / cc[p] for p in others]) + 1.0 / cc[ref]
        return y, m, V

    obs = []  # (week, kind 0 poll / 1 result, pollster, cycle, y, m, V, n)
    n_skipped = 0
    for i in range(len(data.poll_t)):
        c = data.poll_counts[i].astype(float)
        r = reported[i] & (c > 0)
        if not r[ref]:
            n_skipped += 1
            continue
        y, m, V = obs_row(c, r)
        obs.append((int(data.poll_t[i]), 0, int(data.poll_pollster[i]), int(data.poll_cycle[i]), y, m, V, c[r].sum()))
    for j in range(len(data.result_t)):
        c = data.result_counts[j].astype(float)
        r = c > 0
        y, m, V = obs_row(c, r)
        # Dirichlet-multinomial at concentration result_concentration: variance (1/c) * n / conc
        V = V * c.sum() / cfg.result_concentration
        obs.append((int(data.result_t[j]), 1, 0, 0, y, m, V, 1.0))
    obs.sort(key=lambda o: (o[0], o[1]))
    obs.append((T - 1, 2, 0, 0, np.zeros(K), np.zeros(K), np.eye(K), 1.0))

    cum = np.vstack([np.zeros((1, P)), np.cumsum(step_mask, axis=0)])
    rows, last = [], -1
    for w, kind, j, cyc, y, m, V, n in obs:
        steps = cum[w + 1] - cum[last + 1]
        last = w
        Vb = (m[:, None] * m[None, :]) * V + np.diag(1.0 - m)
        W = np.linalg.inv(np.linalg.cholesky(Vb))
        rows.append((steps, kind, j, cyc, y, m, W, n))
    keys = ("steps", "kind", "pollster", "cycle", "y", "m", "W", "n")
    ev = {k: np.array([r[i] for r in rows]) for i, k in enumerate(keys)}
    S0 = 1.5**2 * (np.eye(P) - np.ones((P, P)) / P)  # level0 ~ ZeroSumNormal(1.5)
    ev.update(A=A, others=np.array(others), ref=ref, P0=A @ S0 @ A.T, n_skipped=n_skipped)
    return ev


def kalman_filter(ev, sigma_rw, house, conc, bias=None):
    """Log-likelihood of every observation and the filtered state at the target (mean, cov)."""
    import jax
    import jax.numpy as jnp

    A = jnp.asarray(ev["A"])
    others, ref = ev["others"], ev["ref"]
    K = len(others)
    h_alr = house[:, others] - house[:, ref][:, None]
    b_alr = None if bias is None else bias[:, others] - bias[:, ref][:, None]

    def step(carry, e):
        x, Pm, ll = carry
        steps, kind, j, cyc, y, m, W, n = e
        Pm = Pm + (A * (sigma_rw**2 * steps)[None, :]) @ A.T
        is_poll = (kind == 0).astype(x.dtype)
        phi = jnp.where(kind == 0, (n + conc) / (1.0 + conc), 1.0)
        off = h_alr[j]
        if b_alr is not None:
            off = off + b_alr[cyc]
        yw = W @ (m * (y - is_poll * off))
        Hw = W * m[None, :]

        def scalar(c, k):
            x, Pm, ll = c
            h = Hw[k]
            Ph = Pm @ h
            s = h @ Ph + phi
            v = yw[k] - h @ x
            g = Ph / s
            return (x + g * v, Pm - jnp.outer(g, Ph), ll - 0.5 * (jnp.log(s) + v * v / s)), None

        (x, Pm, ll), _ = jax.lax.scan(scalar, (x, Pm, ll), jnp.arange(K), unroll=K)
        # the unobserved slots each added log(phi) with no signal; the whitening adds log|W|
        ll = ll + 0.5 * jnp.sum(1.0 - m) * jnp.log(phi) + jnp.sum(jnp.log(jnp.abs(jnp.diag(W))))
        return (x, 0.5 * (Pm + Pm.T), ll), None

    xs = tuple(jnp.asarray(ev[k]) for k in ("steps", "kind", "pollster", "cycle", "y", "m", "W", "n"))
    init = (jnp.zeros(K), jnp.asarray(ev["P0"]), jnp.asarray(0.0))
    (x, Pm, ll), _ = jax.lax.scan(step, init, xs)
    return ll, x, Pm


def numpyro_model(ev, n_pollsters, n_cycles, n_parties, cfg: RWConfig):
    import numpyro
    import numpyro.distributions as dist

    sigma_rw = numpyro.sample("sigma_rw", dist.HalfNormal(cfg.innovation_sd_prior).expand([n_parties]).to_event(1))
    sigma_house = numpyro.sample("sigma_house", dist.HalfNormal(cfg.house_sd_prior).expand([n_parties]).to_event(1))
    house_raw = numpyro.sample("house_raw", dist.ZeroSumNormal(1.0, event_shape=(n_pollsters, n_parties)))
    house = numpyro.deterministic("house", house_raw * sigma_house[None, :])
    a, b = cfg.poll_concentration_prior
    conc = numpyro.sample("concentration_polls", dist.Gamma(a, b))
    bias = None
    if cfg.persistent_bias or cfg.bias_sd_prior > 1e-3:
        sigma_bias = numpyro.sample("sigma_bias", dist.HalfNormal(cfg.bias_sd_prior))
        # zero-sum over parties only, one vector per cycle (as poll_rw_model's bias_raw)
        bias_raw = numpyro.sample(
            "bias_raw", dist.ZeroSumNormal(1.0, event_shape=(n_parties,)).expand([n_cycles]).to_event(1)
        )
        bias = sigma_bias * bias_raw
        if cfg.persistent_bias:
            bias_mean = numpyro.sample("bias_mean", dist.ZeroSumNormal(cfg.persistent_bias_sd, event_shape=(n_parties,)))
            bias = bias + bias_mean[None, :]
        bias = numpyro.deterministic("bias", bias)
    ll, _, _ = kalman_filter(ev, sigma_rw, house, conc, bias)
    numpyro.factor("kalman", ll)


def fit_rw_kalman(data: RWData, cfg: RWConfig, draws: int, tune: int, chains: int, seed: int, target_accept: float = 0.9):
    """Sample the hyperparameters; return (idata, national share draws at the target, events)."""
    if cfg.momentum_bias_sd > 0 or cfg.house_drift_sd_prior > 0 or cfg.pollster_concentration or cfg.bloc_swing_sd_prior > 0:
        raise NotImplementedError("rw_kf implements the random walk, house effects and industry bias only")
    import arviz as az

    # One CPU device per chain, as pymc.sampling.jax sets up for the exact fits (it asks XLA for
    # 100). Without it JAX sees one device and the four chains run vectorized on one core, which
    # is how the 24 Nov Mac grid ran them: 2011 at 56 days took 170 s that way against 65 s in
    # parallel (218 s sequential). It has no effect once JAX has started.
    import numpyro

    numpyro.set_host_device_count(max(chains, 1))
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    ev = build_events(data, cfg)
    P = len(data.parties)
    args = (ev, len(data.pollsters), len(data.cycles), P, cfg)
    method = "parallel" if jax.local_device_count() >= chains else "vectorized"
    mcmc = MCMC(
        NUTS(numpyro_model, target_accept_prob=target_accept),
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        chain_method=method,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), *args, extra_fields=("diverging", "num_steps", "adapt_state.step_size"))
    idata = az.from_numpyro(mcmc, log_likelihood=False)
    st = mcmc.get_extra_fields(group_by_chain=True)
    idata.sample_stats["n_steps"] = (("chain", "draw"), np.asarray(st["num_steps"]))
    idata.sample_stats["step_size"] = (("chain", "draw"), np.asarray(st["adapt_state.step_size"]))
    idata.sample_stats["tree_depth"] = (("chain", "draw"), np.log2(np.asarray(st["num_steps"]) + 1.0))

    post = mcmc.get_samples()
    has_bias = "bias" in post

    def final_state(s, h, c, b):
        _, x, Pm = kalman_filter(ev, s, h, c, b if has_bias else None)
        return x, Pm

    S = post["sigma_rw"].shape[0]
    bias_draws = post["bias"] if has_bias else jnp.zeros((S, 1, 1))
    xs, Ps = jax.jit(jax.vmap(final_state))(post["sigma_rw"], post["house"], post["concentration_polls"], bias_draws)
    xs, Ps = np.asarray(xs), np.asarray(Ps)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Ps + 1e-12 * np.eye(Ps.shape[1])[None])
    z = xs + np.einsum("sij,sj->si", L, rng.standard_normal(xs.shape))
    eta = np.zeros((S, P))
    eta[:, ev["others"]] = z
    w = np.exp(eta - eta.max(1, keepdims=True)) * data.exists[data.target_t][None, :]
    national = w / w.sum(1, keepdims=True)
    return idata, national, ev

