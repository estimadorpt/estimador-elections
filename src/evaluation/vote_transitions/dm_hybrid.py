"""Hybrid ecological inference: parish results + EEP individual transitions (+ Católica exit poll).

Transition probabilities beta_r[i, j] (row i = status at a, column j = status at b) for each mainland NUTS-II
region r, logits theta + delta_r (delta shrunk to 0, sd TAU). Rows: 9 categories + NEW; columns: 9
categories + EXIT (the vote-moves thread's net-churn convention: per parish NEW = max(0, reg_b - reg_a),
EXIT = max(0, reg_a - reg_b); NEW -> EXIT is a structural zero).

Likelihood pieces:
  parish:  y_p ~ Dirichlet-multinomial(N_p, mu_p / N_p, kappa), mu_p = x_p @ beta_r(p), kappa estimated
  EEP:     respondent (r, i, j) adds log beta_r[i, j] - log(1 - beta_r[i, EXIT]), weighted by the survey
           weight times a factor that matches the b result, each (r, i) row scaled to n / DEFF
  exit poll (2024-2025): composition of each 2025 voting column by recalled 2024 vote, national,
           ABST and NEW merged, effective n = 35,022 / EP_DEFF spread by column size
"""
import sys, time
import numpy as np, pandas as pd, torch
from .survey import CATS, respondents, margins, table, ipf_factors
from .bounds import parish_pair, ROWS, COLS
from .exitpoll import composition as ep_composition

torch.set_default_dtype(torch.float64)
TAU, THETA_SD, DEFF, EP_DEFF, EP_N = 0.5, 2.0, 1.25, 10.0, 35022
REGIONS = [1, 2, 3, 4, 5]


class Data:
    def __init__(self, a, b):
        X, Y, reg = parish_pair(a, b)
        keep_rows = [i for i, k in enumerate(ROWS) if X[k].sum() > 0]
        self.rows = [ROWS[i] for i in keep_rows]
        self.cols = COLS
        self.X = torch.tensor(X[self.rows].values)
        self.Y = torch.tensor(Y[self.cols].values)
        self.N = self.X.sum(1)
        self.reg = torch.tensor(reg.values - 1)
        R, C = len(self.rows), len(self.cols)
        mask = torch.zeros(R, C)
        if 'NEW' in self.rows:
            mask[self.rows.index('NEW'), self.cols.index('EXIT')] = -torch.inf
        self.mask = mask
        self.a, self.b = a, b
        self.S = None
        self.ep = None

    def add_eep(self):
        u = respondents(self.a, self.b)
        u = u[u.usable & u.prev.isin(self.rows)].copy()
        sa, sb, s_new = margins(self.a, self.b)
        rows9 = [k for k in CATS + ['NEW'] if k in self.rows]
        rt = pd.Series(np.r_[sa.values * (1 - s_new), s_new], CATS + ['NEW']).reindex(rows9).values
        cols9 = [k for k in CATS if sb[k] > 0]
        M = table(u, rows9, cols9) * (len(u) / u.w.sum())
        M = M + 1.0 * sb[cols9].values[None, :] / sb[cols9].sum()
        _, bf = ipf_factors(M, rt / rt.sum(), sb[cols9].values / sb[cols9].sum())
        u['w2'] = u.w * pd.Series(bf, cols9).reindex(u.cur).values
        S = np.zeros((5, len(self.rows), len(self.cols)))
        for (r, i), g in u.groupby(['region', 'prev']):
            dist = g.groupby('cur').w2.sum().reindex(self.cols).fillna(0).values
            S[int(r) - 1, self.rows.index(i)] = dist / dist.sum() * len(g) / DEFF
        self.S = torch.tensor(S)
        self.survey = u
        return self

    def add_exitpoll(self):
        C = ep_composition()  # index CATS (from 2024), columns 2025 voting categories, share_of_to
        _, sb, _ = margins(2024, 2025)
        voters = [k for k in CATS if k != 'ABST']
        vs = sb[voters] / sb[voters].sum()
        n = torch.tensor((vs * EP_N / EP_DEFF).values)
        comp = torch.tensor(C[voters].reindex(CATS).values)  # 9 x 8
        self.ep = (voters, comp, n)
        return self


def unpack(params, d):
    R, C = len(d.rows), len(d.cols)
    theta = params[:R * C].view(R, C)
    delta = params[R * C:R * C + 5 * R * C].view(5, R, C)
    logk = params[-1]
    logits = theta[None] + delta + d.mask[None]
    beta = torch.softmax(logits, dim=2)
    return theta, delta, logk, beta


def negloglik(params, d, use_parish=True, use_survey=True, use_ep=True, Xw=None, Yw=None, Sw=None, epw=None):
    theta, delta, logk, beta = unpack(params, d)
    X = d.X if Xw is None else Xw
    Y = d.Y if Yw is None else Yw
    nll = (theta ** 2).sum() / (2 * THETA_SD ** 2) + (delta ** 2).sum() / (2 * TAU ** 2)
    if use_parish:
        kappa = torch.exp(logk)
        B = beta[d.reg]  # P x R x C
        mu = torch.einsum('pr,prc->pc', X, B)
        N = X.sum(1)
        pi = (mu / N[:, None]).clamp_min(1e-12)
        ll = (torch.lgamma(kappa) - torch.lgamma(N + kappa)).sum() + \
             (torch.lgamma(Y + kappa * pi) - torch.lgamma(kappa * pi)).sum()
        nll = nll - ll
    if use_survey and d.S is not None:
        S = d.S if Sw is None else Sw
        ex = d.cols.index('EXIT')
        lp = torch.log(beta.clamp_min(1e-15)) - torch.log1p(-beta[:, :, ex:ex + 1])
        lp = torch.where(S > 0, lp, torch.zeros_like(lp))
        nll = nll - (S * lp).sum()
    if use_ep and d.ep is not None:
        voters, comp, n = d.ep if epw is None else epw
        regX = torch.zeros(5, len(d.rows))
        regX.index_add_(0, d.reg, X)
        F = torch.einsum('rk,rkc->kc', regX, beta)  # national flows, rows x cols
        jj = [d.cols.index(v) for v in voters]
        F = F[:, jj]
        # merge NEW into ABST for the exit poll's recalled categories
        idx = [d.rows.index(k) if k in d.rows else None for k in CATS]
        Fm = torch.stack([F[i] if i is not None else torch.zeros(len(jj)) for i in idx])
        if 'NEW' in d.rows:
            Fm[CATS.index('ABST')] = Fm[CATS.index('ABST')] + F[d.rows.index('NEW')]
        pi = Fm / Fm.sum(0, keepdim=True)
        nll = nll - (n[None, :] * comp * torch.log(pi.clamp_min(1e-15))).sum()
    return nll


def fit(d, init=None, max_iter=3000, **kw):
    R, C = len(d.rows), len(d.cols)
    n = R * C + 5 * R * C + 1
    params = torch.zeros(n) if init is None else init.clone()
    if init is None:
        params[-1] = np.log(200.0)
        # start from independence: theta = log of the b column shares
        cs = d.Y.sum(0) / d.Y.sum()
        params[:R * C] = torch.log(cs)[None, :].repeat(R, 1).flatten()
        params[:R * C] = params[:R * C] - params[:R * C].view(R, C).mean(1, keepdim=True).repeat(1, C).flatten()
    params.requires_grad_(True)
    opt = torch.optim.LBFGS([params], lr=1, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-12,
                            history_size=50, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = negloglik(params, d, **kw)
        loss.backward()
        return loss
    opt.step(closure)
    return params.detach()


def national(params, d, X=None):
    _, _, logk, beta = unpack(params, d)
    X = d.X if X is None else X
    regX = torch.zeros(5, len(d.rows))
    regX.index_add_(0, d.reg, X)
    F = torch.einsum('rk,rkc->kc', regX, beta)
    return (F / F.sum(1, keepdim=True)).numpy(), float(torch.exp(logk))


VARIANTS = {
    'ei_parish': dict(use_parish=True, use_survey=False, use_ep=False),
    'hybrid_eep': dict(use_parish=True, use_survey=True, use_ep=False),
    'survey_only_eep': dict(use_parish=False, use_survey=True, use_ep=False),
    'hybrid_exitpoll': dict(use_parish=True, use_survey=False, use_ep=True),
}


def boot_inputs(d, rng):
    """Resample parishes within region, survey respondents, and exit-poll compositions."""
    reg = d.reg.numpy()
    idx = np.concatenate([rng.choice(np.where(reg == r)[0], (reg == r).sum()) for r in range(5)])
    cnt = np.bincount(idx, minlength=len(reg))
    Xw = d.X * torch.tensor(cnt)[:, None]
    Yw = d.Y * torch.tensor(cnt)[:, None]
    Sw, epw = None, None
    if d.S is not None:
        S = d.S.numpy()
        Sw = np.zeros_like(S)
        for r in range(5):
            for i in range(S.shape[1]):
                tot = S[r, i].sum()
                if tot > 0:
                    k = max(1, int(round(tot)))
                    Sw[r, i] = rng.multinomial(k, S[r, i] / tot) * tot / k
        Sw = torch.tensor(Sw)
    if d.ep is not None:
        voters, comp, n = d.ep
        c = comp.numpy()
        draws = np.stack([rng.dirichlet(c[:, j] * n[j].item() + 1e-3) for j in range(c.shape[1])], 1)
        epw = (voters, torch.tensor(draws), n)
    return Xw, Yw, Sw, epw


def run_pair(a, b, variants, B=40, seed=0):
    d = Data(a, b)
    if (a, b) != (2024, 2025):
        d.add_eep()
    else:
        d.add_exitpoll()
    rng = np.random.default_rng(seed)
    out = []
    for name in variants:
        kw = VARIANTS[name]
        t0 = time.time()
        p0 = fit(d, **kw)
        nat, kappa = national(p0, d)
        ex = d.cols.index('EXIT')
        if not kw['use_parish']:  # survey only: report conditional on staying in the roll
            nat = nat.copy(); nat[:, ex] = 0; nat = nat / nat.sum(1, keepdims=True)
        boots = []
        for bi in range(B):
            Xw, Yw, Sw, epw = boot_inputs(d, rng)
            pb = fit(d, init=p0, max_iter=600, Xw=Xw, Yw=Yw, Sw=Sw, epw=epw, **kw)
            nb, _ = national(pb, d, X=Xw)
            if not kw['use_parish']:
                nb[:, ex] = 0; nb = nb / nb.sum(1, keepdims=True)
            boots.append(nb)
        lo, hi = np.percentile(np.stack(boots), [5, 95], axis=0)
        n_from = None
        if d.S is not None:
            n_from = d.survey.prev.value_counts().reindex(d.rows).fillna(0).astype(int).values
        for i, fr in enumerate(d.rows):
            for j, to in enumerate(d.cols):
                out.append(dict(pair=f'{a}-{b}', source=name, **{'from': fr}, to=to, share_of_from=nat[i, j],
                                n_from=(n_from[i] if n_from is not None else np.nan), lo=lo[i, j], hi=hi[i, j]))
        print(f'{a}-{b} {name}: kappa {kappa:.0f}, {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(out)


def main(out_dir, B=10):
    res = []
    for a, b in [(2015, 2019), (2019, 2022), (2022, 2024)]:
        res.append(run_pair(a, b, ['ei_parish', 'hybrid_eep', 'survey_only_eep'], B=B))
    res.append(run_pair(2024, 2025, ['ei_parish', 'hybrid_exitpoll'], B=B))
    out = pd.concat(res)
    out.to_csv(f'{out_dir}/transitions_dm_hybrid.csv', index=False)
    return out


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.', int(sys.argv[2]) if len(sys.argv) > 2 else 10)
