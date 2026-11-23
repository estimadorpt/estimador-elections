"""Can individual survey microdata improve the district step? Four tests against blend swing.

The district step turns a national vote into 20 district votes. Blend swing (half proportional,
half uniform swing from the last result) is the baseline. These tests ask whether the
Estudo Eleitoral Português (EEP, ICS/APIS, post-election surveys 2002-2024) poststratified on
Census 2021 cells (sex x age x education by district) adds anything, given the TRUE national
result (the same "oracle" setting as `backtest.py --district-oracle`).

    python -m src.evaluation.survey_districts --ceiling     # demographic and regional swing
    python -m src.evaluation.survey_districts --new-party   # survey geography for young parties
    python -m src.evaluation.survey_districts --recall      # recalled-vote weighting (national)
    python -m src.evaluation.survey_districts --regions     # upper bound for regional crosstabs
    python -m src.evaluation.survey_districts --all

Inputs (not in this repository; the EEP licence is non-commercial research):
    EEP_DTA          EEP_2002-2024.dta (default ../estimador-data/data/surveys/archive/apis/)
    INE_CONSTRAINTS  folder with person_sex_age15_educ_2021.parquet and
                     person_sex_nationality_group_2021.parquet
                     (default ../estimador-microsynthesis/data/interim/constraints/ine/)

Model: a multinomial logit of vote (8 parties, PS reference) on sex, 7 age bands, 3 education
levels and 5 NUTS-II regions, one coefficient set per survey wave, tied across waves by a
random-walk prior (sd TAU per wave) and fitted by penalised maximum likelihood. A district's
survey share is the model's cell probabilities weighted by its census cells (Portuguese
nationals 18+). The EEP covers the mainland only, so the islands keep blend swing throughout.
Results and reading: docs/SURVEY_MICRODATA_DISTRICTS.md.
"""

import argparse
import contextlib
import io
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from src.evaluation.backtest import load_truth
from src.processing.district_swing import blend_swing, seats_from_district_shares

warnings.filterwarnings("ignore", category=RuntimeWarning)

PARTIES = ["PS", "CH", "IL", "BE", "CDU", "PAN", "L", "AD"]
WAVES = [2002, 2005, 2009, 2011, 2015, 2019, 2022, 2024]
ELECTION = {2009: "2009-09-27", 2011: "2011-06-05", 2015: "2015-10-04", 2019: "2019-10-06",
            2022: "2022-01-30", 2024: "2024-03-10", 2025: "2025-05-18"}
AGE_EDGES = [18, 25, 35, 45, 55, 65, 75, 200]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]

# INE district code (first two digits of DICOFRE) -> circle name as in the backtest truth
DISTRICT = {1: "Aveiro", 2: "Beja", 3: "Braga", 4: "Bragança", 5: "Castelo Branco", 6: "Coimbra",
            7: "Évora", 8: "Faro", 9: "Guarda", 10: "Leiria", 11: "Lisboa", 12: "Portalegre",
            13: "Porto", 14: "Santarém", 15: "Setúbal", 16: "Viana do Castelo", 17: "Vila Real",
            18: "Viseu"}
# NUTS-II as coded in the EEP (Q56_1): 1 Norte, 2 Centro, 3 Lisboa e Vale do Tejo, 4 Alentejo,
# 5 Algarve; 6 Madeira, 7 Açores (not sampled by the EEP)
REGION = {"Aveiro": 2, "Beja": 4, "Braga": 1, "Bragança": 1, "Castelo Branco": 2, "Coimbra": 2,
          "Évora": 4, "Faro": 5, "Guarda": 2, "Leiria": 2, "Lisboa": 3, "Portalegre": 4, "Porto": 1,
          "Santarém": 3, "Setúbal": 3, "Viana do Castelo": 1, "Vila Real": 1, "Viseu": 2,
          "Madeira": 6, "Açores": 7}
# EEP party codes (Q21_2 vote, Q21_7 previous vote) -> model parties
VOTE = {1: "BE", 2: "AD", 3: "CDU", 4: "CDU", 5: "CDU", 6: "AD", 7: "PS", 13: "CH", 14: "PAN",
        15: "L", 16: "IL", 29: "AD", 32: "AD", 33: "AD", 35: "AD", 36: "AD"}
OTHER = {8, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 34, 37, 38, 90}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EEP_DTA = os.environ.get(
    "EEP_DTA", os.path.join(ROOT, "..", "estimador-data", "data", "surveys", "archive", "apis", "EEP_2002-2024.dta"))
INE_DIR = os.environ.get(
    "INE_CONSTRAINTS",
    os.path.join(ROOT, "..", "estimador-microsynthesis", "data", "interim", "constraints", "ine"))


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------


def census_cells() -> pd.DataFrame:
    """Portuguese nationals aged 18+ by district x sex x age band x education (Census 2021).

    Foreigners are removed parish by parish and sex by sex (INE gives nationality only by sex),
    assuming they spread like nationals over age and education; 18-19 is 2/5 of 15-19."""
    x = pd.read_parquet(os.path.join(INE_DIR, "person_sex_age15_educ_2021.parquet"))
    nat = pd.read_parquet(os.path.join(INE_DIR, "person_sex_nationality_group_2021.parquet"))
    nat = nat.pivot_table(index=["freguesia", "sex"], columns="nationality_group", values="value", aggfunc="sum")
    nat = nat.fillna(0)
    pt = (nat["PT"] / nat.sum(axis=1)).rename("pt").reset_index()
    x = x.merge(pt, on=["freguesia", "sex"], how="left")
    lo = x["age_15grp"].str.extract(r"^(\d+)")[0].astype(int)
    x["n"] = x["value"] * x["pt"].fillna(1.0) * np.where(lo == 15, 2 / 5, 1.0)
    x["age"] = pd.cut(lo.clip(lower=18), AGE_EDGES, right=False, labels=AGE_LABELS)
    e = x["education_level_code"]
    x["edu"] = np.where(e.isin(["1", "21", "22", "23"]), "basic", np.where(e.isin(["3", "4"]), "secondary", "tertiary"))
    code = x["freguesia"].str[:2].astype(int)
    x["district"] = code.map(DISTRICT)
    x.loc[code.between(30, 39), "district"] = "Madeira"
    x.loc[code.between(40, 49), "district"] = "Açores"
    x["female"] = (x["sex"] == "M").astype(int)  # INE: H homens, M mulheres
    cells = x.groupby(["district", "female", "age", "edu"], observed=True)["n"].sum().reset_index()
    cells["region"] = cells["district"].map(REGION)
    return cells


def load_eep(all_rows: bool = False) -> pd.DataFrame:
    import pyreadstat

    cols = ["ano", "Q21_1", "Q21_2", "Q21_7", "Q42_1", "Q43", "Q44_2", "Q56_1", "Q5_3", "Q5_4", "Q4_1"]
    df, _ = pyreadstat.read_dta(EEP_DTA, usecols=cols)
    out = pd.DataFrame({"year": df["ano"].astype(int)})
    for src, dst in (("Q21_2", "vote"), ("Q21_7", "prev_vote")):
        out[dst] = df[src].map(VOTE)
        out.loc[df[src].isin(OTHER), dst] = "OTH"
    out["age"] = pd.cut(df["Q42_1"].where(df["Q42_1"].between(18, 110)), AGE_EDGES, right=False, labels=AGE_LABELS)
    out["female"] = df["Q43"].map({1: 0, 2: 1})
    out["edu"] = df["Q44_2"].map({1: "basic", 2: "secondary", 3: "tertiary"})
    out["region"] = df["Q56_1"]
    d3 = df["Q5_3"].map({k + 1: v for k, v in DISTRICT.items()})
    out["district"] = df["Q5_4"].map(DISTRICT).fillna(d3)
    out["w"] = df["Q4_1"]
    if all_rows:
        return out
    keep = out.vote.isin(PARTIES) & out.region.between(1, 5) & out.age.notna() & out.female.notna() & out.edu.notna()
    return out[keep & out.year.isin(WAVES)].copy()


def availability(national) -> np.ndarray:
    """(W, K): which parties stood (>0.3% nationally) at each wave's election."""
    A = np.zeros((len(WAVES), len(PARTIES)), bool)
    for i, w in enumerate(WAVES):
        if w in ELECTION:
            A[i] = national[pd.Timestamp(ELECTION[w])].values > 0.003
        else:
            A[i] = [p in ("PS", "AD", "BE", "CDU") for p in PARTIES]
    return A


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------


def design(df: pd.DataFrame) -> pd.DataFrame:
    cols = {"icpt": np.ones(len(df)), "female": df["female"].values.astype(float)}
    for a in AGE_LABELS:
        if a != "35-44":
            cols[f"age_{a}"] = (df["age"].astype(str).values == a).astype(float)
    for e in ("secondary", "tertiary"):
        cols[f"edu_{e}"] = (df["edu"].values == e).astype(float)
    for r in (2, 3, 4, 5):
        cols[f"reg_{r}"] = (df["region"].values == r).astype(float)
    return pd.DataFrame(cols, index=df.index)


class WaveMNL:
    """Multinomial logit with one coefficient set per wave and a random-walk prior across waves."""

    def __init__(self, tau_demo=1.0, tau_reg=0.3, tau_icpt=3.0, s0=1.5, s0_icpt=5.0):
        self.tau_demo, self.tau_reg, self.tau_icpt, self.s0, self.s0_icpt = tau_demo, tau_reg, tau_icpt, s0, s0_icpt

    def fit(self, df: pd.DataFrame, waves: List[int], avail: np.ndarray) -> "WaveMNL":
        X = design(df)
        self.names = list(X.columns)
        J, K, W = X.shape[1], len(PARTIES), len(waves)
        self.waves, self.avail = list(waves), np.asarray(avail, bool)
        wi = df["year"].map({w: i for i, w in enumerate(waves)}).values
        yi = df["vote"].map({p: i for i, p in enumerate(PARTIES)}).values
        Xv = X.values
        tau = np.full(J, self.tau_demo)
        tau[0] = self.tau_icpt
        tau[self.reg_idx] = self.tau_reg
        s0 = np.full(J, self.s0)
        s0[0] = self.s0_icpt
        mask = np.where(self.avail, 0.0, -np.inf)
        free = np.ones((W, K, J), bool)
        free[:, 0, :] = False
        groups = [np.where(wi == w)[0] for w in range(W)]

        def f(theta):
            B = np.zeros((W, K, J))
            B[free] = theta
            nll, G = 0.0, np.zeros_like(B)
            for w, idx in enumerate(groups):
                if len(idx) == 0:
                    continue
                eta = Xv[idx] @ B[w].T + mask[w]
                lse = logsumexp(eta, axis=1)
                p = np.exp(eta - lse[:, None])
                nll -= (eta[np.arange(len(idx)), yi[idx]] - lse).sum()
                p[np.arange(len(idx)), yi[idx]] -= 1
                G[w] += p.T @ Xv[idx]
            pen = 0.5 * ((B[0] / s0) ** 2).sum()
            G[0] += B[0] / s0**2
            D = np.diff(B, axis=0) / tau
            pen += 0.5 * (D**2).sum()
            G[1:] += D / tau
            G[:-1] -= D / tau
            return nll + pen, G[free]

        r = minimize(f, np.zeros(free.sum()), jac=True, method="L-BFGS-B", options=dict(maxiter=5000, maxfun=20000))
        self.B = np.zeros((W, K, J))
        self.B[free] = r.x
        self.converged = bool(r.success)
        return self

    @property
    def reg_idx(self):
        return [i for i, n in enumerate(self.names) if n.startswith("reg_")]

    @property
    def demo_idx(self):
        return [i for i, n in enumerate(self.names) if n.startswith(("female", "age_", "edu_"))]

    def district_offsets(self, B_w: np.ndarray, avail_w: np.ndarray, cells: pd.DataFrame, districts) -> pd.DataFrame:
        """log(district share / national share) on the mainland implied by coefficients B_w (K, J)."""
        main = cells[cells.region <= 5]
        eta = design(main).values @ B_w.T + np.where(avail_w, 0.0, -np.inf)
        P = np.exp(eta - logsumexp(eta, axis=1)[:, None])
        num = pd.DataFrame(P * main["n"].values[:, None], columns=PARTIES).groupby(main["district"].values).sum()
        N = main.groupby("district")["n"].sum()
        pi = num.div(N, axis=0)
        nat = num.sum() / N.sum()
        return (np.log(pi.clip(lower=1e-9)) - np.log(nat.clip(lower=1e-9))).reindex(districts)


# ----------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------


def _score(pred, true_d, seats_map):
    ps = seats_from_district_shares(pred[None], seats_map.values)[0]
    ts = seats_from_district_shares(true_d[None], seats_map.values)[0]
    return float(np.abs(pred - true_d).mean() * 100), float(np.abs(ps - ts).sum() / 2)


def ceiling(tau_demo=1.0, tau_reg=0.3) -> pd.DataFrame:
    """Blend swing times exp(lam * Delta), where Delta is the change in district offsets implied by
    the change in survey coefficients between the EEP of the last election and the EEP of the
    target election itself (post-election, so an upper bound for any pre-election survey).

    demo: only the sex/age/education coefficients change (composition x demographic swing);
    region: only the NUTS-II coefficients change; full: everything changes."""
    district, national, seats_map = load_truth()
    districts = list(seats_map.index)
    A = availability(national)
    m = WaveMNL(tau_demo, tau_reg).fit(load_eep(), WAVES, A)
    cells = census_cells()
    rows = []
    for prev, E in [(2011, 2015), (2015, 2019), (2019, 2022), (2022, 2024)]:
        ip, iE = WAVES.index(prev), WAVES.index(E)
        tp, tE = pd.Timestamp(ELECTION[prev]), pd.Timestamp(ELECTION[E])
        Bp, BE = m.B[ip], m.B[iE]
        o_p = m.district_offsets(Bp, A[ip], cells, districts)
        deltas = {}
        for name, idx in (("demo", m.demo_idx), ("region", m.reg_idx)):
            Bcf = Bp.copy()
            Bcf[:, idx] = BE[:, idx]
            deltas[name] = m.district_offsets(Bcf, A[ip], cells, districts) - o_p
        deltas["full"] = m.district_offsets(BE, A[iE], cells, districts) - m.district_offsets(Bp, A[iE], cells, districts)
        base = blend_swing(national[tE].values[None], national[tp].values, district[tp].values, 0.5)[0]
        true_d = district[tE].values
        mae, seats = _score(base, true_d, seats_map)
        rows.append(dict(election=E, variant="blend", lam=0.0, district_mae_pp=mae, seats_misallocated=seats))
        for name, D in deltas.items():
            Dv = D.fillna(0).values.copy()
            Dv[:, ~A[ip]] = 0  # a party absent last time has no swing to adjust
            for lam in (0.5, 1.0):
                x = base * np.exp(lam * Dv)
                mae, seats = _score(x / x.sum(1, keepdims=True), true_d, seats_map)
                rows.append(dict(election=E, variant=name, lam=lam, district_mae_pp=mae, seats_misallocated=seats))
    return pd.DataFrame(rows)


NEW_PARTY_CASES = [(2019, "PAN"), (2019, "L"), (2022, "CH"), (2022, "IL"), (2022, "L"), (2024, "CH"), (2024, "IL"), (2024, "L")]


def new_party(oracle: bool, demo_only: bool, tau_demo=1.0, tau_reg=0.3) -> pd.DataFrame:
    """Mainland district MAE of one young party's share under: blend, uniform swing, no geography
    (national share everywhere), the survey's structural geography (MrsP offsets), and a 50/50
    mix of survey geography and blend. oracle=True fits on the target election's own survey;
    otherwise on the surveys published before it."""
    district, national, seats_map = load_truth()
    districts = list(seats_map.index)
    mainland = np.array([REGION[d] <= 5 for d in districts])
    A = availability(national)
    e = load_eep()
    cells = census_cells()
    fits: Dict[int, WaveMNL] = {}
    rows = []
    for E, party in NEW_PARTY_CASES:
        k = PARTIES.index(party)
        iE = WAVES.index(E)
        tE, tp = pd.Timestamp(ELECTION[E]), pd.Timestamp(ELECTION[WAVES[iE - 1]])
        key = E if oracle else WAVES[iE - 1]
        ws = [w for w in WAVES if w <= key]
        if key not in fits:
            fits[key] = WaveMNL(tau_demo, tau_reg).fit(e[e.year.isin(ws)], ws, A[: len(ws)])
        m = fits[key]
        Bw = m.B[len(ws) - 1].copy()
        if demo_only:
            Bw[:, m.reg_idx] = 0
        o = m.district_offsets(Bw, A[len(ws) - 1] | (np.arange(len(PARTIES)) == k), cells, districts)[party].values
        true_d = district[tE].values
        base = blend_swing(national[tE].values[None], national[tp].values, district[tp].values, 0.5)[0]
        unif = blend_swing(national[tE].values[None], national[tp].values, district[tp].values, 0.0)[0]
        struct = national[tE].values[k] * np.exp(o)
        rest = np.delete(np.arange(len(PARTIES)), k)

        def with_party(val):
            pred = base.copy()
            for di in np.where(mainland)[0]:
                pred[di, k] = val[di]
                pred[di, rest] = base[di, rest] / base[di, rest].sum() * (1 - val[di])
            return pred

        rules = {"blend": base, "uniform": unif, "no geography": with_party(np.full(len(districts), national[tE].values[k])),
                 "survey geography": with_party(struct), "50/50 survey + blend": with_party(0.5 * struct + 0.5 * base[:, k])}
        row = dict(case=f"{party} {E}", respondents=int((e[e.year == key].vote == party).sum()))
        for name, pred in rules.items():
            row[name] = float(np.abs(pred[mainland, k] - true_d[mainland, k]).mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows).set_index("case")


def _mainland_truth(years) -> Dict[int, pd.Series]:
    from src.data.flexible_loaders import load_election_results_flexible

    with contextlib.redirect_stdout(io.StringIO()):
        raw = load_election_results_flexible([ELECTION[y] for y in years], PARTIES, aggregation_level="district")
    raw["election_date"] = pd.to_datetime(raw["election_date"])
    main = raw[raw.Circulo.map(REGION).fillna(9) <= 5]
    return {t.year: g[PARTIES].sum() / g[PARTIES].values.sum() for t, g in main.groupby("election_date")}


def recall_weighting() -> pd.DataFrame:
    """Error of the EEP's reported vote (mainland) against the mainland result: raw, with the
    survey's sociodemographic weight, weighted so recalled previous vote matches the previous
    result (respondents who recall a party; factors capped at 5), and both."""
    pairs = [(2011, 2009), (2015, 2011), (2019, 2015), (2022, 2019), (2024, 2022)]
    truth = _mainland_truth(sorted({y for p in pairs for y in p}))
    e = load_eep(all_rows=True)
    e = e[e.region.between(1, 5)]
    rows = []
    for E, P in pairs:
        s = e[(e.year == E) & e.vote.isin(PARTIES)]
        demo_w = s["w"].fillna(1.0).values
        known = s.prev_vote.isin(PARTIES).values
        variants = {"raw": np.ones(len(s)), "sociodemographic weight": demo_w}
        for name, w0 in (("recalled-vote weight", np.ones(len(s))), ("both", demo_w)):
            w = w0.copy()
            sh = pd.Series(w[known], index=s.prev_vote.values[known]).groupby(level=0).sum()
            f = (truth[P].reindex(sh.index) / (sh / sh.sum())).fillna(1.0).clip(upper=5)
            w[known] *= pd.Series(s.prev_vote.values[known]).map(f).values
            variants[name] = w
        big3 = truth[E].sort_values(ascending=False).index[:3]
        for name, w in variants.items():
            est = pd.Series(w, index=s.vote.values).groupby(level=0).sum().reindex(PARTIES).fillna(0)
            err = (est / est.sum() - truth[E]) * 100
            rows.append(dict(election=E, variant=name, respondents=len(s), big3_mae_pp=float(err[big3].abs().mean()),
                             PS=err["PS"], AD=err["AD"], CH=err["CH"]))
    return pd.DataFrame(rows)


def regional_bound(n_values=(None, 40000, 20000, 10000, 5000, 2000), shrinks=(0.25, 0.5, 1.0), deff=2.0, reps=40, seed=1):
    """Blend swing applied within each mainland NUTS-II region from that region's vote, either
    known exactly (n=None, the upper bound for regional poll crosstabs) or measured with sampling
    noise from n pooled respondents (design effect deff) and shrunk toward the national blend."""
    district, national, seats_map = load_truth()
    districts = list(seats_map.index)
    years = [2011, 2015, 2019, 2022, 2024, 2025]
    from src.data.flexible_loaders import load_election_results_flexible

    with contextlib.redirect_stdout(io.StringIO()):
        raw = load_election_results_flexible([ELECTION[y] for y in years], PARTIES, aggregation_level="district")
    raw["election_date"] = pd.to_datetime(raw["election_date"])
    reg = np.array([REGION[d] for d in districts])
    mainland = reg <= 5
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        for shrink in shrinks if n is not None else (1.0,):
            maes, seats = [], []
            for prev, E in zip(years[:-1], years[1:]):
                tp, tE = pd.Timestamp(ELECTION[prev]), pd.Timestamp(ELECTION[E])
                base = blend_swing(national[tE].values[None], national[tp].values, district[tp].values, 0.5)[0]
                cnt = {t: raw[raw.election_date == t].set_index("Circulo")[PARTIES].astype(float).reindex(districts) for t in (tp, tE)}
                true_d = district[tE].values
                for _ in range(1 if n is None else reps):
                    pred = base.copy()
                    for r in range(1, 6):
                        idx = np.where(reg == r)[0]
                        rp = cnt[tp].iloc[idx].sum().values / cnt[tp].iloc[idx].values.sum()
                        rE = cnt[tE].iloc[idx].sum().values / cnt[tE].iloc[idx].values.sum()
                        if n is None:
                            est = rE
                        else:
                            neff = int(n * cnt[tE].iloc[idx].values.sum() / cnt[tE].values.sum() / deff)
                            est = rng.multinomial(neff, rE / rE.sum()) / neff
                        implied = blend_swing(national[tE].values[None], national[tp].values, rp[None], 0.5)[0, 0]
                        est = shrink * est + (1 - shrink) * implied
                        pred[idx] = blend_swing(est[None], rp, district[tp].values[idx], 0.5)[0]
                    maes.append(np.abs(pred - true_d)[mainland].mean() * 100)
                    seats.append(_score(pred, true_d, seats_map)[1])
            rows.append(dict(pooled_n="exact" if n is None else n, shrink=shrink,
                             mainland_mae_pp=float(np.mean(maes)), seats_misallocated=float(np.mean(seats))))
    # the baseline on the same footing
    maes, seats = [], []
    for prev, E in zip(years[:-1], years[1:]):
        tp, tE = pd.Timestamp(ELECTION[prev]), pd.Timestamp(ELECTION[E])
        base = blend_swing(national[tE].values[None], national[tp].values, district[tp].values, 0.5)[0]
        maes.append(np.abs(base - district[tE].values)[mainland].mean() * 100)
        seats.append(_score(base, district[tE].values, seats_map)[1])
    rows.insert(0, dict(pooled_n="none (blend)", shrink=np.nan, mainland_mae_pp=float(np.mean(maes)), seats_misallocated=float(np.mean(seats))))
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("ceiling", "new-party", "recall", "regions", "all"):
        ap.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 200)
    if args.ceiling or args.all:
        print("\n== Ceiling: demographic / regional swing from the EEP of both elections (true national)")
        for tau in ((1.0, 0.3), (3.0, 0.1)):
            r = ceiling(*tau)
            print(f"   prior sd: demographic {tau[0]}, region {tau[1]}")
            print(r.pivot_table(index=["variant", "lam"], columns="election", values=["district_mae_pp", "seats_misallocated"]).round(3).to_string())
    if args.new_party or args.all:
        for oracle in (True, False):
            for demo_only in (False, True):
                print(f"\n== Young parties, mainland district MAE (pp): {'same-election survey (oracle)' if oracle else 'surveys before the election'}, "
                      f"{'demographics only' if demo_only else 'demographics + region'}")
                print(new_party(oracle, demo_only).round(2).to_string())
    if args.recall or args.all:
        print("\n== Recalled-vote weighting of the EEP (national, mainland), error in pp")
        r = recall_weighting()
        print(r.pivot(index="variant", columns="election", values="big3_mae_pp").round(2).to_string())
        print(r.round(1).to_string(index=False))
    if args.regions or args.all:
        print("\n== Regional crosstabs: blend within NUTS-II regions from the regional vote (2015-2025)")
        print(regional_bound().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
