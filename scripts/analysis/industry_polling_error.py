"""Industry polling error per party and election, its persistence, and late shifts (read-only).

Polls: every poll the backtest dataset loads (Marktest history + Wikipedia since 2025), shares
among the 8 modelled parties; a party a poll did not report is left out of that poll's average.
Truth: national shares among the same 8 parties over the 20 domestic districts (load_truth).
Model estimates: per-cycle industry bias from saved rw fits with a per-cycle bias form.
"""
import json
import sys

import numpy as np
import pandas as pd

from src.evaluation.backtest import PARTIES, load_truth, make_dataset



def md(t):
    t = t.copy()
    t.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    head = "| " + " | ".join([str(t.index.name or "")] + list(t.columns)) + " |"
    sep = "|" + "---|" * (len(t.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v))) for v in r]) + " |" for i, r in zip(t.index, t.values)]
    return "\n".join([head, sep] + body)

OUT = sys.argv[1]
ELECTIONS = ["2011-06-05", "2015-10-04", "2019-10-06", "2022-01-30", "2024-03-10", "2025-05-18"]
LEFT, RIGHT = ["PS", "BE", "CDU", "L", "PAN"], ["AD", "CH", "IL"]
NEW = {("PAN", "2011"), ("PAN", "2015"), ("L", "2015"), ("L", "2019"), ("CH", "2019"), ("CH", "2022"), ("IL", "2019"), ("IL", "2022")}

district, national, _ = load_truth()
polls = make_dataset("2025-05-18", pd.Timestamp("2025-05-18")).polls_train.copy()
polls["date"] = pd.to_datetime(polls["date"])
cnt = polls[PARTIES].astype(float)
share = cnt.div(cnt.sum(1), axis=0) * 100
share = share.where(cnt > 0)  # unreported party: out of that poll's average


def window(E, lo, hi):
    d = (pd.Timestamp(E) - polls["date"]).dt.days
    m = (d > lo) & (d <= hi)
    return share[m].mean(0), int(m.sum())


rows = []
for E in ELECTIONS:
    y = national[pd.Timestamp(E)] * 100
    w1, n1 = window(E, 0, 7)
    w3, n3 = window(E, 0, 21)
    w8, n8 = window(E, 42, 84)
    for p in PARTIES:
        if y[p] < 0.3:
            continue
        rows.append(dict(election=E[:4], party=p, result=y[p], final_week=w1[p] - y[p], final_3w=w3[p] - y[p],
                         wk8=w8[p] - y[p], late_shift=w3[p] - w8[p], n_week=n1, n_3w=n3, n_8wk=n8))
df = pd.DataFrame(rows)

# model's per-cycle industry bias (log-ratio) -> pp at the result: p_j (b_j - sum_k p_k b_k)
model_bias = {}
for label, path in [
    ("rw_persistent (mean + per-cycle), 2025-target fit, 55f2437",
     "/Users/bernardocaldas/code/estimador-elections-backtest/outputs/backtest/WIP_grid_exp/rw_persistent__2025-05-18__h007.json"),
    ("rw (per-cycle around 0), 2025-target fit, 8ec4578",
     "/Users/bernardocaldas/code/estimador-elections-backtest/outputs/backtest/WIP_grid_v1/rw__2025-05-18__h007.json"),
]:
    b = json.load(open(path))["extra"]["bias_by_cycle"]
    for E in ELECTIONS[:-1]:  # the 2025 cycle is the target: not pinned by its result
        if E not in b:
            continue
        p = national[pd.Timestamp(E)]
        bb = pd.Series(b[E]).reindex(PARTIES).fillna(0.0)
        pp = p * (bb - (p * bb).sum()) * 100
        for party in PARTIES:
            if p[party] * 100 >= 0.3:
                model_bias[(label, E[:4], party)] = pp[party]
df["model_persistent"] = [model_bias.get((list(dict.fromkeys(k[0] for k in model_bias))[0], e, p), np.nan) for e, p in zip(df.election, df.party)]
df["model_percycle"] = [model_bias.get((list(dict.fromkeys(k[0] for k in model_bias))[1], e, p), np.nan) for e, p in zip(df.election, df.party)]


def persistence(d, col):
    d = d.sort_values("election")
    pairs = []
    for p, g in d.groupby("party"):
        v = g[col].dropna().values
        pairs += [(v[i], v[i + 1]) for i in range(len(v) - 1)]
    a = np.array(pairs)
    return len(a), float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), float(np.mean(np.sign(a[:, 0]) == np.sign(a[:, 1])))


lines = ["# Industry polling error, Portuguese legislative elections 2011-2025", "",
         "Error = poll average minus result, pp, among the 8 modelled parties (negative = polls underestimated).",
         "Windows: final week (0-7 days), final 3 weeks (0-21 days), 8 weeks out (42-84 days before, i.e. 6-12 weeks). "
         "A party a poll did not report is left out of that poll's average.", ""]
for col, title in [("final_3w", "Final 3 weeks"), ("final_week", "Final week"), ("model_persistent", "Model: per-cycle bias at the result (rw_persistent)"),
                   ("model_percycle", "Model: per-cycle bias at the result (rw per-cycle, older code)"), ("late_shift", "Late shift: final 3 weeks minus 8 weeks out")]:
    t = df.pivot(index="party", columns="election", values=col).reindex(PARTIES)
    t["mean"] = t.mean(1)
    t["sign"] = [" ".join("+" if x > 0.05 else ("-" if x < -0.05 else "0") for x in r.dropna().values[:-1]) for _, r in t.iterrows()]
    lines += [f"## {title}", "", md(t.round(2)), ""]
n = df.groupby("election")[["n_week", "n_3w", "n_8wk"]].first()
lines += ["Polls per window:", "", md(n), ""]

lines += ["## Persistence: does a party's error predict its next one?", "",
          "| measure | pairs | correlation | sign agreement |", "|---|---|---|---|"]
for col in ["final_3w", "final_week", "model_persistent"]:
    k, r, s = persistence(df, col)
    lines.append(f"| {col}, all parties | {k} | {r:.2f} | {s:.2f} |")
    k, r, s = persistence(df[~df.party.isin(["PAN", "L"])], col)
    lines.append(f"| {col}, without PAN and L | {k} | {r:.2f} | {s:.2f} |")
lines.append("")

bloc = []
for E in ELECTIONS:
    g = df[df.election == E[:4]]
    for name, members in (("left", LEFT), ("right", RIGHT)):
        gg = g[g.party.isin(members)]
        bloc.append(dict(election=E[:4], bloc=name, final_3w=gg.final_3w.sum(), final_week=gg.final_week.sum(), late_shift=gg.late_shift.sum()))
bl = pd.DataFrame(bloc)
bp = []
for name, g in bl.groupby("bloc"):
    v = g.sort_values("election").final_3w.values
    bp += [(v[i], v[i + 1]) for i in range(len(v) - 1)]
bp = np.array(bp)
lines += [f"Bloc persistence (final 3 weeks): {len(bp)} pairs, correlation {np.corrcoef(bp[:,0], bp[:,1])[0,1]:.2f}, sign agreement {np.mean(np.sign(bp[:,0]) == np.sign(bp[:,1])):.2f}", ""]
lines += ["## Blocs (sum of party errors, pp)", "", md(bl.pivot(index="bloc", columns="election", values="final_3w").round(2)), "",
          "Late shift by bloc (final 3 weeks minus 8 weeks out):", "",
          md(bl.pivot(index="bloc", columns="election", values="late_shift").round(2)), ""]
df["new"] = [(p, e) in NEW for p, e in zip(df.party, df.election)]
nv = df.groupby("new")[["final_3w", "final_week"]].agg(["mean", lambda x: x.abs().mean(), "count"]).round(2)
lines += ["## New parties (first two elections) vs established", "", md(nv), ""]
ls = df.groupby("party").agg(abs_late_shift=("late_shift", lambda x: x.abs().mean()), abs_final_week=("final_week", lambda x: x.abs().mean()),
                             abs_final_3w=("final_3w", lambda x: x.abs().mean())).reindex(PARTIES).round(2)
lines += ["## Late shift vs final-week error (mean absolute, pp)", "", md(ls), ""]
open(OUT, "w").write("\n".join(lines))
print("\n".join(lines))
