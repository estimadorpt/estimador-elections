"""Evaluate the dynamics grid against the decision rule stated before the results (23 Nov 2026).

Convergence (per fit): 0 divergences, and the forecast draws themselves (national shares,
chain-major (4, n, P)) with max rhat <= 1.01 and min bulk ESS >= 400. Failing fits are left out.
The short-term scale's own rhat/ESS is reported, not gated (a slowly mixing nuisance scale).

A candidate C (dyn_ou4s primary; dyn_ou2s, dyn_ou8s neighbours; dyn_m32_4s at h7) "wins" over
dyn_rw only if:
  A. it beats dyn_rw on election-day CRPS at >= 2 horizons in >= 3 of 5 elections
     (both fits converged; an election needs >= 2 comparable horizons to count as won);
  B. it is not worse on the polls published after each cutoff: paired per-poll CRPS
     differences (C - RW) pooled over elections and horizons; "worse" = 95% bootstrap CI of
     the mean difference entirely above 0;
  C. both neighbours (2 and 8 weeks) point the same way: mean CRPS difference vs RW < 0.
Usage: python scripts/dynamics/eval_dyn.py outputs/backtest/WIP_dyn_v2
"""

import glob
import json
import os
import sys

import arviz as az
import numpy as np
import pandas as pd

OUT = sys.argv[1]
MODELS = ["dyn_rw", "dyn_ou4s", "dyn_ou2s", "dyn_ou8s", "dyn_m32_4s", "poll_average", "last_result"]
rows, polls = [], []
for f in sorted(glob.glob(os.path.join(OUT, "*__h0*.json"))):
    d = json.load(open(f))
    s = d["summary"]
    if s["model"] not in MODELS:
        continue
    npy = os.path.join(OUT, f"{s['fit']}__{s['election']}__h{int(s['horizon']):03d}__national.npy")
    conv = dict(draw_rhat=np.nan, draw_ess=np.nan, converged=s["model"] in ("poll_average", "last_result"))
    if os.path.exists(npy) and s["model"].startswith("dyn"):
        x = np.load(npy)
        y = x.reshape(4, x.shape[0] // 4, x.shape[1])
        keep = y.std(axis=(0, 1)) > 1e-9  # parties with no draws spread (absent) carry no rhat
        rh = max(float(az.rhat(y[:, :, j])) for j in np.where(keep)[0])
        es = min(float(az.ess(y[:, :, j])) for j in np.where(keep)[0])
        conv = dict(draw_rhat=rh, draw_ess=es, converged=(s.get("diag_divergences", 1) == 0 and rh <= 1.01 and es >= 400))
    rows.append(dict(model=s["model"], election=s["election"], h=int(s["horizon"]), crps=s["nat_crps"], mae=s["nat_mae"],
                     seat_mae=s["seat_mae"], post_crps=s.get("post_poll_crps", np.nan), post_n=s.get("post_n_polls", 0),
                     div=s.get("diag_divergences"), max_rhat=s.get("diag_max_rhat"), min_ess=s.get("diag_min_ess_bulk"),
                     ss_mean=s.get("sigma_short_mean"), ss_rhat=s.get("sigma_short_max_rhat"), ss_ess=s.get("sigma_short_min_ess"), **conv))
    for p in d["extra"].get("post_polls", []):
        polls.append(dict(model=s["model"], election=s["election"], h=int(s["horizon"]), key=(p["date"], p["pollster"], p["n"]), crps=p["crps"]))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("FITS\n", df.sort_values(["election", "h", "model"]).round(3).to_string(index=False))
print("\nnot converged:", df[~df.converged][["model", "election", "h", "draw_rhat", "draw_ess", "div"]].to_string(index=False))

ok = df[df.converged]
pv = ok.pivot_table(index=["election", "h"], columns="model", values="crps")
print("\nELECTION-DAY CRPS (converged fits)\n", pv.round(3).to_string())
print("\nMEAN BY HORIZON\n", ok.pivot_table(index="h", columns="model", values=["crps", "mae", "seat_mae", "post_crps"]).round(3).to_string())

pp = pd.DataFrame(polls)
rng = np.random.default_rng(0)


def paired(c):
    a = pp[pp.model == c].set_index(["election", "h", "key"])["crps"]
    b = pp[pp.model == "dyn_rw"].set_index(["election", "h", "key"])["crps"]
    conv_c = set(map(tuple, ok[ok.model == c][["election", "h"]].values))
    conv_b = set(map(tuple, ok[ok.model == "dyn_rw"][["election", "h"]].values))
    j = a.to_frame("c").join(b.to_frame("rw"), how="inner")
    j = j[[(e, h) in conv_c and (e, h) in conv_b for e, h, _ in j.index]]
    diff = (j["c"] - j["rw"]).values
    if len(diff) == 0:
        return np.nan, (np.nan, np.nan), 0
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
    return float(diff.mean()), tuple(np.quantile(boots, [0.025, 0.975])), len(diff)


print("\nDECISION RULE (vs dyn_rw)")
verdict = {}
for c in ["dyn_ou4s", "dyn_ou2s", "dyn_ou8s", "dyn_m32_4s"]:
    if c not in pv.columns or "dyn_rw" not in pv.columns:
        print(c, "no comparable fits")
        continue
    sub = pv[[c, "dyn_rw"]].dropna()
    won = 0
    per_e = {}
    for e, g in sub.groupby(level=0):
        wins = int((g[c] < g["dyn_rw"]).sum())
        per_e[e] = f"{wins}/{len(g)}"
        won += int(len(g) >= 2 and wins >= 2)
    mean_diff = float((sub[c] - sub["dyn_rw"]).mean())
    pm, ci, n = paired(c)
    A = won >= 3
    B = not (ci[0] > 0)
    verdict[c] = dict(A=A, B=B, mean_crps_diff=mean_diff)
    print(f"{c}: elections won {won}/5 {per_e}  A={A} | mean CRPS diff {mean_diff:+.3f} | post-cutoff polls paired diff {pm:+.3f} CI [{ci[0]:+.3f}, {ci[1]:+.3f}] n={n} B={B}")
if "dyn_ou4s" in verdict:
    v = verdict["dyn_ou4s"]
    neigh = [verdict.get(k, {}).get("mean_crps_diff", np.nan) for k in ("dyn_ou2s", "dyn_ou8s")]
    C = all(np.isfinite(neigh)) and all(x < 0 for x in neigh)
    print(f"\nPRIMARY dyn_ou4s: A={v['A']} B={v['B']} C(neighbours {neigh[0]:+.3f}, {neigh[1]:+.3f} both < 0)={C}  =>  "
          f"{'SHORT TERM WINS' if (v['A'] and v['B'] and C) else 'RW STAYS (rule not met)'}")
