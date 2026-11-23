"""Fixture for the website simulator's engine test: blend swing and D'Hondt on random scenarios.

The website ports `src/processing/district_swing.blend_swing` to TypeScript
(estimador-web `src/lib/simulator/engine.ts`). This script runs the Python on a few random
national vote vectors and writes the inputs and outputs, so the TypeScript test checks the
port against the original. Needs the simulator file (`python -m src.simulator_export`) and
`district_swing.py` (on the backtest branch until it merges).

Usage: python scripts/simulator_fixture.py SIMULATOR_JSON OUT_JSON
"""

import json
import sys

import numpy as np

sys.path.insert(0, ".")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("district_swing", "src/processing/district_swing.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)


def main(sim_path, out_path, n=6, seed=7):
    sim = json.load(open(sim_path, encoding="utf-8"))
    swing = sim["movable"] + [sim["others"]]
    fixed = sim["fixed"]
    circles = [c for c in sim["circles"] if not c["fixed"]]
    votes = np.array([[c["votes"][p] for p in swing] for c in circles], float)
    fixed_votes = np.array([[c["votes"][p] for p in fixed] for c in circles], float)
    total = votes.sum(1) + fixed_votes.sum(1)
    fixed_share = fixed_votes / total[:, None]
    last_district = votes / votes.sum(1, keepdims=True)
    last_national = votes.sum(0) / votes.sum()
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        # a national vector near 2025 (and one with a party at zero, one with a big move)
        nat = last_national * np.exp(rng.normal(0, 0.35, len(swing)))
        if i == 1:
            nat[swing.index("BE")] = 0.0
        if i == 2:
            nat[swing.index("CH")] *= 1.6
        nat = nat / nat.sum()
        core = ds.blend_swing(nat[None, :], last_national, last_district)[0]
        shares = core * (1 - fixed_share.sum(1, keepdims=True))
        full = np.concatenate([shares, fixed_share], axis=1)
        competing = [j for j, p in enumerate(swing + fixed) if p != sim["others"]]
        seats = []
        for d, c in enumerate(circles):
            s = ds.dhondt_draws(full[d, competing][None, :], c["seats"])[0]
            seats.append({ (swing + fixed)[j]: int(k) for j, k in zip(competing, s) if k})
        cases.append({"national": dict(zip(swing, nat.tolist())),
                      "shares": [dict(zip(swing + fixed, row.tolist())) for row in full],
                      "seats": seats})
    json.dump({"circles": [c["name"] for c in circles], "cases": cases}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {out_path}: {len(cases)} cases")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
