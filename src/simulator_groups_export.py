"""The seat simulator's group file: 2025 voters by circle, sex, age, education and vote.

"Mover votos por grupos" on estimador-web's /eleicoes/legislativas lets a reader move a share of
a group (say, a fifth of the under-35s who abstained) to another list or to abstention. It needs,
for each of the 20 domestic circles, how many people of each group voted for each list, blank or
null, or abstained in 2025.

The input is the survey thread's `group_rates_2025.csv` (estimador-elections PR #49,
`src/evaluation/vote_transitions/group_rates.py`): Censos 2021 resident citizens aged 18+ per district, by sex ×
7 age bands × 3 education levels; group gaps from a multinomial logit on the Estudo Eleitoral
Português 2024; one shift per list and district so the cells add up exactly to the 2025 result
(abstention = resident citizens − voters). This module checks it against the official result and
writes it in the website's compact form: per circle, one row of counts per group, in the order
sex, then age, then education, and the options in `OPTIONS` order.

Usage: python -m src.simulator_groups_export --rates group_rates_2025.csv \
    --out ../estimador-web/public/data/elections/parliamentary-2025/simulator-groups.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date

import pandas as pd

from src.simulator_export import FIXED, LISTS, MOVABLE, OTHERS, parish_lists

SEXES = ["men", "women"]
AGES = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
EDUCATION = ["basic", "secondary", "tertiary"]
OPTIONS = MOVABLE + FIXED + [OTHERS, "BLANK_NULL", "ABS"]


def build(rates_path: str) -> dict:
    r = pd.read_csv(rates_path)
    missing = set(OPTIONS) ^ set(r["party"].unique())
    if missing:
        raise ValueError(f"options differ from the site's: {sorted(missing)}")
    official = parish_lists(2025).groupby("circle")[LISTS + ["voters"]].sum()
    official["BLANK_NULL"] = official["voters"] - official[LISTS].sum(axis=1)
    circles = []
    for name, d in r.groupby("district", sort=True):
        t = d.pivot_table(index=["sex", "age", "edu"], columns="party", values="count", aggfunc="sum")
        t = t.reindex(pd.MultiIndex.from_product([SEXES, AGES, EDUCATION]), columns=OPTIONS)
        if t.isna().any().any():
            raise ValueError(f"{name}: a group or an option is missing")
        # The cells must add up to the official 2025 result of the circle.
        for p in LISTS + ["BLANK_NULL"]:
            if abs(t[p].sum() - official.loc[name, p]) > 0.5:
                raise AssertionError(f"{name} {p}: {t[p].sum():.1f} vs official {official.loc[name, p]}")
        circles.append({"name": name, "cells": [[int(round(v)) for v in row] for row in t.to_numpy()]})
    if len(circles) != 20:
        raise AssertionError(f"expected the 20 domestic circles, found {len(circles)}")
    return {
        "version": 1,
        "election": "2025-05-18",
        "generated": date.today().isoformat(),
        "sources": [
            "Censos 2021 (INE): cidadãos portugueses residentes com 18 ou mais anos, por sexo, idade e escolaridade",
            "Estudo Eleitoral Português 2024 (ICS): diferenças entre grupos",
            "Resultados oficiais de 2025 por círculo",
        ],
        "sex": SEXES,
        "age": AGES,
        "edu": EDUCATION,
        "options": OPTIONS,
        "circles": circles,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rates", required=True, help="group_rates_2025.csv from src/evaluation/vote_transitions/group_rates.py (PR #49)")
    ap.add_argument("--out", default="outputs/simulator/simulator-groups.json")
    args = ap.parse_args(argv)
    data = build(args.rates)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {args.out}: {len(data['circles'])} circles x {len(data['circles'][0]['cells'])} groups x {len(OPTIONS)} options")


if __name__ == "__main__":
    main()
