"""The seat simulator's data file: the 2025 legislative result in all 22 circles.

The website's simulator (estimador-web, "Mover votos" on /eleicoes/legislativas) starts from
the 2025 result and moves national votes between lists. It needs, for every circle, the votes
of each list, the seats, and the national shares of earlier elections for its presets. This
module writes that file; the browser does every calculation.

Sources:
- the 20 domestic circles: `data/legislativas_2025.parquet` (parish results) summed to circles;
  with the 2025 seat map they reproduce the official 226 domestic seats exactly;
- the two emigration circles: `data/legislativas_2025_emigracao.csv` (Mapa Oficial n.º 2-A/2025,
  as tabled on Wikipedia's constituency pages, checked against their totals);
- the seat map: `data/legislativas_2025_circulos_deputados.csv` (CNE Mapa Oficial n.º 1/2025;
  Setúbal 19 and Viana do Castelo 5 since 2024).

Lists: the eight parties the reader can move (AD, PS, CH, IL, L, CDU, BE, PAN), JPP (held at
its 2025 share in every circle: it stands to win only in Madeira and must not swing nationally)
and OUT, every other list together (moved as one; never given seats, since no single one of
them came near a seat in 2025).

Usage: python -m src.simulator_export --out ../estimador-web/public/data/elections/parliamentary-2025/simulator.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

MOVABLE = ["AD", "PS", "CH", "IL", "L", "CDU", "BE", "PAN"]
FIXED = ["JPP"]
OTHERS = "OUT"
LISTS = MOVABLE + FIXED + [OTHERS]
EMIGRATION = ["Europa", "Fora da Europa"]

# Columns of the parish files that are not lists.
META = {
    "territory_code",
    "territory_name",
    "territoryCode",
    "territoryName",
    "number_voters",
    "percentage_voters",
    "subscribed_voters",
    "null_votes",
    "blank_votes",
}

# The AD of each year: the PSD and CDS lists and their regional coalitions together.
AD_COLUMNS = ["PPD/PSD.CDS-PP.PPM", "PPD/PSD.CDS-PP", "PPD/PSD", "CDS-PP"]
RENAME = {"PCP-PEV": "CDU", "B.E.": "BE"}

DISTRICTS = {
    "01": "Aveiro",
    "02": "Beja",
    "03": "Braga",
    "04": "Bragança",
    "05": "Castelo Branco",
    "06": "Coimbra",
    "07": "Évora",
    "08": "Faro",
    "09": "Guarda",
    "10": "Leiria",
    "11": "Lisboa",
    "12": "Portalegre",
    "13": "Porto",
    "14": "Santarém",
    "15": "Setúbal",
    "16": "Viana do Castelo",
    "17": "Vila Real",
    "18": "Viseu",
}


def circle_of(territory_code: str) -> str:
    """The circle of a parish code (LOCAL-DDMMFF): mainland districts, Madeira (3x), Açores (4x)."""
    code = territory_code.replace("LOCAL-", "")[:2]
    if code in DISTRICTS:
        return DISTRICTS[code]
    if code.startswith("3"):
        return "Madeira"
    if code.startswith("4"):
        return "Açores"
    raise ValueError(f"Unknown circle for parish {territory_code}")


def parish_lists(year: int) -> pd.DataFrame:
    """Votes per parish and list (AD merged, every other small list in OUT), with the circle."""
    d = pd.read_parquet(os.path.join(DATA_DIR, f"legislativas_{year}.parquet"))
    d = d.rename(columns={"territoryCode": "territory_code", "territoryName": "territory_name"})
    cols = [c for c in d.columns if c not in META]
    x = d[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out = pd.DataFrame(index=d.index)
    out["AD"] = x[[c for c in AD_COLUMNS if c in x]].sum(axis=1)
    named = {RENAME.get(c, c): c for c in cols}
    for p in MOVABLE[1:] + FIXED:
        out[p] = x[named[p]] if p in named else 0.0
    used = set(c for c in AD_COLUMNS if c in x) | {named[p] for p in MOVABLE[1:] + FIXED if p in named}
    out[OTHERS] = x[[c for c in cols if c not in used]].sum(axis=1)
    out["circle"] = d["territory_code"].map(circle_of)
    out["registered"] = pd.to_numeric(d["subscribed_voters"], errors="coerce").fillna(0)
    out["voters"] = pd.to_numeric(d["number_voters"], errors="coerce").fillna(0)
    return out


def emigration_lists() -> pd.DataFrame:
    e = pd.read_csv(os.path.join(DATA_DIR, "legislativas_2025_emigracao.csv"))
    e["list"] = e["Lista"].where(e["Lista"].isin(MOVABLE + FIXED), OTHERS)
    return e.pivot_table(index="Circulo", columns="list", values="Votos", aggfunc="sum").reindex(columns=LISTS).fillna(0)


def seat_map() -> pd.Series:
    s = pd.read_csv(os.path.join(DATA_DIR, "legislativas_2025_circulos_deputados.csv"), encoding="utf-8-sig")
    return s.set_index("Circulo")["Deputados"]


def dhondt(votes: Dict[str, float], seats: int, competing: Sequence[str]) -> Tuple[Dict[str, int], dict]:
    """D'Hondt among `competing` lists, with the last seat and the runner-up.

    Ties go to the list with fewer votes, as the electoral law says (Lei n.º 14/79, art. 16.º),
    then to the earlier one in `competing`; the website's engine copies the rule. Returns the
    seats per list and the last seat: its list, quotient, the runner-up (the next quotient in
    line, of another list) and the extra votes the runner-up needed to take the seat,
    everything else equal.
    """
    quotients = []
    for i, p in enumerate(competing):
        v = float(votes.get(p, 0.0))
        if v <= 0:
            continue
        for k in range(1, seats + 1):
            # reverse sort: larger quotient first, then fewer votes, then the earlier list
            quotients.append((v / k, -v, -i, p, k))
    quotients.sort(reverse=True)
    won = {p: 0 for p in competing}
    for q in quotients[:seats]:
        won[q[3]] += 1
    if len(quotients) < seats:
        raise ValueError("no list has votes")
    last = quotients[seats - 1]
    # The runner-up is the best next quotient of a list other than the holder: the holder's
    # own next quotient cannot take the seat from itself.
    runner = next((q for q in quotients[seats:] if q[3] != last[3]), None)
    margin = None
    if runner is not None:
        margin = {"list": runner[3], "votes": votes_to_pass(last, runner)}
    return won, {"list": last[3], "quotient": last[0], "runnerUp": margin}


def votes_to_pass(last: tuple, runner: tuple) -> int:
    """The fewest extra votes that give the runner-up the last seat, everything else equal.

    It needs a larger quotient, or an equal one with fewer votes than the holder (the law's
    tie rule; with equal votes too the law does not say, so it is not counted as a win). The
    tolerance keeps a whole-vote gap whole despite float error.
    """
    gap = (last[0] - runner[0]) * runner[4]
    n = int(round(gap))
    if abs(gap - n) < 1e-6:
        return n if -runner[1] + n < -last[1] else n + 1
    return int(np.floor(gap)) + 1


def national_shares(year: int) -> Dict[str, float]:
    """Domestic national shares of the movable lists and OUT (JPP left out), summing to 1."""
    p = parish_lists(year)
    tot = p[MOVABLE + [OTHERS]].sum()
    return {k: float(v) for k, v in (tot / tot.sum()).items()}


def build() -> dict:
    seats = seat_map()
    parish = parish_lists(2025)
    dom = parish.groupby("circle")[LISTS + ["registered", "voters"]].sum()
    emi = emigration_lists()
    circles: List[dict] = []
    totals = {p: 0 for p in LISTS}
    for name, k in seats.items():
        fixed = name in EMIGRATION
        row = emi.loc[name] if fixed else dom.loc[name]
        votes = {p: int(round(float(row[p]))) for p in LISTS}
        won, last = dhondt(votes, int(k), MOVABLE + FIXED)
        for p, n in won.items():
            totals[p] += n
        circles.append(
            {
                "name": name,
                "seats": int(k),
                "fixed": fixed,
                "votes": votes,
                "seats2025": {p: n for p, n in won.items() if n},
                "lastSeat": {"list": last["list"], "runnerUp": last["runnerUp"]},
            }
        )
    expected = {"AD": 91, "PS": 58, "CH": 60, "IL": 9, "L": 6, "CDU": 3, "BE": 1, "PAN": 1, "JPP": 1}
    got = {p: n for p, n in totals.items() if n}
    if got != expected:
        raise AssertionError(f"2025 seats not reproduced: {got}")
    return {
        "version": 1,
        "election": "2025-05-18",
        "generated": date.today().isoformat(),
        "seatMap": "CNE, Mapa Oficial n.º 1/2025",
        "sources": [
            "Resultados por freguesia, eleicoes.mai.gov.pt (20 círculos do território nacional)",
            "Mapa Oficial n.º 2-A/2025 (círculos da Europa e de Fora da Europa)",
        ],
        "movable": MOVABLE,
        "fixed": FIXED,
        "others": OTHERS,
        "seats2025": got,
        "circles": circles,
        "presets": {str(y): national_shares(y) for y in (2022, 2024)},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="outputs/simulator/simulator.json")
    args = ap.parse_args(argv)
    data = build()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {args.out}: {len(data['circles'])} circles, seats {data['seats2025']}")


if __name__ == "__main__":
    main()
