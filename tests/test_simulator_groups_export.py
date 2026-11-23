"""The seat simulator's group file (src/simulator_groups_export.py)."""

import pandas as pd
import pytest

from src.simulator_export import LISTS, parish_lists
from src.simulator_groups_export import AGES, EDUCATION, OPTIONS, SEXES, build


def _rates(tmp_path, nudge=0.0):
    """A rates file whose cells add up to the official 2025 result: each circle's whole vote in one
    group (men, 35-44, basic), every other group empty, and some abstention spread evenly."""
    official = parish_lists(2025).groupby("circle")[LISTS + ["voters"]].sum()
    rows = []
    for circle, o in official.iterrows():
        totals = {p: float(o[p]) for p in LISTS}
        totals["BLANK_NULL"] = float(o["voters"] - sum(o[p] for p in LISTS))
        totals["ABS"] = 1000.0
        for s in SEXES:
            for a in AGES:
                for e in EDUCATION:
                    home = (s, a, e) == ("men", "35-44", "basic")
                    for p in OPTIONS:
                        if p == "ABS":
                            count = totals["ABS"] / (len(SEXES) * len(AGES) * len(EDUCATION))
                        else:
                            count = totals[p] if home else 0.0
                        rows.append({"district": circle, "sex": s, "age": a, "edu": e, "party": p, "count": count})
    df = pd.DataFrame(rows)
    df.loc[0, "count"] += nudge
    path = tmp_path / "rates.csv"
    df.to_csv(path, index=False)
    return path


def test_build_writes_twenty_circles_of_42_groups_in_the_sites_order(tmp_path):
    data = build(str(_rates(tmp_path)))
    assert data["options"] == OPTIONS and OPTIONS[-2:] == ["BLANK_NULL", "ABS"]
    assert len(data["circles"]) == 20
    assert all(len(c["cells"]) == 42 and all(len(r) == len(OPTIONS) for r in c["cells"]) for c in data["circles"])
    assert not {"Europa", "Fora da Europa"} & {c["name"] for c in data["circles"]}


def test_build_refuses_cells_that_do_not_add_up_to_the_official_result(tmp_path):
    with pytest.raises(AssertionError):
        build(str(_rates(tmp_path, nudge=10.0)))
