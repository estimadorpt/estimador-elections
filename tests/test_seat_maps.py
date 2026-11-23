"""The seat map per election, recomputed from registered voters."""

import pandas as pd

from src.data.loaders import load_district_config
from src.data.seat_maps import dhondt_apportion, seat_map


def test_dhondt_apportion_small_case():
    # 7 seats, weights 100/80/30: the 7 largest averages are a 100, b 80, a 50, b 40, a 33.3,
    # c 30, b 26.7 -> 3, 3, 1
    assert dhondt_apportion({"a": 100, "b": 80, "c": 30}, 7).to_dict() == {"a": 3, "b": 3, "c": 1}


def test_maps_by_election():
    m = {y: seat_map(d) for y, d in [(2022, "2022-01-30"), (2024, "2024-03-10"), (2025, "2025-05-18"), (2015, "2015-10-04")]}
    for y, s in m.items():
        assert len(s) == 20 and int(s.sum()) == 226, y
    # 2024: Setúbal gains a seat, Viana do Castelo loses one; 2025 keeps the 2024 map
    assert (m[2024]["Setúbal"], m[2024]["Viana do Castelo"]) == (19, 5)
    assert (m[2022]["Setúbal"], m[2022]["Viana do Castelo"]) == (18, 6)
    assert m[2025].equals(m[2024])
    # 2015 is the map before Lisbon and Porto gained seats from Viseu and Guarda (2019)
    assert (m[2015]["Lisboa"], m[2015]["Porto"], m[2015]["Viseu"], m[2015]["Guarda"]) == (47, 39, 9, 4)


def test_seat_file_is_the_latest_map():
    """The file the live forecasts read is the map of the latest election (2024 and 2025)."""
    cfg = pd.Series(load_district_config()).sort_index()
    assert cfg.to_dict() == seat_map("2025-05-18").to_dict()
