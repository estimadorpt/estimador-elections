"""The seat simulator's data file (src/simulator_export.py)."""

from src.simulator_export import build, dhondt


def test_dhondt_names_the_next_list_and_its_margin():
    # 100 000, 80 000, 30 000 for 7 seats: B's third quotient (26 667) takes the last seat;
    # A's fourth (25 000) is next and needs (26 666.7 - 25 000) x 4 = 6 667 more votes, plus one.
    won, last = dhondt({"A": 100000, "B": 80000, "C": 30000}, 7, ["A", "B", "C"])
    assert {p: n for p, n in won.items() if n} == {"A": 3, "B": 3, "C": 1}
    assert last["list"] == "B"
    assert last["runnerUp"] == {"list": "A", "votes": 6667}


def test_build_reproduces_the_2025_parliament_and_its_closest_last_seat():
    data = build()  # raises if the official 2025 seats are not reproduced
    assert len(data["circles"]) == 22
    assert sum(c["seats"] for c in data["circles"]) == 230
    seats = {c["name"]: c["seats"] for c in data["circles"]}
    assert seats["Setúbal"] == 19 and seats["Viana do Castelo"] == 5
    santarem = next(c for c in data["circles"] if c["name"] == "Santarém")
    assert santarem["lastSeat"] == {"list": "AD", "runnerUp": {"list": "PS", "votes": 493}}
    for year in ("2022", "2024"):
        assert abs(sum(data["presets"][year].values()) - 1) < 1e-9
