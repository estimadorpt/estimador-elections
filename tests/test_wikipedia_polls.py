"""Tests for the Wikipedia poll source, exit-poll filtering and the AD results fix."""

import hashlib
import os
from datetime import date

import pandas as pd
import pytest

from src.config import DATA_DIR
from src.data.coalition_manager import CoalitionManager
from src.data.loaders import combine_poll_sources, drop_election_day_polls
from src.data.wikipedia_polls import (
    effective_sample_sizes,
    load_wikipedia_polls,
    parse_fieldwork,
    parse_wikitext_poll_table,
)

SNAPSHOT = os.path.join(DATA_DIR, "wikipedia", "legislative_polls_2026-11-23.wikitext")
SNAPSHOT_SHA256 = "8ad812874fd5d7e778fe2b1d4e33c515922902f8c33d09fe920c9afcde4259ca"
CSV = os.path.join(DATA_DIR, "wikipedia_legislative_polls.csv")


@pytest.fixture(scope="module")
def parsed():
    with open(SNAPSHOT, encoding="utf-8") as fh:
        return parse_wikitext_poll_table(fh.read())


def test_snapshot_is_the_fetched_page():
    with open(SNAPSHOT, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == SNAPSHOT_SHA256


def test_parses_every_poll_and_skips_results(parsed):
    # 50 polls from June 2025 to November 2026; the 2025 legislative and local
    # election rows are results, not polls.
    assert len(parsed) == 50
    assert parsed["fieldwork_end"].min() == date(2025, 6, 15)
    assert parsed["fieldwork_end"].max() == date(2026, 11, 18)
    assert date(2025, 10, 12) not in set(parsed["fieldwork_end"])
    assert date(2025, 5, 18) not in set(parsed["fieldwork_end"])


def test_values_match_the_page(parsed):
    latest = parsed.iloc[-1]
    assert latest["pollster"] == "Intercampus"
    assert latest["sample_size"] == 609
    assert (latest["AD"], latest["PS"], latest["CH"], latest["IL"]) == (22.9, 28.2, 23.8, 8.9)
    # A comma in the sample size and a bold, shaded leader
    big = parsed[(parsed["pollster"] == "Aximage") & (parsed["fieldwork_end"] == date(2026, 11, 2))].iloc[0]
    assert big["sample_size"] == 1063 and big["PS"] == 28.8 and big["turnout"] == 80.1
    # "<1" is read as half a point
    cesop = parsed[(parsed["pollster"] == "CESOP-U.Católica") & (parsed["fieldwork_end"] == date(2025, 12, 12))].iloc[0]
    assert cesop["PAN"] == 0.5


def test_shares_sum_to_about_100(parsed):
    totals = parsed[["AD", "PS", "CH", "IL", "L", "CDU", "BE", "PAN", "JPP", "Others"]].sum(axis=1)
    assert totals.between(99, 101).all()


def test_pollster_names_follow_the_marktest_list(parsed):
    assert set(parsed["pollster"]) == {
        "Intercampus", "Aximage", "Pitagórica", "ICS/ISCTE/GFK Metris", "CESOP-U.Católica",
    }


def test_committed_csv_matches_the_snapshot(parsed):
    committed = pd.read_csv(CSV, parse_dates=["fieldwork_start", "fieldwork_end"])
    committed["footnote"] = committed["footnote"].fillna("")
    rebuilt = parsed.copy()
    rebuilt["fieldwork_start"] = pd.to_datetime(rebuilt["fieldwork_start"])
    rebuilt["fieldwork_end"] = pd.to_datetime(rebuilt["fieldwork_end"])
    pd.testing.assert_frame_equal(
        committed.reset_index(drop=True), rebuilt.reset_index(drop=True), check_dtype=False
    )


@pytest.mark.parametrize("text,expected", [
    ("11–18 Nov 2026", (date(2026, 11, 11), date(2026, 11, 18))),
    ("19 Oct–13 Nov 2026", (date(2026, 10, 19), date(2026, 11, 13))),
    ("28 Dec 2025–3 Jan 2026", (date(2025, 12, 28), date(2026, 1, 3))),
    ("28 Dec–3 Jan 2026", (date(2025, 12, 28), date(2026, 1, 3))),
    (" 12 Oct 2025", (date(2025, 10, 12), date(2025, 10, 12))),
])
def test_parse_fieldwork(text, expected):
    assert parse_fieldwork(text) == expected


def test_rolling_tracker_counts_only_new_interviews():
    polls = pd.DataFrame({
        "pollster": ["Pitagórica"] * 3 + ["Intercampus"] * 2,
        "fieldwork_start": pd.to_datetime(["2026-08-10", "2026-08-18", "2026-08-24", "2026-07-09", "2026-07-10"]),
        "fieldwork_end": pd.to_datetime(["2026-09-04", "2026-09-12", "2026-09-18", "2026-07-14", "2026-07-31"]),
        "sample_size": [804, 804, 804, 604, 803],
    })
    n = effective_sample_sizes(polls)
    assert n.iloc[0] == 804  # first release: every interview is new
    assert n.iloc[1] == round(804 * 8 / 26)  # 8 new days of a 26-day window
    assert n.iloc[2] == round(804 * 6 / 26)
    # Two separate Intercampus studies whose fieldwork overlaps keep their n
    assert list(n.iloc[3:]) == [604, 803]


def test_tracker_effective_sizes_in_the_data():
    polls = load_wikipedia_polls(CSV)
    tracker = polls[(polls["pollster"] == "Pitagórica") & (polls["published_sample_size"] == 804)]
    assert len(tracker) == 11
    later = tracker.iloc[1:]["sample_size"]
    assert later.between(150, 260).all()
    assert abs(later.mean() - 201) < 20


def test_combine_keeps_marktest_and_adds_new_polls():
    marktest = pd.DataFrame({
        "date": pd.to_datetime(["2025-06-14", "2025-09-05"]),
        "pollster": ["Intercampus", "* Aximage"],
        "sample_size": [616, 570],
        "PS": [0.218, 0.236], "CH": [0.217, 0.268], "IL": [0.084, 0.062], "BE": [0.018, 0.024],
        "CDU": [0.034, 0.031], "PAN": [0.039, 0.017], "L": [0.074, 0.065], "AD": [0.299, 0.259],
    })
    wikipedia = load_wikipedia_polls(CSV)
    combined = combine_poll_sources(marktest, wikipedia)
    assert list(combined.columns) == list(marktest.columns)
    # Intercampus ends 15 Jun on Wikipedia and 14 Jun on Marktest: the same poll
    assert len(combined[(combined["pollster"] == "Intercampus") & (combined["date"] <= "2025-06-30")]) == 1
    assert len(combined[(combined["date"] >= "2025-09-01") & (combined["date"] <= "2025-09-10")]) == 1
    assert len(combined) == len(marktest) + len(wikipedia) - 2


def test_drop_election_day_polls():
    polls = pd.DataFrame({
        "date": pd.to_datetime(["2025-05-15", "2025-05-18", "2025-05-18", "2025-06-14"]),
        "pollster": ["Pitagórica", "Intercampus", "CESOP-U.Católica", "Intercampus"],
        "sample_size": [810, 23365, 39103, 616],
    })
    kept = drop_election_day_polls(polls, ["2025-05-18", "2024-03-10"])
    assert list(kept["sample_size"]) == [810, 616]


@pytest.mark.parametrize("columns,expected", [
    # 2019: PSD and CDS ran apart everywhere
    (["PS", "PPD/PSD", "CDS-PP", "PCP-PEV", "CH"], ["PPD/PSD", "CDS-PP"]),
    # 2022: apart on the mainland, together in Madeira and (with PPM) the Azores
    (["PPD/PSD.CDS-PP.PPM", "PCP-PEV", "PPD/PSD", "CDS-PP", "PPD/PSD.CDS-PP", "PPM"],
     ["PPD/PSD.CDS-PP.PPM", "PPD/PSD", "CDS-PP", "PPD/PSD.CDS-PP"]),
    # 2015: PàF on the mainland, CDS-PP.PPM in the Azores
    (["PPD/PSD", "CDS-PP.PPM", "PCP-PEV", "PPD/PSD.CDS-PP", "CDS-PP"],
     ["PPD/PSD", "CDS-PP.PPM", "PPD/PSD.CDS-PP", "CDS-PP"]),
])
def test_ad_sums_every_list_it_ran_under(columns, expected):
    manager = CoalitionManager()
    assert manager.resolve_historical_columns_to_target("AD", columns) == expected
    row = pd.DataFrame([{c: 10.0 for c in columns}])
    mapped = manager.map_election_data_columns(row, ["AD", "PS", "CDU"], election_type="parliamentary")
    assert mapped["AD"].iloc[0] == 10.0 * len(expected)
    assert mapped["CDU"].iloc[0] == 10.0  # one list, unchanged
