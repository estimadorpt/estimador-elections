import pytest
import pandas as pd
from src.data.dataset import ElectionDataset

def test_find_closest_election_date():
    """Test if polls after the last historical election are assigned the target date."""
    
    # Create a dataset with a future election date
    future_election = "2025-10-05"  # A date not in historical_election_dates
    dataset = ElectionDataset(
        election_date=future_election,
        baseline_timescales=[365],
        election_timescales=[60],
    )
    
    # Create test poll rows with different dates
    last_historical = pd.to_datetime(dataset.historical_election_dates[0])  # 2024-03-10
    
    # Test case 1: Poll before last historical election
    test_row = pd.Series({"date": last_historical - pd.Timedelta(days=30)})
    result = dataset.find_closest_election_date(test_row)
    assert result == last_historical, f"Poll before last election should point to last election {last_historical}"
    
    # Test case 2: Poll after last historical election
    test_row = pd.Series({"date": last_historical + pd.Timedelta(days=1)})
    result = dataset.find_closest_election_date(test_row)
    assert result == pd.to_datetime(future_election), f"Poll after last election should point to future election {future_election}"
    
    # Test case 3: Poll far in the past
    test_row = pd.Series({"date": pd.to_datetime("2010-01-01")})
    result = dataset.find_closest_election_date(test_row)
    assert result == pd.to_datetime("2011-06-05"), "Poll in distant past should point to appropriate historical election"
    
    # Also test using the actual polls loaded by the dataset
    if len(dataset.polls) > 0:
        # Check if any polls exist with dates after the last historical election
        future_polls = dataset.polls[dataset.polls["date"] > last_historical]
        if len(future_polls) > 0:
            # All future polls should be assigned to the target election date
            assert pd.to_datetime(future_election) in future_polls["election_date"].unique(), \
                "Future polls are not correctly assigned to the target election date" 

def test_2025_is_a_past_election_and_new_polls_join_the_next_cycle():
    """2025 is history; polls since then come from Wikipedia and point at the next election."""
    target = "2029-10-07"
    dataset = ElectionDataset(
        election_date=target,
        baseline_timescales=[365],
        election_timescales=[60],
    )
    assert dataset.historical_election_dates[0] == "2025-05-18"
    assert dataset.government_parties["2025-05-18"] == ["AD"]

    polls = dataset.polls
    # No exit polls: nothing dated on an election day
    election_days = pd.to_datetime(dataset.historical_election_dates)
    assert not polls["date"].isin(election_days).any()
    assert polls["sample_size"].max() < 5000

    after = polls[polls["date"] > pd.to_datetime("2025-05-18")]
    assert (after["election_date"] == pd.to_datetime(target)).all()
    # Marktest stops in September 2025; Wikipedia carries the rest
    assert after["date"].max() >= pd.to_datetime("2026-11-18")
    assert len(after) == 50

    # The 2025 result counts every AD list (it was 1% when only the Azores list was read)
    results = dataset.results_national.set_index(pd.to_datetime(dataset.results_national["election_date"]))
    parties = dataset.political_families
    shares = results[parties].div(results[parties].sum(axis=1), axis=0)
    assert 0.33 < shares.loc["2025-05-18", "AD"] < 0.36
    assert 0.28 < shares.loc["2022-01-30", "AD"] < 0.33
    assert 0.30 < shares.loc["2019-10-06", "AD"] < 0.36
