from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.models.dynamic_gp_election_model import DynamicGPElectionModel


def _stub(polls, results, calendar):
    return SimpleNamespace(
        polls_train=polls,
        dataset=SimpleNamespace(results_national=results),
        coords={"calendar_time": pd.DatetimeIndex(calendar).strftime("%Y-%m-%d")},
        political_families=["PS", "CH", "IL"],
    )


def test_existence_mask_starts_at_first_appearance():
    polls = pd.DataFrame(
        {"date": pd.to_datetime(["2019-01-01", "2019-06-01"]), "PS": [300, 300], "CH": [0, 20], "IL": [0, 0]}
    )
    results = pd.DataFrame({"election_date": pd.to_datetime(["2015-10-04"]), "PS": [1_700_000], "CH": [0], "IL": [0]})
    calendar = ["2015-10-04", "2019-01-01", "2019-05-15", "2019-06-01", "2019-10-06"]
    mask = DynamicGPElectionModel._existence_log_mask(_stub(polls, results, calendar))
    assert (mask[:, 0] == 0).all()  # PS always exists
    # Chega first seen 2019-06-01, so it exists from 30 days before (2019-05-02)
    assert mask[:, 1].tolist() == [-30.0, -30.0, 0.0, 0.0, 0.0]
    assert (mask[:, 2] == -30).all()  # IL never seen: no votes in the forecast either


def test_existence_mask_reads_results_too():
    polls = pd.DataFrame({"date": pd.to_datetime(["2016-01-01"]), "PS": [300], "CH": [0], "IL": [0]})
    results = pd.DataFrame(
        {"election_date": pd.to_datetime(["2015-10-04"]), "PS": [1_700_000], "CH": [0], "IL": [40_000]}
    )
    mask = DynamicGPElectionModel._existence_log_mask(_stub(polls, results, ["2015-08-01", "2015-09-10"]))
    # IL (and PS) exist from 30 days before their first result, 2015-09-04
    assert mask[:, 2].tolist() == [-30.0, 0.0]
    assert mask[:, 0].tolist() == [-30.0, 0.0]
    assert (mask[:, 1] == -30).all()
