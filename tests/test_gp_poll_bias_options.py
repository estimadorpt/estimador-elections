import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import make_dataset
from src.models.dynamic_gp_election_model import DynamicGPElectionModel


@pytest.fixture(scope="module")
def dataset():
    with contextlib.redirect_stdout(io.StringIO()):
        return make_dataset("2024-03-10", pd.Timestamp("2024-01-14"))


def _build(ds, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        m = DynamicGPElectionModel(ds, **kwargs)
        model = m.build_model()
    return model


def test_no_poll_bias_has_no_bias_variables(dataset):
    model = _build(dataset, use_poll_bias=False)
    names = set(model.named_vars)
    assert "poll_bias_raw" not in names and "sigma_poll_bias" not in names
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_poll_bias_per_cycle_with_and_without_carried_mean(dataset):
    model = _build(dataset, poll_bias_per_cycle=True)
    assert {"poll_bias_raw", "poll_bias_cycle", "sigma_poll_bias_cycle"} <= set(model.named_vars)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    model0 = _build(dataset, poll_bias_per_cycle=True, poll_bias_mean_sd=0.0)
    assert "poll_bias_raw" not in model0.named_vars and "poll_bias_cycle" in model0.named_vars
