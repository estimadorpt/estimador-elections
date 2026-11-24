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


def test_centred_house_effects_sum_to_zero_over_pollsters_too(dataset):
    import pymc as pm

    # (the per-party scale breaks the zero sum over parties in both forms; softmax ignores it)
    he = pm.draw(_build(dataset, center_house_effects=True)["house_effects"], random_seed=1)
    assert np.allclose(he.sum(axis=0), 0, atol=1e-8)
    he0 = pm.draw(_build(dataset)["house_effects"], random_seed=1)
    assert not np.allclose(he0.sum(axis=0), 0, atol=1e-3)  # the original form: firms not centred
