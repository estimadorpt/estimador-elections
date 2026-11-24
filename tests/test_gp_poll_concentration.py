from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytensor

from src.models.dynamic_gp_election_model import DynamicGPElectionModel


def _concentration(pollster_concentration, undecided_variance, u):
    polls = pd.DataFrame({"pollster": ["A", "A", "B"], "date": pd.to_datetime(["2024-01-01"] * 3)})
    stub = SimpleNamespace(
        pollster_concentration=pollster_concentration,
        undecided_variance=undecided_variance,
        polls_train=polls,
        _undecided_shares=lambda p: np.asarray(u, float),
    )
    with pm.Model(coords={"pollsters": ["A", "B"], "observations": [0, 1, 2]}) as model:
        dc = {
            "pollster_idx": pm.Data("pollster_idx", np.array([0, 0, 1]), dims="observations"),
            "observed_N_polls": pm.Data("observed_N_polls", np.array([1000, 1000, 1000]), dims="observations"),
        }
        base = pm.Gamma("concentration_polls", alpha=2, beta=0.01)
        conc = DynamicGPElectionModel._poll_concentration(stub, base, dc)
        point = model.initial_point()
        point["concentration_polls_log__"] = np.log(200.0)
        out = model.replace_rvs_by_values([conc])[0]
        fn = pytensor.function(model.value_vars, out, on_unused_input="ignore")
        value = fn(**{v.name: point[v.name] for v in model.value_vars})
    return np.asarray(value), model


def test_off_returns_the_shared_scalar():
    value, _ = _concentration(False, False, [0, 0, 0])
    assert value.shape == () and np.isclose(value, 200.0)


def test_undecided_lowers_concentration_only_where_there_are_undecided():
    value, model = _concentration(False, True, [0.0, 0.2, 0.4])
    value = value[:, 0]
    assert np.isclose(value[0], 200.0)  # u = 0: the model as before
    assert value[2] < value[1] < 200.0
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_pollster_concentration_is_per_firm():
    value, model = _concentration(True, False, [0, 0, 0])
    assert value.shape == (3, 1)
    assert np.isclose(value[0, 0], value[1, 0])  # same firm
    assert "pollster_concentration" in model.named_vars
