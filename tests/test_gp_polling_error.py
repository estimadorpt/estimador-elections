"""DynamicGPElectionModel.apply_polling_error: the same shift per draw for national and districts."""

from types import SimpleNamespace

import numpy as np
import xarray as xr

from src.models.dynamic_gp_election_model import DynamicGPElectionModel

PARTIES = ["PS", "CH", "AD", "PAN"]


def _stub(sd):
    return SimpleNamespace(polling_error_sd=sd, polling_error_seed=1, _polling_error_eps={})


def _national(n_chain=2, n_draw=500):
    base = np.array([0.3, 0.25, 0.45, 0.0])
    return xr.DataArray(
        np.broadcast_to(base, (n_chain, n_draw, 4)).copy(),
        dims=("chain", "draw", "parties_complete"),
        coords={"parties_complete": PARTIES},
        name="latent_popularity_national",
    )


def test_off_by_default():
    nat = _national()
    assert DynamicGPElectionModel.apply_polling_error(_stub(0.0), nat) is nat


def test_national_and_district_draws_move_together():
    stub = _stub(0.2)
    nat = _national()
    dist = nat.expand_dims(districts=["Lisboa", "Porto"]).transpose("chain", "draw", "districts", "parties_complete")
    n2 = DynamicGPElectionModel.apply_polling_error(stub, nat)
    d2 = DynamicGPElectionModel.apply_polling_error(stub, dist)
    np.testing.assert_allclose(n2.sum("parties_complete"), 1.0)
    np.testing.assert_allclose(d2.sel(districts="Porto").values, n2.values, atol=1e-12)
    assert float(n2.sel(parties_complete="PAN").max()) < 1e-6  # absent party stays absent
    assert float(n2.sel(parties_complete="PS").std()) > 0.02
    assert d2.dims == dist.dims
    assert n2.name == "latent_popularity_national"
