import numpy as np
import pandas as pd

from src.evaluation.ecological_covariates import _polygon_area_km2, shares


def test_polygon_area_one_degree_square_at_equator():
    ring = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    assert abs(_polygon_area_km2([ring]) - 12364) < 50  # 111.2 km x 111.2 km


def test_polygon_area_subtracts_holes():
    outer = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]
    hole = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]]
    assert _polygon_area_km2([outer, hole]) < _polygon_area_km2([outer])


def test_shares_are_computed_from_summed_counts():
    cols = {c: [0, 0] for c in [
        "male20", "pop20", "age20_34", "age35_49", "age50_64", "age65p", "age75p", "edu_none", "edu_basic1",
        "edu_basic", "edu_secondary", "edu_tertiary", "pop25", "edu_tertiary25_44", "pop25_44", "employed",
        "unemployed", "pop15", "foreign", "pop_all", "rel_none", "rel_catholic", "rel_answered", "ind_agri",
        "ind_manuf", "ind_construction", "ind_public", "ind_tourism", "emp_ind_total", "occ_prof", "occ_manual",
        "emp_occ_total", "inc_pension", "inc_social", "inc_total", "hh_tenant", "hh_total", "area_km2"]}
    df = pd.DataFrame(cols, index=["a", "b"])
    df["pop25"] = [100, 300]
    df["edu_tertiary"] = [50, 30]
    df["pop_all"] = [1000, 1000]
    df["area_km2"] = [10, 10]
    total = shares(df.sum().to_frame().T)
    # 80 / 400, not the mean of 0.5 and 0.1
    assert np.isclose(total["edu_tertiary"].iloc[0], 0.2)
    assert np.isclose(total["log_density"].iloc[0], np.log(100))
