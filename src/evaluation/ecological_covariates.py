"""Parish covariates from INE Censos 2021, for the ecological (demographic) district checks.

Reads the INE parish tables that estimador-microsynthesis keeps in
`data/interim/constraints/ine/` (one row per parish x category, counts) and the parish
geometry the website publishes (`estimador-web/public/data/population-geography/parishes`),
and writes one row per parish (DICOFRE code, the 3,092 parishes of CAOP 2013-2024, the same
codes as the legislative results 2015-2025) with counts and shares.

Counts are kept next to the shares so that municipalities and districts can be built by
summing counts, never by averaging shares.

    python -m src.evaluation.ecological_covariates INE_DIR GEO_DIR OUT.parquet
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0088


def _polygon_area_km2(rings) -> float:
    """Area of a lon/lat polygon (outer ring minus holes) on a sphere, small-polygon formula."""
    total = 0.0
    for k, ring in enumerate(rings):
        xy = np.asarray(ring, float)
        lon, lat = np.radians(xy[:, 0]), np.radians(xy[:, 1])
        # spherical excess approximation (Chamberlain & Duquette 2007)
        a = np.sum((lon[1:] - lon[:-1]) * (2 + np.sin(lat[:-1]) + np.sin(lat[1:])))
        area = abs(a) * EARTH_R_KM**2 / 2
        total += area if k == 0 else -area
    return total


def parish_areas(geo_dir: str) -> pd.Series:
    out = {}
    for f in glob.glob(os.path.join(geo_dir, "*.json")):
        fc = json.load(open(f))
        for feat in fc["features"]:
            g = feat["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            out[feat["properties"]["code"]] = sum(_polygon_area_km2(p) for p in polys)
    return pd.Series(out, name="area_km2")


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot_table(index="freguesia", columns=col, values="value", aggfunc="sum", fill_value=0)


AGE_ORDER = [
    "15 - 19 anos", "20 - 24 anos", "25 - 29 anos", "30 - 34 anos", "35 - 39 anos", "40 - 44 anos",
    "45 - 49 anos", "50 - 54 anos", "55 - 59 anos", "60 - 64 anos", "65 - 69 anos", "70 - 74 anos",
    "75 ou mais anos",
]


def build(ine_dir: str, geo_dir: str) -> pd.DataFrame:
    rd = lambda n: pd.read_parquet(os.path.join(ine_dir, n))  # noqa: E731

    ed = rd("person_sex_age15_educ_2021.parquet")
    ed["age_i"] = ed["age_15grp"].map({a: i for i, a in enumerate(AGE_ORDER)})
    adult = ed[ed.age_i >= 1]  # 20+ (stand-in for the 18+ electorate)
    out = pd.DataFrame(index=sorted(ed.freguesia.unique()))
    out["pop20"] = adult.groupby("freguesia").value.sum()
    out["male20"] = adult[adult.sex == "H"].groupby("freguesia").value.sum()
    ages = adult.groupby(["freguesia", "age_i"]).value.sum().unstack(fill_value=0)
    out["age20_34"] = ages[[1, 2, 3]].sum(axis=1)
    out["age35_49"] = ages[[4, 5, 6]].sum(axis=1)
    out["age50_64"] = ages[[7, 8, 9]].sum(axis=1)
    out["age65p"] = ages[[10, 11, 12]].sum(axis=1)
    out["age75p"] = ages[[12]].sum(axis=1)
    a25 = ed[ed.age_i >= 2]
    out["pop25"] = a25.groupby("freguesia").value.sum()
    ecode = a25.groupby(["freguesia", "education_level_code"]).value.sum().unstack(fill_value=0)
    ecode.columns = ecode.columns.astype(str)
    out["edu_none"] = ecode["1"]
    out["edu_basic1"] = ecode["1"] + ecode["21"]  # none or 1st cycle (4 years)
    out["edu_basic"] = ecode[["1", "21", "22", "23"]].sum(axis=1)
    out["edu_secondary"] = ecode[["3", "4"]].sum(axis=1)
    out["edu_tertiary"] = ecode[["51", "52", "53", "54", "55"]].sum(axis=1)
    # young adults' tertiary attainment (25-44) separates "old and unschooled" from class
    y = ed[(ed.age_i >= 2) & (ed.age_i <= 6)]
    out["pop25_44"] = y.groupby("freguesia").value.sum()
    out["edu_tertiary25_44"] = y[y.education_level_code.astype(str).str.startswith("5")].groupby("freguesia").value.sum()

    lab = rd("person_sex_age15_labour3_educ5_2021.parquet")
    lab = _pivot(lab, "employment_status_coarse3")
    lab.columns = lab.columns.astype(str)
    out["pop15"] = lab.sum(axis=1)
    out["employed"] = lab["11"]
    out["unemployed"] = lab["12"]
    out["inactive"] = lab["2"]

    nat = _pivot(rd("person_sex_nationality_group_2021.parquet"), "nationality_group")
    out["pop_all"] = nat.sum(axis=1)
    out["foreign"] = nat["Foreign"]

    rel = _pivot(rd("person_religion15_group_2021.parquet"), "religion_group")
    out["rel_answered"] = rel[["Catholic", "No religion", "Other"]].sum(axis=1)
    out["rel_catholic"] = rel["Catholic"]
    out["rel_none"] = rel["No religion"]

    ind = _pivot(rd("person_employed_sex_industry_2021.parquet"), "industry_section_code")
    out["emp_ind_total"] = ind.sum(axis=1)
    out["ind_agri"] = ind["A"]
    out["ind_manuf"] = ind[["B", "C", "D", "E"]].sum(axis=1)
    out["ind_construction"] = ind["F"]
    out["ind_public"] = ind[["O", "P", "Q"]].sum(axis=1)
    out["ind_tourism"] = ind["I"]

    occ = _pivot(rd("person_employed_sex_occupation_2021.parquet"), "occupation_major")
    occ.columns = occ.columns.astype(str)
    out["emp_occ_total"] = occ.sum(axis=1)
    out["occ_prof"] = occ[["1", "2"]].sum(axis=1)  # managers, professionals
    out["occ_manual"] = occ[["7", "8", "9"]].sum(axis=1)  # craft, operators, elementary

    inc = _pivot(rd("person_sex_age15_income_source_2021.parquet"), "income_source_code")
    inc.columns = inc.columns.astype(str)
    out["inc_total"] = inc.sum(axis=1)
    out["inc_pension"] = inc["2"]
    out["inc_social"] = inc[["3", "4", "5"]].sum(axis=1)  # unemployment benefit, RSI, other social support

    ten = _pivot(rd("household_hh_tenure_2021.parquet"), "hh_tenure_code")
    ten.columns = ten.columns.astype(str)
    out["hh_total"] = ten.sum(axis=1)
    out["hh_tenant"] = ten["3"]

    out["area_km2"] = parish_areas(geo_dir)
    out.index.name = "freguesia"
    return out.fillna(0)


# share = numerator / denominator, computed after summing counts to any level
SHARES = {
    "male": ("male20", "pop20"),
    "age20_34": ("age20_34", "pop20"),
    "age35_49": ("age35_49", "pop20"),
    "age50_64": ("age50_64", "pop20"),
    "age65p": ("age65p", "pop20"),
    "age75p": ("age75p", "pop20"),
    "edu_none": ("edu_none", "pop25"),
    "edu_basic1": ("edu_basic1", "pop25"),
    "edu_basic": ("edu_basic", "pop25"),
    "edu_secondary": ("edu_secondary", "pop25"),
    "edu_tertiary": ("edu_tertiary", "pop25"),
    "edu_tertiary25_44": ("edu_tertiary25_44", "pop25_44"),
    "employed": ("employed", "pop15"),
    "unemployment": ("unemployed", None),  # / (employed + unemployed)
    "foreign": ("foreign", "pop_all"),
    "rel_none": ("rel_none", "rel_answered"),
    "rel_catholic": ("rel_catholic", "rel_answered"),
    "ind_agri": ("ind_agri", "emp_ind_total"),
    "ind_manuf": ("ind_manuf", "emp_ind_total"),
    "ind_construction": ("ind_construction", "emp_ind_total"),
    "ind_public": ("ind_public", "emp_ind_total"),
    "ind_tourism": ("ind_tourism", "emp_ind_total"),
    "occ_prof": ("occ_prof", "emp_occ_total"),
    "occ_manual": ("occ_manual", "emp_occ_total"),
    "inc_pension": ("inc_pension", "inc_total"),
    "inc_social": ("inc_social", "inc_total"),
    "hh_tenant": ("hh_tenant", "hh_total"),
}


def shares(counts: pd.DataFrame) -> pd.DataFrame:
    """Shares (and log density) from summed counts, at whatever level `counts` is."""
    out = pd.DataFrame(index=counts.index)
    for k, (num, den) in SHARES.items():
        if den is None:
            out[k] = counts[num] / (counts["employed"] + counts[num]).clip(lower=1)
        else:
            out[k] = counts[num] / counts[den].clip(lower=1)
    out["log_density"] = np.log((counts["pop_all"] / counts["area_km2"].clip(lower=1e-3)).clip(lower=0.1))
    return out


if __name__ == "__main__":
    ine_dir, geo_dir, dest = sys.argv[1:4]
    df = build(ine_dir, geo_dir)
    df.to_parquet(dest)
    print(df.shape, "parishes; total population", int(df.pop_all.sum()), "; area km2", round(df.area_km2.sum()))
