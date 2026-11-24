"""Seats per domestic constituency (círculo) for each legislative election.

The law (Lei Eleitoral da Assembleia da República, art. 13) shares the 226 domestic seats among
the 20 domestic círculos in proportion to their registered voters, by D'Hondt; Europe and
Outside Europe elect 2 each. The map therefore moves between elections: Setúbal gained a seat
and Viana do Castelo lost one in 2024, Lisbon and Porto gained one each in 2019 (from Viseu and
Guarda), and so on.

`seat_map(election_date)` recomputes an election's map from the registered voters in its own
results file (`legislativas_<year>.parquet`, column `subscribed_voters`). The CNE uses the
registry at an earlier reference date, so a near tie could in principle fall the other way.
The recomputed 2024 map has Setúbal's new seat, and the 2019 and 2022 maps equal the file this
repository used until November 2026. The closest call is 2009 (Coimbra's 10th seat over Vila
Real's 6th, by one voter in 40,000); the next closest is 2022 (0.25%).
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

from src.config import DATA_DIR

DOMESTIC_SEATS = 226

_DISTRICT_BY_CODE = {
    "01": "Aveiro", "02": "Beja", "03": "Braga", "04": "Bragança", "05": "Castelo Branco",
    "06": "Coimbra", "07": "Évora", "08": "Faro", "09": "Guarda", "10": "Leiria", "11": "Lisboa",
    "12": "Portalegre", "13": "Porto", "14": "Santarém", "15": "Setúbal", "16": "Viana do Castelo",
    "17": "Vila Real", "18": "Viseu",
}


def circulo_of(territory_code) -> str | None:
    """Domestic círculo of a parish code (DICOFRE, with or without the 'LOCAL-' prefix)."""
    code = str(territory_code).replace("LOCAL-", "")
    if code[:2] in _DISTRICT_BY_CODE:
        return _DISTRICT_BY_CODE[code[:2]]
    if code[:2] in ("31", "32"):
        return "Madeira"
    if code[:1] == "4":
        return "Açores"
    return None


def dhondt_apportion(weights: dict, seats: int) -> pd.Series:
    """Share `seats` among the keys of `weights` by D'Hondt (highest averages)."""
    quotients = sorted(((w / d, k) for k, w in weights.items() for d in range(1, seats + 1)), reverse=True)
    return pd.Series([k for _, k in quotients[:seats]]).value_counts().reindex(list(weights), fill_value=0)


@lru_cache(maxsize=None)
def seat_map(election_date: str) -> pd.Series:
    """Seats per domestic círculo for the legislative election held on `election_date`
    (YYYY-MM-DD), from the registered voters in that election's results file."""
    year = pd.Timestamp(election_date).year
    df = pd.read_parquet(os.path.join(DATA_DIR, f"legislativas_{year}.parquet"))
    code_col = "territory_code" if "territory_code" in df.columns else "territoryCode"
    circ = df[code_col].map(circulo_of)
    if circ.isna().any():
        raise ValueError(f"{year}: {int(circ.isna().sum())} rows with no domestic círculo")
    registered = df.groupby(circ)["subscribed_voters"].sum()
    return dhondt_apportion(registered.to_dict(), DOMESTIC_SEATS).sort_index().rename("Deputados")
