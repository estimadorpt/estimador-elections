"""Undecided share of each poll, from the ERC deposits.

The shares the model reads are among decided respondents: each firm drops or redistributes its
"não sabe / não responde" before publishing. The law requires the raw figure in every ERC
deposit (Lei 10/2000, art. 6), and `data/erc_polls_*_geo.parquet` holds what our extraction
captured: party rows plus rows for the undecided, abstention and blank/null answers.

`undecided_table()` keeps the legislative, national deposits whose answers sum to about 100% and
that report an undecided share. `poll_undecided_shares()` matches the model's polls to them (same
firm, fieldwork end within a few days) and fills the rest with the firm's median, then the median
of all firms, so a poll's undecided share is never left unknown. Coverage is uneven: most polls
before 2020, few in 2024 and 2025 (an extraction gap, not polls without undecided).
"""

from __future__ import annotations

import glob
import os
import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "..", "data")

_UNDECIDED = r"indecis|ns/nr|não sabe|nao sabe|n\.s|dúvida|duvida|não respond|recusa|nenhum"
_ABSTAIN = r"absten|não vota|nao vota"
_BLANK = r"branco|nulo"

# firm keys shared by the Marktest/Wikipedia spellings and the ERC deposit spellings
_FIRMS = [
    ("cesop", ("cesop", "catolica", "universidade", "ucp")),
    ("ics", ("ics", "iscte", "gfk", "metris")),
    ("intercampus", ("intercampus",)),
    ("aximage", ("aximage",)),
    ("eurosondagem", ("eurosondagem",)),
    ("pitagorica", ("pitagorica",)),
    ("marktest", ("marktest",)),
    ("consulmark2", ("consulmark",)),
    ("duplimetrica", ("duplimetrica",)),
    ("multidados", ("multidados",)),
    ("domp", ("domp",)),
    ("euroexpansao", ("euroexpansao",)),
]


def firm_key(name: str) -> str:
    """A pollster name reduced to one key per firm ("CESOP-U.Católica" -> "cesop")."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    for key, needles in _FIRMS:
        if any(n in s for n in needles):
            return key
    return s.strip()


def undecided_table(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """One row per ERC deposit with a usable undecided share (0-1) for a national legislative poll."""
    files = sorted(glob.glob(os.path.join(data_dir, "erc_polls_*_geo.parquet")))
    if not files:
        return pd.DataFrame(columns=["deposit_number", "firm", "fieldwork_end", "sample_size", "undecided"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    leg = df[
        df["election_type"].str.contains("Legisl", na=False)
        & df["geographic_scope"].str.contains("Nacional", na=False)
    ].copy()
    party = leg["party"].astype(str)
    leg["cat"] = np.select(
        [
            party.str.contains(_UNDECIDED, case=False, regex=True),
            party.str.contains(_ABSTAIN, case=False, regex=True),
            party.str.contains(_BLANK, case=False, regex=True),
        ],
        ["undecided", "abstain", "blank_null"],
        default="party",
    )
    leg["fieldwork_end"] = pd.to_datetime(leg["fieldwork_end"], errors="coerce")
    g = (
        leg.groupby(["deposit_number", "pollster", "fieldwork_end", "sample_size", "cat"])["vote_intention_pct"]
        .sum()
        .unstack("cat")
        .reindex(columns=["party", "undecided", "abstain", "blank_null"], fill_value=0.0)
        .fillna(0.0)
        .reset_index()
    )
    total = g[["party", "undecided", "abstain", "blank_null"]].sum(axis=1)
    keep = (total > 95) & (total < 105) & (g["undecided"] > 0)
    out = g.loc[keep, ["deposit_number", "pollster", "fieldwork_end", "sample_size"]].copy()
    out["undecided"] = (g.loc[keep, "undecided"] / total[keep]).values
    out["firm"] = out["pollster"].map(firm_key)
    return out.drop(columns="pollster").dropna(subset=["fieldwork_end"]).reset_index(drop=True)


def poll_undecided_shares(
    polls: pd.DataFrame,
    table: Optional[pd.DataFrame] = None,
    max_days: int = 5,
    impute: bool = True,
) -> np.ndarray:
    """Undecided share (0-1) for each row of `polls` (columns `pollster`, `date`).

    A poll takes the deposit of the same firm whose fieldwork ended nearest its date, within
    `max_days`. With `impute`, unmatched polls get their firm's median over the matched polls, else
    the median over all matched polls (0.15 if nothing matched).
    """
    table = undecided_table() if table is None else table
    firms = polls["pollster"].map(firm_key).to_numpy()
    dates = pd.to_datetime(polls["date"]).to_numpy()
    u = np.full(len(polls), np.nan)
    by_firm = {f: t for f, t in table.groupby("firm")}
    for i, (f, d) in enumerate(zip(firms, dates)):
        t = by_firm.get(f)
        if t is None:
            continue
        gap = np.abs((t["fieldwork_end"].to_numpy() - d) / np.timedelta64(1, "D"))
        j = int(np.argmin(gap))
        if gap[j] <= max_days:
            u[i] = float(t["undecided"].iloc[j])
    if impute:
        known = ~np.isnan(u)
        overall = float(np.median(u[known])) if known.any() else 0.15
        firm_median = pd.Series(u).groupby(firms).median()
        for i in np.where(~known)[0]:
            m = firm_median.get(firms[i], np.nan)
            u[i] = overall if pd.isna(m) else float(m)
    return np.clip(u, 0.0, 0.6)
