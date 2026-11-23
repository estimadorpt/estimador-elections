"""Polls from Wikipedia's "Opinion polling for the next Portuguese legislative election".

Marktest's poll list (``data/marktest_polls.csv``) is the model's main source,
but it needs a manual login and was last collected in September 2025. Wikipedia's
table lists every national poll since the 18 May 2025 election, within days of
publication. This module turns a saved copy of that table's wikitext into a CSV
(``data/wikipedia_legislative_polls.csv``) and loads that CSV in the model's poll
format.

The saved wikitext is kept verbatim under ``data/wikipedia/`` so the CSV can be
rebuilt and checked:

    python -m src.data.wikipedia_polls data/wikipedia/<snapshot>.wikitext \
        data/wikipedia_legislative_polls.csv

Shares are the ones Wikipedia prints. Its editors rescale polls that report
undecided voters so that the parties sum to 100 without them, the same
decided-voter basis as the Marktest list.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Column order of the main table on the page (as of November 2026).
TABLE_COLUMNS = [
    "pollster_cell", "fieldwork", "sample_size", "turnout",
    "AD", "PS", "CH", "IL", "L", "CDU", "BE", "PAN", "JPP", "Others", "Lead",
]
PARTY_COLUMNS = ["AD", "PS", "CH", "IL", "L", "CDU", "BE", "PAN", "JPP", "Others"]

# Wikipedia's pollster names -> the names the Marktest list uses, so that each
# pollster keeps one house effect across both sources.
POLLSTER_NAMES = {
    "Intercampus": "Intercampus",
    "Aximage": "Aximage",
    "Pitagórica": "Pitagórica",
    "ICS/ISCTE": "ICS/ISCTE/GFK Metris",
    "CESOP–UCP": "CESOP-U.Católica",
    "CESOP-UCP": "CESOP-U.Católica",
    "CESOP–Católica": "CESOP-U.Católica",
    "Consulmark2": "Consulmark2",
    "Duplimétrica": "Duplimétrica/IPESPE",
}

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

CSV_COLUMNS = [
    "pollster", "pollster_wikipedia", "fieldwork_start", "fieldwork_end",
    "sample_size", "turnout", *PARTY_COLUMNS, "source_url", "footnote",
]


def _strip_cell(cell: str) -> str:
    """Return a cell's displayed text: drop attributes, bold marks and <br/> suffixes."""
    text = cell.strip()
    # A cell with attributes reads `style="..."| value` or `{{Party shading/PS}} | value`.
    if "|" in text and not text.startswith("["):
        # Split on the last bar that is not inside {{...}} or [[...]].
        depth = 0
        split_at = -1
        for i, ch in enumerate(text):
            if text.startswith("{{", i) or text.startswith("[[", i):
                depth += 1
            elif text.startswith("}}", i) or text.startswith("]]", i):
                depth -= 1
            elif ch == "|" and depth == 0:
                split_at = i
        if split_at >= 0:
            text = text[split_at + 1:]
    text = re.split(r"<br\s*/?>", text)[0]
    return text.replace("'''", "").strip()


def _parse_number(text: str) -> float:
    text = _strip_cell(text)
    if text in {"", "?", "{{N/A}}", "–", "-"}:
        return np.nan
    if text.startswith("<"):
        # "<1" is printed for a party below one point: count it as half a point.
        return float(text[1:].replace(",", "")) / 2
    return float(text.replace(",", ""))


def parse_fieldwork(text: str) -> tuple[date, date]:
    """Parse '11–18 Nov 2026', '19 Oct–13 Nov 2026', '28 Dec 2025–3 Jan 2026' or '12 Oct 2025'."""
    text = _strip_cell(text).replace("–", "-").replace("—", "-")
    parts = [p.strip() for p in text.split("-")]
    end_tokens = parts[-1].split()
    if len(end_tokens) != 3:
        raise ValueError(f"Unrecognised fieldwork date: {text!r}")
    end = date(int(end_tokens[2]), MONTHS[end_tokens[1][:3].lower()], int(end_tokens[0]))
    if len(parts) == 1:
        return end, end
    start_tokens = parts[0].split()
    day = int(start_tokens[0])
    month = MONTHS[start_tokens[1][:3].lower()] if len(start_tokens) >= 2 else end.month
    year = int(start_tokens[2]) if len(start_tokens) == 3 else end.year
    start = date(year, month, day)
    if start > end:  # e.g. "28 Dec–3 Jan 2026" written without the first year
        start = date(year - 1, month, day)
    return start, end


def _main_table(wikitext: str) -> str:
    """Return the first wikitable under the ===Polling=== heading."""
    start = wikitext.find("{|")
    if start < 0:
        raise ValueError("No table found in the wikitext")
    end = wikitext.find("\n|}", start)
    return wikitext[start:end if end >= 0 else None]


def parse_wikitext_poll_table(wikitext: str) -> pd.DataFrame:
    """Parse the main voting-intention table into one row per poll.

    Election results (rows whose first cell is a bold link) and separator rows
    are skipped. Shares stay in percent, as printed.
    """
    table = _main_table(wikitext)
    rows = re.split(r"\n\|-[^\n]*", table)
    records: List[Dict] = []
    for row in rows:
        lines = [ln for ln in row.split("\n") if ln.startswith("|") and not ln.startswith("|}")]
        if not lines or lines[0].startswith("|+"):
            continue
        cells = [ln[1:] for ln in lines]
        first = cells[0].strip()
        if not first.startswith("["):
            continue  # header, separator or an election-result row
        if len(cells) != len(TABLE_COLUMNS):
            raise ValueError(f"Expected {len(TABLE_COLUMNS)} cells, got {len(cells)}: {first[:80]}")
        link = re.match(r"\[(\S+)\s+([^\]]+)\]", first)
        if not link:
            raise ValueError(f"Unrecognised pollster cell: {first[:80]}")
        footnote = re.search(r"\{\{efn\|name=([^}]+)\}\}", first)
        start, end = parse_fieldwork(cells[1])
        record = {
            "pollster_wikipedia": link.group(2).strip(),
            "source_url": link.group(1),
            "footnote": footnote.group(1) if footnote else "",
            "fieldwork_start": start,
            "fieldwork_end": end,
            "sample_size": _parse_number(cells[2]),
            "turnout": _parse_number(cells[3]),
        }
        for column, cell in zip(TABLE_COLUMNS[4:14], cells[4:14]):
            record[column] = _parse_number(cell)
        records.append(record)

    df = pd.DataFrame(records)
    unknown = sorted(set(df["pollster_wikipedia"]) - set(POLLSTER_NAMES))
    if unknown:
        raise ValueError(f"Pollsters without a name mapping: {unknown}. Add them to POLLSTER_NAMES.")
    df["pollster"] = df["pollster_wikipedia"].map(POLLSTER_NAMES)
    df = df.sort_values(["fieldwork_end", "pollster"]).reset_index(drop=True)
    return df[CSV_COLUMNS]


def effective_sample_sizes(df: pd.DataFrame) -> pd.Series:
    """Count each interview once when a pollster's fieldwork windows overlap.

    Pitagórica's weekly tracker publishes a rolling four-week sample (n = 804),
    so three quarters of each release's interviews were already in the previous
    one. A release is treated as rolling when it has the same published n as the
    pollster's previous poll and at least half of its window overlaps that poll's
    window. It then counts only the share of its window that is new: n * days
    since the previous fieldwork end / days in its own window (about 200 of 804
    for the weekly tracker). Every other poll keeps its n, including separate
    studies whose fieldwork happens to overlap.
    """
    effective = df["sample_size"].astype(float).copy()
    ordered = df.sort_values(["fieldwork_end", "fieldwork_start"])
    for _, group in ordered.groupby("pollster"):
        previous = None
        for idx, row in group.iterrows():
            start = pd.Timestamp(row["fieldwork_start"])
            end = pd.Timestamp(row["fieldwork_end"])
            if previous is not None:
                prev_end = pd.Timestamp(previous["fieldwork_end"])
                window_days = (end - start).days + 1
                overlap_days = (prev_end - start).days + 1
                same_n = row["sample_size"] == previous["sample_size"]
                if same_n and overlap_days * 2 >= window_days:
                    new_days = max((end - prev_end).days, 0)
                    effective.loc[idx] = row["sample_size"] * min(1.0, new_days / window_days)
            previous = row
    return effective.round().astype(int)


def load_wikipedia_polls(path: Optional[str] = None) -> pd.DataFrame:
    """Load the parsed CSV in the model's format: date, pollster, sample_size, shares in 0-1.

    ``date`` is the fieldwork end, as in the Marktest list. ``sample_size`` is the
    effective size from :func:`effective_sample_sizes`; the published one stays in
    ``published_sample_size``.
    """
    from src.config import DATA_DIR

    path = path or os.path.join(DATA_DIR, "wikipedia_legislative_polls.csv")
    df = pd.read_csv(path, parse_dates=["fieldwork_start", "fieldwork_end"])
    df["published_sample_size"] = df["sample_size"]
    df["sample_size"] = effective_sample_sizes(df)
    out = pd.DataFrame({
        "date": df["fieldwork_end"],
        "pollster": df["pollster"],
        "sample_size": df["sample_size"],
        "published_sample_size": df["published_sample_size"],
        "fieldwork_start": df["fieldwork_start"],
    })
    for party in ["PS", "CH", "IL", "BE", "CDU", "PAN", "L", "AD"]:
        out[party] = df[party].fillna(0) / 100
    return out.sort_values("date").reset_index(drop=True)


def main(argv: List[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    with open(argv[1], encoding="utf-8") as fh:
        polls = parse_wikitext_poll_table(fh.read())
    polls.to_csv(argv[2], index=False)
    print(f"Wrote {len(polls)} polls ({polls['fieldwork_end'].min()} to {polls['fieldwork_end'].max()}) to {argv[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
