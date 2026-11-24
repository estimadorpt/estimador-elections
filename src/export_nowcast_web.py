"""Export a random-walk nowcast run to the files estimador-web's live page reads.

    pixi run export-web outputs/ACCEPTED_legislative_nowcast_rw_v1_nobias_20261124 --web ~/code/estimador-web-nowcast-publish

The run folder is what `src/forecast_rw.py` writes (run.json, national_summary.csv,
national_draws.npy, seat_forecast_simulations.json, contested_summary.json, trace.zarr), plus
the frozen poll file the run read (input_polls.csv, copied in by the daily job). The export
refuses a run that would put a wrong page online: a failed convergence gate, a bias form other
than the published one, a target other than the last poll, missing files, seat draws that do not
add up to 230, or shares that do not add up to 1.

Written under the web checkout:

    public/data/elections/legislative-nowcast/nowcast.json          what the page prints
    public/data/elections/legislative-nowcast/district_forecast.json  mean district shares
    public/data/elections/legislative-nowcast/contested_summary.json  per-district seat changes
    data/build-only/elections/legislative-nowcast/seat_forecast_simulations.json  8 000 seat draws

Shares are fractions of the vote for the eight modelled parties (AD, PS, CH, IL, L, CDU, BE,
PAN), renormalised without the others, as the model fits them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

PUBLIC_DIR = os.path.join("public", "data", "elections", "legislative-nowcast")
BUILD_ONLY_DIR = os.path.join("data", "build-only", "elections", "legislative-nowcast")
SEATS = 230
SCHEMA = 1
PUBLISHED_BIAS_FORM = "none"
REQUIRED = (
    "run.json",
    "national_summary.csv",
    "national_draws.npy",
    "seat_forecast_simulations.json",
    "contested_summary.json",
    "input_polls.csv",
    "trace.zarr",
)
# Poll columns in the feed (estimador-data data/polls/feed/legislative_polls.csv) the page lists.
POLL_PARTY_COLUMNS = ("AD", "PS", "CH", "IL", "L", "CDU", "BE", "PAN", "JPP", "Others")


class ExportRefused(RuntimeError):
    """The run must not be published; the message says why."""


def _git_head(path: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_run(run_dir: str) -> dict:
    """run.json, after the checks that make a run publishable."""
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(run_dir, f))]
    if missing:
        raise ExportRefused(f"missing in {run_dir}: {', '.join(missing)}")
    with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
        run = json.load(f)
    diag = run.get("diagnostics", {})
    if diag.get("passes_gate") is not True:
        raise ExportRefused(
            f"convergence gate not passed (max r-hat {diag.get('max_rhat')}, min ESS bulk "
            f"{diag.get('min_ess_bulk')}, tail {diag.get('min_ess_tail')}, divergences {diag.get('divergences')})"
        )
    if diag.get("divergences", 1) != 0:
        raise ExportRefused(f"{diag.get('divergences')} divergences")
    if run.get("bias_form") != PUBLISHED_BIAS_FORM:
        raise ExportRefused(f"bias form {run.get('bias_form')!r} is a scenario, not the published nowcast")
    if run.get("target") != run.get("last_poll") or run.get("horizon_days") != 0:
        raise ExportRefused("not a nowcast: the target is not the day of the last poll")
    return run


def seat_draws(run_dir: str, parties: list[str]) -> list[dict]:
    with open(os.path.join(run_dir, "seat_forecast_simulations.json"), encoding="utf-8") as f:
        draws = json.load(f)
    if not draws:
        raise ExportRefused("no seat draws")
    for i, row in enumerate(draws):
        if "original_sample_id" not in row:
            raise ExportRefused(f"seat draw {i} has no original_sample_id")
        total = sum(int(row.get(p, 0)) for p in parties)
        if total != SEATS:
            raise ExportRefused(f"seat draw {i} adds up to {total} seats, not {SEATS}")
    return draws


def national_rows(run_dir: str, parties: list[str]) -> list[dict]:
    summary = pd.read_csv(os.path.join(run_dir, "national_summary.csv")).set_index("party")
    if sorted(summary.index) != sorted(parties):
        raise ExportRefused(f"national_summary.csv parties {sorted(summary.index)} != {sorted(parties)}")
    total = summary["mean"].sum()
    if abs(total - 100) > 0.5:
        raise ExportRefused(f"mean shares add up to {total:.2f}%, not 100%")
    rows = []
    for party, r in summary.sort_values("median", ascending=False).iterrows():
        rows.append({"party": str(party), **{k: round(float(r[k]) / 100, 4) for k in ("median", "mean", "lo90", "hi90", "lo50", "hi50")}})
    return rows


def trend_series(run_dir: str, since: str, interval: float = 0.9) -> dict:
    """The latent poll average (softmax of eta) on every saved week from `since`: mean and the
    central `interval` band. No polling error is added here: this is where the polls stand,
    not the range of the result."""
    import arviz as az

    idata = az.from_zarr(os.path.join(run_dir, "trace.zarr"))
    eta = idata.posterior["eta"]
    times = pd.to_datetime(eta.coords["time"].values)
    # The grid week nearest the election (earlier on a tie, as _to_grid_index picks it) is the
    # one the result is fitted to; the light trace always keeps it. It is dated as the election.
    gap = np.abs((times - pd.Timestamp(since)).days.values)
    first = int(np.argmin(gap))
    if gap[first] > 3:
        raise ExportRefused(f"the saved latent path has no week at the {since} election")
    keep = np.arange(first, len(times))
    parties = [str(p) for p in eta.coords["parties"].values]
    x = eta.isel(time=keep).stack(s=("chain", "draw")).transpose("s", "time", "parties").values.astype(float)
    x = x - x.max(-1, keepdims=True)
    shares = np.exp(x) / np.exp(x).sum(-1, keepdims=True)  # (S, T, P); every party exists after 2025
    lo, hi = (1 - interval) / 2, 1 - (1 - interval) / 2
    q = np.quantile(shares, [lo, hi], axis=0)
    mean = shares.mean(0)
    out = {}
    for j, p in enumerate(parties):
        out[p] = {
            "mean": [round(float(v), 4) for v in mean[:, j]],
            "low": [round(float(v), 4) for v in q[0, :, j]],
            "high": [round(float(v), 4) for v in q[1, :, j]],
        }
    dates = [str(t.date()) for t in times[keep]]
    dates[0] = str(pd.Timestamp(since).date())
    return {
        "dates": dates,
        "interval": interval,
        "includes_polling_error": False,
        "parties": out,
    }


def district_forecast(run_dir: str, run: dict, parties: list[str]) -> list[dict]:
    """Mean district shares from the run's national draws, by the same blend swing and seeded
    noise the seat simulation used (district_posterior_from_national)."""
    import xarray as xr

    from src.forecast_rw import PLACEHOLDER_ELECTION
    from src.processing.district_swing import district_posterior_from_national

    nat = np.load(os.path.join(run_dir, "national_draws.npy"))
    chains, draws = int(run["chains"]), int(run["draws"])
    if nat.shape != (chains * draws, len(parties)):
        raise ExportRefused(f"national_draws.npy has shape {nat.shape}, expected {(chains * draws, len(parties))}")
    national = xr.DataArray(
        nat.reshape(chains, draws, len(parties)),
        dims=("chain", "draw", "parties_complete"),
        coords={"chain": np.arange(chains), "draw": np.arange(draws), "parties_complete": parties},
    )
    districts = district_posterior_from_national(national, PLACEHOLDER_ELECTION)
    mean = districts.mean(("chain", "draw"))
    rows = []
    for d in mean.coords["districts"].values:
        probs = {p: round(float(mean.sel(districts=d, parties_complete=p)), 4) for p in parties}
        rows.append({"district_name": str(d), "winning_party": max(probs, key=probs.get), "probs": probs})
    return rows


def previous_result(previous: str, parties: list[str]) -> dict:
    from src.evaluation.backtest import load_truth

    _, national, _ = load_truth()
    t = pd.Timestamp(previous)
    if t not in national:
        raise ExportRefused(f"no national result for {previous}")
    s = national[t][parties]
    s = s / s.sum()
    return {"date": previous, "shares": {p: round(float(s[p]), 4) for p in parties}}


def _num(value: str) -> Optional[float]:
    try:
        return float(value) if value not in ("", None) else None
    except ValueError:
        return None


def polls_since(run_dir: str, previous: str) -> list[dict]:
    """The polls since the last election, as published (shares of each poll's own total)."""
    rows = []
    with open(os.path.join(run_dir, "input_polls.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            end = r.get("fieldwork_end") or ""
            if end <= previous:
                continue
            shares = {p: round(v / 100, 4) for p in POLL_PARTY_COLUMNS if (v := _num(r.get(p, ""))) is not None}
            n = _num(r.get("sample_size", ""))
            rows.append(
                {
                    "pollster": r.get("pollster", ""),
                    "fieldwork_start": r.get("fieldwork_start") or None,
                    "fieldwork_end": end,
                    "sample_size": int(n) if n else None,
                    "shares": shares,
                    "source_url": r.get("source_url") or None,
                }
            )
    rows.sort(key=lambda r: (r["fieldwork_end"], r["pollster"]), reverse=True)
    return rows


def fit_commit(run_dir: str) -> Optional[str]:
    """The commit the fit ran at, as the daily job records it beside the run (fit_commit.txt)."""
    path = os.path.join(run_dir, "fit_commit.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip() or None


def build(run_dir: str, code_dir: str = ".", fit_code: Optional[str] = None) -> dict:
    """Every file of the export, as {relative path: content}."""
    run = check_run(run_dir)
    with open(os.path.join(run_dir, "contested_summary.json"), encoding="utf-8") as f:
        contested = json.load(f)
    if not contested.get("districts"):
        raise ExportRefused("contested_summary.json has no districts")
    previous = next(iter(contested["districts"].values())).get("previous_election")
    if not previous:
        raise ExportRefused("contested_summary.json does not name the previous election")
    from src.evaluation.backtest import PARTIES  # the column order of national_draws.npy

    parties = [str(p) for p in PARTIES]
    seats = seat_draws(run_dir, parties)
    national = national_rows(run_dir, parties)
    polls = polls_since(run_dir, previous)
    if len(polls) != int(run.get("n_polls_since_last_election", -1)):
        raise ExportRefused(
            f"input_polls.csv has {len(polls)} polls since {previous}, run.json says {run.get('n_polls_since_last_election')}"
        )
    diag = run["diagnostics"]
    nowcast = {
        "schema": SCHEMA,
        "kind": "nowcast",
        "run": {
            "id": os.path.basename(os.path.normpath(run_dir)),
            "code": fit_code or fit_commit(run_dir),
            "export_code": _git_head(code_dir),
            "model": run.get("model"),
            "bias_form": run.get("bias_form"),
            "as_of": run.get("as_of"),
            "last_poll": run.get("last_poll"),
            "target": run.get("target"),
            "polling_error_sd": run.get("polling_error_sd"),
            "districts": run.get("districts"),
            "chains": run.get("chains"),
            "draws": run.get("draws"),
            "seat_draws": len(seats),
            "polls_since_previous": len(polls),
            "diagnostics": {
                k: diag.get(k) for k in ("divergences", "max_rhat", "min_ess_bulk", "min_ess_tail", "n_values", "passes_gate")
            },
            "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        },
        "parties": [r["party"] for r in national],
        "previous": previous_result(previous, parties),
        "national": national,
        "trend": trend_series(run_dir, previous),
        "polls": polls,
    }
    return {
        os.path.join(PUBLIC_DIR, "nowcast.json"): nowcast,
        os.path.join(PUBLIC_DIR, "district_forecast.json"): district_forecast(run_dir, run, parties),
        os.path.join(PUBLIC_DIR, "contested_summary.json"): contested,
        os.path.join(BUILD_ONLY_DIR, "seat_forecast_simulations.json"): seats,
    }


def write(files: dict, web_root: str) -> list[str]:
    written = []
    for rel, content in files.items():
        path = os.path.join(web_root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            if rel.endswith("seat_forecast_simulations.json"):
                json.dump(content, f, ensure_ascii=False, separators=(",", ":"))
            else:
                json.dump(content, f, ensure_ascii=False, indent=1)
            f.write("\n")
        os.replace(tmp, path)
        written.append(path)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a forecast_rw output folder")
    ap.add_argument("--web", help="the estimador-web checkout to write into (not needed with --check-only)")
    ap.add_argument("--fit-commit", default=None, help="the commit the fit ran at (default: the run's fit_commit.txt)")
    ap.add_argument("--check-only", action="store_true", help="run every check, write nothing")
    args = ap.parse_args(argv)
    if not args.check_only and not args.web:
        ap.error("--web is required unless --check-only")
    try:
        files = build(args.run_dir, code_dir=os.path.dirname(os.path.abspath(__file__)), fit_code=args.fit_commit)
    except ExportRefused as e:
        print(f"EXPORT REFUSED: {e}", file=sys.stderr)
        return 2
    if args.check_only:
        print(f"export ok, {len(files)} files (not written)")
        return 0
    for path in write(files, args.web):
        print(f"wrote {path}")
    head = files[os.path.join(PUBLIC_DIR, "nowcast.json")]
    print(json.dumps({"last_poll": head["run"]["last_poll"], "national": {r["party"]: r["median"] for r in head["national"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
