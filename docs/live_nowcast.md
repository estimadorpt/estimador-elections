# The live legislative nowcast

estimador.pt/pt/eleicoes/proximas-legislativas shows what the polls would give if the next
parliamentary election were held on the day of the latest poll. No election has been called; the
page is a nowcast, not a forecast to a date. This runbook covers the job that keeps it current.

## What runs

`pt.estimador.nowcast` (launchd, every hour, `ops/launchd/pt.estimador.nowcast.plist`) runs
`scripts/nowcast_daily.sh` in the dedicated checkout `~/code/estimador-elections-live`. That
checkout is pinned to a reviewed commit and never pulled by the job: move it deliberately
(`git fetch && git checkout --detach origin/main`) after a model change is merged.

Each run:

1. Reads the poll feed, `~/code/estimador-data/data/polls/feed/legislative_polls.csv`, written
   hourly by `pt.estimador.poll-feed` (estimador-data). It hashes the polls since the 18 May 2025
   election (pollster, fieldwork, sample, shares; not the notes or entry times). The same hash as
   the last published run means nothing new: the run exits.
2. Stops with an alert if the feed lists more than three polls fewer than the published run read
   (one to three fewer is logged: a hand entry Wikipedia later lists merges into one row), and
   alerts once a day if the feed file is more than 26 hours old. Polls that were fitted already
   are not fitted again: a failed fit waits for new polls (or `--force`), and an accepted run that
   did not get online is published again.
3. Copies the feed into a new `outputs/WIP_legislative_nowcast_rw_<date>_<time>/input_polls.csv`,
   records the commit in `fit_commit.txt`, and runs `python -m src.forecast_rw --as-of <today>
   --draws 2000 --tune 2000 --chains 4` on that copy (about 50 minutes).
4. Renames the folder `ACCEPTED_legislative_nowcast_rw_v1_nobias_<date>` only if the fit exited 0
   and `run.json` passes the gate (r-hat at most 1.01 and ESS at least 400 on every value).
5. Resets the web worktree `~/code/estimador-web-nowcast-publish` to `origin/main`, exports the
   run (`src/export_nowcast_web.py`, which refuses a scenario bias form, a target other than the
   last poll, seats that do not add to 230, shares that do not add to 1, or a poll count that
   disagrees with `run.json`), runs estimador-web's `scripts/validate-data.mjs`, commits the four
   files and pushes to `main` (fetch, rebase and retry up to four times: the football job pushes
   to the same branch). Azure deploys on push.
6. Sends an ntfy alert (topic `bc-estimador`) with the new medians, or with the reason nothing
   was published (the same failure is not repeated within 6 hours). On any failure the last good
   estimate stays online.

The page lists the feed's rows the loader keeps (a sample size, a fieldwork end, one row per
pollster and fieldwork end, Wikipedia's first); the model's own count can differ by a poll the
Marktest history already holds, and the export refuses a gap of more than three.

Runs older than 30 days keep everything but `trace.zarr` (about 75 MB each); a run stays
reproducible from `input_polls.csv`, `fit_commit.txt` and the seed in `run.json`.

## Files

| Where | What |
|---|---|
| `outputs/nowcast_daily.log` | one line per step, every run |
| `outputs/nowcast_daily.state` | the poll hash and folder of the last published run |
| `outputs/nowcast_daily.attempt` | the poll hash, outcome (`failed` or `accepted`) and folder of the last fit |
| `~/Library/Logs/estimador-nowcast{,.err}.log` | launchd's stdout and stderr |
| estimador-web `public/data/elections/legislative-nowcast/` | `nowcast.json`, `district_forecast.json`, `contested_summary.json` |
| estimador-web `data/build-only/elections/legislative-nowcast/` | `seat_forecast_simulations.json` (read at build only) |

## By hand

```bash
cd ~/code/estimador-elections-live
scripts/nowcast_daily.sh --no-push                 # a full run that commits in the web worktree but does not push
scripts/nowcast_daily.sh --force                   # refit and publish even without new polls
scripts/nowcast_daily.sh --publish-run outputs/ACCEPTED_...   # publish an existing run, no fit
launchctl kickstart gui/$(id -u)/pt.estimador.nowcast         # the scheduled job, now
```

## Install

```bash
git -C ~/code/estimador-web worktree add --detach ~/code/estimador-web-nowcast-publish origin/main
cp ~/code/estimador-elections-live/ops/launchd/pt.estimador.nowcast.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/pt.estimador.nowcast.plist
```

## When an election is called

The page and the job read the estimate at the last poll. Once a date is set, the forecast to that
date (`pixi run forecast <YYYY-MM-DD>`) replaces it; the job, the export's "not a nowcast" check
and the page's copy all need that change together.
