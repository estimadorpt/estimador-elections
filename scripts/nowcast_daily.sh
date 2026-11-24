#!/bin/bash
# The live legislative nowcast on estimador.pt (/eleicoes/proximas-legislativas):
# when the poll feed holds polls the published estimate has not read, refit the
# random walk, check the gate, export the page's files and push them to
# estimador-web's main (Azure deploys on push). Run by launchd
# (ops/launchd/pt.estimador.nowcast.plist) every hour; a run with no new polls
# only checks the feed and exits.
#
#   scripts/nowcast_daily.sh             # the scheduled run
#   scripts/nowcast_daily.sh --no-push   # everything up to the commit, nothing pushed
#   scripts/nowcast_daily.sh --force     # refit even if the polls have not changed
#   scripts/nowcast_daily.sh --publish-run outputs/ACCEPTED_...  # export and push an existing run, no fit
#
# Nothing reaches the site unless every step passes: the fit exits 0, run.json's gate passes
# (r-hat <= 1.01 and ESS >= 400 on every value, no divergences), the export's own checks pass
# (published bias form, seats add to 230, shares add to 1, poll count matches) and the web
# validator passes. Otherwise the last good estimate stays online and an ntfy alert says why.
#
# Paths (override with the environment):
#   ELECTIONS  this checkout, pinned to a reviewed commit (never Bernardo's ~/code/estimador-elections)
#   FEED       estimador-data's poll feed CSV (written hourly by pt.estimador.poll-feed)
#   WEB        a detached worktree of estimador-web used only by this job; it is reset to origin/main
#              on every run, so never edit it by hand
set -uo pipefail

ELECTIONS=${ELECTIONS:-$HOME/code/estimador-elections-live}
FEED=${FEED:-$HOME/code/estimador-data/data/polls/feed/legislative_polls.csv}
WEB=${WEB:-$HOME/code/estimador-web-nowcast-publish}
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
NTFY_TOPIC=${NTFY_TOPIC:-bc-estimador}
FEED_MAX_AGE_HOURS=${FEED_MAX_AGE_HOURS:-26}
PREVIOUS_ELECTION=2025-05-18
OUTPUTS=$ELECTIONS/outputs
LOG=$OUTPUTS/nowcast_daily.log
STATE=$OUTPUTS/nowcast_daily.state      # "<poll hash> <run folder>" of the last published run
STALE_MARK=$OUTPUTS/nowcast_daily.stale_alerted
ATTEMPT=$OUTPUTS/nowcast_daily.attempt  # "<poll hash> <failed|accepted> <run folder>" of the last fit
ALERTED=$OUTPUTS/nowcast_daily.alerted  # the last failure alert, so a repeat within 6 h stays quiet
LOCK=$OUTPUTS/.nowcast_daily.lock

PUSH=1
FORCE=0
PUBLISH_RUN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --no-push) PUSH=0 ;;
    --force) FORCE=1 ;;
    --publish-run) PUBLISH_RUN="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 64 ;;
  esac
  shift
done

mkdir -p "$OUTPUTS"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts) $*" >> "$LOG"; }
# ntfy topics are public to anyone who knows the name: no home-folder paths in the text.
ntfy() { [ -n "$NTFY_TOPIC" ] || return 0; curl -s -m 20 -H "Title: $1" -d "${2//$HOME/~}" "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true; }
fail() {
  log "STOPPED: $1"
  if [ "$(cat "$ALERTED" 2>/dev/null)" != "$1" ] || [ -z "$(find "$ALERTED" -mmin -360 2>/dev/null)" ]; then
    ntfy "estimador nowcast: not published" "$1"
    printf '%s' "$1" > "$ALERTED"
  fi
  exit 1
}

if ! mkdir "$LOCK" 2>/dev/null; then
  # launchd never starts a second copy, but a hand run might overlap the scheduled one.
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +240 2>/dev/null)" ]; then rmdir "$LOCK"; mkdir "$LOCK"; else log "another run holds $LOCK"; exit 0; fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

cd "$ELECTIONS" || fail "no checkout at $ELECTIONS"
log "START $(git log --oneline -1) push=$PUSH force=$FORCE${PUBLISH_RUN:+ publish-run=$PUBLISH_RUN}"

# What the polls are, independent of columns that change without a new poll (entered_at, notes):
# the hash of every poll since the last election, and how many there are.
poll_digest() {
  /usr/bin/python3 -I - "$1" "$PREVIOUS_ELECTION" <<'PY'
import csv, hashlib, sys
path, since = sys.argv[1], sys.argv[2]
keys = ("pollster", "fieldwork_start", "fieldwork_end", "sample_size", "AD", "PS", "CH", "IL", "L", "CDU", "BE", "PAN", "JPP", "Others")
rows = sorted(tuple((r.get(k) or "").strip() for k in keys) for r in csv.DictReader(open(path, encoding="utf-8")) if (r.get("fieldwork_end") or "") > since)
print(hashlib.sha256(repr(rows).encode()).hexdigest()[:16], len(rows))
PY
}

publish() {
  # Export run folder $1 into the publish worktree, validate, commit and push.
  local run="$1"
  local as_of last_poll
  [ -d "$WEB/.git" ] || [ -f "$WEB/.git" ] || fail "no web checkout at $WEB"
  git -C "$WEB" fetch -q origin main || fail "git fetch of estimador-web failed"
  git -C "$WEB" checkout -q --detach origin/main && git -C "$WEB" reset -q --hard origin/main || fail "could not reset $WEB to origin/main"
  "$PIXI" run python -m src.export_nowcast_web "$run" --web "$WEB" >> "$LOG" 2>&1 || fail "export refused $run (see $LOG)"
  (cd "$WEB" && node scripts/validate-data.mjs >> "$LOG" 2>&1) || fail "validate-data refused the export of $run (see $LOG)"
  git -C "$WEB" add public/data/elections/legislative-nowcast data/build-only/elections/legislative-nowcast
  if git -C "$WEB" diff --cached --quiet; then log "export of $run matches main: nothing to push"; return 0; fi
  as_of=$(/usr/bin/python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['as_of'])" "$run/run.json")
  last_poll=$(/usr/bin/python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['last_poll'])" "$run/run.json")
  git -C "$WEB" -c user.name="estimador nowcast" commit -q -m "data: legislative nowcast of $as_of (polls to $last_poll)" -m "Run $(basename "$run"), fit at $(cat "$run/fit_commit.txt" 2>/dev/null || echo unknown)." || fail "commit failed"
  if [ "$PUSH" = 0 ]; then log "NOT PUSHED (--no-push): $(git -C "$WEB" log --oneline -1)"; return 0; fi
  local attempt=1
  until git -C "$WEB" push -q origin HEAD:main >> "$LOG" 2>&1; do
    [ $attempt -ge 4 ] && fail "push to estimador-web main failed 4 times"
    sleep $((attempt * 20))
    git -C "$WEB" fetch -q origin main && git -C "$WEB" rebase -q origin/main >> "$LOG" 2>&1 || fail "rebase onto main failed"
    attempt=$((attempt + 1))
  done
  log "PUSHED $(git -C "$WEB" log --oneline -1)"
  local summary
  summary=$(/usr/bin/python3 -I -c "
import json,sys
n=json.load(open(sys.argv[1]))
print(', '.join(f\"{r['party']} {r['median']*100:.1f}\" for r in n['national'][:4]))" "$WEB/public/data/elections/legislative-nowcast/nowcast.json")
  ntfy "estimador nowcast published" "Polls to $last_poll: $summary. The site deploys in about 15 minutes."
}

if [ -n "$PUBLISH_RUN" ]; then
  publish "$PUBLISH_RUN"
  log "ALL DONE"
  exit 0
fi

# 1. The feed.
[ -s "$FEED" ] || fail "the poll feed $FEED is missing or empty"
read -r digest npolls <<< "$(poll_digest "$FEED")" || fail "could not read $FEED"
[ -n "${npolls:-}" ] || fail "could not read $FEED"
if [ -n "$(find "$FEED" -mmin +$((FEED_MAX_AGE_HOURS * 60)) 2>/dev/null)" ]; then
  # Once a day at most: the hourly feed job (pt.estimador.poll-feed) has stopped.
  if [ -z "$(find "$STALE_MARK" -mmin -1440 2>/dev/null)" ]; then
    ntfy "estimador poll feed is stale" "$FEED was last written $(stat -f '%Sm' -t '%F %R' "$FEED"); the nowcast cannot see new polls until the feed runs again."
    touch "$STALE_MARK"
  fi
  log "feed is older than ${FEED_MAX_AGE_HOURS} h"
fi
last_digest=""; last_run=""
[ -f "$STATE" ] && read -r last_digest last_run < "$STATE"
if [ -n "$last_run" ] && [ -f "$OUTPUTS/$last_run/input_polls.csv" ]; then
  read -r _ last_n <<< "$(poll_digest "$OUTPUTS/$last_run/input_polls.csv")"
  # A hand entry that Wikipedia later lists can merge into one row; a bigger drop is a broken feed.
  if [ "$npolls" -lt "$((last_n - 3))" ]; then
    fail "the feed lists $npolls polls since $PREVIOUS_ELECTION, far fewer than the $last_n the published run read"
  elif [ "$npolls" -lt "$last_n" ]; then
    log "the feed lists $npolls polls, $last_n before (a merged entry?); going on"
  fi
fi
if [ "$digest" = "$last_digest" ] && [ "$FORCE" = 0 ]; then
  log "no new polls ($npolls since $PREVIOUS_ELECTION, digest $digest)"
  exit 0
fi
# The same polls were fitted already: a failed fit fails the same way again (the seed is fixed),
# so it waits for new polls or --force; an accepted run that did not get online is published again.
attempt_digest=""; attempt_status=""; attempt_run=""
[ -f "$ATTEMPT" ] && read -r attempt_digest attempt_status attempt_run < "$ATTEMPT"
if [ "$digest" = "$attempt_digest" ] && [ "$FORCE" = 0 ]; then
  if [ "$attempt_status" = "accepted" ] && [ -d "$OUTPUTS/$attempt_run" ]; then
    log "publishing $attempt_run again (accepted, not yet online)"
    publish "$OUTPUTS/$attempt_run"
    [ "$PUSH" = 1 ] && echo "$digest $attempt_run" > "$STATE"
    log "ALL DONE"
    exit 0
  fi
  log "these polls failed in $attempt_run already; waiting for new polls (or --force)"
  exit 0
fi

# 2. The fit, on a frozen copy of the feed.
today=$(date +%F)
out="outputs/WIP_legislative_nowcast_rw_$(date +%Y%m%d_%H%M)"
mkdir -p "$out"
cp "$FEED" "$out/input_polls.csv"
git rev-parse HEAD > "$out/fit_commit.txt"
touch "$out/.made_by_nowcast_daily"
log "FIT_START $out ($npolls polls since $PREVIOUS_ELECTION, digest $digest)"
start=$(date +%s)
ESTIMADOR_WIKIPEDIA_POLLS="$PWD/$out/input_polls.csv" /usr/bin/caffeinate -is \
  "$PIXI" run python -m src.forecast_rw --out "$out" --as-of "$today" --draws 2000 --tune 2000 --chains 4 > "$out/fit.log" 2>&1
rc=$?
log "FIT_DONE rc=$rc in $(( $(date +%s) - start )) s"
if [ $rc -ne 0 ]; then echo "$digest failed $(basename "$out")" > "$ATTEMPT"; fail "the fit exited $rc (see $ELECTIONS/$out/fit.log)"; fi
gate=$(/usr/bin/python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['diagnostics'].get('passes_gate'))" "$out/run.json" 2>/dev/null)
if [ "$gate" != "True" ]; then echo "$digest failed $(basename "$out")" > "$ATTEMPT"; fail "the fit in $out did not pass the convergence gate (passes_gate=$gate)"; fi

# 3. Accepted: the gate is the acceptance rule (CLAUDE.md, "When to mark as ACCEPTED").
accepted="outputs/ACCEPTED_legislative_nowcast_rw_v1_nobias_$(date +%Y%m%d)"
[ -e "$accepted" ] && accepted="${accepted}_$(date +%H%M)"
mv "$out" "$accepted" || fail "could not rename $out"
echo "$digest accepted $(basename "$accepted")" > "$ATTEMPT"
log "ACCEPTED $accepted"

# 4. Publish, then remember what was published.
publish "$ELECTIONS/$accepted"
[ "$PUSH" = 1 ] && echo "$digest $(basename "$accepted")" > "$STATE"

# 5. Keep the disk in check: drop the trace (about 75 MB) from this job's runs after 30 days; the
#    run stays reproducible from input_polls.csv, fit_commit.txt and the seed in run.json.
find "$OUTPUTS" -maxdepth 2 -name .made_by_nowcast_daily -mtime +30 2>/dev/null | while read -r mark; do
  rm -rf "$(dirname "$mark")/trace.zarr"
done
log "ALL DONE"
