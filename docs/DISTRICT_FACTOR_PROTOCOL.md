# District factor test protocol

Agreed 23 November 2026. Code: `src/evaluation/district_protocol.py` (about 15 seconds per run).

A change to the district step (a new factor, or a new noise model) is re-tuned together with
everything it touches and kept only if it wins on the predictive distribution in most elections,
with the true national result and at the forecast horizons. Seats misallocated is reported but
never decides: its paired difference has an sd of about 2.5 seats per election, more than any
realistic district factor can gain.

## The rule

1. **Joint tuning.** The blend weight, the noise scale and the factor's own settings are chosen
   for each target by leave-one-election-out on district-share CRPS. The baseline gets the same
   search. A chosen value on the edge of its grid prints a warning: widen the grid and re-run.
2. **Stage 0, ceiling** (factors only). Fit the factor on the target election itself, leaving
   each district out. Stop if district-share CRPS improves by less than 10%.
3. **Stage A, true national result**, 2011–2025. Primary: seat-weighted district-share CRPS
   (parties on the ballot). Gate: district-share 90% coverage between 0.80 and 0.97.
4. **Stage B, forecast national.** The saved `rw_nobias` draws at 7, 28 and 56 days, with the
   calibrated polling error, through the same district step with the stage A settings. Primary:
   district-seat RPS. Secondary: seat-total RPS and 90% coverage (gate 0.85).
5. **Keep** only if stage A is better in at least 4 elections with a mean gain of at least 3% and
   a one-sided sign-flip p below 0.05 over region × election clusters; stage B is better at two or
   more horizons in at least 3 of 5 elections; seat-total RPS is no more than 1% worse at any
   horizon; and both coverage gates hold.
6. **Amendments** (agreed 23 November 2026, after the first run). A target where either side
   calibrates on earlier elections and the data hold fewer than two of them (2011: the results
   start in 2009) is reported but not counted. The coverage gates bind the baseline too: against a
   baseline that fails them there is no verdict (`keep` is `None`); fix the baseline first.

The baseline is the share-scale district noise (`sqrt`, below). `--baseline current` runs the
log-share noise it replaced.

```bash
python -m src.evaluation.district_protocol --stage AB \
  --national-dir outputs/backtest/WIP_grid_v1 --out outputs/district_protocol
```

## First run: district noise on the √p scale

The current noise is a party log sd from past elections, pooled toward the all-party value
(`calibrate_noise`), times a tuned scale k. The candidate is one parameter c on the share scale,
s = μ + c√μ z. A third form adds a region-wide share φ = 0.3 of the noise.

**Verdict: passes the rule as written, narrowly. The gain is calibration, not accuracy.** Adopted
on 23 November 2026, with the amendments above: the model and the backtest now draw district noise
with `add_sqrt_noise`, c from `calibrate_sqrt_noise` on earlier elections only (0.03 for the next
election; 0.025–0.035 for the 2015–2025 targets).

Stage A, district-share CRPS in pp (settings chosen leave-one-election-out):

| | 2011 | 2015 | 2019 | 2022 | 2024 | 2025 | 90% coverage |
|---|---|---|---|---|---|---|---|
| Current (k 0.4–0.6) | 0.720 | 1.251 | 0.558 | 0.442 | 0.510 | 0.371 | 0.62 |
| √p (c 0.03–0.035) | **0.571** | **1.190** | **0.524** | **0.436** | 0.521 | 0.387 | 0.85 |

Better in 4 of 6, mean gain 4.5%, sign-flip p 0.005. Two things weaken it:

- **2011 carries half the gain.** The results start in 2009, so for 2011 the current noise has no
  earlier pair to calibrate on and falls back to 0.25 for every party. On 2015–2025 alone, √p is
  better in 3 of 5 with a mean gain of 1.3%, which would fail.
- **The verdict moved with the grid.** With k searched down to 0.6 only, every choice sat on the
  edge and √p failed stage A (3 of 6, p 0.22). Searching down to 0.3, leave-one-out picks 0.4,
  which is worse on held-out 2015. The edge warning now catches this.

What does not move: tuned for CRPS, the current noise covers 58–62% of district shares in its
90% interval, because one scale cannot fix a shape that is too wide for the big parties and too
narrow for the small ones. √p covers 85%. At k = 1 the current noise covers 82% but its CRPS is
7% worse than √p's on 2015–2025.

Stage B, averages over 2015–2025:

| Horizon | District-seat RPS | Seat-total RPS | Seat-total 90% coverage |
|---|---|---|---|
| 7 days | 19.94 → 19.92 | 2.161 → 2.148 | 0.84 → 0.94 |
| 28 days | 25.35 → 25.27 | 3.051 → 3.033 | 0.92 → 0.94 |
| 56 days | 27.74 → 27.69 | 3.376 → 3.363 | 0.89 → 0.92 |

Better at two or more horizons in 3 of 5 elections. Once the national forecast is uncertain, the
form of the district noise changes seat scores by under 0.5%.

The region-wide share (φ 0.3) scores the same as √p on every line, as expected: it changes how
districts move together (majority and bloc probabilities), which none of these scores sees.

### The model's rule against the tuned one

The model cannot tune c leave-one-election-out: it takes c from the elections before the target
(`sqrt_hist`, a normal approximation to the CRPS). It scores like the tuned version: stage A share
CRPS 0.607 against 0.605, 90% coverage 0.82 against 0.85, and in stage B seat-total coverage 0.92 to
0.94 at every horizon, the same as the tuned √p.

Under the amended rule, the log-share noise could not be a baseline: tuned for CRPS it fails both
coverage gates (0.58 for district shares, 0.84 for seat totals at 7 days), so the rule gives no
verdict against it.

## Regional poll crosstabs: measured, and stopped

Code: `src/evaluation/crosstab_measurement.py` (reads the ERC deposits and the NUTS file, which live
outside this repository; the command is in its docstring).

The stop rule was fixed before measuring: stop if the pooled effective sample is under 10,000, or if
a pollster's regional deviation from its own national figure lies more than 0.07 (rms, log scale)
from the true deviation. **The sample fails in every election; the bias sits at the threshold overall
and above it in 2024 and 2025. Regional crosstabs are not modelled.**

| Election | Pollsters with regional tables | Respondents naming a party | Bias, rms log (90% interval) |
|---|---|---|---|
| 2011 | 3 | 6,600 | 0.050 (0.021–0.072) |
| 2015 | 3 | 9,100 | 0.045 (0.037–0.057) |
| 2022 | 1 | 600 | not measured |
| 2024 | 3 | 5,500 | 0.077 (0.059–0.093) |
| 2025 | 4 | 7,200 | 0.098 (0.045–0.186) |

- **The data.** 156 deposits fall in the final 8 weeks before these elections; 61 hold a usable
  regional vote table. 2019, and all but one 2022 deposit, were never read by the v2 parser. CESOP,
  ICS/ISCTE and most Aximage deposits have no regional vote table. 13 tables were misread: 10
  Pitagórica tracker tables had every value one label to the left (realigned), 3 Consulmark tables
  repeat the total inside the table (dropped). Region definitions come from the fichas técnicas:
  Marktest's and Intercampus's 2011 "Grande Lisboa" and "Grande Porto" are the old NUTS-III areas
  (20% and 11–13% of their samples), not the metropolitan areas.
- **Sample.** Distinct interviews that name a party (a rolling tracker counts only its new
  interviews). The measured design effect is 0.80 (0.65–1.04): a pollster's regional figures scatter
  less from table to table than random samples would, probably from weighting. That lifts 2015 to
  11,400 and leaves the others under 10,000. If every poll in the final 8 weeks had published a usable
  regional table, the pool would still be 12,000–19,000, short of the 20,000–40,000 the simulation
  needed for a gain of 0.4–1.4 seats with no bias at all.
- **Bias.** 0.072 (0.052–0.096) over all pollster-elections, 0.082 if the sampling correction uses the
  measured design effect. By pollster it runs from about zero (Eurosondagem 2015, Intercampus 2025) to
  0.19 (Aximage 2024) and 0.30 (Metris 2025, eight weeks out).
- **Pooling does not remove it, and history cannot correct it.** On the regions several pollsters
  share, their pooled error is 0.10 in 2024 and 0.08 in 2025 (in 2024 every pollster had IL too high
  in Porto and too low in Lisbon); the simulation halved the gain at 0.05 and fell below blend swing
  at 0.10. A pollster's regional errors do not repeat from one election to the next (correlation
  −0.39 to +0.17 over 8–20 cells), so house effects fitted on earlier elections, the planned model,
  would have nothing to learn.

What would reopen it: 2019 and 2022 read by the parser, a pool of regional tables above 20,000
respondents, and regional errors under 0.05 that repeat by pollster.
