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

```bash
python -m src.evaluation.district_protocol --stage AB \
  --national-dir outputs/backtest/WIP_grid_v1 --out outputs/district_protocol
```

## First run: district noise on the √p scale

The current noise is a party log sd from past elections, pooled toward the all-party value
(`calibrate_noise`), times a tuned scale k. The candidate is one parameter c on the share scale,
s = μ + c√μ z. A third form adds a region-wide share φ = 0.3 of the noise.

**Verdict: passes the rule as written, narrowly. The gain is calibration, not accuracy.**

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
