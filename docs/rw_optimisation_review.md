# Random-walk model: optimisation review (design, 24 Nov 2026)

Scope: make `src/models/poll_rw_model.py` (the settled national architecture: plain random walk,
blend districts, calibrated polling error) faster and its convergence more robust, **without
changing its forecasts**. The bias form (none vs static) and the per-pollster-noise / house-drift
options are being settled separately by the architecture grid (`WIP_grid_arch`) and are out of
scope here; every item below applies to whichever form wins.

## 1. What a fit costs today (measured)

| Fit | Draws/tune × chains | Weeks × parties | Step size | Tree depth | Wall time | Slowest value |
|---|---|---|---|---|---|---|
| Nowcast (ACCEPTED_legislative_nowcast_rw_v1_nobias) | 2000/2000 × 4 | ~935 × 8 | 0.0046 | 10.0 | ~20 min (shared Mac) | `sigma_rw[CDU]` bulk ESS 1,695 of 8,000 |
| Backtest 2025 h56 (diagnostic refit) | 500/500 × 4 | 817 × 8 | 0.0054 | 10.0 | 12 min (idle Mac) | `sigma_rw[CDU]` ESS 539 of 2,000 |
| Backtest grids (159 RW fits) | 500/500 × 4 | — | — | 9.4–9.6 | ~8 min median | median min ESS ~400, rhat ≤ 1.04 |

So every draw costs ~1,000 gradient evaluations (tree depth 10), and each process also spreads
over ~7 cores (XLA intra-op threads). The latent weekly steps mix very well (ESS ≥ 3,000 of
2,000 draws); the slow values are a few scale parameters (CDU's volatility, house-effect spreads)
and the house effects of the biggest pollsters on PS/AD. The common-shift direction of the
softmax is not a problem (checked 23 Nov).

## 2. Where the cost comes from

**Measured (cloud probes, 23 Nov; numpyro NUTS, rw_nobias, h56, 2 chains x 300/300):**

| Probe | Step size | Tree depth |
|---|---|---|
| 2025, current model | 0.0052 | 10 (1,023 gradients per draw) |
| 2025, all results removed | 0.0080 | 9 |
| 2025, all polls removed | 0.0076 | 9 |
| 2011, result concentration 2,000 vs 20,000 | 0.0106 vs 0.0092 | 9 vs 9 |
| 2011, zero-sum start levels over the founding parties only (entrants get their own entry level, Normal(0, 1.7); exact rewrite) | 0.039 | 7 (4x fewer gradients) |
| 2025, same zero-sum start-level rewrite | no change | — |
| 2011, plus Haar-wavelet steps on top | small further gain | — |
| nutpie, diagonal or low-rank adaptation | bigger steps | but r-hat 1.15-1.23 at 300 tuning draws |

Shares matched the current model in every exact rewrite.

**Reading:** the election results and the polls are **two roughly equal sources of stiffness**;
either one alone still forces depth 9. Both pin *sums of hundreds of non-centred weekly steps*,
directions a diagonal mass matrix cannot rescale. So my first hypothesis (results alone) is only
half the story: an anchor-and-bridge rewrite for the results alone should reach depth 9 at
best (~2x), not 5-10x. A larger gain needs the poll pinning handled as well: a multiscale or
bridge basis for the whole path, or integrating the path out. The start-level fix is free and
exact and helps where entrants dominate (2011), not on recent targets.

The long grid (800-935 weekly steps from 2009, mostly quiet mid-cycle weeks) multiplies the cost
per gradient but is not what sets the step size.

## 3. Candidate optimisations, in the order I would test them

| # | Change | Expected gain | Changes the model? | Risk |
|---|---|---|---|---|
| 0 | **Zero-sum start levels over the founding parties only**, entrants with their own entry level (measured in the cloud: exact, 4x on 2011, no change on 2025) | up to 4x on early targets | no | low; already prototyped |
| 1 | **Multiscale / bridge basis for the whole path** (anchor the walk at elections and at a coarse set of knots, Brownian bridges between them; or Haar-wavelet steps), so neither result nor poll pinning acts on long sums of steps | from depth 10 towards 6-7 only if both pinning sources are handled; results-only anchoring ~2x at best (depth 9) | no (same joint density, reparameterised) | moderate implementation work |
| 2 | **Dense mass matrix on the hyperparameters** (numpyro `dense_mass` block for `sigma_rw`, `sigma_house`, `concentration_polls`) | better mixing of the slowest values (CDU volatility, house spreads) | no | low |
| 3 | **float32** instead of float64 under JAX | ~1.5-2x per gradient | no (numerics only) | check the Dirichlet-multinomial `gammaln` terms and the 20,000 concentration |
| 4 | **Thread layout**: one XLA thread per chain and more fits in parallel in grids | throughput of grids, not single fits | no | none |
| 5 | **Coarser steps between campaigns** (e.g. 4-week steps outside the 26 weeks before each election, innovations scaled by sqrt(dt)) | grid ~2-3x shorter | slightly (time resolution mid-cycle) | house effects of mid-cycle polls; validate |
| 6 | **Partial pooling of party volatilities** (log-normal around a common scale) | small parties (CDU, PAN) mix better | yes (prior) | low; forecasts must not move |
| 7 | **Integrate the path out** (Kalman filter on a logistic-normal approximation of each poll, then forward-filter-backward-sample the path) | **10x+**: only ~150 parameters sampled | yes (likelihood approximated) | high; keep only if forecasts match the exact model |
| 8 | **Fewer draws** once 1-3 raise ESS per draw | proportional | no | none if the ESS gate still passes |

Not worth doing: removing the softmax common-shift direction (mixes fine), and learned short
timescales (do not mix; no detectable forecast gain, see `scripts/dynamics`).

## 4. Protocol for every change

* **Benchmark set**: 2025 h7, 2022 h28, 2015 h56 (backtest) and the current nowcast, 1000/1000 x 4
  chains, same seed, run one at a time on an otherwise idle Mac.
* **Speed**: wall time, mean tree depth, step size, and **min bulk ESS over all values per
  minute** (the nowcast gate's own measure), plus ESS per minute of the forecast shares.
* **Convergence**: the nowcast gate (0 divergences, every value rhat <= 1.01, ESS >= 400).
* **Same forecasts**: per fit, |delta national CRPS| <= 0.0075 (the measured difference between
  chains 1-2 and 3-4 of the same RW fit) and every party's median within 0.2 pp of the baseline.
  A change that fails this is a model change and goes through the backtest grid instead.
* Adopt a change only if it passes "same forecasts" and improves min-ESS-per-minute by >= 1.5x.

## 5. Order of work once the Mac is free

The stiffness probe is done (section 2), so:
1. Item 0 (free, exact) on the benchmark set.
2. Items 3-4 (float32, thread layout): cheap and independent of the geometry.
3. Item 2 (dense mass on the hyperparameters) for the slow scales; nutpie's low-rank adaptation
   is worth one retry with full tuning (the cloud probe used only 300 tuning draws).
4. Item 1 (multiscale/bridge basis for the whole path), since the polls pin the path as much as
   the results do.
5. Item 5 if fits are still > 5 min; item 7 only if a daily nowcast or a much larger backtest
   needs another order of magnitude.
