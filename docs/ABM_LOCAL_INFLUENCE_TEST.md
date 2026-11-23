# Pre-registered test: does local social influence add anything to blend swing?

Written 23 November 2026, before any of the code below was run. The decision rule in section 5
is fixed; results go in a separate section added after the run, and nothing above it changes.

## 1. Why this test

Agent-based models of voting (the voter model and its noisy, zealot and majority-rule
variants; bounded-confidence models; LLM agents talking to each other) all rest on one
mechanism that a swing rule does not have: a voter's choice moves toward the choices of the
people around them. In aggregate data that mechanism has two forecastable signatures:

1. **Convergence.** A party's share in a parish moves toward its share in the neighbouring
   parishes (the drift term of the voter model, ds_i/dt ∝ s̄_N(i) − s_i).
2. **Diffusion.** Where a party just gained, the parishes next door gain more in the following
   election (a front moving outwards from strongholds).

If neither signature predicts the next election better than blend swing, a social-influence
ABM cannot improve the district forecast, however it is built, because whatever it adds at
district level has to come through these terms. Most seat error is national polling error
(about 8.6 of 11.4 seats), which no voter-level mechanism touches, so this test only asks about
the district step, as the ecological and survey threads did.

## 2. Data

- Parish results for 2015, 2019, 2022, 2024 and 2025 (`load_parish_level_results`, 3,092
  parishes with one code map), eight parties as in `backtest.PARTIES`.
- Neighbours: queen contiguity from the parish polygons the website serves
  (`estimador-web/public/data/population-geography/parishes/*.json`), with a 30 m buffer to
  absorb the web simplification. A parish with no neighbour (small islands) keeps its own share
  as its neighbour average, so it gets no correction.
- The true national result of the target election (oracle), as in `backtest --district-oracle`.

## 3. Rules compared

All corrections act on the log share of party p in parish i, after parish-level blend swing
(B1) and before renormalising. Shares at t−1 use (votes + 0.5) / (total + 4) so empty cells
have a log.

- **B0**: blend swing at district level (the production rule).
- **B1**: blend swing applied to each parish, aggregated to districts with t−1 votes.
- **M0, denoising control**: own t−1 share shrunk toward the neighbour average by its
  binomial sampling variance only (empirical Bayes, one prior variance per party and election
  estimated from neighbour gaps). This is what "convergence" looks like with no influence at
  all, just noise in small parishes.
- **M1, convergence**: β · (log s̄_N,t−1 − log s_i,t−1).
- **M2, diffusion**: γ · Δlog s̄_N(t−2→t−1) + δ · Δlog s_i(t−2→t−1) (neighbours' last change
  and the parish's own last change, so momentum is not mistaken for spread).
- **M1+M2** together.

Coefficients are pooled over parties (one β, one γ, one δ), estimated by least squares on the
parish residual log(true/B1), weighted by the parish's votes at t. A party-specific version is
reported as secondary.

**Learned (the real forecast):** coefficients fitted only on earlier targets. M1 targets 2022
(fit on 2019), 2024 (2019–2022) and 2025 (2019–2024). M2 needs t−2, so it targets 2024 (fit on
2022) and 2025 (2022–2024). **Oracle (ceiling):** coefficients fitted on the target itself.

## 4. Scores

- District MAE, pp, over the 20 domestic districts × 8 parties (the district oracle's score).
- District seats misallocated and national seat error (D'Hondt with the 2024 seat map).
- Parish MAE, pp, weighted by votes at t (where a local mechanism should show most).

## 5. Decision rule (fixed)

- **Forecast value.** A mechanism earns a place in the district step only if its *learned*
  version improves mean district MAE over its targets by at least 0.03 pp against B0, reduces
  mean seats misallocated, and worsens no single target by more than 0.02 pp.
- **Evidence of a local mechanism (explanation, not forecast).** M1 beats the denoising
  control M0 by at least 3% of parish MAE in the learned setting, or γ is positive with a 95%
  interval above zero in every M2 fit.
- If neither holds, a social-influence ABM is not built for forecasting, and the ABM thread
  moves to scenarios and explanation only.

## 6. Known limit

A positive γ cannot tell contagion from neighbouring parishes sharing a local trend (Manski's
reflection problem): aggregate results identify the forecasting question, not the mechanism.
A descriptive check reports Chega's 2019→2022 and 2022→2024 gains against its own and its
neighbours' earlier share, in parishes with at least 1,000 voters, with district fixed
effects. It informs the explanation; it is not a gate.

    python -m src.evaluation.local_influence --all
