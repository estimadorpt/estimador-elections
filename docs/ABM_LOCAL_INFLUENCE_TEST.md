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

## 7. Addendum, written before running it: strategic desertion in small districts

The other voter-level mechanism that could move district shares is strategic voting: voters
desert a party that cannot win a seat in their district and come back when it can (Cox's
M+1 logic; Gschwend 2007 estimates that a party which won no seat in a Portuguese district
loses about 3% of its vote there, more where magnitude is small). The static part of that is
already in blend swing, because last election's offsets include last election's desertion. What
blend cannot see is a change in viability: a party crossing a small district's effective
threshold should gain more than its swing, and one falling below it should lose more.

- **Viability** v(s, M) = logistic((s − T) / (0.2 T)), with T = 0.75 / (M + 1), the effective
  threshold for magnitude M (2024 seat map).
- **Feature** Δv = v(B0's predicted share at t, M) − v(true share at t−1, M); everything in it
  is known before the election once the national result is given.
- **Rule S**: B0 with θ · Δv added to the log share, then renormalised. θ is pooled over
  parties, fitted by least squares on log(true/B0) weighted by district magnitude, on earlier
  transitions only. District results 2009–2025 give transitions from 2009→2011; learned
  targets are 2015, 2019, 2022, 2024 and 2025. An oracle fit on the target is reported as
  the ceiling.
- **Decision**: the same as section 5 (at least 0.03 pp better mean district MAE than B0 over
  the learned targets, fewer seats misallocated, no target worse by more than 0.02 pp).

---

## Results (added after the run, 23 November 2026)

**Verdict: no mechanism earns a place in the district step, and the pre-registered evidence
of a local mechanism does not survive a corrected fit.** A social-influence ABM is not built
for forecasting.

### Review and correction

A review of the first run (Fable, 23 November) found two faults in the fit, both in the code,
not the specification's intent:

1. Cells with no votes at t−1 or t (a party with no list, or none cast) entered the fit with
   log residuals of 5 to 11. They are 1–1.6% of the weight but 40–79% of the weighted sum of
   squares, and pushed β up (0.40–0.47; 0.32–0.34 without them).
2. The fit had no party intercept. Both M2 features carry the party's national log change, so
   least squares forced γ ≈ −δ, which made γ look positive and significant in every fit.

The corrected fit drops those cells and absorbs a party × election mean
(`FIT` in `local_influence.py`); `--as-registered` reproduces the first run. Both are reported.
Everything below is the corrected fit unless marked.

### Forecast value (learned coefficients, true national result)

| Rule | Targets | District MAE, pp | vs B0 | Seats misallocated | vs B0 |
|---|---|---|---|---|---|
| B0 district blend | 2022–2025 | 0.655 | | 5.33 | |
| B1 parish blend | 2022–2025 | 0.641 | −0.014 | 5.67 | +0.33 |
| M0 denoised | 2022–2025 | 0.644 | −0.011 | 5.67 | +0.33 |
| M1 pooled (primary) | 2022–2025 | 0.658 | +0.003 | 5.33 | 0 |
| M1 party-specific (secondary) | 2022–2025 | 0.643 | −0.012 | 5.67 | +0.33 |
| M1 party, party-vote weights (post hoc) | 2022–2025 | 0.640 | −0.015 | 5.33 | 0 |
| B0 district blend | 2024–2025 | 0.613 | | 5.5 | |
| M2 pooled (primary) | 2024–2025 | 0.712 | +0.099 | 8.0 | +2.5 |
| M2 party-specific (secondary) | 2024–2025 | 0.607 | −0.006 | 5.5 | 0 |
| M2 party, party-vote weights (post hoc) | 2024–2025 | 0.607 | −0.006 | 5.0 | −0.5 |
| M2 party, party-vote weights, **oracle** | 2024–2025 | 0.565 | −0.048 | 3.5 | −2.0 |

As registered (first run), M1 pooled was 0.670 and M2 pooled 0.637 (7.5 seats): worse than B0
either way. The review also tried share-scale weights (N·b1²) and a direct grid search on
district MAE over earlier targets: the best learned result was 0.606 against 0.613, with more
seats wrong, and the grid search picked M2 coefficients of about zero.

No rule meets the bar of 0.03 pp. The oracle ceiling (0.048 pp, fitted on the target itself)
shows there is some in-sample signal, but its coefficients do not carry from one election to
the next: the M2 fit on 2022 applied to 2024 gives 0.851 pp and 10 seats misallocated.

### Evidence of a local mechanism

- **M1 against the denoising control M0, parish MAE (learned, 2022–2025):** best M1 1.036 pp
  against M0 1.028. Convergence toward neighbours does no better than removing sampling noise.
  Fails.
- **γ, the neighbours' last change:** as registered, positive with an interval above zero in
  every fit (+0.52, +0.38 learned; +0.28 to +0.52 oracle), which meets the letter of the
  criterion. That result came from the missing intercept. Corrected, γ is −0.001 (0.041) on
  2022, +0.22 (0.04) on 2024 and +0.13 (0.02) on 2025: not above zero in every fit. Fails.
- The review adds that the own-change coefficient δ (−0.04 to −0.34) is about what binomial noise
  in the t−1 share predicts, and that running the regression backwards in time gives the same
  pattern. A diffusion front should not run backwards. The neighbourhood-wide term γ + δ is
  positive only for 2022→2024: one unreplicated transition.

### Chega's spread (descriptive)

Chega's change in pp on its own and its neighbours' earlier share, within districts,
municipality-clustered SE:

| Change | Parishes ≥ 1,000 voters: neighbours | Parishes ≥ 5,000 voters: neighbours |
|---|---|---|
| 2019→2022 | +0.48 (0.20) | +0.17 (0.35) |
| 2022→2024 | +0.18 (0.08) | +0.13 (0.13) |
| 2024→2025 | −0.07 (0.03) | −0.09 (0.04) |

The neighbour effect shrinks where parishes are big enough for their own share to be measured
well, and turns negative once Chega is established. Its growth followed its own earlier share
(own-share coefficient +1.2 to +1.3 in large parishes before 2024) plus a uniform rise: no sign
of spread from parish to parish.

### Strategic desertion

θ is between −0.10 and +0.07 with standard errors of 0.03–0.09 in every fit; the learned rule
changes mean district MAE from 0.794 to 0.790 pp (7.4 to 7.2 seats). The test cannot tell zero
from Gschwend's 3% (θ ≈ 0.03), but even that size moves district shares by 0.01–0.06 pp, so
true or not, desertion is too small to matter for districts. Restricting to small districts
gives a negative θ.
