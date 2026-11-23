# Survey microdata for district votes: tested, not worth building

23 November 2026. Code: `src/evaluation/survey_districts.py` (`--all` runs everything in about a minute).

## Question

Can individual survey responses, poststratified on census cells (MRP or MrsP), set better district
priors than blend swing, especially for parties with a short history (Chega, IL, Livre)?

## Answer

No. Every test was run in the most favourable setting available: the true national result, and in
most tests the post-election survey of the election being predicted, which no forecast will ever
have. Even there, survey microdata do not beat blend swing on district error or seats.

| Test | Blend swing | Best survey variant | Verdict |
|---|---|---|---|
| Demographic swing, true (2015–2024) | 0.851 pp, 8.25 seats | 0.833 pp, 8.50 seats | gain under 0.02 pp, seats worse |
| Survey geography for young parties (8 cases) | best in 7 of 8 | best in 1 of 8 (CH 2024, 0.04 pp) | worse than blend |
| Regional swing, from the same surveys | 0.851 pp | 0.846 pp at best, up to 1.10 | noise |

District error is the mean absolute error over 20 districts × 8 parties, as in
`backtest.py --district-oracle`; seats are seats misallocated across districts.

## Data

- **Estudo Eleitoral Português 2002–2024** (ICS, APIS; 8 post-election waves, 11,330 respondents,
  500–800 per wave who name one of the eight parties). Vote, previous vote, sex, age, education,
  NUTS-II region, district. Mainland only: the Azores and Madeira have 0–3 respondents per wave.
  The sample is clustered: in 2019 Bragança, Guarda and Leiria have no respondents; in 2024 Beja,
  Bragança, Guarda and Leiria have none.
- **Census 2021** (INE): sex × 13 age bands × 11 education levels for every parish, minus
  foreign nationals, aggregated to district × sex × 7 age bands × 3 education levels.
- ESS rounds 9–11 were staged but not used: they have region but no district, and the tests
  below fail on the mechanism, not on sample size, so more respondents would not change the result.
- CSES Module 6 Portugal is the EEP 2022 and 2024 sample again. EES 2024 cannot be redistributed.

## Model

Multinomial logit of vote on sex, age band, education and region, one set of coefficients per
wave tied across waves by a random-walk prior (Gao et al. 2021). A district's survey share is the
model's cell probabilities weighted by the district's census cells. Results are shown for two
settings of the prior; neither changes the conclusion.

## 1. Ceiling: the true demographic swing

Between consecutive EEP waves, the change in the sex, age and education coefficients is applied to
each district's census composition. That gives the district swing that demographic change alone
would produce. Blend swing is multiplied by it (λ = 0.5 or 1), with the true national result.

| District error, pp | 2015 | 2019 | 2022 | 2024 |
|---|---|---|---|---|
| Blend swing | 1.143 | 0.862 | 0.739 | 0.659 |
| + demographic swing (λ 0.5) | 1.146 | 0.840 | 0.724 | 0.662 |
| + regional swing from the survey (λ 0.5) | 1.305 | 0.898 | 0.699 | 0.757 |
| + both (λ 0.5, tight regional prior) | 1.157 | 0.835 | 0.700 | 0.641 |

The age gaps are large: in 2024, Chega had 37–39% among under-35s and 5–10% among over-65s in the
EEP. But districts differ too little in age, sex and education for these gaps to move district
shares. The demographic swing changes a district's share of a party by 1–5% of that share, about
0.1–0.3 pp for Chega, against blend errors of 1–2 pp. The regional swing measured by the survey is
noise: with 50–120 respondents per region in Alentejo and Algarve, it makes things worse unless it
is shrunk almost to nothing.

## 2. Young parties: survey geography against their last offsets

For a party with one or two elections behind it, the district offsets implied by the survey
(MrsP) replace or are averaged with blend swing. Mainland district error for that party, in pp,
fitting on the target election's own survey:

| Case | Respondents | Blend | Uniform | No geography | Survey geography | 50/50 |
|---|---|---|---|---|---|---|
| CH 2022 | 31 | **1.01** | 1.46 | 1.92 | 2.11 | 1.33 |
| CH 2024 | 118 | 1.36 | **1.12** | 2.56 | 3.12 | 1.38 |
| IL 2022 | 16 | **0.72** | 1.31 | 1.88 | 1.41 | 1.03 |
| IL 2024 | 19 | **0.37** | 0.39 | 1.83 | 1.16 | 0.66 |
| L 2024 | 10 | **0.35** | 0.78 | 1.32 | 0.84 | 0.59 |
| PAN 2019 | 15 | **0.33** | 0.60 | 1.16 | 1.04 | 0.63 |

(Demographics only; adding region is worse. With the surveys published before each election,
which is what a forecast would have, the survey geography is worse again: CH 2024 2.92, CH 2022
2.03.) Livre 2019 and 2022 behave like Livre 2024.

The survey geography is barely better than no geography at all, and for Chega it is worse. The
correlation between survey-implied and actual district offsets shows why:

| Party | Correlation, demographics only (2015–2024) | Spread of offsets, actual vs survey (sd, log) |
|---|---|---|
| IL, Livre, PAN, BE | 0.6 to 0.95 | 0.27–0.46 vs 0.05–0.13 |
| PS | 0.1 to 0.3 | 0.09 vs 0.03 |
| AD, CDU, Chega | −0.5 to 0.2 | 0.29–0.72 vs 0.02–0.05 |

For the urban, educated parties the direction is right but the survey sees a quarter of the
spread. For Chega it points the wrong way: its voters are young, but it is strongest in the old
South. Its geography is regional, not compositional, and the last result already holds it.

## 3. Side test: weighting by recalled vote (national, not districts)

Each EEP wave is weighted so that recalled previous vote matches the previous result, and its
reported vote is compared with the mainland result. Big-three mean absolute error, pp:

| | 2011 | 2015 | 2019 | 2022 | 2024 |
|---|---|---|---|---|---|
| Raw | 3.2 | 1.3 | 9.3 | 7.8 | 3.4 |
| Recalled-vote weight | 1.9 | 2.7 | 0.8 | 1.6 | 3.0 |

Recall weighting removes gross sample skews (the 2019 and 2022 waves over-report PS by 13 pp) but
overshoots when nothing is wrong (2015: AD +4.9 after weighting) and in 2024 swaps a PS +5.5 for an
AD +4.0. Published polls are already weighted, so their errors look like the 2015 and 2024 rows,
not the 2019 and 2022 ones. This is a post-election survey, not a poll; the real test needs the
recalled-vote × intention tables in the ERC deposits (Pitagórica's tracker publishes the full
matrix), which the parser work would provide.

## 4. Where district gains could still come from: regional crosstabs

If the five mainland NUTS-II regional results were known exactly, blend swing applied within each
region would cut mainland district error from 0.70 to 0.49 pp and misallocated seats from 7.4 to
5.0 per election (2015–2025). That is the most any regional information can give. With realistic
sampling noise (design effect 2) and shrinkage toward the national swing:

| Pooled respondents with a regional crosstab | Mainland error, pp | Seats misallocated |
|---|---|---|
| None (blend) | 0.695 | 7.4 |
| 5,000 | 0.69 | 7.3 |
| 10,000 | 0.65 | 7.0 |
| 20,000 | 0.64 | 6.6 |
| 40,000 | 0.60 | 6.0 |
| Exact | 0.49 | 5.0 |

This ignores pollster regional house effects and mismatched region definitions, so it is
optimistic. A final month gives perhaps 5,000–10,000 respondents with regional tables (Intercampus,
Aximage), worth about half a seat at best. Exit polls have far larger samples but come too late.

## Recommendation

1. Keep blend swing. Do not build a survey-based district model.
2. The parameters that matter are regional and historical, which is ecological territory: a district
   or parish regression on past results and covariates (education share tracks IL, Livre, PAN and
   BE; the North/South axis tracks Chega, AD and CDU) has thousands of parishes of real votes,
   where the best survey has 500–800 respondents.
3. If the ERC parser extracts regional crosstabs anyway, add them as noisy regional observations
   with strong shrinkage; expect half a seat at most.
4. Recalled-vote weighting of polls is worth one test once recall × intention tables are
   extracted, but the EEP evidence says it can overshoot when the poll is already balanced.
5. Licence: the EEP is for non-commercial research only, so any published output derived from it
   would need ICS's permission. With these results there is nothing to publish.
