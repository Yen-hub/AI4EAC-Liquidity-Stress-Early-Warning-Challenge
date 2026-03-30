# Solution Report

## Objective

Predict the probability that a customer will experience liquidity stress within 30 days after the observation date.

Competition metric:

- `60%` Log Loss
- `40%` ROC-AUC

This meant the solution needed to balance **probability calibration** and **ranking strength**. A purely AUC-driven model was not enough, and a highly calibrated but weak ranker was also not enough.

## Dataset Framing

The training data provides monthly behavioral history (`m1` to `m6`) along with customer-level fields such as age, ARPU, and activity indicators. No explicit observation timestamp column was available in the released files, so the approach relied on feature engineering over the six historical monthly blocks instead of true time-based folds.

## Experiment Progression

### Stage 1. Reproducible GBDT Ensemble

Implemented in `train_competition_pipeline.py`.

Key ideas:

- generate six-month aggregates, ranges, slopes, recent-vs-history ratios, and activity counts
- derive inflow/outflow interaction features
- train `XGBoost`, `LightGBM`, and `CatBoost`
- compare safe vs aggressive logit blends

Local repeated-CV summary:

| Model | Log Loss | AUC | Weighted Proxy |
| --- | ---: | ---: | ---: |
| Safe blend | 0.258784 | 0.899022 | 0.804338 |
| Aggressive blend | 0.258824 | 0.899054 | 0.804327 |
| CatBoost | 0.259016 | 0.898543 | 0.804007 |

Takeaway:

- `CatBoost` was the backbone.
- `LightGBM` added small but consistent lift.
- `XGBoost` added little marginal value locally.

### Stage 2. Pseudo-Label CatBoost Family

Implemented in:

- `train_catboost_pseudolabel.py`
- `quick_pseudo_runner.py`
- `quick_pseudo_gpu_runner.py`
- `quick_pseudo_posheavy_gpu_runner.py`

Key ideas:

- use agreement between strong existing submissions to select confident pseudo-labels
- re-train CatBoost with weighted pseudo rows
- test both balanced pseudo-labeling and positive-heavy pseudo-labeling

Important observation:

- the balanced pseudo-label family was more stable than strong on the public board
- the positive-heavy GPU variant added more useful diversity and created a better basis for final hybrids

### Stage 3. Blend Refinement

This stage combined:

- the best public-facing hybrid at that time
- `submission2.csv`
- the positive-heavy pseudo-label CatBoost model

Two refinement families were tested:

1. **Logit-space calibration blends**
2. **Rank-based blends**

The calibration family improved Log Loss in tiny increments and likely reduced private-board risk.

The rank-based family produced the biggest public jump:

- `submission_rankblend_cur_sub2_pos_65_20_15.csv`
  - Public score: `0.706513722`
  - `Target RAUC`: `0.903993706`
  - `Target Log Loss`: `0.252983981`

The strongest calibration-stable alternative was:

- `submission_winner_posheavy_submission2_logit65_25_10_bias015.csv`
  - Public score: `0.706035355`
  - `Target RAUC`: `0.903704478`
  - `Target Log Loss`: `0.253343672`

## Final Recommendation

For a two-submission private-leaderboard hedge, the recommended pair is:

1. `final_submissions/submission_rankblend_cur_sub2_pos_65_20_15.csv`
2. `final_submissions/submission_winner_posheavy_submission2_logit65_25_10_bias015.csv`

Why this pair:

- the first file was the highest public performer
- the second file is a more disciplined calibration-oriented hedge
- together they cover both upside and generalization risk better than two near-duplicates

## What Did Not Make The Final Cut

These were useful explorations, but they were not the final handoff:

- `TabICL` stacking
  - interesting, but operationally expensive and not successful enough in the available time window
- orthogonal anchor rankers from `train_new_rankers.py`
  - gave diversity, but the strongest public test still dropped materially
- endless micro-calibration sweeps
  - improved only in the fifth decimal and were not enough on their own

## Generalization View

The best public scorer is not automatically the safest private-board choice.

Our internal read at handoff time:

- `submission_rankblend_cur_sub2_pos_65_20_15.csv`
  - higher upside
  - more public-board fitting risk because it came from explicit rank refinement
- `submission_winner_posheavy_submission2_logit65_25_10_bias015.csv`
  - lower upside
  - stronger probability-discipline profile and likely better private-board safety

## Next Steps

If this work is extended after the sprint, the highest-value unexplored directions are:

- a new orthogonal base learner such as `TabM`
- a deep tabular family such as `GANDALF`, `GATE`, `FT-Transformer`, or `ExcelFormer`
- principled OOF-only calibration such as beta calibration or Venn-Abers on the final stack

Those directions were identified, but not fully exploited before the submission window closed.
