# AI4EAC Liquidity Stress Early Warning Challenge

This repository contains our working solution package for the **AI4EAC Liquidity Stress Early Warning Challenge** on Zindi. The goal is to predict the probability that a customer will experience liquidity stress within 30 days of the observation date.

The competition uses a multi-metric evaluation setup:

- `Log Loss` weighted at `60%`
- `ROC-AUC` weighted at `40%`

That scoring mix makes probability quality just as important as ranking quality. Our work therefore focused on two things:

1. Building stronger tabular models with reproducible feature engineering and cross-validation.
2. Improving probability quality and ranking through controlled blending, pseudo-labeling, and late-stage calibration/rank refinement.

## Repository Highlights

- Reproducible training pipeline for `XGBoost`, `LightGBM`, and `CatBoost`
- Temporal-style feature engineering from the `m1` to `m6` monthly history blocks
- CatBoost pseudo-label pipelines for fast and positive-heavy GPU variants
- Experimental orthogonal rankers and stacking probes
- Final tracked submission files for the strongest public-board candidates

## Best Submission Candidates

These are the two finalists we would keep for private leaderboard hedging:

| File | Role | Notes |
| --- | --- | --- |
| `final_submissions/submission_rankblend_cur_sub2_pos_65_20_15.csv` | Highest public-upside candidate | Best observed public leaderboard result during the sprint |
| `final_submissions/submission_winner_posheavy_submission2_logit65_25_10_bias015.csv` | Safer private-board hedge | Stronger calibration discipline and lower overfitting risk |

Observed public leaderboard notes from the sprint:

- `submission_rankblend_cur_sub2_pos_65_20_15.csv`
  - Public score: `0.706513722`
  - `Target RAUC`: `0.903993706`
  - `Target Log Loss`: `0.252983981`
- `submission_winner_posheavy_submission2_logit65_25_10_bias015.csv`
  - Public score: `0.706035355`
  - `Target RAUC`: `0.903704478`
  - `Target Log Loss`: `0.253343672`

## Solution Overview

### 1. Baseline Ensemble Pipeline

The main reproducible pipeline lives in `train_competition_pipeline.py`. It:

- engineers aggregate, trend, ratio, and interaction features from the `m1` to `m6` history
- trains `XGBoost`, `LightGBM`, and `CatBoost`
- evaluates with repeated stratified CV
- optimizes a local proxy of the competition metric
- exports safe and aggressive submission blends

Best local OOF result from this stage:

- `safe_blend`: `logloss=0.258784`, `auc=0.899022`, `weighted_proxy=0.804338`

### 2. Pseudo-Labeling With CatBoost

We then used submission agreement to select high-confidence pseudo-labels from test data and retrained CatBoost variants:

- `quick_pseudo_runner.py`: fast CPU probe
- `quick_pseudo_gpu_runner.py`: fast GPU pseudo-label model
- `train_catboost_pseudolabel.py`: larger balanced pseudo-label ensemble
- `quick_pseudo_posheavy_gpu_runner.py`: positive-heavy GPU pseudo-label variant

The positive-heavy GPU run gave a useful new direction because it changed both prediction shape and ranking enough to create better hybrid submissions.

### 3. Final Rank/Calibration Refinement

The late-stage finalists came from combining three sources:

- the best public-facing blend at that point
- `submission2.csv`
- the positive-heavy pseudo-label CatBoost variant

Two refinement families were the most useful:

- **logit-space calibration blends** for safer probability control
- **rank-based blends** to improve ordering while preserving a strong anchor distribution

The best public result came from the rank-blend family, while the best private-board hedge came from the calibrated logit-blend family.

## Project Structure

| Path | Description |
| --- | --- |
| `Main.ipynb` | Original working notebook |
| `train_competition_pipeline.py` | Main reproducible training pipeline |
| `train_catboost_pseudolabel.py` | Balanced pseudo-label CatBoost ensemble |
| `quick_pseudo_runner.py` | Fast CPU pseudo-label probe |
| `quick_pseudo_gpu_runner.py` | Fast GPU pseudo-label probe |
| `quick_pseudo_posheavy_gpu_runner.py` | Positive-heavy GPU pseudo-label training |
| `train_new_rankers.py` | Orthogonal ranking-source experiments |
| `train_tabicl_stack.py` | Experimental TabICL stacker |
| `docs/SOLUTION_REPORT.md` | Professional write-up of the experiment progression |
| `final_submissions/` | Final tracked submission files |

## Environment

The scripts were developed in Python and depend on the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `catboost`
- `lightgbm`
- `xgboost`
- `torch`
- `tabicl`

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running The Pipelines

Baseline ensemble:

```bash
python train_competition_pipeline.py
```

Balanced pseudo-label CatBoost ensemble:

```bash
python train_catboost_pseudolabel.py
```

Fast GPU pseudo-label probe:

```bash
python quick_pseudo_gpu_runner.py
```

Positive-heavy GPU pseudo-label probe:

```bash
python quick_pseudo_posheavy_gpu_runner.py
```

Orthogonal ranking-source experiments:

```bash
python train_new_rankers.py
```

## Notes

- The repository keeps the original challenge data and notebook layout intact.
- Large experiment artifact folders are intentionally ignored to keep Git history manageable.
- The tracked final submission files are the handoff-ready deliverables from the sprint.

