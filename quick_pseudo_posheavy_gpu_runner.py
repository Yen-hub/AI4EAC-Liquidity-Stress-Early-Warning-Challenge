from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from train_catboost_pseudolabel import logit_average, select_pseudolabels
from train_competition_pipeline import build_submission, clip_probs, prepare_datasets


def main() -> None:
    base = Path(__file__).resolve().parent
    artifacts = base / "artifacts_cb_posheavy_gpu"
    artifacts.mkdir(parents=True, exist_ok=True)

    X, y, X_test, test_ids, cat_cols = prepare_datasets()
    submission2 = clip_probs(pd.read_csv(base / "submission2.csv")["TargetLogLoss"].to_numpy())
    aggressive = clip_probs(
        pd.read_csv(base / "artifacts" / "submissions" / "submission_aggressive_auc_blend.csv")["TargetLogLoss"].to_numpy()
    )
    winner = clip_probs(
        pd.read_csv(base / "artifacts_cb_quick_gpu" / "submission_bestpublic_quickgpu_logit50.csv")["TargetLogLoss"].to_numpy()
    )

    consensus = 0.5 * (submission2 + aggressive)
    disagreement = np.abs(submission2 - aggressive)

    pseudo_idx, pseudo_labels, pseudo_weights, pseudo_summary = select_pseudolabels(
        consensus=consensus,
        disagreement=disagreement,
        pos_count=420,
        neg_count=100,
        pos_pool=1000,
        neg_pool=500,
        pos_diff_max=0.07,
        neg_diff_max=0.008,
        pos_weight=0.12,
        neg_weight=0.08,
    )
    print(json.dumps({"pseudo_summary": pseudo_summary}, indent=2), flush=True)

    X_aug = pd.concat([X, X_test.iloc[pseudo_idx].copy().reset_index(drop=True)], axis=0, ignore_index=True)
    y_aug = pd.concat([y, pd.Series(pseudo_labels, name=y.name)], axis=0, ignore_index=True)
    sample_weight = np.ones(len(X_aug), dtype=float)
    sample_weight[len(X) :] = pseudo_weights

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=1000,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=6.0,
        bagging_temperature=0.5,
        random_seed=2026,
        allow_writing_files=False,
        verbose=False,
        task_type="GPU",
        devices="0",
    )
    print("Training positive-heavy pseudo CatBoost on GPU", flush=True)
    model.fit(X_aug, y_aug, sample_weight=sample_weight, cat_features=cat_cols, verbose=False)

    test_preds = clip_probs(model.predict_proba(X_test)[:, 1])
    blend_50 = logit_average({"winner": winner, "posheavy": test_preds}, {"winner": 0.5, "posheavy": 0.5})
    blend_65 = logit_average({"winner": winner, "posheavy": test_preds}, {"winner": 0.65, "posheavy": 0.35})

    build_submission(test_ids, test_preds).to_csv(
        artifacts / "submission_posheavy_gpu_pseudo_catboost.csv",
        index=False,
    )
    build_submission(test_ids, blend_50).to_csv(
        artifacts / "submission_winner_posheavy_logit50.csv",
        index=False,
    )
    build_submission(test_ids, blend_65).to_csv(
        artifacts / "submission_winner_posheavy_logit65_35.csv",
        index=False,
    )

    summary = {
        "pseudo_summary": pseudo_summary,
        "diversity": {
            "corr_with_winner": float(np.corrcoef(winner, test_preds)[0, 1]),
            "mad_with_winner": float(np.mean(np.abs(winner - test_preds))),
            "mean_posheavy": float(test_preds.mean()),
            "mean_winner": float(winner.mean()),
        },
        "submissions": [
            str(artifacts / "submission_posheavy_gpu_pseudo_catboost.csv"),
            str(artifacts / "submission_winner_posheavy_logit50.csv"),
            str(artifacts / "submission_winner_posheavy_logit65_35.csv"),
        ],
    }
    (artifacts / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
