from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from train_catboost_pseudolabel import load_teacher_predictions, logit_average, select_pseudolabels
from train_competition_pipeline import build_submission, clip_probs, prepare_datasets


def main() -> None:
    base = Path(__file__).resolve().parent
    artifacts = base / "artifacts_cb_quick_gpu"
    artifacts.mkdir(parents=True, exist_ok=True)

    X, y, X_test, test_ids, cat_cols = prepare_datasets()
    teachers = load_teacher_predictions()
    consensus = 0.5 * (teachers["submission2"] + teachers["aggressive"])
    disagreement = np.abs(teachers["submission2"] - teachers["aggressive"])
    pseudo_idx, pseudo_labels, pseudo_weights, pseudo_summary = select_pseudolabels(
        consensus=consensus,
        disagreement=disagreement,
        pos_count=280,
        neg_count=280,
        pos_pool=800,
        neg_pool=1200,
        pos_diff_max=0.05,
        neg_diff_max=0.01,
        pos_weight=0.08,
        neg_weight=0.18,
    )
    print(json.dumps({"pseudo_summary": pseudo_summary}, indent=2), flush=True)

    X_aug = pd.concat([X, X_test.iloc[pseudo_idx].copy().reset_index(drop=True)], axis=0, ignore_index=True)
    y_aug = pd.concat([y, pd.Series(pseudo_labels, name=y.name)], axis=0, ignore_index=True)
    weights = np.ones(len(X_aug), dtype=float)
    weights[len(X) :] = pseudo_weights

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=900,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=5.0,
        bagging_temperature=0.0,
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
        task_type="GPU",
        devices="0",
    )
    print("Training quick pseudo CatBoost on GPU", flush=True)
    model.fit(X_aug, y_aug, sample_weight=weights, cat_features=cat_cols, verbose=False)

    test_preds = clip_probs(model.predict_proba(X_test)[:, 1])
    best_public = teachers["best_public"]
    submission2 = teachers["submission2"]
    best_public_blend = logit_average(
        {"best_public": best_public, "quick_gpu": test_preds},
        {"best_public": 0.5, "quick_gpu": 0.5},
    )
    submission2_blend = logit_average(
        {"submission2": submission2, "quick_gpu": test_preds},
        {"submission2": 0.5, "quick_gpu": 0.5},
    )

    build_submission(test_ids, test_preds).to_csv(artifacts / "submission_quick_gpu_pseudo_catboost.csv", index=False)
    build_submission(test_ids, best_public_blend).to_csv(
        artifacts / "submission_bestpublic_quickgpu_logit50.csv",
        index=False,
    )
    build_submission(test_ids, submission2_blend).to_csv(
        artifacts / "submission_submission2_quickgpu_logit50.csv",
        index=False,
    )

    summary = {
        "pseudo_summary": pseudo_summary,
        "corr_with_best_public": float(np.corrcoef(best_public, test_preds)[0, 1]),
        "mad_with_best_public": float(np.mean(np.abs(best_public - test_preds))),
        "corr_with_submission2": float(np.corrcoef(submission2, test_preds)[0, 1]),
        "mad_with_submission2": float(np.mean(np.abs(submission2 - test_preds))),
        "submissions": [
            str(artifacts / "submission_quick_gpu_pseudo_catboost.csv"),
            str(artifacts / "submission_bestpublic_quickgpu_logit50.csv"),
            str(artifacts / "submission_submission2_quickgpu_logit50.csv"),
        ],
    }
    (artifacts / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
