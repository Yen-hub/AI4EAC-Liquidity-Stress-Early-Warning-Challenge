from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_catboost_pseudolabel import MODEL_CONFIGS, build_catboost, load_teacher_predictions, logit_average, select_pseudolabels
from train_competition_pipeline import build_submission, clip_probs, prepare_datasets


def main() -> None:
    base = Path(__file__).resolve().parent
    artifacts = base / "artifacts_cb_quick"
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

    config = MODEL_CONFIGS[0]
    X_aug = pd.concat([X, X_test.iloc[pseudo_idx].copy().reset_index(drop=True)], axis=0, ignore_index=True)
    y_aug = pd.concat([y, pd.Series(pseudo_labels, name=y.name)], axis=0, ignore_index=True)
    weights = np.ones(len(X_aug), dtype=float)
    weights[len(X) :] = pseudo_weights

    model = build_catboost(config, seed=42, iterations=1400, threads=6)
    print("Training quick pseudo CatBoost", flush=True)
    model.fit(X_aug, y_aug, sample_weight=weights, cat_features=cat_cols, verbose=False)

    test_preds = clip_probs(model.predict_proba(X_test)[:, 1])
    best_public = teachers["best_public"]
    best_public_blend = logit_average(
        {"best_public": best_public, "quick_pseudo": test_preds},
        {"best_public": 0.5, "quick_pseudo": 0.5},
    )

    build_submission(test_ids, test_preds).to_csv(artifacts / "submission_quick_pseudo_catboost.csv", index=False)
    build_submission(test_ids, best_public_blend).to_csv(
        artifacts / "submission_bestpublic_quick_pseudo_logit50.csv",
        index=False,
    )

    summary = {
        "pseudo_summary": pseudo_summary,
        "corr_with_best_public": float(np.corrcoef(best_public, test_preds)[0, 1]),
        "mad_with_best_public": float(np.mean(np.abs(best_public - test_preds))),
        "submissions": [
            str(artifacts / "submission_quick_pseudo_catboost.csv"),
            str(artifacts / "submission_bestpublic_quick_pseudo_logit50.csv"),
        ],
    }
    (artifacts / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
