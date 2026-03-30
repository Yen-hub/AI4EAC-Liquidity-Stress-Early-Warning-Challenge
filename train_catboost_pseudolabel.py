from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

from train_competition_pipeline import build_submission, clip_probs, competition_metrics, prepare_datasets

DATA_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = DATA_DIR / "artifacts_cb_pseudo"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"

TARGET = "liquidity_stress_next_30d"
RANDOM_STATE = 42

BEST_PUBLIC_PATH = DATA_DIR / "artifacts" / "submissions" / "submission_blend_s2_50_agg_50.csv"
SUBMISSION2_PATH = DATA_DIR / "submission2.csv"
AGGRESSIVE_PATH = DATA_DIR / "artifacts" / "submissions" / "submission_aggressive_auc_blend.csv"

MODEL_CONFIGS = [
    {"name": "cb_d6", "depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5.0, "bagging_temperature": 0.0},
    {"name": "cb_d7", "depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 7.0, "bagging_temperature": 0.5},
    {"name": "cb_d8", "depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 10.0, "bagging_temperature": 1.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CatBoost seed ensemble with agreement pseudo-labels.")
    parser.add_argument("--artifacts-dir", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1400)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--pos-count", type=int, default=280)
    parser.add_argument("--neg-count", type=int, default=280)
    parser.add_argument("--pos-pool", type=int, default=800)
    parser.add_argument("--neg-pool", type=int, default=1200)
    parser.add_argument("--pos-diff-max", type=float, default=0.05)
    parser.add_argument("--neg-diff-max", type=float, default=0.01)
    parser.add_argument("--pos-weight", type=float, default=0.08)
    parser.add_argument("--neg-weight", type=float, default=0.18)
    return parser.parse_args()


def logit_average(pred_map: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = np.zeros(len(next(iter(pred_map.values()))), dtype=float)
    for name, preds in pred_map.items():
        probs = clip_probs(preds)
        total += weights[name] * np.log(probs / (1.0 - probs))
    return clip_probs(1.0 / (1.0 + np.exp(-total)))


def load_teacher_predictions() -> dict[str, np.ndarray]:
    return {
        "best_public": clip_probs(pd.read_csv(BEST_PUBLIC_PATH)["TargetLogLoss"].to_numpy()),
        "submission2": clip_probs(pd.read_csv(SUBMISSION2_PATH)["TargetLogLoss"].to_numpy()),
        "aggressive": clip_probs(pd.read_csv(AGGRESSIVE_PATH)["TargetLogLoss"].to_numpy()),
    }


def select_pseudolabels(
    consensus: np.ndarray,
    disagreement: np.ndarray,
    *,
    pos_count: int,
    neg_count: int,
    pos_pool: int,
    neg_pool: int,
    pos_diff_max: float,
    neg_diff_max: float,
    pos_weight: float,
    neg_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    order = np.argsort(consensus)

    low_pool_idx = order[:neg_pool]
    low_keep = low_pool_idx[disagreement[low_pool_idx] <= neg_diff_max]
    if len(low_keep) < neg_count:
        ranked = low_pool_idx[np.argsort(disagreement[low_pool_idx])]
        low_keep = ranked[:neg_count]
    else:
        low_keep = low_keep[:neg_count]

    high_pool_idx = order[-pos_pool:][::-1]
    high_keep = high_pool_idx[disagreement[high_pool_idx] <= pos_diff_max]
    if len(high_keep) < pos_count:
        ranked = high_pool_idx[np.argsort(disagreement[high_pool_idx])]
        high_keep = ranked[:pos_count]
    else:
        high_keep = high_keep[:pos_count]

    pseudo_idx = np.concatenate([low_keep, high_keep])
    pseudo_labels = np.concatenate(
        [np.zeros(len(low_keep), dtype=int), np.ones(len(high_keep), dtype=int)]
    )

    low_weights = neg_weight * np.clip(1.0 - disagreement[low_keep] / max(neg_diff_max, 1e-6), 0.35, 1.0)
    high_weights = pos_weight * np.clip(1.0 - disagreement[high_keep] / max(pos_diff_max, 1e-6), 0.35, 1.0)
    pseudo_weights = np.concatenate([low_weights, high_weights])

    summary = {
        "selected_total": int(len(pseudo_idx)),
        "selected_negatives": int(len(low_keep)),
        "selected_positives": int(len(high_keep)),
        "negative_prob_max": float(consensus[low_keep].max()) if len(low_keep) else None,
        "positive_prob_min": float(consensus[high_keep].min()) if len(high_keep) else None,
        "negative_disagreement_max": float(disagreement[low_keep].max()) if len(low_keep) else None,
        "positive_disagreement_max": float(disagreement[high_keep].max()) if len(high_keep) else None,
    }
    return pseudo_idx, pseudo_labels, pseudo_weights, summary


def build_catboost(config: dict[str, float], seed: int, iterations: int, threads: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=iterations,
        learning_rate=config["learning_rate"],
        depth=int(config["depth"]),
        l2_leaf_reg=config["l2_leaf_reg"],
        bagging_temperature=config["bagging_temperature"],
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=threads,
    )


def train_cv_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    *,
    pseudo_idx: np.ndarray,
    pseudo_labels: np.ndarray,
    pseudo_weights: np.ndarray,
    splits: int,
    iterations: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X), dtype=float)
    oof_counts = np.zeros(len(X), dtype=float)
    test_preds_accum = []

    pseudo_frame = X_test.iloc[pseudo_idx].copy().reset_index(drop=True)
    pseudo_target = pd.Series(pseudo_labels, name=TARGET)

    for model_offset, config in enumerate(MODEL_CONFIGS):
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            print(
                f"Training {config['name']} fold {fold}/{splits} "
                f"with {len(pseudo_idx)} pseudo rows",
                flush=True,
            )
            X_train = pd.concat([X.iloc[train_idx].copy(), pseudo_frame], axis=0, ignore_index=True)
            y_train = pd.concat([y.iloc[train_idx].copy(), pseudo_target], axis=0, ignore_index=True)
            X_valid = X.iloc[valid_idx].copy()
            y_valid = y.iloc[valid_idx].copy()

            sample_weight = np.ones(len(X_train), dtype=float)
            sample_weight[len(train_idx) :] = pseudo_weights

            model = build_catboost(
                config=config,
                seed=RANDOM_STATE + model_offset * 100 + fold,
                iterations=iterations,
                threads=threads,
            )
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=(X_valid, y_valid),
                cat_features=cat_cols,
                use_best_model=True,
                early_stopping_rounds=120,
                verbose=False,
            )

            oof_preds[valid_idx] += model.predict_proba(X_valid)[:, 1]
            oof_counts[valid_idx] += 1.0
            test_preds_accum.append(model.predict_proba(X_test)[:, 1])

    oof_preds = oof_preds / np.maximum(oof_counts, 1.0)
    test_preds = np.mean(test_preds_accum, axis=0)
    return clip_probs(oof_preds), clip_probs(test_preds), competition_metrics(y, oof_preds)


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    predictions_dir = artifacts_dir / "predictions"
    submissions_dir = artifacts_dir / "submissions"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    X, y, X_test, test_ids, cat_cols = prepare_datasets()
    teachers = load_teacher_predictions()
    consensus = 0.5 * (teachers["submission2"] + teachers["aggressive"])
    disagreement = np.abs(teachers["submission2"] - teachers["aggressive"])

    pseudo_idx, pseudo_labels, pseudo_weights, pseudo_summary = select_pseudolabels(
        consensus=consensus,
        disagreement=disagreement,
        pos_count=args.pos_count,
        neg_count=args.neg_count,
        pos_pool=args.pos_pool,
        neg_pool=args.neg_pool,
        pos_diff_max=args.pos_diff_max,
        neg_diff_max=args.neg_diff_max,
        pos_weight=args.pos_weight,
        neg_weight=args.neg_weight,
    )
    print(json.dumps({"pseudo_summary": pseudo_summary}, indent=2), flush=True)

    pseudo_oof, pseudo_test, pseudo_metrics = train_cv_ensemble(
        X=X,
        y=y,
        X_test=X_test,
        cat_cols=cat_cols,
        pseudo_idx=pseudo_idx,
        pseudo_labels=pseudo_labels,
        pseudo_weights=pseudo_weights,
        splits=args.splits,
        iterations=args.iterations,
        threads=args.threads,
    )
    print(
        "pseudo_catboost: "
        f"logloss={pseudo_metrics['logloss']:.6f} "
        f"auc={pseudo_metrics['auc']:.6f} "
        f"weighted_proxy={pseudo_metrics['weighted_proxy']:.6f}",
        flush=True,
    )

    best_public_test = teachers["best_public"]
    best_public_hybrid = logit_average(
        {"best_public": best_public_test, "pseudo_catboost": pseudo_test},
        {"best_public": 0.5, "pseudo_catboost": 0.5},
    )

    corr_with_best_public = float(np.corrcoef(best_public_test, pseudo_test)[0, 1])
    mad_with_best_public = float(np.mean(np.abs(best_public_test - pseudo_test)))

    metrics_df = pd.DataFrame(
        [
            {"model": "pseudo_catboost", **pseudo_metrics},
        ]
    )
    metrics_df.to_csv(artifacts_dir / "model_metrics.csv", index=False)

    pd.DataFrame(
        {
            "pseudo_catboost": pseudo_oof,
            TARGET: y.to_numpy(),
        }
    ).to_csv(predictions_dir / "oof_predictions.csv", index=False)

    pd.DataFrame(
        {
            "ID": test_ids,
            "best_public": best_public_test,
            "pseudo_catboost": pseudo_test,
            "best_public_pseudo_logit50": best_public_hybrid,
        }
    ).to_csv(predictions_dir / "test_predictions.csv", index=False)

    build_submission(test_ids, pseudo_test).to_csv(
        submissions_dir / "submission_pseudo_catboost_agreement.csv",
        index=False,
    )
    build_submission(test_ids, best_public_hybrid).to_csv(
        submissions_dir / "submission_bestpublic_pseudo_logit50.csv",
        index=False,
    )

    summary = {
        "config": vars(args),
        "pseudo_summary": pseudo_summary,
        "pseudo_metrics": pseudo_metrics,
        "diversity": {
            "corr_with_best_public": corr_with_best_public,
            "mad_with_best_public": mad_with_best_public,
        },
        "submissions": [
            str(submissions_dir / "submission_pseudo_catboost_agreement.csv"),
            str(submissions_dir / "submission_bestpublic_pseudo_logit50.csv"),
        ],
    }
    (artifacts_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
