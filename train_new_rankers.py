from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_competition_pipeline import prepare_datasets

DATA_DIR = Path(__file__).resolve().parent
BEST_SUBMISSION_PATH = DATA_DIR / "artifacts_cb_posheavy_gpu" / "submission_rankblend_cur_sub2_pos_65_20_15.csv"
TARGET = "liquidity_stress_next_30d"
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train new ranking sources and create anchored rank-blend submissions.")
    parser.add_argument("--artifacts-dir", type=str, default=str(DATA_DIR / "artifacts_new_rankers"))
    parser.add_argument("--splits", type=int, default=3)
    return parser.parse_args()


def clip_probs(preds: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(preds, dtype=float), EPS, 1.0 - EPS)


def competition_metrics(y_true: pd.Series | np.ndarray, preds: np.ndarray) -> dict[str, float]:
    probs = clip_probs(preds)
    loss = log_loss(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    weighted_proxy = 0.6 * (1.0 - loss) + 0.4 * auc
    return {
        "logloss": float(loss),
        "auc": float(auc),
        "weighted_proxy": float(weighted_proxy),
    }


def build_submission(ids: pd.Series, preds: np.ndarray) -> pd.DataFrame:
    probs = clip_probs(preds)
    return pd.DataFrame({"ID": ids, "TargetLogLoss": probs, "TargetRAUC": probs})


def normalized_ranks(arr: np.ndarray) -> np.ndarray:
    return pd.Series(arr).rank(method="average", pct=True).to_numpy()


def rank_blend(anchor_probs: np.ndarray, components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    rank_score = np.zeros(len(anchor_probs), dtype=float)
    for name, preds in components.items():
        rank_score += weights[name] * normalized_ranks(preds)
    order = np.argsort(rank_score)
    out = np.empty_like(anchor_probs)
    out[order] = np.sort(anchor_probs)
    return clip_probs(out)


def encode_for_trees(frame: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    encoded = frame.copy()
    for col in cat_cols:
        encoded[col] = encoded[col].cat.codes.astype("int32")
    return encoded


def train_catboost_auc(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X), dtype=float)
    test_preds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"CatBoost AUC fold {fold}/{n_splits}", flush=True)
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=700,
            learning_rate=0.04,
            depth=5,
            l2_leaf_reg=4.0,
            random_strength=0.5,
            bagging_temperature=1.0,
            auto_class_weights="Balanced",
            max_ctr_complexity=1,
            border_count=64,
            one_hot_max_size=2,
            gpu_ram_part=0.8,
            random_seed=42 + fold,
            allow_writing_files=False,
            verbose=False,
            task_type="GPU",
            devices="0",
        )
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_cols,
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False,
        )
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_preds.append(model.predict_proba(X_test)[:, 1])

    test = np.mean(test_preds, axis=0)
    return clip_probs(oof), clip_probs(test), competition_metrics(y, oof)


def train_extra_trees(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
    oof = np.zeros(len(X), dtype=float)
    test_preds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"ExtraTrees fold {fold}/{n_splits}", flush=True)
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()

        model = ExtraTreesClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=100 + fold,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        oof[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_preds.append(model.predict_proba(X_test)[:, 1])

    test = np.mean(test_preds, axis=0)
    return clip_probs(oof), clip_probs(test), competition_metrics(y, oof)


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    submissions_dir = artifacts_dir / "submissions"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    X, y, X_test, ids, cat_cols = prepare_datasets()
    best_submission = pd.read_csv(BEST_SUBMISSION_PATH)
    anchor_test = clip_probs(best_submission["TargetLogLoss"].to_numpy())

    X_tree = encode_for_trees(X, cat_cols)
    X_test_tree = encode_for_trees(X_test, cat_cols)

    cat_auc_oof, cat_auc_test, cat_auc_metrics = train_catboost_auc(X, y, X_test, cat_cols, args.splits)
    et_oof, et_test, et_metrics = train_extra_trees(X_tree, y, X_test_tree, args.splits)

    corr_report = {
        "cat_auc_vs_anchor": float(np.corrcoef(anchor_test, cat_auc_test)[0, 1]),
        "et_vs_anchor": float(np.corrcoef(anchor_test, et_test)[0, 1]),
        "cat_auc_vs_et": float(np.corrcoef(cat_auc_test, et_test)[0, 1]),
    }

    candidates = {
        "submission_rankblend_anchor_catauc_80_20.csv": rank_blend(
            anchor_test,
            {"anchor": anchor_test, "cat_auc": cat_auc_test},
            {"anchor": 0.80, "cat_auc": 0.20},
        ),
        "submission_rankblend_anchor_catauc_et_70_20_10.csv": rank_blend(
            anchor_test,
            {"anchor": anchor_test, "cat_auc": cat_auc_test, "et": et_test},
            {"anchor": 0.70, "cat_auc": 0.20, "et": 0.10},
        ),
        "submission_rankblend_anchor_catauc_et_60_25_15.csv": rank_blend(
            anchor_test,
            {"anchor": anchor_test, "cat_auc": cat_auc_test, "et": et_test},
            {"anchor": 0.60, "cat_auc": 0.25, "et": 0.15},
        ),
    }

    for name, preds in candidates.items():
        build_submission(ids, preds).to_csv(submissions_dir / name, index=False)

    pd.DataFrame(
        [
            {"model": "catboost_auc", **cat_auc_metrics},
            {"model": "extra_trees", **et_metrics},
        ]
    ).to_csv(artifacts_dir / "model_metrics.csv", index=False)

    pd.DataFrame(
        {
            "catboost_auc": cat_auc_oof,
            "extra_trees": et_oof,
            TARGET: y.to_numpy(),
        }
    ).to_csv(artifacts_dir / "oof_predictions.csv", index=False)

    pd.DataFrame(
        {
            "ID": ids,
            "anchor_best": anchor_test,
            "catboost_auc": cat_auc_test,
            "extra_trees": et_test,
        }
    ).to_csv(artifacts_dir / "test_predictions.csv", index=False)

    summary = {
        "config": vars(args),
        "catboost_auc_metrics": cat_auc_metrics,
        "extra_trees_metrics": et_metrics,
        "correlations": corr_report,
        "submissions": [str(submissions_dir / name) for name in candidates],
    }
    (artifacts_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
