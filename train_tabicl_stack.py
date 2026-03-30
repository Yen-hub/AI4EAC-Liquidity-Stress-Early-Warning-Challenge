from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabicl import TabICLClassifier
from catboost import CatBoostClassifier

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "Test.csv"
BASE_PREDICTIONS_DIR = DATA_DIR / "artifacts" / "predictions"
DEFAULT_ARTIFACTS_DIR = DATA_DIR / "artifacts_tabicl"

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
RANDOM_STATE = 42
EPS = 1e-6


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


def logit_average(pred_map: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = np.zeros(len(next(iter(pred_map.values()))), dtype=float)
    for name, preds in pred_map.items():
        probs = clip_probs(preds)
        total += weights[name] * np.log(probs / (1.0 - probs))
    return clip_probs(1.0 / (1.0 + np.exp(-total)))


def load_raw_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X = train.drop(columns=[TARGET, ID_COL]).copy()
    y = train[TARGET].copy().reset_index(drop=True)
    X_test = test.drop(columns=[ID_COL]).copy()
    ids = test[ID_COL].copy().reset_index(drop=True)

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X[col] = X[col].fillna("Unknown").astype(str)
        X_test[col] = X_test[col].fillna("Unknown").astype(str)

    return X.reset_index(drop=True), y, X_test.reset_index(drop=True), ids


def load_base_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_csv(BASE_PREDICTIONS_DIR / "oof_predictions.csv")
    test = pd.read_csv(BASE_PREDICTIONS_DIR / "test_predictions.csv")
    return oof, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TabICLv2 stacking experiments.")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--valid-chunk-size", type=int, default=2000)
    parser.add_argument("--test-chunk-size", type=int, default=1500)
    parser.add_argument("--meta-catboost-iterations", type=int, default=400)
    parser.add_argument("--offload-mode", type=str, default="auto")
    parser.add_argument("--skip-pseudolabel", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def build_tabicl(
    seed: int,
    n_estimators: int,
    batch_size: int,
    offload_mode: str,
    offload_dir: Path,
) -> TabICLClassifier:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabICLClassifier(
        n_estimators=n_estimators,
        softmax_temperature=0.9,
        average_logits=True,
        random_state=seed,
        verbose=False,
        device=device,
        kv_cache=False,
        batch_size=batch_size,
        offload_mode=offload_mode,
        disk_offload_dir=str(offload_dir) if offload_mode == "disk" else None,
    )


def tabicl_predict_proba_chunked(
    model: TabICLClassifier,
    X: pd.DataFrame,
    chunk_size: int = 2000,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        chunk_preds = model.predict_proba(X.iloc[start:stop])[:, 1]
        outputs.append(chunk_preds)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return clip_probs(np.concatenate(outputs))


def run_tabicl_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    *,
    n_splits: int,
    n_estimators: int,
    batch_size: int,
    offload_mode: str,
    valid_chunk_size: int,
    test_chunk_size: int,
    artifacts_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X), dtype=float)
    test_fold_preds = []
    fold_dir = artifacts_dir / "fold_cache"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        valid_cache_path = fold_dir / f"tabicl_valid_fold{fold}.npy"
        test_cache_path = fold_dir / f"tabicl_test_fold{fold}.npy"

        if valid_cache_path.exists() and test_cache_path.exists():
            valid_preds = np.load(valid_cache_path)
            fold_test_preds = np.load(test_cache_path)
            print(f"Reusing cached TabICL fold {fold}/{n_splits}")
        else:
            print(f"Running TabICL fold {fold}/{n_splits}")
            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()

            model = build_tabicl(
                RANDOM_STATE + fold,
                n_estimators=n_estimators,
                batch_size=batch_size,
                offload_mode=offload_mode,
                offload_dir=artifacts_dir / "tabicl_offload",
            )
            model.fit(X_train, y_train)

            valid_preds = tabicl_predict_proba_chunked(model, X_valid, chunk_size=valid_chunk_size)
            fold_test_preds = tabicl_predict_proba_chunked(model, X_test, chunk_size=test_chunk_size)
            np.save(valid_cache_path, valid_preds)
            np.save(test_cache_path, fold_test_preds)

        oof_preds[valid_idx] = valid_preds
        test_fold_preds.append(fold_test_preds)

    test_preds = np.mean(test_fold_preds, axis=0)
    return clip_probs(oof_preds), clip_probs(test_preds), competition_metrics(y, oof_preds)


def run_meta_logistic_cv(
    meta_train: pd.DataFrame,
    y: pd.Series,
    meta_test: pd.DataFrame,
    *,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(meta_train), dtype=float)
    test_fold_preds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(meta_train, y), start=1):
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(meta_train.iloc[train_idx], y.iloc[train_idx])
        oof_preds[valid_idx] = model.predict_proba(meta_train.iloc[valid_idx])[:, 1]
        test_fold_preds.append(model.predict_proba(meta_test)[:, 1])

    test_preds = np.mean(test_fold_preds, axis=0)
    return clip_probs(oof_preds), clip_probs(test_preds), competition_metrics(y, oof_preds)


def run_meta_catboost_cv(
    meta_train: pd.DataFrame,
    y: pd.Series,
    meta_test: pd.DataFrame,
    *,
    n_splits: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(meta_train), dtype=float)
    test_fold_preds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(meta_train, y), start=1):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=iterations,
            learning_rate=0.05,
            depth=3,
            l2_leaf_reg=5.0,
            random_seed=RANDOM_STATE + fold,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            meta_train.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=(meta_train.iloc[valid_idx], y.iloc[valid_idx]),
            use_best_model=True,
            early_stopping_rounds=40,
            verbose=False,
        )
        oof_preds[valid_idx] = model.predict_proba(meta_train.iloc[valid_idx])[:, 1]
        test_fold_preds.append(model.predict_proba(meta_test)[:, 1])

    test_preds = np.mean(test_fold_preds, axis=0)
    return clip_probs(oof_preds), clip_probs(test_preds), competition_metrics(y, oof_preds)


def build_submission(ids: pd.Series, preds: np.ndarray) -> pd.DataFrame:
    probs = clip_probs(preds)
    return pd.DataFrame(
        {
            ID_COL: ids,
            "TargetLogLoss": probs,
            "TargetRAUC": probs,
        }
    )


def maybe_build_pseudolabel_variant(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    ids: pd.Series,
    tabicl_test_preds: np.ndarray,
    tabicl_oof: np.ndarray,
    base_oof: pd.DataFrame,
    *,
    n_splits: int,
) -> tuple[str | None, np.ndarray | None, dict[str, float] | None]:
    corr_with_cat = float(np.corrcoef(tabicl_oof, base_oof["catboost"].to_numpy())[0, 1])
    if corr_with_cat >= 0.98:
        return None, None, None

    low_thr = float(np.quantile(tabicl_test_preds, 0.01))
    high_thr = float(np.quantile(tabicl_test_preds, 0.99))
    pseudo_mask = (tabicl_test_preds <= low_thr) | (tabicl_test_preds >= high_thr)
    pseudo_labels = (tabicl_test_preds[pseudo_mask] >= high_thr).astype(int)

    X_aug = pd.concat([X, X_test.loc[pseudo_mask]], axis=0, ignore_index=True)
    y_aug = pd.concat([y, pd.Series(pseudo_labels)], axis=0, ignore_index=True)
    sample_weight = np.ones(len(X_aug), dtype=float)
    sample_weight[len(X) :] = 0.15

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X), dtype=float)
    test_fold_preds = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=900,
            learning_rate=0.04,
            depth=6,
            random_seed=RANDOM_STATE + fold,
            verbose=False,
            allow_writing_files=False,
        )
        train_mask = np.zeros(len(X_aug), dtype=bool)
        train_mask[train_idx] = True
        train_mask[len(X) :] = True

        model.fit(
            X_aug.loc[train_mask],
            y_aug.loc[train_mask],
            sample_weight=sample_weight[train_mask],
            eval_set=(X.iloc[valid_idx], y.iloc[valid_idx]),
            use_best_model=True,
            early_stopping_rounds=60,
            verbose=False,
        )
        oof_preds[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        test_fold_preds.append(model.predict_proba(X_test)[:, 1])

    test_preds = np.mean(test_fold_preds, axis=0)
    return "tabicl_pseudolabel_catboost", clip_probs(test_preds), competition_metrics(y, oof_preds)


def main() -> None:
    args = parse_args()
    if args.fast:
        args.splits = 3
        args.n_estimators = min(args.n_estimators, 4)
        args.valid_chunk_size = min(args.valid_chunk_size, 1500)
        args.test_chunk_size = min(args.test_chunk_size, 1000)
        args.meta_catboost_iterations = min(args.meta_catboost_iterations, 250)
        args.offload_mode = "cpu"
        args.skip_pseudolabel = True

    artifacts_dir = Path(args.artifacts_dir)
    predictions_dir = artifacts_dir / "predictions"
    submissions_dir = artifacts_dir / "submissions"

    artifacts_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    X, y, X_test, ids = load_raw_data()
    base_oof, base_test = load_base_predictions()

    tabicl_oof, tabicl_test, tabicl_metrics = run_tabicl_cv(
        X,
        y,
        X_test,
        n_splits=args.splits,
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        offload_mode=args.offload_mode,
        valid_chunk_size=args.valid_chunk_size,
        test_chunk_size=args.test_chunk_size,
        artifacts_dir=artifacts_dir,
    )
    print(
        "tabicl: "
        f"logloss={tabicl_metrics['logloss']:.6f} "
        f"auc={tabicl_metrics['auc']:.6f} "
        f"weighted_proxy={tabicl_metrics['weighted_proxy']:.6f}"
    )

    meta_train = pd.DataFrame(
        {
            "catboost": base_oof["catboost"],
            "lightgbm": base_oof["lightgbm"],
            "tabicl": tabicl_oof,
        }
    )
    meta_test = pd.DataFrame(
        {
            "catboost": base_test["catboost"],
            "lightgbm": base_test["lightgbm"],
            "tabicl": tabicl_test,
        }
    )

    stack_log_oof, stack_log_test, stack_log_metrics = run_meta_logistic_cv(
        meta_train,
        y,
        meta_test,
        n_splits=args.splits,
    )
    stack_cat_oof, stack_cat_test, stack_cat_metrics = run_meta_catboost_cv(
        meta_train,
        y,
        meta_test,
        n_splits=args.splits,
        iterations=args.meta_catboost_iterations,
    )

    print(
        "stack_logistic: "
        f"logloss={stack_log_metrics['logloss']:.6f} "
        f"auc={stack_log_metrics['auc']:.6f} "
        f"weighted_proxy={stack_log_metrics['weighted_proxy']:.6f}"
    )
    print(
        "stack_catboost: "
        f"logloss={stack_cat_metrics['logloss']:.6f} "
        f"auc={stack_cat_metrics['auc']:.6f} "
        f"weighted_proxy={stack_cat_metrics['weighted_proxy']:.6f}"
    )

    # `submission2` has no OOF counterpart, so use it only as a test-time hedge.
    submission2 = pd.read_csv(DATA_DIR / "submission2.csv")
    submission2_test = clip_probs(submission2["TargetLogLoss"].to_numpy())
    blended_with_submission2 = logit_average(
        {
            "submission2": submission2_test,
            "stack_logistic": stack_log_test,
        },
        {"submission2": 0.5, "stack_logistic": 0.5},
    )

    corr_report = {
        "tabicl_vs_catboost": float(np.corrcoef(tabicl_oof, base_oof["catboost"].to_numpy())[0, 1]),
        "tabicl_vs_lightgbm": float(np.corrcoef(tabicl_oof, base_oof["lightgbm"].to_numpy())[0, 1]),
        "stack_logistic_vs_catboost": float(np.corrcoef(stack_log_oof, base_oof["catboost"].to_numpy())[0, 1]),
    }

    metrics_rows = [
        {"model": "tabicl", **tabicl_metrics},
        {"model": "stack_logistic", **stack_log_metrics},
        {"model": "stack_catboost", **stack_cat_metrics},
    ]

    submissions = {
        "submission_tabicl.csv": build_submission(ids, tabicl_test),
        "submission_stack_logistic.csv": build_submission(ids, stack_log_test),
        "submission_stack_catboost.csv": build_submission(ids, stack_cat_test),
        "submission_submission2_stack50_logit.csv": build_submission(ids, blended_with_submission2),
    }

    pseudo_name, pseudo_test_preds, pseudo_metrics = (None, None, None)
    if not args.skip_pseudolabel:
        pseudo_name, pseudo_test_preds, pseudo_metrics = maybe_build_pseudolabel_variant(
            X=X,
            y=y,
            X_test=X_test,
            ids=ids,
            tabicl_test_preds=tabicl_test,
            tabicl_oof=tabicl_oof,
            base_oof=base_oof,
            n_splits=args.splits,
        )
    if pseudo_name and pseudo_test_preds is not None and pseudo_metrics is not None:
        metrics_rows.append({"model": pseudo_name, **pseudo_metrics})
        submissions[f"submission_{pseudo_name}.csv"] = build_submission(ids, pseudo_test_preds)
        print(
            f"{pseudo_name}: "
            f"logloss={pseudo_metrics['logloss']:.6f} "
            f"auc={pseudo_metrics['auc']:.6f} "
            f"weighted_proxy={pseudo_metrics['weighted_proxy']:.6f}"
        )

    pd.DataFrame(metrics_rows).sort_values("weighted_proxy", ascending=False).to_csv(
        artifacts_dir / "model_metrics.csv",
        index=False,
    )

    oof_df = meta_train.copy()
    oof_df["tabicl"] = tabicl_oof
    oof_df["stack_logistic"] = stack_log_oof
    oof_df["stack_catboost"] = stack_cat_oof
    oof_df[TARGET] = y.to_numpy()
    oof_df.to_csv(predictions_dir / "oof_predictions.csv", index=False)

    test_df = meta_test.copy()
    test_df.insert(0, ID_COL, ids)
    test_df["tabicl"] = tabicl_test
    test_df["stack_logistic"] = stack_log_test
    test_df["stack_catboost"] = stack_cat_test
    test_df["submission2_stack50_logit"] = blended_with_submission2
    if pseudo_name and pseudo_test_preds is not None:
        test_df[pseudo_name] = pseudo_test_preds
    test_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

    for name, frame in submissions.items():
        frame.to_csv(submissions_dir / name, index=False)

    summary = {
        "config": {
            "splits": args.splits,
            "n_estimators": args.n_estimators,
            "batch_size": args.batch_size,
            "offload_mode": args.offload_mode,
            "valid_chunk_size": args.valid_chunk_size,
            "test_chunk_size": args.test_chunk_size,
            "meta_catboost_iterations": args.meta_catboost_iterations,
            "skip_pseudolabel": args.skip_pseudolabel,
        },
        "tabicl_metrics": tabicl_metrics,
        "stack_logistic_metrics": stack_log_metrics,
        "stack_catboost_metrics": stack_cat_metrics,
        "correlations": corr_report,
        "submissions": [str(submissions_dir / name) for name in submissions],
        "note": "submission2 is only used as a test-time hedge because no train OOF is available.",
    }
    (artifacts_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
