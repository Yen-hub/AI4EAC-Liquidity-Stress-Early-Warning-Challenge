from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "Test.csv"

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5
CV_SEEDS = [42, 2026]
EPS = 1e-6


def clip_probs(preds: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(preds, dtype=float), EPS, 1.0 - EPS)


def competition_metrics(y_true: pd.Series, preds: np.ndarray) -> dict[str, float]:
    probs = clip_probs(preds)
    loss = log_loss(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    weighted_proxy = 0.6 * (1.0 - loss) + 0.4 * auc
    return {
        "logloss": float(loss),
        "auc": float(auc),
        "weighted_proxy": float(weighted_proxy),
    }


def monthly_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for col in columns:
        match = re.match(r"^m([1-6])_(.+)$", col)
        if not match:
            continue
        suffix = match.group(2)
        groups.setdefault(suffix, []).append(col)
    for suffix, cols in groups.items():
        cols.sort(key=lambda name: int(re.match(r"^m([1-6])_", name).group(1)))
    return groups


def add_temporal_group_features(frame: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    month_index = np.arange(6, 0, -1, dtype=float)
    centered_months = month_index - month_index.mean()
    slope_denom = float(np.square(centered_months).sum())
    feature_map: dict[str, np.ndarray] = {}

    for suffix, cols in groups.items():
        values = frame[cols].astype(float)
        values_np = values.to_numpy()
        recent = values_np[:, 0]
        oldest = values_np[:, -1]
        history_mean = values_np[:, 1:].mean(axis=1)
        mean_6m = values_np.mean(axis=1)
        std_6m = values_np.std(axis=1)
        slope = (values_np * centered_months).sum(axis=1) / slope_denom

        feature_map[f"{suffix}_sum_6m"] = values_np.sum(axis=1)
        feature_map[f"{suffix}_mean_6m"] = mean_6m
        feature_map[f"{suffix}_std_6m"] = std_6m
        feature_map[f"{suffix}_min_6m"] = values_np.min(axis=1)
        feature_map[f"{suffix}_max_6m"] = values_np.max(axis=1)
        feature_map[f"{suffix}_range_6m"] = feature_map[f"{suffix}_max_6m"] - feature_map[f"{suffix}_min_6m"]
        feature_map[f"{suffix}_trend_slope"] = slope
        feature_map[f"{suffix}_recent_minus_oldest"] = recent - oldest
        feature_map[f"{suffix}_recent_ratio_oldest"] = (recent + 1.0) / (oldest + 1.0)
        feature_map[f"{suffix}_recent_ratio_history"] = (recent + 1.0) / (history_mean + 1.0)
        feature_map[f"{suffix}_zero_months"] = (values_np == 0).sum(axis=1)
        feature_map[f"{suffix}_active_months"] = (values_np > 0).sum(axis=1)
        feature_map[f"{suffix}_cv_6m"] = std_6m / (mean_6m + 1.0)

    return pd.concat([frame, pd.DataFrame(feature_map, index=frame.index)], axis=1)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()
    groups = monthly_feature_groups(engineered.columns.tolist())
    engineered = add_temporal_group_features(engineered, groups)

    inflow_families = [
        "deposit_total_value",
        "received_total_value",
        "transfer_from_bank_total_value",
    ]
    outflow_families = [
        "withdraw_total_value",
        "merchantpay_total_value",
        "paybill_total_value",
        "mm_send_total_value",
    ]
    interaction_groups: dict[str, list[str]] = {}
    interaction_features: dict[str, np.ndarray] = {}
    ratio_groups: dict[str, list[str]] = {}
    ratio_features: dict[str, np.ndarray] = {}
    counterparties_by_family = {
        "deposit": "agents",
        "merchantpay": "merchants",
        "mm_send": "recipients",
        "paybill": "companies",
        "received": "senders",
        "transfer_from_bank": "banks",
        "withdraw": "agents",
    }

    for month in range(1, 7):
        inflow_cols = [f"m{month}_{name}" for name in inflow_families if f"m{month}_{name}" in engineered.columns]
        outflow_cols = [f"m{month}_{name}" for name in outflow_families if f"m{month}_{name}" in engineered.columns]
        balance_col = f"m{month}_daily_avg_bal"

        inflow_col = f"m{month}_inflow_total_value"
        outflow_col = f"m{month}_outflow_total_value"
        netflow_col = f"m{month}_netflow_value"
        pressure_col = f"m{month}_outflow_to_inflow_ratio"
        coverage_col = f"m{month}_balance_to_outflow_ratio"

        interaction_features[inflow_col] = engineered[inflow_cols].sum(axis=1).to_numpy()
        interaction_features[outflow_col] = engineered[outflow_cols].sum(axis=1).to_numpy()
        interaction_features[netflow_col] = interaction_features[inflow_col] - interaction_features[outflow_col]
        interaction_features[pressure_col] = (interaction_features[outflow_col] + 1.0) / (
            interaction_features[inflow_col] + 1.0
        )
        interaction_features[coverage_col] = (engineered[balance_col].to_numpy() + 1.0) / (
            interaction_features[outflow_col] + 1.0
        )

        interaction_groups.setdefault("inflow_total_value", []).append(inflow_col)
        interaction_groups.setdefault("outflow_total_value", []).append(outflow_col)
        interaction_groups.setdefault("netflow_value", []).append(netflow_col)
        interaction_groups.setdefault("outflow_to_inflow_ratio", []).append(pressure_col)
        interaction_groups.setdefault("balance_to_outflow_ratio", []).append(coverage_col)

        inflow_volume_cols = [
            f"m{month}_{name}"
            for name in ("deposit_volume", "received_volume", "transfer_from_bank_volume")
            if f"m{month}_{name}" in engineered.columns
        ]
        outflow_volume_cols = [
            f"m{month}_{name}"
            for name in ("withdraw_volume", "merchantpay_volume", "paybill_volume", "mm_send_volume")
            if f"m{month}_{name}" in engineered.columns
        ]
        inflow_volume_col = f"m{month}_inflow_volume"
        outflow_volume_col = f"m{month}_outflow_volume"
        volume_pressure_col = f"m{month}_outflow_to_inflow_volume_ratio"
        interaction_features[inflow_volume_col] = engineered[inflow_volume_cols].sum(axis=1).to_numpy()
        interaction_features[outflow_volume_col] = engineered[outflow_volume_cols].sum(axis=1).to_numpy()
        interaction_features[volume_pressure_col] = (interaction_features[outflow_volume_col] + 1.0) / (
            interaction_features[inflow_volume_col] + 1.0
        )
        interaction_groups.setdefault("inflow_volume", []).append(inflow_volume_col)
        interaction_groups.setdefault("outflow_volume", []).append(outflow_volume_col)
        interaction_groups.setdefault("outflow_to_inflow_volume_ratio", []).append(volume_pressure_col)

        for family, counterparty_suffix in counterparties_by_family.items():
            total_col = f"m{month}_{family}_total_value"
            volume_col = f"m{month}_{family}_volume"
            highest_col = f"m{month}_{family}_highest_amount"
            counterparty_col = f"m{month}_{family}_{counterparty_suffix}"

            if total_col in engineered.columns and volume_col in engineered.columns:
                avg_ticket_col = f"m{month}_{family}_avg_ticket"
                ratio_features[avg_ticket_col] = (engineered[total_col].to_numpy() + 1.0) / (
                    engineered[volume_col].to_numpy() + 1.0
                )
                ratio_groups.setdefault(f"{family}_avg_ticket", []).append(avg_ticket_col)

            if highest_col in engineered.columns and total_col in engineered.columns:
                highest_share_col = f"m{month}_{family}_highest_share"
                ratio_features[highest_share_col] = (engineered[highest_col].to_numpy() + 1.0) / (
                    engineered[total_col].to_numpy() + 1.0
                )
                ratio_groups.setdefault(f"{family}_highest_share", []).append(highest_share_col)

            if counterparty_col in engineered.columns and volume_col in engineered.columns:
                counterparty_rate_col = f"m{month}_{family}_counterparty_per_txn"
                ratio_features[counterparty_rate_col] = (engineered[counterparty_col].to_numpy() + 1.0) / (
                    engineered[volume_col].to_numpy() + 1.0
                )
                ratio_groups.setdefault(f"{family}_counterparty_per_txn", []).append(counterparty_rate_col)

    interaction_features["profile_age_x_arpu"] = (engineered["age"] * engineered["arpu"]).to_numpy()
    interaction_features["activity_x_arpu"] = (engineered["x_90_d_activity_rate"] * engineered["arpu"]).to_numpy()
    interaction_features["balance_m1_to_m6_ratio"] = (
        (engineered["m1_daily_avg_bal"] + 1.0) / (engineered["m6_daily_avg_bal"] + 1.0)
    ).to_numpy()

    engineered = pd.concat([engineered, pd.DataFrame(interaction_features, index=engineered.index)], axis=1)
    engineered = pd.concat([engineered, pd.DataFrame(ratio_features, index=engineered.index)], axis=1)
    engineered = add_temporal_group_features(engineered, interaction_groups)
    engineered = add_temporal_group_features(engineered, ratio_groups)
    return engineered


def prepare_datasets() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    y = train[TARGET].copy()
    train_features = engineer_features(train.drop(columns=[TARGET]))
    test_features = engineer_features(test.copy())

    cat_cols = train_features.select_dtypes(include="object").columns.tolist()
    cat_cols = [col for col in cat_cols if col != ID_COL]
    num_cols = [col for col in train_features.columns if col not in cat_cols + [ID_COL]]

    train_features[num_cols] = train_features[num_cols].replace([np.inf, -np.inf], np.nan)
    test_features[num_cols] = test_features[num_cols].replace([np.inf, -np.inf], np.nan)

    medians = train_features[num_cols].median()
    train_features[num_cols] = train_features[num_cols].fillna(medians)
    test_features[num_cols] = test_features[num_cols].fillna(medians)

    for col in cat_cols:
        train_features[col] = train_features[col].fillna("Unknown").astype("category")
        test_features[col] = test_features[col].fillna("Unknown").astype("category")
        categories = train_features[col].cat.categories.union(test_features[col].cat.categories)
        train_features[col] = train_features[col].cat.set_categories(categories)
        test_features[col] = test_features[col].cat.set_categories(categories)

    X = train_features.drop(columns=[ID_COL]).reset_index(drop=True)
    X_test = test_features.drop(columns=[ID_COL]).reset_index(drop=True)
    test_ids = test_features[ID_COL].reset_index(drop=True)
    return X, y.reset_index(drop=True), X_test, test_ids, cat_cols


def build_xgboost(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_bin=256,
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=-1,
    )


def build_lightgbm(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def build_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )


def train_model_cv(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cat_cols: list[str],
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    oof_preds = np.zeros(len(X), dtype=float)
    oof_counts = np.zeros(len(X), dtype=float)
    test_fold_preds = []
    cat_indices = [X.columns.get_loc(col) for col in cat_cols]

    for split_seed in CV_SEEDS:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()
            y_valid = y.iloc[valid_idx].copy()

            if model_name == "xgboost":
                model = build_xgboost(split_seed + fold)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            elif model_name == "lightgbm":
                model = build_lightgbm(split_seed + fold)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric="binary_logloss",
                    categorical_feature=cat_indices,
                    callbacks=[early_stopping(100), log_evaluation(0)],
                )
            elif model_name == "catboost":
                model = build_catboost(split_seed + fold)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    cat_features=cat_cols,
                    use_best_model=True,
                    early_stopping_rounds=100,
                    verbose=False,
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")

            oof_preds[valid_idx] += model.predict_proba(X_valid)[:, 1]
            oof_counts[valid_idx] += 1.0
            test_fold_preds.append(model.predict_proba(X_test)[:, 1])

    oof_preds = oof_preds / np.maximum(oof_counts, 1.0)
    test_preds = np.mean(test_fold_preds, axis=0)
    metrics = competition_metrics(y, oof_preds)
    return clip_probs(oof_preds), clip_probs(test_preds), metrics


def optimize_blend_weights(
    predictions: dict[str, np.ndarray],
    y: pd.Series,
    objective: str,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, str]:
    model_names = list(predictions.keys())
    best_score = float("-inf")
    best_weights: dict[str, float] = {}
    best_blend = np.zeros(len(y), dtype=float)
    best_mode = "prob"

    def blend_with_weights(weights: np.ndarray, mode: str) -> np.ndarray:
        if mode == "prob":
            blend = np.zeros(len(y), dtype=float)
            for idx, model_name in enumerate(model_names):
                blend += weights[idx] * predictions[model_name]
            return clip_probs(blend)
        if mode == "logit":
            logit_blend = np.zeros(len(y), dtype=float)
            for idx, model_name in enumerate(model_names):
                preds = clip_probs(predictions[model_name])
                logit_blend += weights[idx] * np.log(preds / (1.0 - preds))
            return clip_probs(1.0 / (1.0 + np.exp(-logit_blend)))
        raise ValueError(f"Unknown blend mode: {mode}")

    grid = np.linspace(0.0, 1.0, 21)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < 0.0 or w3 > 1.0:
                continue
            weights = np.array([w1, w2, w3], dtype=float)
            for mode in ("prob", "logit"):
                blend = blend_with_weights(weights, mode)
                metrics = competition_metrics(y, blend)
                score = metrics[objective]
                if score > best_score:
                    best_score = score
                    best_weights = {model_names[idx]: float(weights[idx]) for idx in range(len(model_names))}
                    best_blend = blend
                    best_mode = mode

    return best_weights, competition_metrics(y, best_blend), best_blend, best_mode


def stack_calibrate(
    model_oof: dict[str, np.ndarray],
    model_test: dict[str, np.ndarray],
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    feature_order = list(model_oof.keys())
    oof_matrix = np.column_stack([model_oof[name] for name in feature_order])
    test_matrix = np.column_stack([model_test[name] for name in feature_order])
    calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
    calibrator.fit(oof_matrix, y)
    calibrated_oof = calibrator.predict_proba(oof_matrix)[:, 1]
    calibrated_test = calibrator.predict_proba(test_matrix)[:, 1]
    return clip_probs(calibrated_oof), clip_probs(calibrated_test)


def build_submission(ids: pd.Series, preds: np.ndarray) -> pd.DataFrame:
    probs = clip_probs(preds)
    return pd.DataFrame(
        {
            ID_COL: ids,
            "TargetLogLoss": probs,
            "TargetRAUC": probs,
        }
    )


def save_outputs(
    artifacts_dir: Path,
    predictions_dir: Path,
    submissions_dir: Path,
    ids: pd.Series,
    y: pd.Series,
    model_oof: dict[str, np.ndarray],
    model_test: dict[str, np.ndarray],
    model_metrics: dict[str, dict[str, float]],
    raw_blend_oof: np.ndarray,
    raw_blend_test: np.ndarray,
    safe_blend_oof: np.ndarray,
    safe_blend_test: np.ndarray,
    safe_blend_name: str,
    safe_blend_metrics: dict[str, float],
    weighted_blend_weights: dict[str, float],
    weighted_blend_mode: str,
    aggressive_blend_oof: np.ndarray,
    aggressive_blend_test: np.ndarray,
    aggressive_blend_metrics: dict[str, float],
    aggressive_blend_weights: dict[str, float],
    aggressive_blend_mode: str,
    n_splits: int,
) -> None:
    artifacts_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for model_name, metrics in model_metrics.items():
        metrics_rows.append({"model": model_name, **metrics})
    metrics_rows.append({"model": "raw_blend", **competition_metrics(y, raw_blend_oof)})
    metrics_rows.append({"model": "safe_blend", **safe_blend_metrics})
    metrics_rows.append({"model": "aggressive_blend", **aggressive_blend_metrics})
    metrics_df = pd.DataFrame(metrics_rows).sort_values("weighted_proxy", ascending=False)
    metrics_df.to_csv(artifacts_dir / "model_metrics.csv", index=False)

    oof_df = pd.DataFrame({model_name: preds for model_name, preds in model_oof.items()})
    oof_df["raw_blend"] = raw_blend_oof
    oof_df["safe_blend"] = safe_blend_oof
    oof_df["aggressive_blend"] = aggressive_blend_oof
    oof_df[TARGET] = y.to_numpy()
    oof_df.to_csv(predictions_dir / "oof_predictions.csv", index=False)

    test_df = pd.DataFrame({ID_COL: ids})
    for model_name, preds in model_test.items():
        test_df[model_name] = preds
    test_df["raw_blend"] = raw_blend_test
    test_df["safe_blend"] = safe_blend_test
    test_df["aggressive_blend"] = aggressive_blend_test
    test_df.to_csv(predictions_dir / "test_predictions.csv", index=False)

    safe_submission = build_submission(ids, safe_blend_test)
    aggressive_submission = build_submission(ids, aggressive_blend_test)
    safe_submission.to_csv(submissions_dir / "submission_safe_blend.csv", index=False)
    aggressive_submission.to_csv(submissions_dir / "submission_aggressive_auc_blend.csv", index=False)

    summary = {
        "n_splits": n_splits,
        "cv_seeds": CV_SEEDS,
        "safe_blend_strategy": safe_blend_name,
        "weighted_blend_weights": weighted_blend_weights,
        "weighted_blend_mode": weighted_blend_mode,
        "aggressive_blend_weights": aggressive_blend_weights,
        "aggressive_blend_mode": aggressive_blend_mode,
        "safe_submission": str(submissions_dir / "submission_safe_blend.csv"),
        "aggressive_submission": str(submissions_dir / "submission_aggressive_auc_blend.csv"),
        "metrics_file": str(artifacts_dir / "model_metrics.csv"),
    }
    (artifacts_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(n_splits: int, run_name: str) -> None:
    artifacts_dir = DATA_DIR / ("artifacts" if not run_name else f"artifacts_{run_name}")
    predictions_dir = artifacts_dir / "predictions"
    submissions_dir = artifacts_dir / "submissions"
    X, y, X_test, test_ids, cat_cols = prepare_datasets()
    model_oof: dict[str, np.ndarray] = {}
    model_test: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, float]] = {}

    for model_name in ("xgboost", "lightgbm", "catboost"):
        oof_preds, test_preds, metrics = train_model_cv(model_name, X, y, X_test, cat_cols, n_splits=n_splits)
        model_oof[model_name] = oof_preds
        model_test[model_name] = test_preds
        model_metrics[model_name] = metrics
        print(
            f"{model_name}: "
            f"logloss={metrics['logloss']:.6f} "
            f"auc={metrics['auc']:.6f} "
            f"weighted_proxy={metrics['weighted_proxy']:.6f}"
        )

    blend_weights, raw_blend_metrics, raw_blend_oof, blend_mode = optimize_blend_weights(
        model_oof,
        y,
        objective="weighted_proxy",
    )
    if blend_mode == "prob":
        raw_blend_test = np.zeros(len(X_test), dtype=float)
        for model_name, weight in blend_weights.items():
            raw_blend_test += weight * model_test[model_name]
        raw_blend_test = clip_probs(raw_blend_test)
    else:
        raw_blend_test = np.zeros(len(X_test), dtype=float)
        for model_name, weight in blend_weights.items():
            preds = clip_probs(model_test[model_name])
            raw_blend_test += weight * np.log(preds / (1.0 - preds))
        raw_blend_test = clip_probs(1.0 / (1.0 + np.exp(-raw_blend_test)))

    stacked_blend_oof, stacked_blend_test = stack_calibrate(model_oof, model_test, y)
    stacked_metrics = competition_metrics(y, stacked_blend_oof)

    if stacked_metrics["weighted_proxy"] > raw_blend_metrics["weighted_proxy"]:
        safe_blend_name = "stacked_calibrated_blend"
        safe_blend_oof = stacked_blend_oof
        safe_blend_test = stacked_blend_test
        safe_metrics = stacked_metrics
    else:
        safe_blend_name = "weighted_raw_blend_fallback"
        safe_blend_oof = raw_blend_oof
        safe_blend_test = raw_blend_test
        safe_metrics = raw_blend_metrics

    aggressive_weights, aggressive_metrics, aggressive_blend_oof, aggressive_mode = optimize_blend_weights(
        model_oof,
        y,
        objective="auc",
    )
    if aggressive_mode == "prob":
        aggressive_blend_test = np.zeros(len(X_test), dtype=float)
        for model_name, weight in aggressive_weights.items():
            aggressive_blend_test += weight * model_test[model_name]
        aggressive_blend_test = clip_probs(aggressive_blend_test)
    else:
        aggressive_blend_test = np.zeros(len(X_test), dtype=float)
        for model_name, weight in aggressive_weights.items():
            preds = clip_probs(model_test[model_name])
            aggressive_blend_test += weight * np.log(preds / (1.0 - preds))
        aggressive_blend_test = clip_probs(1.0 / (1.0 + np.exp(-aggressive_blend_test)))

    print(f"raw_blend mode={blend_mode} weights={blend_weights}")
    print(
        "raw_blend: "
        f"logloss={raw_blend_metrics['logloss']:.6f} "
        f"auc={raw_blend_metrics['auc']:.6f} "
        f"weighted_proxy={raw_blend_metrics['weighted_proxy']:.6f}"
    )
    print(
        f"{safe_blend_name}: "
        f"logloss={safe_metrics['logloss']:.6f} "
        f"auc={safe_metrics['auc']:.6f} "
        f"weighted_proxy={safe_metrics['weighted_proxy']:.6f}"
    )
    print(f"aggressive_blend mode={aggressive_mode} weights={aggressive_weights}")
    print(
        "aggressive_blend: "
        f"logloss={aggressive_metrics['logloss']:.6f} "
        f"auc={aggressive_metrics['auc']:.6f} "
        f"weighted_proxy={aggressive_metrics['weighted_proxy']:.6f}"
    )

    save_outputs(
        artifacts_dir=artifacts_dir,
        predictions_dir=predictions_dir,
        submissions_dir=submissions_dir,
        ids=test_ids,
        y=y,
        model_oof=model_oof,
        model_test=model_test,
        model_metrics=model_metrics,
        raw_blend_oof=raw_blend_oof,
        raw_blend_test=raw_blend_test,
        safe_blend_oof=safe_blend_oof,
        safe_blend_test=safe_blend_test,
        safe_blend_name=safe_blend_name,
        safe_blend_metrics=safe_metrics,
        weighted_blend_weights=blend_weights,
        weighted_blend_mode=blend_mode,
        aggressive_blend_oof=aggressive_blend_oof,
        aggressive_blend_test=aggressive_blend_test,
        aggressive_blend_metrics=aggressive_metrics,
        aggressive_blend_weights=aggressive_weights,
        aggressive_blend_mode=aggressive_mode,
        n_splits=n_splits,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()
    main(n_splits=args.folds, run_name=args.run_name)
