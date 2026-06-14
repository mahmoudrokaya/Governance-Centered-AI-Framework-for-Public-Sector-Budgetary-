"""
05_baseline_and_ai_model_experiments.py

Purpose
-------
Run reproducible baseline, standalone AI, and governance-centered model experiments
using the panel-level labeled dataset produced by:

    04C_define_panel_governance_metrics_and_labels.py

This script directly addresses reviewer concerns about:
1. baseline definition,
2. standalone AI definition,
3. governance-centered method definition,
4. model inputs,
5. target labels,
6. train/test protocol,
7. validation protocol,
8. hyperparameters,
9. confidence intervals,
10. statistical comparison.

Input folder
------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\04C_define_panel_governance_metrics_and_labels

Main input
----------
panel_governance_labeled_dataset.csv

Output folder
-------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\05_baseline_and_ai_model_experiments

Main target
-----------
panel_fiscal_anomaly_label

Observation unit
----------------
year × period × panel_entity

Compared methods
----------------
1. Conventional statistical baseline
   - LogisticRegression using only conventional fiscal value features.

2. Standalone AI model
   - RandomForestClassifier using all numeric analytical features except governance
     metric columns and target-derived columns.

3. Governance-centered AI model
   - RandomForestClassifier using analytical features plus governance-awareness
     features such as traceability, interpretability, governance alignment,
     decision usability, audit readiness, and temporal support.

Validation design
-----------------
- Main evaluation: GroupShuffleSplit by year_period_key.
  This prevents the same year-period group from being split across train/test.
- Repeated validation: repeated GroupShuffleSplit.
- Confidence intervals: bootstrap resampling of test predictions.
- Statistical comparison: paired bootstrap difference in ROC-AUC, F1, and
  balanced accuracy.

Outputs
-------
1. model_performance_summary.csv
2. repeated_validation_results.csv
3. bootstrap_confidence_intervals.csv
4. paired_model_comparisons.csv
5. confusion_matrices.csv
6. prediction_outputs.csv
7. feature_importance_governance_model.csv
8. experiment_protocol.csv
9. hyperparameter_table.csv
10. selected_feature_sets.csv
11. experiment_summary.json
12. experiment_report.txt
13. figures:
    - roc_curve_comparison.png
    - pr_curve_comparison.png
    - feature_importance_governance_model.png

How to run
----------
pip install pandas numpy scikit-learn matplotlib openpyxl
python 05_baseline_and_ai_model_experiments.py

Scientific note
---------------
Because labels are formula-derived from governance/risk components, performance must
be interpreted as internal reproducibility and comparative consistency, not as proof
of external clinical or policy ground truth. This limitation is explicitly reported.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")

INPUT_DIR = BASE_DIR / "Results" / "04C_define_panel_governance_metrics_and_labels"
INPUT_DATASET = INPUT_DIR / "panel_governance_labeled_dataset.csv"

RESULTS_DIR = BASE_DIR / "Results" / "05_baseline_and_ai_model_experiments"
FIG_DIR = RESULTS_DIR / "figures"

TARGET_COL = "panel_fiscal_anomaly_label"
GROUP_COL = "year_period_key"
RANDOM_STATE = 42

TEST_SIZE = 0.25
N_REPEATS = 30
N_BOOTSTRAP = 1000

# For speed and reproducibility.
N_JOBS = -1


# =============================================================================
# Utility
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def is_numeric_feature(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return np.nan


def get_proba_or_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1]
        return p[:, -1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return np.asarray(s)
    return model.predict(X)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "roc_auc": safe_roc_auc(y_true, y_score),
        "average_precision": safe_average_precision(y_true, y_score),
    }


def save_fig(fig: plt.Figure, name: str) -> str:
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# =============================================================================
# Data loading and feature sets
# =============================================================================

def load_dataset() -> pd.DataFrame:
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_DATASET}\n"
            "Run 04C_define_panel_governance_metrics_and_labels.py first."
        )

    df = pd.read_csv(INPUT_DATASET)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COL}")

    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column missing: {GROUP_COL}")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def identify_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    id_cols = {
        "panel_observation_id",
        "year_period_key",
        "period",
        "category",
        "normalized_indicator",
        "panel_entity",
        "unit_mode",
        "panel_risk_level",
    }

    target_or_leakage_prefixes = [
        "panel_fiscal_anomaly_label",
        "panel_multi_component_anomaly_label",
        "panel_anomaly_threshold",
        "label_high_",
    ]

    target_or_leakage_cols = set()
    for c in df.columns:
        if c == TARGET_COL:
            target_or_leakage_cols.add(c)
        if any(c.startswith(p) for p in target_or_leakage_prefixes):
            target_or_leakage_cols.add(c)
        # Exclude risk score and risk components from model input because labels are derived from them.
        if c.startswith("risk_component_"):
            target_or_leakage_cols.add(c)
        if c == "panel_fiscal_risk_score_0_100":
            target_or_leakage_cols.add(c)
        if c.startswith("panel_fiscal_anomaly_label_q"):
            target_or_leakage_cols.add(c)

    numeric_cols = [
        c for c in df.columns
        if c not in id_cols
        and c not in target_or_leakage_cols
        and is_numeric_feature(df, c)
    ]

    # Conventional baseline: use only raw value and temporal numeric fiscal features.
    conventional_keywords = [
        "value_mean", "value_median", "value_sum", "value_min", "value_max",
        "value_std", "value_count", "value_lag", "value_change", "value_pct_change",
        "value_rolling", "value_winsorized"
    ]
    conventional_features = [
        c for c in numeric_cols
        if any(k in c for k in conventional_keywords)
    ]

    # Standalone AI: all numeric analytical features, excluding governance metrics.
    governance_keywords = [
        "traceability",
        "interpretability",
        "governance",
        "audit",
        "decision",
        "temporal_support",
        "risk_signal_completeness",
    ]
    standalone_features = [
        c for c in numeric_cols
        if not any(k in c for k in governance_keywords)
    ]

    # Governance-centered: standalone + governance metric features.
    governance_features = [
        c for c in numeric_cols
        if any(k in c for k in governance_keywords)
    ]
    governance_centered_features = standalone_features + governance_features

    # Add year/period as allowed contextual predictors for all.
    context_features = [c for c in ["year", "period_order"] if c in df.columns and c not in target_or_leakage_cols]

    conventional_features = sorted(list(dict.fromkeys(context_features + conventional_features)))
    standalone_features = sorted(list(dict.fromkeys(context_features + standalone_features)))
    governance_centered_features = sorted(list(dict.fromkeys(context_features + governance_centered_features)))

    return {
        "conventional_statistical_baseline": conventional_features,
        "standalone_ai_model": standalone_features,
        "governance_centered_ai_model": governance_centered_features,
    }


def make_preprocessor(features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, features),
        ],
        remainder="drop",
    )


def make_models(feature_sets: Dict[str, List[str]]) -> Dict[str, Pipeline]:
    models = {}

    # Conventional statistical baseline.
    models["conventional_statistical_baseline"] = Pipeline(steps=[
        ("preprocess", make_preprocessor(feature_sets["conventional_statistical_baseline"])),
        ("model", LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            C=1.0,
            random_state=RANDOM_STATE,
        )),
    ])

    # Standalone AI.
    models["standalone_ai_model"] = Pipeline(steps=[
        ("preprocess", make_preprocessor(feature_sets["standalone_ai_model"])),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )),
    ])

    # Governance-centered AI.
    models["governance_centered_ai_model"] = Pipeline(steps=[
        ("preprocess", make_preprocessor(feature_sets["governance_centered_ai_model"])),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )),
    ])

    # Trivial reference, not main comparator.
    common_features = feature_sets["conventional_statistical_baseline"]
    models["majority_dummy_reference"] = Pipeline(steps=[
        ("preprocess", make_preprocessor(common_features)),
        ("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)),
    ])

    return models


# =============================================================================
# Evaluation
# =============================================================================

def make_main_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_dummy = df[[GROUP_COL]]
    y = df[TARGET_COL].values
    groups = df[GROUP_COL].values
    train_idx, test_idx = next(splitter.split(X_dummy, y, groups))
    return train_idx, test_idx


def evaluate_models_once(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[TARGET_COL].values

    X_train = df.iloc[train_idx]
    X_test = df.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    perf_rows = []
    pred_rows = []
    cm_rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_score = get_proba_or_score(model, X_test)
        y_pred = model.predict(X_test)

        metrics = metric_dict(y_test, y_pred, y_score)
        row = {
            "model": model_name,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_positive": int(y_train.sum()),
            "train_negative": int(len(y_train) - y_train.sum()),
            "test_positive": int(y_test.sum()),
            "test_negative": int(len(y_test) - y_test.sum()),
        }
        row.update(metrics)
        perf_rows.append(row)

        for local_i, global_i in enumerate(test_idx):
            pred_rows.append({
                "model": model_name,
                "row_index": int(global_i),
                "panel_observation_id": df.iloc[global_i].get("panel_observation_id", ""),
                "year": df.iloc[global_i].get("year", np.nan),
                "period": df.iloc[global_i].get("period", ""),
                "year_period_key": df.iloc[global_i].get("year_period_key", ""),
                "category": df.iloc[global_i].get("category", ""),
                "normalized_indicator": df.iloc[global_i].get("normalized_indicator", ""),
                "panel_entity": df.iloc[global_i].get("panel_entity", ""),
                "y_true": int(y_test[local_i]),
                "y_pred": int(y_pred[local_i]),
                "y_score": float(y_score[local_i]),
            })

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_rows.extend([
            {"model": model_name, "cell": "tn", "count": int(cm[0, 0])},
            {"model": model_name, "cell": "fp", "count": int(cm[0, 1])},
            {"model": model_name, "cell": "fn", "count": int(cm[1, 0])},
            {"model": model_name, "cell": "tp", "count": int(cm[1, 1])},
        ])

    return pd.DataFrame(perf_rows), pd.DataFrame(pred_rows), pd.DataFrame(cm_rows)


def repeated_group_validation(
    df: pd.DataFrame,
    models: Dict[str, Pipeline],
    n_repeats: int = N_REPEATS,
) -> pd.DataFrame:
    rows = []
    y = df[TARGET_COL].values
    groups = df[GROUP_COL].values

    for rep in range(n_repeats):
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE + rep)
        train_idx, test_idx = next(splitter.split(df[[GROUP_COL]], y, groups))

        X_train = df.iloc[train_idx]
        X_test = df.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_score = get_proba_or_score(model, X_test)
                y_pred = model.predict(X_test)
                metrics = metric_dict(y_test, y_pred, y_score)
            except Exception as e:
                metrics = {
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "mcc": np.nan,
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                }

            row = {
                "repeat": rep + 1,
                "model": model_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_positive": int(y_train.sum()),
                "test_positive": int(y_test.sum()),
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_ci_from_predictions(preds: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for model_name, sdf in preds.groupby("model"):
        y_true = sdf["y_true"].values
        y_pred = sdf["y_pred"].values
        y_score = sdf["y_score"].values
        n = len(sdf)

        boot_metrics: Dict[str, List[float]] = {
            "accuracy": [],
            "balanced_accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "mcc": [],
            "roc_auc": [],
            "average_precision": [],
        }

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            yp = y_pred[idx]
            ys = y_score[idx]
            m = metric_dict(yt, yp, ys)
            for k in boot_metrics:
                boot_metrics[k].append(m[k])

        for metric, values in boot_metrics.items():
            arr = np.asarray(values, dtype=float)
            rows.append({
                "model": model_name,
                "metric": metric,
                "mean": float(np.nanmean(arr)),
                "ci_lower_2_5": float(np.nanpercentile(arr, 2.5)),
                "ci_upper_97_5": float(np.nanpercentile(arr, 97.5)),
                "n_bootstrap": int(n_bootstrap),
            })

    return pd.DataFrame(rows)


def paired_bootstrap_comparison(preds: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    pivot_score = preds.pivot_table(index="row_index", columns="model", values="y_score", aggfunc="first")
    pivot_pred = preds.pivot_table(index="row_index", columns="model", values="y_pred", aggfunc="first")
    y_true = preds.drop_duplicates("row_index").set_index("row_index")["y_true"]
    common_index = pivot_score.dropna().index.intersection(y_true.index)

    models = [c for c in pivot_score.columns if c != "majority_dummy_reference"]
    rows = []

    comparison_pairs = [
        ("governance_centered_ai_model", "conventional_statistical_baseline"),
        ("governance_centered_ai_model", "standalone_ai_model"),
        ("standalone_ai_model", "conventional_statistical_baseline"),
    ]

    for model_a, model_b in comparison_pairs:
        if model_a not in models or model_b not in models:
            continue

        idx_all = np.array(common_index)
        n = len(idx_all)
        if n == 0:
            continue

        diffs = {
            "roc_auc": [],
            "f1": [],
            "balanced_accuracy": [],
            "average_precision": [],
        }

        for _ in range(n_bootstrap):
            sampled = rng.choice(idx_all, size=n, replace=True)
            yt = y_true.loc[sampled].values

            score_a = pivot_score.loc[sampled, model_a].values
            score_b = pivot_score.loc[sampled, model_b].values
            pred_a = pivot_pred.loc[sampled, model_a].values
            pred_b = pivot_pred.loc[sampled, model_b].values

            ma = metric_dict(yt, pred_a, score_a)
            mb = metric_dict(yt, pred_b, score_b)

            for metric in diffs:
                diffs[metric].append(ma[metric] - mb[metric])

        for metric, values in diffs.items():
            arr = np.asarray(values, dtype=float)
            rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric,
                "mean_difference_a_minus_b": float(np.nanmean(arr)),
                "ci_lower_2_5": float(np.nanpercentile(arr, 2.5)),
                "ci_upper_97_5": float(np.nanpercentile(arr, 97.5)),
                "probability_a_greater_than_b": float(np.nanmean(arr > 0)),
                "n_bootstrap": int(n_bootstrap),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Feature importance and plots
# =============================================================================

def governance_model_feature_importance(
    df: pd.DataFrame,
    model: Pipeline,
    features: List[str],
    test_idx: np.ndarray,
) -> pd.DataFrame:
    X_test = df.iloc[test_idx]
    y_test = df.iloc[test_idx][TARGET_COL].values

    try:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=20,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=N_JOBS,
        )

        imp = pd.DataFrame({
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)

        return imp
    except Exception:
        # Fallback to RF internal importances after preprocessing.
        try:
            rf = model.named_steps["model"]
            if hasattr(rf, "feature_importances_"):
                return pd.DataFrame({
                    "feature": features,
                    "importance_mean": rf.feature_importances_,
                    "importance_std": np.nan,
                }).sort_values("importance_mean", ascending=False)
        except Exception:
            pass

    return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])


def plot_roc_curves(preds: pd.DataFrame) -> str:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for model_name, sdf in preds.groupby("model"):
        y_true = sdf["y_true"].values
        y_score = sdf["y_score"].values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = safe_roc_auc(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(fontsize=8)
    return save_fig(fig, "roc_curve_comparison.png")


def plot_pr_curves(preds: pd.DataFrame) -> str:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    for model_name, sdf in preds.groupby("model"):
        y_true = sdf["y_true"].values
        y_score = sdf["y_score"].values
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = safe_average_precision(y_true, y_score)
        ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(fontsize=8)
    return save_fig(fig, "pr_curve_comparison.png")


def plot_feature_importance(imp: pd.DataFrame) -> str:
    top = imp.head(20).copy()
    if top.empty:
        return ""

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    top = top.sort_values("importance_mean", ascending=True)
    ax.barh(top["feature"], top["importance_mean"])
    ax.set_xlabel("Permutation Importance")
    ax.set_title("Top Governance-Centered Model Features")
    return save_fig(fig, "feature_importance_governance_model.png")


# =============================================================================
# Protocol tables
# =============================================================================

def build_protocol_table(df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "item": "observation_unit",
            "value": "year × period × panel_entity",
            "description": "Each row is one fiscal indicator entity observed in a fiscal reporting period."
        },
        {
            "item": "target_label",
            "value": TARGET_COL,
            "description": "Formula-based anomaly label defined in script 04C."
        },
        {
            "item": "sample_size",
            "value": int(len(df)),
            "description": "Total panel observations used for experiments."
        },
        {
            "item": "positive_count",
            "value": int(df[TARGET_COL].sum()),
            "description": "Number of anomaly-positive observations."
        },
        {
            "item": "negative_count",
            "value": int(len(df) - df[TARGET_COL].sum()),
            "description": "Number of anomaly-negative observations."
        },
        {
            "item": "grouping_variable",
            "value": GROUP_COL,
            "description": "Used for group split to avoid leakage across the same fiscal reporting period."
        },
        {
            "item": "main_split",
            "value": f"GroupShuffleSplit(test_size={TEST_SIZE}, random_state={RANDOM_STATE})",
            "description": "Main train/test evaluation protocol."
        },
        {
            "item": "repeated_validation",
            "value": f"{N_REPEATS} repeated GroupShuffleSplit evaluations",
            "description": "Repeated validation for robustness."
        },
        {
            "item": "bootstrap_confidence_intervals",
            "value": f"{N_BOOTSTRAP} bootstrap resamples of test predictions",
            "description": "Used for metric confidence intervals and paired model comparisons."
        },
        {
            "item": "conventional_features",
            "value": len(feature_sets["conventional_statistical_baseline"]),
            "description": "Number of features used in conventional statistical baseline."
        },
        {
            "item": "standalone_ai_features",
            "value": len(feature_sets["standalone_ai_model"]),
            "description": "Number of features used in standalone AI model."
        },
        {
            "item": "governance_centered_features",
            "value": len(feature_sets["governance_centered_ai_model"]),
            "description": "Number of features used in governance-centered model."
        },
    ])


def build_hyperparameter_table() -> pd.DataFrame:
    rows = [
        {
            "model": "conventional_statistical_baseline",
            "algorithm": "LogisticRegression",
            "hyperparameters": "max_iter=2000; solver=liblinear; class_weight=balanced; C=1.0; random_state=42",
            "purpose": "Transparent conventional statistical comparator."
        },
        {
            "model": "standalone_ai_model",
            "algorithm": "RandomForestClassifier",
            "hyperparameters": "n_estimators=300; max_depth=6; min_samples_split=8; min_samples_leaf=4; class_weight=balanced_subsample; random_state=42",
            "purpose": "Standalone AI comparator using analytical features without governance metrics."
        },
        {
            "model": "governance_centered_ai_model",
            "algorithm": "RandomForestClassifier",
            "hyperparameters": "n_estimators=400; max_depth=7; min_samples_split=8; min_samples_leaf=4; class_weight=balanced_subsample; random_state=42",
            "purpose": "Governance-centered model using analytical + governance features."
        },
        {
            "model": "majority_dummy_reference",
            "algorithm": "DummyClassifier",
            "hyperparameters": "strategy=most_frequent; random_state=42",
            "purpose": "Trivial reference only; not a scientific comparator."
        },
    ]
    return pd.DataFrame(rows)


def build_feature_set_table(feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for model_name, features in feature_sets.items():
        for f in features:
            rows.append({
                "model": model_name,
                "feature": f,
            })
    return pd.DataFrame(rows)


def write_report(summary: Dict[str, Any], perf: pd.DataFrame, ci: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    lines = []
    lines.append("BASELINE AND AI MODEL EXPERIMENT REPORT")
    lines.append("=" * 80)
    lines.append(f"Run time: {summary['run_time']}")
    lines.append(f"Input dataset: {summary['input_dataset']}")
    lines.append(f"Results directory: {summary['results_dir']}")
    lines.append("")
    lines.append("1. Dataset and protocol")
    lines.append("-" * 80)
    lines.append(f"Observation unit: {summary['observation_unit']}")
    lines.append(f"Total observations: {summary['n_observations']}")
    lines.append(f"Positive observations: {summary['positive_count']}")
    lines.append(f"Negative observations: {summary['negative_count']}")
    lines.append(f"Positive rate: {summary['positive_rate']:.6f}")
    lines.append(f"Group split variable: {summary['group_variable']}")
    lines.append(f"Train observations: {summary['n_train']}")
    lines.append(f"Test observations: {summary['n_test']}")
    lines.append("")
    lines.append("2. Main performance")
    lines.append("-" * 80)

    for _, r in perf.iterrows():
        lines.append(
            f"{r['model']}: ROC-AUC={r['roc_auc']:.4f}, AP={r['average_precision']:.4f}, "
            f"F1={r['f1']:.4f}, Balanced Acc={r['balanced_accuracy']:.4f}, "
            f"Precision={r['precision']:.4f}, Recall={r['recall']:.4f}, MCC={r['mcc']:.4f}"
        )

    lines.append("")
    lines.append("3. Confidence intervals")
    lines.append("-" * 80)
    for _, r in ci.iterrows():
        if r["metric"] in ["roc_auc", "average_precision", "f1", "balanced_accuracy"]:
            lines.append(
                f"{r['model']} | {r['metric']}: mean={r['mean']:.4f}, "
                f"95% CI [{r['ci_lower_2_5']:.4f}, {r['ci_upper_97_5']:.4f}]"
            )

    lines.append("")
    lines.append("4. Paired comparisons")
    lines.append("-" * 80)
    for _, r in comparisons.iterrows():
        lines.append(
            f"{r['model_a']} minus {r['model_b']} | {r['metric']}: "
            f"mean diff={r['mean_difference_a_minus_b']:.4f}, "
            f"95% CI [{r['ci_lower_2_5']:.4f}, {r['ci_upper_97_5']:.4f}], "
            f"P(diff>0)={r['probability_a_greater_than_b']:.4f}"
        )

    lines.append("")
    lines.append("5. Scientific limitation")
    lines.append("-" * 80)
    lines.append(
        "The target label is formula-derived from governance/risk metrics rather than an external "
        "expert-annotated ground truth. Therefore, the results should be interpreted as evidence "
        "of reproducible internal analytical consistency and comparative behavior, not as external "
        "policy outcome validation."
    )

    (RESULTS_DIR / "experiment_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("05 BASELINE AND AI MODEL EXPERIMENTS")
    print("=" * 80)
    print(f"Input dataset:     {INPUT_DATASET}")
    print(f"Results directory: {RESULTS_DIR}")
    print("-" * 80)

    df = load_dataset()
    feature_sets = identify_feature_sets(df)
    models = make_models(feature_sets)

    train_idx, test_idx = make_main_split(df)

    perf, preds, cms = evaluate_models_once(df, models, train_idx, test_idx)

    repeated = repeated_group_validation(df, models, n_repeats=N_REPEATS)
    ci = bootstrap_ci_from_predictions(preds, n_bootstrap=N_BOOTSTRAP)
    comparisons = paired_bootstrap_comparison(preds, n_bootstrap=N_BOOTSTRAP)

    # Fit governance model on main split for feature importance.
    gov_model = models["governance_centered_ai_model"]
    gov_model.fit(df.iloc[train_idx], df.iloc[train_idx][TARGET_COL].values)
    gov_importance = governance_model_feature_importance(
        df=df,
        model=gov_model,
        features=feature_sets["governance_centered_ai_model"],
        test_idx=test_idx,
    )

    protocol = build_protocol_table(df, feature_sets)
    hyperparams = build_hyperparameter_table()
    feature_set_table = build_feature_set_table(feature_sets)

    roc_path = plot_roc_curves(preds)
    pr_path = plot_pr_curves(preds)
    importance_path = plot_feature_importance(gov_importance)

    # Save outputs.
    perf.to_csv(RESULTS_DIR / "model_performance_summary.csv", index=False, encoding="utf-8-sig")
    repeated.to_csv(RESULTS_DIR / "repeated_validation_results.csv", index=False, encoding="utf-8-sig")
    ci.to_csv(RESULTS_DIR / "bootstrap_confidence_intervals.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(RESULTS_DIR / "paired_model_comparisons.csv", index=False, encoding="utf-8-sig")
    cms.to_csv(RESULTS_DIR / "confusion_matrices.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(RESULTS_DIR / "prediction_outputs.csv", index=False, encoding="utf-8-sig")
    gov_importance.to_csv(RESULTS_DIR / "feature_importance_governance_model.csv", index=False, encoding="utf-8-sig")
    protocol.to_csv(RESULTS_DIR / "experiment_protocol.csv", index=False, encoding="utf-8-sig")
    hyperparams.to_csv(RESULTS_DIR / "hyperparameter_table.csv", index=False, encoding="utf-8-sig")
    feature_set_table.to_csv(RESULTS_DIR / "selected_feature_sets.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(RESULTS_DIR / "experiment_outputs_preview.xlsx", engine="openpyxl") as writer:
        perf.to_excel(writer, sheet_name="main_performance", index=False)
        repeated.to_excel(writer, sheet_name="repeated_validation", index=False)
        ci.to_excel(writer, sheet_name="bootstrap_ci", index=False)
        comparisons.to_excel(writer, sheet_name="paired_comparisons", index=False)
        cms.to_excel(writer, sheet_name="confusion_matrices", index=False)
        gov_importance.head(100).to_excel(writer, sheet_name="feature_importance", index=False)
        protocol.to_excel(writer, sheet_name="protocol", index=False)
        hyperparams.to_excel(writer, sheet_name="hyperparameters", index=False)

    y = df[TARGET_COL].values
    y_train = y[train_idx]
    y_test = y[test_idx]

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "input_dataset": str(INPUT_DATASET),
        "results_dir": str(RESULTS_DIR),
        "observation_unit": "year × period × panel_entity",
        "target_label": TARGET_COL,
        "n_observations": int(len(df)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "positive_rate": float(y.mean()),
        "group_variable": GROUP_COL,
        "n_unique_groups": int(df[GROUP_COL].nunique()),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_positive": int(y_train.sum()),
        "train_negative": int(len(y_train) - y_train.sum()),
        "test_positive": int(y_test.sum()),
        "test_negative": int(len(y_test) - y_test.sum()),
        "n_repeated_validations": int(N_REPEATS),
        "n_bootstrap": int(N_BOOTSTRAP),
        "feature_counts": {k: len(v) for k, v in feature_sets.items()},
        "figures": {
            "roc_curve_comparison": roc_path,
            "pr_curve_comparison": pr_path,
            "feature_importance_governance_model": importance_path,
        },
        "outputs": {
            "model_performance_summary": str(RESULTS_DIR / "model_performance_summary.csv"),
            "repeated_validation_results": str(RESULTS_DIR / "repeated_validation_results.csv"),
            "bootstrap_confidence_intervals": str(RESULTS_DIR / "bootstrap_confidence_intervals.csv"),
            "paired_model_comparisons": str(RESULTS_DIR / "paired_model_comparisons.csv"),
            "prediction_outputs": str(RESULTS_DIR / "prediction_outputs.csv"),
            "experiment_report": str(RESULTS_DIR / "experiment_report.txt"),
        }
    }

    with (RESULTS_DIR / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_report(summary, perf, ci, comparisons)

    print("[OK] Wrote model_performance_summary.csv")
    print("[OK] Wrote repeated_validation_results.csv")
    print("[OK] Wrote bootstrap_confidence_intervals.csv")
    print("[OK] Wrote paired_model_comparisons.csv")
    print("[OK] Wrote confusion_matrices.csv")
    print("[OK] Wrote prediction_outputs.csv")
    print("[OK] Wrote feature_importance_governance_model.csv")
    print("[OK] Wrote experiment_protocol.csv")
    print("[OK] Wrote hyperparameter_table.csv")
    print("[OK] Wrote selected_feature_sets.csv")
    print("[OK] Wrote experiment_outputs_preview.xlsx")
    print("[OK] Wrote figures")
    print("[OK] Wrote experiment_summary.json")
    print("[OK] Wrote experiment_report.txt")
    print("-" * 80)
    print(f"Observations: {len(df)}")
    print(f"Positive labels: {int(y.sum())}")
    print(f"Negative labels: {int(len(y)-y.sum())}")
    print(f"Train/Test: {len(train_idx)}/{len(test_idx)}")
    print("Main performance:")
    print(perf[["model", "roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
