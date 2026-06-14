r"""
05B_leakage_audit_and_feature_purity_analysis.py

Purpose
-------
Audit possible label leakage in the experiments from:

    05_baseline_and_ai_model_experiments.py

The main concern is that the target label:

    panel_fiscal_anomaly_label

was derived from a formula-based risk score using risk components, and some features
used by models may be direct or indirect inputs to the label-generation process.

This script:
1. Audits all available features.
2. Classifies features as:
   - target_or_label_column
   - direct_leakage
   - governance_metric_leakage
   - indirect_formula_related
   - metadata_or_identifier
   - retained_pure_feature
3. Rebuilds three feature sets:
   - original feature set
   - leakage-safe feature set
   - ultra-conservative feature set
4. Re-runs baseline/AI/governance experiments under leakage-safe conditions.
5. Compares original vs leakage-safe performance.
6. Produces reviewer-facing evidence that the analysis either remains defensible or
   must be reported as internal consistency only.

Input folder
------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\04C_define_panel_governance_metrics_and_labels

Main input
----------
panel_governance_labeled_dataset.csv

Optional comparison input
-------------------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\05_baseline_and_ai_model_experiments\\model_performance_summary.csv

Output folder
-------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\05B_leakage_audit_and_feature_purity_analysis

Outputs
-------
1. leakage_feature_audit.csv
2. leakage_safe_feature_sets.csv
3. leakage_safe_model_performance.csv
4. ultra_conservative_model_performance.csv
5. leakage_safe_repeated_validation.csv
6. leakage_safe_bootstrap_confidence_intervals.csv
7. leakage_safe_paired_model_comparisons.csv
8. original_vs_leakage_safe_comparison.csv
9. leakage_safe_predictions.csv
10. leakage_audit_summary.json
11. leakage_audit_report.txt
12. figures:
    - leakage_safe_roc_curve_comparison.png
    - leakage_safe_pr_curve_comparison.png

How to run
----------
pip install pandas numpy scikit-learn matplotlib openpyxl
python 05B_leakage_audit_and_feature_purity_analysis.py

Scientific interpretation
-------------------------
If performance collapses after leakage removal, the original results should be
reported only as internal reproducibility of the scoring rule. If performance remains
reasonable, the AI models are less vulnerable to the label-leakage criticism.
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")

INPUT_DIR = BASE_DIR / "Results" / "04C_define_panel_governance_metrics_and_labels"
INPUT_DATASET = INPUT_DIR / "panel_governance_labeled_dataset.csv"

ORIGINAL_RESULTS_DIR = BASE_DIR / "Results" / "05_baseline_and_ai_model_experiments"
ORIGINAL_PERFORMANCE = ORIGINAL_RESULTS_DIR / "model_performance_summary.csv"

RESULTS_DIR = BASE_DIR / "Results" / "05B_leakage_audit_and_feature_purity_analysis"
FIG_DIR = RESULTS_DIR / "figures"

TARGET_COL = "panel_fiscal_anomaly_label"
GROUP_COL = "year_period_key"

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_REPEATS = 30
N_BOOTSTRAP = 1000
N_JOBS = -1


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def is_numeric_feature(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return np.nan


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "roc_auc": safe_auc(y_true, y_score),
        "average_precision": safe_ap(y_true, y_score),
    }


def save_fig(fig: plt.Figure, name: str) -> str:
    out = FIG_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# =============================================================================
# Loading
# =============================================================================

def load_dataset() -> pd.DataFrame:
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(
            f"Missing input dataset:\n{INPUT_DATASET}\n"
            "Run 04C_define_panel_governance_metrics_and_labels.py first."
        )

    df = pd.read_csv(INPUT_DATASET)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    if GROUP_COL not in df.columns:
        raise ValueError(f"Group column not found: {GROUP_COL}")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


# =============================================================================
# Leakage feature audit
# =============================================================================

def classify_feature(col: str) -> Tuple[str, str]:
    """
    Return leakage_class, reason.
    """
    c = col.lower()

    identifiers = {
        "panel_observation_id",
        "year_period_key",
        "period",
        "category",
        "normalized_indicator",
        "panel_entity",
        "unit_mode",
        "panel_risk_level",
    }

    if col in identifiers:
        return "metadata_or_identifier", "Identifier or descriptor, not used as numeric model feature."

    if col == TARGET_COL:
        return "target_or_label_column", "Main target label."

    # Any alternative labels or thresholds.
    if "anomaly_label" in c or c.startswith("label_high_") or "risk_level" in c:
        return "target_or_label_column", "Alternative label or class assignment."

    if "threshold" in c:
        return "target_or_label_column", "Threshold used to assign labels."

    # Direct risk score or risk components used to construct target.
    if c == "panel_fiscal_risk_score_0_100":
        return "direct_leakage", "Direct continuous score used to define the target label."

    if c.startswith("risk_component_"):
        return "direct_leakage", "Risk component used in the target-generating formula."

    # Governance metrics are not the target, but they were defined after the risk framework and may be post-hoc outputs.
    governance_keywords = [
        "traceability",
        "interpretability",
        "governance",
        "audit_readiness",
        "decision_usability",
        "risk_signal_completeness",
        "temporal_support",
    ]
    if any(k in c for k in governance_keywords):
        return "governance_metric_leakage", "Governance metric output; excluded from leakage-safe AI features."

    # Direct inputs to risk formula, because 04C formula uses these variables.
    direct_formula_inputs = [
        "value_abs_robust_z_by_entity",
        "value_robust_z_by_entity",
        "value_robust_z_by_period",
        "value_pct_change_lag_1",
        "value_change_lag_1",
        "value_rolling_std_2",
        "value_rolling_std_4",
        "n_source_files",
        "n_source_sheets",
        "value_count",
    ]
    if any(k in c for k in direct_formula_inputs):
        return "direct_leakage", "Direct input to the panel fiscal-risk formula."

    # Highly related transformations that are close to label generation.
    indirect_formula_related = [
        "value_minmax_by_entity",
        "value_rolling_mean",
        "value_lag_",
        "value_change_lag_",
        "value_pct_change_lag_",
        "source_file",
        "source_sheet",
        "n_value_columns",
        "n_raw_labels",
    ]
    if any(k in c for k in indirect_formula_related):
        return "indirect_formula_related", "Closely related to the risk formula or traceability formula."

    # Raw fiscal values are retained in leakage-safe setting but removed in ultra-conservative if desired.
    raw_value_keywords = [
        "value_mean",
        "value_median",
        "value_sum",
        "value_min",
        "value_max",
        "value_std",
        "value_winsorized",
    ]
    if any(k in c for k in raw_value_keywords):
        return "retained_pure_feature", "Raw or aggregated fiscal value feature retained for leakage-safe modeling."

    # Context fields.
    if c in {"year", "period_order", "is_core_indicator", "is_revenue_related", "is_expenditure_related", "is_debt_related", "is_surplus_deficit_related"}:
        return "retained_pure_feature", "Contextual or semantic descriptor available before scoring."

    if col == GROUP_COL:
        return "metadata_or_identifier", "Group-splitting variable."

    return "retained_pure_feature", "No direct match to leakage patterns."


def build_leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        leak_class, reason = classify_feature(col)
        rows.append({
            "feature": col,
            "leakage_class": leak_class,
            "reason": reason,
            "numeric": bool(is_numeric_feature(df, col)),
            "non_missing_count": int(df[col].notna().sum()),
            "missing_count": int(df[col].isna().sum()),
        })
    return pd.DataFrame(rows)


def build_feature_sets_from_audit(df: pd.DataFrame, audit: pd.DataFrame) -> Dict[str, List[str]]:
    numeric = audit[audit["numeric"] == True].copy()

    # Exclude labels and metadata.
    usable = numeric[
        ~numeric["leakage_class"].isin(["target_or_label_column", "metadata_or_identifier"])
    ].copy()

    # Original reconstructed feature set approximates previous script but still excludes direct target/risk score leakage.
    original_like = usable[
        ~usable["feature"].str.startswith("panel_fiscal_anomaly_label", na=False)
    ]["feature"].tolist()

    # Leakage-safe excludes all direct leakage and governance outputs.
    leakage_safe = usable[
        ~usable["leakage_class"].isin([
            "direct_leakage",
            "governance_metric_leakage",
        ])
    ]["feature"].tolist()

    # Ultra-conservative excludes direct, governance, and indirect formula-related.
    ultra_conservative = usable[
        ~usable["leakage_class"].isin([
            "direct_leakage",
            "governance_metric_leakage",
            "indirect_formula_related",
        ])
    ]["feature"].tolist()

    # Method-specific sets.
    conventional_keywords = ["value_mean", "value_median", "value_sum", "value_min", "value_max", "value_std", "value_winsorized"]
    context = [c for c in ["year", "period_order", "is_core_indicator", "is_revenue_related", "is_expenditure_related", "is_debt_related", "is_surplus_deficit_related"] if c in df.columns]

    conventional_safe = [c for c in leakage_safe if any(k in c for k in conventional_keywords) or c in context]
    standalone_safe = leakage_safe
    governance_safe = leakage_safe  # no governance outputs in safe set by design.

    conventional_ultra = [c for c in ultra_conservative if any(k in c for k in conventional_keywords) or c in context]
    standalone_ultra = ultra_conservative
    governance_ultra = ultra_conservative

    # Avoid empty sets.
    if not conventional_safe:
        conventional_safe = [c for c in context if c in df.columns]
    if not conventional_ultra:
        conventional_ultra = [c for c in context if c in df.columns]

    return {
        "leakage_safe__conventional_statistical_baseline": sorted(list(dict.fromkeys(conventional_safe))),
        "leakage_safe__standalone_ai_model": sorted(list(dict.fromkeys(standalone_safe))),
        "leakage_safe__governance_centered_ai_model": sorted(list(dict.fromkeys(governance_safe))),
        "ultra_conservative__conventional_statistical_baseline": sorted(list(dict.fromkeys(conventional_ultra))),
        "ultra_conservative__standalone_ai_model": sorted(list(dict.fromkeys(standalone_ultra))),
        "ultra_conservative__governance_centered_ai_model": sorted(list(dict.fromkeys(governance_ultra))),
    }


def feature_sets_to_table(feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for model, features in feature_sets.items():
        for f in features:
            rows.append({
                "feature_set_model": model,
                "feature": f,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Modeling
# =============================================================================

def make_preprocessor(features: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), features)
        ],
        remainder="drop",
    )


def make_model(model_type: str, features: List[str]) -> Pipeline:
    if "conventional_statistical_baseline" in model_type:
        clf = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            C=1.0,
            random_state=RANDOM_STATE,
        )
    elif "standalone_ai_model" in model_type:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
    elif "governance_centered_ai_model" in model_type:
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
    else:
        clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)

    return Pipeline(steps=[
        ("preprocess", make_preprocessor(features)),
        ("model", clf),
    ])


def get_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1]
        return p[:, -1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return np.asarray(model.predict(X))


def main_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y = df[TARGET_COL].values
    groups = df[GROUP_COL].values
    train_idx, test_idx = next(splitter.split(df[[GROUP_COL]], y, groups))
    return train_idx, test_idx


def run_model_set(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    prefix: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[TARGET_COL].values
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_train = df.iloc[train_idx]
    X_test = df.iloc[test_idx]

    perf_rows = []
    pred_rows = []
    cm_rows = []

    selected = {k: v for k, v in feature_sets.items() if k.startswith(prefix)}

    for model_name, features in selected.items():
        model_short = model_name.replace(prefix + "__", "")
        model = make_model(model_short, features)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = get_score(model, X_test)
        metrics = metric_dict(y_test, y_pred, y_score)

        row = {
            "analysis_set": prefix,
            "model": model_short,
            "feature_count": int(len(features)),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_positive": int(y_train.sum()),
            "test_positive": int(y_test.sum()),
        }
        row.update(metrics)
        perf_rows.append(row)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_rows.extend([
            {"analysis_set": prefix, "model": model_short, "cell": "tn", "count": int(cm[0, 0])},
            {"analysis_set": prefix, "model": model_short, "cell": "fp", "count": int(cm[0, 1])},
            {"analysis_set": prefix, "model": model_short, "cell": "fn", "count": int(cm[1, 0])},
            {"analysis_set": prefix, "model": model_short, "cell": "tp", "count": int(cm[1, 1])},
        ])

        for local_i, global_i in enumerate(test_idx):
            pred_rows.append({
                "analysis_set": prefix,
                "model": model_short,
                "row_index": int(global_i),
                "panel_observation_id": df.iloc[global_i].get("panel_observation_id", ""),
                "year": df.iloc[global_i].get("year", np.nan),
                "period": df.iloc[global_i].get("period", ""),
                "year_period_key": df.iloc[global_i].get("year_period_key", ""),
                "panel_entity": df.iloc[global_i].get("panel_entity", ""),
                "y_true": int(y_test[local_i]),
                "y_pred": int(y_pred[local_i]),
                "y_score": float(y_score[local_i]),
            })

    return pd.DataFrame(perf_rows), pd.DataFrame(pred_rows), pd.DataFrame(cm_rows)


def repeated_validation(df: pd.DataFrame, feature_sets: Dict[str, List[str]], prefix: str, n_repeats: int = N_REPEATS) -> pd.DataFrame:
    rows = []
    y = df[TARGET_COL].values
    groups = df[GROUP_COL].values
    selected = {k: v for k, v in feature_sets.items() if k.startswith(prefix)}

    for rep in range(n_repeats):
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE + rep)
        train_idx, test_idx = next(splitter.split(df[[GROUP_COL]], y, groups))

        X_train = df.iloc[train_idx]
        X_test = df.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name, features in selected.items():
            model_short = model_name.replace(prefix + "__", "")
            model = make_model(model_short, features)
            try:
                model.fit(X_train, y_train)
                yp = model.predict(X_test)
                ys = get_score(model, X_test)
                metrics = metric_dict(y_test, yp, ys)
            except Exception:
                metrics = {k: np.nan for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "average_precision"]}

            row = {
                "analysis_set": prefix,
                "repeat": rep + 1,
                "model": model_short,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_positive": int(y_train.sum()),
                "test_positive": int(y_test.sum()),
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_ci(preds: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for (analysis_set, model), sdf in preds.groupby(["analysis_set", "model"]):
        y_true = sdf["y_true"].values
        y_pred = sdf["y_pred"].values
        y_score = sdf["y_score"].values
        n = len(sdf)

        metrics = {k: [] for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "average_precision"]}

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            m = metric_dict(y_true[idx], y_pred[idx], y_score[idx])
            for k in metrics:
                metrics[k].append(m[k])

        for metric, vals in metrics.items():
            arr = np.asarray(vals, dtype=float)
            rows.append({
                "analysis_set": analysis_set,
                "model": model,
                "metric": metric,
                "mean": float(np.nanmean(arr)),
                "ci_lower_2_5": float(np.nanpercentile(arr, 2.5)),
                "ci_upper_97_5": float(np.nanpercentile(arr, 97.5)),
                "n_bootstrap": int(n_bootstrap),
            })

    return pd.DataFrame(rows)


def paired_comparisons(preds: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for analysis_set, sdf_all in preds.groupby("analysis_set"):
        pivot_score = sdf_all.pivot_table(index="row_index", columns="model", values="y_score", aggfunc="first")
        pivot_pred = sdf_all.pivot_table(index="row_index", columns="model", values="y_pred", aggfunc="first")
        y_true = sdf_all.drop_duplicates("row_index").set_index("row_index")["y_true"]
        common_index = pivot_score.dropna().index.intersection(y_true.index)

        pairs = [
            ("governance_centered_ai_model", "conventional_statistical_baseline"),
            ("governance_centered_ai_model", "standalone_ai_model"),
            ("standalone_ai_model", "conventional_statistical_baseline"),
        ]

        for a, b in pairs:
            if a not in pivot_score.columns or b not in pivot_score.columns:
                continue

            idx_all = np.array(common_index)
            n = len(idx_all)
            if n == 0:
                continue

            diffs = {k: [] for k in ["roc_auc", "average_precision", "f1", "balanced_accuracy"]}

            for _ in range(n_bootstrap):
                sampled = rng.choice(idx_all, size=n, replace=True)
                yt = y_true.loc[sampled].values
                ma = metric_dict(yt, pivot_pred.loc[sampled, a].values, pivot_score.loc[sampled, a].values)
                mb = metric_dict(yt, pivot_pred.loc[sampled, b].values, pivot_score.loc[sampled, b].values)

                for k in diffs:
                    diffs[k].append(ma[k] - mb[k])

            for metric, vals in diffs.items():
                arr = np.asarray(vals, dtype=float)
                rows.append({
                    "analysis_set": analysis_set,
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "mean_difference_a_minus_b": float(np.nanmean(arr)),
                    "ci_lower_2_5": float(np.nanpercentile(arr, 2.5)),
                    "ci_upper_97_5": float(np.nanpercentile(arr, 97.5)),
                    "probability_a_greater_than_b": float(np.nanmean(arr > 0)),
                    "n_bootstrap": int(n_bootstrap),
                })

    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================

def plot_roc(preds: pd.DataFrame, prefix: str, filename: str) -> str:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    sdf_all = preds[preds["analysis_set"] == prefix]
    for model, sdf in sdf_all.groupby("model"):
        yt = sdf["y_true"].values
        ys = sdf["y_score"].values
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        auc = safe_auc(yt, ys)
        ax.plot(fpr, tpr, label=f"{model} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve Comparison ({prefix})")
    ax.legend(fontsize=8)
    return save_fig(fig, filename)


def plot_pr(preds: pd.DataFrame, prefix: str, filename: str) -> str:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    sdf_all = preds[preds["analysis_set"] == prefix]
    for model, sdf in sdf_all.groupby("model"):
        yt = sdf["y_true"].values
        ys = sdf["y_score"].values
        if len(np.unique(yt)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(yt, ys)
        ap = safe_ap(yt, ys)
        ax.plot(recall, precision, label=f"{model} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve Comparison ({prefix})")
    ax.legend(fontsize=8)
    return save_fig(fig, filename)


# =============================================================================
# Reporting
# =============================================================================

def compare_original_to_safe(original_path: Path, safe_perf: pd.DataFrame) -> pd.DataFrame:
    if not original_path.exists():
        return pd.DataFrame()

    original = pd.read_csv(original_path)
    if original.empty:
        return pd.DataFrame()

    safe = safe_perf[safe_perf["analysis_set"] == "leakage_safe"].copy()

    # Align model names.
    merged = original.merge(
        safe,
        on="model",
        how="inner",
        suffixes=("_original", "_leakage_safe")
    )

    metrics = ["roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]
    rows = []
    for _, r in merged.iterrows():
        for m in metrics:
            rows.append({
                "model": r["model"],
                "metric": m,
                "original": r.get(f"{m}_original", np.nan),
                "leakage_safe": r.get(f"{m}_leakage_safe", np.nan),
                "difference_leakage_safe_minus_original": r.get(f"{m}_leakage_safe", np.nan) - r.get(f"{m}_original", np.nan),
            })

    return pd.DataFrame(rows)


def write_report(summary: Dict[str, Any], audit_counts: pd.DataFrame, safe_perf: pd.DataFrame, ultra_perf: pd.DataFrame, original_comparison: pd.DataFrame) -> None:
    lines = []
    lines.append("LEAKAGE AUDIT AND FEATURE PURITY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Run time: {summary['run_time']}")
    lines.append(f"Input dataset: {summary['input_dataset']}")
    lines.append(f"Results directory: {summary['results_dir']}")
    lines.append("")
    lines.append("1. Leakage audit")
    lines.append("-" * 80)
    lines.append(f"Total columns audited: {summary['total_columns_audited']}")
    lines.append(f"Numeric columns audited: {summary['numeric_columns_audited']}")
    for _, r in audit_counts.iterrows():
        lines.append(f"{r['leakage_class']}: {r['count']}")
    lines.append("")
    lines.append("2. Leakage-safe main performance")
    lines.append("-" * 80)
    for _, r in safe_perf.iterrows():
        lines.append(
            f"{r['model']} | features={r['feature_count']}: "
            f"ROC-AUC={r['roc_auc']:.4f}, AP={r['average_precision']:.4f}, "
            f"F1={r['f1']:.4f}, Balanced Acc={r['balanced_accuracy']:.4f}, MCC={r['mcc']:.4f}"
        )
    lines.append("")
    lines.append("3. Ultra-conservative main performance")
    lines.append("-" * 80)
    for _, r in ultra_perf.iterrows():
        lines.append(
            f"{r['model']} | features={r['feature_count']}: "
            f"ROC-AUC={r['roc_auc']:.4f}, AP={r['average_precision']:.4f}, "
            f"F1={r['f1']:.4f}, Balanced Acc={r['balanced_accuracy']:.4f}, MCC={r['mcc']:.4f}"
        )
    lines.append("")
    lines.append("4. Original vs leakage-safe comparison")
    lines.append("-" * 80)
    if original_comparison.empty:
        lines.append("Original performance file was not found or could not be aligned.")
    else:
        for _, r in original_comparison.iterrows():
            lines.append(
                f"{r['model']} | {r['metric']}: original={r['original']}, "
                f"leakage_safe={r['leakage_safe']}, "
                f"difference={r['difference_leakage_safe_minus_original']}"
            )
    lines.append("")
    lines.append("5. Interpretation")
    lines.append("-" * 80)
    lines.append("If leakage-safe performance remains acceptable, the experimental design is less vulnerable")
    lines.append("to the criticism that the model only learns the label-generation formula. If performance")
    lines.append("drops sharply, the paper should frame results as internal scoring reproducibility rather")
    lines.append("than independent predictive validation.")

    (RESULTS_DIR / "leakage_audit_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("05B LEAKAGE AUDIT AND FEATURE PURITY ANALYSIS")
    print("=" * 80)
    print(f"Input dataset:     {INPUT_DATASET}")
    print(f"Results directory: {RESULTS_DIR}")
    print("-" * 80)

    df = load_dataset()
    audit = build_leakage_audit(df)
    feature_sets = build_feature_sets_from_audit(df, audit)
    feature_set_table = feature_sets_to_table(feature_sets)

    train_idx, test_idx = main_split(df)

    safe_perf, safe_preds, safe_cms = run_model_set(df, feature_sets, "leakage_safe", train_idx, test_idx)
    ultra_perf, ultra_preds, ultra_cms = run_model_set(df, feature_sets, "ultra_conservative", train_idx, test_idx)

    all_preds = pd.concat([safe_preds, ultra_preds], ignore_index=True)
    all_perf = pd.concat([safe_perf, ultra_perf], ignore_index=True)
    all_cms = pd.concat([safe_cms, ultra_cms], ignore_index=True)

    repeated_safe = repeated_validation(df, feature_sets, "leakage_safe", n_repeats=N_REPEATS)
    ci_safe = bootstrap_ci(safe_preds, n_bootstrap=N_BOOTSTRAP)
    comparisons_safe = paired_comparisons(safe_preds, n_bootstrap=N_BOOTSTRAP)

    original_comparison = compare_original_to_safe(ORIGINAL_PERFORMANCE, safe_perf)

    audit_counts = (
        audit.groupby("leakage_class", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    roc_path = plot_roc(all_preds, "leakage_safe", "leakage_safe_roc_curve_comparison.png")
    pr_path = plot_pr(all_preds, "leakage_safe", "leakage_safe_pr_curve_comparison.png")
    roc_ultra_path = plot_roc(all_preds, "ultra_conservative", "ultra_conservative_roc_curve_comparison.png")
    pr_ultra_path = plot_pr(all_preds, "ultra_conservative", "ultra_conservative_pr_curve_comparison.png")

    # Save outputs.
    audit.to_csv(RESULTS_DIR / "leakage_feature_audit.csv", index=False, encoding="utf-8-sig")
    feature_set_table.to_csv(RESULTS_DIR / "leakage_safe_feature_sets.csv", index=False, encoding="utf-8-sig")
    safe_perf.to_csv(RESULTS_DIR / "leakage_safe_model_performance.csv", index=False, encoding="utf-8-sig")
    ultra_perf.to_csv(RESULTS_DIR / "ultra_conservative_model_performance.csv", index=False, encoding="utf-8-sig")
    all_perf.to_csv(RESULTS_DIR / "all_purity_model_performance.csv", index=False, encoding="utf-8-sig")
    repeated_safe.to_csv(RESULTS_DIR / "leakage_safe_repeated_validation.csv", index=False, encoding="utf-8-sig")
    ci_safe.to_csv(RESULTS_DIR / "leakage_safe_bootstrap_confidence_intervals.csv", index=False, encoding="utf-8-sig")
    comparisons_safe.to_csv(RESULTS_DIR / "leakage_safe_paired_model_comparisons.csv", index=False, encoding="utf-8-sig")
    all_cms.to_csv(RESULTS_DIR / "leakage_safe_confusion_matrices.csv", index=False, encoding="utf-8-sig")
    all_preds.to_csv(RESULTS_DIR / "leakage_safe_predictions.csv", index=False, encoding="utf-8-sig")
    original_comparison.to_csv(RESULTS_DIR / "original_vs_leakage_safe_comparison.csv", index=False, encoding="utf-8-sig")
    audit_counts.to_csv(RESULTS_DIR / "leakage_class_counts.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(RESULTS_DIR / "leakage_audit_preview.xlsx", engine="openpyxl") as writer:
        audit.to_excel(writer, sheet_name="feature_audit", index=False)
        feature_set_table.to_excel(writer, sheet_name="feature_sets", index=False)
        safe_perf.to_excel(writer, sheet_name="safe_performance", index=False)
        ultra_perf.to_excel(writer, sheet_name="ultra_performance", index=False)
        ci_safe.to_excel(writer, sheet_name="safe_ci", index=False)
        comparisons_safe.to_excel(writer, sheet_name="safe_comparisons", index=False)
        original_comparison.to_excel(writer, sheet_name="original_vs_safe", index=False)
        audit_counts.to_excel(writer, sheet_name="audit_counts", index=False)

    y = df[TARGET_COL].values

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "input_dataset": str(INPUT_DATASET),
        "results_dir": str(RESULTS_DIR),
        "target_label": TARGET_COL,
        "n_observations": int(len(df)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "positive_rate": float(y.mean()),
        "group_variable": GROUP_COL,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "total_columns_audited": int(len(audit)),
        "numeric_columns_audited": int(audit["numeric"].sum()),
        "leakage_class_counts": audit_counts.to_dict(orient="records"),
        "feature_counts": {k: len(v) for k, v in feature_sets.items()},
        "figures": {
            "leakage_safe_roc": roc_path,
            "leakage_safe_pr": pr_path,
            "ultra_conservative_roc": roc_ultra_path,
            "ultra_conservative_pr": pr_ultra_path,
        },
        "outputs": {
            "leakage_feature_audit": str(RESULTS_DIR / "leakage_feature_audit.csv"),
            "leakage_safe_model_performance": str(RESULTS_DIR / "leakage_safe_model_performance.csv"),
            "ultra_conservative_model_performance": str(RESULTS_DIR / "ultra_conservative_model_performance.csv"),
            "original_vs_leakage_safe_comparison": str(RESULTS_DIR / "original_vs_leakage_safe_comparison.csv"),
            "leakage_audit_report": str(RESULTS_DIR / "leakage_audit_report.txt"),
        }
    }

    with (RESULTS_DIR / "leakage_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_report(summary, audit_counts, safe_perf, ultra_perf, original_comparison)

    print("[OK] Wrote leakage_feature_audit.csv")
    print("[OK] Wrote leakage_safe_feature_sets.csv")
    print("[OK] Wrote leakage_safe_model_performance.csv")
    print("[OK] Wrote ultra_conservative_model_performance.csv")
    print("[OK] Wrote all_purity_model_performance.csv")
    print("[OK] Wrote leakage_safe_repeated_validation.csv")
    print("[OK] Wrote leakage_safe_bootstrap_confidence_intervals.csv")
    print("[OK] Wrote leakage_safe_paired_model_comparisons.csv")
    print("[OK] Wrote leakage_safe_confusion_matrices.csv")
    print("[OK] Wrote leakage_safe_predictions.csv")
    print("[OK] Wrote original_vs_leakage_safe_comparison.csv")
    print("[OK] Wrote leakage_class_counts.csv")
    print("[OK] Wrote leakage_audit_preview.xlsx")
    print("[OK] Wrote figures")
    print("[OK] Wrote leakage_audit_summary.json")
    print("[OK] Wrote leakage_audit_report.txt")
    print("-" * 80)
    print(f"Observations: {len(df)}")
    print(f"Positive labels: {int(y.sum())}")
    print(f"Negative labels: {int(len(y)-y.sum())}")
    print("Leakage-safe performance:")
    print(safe_perf[["model", "feature_count", "roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]].to_string(index=False))
    print("Ultra-conservative performance:")
    print(ultra_perf[["model", "feature_count", "roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
