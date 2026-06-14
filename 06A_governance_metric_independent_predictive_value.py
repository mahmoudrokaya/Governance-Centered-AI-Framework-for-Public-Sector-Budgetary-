# -*- coding: utf-8 -*-
r"""
06A GOVERNANCE METRIC INDEPENDENT PREDICTIVE VALUE

Project:
Transforming AI-Driven Solutions for Public Financial Governance
and Budget Performance Analysis

Purpose:
Study A evaluates the independent predictive value of each governance metric.

For each governance metric, the script compares:

1. Leakage-safe baseline features only
2. Leakage-safe baseline features + one governance metric

It reports:
- ROC-AUC
- Average Precision
- F1
- Balanced Accuracy
- MCC
- Delta metrics relative to baseline
- Repeated validation results
- Bootstrap confidence intervals
- Paired probability comparison
- Publication-ready CSV, Excel, and figure outputs

Important:
This study is intentionally separate from leakage-safe modeling.
Governance metrics were excluded from leakage-safe modeling because they are
derived governance outputs. Here, they are evaluated explicitly as candidate
governance indicators to determine whether each adds independent empirical value.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from scipy.stats import wilcoxon


warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")
RESULTS_DIR = BASE_DIR / "Results"
INPUT_DIR_04C = RESULTS_DIR / "04C_define_panel_governance_metrics_and_labels"
INPUT_DIR_05B = RESULTS_DIR / "05B_leakage_audit_and_feature_purity_analysis"

OUTPUT_DIR = RESULTS_DIR / "06A_governance_metric_independent_predictive_value"
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = INPUT_DIR_04C / "panel_governance_labeled_dataset.csv"
METRIC_DEFINITIONS_PATH = INPUT_DIR_04C / "panel_governance_metric_definitions.csv"
LEAKAGE_AUDIT_PATH = INPUT_DIR_05B / "leakage_feature_audit.csv"

TARGET_COL = "panel_fiscal_anomaly_label"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def safe_average_precision(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def make_model(model_type: str = "random_forest"):
    """
    Random Forest is used as the main AI model because previous stages used
    Random Forest for the standalone and governance-centered AI configurations.

    Logistic regression is also available for sensitivity checking if needed.
    """
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_type == "logistic":
        clf = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ]
    )
    return pipe


def get_baseline_features(df: pd.DataFrame, leakage_audit: pd.DataFrame):
    """
    Uses the leakage audit classification from Stage 05B.

    Baseline features are only those classified as retained_pure_feature.
    This avoids using direct label-construction variables in Study A.
    """
    pure_features = leakage_audit.loc[
        (leakage_audit["leakage_class"] == "retained_pure_feature")
        & (leakage_audit["numeric"] == True),
        "feature",
    ].tolist()

    pure_features = [c for c in pure_features if c in df.columns and c != TARGET_COL]

    if len(pure_features) == 0:
        raise ValueError("No retained pure features were found from leakage_feature_audit.csv")

    return pure_features


def get_governance_metrics(df: pd.DataFrame, leakage_audit: pd.DataFrame):
    """
    Governance metrics are selected from the leakage audit class:
    governance_metric_leakage.

    They are called leakage-related in Stage 05B because they are derived
    governance outputs and were excluded from leakage-safe AI modeling.

    In this Study A script, they are evaluated one-by-one as candidate
    governance indicators.
    """
    governance_metrics = leakage_audit.loc[
        (leakage_audit["leakage_class"] == "governance_metric_leakage")
        & (leakage_audit["numeric"] == True),
        "feature",
    ].tolist()

    governance_metrics = [c for c in governance_metrics if c in df.columns]

    if len(governance_metrics) == 0:
        governance_metrics = [
            c for c in df.columns
            if c.startswith("panel_")
            and (
                "governance" in c
                or "traceability" in c
                or "interpretability" in c
                or "audit_readiness" in c
                or "risk_signal" in c
                or "temporal_support" in c
                or "decision_usability" in c
            )
        ]

    if len(governance_metrics) == 0:
        raise ValueError("No governance metric columns were found.")

    return governance_metrics


def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=1000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    records = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue

        m = compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        records.append(m)

    boot = pd.DataFrame(records)

    ci_records = []
    for metric in ["roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]:
        ci_records.append(
            {
                "metric": metric,
                "mean": boot[metric].mean(),
                "ci_lower_2_5": boot[metric].quantile(0.025),
                "ci_upper_97_5": boot[metric].quantile(0.975),
                "n_bootstrap_valid": len(boot),
            }
        )

    return pd.DataFrame(ci_records)


def repeated_validation(df, features, target_col, model_name, n_splits=5, n_repeats=10):
    X = df[features].copy()
    y = df[target_col].astype(int).copy()

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    rows = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model = make_model("random_forest")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        m = compute_metrics(y_test, y_pred, y_prob)
        rows.append(
            {
                "model": model_name,
                "fold_id": fold_id,
                "n_features": len(features),
                **m,
            }
        )

    return pd.DataFrame(rows)


def get_metric_formula(metric_name, metric_definitions):
    if metric_definitions is None:
        return ""

    if "metric_name" not in metric_definitions.columns:
        return ""

    row = metric_definitions.loc[metric_definitions["metric_name"] == metric_name]
    if row.empty:
        return ""

    formula = row.iloc[0].get("formula", "")
    return formula


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("06A GOVERNANCE METRIC INDEPENDENT PREDICTIVE VALUE")

    print(f"Dataset:              {DATASET_PATH}")
    print(f"Leakage audit:        {LEAKAGE_AUDIT_PATH}")
    print(f"Metric definitions:   {METRIC_DEFINITIONS_PATH}")
    print(f"Results directory:    {OUTPUT_DIR}")
    print("-" * 80)

    require_file(DATASET_PATH)
    require_file(LEAKAGE_AUDIT_PATH)

    df = pd.read_csv(DATASET_PATH)
    leakage_audit = pd.read_csv(LEAKAGE_AUDIT_PATH)

    metric_definitions = None
    if METRIC_DEFINITIONS_PATH.exists():
        metric_definitions = pd.read_csv(METRIC_DEFINITIONS_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    baseline_features = get_baseline_features(df, leakage_audit)
    governance_metrics = get_governance_metrics(df, leakage_audit)

    print(f"Observations:         {len(df)}")
    print(f"Positive labels:      {int(df[TARGET_COL].sum())}")
    print(f"Negative labels:      {int((1 - df[TARGET_COL]).sum())}")
    print(f"Baseline features:    {len(baseline_features)}")
    print(f"Governance metrics:   {len(governance_metrics)}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # Fixed train/test split for paired comparison
    # -------------------------------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )

    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    all_model_rows = []
    all_prediction_rows = []
    all_bootstrap_rows = []
    all_repeated_rows = []
    all_importance_rows = []
    all_pairwise_rows = []

    # -------------------------------------------------------------------------
    # Baseline model
    # -------------------------------------------------------------------------

    baseline_model_name = "leakage_safe_baseline_only"

    baseline_model = make_model("random_forest")
    baseline_model.fit(train_df[baseline_features], y_train)

    baseline_prob = baseline_model.predict_proba(test_df[baseline_features])[:, 1]
    baseline_pred = (baseline_prob >= 0.5).astype(int)

    baseline_metrics = compute_metrics(y_test, baseline_pred, baseline_prob)

    all_model_rows.append(
        {
            "model": baseline_model_name,
            "added_governance_metric": "none",
            "feature_count": len(baseline_features),
            "baseline_feature_count": len(baseline_features),
            "governance_feature_count": 0,
            "formula": "",
            **baseline_metrics,
        }
    )

    base_pred_df = pd.DataFrame(
        {
            "panel_observation_id": test_df.get(
                "panel_observation_id",
                pd.Series(test_df.index, index=test_df.index),
            ).values,
            "model": baseline_model_name,
            "added_governance_metric": "none",
            "y_true": y_test.values,
            "y_prob": baseline_prob,
            "y_pred": baseline_pred,
        }
    )
    all_prediction_rows.append(base_pred_df)

    boot = bootstrap_ci(y_test.values, baseline_pred, baseline_prob)
    boot.insert(0, "model", baseline_model_name)
    boot.insert(1, "added_governance_metric", "none")
    all_bootstrap_rows.append(boot)

    repeated_base = repeated_validation(
        df=df,
        features=baseline_features,
        target_col=TARGET_COL,
        model_name=baseline_model_name,
    )
    repeated_base["added_governance_metric"] = "none"
    all_repeated_rows.append(repeated_base)

    print(f"[OK] Evaluated baseline model")

    # -------------------------------------------------------------------------
    # Baseline + one governance metric at a time
    # -------------------------------------------------------------------------

    for metric_col in governance_metrics:
        model_name = f"baseline_plus__{metric_col}"

        features = baseline_features + [metric_col]

        model = make_model("random_forest")
        model.fit(train_df[features], y_train)

        y_prob = model.predict_proba(test_df[features])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        m = compute_metrics(y_test, y_pred, y_prob)

        formula = get_metric_formula(metric_col, metric_definitions)

        all_model_rows.append(
            {
                "model": model_name,
                "added_governance_metric": metric_col,
                "feature_count": len(features),
                "baseline_feature_count": len(baseline_features),
                "governance_feature_count": 1,
                "formula": formula,
                **m,
            }
        )

        pred_df = pd.DataFrame(
            {
                "panel_observation_id": test_df.get(
                    "panel_observation_id",
                    pd.Series(test_df.index, index=test_df.index),
                ).values,
                "model": model_name,
                "added_governance_metric": metric_col,
                "y_true": y_test.values,
                "baseline_prob": baseline_prob,
                "y_prob": y_prob,
                "y_pred": y_pred,
                "probability_difference_from_baseline": y_prob - baseline_prob,
            }
        )
        all_prediction_rows.append(pred_df)

        boot = bootstrap_ci(y_test.values, y_pred, y_prob)
        boot.insert(0, "model", model_name)
        boot.insert(1, "added_governance_metric", metric_col)
        all_bootstrap_rows.append(boot)

        repeated = repeated_validation(
            df=df,
            features=features,
            target_col=TARGET_COL,
            model_name=model_name,
        )
        repeated["added_governance_metric"] = metric_col
        all_repeated_rows.append(repeated)

        # Permutation importance on test split
        try:
            perm = permutation_importance(
                model,
                test_df[features],
                y_test,
                scoring="roc_auc",
                n_repeats=30,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            importance_df = pd.DataFrame(
                {
                    "model": model_name,
                    "added_governance_metric": metric_col,
                    "feature": features,
                    "permutation_importance_mean": perm.importances_mean,
                    "permutation_importance_std": perm.importances_std,
                }
            )

            importance_df["is_added_governance_metric"] = (
                importance_df["feature"] == metric_col
            )
            all_importance_rows.append(importance_df)

        except Exception as exc:
            all_importance_rows.append(
                pd.DataFrame(
                    [
                        {
                            "model": model_name,
                            "added_governance_metric": metric_col,
                            "feature": metric_col,
                            "permutation_importance_mean": np.nan,
                            "permutation_importance_std": np.nan,
                            "is_added_governance_metric": True,
                            "error": str(exc),
                        }
                    ]
                )
            )

        # Paired comparison against baseline probabilities
        try:
            stat, p_value = wilcoxon(
                np.abs(y_test.values - baseline_prob),
                np.abs(y_test.values - y_prob),
                zero_method="wilcox",
                alternative="two-sided",
            )
        except Exception:
            stat, p_value = np.nan, np.nan

        all_pairwise_rows.append(
            {
                "added_governance_metric": metric_col,
                "comparison": "absolute_probability_error_vs_baseline",
                "wilcoxon_statistic": stat,
                "p_value": p_value,
                "mean_abs_error_baseline": np.mean(np.abs(y_test.values - baseline_prob)),
                "mean_abs_error_metric_model": np.mean(np.abs(y_test.values - y_prob)),
                "delta_mean_abs_error": (
                    np.mean(np.abs(y_test.values - y_prob))
                    - np.mean(np.abs(y_test.values - baseline_prob))
                ),
            }
        )

        print(f"[OK] Evaluated {metric_col}")

    # -------------------------------------------------------------------------
    # Assemble results
    # -------------------------------------------------------------------------

    performance_df = pd.DataFrame(all_model_rows)

    baseline_row = performance_df.loc[
        performance_df["added_governance_metric"] == "none"
    ].iloc[0]

    for metric in ["roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]:
        performance_df[f"delta_{metric}"] = performance_df[metric] - baseline_row[metric]

    performance_df["improves_roc_auc"] = performance_df["delta_roc_auc"] > 0
    performance_df["improves_f1"] = performance_df["delta_f1"] > 0
    performance_df["improves_balanced_accuracy"] = (
        performance_df["delta_balanced_accuracy"] > 0
    )

    repeated_df = pd.concat(all_repeated_rows, ignore_index=True)
    bootstrap_df = pd.concat(all_bootstrap_rows, ignore_index=True)
    predictions_df = pd.concat(all_prediction_rows, ignore_index=True)
    pairwise_df = pd.DataFrame(all_pairwise_rows)

    if all_importance_rows:
        importance_df = pd.concat(all_importance_rows, ignore_index=True)
    else:
        importance_df = pd.DataFrame()

    # Repeated validation summary
    repeated_summary = (
        repeated_df
        .groupby(["model", "added_governance_metric"], as_index=False)
        [["roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]]
        .agg(["mean", "std"])
    )

    repeated_summary.columns = [
        "_".join([c for c in col if c]) for col in repeated_summary.columns
    ]
    repeated_summary = repeated_summary.reset_index()

    base_repeated = repeated_summary.loc[
        repeated_summary["added_governance_metric"] == "none"
    ].iloc[0]

    for metric in ["roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]:
        repeated_summary[f"delta_{metric}_mean"] = (
            repeated_summary[f"{metric}_mean"] - base_repeated[f"{metric}_mean"]
        )

    # Governance metric contribution table
    contribution_cols = [
        "added_governance_metric",
        "roc_auc",
        "delta_roc_auc",
        "average_precision",
        "delta_average_precision",
        "f1",
        "delta_f1",
        "balanced_accuracy",
        "delta_balanced_accuracy",
        "mcc",
        "delta_mcc",
        "formula",
    ]

    contribution_df = performance_df.loc[
        performance_df["added_governance_metric"] != "none",
        contribution_cols,
    ].copy()

    contribution_df = contribution_df.sort_values(
        by=["delta_roc_auc", "delta_f1", "delta_balanced_accuracy"],
        ascending=False,
    )

    # Add paired p-values
    contribution_df = contribution_df.merge(
        pairwise_df[
            [
                "added_governance_metric",
                "p_value",
                "delta_mean_abs_error",
            ]
        ],
        on="added_governance_metric",
        how="left",
    )

    contribution_df["evidence_interpretation"] = np.select(
        [
            (contribution_df["delta_roc_auc"] > 0.01) & (contribution_df["p_value"] < 0.05),
            (contribution_df["delta_roc_auc"] > 0.00),
            (contribution_df["delta_roc_auc"] == 0.00),
            (contribution_df["delta_roc_auc"] < 0.00),
        ],
        [
            "adds statistically supported predictive value",
            "adds small positive predictive value",
            "no observable predictive change",
            "reduces predictive performance",
        ],
        default="mixed or inconclusive evidence",
    )

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------

    performance_path = OUTPUT_DIR / "studyA_independent_predictive_value_performance.csv"
    contribution_path = OUTPUT_DIR / "studyA_governance_metric_contribution_summary.csv"
    repeated_path = OUTPUT_DIR / "studyA_repeated_validation_results.csv"
    repeated_summary_path = OUTPUT_DIR / "studyA_repeated_validation_summary.csv"
    bootstrap_path = OUTPUT_DIR / "studyA_bootstrap_confidence_intervals.csv"
    predictions_path = OUTPUT_DIR / "studyA_prediction_outputs.csv"
    pairwise_path = OUTPUT_DIR / "studyA_paired_probability_comparisons.csv"
    importance_path = OUTPUT_DIR / "studyA_permutation_importance.csv"
    feature_sets_path = OUTPUT_DIR / "studyA_feature_sets.csv"

    performance_df.to_csv(performance_path, index=False)
    contribution_df.to_csv(contribution_path, index=False)
    repeated_df.to_csv(repeated_path, index=False)
    repeated_summary.to_csv(repeated_summary_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    feature_rows = []
    for feature in baseline_features:
        feature_rows.append(
            {
                "feature_set": "leakage_safe_baseline",
                "feature": feature,
                "role": "baseline_feature",
            }
        )

    for metric_col in governance_metrics:
        feature_rows.append(
            {
                "feature_set": f"baseline_plus__{metric_col}",
                "feature": metric_col,
                "role": "added_governance_metric",
            }
        )

    feature_sets_df = pd.DataFrame(feature_rows)
    feature_sets_df.to_csv(feature_sets_path, index=False)

    # Excel preview
    excel_path = OUTPUT_DIR / "studyA_governance_metric_validation_preview.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        performance_df.to_excel(writer, sheet_name="performance", index=False)
        contribution_df.to_excel(writer, sheet_name="contribution", index=False)
        repeated_summary.to_excel(writer, sheet_name="repeated_summary", index=False)
        bootstrap_df.to_excel(writer, sheet_name="bootstrap_ci", index=False)
        pairwise_df.to_excel(writer, sheet_name="paired_tests", index=False)
        feature_sets_df.to_excel(writer, sheet_name="feature_sets", index=False)

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    plot_df = contribution_df.copy()

    plt.figure(figsize=(12, 7))
    plt.barh(
        plot_df["added_governance_metric"],
        plot_df["delta_roc_auc"],
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("Delta ROC-AUC versus leakage-safe baseline")
    plt.ylabel("Governance metric added individually")
    plt.title("Study A: Independent Predictive Value of Governance Metrics")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "studyA_delta_roc_auc_by_metric.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.barh(
        plot_df["added_governance_metric"],
        plot_df["delta_f1"],
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("Delta F1 versus leakage-safe baseline")
    plt.ylabel("Governance metric added individually")
    plt.title("Study A: F1 Change After Adding Each Governance Metric")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "studyA_delta_f1_by_metric.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.barh(
        plot_df["added_governance_metric"],
        plot_df["delta_balanced_accuracy"],
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("Delta Balanced Accuracy versus leakage-safe baseline")
    plt.ylabel("Governance metric added individually")
    plt.title("Study A: Balanced Accuracy Change After Adding Each Governance Metric")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "studyA_delta_balanced_accuracy_by_metric.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Report and summary JSON
    # -------------------------------------------------------------------------

    summary = {
        "script": "06A_governance_metric_independent_predictive_value.py",
        "objective": "Evaluate each governance metric independently by adding it to the leakage-safe baseline feature set.",
        "dataset_path": str(DATASET_PATH),
        "leakage_audit_path": str(LEAKAGE_AUDIT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "observations": int(len(df)),
        "positive_labels": int(df[TARGET_COL].sum()),
        "negative_labels": int((1 - df[TARGET_COL]).sum()),
        "baseline_feature_count": int(len(baseline_features)),
        "governance_metric_count": int(len(governance_metrics)),
        "governance_metrics": governance_metrics,
        "baseline_metrics": {
            k: float(v) if pd.notna(v) else None
            for k, v in baseline_metrics.items()
        },
        "best_metric_by_delta_roc_auc": (
            contribution_df.iloc[0]["added_governance_metric"]
            if not contribution_df.empty else None
        ),
        "best_delta_roc_auc": (
            float(contribution_df.iloc[0]["delta_roc_auc"])
            if not contribution_df.empty else None
        ),
    }

    with open(OUTPUT_DIR / "studyA_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    report_lines = []
    report_lines.append("06A GOVERNANCE METRIC INDEPENDENT PREDICTIVE VALUE")
    report_lines.append("=" * 80)
    report_lines.append(f"Dataset: {DATASET_PATH}")
    report_lines.append(f"Leakage audit: {LEAKAGE_AUDIT_PATH}")
    report_lines.append(f"Results directory: {OUTPUT_DIR}")
    report_lines.append("-" * 80)
    report_lines.append(f"Observations: {len(df)}")
    report_lines.append(f"Positive labels: {int(df[TARGET_COL].sum())}")
    report_lines.append(f"Negative labels: {int((1 - df[TARGET_COL]).sum())}")
    report_lines.append(f"Baseline feature count: {len(baseline_features)}")
    report_lines.append(f"Governance metrics evaluated: {len(governance_metrics)}")
    report_lines.append("")
    report_lines.append("Baseline performance:")
    for k, v in baseline_metrics.items():
        report_lines.append(f"  {k}: {v:.6f}")
    report_lines.append("")
    report_lines.append("Governance metric contribution summary:")
    report_lines.append(
        contribution_df[
            [
                "added_governance_metric",
                "delta_roc_auc",
                "delta_f1",
                "delta_balanced_accuracy",
                "p_value",
                "evidence_interpretation",
            ]
        ].to_string(index=False)
    )
    report_lines.append("")
    report_lines.append("Interpretation:")
    report_lines.append(
        "A governance metric is considered empirically useful only if adding it "
        "to the leakage-safe baseline improves predictive performance and the "
        "paired probability-error comparison does not indicate that the change is "
        "random or harmful. Metrics with negligible or negative deltas should be "
        "reported as governance-reporting indicators rather than as independent "
        "predictive contributors."
    )

    with open(OUTPUT_DIR / "studyA_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("-" * 80)
    print(f"[OK] Wrote {performance_path.name}")
    print(f"[OK] Wrote {contribution_path.name}")
    print(f"[OK] Wrote {repeated_path.name}")
    print(f"[OK] Wrote {repeated_summary_path.name}")
    print(f"[OK] Wrote {bootstrap_path.name}")
    print(f"[OK] Wrote {predictions_path.name}")
    print(f"[OK] Wrote {pairwise_path.name}")
    print(f"[OK] Wrote {importance_path.name}")
    print(f"[OK] Wrote {feature_sets_path.name}")
    print(f"[OK] Wrote {excel_path.name}")
    print(f"[OK] Wrote figures")
    print(f"[OK] Wrote studyA_summary.json")
    print(f"[OK] Wrote studyA_report.txt")
    print("-" * 80)

    print("Baseline metrics:")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("-" * 80)
    print("Top governance metrics by delta ROC-AUC:")
    print(
        contribution_df[
            [
                "added_governance_metric",
                "delta_roc_auc",
                "delta_f1",
                "delta_balanced_accuracy",
                "p_value",
                "evidence_interpretation",
            ]
        ].to_string(index=False)
    )

    print("=" * 80)


if __name__ == "__main__":
    main()