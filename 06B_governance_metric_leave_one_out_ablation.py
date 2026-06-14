# -*- coding: utf-8 -*-
"""
06B GOVERNANCE METRIC LEAVE-ONE-OUT ABLATION

Study B:
Evaluate the contribution of each governance metric by comparing:

1. Full governance model:
   leakage-safe baseline features + all governance metrics

2. Leave-one-metric-out models:
   leakage-safe baseline features + all governance metrics except one

Positive delta means:
Removing the metric reduced performance, so the metric contributes useful information.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

OUTPUT_DIR = RESULTS_DIR / "06B_governance_metric_leave_one_out_ablation"
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = INPUT_DIR_04C / "panel_governance_labeled_dataset.csv"
LEAKAGE_AUDIT_PATH = INPUT_DIR_05B / "leakage_feature_audit.csv"
METRIC_DEFINITIONS_PATH = INPUT_DIR_04C / "panel_governance_metric_definitions.csv"

TARGET_COL = "panel_fiscal_anomaly_label"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title):
    print("=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def make_model():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_average_precision(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "average_precision": safe_average_precision(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=1000):
    rng = np.random.default_rng(RANDOM_STATE)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    rows = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)

        if len(np.unique(y_true[idx])) < 2:
            continue

        rows.append(compute_metrics(y_true[idx], y_pred[idx], y_prob[idx]))

    boot = pd.DataFrame(rows)

    out = []
    for metric in ["roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]:
        out.append(
            {
                "metric": metric,
                "mean": boot[metric].mean(),
                "ci_lower_2_5": boot[metric].quantile(0.025),
                "ci_upper_97_5": boot[metric].quantile(0.975),
                "n_bootstrap_valid": len(boot),
            }
        )

    return pd.DataFrame(out)


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
        model = make_model()

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
                "feature_count": len(features),
                **m,
            }
        )

    return pd.DataFrame(rows)


def get_leakage_safe_features(df, leakage_audit):
    features = leakage_audit.loc[
        (leakage_audit["leakage_class"] == "retained_pure_feature")
        & (leakage_audit["numeric"] == True),
        "feature",
    ].tolist()

    features = [c for c in features if c in df.columns and c != TARGET_COL]

    if len(features) == 0:
        raise ValueError("No leakage-safe baseline features found.")

    return features


def get_governance_metrics(df, leakage_audit):
    metrics = leakage_audit.loc[
        (leakage_audit["leakage_class"] == "governance_metric_leakage")
        & (leakage_audit["numeric"] == True),
        "feature",
    ].tolist()

    metrics = [c for c in metrics if c in df.columns]

    if len(metrics) == 0:
        metrics = [
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

    if len(metrics) == 0:
        raise ValueError("No governance metric columns found.")

    return metrics


def get_formula(metric_name, metric_definitions):
    if metric_definitions is None:
        return ""

    if "metric_name" not in metric_definitions.columns:
        return ""

    matched = metric_definitions.loc[
        metric_definitions["metric_name"] == metric_name
    ]

    if matched.empty:
        return ""

    if "formula" in matched.columns:
        return matched.iloc[0]["formula"]

    return ""


def interpretation_from_delta(delta_auc, delta_f1, delta_bacc):
    if delta_auc > 0.01:
        return "strong contributor"
    if delta_auc > 0.003 and (delta_f1 > 0 or delta_bacc > 0):
        return "moderate contributor"
    if delta_auc > 0:
        return "minor contributor"
    if abs(delta_auc) < 1e-8 and abs(delta_f1) < 1e-8 and abs(delta_bacc) < 1e-8:
        return "no measurable contribution"
    if delta_auc < 0:
        return "possible harmful or redundant metric"
    return "mixed or inconclusive contribution"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("06B GOVERNANCE METRIC LEAVE-ONE-OUT ABLATION")

    print(f"Dataset:            {DATASET_PATH}")
    print(f"Leakage audit:      {LEAKAGE_AUDIT_PATH}")
    print(f"Metric definitions: {METRIC_DEFINITIONS_PATH}")
    print(f"Results directory:  {OUTPUT_DIR}")
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

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    leakage_safe_features = get_leakage_safe_features(df, leakage_audit)
    governance_metrics = get_governance_metrics(df, leakage_audit)

    full_features = leakage_safe_features + governance_metrics

    print(f"Observations:              {len(df)}")
    print(f"Positive labels:           {int(df[TARGET_COL].sum())}")
    print(f"Negative labels:           {int((1 - df[TARGET_COL]).sum())}")
    print(f"Leakage-safe features:     {len(leakage_safe_features)}")
    print(f"Governance metrics:        {len(governance_metrics)}")
    print(f"Full model feature count:  {len(full_features)}")
    print("-" * 80)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )

    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    performance_rows = []
    prediction_rows = []
    bootstrap_rows = []
    repeated_rows = []
    feature_set_rows = []
    pairwise_rows = []
    importance_rows = []

    # =========================================================================
    # FULL GOVERNANCE MODEL
    # =========================================================================

    full_model_name = "full_governance_model"

    full_model = make_model()
    full_model.fit(train_df[full_features], y_train)

    full_prob = full_model.predict_proba(test_df[full_features])[:, 1]
    full_pred = (full_prob >= 0.5).astype(int)

    full_metrics = compute_metrics(y_test, full_pred, full_prob)

    performance_rows.append(
        {
            "model": full_model_name,
            "ablation_type": "none",
            "removed_governance_metric": "none",
            "feature_count": len(full_features),
            "leakage_safe_feature_count": len(leakage_safe_features),
            "governance_metric_count": len(governance_metrics),
            **full_metrics,
        }
    )

    full_predictions = pd.DataFrame(
        {
            "panel_observation_id": test_df.get(
                "panel_observation_id",
                pd.Series(test_df.index, index=test_df.index),
            ).values,
            "model": full_model_name,
            "removed_governance_metric": "none",
            "y_true": y_test.values,
            "y_prob": full_prob,
            "y_pred": full_pred,
        }
    )
    prediction_rows.append(full_predictions)

    boot = bootstrap_ci(y_test.values, full_pred, full_prob)
    boot.insert(0, "model", full_model_name)
    boot.insert(1, "removed_governance_metric", "none")
    bootstrap_rows.append(boot)

    repeated = repeated_validation(
        df=df,
        features=full_features,
        target_col=TARGET_COL,
        model_name=full_model_name,
    )
    repeated["removed_governance_metric"] = "none"
    repeated_rows.append(repeated)

    for f in full_features:
        role = "governance_metric" if f in governance_metrics else "leakage_safe_baseline"
        feature_set_rows.append(
            {
                "model": full_model_name,
                "feature": f,
                "role": role,
                "removed_governance_metric": "none",
            }
        )

    try:
        perm = permutation_importance(
            full_model,
            test_df[full_features],
            y_test,
            scoring="roc_auc",
            n_repeats=30,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        imp_df = pd.DataFrame(
            {
                "model": full_model_name,
                "feature": full_features,
                "permutation_importance_mean": perm.importances_mean,
                "permutation_importance_std": perm.importances_std,
                "is_governance_metric": [f in governance_metrics for f in full_features],
            }
        )

        importance_rows.append(imp_df)

    except Exception as exc:
        importance_rows.append(
            pd.DataFrame(
                [
                    {
                        "model": full_model_name,
                        "feature": "",
                        "permutation_importance_mean": np.nan,
                        "permutation_importance_std": np.nan,
                        "is_governance_metric": np.nan,
                        "error": str(exc),
                    }
                ]
            )
        )

    print("[OK] Evaluated full governance model")

    # =========================================================================
    # LEAVE-ONE-METRIC-OUT ABLATION
    # =========================================================================

    for removed_metric in governance_metrics:
        model_name = f"leave_out__{removed_metric}"

        reduced_features = [
            f for f in full_features
            if f != removed_metric
        ]

        model = make_model()
        model.fit(train_df[reduced_features], y_train)

        y_prob = model.predict_proba(test_df[reduced_features])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        m = compute_metrics(y_test, y_pred, y_prob)

        performance_rows.append(
            {
                "model": model_name,
                "ablation_type": "leave_one_metric_out",
                "removed_governance_metric": removed_metric,
                "feature_count": len(reduced_features),
                "leakage_safe_feature_count": len(leakage_safe_features),
                "governance_metric_count": len(governance_metrics) - 1,
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
                "removed_governance_metric": removed_metric,
                "y_true": y_test.values,
                "full_model_prob": full_prob,
                "y_prob": y_prob,
                "y_pred": y_pred,
                "probability_difference_from_full_model": y_prob - full_prob,
            }
        )
        prediction_rows.append(pred_df)

        boot = bootstrap_ci(y_test.values, y_pred, y_prob)
        boot.insert(0, "model", model_name)
        boot.insert(1, "removed_governance_metric", removed_metric)
        bootstrap_rows.append(boot)

        repeated = repeated_validation(
            df=df,
            features=reduced_features,
            target_col=TARGET_COL,
            model_name=model_name,
        )
        repeated["removed_governance_metric"] = removed_metric
        repeated_rows.append(repeated)

        for f in reduced_features:
            role = "governance_metric" if f in governance_metrics else "leakage_safe_baseline"
            feature_set_rows.append(
                {
                    "model": model_name,
                    "feature": f,
                    "role": role,
                    "removed_governance_metric": removed_metric,
                }
            )

        try:
            stat, p_value = wilcoxon(
                np.abs(y_test.values - full_prob),
                np.abs(y_test.values - y_prob),
                zero_method="wilcox",
                alternative="two-sided",
            )
        except Exception:
            stat, p_value = np.nan, np.nan

        pairwise_rows.append(
            {
                "removed_governance_metric": removed_metric,
                "comparison": "absolute_probability_error_full_model_vs_ablated_model",
                "wilcoxon_statistic": stat,
                "p_value": p_value,
                "mean_abs_error_full_model": np.mean(np.abs(y_test.values - full_prob)),
                "mean_abs_error_ablated_model": np.mean(np.abs(y_test.values - y_prob)),
                "delta_mean_abs_error_after_removal": (
                    np.mean(np.abs(y_test.values - y_prob))
                    - np.mean(np.abs(y_test.values - full_prob))
                ),
            }
        )

        print(f"[OK] Evaluated leave-out model: {removed_metric}")

    # =========================================================================
    # ASSEMBLE RESULTS
    # =========================================================================

    performance_df = pd.DataFrame(performance_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    bootstrap_df = pd.concat(bootstrap_rows, ignore_index=True)
    repeated_df = pd.concat(repeated_rows, ignore_index=True)
    feature_sets_df = pd.DataFrame(feature_set_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    if importance_rows:
        importance_df = pd.concat(importance_rows, ignore_index=True)
    else:
        importance_df = pd.DataFrame()

    full_row = performance_df.loc[
        performance_df["removed_governance_metric"] == "none"
    ].iloc[0]

    delta_rows = []

    for _, row in performance_df.iterrows():
        if row["removed_governance_metric"] == "none":
            continue

        removed_metric = row["removed_governance_metric"]

        delta_roc_auc = full_row["roc_auc"] - row["roc_auc"]
        delta_average_precision = full_row["average_precision"] - row["average_precision"]
        delta_f1 = full_row["f1"] - row["f1"]
        delta_balanced_accuracy = full_row["balanced_accuracy"] - row["balanced_accuracy"]
        delta_mcc = full_row["mcc"] - row["mcc"]

        importance_score = (
            4 * delta_roc_auc
            + 2 * delta_f1
            + 2 * delta_balanced_accuracy
            + 1 * delta_mcc
        )

        delta_rows.append(
            {
                "removed_governance_metric": removed_metric,
                "formula": get_formula(removed_metric, metric_definitions),
                "full_roc_auc": full_row["roc_auc"],
                "ablated_roc_auc": row["roc_auc"],
                "delta_roc_auc": delta_roc_auc,
                "full_average_precision": full_row["average_precision"],
                "ablated_average_precision": row["average_precision"],
                "delta_average_precision": delta_average_precision,
                "full_f1": full_row["f1"],
                "ablated_f1": row["f1"],
                "delta_f1": delta_f1,
                "full_balanced_accuracy": full_row["balanced_accuracy"],
                "ablated_balanced_accuracy": row["balanced_accuracy"],
                "delta_balanced_accuracy": delta_balanced_accuracy,
                "full_mcc": full_row["mcc"],
                "ablated_mcc": row["mcc"],
                "delta_mcc": delta_mcc,
                "importance_score": importance_score,
                "reviewer_interpretation": interpretation_from_delta(
                    delta_roc_auc,
                    delta_f1,
                    delta_balanced_accuracy,
                ),
            }
        )

    delta_df = pd.DataFrame(delta_rows)

    delta_df = delta_df.merge(
        pairwise_df[
            [
                "removed_governance_metric",
                "p_value",
                "delta_mean_abs_error_after_removal",
            ]
        ],
        on="removed_governance_metric",
        how="left",
    )

    delta_df = delta_df.sort_values(
        by=["importance_score", "delta_roc_auc", "delta_f1"],
        ascending=False,
    )

    ranking_df = delta_df.copy()
    ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))

    repeated_summary = (
        repeated_df
        .groupby(["model", "removed_governance_metric"], as_index=False)
        [["roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]]
        .agg(["mean", "std"])
    )

    repeated_summary.columns = [
        "_".join([c for c in col if c]) for col in repeated_summary.columns
    ]
    repeated_summary = repeated_summary.reset_index()

    full_repeated_row = repeated_summary.loc[
        repeated_summary["removed_governance_metric"] == "none"
    ].iloc[0]

    for metric in ["roc_auc", "average_precision", "f1", "balanced_accuracy", "mcc"]:
        repeated_summary[f"delta_{metric}_mean_vs_full"] = (
            full_repeated_row[f"{metric}_mean"] - repeated_summary[f"{metric}_mean"]
        )

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    performance_path = OUTPUT_DIR / "ablation_performance_summary.csv"
    delta_path = OUTPUT_DIR / "ablation_delta_summary.csv"
    repeated_path = OUTPUT_DIR / "ablation_repeated_validation.csv"
    repeated_summary_path = OUTPUT_DIR / "ablation_repeated_validation_summary.csv"
    bootstrap_path = OUTPUT_DIR / "ablation_bootstrap_confidence_intervals.csv"
    predictions_path = OUTPUT_DIR / "ablation_prediction_outputs.csv"
    pairwise_path = OUTPUT_DIR / "ablation_paired_probability_comparisons.csv"
    feature_sets_path = OUTPUT_DIR / "ablation_feature_sets.csv"
    ranking_path = OUTPUT_DIR / "ablation_metric_rankings.csv"
    importance_path = OUTPUT_DIR / "ablation_permutation_importance_full_model.csv"

    performance_df.to_csv(performance_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    repeated_df.to_csv(repeated_path, index=False)
    repeated_summary.to_csv(repeated_summary_path, index=False)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    feature_sets_df.to_csv(feature_sets_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    excel_path = OUTPUT_DIR / "ablation_preview.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        performance_df.to_excel(writer, sheet_name="performance", index=False)
        delta_df.to_excel(writer, sheet_name="delta_summary", index=False)
        ranking_df.to_excel(writer, sheet_name="rankings", index=False)
        repeated_summary.to_excel(writer, sheet_name="repeated_summary", index=False)
        bootstrap_df.to_excel(writer, sheet_name="bootstrap_ci", index=False)
        pairwise_df.to_excel(writer, sheet_name="paired_tests", index=False)
        importance_df.to_excel(writer, sheet_name="importance", index=False)
        feature_sets_df.to_excel(writer, sheet_name="feature_sets", index=False)

    # =========================================================================
    # FIGURES
    # =========================================================================

    plot_df = delta_df.sort_values("delta_roc_auc", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(plot_df["removed_governance_metric"], plot_df["delta_roc_auc"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Performance drop after metric removal: Delta ROC-AUC")
    plt.ylabel("Removed governance metric")
    plt.title("Study B: Leave-One-Governance-Metric-Out Ablation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delta_roc_auc.png", dpi=300)
    plt.close()

    plot_df = delta_df.sort_values("delta_f1", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(plot_df["removed_governance_metric"], plot_df["delta_f1"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Performance drop after metric removal: Delta F1")
    plt.ylabel("Removed governance metric")
    plt.title("Study B: F1 Contribution of Governance Metrics")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delta_f1.png", dpi=300)
    plt.close()

    plot_df = delta_df.sort_values("delta_balanced_accuracy", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(
        plot_df["removed_governance_metric"],
        plot_df["delta_balanced_accuracy"],
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("Performance drop after metric removal: Delta Balanced Accuracy")
    plt.ylabel("Removed governance metric")
    plt.title("Study B: Balanced Accuracy Contribution of Governance Metrics")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delta_balanced_accuracy.png", dpi=300)
    plt.close()

    plot_df = delta_df.sort_values("importance_score", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(plot_df["removed_governance_metric"], plot_df["importance_score"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Composite importance score")
    plt.ylabel("Removed governance metric")
    plt.title("Study B: Composite Governance Metric Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "composite_importance_score.png", dpi=300)
    plt.close()

    # =========================================================================
    # SUMMARY AND REPORT
    # =========================================================================

    summary = {
        "script": "06B_governance_metric_leave_one_out_ablation.py",
        "objective": "Evaluate contribution of each governance metric by removing one metric at a time from the full governance model.",
        "dataset_path": str(DATASET_PATH),
        "leakage_audit_path": str(LEAKAGE_AUDIT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "observations": int(len(df)),
        "positive_labels": int(df[TARGET_COL].sum()),
        "negative_labels": int((1 - df[TARGET_COL]).sum()),
        "leakage_safe_feature_count": int(len(leakage_safe_features)),
        "governance_metric_count": int(len(governance_metrics)),
        "full_model_feature_count": int(len(full_features)),
        "full_model_metrics": {
            k: float(v) if pd.notna(v) else None
            for k, v in full_metrics.items()
        },
        "top_contributor": (
            ranking_df.iloc[0]["removed_governance_metric"]
            if not ranking_df.empty else None
        ),
        "top_contributor_delta_roc_auc": (
            float(ranking_df.iloc[0]["delta_roc_auc"])
            if not ranking_df.empty else None
        ),
        "top_contributor_importance_score": (
            float(ranking_df.iloc[0]["importance_score"])
            if not ranking_df.empty else None
        ),
    }

    with open(OUTPUT_DIR / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    report_lines = []
    report_lines.append("06B GOVERNANCE METRIC LEAVE-ONE-OUT ABLATION")
    report_lines.append("=" * 80)
    report_lines.append(f"Dataset: {DATASET_PATH}")
    report_lines.append(f"Leakage audit: {LEAKAGE_AUDIT_PATH}")
    report_lines.append(f"Results directory: {OUTPUT_DIR}")
    report_lines.append("-" * 80)
    report_lines.append(f"Observations: {len(df)}")
    report_lines.append(f"Positive labels: {int(df[TARGET_COL].sum())}")
    report_lines.append(f"Negative labels: {int((1 - df[TARGET_COL]).sum())}")
    report_lines.append(f"Leakage-safe baseline features: {len(leakage_safe_features)}")
    report_lines.append(f"Governance metrics evaluated: {len(governance_metrics)}")
    report_lines.append(f"Full governance model features: {len(full_features)}")
    report_lines.append("")
    report_lines.append("Full governance model performance:")
    for k, v in full_metrics.items():
        report_lines.append(f"  {k}: {v:.6f}")
    report_lines.append("")
    report_lines.append("Leave-one-metric-out contribution ranking:")
    report_lines.append(
        ranking_df[
            [
                "rank",
                "removed_governance_metric",
                "delta_roc_auc",
                "delta_f1",
                "delta_balanced_accuracy",
                "delta_mcc",
                "importance_score",
                "p_value",
                "reviewer_interpretation",
            ]
        ].to_string(index=False)
    )
    report_lines.append("")
    report_lines.append("Reviewer-facing interpretation:")
    report_lines.append(
        "A positive delta indicates that removing the governance metric reduced "
        "model performance, meaning the metric contributed useful predictive "
        "information within the full governance model. Metrics with near-zero "
        "or negative deltas should be interpreted as weak, redundant, or mainly "
        "descriptive governance-reporting indicators rather than essential "
        "predictive components."
    )

    with open(OUTPUT_DIR / "ablation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # =========================================================================
    # CONSOLE OUTPUT
    # =========================================================================

    print("-" * 80)
    print(f"[OK] Wrote {performance_path.name}")
    print(f"[OK] Wrote {delta_path.name}")
    print(f"[OK] Wrote {repeated_path.name}")
    print(f"[OK] Wrote {repeated_summary_path.name}")
    print(f"[OK] Wrote {bootstrap_path.name}")
    print(f"[OK] Wrote {predictions_path.name}")
    print(f"[OK] Wrote {pairwise_path.name}")
    print(f"[OK] Wrote {feature_sets_path.name}")
    print(f"[OK] Wrote {ranking_path.name}")
    print(f"[OK] Wrote {importance_path.name}")
    print(f"[OK] Wrote {excel_path.name}")
    print(f"[OK] Wrote figures")
    print(f"[OK] Wrote ablation_summary.json")
    print(f"[OK] Wrote ablation_report.txt")
    print("-" * 80)

    print("Full governance model metrics:")
    for k, v in full_metrics.items():
        print(f"  {k}: {v:.6f}")

    print("-" * 80)
    print("Leave-one-out contribution ranking:")
    print(
        ranking_df[
            [
                "rank",
                "removed_governance_metric",
                "delta_roc_auc",
                "delta_f1",
                "delta_balanced_accuracy",
                "importance_score",
                "p_value",
                "reviewer_interpretation",
            ]
        ].to_string(index=False)
    )
    print("=" * 80)


if __name__ == "__main__":
    main()