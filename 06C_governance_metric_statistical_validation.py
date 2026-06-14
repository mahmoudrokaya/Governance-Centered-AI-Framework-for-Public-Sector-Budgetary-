# -*- coding: utf-8 -*-
r"""
06C GOVERNANCE METRIC STATISTICAL VALIDATION

Study C:
Statistically validate the governance metrics themselves.

This script evaluates whether each governance metric is empirically associated
with the fiscal anomaly label.

It does NOT train prediction models.

It computes:

1. Group summaries:
   - non-anomaly mean / median / SD
   - anomaly mean / median / SD
   - mean difference

2. Statistical tests:
   - Mann-Whitney U test
   - Welch t-test
   - Spearman correlation
   - Point-biserial correlation
   - Mutual information

3. Effect sizes:
   - Cohen's d
   - Hedges' g
   - Cliff's delta
   - Rank-biserial correlation

4. Bootstrap confidence intervals:
   - mean difference CI
   - median difference CI
   - Cohen's d CI

5. Multiple-testing correction:
   - Benjamini-Hochberg FDR

Outputs are designed to directly support reviewer responses regarding
empirical justification of governance metrics.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    mannwhitneyu,
    ttest_ind,
    spearmanr,
    pointbiserialr,
)

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")
RESULTS_DIR = BASE_DIR / "Results"

INPUT_DIR_04C = RESULTS_DIR / "04C_define_panel_governance_metrics_and_labels"
INPUT_DIR_05B = RESULTS_DIR / "05B_leakage_audit_and_feature_purity_analysis"

OUTPUT_DIR = RESULTS_DIR / "06C_governance_metric_statistical_validation"
FIG_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = INPUT_DIR_04C / "panel_governance_labeled_dataset.csv"
LEAKAGE_AUDIT_PATH = INPUT_DIR_05B / "leakage_feature_audit.csv"
METRIC_DEFINITIONS_PATH = INPUT_DIR_04C / "panel_governance_metric_definitions.csv"

TARGET_COL = "panel_fiscal_anomaly_label"
RANDOM_STATE = 42
N_BOOTSTRAP = 2000


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print("=" * 80)
    print(title)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


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

    row = metric_definitions.loc[
        metric_definitions["metric_name"] == metric_name
    ]

    if row.empty:
        return ""

    if "formula" in row.columns:
        return row.iloc[0]["formula"]

    return ""


def cohen_d(x0, x1):
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]

    n0 = len(x0)
    n1 = len(x1)

    if n0 < 2 or n1 < 2:
        return np.nan

    s0 = np.var(x0, ddof=1)
    s1 = np.var(x1, ddof=1)

    pooled = np.sqrt(((n0 - 1) * s0 + (n1 - 1) * s1) / (n0 + n1 - 2))

    if pooled == 0:
        return np.nan

    return (np.mean(x1) - np.mean(x0)) / pooled


def hedges_g_from_d(d, n0, n1):
    if np.isnan(d):
        return np.nan

    df = n0 + n1 - 2

    if df <= 0:
        return np.nan

    correction = 1 - (3 / (4 * df - 1))

    return d * correction


def cliffs_delta(x0, x1):
    """
    Cliff's delta:
    Positive value means values in anomaly group tend to be higher.
    """
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]

    n0 = len(x0)
    n1 = len(x1)

    if n0 == 0 or n1 == 0:
        return np.nan

    greater = 0
    lower = 0

    for value in x1:
        greater += np.sum(value > x0)
        lower += np.sum(value < x0)

    return (greater - lower) / (n0 * n1)


def rank_biserial_from_u(u_stat, n0, n1):
    if n0 == 0 or n1 == 0:
        return np.nan

    return (2 * u_stat) / (n0 * n1) - 1


def bootstrap_metric_ci(x0, x1, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_STATE)

    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    x0 = x0[~np.isnan(x0)]
    x1 = x1[~np.isnan(x1)]

    n0 = len(x0)
    n1 = len(x1)

    rows = []

    if n0 == 0 or n1 == 0:
        return {
            "mean_difference_ci_lower": np.nan,
            "mean_difference_ci_upper": np.nan,
            "median_difference_ci_lower": np.nan,
            "median_difference_ci_upper": np.nan,
            "cohen_d_ci_lower": np.nan,
            "cohen_d_ci_upper": np.nan,
            "n_bootstrap": 0,
        }

    for _ in range(n_bootstrap):
        s0 = rng.choice(x0, size=n0, replace=True)
        s1 = rng.choice(x1, size=n1, replace=True)

        rows.append(
            {
                "mean_difference": np.mean(s1) - np.mean(s0),
                "median_difference": np.median(s1) - np.median(s0),
                "cohen_d": cohen_d(s0, s1),
            }
        )

    boot = pd.DataFrame(rows)

    return {
        "mean_difference_ci_lower": boot["mean_difference"].quantile(0.025),
        "mean_difference_ci_upper": boot["mean_difference"].quantile(0.975),
        "median_difference_ci_lower": boot["median_difference"].quantile(0.025),
        "median_difference_ci_upper": boot["median_difference"].quantile(0.975),
        "cohen_d_ci_lower": boot["cohen_d"].quantile(0.025),
        "cohen_d_ci_upper": boot["cohen_d"].quantile(0.975),
        "n_bootstrap": len(boot),
    }


def benjamini_hochberg(p_values):
    """
    Returns adjusted p-values using Benjamini-Hochberg FDR.
    """
    p = np.asarray(p_values, dtype=float)

    adjusted = np.full_like(p, np.nan, dtype=float)

    valid = ~np.isnan(p)
    p_valid = p[valid]

    if len(p_valid) == 0:
        return adjusted

    order = np.argsort(p_valid)
    ranked_p = p_valid[order]

    n = len(ranked_p)
    adjusted_ranked = np.empty(n)

    for i in range(n):
        rank = i + 1
        adjusted_ranked[i] = ranked_p[i] * n / rank

    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.minimum(adjusted_ranked, 1.0)

    temp = np.empty(n)
    temp[order] = adjusted_ranked

    adjusted[valid] = temp

    return adjusted


def evidence_strength(row):
    p = row["mannwhitney_fdr_p"]
    d = abs(row["cohen_d"])
    cd = abs(row["cliffs_delta"])

    if pd.notna(p) and p < 0.05 and (d >= 0.5 or cd >= 0.33):
        return "strong empirical validity evidence"

    if pd.notna(p) and p < 0.05 and (d >= 0.2 or cd >= 0.147):
        return "moderate empirical validity evidence"

    if pd.notna(p) and p < 0.05:
        return "statistically significant but small effect"

    if pd.notna(p) and p >= 0.05 and (d >= 0.2 or cd >= 0.147):
        return "non-significant but non-trivial effect"

    return "weak or no empirical validity evidence"


def effect_size_label_cohen(d):
    ad = abs(d)

    if pd.isna(ad):
        return "undefined"

    if ad < 0.2:
        return "negligible"

    if ad < 0.5:
        return "small"

    if ad < 0.8:
        return "medium"

    return "large"


def effect_size_label_cliffs(delta):
    ad = abs(delta)

    if pd.isna(ad):
        return "undefined"

    if ad < 0.147:
        return "negligible"

    if ad < 0.33:
        return "small"

    if ad < 0.474:
        return "medium"

    return "large"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("06C GOVERNANCE METRIC STATISTICAL VALIDATION")

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

    governance_metrics = get_governance_metrics(df, leakage_audit)

    print(f"Observations:        {len(df)}")
    print(f"Positive anomalies:  {int(df[TARGET_COL].sum())}")
    print(f"Negative anomalies:  {int((1 - df[TARGET_COL]).sum())}")
    print(f"Metrics evaluated:   {len(governance_metrics)}")
    print("-" * 80)

    y = df[TARGET_COL].astype(int).values

    rows = []
    distribution_rows = []

    for metric in governance_metrics:
        values = pd.to_numeric(df[metric], errors="coerce")
        valid_mask = values.notna()

        temp = pd.DataFrame(
            {
                "metric": metric,
                "value": values,
                "label": df[TARGET_COL],
            }
        ).dropna()

        x0 = temp.loc[temp["label"] == 0, "value"].values
        x1 = temp.loc[temp["label"] == 1, "value"].values

        n0 = len(x0)
        n1 = len(x1)

        mean0 = np.mean(x0) if n0 > 0 else np.nan
        mean1 = np.mean(x1) if n1 > 0 else np.nan

        median0 = np.median(x0) if n0 > 0 else np.nan
        median1 = np.median(x1) if n1 > 0 else np.nan

        sd0 = np.std(x0, ddof=1) if n0 > 1 else np.nan
        sd1 = np.std(x1, ddof=1) if n1 > 1 else np.nan

        q1_0 = np.quantile(x0, 0.25) if n0 > 0 else np.nan
        q3_0 = np.quantile(x0, 0.75) if n0 > 0 else np.nan

        q1_1 = np.quantile(x1, 0.25) if n1 > 0 else np.nan
        q3_1 = np.quantile(x1, 0.75) if n1 > 0 else np.nan

        try:
            u_stat, mw_p = mannwhitneyu(x1, x0, alternative="two-sided")
        except Exception:
            u_stat, mw_p = np.nan, np.nan

        try:
            t_stat, t_p = ttest_ind(x1, x0, equal_var=False, nan_policy="omit")
        except Exception:
            t_stat, t_p = np.nan, np.nan

        try:
            spearman_r, spearman_p = spearmanr(temp["value"], temp["label"])
        except Exception:
            spearman_r, spearman_p = np.nan, np.nan

        try:
            pb_r, pb_p = pointbiserialr(temp["label"], temp["value"])
        except Exception:
            pb_r, pb_p = np.nan, np.nan

        try:
            mi = mutual_info_classif(
                temp[["value"]],
                temp["label"].astype(int),
                discrete_features=False,
                random_state=RANDOM_STATE,
            )[0]
        except Exception:
            mi = np.nan

        d = cohen_d(x0, x1)
        g = hedges_g_from_d(d, n0, n1)
        cd = cliffs_delta(x0, x1)
        rb = rank_biserial_from_u(u_stat, n0, n1)

        ci = bootstrap_metric_ci(x0, x1, N_BOOTSTRAP)

        rows.append(
            {
                "metric": metric,
                "formula": get_formula(metric, metric_definitions),
                "n_non_anomaly": n0,
                "n_anomaly": n1,
                "mean_non_anomaly": mean0,
                "mean_anomaly": mean1,
                "mean_difference_anomaly_minus_non": mean1 - mean0,
                "median_non_anomaly": median0,
                "median_anomaly": median1,
                "median_difference_anomaly_minus_non": median1 - median0,
                "sd_non_anomaly": sd0,
                "sd_anomaly": sd1,
                "q1_non_anomaly": q1_0,
                "q3_non_anomaly": q3_0,
                "q1_anomaly": q1_1,
                "q3_anomaly": q3_1,
                "mannwhitney_u": u_stat,
                "mannwhitney_p": mw_p,
                "welch_t": t_stat,
                "welch_p": t_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "point_biserial_r": pb_r,
                "point_biserial_p": pb_p,
                "mutual_information": mi,
                "cohen_d": d,
                "hedges_g": g,
                "cliffs_delta": cd,
                "rank_biserial_correlation": rb,
                **ci,
            }
        )

        for group_label, arr in [("non_anomaly", x0), ("anomaly", x1)]:
            distribution_rows.append(
                {
                    "metric": metric,
                    "group": group_label,
                    "n": len(arr),
                    "mean": np.mean(arr) if len(arr) else np.nan,
                    "median": np.median(arr) if len(arr) else np.nan,
                    "std": np.std(arr, ddof=1) if len(arr) > 1 else np.nan,
                    "min": np.min(arr) if len(arr) else np.nan,
                    "q1": np.quantile(arr, 0.25) if len(arr) else np.nan,
                    "q3": np.quantile(arr, 0.75) if len(arr) else np.nan,
                    "max": np.max(arr) if len(arr) else np.nan,
                }
            )

        print(f"[OK] Validated metric: {metric}")

    validation_df = pd.DataFrame(rows)
    distribution_df = pd.DataFrame(distribution_rows)

    validation_df["mannwhitney_fdr_p"] = benjamini_hochberg(
        validation_df["mannwhitney_p"].values
    )
    validation_df["welch_fdr_p"] = benjamini_hochberg(
        validation_df["welch_p"].values
    )
    validation_df["spearman_fdr_p"] = benjamini_hochberg(
        validation_df["spearman_p"].values
    )
    validation_df["point_biserial_fdr_p"] = benjamini_hochberg(
        validation_df["point_biserial_p"].values
    )

    validation_df["cohen_d_magnitude"] = validation_df["cohen_d"].apply(
        effect_size_label_cohen
    )
    validation_df["cliffs_delta_magnitude"] = validation_df["cliffs_delta"].apply(
        effect_size_label_cliffs
    )

    validation_df["evidence_strength"] = validation_df.apply(
        evidence_strength,
        axis=1,
    )

    validation_df["direction"] = np.where(
        validation_df["mean_difference_anomaly_minus_non"] > 0,
        "higher in anomaly group",
        np.where(
            validation_df["mean_difference_anomaly_minus_non"] < 0,
            "lower in anomaly group",
            "no mean difference",
        ),
    )

    validation_df = validation_df.sort_values(
        by=[
            "mannwhitney_fdr_p",
            "mutual_information",
            "cohen_d",
        ],
        ascending=[True, False, False],
    )

    validation_df.insert(0, "rank", range(1, len(validation_df) + 1))

    # =========================================================================
    # SAVE TABLES
    # =========================================================================

    validation_path = OUTPUT_DIR / "studyC_metric_statistical_validation.csv"
    distribution_path = OUTPUT_DIR / "studyC_metric_group_distributions.csv"
    ranking_path = OUTPUT_DIR / "studyC_metric_validity_ranking.csv"

    validation_df.to_csv(validation_path, index=False)
    distribution_df.to_csv(distribution_path, index=False)

    ranking_cols = [
        "rank",
        "metric",
        "direction",
        "mean_non_anomaly",
        "mean_anomaly",
        "mean_difference_anomaly_minus_non",
        "mean_difference_ci_lower",
        "mean_difference_ci_upper",
        "mannwhitney_p",
        "mannwhitney_fdr_p",
        "cohen_d",
        "cohen_d_ci_lower",
        "cohen_d_ci_upper",
        "cohen_d_magnitude",
        "cliffs_delta",
        "cliffs_delta_magnitude",
        "spearman_r",
        "spearman_fdr_p",
        "point_biserial_r",
        "point_biserial_fdr_p",
        "mutual_information",
        "evidence_strength",
        "formula",
    ]

    ranking_df = validation_df[ranking_cols].copy()
    ranking_df.to_csv(ranking_path, index=False)

    excel_path = OUTPUT_DIR / "studyC_statistical_validation_preview.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        validation_df.to_excel(writer, sheet_name="validation_full", index=False)
        ranking_df.to_excel(writer, sheet_name="validity_ranking", index=False)
        distribution_df.to_excel(writer, sheet_name="group_distributions", index=False)

    # =========================================================================
    # FIGURES
    # =========================================================================

    plot_df = ranking_df.sort_values("cohen_d", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(plot_df["metric"], plot_df["cohen_d"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Cohen's d: anomaly minus non-anomaly")
    plt.ylabel("Governance metric")
    plt.title("Study C: Effect Size of Governance Metrics by Anomaly Status")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cohen_d_by_metric.png", dpi=300)
    plt.close()

    plot_df = ranking_df.sort_values("cliffs_delta", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(plot_df["metric"], plot_df["cliffs_delta"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Cliff's delta: anomaly group versus non-anomaly group")
    plt.ylabel("Governance metric")
    plt.title("Study C: Nonparametric Effect Size by Governance Metric")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cliffs_delta_by_metric.png", dpi=300)
    plt.close()

    plot_df = ranking_df.sort_values("mutual_information", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(plot_df["metric"], plot_df["mutual_information"])
    plt.xlabel("Mutual information with anomaly label")
    plt.ylabel("Governance metric")
    plt.title("Study C: Mutual Information of Governance Metrics")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mutual_information_by_metric.png", dpi=300)
    plt.close()

    plot_df = ranking_df.sort_values("spearman_r", ascending=True)

    plt.figure(figsize=(12, 7))
    plt.barh(plot_df["metric"], plot_df["spearman_r"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Spearman correlation with anomaly label")
    plt.ylabel("Governance metric")
    plt.title("Study C: Rank Correlation of Governance Metrics with Anomaly Status")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "spearman_correlation_by_metric.png", dpi=300)
    plt.close()

    # =========================================================================
    # SUMMARY JSON AND TEXT REPORT
    # =========================================================================

    significant_count = int((ranking_df["mannwhitney_fdr_p"] < 0.05).sum())
    moderate_or_strong_count = int(
        ranking_df["evidence_strength"].isin(
            [
                "strong empirical validity evidence",
                "moderate empirical validity evidence",
            ]
        ).sum()
    )

    summary = {
        "script": "06C_governance_metric_statistical_validation.py",
        "objective": "Statistically validate governance metrics against fiscal anomaly labels without training prediction models.",
        "dataset_path": str(DATASET_PATH),
        "leakage_audit_path": str(LEAKAGE_AUDIT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "observations": int(len(df)),
        "positive_anomalies": int(df[TARGET_COL].sum()),
        "negative_anomalies": int((1 - df[TARGET_COL]).sum()),
        "governance_metric_count": int(len(governance_metrics)),
        "fdr_significant_metric_count": significant_count,
        "moderate_or_strong_validity_metric_count": moderate_or_strong_count,
        "top_metric": ranking_df.iloc[0]["metric"] if len(ranking_df) else None,
        "top_metric_evidence_strength": ranking_df.iloc[0]["evidence_strength"] if len(ranking_df) else None,
    }

    with open(OUTPUT_DIR / "studyC_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    report_lines = []
    report_lines.append("06C GOVERNANCE METRIC STATISTICAL VALIDATION")
    report_lines.append("=" * 80)
    report_lines.append(f"Dataset: {DATASET_PATH}")
    report_lines.append(f"Leakage audit: {LEAKAGE_AUDIT_PATH}")
    report_lines.append(f"Results directory: {OUTPUT_DIR}")
    report_lines.append("-" * 80)
    report_lines.append(f"Observations: {len(df)}")
    report_lines.append(f"Positive anomalies: {int(df[TARGET_COL].sum())}")
    report_lines.append(f"Negative anomalies: {int((1 - df[TARGET_COL]).sum())}")
    report_lines.append(f"Governance metrics evaluated: {len(governance_metrics)}")
    report_lines.append(f"FDR-significant metrics: {significant_count}")
    report_lines.append(f"Moderate/strong validity metrics: {moderate_or_strong_count}")
    report_lines.append("")
    report_lines.append("Metric validity ranking:")
    report_lines.append(
        ranking_df[
            [
                "rank",
                "metric",
                "mean_difference_anomaly_minus_non",
                "mannwhitney_fdr_p",
                "cohen_d",
                "cohen_d_magnitude",
                "cliffs_delta",
                "cliffs_delta_magnitude",
                "spearman_r",
                "mutual_information",
                "evidence_strength",
            ]
        ].to_string(index=False)
    )
    report_lines.append("")
    report_lines.append("Reviewer-facing interpretation:")
    report_lines.append(
        "Study C evaluates construct-level empirical validity of the governance "
        "metrics without relying on predictive model training. Metrics that show "
        "FDR-significant group differences, non-trivial effect sizes, and stable "
        "bootstrap confidence intervals can be defended as empirically meaningful "
        "governance indicators. Metrics with weak or non-significant evidence "
        "should be reported as descriptive governance documentation indicators or "
        "removed from the predictive governance model."
    )

    with open(OUTPUT_DIR / "studyC_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # =========================================================================
    # CONSOLE OUTPUT
    # =========================================================================

    print("-" * 80)
    print(f"[OK] Wrote {validation_path.name}")
    print(f"[OK] Wrote {distribution_path.name}")
    print(f"[OK] Wrote {ranking_path.name}")
    print(f"[OK] Wrote {excel_path.name}")
    print(f"[OK] Wrote figures")
    print(f"[OK] Wrote studyC_summary.json")
    print(f"[OK] Wrote studyC_report.txt")
    print("-" * 80)

    print("Metric validity ranking:")
    print(
        ranking_df[
            [
                "rank",
                "metric",
                "mannwhitney_fdr_p",
                "cohen_d",
                "cohen_d_magnitude",
                "cliffs_delta",
                "cliffs_delta_magnitude",
                "mutual_information",
                "evidence_strength",
            ]
        ].to_string(index=False)
    )
    print("=" * 80)


if __name__ == "__main__":
    main()