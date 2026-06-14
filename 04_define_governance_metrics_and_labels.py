r"""
04_define_governance_metrics_and_labels.py

Purpose
-------
Define formula-based fiscal risk indicators, anomaly labels, and governance-oriented
metrics from the model-ready MoF dataset produced by 03_build_long_format_budget_dataset.py.

This script addresses reviewer concerns by defining:
- target variable and anomaly thresholds,
- fiscal risk score and its components,
- interpretability score,
- governance alignment index,
- traceability index,
- decision usability score,
- temporal analytical consistency,
- threshold sensitivity.

Input folder
------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\03_build_long_format_budget_dataset

Output folder
-------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\04_define_governance_metrics_and_labels
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")
INPUT_DIR = BASE_DIR / "Results" / "03_build_long_format_budget_dataset"
RESULTS_DIR = BASE_DIR / "Results" / "04_define_governance_metrics_and_labels"

CLEANED_LONG_PATH = INPUT_DIR / "cleaned_long_dataset.csv"
WIDE_PATH = INPUT_DIR / "model_ready_wide_dataset.csv"
FEATURE_DICTIONARY_PATH = INPUT_DIR / "model_feature_dictionary.csv"
OBS_COUNTS_PATH = INPUT_DIR / "observation_unit_counts.csv"

VALID_PERIODS = ["Q1", "Q2", "Q3", "Q4", "End-Year/Annual"]
PERIOD_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "End-Year/Annual": 5}
DEFAULT_HIGH_RISK_QUANTILE = 0.75

RISK_WEIGHTS = {
    "revenue_expenditure_imbalance": 0.25,
    "deficit_pressure": 0.20,
    "debt_pressure": 0.20,
    "revenue_composition_pressure": 0.15,
    "expenditure_volatility": 0.20,
}
TRACEABILITY_WEIGHTS = {"source_file_coverage": 0.40, "source_sheet_coverage": 0.30, "record_coverage": 0.30}
INTERPRETABILITY_WEIGHTS = {"known_category_coverage": 0.35, "known_indicator_coverage": 0.35, "feature_nonmissing_coverage": 0.30}
GOVERNANCE_ALIGNMENT_WEIGHTS = {"traceability_index": 0.40, "interpretability_score": 0.35, "audit_readiness_score": 0.25}
DECISION_USABILITY_WEIGHTS = {"governance_alignment_index": 0.40, "risk_signal_completeness": 0.35, "temporal_comparability": 0.25}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).replace("\n", " ").replace("\r", " ").replace("\t", " ")).strip()


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def minmax_scale(series: pd.Series) -> pd.Series:
    x = safe_numeric(series)
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    mn = x.min(skipna=True)
    mx = x.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - mn) / (mx - mn)).clip(0, 1).fillna(0)


def ratio_safe(num: pd.Series, den: pd.Series) -> pd.Series:
    out = safe_numeric(num) / safe_numeric(den).replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)


def make_safe_feature_name(s: str) -> str:
    s = clean_text(s).lower().replace("/", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "unknown"


def find_first_existing_column(df: pd.DataFrame, contains_all: List[str], contains_any: Optional[List[str]] = None) -> Optional[str]:
    contains_any = contains_any or []
    candidates = []
    for c in df.columns:
        low = c.lower()
        if all(k.lower() in low for k in contains_all):
            if not contains_any or any(k.lower() in low for k in contains_any):
                candidates.append(c)
    if not candidates:
        return None
    for priority in ["value_mean", "value_median", "category_value_mean", "value_sum", "category_value_sum"]:
        for c in candidates:
            if priority in c.lower():
                return c
    return candidates[0]


def get_col_or_nan(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return safe_numeric(df[col])


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    protected = {"observation_id", "year", "period", "period_order"}
    return [c for c in df.columns if c not in protected and pd.api.types.is_numeric_dtype(df[c])]


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [str(p) for p in [CLEANED_LONG_PATH, WIDE_PATH, FEATURE_DICTIONARY_PATH, OBS_COUNTS_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Required files are missing. Run 03_build_long_format_budget_dataset.py first:\n" + "\n".join(missing))
    return (
        pd.read_csv(CLEANED_LONG_PATH),
        pd.read_csv(WIDE_PATH),
        pd.read_csv(FEATURE_DICTIONARY_PATH),
        pd.read_csv(OBS_COUNTS_PATH),
    )


def filter_valid_reporting_periods(cleaned_long: pd.DataFrame, wide: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    excluded_long = cleaned_long.loc[~cleaned_long["period"].isin(VALID_PERIODS)].copy()
    cleaned_valid = cleaned_long.loc[cleaned_long["period"].isin(VALID_PERIODS)].copy()
    wide_valid = wide.loc[wide["period"].isin(VALID_PERIODS)].copy()
    cleaned_valid["period_order"] = cleaned_valid["period"].map(PERIOD_ORDER)
    wide_valid["period_order"] = wide_valid["period"].map(PERIOD_ORDER)
    return (
        cleaned_valid.sort_values(["year", "period_order"]).reset_index(drop=True),
        wide_valid.sort_values(["year", "period_order"]).reset_index(drop=True),
        excluded_long,
    )


def identify_core_columns(wide: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "revenue": find_first_existing_column(wide, ["revenue"], ["total_revenue", "revenue__revenue", "category__revenue"]),
        "oil_revenue": find_first_existing_column(wide, ["oil"], ["oil_revenue"]),
        "non_oil_revenue": find_first_existing_column(wide, ["non_oil"], ["non_oil_revenue"]),
        "expenditure": find_first_existing_column(wide, ["expenditure"], ["total_expenditure", "category__expenditure", "expenditure__expenditure"]),
        "surplus_deficit": find_first_existing_column(wide, ["surplus"], ["deficit", "surplus_deficit", "category__surplus"]),
        "debt": find_first_existing_column(wide, ["debt"], ["public_debt", "category__debt", "debt__debt"]),
    }


def compute_fiscal_risk_components(wide: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    out = wide.copy().sort_values(["year", "period_order"]).reset_index(drop=True)
    cols = identify_core_columns(out)

    revenue = get_col_or_nan(out, cols["revenue"])
    oil_revenue = get_col_or_nan(out, cols["oil_revenue"])
    non_oil_revenue = get_col_or_nan(out, cols["non_oil_revenue"])
    expenditure = get_col_or_nan(out, cols["expenditure"])
    surplus_deficit = get_col_or_nan(out, cols["surplus_deficit"])
    debt = get_col_or_nan(out, cols["debt"])

    imbalance = ratio_safe((expenditure - revenue).abs(), revenue.abs())
    out["risk_component_revenue_expenditure_imbalance_raw"] = imbalance
    out["risk_component_revenue_expenditure_imbalance"] = minmax_scale(imbalance)

    fallback_deficit = revenue - expenditure
    deficit_amount = surplus_deficit.where(surplus_deficit.notna(), fallback_deficit)
    deficit_pressure = (-deficit_amount).clip(lower=0)
    deficit_ratio = ratio_safe(deficit_pressure, revenue.abs())
    out["risk_component_deficit_pressure_raw"] = deficit_ratio
    out["risk_component_deficit_pressure"] = minmax_scale(deficit_ratio)

    debt_ratio = ratio_safe(debt.abs(), revenue.abs())
    out["risk_component_debt_pressure_raw"] = debt_ratio
    out["risk_component_debt_pressure"] = minmax_scale(debt_ratio)

    oil_share = ratio_safe(oil_revenue.abs(), revenue.abs())
    non_oil_share = ratio_safe(non_oil_revenue.abs(), revenue.abs())
    composition_pressure = oil_share.fillna(0) - non_oil_share.fillna(0)
    out["oil_revenue_share"] = oil_share
    out["non_oil_revenue_share"] = non_oil_share
    out["risk_component_revenue_composition_pressure_raw"] = composition_pressure
    out["risk_component_revenue_composition_pressure"] = minmax_scale(composition_pressure)

    exp_change = expenditure.pct_change().replace([np.inf, -np.inf], np.nan)
    out["expenditure_qoq_change"] = exp_change
    out["risk_component_expenditure_volatility_raw"] = exp_change.abs()
    out["risk_component_expenditure_volatility"] = minmax_scale(exp_change.abs())

    score = pd.Series(np.zeros(len(out)), index=out.index)
    for name, weight in RISK_WEIGHTS.items():
        score += weight * out[f"risk_component_{name}"]
    out["fiscal_risk_score_0_100"] = (score * 100).clip(0, 100)
    return out, cols


def assign_anomaly_labels(df: pd.DataFrame, high_risk_quantile: float = DEFAULT_HIGH_RISK_QUANTILE) -> pd.DataFrame:
    out = df.copy()
    threshold = out["fiscal_risk_score_0_100"].quantile(high_risk_quantile)
    out["high_risk_threshold_quantile"] = high_risk_quantile
    out["high_risk_threshold_value"] = threshold
    out["fiscal_anomaly_label"] = (out["fiscal_risk_score_0_100"] >= threshold).astype(int)

    for comp in RISK_WEIGHTS.keys():
        col = f"risk_component_{comp}"
        thr = out[col].quantile(high_risk_quantile)
        out[f"label_high_{comp}"] = (out[col] >= thr).astype(int)

    comp_label_cols = [f"label_high_{comp}" for comp in RISK_WEIGHTS.keys()]
    out["multi_component_anomaly_label"] = (out[comp_label_cols].sum(axis=1) >= 2).astype(int)

    q33 = out["fiscal_risk_score_0_100"].quantile(0.33)
    q67 = out["fiscal_risk_score_0_100"].quantile(0.67)
    out["risk_level"] = np.select(
        [out["fiscal_risk_score_0_100"] <= q33, out["fiscal_risk_score_0_100"] <= q67, out["fiscal_risk_score_0_100"] > q67],
        ["low", "moderate", "high"],
        default="unknown",
    )
    return out


def compute_traceability_index(df: pd.DataFrame) -> pd.Series:
    source_files = minmax_scale(df.get("n_source_files", pd.Series(np.zeros(len(df)))))
    source_sheets = minmax_scale(df.get("n_source_sheets", pd.Series(np.zeros(len(df)))))
    records = minmax_scale(df.get("n_long_records", pd.Series(np.zeros(len(df)))))
    return (
        TRACEABILITY_WEIGHTS["source_file_coverage"] * source_files
        + TRACEABILITY_WEIGHTS["source_sheet_coverage"] * source_sheets
        + TRACEABILITY_WEIGHTS["record_coverage"] * records
    ).clip(0, 1)


def compute_interpretability_score(df: pd.DataFrame, cleaned_valid: pd.DataFrame) -> pd.Series:
    if cleaned_valid.empty:
        return pd.Series(np.zeros(len(df)), index=df.index)
    tmp = cleaned_valid.copy()
    tmp["observation_id"] = tmp["year"].astype(str) + "__" + tmp["period"].astype(str).map(make_safe_feature_name)
    tmp["known_category"] = ~tmp["category"].astype(str).str.lower().isin(["other", "", "unknown"])
    tmp["known_indicator"] = ~tmp["normalized_indicator"].astype(str).str.lower().isin(["other", "unknown_feature", "unknown", ""])
    obs_cov = tmp.groupby("observation_id").agg(
        known_category_coverage=("known_category", "mean"),
        known_indicator_coverage=("known_indicator", "mean"),
    ).reset_index()
    merged = df[["observation_id"]].merge(obs_cov, on="observation_id", how="left")
    feature_cols = get_feature_columns(df)
    nonmissing = df[feature_cols].notna().mean(axis=1) if feature_cols else pd.Series(np.zeros(len(df)), index=df.index)
    return (
        INTERPRETABILITY_WEIGHTS["known_category_coverage"] * merged["known_category_coverage"].fillna(0)
        + INTERPRETABILITY_WEIGHTS["known_indicator_coverage"] * merged["known_indicator_coverage"].fillna(0)
        + INTERPRETABILITY_WEIGHTS["feature_nonmissing_coverage"] * nonmissing
    ).clip(0, 1)


def compute_audit_readiness_score(df: pd.DataFrame) -> pd.Series:
    required = ["n_long_records", "n_unique_labels", "n_source_files", "n_source_sheets", "n_categories", "n_indicators"]
    parts = []
    for col in required:
        if col in df.columns:
            parts.append((safe_numeric(df[col]) > 0).astype(float))
        else:
            parts.append(pd.Series(np.zeros(len(df)), index=df.index))
    return pd.concat(parts, axis=1).mean(axis=1).clip(0, 1)


def compute_risk_signal_completeness(df: pd.DataFrame) -> pd.Series:
    parts = []
    for comp in RISK_WEIGHTS.keys():
        col = f"risk_component_{comp}"
        parts.append(df[col].notna().astype(float) if col in df.columns else pd.Series(np.zeros(len(df)), index=df.index))
    return pd.concat(parts, axis=1).mean(axis=1).clip(0, 1)


def compute_temporal_comparability(df: pd.DataFrame) -> pd.Series:
    out = df.copy().sort_values(["year", "period_order"])
    out["has_previous_observation"] = 0.0
    if len(out) > 1:
        out.loc[out.index[1:], "has_previous_observation"] = 1.0
    return out.sort_index()["has_previous_observation"].clip(0, 1)


def compute_governance_metrics(df: pd.DataFrame, cleaned_valid: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["traceability_index_0_1"] = compute_traceability_index(out)
    out["interpretability_score_0_1"] = compute_interpretability_score(out, cleaned_valid)
    out["audit_readiness_score_0_1"] = compute_audit_readiness_score(out)
    out["governance_alignment_index_0_1"] = (
        GOVERNANCE_ALIGNMENT_WEIGHTS["traceability_index"] * out["traceability_index_0_1"]
        + GOVERNANCE_ALIGNMENT_WEIGHTS["interpretability_score"] * out["interpretability_score_0_1"]
        + GOVERNANCE_ALIGNMENT_WEIGHTS["audit_readiness_score"] * out["audit_readiness_score_0_1"]
    ).clip(0, 1)
    out["risk_signal_completeness_0_1"] = compute_risk_signal_completeness(out)
    out["temporal_comparability_0_1"] = compute_temporal_comparability(out)
    out["decision_usability_score_0_1"] = (
        DECISION_USABILITY_WEIGHTS["governance_alignment_index"] * out["governance_alignment_index_0_1"]
        + DECISION_USABILITY_WEIGHTS["risk_signal_completeness"] * out["risk_signal_completeness_0_1"]
        + DECISION_USABILITY_WEIGHTS["temporal_comparability"] * out["temporal_comparability_0_1"]
    ).clip(0, 1)
    out["interpretability_score_1_5"] = 1 + 4 * out["interpretability_score_0_1"]
    out["decision_usability_score_1_5"] = 1 + 4 * out["decision_usability_score_0_1"]
    return out


def compute_temporal_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    out = df.copy().sort_values(["year", "period_order"]).reset_index(drop=True)
    out["risk_score_absolute_change"] = out["fiscal_risk_score_0_100"].diff().abs()
    max_change = out["risk_score_absolute_change"].max(skipna=True)
    if pd.isna(max_change) or max_change == 0:
        out["temporal_consistency_component"] = 1.0
    else:
        out["temporal_consistency_component"] = (1.0 - out["risk_score_absolute_change"] / max_change).clip(0, 1)
    return out, float(out["temporal_consistency_component"].mean(skipna=True))


def build_metric_definitions(core_cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    rows = [
        ["fiscal_risk_score_0_100", "100 × Σ(w_j × normalized_risk_component_j), weights: imbalance=0.25, deficit=0.20, debt=0.20, revenue composition=0.15, expenditure volatility=0.20.", "0-100", "Higher values indicate stronger fiscal risk signal.", "Defines output-generation procedure."],
        ["fiscal_anomaly_label", "1 if fiscal_risk_score_0_100 ≥ empirical 75th percentile; otherwise 0.", "0/1", "Formula-based high-risk anomaly label.", "Defines target variable and anomaly threshold."],
        ["traceability_index_0_1", "0.40 × normalized source-file coverage + 0.30 × normalized source-sheet coverage + 0.30 × normalized long-record coverage.", "0-1", "Higher values indicate stronger documented provenance.", "Defines traceability mathematically."],
        ["interpretability_score_0_1", "0.35 × known category coverage + 0.35 × known indicator coverage + 0.30 × feature non-missing coverage.", "0-1", "Higher values indicate stronger explanation readiness.", "Defines interpretability mathematically."],
        ["interpretability_score_1_5", "1 + 4 × interpretability_score_0_1.", "1-5", "Manuscript-compatible interpretability score.", "Explains the 1-5 scale."],
        ["governance_alignment_index_0_1", "0.40 × traceability_index + 0.35 × interpretability_score + 0.25 × audit_readiness_score.", "0-1", "Governance compatibility, auditability, and explanation readiness.", "Defines governance alignment mathematically."],
        ["decision_usability_score_0_1", "0.40 × governance_alignment_index + 0.35 × risk_signal_completeness + 0.25 × temporal_comparability.", "0-1", "Practical usability for governance-aware decision support.", "Defines decision usability mathematically."],
        ["decision_usability_score_1_5", "1 + 4 × decision_usability_score_0_1.", "1-5", "Manuscript-compatible usability score.", "Explains the 1-5 scale."],
        ["temporal_consistency", "Mean of [1 - absolute risk-score change / maximum absolute risk-score change] across consecutive observations.", "0-1", "Higher values indicate smoother temporal behavior.", "Defines temporal analytical consistency."],
    ]
    for concept, col in core_cols.items():
        rows.append([f"data_column_mapping__{concept}", f"Mapped to source feature column: {col}", "source feature", "Automatic input-feature mapping.", "Documents feature construction."])
    return pd.DataFrame(rows, columns=["metric_name", "formula", "scale", "interpretation", "reviewer_concern_addressed"])


def build_anomaly_label_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        {"label_name": "fiscal_anomaly_label", "positive_class_rule": "fiscal_risk_score_0_100 >= 75th percentile", "negative_class_rule": "fiscal_risk_score_0_100 < 75th percentile", "threshold_type": "empirical quantile", "default_threshold": DEFAULT_HIGH_RISK_QUANTILE, "purpose": "Main binary target."},
        {"label_name": "multi_component_anomaly_label", "positive_class_rule": "At least two of the five risk components exceed their 75th percentile thresholds.", "negative_class_rule": "Fewer than two high component flags.", "threshold_type": "component empirical quantiles", "default_threshold": DEFAULT_HIGH_RISK_QUANTILE, "purpose": "Robustness target."},
        {"label_name": "risk_level", "positive_class_rule": "Low/moderate/high using 33rd and 67th percentile cutoffs.", "negative_class_rule": "Not applicable.", "threshold_type": "tertile thresholds", "default_threshold": "0.33 and 0.67", "purpose": "Ordinal descriptive grouping."},
    ])


def build_metric_summary(df: pd.DataFrame, temporal_consistency: float) -> pd.DataFrame:
    metrics = [
        "fiscal_risk_score_0_100", "traceability_index_0_1", "interpretability_score_0_1",
        "interpretability_score_1_5", "governance_alignment_index_0_1", "decision_usability_score_0_1",
        "decision_usability_score_1_5", "risk_signal_completeness_0_1", "temporal_comparability_0_1",
    ]
    rows = []
    for m in metrics:
        if m in df.columns:
            x = safe_numeric(df[m])
            rows.append({
                "metric": m, "n": int(x.notna().sum()),
                "mean": float(x.mean()) if x.notna().any() else np.nan,
                "std": float(x.std(ddof=1)) if x.notna().sum() > 1 else np.nan,
                "min": float(x.min()) if x.notna().any() else np.nan,
                "q25": float(x.quantile(0.25)) if x.notna().any() else np.nan,
                "median": float(x.median()) if x.notna().any() else np.nan,
                "q75": float(x.quantile(0.75)) if x.notna().any() else np.nan,
                "max": float(x.max()) if x.notna().any() else np.nan,
            })
    rows.append({"metric": "temporal_consistency", "n": int(len(df)), "mean": temporal_consistency, "std": np.nan, "min": np.nan, "q25": np.nan, "median": np.nan, "q75": np.nan, "max": np.nan})
    return pd.DataFrame(rows)


def build_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        thr = df["fiscal_risk_score_0_100"].quantile(q)
        lab = (df["fiscal_risk_score_0_100"] >= thr).astype(int)
        rows.append({"risk_score_quantile_threshold": q, "threshold_value": float(thr), "positive_count": int(lab.sum()), "negative_count": int((1 - lab).sum()), "positive_rate": float(lab.mean())})
    return pd.DataFrame(rows)


def build_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for label in ["fiscal_anomaly_label", "multi_component_anomaly_label", "risk_level"]:
        counts = df[label].value_counts(dropna=False).reset_index()
        counts.columns = ["class_value", "count"]
        counts["label_name"] = label
        counts["percentage"] = counts["count"] / counts["count"].sum()
        parts.append(counts[["label_name", "class_value", "count", "percentage"]])
    return pd.concat(parts, ignore_index=True)


def build_reviewer_methodological_table(df: pd.DataFrame, core_cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"reviewer_issue": "AI model target variable was not defined.", "resolution_in_this_script": "Defined fiscal_anomaly_label using the 75th percentile of fiscal_risk_score_0_100.", "output_file": "governance_labeled_dataset.csv; anomaly_label_definitions.csv"},
        {"reviewer_issue": "Risk indicators and output-generation process were unclear.", "resolution_in_this_script": "Defined five normalized risk components and a weighted composite fiscal_risk_score_0_100.", "output_file": "governance_metric_definitions.csv; governance_scores_by_observation.csv"},
        {"reviewer_issue": "Interpretability and governance metrics were not mathematically formulated.", "resolution_in_this_script": "Defined traceability, interpretability, governance alignment, decision usability, and temporal consistency using explicit formulas.", "output_file": "governance_metric_definitions.csv"},
        {"reviewer_issue": "The 0-1 and 1-5 scales were not explained.", "resolution_in_this_script": "Defined 0-1 indices and converted 1-5 scores using 1 + 4 × score.", "output_file": "governance_metric_definitions.csv"},
        {"reviewer_issue": "No sensitivity analysis for thresholds was provided.", "resolution_in_this_script": "Generated label distributions across risk-score quantile thresholds from 0.60 to 0.90.", "output_file": "governance_threshold_sensitivity.csv"},
        {"reviewer_issue": "Input features were not documented.", "resolution_in_this_script": f"Mapped core fiscal concepts to data columns: {json.dumps(core_cols, ensure_ascii=False)}", "output_file": "governance_metric_definitions.csv"},
        {"reviewer_issue": "Number of observations and labels were unclear.", "resolution_in_this_script": f"Produced {len(df)} valid reporting-period observations with formula-based labels.", "output_file": "label_distribution.csv; governance_summary.json"},
    ])


def write_report(summary: Dict[str, Any], metric_summary: pd.DataFrame, label_distribution: pd.DataFrame) -> None:
    lines = [
        "GOVERNANCE METRICS AND LABELS REPORT",
        "=" * 80,
        f"Run time: {summary['run_time']}",
        f"Input directory: {summary['input_dir']}",
        f"Results directory: {summary['results_dir']}",
        "",
        "1. Observation set",
        "-" * 80,
        f"Input model-ready observations: {summary['input_model_ready_observations']}",
        f"Valid reporting-period observations retained: {summary['valid_reporting_period_observations']}",
        f"Excluded non-standard period observations: {summary['excluded_nonstandard_period_observations']}",
        f"Years retained: {summary['years_retained']}",
        f"Periods retained: {summary['periods_retained']}",
        "",
        "2. Target labels",
        "-" * 80,
        f"Main anomaly threshold quantile: {summary['main_anomaly_threshold_quantile']}",
        f"Main anomaly threshold value: {summary['main_anomaly_threshold_value']:.6f}",
        f"Positive anomaly observations: {summary['positive_anomaly_count']}",
        f"Negative anomaly observations: {summary['negative_anomaly_count']}",
        "",
        "3. Core fiscal column mapping",
        "-" * 80,
    ]
    for k, v in summary["core_column_mapping"].items():
        lines.append(f"{k}: {v}")
    lines += ["", "4. Metric summary", "-" * 80]
    for _, r in metric_summary.iterrows():
        lines.append(f"{r['metric']}: n={r['n']}, mean={r['mean']}, std={r['std']}, min={r['min']}, median={r['median']}, max={r['max']}")
    lines += ["", "5. Label distribution", "-" * 80]
    for _, r in label_distribution.iterrows():
        lines.append(f"{r['label_name']} | {r['class_value']}: count={r['count']}, percentage={r['percentage']:.4f}")
    lines += ["", "6. Next script", "-" * 80, "05_baseline_and_ai_model_experiments.py"]
    (RESULTS_DIR / "governance_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("=" * 80)
    print("04 DEFINE GOVERNANCE METRICS AND LABELS")
    print("=" * 80)
    print(f"Input directory:   {INPUT_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("-" * 80)

    cleaned_long, wide, feature_dict, obs_counts = load_inputs()
    cleaned_valid, wide_valid, excluded_long = filter_valid_reporting_periods(cleaned_long, wide)

    fiscal_df, core_cols = compute_fiscal_risk_components(wide_valid)
    fiscal_df = assign_anomaly_labels(fiscal_df)
    fiscal_df = compute_governance_metrics(fiscal_df, cleaned_valid)
    temporal_df, temporal_consistency = compute_temporal_consistency(fiscal_df)

    metric_defs = build_metric_definitions(core_cols)
    anomaly_defs = build_anomaly_label_definitions()
    metric_summary = build_metric_summary(fiscal_df, temporal_consistency)
    threshold_sensitivity = build_threshold_sensitivity(fiscal_df)
    label_distribution = build_label_distribution(fiscal_df)
    reviewer_table = build_reviewer_methodological_table(fiscal_df, core_cols)

    score_cols = [
        "observation_id", "year", "period", "period_order", "fiscal_risk_score_0_100",
        "fiscal_anomaly_label", "multi_component_anomaly_label", "risk_level",
        "traceability_index_0_1", "interpretability_score_0_1", "interpretability_score_1_5",
        "audit_readiness_score_0_1", "governance_alignment_index_0_1",
        "decision_usability_score_0_1", "decision_usability_score_1_5",
        "risk_signal_completeness_0_1", "temporal_comparability_0_1",
    ]
    score_cols = [c for c in score_cols if c in fiscal_df.columns]
    scores_by_obs = fiscal_df[score_cols].copy()

    fiscal_df.to_csv(RESULTS_DIR / "governance_labeled_dataset.csv", index=False, encoding="utf-8-sig")
    metric_defs.to_csv(RESULTS_DIR / "governance_metric_definitions.csv", index=False, encoding="utf-8-sig")
    anomaly_defs.to_csv(RESULTS_DIR / "anomaly_label_definitions.csv", index=False, encoding="utf-8-sig")
    scores_by_obs.to_csv(RESULTS_DIR / "governance_scores_by_observation.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(RESULTS_DIR / "governance_metric_summary.csv", index=False, encoding="utf-8-sig")
    threshold_sensitivity.to_csv(RESULTS_DIR / "governance_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    temporal_df.to_csv(RESULTS_DIR / "temporal_consistency_results.csv", index=False, encoding="utf-8-sig")
    reviewer_table.to_csv(RESULTS_DIR / "reviewer_methodological_table.csv", index=False, encoding="utf-8-sig")
    label_distribution.to_csv(RESULTS_DIR / "label_distribution.csv", index=False, encoding="utf-8-sig")
    excluded_long.to_csv(RESULTS_DIR / "excluded_nonstandard_period_records.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(RESULTS_DIR / "governance_metrics_and_labels_preview.xlsx", engine="openpyxl") as writer:
        scores_by_obs.to_excel(writer, sheet_name="scores_by_observation", index=False)
        metric_defs.to_excel(writer, sheet_name="metric_definitions", index=False)
        anomaly_defs.to_excel(writer, sheet_name="label_definitions", index=False)
        metric_summary.to_excel(writer, sheet_name="metric_summary", index=False)
        threshold_sensitivity.to_excel(writer, sheet_name="threshold_sensitivity", index=False)
        label_distribution.to_excel(writer, sheet_name="label_distribution", index=False)
        reviewer_table.to_excel(writer, sheet_name="reviewer_table", index=False)

    threshold_value = float(fiscal_df["high_risk_threshold_value"].iloc[0]) if len(fiscal_df) else float("nan")
    positive_count = int(fiscal_df["fiscal_anomaly_label"].sum()) if len(fiscal_df) else 0
    negative_count = int(len(fiscal_df) - positive_count)

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(INPUT_DIR),
        "results_dir": str(RESULTS_DIR),
        "input_model_ready_observations": int(len(wide)),
        "valid_reporting_period_observations": int(len(fiscal_df)),
        "excluded_nonstandard_period_observations": int(len(wide) - len(wide_valid)),
        "excluded_nonstandard_long_records": int(len(excluded_long)),
        "years_retained": sorted(fiscal_df["year"].dropna().astype(int).unique().tolist()) if len(fiscal_df) else [],
        "periods_retained": sorted(fiscal_df["period"].dropna().astype(str).unique().tolist(), key=lambda x: PERIOD_ORDER.get(x, 99)) if len(fiscal_df) else [],
        "main_anomaly_threshold_quantile": DEFAULT_HIGH_RISK_QUANTILE,
        "main_anomaly_threshold_value": threshold_value,
        "positive_anomaly_count": positive_count,
        "negative_anomaly_count": negative_count,
        "temporal_consistency_0_1": temporal_consistency,
        "core_column_mapping": core_cols,
        "risk_weights": RISK_WEIGHTS,
        "traceability_weights": TRACEABILITY_WEIGHTS,
        "interpretability_weights": INTERPRETABILITY_WEIGHTS,
        "governance_alignment_weights": GOVERNANCE_ALIGNMENT_WEIGHTS,
        "decision_usability_weights": DECISION_USABILITY_WEIGHTS,
    }
    with (RESULTS_DIR / "governance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_report(summary, metric_summary, label_distribution)

    print("[OK] Wrote governance_labeled_dataset.csv")
    print("[OK] Wrote governance_metric_definitions.csv")
    print("[OK] Wrote anomaly_label_definitions.csv")
    print("[OK] Wrote governance_scores_by_observation.csv")
    print("[OK] Wrote governance_metric_summary.csv")
    print("[OK] Wrote governance_threshold_sensitivity.csv")
    print("[OK] Wrote temporal_consistency_results.csv")
    print("[OK] Wrote reviewer_methodological_table.csv")
    print("[OK] Wrote label_distribution.csv")
    print("[OK] Wrote excluded_nonstandard_period_records.csv")
    print("[OK] Wrote governance_metrics_and_labels_preview.xlsx")
    print("[OK] Wrote governance_summary.json")
    print("[OK] Wrote governance_report.txt")
    print("-" * 80)
    print(f"Valid observations retained: {len(fiscal_df)}")
    print(f"Positive anomaly count: {positive_count}")
    print(f"Negative anomaly count: {negative_count}")
    print(f"Anomaly threshold value: {threshold_value:.6f}")
    print(f"Temporal consistency: {temporal_consistency:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
