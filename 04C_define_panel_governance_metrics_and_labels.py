r"""
04C_define_panel_governance_metrics_and_labels.py

Purpose
-------
Define formula-based fiscal-risk labels and governance-oriented metrics on the
expanded panel dataset produced by:

    04B_dataset_expansion_and_panel_construction_1.py
    or
    04B_dataset_expansion_and_panel_construction.py

This script replaces the earlier 16-observation governance-labeling design with a
larger and more defensible panel design.

Input folder
------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\04B_dataset_expansion_and_panel_construction

Main inputs
-----------
1. expanded_panel_dataset.csv
2. expanded_panel_feature_matrix.csv
3. expanded_panel_feature_dictionary.csv
4. expanded_panel_entity_dictionary.csv

Output folder
-------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\04C_define_panel_governance_metrics_and_labels

Outputs
-------
1. panel_governance_labeled_dataset.csv
2. panel_governance_scores_by_observation.csv
3. panel_governance_metric_definitions.csv
4. panel_anomaly_label_definitions.csv
5. panel_label_distribution.csv
6. panel_governance_metric_summary.csv
7. panel_threshold_sensitivity.csv
8. panel_temporal_consistency_by_entity.csv
9. panel_reviewer_methodological_table.csv
10. panel_governance_summary.json
11. panel_governance_report.txt
12. panel_governance_metrics_preview.xlsx

Observation unit
----------------
year × period × panel_entity

Main target label
-----------------
panel_fiscal_anomaly_label:
    1 if panel_fiscal_risk_score_0_100 >= 75th percentile
    0 otherwise

Scientific note
---------------
This script defines labels and metrics only. It does not train a model. The output is
intended to be used by the next experiment script for baseline, standalone AI, and
governance-centered model comparison.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")

INPUT_DIR = BASE_DIR / "Results" / "04B_dataset_expansion_and_panel_construction"
RESULTS_DIR = BASE_DIR / "Results" / "04C_define_panel_governance_metrics_and_labels"

PANEL_DATASET_PATH = INPUT_DIR / "expanded_panel_dataset.csv"
FEATURE_MATRIX_PATH = INPUT_DIR / "expanded_panel_feature_matrix.csv"
FEATURE_DICTIONARY_PATH = INPUT_DIR / "expanded_panel_feature_dictionary.csv"
ENTITY_DICTIONARY_PATH = INPUT_DIR / "expanded_panel_entity_dictionary.csv"

PERIOD_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
    "End-Year/Annual": 5,
    "Mid-Year/H1": 6,
}

MAIN_ANOMALY_QUANTILE = 0.75

# Weighted risk components. These are formula-based and fixed before modeling.
RISK_WEIGHTS = {
    "entity_deviation": 0.30,
    "period_deviation": 0.20,
    "temporal_change": 0.20,
    "rolling_instability": 0.15,
    "source_uncertainty": 0.15,
}

# Governance scores are formula-based proxies from traceability, completeness, and interpretability.
TRACEABILITY_WEIGHTS = {
    "source_file_support": 0.35,
    "source_sheet_support": 0.35,
    "source_value_count": 0.30,
}

INTERPRETABILITY_WEIGHTS = {
    "core_indicator_flag": 0.25,
    "known_category_flag": 0.25,
    "known_indicator_flag": 0.25,
    "nonmissing_feature_coverage": 0.25,
}

GOVERNANCE_ALIGNMENT_WEIGHTS = {
    "traceability_index": 0.35,
    "interpretability_index": 0.35,
    "audit_readiness_index": 0.30,
}

DECISION_USABILITY_WEIGHTS = {
    "governance_alignment_index": 0.35,
    "risk_signal_completeness": 0.35,
    "temporal_support_index": 0.30,
}


# =============================================================================
# Utility functions
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def minmax_scale(s: pd.Series) -> pd.Series:
    x = safe_numeric(s)
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    mn = x.min(skipna=True)
    mx = x.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - mn) / (mx - mn)).clip(0, 1).fillna(0)


def inverse_minmax_scale(s: pd.Series) -> pd.Series:
    return 1 - minmax_scale(s)


def clipped_abs(s: pd.Series) -> pd.Series:
    return safe_numeric(s).abs().fillna(0)


def get_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return safe_numeric(df[col]).fillna(default)
    return pd.Series(np.full(len(df), default), index=df.index)


def feature_nonmissing_coverage(df: pd.DataFrame) -> pd.Series:
    excluded = {
        "panel_observation_id", "year", "period", "period_order",
        "year_period_key", "category", "normalized_indicator",
        "panel_entity", "unit_mode"
    }
    candidate_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    if not candidate_cols:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[candidate_cols].notna().mean(axis=1).fillna(0)


def temporal_support_index(df: pd.DataFrame) -> pd.Series:
    support_cols = [
        "value_lag_1",
        "value_change_lag_1",
        "value_pct_change_lag_1",
        "value_rolling_mean_2",
        "value_rolling_std_2",
        "value_rolling_mean_4",
        "value_rolling_std_4",
    ]
    cols = [c for c in support_cols if c in df.columns]
    if not cols:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[cols].notna().mean(axis=1).fillna(0)


# =============================================================================
# Load inputs
# =============================================================================

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [
        str(p) for p in [
            PANEL_DATASET_PATH,
            FEATURE_MATRIX_PATH,
            FEATURE_DICTIONARY_PATH,
            ENTITY_DICTIONARY_PATH,
        ]
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required input files are missing. Run 04B_dataset_expansion_and_panel_construction first.\n"
            + "\n".join(missing)
        )

    panel = pd.read_csv(PANEL_DATASET_PATH)
    feature_matrix = pd.read_csv(FEATURE_MATRIX_PATH)
    feature_dict = pd.read_csv(FEATURE_DICTIONARY_PATH)
    entity_dict = pd.read_csv(ENTITY_DICTIONARY_PATH)

    return panel, feature_matrix, feature_dict, entity_dict


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["panel_observation_id", "year", "period", "period_order", "year_period_key",
                "category", "normalized_indicator", "panel_entity"]:
        if col not in out.columns:
            out[col] = ""

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["period_order"] = pd.to_numeric(out["period_order"], errors="coerce")
    out = out.dropna(subset=["year", "period_order"]).copy()
    out["year"] = out["year"].astype(int)
    out["period_order"] = out["period_order"].astype(int)

    out["category"] = out["category"].map(clean_text)
    out["normalized_indicator"] = out["normalized_indicator"].map(clean_text)
    out["panel_entity"] = out["panel_entity"].map(clean_text)

    out = out.sort_values(["panel_entity", "year", "period_order"]).reset_index(drop=True)

    return out


# =============================================================================
# Risk score and labels
# =============================================================================

def compute_panel_risk_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Component 1: within-entity deviation.
    out["risk_component_entity_deviation_raw"] = clipped_abs(get_series(out, "value_abs_robust_z_by_entity"))
    out["risk_component_entity_deviation"] = minmax_scale(out["risk_component_entity_deviation_raw"])

    # Component 2: within-period cross-sectional deviation.
    out["risk_component_period_deviation_raw"] = clipped_abs(get_series(out, "value_robust_z_by_period"))
    out["risk_component_period_deviation"] = minmax_scale(out["risk_component_period_deviation_raw"])

    # Component 3: temporal change.
    pct_change = clipped_abs(get_series(out, "value_pct_change_lag_1"))
    abs_change = clipped_abs(get_series(out, "value_change_lag_1"))
    out["risk_component_temporal_change_raw"] = 0.5 * minmax_scale(pct_change) + 0.5 * minmax_scale(abs_change)
    out["risk_component_temporal_change"] = minmax_scale(out["risk_component_temporal_change_raw"])

    # Component 4: rolling instability.
    rolling_std2 = clipped_abs(get_series(out, "value_rolling_std_2"))
    rolling_std4 = clipped_abs(get_series(out, "value_rolling_std_4"))
    out["risk_component_rolling_instability_raw"] = 0.5 * minmax_scale(rolling_std2) + 0.5 * minmax_scale(rolling_std4)
    out["risk_component_rolling_instability"] = minmax_scale(out["risk_component_rolling_instability_raw"])

    # Component 5: source uncertainty.
    # Lower source support and lower count imply higher uncertainty.
    n_files = get_series(out, "n_source_files")
    n_sheets = get_series(out, "n_source_sheets")
    v_count = get_series(out, "value_count")
    support_score = (
        0.35 * minmax_scale(n_files)
        + 0.35 * minmax_scale(n_sheets)
        + 0.30 * minmax_scale(v_count)
    )
    out["risk_component_source_uncertainty_raw"] = 1 - support_score
    out["risk_component_source_uncertainty"] = out["risk_component_source_uncertainty_raw"].clip(0, 1).fillna(0)

    # Weighted risk score.
    score = pd.Series(np.zeros(len(out)), index=out.index)
    for component, weight in RISK_WEIGHTS.items():
        score += weight * out[f"risk_component_{component}"]

    out["panel_fiscal_risk_score_0_100"] = (100 * score).clip(0, 100)

    return out


def assign_panel_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    main_thr = out["panel_fiscal_risk_score_0_100"].quantile(MAIN_ANOMALY_QUANTILE)
    out["panel_anomaly_threshold_quantile"] = MAIN_ANOMALY_QUANTILE
    out["panel_anomaly_threshold_value"] = main_thr
    out["panel_fiscal_anomaly_label"] = (out["panel_fiscal_risk_score_0_100"] >= main_thr).astype(int)

    # Alternative threshold labels for robustness.
    for q in [0.60, 0.70, 0.80, 0.90]:
        thr = out["panel_fiscal_risk_score_0_100"].quantile(q)
        out[f"panel_fiscal_anomaly_label_q{int(q*100)}"] = (out["panel_fiscal_risk_score_0_100"] >= thr).astype(int)

    # Multi-component anomaly.
    component_flag_cols = []
    for comp in RISK_WEIGHTS.keys():
        col = f"risk_component_{comp}"
        thr = out[col].quantile(MAIN_ANOMALY_QUANTILE)
        flag_col = f"label_high_{comp}"
        out[flag_col] = (out[col] >= thr).astype(int)
        component_flag_cols.append(flag_col)

    out["panel_multi_component_anomaly_label"] = (out[component_flag_cols].sum(axis=1) >= 2).astype(int)

    # Risk levels using tertiles.
    q33 = out["panel_fiscal_risk_score_0_100"].quantile(0.33)
    q67 = out["panel_fiscal_risk_score_0_100"].quantile(0.67)
    out["panel_risk_level"] = np.select(
        [
            out["panel_fiscal_risk_score_0_100"] <= q33,
            out["panel_fiscal_risk_score_0_100"] <= q67,
            out["panel_fiscal_risk_score_0_100"] > q67,
        ],
        ["low", "moderate", "high"],
        default="unknown",
    )

    return out


# =============================================================================
# Governance metrics
# =============================================================================

def compute_panel_governance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    n_files = get_series(out, "n_source_files")
    n_sheets = get_series(out, "n_source_sheets")
    v_count = get_series(out, "value_count")

    out["panel_traceability_index_0_1"] = (
        TRACEABILITY_WEIGHTS["source_file_support"] * minmax_scale(n_files)
        + TRACEABILITY_WEIGHTS["source_sheet_support"] * minmax_scale(n_sheets)
        + TRACEABILITY_WEIGHTS["source_value_count"] * minmax_scale(v_count)
    ).clip(0, 1)

    known_category = (~out["category"].str.lower().isin(["", "other", "unknown"])).astype(float)
    known_indicator = (~out["normalized_indicator"].str.lower().isin(["", "other", "unknown"])).astype(float)
    core_indicator = get_series(out, "is_core_indicator").clip(0, 1)
    nonmissing = feature_nonmissing_coverage(out)

    out["panel_interpretability_index_0_1"] = (
        INTERPRETABILITY_WEIGHTS["core_indicator_flag"] * core_indicator
        + INTERPRETABILITY_WEIGHTS["known_category_flag"] * known_category
        + INTERPRETABILITY_WEIGHTS["known_indicator_flag"] * known_indicator
        + INTERPRETABILITY_WEIGHTS["nonmissing_feature_coverage"] * nonmissing
    ).clip(0, 1)

    # Audit readiness: enough source support + category/indicator descriptors + numeric value.
    has_value = out["value_mean"].notna().astype(float) if "value_mean" in out.columns else pd.Series(np.zeros(len(out)), index=out.index)
    has_sources = ((n_files > 0) & (n_sheets > 0) & (v_count > 0)).astype(float)
    out["panel_audit_readiness_index_0_1"] = (
        0.35 * has_value
        + 0.35 * has_sources
        + 0.15 * known_category
        + 0.15 * known_indicator
    ).clip(0, 1)

    out["panel_governance_alignment_index_0_1"] = (
        GOVERNANCE_ALIGNMENT_WEIGHTS["traceability_index"] * out["panel_traceability_index_0_1"]
        + GOVERNANCE_ALIGNMENT_WEIGHTS["interpretability_index"] * out["panel_interpretability_index_0_1"]
        + GOVERNANCE_ALIGNMENT_WEIGHTS["audit_readiness_index"] * out["panel_audit_readiness_index_0_1"]
    ).clip(0, 1)

    component_cols = [f"risk_component_{c}" for c in RISK_WEIGHTS.keys()]
    available_components = [out[c].notna().astype(float) for c in component_cols if c in out.columns]
    if available_components:
        out["panel_risk_signal_completeness_0_1"] = pd.concat(available_components, axis=1).mean(axis=1).clip(0, 1)
    else:
        out["panel_risk_signal_completeness_0_1"] = 0.0

    out["panel_temporal_support_index_0_1"] = temporal_support_index(out)

    out["panel_decision_usability_index_0_1"] = (
        DECISION_USABILITY_WEIGHTS["governance_alignment_index"] * out["panel_governance_alignment_index_0_1"]
        + DECISION_USABILITY_WEIGHTS["risk_signal_completeness"] * out["panel_risk_signal_completeness_0_1"]
        + DECISION_USABILITY_WEIGHTS["temporal_support_index"] * out["panel_temporal_support_index_0_1"]
    ).clip(0, 1)

    # Manuscript-compatible 1-5 scales.
    out["panel_interpretability_score_1_5"] = 1 + 4 * out["panel_interpretability_index_0_1"]
    out["panel_decision_usability_score_1_5"] = 1 + 4 * out["panel_decision_usability_index_0_1"]

    return out


def compute_temporal_consistency_by_entity(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    out = df.copy()
    out = out.sort_values(["panel_entity", "year", "period_order"]).reset_index(drop=True)

    out["risk_score_abs_change_within_entity"] = (
        out.groupby("panel_entity")["panel_fiscal_risk_score_0_100"]
        .diff()
        .abs()
    )

    max_change_by_entity = out.groupby("panel_entity")["risk_score_abs_change_within_entity"].transform("max")
    out["temporal_consistency_component"] = 1 - (
        out["risk_score_abs_change_within_entity"] / max_change_by_entity.replace({0: np.nan})
    )
    out["temporal_consistency_component"] = out["temporal_consistency_component"].clip(0, 1)

    # If entity has only one point or no change, set neutral high consistency.
    out["temporal_consistency_component"] = out["temporal_consistency_component"].fillna(1.0)

    entity_summary = (
        out.groupby("panel_entity", dropna=False)
        .agg(
            n_observations=("panel_observation_id", "count"),
            mean_temporal_consistency=("temporal_consistency_component", "mean"),
            mean_risk_score=("panel_fiscal_risk_score_0_100", "mean"),
            max_risk_score=("panel_fiscal_risk_score_0_100", "max"),
            anomaly_count=("panel_fiscal_anomaly_label", "sum"),
        )
        .reset_index()
        .sort_values(["mean_temporal_consistency", "n_observations"], ascending=[True, False])
    )

    overall = float(out["temporal_consistency_component"].mean()) if len(out) else float("nan")

    return entity_summary, overall


# =============================================================================
# Definition and summary outputs
# =============================================================================

def build_metric_definitions() -> pd.DataFrame:
    rows = [
        {
            "metric_name": "panel_fiscal_risk_score_0_100",
            "formula": "100 × [0.30×entity_deviation + 0.20×period_deviation + 0.20×temporal_change + 0.15×rolling_instability + 0.15×source_uncertainty]",
            "scale": "0-100",
            "interpretation": "Higher values indicate stronger panel-level fiscal anomaly/risk signal.",
            "reviewer_issue_addressed": "Defines output-generation process and target risk score."
        },
        {
            "metric_name": "panel_fiscal_anomaly_label",
            "formula": "1 if panel_fiscal_risk_score_0_100 ≥ empirical 75th percentile; otherwise 0.",
            "scale": "binary 0/1",
            "interpretation": "Main supervised-learning target for later experiments.",
            "reviewer_issue_addressed": "Defines risk labels and anomaly thresholds."
        },
        {
            "metric_name": "risk_component_entity_deviation",
            "formula": "Min-max normalized absolute robust z-score of value_mean within the same panel_entity.",
            "scale": "0-1",
            "interpretation": "Detects unusual values relative to the same fiscal entity over time.",
            "reviewer_issue_addressed": "Defines anomaly component mathematically."
        },
        {
            "metric_name": "risk_component_period_deviation",
            "formula": "Min-max normalized absolute robust z-score of value_mean within the same year-period cross-section.",
            "scale": "0-1",
            "interpretation": "Detects unusual values relative to other fiscal entities in the same reporting period.",
            "reviewer_issue_addressed": "Defines anomaly component mathematically."
        },
        {
            "metric_name": "risk_component_temporal_change",
            "formula": "Min-max normalized average of absolute lag-1 percentage change and absolute lag-1 value change.",
            "scale": "0-1",
            "interpretation": "Captures abrupt movement relative to prior observation for the same entity.",
            "reviewer_issue_addressed": "Defines temporal anomaly measurement."
        },
        {
            "metric_name": "risk_component_rolling_instability",
            "formula": "Min-max normalized average of rolling standard deviation over 2-period and 4-period windows.",
            "scale": "0-1",
            "interpretation": "Captures short-run instability in the fiscal entity.",
            "reviewer_issue_addressed": "Defines robustness/instability measurement."
        },
        {
            "metric_name": "risk_component_source_uncertainty",
            "formula": "1 - [0.35×normalized source-file count + 0.35×normalized source-sheet count + 0.30×normalized value count].",
            "scale": "0-1",
            "interpretation": "Higher values indicate lower source support and greater data uncertainty.",
            "reviewer_issue_addressed": "Defines traceability-aware uncertainty."
        },
        {
            "metric_name": "panel_traceability_index_0_1",
            "formula": "0.35×normalized source-file count + 0.35×normalized source-sheet count + 0.30×normalized value count.",
            "scale": "0-1",
            "interpretation": "Higher values indicate stronger source provenance.",
            "reviewer_issue_addressed": "Defines traceability index."
        },
        {
            "metric_name": "panel_interpretability_index_0_1",
            "formula": "0.25×core-indicator flag + 0.25×known-category flag + 0.25×known-indicator flag + 0.25×nonmissing feature coverage.",
            "scale": "0-1",
            "interpretation": "Higher values indicate stronger interpretability and semantic clarity.",
            "reviewer_issue_addressed": "Defines interpretability score."
        },
        {
            "metric_name": "panel_governance_alignment_index_0_1",
            "formula": "0.35×traceability index + 0.35×interpretability index + 0.30×audit-readiness index.",
            "scale": "0-1",
            "interpretation": "Higher values indicate stronger governance compatibility.",
            "reviewer_issue_addressed": "Defines governance alignment index."
        },
        {
            "metric_name": "panel_decision_usability_index_0_1",
            "formula": "0.35×governance alignment + 0.35×risk signal completeness + 0.30×temporal support.",
            "scale": "0-1",
            "interpretation": "Higher values indicate stronger practical usability for governance-aware decision support.",
            "reviewer_issue_addressed": "Defines decision usability."
        },
        {
            "metric_name": "panel_interpretability_score_1_5",
            "formula": "1 + 4×panel_interpretability_index_0_1.",
            "scale": "1-5",
            "interpretation": "Converted interpretability score for manuscript reporting.",
            "reviewer_issue_addressed": "Explains the 1-5 scale."
        },
        {
            "metric_name": "panel_decision_usability_score_1_5",
            "formula": "1 + 4×panel_decision_usability_index_0_1.",
            "scale": "1-5",
            "interpretation": "Converted usability score for manuscript reporting.",
            "reviewer_issue_addressed": "Explains the 1-5 scale."
        },
    ]
    return pd.DataFrame(rows)


def build_label_definitions() -> pd.DataFrame:
    rows = [
        {
            "label_name": "panel_fiscal_anomaly_label",
            "positive_rule": "panel_fiscal_risk_score_0_100 >= 75th percentile",
            "negative_rule": "panel_fiscal_risk_score_0_100 < 75th percentile",
            "threshold_type": "empirical quantile",
            "default_threshold": MAIN_ANOMALY_QUANTILE,
            "purpose": "Main target for later classification experiments."
        },
        {
            "label_name": "panel_multi_component_anomaly_label",
            "positive_rule": "At least two risk components exceed their component-specific 75th percentile.",
            "negative_rule": "Fewer than two high component flags.",
            "threshold_type": "component-wise empirical quantile",
            "default_threshold": MAIN_ANOMALY_QUANTILE,
            "purpose": "Alternative stricter anomaly label."
        },
        {
            "label_name": "panel_risk_level",
            "positive_rule": "Low, moderate, high assigned by 33rd and 67th percentiles of panel_fiscal_risk_score_0_100.",
            "negative_rule": "Not applicable.",
            "threshold_type": "tertile",
            "default_threshold": "0.33 and 0.67",
            "purpose": "Ordinal descriptive risk grouping."
        },
    ]
    return pd.DataFrame(rows)


def build_metric_summary(df: pd.DataFrame, overall_temporal_consistency: float) -> pd.DataFrame:
    metric_cols = [
        "panel_fiscal_risk_score_0_100",
        "risk_component_entity_deviation",
        "risk_component_period_deviation",
        "risk_component_temporal_change",
        "risk_component_rolling_instability",
        "risk_component_source_uncertainty",
        "panel_traceability_index_0_1",
        "panel_interpretability_index_0_1",
        "panel_interpretability_score_1_5",
        "panel_audit_readiness_index_0_1",
        "panel_governance_alignment_index_0_1",
        "panel_risk_signal_completeness_0_1",
        "panel_temporal_support_index_0_1",
        "panel_decision_usability_index_0_1",
        "panel_decision_usability_score_1_5",
    ]

    rows = []
    for col in metric_cols:
        if col not in df.columns:
            continue
        x = safe_numeric(df[col])
        rows.append({
            "metric": col,
            "n": int(x.notna().sum()),
            "mean": float(x.mean()) if x.notna().any() else np.nan,
            "std": float(x.std(ddof=1)) if x.notna().sum() > 1 else np.nan,
            "min": float(x.min()) if x.notna().any() else np.nan,
            "q25": float(x.quantile(0.25)) if x.notna().any() else np.nan,
            "median": float(x.median()) if x.notna().any() else np.nan,
            "q75": float(x.quantile(0.75)) if x.notna().any() else np.nan,
            "max": float(x.max()) if x.notna().any() else np.nan,
        })

    rows.append({
        "metric": "overall_temporal_consistency_by_entity",
        "n": int(len(df)),
        "mean": overall_temporal_consistency,
        "std": np.nan,
        "min": np.nan,
        "q25": np.nan,
        "median": np.nan,
        "q75": np.nan,
        "max": np.nan,
    })

    return pd.DataFrame(rows)


def build_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    labels = [
        "panel_fiscal_anomaly_label",
        "panel_multi_component_anomaly_label",
        "panel_risk_level",
        "panel_fiscal_anomaly_label_q60",
        "panel_fiscal_anomaly_label_q70",
        "panel_fiscal_anomaly_label_q80",
        "panel_fiscal_anomaly_label_q90",
    ]

    rows = []
    for label in labels:
        if label not in df.columns:
            continue
        counts = df[label].value_counts(dropna=False).reset_index()
        counts.columns = ["class_value", "count"]
        counts["label_name"] = label
        counts["percentage"] = counts["count"] / counts["count"].sum()
        rows.append(counts[["label_name", "class_value", "count", "percentage"]])

    if not rows:
        return pd.DataFrame(columns=["label_name", "class_value", "count", "percentage"])

    return pd.concat(rows, ignore_index=True)


def build_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        thr = df["panel_fiscal_risk_score_0_100"].quantile(q)
        lab = (df["panel_fiscal_risk_score_0_100"] >= thr).astype(int)
        rows.append({
            "risk_score_quantile_threshold": q,
            "threshold_value": float(thr),
            "positive_count": int(lab.sum()),
            "negative_count": int((1 - lab).sum()),
            "positive_rate": float(lab.mean()),
        })
    return pd.DataFrame(rows)


def build_reviewer_methodological_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "reviewer_issue": "The study did not define the model target variable or anomaly labels.",
            "resolution": "Defined panel_fiscal_anomaly_label using the empirical 75th percentile of a formula-based panel_fiscal_risk_score_0_100.",
            "evidence_file": "panel_anomaly_label_definitions.csv; panel_label_distribution.csv",
        },
        {
            "reviewer_issue": "The dataset size was too small for statistical or AI experiments.",
            "resolution": f"Expanded the observation unit to year × period × panel_entity, producing {len(df)} panel observations.",
            "evidence_file": "panel_governance_labeled_dataset.csv",
        },
        {
            "reviewer_issue": "Governance metrics were not operationally defined.",
            "resolution": "Defined traceability, interpretability, governance alignment, audit readiness, and decision usability with explicit formulas.",
            "evidence_file": "panel_governance_metric_definitions.csv",
        },
        {
            "reviewer_issue": "The 0-1 and 1-5 scales were unclear.",
            "resolution": "Defined all base governance indices on 0-1 scales and converted manuscript scores using 1 + 4 × index.",
            "evidence_file": "panel_governance_metric_definitions.csv",
        },
        {
            "reviewer_issue": "No threshold sensitivity analysis was provided.",
            "resolution": "Generated threshold sensitivity from the 50th to 95th percentile of the panel risk score.",
            "evidence_file": "panel_threshold_sensitivity.csv",
        },
        {
            "reviewer_issue": "Temporal consistency was not defined.",
            "resolution": "Computed entity-level temporal consistency using normalized risk-score change over time.",
            "evidence_file": "panel_temporal_consistency_by_entity.csv",
        },
    ])


def write_report(summary: Dict[str, Any], metric_summary: pd.DataFrame, label_distribution: pd.DataFrame) -> None:
    lines = []
    lines.append("PANEL GOVERNANCE METRICS AND LABELS REPORT")
    lines.append("=" * 80)
    lines.append(f"Run time: {summary['run_time']}")
    lines.append(f"Input directory: {summary['input_dir']}")
    lines.append(f"Results directory: {summary['results_dir']}")
    lines.append("")
    lines.append("1. Observation unit")
    lines.append("-" * 80)
    lines.append(f"Observation unit: {summary['observation_unit']}")
    lines.append(f"Panel observations: {summary['panel_observations']}")
    lines.append(f"Unique panel entities: {summary['unique_panel_entities']}")
    lines.append(f"Unique year-period combinations: {summary['unique_year_periods']}")
    lines.append("")
    lines.append("2. Target labels")
    lines.append("-" * 80)
    lines.append(f"Main anomaly threshold quantile: {summary['main_anomaly_threshold_quantile']}")
    lines.append(f"Main anomaly threshold value: {summary['main_anomaly_threshold_value']:.6f}")
    lines.append(f"Positive anomaly observations: {summary['positive_anomaly_count']}")
    lines.append(f"Negative anomaly observations: {summary['negative_anomaly_count']}")
    lines.append(f"Positive anomaly rate: {summary['positive_anomaly_rate']:.6f}")
    lines.append("")
    lines.append("3. Metric summary")
    lines.append("-" * 80)
    for _, r in metric_summary.iterrows():
        lines.append(
            f"{r['metric']}: n={r['n']}, mean={r['mean']}, std={r['std']}, "
            f"min={r['min']}, median={r['median']}, max={r['max']}"
        )
    lines.append("")
    lines.append("4. Label distribution")
    lines.append("-" * 80)
    for _, r in label_distribution.iterrows():
        lines.append(
            f"{r['label_name']} | {r['class_value']}: count={r['count']}, percentage={r['percentage']:.4f}"
        )
    lines.append("")
    lines.append("5. Next script")
    lines.append("-" * 80)
    lines.append("05_baseline_and_ai_model_experiments.py")
    lines.append("Use panel_governance_labeled_dataset.csv and panel_fiscal_anomaly_label.")

    (RESULTS_DIR / "panel_governance_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("04C DEFINE PANEL GOVERNANCE METRICS AND LABELS")
    print("=" * 80)
    print(f"Input directory:   {INPUT_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("-" * 80)

    panel, feature_matrix, feature_dict, entity_dict = load_inputs()
    fm = prepare_feature_matrix(feature_matrix)

    labeled = compute_panel_risk_components(fm)
    labeled = assign_panel_labels(labeled)
    labeled = compute_panel_governance_metrics(labeled)

    temporal_by_entity, overall_temporal_consistency = compute_temporal_consistency_by_entity(labeled)

    metric_defs = build_metric_definitions()
    label_defs = build_label_definitions()
    metric_summary = build_metric_summary(labeled, overall_temporal_consistency)
    label_distribution = build_label_distribution(labeled)
    threshold_sensitivity = build_threshold_sensitivity(labeled)
    reviewer_table = build_reviewer_methodological_table(labeled)

    score_cols = [
        "panel_observation_id", "year", "period", "period_order", "year_period_key",
        "category", "normalized_indicator", "panel_entity",
        "panel_fiscal_risk_score_0_100", "panel_fiscal_anomaly_label",
        "panel_multi_component_anomaly_label", "panel_risk_level",
        "panel_traceability_index_0_1",
        "panel_interpretability_index_0_1",
        "panel_interpretability_score_1_5",
        "panel_audit_readiness_index_0_1",
        "panel_governance_alignment_index_0_1",
        "panel_risk_signal_completeness_0_1",
        "panel_temporal_support_index_0_1",
        "panel_decision_usability_index_0_1",
        "panel_decision_usability_score_1_5",
    ]
    score_cols = [c for c in score_cols if c in labeled.columns]
    scores_by_obs = labeled[score_cols].copy()

    # Save outputs.
    labeled.to_csv(RESULTS_DIR / "panel_governance_labeled_dataset.csv", index=False, encoding="utf-8-sig")
    scores_by_obs.to_csv(RESULTS_DIR / "panel_governance_scores_by_observation.csv", index=False, encoding="utf-8-sig")
    metric_defs.to_csv(RESULTS_DIR / "panel_governance_metric_definitions.csv", index=False, encoding="utf-8-sig")
    label_defs.to_csv(RESULTS_DIR / "panel_anomaly_label_definitions.csv", index=False, encoding="utf-8-sig")
    label_distribution.to_csv(RESULTS_DIR / "panel_label_distribution.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(RESULTS_DIR / "panel_governance_metric_summary.csv", index=False, encoding="utf-8-sig")
    threshold_sensitivity.to_csv(RESULTS_DIR / "panel_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    temporal_by_entity.to_csv(RESULTS_DIR / "panel_temporal_consistency_by_entity.csv", index=False, encoding="utf-8-sig")
    reviewer_table.to_csv(RESULTS_DIR / "panel_reviewer_methodological_table.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(RESULTS_DIR / "panel_governance_metrics_preview.xlsx", engine="openpyxl") as writer:
        scores_by_obs.head(1500).to_excel(writer, sheet_name="scores_by_observation", index=False)
        metric_defs.to_excel(writer, sheet_name="metric_definitions", index=False)
        label_defs.to_excel(writer, sheet_name="label_definitions", index=False)
        label_distribution.to_excel(writer, sheet_name="label_distribution", index=False)
        metric_summary.to_excel(writer, sheet_name="metric_summary", index=False)
        threshold_sensitivity.to_excel(writer, sheet_name="threshold_sensitivity", index=False)
        temporal_by_entity.to_excel(writer, sheet_name="temporal_consistency", index=False)
        reviewer_table.to_excel(writer, sheet_name="reviewer_table", index=False)

    thr = float(labeled["panel_anomaly_threshold_value"].iloc[0]) if len(labeled) else float("nan")
    pos = int(labeled["panel_fiscal_anomaly_label"].sum()) if len(labeled) else 0
    neg = int(len(labeled) - pos)
    pos_rate = float(pos / len(labeled)) if len(labeled) else float("nan")

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(INPUT_DIR),
        "results_dir": str(RESULTS_DIR),
        "observation_unit": "year × period × panel_entity",
        "panel_observations": int(len(labeled)),
        "unique_panel_entities": int(labeled["panel_entity"].nunique()) if len(labeled) else 0,
        "unique_year_periods": int(labeled["year_period_key"].nunique()) if len(labeled) else 0,
        "years_retained": sorted(labeled["year"].dropna().astype(int).unique().tolist()) if len(labeled) else [],
        "periods_retained": sorted(labeled["period"].dropna().astype(str).unique().tolist(), key=lambda p: PERIOD_ORDER.get(p, 99)) if len(labeled) else [],
        "main_anomaly_threshold_quantile": MAIN_ANOMALY_QUANTILE,
        "main_anomaly_threshold_value": thr,
        "positive_anomaly_count": pos,
        "negative_anomaly_count": neg,
        "positive_anomaly_rate": pos_rate,
        "overall_temporal_consistency_by_entity": overall_temporal_consistency,
        "risk_weights": RISK_WEIGHTS,
        "traceability_weights": TRACEABILITY_WEIGHTS,
        "interpretability_weights": INTERPRETABILITY_WEIGHTS,
        "governance_alignment_weights": GOVERNANCE_ALIGNMENT_WEIGHTS,
        "decision_usability_weights": DECISION_USABILITY_WEIGHTS,
        "outputs": {
            "panel_governance_labeled_dataset": str(RESULTS_DIR / "panel_governance_labeled_dataset.csv"),
            "panel_governance_scores_by_observation": str(RESULTS_DIR / "panel_governance_scores_by_observation.csv"),
            "panel_governance_metric_definitions": str(RESULTS_DIR / "panel_governance_metric_definitions.csv"),
            "panel_anomaly_label_definitions": str(RESULTS_DIR / "panel_anomaly_label_definitions.csv"),
            "panel_label_distribution": str(RESULTS_DIR / "panel_label_distribution.csv"),
            "panel_governance_report": str(RESULTS_DIR / "panel_governance_report.txt"),
        }
    }

    with (RESULTS_DIR / "panel_governance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_report(summary, metric_summary, label_distribution)

    print("[OK] Wrote panel_governance_labeled_dataset.csv")
    print("[OK] Wrote panel_governance_scores_by_observation.csv")
    print("[OK] Wrote panel_governance_metric_definitions.csv")
    print("[OK] Wrote panel_anomaly_label_definitions.csv")
    print("[OK] Wrote panel_label_distribution.csv")
    print("[OK] Wrote panel_governance_metric_summary.csv")
    print("[OK] Wrote panel_threshold_sensitivity.csv")
    print("[OK] Wrote panel_temporal_consistency_by_entity.csv")
    print("[OK] Wrote panel_reviewer_methodological_table.csv")
    print("[OK] Wrote panel_governance_metrics_preview.xlsx")
    print("[OK] Wrote panel_governance_summary.json")
    print("[OK] Wrote panel_governance_report.txt")
    print("-" * 80)
    print(f"Panel observations: {len(labeled)}")
    print(f"Positive anomaly count: {pos}")
    print(f"Negative anomaly count: {neg}")
    print(f"Positive anomaly rate: {pos_rate:.6f}")
    print(f"Anomaly threshold value: {thr:.6f}")
    print(f"Overall temporal consistency: {overall_temporal_consistency:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
