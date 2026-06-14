r"""
04B_dataset_expansion_and_panel_construction.py

Purpose
-------
Construct an expanded panel dataset from the repaired long-format MoF budget data.

Why this script is needed
-------------------------
The repaired observation-level dataset contains only 16 reporting-period observations
(year + period). This is too small to justify supervised AI experiments.

This script expands the analytical unit from:

    year × period

to:

    year × period × fiscal indicator entity

where each fiscal indicator/entity is derived from the repaired long-format MoF data.

This provides a larger panel suitable for:
1. transparent fiscal-risk scoring,
2. anomaly detection,
3. baseline comparisons,
4. cautious supervised modeling,
5. uncertainty and sensitivity analysis.

Input folder
------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\03B_repair_periods_and_feature_mapping

Main input
----------
repaired_cleaned_long_dataset.csv

Output folder
-------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\04B_dataset_expansion_and_panel_construction

Outputs
-------
1. expanded_panel_dataset.csv
2. expanded_panel_feature_matrix.csv
3. expanded_panel_feature_dictionary.csv
4. expanded_panel_entity_dictionary.csv
5. expanded_panel_quality_report.csv
6. expanded_panel_coverage_by_year_period.csv
7. expanded_panel_coverage_by_entity.csv
8. expanded_panel_temporal_features.csv
9. expanded_panel_summary.json
10. expanded_panel_report.txt
11. expanded_panel_preview.xlsx

Observation unit
----------------
year × period × panel_entity

The panel_entity is a normalized fiscal indicator/entity constructed from:
    category + normalized_indicator + cleaned fiscal label

How to run
----------
pip install pandas numpy openpyxl
python 04B_dataset_expansion_and_panel_construction.py

Scientific note
---------------
This script does not train a model. It creates a larger empirical panel so later
scripts can define labels, train baselines, and run validation more defensibly.
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


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")

INPUT_DIR = BASE_DIR / "Results" / "03B_repair_periods_and_feature_mapping"
INPUT_REPAIRED_LONG = INPUT_DIR / "repaired_cleaned_long_dataset.csv"

RESULTS_DIR = BASE_DIR / "Results" / "04B_dataset_expansion_and_panel_construction"

VALID_PERIODS = ["Q1", "Q2", "Q3", "Q4", "End-Year/Annual", "Mid-Year/H1"]

PERIOD_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
    "End-Year/Annual": 5,
    "Mid-Year/H1": 6,
}

CORE_INDICATORS = {
    "total_revenue",
    "oil_revenue",
    "non_oil_revenue",
    "total_expenditure",
    "surplus_deficit",
    "public_debt",
    "domestic_debt",
    "external_debt",
}

# Winsorization limits for robust feature construction.
LOW_Q = 0.01
HIGH_Q = 0.99


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


def make_safe_name(x: Any, max_len: int = 100) -> str:
    s = clean_text(x).lower()
    s = s.replace("/", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "unknown"
    return s[:max_len].strip("_")


def normalize_label_for_entity(label: Any) -> str:
    s = clean_text(label).lower()

    # Remove common bilingual separators and repeated spaces.
    s = s.replace("|", " ")
    s = re.sub(r"\s+", " ", s)

    # Remove very generic fragments.
    generic = [
        "amount", "million sar", "sar million", "actual", "budget",
        "q1", "q2", "q3", "q4", "2021", "2022", "2023", "2024",
        "2025", "2026", "total", "item", "column"
    ]
    for g in generic:
        s = re.sub(rf"\b{re.escape(g)}\b", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        s = "generic"
    return make_safe_name(s, max_len=80)


def coerce_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def winsorize_series(s: pd.Series, low_q: float = LOW_Q, high_q: float = HIGH_Q) -> pd.Series:
    x = coerce_numeric_series(s)
    if x.notna().sum() < 4:
        return x
    lo = x.quantile(low_q)
    hi = x.quantile(high_q)
    return x.clip(lower=lo, upper=hi)


def safe_pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    cur = coerce_numeric_series(current)
    prev = coerce_numeric_series(previous)
    out = (cur - prev) / prev.replace({0: np.nan}).abs()
    return out.replace([np.inf, -np.inf], np.nan)


def robust_z_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    def transform_group(s: pd.Series) -> pd.Series:
        x = coerce_numeric_series(s)
        med = x.median(skipna=True)
        mad = (x - med).abs().median(skipna=True)

        if pd.isna(mad) or mad == 0:
            sd = x.std(skipna=True)
            if pd.isna(sd) or sd == 0:
                return pd.Series(np.zeros(len(x)), index=x.index)
            return ((x - x.mean(skipna=True)) / sd).fillna(0)

        return (0.6745 * (x - med) / mad).fillna(0)

    return df.groupby(group_col, group_keys=False)[value_col].apply(transform_group)


def minmax_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    def transform_group(s: pd.Series) -> pd.Series:
        x = coerce_numeric_series(s)
        mn = x.min(skipna=True)
        mx = x.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series(np.zeros(len(x)), index=x.index)
        return ((x - mn) / (mx - mn)).clip(0, 1).fillna(0)

    return df.groupby(group_col, group_keys=False)[value_col].apply(transform_group)


# =============================================================================
# Loading and panel entity construction
# =============================================================================

def load_repaired_long() -> pd.DataFrame:
    if not INPUT_REPAIRED_LONG.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_REPAIRED_LONG}\n"
            "Run 03B_repair_periods_and_feature_mapping.py first."
        )

    df = pd.read_csv(INPUT_REPAIRED_LONG)
    return df


def prepare_long(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [
        "year", "period", "period_order", "category", "normalized_indicator",
        "label", "value", "unit", "source_file", "sheet_name", "value_column"
    ]

    for col in required:
        if col not in out.columns:
            out[col] = np.nan

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["year", "value"]).copy()
    out["year"] = out["year"].astype(int)

    out["period"] = out["period"].map(clean_text)
    out = out[out["period"].isin(VALID_PERIODS)].copy()
    out["period_order"] = out["period"].map(PERIOD_ORDER).astype(int)

    out["category"] = out["category"].map(clean_text).replace("", "Other")
    out["normalized_indicator"] = out["normalized_indicator"].map(clean_text).replace("", "other")
    out["label"] = out["label"].map(clean_text).replace("", "Unlabeled")
    out["unit"] = out["unit"].map(clean_text).replace("", "unspecified")
    out["source_file"] = out["source_file"].map(clean_text)
    out["sheet_name"] = out["sheet_name"].map(clean_text)
    out["value_column"] = out["value_column"].map(clean_text)

    out["label_entity"] = out["label"].map(normalize_label_for_entity)

    out["panel_entity"] = (
        out["category"].map(make_safe_name)
        + "__"
        + out["normalized_indicator"].map(make_safe_name)
        + "__"
        + out["label_entity"]
    )

    out["panel_entity"] = out["panel_entity"].map(lambda x: make_safe_name(x, max_len=160))

    out["observation_id"] = (
        out["year"].astype(str)
        + "__"
        + out["period"].map(make_safe_name)
        + "__"
        + out["panel_entity"]
    )

    out = out.sort_values(["panel_entity", "year", "period_order", "source_file", "sheet_name"]).reset_index(drop=True)
    return out


# =============================================================================
# Panel aggregation and temporal features
# =============================================================================

def aggregate_to_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long extracted rows into panel observations:
        year × period × panel_entity
    """
    if df.empty:
        return pd.DataFrame()

    group_cols = ["year", "period", "period_order", "category", "normalized_indicator", "panel_entity"]

    panel = (
        df.groupby(group_cols, dropna=False)
        .agg(
            value_mean=("value", "mean"),
            value_median=("value", "median"),
            value_sum=("value", "sum"),
            value_min=("value", "min"),
            value_max=("value", "max"),
            value_std=("value", "std"),
            value_count=("value", "count"),
            n_source_files=("source_file", "nunique"),
            n_source_sheets=("sheet_name", "nunique"),
            n_value_columns=("value_column", "nunique"),
            n_raw_labels=("label", "nunique"),
            unit_mode=("unit", lambda s: s.mode().iloc[0] if not s.mode().empty else "unspecified"),
            example_label=("label", lambda s: " || ".join(s.dropna().astype(str).drop_duplicates().head(3).tolist())),
            source_files=("source_file", lambda s: " || ".join(s.dropna().astype(str).drop_duplicates().head(10).tolist())),
            source_sheets=("sheet_name", lambda s: " || ".join(s.dropna().astype(str).drop_duplicates().head(10).tolist())),
        )
        .reset_index()
    )

    panel["panel_observation_id"] = (
        panel["year"].astype(str)
        + "__"
        + panel["period"].map(make_safe_name)
        + "__"
        + panel["panel_entity"]
    )

    panel = panel.sort_values(["panel_entity", "year", "period_order"]).reset_index(drop=True)

    return panel


def add_temporal_features(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel

    out = panel.copy()
    out = out.sort_values(["panel_entity", "year", "period_order"]).reset_index(drop=True)

    out["value_winsorized"] = winsorize_series(out["value_mean"])

    # Lag features within each entity.
    for lag in [1, 2, 4]:
        out[f"value_lag_{lag}"] = out.groupby("panel_entity")["value_mean"].shift(lag)
        out[f"value_change_lag_{lag}"] = out["value_mean"] - out[f"value_lag_{lag}"]
        out[f"value_pct_change_lag_{lag}"] = safe_pct_change(out["value_mean"], out[f"value_lag_{lag}"])

    # Rolling features.
    out["value_rolling_mean_2"] = (
        out.groupby("panel_entity")["value_mean"]
        .transform(lambda s: s.rolling(window=2, min_periods=1).mean())
    )
    out["value_rolling_std_2"] = (
        out.groupby("panel_entity")["value_mean"]
        .transform(lambda s: s.rolling(window=2, min_periods=2).std())
    )

    out["value_rolling_mean_4"] = (
        out.groupby("panel_entity")["value_mean"]
        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())
    )
    out["value_rolling_std_4"] = (
        out.groupby("panel_entity")["value_mean"]
        .transform(lambda s: s.rolling(window=4, min_periods=2).std())
    )

    # Entity-normalized anomaly-oriented features.
    out["value_robust_z_by_entity"] = robust_z_by_group(out, "panel_entity", "value_mean")
    out["value_abs_robust_z_by_entity"] = out["value_robust_z_by_entity"].abs()
    out["value_minmax_by_entity"] = minmax_by_group(out, "panel_entity", "value_mean")

    # Cross-sectional within-period normalization.
    out["value_robust_z_by_period"] = robust_z_by_group(out, "year_period_key", "value_mean") if "year_period_key" in out.columns else 0

    return out


def add_panel_keys(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["year_period_key"] = out["year"].astype(str) + "__" + out["period"].map(make_safe_name)
    out["is_core_indicator"] = out["normalized_indicator"].isin(CORE_INDICATORS).astype(int)
    out["is_revenue_related"] = out["category"].str.lower().str.contains("revenue", na=False).astype(int)
    out["is_expenditure_related"] = out["category"].str.lower().str.contains("expenditure", na=False).astype(int)
    out["is_debt_related"] = out["category"].str.lower().str.contains("debt", na=False).astype(int)
    out["is_surplus_deficit_related"] = out["category"].str.lower().str.contains("surplus|deficit", regex=True, na=False).astype(int)
    return out


def build_feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric feature matrix suitable for baseline/AI scripts.

    It includes identifiers, labels/categories, and numeric features.
    Labels are NOT created here; this is only a feature table.
    """
    if panel.empty:
        return pd.DataFrame()

    keep_cols = [
        "panel_observation_id",
        "year",
        "period",
        "period_order",
        "year_period_key",
        "category",
        "normalized_indicator",
        "panel_entity",
        "unit_mode",
        "value_mean",
        "value_median",
        "value_sum",
        "value_min",
        "value_max",
        "value_std",
        "value_count",
        "n_source_files",
        "n_source_sheets",
        "n_value_columns",
        "n_raw_labels",
        "value_winsorized",
        "value_lag_1",
        "value_lag_2",
        "value_lag_4",
        "value_change_lag_1",
        "value_change_lag_2",
        "value_change_lag_4",
        "value_pct_change_lag_1",
        "value_pct_change_lag_2",
        "value_pct_change_lag_4",
        "value_rolling_mean_2",
        "value_rolling_std_2",
        "value_rolling_mean_4",
        "value_rolling_std_4",
        "value_robust_z_by_entity",
        "value_abs_robust_z_by_entity",
        "value_minmax_by_entity",
        "value_robust_z_by_period",
        "is_core_indicator",
        "is_revenue_related",
        "is_expenditure_related",
        "is_debt_related",
        "is_surplus_deficit_related",
    ]

    keep_cols = [c for c in keep_cols if c in panel.columns]
    fm = panel[keep_cols].copy()

    # Replace inf in numeric columns.
    for c in fm.columns:
        if pd.api.types.is_numeric_dtype(fm[c]):
            fm[c] = fm[c].replace([np.inf, -np.inf], np.nan)

    return fm


# =============================================================================
# Dictionaries and reports
# =============================================================================

def build_entity_dictionary(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    entity_dict = (
        panel.groupby(["panel_entity", "category", "normalized_indicator"], dropna=False)
        .agg(
            n_observations=("panel_observation_id", "count"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_periods=("period", "nunique"),
            n_source_files=("n_source_files", "sum"),
            example_label=("example_label", lambda s: " || ".join(s.dropna().astype(str).drop_duplicates().head(3).tolist())),
        )
        .reset_index()
        .sort_values(["category", "normalized_indicator", "panel_entity"])
    )

    return entity_dict


def build_feature_dictionary(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    if feature_matrix.empty:
        return pd.DataFrame()

    identifier_cols = {
        "panel_observation_id", "year", "period", "period_order", "year_period_key",
        "category", "normalized_indicator", "panel_entity", "unit_mode"
    }

    rows = []
    for col in feature_matrix.columns:
        if col in identifier_cols:
            role = "identifier_or_descriptor"
        elif col.startswith("is_"):
            role = "binary_descriptor"
        elif "lag" in col:
            role = "temporal_lag_feature"
        elif "rolling" in col:
            role = "temporal_rolling_feature"
        elif "robust_z" in col or "minmax" in col:
            role = "normalized_anomaly_feature"
        elif col.startswith("n_") or col.endswith("_count"):
            role = "coverage_or_traceability_feature"
        elif col.startswith("value_"):
            role = "value_feature"
        else:
            role = "derived_feature"

        numeric = pd.api.types.is_numeric_dtype(feature_matrix[col])

        rows.append({
            "feature_name": col,
            "role": role,
            "numeric": bool(numeric),
            "non_missing_count": int(feature_matrix[col].notna().sum()),
            "missing_count": int(feature_matrix[col].isna().sum()),
            "mean": float(feature_matrix[col].mean()) if numeric and feature_matrix[col].notna().any() else np.nan,
            "std": float(feature_matrix[col].std(ddof=1)) if numeric and feature_matrix[col].notna().sum() > 1 else np.nan,
            "min": float(feature_matrix[col].min()) if numeric and feature_matrix[col].notna().any() else np.nan,
            "max": float(feature_matrix[col].max()) if numeric and feature_matrix[col].notna().any() else np.nan,
            "description": describe_feature(col),
        })

    return pd.DataFrame(rows)


def describe_feature(col: str) -> str:
    descriptions = {
        "panel_observation_id": "Unique panel observation identifier: year + period + panel entity.",
        "year": "Fiscal year.",
        "period": "Fiscal reporting period.",
        "period_order": "Numeric order of fiscal reporting period.",
        "year_period_key": "Fiscal year-period grouping key.",
        "category": "Repaired fiscal category.",
        "normalized_indicator": "Repaired normalized fiscal indicator.",
        "panel_entity": "Fiscal indicator/entity used as panel unit.",
        "value_mean": "Mean extracted numeric value for the entity in a year-period.",
        "value_median": "Median extracted numeric value for the entity in a year-period.",
        "value_sum": "Sum of extracted numeric values for the entity in a year-period.",
        "value_count": "Number of long-format source values aggregated into the panel observation.",
        "n_source_files": "Number of source files supporting the panel observation.",
        "n_source_sheets": "Number of source sheets supporting the panel observation.",
        "value_robust_z_by_entity": "Robust z-score of value_mean relative to the same fiscal entity over time.",
        "value_abs_robust_z_by_entity": "Absolute robust z-score, useful as an anomaly-strength feature.",
        "value_minmax_by_entity": "Min-max normalized value_mean within the same fiscal entity.",
    }
    return descriptions.get(col, "Derived panel feature.")


def build_quality_report(raw: pd.DataFrame, panel: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, value: Any, interpretation: str) -> None:
        rows.append({
            "check": check,
            "value": value,
            "interpretation": interpretation,
        })

    add("input_repaired_long_records", int(len(raw)), "Rows loaded from repaired long-format dataset.")
    add("expanded_panel_observations", int(len(panel)), "Rows in expanded year-period-entity panel dataset.")
    add("feature_matrix_rows", int(len(fm)), "Rows in expanded feature matrix.")
    add("feature_matrix_columns", int(fm.shape[1]) if not fm.empty else 0, "Columns in expanded feature matrix.")

    if not panel.empty:
        add("unique_years", int(panel["year"].nunique()), "Number of fiscal years represented.")
        add("unique_periods", int(panel["period"].nunique()), "Number of fiscal reporting period types represented.")
        add("unique_year_periods", int(panel["year_period_key"].nunique()), "Number of unique fiscal year-period combinations.")
        add("unique_panel_entities", int(panel["panel_entity"].nunique()), "Number of fiscal indicator entities represented.")
        add("unique_categories", int(panel["category"].nunique()), "Number of fiscal categories represented.")
        add("unique_normalized_indicators", int(panel["normalized_indicator"].nunique()), "Number of normalized indicators represented.")
        add("mean_observations_per_entity", float(panel.groupby("panel_entity").size().mean()), "Average temporal coverage per fiscal entity.")
        add("median_observations_per_entity", float(panel.groupby("panel_entity").size().median()), "Median temporal coverage per fiscal entity.")
        add("core_indicator_observations", int(panel["is_core_indicator"].sum()) if "is_core_indicator" in panel else 0, "Panel observations belonging to core fiscal indicators.")

    if not fm.empty:
        numeric_cols = [c for c in fm.columns if pd.api.types.is_numeric_dtype(fm[c])]
        add("numeric_feature_columns", int(len(numeric_cols)), "Numeric columns usable by later statistical/AI scripts.")
        if numeric_cols:
            missing_rate = fm[numeric_cols].isna().mean().mean()
            add("mean_numeric_missing_rate", float(missing_rate), "Average missingness across numeric feature columns.")

    return pd.DataFrame(rows)


def build_coverage_by_year_period(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    return (
        panel.groupby(["year", "period", "period_order"], dropna=False)
        .agg(
            n_panel_observations=("panel_observation_id", "count"),
            n_entities=("panel_entity", "nunique"),
            n_categories=("category", "nunique"),
            n_indicators=("normalized_indicator", "nunique"),
            n_core_indicator_observations=("is_core_indicator", "sum"),
            mean_value_count=("value_count", "mean"),
            mean_source_files=("n_source_files", "mean"),
            mean_source_sheets=("n_source_sheets", "mean"),
        )
        .reset_index()
        .sort_values(["year", "period_order"])
    )


def build_coverage_by_entity(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    return (
        panel.groupby(["panel_entity", "category", "normalized_indicator"], dropna=False)
        .agg(
            n_observations=("panel_observation_id", "count"),
            n_years=("year", "nunique"),
            n_periods=("period", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            mean_value=("value_mean", "mean"),
            std_value=("value_mean", "std"),
            mean_abs_robust_z=("value_abs_robust_z_by_entity", "mean"),
            max_abs_robust_z=("value_abs_robust_z_by_entity", "max"),
            mean_source_files=("n_source_files", "mean"),
            mean_source_sheets=("n_source_sheets", "mean"),
        )
        .reset_index()
        .sort_values(["n_observations", "category", "normalized_indicator"], ascending=[False, True, True])
    )


def write_report(summary: Dict[str, Any], quality: pd.DataFrame) -> None:
    lines = []
    lines.append("EXPANDED PANEL DATASET CONSTRUCTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Run time: {summary['run_time']}")
    lines.append(f"Input file: {summary['input_file']}")
    lines.append(f"Results directory: {summary['results_dir']}")
    lines.append("")
    lines.append("1. Observation unit")
    lines.append("-" * 80)
    lines.append("Expanded panel observation unit: year × period × panel_entity.")
    lines.append("This expands the analytical design beyond year × period only.")
    lines.append("")
    lines.append("2. Main counts")
    lines.append("-" * 80)
    lines.append(f"Input repaired long records: {summary['input_repaired_long_records']}")
    lines.append(f"Expanded panel observations: {summary['expanded_panel_observations']}")
    lines.append(f"Feature matrix rows: {summary['feature_matrix_rows']}")
    lines.append(f"Feature matrix columns: {summary['feature_matrix_columns']}")
    lines.append(f"Numeric feature columns: {summary['numeric_feature_columns']}")
    lines.append("")
    lines.append("3. Coverage")
    lines.append("-" * 80)
    lines.append(f"Years retained: {summary['years_retained']}")
    lines.append(f"Periods retained: {summary['periods_retained']}")
    lines.append(f"Unique year-period combinations: {summary['unique_year_periods']}")
    lines.append(f"Unique panel entities: {summary['unique_panel_entities']}")
    lines.append(f"Categories retained: {summary['categories_retained']}")
    lines.append(f"Indicators retained: {summary['indicators_retained']}")
    lines.append("")
    lines.append("4. Quality checks")
    lines.append("-" * 80)
    if not quality.empty:
        for _, r in quality.iterrows():
            lines.append(f"{r['check']}: {r['value']} -- {r['interpretation']}")
    lines.append("")
    lines.append("5. Reviewer-facing implication")
    lines.append("-" * 80)
    lines.append("The empirical unit is no longer restricted to 16 year-period observations.")
    lines.append("The expanded panel provides multiple fiscal indicator entities within each period,")
    lines.append("which can support more defensible baseline comparisons and cautious anomaly modeling.")
    lines.append("")
    lines.append("6. Next step")
    lines.append("-" * 80)
    lines.append("Write 04C_define_panel_governance_metrics_and_labels.py or update script 04 to use:")
    lines.append(str(RESULTS_DIR / "expanded_panel_feature_matrix.csv"))

    (RESULTS_DIR / "expanded_panel_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("04B DATASET EXPANSION AND PANEL CONSTRUCTION")
    print("=" * 80)
    print(f"Input file:        {INPUT_REPAIRED_LONG}")
    print(f"Results directory: {RESULTS_DIR}")
    print("-" * 80)

    raw = load_repaired_long()
    prepared = prepare_long(raw)
    panel = aggregate_to_panel(prepared)
    panel = add_panel_keys(panel)
    panel = add_temporal_features(panel)

    feature_matrix = build_feature_matrix(panel)
    entity_dictionary = build_entity_dictionary(panel)
    feature_dictionary = build_feature_dictionary(feature_matrix)
    quality_report = build_quality_report(prepared, panel, feature_matrix)
    coverage_year_period = build_coverage_by_year_period(panel)
    coverage_entity = build_coverage_by_entity(panel)

    # Save outputs.
    panel.to_csv(RESULTS_DIR / "expanded_panel_dataset.csv", index=False, encoding="utf-8-sig")
    feature_matrix.to_csv(RESULTS_DIR / "expanded_panel_feature_matrix.csv", index=False, encoding="utf-8-sig")
    feature_dictionary.to_csv(RESULTS_DIR / "expanded_panel_feature_dictionary.csv", index=False, encoding="utf-8-sig")
    entity_dictionary.to_csv(RESULTS_DIR / "expanded_panel_entity_dictionary.csv", index=False, encoding="utf-8-sig")
    quality_report.to_csv(RESULTS_DIR / "expanded_panel_quality_report.csv", index=False, encoding="utf-8-sig")
    coverage_year_period.to_csv(RESULTS_DIR / "expanded_panel_coverage_by_year_period.csv", index=False, encoding="utf-8-sig")
    coverage_entity.to_csv(RESULTS_DIR / "expanded_panel_coverage_by_entity.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(RESULTS_DIR / "expanded_panel_temporal_features.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(RESULTS_DIR / "expanded_panel_preview.xlsx", engine="openpyxl") as writer:
        panel.head(1500).to_excel(writer, sheet_name="panel_dataset_preview", index=False)
        feature_matrix.head(1500).to_excel(writer, sheet_name="feature_matrix_preview", index=False)
        entity_dictionary.to_excel(writer, sheet_name="entity_dictionary", index=False)
        feature_dictionary.to_excel(writer, sheet_name="feature_dictionary", index=False)
        quality_report.to_excel(writer, sheet_name="quality_report", index=False)
        coverage_year_period.to_excel(writer, sheet_name="coverage_year_period", index=False)
        coverage_entity.head(1500).to_excel(writer, sheet_name="coverage_entity", index=False)

    years = sorted(panel["year"].dropna().astype(int).unique().tolist()) if not panel.empty else []
    periods = sorted(panel["period"].dropna().astype(str).unique().tolist(), key=lambda x: PERIOD_ORDER.get(x, 99)) if not panel.empty else []
    categories = sorted(panel["category"].dropna().astype(str).unique().tolist()) if not panel.empty else []
    indicators = sorted(panel["normalized_indicator"].dropna().astype(str).unique().tolist()) if not panel.empty else []
    numeric_cols = [c for c in feature_matrix.columns if pd.api.types.is_numeric_dtype(feature_matrix[c])] if not feature_matrix.empty else []

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(INPUT_REPAIRED_LONG),
        "results_dir": str(RESULTS_DIR),
        "input_repaired_long_records": int(len(raw)),
        "prepared_long_records": int(len(prepared)),
        "expanded_panel_observations": int(len(panel)),
        "feature_matrix_rows": int(len(feature_matrix)),
        "feature_matrix_columns": int(feature_matrix.shape[1]) if not feature_matrix.empty else 0,
        "numeric_feature_columns": int(len(numeric_cols)),
        "years_retained": years,
        "periods_retained": periods,
        "unique_year_periods": int(panel["year_period_key"].nunique()) if not panel.empty else 0,
        "unique_panel_entities": int(panel["panel_entity"].nunique()) if not panel.empty else 0,
        "categories_retained": categories,
        "indicators_retained": indicators,
        "observation_unit": "year × period × panel_entity",
        "outputs": {
            "expanded_panel_dataset": str(RESULTS_DIR / "expanded_panel_dataset.csv"),
            "expanded_panel_feature_matrix": str(RESULTS_DIR / "expanded_panel_feature_matrix.csv"),
            "expanded_panel_feature_dictionary": str(RESULTS_DIR / "expanded_panel_feature_dictionary.csv"),
            "expanded_panel_entity_dictionary": str(RESULTS_DIR / "expanded_panel_entity_dictionary.csv"),
            "expanded_panel_quality_report": str(RESULTS_DIR / "expanded_panel_quality_report.csv"),
            "expanded_panel_coverage_by_year_period": str(RESULTS_DIR / "expanded_panel_coverage_by_year_period.csv"),
            "expanded_panel_coverage_by_entity": str(RESULTS_DIR / "expanded_panel_coverage_by_entity.csv"),
            "expanded_panel_report": str(RESULTS_DIR / "expanded_panel_report.txt"),
        },
    }

    with (RESULTS_DIR / "expanded_panel_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_report(summary, quality_report)

    print("[OK] Wrote expanded_panel_dataset.csv")
    print("[OK] Wrote expanded_panel_feature_matrix.csv")
    print("[OK] Wrote expanded_panel_feature_dictionary.csv")
    print("[OK] Wrote expanded_panel_entity_dictionary.csv")
    print("[OK] Wrote expanded_panel_quality_report.csv")
    print("[OK] Wrote expanded_panel_coverage_by_year_period.csv")
    print("[OK] Wrote expanded_panel_coverage_by_entity.csv")
    print("[OK] Wrote expanded_panel_temporal_features.csv")
    print("[OK] Wrote expanded_panel_preview.xlsx")
    print("[OK] Wrote expanded_panel_summary.json")
    print("[OK] Wrote expanded_panel_report.txt")
    print("-" * 80)
    print(f"Expanded panel observations: {len(panel)}")
    print(f"Feature matrix rows: {len(feature_matrix)}")
    print(f"Feature matrix columns: {feature_matrix.shape[1] if not feature_matrix.empty else 0}")
    print(f"Numeric feature columns: {len(numeric_cols)}")
    print(f"Unique panel entities: {summary['unique_panel_entities']}")
    print(f"Unique year-period combinations: {summary['unique_year_periods']}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
