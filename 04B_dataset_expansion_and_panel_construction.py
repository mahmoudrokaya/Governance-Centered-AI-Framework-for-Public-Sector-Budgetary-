r"""
04B_dataset_expansion_and_panel_construction.py

Purpose
-------
Construct an expanded fiscal panel dataset from the repaired long-format MoF data.

Why this script is needed
-------------------------
Script 03B correctly repaired periods and fiscal indicators, but the observation unit
remained "year + period", producing only 16 model-ready observations.

For reviewer-facing empirical analysis, this is too small for supervised AI or
train/validation/test experiments.

This script expands the observation unit to:

    year + period + fiscal indicator entity

where each fiscal indicator/category/label row becomes a panel entity observed across
time. This produces a larger panel suitable for:

1. descriptive statistical baselines,
2. anomaly detection,
3. supervised classification with caution,
4. rolling/blocked validation,
5. confidence intervals and sensitivity analysis.

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
3. expanded_panel_entity_dictionary.csv
4. expanded_panel_data_quality.csv
5. expanded_panel_observation_counts.csv
6. expanded_panel_temporal_coverage.csv
7. expanded_panel_split_plan.csv
8. expanded_panel_summary.json
9. expanded_panel_report.txt
10. expanded_panel_preview.xlsx

How to run
----------
pip install pandas numpy openpyxl
python 04B_dataset_expansion_and_panel_construction.py

Scientific position
-------------------
This script creates a larger empirical panel dataset. It does not yet define anomaly
labels or train AI models. Labels should be created after this panel is confirmed.
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

# Minimum number of time points for an entity to be retained in model matrix.
MIN_ENTITY_TIME_POINTS = 2

# Cap extreme ratios to avoid unstable derived features.
RATIO_CLIP_ABS = 10.0


# =============================================================================
# Utilities
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_safe_key(x: Any) -> str:
    s = clean_text(x).lower()
    s = s.replace("/", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def robust_z_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    def transform(x: pd.Series) -> pd.Series:
        med = x.median(skipna=True)
        mad = (x - med).abs().median(skipna=True)
        if pd.isna(mad) or mad == 0:
            sd = x.std(skipna=True)
            if pd.isna(sd) or sd == 0:
                return pd.Series(np.zeros(len(x)), index=x.index)
            return (x - x.mean(skipna=True)) / sd
        return 0.6745 * (x - med) / mad

    return df.groupby(group_col, group_keys=False)[value_col].apply(transform)


def pct_change_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    return (
        df.sort_values(["entity_id", "time_index"])
        .groupby(group_col)[value_col]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .clip(-RATIO_CLIP_ABS, RATIO_CLIP_ABS)
    )


def diff_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
    return (
        df.sort_values(["entity_id", "time_index"])
        .groupby(group_col)[value_col]
        .diff()
    )


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# =============================================================================
# Loading and panel entity construction
# =============================================================================

def load_input() -> pd.DataFrame:
    if not INPUT_REPAIRED_LONG.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_REPAIRED_LONG}\n"
            "Run 03B_repair_periods_and_feature_mapping.py first."
        )
    df = pd.read_csv(INPUT_REPAIRED_LONG)
    return df


def prepare_long(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [
        "year", "period", "period_order", "category", "normalized_indicator",
        "label", "value", "unit", "source_file", "sheet_name",
        "source_relative_path", "value_column"
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

    for col in ["category", "normalized_indicator", "label", "unit", "source_file", "sheet_name", "value_column"]:
        out[col] = out[col].map(clean_text)

    # Time index based on year and period order.
    # This preserves the temporal order even if Annual is included.
    out["time_index"] = out["year"] * 10 + out["period_order"]

    return out.sort_values(["year", "period_order", "category", "normalized_indicator", "label"]).reset_index(drop=True)


def build_entity_id(df: pd.DataFrame) -> pd.Series:
    """
    Entity is deliberately defined using category + normalized_indicator + label.
    This creates a panel in which each fiscal line item can be tracked across periods.
    """
    return (
        df["category"].map(make_safe_key)
        + "__"
        + df["normalized_indicator"].map(make_safe_key)
        + "__"
        + df["label"].map(make_safe_key)
    )


def construct_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entity_id"] = build_entity_id(out)
    out["observation_id"] = (
        out["year"].astype(str)
        + "__"
        + out["period"].map(make_safe_key)
        + "__"
        + out["entity_id"]
    )

    # Aggregate duplicate entity-period rows from multiple sheets/sources.
    panel = (
        out.groupby(
            ["observation_id", "entity_id", "year", "period", "period_order", "time_index",
             "category", "normalized_indicator", "label", "unit"],
            dropna=False
        )
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
            source_files=("source_file", lambda x: " | ".join(sorted(set(map(str, x))))[:1000]),
            source_sheets=("sheet_name", lambda x: " | ".join(sorted(set(map(str, x))))[:1000]),
        )
        .reset_index()
    )

    panel["value_std"] = panel["value_std"].fillna(0)

    # Entity coverage.
    entity_counts = panel.groupby("entity_id")["time_index"].nunique()
    panel["entity_time_points"] = panel["entity_id"].map(entity_counts)

    return panel.sort_values(["entity_id", "time_index"]).reset_index(drop=True)


def add_panel_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out = out.sort_values(["entity_id", "time_index"]).reset_index(drop=True)

    # Core temporal features within each entity.
    out["value_lag1"] = out.groupby("entity_id")["value_mean"].shift(1)
    out["value_diff_lag1"] = out["value_mean"] - out["value_lag1"]
    out["value_pct_change_lag1"] = pct_change_by_group(out, "value_mean", "entity_id")

    out["value_rolling_mean_2"] = (
        out.groupby("entity_id")["value_mean"]
        .rolling(window=2, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["value_rolling_std_2"] = (
        out.groupby("entity_id")["value_mean"]
        .rolling(window=2, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    out["entity_robust_z"] = robust_z_by_group(out, "value_mean", "entity_id")
    out["entity_robust_abs_z"] = out["entity_robust_z"].abs()

    # Cross-sectional rank within same year-period.
    out["period_value_rank_pct"] = (
        out.groupby(["year", "period"])["value_mean"]
        .rank(pct=True)
    )

    # Category-level relative magnitude.
    cat_period_sum = (
        out.groupby(["year", "period", "category"])["value_mean"]
        .transform(lambda x: x.abs().sum())
        .replace({0: np.nan})
    )
    out["category_abs_share"] = (out["value_mean"].abs() / cat_period_sum).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Provenance and traceability features.
    out["traceability_record_density"] = out["value_count"]
    out["traceability_source_density"] = out["n_source_files"] + out["n_source_sheets"]

    # Missingness indicators for temporal features.
    out["has_lag1"] = out["value_lag1"].notna().astype(int)
    out["has_pct_change"] = out["value_pct_change_lag1"].notna().astype(int)

    # Fill only derived temporal NAs where appropriate.
    out["value_diff_lag1"] = out["value_diff_lag1"].fillna(0)
    out["value_pct_change_lag1"] = out["value_pct_change_lag1"].fillna(0)
    out["entity_robust_z"] = out["entity_robust_z"].fillna(0)
    out["entity_robust_abs_z"] = out["entity_robust_abs_z"].fillna(0)

    return out.sort_values(["year", "period_order", "category", "normalized_indicator", "entity_id"]).reset_index(drop=True)


def build_feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Feature matrix for downstream label construction and modeling.
    Retains panel rows with minimum temporal support.
    """
    out = panel.copy()
    out = out[out["entity_time_points"] >= MIN_ENTITY_TIME_POINTS].copy()

    # Encode simple categorical variables as stable integer codes for later modeling.
    for col in ["category", "normalized_indicator", "unit", "period"]:
        out[f"{col}_code"] = pd.factorize(out[col].astype(str))[0]

    feature_cols = [
        "year",
        "period_order",
        "time_index",
        "category_code",
        "normalized_indicator_code",
        "unit_code",
        "period_code",
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
        "entity_time_points",
        "value_lag1",
        "value_diff_lag1",
        "value_pct_change_lag1",
        "value_rolling_mean_2",
        "value_rolling_std_2",
        "entity_robust_z",
        "entity_robust_abs_z",
        "period_value_rank_pct",
        "category_abs_share",
        "traceability_record_density",
        "traceability_source_density",
        "has_lag1",
        "has_pct_change",
    ]

    # Guarantee all columns exist.
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0

    matrix_cols = [
        "observation_id", "entity_id", "year", "period", "period_order",
        "category", "normalized_indicator", "label", "unit"
    ] + feature_cols

    matrix = out[matrix_cols].copy()

    # Fill feature missingness in a reproducible way.
    numeric_cols = [c for c in feature_cols if c in matrix.columns]
    for col in numeric_cols:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
        if matrix[col].isna().any():
            matrix[col] = matrix[col].fillna(matrix[col].median() if matrix[col].notna().any() else 0)

    return matrix.reset_index(drop=True)


# =============================================================================
# Reports
# =============================================================================

def build_entity_dictionary(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    entity_dict = (
        panel.groupby(["entity_id", "category", "normalized_indicator", "label", "unit"], dropna=False)
        .agg(
            n_observations=("observation_id", "count"),
            n_years=("year", "nunique"),
            n_periods=("period", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            mean_value=("value_mean", "mean"),
            median_value=("value_mean", "median"),
            std_value=("value_mean", "std"),
            min_value=("value_mean", "min"),
            max_value=("value_mean", "max"),
            n_source_files=("n_source_files", "max"),
            n_source_sheets=("n_source_sheets", "max"),
        )
        .reset_index()
        .sort_values(["category", "normalized_indicator", "entity_id"])
    )
    return entity_dict


def build_quality_report(raw: pd.DataFrame, panel: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(name: str, value: Any, interpretation: str) -> None:
        rows.append({"check": name, "value": value, "interpretation": interpretation})

    add("input_repaired_long_records", int(len(raw)), "Rows loaded from repaired long-format dataset.")
    add("expanded_panel_observations", int(len(panel)), "Rows after aggregating to year-period-entity observation unit.")
    add("feature_matrix_observations", int(len(matrix)), f"Panel observations with at least {MIN_ENTITY_TIME_POINTS} entity time points.")
    add("unique_entities_panel", int(panel["entity_id"].nunique()) if not panel.empty else 0, "Number of fiscal indicator entities in expanded panel.")
    add("unique_entities_matrix", int(matrix["entity_id"].nunique()) if not matrix.empty else 0, "Number of fiscal indicator entities retained in feature matrix.")
    add("unique_years", int(panel["year"].nunique()) if not panel.empty else 0, "Fiscal years represented.")
    add("unique_periods", int(panel["period"].nunique()) if not panel.empty else 0, "Fiscal reporting periods represented.")
    add("unique_categories", int(panel["category"].nunique()) if not panel.empty else 0, "Fiscal categories represented.")
    add("unique_indicators", int(panel["normalized_indicator"].nunique()) if not panel.empty else 0, "Normalized fiscal indicators represented.")
    add("min_entity_time_points_required", MIN_ENTITY_TIME_POINTS, "Minimum time support required for model feature matrix.")
    add("matrix_numeric_feature_count", int(len([c for c in matrix.columns if pd.api.types.is_numeric_dtype(matrix[c])])) if not matrix.empty else 0, "Numeric columns available for later modeling.")

    return pd.DataFrame(rows)


def build_observation_counts(panel: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(item: str, count: int, definition: str) -> None:
        rows.append({"item": item, "count": count, "definition": definition})

    add("expanded_panel_observations", int(len(panel)), "Observation unit: year + fiscal period + fiscal indicator entity.")
    add("expanded_panel_feature_matrix_observations", int(len(matrix)), f"Panel observations retained after requiring at least {MIN_ENTITY_TIME_POINTS} time points per entity.")
    add("unique_entities", int(panel["entity_id"].nunique()) if not panel.empty else 0, "Unique fiscal entities tracked across time.")
    add("unique_years", int(panel["year"].nunique()) if not panel.empty else 0, "Years represented in panel.")
    add("unique_periods", int(panel["period"].nunique()) if not panel.empty else 0, "Fiscal reporting periods represented in panel.")
    add("unique_categories", int(panel["category"].nunique()) if not panel.empty else 0, "Fiscal categories represented in panel.")
    add("unique_indicators", int(panel["normalized_indicator"].nunique()) if not panel.empty else 0, "Normalized fiscal indicators represented in panel.")

    if not matrix.empty:
        add("numeric_feature_columns", int(len([c for c in matrix.columns if pd.api.types.is_numeric_dtype(matrix[c])])), "Numeric feature columns available in expanded panel matrix.")

    return pd.DataFrame(rows)


def build_temporal_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()

    coverage = (
        panel.groupby(["category", "normalized_indicator"], dropna=False)
        .agg(
            n_entities=("entity_id", "nunique"),
            n_observations=("observation_id", "count"),
            n_years=("year", "nunique"),
            n_periods=("period", "nunique"),
            min_year=("year", "min"),
            max_year=("year", "max"),
            median_entity_time_points=("entity_time_points", "median"),
            max_entity_time_points=("entity_time_points", "max"),
        )
        .reset_index()
        .sort_values(["category", "normalized_indicator"])
    )
    return coverage


def build_split_plan(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Reviewer-facing split plan, not a model split yet.

    Since this is panel data, random splitting can leak temporal information.
    We recommend blocked temporal validation.
    """
    if matrix.empty:
        return pd.DataFrame()

    years = sorted(matrix["year"].unique().tolist())

    rows = []
    for test_year in years:
        train_years = [y for y in years if y < test_year]
        if len(train_years) == 0:
            continue

        train_n = int(matrix[matrix["year"].isin(train_years)].shape[0])
        test_n = int(matrix[matrix["year"] == test_year].shape[0])

        rows.append({
            "split_name": f"rolling_train_until_{test_year-1}_test_{test_year}",
            "train_years": ", ".join(map(str, train_years)),
            "test_year": int(test_year),
            "train_observations": train_n,
            "test_observations": test_n,
            "split_type": "blocked_temporal_out_of_year_validation",
            "leakage_control": "Training years occur strictly before test year.",
        })

    # Also propose final holdout if enough years.
    if len(years) >= 4:
        holdout_year = years[-1]
        train_years = years[:-1]
        rows.append({
            "split_name": f"final_holdout_{holdout_year}",
            "train_years": ", ".join(map(str, train_years)),
            "test_year": int(holdout_year),
            "train_observations": int(matrix[matrix["year"].isin(train_years)].shape[0]),
            "test_observations": int(matrix[matrix["year"] == holdout_year].shape[0]),
            "split_type": "final_temporal_holdout",
            "leakage_control": "Most recent year is held out for final evaluation.",
        })

    return pd.DataFrame(rows)


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
    lines.append("Expanded observation unit: year + fiscal period + fiscal indicator entity.")
    lines.append("This expands the earlier year-period dataset into a panel dataset for empirical testing.")
    lines.append("")
    lines.append("2. Main counts")
    lines.append("-" * 80)
    lines.append(f"Input repaired long records: {summary['input_repaired_long_records']}")
    lines.append(f"Expanded panel observations: {summary['expanded_panel_observations']}")
    lines.append(f"Feature matrix observations: {summary['feature_matrix_observations']}")
    lines.append(f"Unique entities: {summary['unique_entities']}")
    lines.append(f"Years retained: {summary['years_retained']}")
    lines.append(f"Periods retained: {summary['periods_retained']}")
    lines.append(f"Categories retained: {summary['categories_retained']}")
    lines.append(f"Indicators retained: {summary['indicators_retained']}")
    lines.append("")
    lines.append("3. Reviewer relevance")
    lines.append("-" * 80)
    lines.append("The earlier reporting-period dataset had only 16 observations. This expanded panel")
    lines.append("uses fiscal indicator entities within each reporting period, which provides a larger")
    lines.append("sample for anomaly detection, baseline comparison, and blocked temporal validation.")
    lines.append("")
    lines.append("4. Quality checks")
    lines.append("-" * 80)
    for _, r in quality.iterrows():
        lines.append(f"{r['check']}: {r['value']} -- {r['interpretation']}")
    lines.append("")
    lines.append("5. Recommended next step")
    lines.append("-" * 80)
    lines.append("Write 04C_define_panel_governance_metrics_and_labels.py.")
    lines.append("That script should define anomaly labels and governance metrics using the expanded panel,")
    lines.append("then 05 can compare baselines, standalone AI, and governance-centered scoring.")

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

    raw = load_input()
    prepared = prepare_long(raw)
    panel = construct_panel(prepared)
    panel = add_panel_features(panel)
    matrix = build_feature_matrix(panel)

    entity_dict = build_entity_dictionary(panel)
    quality = build_quality_report(prepared, panel, matrix)
    counts = build_observation_counts(panel, matrix)
    temporal_coverage = build_temporal_coverage(panel)
    split_plan = build_split_plan(matrix)

    # Save outputs.
    panel.to_csv(RESULTS_DIR / "expanded_panel_dataset.csv", index=False, encoding="utf-8-sig")
    matrix.to_csv(RESULTS_DIR / "expanded_panel_feature_matrix.csv", index=False, encoding="utf-8-sig")
    entity_dict.to_csv(RESULTS_DIR / "expanded_panel_entity_dictionary.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(RESULTS_DIR / "expanded_panel_data_quality.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(RESULTS_DIR / "expanded_panel_observation_counts.csv", index=False, encoding="utf-8-sig")
    temporal_coverage.to_csv(RESULTS_DIR / "expanded_panel_temporal_coverage.csv", index=False, encoding="utf-8-sig")
    split_plan.to_csv(RESULTS_DIR / "expanded_panel_split_plan.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(RESULTS_DIR / "expanded_panel_preview.xlsx", engine="openpyxl") as writer:
        panel.head(2000).to_excel(writer, sheet_name="expanded_panel", index=False)
        matrix.head(2000).to_excel(writer, sheet_name="feature_matrix", index=False)
        entity_dict.head(1000).to_excel(writer, sheet_name="entity_dictionary", index=False)
        quality.to_excel(writer, sheet_name="quality", index=False)
        counts.to_excel(writer, sheet_name="counts", index=False)
        temporal_coverage.to_excel(writer, sheet_name="temporal_coverage", index=False)
        split_plan.to_excel(writer, sheet_name="split_plan", index=False)

    years = sorted(panel["year"].dropna().astype(int).unique().tolist()) if not panel.empty else []
    periods = sorted(panel["period"].dropna().unique().tolist(), key=lambda p: PERIOD_ORDER.get(p, 99)) if not panel.empty else []
    cats = sorted(panel["category"].dropna().unique().tolist()) if not panel.empty else []
    inds = sorted(panel["normalized_indicator"].dropna().unique().tolist()) if not panel.empty else []

    numeric_feature_cols = [c for c in matrix.columns if pd.api.types.is_numeric_dtype(matrix[c])] if not matrix.empty else []

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(INPUT_REPAIRED_LONG),
        "results_dir": str(RESULTS_DIR),
        "input_repaired_long_records": int(len(raw)),
        "prepared_valid_long_records": int(len(prepared)),
        "expanded_panel_observations": int(len(panel)),
        "feature_matrix_observations": int(len(matrix)),
        "unique_entities": int(panel["entity_id"].nunique()) if not panel.empty else 0,
        "unique_entities_in_feature_matrix": int(matrix["entity_id"].nunique()) if not matrix.empty else 0,
        "numeric_feature_columns": int(len(numeric_feature_cols)),
        "years_retained": years,
        "periods_retained": periods,
        "categories_retained": cats,
        "indicators_retained": inds,
        "minimum_entity_time_points_required": MIN_ENTITY_TIME_POINTS,
        "observation_unit": "year + fiscal period + fiscal indicator entity",
        "recommended_validation": "blocked temporal validation by year; avoid random splitting unless leakage is explicitly controlled",
        "outputs": {
            "expanded_panel_dataset": str(RESULTS_DIR / "expanded_panel_dataset.csv"),
            "expanded_panel_feature_matrix": str(RESULTS_DIR / "expanded_panel_feature_matrix.csv"),
            "expanded_panel_entity_dictionary": str(RESULTS_DIR / "expanded_panel_entity_dictionary.csv"),
            "expanded_panel_data_quality": str(RESULTS_DIR / "expanded_panel_data_quality.csv"),
            "expanded_panel_observation_counts": str(RESULTS_DIR / "expanded_panel_observation_counts.csv"),
            "expanded_panel_temporal_coverage": str(RESULTS_DIR / "expanded_panel_temporal_coverage.csv"),
            "expanded_panel_split_plan": str(RESULTS_DIR / "expanded_panel_split_plan.csv"),
            "expanded_panel_report": str(RESULTS_DIR / "expanded_panel_report.txt"),
        },
    }

    with (RESULTS_DIR / "expanded_panel_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_report(summary, quality)

    print("[OK] Wrote expanded_panel_dataset.csv")
    print("[OK] Wrote expanded_panel_feature_matrix.csv")
    print("[OK] Wrote expanded_panel_entity_dictionary.csv")
    print("[OK] Wrote expanded_panel_data_quality.csv")
    print("[OK] Wrote expanded_panel_observation_counts.csv")
    print("[OK] Wrote expanded_panel_temporal_coverage.csv")
    print("[OK] Wrote expanded_panel_split_plan.csv")
    print("[OK] Wrote expanded_panel_preview.xlsx")
    print("[OK] Wrote expanded_panel_summary.json")
    print("[OK] Wrote expanded_panel_report.txt")
    print("-" * 80)
    print(f"Input repaired long records: {len(raw)}")
    print(f"Expanded panel observations: {len(panel)}")
    print(f"Feature matrix observations: {len(matrix)}")
    print(f"Unique panel entities: {summary['unique_entities']}")
    print(f"Numeric feature columns: {summary['numeric_feature_columns']}")
    print(f"Years retained: {years}")
    print(f"Periods retained: {periods}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
