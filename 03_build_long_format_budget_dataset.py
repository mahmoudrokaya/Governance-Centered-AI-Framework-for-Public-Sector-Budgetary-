r"""
03_build_long_format_budget_dataset.py

Purpose
-------
Build a clean, model-ready fiscal dataset from the standardized long-format MoF
dataset produced by:

    02_extract_and_standardize_mof_tables.py

This script is designed to address key reviewer concerns:
1. Define the number of usable observations.
2. Define the observation unit.
3. Fix missing fiscal periods where defensible.
4. Remove duplicate extracted records transparently.
5. Build cleaned long-format and wide-format analytical datasets.
6. Produce reviewer-facing counts and data-quality evidence.
7. Prepare feature-ready fiscal indicators for later governance scoring,
   baseline modeling, anomaly detection, and validation scripts.

Input
-----
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\02_extract_and_standardize_mof_tables\\standardized_long_dataset.csv

Output folder
-------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\03_build_long_format_budget_dataset

Outputs
-------
1. cleaned_long_dataset.csv
2. model_ready_wide_dataset.csv
3. annual_quarterly_summary.csv
4. duplicate_records_removed.csv
5. missing_period_resolution_log.csv
6. observation_unit_counts.csv
7. category_period_coverage.csv
8. model_feature_dictionary.csv
9. build_summary.json
10. build_report.txt

How to run
----------
pip install pandas numpy openpyxl
python 03_build_long_format_budget_dataset.py

Scientific position
-------------------
This script does not invent labels and does not yet train AI models. It prepares a
clean analytical dataset. Target labels and governance metrics should be defined in
the next script.
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

INPUT_LONG_DATASET = (
    BASE_DIR
    / "Results"
    / "02_extract_and_standardize_mof_tables"
    / "standardized_long_dataset.csv"
)

RESULTS_DIR = BASE_DIR / "Results" / "03_build_long_format_budget_dataset"

PERIOD_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
    "End-Year/Annual": 5,
    "Mid-Year/H1": 6,
    "Unknown": 99,
}

CORE_CATEGORIES = [
    "Revenue",
    "Oil revenue",
    "Non-oil revenue",
    "Expenditure",
    "Surplus/Deficit",
    "Debt",
]

# Label patterns for safer indicator grouping.
INDICATOR_PATTERNS = {
    "total_revenue": [
        r"\btotal\s+revenue", r"\brevenues?\b", r"\btotal\s+revenues?\b"
    ],
    "oil_revenue": [
        r"\boil\s+revenue", r"\boil\s+revenues", r"\boil\b"
    ],
    "non_oil_revenue": [
        r"\bnon[-\s]?oil\s+revenue", r"\bnon[-\s]?oil\s+revenues"
    ],
    "total_expenditure": [
        r"\btotal\s+expenditure", r"\btotal\s+expenditures", r"\bexpenditure\b", r"\bexpenses?\b"
    ],
    "surplus_deficit": [
        r"\bsurplus", r"\bdeficit", r"\bbudget\s+surplus", r"\bbudget\s+deficit"
    ],
    "public_debt": [
        r"\bpublic\s+debt", r"\bdebt\b"
    ],
    "domestic_debt": [
        r"\bdomestic\s+debt"
    ],
    "external_debt": [
        r"\bexternal\s+debt"
    ],
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


def normalize_period(x: Any) -> str:
    s = clean_text(x)
    if not s:
        return "Unknown"

    low = s.lower()
    if low in {"nan", "none", "na", "null"}:
        return "Unknown"

    if re.search(r"\bq\s*1\b|\bq1\b|first\s+quarter", low):
        return "Q1"
    if re.search(r"\bq\s*2\b|\bq2\b|second\s+quarter", low):
        return "Q2"
    if re.search(r"\bq\s*3\b|\bq3\b|third\s+quarter", low):
        return "Q3"
    if re.search(r"\bq\s*4\b|\bq4\b|fourth\s+quarter", low):
        return "Q4"
    if re.search(r"end[\s\-_]*year|annual|year[\s\-_]*end|final", low):
        return "End-Year/Annual"
    if re.search(r"mid[\s\-_]*year|half[\s\-_]*year|h1|semi", low):
        return "Mid-Year/H1"

    return s


def infer_period_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    p = normalize_period(text)
    return None if p == "Unknown" else p


def infer_year_from_text(text: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", str(text))
    if m:
        return int(m.group(1))
    return None


def standardize_category(x: Any, label: str = "") -> str:
    cat = clean_text(x)
    low = f"{cat} {label}".lower()

    if "non-oil" in low or "non oil" in low:
        return "Non-oil revenue"
    if "oil" in low and "revenue" in low:
        return "Oil revenue"
    if "revenue" in low:
        return "Revenue"
    if "expenditure" in low or "expense" in low:
        return "Expenditure"
    if "deficit" in low or "surplus" in low:
        return "Surplus/Deficit"
    if "debt" in low:
        return "Debt"
    return cat if cat else "Other"


def normalize_indicator(label: Any, category: Any, value_column: Any = "") -> str:
    text = f"{clean_text(label)} {clean_text(category)} {clean_text(value_column)}".lower()

    # Specific indicators first.
    for indicator, patterns in INDICATOR_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                return indicator

    # Fallback by category.
    cat = standardize_category(category, str(label))
    fallback = cat.lower().replace("/", "_").replace("-", "_").replace(" ", "_")
    fallback = re.sub(r"[^a-z0-9_]+", "", fallback)
    return fallback or "other"


def make_safe_feature_name(s: str) -> str:
    s = clean_text(s).lower()
    s = s.replace("/", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_feature"


def coerce_numeric(x: Any) -> Optional[float]:
    try:
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def add_period_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["period_order"] = out["period"].map(PERIOD_ORDER).fillna(90).astype(int)
    return out


# =============================================================================
# Core cleaning
# =============================================================================

def load_input() -> pd.DataFrame:
    if not INPUT_LONG_DATASET.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_LONG_DATASET}\n"
            "Run 02_extract_and_standardize_mof_tables.py first."
        )

    df = pd.read_csv(INPUT_LONG_DATASET)
    return df


def fix_missing_year_period(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resolve missing years/periods using defensible source metadata:
    - existing period column
    - file_period_guess
    - value_column
    - source_file
    - sheet_name

    Logs every changed record.
    """
    out = df.copy()
    log_rows: List[Dict[str, Any]] = []

    # Ensure required columns.
    for col in ["period", "file_period_guess", "value_column", "source_file", "sheet_name"]:
        if col not in out.columns:
            out[col] = ""

    for idx, row in out.iterrows():
        old_period = normalize_period(row.get("period", ""))
        new_period = old_period
        method = "kept_original"

        if old_period == "Unknown":
            candidates = [
                ("file_period_guess", row.get("file_period_guess", "")),
                ("value_column", row.get("value_column", "")),
                ("source_file", row.get("source_file", "")),
                ("sheet_name", row.get("sheet_name", "")),
            ]

            for source_name, text in candidates:
                inferred = infer_period_from_text(str(text))
                if inferred:
                    new_period = inferred
                    method = f"inferred_from_{source_name}"
                    break

        if new_period == "Unknown":
            # If the source is annual/end-budget, set annual if source text suggests it.
            combined = " ".join([
                str(row.get("source_file", "")),
                str(row.get("sheet_name", "")),
                str(row.get("value_column", "")),
            ])
            inferred = infer_period_from_text(combined)
            if inferred:
                new_period = inferred
                method = "inferred_from_combined_source_text"

        if new_period != old_period or old_period == "Unknown":
            log_rows.append({
                "row_index": int(idx),
                "source_file": row.get("source_file", ""),
                "sheet_name": row.get("sheet_name", ""),
                "label": row.get("label", ""),
                "value_column": row.get("value_column", ""),
                "old_period": old_period,
                "new_period": new_period,
                "resolution_method": method,
            })

        out.at[idx, "period"] = new_period

        # Year repair if needed.
        if "year" in out.columns:
            y = pd.to_numeric(row.get("year", np.nan), errors="coerce")
        else:
            y = np.nan

        if pd.isna(y):
            combined = " ".join([
                str(row.get("file_year_guess", "")),
                str(row.get("value_column", "")),
                str(row.get("source_file", "")),
                str(row.get("sheet_name", "")),
            ])
            inferred_y = infer_year_from_text(combined)
            if inferred_y is not None:
                out.at[idx, "year"] = inferred_y

    log_df = pd.DataFrame(log_rows)
    return out, log_df


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove exact analytical duplicates, preserving the first occurrence.

    Duplicate key represents same extracted observation:
    year, period, category, normalized_indicator, label, value, unit.
    """
    out = df.copy()

    dup_key = [
        "year",
        "period",
        "category",
        "normalized_indicator",
        "label",
        "value",
        "unit",
    ]

    for c in dup_key:
        if c not in out.columns:
            out[c] = ""

    duplicated_mask = out.duplicated(subset=dup_key, keep="first")
    duplicate_records = out.loc[duplicated_mask].copy()
    cleaned = out.loc[~duplicated_mask].copy()

    return cleaned.reset_index(drop=True), duplicate_records.reset_index(drop=True)


def clean_long_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    # Harmonize required fields.
    required_cols = [
        "source_file", "source_relative_path", "sheet_name", "source_sha256",
        "file_year_guess", "file_period_guess", "year", "period", "label",
        "category", "value_column", "value", "raw_value", "unit"
    ]

    for c in required_cols:
        if c not in out.columns:
            out[c] = np.nan

    out["label"] = out["label"].map(clean_text)
    out["value_column"] = out["value_column"].map(clean_text)
    out["source_file"] = out["source_file"].map(clean_text)
    out["sheet_name"] = out["sheet_name"].map(clean_text)
    out["unit"] = out["unit"].map(clean_text).replace("", "unspecified")

    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    # Remove unusable rows.
    out = out.dropna(subset=["year", "value"]).copy()
    out["year"] = out["year"].astype(int)

    # Fix categories before period resolution.
    out["category"] = [
        standardize_category(cat, label)
        for cat, label in zip(out["category"], out["label"])
    ]

    out["period"] = out["period"].apply(normalize_period)

    out, period_log = fix_missing_year_period(out)

    out["period"] = out["period"].apply(normalize_period)
    out["category"] = [
        standardize_category(cat, label)
        for cat, label in zip(out["category"], out["label"])
    ]

    out["normalized_indicator"] = [
        normalize_indicator(label, cat, vc)
        for label, cat, vc in zip(out["label"], out["category"], out["value_column"])
    ]

    # Remove remaining fully vague rows if label is empty/unlabeled and category is Other.
    vague = (
        out["label"].str.lower().isin(["", "unlabeled row"])
        & out["category"].str.lower().eq("other")
    )
    out = out.loc[~vague].copy()

    # Sort before duplicate removal.
    out = add_period_sort(out)
    out = out.sort_values(
        ["year", "period_order", "category", "normalized_indicator", "label", "source_file", "sheet_name"],
        na_position="last"
    ).reset_index(drop=True)

    cleaned, duplicates = remove_duplicates(out)

    return cleaned, duplicates, period_log


# =============================================================================
# Feature building
# =============================================================================

def build_observation_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["year"].astype(str)
        + "__"
        + df["period"].astype(str).map(make_safe_feature_name)
    )


def aggregate_long_to_observation_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready wide dataset.

    Observation unit:
        fiscal reporting period = year + period

    Features:
        Aggregated values by normalized_indicator and category.
        Because multiple source rows may exist per indicator/period, we compute:
        - mean
        - median
        - sum
        - count

    This gives a transparent model matrix while preserving extraction provenance.
    """
    if df.empty:
        return pd.DataFrame()

    temp = df.copy()
    temp["observation_id"] = build_observation_id(temp)
    temp["feature_base"] = (
        temp["category"].map(make_safe_feature_name)
        + "__"
        + temp["normalized_indicator"].map(make_safe_feature_name)
    )

    grouped = (
        temp
        .groupby(["observation_id", "year", "period", "period_order", "feature_base"], dropna=False)
        .agg(
            value_mean=("value", "mean"),
            value_median=("value", "median"),
            value_sum=("value", "sum"),
            value_count=("value", "count"),
            source_file_count=("source_file", "nunique"),
            sheet_count=("sheet_name", "nunique"),
        )
        .reset_index()
    )

    wide_parts = []

    for metric in ["value_mean", "value_median", "value_sum", "value_count", "source_file_count", "sheet_count"]:
        pivot = grouped.pivot_table(
            index=["observation_id", "year", "period", "period_order"],
            columns="feature_base",
            values=metric,
            aggfunc="first",
        )
        pivot.columns = [f"{make_safe_feature_name(c)}__{metric}" for c in pivot.columns]
        wide_parts.append(pivot)

    wide = pd.concat(wide_parts, axis=1).reset_index()

    # Add high-level totals by category.
    cat_summary = (
        temp
        .groupby(["observation_id", "year", "period", "period_order", "category"], dropna=False)
        .agg(
            category_value_mean=("value", "mean"),
            category_value_sum=("value", "sum"),
            category_value_count=("value", "count"),
        )
        .reset_index()
    )

    for metric in ["category_value_mean", "category_value_sum", "category_value_count"]:
        pivot = cat_summary.pivot_table(
            index=["observation_id", "year", "period", "period_order"],
            columns="category",
            values=metric,
            aggfunc="first",
        )
        pivot.columns = [f"category__{make_safe_feature_name(c)}__{metric}" for c in pivot.columns]
        wide = wide.merge(pivot.reset_index(), on=["observation_id", "year", "period", "period_order"], how="left")

    # Add record provenance counts.
    provenance = (
        temp
        .groupby(["observation_id", "year", "period", "period_order"], dropna=False)
        .agg(
            n_long_records=("value", "count"),
            n_unique_labels=("label", "nunique"),
            n_source_files=("source_file", "nunique"),
            n_source_sheets=("sheet_name", "nunique"),
            n_categories=("category", "nunique"),
            n_indicators=("normalized_indicator", "nunique"),
        )
        .reset_index()
    )

    wide = provenance.merge(wide, on=["observation_id", "year", "period", "period_order"], how="left")
    wide = wide.sort_values(["year", "period_order"]).reset_index(drop=True)

    return wide


def build_annual_quarterly_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    if cleaned.empty:
        return pd.DataFrame()

    summary = (
        cleaned
        .groupby(["year", "period", "period_order", "category"], dropna=False)
        .agg(
            n_records=("value", "count"),
            n_labels=("label", "nunique"),
            n_indicators=("normalized_indicator", "nunique"),
            n_source_files=("source_file", "nunique"),
            value_mean=("value", "mean"),
            value_median=("value", "median"),
            value_std=("value", "std"),
            value_min=("value", "min"),
            value_max=("value", "max"),
        )
        .reset_index()
        .sort_values(["year", "period_order", "category"])
    )

    return summary


def build_observation_unit_counts(cleaned: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "item": "cleaned_long_records",
        "count": int(len(cleaned)),
        "definition": "Rows in cleaned long-format dataset after resolving periods and removing analytical duplicates."
    })

    rows.append({
        "item": "model_ready_observations",
        "count": int(len(wide)),
        "definition": "Unique fiscal reporting periods used as observation units: year + period."
    })

    if not cleaned.empty:
        rows.append({
            "item": "unique_years",
            "count": int(cleaned["year"].nunique()),
            "definition": "Number of fiscal years retained."
        })
        rows.append({
            "item": "unique_periods",
            "count": int(cleaned["period"].nunique()),
            "definition": "Number of fiscal reporting period types retained."
        })
        rows.append({
            "item": "unique_categories",
            "count": int(cleaned["category"].nunique()),
            "definition": "Number of fiscal categories retained."
        })
        rows.append({
            "item": "unique_indicators",
            "count": int(cleaned["normalized_indicator"].nunique()),
            "definition": "Number of normalized fiscal indicators retained."
        })
        rows.append({
            "item": "unique_source_files",
            "count": int(cleaned["source_file"].nunique()),
            "definition": "Number of source Excel/CSV files contributing cleaned observations."
        })
        rows.append({
            "item": "unique_source_sheets",
            "count": int(cleaned[["source_file", "sheet_name"]].drop_duplicates().shape[0]),
            "definition": "Number of source workbook sheets contributing cleaned observations."
        })

    return pd.DataFrame(rows)


def build_category_period_coverage(cleaned: pd.DataFrame) -> pd.DataFrame:
    if cleaned.empty:
        return pd.DataFrame()

    coverage = (
        cleaned
        .groupby(["year", "period", "period_order", "category"], dropna=False)
        .agg(
            n_records=("value", "count"),
            n_indicators=("normalized_indicator", "nunique"),
            n_labels=("label", "nunique"),
            n_source_files=("source_file", "nunique"),
        )
        .reset_index()
        .sort_values(["year", "period_order", "category"])
    )
    return coverage


def build_feature_dictionary(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()

    rows = []
    protected = {"observation_id", "year", "period", "period_order"}

    for col in wide.columns:
        if col in protected:
            role = "identifier"
        elif col.startswith("n_"):
            role = "provenance_count"
        elif "__value_mean" in col:
            role = "feature_mean"
        elif "__value_median" in col:
            role = "feature_median"
        elif "__value_sum" in col:
            role = "feature_sum"
        elif "__value_count" in col:
            role = "feature_count"
        elif "__source_file_count" in col or "__sheet_count" in col:
            role = "feature_provenance"
        elif col.startswith("category__"):
            role = "category_aggregate_feature"
        else:
            role = "derived_feature"

        numeric = pd.api.types.is_numeric_dtype(wide[col])

        rows.append({
            "feature_name": col,
            "role": role,
            "numeric": bool(numeric),
            "non_missing_count": int(wide[col].notna().sum()),
            "missing_count": int(wide[col].isna().sum()),
            "mean": float(wide[col].mean()) if numeric and wide[col].notna().any() else np.nan,
            "std": float(wide[col].std(ddof=1)) if numeric and wide[col].notna().sum() > 1 else np.nan,
            "min": float(wide[col].min()) if numeric and wide[col].notna().any() else np.nan,
            "max": float(wide[col].max()) if numeric and wide[col].notna().any() else np.nan,
            "description": describe_feature(col, role),
        })

    return pd.DataFrame(rows)


def describe_feature(col: str, role: str) -> str:
    if col == "observation_id":
        return "Unique observation key defined as fiscal year plus reporting period."
    if col == "year":
        return "Fiscal year."
    if col == "period":
        return "Fiscal reporting period."
    if col == "period_order":
        return "Numeric ordering of reporting periods."
    if role == "provenance_count":
        return "Count describing data coverage or extraction provenance for this observation."
    if role.startswith("feature_"):
        return "Aggregated fiscal indicator feature derived from standardized MoF long-format records."
    if role == "category_aggregate_feature":
        return "High-level fiscal category aggregate feature."
    return "Derived analytical field."


# =============================================================================
# Reporting
# =============================================================================

def make_build_summary(
    raw_df: pd.DataFrame,
    cleaned: pd.DataFrame,
    duplicates: pd.DataFrame,
    period_log: pd.DataFrame,
    wide: pd.DataFrame,
) -> Dict[str, Any]:

    years = sorted(cleaned["year"].dropna().astype(int).unique().tolist()) if not cleaned.empty else []
    periods = sorted(cleaned["period"].dropna().astype(str).unique().tolist(), key=lambda x: PERIOD_ORDER.get(x, 90)) if not cleaned.empty else []
    categories = sorted(cleaned["category"].dropna().astype(str).unique().tolist()) if not cleaned.empty else []

    unresolved_periods = int((cleaned["period"] == "Unknown").sum()) if not cleaned.empty else 0

    n_model_features = 0
    if not wide.empty:
        n_model_features = int(len([c for c in wide.columns if c not in {"observation_id", "year", "period", "period_order"}]))

    return {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "input_dataset": str(INPUT_LONG_DATASET),
        "results_dir": str(RESULTS_DIR),
        "input_long_records": int(len(raw_df)),
        "cleaned_long_records": int(len(cleaned)),
        "duplicates_removed": int(len(duplicates)),
        "period_resolution_log_records": int(len(period_log)),
        "unresolved_period_records_after_cleaning": unresolved_periods,
        "model_ready_observations": int(len(wide)),
        "model_ready_features_excluding_identifiers": n_model_features,
        "years_detected": years,
        "periods_detected": periods,
        "categories_detected": categories,
        "observation_unit": "Fiscal reporting period defined as year + period.",
        "scientific_note": (
            "This script prepares a clean analytical dataset. It does not create target labels, "
            "does not compute governance scores, and does not train AI models. These should be "
            "defined explicitly in later scripts."
        ),
        "outputs": {
            "cleaned_long_dataset": str(RESULTS_DIR / "cleaned_long_dataset.csv"),
            "model_ready_wide_dataset": str(RESULTS_DIR / "model_ready_wide_dataset.csv"),
            "annual_quarterly_summary": str(RESULTS_DIR / "annual_quarterly_summary.csv"),
            "duplicate_records_removed": str(RESULTS_DIR / "duplicate_records_removed.csv"),
            "missing_period_resolution_log": str(RESULTS_DIR / "missing_period_resolution_log.csv"),
            "observation_unit_counts": str(RESULTS_DIR / "observation_unit_counts.csv"),
            "category_period_coverage": str(RESULTS_DIR / "category_period_coverage.csv"),
            "model_feature_dictionary": str(RESULTS_DIR / "model_feature_dictionary.csv"),
            "build_report": str(RESULTS_DIR / "build_report.txt"),
        }
    }


def write_build_report(summary: Dict[str, Any], obs_counts: pd.DataFrame) -> None:
    lines = []
    lines.append("MODEL-READY LONG-FORMAT BUDGET DATASET BUILD REPORT")
    lines.append("=" * 80)
    lines.append(f"Run time: {summary['run_time']}")
    lines.append(f"Input dataset: {summary['input_dataset']}")
    lines.append(f"Results directory: {summary['results_dir']}")
    lines.append("")
    lines.append("1. Reviewer-facing observation unit")
    lines.append("-" * 80)
    lines.append(f"Observation unit: {summary['observation_unit']}")
    lines.append(f"Model-ready observations: {summary['model_ready_observations']}")
    lines.append(f"Model-ready features excluding identifiers: {summary['model_ready_features_excluding_identifiers']}")
    lines.append("")
    lines.append("2. Cleaning summary")
    lines.append("-" * 80)
    lines.append(f"Input long records: {summary['input_long_records']}")
    lines.append(f"Cleaned long records: {summary['cleaned_long_records']}")
    lines.append(f"Duplicates removed: {summary['duplicates_removed']}")
    lines.append(f"Period-resolution log records: {summary['period_resolution_log_records']}")
    lines.append(f"Unresolved period records after cleaning: {summary['unresolved_period_records_after_cleaning']}")
    lines.append("")
    lines.append("3. Coverage")
    lines.append("-" * 80)
    lines.append(f"Years detected: {summary['years_detected']}")
    lines.append(f"Periods detected: {summary['periods_detected']}")
    lines.append(f"Categories detected: {summary['categories_detected']}")
    lines.append("")
    lines.append("4. Counts")
    lines.append("-" * 80)

    if not obs_counts.empty:
        for _, r in obs_counts.iterrows():
            lines.append(f"{r['item']}: {r['count']} -- {r['definition']}")

    lines.append("")
    lines.append("5. Scientific note")
    lines.append("-" * 80)
    lines.append(summary["scientific_note"])
    lines.append("")
    lines.append("6. Next script")
    lines.append("-" * 80)
    lines.append("04_define_governance_metrics_and_labels.py")
    lines.append("This should define formula-based risk indicators, anomaly labels, governance metrics,")
    lines.append("and reviewer-facing rubrics before any AI model is trained.")

    (RESULTS_DIR / "build_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("03 BUILD LONG-FORMAT MODEL-READY BUDGET DATASET")
    print("=" * 80)
    print(f"Input dataset:     {INPUT_LONG_DATASET}")
    print(f"Results directory: {RESULTS_DIR}")
    print("-" * 80)

    raw_df = load_input()
    cleaned, duplicates, period_log = clean_long_dataset(raw_df)
    wide = aggregate_long_to_observation_level(cleaned)
    annual_quarterly_summary = build_annual_quarterly_summary(cleaned)
    obs_counts = build_observation_unit_counts(cleaned, wide)
    category_period_coverage = build_category_period_coverage(cleaned)
    feature_dictionary = build_feature_dictionary(wide)

    # Save outputs.
    cleaned.to_csv(RESULTS_DIR / "cleaned_long_dataset.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(RESULTS_DIR / "model_ready_wide_dataset.csv", index=False, encoding="utf-8-sig")
    annual_quarterly_summary.to_csv(RESULTS_DIR / "annual_quarterly_summary.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(RESULTS_DIR / "duplicate_records_removed.csv", index=False, encoding="utf-8-sig")
    period_log.to_csv(RESULTS_DIR / "missing_period_resolution_log.csv", index=False, encoding="utf-8-sig")
    obs_counts.to_csv(RESULTS_DIR / "observation_unit_counts.csv", index=False, encoding="utf-8-sig")
    category_period_coverage.to_csv(RESULTS_DIR / "category_period_coverage.csv", index=False, encoding="utf-8-sig")
    feature_dictionary.to_csv(RESULTS_DIR / "model_feature_dictionary.csv", index=False, encoding="utf-8-sig")

    # Optional Excel workbook for inspection.
    with pd.ExcelWriter(RESULTS_DIR / "model_ready_dataset_preview.xlsx", engine="openpyxl") as writer:
        cleaned.head(1000).to_excel(writer, sheet_name="cleaned_long_preview", index=False)
        wide.to_excel(writer, sheet_name="model_ready_wide", index=False)
        annual_quarterly_summary.to_excel(writer, sheet_name="annual_quarterly_summary", index=False)
        obs_counts.to_excel(writer, sheet_name="observation_counts", index=False)
        category_period_coverage.head(1000).to_excel(writer, sheet_name="coverage_preview", index=False)
        feature_dictionary.to_excel(writer, sheet_name="feature_dictionary", index=False)

    summary = make_build_summary(raw_df, cleaned, duplicates, period_log, wide)

    with (RESULTS_DIR / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_build_report(summary, obs_counts)

    print("[OK] Wrote cleaned_long_dataset.csv")
    print("[OK] Wrote model_ready_wide_dataset.csv")
    print("[OK] Wrote annual_quarterly_summary.csv")
    print("[OK] Wrote duplicate_records_removed.csv")
    print("[OK] Wrote missing_period_resolution_log.csv")
    print("[OK] Wrote observation_unit_counts.csv")
    print("[OK] Wrote category_period_coverage.csv")
    print("[OK] Wrote model_feature_dictionary.csv")
    print("[OK] Wrote model_ready_dataset_preview.xlsx")
    print("[OK] Wrote build_summary.json")
    print("[OK] Wrote build_report.txt")
    print("-" * 80)
    print(f"Input long records: {summary['input_long_records']}")
    print(f"Cleaned long records: {summary['cleaned_long_records']}")
    print(f"Duplicates removed: {summary['duplicates_removed']}")
    print(f"Model-ready observations: {summary['model_ready_observations']}")
    print(f"Model-ready features excluding identifiers: {summary['model_ready_features_excluding_identifiers']}")
    print(f"Unresolved period records after cleaning: {summary['unresolved_period_records_after_cleaning']}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
