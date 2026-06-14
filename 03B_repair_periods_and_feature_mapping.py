r"""
03B_repair_periods_and_feature_mapping.py

Purpose
-------
Repair invalid fiscal period labels and rebuild a cleaner model-ready dataset before
governance labels and AI experiments are computed.

Why this script is needed
-------------------------
After 03_build_long_format_budget_dataset.py, the build report showed invalid period
values such as "Public Debt", "Column_3", "Item", and Arabic debt labels. These are
not fiscal reporting periods; they are misread table labels/headers.

This script fixes that problem by:
1. Loading cleaned_long_dataset.csv from script 03.
2. Reassigning the fiscal period using a strict hierarchy:
   period -> file_period_guess -> source_file -> sheet_name -> value_column.
3. Keeping only valid fiscal periods:
   Q1, Q2, Q3, Q4, End-Year/Annual, Mid-Year/H1.
4. Rebuilding normalized fiscal indicators with better mapping.
5. Rebuilding a wide model-ready dataset.
6. Producing a clean feature-mapping report so script 04 can identify revenue,
   oil revenue, non-oil revenue, expenditure, surplus/deficit, and debt more reliably.

Input folder
------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\03_build_long_format_budget_dataset

Main input
----------
cleaned_long_dataset.csv

Output folder
-------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\03B_repair_periods_and_feature_mapping

Outputs
-------
1. repaired_cleaned_long_dataset.csv
2. repaired_model_ready_wide_dataset.csv
3. repaired_model_feature_dictionary.csv
4. period_repair_log.csv
5. invalid_period_records.csv
6. concept_mapping_candidates.csv
7. repaired_observation_unit_counts.csv
8. repaired_category_period_coverage.csv
9. repair_summary.json
10. repair_report.txt
11. repaired_dataset_preview.xlsx

How to run
----------
pip install pandas numpy openpyxl
python 03B_repair_periods_and_feature_mapping.py

Important
---------
After running this script, update script 04 to read from:
Results\\03B_repair_periods_and_feature_mapping
instead of:
Results\\03_build_long_format_budget_dataset
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

INPUT_DIR = BASE_DIR / "Results" / "03_build_long_format_budget_dataset"
INPUT_CLEANED_LONG = INPUT_DIR / "cleaned_long_dataset.csv"

RESULTS_DIR = BASE_DIR / "Results" / "03B_repair_periods_and_feature_mapping"

VALID_PERIODS = ["Q1", "Q2", "Q3", "Q4", "End-Year/Annual", "Mid-Year/H1"]

PERIOD_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
    "End-Year/Annual": 5,
    "Mid-Year/H1": 6,
}

INVALID_PERIOD_TOKENS = {
    "", "nan", "none", "null", "unknown",
    "column_1", "column_2", "column_3", "column_4", "column_5", "column_6",
    "item", "total", "revenues*", "revenues", "revenue",
    "public debt", "domestic debt", "external debt",
    "الدين العام", "الدين الداخلي domestic debt", "الدين الخارجي external debt",
    "الإجمالي", "البيان",
}

CONCEPT_PATTERNS: Dict[str, List[str]] = {
    "total_revenue": [
        r"\btotal\s+revenue[s]?\b",
        r"\brevenue[s]?\b",
        r"الإيرادات",
    ],
    "oil_revenue": [
        r"\boil\s+revenue[s]?\b",
        r"\boil\b",
        r"الإيرادات النفطية",
    ],
    "non_oil_revenue": [
        r"\bnon[-\s]?oil\s+revenue[s]?\b",
        r"\bnon[-\s]?oil\b",
        r"الإيرادات غير النفطية",
    ],
    "total_expenditure": [
        r"\btotal\s+expenditure[s]?\b",
        r"\bexpenditure[s]?\b",
        r"\bexpense[s]?\b",
        r"المصروفات",
        r"النفقات",
    ],
    "surplus_deficit": [
        r"\bsurplus\b",
        r"\bdeficit\b",
        r"\bbudget\s+surplus\b",
        r"\bbudget\s+deficit\b",
        r"الفائض",
        r"العجز",
    ],
    "public_debt": [
        r"\bpublic\s+debt\b",
        r"\bdebt\b",
        r"الدين العام",
    ],
    "domestic_debt": [
        r"\bdomestic\s+debt\b",
        r"الدين الداخلي",
    ],
    "external_debt": [
        r"\bexternal\s+debt\b",
        r"الدين الخارجي",
    ],
}


# =============================================================================
# Utilities
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


def normalize_for_match(x: Any) -> str:
    s = clean_text(x).lower()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_safe_feature_name(x: Any) -> str:
    s = clean_text(x).lower()
    s = s.replace("/", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def infer_period_from_text(text: Any) -> Optional[str]:
    t = normalize_for_match(text)
    if not t:
        return None

    # Strong quarter patterns.
    if re.search(r"(^|\b)q\s*1(\b|$)|q1|first\s+quarter|quarter\s*1", t):
        return "Q1"
    if re.search(r"(^|\b)q\s*2(\b|$)|q2|second\s+quarter|quarter\s*2", t):
        return "Q2"
    if re.search(r"(^|\b)q\s*3(\b|$)|q3|third\s+quarter|quarter\s*3", t):
        return "Q3"
    if re.search(r"(^|\b)q\s*4(\b|$)|q4|fourth\s+quarter|quarter\s*4", t):
        return "Q4"

    # Annual / mid-year.
    if re.search(r"end[\s\-_]*year|year[\s\-_]*end|annual|fy|final|until\s+q4|end\s+bud", t):
        return "End-Year/Annual"
    if re.search(r"mid[\s\-_]*year|half[\s\-_]*year|semi[\s\-_]*annual|h1|until\s+q2|mid\s+bud", t):
        return "Mid-Year/H1"

    return None


def is_valid_period(x: Any) -> bool:
    return clean_text(x) in VALID_PERIODS


def is_invalid_period_token(x: Any) -> bool:
    return normalize_for_match(x) in INVALID_PERIOD_TOKENS


def infer_year_from_text(text: Any) -> Optional[int]:
    m = re.search(r"(20\d{2})", str(text))
    if m:
        return int(m.group(1))
    return None


def classify_category(label: Any, original_category: Any = "") -> str:
    text = normalize_for_match(f"{label} {original_category}")

    if re.search(r"non[-\s]?oil|غير النفطية", text):
        return "Non-oil revenue"
    if re.search(r"oil|النفطية", text) and re.search(r"revenue|إيراد|الإيرادات", text):
        return "Oil revenue"
    if re.search(r"revenue|إيراد|الإيرادات", text):
        return "Revenue"
    if re.search(r"expenditure|expense|المصروفات|النفقات", text):
        return "Expenditure"
    if re.search(r"surplus|deficit|الفائض|العجز", text):
        return "Surplus/Deficit"
    if re.search(r"debt|الدين", text):
        return "Debt"
    return clean_text(original_category) if clean_text(original_category) else "Other"


def normalize_indicator(label: Any, category: Any, value_column: Any = "", sheet_name: Any = "") -> str:
    text = normalize_for_match(f"{label} {category} {value_column} {sheet_name}")

    # Specific before general.
    if any(re.search(p, text) for p in CONCEPT_PATTERNS["non_oil_revenue"]):
        return "non_oil_revenue"
    if any(re.search(p, text) for p in CONCEPT_PATTERNS["oil_revenue"]):
        return "oil_revenue"
    if any(re.search(p, text) for p in CONCEPT_PATTERNS["total_revenue"]):
        return "total_revenue"
    if any(re.search(p, text) for p in CONCEPT_PATTERNS["total_expenditure"]):
        return "total_expenditure"
    if any(re.search(p, text) for p in CONCEPT_PATTERNS["surplus_deficit"]):
        return "surplus_deficit"
    if any(re.search(p, text) for p in CONCEPT_PATTERNS["domestic_debt"]):
        return "domestic_debt"
    if any(re.search(p, text) for p in CONCEPT_PATTERNS["external_debt"]):
        return "external_debt"
    if any(re.search(p, text) for p in CONCEPT_PATTERNS["public_debt"]):
        return "public_debt"

    cat = classify_category(label, category)
    return make_safe_feature_name(cat)


def coerce_numeric(x: Any) -> Optional[float]:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return None
    return float(v)


# =============================================================================
# Load and repair
# =============================================================================

def load_input() -> pd.DataFrame:
    if not INPUT_CLEANED_LONG.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_CLEANED_LONG}\n"
            "Run 03_build_long_format_budget_dataset.py first."
        )
    return pd.read_csv(INPUT_CLEANED_LONG)


def repair_periods(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    log_rows: List[Dict[str, Any]] = []

    required = ["period", "file_period_guess", "source_file", "sheet_name", "value_column", "label"]
    for col in required:
        if col not in out.columns:
            out[col] = ""

    for idx, row in out.iterrows():
        old_period = clean_text(row.get("period", ""))
        new_period = old_period if is_valid_period(old_period) else None
        method = "kept_valid_period" if new_period else "needs_repair"

        if new_period is None:
            # Strict hierarchy. Do not use label first because labels caused false periods.
            candidates = [
                ("file_period_guess", row.get("file_period_guess", "")),
                ("source_file", row.get("source_file", "")),
                ("sheet_name", row.get("sheet_name", "")),
                ("value_column", row.get("value_column", "")),
            ]

            for cname, ctext in candidates:
                inferred = infer_period_from_text(ctext)
                if inferred:
                    new_period = inferred
                    method = f"inferred_from_{cname}"
                    break

        if new_period is None:
            # Last fallback: year-end sheets often lack quarter but belong to annual reports.
            combined = " ".join([
                str(row.get("source_file", "")),
                str(row.get("sheet_name", "")),
                str(row.get("value_column", "")),
            ])
            inferred = infer_period_from_text(combined)
            if inferred:
                new_period = inferred
                method = "inferred_from_combined_source_text"

        if new_period is None:
            new_period = "Invalid/Unresolved"
            method = "unresolved_excluded"

        if old_period != new_period or method != "kept_valid_period":
            log_rows.append({
                "row_index": int(idx),
                "source_file": row.get("source_file", ""),
                "sheet_name": row.get("sheet_name", ""),
                "label": row.get("label", ""),
                "value_column": row.get("value_column", ""),
                "old_period": old_period,
                "new_period": new_period,
                "repair_method": method,
            })

        out.at[idx, "period"] = new_period

    invalid = out.loc[~out["period"].isin(VALID_PERIODS)].copy()
    repaired = out.loc[out["period"].isin(VALID_PERIODS)].copy()

    # Recalculate period order.
    repaired["period_order"] = repaired["period"].map(PERIOD_ORDER).astype(int)

    log_df = pd.DataFrame(log_rows)
    return repaired.reset_index(drop=True), invalid.reset_index(drop=True), log_df


def repair_years(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "year" not in out.columns:
        out["year"] = np.nan

    out["year"] = pd.to_numeric(out["year"], errors="coerce")

    for idx, row in out[out["year"].isna()].iterrows():
        combined = " ".join([
            str(row.get("file_year_guess", "")),
            str(row.get("source_file", "")),
            str(row.get("sheet_name", "")),
            str(row.get("value_column", "")),
        ])
        y = infer_year_from_text(combined)
        if y:
            out.at[idx, "year"] = y

    out = out.dropna(subset=["year"]).copy()
    out["year"] = out["year"].astype(int)
    return out


def repair_categories_and_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["label", "category", "value_column", "sheet_name"]:
        if col not in out.columns:
            out[col] = ""

    out["category_repaired"] = [
        classify_category(label, cat)
        for label, cat in zip(out["label"], out["category"])
    ]

    out["normalized_indicator_repaired"] = [
        normalize_indicator(label, cat, vc, sh)
        for label, cat, vc, sh in zip(
            out["label"], out["category_repaired"], out["value_column"], out["sheet_name"]
        )
    ]

    # Replace previous fields with repaired versions.
    out["category"] = out["category_repaired"]
    out["normalized_indicator"] = out["normalized_indicator_repaired"]

    return out


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key = [
        "year", "period", "category", "normalized_indicator",
        "label", "value_column", "value", "unit", "source_file", "sheet_name"
    ]

    for col in key:
        if col not in df.columns:
            df[col] = ""

    dup_mask = df.duplicated(subset=key, keep="first")
    dups = df.loc[dup_mask].copy()
    clean = df.loc[~dup_mask].copy()
    return clean.reset_index(drop=True), dups.reset_index(drop=True)


# =============================================================================
# Rebuild wide model-ready dataset
# =============================================================================

def build_observation_id(df: pd.DataFrame) -> pd.Series:
    return df["year"].astype(str) + "__" + df["period"].astype(str).map(make_safe_feature_name)


def build_wide_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    temp = df.copy()
    temp["observation_id"] = build_observation_id(temp)
    temp["feature_base"] = (
        temp["category"].map(make_safe_feature_name)
        + "__"
        + temp["normalized_indicator"].map(make_safe_feature_name)
    )

    # Aggregate repeated extracted values for the same reporting period and concept.
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

    index_cols = ["observation_id", "year", "period", "period_order"]

    pivots = []
    for metric in ["value_mean", "value_median", "value_sum", "value_count", "source_file_count", "sheet_count"]:
        pivot = grouped.pivot_table(
            index=index_cols,
            columns="feature_base",
            values=metric,
            aggfunc="first",
        )
        pivot.columns = [f"{make_safe_feature_name(c)}__{metric}" for c in pivot.columns]
        pivots.append(pivot)

    wide = pd.concat(pivots, axis=1).reset_index()

    provenance = (
        temp
        .groupby(index_cols, dropna=False)
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

    wide = provenance.merge(wide, on=index_cols, how="left")
    wide = wide.sort_values(["year", "period_order"]).reset_index(drop=True)
    return wide


def build_feature_dictionary(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame()

    protected = {"observation_id", "year", "period", "period_order"}
    rows = []

    for col in wide.columns:
        if col in protected:
            role = "identifier"
        elif col.startswith("n_"):
            role = "provenance_count"
        elif "__value_mean" in col:
            role = "model_feature_mean"
        elif "__value_median" in col:
            role = "model_feature_median"
        elif "__value_sum" in col:
            role = "model_feature_sum"
        elif "__value_count" in col:
            role = "model_feature_count"
        elif "__source_file_count" in col or "__sheet_count" in col:
            role = "feature_traceability"
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
        })

    return pd.DataFrame(rows)


# =============================================================================
# Concept mapping diagnostics
# =============================================================================

def build_concept_mapping_candidates(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for concept, patterns in CONCEPT_PATTERNS.items():
        for col in wide.columns:
            low = normalize_for_match(col)
            matched = any(re.search(p, low) for p in patterns)
            if matched:
                preferred = (
                    "__value_mean" in col
                    or "__value_median" in col
                    or "__value_sum" in col
                )
                rows.append({
                    "concept": concept,
                    "candidate_feature": col,
                    "preferred_numeric_value_feature": bool(preferred),
                    "non_missing_count": int(wide[col].notna().sum()) if col in wide else 0,
                    "mean": float(pd.to_numeric(wide[col], errors="coerce").mean()) if col in wide and pd.api.types.is_numeric_dtype(wide[col]) else np.nan,
                })

    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["concept", "preferred_numeric_value_feature", "non_missing_count"],
            ascending=[True, False, False]
        )
    return candidates


def build_observation_counts(df: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "item": "repaired_cleaned_long_records",
            "count": int(len(df)),
            "definition": "Long-format rows after strict period repair, invalid-period exclusion, and duplicate removal."
        },
        {
            "item": "repaired_model_ready_observations",
            "count": int(len(wide)),
            "definition": "Unique reporting-period observations defined as year + valid period."
        },
        {
            "item": "unique_years",
            "count": int(df["year"].nunique()) if not df.empty else 0,
            "definition": "Number of fiscal years retained."
        },
        {
            "item": "unique_periods",
            "count": int(df["period"].nunique()) if not df.empty else 0,
            "definition": "Number of valid fiscal period types retained."
        },
        {
            "item": "unique_categories",
            "count": int(df["category"].nunique()) if not df.empty else 0,
            "definition": "Number of repaired fiscal categories retained."
        },
        {
            "item": "unique_indicators",
            "count": int(df["normalized_indicator"].nunique()) if not df.empty else 0,
            "definition": "Number of repaired normalized indicators retained."
        },
    ]
    return pd.DataFrame(rows)


def build_coverage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(["year", "period", "period_order", "category", "normalized_indicator"], dropna=False)
        .agg(
            n_records=("value", "count"),
            n_labels=("label", "nunique"),
            n_source_files=("source_file", "nunique"),
            n_source_sheets=("sheet_name", "nunique"),
            value_mean=("value", "mean"),
            value_median=("value", "median"),
            value_std=("value", "std"),
            value_min=("value", "min"),
            value_max=("value", "max"),
        )
        .reset_index()
        .sort_values(["year", "period_order", "category", "normalized_indicator"])
    )


# =============================================================================
# Reporting
# =============================================================================

def write_report(summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("PERIOD REPAIR AND FEATURE-MAPPING REPORT")
    lines.append("=" * 80)
    lines.append(f"Run time: {summary['run_time']}")
    lines.append(f"Input file: {summary['input_file']}")
    lines.append(f"Results directory: {summary['results_dir']}")
    lines.append("")
    lines.append("1. Repair summary")
    lines.append("-" * 80)
    lines.append(f"Input long records: {summary['input_long_records']}")
    lines.append(f"Records retained after valid-period repair: {summary['records_after_valid_period_repair']}")
    lines.append(f"Invalid-period records excluded: {summary['invalid_period_records_excluded']}")
    lines.append(f"Duplicates removed after repair: {summary['duplicates_removed_after_repair']}")
    lines.append(f"Final repaired long records: {summary['final_repaired_long_records']}")
    lines.append("")
    lines.append("2. Model-ready dataset")
    lines.append("-" * 80)
    lines.append(f"Model-ready observations: {summary['model_ready_observations']}")
    lines.append(f"Model-ready features excluding identifiers: {summary['model_ready_features_excluding_identifiers']}")
    lines.append(f"Years retained: {summary['years_retained']}")
    lines.append(f"Periods retained: {summary['periods_retained']}")
    lines.append(f"Categories retained: {summary['categories_retained']}")
    lines.append(f"Indicators retained: {summary['indicators_retained']}")
    lines.append("")
    lines.append("3. Why this matters")
    lines.append("-" * 80)
    lines.append("Invalid fiscal periods such as debt labels, item names, or generic column names were excluded")
    lines.append("or repaired using file-level and sheet-level reporting-period evidence. The resulting dataset")
    lines.append("contains only valid fiscal reporting periods and is safer for governance metric computation.")
    lines.append("")
    lines.append("4. Next step")
    lines.append("-" * 80)
    lines.append("Update 04_define_governance_metrics_and_labels.py so INPUT_DIR points to:")
    lines.append(str(RESULTS_DIR))
    lines.append("Then rerun script 04.")

    (RESULTS_DIR / "repair_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("03B REPAIR PERIODS AND FEATURE MAPPING")
    print("=" * 80)
    print(f"Input file:        {INPUT_CLEANED_LONG}")
    print(f"Results directory: {RESULTS_DIR}")
    print("-" * 80)

    raw = load_input()
    raw_n = len(raw)

    # Ensure numeric value.
    raw["value"] = pd.to_numeric(raw.get("value", np.nan), errors="coerce")
    raw = raw.dropna(subset=["value"]).copy()

    year_fixed = repair_years(raw)
    period_repaired, invalid_period_records, period_log = repair_periods(year_fixed)
    indicator_repaired = repair_categories_and_indicators(period_repaired)
    final_long, duplicates = remove_duplicates(indicator_repaired)

    final_long = final_long.sort_values(["year", "period_order", "category", "normalized_indicator", "label"]).reset_index(drop=True)

    wide = build_wide_dataset(final_long)
    feature_dict = build_feature_dictionary(wide)
    concept_candidates = build_concept_mapping_candidates(wide)
    obs_counts = build_observation_counts(final_long, wide)
    coverage = build_coverage(final_long)

    # Save outputs.
    final_long.to_csv(RESULTS_DIR / "repaired_cleaned_long_dataset.csv", index=False, encoding="utf-8-sig")
    wide.to_csv(RESULTS_DIR / "repaired_model_ready_wide_dataset.csv", index=False, encoding="utf-8-sig")
    feature_dict.to_csv(RESULTS_DIR / "repaired_model_feature_dictionary.csv", index=False, encoding="utf-8-sig")
    period_log.to_csv(RESULTS_DIR / "period_repair_log.csv", index=False, encoding="utf-8-sig")
    invalid_period_records.to_csv(RESULTS_DIR / "invalid_period_records.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(RESULTS_DIR / "duplicates_removed_after_repair.csv", index=False, encoding="utf-8-sig")
    concept_candidates.to_csv(RESULTS_DIR / "concept_mapping_candidates.csv", index=False, encoding="utf-8-sig")
    obs_counts.to_csv(RESULTS_DIR / "repaired_observation_unit_counts.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(RESULTS_DIR / "repaired_category_period_coverage.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(RESULTS_DIR / "repaired_dataset_preview.xlsx", engine="openpyxl") as writer:
        final_long.head(1000).to_excel(writer, sheet_name="repaired_long_preview", index=False)
        wide.to_excel(writer, sheet_name="repaired_wide", index=False)
        feature_dict.to_excel(writer, sheet_name="feature_dictionary", index=False)
        concept_candidates.to_excel(writer, sheet_name="concept_candidates", index=False)
        obs_counts.to_excel(writer, sheet_name="observation_counts", index=False)
        coverage.head(1000).to_excel(writer, sheet_name="coverage_preview", index=False)

    years = sorted(final_long["year"].dropna().astype(int).unique().tolist()) if not final_long.empty else []
    periods = sorted(final_long["period"].dropna().unique().tolist(), key=lambda p: PERIOD_ORDER.get(p, 99)) if not final_long.empty else []
    cats = sorted(final_long["category"].dropna().unique().tolist()) if not final_long.empty else []
    inds = sorted(final_long["normalized_indicator"].dropna().unique().tolist()) if not final_long.empty else []

    feature_count = len([c for c in wide.columns if c not in {"observation_id", "year", "period", "period_order"}]) if not wide.empty else 0

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(INPUT_CLEANED_LONG),
        "results_dir": str(RESULTS_DIR),
        "input_long_records": int(raw_n),
        "records_after_year_repair": int(len(year_fixed)),
        "records_after_valid_period_repair": int(len(period_repaired)),
        "invalid_period_records_excluded": int(len(invalid_period_records)),
        "duplicates_removed_after_repair": int(len(duplicates)),
        "final_repaired_long_records": int(len(final_long)),
        "period_repair_log_records": int(len(period_log)),
        "model_ready_observations": int(len(wide)),
        "model_ready_features_excluding_identifiers": int(feature_count),
        "years_retained": years,
        "periods_retained": periods,
        "categories_retained": cats,
        "indicators_retained": inds,
        "outputs": {
            "repaired_cleaned_long_dataset": str(RESULTS_DIR / "repaired_cleaned_long_dataset.csv"),
            "repaired_model_ready_wide_dataset": str(RESULTS_DIR / "repaired_model_ready_wide_dataset.csv"),
            "repaired_model_feature_dictionary": str(RESULTS_DIR / "repaired_model_feature_dictionary.csv"),
            "period_repair_log": str(RESULTS_DIR / "period_repair_log.csv"),
            "invalid_period_records": str(RESULTS_DIR / "invalid_period_records.csv"),
            "concept_mapping_candidates": str(RESULTS_DIR / "concept_mapping_candidates.csv"),
            "repair_report": str(RESULTS_DIR / "repair_report.txt"),
        }
    }

    with (RESULTS_DIR / "repair_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_report(summary)

    print("[OK] Wrote repaired_cleaned_long_dataset.csv")
    print("[OK] Wrote repaired_model_ready_wide_dataset.csv")
    print("[OK] Wrote repaired_model_feature_dictionary.csv")
    print("[OK] Wrote period_repair_log.csv")
    print("[OK] Wrote invalid_period_records.csv")
    print("[OK] Wrote duplicates_removed_after_repair.csv")
    print("[OK] Wrote concept_mapping_candidates.csv")
    print("[OK] Wrote repaired_observation_unit_counts.csv")
    print("[OK] Wrote repaired_category_period_coverage.csv")
    print("[OK] Wrote repaired_dataset_preview.xlsx")
    print("[OK] Wrote repair_summary.json")
    print("[OK] Wrote repair_report.txt")
    print("-" * 80)
    print(f"Input long records: {raw_n}")
    print(f"Final repaired long records: {len(final_long)}")
    print(f"Invalid-period records excluded: {len(invalid_period_records)}")
    print(f"Duplicates removed after repair: {len(duplicates)}")
    print(f"Model-ready observations: {len(wide)}")
    print(f"Model-ready features excluding identifiers: {feature_count}")
    print(f"Periods retained: {periods}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
