r"""
02_extract_and_standardize_mof_tables.py

Purpose
-------
Extract and standardize the downloaded Saudi Ministry of Finance (MoF) Excel/CSV
budget-performance tables into a single long-format analytical dataset.

Why this script is needed
-------------------------
The audit showed that after web crawling, the project now has multi-year MoF data
covering 2017-2026. However, the Excel sheets have messy headers such as
"Unnamed: 1", "Unnamed: 2", etc. This script tries to identify real table headers,
clean rows, infer fiscal concepts, and convert all extractable tables into a unified
long-format structure.

Input folder
------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Data

Output folder
-------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\02_extract_and_standardize_mof_tables

Main outputs
------------
1. raw_sheet_profiles.csv
2. detected_headers.csv
3. standardized_long_dataset.csv
4. wide_clean_tables.xlsx
5. variable_dictionary.csv
6. data_quality_report.csv
7. extraction_summary.json
8. extraction_report.txt

How to run
----------
pip install pandas numpy openpyxl
python 02_extract_and_standardize_mof_tables.py

Important note
--------------
This script is intentionally conservative. It does not fabricate labels or AI targets.
It only creates a transparent analyzable dataset from official MoF tables.
"""

from __future__ import annotations

import json
import re
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results" / "02_extract_and_standardize_mof_tables"

SUPPORTED_EXCEL = {".xlsx", ".xls", ".xlsm"}
SUPPORTED_CSV = {".csv"}

MAX_HEADER_SCAN_ROWS = 15
MIN_HEADER_NONEMPTY_CELLS = 2
MIN_DATA_ROWS_AFTER_HEADER = 2
MAX_EXCEL_OUTPUT_SHEETS = 100


# =============================================================================
# Utilities
# =============================================================================

def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\n", " ")
    s = s.replace("\r", " ")
    s = s.replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_col_name(x: Any) -> str:
    s = clean_text(x)
    if not s or s.lower().startswith("unnamed"):
        return ""
    return s


def make_unique_columns(cols: List[str]) -> List[str]:
    out = []
    seen: Dict[str, int] = {}
    for i, c in enumerate(cols):
        c = clean_text(c)
        if not c:
            c = f"Column_{i+1}"
        base = c
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            c = f"{base}_{seen[base]}"
        out.append(c)
    return out


def infer_year_period_from_name(text: str) -> Tuple[Optional[int], Optional[str]]:
    lower = text.lower()

    year = None
    y = re.search(r"(20\d{2})", lower)
    if y:
        year = int(y.group(1))

    period = None
    patterns = [
        (r"\bq\s*1\b|\bq1\b|first\s+quarter|quarter\s*1", "Q1"),
        (r"\bq\s*2\b|\bq2\b|second\s+quarter|quarter\s*2", "Q2"),
        (r"\bq\s*3\b|\bq3\b|third\s+quarter|quarter\s*3", "Q3"),
        (r"\bq\s*4\b|\bq4\b|fourth\s+quarter|quarter\s*4", "Q4"),
        (r"mid[\s\-_]*year|half[\s\-_]*year|semi[\s\-_]*annual|h1", "Mid-Year/H1"),
        (r"end[\s\-_]*year|year[\s\-_]*end|annual|final|fy", "End-Year/Annual"),
    ]
    for pat, lab in patterns:
        if re.search(pat, lower):
            period = lab
            break

    return year, period


def parse_numeric_value(x: Any) -> Optional[float]:
    if pd.isna(x):
        return None

    s = str(x).strip()
    if not s:
        return None

    # Handle accounting negatives and Arabic minus variants.
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace(",", "")
    s = s.replace("%", "")
    s = s.replace("SAR", "")
    s = s.replace("sar", "")
    s = s.replace("million", "")
    s = s.replace("Million", "")
    s = s.strip()

    if re.match(r"^\(.*\)$", s):
        s = "-" + s[1:-1]

    # Keep only numeric-like fragments if text mixed with numbers.
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None

    try:
        return float(m.group(0))
    except Exception:
        return None


def numeric_fraction(row: pd.Series) -> float:
    if len(row) == 0:
        return 0.0
    vals = [parse_numeric_value(v) for v in row.tolist()]
    return sum(v is not None for v in vals) / len(vals)


def nonempty_count(row: pd.Series) -> int:
    return sum(bool(clean_text(v)) for v in row.tolist())


def looks_like_header_row(row: pd.Series) -> float:
    """
    Score a row as a possible header row.
    Higher is better.
    """
    values = [clean_text(v) for v in row.tolist()]
    nonempty = sum(bool(v) for v in values)
    if nonempty < MIN_HEADER_NONEMPTY_CELLS:
        return -999.0

    num_frac = numeric_fraction(row)
    text_frac = 1.0 - num_frac

    keyword_score = 0
    header_keywords = [
        "revenue", "revenues", "expenditure", "expenditures", "deficit",
        "surplus", "debt", "actual", "budget", "sector", "item",
        "classification", "q1", "q2", "q3", "q4", "quarter", "2020",
        "2021", "2022", "2023", "2024", "2025", "2026", "amount",
        "million", "sar", "chapter", "economic", "source"
    ]

    joined = " ".join(values).lower()
    for kw in header_keywords:
        if kw in joined:
            keyword_score += 1

    # Penalize rows that are mainly numeric, because they are likely data rows.
    return nonempty + 2.0 * keyword_score + 3.0 * text_frac - 4.0 * num_frac


def detect_header_row(df_raw: pd.DataFrame) -> int:
    max_rows = min(MAX_HEADER_SCAN_ROWS, len(df_raw))
    best_idx = 0
    best_score = -999999.0

    for idx in range(max_rows):
        row = df_raw.iloc[idx]
        score = looks_like_header_row(row)
        if score > best_score:
            best_score = score
            best_idx = idx

    return int(best_idx)


def flatten_possible_multirow_header(df_raw: pd.DataFrame, header_idx: int) -> List[str]:
    """
    Combines one or two rows as header if the next row contains useful header fragments.
    """
    row1 = [normalize_col_name(v) for v in df_raw.iloc[header_idx].tolist()]
    cols = row1

    if header_idx + 1 < len(df_raw):
        row2 = [normalize_col_name(v) for v in df_raw.iloc[header_idx + 1].tolist()]
        row2_num_frac = numeric_fraction(df_raw.iloc[header_idx + 1])
        row2_nonempty = sum(bool(x) for x in row2)

        # If the next row is still mostly text and contains header hints, combine.
        if row2_nonempty >= 2 and row2_num_frac < 0.40:
            combined = []
            for a, b in zip(row1, row2):
                if a and b and a.lower() != b.lower():
                    combined.append(f"{a} {b}")
                elif a:
                    combined.append(a)
                elif b:
                    combined.append(b)
                else:
                    combined.append("")
            cols = combined

    return make_unique_columns(cols)


def clean_table(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, int, List[str]]:
    if df_raw.empty:
        return pd.DataFrame(), 0, []

    # Remove fully empty rows and columns.
    df = df_raw.dropna(how="all").dropna(axis=1, how="all").copy()

    if df.empty:
        return pd.DataFrame(), 0, []

    header_idx = detect_header_row(df)
    cols = flatten_possible_multirow_header(df, header_idx)

    data_start = header_idx + 1

    # If next row was used as header-like, skip it when it is mostly text.
    if header_idx + 1 < len(df) and numeric_fraction(df.iloc[header_idx + 1]) < 0.40:
        if nonempty_count(df.iloc[header_idx + 1]) >= 2:
            data_start = header_idx + 2

    cleaned = df.iloc[data_start:].copy()
    cleaned.columns = cols[:len(cleaned.columns)]

    # Remove repeated header rows and fully empty rows.
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.loc[:, [c for c in cleaned.columns if clean_text(c)]]

    # Drop rows that are almost entirely empty.
    if not cleaned.empty:
        keep = cleaned.apply(lambda r: nonempty_count(r) >= 2, axis=1)
        cleaned = cleaned[keep].copy()

    return cleaned.reset_index(drop=True), header_idx, cols


def classify_indicator(text: str) -> str:
    t = text.lower()

    if "oil" in t and "non" not in t and "revenue" in t:
        return "Oil revenue"
    if ("non-oil" in t or "non oil" in t) and "revenue" in t:
        return "Non-oil revenue"
    if "revenue" in t:
        return "Revenue"
    if "expenditure" in t or "expense" in t or "expenses" in t:
        return "Expenditure"
    if "deficit" in t or "surplus" in t:
        return "Surplus/Deficit"
    if "debt" in t:
        return "Debt"
    if "compensation" in t or "employees" in t or "wages" in t:
        return "Compensation of employees"
    if "goods" in t or "services" in t:
        return "Goods and services"
    if "social" in t or "benefits" in t:
        return "Social benefits"
    if "asset" in t or "capital" in t:
        return "Capital / assets"
    return "Other"


def infer_unit_from_context(text: str) -> str:
    t = text.lower()
    if "%" in t or "percent" in t or "percentage" in t:
        return "percent"
    if "million" in t or "sar" in t:
        return "million_sar"
    return "unspecified"


def find_label_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for c in df.columns:
        s = df[c]
        values = s.dropna().astype(str).map(clean_text)
        if values.empty:
            continue

        numeric_like_count = sum(parse_numeric_value(v) is not None for v in values)
        numeric_share = numeric_like_count / len(values)

        if numeric_share < 0.40:
            candidates.append(c)

    return candidates[:3]


def find_value_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for c in df.columns:
        s = df[c]
        values = s.dropna()
        if values.empty:
            continue
        numeric_like_count = sum(parse_numeric_value(v) is not None for v in values)
        numeric_share = numeric_like_count / len(values)

        if numeric_share >= 0.50:
            candidates.append(c)

    return candidates


def infer_value_period(column_name: str, default_period: Optional[str]) -> Optional[str]:
    text = column_name.lower()
    _, p = infer_year_period_from_name(text)
    if p:
        return p
    return default_period


def infer_value_year(column_name: str, default_year: Optional[int]) -> Optional[int]:
    y, _ = infer_year_period_from_name(column_name)
    if y:
        return y
    return default_year


def table_to_long(
    cleaned: pd.DataFrame,
    source_file: Path,
    sheet_name: str,
    file_year: Optional[int],
    file_period: Optional[str],
) -> pd.DataFrame:
    if cleaned.empty:
        return pd.DataFrame()

    label_cols = find_label_columns(cleaned)
    value_cols = find_value_columns(cleaned)

    if not value_cols:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for row_idx, row in cleaned.iterrows():
        label_parts = []
        for c in label_cols:
            txt = clean_text(row.get(c, ""))
            if txt:
                label_parts.append(txt)

        row_label = " | ".join(label_parts).strip()
        if not row_label:
            row_label = "Unlabeled row"

        category = classify_indicator(row_label + " " + sheet_name)
        unit_from_label = infer_unit_from_context(row_label + " " + sheet_name)

        for vc in value_cols:
            raw_value = row.get(vc, None)
            value = parse_numeric_value(raw_value)

            if value is None:
                continue

            value_year = infer_value_year(str(vc), file_year)
            value_period = infer_value_period(str(vc), file_period)
            unit = infer_unit_from_context(str(vc))
            if unit == "unspecified":
                unit = unit_from_label

            rows.append({
                "source_file": source_file.name,
                "source_relative_path": str(source_file.relative_to(DATA_DIR)),
                "sheet_name": str(sheet_name),
                "source_sha256": sha256_file(source_file),
                "file_year_guess": file_year,
                "file_period_guess": file_period,
                "year": value_year,
                "period": value_period,
                "row_number_in_cleaned_table": int(row_idx),
                "label": row_label,
                "category": category,
                "value_column": str(vc),
                "value": value,
                "raw_value": clean_text(raw_value),
                "unit": unit,
                "label_columns_used": " | ".join(label_cols),
            })

    return pd.DataFrame(rows)


def read_workbook_or_csv(path: Path) -> Dict[str, pd.DataFrame]:
    ext = path.suffix.lower()

    if ext in SUPPORTED_EXCEL:
        try:
            return pd.read_excel(path, sheet_name=None, header=None, engine=None)
        except Exception as e:
            print(f"[WARN] Cannot read Excel: {path.name} -> {e}")
            return {}

    if ext in SUPPORTED_CSV:
        for enc in ["utf-8-sig", "utf-8", "cp1256", "latin1"]:
            try:
                return {"CSV": pd.read_csv(path, header=None, encoding=enc)}
            except Exception:
                pass
        print(f"[WARN] Cannot read CSV: {path.name}")
        return {}

    return {}


def collect_tabular_files() -> List[Path]:
    files = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXCEL | SUPPORTED_CSV:
            files.append(p)
    return sorted(files)


def build_variable_dictionary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    rows = []
    for category, sdf in long_df.groupby("category", dropna=False):
        rows.append({
            "category": category,
            "n_records": int(len(sdf)),
            "n_years": int(sdf["year"].nunique(dropna=True)),
            "years": ", ".join(map(str, sorted(sdf["year"].dropna().astype(int).unique()))) if sdf["year"].notna().any() else "",
            "periods": ", ".join(sorted(sdf["period"].dropna().astype(str).unique())),
            "units": ", ".join(sorted(sdf["unit"].dropna().astype(str).unique())),
            "example_labels": " || ".join(sdf["label"].dropna().astype(str).head(5).tolist()),
        })

    return pd.DataFrame(rows).sort_values(["category"])


def build_quality_report(long_df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add_check(name: str, value: Any, interpretation: str) -> None:
        checks.append({
            "check": name,
            "value": value,
            "interpretation": interpretation,
        })

    add_check("total_long_records", int(len(long_df)), "Number of numeric observations extracted into the standardized long dataset.")

    if long_df.empty:
        return pd.DataFrame(checks)

    add_check("unique_source_files", int(long_df["source_file"].nunique()), "Number of Excel/CSV files contributing data.")
    add_check("unique_sheets", int(long_df[["source_file", "sheet_name"]].drop_duplicates().shape[0]), "Number of source sheets contributing data.")
    add_check("unique_years", int(long_df["year"].nunique(dropna=True)), "Number of detected fiscal years.")
    add_check("year_range", f"{long_df['year'].min()} to {long_df['year'].max()}", "Detected year range after extraction.")
    add_check("unique_periods", int(long_df["period"].nunique(dropna=True)), "Number of detected fiscal periods.")
    add_check("records_missing_year", int(long_df["year"].isna().sum()), "Records where year could not be inferred.")
    add_check("records_missing_period", int(long_df["period"].isna().sum()), "Records where period could not be inferred.")
    add_check("unique_categories", int(long_df["category"].nunique(dropna=True)), "Number of inferred fiscal indicator categories.")
    add_check("duplicate_source_label_value_records", int(long_df.duplicated(subset=["source_file", "sheet_name", "year", "period", "label", "value_column", "value"]).sum()), "Possible duplicate extracted records.")

    return pd.DataFrame(checks)


def write_report(summary: Dict[str, Any], quality_df: pd.DataFrame) -> None:
    lines = []
    lines.append("MOF TABLE EXTRACTION AND STANDARDIZATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Run time: {summary['run_time']}")
    lines.append(f"Data directory: {summary['data_dir']}")
    lines.append(f"Results directory: {summary['results_dir']}")
    lines.append("")
    lines.append("1. Extraction coverage")
    lines.append("-" * 80)
    lines.append(f"Tabular files scanned: {summary['tabular_files_scanned']}")
    lines.append(f"Sheets/CSVs scanned: {summary['sheets_scanned']}")
    lines.append(f"Sheets with extracted long records: {summary['sheets_with_long_records']}")
    lines.append(f"Long-format records extracted: {summary['long_records_extracted']}")
    lines.append("")
    lines.append("2. Dataset readiness")
    lines.append("-" * 80)
    lines.append(f"Years detected: {summary['years_detected']}")
    lines.append(f"Periods detected: {summary['periods_detected']}")
    lines.append(f"Categories detected: {summary['categories_detected']}")
    lines.append("")
    lines.append("3. Quality checks")
    lines.append("-" * 80)

    if not quality_df.empty:
        for _, r in quality_df.iterrows():
            lines.append(f"{r['check']}: {r['value']} -- {r['interpretation']}")

    lines.append("")
    lines.append("4. Recommended next step")
    lines.append("-" * 80)
    lines.append("Proceed to 03_build_long_format_budget_dataset.py to construct analytical features,")
    lines.append("define official observation units, remove duplicates, and prepare model-ready matrices.")

    (RESULTS_DIR / "extraction_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dirs()

    print("=" * 80)
    print("02 EXTRACT AND STANDARDIZE MOF TABLES")
    print("=" * 80)
    print(f"Data directory:    {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("-" * 80)

    files = collect_tabular_files()
    print(f"Tabular files found: {len(files)}")

    sheet_profile_rows: List[Dict[str, Any]] = []
    header_rows: List[Dict[str, Any]] = []
    long_frames: List[pd.DataFrame] = []
    clean_tables_for_excel: Dict[str, pd.DataFrame] = {}

    sheets_scanned = 0
    sheets_with_long = 0

    for fidx, path in enumerate(files, start=1):
        year, period = infer_year_period_from_name(path.name)
        workbook = read_workbook_or_csv(path)

        if not workbook:
            continue

        for sheet_name, raw_df in workbook.items():
            sheets_scanned += 1

            cleaned, header_idx, cols = clean_table(raw_df)

            sheet_profile_rows.append({
                "source_file": path.name,
                "source_relative_path": str(path.relative_to(DATA_DIR)),
                "sheet_name": str(sheet_name),
                "file_year_guess": year,
                "file_period_guess": period,
                "raw_rows": int(raw_df.shape[0]),
                "raw_columns": int(raw_df.shape[1]),
                "cleaned_rows": int(cleaned.shape[0]),
                "cleaned_columns": int(cleaned.shape[1]),
                "detected_header_row_zero_based": int(header_idx),
                "value_columns_detected": " | ".join(find_value_columns(cleaned)) if not cleaned.empty else "",
                "label_columns_detected": " | ".join(find_label_columns(cleaned)) if not cleaned.empty else "",
            })

            header_rows.append({
                "source_file": path.name,
                "sheet_name": str(sheet_name),
                "detected_header_row_zero_based": int(header_idx),
                "detected_columns": " | ".join(cols),
            })

            long_df = table_to_long(
                cleaned=cleaned,
                source_file=path,
                sheet_name=str(sheet_name),
                file_year=year,
                file_period=period,
            )

            if not long_df.empty:
                sheets_with_long += 1
                long_frames.append(long_df)

                if len(clean_tables_for_excel) < MAX_EXCEL_OUTPUT_SHEETS:
                    safe_name = re.sub(r"[\[\]\*\?/\\:]", "_", f"{path.stem[:15]}_{str(sheet_name)[:12]}")[:31]
                    clean_tables_for_excel[safe_name] = cleaned.head(500)

        print(f"[{fidx}/{len(files)}] Processed: {path.name}")

    raw_sheet_profiles = pd.DataFrame(sheet_profile_rows)
    detected_headers = pd.DataFrame(header_rows)

    if long_frames:
        long_dataset = pd.concat(long_frames, ignore_index=True)
    else:
        long_dataset = pd.DataFrame(columns=[
            "source_file", "source_relative_path", "sheet_name", "source_sha256",
            "file_year_guess", "file_period_guess", "year", "period",
            "row_number_in_cleaned_table", "label", "category", "value_column",
            "value", "raw_value", "unit", "label_columns_used"
        ])

    # Basic cleanup and ordering.
    if not long_dataset.empty:
        long_dataset["year"] = pd.to_numeric(long_dataset["year"], errors="coerce")
        long_dataset["value"] = pd.to_numeric(long_dataset["value"], errors="coerce")
        long_dataset = long_dataset.dropna(subset=["value"]).copy()
        long_dataset = long_dataset.sort_values(
            by=["year", "period", "category", "label", "source_file"],
            na_position="last"
        ).reset_index(drop=True)

    variable_dictionary = build_variable_dictionary(long_dataset)
    quality_report = build_quality_report(long_dataset)

    # Save outputs.
    raw_sheet_profiles.to_csv(RESULTS_DIR / "raw_sheet_profiles.csv", index=False, encoding="utf-8-sig")
    detected_headers.to_csv(RESULTS_DIR / "detected_headers.csv", index=False, encoding="utf-8-sig")
    long_dataset.to_csv(RESULTS_DIR / "standardized_long_dataset.csv", index=False, encoding="utf-8-sig")
    variable_dictionary.to_csv(RESULTS_DIR / "variable_dictionary.csv", index=False, encoding="utf-8-sig")
    quality_report.to_csv(RESULTS_DIR / "data_quality_report.csv", index=False, encoding="utf-8-sig")

    if clean_tables_for_excel:
        with pd.ExcelWriter(RESULTS_DIR / "wide_clean_tables.xlsx", engine="openpyxl") as writer:
            for sheet_name, df in clean_tables_for_excel.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    years_detected = []
    if not long_dataset.empty and long_dataset["year"].notna().any():
        years_detected = sorted(long_dataset["year"].dropna().astype(int).unique().tolist())

    periods_detected = []
    if not long_dataset.empty:
        periods_detected = sorted(long_dataset["period"].dropna().astype(str).unique().tolist())

    categories_detected = []
    if not long_dataset.empty:
        categories_detected = sorted(long_dataset["category"].dropna().astype(str).unique().tolist())

    summary = {
        "run_time": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(DATA_DIR),
        "results_dir": str(RESULTS_DIR),
        "tabular_files_scanned": int(len(files)),
        "sheets_scanned": int(sheets_scanned),
        "sheets_with_long_records": int(sheets_with_long),
        "long_records_extracted": int(len(long_dataset)),
        "years_detected": years_detected,
        "periods_detected": periods_detected,
        "categories_detected": categories_detected,
        "outputs": {
            "raw_sheet_profiles": str(RESULTS_DIR / "raw_sheet_profiles.csv"),
            "detected_headers": str(RESULTS_DIR / "detected_headers.csv"),
            "standardized_long_dataset": str(RESULTS_DIR / "standardized_long_dataset.csv"),
            "wide_clean_tables": str(RESULTS_DIR / "wide_clean_tables.xlsx"),
            "variable_dictionary": str(RESULTS_DIR / "variable_dictionary.csv"),
            "data_quality_report": str(RESULTS_DIR / "data_quality_report.csv"),
            "extraction_report": str(RESULTS_DIR / "extraction_report.txt"),
        },
    }

    with (RESULTS_DIR / "extraction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_report(summary, quality_report)

    print("[OK] Wrote raw_sheet_profiles.csv")
    print("[OK] Wrote detected_headers.csv")
    print("[OK] Wrote standardized_long_dataset.csv")
    print("[OK] Wrote variable_dictionary.csv")
    print("[OK] Wrote data_quality_report.csv")
    if clean_tables_for_excel:
        print("[OK] Wrote wide_clean_tables.xlsx")
    print("[OK] Wrote extraction_summary.json")
    print("[OK] Wrote extraction_report.txt")
    print("-" * 80)
    print(f"Long-format records extracted: {len(long_dataset)}")
    print(f"Years detected: {years_detected}")
    print(f"Periods detected: {periods_detected}")
    print(f"Categories detected: {categories_detected}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
