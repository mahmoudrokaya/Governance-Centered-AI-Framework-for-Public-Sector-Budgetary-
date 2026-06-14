r"""
01_data_inventory_and_audit.py

Purpose
-------
Audit the available Saudi Ministry of Finance (MoF) budget-performance data before
building any new empirical model.

This script answers the reviewers' first methodological concern:
- What files exist?
- What years / quarters are covered?
- Which files are Excel/PDF/CSV?
- Which Excel sheets and variables are available?
- How many extractable rows and columns exist?
- Which numeric variables can support analysis?
- Is the dataset large enough for machine learning, or only for descriptive/rule-based analysis?
- What evidence can be reported in the paper before building the proposed method?

Expected local paths
--------------------
Data folder:
    E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Data

Script folder:
    E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Scripts

Result folder created automatically:
    E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\01_data_inventory_and_audit

Outputs
-------
1. file_inventory.csv
2. excel_sheet_inventory.csv
3. excel_column_inventory.csv
4. numeric_variable_summary.csv
5. pdf_inventory.csv
6. extracted_excel_preview.xlsx
7. audit_summary.json
8. audit_report.txt

Notes
-----
- This script does NOT train any model.
- It only audits the current data and tells us what is scientifically defensible.
- PDF table extraction is optional and not required here. PDFs are inventoried only.
"""

from __future__ import annotations

import json
import re
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")
DATA_DIR = BASE_DIR / "Data"
SCRIPTS_DIR = BASE_DIR / "Scripts"
RESULTS_DIR = BASE_DIR / "Results" / "01_data_inventory_and_audit"

SUPPORTED_EXCEL = {".xlsx", ".xls", ".xlsm"}
SUPPORTED_CSV = {".csv"}
SUPPORTED_PDF = {".pdf"}
SUPPORTED_DATA_EXTENSIONS = SUPPORTED_EXCEL | SUPPORTED_CSV | SUPPORTED_PDF

MAX_PREVIEW_ROWS_PER_SHEET = 20


# =============================================================================
# Utilities
# =============================================================================

def ensure_directories() -> None:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def file_hash(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def safe_read_text_fragment(path: Path, n_bytes: int = 4096) -> str:
    try:
        raw = path.read_bytes()[:n_bytes]
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def infer_year_quarter_from_name(name: str) -> Tuple[Optional[int], Optional[str]]:
    lower = name.lower()

    year = None
    ymatch = re.search(r"(20\d{2})", lower)
    if ymatch:
        year = int(ymatch.group(1))

    quarter = None
    patterns = [
        (r"\bq\s*1\b|\bq1\b|quarter\s*1|first\s+quarter", "Q1"),
        (r"\bq\s*2\b|\bq2\b|quarter\s*2|second\s+quarter", "Q2"),
        (r"\bq\s*3\b|\bq3\b|quarter\s*3|third\s+quarter", "Q3"),
        (r"\bq\s*4\b|\bq4\b|quarter\s*4|fourth\s+quarter", "Q4"),
        (r"mid[\s\-_]*year|half[\s\-_]*year|h1", "Mid-Year/H1"),
        (r"end[\s\-_]*year|year[\s\-_]*end|annual|fy", "End-Year/Annual"),
    ]
    for pat, label in patterns:
        if re.search(pat, lower):
            quarter = label
            break

    return year, quarter


def classify_file_role(name: str) -> str:
    lower = name.lower()
    if "infographic" in lower or "info" in lower:
        return "infographic_or_visual_summary"
    if "budget" in lower and "performance" in lower:
        return "budget_performance_report"
    if "performance" in lower:
        return "performance_report"
    if "statement" in lower:
        return "statement"
    return "unknown"


def clean_column_name(col: Any) -> str:
    text = str(col).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text


def is_probably_numeric_series(s: pd.Series) -> bool:
    if s.empty:
        return False

    converted = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("−", "-", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    numeric = pd.to_numeric(converted, errors="coerce")
    non_null = s.notna().sum()
    if non_null == 0:
        return False
    return numeric.notna().sum() / non_null >= 0.60


def coerce_numeric(s: pd.Series) -> pd.Series:
    converted = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("−", "-", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(converted, errors="coerce")


def read_excel_safely(path: Path) -> Dict[str, pd.DataFrame]:
    try:
        return pd.read_excel(path, sheet_name=None, engine=None)
    except Exception as e:
        print(f"[WARN] Failed to read Excel file: {path.name} -> {e}")
        return {}


def read_csv_safely(path: Path) -> Optional[pd.DataFrame]:
    encodings = ["utf-8-sig", "utf-8", "cp1256", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    print(f"[WARN] Failed to read CSV file: {path.name}")
    return None


# =============================================================================
# Inventory
# =============================================================================

def collect_files() -> List[Path]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    files = [
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
    ]
    return sorted(files)


def build_file_inventory(files: List[Path]) -> pd.DataFrame:
    rows = []

    for p in files:
        year, quarter = infer_year_quarter_from_name(p.name)

        try:
            stat = p.stat()
            size_bytes = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        except Exception:
            size_bytes = None
            modified = None

        try:
            sha256 = file_hash(p)
        except Exception:
            sha256 = None

        rows.append({
            "file_name": p.name,
            "relative_path": str(p.relative_to(DATA_DIR)),
            "absolute_path": str(p),
            "extension": p.suffix.lower(),
            "file_role_guess": classify_file_role(p.name),
            "year_guess": year,
            "period_guess": quarter,
            "size_bytes": size_bytes,
            "modified_time": modified,
            "sha256": sha256,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Excel / CSV audit
# =============================================================================

def audit_tabular_files(files: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    sheet_rows: List[Dict[str, Any]] = []
    column_rows: List[Dict[str, Any]] = []
    numeric_rows: List[Dict[str, Any]] = []
    preview_sheets: Dict[str, pd.DataFrame] = {}

    for p in files:
        ext = p.suffix.lower()
        tables: Dict[str, pd.DataFrame] = {}

        if ext in SUPPORTED_EXCEL:
            tables = read_excel_safely(p)
        elif ext in SUPPORTED_CSV:
            df = read_csv_safely(p)
            if df is not None:
                tables = {"CSV": df}

        if not tables:
            continue

        year, period = infer_year_quarter_from_name(p.name)

        for sheet_name, df in tables.items():
            if df is None:
                continue

            original_shape = df.shape

            # Drop fully empty rows and columns for audit only.
            df2 = df.dropna(how="all").dropna(axis=1, how="all").copy()
            df2.columns = [clean_column_name(c) for c in df2.columns]

            non_empty_rows = int(df2.dropna(how="all").shape[0])
            non_empty_cols = int(df2.dropna(axis=1, how="all").shape[1])

            sheet_key = f"{p.stem[:20]}__{str(sheet_name)[:20]}"
            preview_sheets[sheet_key[:31]] = df2.head(MAX_PREVIEW_ROWS_PER_SHEET)

            numeric_col_count = 0
            text_col_count = 0

            for col in df2.columns:
                s = df2[col]
                non_null = int(s.notna().sum())
                unique_count = int(s.nunique(dropna=True))
                numeric_like = is_probably_numeric_series(s)

                if numeric_like:
                    numeric_col_count += 1
                    num = coerce_numeric(s)
                    numeric_rows.append({
                        "file_name": p.name,
                        "sheet_name": sheet_name,
                        "year_guess": year,
                        "period_guess": period,
                        "column_name": col,
                        "non_null_count": int(num.notna().sum()),
                        "mean": float(num.mean()) if num.notna().any() else np.nan,
                        "std": float(num.std(ddof=1)) if num.notna().sum() > 1 else np.nan,
                        "min": float(num.min()) if num.notna().any() else np.nan,
                        "q25": float(num.quantile(0.25)) if num.notna().any() else np.nan,
                        "median": float(num.median()) if num.notna().any() else np.nan,
                        "q75": float(num.quantile(0.75)) if num.notna().any() else np.nan,
                        "max": float(num.max()) if num.notna().any() else np.nan,
                    })
                else:
                    text_col_count += 1

                examples = (
                    s.dropna()
                    .astype(str)
                    .head(5)
                    .tolist()
                )

                column_rows.append({
                    "file_name": p.name,
                    "sheet_name": sheet_name,
                    "year_guess": year,
                    "period_guess": period,
                    "column_name": col,
                    "non_null_count": non_null,
                    "unique_count": unique_count,
                    "numeric_like": bool(numeric_like),
                    "example_values": " | ".join(examples),
                })

            sheet_rows.append({
                "file_name": p.name,
                "sheet_name": sheet_name,
                "year_guess": year,
                "period_guess": period,
                "original_rows": original_shape[0],
                "original_columns": original_shape[1],
                "non_empty_rows": non_empty_rows,
                "non_empty_columns": non_empty_cols,
                "numeric_like_columns": numeric_col_count,
                "text_like_columns": text_col_count,
            })

    return (
        pd.DataFrame(sheet_rows),
        pd.DataFrame(column_rows),
        pd.DataFrame(numeric_rows),
        preview_sheets,
    )


# =============================================================================
# PDF audit
# =============================================================================

def audit_pdfs(files: List[Path]) -> pd.DataFrame:
    rows = []

    for p in files:
        if p.suffix.lower() != ".pdf":
            continue

        year, period = infer_year_quarter_from_name(p.name)

        # We do not depend on PDF parsing libraries here.
        # We only inspect file metadata and header.
        fragment = safe_read_text_fragment(p, n_bytes=2048)
        header_hint = fragment[:50].replace("\n", " ")

        rows.append({
            "file_name": p.name,
            "relative_path": str(p.relative_to(DATA_DIR)),
            "year_guess": year,
            "period_guess": period,
            "size_bytes": p.stat().st_size if p.exists() else None,
            "pdf_header_hint": header_hint,
            "table_extraction_status": "not_attempted_in_audit_script",
            "note": "Use later script with tabula/camelot/pdfplumber if PDF tables are required.",
        })

    return pd.DataFrame(rows)


# =============================================================================
# Scientific sufficiency assessment
# =============================================================================

def assess_scientific_sufficiency(
    file_inventory: pd.DataFrame,
    sheet_inventory: pd.DataFrame,
    column_inventory: pd.DataFrame,
    numeric_summary: pd.DataFrame,
) -> Dict[str, Any]:
    years = sorted([
        int(x) for x in file_inventory["year_guess"].dropna().unique().tolist()
    ]) if not file_inventory.empty and "year_guess" in file_inventory else []

    periods = sorted([
        str(x) for x in file_inventory["period_guess"].dropna().unique().tolist()
    ]) if not file_inventory.empty and "period_guess" in file_inventory else []

    n_files = int(len(file_inventory))
    n_excel_csv_files = int(file_inventory["extension"].isin(list(SUPPORTED_EXCEL | SUPPORTED_CSV)).sum()) if not file_inventory.empty else 0
    n_pdf_files = int((file_inventory["extension"] == ".pdf").sum()) if not file_inventory.empty else 0

    n_sheets = int(len(sheet_inventory))
    total_non_empty_rows = int(sheet_inventory["non_empty_rows"].sum()) if not sheet_inventory.empty else 0
    total_numeric_columns = int(sheet_inventory["numeric_like_columns"].sum()) if not sheet_inventory.empty else 0

    unique_numeric_variables = (
        sorted(numeric_summary["column_name"].dropna().astype(str).unique().tolist())
        if not numeric_summary.empty else []
    )

    # Conservative interpretation for reviewers:
    # With only 2023 quarterly reports, ML claims are weak unless rows come from many sectors/items.
    distinct_periods = len(periods)
    distinct_years = len(years)

    if distinct_years <= 1 and distinct_periods <= 6:
        ml_sufficiency = "weak_for_supervised_machine_learning"
        recommended_design = (
            "Use transparent rule-based/statistical governance scoring, or expand the dataset "
            "to multiple years/sectors before claiming supervised AI model training."
        )
    elif total_non_empty_rows < 100:
        ml_sufficiency = "limited_for_machine_learning"
        recommended_design = (
            "Avoid strong ML claims. Use descriptive statistics, deterministic indicators, "
            "sensitivity analysis, and clearly defined formulas."
        )
    else:
        ml_sufficiency = "potentially_usable_after_cleaning"
        recommended_design = (
            "Proceed to construct a unified long-format dataset, define labels/thresholds, "
            "and then evaluate baseline/statistical/AI/proposed methods using a reproducible split."
        )

    reviewer_implications = {
        "ai_model_specification": (
            "Current audit does not prove that a supervised AI model is justified. "
            "A later script must define the target variable, input variables, labels, algorithm, "
            "hyperparameters, and output-generation process."
        ),
        "metric_definition": (
            "Governance metrics must be formula-based or evaluator-rubric-based. "
            "The audit only identifies available variables; it does not justify scores such as 4.6/5."
        ),
        "validation_protocol": (
            "Validation must match the actual number of independent observations. "
            "If only quarterly records exist, k-fold supervised validation is not defensible."
        ),
        "statistical_evidence": (
            "Later scripts must produce uncertainty estimates, sensitivity analysis, and confidence intervals "
            "where statistically meaningful."
        ),
    }

    return {
        "audit_time": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(DATA_DIR),
        "results_dir": str(RESULTS_DIR),
        "n_files": n_files,
        "n_excel_or_csv_files": n_excel_csv_files,
        "n_pdf_files": n_pdf_files,
        "years_detected": years,
        "periods_detected": periods,
        "n_tabular_sheets_or_csvs": n_sheets,
        "total_non_empty_rows_across_sheets": total_non_empty_rows,
        "total_numeric_like_columns_across_sheets": total_numeric_columns,
        "unique_numeric_variable_count": len(unique_numeric_variables),
        "unique_numeric_variables": unique_numeric_variables[:200],
        "machine_learning_sufficiency": ml_sufficiency,
        "recommended_next_design": recommended_design,
        "reviewer_implications": reviewer_implications,
    }


def write_text_report(summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("DATA INVENTORY AND AUDIT REPORT")
    lines.append("=" * 80)
    lines.append(f"Audit time: {summary['audit_time']}")
    lines.append(f"Data directory: {summary['data_dir']}")
    lines.append(f"Results directory: {summary['results_dir']}")
    lines.append("")
    lines.append("1. File coverage")
    lines.append("-" * 80)
    lines.append(f"Total supported files: {summary['n_files']}")
    lines.append(f"Excel/CSV files: {summary['n_excel_or_csv_files']}")
    lines.append(f"PDF files: {summary['n_pdf_files']}")
    lines.append(f"Years detected: {summary['years_detected']}")
    lines.append(f"Periods detected: {summary['periods_detected']}")
    lines.append("")
    lines.append("2. Tabular data coverage")
    lines.append("-" * 80)
    lines.append(f"Tabular sheets/CSVs detected: {summary['n_tabular_sheets_or_csvs']}")
    lines.append(f"Total non-empty rows across sheets: {summary['total_non_empty_rows_across_sheets']}")
    lines.append(f"Total numeric-like columns across sheets: {summary['total_numeric_like_columns_across_sheets']}")
    lines.append(f"Unique numeric variable count: {summary['unique_numeric_variable_count']}")
    lines.append("")
    lines.append("3. Machine-learning sufficiency assessment")
    lines.append("-" * 80)
    lines.append(f"Assessment: {summary['machine_learning_sufficiency']}")
    lines.append(f"Recommended next design: {summary['recommended_next_design']}")
    lines.append("")
    lines.append("4. Reviewer-facing implications")
    lines.append("-" * 80)
    for k, v in summary["reviewer_implications"].items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("5. Next scripts recommended")
    lines.append("-" * 80)
    lines.append("02_extract_and_standardize_mof_tables.py")
    lines.append("03_build_long_format_budget_dataset.py")
    lines.append("04_define_governance_metrics_and_labels.py")
    lines.append("05_baseline_and_ai_model_experiments.py")
    lines.append("06_validation_uncertainty_and_sensitivity.py")

    (RESULTS_DIR / "audit_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_directories()

    print("=" * 80)
    print("01 DATA INVENTORY AND AUDIT")
    print("=" * 80)
    print(f"Data directory:    {DATA_DIR}")
    print(f"Scripts directory: {SCRIPTS_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print("-" * 80)

    files = collect_files()
    print(f"Supported files found: {len(files)}")

    file_inventory = build_file_inventory(files)
    file_inventory.to_csv(RESULTS_DIR / "file_inventory.csv", index=False, encoding="utf-8-sig")

    sheet_inventory, column_inventory, numeric_summary, preview_sheets = audit_tabular_files(files)

    sheet_inventory.to_csv(RESULTS_DIR / "excel_sheet_inventory.csv", index=False, encoding="utf-8-sig")
    column_inventory.to_csv(RESULTS_DIR / "excel_column_inventory.csv", index=False, encoding="utf-8-sig")
    numeric_summary.to_csv(RESULTS_DIR / "numeric_variable_summary.csv", index=False, encoding="utf-8-sig")

    pdf_inventory = audit_pdfs(files)
    pdf_inventory.to_csv(RESULTS_DIR / "pdf_inventory.csv", index=False, encoding="utf-8-sig")

    if preview_sheets:
        preview_path = RESULTS_DIR / "extracted_excel_preview.xlsx"
        with pd.ExcelWriter(preview_path, engine="openpyxl") as writer:
            for sheet_name, df in preview_sheets.items():
                safe_sheet = re.sub(r"[\[\]\*\?/\\:]", "_", sheet_name)[:31]
                df.to_excel(writer, sheet_name=safe_sheet, index=False)

    summary = assess_scientific_sufficiency(
        file_inventory=file_inventory,
        sheet_inventory=sheet_inventory,
        column_inventory=column_inventory,
        numeric_summary=numeric_summary,
    )

    with (RESULTS_DIR / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_text_report(summary)

    print("[OK] Wrote file_inventory.csv")
    print("[OK] Wrote excel_sheet_inventory.csv")
    print("[OK] Wrote excel_column_inventory.csv")
    print("[OK] Wrote numeric_variable_summary.csv")
    print("[OK] Wrote pdf_inventory.csv")
    if preview_sheets:
        print("[OK] Wrote extracted_excel_preview.xlsx")
    print("[OK] Wrote audit_summary.json")
    print("[OK] Wrote audit_report.txt")
    print("-" * 80)
    print(f"Machine-learning sufficiency: {summary['machine_learning_sufficiency']}")
    print(f"Recommended next design: {summary['recommended_next_design']}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
