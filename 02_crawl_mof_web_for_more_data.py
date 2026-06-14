r"""
02_crawl_mof_web_for_more_data.py

Purpose
-------
Crawl official Saudi Ministry of Finance (MoF) financial report pages and download
additional public budget-performance files to strengthen the empirical dataset.

This script is designed to address the reviewers' concern that the previous dataset used
only 2023 quarterly reports. It expands the available data across multiple years when
files are publicly accessible from the MoF portal.

Download target
---------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Data

Result folder
-------------
E:\\47\\471\\New Papers\\Transforming AI-Driven Solutions\\Codes\\Results\\02_crawl_mof_web_for_more_data

Main outputs
------------
1. Downloaded PDF/XLS/XLSX/CSV files saved inside:
   Data\\MoF_Web_Crawl

2. Crawl manifests saved inside:
   Results\\02_crawl_mof_web_for_more_data
   - crawl_seed_pages.csv
   - discovered_links.csv
   - downloaded_files.csv
   - failed_downloads.csv
   - crawl_summary.json
   - crawl_report.txt

How to run
----------
pip install requests beautifulsoup4 pandas lxml openpyxl
python 02_crawl_mof_web_for_more_data.py

Notes
-----
- The script crawls only official mof.gov.sa financial-report pages by default.
- It does not scrape private or restricted data.
- It respects a delay between requests.
- It avoids re-downloading existing files unless FORCE_DOWNLOAD=True.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"E:\47\471\New Papers\Transforming AI-Driven Solutions\Codes")
DATA_DIR = BASE_DIR / "Data"
DOWNLOAD_DIR = DATA_DIR / "MoF_Web_Crawl"

RESULTS_DIR = BASE_DIR / "Results" / "02_crawl_mof_web_for_more_data"

# Saudi MoF official financial-report portal.
OFFICIAL_DOMAIN = "www.mof.gov.sa"
BASE_FINANCIAL_REPORT = "https://www.mof.gov.sa/en/financialreport"

# Extend years if needed.
YEARS_TO_CRAWL = list(range(2016, 2026 + 1))

# Crawl behavior.
REQUEST_DELAY_SECONDS = 0.8
TIMEOUT_SECONDS = 60
MAX_INTERNAL_PAGES_PER_YEAR = 30
FORCE_DOWNLOAD = False

# Only download research-relevant files.
DOWNLOAD_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 ResearchBudgetCrawler/1.0"
    )
}


# =============================================================================
# Utility functions
# =============================================================================

def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url: str) -> str:
    url = url.strip()
    url = url.split("#")[0]
    return url


def is_official_mof_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower() in {"mof.gov.sa", "www.mof.gov.sa"}
    except Exception:
        return False


def is_downloadable_file(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in DOWNLOAD_EXTENSIONS)


def infer_extension(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in DOWNLOAD_EXTENSIONS:
        if path.endswith(ext):
            return ext
    return ""


def infer_year_from_text(text: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", text)
    if m:
        return int(m.group(1))
    return None


def infer_period_from_text(text: str) -> Optional[str]:
    lower = text.lower()

    patterns = [
        (r"\bq\s*1\b|\bq1\b|first\s+quarter|quarter\s*1", "Q1"),
        (r"\bq\s*2\b|\bq2\b|second\s+quarter|quarter\s*2", "Q2"),
        (r"\bq\s*3\b|\bq3\b|third\s+quarter|quarter\s*3", "Q3"),
        (r"\bq\s*4\b|\bq4\b|fourth\s+quarter|quarter\s*4", "Q4"),
        (r"mid[\s\-_]*year|half[\s\-_]*year|semi[\s\-_]*annual|h1", "Mid-Year/H1"),
        (r"end[\s\-_]*year|year[\s\-_]*end|annual|final|fy", "End-Year/Annual"),
        (r"budget\s+statement", "Budget Statement"),
    ]

    for pat, label in patterns:
        if re.search(pat, lower):
            return label
    return None


def classify_file_type(text: str, url: str) -> str:
    lower = f"{text} {url}".lower()
    if "infographic" in lower:
        return "infographic"
    if "budget performance" in lower or "performance report" in lower:
        return "budget_performance_report"
    if "budget statement" in lower:
        return "budget_statement"
    if "financial report" in lower:
        return "financial_report"
    return "other_financial_file"


def safe_filename(url: str, link_text: str = "") -> str:
    parsed = urlparse(url)
    raw_name = unquote(Path(parsed.path).name) or "downloaded_file"
    raw_name = re.sub(r"[^\w\-.]+", "_", raw_name, flags=re.UNICODE)
    raw_name = raw_name.strip("._") or "downloaded_file"

    year = infer_year_from_text(url + " " + link_text)
    period = infer_period_from_text(url + " " + link_text)
    ext = infer_extension(url)

    prefix_parts = []
    if year:
        prefix_parts.append(str(year))
    if period:
        prefix_parts.append(period.replace("/", "_").replace(" ", "_"))

    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    prefix = "__".join(prefix_parts + [h]) if prefix_parts else h

    if ext and not raw_name.lower().endswith(ext):
        raw_name = raw_name + ext

    return f"{prefix}__{raw_name}"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def http_get(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS, allow_redirects=True)
    response.raise_for_status()
    return response


def extract_links_from_html(html: str, base_url: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        text = a.get_text(" ", strip=True)
        if not href:
            continue

        full = normalize_url(urljoin(base_url, href))
        if not is_official_mof_url(full):
            continue

        links.append({
            "source_page": base_url,
            "url": full,
            "link_text": text,
        })

    # Deduplicate while preserving useful text.
    seen = set()
    unique = []
    for item in links:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)

    return unique


def build_seed_pages() -> List[str]:
    seeds = {BASE_FINANCIAL_REPORT}

    for year in YEARS_TO_CRAWL:
        seeds.add(f"{BASE_FINANCIAL_REPORT}/{year}/Pages/default.aspx")
        seeds.add(f"{BASE_FINANCIAL_REPORT}/{year}")
        seeds.add(f"{BASE_FINANCIAL_REPORT}/{year}/")

    return sorted(seeds)


# =============================================================================
# Crawling
# =============================================================================

def crawl_seed_pages() -> Tuple[pd.DataFrame, pd.DataFrame]:
    session = requests.Session()

    seed_pages = build_seed_pages()
    seed_rows = []
    discovered_rows = []

    for seed in seed_pages:
        print(f"[CRAWL] {seed}")

        try:
            response = http_get(session, seed)
            status = response.status_code
            content_type = response.headers.get("Content-Type", "")
            html = response.text

            seed_rows.append({
                "seed_page": seed,
                "status": "success",
                "http_status": status,
                "content_type": content_type,
                "error": "",
            })

            links = extract_links_from_html(html, seed)

            # Save links from the seed.
            for item in links:
                discovered_rows.append(item)

            # Crawl a limited number of internal financial-report pages from this seed.
            internal_pages = [
                item["url"] for item in links
                if "financialreport" in item["url"].lower()
                and not is_downloadable_file(item["url"])
                and item["url"].lower().endswith((".aspx", "/"))
            ]

            internal_pages = list(dict.fromkeys(internal_pages))[:MAX_INTERNAL_PAGES_PER_YEAR]

            for page in internal_pages:
                if page == seed:
                    continue

                print(f"  [INTERNAL] {page}")
                try:
                    time.sleep(REQUEST_DELAY_SECONDS)
                    r2 = http_get(session, page)
                    links2 = extract_links_from_html(r2.text, page)
                    for item2 in links2:
                        discovered_rows.append(item2)
                except Exception as e:
                    discovered_rows.append({
                        "source_page": page,
                        "url": "",
                        "link_text": f"INTERNAL_PAGE_ERROR: {e}",
                    })

        except Exception as e:
            seed_rows.append({
                "seed_page": seed,
                "status": "failed",
                "http_status": "",
                "content_type": "",
                "error": str(e),
            })

        time.sleep(REQUEST_DELAY_SECONDS)

    seed_df = pd.DataFrame(seed_rows)
    discovered_df = pd.DataFrame(discovered_rows)

    if not discovered_df.empty:
        discovered_df["url"] = discovered_df["url"].fillna("").astype(str)
        discovered_df = discovered_df[discovered_df["url"].str.len() > 0].copy()
        discovered_df = discovered_df.drop_duplicates(subset=["url"]).reset_index(drop=True)

        discovered_df["extension"] = discovered_df["url"].apply(infer_extension)
        discovered_df["downloadable"] = discovered_df["url"].apply(is_downloadable_file)
        discovered_df["year_guess"] = discovered_df.apply(
            lambda r: infer_year_from_text(str(r["url"]) + " " + str(r["link_text"])),
            axis=1,
        )
        discovered_df["period_guess"] = discovered_df.apply(
            lambda r: infer_period_from_text(str(r["url"]) + " " + str(r["link_text"])),
            axis=1,
        )
        discovered_df["file_type_guess"] = discovered_df.apply(
            lambda r: classify_file_type(str(r["link_text"]), str(r["url"])),
            axis=1,
        )

    return seed_df, discovered_df


# =============================================================================
# Downloading
# =============================================================================

def download_discovered_files(discovered_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if discovered_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    files_df = discovered_df[discovered_df["downloadable"] == True].copy()
    files_df = files_df[files_df["extension"].isin(DOWNLOAD_EXTENSIONS)].copy()

    session = requests.Session()

    downloaded_rows = []
    failed_rows = []

    for _, row in files_df.iterrows():
        url = str(row["url"])
        link_text = str(row.get("link_text", ""))
        filename = safe_filename(url, link_text)
        out_path = DOWNLOAD_DIR / filename

        if out_path.exists() and out_path.stat().st_size > 0 and not FORCE_DOWNLOAD:
            print(f"[SKIP] {out_path.name}")
            downloaded_rows.append({
                "url": url,
                "source_page": row.get("source_page", ""),
                "link_text": link_text,
                "local_path": str(out_path),
                "file_name": out_path.name,
                "extension": out_path.suffix.lower(),
                "year_guess": row.get("year_guess", None),
                "period_guess": row.get("period_guess", None),
                "file_type_guess": row.get("file_type_guess", ""),
                "size_bytes": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
                "download_status": "already_exists",
            })
            continue

        print(f"[DOWNLOAD] {url}")

        try:
            response = http_get(session, url)
            content = response.content

            if not content or len(content) < 100:
                raise ValueError("Downloaded content is unexpectedly small.")

            out_path.write_bytes(content)

            downloaded_rows.append({
                "url": url,
                "source_page": row.get("source_page", ""),
                "link_text": link_text,
                "local_path": str(out_path),
                "file_name": out_path.name,
                "extension": out_path.suffix.lower(),
                "year_guess": row.get("year_guess", None),
                "period_guess": row.get("period_guess", None),
                "file_type_guess": row.get("file_type_guess", ""),
                "size_bytes": out_path.stat().st_size,
                "sha256": sha256_file(out_path),
                "download_status": "downloaded",
            })

        except Exception as e:
            failed_rows.append({
                "url": url,
                "source_page": row.get("source_page", ""),
                "link_text": link_text,
                "year_guess": row.get("year_guess", None),
                "period_guess": row.get("period_guess", None),
                "file_type_guess": row.get("file_type_guess", ""),
                "error": str(e),
            })

        time.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(downloaded_rows), pd.DataFrame(failed_rows)


# =============================================================================
# Reporting
# =============================================================================

def write_summary(
    seed_df: pd.DataFrame,
    discovered_df: pd.DataFrame,
    downloaded_df: pd.DataFrame,
    failed_df: pd.DataFrame,
) -> None:
    years_downloaded = []
    periods_downloaded = []

    if not downloaded_df.empty:
        years_downloaded = sorted([
            int(x) for x in downloaded_df["year_guess"].dropna().unique().tolist()
            if str(x).strip() not in {"", "nan", "None"}
        ])

        periods_downloaded = sorted([
            str(x) for x in downloaded_df["period_guess"].dropna().unique().tolist()
            if str(x).strip() not in {"", "nan", "None"}
        ])

    summary = {
        "crawl_time": datetime.now().isoformat(timespec="seconds"),
        "official_domain": OFFICIAL_DOMAIN,
        "base_financial_report_url": BASE_FINANCIAL_REPORT,
        "data_dir": str(DATA_DIR),
        "download_dir": str(DOWNLOAD_DIR),
        "results_dir": str(RESULTS_DIR),
        "years_requested": YEARS_TO_CRAWL,
        "seed_pages_attempted": int(len(seed_df)),
        "seed_pages_successful": int((seed_df["status"] == "success").sum()) if not seed_df.empty else 0,
        "links_discovered": int(len(discovered_df)),
        "downloadable_links_discovered": int(discovered_df["downloadable"].sum()) if not discovered_df.empty else 0,
        "files_downloaded_or_existing": int(len(downloaded_df)),
        "failed_downloads": int(len(failed_df)),
        "years_downloaded": years_downloaded,
        "periods_downloaded": periods_downloaded,
        "downloaded_extensions": (
            downloaded_df["extension"].value_counts().to_dict()
            if not downloaded_df.empty else {}
        ),
        "downloaded_file_type_guesses": (
            downloaded_df["file_type_guess"].value_counts().to_dict()
            if not downloaded_df.empty else {}
        ),
        "next_step": (
            "Run 01_data_inventory_and_audit.py again after this crawl. "
            "If multiple years are now detected, proceed to table extraction and long-format dataset construction."
        ),
    }

    with (RESULTS_DIR / "crawl_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("MOF WEB CRAWL REPORT")
    lines.append("=" * 80)
    lines.append(f"Crawl time: {summary['crawl_time']}")
    lines.append(f"Official source domain: {summary['official_domain']}")
    lines.append(f"Base URL: {summary['base_financial_report_url']}")
    lines.append(f"Download directory: {summary['download_dir']}")
    lines.append("")
    lines.append("1. Crawl coverage")
    lines.append("-" * 80)
    lines.append(f"Years requested: {summary['years_requested']}")
    lines.append(f"Seed pages attempted: {summary['seed_pages_attempted']}")
    lines.append(f"Seed pages successful: {summary['seed_pages_successful']}")
    lines.append(f"Links discovered: {summary['links_discovered']}")
    lines.append(f"Downloadable links discovered: {summary['downloadable_links_discovered']}")
    lines.append("")
    lines.append("2. Downloads")
    lines.append("-" * 80)
    lines.append(f"Files downloaded or already existing: {summary['files_downloaded_or_existing']}")
    lines.append(f"Failed downloads: {summary['failed_downloads']}")
    lines.append(f"Years downloaded: {summary['years_downloaded']}")
    lines.append(f"Periods downloaded: {summary['periods_downloaded']}")
    lines.append(f"Downloaded extensions: {summary['downloaded_extensions']}")
    lines.append(f"Downloaded file-type guesses: {summary['downloaded_file_type_guesses']}")
    lines.append("")
    lines.append("3. Next step")
    lines.append("-" * 80)
    lines.append(summary["next_step"])

    (RESULTS_DIR / "crawl_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_directories()

    print("=" * 80)
    print("02 CRAWL MOF WEB FOR MORE DATA")
    print("=" * 80)
    print(f"Data directory:     {DATA_DIR}")
    print(f"Download directory: {DOWNLOAD_DIR}")
    print(f"Results directory:  {RESULTS_DIR}")
    print("-" * 80)

    seed_df, discovered_df = crawl_seed_pages()

    seed_path = RESULTS_DIR / "crawl_seed_pages.csv"
    discovered_path = RESULTS_DIR / "discovered_links.csv"

    seed_df.to_csv(seed_path, index=False, encoding="utf-8-sig")
    discovered_df.to_csv(discovered_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote {seed_path}")
    print(f"[OK] Wrote {discovered_path}")

    downloaded_df, failed_df = download_discovered_files(discovered_df)

    downloaded_path = RESULTS_DIR / "downloaded_files.csv"
    failed_path = RESULTS_DIR / "failed_downloads.csv"

    downloaded_df.to_csv(downloaded_path, index=False, encoding="utf-8-sig")
    failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote {downloaded_path}")
    print(f"[OK] Wrote {failed_path}")

    write_summary(seed_df, discovered_df, downloaded_df, failed_df)

    print("[OK] Wrote crawl_summary.json")
    print("[OK] Wrote crawl_report.txt")
    print("-" * 80)
    print(f"Downloaded / existing files: {len(downloaded_df)}")
    print(f"Failed downloads: {len(failed_df)}")
    print("Next: run 01_data_inventory_and_audit.py again.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)
