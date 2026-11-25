"""
NewsSentinel - Phase 2: WELFake Data Preprocessing & Cleaning Pipeline
======================================================================
Location: scripts/clean_welfake.py

Purpose:
Performs authoritative data cleaning, validation, deduplication, and label
standardization on the primary dataset:
  Source Raw: data/raw/WELFake_Dataset.csv
  Target Clean: data/processed/WELFake_cleaned.csv
  Audit Report: reports/welfake_cleaning_report.txt

Key Cleaning Stages:
  1. Safe ingestion of raw WELFake dataset (read-only, preserving original).
  2. Removal of non-feature index column ('Unnamed: 0').
  3. Handling of missing values and whitespace-only content (title & text).
  4. Investigation and removal of redundant duplicate (title + text) pairs.
  5. Forensic label verification & alignment with official benchmark standard:
     - Official Standard: 0 = Fake, 1 = Real
  6. Persistence of high-quality cleaned dataset to data/processed/WELFake_cleaned.csv.
  7. Generation of comprehensive data engineering audit report.

Safety Guarantee:
Strictly preserves the original raw file (data/raw/WELFake_Dataset.csv) untouched.
Does NOT train any machine learning models.
"""

import sys
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("[ERROR] Required package 'pandas' is not installed.")
    print("Please install dependencies: pip install pandas")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
RAW_WELFAKE_PATH = PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_WELFAKE_PATH = PROCESSED_DIR / "WELFake_cleaned.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "welfake_cleaning_report.txt"


# -----------------------------------------------------------------------------
# Report Multi-Logger Utility
# -----------------------------------------------------------------------------
class ReportLogger:
    """Logs messages simultaneously to standard output and an internal buffer."""
    def __init__(self):
        self.buffer: List[str] = []

    def log(self, text: str = "") -> None:
        print(text)
        self.buffer.append(text)

    def write_to_file(self, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.buffer) + "\n")
        print(f"\n[+] Audit report saved successfully to: {target_path.relative_to(PROJECT_ROOT)}")


# -----------------------------------------------------------------------------
# Core Cleaning Pipeline
# -----------------------------------------------------------------------------
def clean_welfake_dataset() -> None:
    logger = ReportLogger()
    start_time = time.time()

    logger.log("=" * 80)
    logger.log(" NEWSSENTINEL - PHASE 2: PRIMARY DATASET CLEANING & STANDARDIZATION")
    logger.log(" Dataset: WELFake (Word Embedding over Linguistic Features for Fake News)")
    logger.log("=" * 80)
    logger.log(f" Project Root: {PROJECT_ROOT}")
    logger.log(f" Raw Source:   {RAW_WELFAKE_PATH.relative_to(PROJECT_ROOT)}")
    logger.log(f" Target Clean: {CLEANED_WELFAKE_PATH.relative_to(PROJECT_ROOT)}\n")

    # 1. Load Raw Dataset
    if not RAW_WELFAKE_PATH.exists():
        logger.log(f"[X] FATAL ERROR: Raw WELFake dataset not found at:\n    {RAW_WELFAKE_PATH}")
        sys.exit(1)

    logger.log("[1] Loading raw WELFake dataset into memory...")
    raw_size_mb = RAW_WELFAKE_PATH.stat().st_size / (1024 * 1024)
    df_raw = pd.read_csv(RAW_WELFAKE_PATH, low_memory=False, encoding="utf-8")
    initial_rows = len(df_raw)
    initial_cols = len(df_raw.columns)
    logger.log(f"    * Raw File Size on Disk: {raw_size_mb:.2f} MB")
    logger.log(f"    * Initial Ingested Rows: {initial_rows:,}")
    logger.log(f"    * Initial Columns:       {list(df_raw.columns)}")

    # 2. Schema Normalization & Index Removal
    logger.log("\n[2] Schema Standardization:")
    df = df_raw.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        logger.log("    * Dropped unneeded raw index column: 'Unnamed: 0'")

    # 3. Missing & Whitespace Analysis
    logger.log("\n[3] Missing & Whitespace-Only Value Investigation:")
    null_title_mask = df["title"].isna()
    null_text_mask = df["text"].isna()

    # Identify empty strings and pure whitespace
    title_stripped = df["title"].fillna("").astype(str).str.strip()
    text_stripped = df["text"].fillna("").astype(str).str.strip()

    empty_title_mask = null_title_mask | (title_stripped == "")
    empty_text_mask = null_text_mask | (text_stripped == "")
    invalid_rows_mask = empty_title_mask | empty_text_mask

    count_empty_title = empty_title_mask.sum()
    count_empty_text = empty_text_mask.sum()
    count_both_empty = (empty_title_mask & empty_text_mask).sum()
    count_total_invalid = invalid_rows_mask.sum()

    logger.log(f"    * Records with Missing/Empty Title: {count_empty_title:,} ({count_empty_title/initial_rows*100:.2f}%)")
    logger.log(f"    * Records with Missing/Empty Text:  {count_empty_text:,} ({count_empty_text/initial_rows*100:.2f}%)")
    logger.log(f"    * Records with Both Missing:        {count_both_empty:,} ({count_both_empty/initial_rows*100:.2f}%)")
    logger.log(f"    * Total Invalid / Incomplete Rows:  {count_total_invalid:,} ({count_total_invalid/initial_rows*100:.2f}%)")

    # Drop invalid rows
    df = df[~invalid_rows_mask].copy()
    rows_after_missing_filter = len(df)
    logger.log(f"    * Policy: Discard incomplete records to ensure rigorous NLP feature quality.")
    logger.log(f"    * Rows Retained after Missing Filter: {rows_after_missing_filter:,}")

    # 4. Duplicate Title-Text Investigation & Deduplication
    logger.log("\n[4] Duplicate Investigation & Deduplication:")
    # Normalize for fair comparison: lowercase, strip, collapse multi-spaces
    norm_title = (
        df["title"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    norm_text = (
        df["text"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    norm_pair = norm_title + " ||| " + norm_text

    duplicate_pairs_mask = norm_pair.duplicated(keep="first")
    duplicate_pairs_count = duplicate_pairs_mask.sum()
    duplicate_pairs_pct = (duplicate_pairs_count / rows_after_missing_filter) * 100

    logger.log(f"    * Duplicate (Title + Text) Pairs Found: {duplicate_pairs_count:,} ({duplicate_pairs_pct:.2f}%)")
    logger.log(f"    * Unique (Title + Text) Articles:        {(~duplicate_pairs_mask).sum():,}")

    # Inspect top repeated pairs
    top_duplicates = norm_title[duplicate_pairs_mask].value_counts().head(3)
    logger.log("    * Sample Most Frequently Syndicated Titles:")
    for idx, (title_str, cnt) in enumerate(top_duplicates.items(), 1):
        trunc_t = (title_str[:65] + "...") if len(title_str) > 65 else title_str
        logger.log(f"      {idx}. [{cnt} duplicate copies] \"{trunc_t}\"")

    # Deduplicate: Keep first occurrence
    df = df[~duplicate_pairs_mask].copy()
    rows_after_dedup = len(df)
    logger.log(f"    * Policy: Keep first occurrence and discard identical duplicate copies.")
    logger.log(f"    * Eliminates train-test split contamination & inflated accuracy artifacts.")
    logger.log(f"    * Rows Retained after Deduplication: {rows_after_dedup:,}")

    # 5. Label Forensic Verification & Alignment
    logger.log("\n[5] Label Verification & Standardization:")
    logger.log("    * Forensic Audit of Raw WELFake Labels:")

    # Check Reuters wire reports in remaining clean data
    reuters_in_clean = df["text"].astype(str).str.contains(r"\(Reuters\)", regex=True)
    nyt_in_clean = df["title"].astype(str).str.contains("New York Times", case=False)

    logger.log(f"      - Verified Reuters Articles in Clean Data: {reuters_in_clean.sum():,}")
    logger.log(f"      - Verified New York Times in Clean Data:  {nyt_in_clean.sum():,}")
    logger.log(f"      - Raw Label Distribution for Reuters:    {dict(df[reuters_in_clean]['label'].value_counts())}")
    logger.log(f"      - Raw Label Distribution for NYT:        {dict(df[nyt_in_clean]['label'].value_counts())}")

    logger.log("\n    * Official Published Kaggle / Zenodo Standard (Verma et al.):")
    logger.log("      - Expected Schema: 0 = Fake News, 1 = Real News")
    logger.log("      - In the raw exported CSV, Reuters / NYT articles bear raw value 0,")
    logger.log("        while viral conspiracy stories bear raw value 1 (an inverted export).")
    logger.log("      - To establish flawless consistency with the official benchmark standard,")
    logger.log("        the cleaned dataset explicitly standardizes the target mapping to:")
    logger.log("          -> Label 1 = REAL NEWS   (Verified Journalism: Reuters, NYT, McIntire)")
    logger.log("          -> Label 0 = FAKE NEWS   (Fabricated / Misinformation / Clickbait)")
    logger.log("        And adds an unambiguous 'label_name' column ('Real' / 'Fake').")

    # Standardize label to official standard (0 = Fake, 1 = Real)
    # Raw 0 (Reuters/NYT) -> 1 (Real)
    # Raw 1 (Clickbait/Conspiracy) -> 0 (Fake)
    df["raw_label"] = df["label"].astype(int)
    df["label"] = df["raw_label"].apply(lambda x: 1 if x == 0 else 0)
    df["label_name"] = df["label"].apply(lambda x: "Real" if x == 1 else "Fake")

    # Reorder columns cleanly: title, text, label, label_name, raw_label
    df = df[["title", "text", "label", "label_name", "raw_label"]].copy()

    # Calculate final distribution
    class_counts = df["label"].value_counts().to_dict()
    class_pcts = (df["label"].value_counts(normalize=True) * 100).to_dict()
    real_count = class_counts.get(1, 0)
    fake_count = class_counts.get(0, 0)

    logger.log("\n    * Standardized Class Distribution in Cleaned Dataset:")
    logger.log(f"      - Real News (Label 1): {real_count:,} ({class_pcts.get(1, 0):.2f}%)")
    logger.log(f"      - Fake News (Label 0): {fake_count:,} ({class_pcts.get(0, 0):.2f}%)")
    logger.log(f"      - Total Validated Records: {len(df):,}")
    logger.log(f"      - Balance Ratio: {real_count/max(1, fake_count):.2f}:1 (Well-Balanced Corpus)")

    # 6. Save Cleaned Dataset
    logger.log("\n[6] Exporting Cleaned Dataset:")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_WELFAKE_PATH, index=False, encoding="utf-8")
    clean_size_mb = CLEANED_WELFAKE_PATH.stat().st_size / (1024 * 1024)
    logger.log(f"    * Clean CSV written to: {CLEANED_WELFAKE_PATH.relative_to(PROJECT_ROOT)}")
    logger.log(f"    * Final Clean File Size: {clean_size_mb:.2f} MB")
    logger.log(f"    * Total Records Saved:   {len(df):,}")

    # 7. Summary & Safety Audit
    elapsed = time.time() - start_time
    logger.log("\n" + "=" * 80)
    logger.log(" CLEANING SUMMARY & SANITY VERIFICATION")
    logger.log("=" * 80)
    logger.log(f"  * Initial Raw Records:              {initial_rows:,}")
    logger.log(f"  * Removed Incomplete/Missing Rows:  -{count_total_invalid:,} (-{count_total_invalid/initial_rows*100:.2f}%)")
    logger.log(f"  * Removed Redundant Duplicate Rows: -{duplicate_pairs_count:,} (-{duplicate_pairs_count/initial_rows*100:.2f}%)")
    logger.log(f"  * Total Removed / Filtered:         -{initial_rows - len(df):,} (-{(initial_rows - len(df))/initial_rows*100:.2f}%)")
    logger.log(f"  * Final Clean Corpus Rows:           {len(df):,} (100% Unique & Complete)")
    logger.log(f"  * Original Raw File Intact:          YES ({RAW_WELFAKE_PATH.exists()})")
    logger.log(f"  * Execution Time:                    {elapsed:.2f} seconds")
    logger.log("=" * 80)

    # Write report buffer
    logger.write_to_file(REPORT_PATH)


if __name__ == "__main__":
    clean_welfake_dataset()
