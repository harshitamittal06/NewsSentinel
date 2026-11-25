"""
NewsSentinel - Phase 2: Primary Dataset Inspection Script
=========================================================
Location: scripts/inspect_datasets.py

Purpose:
Performs a comprehensive, non-destructive exploratory inspection of the primary
project datasets:
  1. Primary Raw:     data/raw/WELFake_Dataset.csv
  2. Primary Cleaned: data/processed/WELFake_cleaned.csv (if generated)

Inspection Scope:
- Shape (rows x columns) and memory consumption
- Column names, data types, and non-null counts
- Missing value audit per column (counts & percentages)
- Exact duplicate rows and duplicate content analysis (title & text columns)
- Label / class distribution and balance
- Readable 5-row sample preview with text truncation

Safety Guarantee:
Strictly READ-ONLY. No datasets are modified, cleaned, overwritten, or deleted.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Ensure project root is in Python path regardless of execution directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("\n[ERROR] Required package 'pandas' is not installed.")
    print("Please install dependencies by running: pip install pandas\n")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Configuration & Primary Dataset Registry
# -----------------------------------------------------------------------------
DATASET_REGISTRY = [
    {
        "id": "welfake_raw",
        "name": "WELFake Dataset (Raw Master Corpus)",
        "path": PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv",
        "description": "72,134 news records collated across Kaggle, McIntire, Reuters, and BuzzFeed.",
        "label_mapping_note": "Raw exported labels: 0=Real, 1=Fake",
    },
    {
        "id": "welfake_cleaned",
        "name": "WELFake Dataset (Cleaned & Standardized)",
        "path": PROJECT_ROOT / "data" / "processed" / "WELFake_cleaned.csv",
        "description": "62,586 deduplicated, complete articles with standardized benchmark labels.",
        "label_mapping_note": "Standardized: 0=Fake, 1=Real (with explicit label_name column)",
    },
]

# Common character encodings to test sequentially
CANDIDATE_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]


# -----------------------------------------------------------------------------
# Formatting Helpers
# -----------------------------------------------------------------------------
def format_bytes(num_bytes: int) -> str:
    """Format byte integers into human-readable strings (KB, MB, GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print an eye-catching terminal banner for section separation."""
    border = "=" * 80
    print(f"\n{border}")
    print(f" {title.upper()}")
    if subtitle:
        print(f" {subtitle}")
    print(f"{border}\n")


def print_subdivider(title: str) -> None:
    """Print a clean subsection divider."""
    print(f"\n--- [ {title} ] " + "-" * (70 - len(title)))


# -----------------------------------------------------------------------------
# Safe Loading Engine
# -----------------------------------------------------------------------------
def load_dataset_safely(file_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Safely load a CSV dataset, automatically attempting common encodings.
    """
    if not file_path.exists():
        err_msg = f"File not found at: {file_path}"
        return None, None, err_msg

    last_error = ""
    for enc in CANDIDATE_ENCODINGS:
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            return df, enc, None
        except UnicodeDecodeError as ude:
            last_error = f"Encoding '{enc}' failed: {ude}"
            continue
        except Exception as ex:
            last_error = f"Error reading with '{enc}': {ex}"
            continue

    return None, None, f"All candidate encodings failed. Last error: {last_error}"


# -----------------------------------------------------------------------------
# Inspection Functions
# -----------------------------------------------------------------------------
def inspect_dataset(info: Dict[str, Any], index: int, total: int) -> Optional[Dict[str, Any]]:
    """
    Inspect a single dataset file and print a structured diagnostic report.
    """
    name = info["name"]
    file_path = info["path"]
    desc = info["description"]
    label_note = info.get("label_mapping_note", "")

    print_header(
        f"Dataset {index} of {total}: {name}",
        f"File: {file_path.relative_to(PROJECT_ROOT) if file_path.exists() else file_path} | {desc}"
    )

    if not file_path.exists():
        print(f"[!] NOTICE: File not yet generated or missing at:\n    {file_path}")
        print("    Run 'python scripts/clean_welfake.py' to generate processed datasets.\n")
        return None

    file_size_bytes = file_path.stat().st_size
    print(f"[*] Physical File Size on Disk: {format_bytes(file_size_bytes)}")
    print(f"[*] Loading dataset into memory...")

    df, encoding_used, error_msg = load_dataset_safely(file_path)

    if df is None:
        print(f"[X] FAILED to load dataset: {error_msg}")
        return None

    print(f"[+] Loaded successfully using encoding: '{encoding_used}'")

    # 1. Dimensions & Memory
    rows, cols = df.shape
    mem_bytes = df.memory_usage(deep=True).sum()
    print_subdivider("1. Dataset Dimensions & Memory Footprint")
    print(f"  * Total Rows:            {rows:,}")
    print(f"  * Total Columns:         {cols}")
    print(f"  * Total Cell Count:      {rows * cols:,}")
    print(f"  * In-Memory Size (Deep): {format_bytes(mem_bytes)}")

    # 2. Columns & Schema
    print_subdivider("2. Columns & Data Types Schema")
    schema_rows = []
    for idx, col in enumerate(df.columns):
        col_type = str(df[col].dtype)
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        null_pct = (null_count / rows) * 100 if rows > 0 else 0.0
        schema_rows.append({
            "#": idx + 1,
            "Column Name": col,
            "Dtype": col_type,
            "Non-Null": f"{non_null:,}",
            "Null Count": f"{null_count:,}",
            "Null %": f"{null_pct:.2f}%"
        })
    schema_df = pd.DataFrame(schema_rows)
    print(schema_df.to_string(index=False))

    # 3. Missing Value Audit
    print_subdivider("3. Missing Value Analysis")
    missing_series = df.isna().sum()
    missing_cols = missing_series[missing_series > 0]
    total_missing_cells = missing_series.sum()
    total_cells = rows * cols
    missing_ratio = (total_missing_cells / total_cells) * 100 if total_cells > 0 else 0.0

    print(f"  * Total Missing Data Cells: {total_missing_cells:,} ({missing_ratio:.2f}% of all cells)")
    if len(missing_cols) == 0:
        print("  * Status: [EXCELLENT] Zero missing values detected across all columns.")
    else:
        print("  * Columns with Missing Values:")
        for col_name, count in missing_cols.items():
            pct = (count / rows) * 100
            print(f"    - '{col_name}': {count:,} missing ({pct:.2f}%)")

    # 4. Duplicate Audit
    print_subdivider("4. Duplication & Redundancy Audit")
    exact_duplicates = df.duplicated().sum()
    exact_dup_pct = (exact_duplicates / rows) * 100 if rows > 0 else 0.0
    print(f"  * Exact Duplicate Rows (all columns): {exact_duplicates:,} ({exact_dup_pct:.2f}%)")

    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else None

    if title_col:
        title_dups = df.duplicated(subset=[title_col]).sum()
        title_dup_pct = (title_dups / rows) * 100 if rows > 0 else 0.0
        unique_titles = df[title_col].nunique(dropna=True)
        print(f"  * Duplicate '{title_col}' values:  {title_dups:,} ({title_dup_pct:.2f}%) [Unique: {unique_titles:,}]")

    if text_col:
        text_dups = df.duplicated(subset=[text_col]).sum()
        text_dup_pct = (text_dups / rows) * 100 if rows > 0 else 0.0
        unique_texts = df[text_col].nunique(dropna=True)
        print(f"  * Duplicate '{text_col}' values:   {text_dups:,} ({text_dup_pct:.2f}%) [Unique: {unique_texts:,}]")

    if title_col and text_col:
        combo_dups = df.duplicated(subset=[title_col, text_col]).sum()
        combo_dup_pct = (combo_dups / rows) * 100 if rows > 0 else 0.0
        print(f"  * Duplicate ({title_col} + {text_col}) pairs: {combo_dups:,} ({combo_dup_pct:.2f}%)")

    # 5. Label Distribution
    print_subdivider("5. Label / Class Distribution")
    if "label" in df.columns:
        vc = df["label"].value_counts(dropna=False)
        pcts = df["label"].value_counts(normalize=True, dropna=False) * 100

        dist_rows = []
        for class_val, count in vc.items():
            class_str = str(class_val)
            dist_rows.append({
                "Class / Label": class_str,
                "Count": f"{count:,}",
                "Percentage": f"{pcts[class_val]:.2f}%"
            })
        dist_df = pd.DataFrame(dist_rows)
        print(dist_df.to_string(index=False))
        if label_note:
            print(f"  * Note: {label_note}")

    if "label_name" in df.columns:
        print("\n  * Human-Readable Class Breakdown ('label_name'):")
        ln_vc = df["label_name"].value_counts(dropna=False)
        ln_pcts = df["label_name"].value_counts(normalize=True, dropna=False) * 100
        for name_val, cnt in ln_vc.items():
            print(f"    - {name_val}: {cnt:,} ({ln_pcts[name_val]:.2f}%)")

    # 6. Sample Preview
    print_subdivider("6. Readable Data Sample Preview (First 5 Rows)")
    preview_df = df.head(5).copy()
    for col in preview_df.columns:
        if preview_df[col].dtype == object or str(preview_df[col].dtype).startswith("str"):
            preview_df[col] = preview_df[col].fillna("<NULL>").astype(str).apply(
                lambda val: (val[:68] + "...") if len(val) > 70 else val
            )

    orig_colwidth = pd.get_option("display.max_colwidth")
    orig_width = pd.get_option("display.width")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(preview_df.to_string())
    pd.set_option("display.max_colwidth", orig_colwidth)
    pd.set_option("display.width", orig_width)

    return {
        "Dataset": name,
        "Rows": rows,
        "Columns": cols,
        "Disk Size": format_bytes(file_size_bytes),
        "Memory (RAM)": format_bytes(mem_bytes),
        "Missing Cells": total_missing_cells,
        "Exact Dups": exact_duplicates,
        "Encoding": encoding_used
    }


def main() -> None:
    print("\n" + "#" * 80)
    print("  NEWSSENTINEL - DATA SCIENCE INTELLIGENCE PLATFORM")
    print("  Phase 2: Primary Dataset Inspection (WELFake)")
    print("#" * 80)
    print(f"  Project Root Directory: {PROJECT_ROOT}")

    total_datasets = len(DATASET_REGISTRY)
    for idx, info in enumerate(DATASET_REGISTRY, 1):
        inspect_dataset(info, idx, total_datasets)

    print("\n[+] Inspection completed successfully.\n")


if __name__ == "__main__":
    main()
