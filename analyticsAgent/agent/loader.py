"""
loader.py — Data Ingestion Module
Handles CSV and XLSX loading with smart type detection.
"""

import pandas as pd
import numpy as np
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


SUPPORTED_FORMATS = [".csv", ".xlsx", ".xls"]


def loadFile(filepath: str) -> pd.DataFrame:
    """Load a CSV or XLSX file into a DataFrame."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}'. Use: {SUPPORTED_FORMATS}")

    print(f"\n📂 Loading file: {path.name}")

    if ext == ".csv":
        df = _load_csv(filepath)
    else:
        df = _load_xlsx(filepath)

    print(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def _load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV with automatic encoding and delimiter detection."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    delimiters = [",", ";", "\t", "|"]

    for encoding in encodings:
        for delimiter in delimiters:
            try:
                df = pd.read_csv(filepath, sep=delimiter, encoding=encoding,
                                  low_memory=False)
                # Valid if more than 1 column or file has content
                if df.shape[1] > 1 or df.shape[0] > 0:
                    print(f"   ↳ Encoding: {encoding}, Delimiter: '{delimiter}'")
                    return df
            except Exception:
                continue

    raise ValueError("Could not parse CSV file. Check encoding or delimiter.")


def _load_xlsx(filepath: str) -> pd.DataFrame:
    """Load XLSX, with sheet selection if multiple sheets exist."""
    xl = pd.ExcelFile(filepath, engine="openpyxl")
    sheets = xl.sheet_names

    if len(sheets) == 1:
        print(f"   ↳ Sheet: '{sheets[0]}'")
        return pd.read_excel(filepath, sheet_name=sheets[0], engine="openpyxl")

    print(f"\n📋 Multiple sheets found: {sheets}")
    for i, sheet in enumerate(sheets):
        print(f"   [{i}] {sheet}")

    choice = input("   Select sheet number (default 0): ").strip()
    idx = int(choice) if choice.isdigit() and int(choice) < len(sheets) else 0
    print(f"   ↳ Sheet: '{sheets[idx]}'")
    return pd.read_excel(filepath, sheet_name=sheets[idx], engine="openpyxl")


def detectColumnTypes(df: pd.DataFrame) -> dict:
    """
    Auto-detect semantic column types:
      - numeric: int/float columns
      - categorical: low-cardinality object/bool columns
      - datetime: date/time columns
      - text: high-cardinality string columns
      - id: likely identifier columns
    """
    types = {}

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        n = len(series.dropna())

        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "datetime"

        elif pd.api.types.is_bool_dtype(series):
            types[col] = "categorical"

        elif pd.api.types.is_numeric_dtype(series):
            # Detect sequential row-index columns (0,1,2,3...) — useless for ML
            is_sequential = (
                pd.api.types.is_integer_dtype(series)
                and nunique == n                           # every value is unique
                and series.dropna().min() == 0            # starts at 0
                and series.dropna().max() == n - 1        # ends at n-1
            )
            # Also catch columns literally named 'index', 'unnamed: 0', or bare 'id'
            is_index_name = col.lower() in ("index", "unnamed: 0", "unnamed:0", "id")

            if is_sequential or (is_index_name and nunique / max(n, 1) > 0.9):
                types[col] = "index"   # row counter — always drop, never model
            elif nunique <= 20 and nunique / max(n, 1) < 0.05:
                types[col] = "categorical"
            else:
                types[col] = "numeric"

        elif pd.api.types.is_object_dtype(series) or series.dtype.kind == 'O' or str(series.dtype) in ('str', 'string'):
            # Try parsing as datetime
            sample = series.dropna().head(100)
            try:
                pd.to_datetime(sample, infer_datetime_format=True)
                types[col] = "datetime"
            except Exception:
                # High cardinality object = text, low = categorical
                if nunique <= 50 or (n > 0 and nunique / n < 0.5):
                    types[col] = "categorical"
                else:
                    types[col] = "text"

            # Detect ID columns by name heuristic
            if any(kw in col.lower() for kw in ["_id", "id_", "uuid", "key", "code"]):
                if nunique / max(n, 1) > 0.9:
                    types[col] = "id"
        else:
            types[col] = "unknown"

    return types


def getDatasetProfile(df: pd.DataFrame, filepath: str) -> dict:
    """Generate a lightweight dataset profile for memory logging."""
    colTypes = detectColumnTypes(df)

    # Compute a file hash for change detection
    with open(filepath, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    profile = {
        "file": os.path.basename(filepath),
        "file_hash": file_hash,
        "loaded_at": datetime.now().isoformat(),
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "column_types": colTypes,
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    return profile


def showInfo(df: pd.DataFrame, colTypes: dict):
    """
    Print a detailed df.info()-style table showing every column,
    its dtype, non-null count, % missing, nunique, and detected semantic type.
    This is always the FIRST thing shown after loading — before any operations.
    """
    n = len(df)
    print("\n" + "═" * 80)
    print("📋  DATASET INFO")
    print("═" * 80)
    print(f"  Rows    : {n:,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"  Memory  : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(f"  Duplicates: {df.duplicated().sum():,}")
    print()

    # Header
    print(f"  {'#':<4} {'Column':<30} {'Dtype':<12} {'Non-Null':>9} {'Missing%':>9} "
          f"{'Unique':>8}  {'Semantic Type':<14}")
    print("  " + "─" * 76)

    icons = {
        "numeric": "🔢", "categorical": "🏷️ ", "datetime": "📅",
        "text": "📝", "id": "🔑", "index": "🔢❌", "unknown": "❓",
    }

    for i, col in enumerate(df.columns):
        non_null  = df[col].notna().sum()
        missing   = n - non_null
        miss_pct  = missing / n * 100 if n else 0
        nunique   = df[col].nunique(dropna=True)
        dtype     = str(df[col].dtype)
        sem_type  = colTypes.get(col, "unknown")
        icon      = icons.get(sem_type, "•")
        miss_str  = f"{miss_pct:.1f}%" if missing > 0 else "—"

        print(f"  {i:<4} {col:<30} {dtype:<12} {non_null:>9,} {miss_str:>9} "
              f"{nunique:>8,}  {icon} {sem_type}")

    print("═" * 80)


def suggestTarget(df: pd.DataFrame, colTypes: dict) -> Optional[str]:
    """
    Heuristically suggest the most likely target column based on:
    - Column name keywords (amount, price, total, sales, revenue, profit,
      churn, target, label, class, score, quantity, units, sold, output)
    - It being numeric or low-cardinality categorical
    - NOT being an ID column
    Returns the suggested column name, or None if no clear candidate.
    """
    keywords = [
        "target", "label", "class", "output",           # generic ML
        "sales", "revenue", "profit", "amount",          # money
        "price", "cost", "total", "income",              # more money
        "churn", "converted", "fraud", "default",        # classification flags
        "quantity", "units", "sold", "demand",           # volume
        "score", "rating", "result", "outcome",          # outcomes
    ]
    candidates = []
    for col in df.columns:
        sem = colTypes.get(col, "unknown")
        if sem in ("id", "index"):
            continue
        col_lower = col.lower().replace(" ", "_")
        for kw in keywords:
            if kw in col_lower:
                # Prefer numeric, then categorical, skip text/datetime
                priority = 0 if sem == "numeric" else (1 if sem == "categorical" else 2)
                candidates.append((priority, col))
                break

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def promptTarget(df: pd.DataFrame, colTypes: dict,
                  suggested: Optional[str]) -> Optional[str]:
    """
    Interactively ask the user to confirm or choose the target column.
    Shows a numbered list of usable columns (non-ID, non-datetime).
    """
    usable = [(i, col) for i, col in enumerate(df.columns)
              if colTypes.get(col, "unknown") not in ("id", "index", "datetime", "text")]

    print("\n🎯  SELECT TARGET COLUMN")
    print("─" * 50)
    for idx, col in usable:
        sem   = colTypes.get(col, "?")
        icon  = {"numeric": "🔢", "categorical": "🏷️ "}.get(sem, "•")
        tag   = "  ← suggested" if col == suggested else ""
        print(f"  [{idx:2d}] {col:<30} {icon} {sem}{tag}")
    print("  [ n] No target / unsupervised analysis")
    print("─" * 50)

    default_hint = f" (default: '{suggested}')" if suggested else " (default: n)"
    choice = input(f"  Enter column number or name{default_hint}: ").strip()

    if choice.lower() in ("n", "no", "none", ""):
        if suggested and choice == "":
            print(f"  ✅ Using suggested target: '{suggested}'")
            return suggested
        print("  ℹ️  No target selected — running unsupervised EDA.")
        return None

    # Match by number
    if choice.isdigit():
        idx = int(choice)
        if idx < len(df.columns):
            selected = df.columns[idx]
            print(f"  ✅ Target set to: '{selected}'")
            return selected

    # Match by name (case-insensitive)
    matches = [col for col in df.columns if col.lower() == choice.lower()]
    if matches:
        print(f"  ✅ Target set to: '{matches[0]}'")
        return matches[0]

    print(f"  ⚠️  '{choice}' not found — no target selected.")
    return None


def printProfile(profile: dict):
    """Pretty-print the dataset profile summary (used for memory logging)."""
    print(f"\n  📁 {profile['file']}  |  "
          f"{profile['rows']:,} rows × {profile['columns']} cols  |  "
          f"{profile['missing_cells']:,} missing cells  |  "
          f"{profile['memory_mb']} MB")