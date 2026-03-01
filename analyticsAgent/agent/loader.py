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


SUPPORTED_FORMATS = [".csv", ".xlsx", ".xls"]


def load_file(filepath: str) -> pd.DataFrame:
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


def detect_column_types(df: pd.DataFrame) -> dict:
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
            # Heuristic: if very few unique values relative to rows → treat as categorical
            if nunique <= 20 and nunique / max(n, 1) < 0.05:
                types[col] = "categorical"
            else:
                types[col] = "numeric"

        elif pd.api.types.is_object_dtype(series):
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


def get_dataset_profile(df: pd.DataFrame, filepath: str) -> dict:
    """Generate a lightweight dataset profile for memory logging."""
    col_types = detect_column_types(df)

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
        "column_types": col_types,
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    return profile


def print_profile(profile: dict):
    """Pretty-print the dataset profile to the CLI."""
    print("\n" + "═" * 50)
    print("📊 DATASET PROFILE")
    print("═" * 50)
    print(f"  File         : {profile['file']}")
    print(f"  Shape        : {profile['rows']:,} rows × {profile['columns']} columns")
    print(f"  Missing cells: {profile['missing_cells']:,}")
    print(f"  Duplicate rows: {profile['duplicate_rows']:,}")
    print(f"  Memory usage : {profile['memory_mb']} MB")

    print("\n  Column Types Detected:")
    type_groups = {}
    for col, ctype in profile["column_types"].items():
        type_groups.setdefault(ctype, []).append(col)

    icons = {
        "numeric": "🔢",
        "categorical": "🏷️ ",
        "datetime": "📅",
        "text": "📝",
        "id": "🔑",
        "unknown": "❓",
    }
    for ctype, cols in type_groups.items():
        icon = icons.get(ctype, "•")
        print(f"    {icon} {ctype.upper()} ({len(cols)}): {', '.join(cols)}")
    print("═" * 50)