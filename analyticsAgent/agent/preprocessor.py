import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import json
import os
from pathlib import Path


class Preprocessor:
    def __init__(self, column_types: dict):
        self.column_types = column_types
        self.label_encoders = {}
        self.scaler = None
        self.imputers = {}
        self.dropped_columns = []
        self.datetime_source_columns = []
        self.fit_columns = []   # columns present when fit() was called
        self._is_fit = False

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame, target_col: str = None,
                      scale: str = "standard") -> pd.DataFrame:
        """
        Fit the preprocessor on training data and return transformed DataFrame.
        scale: 'standard' | 'minmax' | None
        """
        print("\n🔧 PREPROCESSING")
        print("─" * 40)

        df = df.copy()
        df = self._drop_id_columns(df)
        df = self._handle_duplicates(df)
        df = self._parse_datetimes(df)
        df = self._engineer_time_features(df)
        df = self._impute_missing(df, fit=True)
        df = self._encode_categoricals(df, target_col, fit=True)
        df = self._drop_high_cardinality_text(df, target_col)

        if scale:
            df = self._scale_numerics(df, target_col, method=scale, fit=True)

        self.fit_columns = list(df.columns)
        self._is_fit = True

        print(f"\n✅ Preprocessing complete → {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Apply previously-fit transformations to new data."""
        if not self._is_fit:
            raise RuntimeError("Call fit_transform() before transform().")

        df = df.copy()
        df = self._drop_id_columns(df)
        df = self._parse_datetimes(df)
        df = self._engineer_time_features(df)
        df = self._impute_missing(df, fit=False)
        df = self._encode_categoricals(df, target_col, fit=False)
        df = self._drop_high_cardinality_text(df, target_col)

        if self.scaler:
            num_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])
                        and c != target_col and c in self.fit_columns]
            if num_cols:
                df[num_cols] = self.scaler.transform(df[num_cols])

        # Align columns to training schema
        for col in self.fit_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in self.fit_columns if c in df.columns]]

        return df

    def save_state(self, path: str):
        """Persist encoder mappings to JSON (scalers need joblib separately)."""
        state = {
            "column_types": self.column_types,
            "dropped_columns": self.dropped_columns,
            "datetime_source_columns": self.datetime_source_columns,
            "fit_columns": self.fit_columns,
            "label_encoders": {
                col: dict(zip(le.classes_.tolist(),
                              le.transform(le.classes_).tolist()))
                for col, le in self.label_encoders.items()
            }
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        print(f"💾 Preprocessor state saved to {path}")

    # ─────────────────────────────────────────────
    # PRIVATE STEPS
    # ─────────────────────────────────────────────

    def _drop_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        id_cols = [c for c, t in self.column_types.items()
                   if t in ("id", "index") and c in df.columns]
        if id_cols:
            df = df.drop(columns=id_cols)
            self.dropped_columns.extend(id_cols)
            print(f"  🗑  Dropped index/ID columns: {id_cols}")
        return df

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates()
        removed = n_before - len(df)
        if removed:
            print(f"  🧹 Removed {removed:,} duplicate rows")
        return df

    def _parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        dt_cols = [c for c, t in self.column_types.items()
                   if t == "datetime" and c in df.columns]
        for col in dt_cols:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True,
                                          errors="coerce")
                self.datetime_source_columns.append(col)
            except Exception:
                pass
        return df

    def _engineer_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract year, month, day, dayofweek, quarter from datetime columns."""
        engineered = []
        for col in self.datetime_source_columns:
            if col not in df.columns:
                continue
            s = df[col]
            df[f"{col}_year"]      = s.dt.year
            df[f"{col}_month"]     = s.dt.month
            df[f"{col}_day"]       = s.dt.day
            df[f"{col}_dayofweek"] = s.dt.dayofweek
            df[f"{col}_quarter"]   = s.dt.quarter
            df[f"{col}_is_weekend"]= s.dt.dayofweek.isin([5, 6]).astype(int)
            engineered.append(col)

        if engineered:
            df = df.drop(columns=engineered)
            print(f"  📅 Engineered time features from: {engineered}")

        return df

    def _impute_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Impute: median for numeric, most_frequent for categorical."""
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                    and df[c].isnull().any()]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                    and df[c].isnull().any()]

        if num_cols:
            if fit:
                self.imputers["numeric"] = SimpleImputer(strategy="median")
                df[num_cols] = self.imputers["numeric"].fit_transform(df[num_cols])
            else:
                df[num_cols] = self.imputers["numeric"].transform(df[num_cols])
            print(f"  🩹 Imputed {len(num_cols)} numeric column(s) with median")

        if cat_cols:
            if fit:
                self.imputers["categorical"] = SimpleImputer(strategy="most_frequent")
                df[cat_cols] = self.imputers["categorical"].fit_transform(df[cat_cols])
            else:
                df[cat_cols] = self.imputers["categorical"].transform(df[cat_cols])
            print(f"  🩹 Imputed {len(cat_cols)} categorical column(s) with mode")

        return df

    def _encode_categoricals(self, df: pd.DataFrame, target_col: str,
                              fit: bool) -> pd.DataFrame:
        """Label-encode low-cardinality categoricals. One-hot for <=10 unique."""
        cat_cols = [c for c in df.columns
                    if not pd.api.types.is_numeric_dtype(df[c])
                    and c != target_col]

        onehot_cols = []
        label_cols  = []

        for col in cat_cols:
            nunique = df[col].nunique()
            if nunique <= 10:
                onehot_cols.append(col)
            else:
                label_cols.append(col)

        # One-hot encoding
        if onehot_cols:
            df = pd.get_dummies(df, columns=onehot_cols, drop_first=True,
                                dtype=int)
            print(f"  🔠 One-hot encoded: {onehot_cols}")

        # Label encoding for higher-cardinality categoricals
        for col in label_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    # Handle unseen labels gracefully
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).map(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        if label_cols:
            print(f"  🔡 Label encoded: {label_cols}")

        return df

    def _drop_high_cardinality_text(self, df: pd.DataFrame,
                                     target_col: str) -> pd.DataFrame:
        """Drop free-text columns (still object dtype after encoding)."""
        text_cols = [c for c in df.columns
                     if pd.api.types.is_object_dtype(df[c]) and c != target_col]
        if text_cols:
            df = df.drop(columns=text_cols)
            print(f"Dropped high-cardinality text columns: {text_cols}")
        return df

    def _scale_numerics(self, df: pd.DataFrame, target_col: str,
                        method: str, fit: bool) -> pd.DataFrame:
        num_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
        if not num_cols:
            return df

        if fit:
            self.scaler = (StandardScaler() if method == "standard"
                           else MinMaxScaler())
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
            print(f"  📏 Scaled {len(num_cols)} numeric column(s) [{method}]")
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])

        return df