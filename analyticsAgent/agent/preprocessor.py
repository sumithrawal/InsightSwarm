import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import json
import os
from pathlib import Path


class Preprocessor:
    def __init__(self, columnTypes: dict):
        self.columnTypes = columnTypes
        self.label_encoders = {}
        self.scaler = None
        self.imputers = {}
        self.dropped_columns = []
        self.datetime_source_columns = []
        self.fit_columns = []   
        self._is_fit = False

    
    
    

    def fitTransform(self, df: pd.DataFrame, target_col: str = None,
                      scale: str = "standard") -> pd.DataFrame:
        """
        Fit the preprocessor on training data and return transformed DataFrame.
        scale: 'standard' | 'minmax' | None
        """
        print("\n PREPROCESSING")
        print("─" * 40)

        df = df.copy()
        df = self._dropIdColumns(df)
        df = self._handleDuplicates(df)
        df = self._parseDatetimes(df)
        df = self._engineerTimeFeatures(df)
        df = self._imputeMissing(df, fit=True)
        df = self._encodeCategoricals(df, target_col, fit=True)
        df = self._dropHighCardinalityText(df, target_col)

        if scale:
            df = self._scaleNumerics(df, target_col, method=scale, fit=True)

        self.fit_columns = list(df.columns)
        self._is_fit = True

        print(f"\n✅ Preprocessing complete → {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Apply previously-fit transformations to new data."""
        if not self._is_fit:
            raise RuntimeError("Call fit_transform() before transform().")

        df = df.copy()
        df = self._dropIdColumns(df)
        df = self._parseDatetimes(df)
        df = self._engineerTimeFeatures(df)
        df = self._imputeMissing(df, fit=False)
        df = self._encodeCategoricals(df, target_col, fit=False)
        df = self._dropHighCardinalityText(df, target_col)

        if self.scaler:
            numCols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])
                        and c != target_col and c in self.fit_columns]
            if numCols:
                df[numCols] = self.scaler.transform(df[numCols])

        
        for col in self.fit_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in self.fit_columns if c in df.columns]]

        return df

    def saveState(self, path: str):
        """Persist encoder mappings to JSON (scalers need joblib separately)."""
        state = {
            "column_types": self.columnTypes,
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
        print(f" Preprocessor state saved to {path}")

    
    
    

    def _dropIdColumns(self, df: pd.DataFrame) -> pd.DataFrame:
        idCols = [c for c, t in self.columnTypes.items()
                   if t in ("id", "index") and c in df.columns]
        if idCols:
            df = df.drop(columns=idCols)
            self.dropped_columns.extend(idCols)
            print(f"    Dropped index/ID columns: {id_cols}")
        return df

    def _handleDuplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        nBefore = len(df)
        df = df.drop_duplicates()
        removed = nBefore - len(df)
        if removed:
            print(f"   Removed {removed:,} duplicate rows")
        return df

    def _parseDatetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        dtCols = [c for c, t in self.columnTypes.items()
                   if t == "datetime" and c in df.columns]
        for col in dtCols:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True,
                                          errors="coerce")
                self.datetime_source_columns.append(col)
            except Exception:
                pass
        return df

    def _engineerTimeFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
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
            print(f"   Engineered time features from: {engineered}")

        return df

    def _imputeMissing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Impute: median for numeric, most_frequent for categorical."""
        numCols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                    and df[c].isnull().any()]
        catCols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                    and df[c].isnull().any()]

        if numCols:
            if fit:
                self.imputers["numeric"] = SimpleImputer(strategy="median")
                df[numCols] = self.imputers["numeric"].fitTransform(df[numCols])
            else:
                df[numCols] = self.imputers["numeric"].transform(df[numCols])
            print(f"   Imputed {len(num_cols)} numeric column(s) with median")

        if catCols:
            if fit:
                self.imputers["categorical"] = SimpleImputer(strategy="most_frequent")
                df[catCols] = self.imputers["categorical"].fitTransform(df[catCols])
            else:
                df[catCols] = self.imputers["categorical"].transform(df[catCols])
            print(f"   Imputed {len(cat_cols)} categorical column(s) with mode")

        return df

    def _encodeCategoricals(self, df: pd.DataFrame, target_col: str,
                              fit: bool) -> pd.DataFrame:
        """Label-encode low-cardinality categoricals. One-hot for <=10 unique."""
        catCols = [c for c in df.columns
                    if not pd.api.types.is_numeric_dtype(df[c])
                    and c != target_col]

        onehotCols = []
        labelCols  = []

        for col in catCols:
            nunique = df[col].nunique()
            if nunique <= 10:
                onehotCols.append(col)
            else:
                labelCols.append(col)

        
        if onehotCols:
            df = pd.get_dummies(df, columns=onehotCols, drop_first=True,
                                dtype=int)
            print(f"   One-hot encoded: {onehot_cols}")

        
        for col in labelCols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fitTransform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).map(
                        lambda x: x if x in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        if labelCols:
            print(f"   Label encoded: {label_cols}")

        return df

    def _dropHighCardinalityText(self, df: pd.DataFrame,
                                     target_col: str) -> pd.DataFrame:
        """Drop free-text columns (still object dtype after encoding)."""
        textCols = [c for c in df.columns
                     if pd.api.types.is_object_dtype(df[c]) and c != target_col]
        if textCols:
            df = df.drop(columns=textCols)
            print(f"Dropped high-cardinality text columns: {text_cols}")
        return df

    def _scaleNumerics(self, df: pd.DataFrame, target_col: str,
                        method: str, fit: bool) -> pd.DataFrame:
        numCols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
        if not numCols:
            return df

        if fit:
            self.scaler = (StandardScaler() if method == "standard"
                           else MinMaxScaler())
            df[numCols] = self.scaler.fitTransform(df[numCols])
            print(f"   Scaled {len(num_cols)} numeric column(s) [{method}]")
        else:
            df[numCols] = self.scaler.transform(df[numCols])

        return df