"""
predictor.py — Phase 3: Predictive Modeling
Auto-detects task type, trains multiple models, evaluates and saves the best one.
Handles skewed targets (log transform), encodes categoricals, and reports metrics.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import joblib

warnings.filterwarnings("ignore")

PALETTE = ["#7c83fd", "#fd7c7c", "#7cfdba", "#fdd47c", "#c47cfd", "#7cd4fd"]
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3d4d",
    "axes.labelcolor": "#c8ccd8",
    "axes.titlecolor": "#ffffff",
    "xtick.color": "#9a9db0",
    "ytick.color": "#9a9db0",
    "text.color": "#c8ccd8",
    "grid.color": "#2a2d3a",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
})


class Predictor:
    """
    Trains, evaluates, and saves the best ML model for a given dataset.
    Supports regression and classification automatically.
    """

    def __init__(self, df, col_types, target_col,
                 model_dir="models", output_dir="outputs"):
        self.df          = df.copy()
        self.col_types   = col_types
        self.target_col  = target_col
        self.model_dir   = model_dir
        self.output_dir  = output_dir
        self.task        = None
        self.log_target  = False
        self.encoders    = {}
        self.feature_cols = []
        self.results     = {}
        self.best_model  = None
        self.best_name   = None
        self._X_test     = None
        self._y_test     = None
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> dict:
        print("\n🤖 PREDICTIVE MODELING — PHASE 3")
        print("═" * 55)

        X, y = self._prepare_data()
        if X is None:
            return {}

        self._detect_task(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"\n📊 Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
        print(f"   Features: {list(X.columns)}")

        self._train_all(X_train, X_test, y_train, y_test)
        self._save_best(X_train, y_train)
        self._plot_results(X_test, y_test)
        self._save_report()

        return self.results

    # ── Data Preparation ─────────────────────────────────────────

    def _prepare_data(self):
        drop_types = {"id", "index", "text"}
        drop_cols  = [c for c, t in self.col_types.items()
                      if t in drop_types and c in self.df.columns]
        df = self.df.drop(columns=drop_cols).dropna(subset=[self.target_col])

        if self.target_col not in df.columns:
            print(f"❌ Target '{self.target_col}' not found.")
            return None, None

        y = df[self.target_col].copy()
        X = df.drop(columns=[self.target_col])

        # Encode string/object columns
        for col in X.columns:
            dtype_str = str(X[col].dtype)
            is_str    = (X[col].dtype.kind == "O" or dtype_str in ("str","string"))
            if is_str:
                X[col] = X[col].fillna("__missing__")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(X[col].median())

        # Log-transform heavily skewed numeric target
        if pd.api.types.is_numeric_dtype(y):
            skew = float(y.skew())
            if abs(skew) > 1:
                print(f"\n📐 Target '{self.target_col}' is heavily skewed (skew={skew:.2f})")
                print(f"   Applying log1p transform → model will learn on log scale")
                y = np.log1p(y)
                self.log_target = True

        self.feature_cols = list(X.columns)
        print(f"\n✅ Data ready: {X.shape[0]:,} rows × {X.shape[1]} features")
        return X, y

    def _detect_task(self, y):
        if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20:
            self.task = "classification"
        else:
            self.task = "regression"

        icon = "📈" if self.task == "regression" else "🏷️ "
        log_note = " (log-transformed)" if self.log_target else ""
        print(f"\n{icon} Task   : {self.task.upper()}")
        print(f"   Target : '{self.target_col}'{log_note}")

    def _get_models(self):
        if self.task == "regression":
            return {
                "Linear Regression":  LinearRegression(),
                "Ridge":              Ridge(alpha=1.0),
                "Lasso":              Lasso(alpha=0.1, max_iter=5000),
                "Decision Tree":      DecisionTreeRegressor(max_depth=8, random_state=42),
                "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
            }
        else:
            return {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
                "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
            }

    def _train_all(self, X_train, X_test, y_train, y_test):
        models    = self._get_models()
        score_key = "r2" if self.task == "regression" else "f1_weighted"
        metric    = "R²" if self.task == "regression" else "F1"
        aux_label = "MAE" if self.task == "regression" else "Accuracy"

        print(f"\n{'─'*55}")
        print(f"  {'Model':<26} {'CV '+metric:>9} {'Test '+metric:>10} {aux_label:>11}")
        print(f"{'─'*55}")

        for name, model in models.items():
            try:
                cv     = cross_val_score(model, X_train, y_train,
                                         cv=5, scoring=score_key, n_jobs=-1)
                cv_m   = cv.mean()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if self.task == "regression":
                    y_true_r = np.expm1(y_test)  if self.log_target else y_test
                    y_pred_r = np.expm1(y_pred)  if self.log_target else y_pred
                    ts   = r2_score(y_test, y_pred)
                    aux  = mean_absolute_error(y_true_r, y_pred_r)
                    rmse = float(np.sqrt(mean_squared_error(y_true_r, y_pred_r)))
                    self.results[name] = {
                        "cv_r2": round(cv_m, 4), "test_r2": round(ts, 4),
                        "mae": round(aux, 4), "rmse": round(rmse, 4), "model": model,
                    }
                else:
                    ts  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    aux = accuracy_score(y_test, y_pred)
                    self.results[name] = {
                        "cv_f1": round(cv_m, 4), "test_f1": round(ts, 4),
                        "accuracy": round(aux, 4), "model": model,
                    }

                print(f"  {name:<26} {cv_m:>9.4f} {ts:>10.4f} {aux:>11.4f}")

            except Exception as e:
                print(f"  {name:<26} ❌ {e}")

        print(f"{'─'*55}")
        self._X_test = X_test
        self._y_test = y_test

    def _save_best(self, X_train, y_train):
        sk = "test_r2" if self.task == "regression" else "test_f1"
        self.best_name  = max(self.results, key=lambda n: self.results[n][sk])
        info            = self.results[self.best_name]
        self.best_model = info["model"]
        self.best_model.fit(X_train, y_train)

        print(f"\n🏆 Best model : {self.best_name}")
        print(f"   Test {sk.split('_')[1].upper():<4} : {info[sk]:.4f}")
        if self.task == "regression":
            print(f"   MAE         : {info['mae']:.4f}")
            print(f"   RMSE        : {info['rmse']:.4f}")

        model_path = os.path.join(self.model_dir, "best_model.pkl")
        meta_path  = os.path.join(self.model_dir, "best_model_meta.json")
        joblib.dump(self.best_model, model_path)

        meta = {
            "model_name": self.best_name, "task": self.task,
            "target_col": self.target_col, "log_target": self.log_target,
            "feature_cols": self.feature_cols,
            "score": info[sk], "trained_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n💾 Saved → {model_path}")
        print(f"   Meta → {meta_path}")

    def _plot_results(self, X_test, y_test):
        print("\n📊 Generating performance charts...")
        sk     = "test_r2" if self.task == "regression" else "test_f1"
        names  = list(self.results.keys())
        scores = [self.results[n][sk] for n in names]
        colors = ["#ffd700" if n == self.best_name else PALETTE[0] for n in names]

        # Chart 1 — Model comparison
        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.7)))
        bars = ax.barh(names, scores, color=colors, alpha=0.85)
        ax.set_xlabel(sk.upper())
        ax.set_title(f"Model Comparison — Test {sk.split('_')[1].upper()}",
                     fontsize=14, fontweight="bold")
        for bar, val in zip(bars, scores):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)
        plt.tight_layout()
        self._save_fig("model_comparison.png")

        # Chart 2 — Actual vs Predicted (regression)
        if self.task == "regression":
            y_pred = self.best_model.predict(X_test)
            yt = np.expm1(y_test) if self.log_target else y_test
            yp = np.expm1(y_pred) if self.log_target else y_pred

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].scatter(yt, yp, alpha=0.35, color=PALETTE[0], s=12)
            lim = [min(float(yt.min()), float(yp.min())),
                   max(float(yt.max()), float(yp.max()))]
            axes[0].plot(lim, lim, "r--", linewidth=1.5, label="Perfect prediction")
            axes[0].set_xlabel("Actual")
            axes[0].set_ylabel("Predicted")
            axes[0].set_title(f"Actual vs Predicted — {self.best_name}",
                               fontsize=12, fontweight="bold")
            axes[0].legend()

            residuals = np.array(yp) - np.array(yt)
            axes[1].scatter(yp, residuals, alpha=0.35, color=PALETTE[1], s=12)
            axes[1].axhline(0, color="white", linestyle="--", linewidth=1)
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("Residual (Predicted − Actual)")
            axes[1].set_title("Residual Plot", fontsize=12, fontweight="bold")
            plt.tight_layout()
            self._save_fig("prediction_vs_actual.png")

        # Chart 3 — Feature importance
        if hasattr(self.best_model, "feature_importances_"):
            imp = self.best_model.feature_importances_
            fd  = pd.DataFrame({"feature": self.feature_cols, "importance": imp})
            fd  = fd.sort_values("importance").tail(20)
            fig, ax = plt.subplots(figsize=(9, max(4, len(fd) * 0.45)))
            bars = ax.barh(fd["feature"], fd["importance"], color=PALETTE[2], alpha=0.85)
            ax.set_title(f"Feature Importance — {self.best_name}",
                         fontsize=13, fontweight="bold")
            for bar, val in zip(bars, fd["importance"]):
                ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8)
            plt.tight_layout()
            self._save_fig("feature_importance.png")

        # Chart 4 — Confusion matrix (classification)
        if self.task == "classification":
            y_pred = self.best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(max(6, cm.shape[0] * 0.8),
                                            max(5, cm.shape[0] * 0.7)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.5)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {self.best_name}",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            self._save_fig("confusion_matrix.png")

    def _save_report(self):
        report = {
            "task": self.task, "target_col": self.target_col,
            "log_target": self.log_target, "features": self.feature_cols,
            "best_model": self.best_name,
            "results": {n: {k: v for k, v in i.items() if k != "model"}
                        for n, i in self.results.items()},
            "generated_at": datetime.now().isoformat(),
        }
        path = os.path.join(self.output_dir, "modeling_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report → {path}")

    def _save_fig(self, filename):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches="tight",
                    facecolor=plt.rcParams["figure.facecolor"])
        plt.close()
        print(f"   💾 {filename}")

    @staticmethod
    def load_and_predict(model_dir, new_data):
        """Load saved model and predict on new data."""
        model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        with open(os.path.join(model_dir, "best_model_meta.json")) as f:
            meta = json.load(f)
        for col in meta["feature_cols"]:
            if col not in new_data.columns:
                new_data[col] = 0
        X    = new_data[meta["feature_cols"]]
        preds = model.predict(X)
        if meta.get("log_target"):
            preds = np.expm1(preds)
        return preds, meta