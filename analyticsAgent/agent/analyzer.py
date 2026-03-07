"""
analyzer.py — Exploratory Data Analysis (EDA) Module
Auto-generates charts, stats, and insights from any dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime


sns.set_theme(style="darkgrid", palette="muted")
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
PALETTE = ["#7c83fd", "#fd7c7c", "#7cfdba", "#fdd47c", "#c47cfd", "#7cd4fd"]


class Analyzer:
    def __init__(self, df: pd.DataFrame, columnTypes: dict,
                 target_col: str = None, output_dir: str = "outputs"):
        self.df          = df.copy()
        self.columnTypes = columnTypes
        self.target_col  = target_col
        self.output_dir  = output_dir
        self.report      = {}   
        os.makedirs(output_dir, exist_ok=True)

    
    
    

    def runFullEda(self) -> dict:
        """Run all EDA steps and return a summary report dict."""
        print("\n EXPLORATORY DATA ANALYSIS")
        print("═" * 50)

        self._summaryStatistics()
        self._missingValueHeatmap()
        self._numericDistributions()
        self._categoricalDistributions()
        self._correlationHeatmap()
        self._targetAnalysis()
        self._timeSeriesPlots()
        self._outlierSummary()
        self._saveReport()

        print(f"\n✅ EDA complete — charts saved to /{self.output_dir}/")
        print("═" * 50)
        return self.report

    
    
    

    def _summaryStatistics(self):
        """Print and store summary stats for all columns."""
        print("\n Summary Statistics")
        numDf = self.df.select_dtypes(include="number")
        if numDf.empty:
            print("   No numeric columns found.")
            return

        stats = numDf.describe().T
        stats["skewness"] = numDf.skew()
        stats["kurtosis"] = numDf.kurtosis()
        print(stats[["mean", "std", "min", "50%", "max", "skewness"]].to_string())

        self.report["summary_stats"] = stats.round(4).to_dict()

        
        skewed = stats[stats["skewness"].abs() > 1].index.tolist()
        if skewed:
            print(f"\n   ⚠️  Highly skewed columns (|skew| > 1): {skewed}")
            self.report["skewed_columns"] = skewed

    def _missingValueHeatmap(self):
        """Visualize missing values across the dataset."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            print("\n✅ No missing values found.")
            self.report["missing_values"] = {}
            return

        print(f"\n Missing Values: {len(missing)} column(s) affected")
        for col, count in missing.items():
            pct = count / len(self.df) * 100
            bar = "█" * int(pct / 2)
            print(f"   {col:30s} {count:5d} ({pct:5.1f}%) {bar}")
        self.report["missing_values"] = missing.to_dict()

        
        fig, ax = plt.subplots(figsize=(10, max(3, len(missing) * 0.5)))
        pctMissing = (missing / len(self.df) * 100).sort_values()
        bars = ax.barh(pctMissing.index, pctMissing.values, color=PALETTE[0], alpha=0.85)
        ax.set_xlabel("% Missing")
        ax.set_title("Missing Value Distribution", fontsize=14, fontweight="bold")
        for bar, val in zip(bars, pctMissing.values):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)
        plt.tight_layout()
        self._saveFig("missing_values.png")

    def _numericDistributions(self):
        dropCols = {c for c, t in self.columnTypes.items() if t in ("id", "index")}
        numCols = [c for c in self.df.select_dtypes(include="number").columns
                    if c != self.target_col and c not in dropCols]
        if not numCols:
            return

        print(f"\n Plotting distributions for {len(numCols)} numeric column(s)...")

        ncols = 3
        nrows = max(1, (len(numCols) + ncols - 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 4.5, nrows * 3.5))
        axes = np.array(axes).flatten()

        for i, col in enumerate(numCols):
            ax = axes[i]
            data = self.df[col].dropna()
            ax.hist(data, bins=30, color=PALETTE[i % len(PALETTE)],
                    alpha=0.7, edgecolor="none", density=True)
            try:
                data.plot.kde(ax=ax, color="white", linewidth=1.5)
            except Exception:
                pass
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            
            ax.axvline(data.mean(), color="#ffd700", linestyle="--",
                       linewidth=1, label=f"mean={data.mean():.2f}")
            ax.axvline(data.median(), color="#ff6b6b", linestyle=":",
                       linewidth=1, label=f"median={data.median():.2f}")
            ax.legend(fontsize=7)

        
        for j in range(len(numCols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Numeric Column Distributions", fontsize=15,
                     fontweight="bold", y=1.01)
        plt.tight_layout()
        self._saveFig("numeric_distributions.png")

    def _categoricalDistributions(self):
        """Bar charts for categorical columns."""
        catCols = [c for c, t in self.columnTypes.items()
                    if t == "categorical" and c in self.df.columns
                    and c != self.target_col]
        if not catCols:
            return

        print(f"\n️  Plotting {len(catCols)} categorical column(s)...")

        ncols = 2
        nrows = max(1, (len(catCols) + 1) // ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 5, nrows * 3.5))
        axes = np.array(axes).flatten()

        for i, col in enumerate(catCols):
            ax = axes[i]
            vc = self.df[col].value_counts().head(15)
            bars = ax.barh(vc.index.astype(str), vc.values,
                           color=PALETTE[i % len(PALETTE)], alpha=0.85)
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.set_xlabel("Count")
            for bar, val in zip(bars, vc.values):
                ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", fontsize=8)

        for j in range(len(catCols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Categorical Column Distributions", fontsize=15,
                     fontweight="bold", y=1.01)
        plt.tight_layout()
        self._saveFig("categorical_distributions.png")

    def _correlationHeatmap(self):
        dropCols = [c for c, t in self.columnTypes.items()
                     if t in ("id", "index") and c in self.df.columns]
        numDf = self.df.drop(columns=dropCols).select_dtypes(include="number")

        if numDf.shape[1] < 2:
            print("\nNot enough numeric columns for a correlation heatmap (need ≥ 2).")
            self.report["top_correlations"] = {}
            return

        print(f"\n Generating correlation heatmap ({numDf.shape[1]} columns)...")
        corr = numDf.corr()

        
        upperMask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        pairs = (
            corr.where(upperMask)
                .stack()                     
                .dropna()                    
                .sort_values(ascending=False)
        )

        if pairs.empty:
            print("   No valid correlation pairs found.")
            self.report["top_correlations"] = {}
            return

        top = pairs.head(5)
        self.report["top_correlations"] = {
            str(k): round(float(v), 4)
            for k, v in top.items()
            if not np.isnan(v)
        }

        print("   Top correlations:")
        for (c1, c2), val in top.items():
            print(f"   {c1} ↔ {c2}: {val:.3f}")

        fig, ax = plt.subplots(figsize=(max(8, numDf.shape[1] * 0.8),
                                        max(6, numDf.shape[1] * 0.7)))
        plotMask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(corr, mask=plotMask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt=".2f", square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._saveFig("correlation_heatmap.png")

    def _targetAnalysis(self):
        """Analyze target column distribution and its relationship with features."""
        if not self.target_col or self.target_col not in self.df.columns:
            return

        print(f"\n Target Column Analysis: '{self.target_col}'")
        target = self.df[self.target_col].dropna()

        if pd.api.types.is_numeric_dtype(target):
            
            print(f"   Type     : Regression (continuous)")
            print(f"   Mean     : {target.mean():.4f}")
            print(f"   Std      : {target.std():.4f}")
            print(f"   Range    : {target.min():.4f} → {target.max():.4f}")
            self.report["target"] = {
                "column": self.target_col, "task": "regression",
                "mean": round(target.mean(), 4),
                "std": round(target.std(), 4),
            }

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(target, bins=40, color=PALETTE[0], alpha=0.8, edgecolor="none")
            axes[0].set_title(f"Target Distribution: {self.target_col}")
            axes[0].axvline(target.mean(), color="#ffd700", linestyle="--", label="mean")
            axes[0].legend()

            axes[1].boxplot(target, vert=False, patch_artist=True,
                            boxprops=dict(facecolor=PALETTE[0], alpha=0.7))
            axes[1].set_title("Target Boxplot")
            plt.tight_layout()

        else:
            
            vc = target.value_counts()
            print(f"   Type     : Classification ({vc.shape[0]} classes)")
            print(f"   Balance  :")
            for cls, cnt in vc.items():
                print(f"     {str(cls):20s}: {cnt} ({cnt/len(target)*100:.1f}%)")
            self.report["target"] = {
                "column": self.target_col, "task": "classification",
                "classes": vc.index.tolist(),
                "class_counts": vc.to_dict(),
            }

            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(vc.index.astype(str), vc.values,
                          color=PALETTE[:len(vc)], alpha=0.85)
            ax.set_title(f"Class Distribution: {self.target_col}", fontsize=13,
                         fontweight="bold")
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(val), ha="center", fontsize=10)
            plt.tight_layout()

        self._saveFig("target_analysis.png")

    def _timeSeriesPlots(self):
        """Plot time series trends for any datetime-derived columns."""
        
        timeMarkers = ["_year", "_month", "_quarter"]
        timeCols = [c for c in self.df.columns
                     if any(m in c for m in timeMarkers)]

        
        dtCols = [c for c in self.df.columns
                   if pd.api.types.is_datetime64_any_dtype(self.df[c])]

        if not dtCols:
            return

        print(f"\n Plotting time series for: {dtCols}")
        numCols = [c for c in self.df.select_dtypes(include="number").columns
                    if c != self.target_col][:4]  

        for dtCol in dtCols:
            if not numCols:
                break
            dfTs = self.df[[dtCol] + numCols].dropna().sort_values(dtCol)
            dfTs = dfTs.set_index(dtCol)

            fig, axes = plt.subplots(len(numCols), 1,
                                      figsize=(12, 2.5 * len(numCols)),
                                      sharex=True)
            axes = [axes] if len(numCols) == 1 else list(axes)

            for ax, col in zip(axes, numCols):
                
                try:
                    ts = dfTs[col].resample("ME").mean()
                except Exception:
                    ts = dfTs[col]
                ax.plot(ts.index, ts.values, color=PALETTE[0],
                        linewidth=1.5, alpha=0.9)
                ax.fill_between(ts.index, ts.values, alpha=0.2, color=PALETTE[0])
                ax.set_ylabel(col, fontsize=9)
                ax.set_title(f"{col} over time", fontsize=10)

            axes[-1].set_xlabel(dtCol)
            fig.suptitle(f"Time Series Trends (by {dtCol})",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            self._saveFig(f"timeseries_{dtCol}.png")

    def _outlierSummary(self):
        """Detect outliers using IQR method — excludes index/id columns."""
        dropCols = {c for c, t in self.columnTypes.items() if t in ("id", "index")}
        numDf = self.df.drop(columns=[c for c in dropCols if c in self.df.columns]
                              ).select_dtypes(include="number")
        if numDf.empty:
            return

        print("\n Outlier Detection (IQR method):")
        outlierInfo = {}

        for col in numDf.columns:
            s = numDf[col].dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            nOut = ((s < lower) | (s > upper)).sum()
            pct = nOut / len(s) * 100
            if nOut > 0:
                print(f"   {col:30s} {nOut:5d} outliers ({pct:.1f}%)")
                outlierInfo[col] = {"count": int(nOut), "pct": round(pct, 2),
                                     "lower_bound": round(lower, 4),
                                     "upper_bound": round(upper, 4)}

        if not outlierInfo:
            print("   No significant outliers detected.")
        self.report["outliers"] = outlierInfo

        
        outlierCols = list(outlierInfo.keys())[:8]
        if outlierCols:
            fig, ax = plt.subplots(figsize=(max(8, len(outlierCols) * 1.2), 5))
            data = [self.df[c].dropna().values for c in outlierCols]
            bp = ax.boxplot(data, labels=outlierCols, patch_artist=True,
                            notch=False, vert=True)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(PALETTE[i % len(PALETTE)])
                patch.set_alpha(0.7)
            ax.set_title("Outlier Overview (Boxplots)", fontsize=14,
                         fontweight="bold")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            self._saveFig("outliers_boxplot.png")

    def _saveReport(self):
        """Save the full EDA report to JSON."""
        self.report["generated_at"] = datetime.now().isoformat()
        report_path = os.path.join(self.output_dir, "eda_report.json")
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        print(f"\n EDA report saved → {report_path}")

    def _saveFig(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches="tight", facecolor=plt.rcParams["figure.facecolor"])
        plt.close()
        print(f"    Saved: {filename}")