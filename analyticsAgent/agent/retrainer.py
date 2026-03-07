"""
retrainer.py — Phase 4: Self-Improvement Loop
The agent retrains itself by combining:
  1. Original training data
  2. New data files (if uploaded)
  3. User corrections (feedback entries)
  4. Feature hints (user-specified column weights)
Then compares the new model vs the previous best — keeps whichever wins.
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from agent.predictor import Predictor
from agent.memory import Memory


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


class Retrainer:
    """
    Self-improvement engine. Merges original data + new data + corrections,
    retrains all models, and promotes the new model only if it beats
    the current saved champion.
    """

    def __init__(self, originalFile: str, target_col: str, colTypes: dict,
                 memory: Memory, model_dir: str = "models",
                 output_dir: str = "outputs"):
        self.originalFile = originalFile
        self.target_col    = target_col
        self.colTypes     = colTypes
        self.memory        = memory
        self.model_dir     = model_dir
        self.output_dir    = output_dir
        os.makedirs(model_dir,  exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    
    
    

    def run(self, newDataFiles: list = None,
            featureWeights: dict = None) -> dict:
        """
        Full retraining pipeline:
          1. Load original data
          2. Apply corrections from feedback log
          3. Merge any new data files
          4. Apply feature weights (boost important columns)
          5. Retrain all models
          6. Compare vs champion — promote if better

        Returns dict with outcome summary.
        """
        print("\n SELF-IMPROVEMENT — PHASE 4")
        print("═" * 55)

        
        df = self._loadOriginal()
        print(f"   Original data : {df.shape[0]:,} rows")

        
        df, nCorrections = self._applyCorrections(df)

        
        if newDataFiles:
            df, nNew = self._mergeNewData(df, newDataFiles)
        else:
            nNew = 0

        print(f"\n Final dataset : {df.shape[0]:,} rows "
              f"(+{nCorrections} corrections, +{nNew} new rows)")

        
        if featureWeights:
            df = self._applyFeatureWeights(df, featureWeights)

        
        metaPath = os.path.join(self.model_dir, "best_model_meta.json")
        prevScore, scoreKey = self._getChampionScore(metaPath)
        print(f"\n Current champion score ({scoreKey}): {prevScore:.4f}")

        
        print(f"\n Retraining on {df.shape[0]:,} rows...")
        predictor = Predictor(df, self.colTypes, self.target_col,
                              model_dir=self.model_dir + "/challenger",
                              output_dir=self.output_dir)
        results   = predictor.run()

        if not results:
            print("❌ Retraining failed — no results.")
            return {"promoted": False, "reason": "training failed"}

        newScore = results.get(predictor.bestName, {}).get(scoreKey, -999)
        print(f"\n Challenger score ({scoreKey}): {newScore:.4f}")

        
        outcome = self._evaluateAndPromote(
            predictor, prevScore, newScore, scoreKey, metaPath
        )

        
        self._plotImprovementHistory(scoreKey)

        
        self.memory.markAllFeedbackApplied()

        
        self.memory.logModelVersion({
            "model_name":  predictor.bestName,
            "task":        predictor.task,
            "target_col":  self.target_col,
            "score_key":   scoreKey,
            "score":       newScore,
            "prev_score":  prevScore,
            "promoted":    outcome["promoted"],
            "n_rows":      df.shape[0],
            "n_corrections": nCorrections,
            "n_new_rows":  nNew,
            "trigger":     "retrain",
            "trained_at":  datetime.now().isoformat(),
        })

        return outcome

    
    
    

    def _loadOriginal(self) -> pd.DataFrame:
        """Load and return the original dataset."""
        from agent.loader import loadFile
        df = loadFile(self.originalFile)
        
        drop = [c for c, t in self.colTypes.items()
                if t in ("id", "index") and c in df.columns]
        return df.drop(columns=drop)

    def _applyCorrections(self, df: pd.DataFrame):
        """
        Apply correction-type feedback entries to the DataFrame.
        A correction entry has:
          data: [{"sku": "AN201-RED-L", "correct_value": 12}, ...]
          OR
          data: [{"row_index": 5, "correct_value": 12}, ...]
        """
        pending = [f for f in self.memory.getFeedback(applied=False)
                   if f.get("type") == "correction"]

        if not pending:
            print("   No corrections to apply.")
            return df, 0

        nApplied = 0
        for fb in pending:
            entries = fb.get("data", [])
            for entry in entries:
                val = entry.get("correct_value")
                if val is None:
                    continue
                
                if "sku" in entry and "SKU Code" in df.columns:
                    mask = df["SKU Code"] == entry["sku"]
                    df.loc[mask, self.target_col] = val
                    nApplied += int(mask.sum())
                
                elif "row_index" in entry:
                    idx = entry["row_index"]
                    if idx in df.index:
                        df.at[idx, self.target_col] = val
                        nApplied += 1

        print(f" Applied {nApplied} correction(s) from {len(pending)} feedback entry(ies)")
        return df, nApplied

    def _mergeNewData(self, df: pd.DataFrame, files: list):
        """Append new data files to the training set."""
        from agent.loader import loadFile
        newRows = 0
        for path in files:
            try:
                newDf = loadFile(path)
                
                drop = [c for c, t in self.colTypes.items()
                        if t in ("id", "index") and c in newDf.columns]
                newDf = newDf.drop(columns=drop, errors="ignore")
                
                common = [c for c in df.columns if c in newDf.columns]
                newDf = newDf[common]
                n = len(newDf)
                df = pd.concat([df, newDf], ignore_index=True)
                newRows += n
                print(f"   ➕ Merged {n:,} rows from {os.path.basename(path)}")
            except Exception as e:
                print(f"   ⚠️  Could not load {path}: {e}")
        return df, newRows

    def _applyFeatureWeights(self, df: pd.DataFrame,
                                featureWeights: dict) -> pd.DataFrame:
        """
        Boost underrepresented signal by duplicating rows where
        important feature values appear.
        feature_weights = {"Category": {"KURTA": 1, "LEHENGA CHOLI": 3}}
        means rows with LEHENGA CHOLI in Category are duplicated 3x.
        """
        extraFrames = [df]
        for col, weightMap in featureWeights.items():
            if col not in df.columns:
                continue
            for val, weight in weightMap.items():
                if weight <= 1:
                    continue
                subset = df[df[col] == val]
                if subset.empty:
                    continue
                for _ in range(weight - 1):
                    extraFrames.append(subset)
                print(f"   ⚖️  Boosted '{col}={val}' ×{weight}")

        result = pd.concat(extraFrames, ignore_index=True)
        print(f"   Dataset after weighting: {len(result):,} rows")
        return result

    
    
    

    def _getChampionScore(self, metaPath: str):
        """Read the current champion's score and score_key from metadata."""
        if not os.path.exists(metaPath):
            return -999, "test_r2"
        with open(metaPath) as f:
            meta = json.load(f)
        scoreKey = "test_r2" if meta.get("task") == "regression" else "test_f1"
        return meta.get("score", -999), scoreKey

    def _evaluateAndPromote(self, predictor, prevScore, newScore,
                               scoreKey, metaPath):
        """
        Compare challenger vs champion.
        Promote challenger if it scores higher by at least 0.5%.
        """
        improvement = newScore - prevScore
        threshold   = 0.005   

        print(f"\n{'─'*55}")
        print(f"  Champion  : {prevScore:.4f}")
        print(f"  Challenger: {newScore:.4f}")
        print(f"  Δ Change  : {improvement:+.4f}")
        print(f"{'─'*55}")

        if improvement >= threshold:
            
            challengerDir = self.model_dir + "/challenger"
            champModel    = os.path.join(self.model_dir, "best_model.pkl")
            champMeta     = metaPath

            
            archiveDir = os.path.join(self.model_dir, "archive")
            os.makedirs(archiveDir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(champModel):
                import shutil
                shutil.copy(champModel,
                            os.path.join(archiveDir, f"model_{ts}.pkl"))

            
            import shutil
            shutil.copy(os.path.join(challengerDir, "best_model.pkl"), champModel)
            shutil.copy(os.path.join(challengerDir, "best_model_meta.json"), champMeta)

            print(f"\n PROMOTED! New champion: {predictor.best_name}")
            print(f"   Improvement: {improvement:+.4f} ({improvement/abs(prevScore)*100:+.1f}%)"
                  if prevScore != 0 else f"   Improvement: {improvement:+.4f}")
            print(f"   Old model archived → {archiveDir}/model_{ts}.pkl")
            return {"promoted": True, "improvement": improvement,
                    "new_score": newScore, "prev_score": prevScore,
                    "best_model": predictor.bestName}
        else:
            print(f"\n  Champion retained (improvement {improvement:+.4f} < threshold {threshold})")
            print(f"   Challenger discarded — champion is still the best model.")
            return {"promoted": False, "improvement": improvement,
                    "new_score": newScore, "prev_score": prevScore,
                    "best_model": "champion retained"}

    
    
    

    def _plotImprovementHistory(self, scoreKey: str):
        """Plot model score over all retraining cycles."""
        versions = self.memory.getModelVersions(self.target_col)
        if len(versions) < 2:
            return

        scores     = [v.get("score", 0) for v in versions]
        labels     = [f"v{i+1}\n{v.get('model_name','')[:8]}"
                      for i, v in enumerate(versions)]
        promoted   = [v.get("promoted", True) for v in versions]
        colors     = ["#ffd700" if p else PALETTE[1] for p in promoted]

        fig, ax = plt.subplots(figsize=(max(8, len(scores) * 1.2), 5))
        ax.plot(range(len(scores)), scores, color=PALETTE[0],
                linewidth=2, marker="o", markersize=8, zorder=3)
        ax.scatter(range(len(scores)), scores, color=colors,
                   s=80, zorder=4)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(scoreKey.upper())
        ax.set_title(f"Model Improvement Over Time — {self.target_col}",
                     fontsize=13, fontweight="bold")

        
        for i, (s, p) in enumerate(zip(scores, promoted)):
            ax.annotate(f"{s:.3f}",
                        (i, s), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8,
                        color="#ffd700" if p else PALETTE[1])

        
        from matplotlib.lines import Line2D
        legend = [
            Line2D([0],[0], marker="o", color="w", markerfacecolor="#ffd700",
                   markersize=9, label="Promoted"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor=PALETTE[1],
                   markersize=9, label="Not promoted"),
        ]
        ax.legend(handles=legend)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "improvement_history.png")
        plt.savefig(path, bbox_inches="tight",
                    facecolor=plt.rcParams["figure.facecolor"])
        plt.close()
        print(f"    improvement_history.png")