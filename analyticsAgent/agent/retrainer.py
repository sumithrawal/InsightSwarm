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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
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

    def __init__(self, original_file: str, target_col: str, col_types: dict,
                 memory: Memory, model_dir: str = "models",
                 output_dir: str = "outputs"):
        self.original_file = original_file
        self.target_col    = target_col
        self.col_types     = col_types
        self.memory        = memory
        self.model_dir     = model_dir
        self.output_dir    = output_dir
        os.makedirs(model_dir,  exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # MAIN ENTRY
    # ─────────────────────────────────────────────

    def run(self, new_data_files: list = None,
            feature_weights: dict = None) -> dict:
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
        print("\n🔁 SELF-IMPROVEMENT — PHASE 4")
        print("═" * 55)

        # 1. Load original data
        df = self._load_original()
        print(f"   Original data : {df.shape[0]:,} rows")

        # 2. Apply corrections
        df, n_corrections = self._apply_corrections(df)

        # 3. Merge new data
        if new_data_files:
            df, n_new = self._merge_new_data(df, new_data_files)
        else:
            n_new = 0

        print(f"\n📦 Final dataset : {df.shape[0]:,} rows "
              f"(+{n_corrections} corrections, +{n_new} new rows)")

        # 4. Apply feature weights (duplicate important rows to boost signal)
        if feature_weights:
            df = self._apply_feature_weights(df, feature_weights)

        # 5. Get current champion score
        meta_path = os.path.join(self.model_dir, "best_model_meta.json")
        prev_score, score_key = self._get_champion_score(meta_path)
        print(f"\n🏆 Current champion score ({score_key}): {prev_score:.4f}")

        # 6. Retrain
        print(f"\n🔄 Retraining on {df.shape[0]:,} rows...")
        predictor = Predictor(df, self.col_types, self.target_col,
                              model_dir=self.model_dir + "/challenger",
                              output_dir=self.output_dir)
        results   = predictor.run()

        if not results:
            print("❌ Retraining failed — no results.")
            return {"promoted": False, "reason": "training failed"}

        new_score = results.get(predictor.best_name, {}).get(score_key, -999)
        print(f"\n📊 Challenger score ({score_key}): {new_score:.4f}")

        # 7. Decide: promote challenger or keep champion
        outcome = self._evaluate_and_promote(
            predictor, prev_score, new_score, score_key, meta_path
        )

        # 8. Plot improvement history
        self._plot_improvement_history(score_key)

        # 9. Mark feedback as applied
        self.memory.mark_all_feedback_applied()

        # 10. Log version
        self.memory.log_model_version({
            "model_name":  predictor.best_name,
            "task":        predictor.task,
            "target_col":  self.target_col,
            "score_key":   score_key,
            "score":       new_score,
            "prev_score":  prev_score,
            "promoted":    outcome["promoted"],
            "n_rows":      df.shape[0],
            "n_corrections": n_corrections,
            "n_new_rows":  n_new,
            "trigger":     "retrain",
            "trained_at":  datetime.now().isoformat(),
        })

        return outcome

    # ─────────────────────────────────────────────
    # DATA PIPELINE
    # ─────────────────────────────────────────────

    def _load_original(self) -> pd.DataFrame:
        """Load and return the original dataset."""
        from agent.loader import load_file
        df = load_file(self.original_file)
        # Drop index-type columns
        drop = [c for c, t in self.col_types.items()
                if t in ("id", "index") and c in df.columns]
        return df.drop(columns=drop)

    def _apply_corrections(self, df: pd.DataFrame):
        """
        Apply correction-type feedback entries to the DataFrame.
        A correction entry has:
          data: [{"sku": "AN201-RED-L", "correct_value": 12}, ...]
          OR
          data: [{"row_index": 5, "correct_value": 12}, ...]
        """
        pending = [f for f in self.memory.get_feedback(applied=False)
                   if f.get("type") == "correction"]

        if not pending:
            print("   No corrections to apply.")
            return df, 0

        n_applied = 0
        for fb in pending:
            entries = fb.get("data", [])
            for entry in entries:
                val = entry.get("correct_value")
                if val is None:
                    continue
                # Match by SKU Code if present
                if "sku" in entry and "SKU Code" in df.columns:
                    mask = df["SKU Code"] == entry["sku"]
                    df.loc[mask, self.target_col] = val
                    n_applied += int(mask.sum())
                # Match by row index
                elif "row_index" in entry:
                    idx = entry["row_index"]
                    if idx in df.index:
                        df.at[idx, self.target_col] = val
                        n_applied += 1

        print(f"   ✏️  Applied {n_applied} correction(s) from {len(pending)} feedback entry(ies)")
        return df, n_applied

    def _merge_new_data(self, df: pd.DataFrame, files: list):
        """Append new data files to the training set."""
        from agent.loader import load_file
        new_rows = 0
        for path in files:
            try:
                new_df = load_file(path)
                # Drop index-type cols
                drop = [c for c, t in self.col_types.items()
                        if t in ("id", "index") and c in new_df.columns]
                new_df = new_df.drop(columns=drop, errors="ignore")
                # Only keep columns that exist in original
                common = [c for c in df.columns if c in new_df.columns]
                new_df = new_df[common]
                n = len(new_df)
                df = pd.concat([df, new_df], ignore_index=True)
                new_rows += n
                print(f"   ➕ Merged {n:,} rows from {os.path.basename(path)}")
            except Exception as e:
                print(f"   ⚠️  Could not load {path}: {e}")
        return df, new_rows

    def _apply_feature_weights(self, df: pd.DataFrame,
                                feature_weights: dict) -> pd.DataFrame:
        """
        Boost underrepresented signal by duplicating rows where
        important feature values appear.
        feature_weights = {"Category": {"KURTA": 1, "LEHENGA CHOLI": 3}}
        means rows with LEHENGA CHOLI in Category are duplicated 3x.
        """
        extra_frames = [df]
        for col, weight_map in feature_weights.items():
            if col not in df.columns:
                continue
            for val, weight in weight_map.items():
                if weight <= 1:
                    continue
                subset = df[df[col] == val]
                if subset.empty:
                    continue
                for _ in range(weight - 1):
                    extra_frames.append(subset)
                print(f"   ⚖️  Boosted '{col}={val}' ×{weight}")

        result = pd.concat(extra_frames, ignore_index=True)
        print(f"   Dataset after weighting: {len(result):,} rows")
        return result

    # ─────────────────────────────────────────────
    # CHAMPION vs CHALLENGER
    # ─────────────────────────────────────────────

    def _get_champion_score(self, meta_path: str):
        """Read the current champion's score and score_key from metadata."""
        if not os.path.exists(meta_path):
            return -999, "test_r2"
        with open(meta_path) as f:
            meta = json.load(f)
        score_key = "test_r2" if meta.get("task") == "regression" else "test_f1"
        return meta.get("score", -999), score_key

    def _evaluate_and_promote(self, predictor, prev_score, new_score,
                               score_key, meta_path):
        """
        Compare challenger vs champion.
        Promote challenger if it scores higher by at least 0.5%.
        """
        improvement = new_score - prev_score
        threshold   = 0.005   # must improve by at least 0.5%

        print(f"\n{'─'*55}")
        print(f"  Champion  : {prev_score:.4f}")
        print(f"  Challenger: {new_score:.4f}")
        print(f"  Δ Change  : {improvement:+.4f}")
        print(f"{'─'*55}")

        if improvement >= threshold:
            # Promote challenger → overwrite champion
            challenger_dir = self.model_dir + "/challenger"
            champ_model    = os.path.join(self.model_dir, "best_model.pkl")
            champ_meta     = meta_path

            # Archive the old champion first
            archive_dir = os.path.join(self.model_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(champ_model):
                import shutil
                shutil.copy(champ_model,
                            os.path.join(archive_dir, f"model_{ts}.pkl"))

            # Copy challenger to champion slot
            import shutil
            shutil.copy(os.path.join(challenger_dir, "best_model.pkl"), champ_model)
            shutil.copy(os.path.join(challenger_dir, "best_model_meta.json"), champ_meta)

            print(f"\n🚀 PROMOTED! New champion: {predictor.best_name}")
            print(f"   Improvement: {improvement:+.4f} ({improvement/abs(prev_score)*100:+.1f}%)"
                  if prev_score != 0 else f"   Improvement: {improvement:+.4f}")
            print(f"   Old model archived → {archive_dir}/model_{ts}.pkl")
            return {"promoted": True, "improvement": improvement,
                    "new_score": new_score, "prev_score": prev_score,
                    "best_model": predictor.best_name}
        else:
            print(f"\n🛡  Champion retained (improvement {improvement:+.4f} < threshold {threshold})")
            print(f"   Challenger discarded — champion is still the best model.")
            return {"promoted": False, "improvement": improvement,
                    "new_score": new_score, "prev_score": prev_score,
                    "best_model": "champion retained"}

    # ─────────────────────────────────────────────
    # IMPROVEMENT HISTORY CHART
    # ─────────────────────────────────────────────

    def _plot_improvement_history(self, score_key: str):
        """Plot model score over all retraining cycles."""
        versions = self.memory.get_model_versions(self.target_col)
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
        ax.set_ylabel(score_key.upper())
        ax.set_title(f"Model Improvement Over Time — {self.target_col}",
                     fontsize=13, fontweight="bold")

        # Annotate each point
        for i, (s, p) in enumerate(zip(scores, promoted)):
            ax.annotate(f"{s:.3f}",
                        (i, s), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8,
                        color="#ffd700" if p else PALETTE[1])

        # Legend
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
        print(f"   💾 improvement_history.png")