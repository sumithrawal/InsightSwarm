"""
memory.py — Agent Memory System
Tracks every run, feedback received, model versions, and performance history.
The agent uses this to learn which strategies work best over time.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path


MEMORYPath   = "feedback/memory.json"
FEEDBACKPath = "feedback/feedback_log.json"
VERSIONSPath = "feedback/model_versions.json"



class Memory:
    """
    Persistent key-value store for the agent's brain.
    Tracks: runs, feedback entries, model version history.
    """

    def __init__(self, baseDir: str = "feedback"):
        self.baseDir      = baseDir
        self.memory_path   = os.path.join(baseDir, "memory.json")
        self.feedback_path = os.path.join(baseDir, "feedback_log.json")
        self.versions_path = os.path.join(baseDir, "model_versions.json")
        os.makedirs(baseDir, exist_ok=True)
        self._memory   = self._load(self.memory_path,   {"runs": []})
        self._feedback = self._load(self.feedback_path, {"entries": []})
        self._versions = self._load(self.versions_path, {"versions": []})

    

    def logRun(self, run: dict):
        """Record any agent run (load/analyze/train/predict/retrain)."""
        run["logged_at"] = datetime.now().isoformat()
        self._memory["runs"].append(run)
        self._save(self.memory_path, self._memory)

    def getRuns(self, run_type: str = None) -> list:
        runs = self._memory.get("runs", [])
        if run_type:
            runs = [r for r in runs if r.get("type") == run_type]
        return runs

    def lastRun(self, run_type: str = None) -> dict | None:
        runs = self.getRuns(run_type)
        return runs[-1] if runs else None

    

    def addFeedback(self, feedback: dict) -> str:
        """
        Store a feedback entry. Returns its unique ID.

        feedback dict keys:
          - type         : "correction" | "feature_hint" | "new_data" | "label"
          - target_col   : which column the feedback is about
          - detail       : human-readable explanation
          - data         : optional dict/list payload (corrections, weights etc.)
          - run_id       : optional — which prediction run this feedback is for
        """
        fid = self._makeId("fb")
        entry = {
            "id":          fid,
            "created_at":  datetime.now().isoformat(),
            "applied":     False,
            **feedback,
        }
        self._feedback["entries"].append(entry)
        self._save(self.feedback_path, self._feedback)
        return fid

    def getFeedback(self, applied: bool = None) -> list:
        entries = self._feedback.get("entries", [])
        if applied is not None:
            entries = [e for e in entries if e.get("applied") == applied]
        return entries

    def markFeedbackApplied(self, fid: str):
        for entry in self._feedback["entries"]:
            if entry["id"] == fid:
                entry["applied"] = True
                entry["applied_at"] = datetime.now().isoformat()
        self._save(self.feedback_path, self._feedback)

    def markAllFeedbackApplied(self):
        for entry in self._feedback["entries"]:
            if not entry.get("applied"):
                entry["applied"] = True
                entry["applied_at"] = datetime.now().isoformat()
        self._save(self.feedback_path, self._feedback)

    def pendingFeedbackCount(self) -> int:
        return len(self.getFeedback(applied=False))

    

    def logModelVersion(self, meta: dict):
        """
        Record a trained model version with its performance metrics.
        meta should include: model_name, task, target_col, score,
                             score_key, trained_at, trigger (initial/retrain)
        """
        vid = self._makeId("v")
        version = {"version_id": vid, **meta}
        self._versions["versions"].append(version)
        self._save(self.versions_path, self._versions)
        return vid

    def getModelVersions(self, target_col: str = None) -> list:
        versions = self._versions.get("versions", [])
        if target_col:
            versions = [v for v in versions if v.get("target_col") == target_col]
        return versions

    def bestKnownScore(self, target_col: str, scoreKey: str) -> float:
        """Return the highest score ever recorded for a target column."""
        versions = self.getModelVersions(target_col)
        scores   = [v.get("score", -999) for v in versions
                    if v.get("score_key") == scoreKey]
        return max(scores) if scores else -999

    

    @staticmethod
    def hashDataframe(df) -> str:
        """Create a fingerprint of a DataFrame for change detection."""
        return hashlib.md5(
            str(df.shape).encode() +
            str(df.columns.tolist()).encode() +
            str(df.head(100).values.tolist()).encode()
        ).hexdigest()[:12]

    

    def printSummary(self):
        runs      = self._memory.get("runs", [])
        feedback  = self._feedback.get("entries", [])
        versions  = self._versions.get("versions", [])
        pending   = [f for f in feedback if not f.get("applied")]

        print("\n" + "═" * 55)
        print("  AGENT MEMORY SUMMARY")
        print("═" * 55)
        print(f"  Total runs       : {len(runs)}")

        
        byType = {}
        for r in runs:
            byType[r.get("type","?")] = byType.get(r.get("type","?"), 0) + 1
        for t, n in sorted(byType.items()):
            print(f"    ↳ {t:<12}: {n}")

        print(f"\n  Feedback entries : {len(feedback)}")
        print(f"    ↳ Pending      : {len(pending)}")
        print(f"    ↳ Applied      : {len(feedback) - len(pending)}")

        print(f"\n  Model versions   : {len(versions)}")
        for v in versions[-5:]:   
            score = v.get('score', '?')
            scoreStr = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"    ↳ [{v['version_id']}] {v.get('model_name','?'):<26} "
                  f"{v.get('score_key','?')}={score_str}  "
                  f"trigger={v.get('trigger','?')}")

        if pending:
            print(f"\n  ⏳ Unapplied feedback:")
            for fb in pending[:5]:
                print(f"    [{fb['id']}] {fb.get('type','?'):12s} "
                      f"— {fb.get('detail','')[:50]}")
        print("═" * 55)

    

    @staticmethod
    def _load(path: str, default: dict) -> dict:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return default

    @staticmethod
    def _save(path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _makeId(prefix: str) -> str:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}_{ts}_{os.urandom(3).hex()}"