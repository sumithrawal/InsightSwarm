"""
reporter.py — Phase 5: HTML Report Generator
Bundles all EDA charts, model results, memory summary, and insights
into a single self-contained HTML report file.
"""

import os
import json
import base64
from datetime import datetime


def _imgToB64(path: str) -> str:
    """Convert a PNG to base64 for embedding in HTML."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""


def _loadJson(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def generateReport(output_dir: str = "outputs",
                    model_dir:  str = "models",
                    feedback_dir: str = "feedback",
                    report_path: str = "outputs/report.html") -> str:
    """
    Generate a self-contained HTML report from all agent outputs.
    Returns the path to the generated report.
    """
    
    edaReport      = _loadJson(os.path.join(output_dir, "eda_report.json"))
    modelingReport = _loadJson(os.path.join(output_dir, "modeling_report.json"))
    memoryData     = _loadJson(os.path.join(feedback_dir, "memory.json"))
    versionsData   = _loadJson(os.path.join(feedback_dir, "model_versions.json"))
    meta            = _loadJson(os.path.join(model_dir, "best_model_meta.json"))
    feedbackData   = _loadJson(os.path.join(feedback_dir, "feedback_log.json"))

    
    chartFiles = [
        ("missing_values.png",          "Missing Values"),
        ("numeric_distributions.png",   "Numeric Distributions"),
        ("categorical_distributions.png","Categorical Distributions"),
        ("correlation_heatmap.png",      "Correlation Heatmap"),
        ("target_analysis.png",          "Target Analysis"),
        ("outliers_boxplot.png",         "Outlier Overview"),
        ("model_comparison.png",         "Model Comparison"),
        ("prediction_vs_actual.png",     "Actual vs Predicted"),
        ("feature_importance.png",       "Feature Importance"),
        ("confusion_matrix.png",         "Confusion Matrix"),
        ("improvement_history.png",      "Improvement History"),
    ]

    chartsHtml = ""
    for fname, title in chartFiles:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            b64 = _imgToB64(fpath)
            chartsHtml += f"""
            <div class="chart-card">
                <div class="chart-title">{title}</div>
                <img src="data:image/png;base64,{b64}" alt="{title}" />
            </div>"""

    
    
    edaRows = ""
    summary = edaReport.get("summary_stats", {})
    if summary:
        cols = list(next(iter(summary.values())).keys()) if summary else []
        edaRows = "<tr><th>Metric</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
        for metric, vals in summary.items():
            edaRows += "<tr><td>" + metric + "</td>"
            for c in cols:
                v = vals.get(c, "")
                edaRows += f"<td>{round(v,4) if isinstance(v,float) else v}</td>"
            edaRows += "</tr>"

    
    missingRows = ""
    for col, cnt in edaReport.get("missing_values", {}).items():
        pct = round(cnt / max(edaReport.get("summary_stats",{}).get("count",{}).get(col,1),1)*100,1)
        bar = "█" * int(pct * 2)
        missingRows += f"<tr><td>{col}</td><td>{cnt}</td><td>{pct}%  {bar}</td></tr>"

    
    outlierRows = ""
    for col, info in edaReport.get("outliers", {}).items():
        outlierRows += (f"<tr><td>{col}</td><td>{info['count']}</td>"
                         f"<td>{info['pct']}%</td>"
                         f"<td>{info['lower_bound']} → {info['upper_bound']}</td></tr>")

    
    modelRows = ""
    bestName = modelingReport.get("best_model", "")
    for name, info in modelingReport.get("results", {}).items():
        task    = modelingReport.get("task", "regression")
        score   = info.get("test_r2" if task == "regression" else "test_f1", "")
        cv      = info.get("cv_r2"   if task == "regression" else "cv_f1", "")
        aux     = info.get("mae", info.get("accuracy", ""))
        badge   = " " if name == bestName else ""
        bold    = "font-weight:bold; color:#ffd700;" if name == bestName else ""
        modelRows += (f"<tr style='{bold}'><td>{name}{badge}</td>"
                       f"<td>{cv}</td><td>{score}</td><td>{aux}</td></tr>")

    
    versionRows = ""
    for v in versionsData.get("versions", []):
        promoted = "✅ Yes" if v.get("promoted", True) else "❌ No"
        score    = round(v.get("score",0), 4)
        prev     = round(v.get("prev_score",0), 4)
        delta    = round(score - prev, 4) if prev else "—"
        versionRows += (f"<tr><td>{v.get('version_id','')}</td>"
                         f"<td>{v.get('model_name','')}</td>"
                         f"<td>{score}</td><td>{prev}</td>"
                         f"<td>{delta}</td><td>{promoted}</td>"
                         f"<td>{v.get('trigger','')}</td>"
                         f"<td>{str(v.get('trained_at',''))[:19]}</td></tr>")

    
    feedbackRows = ""
    for fb in feedbackData.get("entries", []):
        applied = "✅" if fb.get("applied") else "⏳"
        detail  = str(fb.get("detail", ""))[:60]
        feedbackRows += (f"<tr><td>{fb.get('id','')}</td>"
                          f"<td>{fb.get('type','')}</td>"
                          f"<td>{detail}</td>"
                          f"<td>{applied}</td>"
                          f"<td>{str(fb.get('created_at',''))[:19]}</td></tr>")

    
    runRows = ""
    for r in memoryData.get("runs", []):
        ts  = str(r.get("logged_at", r.get("generated_at","?")))[:19]
        runRows += (f"<tr><td>{r.get('type','').upper()}</td>"
                     f"<td>{r.get('file','')}</td>"
                     f"<td>{r.get('target','—')}</td>"
                     f"<td>{ts}</td></tr>")

    
    task         = modelingReport.get("task", "—")
    target_col   = meta.get("target_col", "—")
    bestScore   = round(meta.get("score", 0), 4)
    scoreLabel  = "R²" if task == "regression" else "F1"
    logNote     = " (log-transformed)" if meta.get("log_target") else ""
    nVersions   = len(versionsData.get("versions", []))
    nFeedback   = len(feedbackData.get("entries", []))
    nPending    = len([f for f in feedbackData.get("entries",[]) if not f.get("applied")])
    skewed       = ", ".join(edaReport.get("skewed_columns", []))
    generatedAt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analytics Agent Report</title>
<style>
  :root {{
    --bg:        #0f1117;
    --surface:   #1a1d27;
    --surface2:  #22263a;
    --border:    #2e3245;
    --text:      #c8ccd8;
    --muted:     #7a7d90;
    --accent:    #7c83fd;
    --gold:      #ffd700;
    --green:     #7cfdba;
    --red:       #fd7c7c;
    --radius:    10px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
  }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(135deg, #1a1d27 0%, #12152a 100%);
    border-bottom: 1px solid var(--border);
    padding: 36px 48px 28px;
  }}
  .header h1 {{
    font-size: 28px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
  }}
  .header h1 span {{ color: var(--accent); }}
  .header p {{ color: var(--muted); margin-top: 6px; font-size: 13px; }}
  .badge {{
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    color: var(--accent);
    margin-right: 8px;
    margin-top: 10px;
  }}

  /* ── Layout ── */
  .container {{ max-width: 1300px; margin: 0 auto; padding: 32px 48px 64px; }}

  /* ── Section ── */
  .section {{ margin-bottom: 48px; }}
  .section-title {{
    font-size: 18px;
    font-weight: 600;
    color: #fff;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--accent);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  /* ── Metric Cards ── */
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
  }}
  .metric-card .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }}
  .metric-card .value {{ font-size: 26px; font-weight: 700; color: var(--accent); margin: 6px 0 4px; }}
  .metric-card .sub   {{ font-size: 11px; color: var(--muted); }}

  /* ── Charts ── */
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 20px;
  }}
  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }}
  .chart-title {{
    padding: 12px 16px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }}
  .chart-card img {{ width: 100%; display: block; }}

  /* ── Tables ── */
  .table-wrap {{ overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead th {{
    background: var(--surface2);
    color: var(--accent);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  tbody tr {{ border-bottom: 1px solid var(--border); }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: var(--surface2); }}
  tbody td {{ padding: 9px 14px; color: var(--text); font-size: 13px; }}

  /* ── Insight Pills ── */
  .insights {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 24px; }}
  .insight-pill {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 13px;
    flex: 1;
    min-width: 240px;
  }}
  .insight-pill .icon {{ margin-right: 6px; }}
  .insight-pill strong {{ color: #fff; }}

  /* ── Footer ── */
  .footer {{
    text-align: center;
    padding: 24px;
    color: var(--muted);
    font-size: 12px;
    border-top: 1px solid var(--border);
    margin-top: 48px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1> <span>Analytics Agent</span> — Full Report</h1>
  <p>Generated: {generatedAt}</p>
  <span class="badge">Target: {target_col}{logNote}</span>
  <span class="badge">Task: {task.upper()}</span>
  <span class="badge">Best: {bestName}</span>
  <span class="badge">{scoreLabel}: {bestScore}</span>
</div>

<div class="container">

  <!-- ── KEY METRICS ── -->
  <div class="section">
    <div class="section-title"> Key Metrics</div>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="label">ML Task</div>
        <div class="value" style="font-size:18px">{task.upper()}</div>
        <div class="sub">Target: {target_col}</div>
      </div>
      <div class="metric-card">
        <div class="label">Best Model</div>
        <div class="value" style="font-size:16px">{bestName}</div>
        <div class="sub">Champion</div>
      </div>
      <div class="metric-card">
        <div class="label">{scoreLabel} Score</div>
        <div class="value">{bestScore}</div>
        <div class="sub">Test set</div>
      </div>
      <div class="metric-card">
        <div class="label">Retrain Cycles</div>
        <div class="value">{nVersions}</div>
        <div class="sub">Model versions</div>
      </div>
      <div class="metric-card">
        <div class="label">Feedback</div>
        <div class="value">{nFeedback}</div>
        <div class="sub">{nPending} pending</div>
      </div>
      <div class="metric-card">
        <div class="label">Skewed Columns</div>
        <div class="value">{len(edaReport.get("skewed_columns",[]))}</div>
        <div class="sub">{skewed or "None"}</div>
      </div>
    </div>
  </div>

  <!-- ── EDA INSIGHTS ── -->
  <div class="section">
    <div class="section-title"> EDA Insights</div>
    <div class="insights">
      {"".join(f'<div class="insight-pill"><span class="icon"></span><strong>Skewed:</strong> {c}</div>' for c in edaReport.get("skewed_columns",[]))}
      {"".join(f'<div class="insight-pill"><span class="icon"></span><strong>{c}:</strong> {n} missing</div>' for c,n in list(edaReport.get("missing_values",{}).items())[:4])}
      {"".join(f'<div class="insight-pill"><span class="icon">⚠️</span><strong>{c}:</strong> {i["pct"]}% outliers</div>' for c,i in edaReport.get("outliers",{}).items())}
      {"".join(f'<div class="insight-pill"><span class="icon"></span><strong>{k}:</strong> {v:+.3f}</div>' for k,v in list(edaReport.get("top_correlations",{}).items())[:3])}
    </div>
  </div>

  <!-- ── CHARTS ── -->
  <div class="section">
    <div class="section-title">️ Charts</div>
    <div class="charts-grid">
      {chartsHtml}
    </div>
  </div>

  <!-- ── SUMMARY STATS ── -->
  {"" if not edaRows else f'''
  <div class="section">
    <div class="section-title"> Summary Statistics</div>
    <div class="table-wrap">
      <table><thead>{edaRows[:edaRows.index("</tr>")+5]}</thead>
      <tbody>{edaRows[edaRows.index("</tr>")+5:]}</tbody></table>
    </div>
  </div>'''}

  <!-- ── MISSING VALUES ── -->
  {"" if not missingRows else f'''
  <div class="section">
    <div class="section-title"> Missing Values</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Column</th><th>Count</th><th>% Missing</th></tr></thead>
        <tbody>{missingRows}</tbody>
      </table>
    </div>
  </div>'''}

  <!-- ── OUTLIERS ── -->
  {"" if not outlierRows else f'''
  <div class="section">
    <div class="section-title">⚠️ Outliers (IQR Method)</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Column</th><th>Count</th><th>%</th><th>IQR Bounds</th></tr></thead>
        <tbody>{outlierRows}</tbody>
      </table>
    </div>
  </div>'''}

  <!-- ── MODEL RESULTS ── -->
  {"" if not modelRows else f'''
  <div class="section">
    <div class="section-title"> Model Results</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>CV Score</th><th>Test Score</th><th>MAE / Accuracy</th></tr></thead>
        <tbody>{modelRows}</tbody>
      </table>
    </div>
  </div>'''}

  <!-- ── VERSION HISTORY ── -->
  {"" if not versionRows else f'''
  <div class="section">
    <div class="section-title"> Model Version History</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Version</th><th>Model</th><th>Score</th><th>Prev Score</th><th>Δ</th><th>Promoted</th><th>Trigger</th><th>Trained</th></tr></thead>
        <tbody>{versionRows}</tbody>
      </table>
    </div>
  </div>'''}

  <!-- ── FEEDBACK LOG ── -->
  {"" if not feedbackRows else f'''
  <div class="section">
    <div class="section-title">✏️ Feedback Log</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>ID</th><th>Type</th><th>Detail</th><th>Applied</th><th>Created</th></tr></thead>
        <tbody>{feedbackRows}</tbody>
      </table>
    </div>
  </div>'''}

  <!-- ── RUN HISTORY ── -->
  {"" if not runRows else f'''
  <div class="section">
    <div class="section-title"> Run History</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Type</th><th>File</th><th>Target</th><th>Timestamp</th></tr></thead>
        <tbody>{runRows}</tbody>
      </table>
    </div>
  </div>'''}

</div>

<div class="footer">
  Analytics Agent — Phase 5 Report &nbsp;|&nbsp;
  Generated {generatedAt} &nbsp;|&nbsp;
  Target: <strong>{target_col}</strong> &nbsp;|&nbsp;
  Best Model: <strong>{bestName}</strong> ({scoreLabel}={bestScore})
</div>

</body>
</html>"""

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path