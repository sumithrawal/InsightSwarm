import click, json, os, sys, traceback
import pandas as pd
from datetime import datetime

def _c(text, fg=None, bold=False):
    return click.style(str(text), fg=fg, bold=bold)

def _header(title):
    click.echo()
    click.echo(_c("═" * 55, fg="bright_black"))
    click.echo(_c(f"{title}", fg="cyan", bold=True))
    click.echo(_c("═" * 55, fg="bright_black"))

def _ok(msg):   click.echo(_c(f"{msg}", fg="green"))
def _warn(msg): click.echo(_c(f"{msg}", fg="yellow"))
def _err(msg):  click.echo(_c(f"{msg}", fg="red"), err=True)
def _info(msg): click.echo(_c(f"{msg}", fg="bright_black"))
def _bullet(label, value, color="cyan"):
    click.echo(f"{_c(label+':', bold=True)} {_c(value, fg=color)}")

VERSION = "1.0.0"


def _imports():
    from agent.loader import (
        loadFile, detectColumnTypes, getDatasetProfile,
        printProfile, showInfo, suggestTarget, promptTarget,
    )
    from agent.preprocessor import Preprocessor
    from agent.analyzer     import Analyzer
    from agent.predictor    import Predictor
    from agent.memory       import Memory
    from agent.retrainer    import Retrainer
    from agent.reporter     import generateReport
    return (loadFile, detectColumnTypes, getDatasetProfile,
            printProfile, showInfo, suggestTarget, promptTarget,
            Preprocessor, Analyzer, Predictor, Memory, Retrainer,
            generateReport)

MEM = None
def _mem():
    global MEM
    if MEM is None:
        from agent.memory import Memory
        MEM = Memory()
    return MEM


def _ingest(file):
    (loadFile, detectColumnTypes, getDatasetProfile,
     printProfile, showInfo, *_) = _imports()
    df        = loadFile(file)
    colTypes = detectColumnTypes(df)
    profile   = getDatasetProfile(df, file)
    showInfo(df, colTypes)
    printProfile(profile)
    return df, colTypes, profile


def _resolveTarget(df, colTypes, target):
    _, _, _, _, _, suggestTarget, promptTarget, *_ = _imports()
    if target:
        if target not in df.columns:
            _warn(f"Column '{target}' not found.")
            _info(f"Available: {list(df.columns)}")
            target = None
        else:
            _ok(f"Using target: '{target}'")
    if not target:
        target = promptTarget(df, colTypes, suggestTarget(df, colTypes))
    return target


def _safeRun(fn):
    """Wrap command body in clean error handling."""
    try:
        fn()
    except FileNotFoundError as e:
        _err(f"File not found: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo()
        _warn("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        _err(f"Unexpected error: {e}")
        click.echo(_c(traceback.format_exc(), fg="bright_black"), err=True)
        sys.exit(1)



@click.group()
@click.version_option(VERSION, "-V", "--version",
                      prog_name="Analytics Agent",
                      message="%(prog)s v%(version)s")
def cli():
    """
    \b
      Analytics Agent v1.0.0
    Self-improving data analysis system.
    """
    pass



@cli.command()
@click.option("--file",   "-f", required=True,  help="Path to CSV or XLSX file")
@click.option("--target", "-t", default=None,   help="Target column (prompted if blank)")
@click.option("--scale",  "-s", default="standard",
              type=click.Choice(["standard", "minmax", "none"]),
              help="Scaling method  [default: standard]")
@click.option("--output", "-o", default=None,   help="Save cleaned data to CSV")
def load(file, target, scale, output):
    """  Load, inspect, and preprocess a dataset."""
    def _run():
        _, _, _, _, _, _, _, Preprocessor, *_ = _imports()
        _header("LOAD & PREPROCESS")
        df, colTypes, profile = _ingest(file)
        target = _resolveTarget(df, colTypes, target)

        preprocessor = Preprocessor(colTypes)
        dfClean = preprocessor.fitTransform(
            df, target_col=target, scale=None if scale == "none" else scale
        )
        os.makedirs("models", exist_ok=True)
        preprocessor.saveState("models/preprocessor_state.json")

        click.echo()
        click.echo(_c("   Preview (first 5 rows):", bold=True))
        click.echo(dfClean.head().to_string())

        if output:
            dfClean.to_csv(output, index=False)
            _ok(f"Saved cleaned data → {output}")

        _mem().logRun({"type": "load", "file": os.path.basename(file),
                        "target": target, "scale": scale,
                        "processed_shape": list(dfClean.shape)})
        _ok("Logged to memory")
    _safeRun(_run)



@cli.command()
@click.option("--file",       "-f", required=True, help="Path to CSV or XLSX file")
@click.option("--target",     "-t", default=None,  help="Target column (prompted if blank)")
@click.option("--output-dir", "-o", default="outputs", help="Directory for charts  [default: outputs]")
def analyze(file, target, output_dir):
    """  Run full EDA — inspect columns, generate charts + report."""
    def _run():
        _, _, _, _, _, _, _, Preprocessor, Analyzer, *_ = _imports()
        _header("EXPLORATORY DATA ANALYSIS")
        df, colTypes, _ = _ingest(file)
        target = _resolveTarget(df, colTypes, target)

        preprocessor = Preprocessor(colTypes)
        df = preprocessor._parseDatetimes(df)

        analyzer = Analyzer(df, colTypes, target_col=target, output_dir=output_dir)
        report   = analyzer.runFullEda()

        click.echo()
        click.echo(_c("   KEY INSIGHTS", bold=True, fg="yellow"))
        click.echo(_c("  " + "─"*46, fg="bright_black"))

        if report.get("skewed_columns"):
            click.echo(f"  {_c(' Skewed:', bold=True)} "
                       f"{_c(str(report['skewed_columns']), fg='yellow')}")
        if report.get("missing_values"):
            worst = max(report["missing_values"], key=report["missing_values"].get)
            cnt   = report["missing_values"][worst]
            click.echo(f"  {_c(' Most missing:', bold=True)} "
                       f"{_c(worst, fg='yellow')} ({cnt} cells)")
        if report.get("top_correlations"):
            click.echo(f"  {_c(' Top correlations:', bold=True)}")
            for pair, val in list(report["top_correlations"].items())[:3]:
                color = "green" if val > 0 else "red"
                click.echo(f"      {pair}  →  {_c(f'{val:+.3f}', fg=color)}")
        if report.get("outliers"):
            wo  = max(report["outliers"], key=lambda c: report["outliers"][c]["pct"])
            pct = report["outliers"][wo]["pct"]
            click.echo(f"  {_c('⚠️  Most outliers:', bold=True)} "
                       f"{_c(wo, fg='yellow')} ({pct}%)")
        if report.get("target"):
            t = report["target"]
            click.echo(f"  {_c(' Task:', bold=True)} "
                       f"{_c(t['task'].upper(), fg='cyan', bold=True)} "
                       f"on '{_c(t['column'], fg='green')}'")

        _mem().logRun({"type": "analyze", "file": os.path.basename(file),
                        "target": target, "output_dir": output_dir,
                        "generated_at": report.get("generated_at")})
        click.echo()
        _ok(f"Logged  |  Charts saved in: ./{output_dir}/")
    _safeRun(_run)



@cli.command()
@click.option("--file",       "-f", required=True, help="Path to CSV or XLSX file")
@click.option("--target",     "-t", default=None,  help="Target column (prompted if blank)")
@click.option("--model-dir",  "-m", default="models",  help="Directory to save model  [default: models]")
@click.option("--output-dir", "-o", default="outputs", help="Directory for charts  [default: outputs]")
def train(file, target, model_dir, output_dir):
    """  Train & compare ML models, save the best one."""
    def _run():
        *_, Predictor, Memory, Retrainer, generateReport = _imports()
        _header("PREDICTIVE MODELING")
        df, colTypes, _ = _ingest(file)
        target = _resolveTarget(df, colTypes, target)
        if not target:
            _err("No target selected. Cannot train.")
            return

        predictor = Predictor(df, colTypes, target,
                              model_dir=model_dir, output_dir=output_dir)
        results = predictor.run()

        sk    = "test_r2" if predictor.task == "regression" else "test_f1"
        score = results.get(predictor.bestName, {}).get(sk)

        click.echo()
        _bullet("Best model", predictor.bestName, "cyan")
        _bullet("Score",
                f"{sk.upper()} = {score:.4f}" if score else "—",
                "green")

        _mem().logModelVersion({
            "model_name": predictor.bestName,
            "task":       predictor.task,
            "target_col": target,
            "score_key":  sk,
            "score":      score,
            "trigger":    "initial",
            "trained_at": datetime.now().isoformat(),
        })
        _mem().logRun({"type": "train", "file": os.path.basename(file),
                        "target": target, "task": predictor.task,
                        "best_model": predictor.bestName, "score": score})
        _ok(f"Logged  |  Charts: ./{output_dir}/")
    _safeRun(_run)



@cli.command()
@click.option("--file",      "-f", required=True, help="CSV/XLSX with new rows to score")
@click.option("--model-dir", "-m", default="models", help="Directory with saved model  [default: models]")
@click.option("--output",    "-o", default="predictions.csv", help="Output CSV path  [default: predictions.csv]")
def predict(file, model_dir, output):
    """  Score new data using the saved best model."""
    def _run():
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        loadFile, detectColumnTypes, _, _, showInfo, *_ = _imports()
        *_, Predictor, Memory, Retrainer, generateReport = _imports()

        _header("PREDICT")
        df        = loadFile(file)
        colTypes = detectColumnTypes(df)
        showInfo(df, colTypes)

        dfEnc = df.copy()
        for col in dfEnc.columns:
            dtypeStr = str(dfEnc[col].dtype)
            if dfEnc[col].dtype.kind == "O" or dtypeStr in ("str", "string"):
                dfEnc[col] = dfEnc[col].fillna("__missing__")
                le = LabelEncoder()
                dfEnc[col] = le.fitTransform(dfEnc[col].astype(str))
            else:
                dfEnc[col] = pd.to_numeric(dfEnc[col], errors="coerce").fillna(0)

        preds, meta = Predictor.loadAndPredict(model_dir, dfEnc)
        tc = meta["target_col"]
        df[f"predicted_{tc}"] = preds
        df.to_csv(output, index=False)

        click.echo()
        _bullet("Model used",   meta["model_name"])
        _bullet("Target",       tc)
        _bullet("Rows scored",  f"{len(preds):,}", "green")
        _bullet("Saved to",     output)
        click.echo()
        click.echo(_c("  Preview:", bold=True))
        click.echo(df[[f"predicted_{tc}"]].head().to_string())

        _mem().logRun({"type": "predict", "file": os.path.basename(file),
                        "model": meta["model_name"], "target": tc,
                        "rows": len(preds), "output": output})
    _safeRun(_run)



@cli.command()
@click.option("--type", "-t", "fb_type",
              type=click.Choice(["correction", "feature_hint", "new_data", "label"]),
              required=True, help="Type of feedback")
@click.option("--target", default=None, help="Target column this feedback relates to")
@click.option("--detail", default="",  help="Human-readable explanation")
@click.option("--data",   default=None,
              help=(
                "JSON payload:\n\n"
                "  correction   → '[{\"sku\":\"AN201-RED-L\",\"correct_value\":12}]'\n"
                "  feature_hint → '{\"Category\":{\"LEHENGA CHOLI\":3}}'\n"
                "  new_data     → '{\"files\":[\"extra.csv\"]}'"
              ))
def feedback(fb_type, target, detail, data):
    """✏️   Submit a correction, feature hint, or new-data signal."""
    def _run():
        _header("FEEDBACK")
        parsed = None
        if data:
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as e:
                _err(f"Cannot parse --data JSON: {e}")
                return

        fid = _mem().addFeedback({
            "type":       fb_type,
            "target_col": target,
            "detail":     detail,
            "data":       parsed,
        })

        _ok(f"Feedback recorded  [id: {_c(fid, fg='cyan')}]")
        _bullet("Type",   fb_type)
        _bullet("Target", target or "(not specified)")
        _bullet("Detail", detail  or "(none)")
        pending = _mem().pendingFeedbackCount()
        click.echo()
        _info(f"{pending} feedback item(s) pending — run "
              f"{_c('retrain', fg='cyan', bold=True)} to apply.")
    _safeRun(_run)



@cli.command()
@click.option("--file",       "-f", required=True,
              help="Original training file (same as used for train)")
@click.option("--target",     "-t", default=None,
              help="Target column (prompted if blank)")
@click.option("--new-data",   "-n", multiple=True,
              help="Extra CSV/XLSX files to merge in (repeatable)")
@click.option("--model-dir",  "-m", default="models")
@click.option("--output-dir", "-o", default="outputs")
def retrain(file, target, new_data, model_dir, output_dir):
    """  Apply feedback + new data, retrain, promote if better."""
    def _run():
        *_, Retrainer, generateReport = _imports()
        _header("SELF-IMPROVEMENT / RETRAIN")
        df, colTypes, _ = _ingest(file)
        target = _resolveTarget(df, colTypes, target)
        if not target:
            _err("No target selected.")
            return

        pending = _mem().getFeedback(applied=False)
        click.echo()
        click.echo(_c(f"   Pending feedback: {len(pending)} item(s)", bold=True))
        for fb in pending:
            click.echo(f"    {_c('['+fb['id']+']', fg='bright_black')} "
                       f"{_c(fb.get('type','?'), fg='cyan'):12s}  "
                       f"{fb.get('detail','')[:60]}")

        featureWeights = {}
        for fb in pending:
            if fb.get("type") == "feature_hint" and isinstance(fb.get("data"), dict):
                featureWeights.update(fb["data"])

        retrainer = Retrainer(
            originalFile=file, target_col=target,
            colTypes=colTypes, memory=_mem(),
            model_dir=model_dir, output_dir=output_dir,
        )
        outcome = retrainer.run(
            newDataFiles  = list(new_data) if new_data else None,
            featureWeights = featureWeights or None,
        )

        click.echo()
        click.echo(_c("  " + "═"*46, fg="bright_black"))
        if outcome.get("promoted"):
            click.echo(_c("   NEW CHAMPION PROMOTED!", fg="green", bold=True))
            _bullet("Model",       outcome.get("best_model"), "cyan")
            _bullet("New score",   f"{outcome.get('new_score',0):.4f}", "green")
            _bullet("Improvement", f"{outcome.get('improvement',0):+.4f}", "green")
        else:
            click.echo(_c("    Champion retained", fg="yellow", bold=True))
            _bullet("Challenger score", f"{outcome.get('new_score',0):.4f}", "yellow")
            _bullet("Champion score",   f"{outcome.get('prev_score',0):.4f}", "cyan")
            _bullet("Δ Change",         f"{outcome.get('improvement',0):+.4f}", "red")
        click.echo(_c("  " + "═"*46, fg="bright_black"))

        _mem().logRun({"type": "retrain", "file": os.path.basename(file),
                        "target": target, "promoted": outcome.get("promoted"),
                        "improvement": outcome.get("improvement"),
                        "new_score": outcome.get("new_score")})
        _ok(f"Logged  |  Charts: ./{output_dir}/")
    _safeRun(_run)



@cli.command()
@click.option("--output-dir",    "-o", default="outputs",  help="Where charts live  [default: outputs]")
@click.option("--model-dir",     "-m", default="models",   help="Where models live  [default: models]")
@click.option("--feedback-dir",  "-f", default="feedback", help="Where feedback lives [default: feedback]")
@click.option("--report-path",   "-r", default="outputs/report.html",
              help="Output HTML path  [default: outputs/report.html]")
def report(output_dir, model_dir, feedback_dir, report_path):
    """  Generate a full self-contained HTML report."""
    def _run():
        *_, generateReport = _imports()
        _header("GENERATE HTML REPORT")
        _info("Bundling all charts, stats, model results and history...")

        path = generateReport(
            output_dir   = output_dir,
            model_dir    = model_dir,
            feedback_dir = feedback_dir,
            report_path  = report_path,
        )
        click.echo()
        _ok(f"Report generated → {_c(path, fg='cyan', bold=True)}")
        _info("Open in any browser — all charts are embedded, no internet needed.")

        _mem().logRun({"type": "report", "report_path": path})
    _safeRun(_run)



@cli.command()
def memory():
    """  Show agent memory — runs, feedback, model versions."""
    _mem().printSummary()



@cli.command()
@click.option("--type", "-t", "run_type", default=None,
              type=click.Choice(["load","analyze","train","predict",
                                 "feedback","retrain","report"]),
              help="Filter by run type")
def history(run_type):
    """  Show all previous runs (optionally filter by type)."""
    runs = _mem().getRuns(run_type)
    if not runs:
        _warn("No runs recorded yet.")
        return

    click.echo()
    header = (f"  {'#':<4} {_c('Type',-10):<18} "
              f"{_c('File',-30):<38} "
              f"{_c('Target',-18):<26} "
              f"{_c('Timestamp')}")
    click.echo(_c(f"   {len(runs)} run(s):", bold=True))
    click.echo(_c("  " + "─"*72, fg="bright_black"))

    for i, run in enumerate(runs):
        ts  = str(run.get("logged_at") or run.get("trained_at") or
                  run.get("generated_at", "?"))[:19]
        fn  = run.get("file", "?")
        tgt = run.get("target") or "—"
        rtype = run.get("type","?").upper()

        typeColor = {
            "TRAIN":"cyan","RETRAIN":"green","ANALYZE":"yellow",
            "LOAD":"white","PREDICT":"magenta","REPORT":"blue",
            "FEEDBACK":"bright_black",
        }.get(rtype, "white")

        click.echo(
            f"  {_c(str(i+1)+'.',fg='bright_black'):<6}"
            f"{_c(rtype, fg=type_color, bold=True):<18}"
            f"{fn:<30}  "
            f"target={_c(tgt, fg='cyan'):<20}  "
            f"{_c(ts, fg='bright_black')}"
        )



if __name__ == "__main__":
    cli()