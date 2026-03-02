import click
import json
import os
from agent.loader import (
    loadFile, detectColumnTypes, getDatasetProfile,
    printProfile, showInfo, suggestTarget, promptTarget,
)
from agent.preprocessor import Preprocessor
from agent.analyzer import Analyzer

MEMORY_FILE = "feedback/memory.json"


def loadMemory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {"runs": []}


def saveMemory(memory: dict):
    os.makedirs("feedback", exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def _ingest(file: str) -> tuple:
    df         = loadFile(file)
    colTypes  = detectColumnTypes(df)
    profile    = getDatasetProfile(df, file)
    showInfo(df, colTypes)
    printProfile(profile)

    return df, colTypes, profile


@click.group()
def cli():
    pass
@cli.command()
@click.option("--file",   "-f", required=True, help="Path to CSV or XLSX file")
@click.option("--target", "-t", default=None,
              help="Target column name (leave blank to be prompted)")
@click.option("--scale",  "-s", default="standard",
              type=click.Choice(["standard", "minmax", "none"]),
              help="Scaling method for numeric features")
@click.option("--output", "-o", default=None,
              help="Save preprocessed data to this CSV path")
def load(file, target, scale, output):
    df, colTypes, profile = _ingest(file)
    if target:
        if target not in df.columns:
            click.echo(f"\n⚠️  Column '{target}' not found in this dataset.")
            click.echo(f"   Available columns: {list(df.columns)}")
            target = None
        else:
            click.echo(f"\n✅ Using provided target: '{target}'")

    if not target:
        suggested = suggestTarget(df, colTypes)
        target    = promptTarget(df, colTypes, suggested)
    scale_method = None if scale == "none" else scale
    preprocessor = Preprocessor(colTypes)
    df_clean     = preprocessor.fit_transform(df, target_col=target,
                                               scale=scale_method)
    os.makedirs("models", exist_ok=True)
    preprocessor.save_state("models/preprocessor_state.json")
    click.echo("\n🔍 Preview — processed data (first 5 rows):")
    click.echo(df_clean.head().to_string())
    if output:
        df_clean.to_csv(output, index=False)
        click.echo(f"\n💾 Saved preprocessed data → {output}")
    memory = loadMemory()
    memory["runs"].append({
        "type":            "load",
        "profile":         profile,
        "target":          target,
        "scale":           scale,
        "processed_shape": list(df_clean.shape),
    })
    saveMemory(memory)
    click.echo(f"\n📝 Run logged to {MEMORY_FILE}")
@cli.command()
@click.option("--file",       "-f", required=True, help="Path to CSV or XLSX file")
@click.option("--target",     "-t", default=None,
              help="Target column name (leave blank to be prompted)")
@click.option("--output-dir", "-o", default="outputs",
              help="Directory to save charts and report")
def analyze(file, target, output_dir):
    df, colTypes, profile = _ingest(file)
    if target:
        if target not in df.columns:
            click.echo(f"\n⚠️  Column '{target}' not found.")
            click.echo(f"   Available columns: {list(df.columns)}")
            target = None
        else:
            click.echo(f"\n✅ Using provided target: '{target}'")

    if not target:
        suggested = suggestTarget(df, colTypes)
        target    = promptTarget(df, colTypes, suggested)

    preprocessor = Preprocessor(colTypes)
    df = preprocessor._parse_datetimes(df)

    analyzer = Analyzer(df, colTypes, target_col=target, output_dir=output_dir)
    report   = analyzer.run_full_eda()
    click.echo("\n💡 KEY INSIGHTS")
    click.echo("─" * 50)

    if report.get("skewed_columns"):
        click.echo(f"  📐 Skewed (consider log transform): {report['skewed_columns']}")

    if report.get("missing_values"):
        worst = max(report["missing_values"], key=report["missing_values"].get)
        count = report["missing_values"][worst]
        pct   = round(count / len(df) * 100, 1)
        click.echo(f"  🩹 Most missing: '{worst}' — {count} cells ({pct}%)")

    if report.get("top_correlations"):
        click.echo("  🔗 Top correlations:")
        for pair, val in list(report["top_correlations"].items())[:3]:
            click.echo(f"     {pair}  →  {val:+.3f}")

    if report.get("outliers"):
        worst_out = max(report["outliers"], key=lambda c: report["outliers"][c]["pct"])
        click.echo(f"  ⚠️  Most outliers: '{worst_out}' "
                   f"({report['outliers'][worst_out]['pct']}% of rows)")

    if report.get("target"):
        t = report["target"]
        click.echo(f"  🎯 ML task detected: {t['task'].upper()} on '{t['column']}'")
    memory = loadMemory()
    memory["runs"].append({
        "type":         "analyze",
        "file":         os.path.basename(file),
        "target":       target,
        "output_dir":   output_dir,
        "generated_at": report.get("generated_at"),
    })
    saveMemory(memory)
    click.echo(f"\n📝 Run logged to {MEMORY_FILE}")
    click.echo(f"🖼️  Charts saved in: ./{output_dir}/")
@cli.command()
def history():
    memory = loadMemory()
    runs   = memory.get("runs", [])
    if not runs:
        click.echo("No runs recorded yet.")
        return

    click.echo(f"\n📜 {len(runs)} run(s) on record:\n")
    for i, run in enumerate(runs):
        p  = run.get("profile", {})
        ts = run.get("generated_at") or p.get("loaded_at", "?")
        fn = run.get("file") or p.get("file", "?")
        tgt = run.get("target") or "—"
        click.echo(f"  [{i+1}] {run['type'].upper():<10} {fn:<32} "
                   f"target={tgt:<20} {str(ts)[:19]}")


if __name__ == "__main__":
    cli()