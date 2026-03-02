import click
import json
import os
from pathlib import Path
from agent.loader import load_file, detect_column_types, get_dataset_profile, print_profile
from agent.preprocessor import Preprocessor
from agent.analyzer import Analyzer

MEMORY_FILE = "feedback/memory.json"


def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {"runs": []}


def save_memory(memory: dict):
    os.makedirs("feedback", exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


@click.group()
def cli():
    """🤖 Analytics Agent — Your self-improving data analysis system."""
    pass


@cli.command()
@click.option("--file", "-f", required=True, help="Path to CSV or XLSX file")
@click.option("--target", "-t", default=None, help="Target column for prediction")
@click.option("--scale", "-s", default="standard",
              type=click.Choice(["standard", "minmax", "none"]),
              help="Scaling method for numeric features")
@click.option("--output", "-o", default=None,
              help="Save preprocessed data to this CSV path")
def load(file, target, scale, output):
    """📂 Load and preprocess a dataset."""
    df = load_file(file)
    col_types = detect_column_types(df)
    profile   = get_dataset_profile(df, file)
    print_profile(profile)

    if target and target not in df.columns:
        click.echo(f"\n⚠️  Target column '{target}' not found.")
        target = None

    scale_method = None if scale == "none" else scale
    preprocessor = Preprocessor(col_types)
    df_clean = preprocessor.fit_transform(df, target_col=target, scale=scale_method)

    os.makedirs("models", exist_ok=True)
    preprocessor.save_state("models/preprocessor_state.json")

    click.echo("\n🔍 Preview of processed data (first 5 rows):")
    click.echo(df_clean.head().to_string())

    if output:
        df_clean.to_csv(output, index=False)
        click.echo(f"\n💾 Saved preprocessed data → {output}")

    memory = load_memory()
    memory["runs"].append({
        "type": "load",
        "profile": profile,
        "target": target,
        "scale": scale,
        "processed_shape": list(df_clean.shape),
    })
    save_memory(memory)
    click.echo(f"\n📝 Run logged to {MEMORY_FILE}")


@cli.command()
@click.option("--file", "-f", required=True, help="Path to CSV or XLSX file")
@click.option("--target", "-t", default=None, help="Target column (improves analysis)")
@click.option("--output-dir", "-o", default="outputs", help="Directory to save charts")
def analyze(file, target, output_dir):
    """📊 Run full Exploratory Data Analysis and generate charts."""
    df = load_file(file)
    col_types = detect_column_types(df)
    profile   = get_dataset_profile(df, file)
    print_profile(profile)

    preprocessor = Preprocessor(col_types)
    df = preprocessor._parse_datetimes(df)

    if target and target not in df.columns:
        click.echo(f"\n⚠️  Target column '{target}' not found.")
        target = None

    analyzer = Analyzer(df, col_types, target_col=target, output_dir=output_dir)
    report   = analyzer.run_full_eda()

    click.echo("\n💡 KEY INSIGHTS")
    click.echo("─" * 40)
    if "skewed_columns" in report:
        click.echo(f"  📐 Skewed columns: {report['skewed_columns']}")
    if "missing_values" in report and report["missing_values"]:
        worst = max(report["missing_values"], key=report["missing_values"].get)
        click.echo(f"  🩹 Most missing: '{worst}' ({report['missing_values'][worst]} cells)")
    if "top_correlations" in report:
        click.echo("  🔗 Top correlations:")
        for pair, val in list(report["top_correlations"].items())[:3]:
            click.echo(f"     {pair} → {val}")
    if "outliers" in report and report["outliers"]:
        worst_out = max(report["outliers"], key=lambda c: report["outliers"][c]["pct"])
        click.echo(f"  ⚠️  Most outliers: '{worst_out}' ({report['outliers'][worst_out]['pct']}%)")
    if "target" in report:
        t = report["target"]
        click.echo(f"  🎯 Task: {t['task'].upper()} on '{t['column']}'")

    memory = load_memory()
    memory["runs"].append({
        "type": "analyze",
        "file": os.path.basename(file),
        "target": target,
        "output_dir": output_dir,
        "generated_at": report.get("generated_at"),
    })
    save_memory(memory)
    click.echo(f"\n📝 Run logged to {MEMORY_FILE}")
    click.echo(f"\n🖼️  Open your charts in: ./{output_dir}/")


@cli.command()
def history():
    """📜 Show history of previous runs."""
    memory = load_memory()
    runs = memory.get("runs", [])
    if not runs:
        click.echo("No runs recorded yet.")
        return
    click.echo(f"\n📜 {len(runs)} run(s) recorded:\n")
    for i, run in enumerate(runs):
        p = run.get("profile", {})
        ts = run.get("generated_at") or p.get("loaded_at", "?")
        click.echo(f"  [{i+1}] {run['type'].upper():10s} | "
                   f"{run.get('file') or p.get('file','?'):30s} | "
                   f"{str(ts)[:19]}")


if __name__ == "__main__":
    cli()