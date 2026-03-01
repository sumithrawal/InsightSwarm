"""
main.py — Analytics Agent CLI Entry Point
Phase 1: Load + Preprocess datasets
"""

import click
import json
import os
from pathlib import Path
from agent.loader import load_file, detect_column_types, get_dataset_profile, print_profile
from agent.preprocessor import Preprocessor

MEMORY_FILE = "feedback/memory.json"


def load_memory():
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

    # ── 1. Load ──────────────────────────────────────
    df = load_file(file)

    # ── 2. Profile ───────────────────────────────────
    col_types = detect_column_types(df)
    profile   = get_dataset_profile(df, file)
    print_profile(profile)

    # ── 3. Preprocess ────────────────────────────────
    if target and target not in df.columns:
        click.echo(f"\n⚠️  Target column '{target}' not found. Skipping target-aware steps.")
        target = None

    scale_method = None if scale == "none" else scale
    preprocessor = Preprocessor(col_types)
    df_clean = preprocessor.fit_transform(df, target_col=target, scale=scale_method)

    # ── 4. Save preprocessor state ───────────────────
    os.makedirs("models", exist_ok=True)
    preprocessor.save_state("models/preprocessor_state.json")

    # ── 5. Preview ───────────────────────────────────
    click.echo("\n🔍 Preview of processed data (first 5 rows):")
    click.echo(df_clean.head().to_string())

    # ── 6. Optionally save output ────────────────────
    if output:
        df_clean.to_csv(output, index=False)
        click.echo(f"\n💾 Saved preprocessed data → {output}")

    # ── 7. Log to memory ─────────────────────────────
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
        click.echo(f"  [{i+1}] {run['type'].upper()} | {p.get('file','?')} | "
                   f"{p.get('rows','?')} rows | {p.get('loaded_at','?')[:19]}")


if __name__ == "__main__":
    cli()