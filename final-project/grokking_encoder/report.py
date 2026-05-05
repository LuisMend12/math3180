from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_report(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    logs = {}
    for path in sorted((run_dir / "logs").glob("*.jsonl")) if (run_dir / "logs").exists() else []:
        logs[path.stem] = read_jsonl(path)

    plot_paths = []
    for phase, rows in logs.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        fig, ax1 = plt.subplots(figsize=(9, 5))
        if "train_loss" in df:
            ax1.plot(df["epoch"], df["train_loss"], label="train_loss")
        if "val_loss" in df:
            ax1.plot(df["epoch"], df["val_loss"], label="val_loss")
        if "mlm_loss" in df:
            ax1.plot(df["epoch"], df["mlm_loss"], label="mlm_loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        if "val_accuracy" in df:
            ax2.plot(df["epoch"], df["val_accuracy"], color="tab:green", label="val_accuracy")
        if "val_macro_f1" in df:
            ax2.plot(df["epoch"], df["val_macro_f1"], color="tab:orange", label="val_macro_f1")
        ax2.set_ylabel("score")
        ax2.legend(loc="lower right")
        fig.tight_layout()
        out = report_dir / f"{phase}_curves.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        plot_paths.append(out)

    benchmark = {}
    for name in ["benchmark_summary.json", "benchmark_torch.json", "benchmark_mlx.json"]:
        path = run_dir / name
        if path.exists():
            benchmark[name] = json.loads(path.read_text())

    evals = {}
    eval_dir = run_dir / "eval"
    if eval_dir.exists():
        for path in sorted(eval_dir.glob("*.json")):
            evals[path.stem] = json.loads(path.read_text())

    lines = [
        "# Tiny MathNet Grokking Encoder Report",
        "",
        "## Run Summary",
        f"- Run directory: `{run_dir}`",
        f"- Logged phases: {', '.join(logs) if logs else 'none'}",
        f"- Evaluation files: {len(evals)}",
        "",
        "## Benchmark",
        "```json",
        json.dumps(benchmark, indent=2, sort_keys=True)[:12000],
        "```",
        "",
        "## Evaluation",
        "```json",
        json.dumps(evals, indent=2, sort_keys=True)[:12000],
        "```",
        "",
        "## Curves",
    ]
    for path in plot_paths:
        lines.append(f"![{path.name}]({path.name})")
    md_path = report_dir / "report.md"
    md_path.write_text("\n".join(lines))

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in lines]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path = report_dir / "report.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2))
    return {"report_md": str(md_path), "report_ipynb": str(nb_path), "plots": [str(p) for p in plot_paths]}
