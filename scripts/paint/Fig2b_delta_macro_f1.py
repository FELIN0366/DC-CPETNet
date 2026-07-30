# -*- coding: utf-8 -*-
"""
Draw the Fig. 2b task-wise delta Macro-F1 forest plot.

Input workbook:
    RESULT2_Table_Fig2A_Fig2B_completed(final).xlsx
Sheet:
    Fig2B_data

The orange series is computed row-wise as the best non-proposed MTL baseline
relative to the single-task model. The blue series is the proposed method
relative to the single-task model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Mandatory publication/export rules: keep text editable in SVG/PDF.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

DEFAULT_EXCEL_PATH = Path(
    "xx_path"
    + "\u8bba\u6587"
    + "/tables/RESULT2_Table_Fig2A_Fig2B_completed(final).xlsx"
)
SHEET_NAME = "Fig2B_data"
DEFAULT_OUTPUT_BASENAME = "Fig2b_delta_macro_f1"

BASELINE_MODELS = ("Shared Bottom", "MMOE", "CGC", "ADATT")
OURS_MODEL = "Our method"

ORANGE = "#ff7f0e"
BLUE = "#1f77b4"
GRID_COLOR = "#e8e8e8"
AXIS_COLOR = "#222222"


def find_delta_columns(df: pd.DataFrame, model: str) -> tuple[str, str]:
    """Return the mean and SD delta columns for one model family."""
    mean_cols = [
        col
        for col in df.columns
        if model in str(col) and "Single" in str(col) and "mean" in str(col).lower()
    ]
    sd_cols = [
        col
        for col in df.columns
        if model in str(col) and "Single" in str(col) and "sd" in str(col).lower()
    ]
    if len(mean_cols) != 1 or len(sd_cols) != 1:
        raise ValueError(
            f"Expected exactly one mean and one SD column for {model}; "
            f"found mean={mean_cols}, sd={sd_cols}"
        )
    return mean_cols[0], sd_cols[0]


def clean_task_label(value: object) -> str:
    task = str(value).strip()
    if task.lower().startswith("mean across"):
        return "Mean across\nt1-t5"
    return task


def build_plot_data(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if df.empty:
        raise ValueError(f"No rows found in sheet {sheet_name!r}")

    task_col = df.columns[0]
    series_cols = {
        model: find_delta_columns(df, model)
        for model in (*BASELINE_MODELS, OURS_MODEL)
    }

    rows = []
    for _, row in df.iterrows():
        baseline_values = {
            model: float(row[series_cols[model][0]])
            for model in BASELINE_MODELS
        }
        best_model = max(baseline_values, key=baseline_values.get)
        best_mean_col, best_sd_col = series_cols[best_model]
        ours_mean_col, ours_sd_col = series_cols[OURS_MODEL]

        rows.append(
            {
                "task": clean_task_label(row[task_col]),
                "best_baseline": best_model,
                "best_baseline_delta": float(row[best_mean_col]),
                "best_baseline_sd": float(row[best_sd_col]),
                "our_method_delta": float(row[ours_mean_col]),
                "our_method_sd": float(row[ours_sd_col]),
            }
        )

    return pd.DataFrame(rows)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.2,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
            "axes.spines.right": True,
            "axes.spines.top": True,
        }
    )


def nice_xlim(plot_df: pd.DataFrame) -> tuple[float, float]:
    lows = np.r_[
        plot_df["best_baseline_delta"] - plot_df["best_baseline_sd"],
        plot_df["our_method_delta"] - plot_df["our_method_sd"],
        0,
    ]
    highs = np.r_[
        plot_df["best_baseline_delta"] + plot_df["best_baseline_sd"],
        plot_df["our_method_delta"] + plot_df["our_method_sd"],
        0,
    ]
    lo = np.floor((float(np.nanmin(lows)) - 0.02) / 0.05) * 0.05
    hi = np.ceil((float(np.nanmax(highs)) + 0.02) / 0.05) * 0.05
    return lo, hi


def annotate_paired_points(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    y_best: np.ndarray,
    y_ours: np.ndarray,
) -> None:
    """Annotate paired deltas, avoiding label collisions when two points are close."""
    for row, y_b, y_o in zip(plot_df.itertuples(index=False), y_best, y_ours):
        best_x = float(row.best_baseline_delta)
        ours_x = float(row.our_method_delta)
        close_pair = abs(best_x - ours_x) < 0.014

        ax.text(
            best_x,
            float(y_b) + 0.085,
            f"{best_x:+.3f}",
            color=ORANGE,
            fontsize=7.5,
            ha="center",
            va="bottom",
            clip_on=False,
        )
        ax.text(
            ours_x,
            float(y_o) + (-0.085 if close_pair else 0.085),
            f"{ours_x:+.3f}",
            color=BLUE,
            fontsize=7.5,
            ha="center",
            va="top" if close_pair else "bottom",
            clip_on=False,
        )


def draw_delta_plot(plot_df: pd.DataFrame) -> plt.Figure:
    set_publication_style()

    fig, ax = plt.subplots(figsize=(6.7, 3.95), constrained_layout=True)
    base_y = np.arange(len(plot_df))[::-1]
    y_best = base_y + 0.085
    y_ours = base_y - 0.085

    ax.errorbar(
        plot_df["best_baseline_delta"],
        y_best,
        xerr=plot_df["best_baseline_sd"],
        fmt="o",
        color=ORANGE,
        ecolor=ORANGE,
        elinewidth=1.6,
        capsize=3,
        capthick=1.2,
        markersize=5.2,
        label=r"$\Delta$(Best Baseline - SingleTask)",
        zorder=3,
    )
    ax.errorbar(
        plot_df["our_method_delta"],
        y_ours,
        xerr=plot_df["our_method_sd"],
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.6,
        capsize=3,
        capthick=1.2,
        markersize=5.2,
        label=r"$\Delta$(Our method - SingleTask)",
        zorder=3,
    )

    annotate_paired_points(ax, plot_df, y_best, y_ours)

    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_yticks(base_y)
    ax.set_yticklabels(plot_df["task"])
    ax.set_ylabel("Task")
    ax.set_xlabel(r"$\Delta$ Macro-F1")

    lo, hi = nice_xlim(plot_df)
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_ylim(-0.45, len(plot_df) - 0.55)

    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
        spine.set_linewidth(1.0)

    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), handlelength=2.0)
    return fig


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot task-wise delta Macro-F1 against the single-task baseline."
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--sheet", default=SHEET_NAME)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / DEFAULT_OUTPUT_BASENAME,
    )
    parser.add_argument("--basename", default=DEFAULT_OUTPUT_BASENAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_df = build_plot_data(args.excel, args.sheet)

    output_base = args.outdir / args.basename
    save_figure(draw_delta_plot(plot_df), output_base)
    plot_df.to_csv(output_base.with_name(output_base.name + "_source_data.csv"), index=False)

    print("Saved:")
    for suffix in (".svg", ".pdf", ".png", ".tiff", "_source_data.csv"):
        path = output_base.with_suffix(suffix) if suffix.startswith(".") else output_base.with_name(output_base.name + suffix)
        print(f"  {path}")
    print("\nPlot data:")
    print(plot_df.to_string(index=False))


if __name__ == "__main__":
    main()

