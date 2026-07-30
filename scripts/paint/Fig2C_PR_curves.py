#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Fig. 2C-style multi-model precision-recall curves for binary CPET tasks.

This uses the confirmed compact ROC layout, but recomputes five-fold PR curves
from each workbook's sample_scores sheet.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


TASK_ORDER = ["t3", "t4", "t5"]

TASK_TITLES = {
    "t3": "t3: Exercise ECG interpretation",
    "t4": "t4: Ventilatory function",
    "t5": "t5: Heart-rate reserve",
}

CONTENT_BASE = (
    Path("xx_path")
    / "\u8bba\u6587"
    / "\u5185\u5bb9\u4fdd\u5b58"
)

DEFAULT_OUTPUT_DIR = Path("outputs") / "fig2c_pr_multimodel_iter026"
DEFAULT_PREFIX = "Fig2C_PR_curves_iter026"

DEFAULT_OUR_METHOD_WORKBOOK = (
    CONTENT_BASE
    / "OurMethods"
    / "iter026_joint_t1_t4_rescue_from_iter022_predbank_t46_t472_locked"
    / "fig2c_roc_raw"
    / "fig2c_roc_curve_data.xlsx"
)

SINGLE_TASK_WORKBOOKS = [
    CONTENT_BASE
    / "SingleTask"
    / "fig2c_roc_single_task_t3_HDSTGCN_nine_graph_"
    "\u6807\u51c6\u5fc3\u7535\u8fd0\u52a8\u8d1f\u8377\u8bd5\u9a8c_final.xlsx",
    CONTENT_BASE
    / "SingleTask"
    / "fig2c_roc_single_task_t4_HDSTGCN_nine_graph_"
    "\u8fd0\u52a8\u4e2d\u6362\u6c14\u80ba\u529f\u80fd_final.xlsx",
    CONTENT_BASE
    / "SingleTask"
    / "fig2c_roc_single_task_t5_HDSTGCN_nine_graph_"
    "\u5fc3\u7387\u50a8\u5907_final.xlsx",
]


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    workbooks: tuple[Path, ...]
    color: str
    linestyle: str
    linewidth: float = 2.2


def default_model_specs(our_method_workbook: Path) -> list[ModelSpec]:
    return [
        ModelSpec("Single-task model", tuple(SINGLE_TASK_WORKBOOKS), "#1F77B4", "-", 2.0),
        ModelSpec(
            "Shared-bottom MTL",
            (CONTENT_BASE / "A1_Shared_Bottom" / "fig2c_roc_curve_data.xlsx",),
            "#8C564B",
            "--",
            2.0,
        ),
        ModelSpec("MMoE", (CONTENT_BASE / "A2_MMOE" / "fig2c_roc_curve_data.xlsx",), "#FF7F0E", "-.", 2.0),
        ModelSpec("CGC", (CONTENT_BASE / "A3_CGC" / "fig2c_roc_curve_data.xlsx",), "#9467BD", "-.", 2.0),
        ModelSpec("AdaTT", (CONTENT_BASE / "A4_ADATT" / "fig2c_roc_curve_data.xlsx",), "#2CA02C", "-", 2.0),
        ModelSpec("Our method", (our_method_workbook,), "#D62728", "-", 2.8),
    ]


def choose_available_font(candidates: Iterable[str]) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            return font
    return "DejaVu Serif"


def set_matplotlib_style() -> None:
    font_family = choose_available_font(("Times New Roman", "Liberation Serif", "DejaVu Serif"))
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_sample_scores(path: Path, display_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")
    df = pd.read_excel(path, sheet_name="sample_scores")
    required = {"task_key", "fold", "y_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} sample_scores missing columns: {sorted(missing)}")

    df = df.copy()
    if "y_true_minor" in df.columns:
        df["y_positive"] = pd.to_numeric(df["y_true_minor"], errors="coerce")
    elif {"y_true", "positive_class"} <= set(df.columns):
        df["y_positive"] = (
            pd.to_numeric(df["y_true"], errors="coerce")
            == pd.to_numeric(df["positive_class"], errors="coerce")
        ).astype(int)
    else:
        raise ValueError(f"{path} needs y_true_minor or y_true + positive_class for PR curves.")

    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["display_name"] = display_name
    return df.dropna(subset=["task_key", "fold", "y_positive", "y_score"])


def load_all_sample_scores(model_specs: list[ModelSpec]) -> pd.DataFrame:
    frames = []
    for spec in model_specs:
        for workbook in spec.workbooks:
            frames.append(read_sample_scores(workbook, spec.display_name))
    return pd.concat(frames, ignore_index=True)


def precision_recall_curve_np(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    positive_total = int(y_sorted.sum())
    if positive_total == 0:
        raise ValueError("Cannot compute PR curve with zero positive samples.")

    distinct = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct, y_sorted.size - 1]

    tp = np.cumsum(y_sorted)[threshold_idxs]
    fp = 1 + threshold_idxs - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positive_total

    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    average_precision = float(np.sum(np.diff(recall) * precision[1:]))
    return recall, precision, average_precision


def interpolate_precision(recall: np.ndarray, precision: np.ndarray, recall_grid: np.ndarray) -> np.ndarray:
    work = pd.DataFrame({"recall": recall, "precision": precision})
    work = work.groupby("recall", as_index=False)["precision"].max().sort_values("recall")
    x = work["recall"].to_numpy(dtype=float)
    y = work["precision"].to_numpy(dtype=float)
    return np.interp(recall_grid, x, y, left=y[0], right=y[-1])


def aggregate_pr_curves(scores: pd.DataFrame, model_specs: list[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    recall_grid = np.linspace(0.0, 1.0, 101)
    curve_rows = []
    summary_rows = []

    for spec in model_specs:
        for task in TASK_ORDER:
            task_model = scores[
                (scores["display_name"] == spec.display_name)
                & (scores["task_key"].astype(str) == task)
            ]
            if task_model.empty:
                raise ValueError(f"Missing sample_scores for {spec.display_name}:{task}")

            fold_precisions = []
            fold_auprcs = []
            fold_prevalence = []

            for fold, fold_df in task_model.groupby("fold", sort=True):
                y_true = fold_df["y_positive"].to_numpy(dtype=int)
                y_score = fold_df["y_score"].to_numpy(dtype=float)
                recall, precision, auprc = precision_recall_curve_np(y_true, y_score)
                fold_precisions.append(interpolate_precision(recall, precision, recall_grid))
                fold_auprcs.append(auprc)
                fold_prevalence.append(float(np.mean(y_true)))

            precision_array = np.vstack(fold_precisions)
            mean_precision = precision_array.mean(axis=0)
            std_precision = precision_array.std(axis=0, ddof=1) if precision_array.shape[0] > 1 else np.zeros_like(recall_grid)

            for recall, precision, std in zip(recall_grid, mean_precision, std_precision):
                curve_rows.append(
                    {
                        "display_name": spec.display_name,
                        "task_key": task,
                        "mean_recall": float(recall),
                        "mean_precision": float(np.clip(precision, 0.0, 1.0)),
                        "std_precision": float(std),
                    }
                )

            summary_rows.append(
                {
                    "display_name": spec.display_name,
                    "task_key": task,
                    "mean_auprc": float(np.mean(fold_auprcs)),
                    "std_auprc": float(np.std(fold_auprcs, ddof=1)) if len(fold_auprcs) > 1 else np.nan,
                    "mean_prevalence": float(np.mean(fold_prevalence)),
                    "n_folds": len(fold_auprcs),
                }
            )

    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def get_auprc(summary: pd.DataFrame, display_name: str, task_key: str) -> float:
    sub = summary[(summary["display_name"] == display_name) & (summary["task_key"] == task_key)]
    if sub.empty:
        return np.nan
    return float(sub["mean_auprc"].iloc[0])


def get_baseline_range(summary: pd.DataFrame, task_key: str) -> tuple[float, float, float]:
    sub = summary[summary["task_key"] == task_key]
    if sub.empty:
        return np.nan, np.nan, np.nan
    prevalence = sub["mean_prevalence"].to_numpy(dtype=float)
    return float(np.min(prevalence)), float(np.mean(prevalence)), float(np.max(prevalence))


def plot_fig2c_pr(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    model_specs: list[ModelSpec],
    output_prefix: Path,
    width: float,
    height: float,
    panel_box_aspect: float,
    wspace: float,
    dpi: int,
) -> None:
    set_matplotlib_style()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(width, height),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for ax, task in zip(axes, TASK_ORDER):
        baseline_min, baseline_mean, baseline_max = get_baseline_range(summary, task)
        if np.isfinite(baseline_min) and np.isfinite(baseline_max) and baseline_max > baseline_min:
            ax.axhspan(
                baseline_min,
                baseline_max,
                color="#BDBDBD",
                alpha=0.18,
                linewidth=0,
                zorder=0,
            )
            baseline_label = f"Chance ({baseline_min:.4f}-{baseline_max:.4f})"
        else:
            baseline_label = f"Chance ({baseline_mean:.4f})"
        ax.axhline(
            baseline_mean,
            color="#8E8E8E",
            linestyle="--",
            linewidth=1.6,
            label=baseline_label,
            zorder=1,
        )

        for spec in model_specs:
            sub = curves[
                (curves["display_name"] == spec.display_name)
                & (curves["task_key"].astype(str) == task)
            ].sort_values("mean_recall")
            if sub.empty:
                continue

            auprc = get_auprc(summary, spec.display_name, task)
            ax.plot(
                sub["mean_recall"].to_numpy(dtype=float),
                np.clip(sub["mean_precision"].to_numpy(dtype=float), 0.0, 1.0),
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                label=f"{spec.display_name} ({auprc:.4f})",
                zorder=3 if spec.display_name != "Our method" else 4,
            )

        ax.set_title(TASK_TITLES[task], pad=8)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.03)
        ax.set_box_aspect(panel_box_aspect)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("Recall (Sensitivity)")
        ax.grid(True, linestyle="--", color="#D0D0D0", linewidth=0.7, alpha=0.60)
        ax.set_axisbelow(True)
        ax.legend(
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.92,
            borderpad=0.35,
            labelspacing=0.28,
            handlelength=1.75,
            handletextpad=0.45,
        )

    axes[0].set_ylabel("Precision")
    fig.subplots_adjust(left=0.055, right=0.995, top=0.865, bottom=0.155, wspace=wspace)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out = output_prefix.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {out}")
    plt.close(fig)


def write_summary(summary: pd.DataFrame, output_path: Path, model_specs: list[ModelSpec]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order_map = {spec.display_name: i for i, spec in enumerate(model_specs)}
    out = summary.copy()
    out["model_order"] = out["display_name"].map(order_map)
    out["task_order"] = out["task_key"].map({task: i for i, task in enumerate(TASK_ORDER)})
    out = out.sort_values(["model_order", "task_order"]).drop(columns=["model_order", "task_order"])
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig. 2C-style multi-model PR curves.")
    parser.add_argument(
        "--our-method-workbook",
        type=Path,
        default=DEFAULT_OUR_METHOD_WORKBOOK,
        help="Our method fig2c_roc_curve_data.xlsx. Defaults to the iter026 locked workbook.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Output filename prefix.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG output DPI.")
    parser.add_argument("--width", type=float, default=13.4, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=4.8, help="Figure height in inches.")
    parser.add_argument(
        "--panel-box-aspect",
        type=float,
        default=0.875,
        help="Panel plotting-area height/width ratio; matches the confirmed ROC template.",
    )
    parser.add_argument("--wspace", type=float, default=0.025, help="Horizontal space between subplots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_specs = default_model_specs(args.our_method_workbook)
    scores = load_all_sample_scores(model_specs)
    curves, summary = aggregate_pr_curves(scores, model_specs)

    output_prefix = args.outdir / args.prefix
    write_summary(summary, output_prefix.with_name(f"{args.prefix}_auprc_check.csv"), model_specs)
    plot_fig2c_pr(
        curves=curves,
        summary=summary,
        model_specs=model_specs,
        output_prefix=output_prefix,
        width=args.width,
        height=args.height,
        panel_box_aspect=args.panel_box_aspect,
        wspace=args.wspace,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

