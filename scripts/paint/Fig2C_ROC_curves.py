#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Fig. 2C multi-model ROC curves for binary structured CPET report tasks.

The script reads the already aggregated five-fold ROC Excel workbooks and
draws the three binary tasks (t3-t5) in the manuscript-style multi-model layout.
Default paths point to the local CPET manuscript figure source folders.
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

DEFAULT_OUTPUT_DIR = CONTENT_BASE / "Result2" / "V1"
DEFAULT_PREFIX = "Fig2C_ROC_curves"

DEFAULT_OUR_METHOD_WORKBOOK = (
    CONTENT_BASE
    / "OurMethods"
    / "iter026_joint_t1_t4_rescue_from_iter022_predbank_t46_t472_locked"
    / "fig2c_roc_raw"
    / "fig2c_roc_curve_data.xlsx"
)

V1_OUR_METHOD_WORKBOOK = (
    CONTENT_BASE
    / "OurMethods"
    / "best_299_4e46105_\u5f53\u524d\u6700\u597d"
    / "fig2c_roc_curve_data.xlsx"
)

SINGLE_TASK_WORKBOOKS = [
    CONTENT_BASE
    / "SingleTask"
    / "fig2c_roc_single_task_t3_HDSTGCN_nine_graph_t3.xlsx",
    CONTENT_BASE
    / "SingleTask"
    / "fig2c_roc_single_task_t4_HDSTGCN_nine_graph_t4.xlsx",
    CONTENT_BASE
    / "SingleTask"
    / "fig2c_roc_single_task_t5_HDSTGCN_nine_graph_t5.xlsx",
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
        ModelSpec(
            "Single-task model",
            tuple(SINGLE_TASK_WORKBOOKS),
            "#1F77B4",
            "-",
            2.0,
        ),
        ModelSpec(
            "Shared-bottom MTL",
            (CONTENT_BASE / "A1_Shared_Bottom" / "fig2c_roc_curve_data.xlsx",),
            "#8C564B",
            "--",
            2.0,
        ),
        ModelSpec(
            "MMoE",
            (CONTENT_BASE / "A2_MMOE" / "fig2c_roc_curve_data.xlsx",),
            "#FF7F0E",
            "-.",
            2.0,
        ),
        ModelSpec(
            "CGC",
            (CONTENT_BASE / "A3_CGC" / "fig2c_roc_curve_data.xlsx",),
            "#9467BD",
            "-.",
            2.0,
        ),
        ModelSpec(
            "AdaTT",
            (CONTENT_BASE / "A4_ADATT" / "fig2c_roc_curve_data.xlsx",),
            "#2CA02C",
            "-",
            2.0,
        ),
        ModelSpec(
            "DC-CPETNet",
            (our_method_workbook,),
            "#D62728",
            "-",
            2.8,
        ),
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


def _read_required_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")
    return pd.read_excel(path, sheet_name=sheet_name)


def load_model_data(spec: ModelSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_frames = []
    auc_frames = []

    for workbook in spec.workbooks:
        roc_mean = _read_required_sheet(workbook, "roc_points_mean")
        auc_summary = _read_required_sheet(workbook, "auc_summary")

        required_mean = {"task_key", "mean_fpr", "mean_tpr", "std_tpr"}
        required_auc = {"task_key", "mean_auc"}
        missing_mean = required_mean - set(roc_mean.columns)
        missing_auc = required_auc - set(auc_summary.columns)
        if missing_mean:
            raise ValueError(f"{workbook} roc_points_mean missing columns: {sorted(missing_mean)}")
        if missing_auc:
            raise ValueError(f"{workbook} auc_summary missing columns: {sorted(missing_auc)}")

        roc_mean = roc_mean.copy()
        auc_summary = auc_summary.copy()
        roc_mean["display_name"] = spec.display_name
        auc_summary["display_name"] = spec.display_name
        roc_mean["source_workbook"] = str(workbook)
        auc_summary["source_workbook"] = str(workbook)
        mean_frames.append(roc_mean)
        auc_frames.append(auc_summary)

    return pd.concat(mean_frames, ignore_index=True), pd.concat(auc_frames, ignore_index=True)


def load_all_data(model_specs: list[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_frames = []
    auc_frames = []
    for spec in model_specs:
        roc_mean, auc_summary = load_model_data(spec)
        mean_frames.append(roc_mean)
        auc_frames.append(auc_summary)

    roc_mean_all = pd.concat(mean_frames, ignore_index=True)
    auc_summary_all = pd.concat(auc_frames, ignore_index=True)

    for col in ["mean_fpr", "mean_tpr", "std_tpr"]:
        roc_mean_all[col] = pd.to_numeric(roc_mean_all[col], errors="coerce")
    for col in ["mean_auc", "std_auc"]:
        if col in auc_summary_all.columns:
            auc_summary_all[col] = pd.to_numeric(auc_summary_all[col], errors="coerce")

    roc_mean_all = roc_mean_all.dropna(subset=["task_key", "mean_fpr", "mean_tpr"])
    auc_summary_all = auc_summary_all.dropna(subset=["task_key", "mean_auc"])
    return roc_mean_all, auc_summary_all


def get_auc(auc_summary: pd.DataFrame, display_name: str, task_key: str) -> float:
    sub = auc_summary[
        (auc_summary["display_name"] == display_name)
        & (auc_summary["task_key"].astype(str) == task_key)
    ]
    if sub.empty:
        return np.nan
    return float(sub["mean_auc"].iloc[0])


def validate_complete_inputs(
    roc_mean: pd.DataFrame,
    auc_summary: pd.DataFrame,
    model_specs: list[ModelSpec],
    allow_missing: bool,
) -> None:
    missing = []
    for spec in model_specs:
        for task in TASK_ORDER:
            has_curve = not roc_mean[
                (roc_mean["display_name"] == spec.display_name)
                & (roc_mean["task_key"].astype(str) == task)
            ].empty
            has_auc = not auc_summary[
                (auc_summary["display_name"] == spec.display_name)
                & (auc_summary["task_key"].astype(str) == task)
            ].empty
            if not has_curve or not has_auc:
                missing.append(f"{spec.display_name}:{task}")

    if missing and not allow_missing:
        raise ValueError(
            "Missing ROC/AUC entries for "
            + ", ".join(missing)
            + ". Use --allow-missing to draw available entries only."
        )


def plot_fig2c(
    roc_mean: pd.DataFrame,
    auc_summary: pd.DataFrame,
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
        ax.plot(
            [0, 1],
            [0, 1],
            color="#8E8E8E",
            linestyle="--",
            linewidth=1.6,
            label="Chance",
            zorder=1,
        )

        for spec in model_specs:
            sub = roc_mean[
                (roc_mean["display_name"] == spec.display_name)
                & (roc_mean["task_key"].astype(str) == task)
            ].sort_values("mean_fpr")
            if sub.empty:
                continue

            auc = get_auc(auc_summary, spec.display_name, task)
            label = f"{spec.display_name} ({auc:.4f})" if np.isfinite(auc) else spec.display_name
            ax.plot(
                sub["mean_fpr"].to_numpy(dtype=float),
                np.clip(sub["mean_tpr"].to_numpy(dtype=float), 0.0, 1.0),
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                label=label,
                zorder=3 if spec.display_name != "Our method" else 4,
            )

        ax.set_title(TASK_TITLES[task], pad=8)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.03)
        ax.set_box_aspect(panel_box_aspect)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis="y", labelleft=True)
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.grid(True, linestyle="--", color="#D0D0D0", linewidth=0.7, alpha=0.60)
        ax.set_axisbelow(True)
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.92,
            borderpad=0.35,
            labelspacing=0.28,
            handlelength=1.75,
            handletextpad=0.45,
        )

        for text in legend.get_texts():
            if text.get_text().startswith("DC-CPETNet"):
                text.set_fontweight("bold")

    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    fig.subplots_adjust(left=0.055, right=0.995, top=0.865, bottom=0.155, wspace=wspace)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out = output_prefix.with_suffix(f".{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved: {out}")

    plt.close(fig)


def write_auc_check_table(auc_summary: pd.DataFrame, output_path: Path, model_specs: list[ModelSpec]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    order_map = {spec.display_name: i for i, spec in enumerate(model_specs)}
    table = auc_summary[["display_name", "task_key", "mean_auc", "std_auc"]].copy()
    table["model_order"] = table["display_name"].map(order_map)
    table["task_order"] = table["task_key"].map({task: i for i, task in enumerate(TASK_ORDER)})
    table = table.sort_values(["model_order", "task_order"]).drop(columns=["model_order", "task_order"])
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig. 2C multi-model ROC curves.")
    parser.add_argument(
        "--our-method-workbook",
        type=Path,
        default=DEFAULT_OUR_METHOD_WORKBOOK,
        help="Our method fig2c_roc_curve_data.xlsx. Defaults to the iter026 locked workbook.",
    )
    parser.add_argument(
        "--use-v1-our-method",
        action="store_true",
        help="Use the older V1 Our method workbook whose AUROCs match the V1 reference image.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Output filename prefix.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG output DPI.")
    parser.add_argument("--width", type=float, default=13.6, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=4.8, help="Figure height in inches.")
    parser.add_argument(
        "--panel-box-aspect",
        type=float,
        default=0.875,
        help="Panel plotting-area height/width ratio. 0.875 matches the provided V1 reference layout.",
    )
    parser.add_argument("--wspace", type=float, default=0.025, help="Horizontal space between subplots.")
    parser.add_argument("--allow-missing", action="store_true", help="Draw available curves if some entries are missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    our_method_workbook = V1_OUR_METHOD_WORKBOOK if args.use_v1_our_method else args.our_method_workbook
    model_specs = default_model_specs(our_method_workbook)

    roc_mean, auc_summary = load_all_data(model_specs)
    validate_complete_inputs(roc_mean, auc_summary, model_specs, allow_missing=args.allow_missing)

    output_prefix = args.outdir / args.prefix
    write_auc_check_table(auc_summary, output_prefix.with_name(f"{args.prefix}_auc_check.csv"), model_specs)
    plot_fig2c(
        roc_mean=roc_mean,
        auc_summary=auc_summary,
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

