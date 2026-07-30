# -*- coding: utf-8 -*-
"""
Plot Fig. 2C ROC curves for binary structured CPET report tasks.

Input Excel:
    fig2c_roc_curve_data.xlsx

Required sheets:
    - roc_points_mean
    - auc_summary

Expected columns in roc_points_mean:
    model_name, model_type, task_key, task_name,
    mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc, positive_class

Expected columns in auc_summary:
    model_name, model_type, task_key, task_name,
    auc_fold1, auc_fold2, auc_fold3, auc_fold4, auc_fold5,
    mean_auc, std_auc, positive_class, n_folds

Output:
    Fig2C_ROC_curves.png
    Fig2C_ROC_curves.pdf
    Fig2C_ROC_curves.svg
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Basic configuration
# =========================

TASK_ORDER = ["t3", "t4", "t5"]

TASK_TITLE_MAP = {
    "t1": "t1: Weber functional class",
    "t2": "t2: Exercise capacity",
    "t3": "t3: Exercise ECG response",
    "t4": "t4: Breathing reserve",
    "t5": "t5: Heart-rate reserve",
    "Mean across t1–t5": "Mean across t1–t5",
}


TASK_TITLE_CN_MAP = {
    "t3": "t3: 标准心电运动负荷试验",
    "t4": "t4: 运动中换气肺功能",
    "t5": "t5: 心率储备",
}

MODEL_ORDER = [
    "Single-task model",
    "t1-t5 MTL baseline",
    "t1–t5 MTL baseline",
    "Our method",
]

MODEL_LABEL_MAP = {
    "Single-task model": "Single-task model",
    "t1-t5 MTL baseline": "t1–t5 MTL baseline",
    "t1–t5 MTL baseline": "t1–t5 MTL baseline",
    "Our method": "Our method",
}

# 期刊图常用：灰色=Single，橙色=MTL，蓝色=Ours
MODEL_COLOR_MAP = {
    "Single-task model": "#7A7A7A",
    "t1-t5 MTL baseline": "#D55E00",
    "t1–t5 MTL baseline": "#D55E00",
    "Our method": "#0072B2",
}

MODEL_LINESTYLE_MAP = {
    "Single-task model": "--",
    "t1-t5 MTL baseline": "-.",
    "t1–t5 MTL baseline": "-.",
    "Our method": "-",
}


# =========================
# 2. Utility functions
# =========================

def set_matplotlib_style():
    """A clean journal-style matplotlib setting."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def read_excel_data(excel_path: str):
    """Read ROC mean points and AUC summary."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    roc_mean = pd.read_excel(excel_path, sheet_name="roc_points_mean")
    auc_summary = pd.read_excel(excel_path, sheet_name="auc_summary")

    # 删除空行
    roc_mean = roc_mean.dropna(subset=["model_name", "task_key", "mean_fpr", "mean_tpr"])
    auc_summary = auc_summary.dropna(subset=["model_name", "task_key", "mean_auc"])

    required_mean_cols = {
        "model_name", "model_type", "task_key", "task_name",
        "mean_fpr", "mean_tpr", "std_tpr", "mean_auc", "std_auc"
    }
    required_auc_cols = {
        "model_name", "model_type", "task_key", "mean_auc", "std_auc"
    }

    missing_mean = required_mean_cols - set(roc_mean.columns)
    missing_auc = required_auc_cols - set(auc_summary.columns)

    if missing_mean:
        raise ValueError(f"`roc_points_mean` 缺少列: {missing_mean}")
    if missing_auc:
        raise ValueError(f"`auc_summary` 缺少列: {missing_auc}")

    # 保证数值列为 numeric
    for col in ["mean_fpr", "mean_tpr", "std_tpr", "mean_auc", "std_auc"]:
        roc_mean[col] = pd.to_numeric(roc_mean[col], errors="coerce")

    for col in ["mean_auc", "std_auc"]:
        auc_summary[col] = pd.to_numeric(auc_summary[col], errors="coerce")

    roc_mean = roc_mean.dropna(subset=["mean_fpr", "mean_tpr"])
    auc_summary = auc_summary.dropna(subset=["mean_auc"])

    return roc_mean, auc_summary


def get_ordered_models(roc_mean: pd.DataFrame):
    """Return model names in desired plotting order."""
    available = list(roc_mean["model_name"].dropna().unique())

    ordered = []
    for m in MODEL_ORDER:
        if m in available and m not in ordered:
            ordered.append(m)

    for m in available:
        if m not in ordered:
            ordered.append(m)

    return ordered


def get_auc_label(auc_summary: pd.DataFrame, model_name: str, task_key: str):
    """Get AUROC mean ± SD text for legend."""
    sub = auc_summary[
        (auc_summary["model_name"] == model_name)
        & (auc_summary["task_key"] == task_key)
    ]

    if sub.empty:
        return "AUROC = N/A"

    mean_auc = float(sub["mean_auc"].iloc[0])
    std_auc = float(sub["std_auc"].iloc[0])

    if np.isnan(std_auc):
        return f"AUROC = {mean_auc:.3f}"

    return f"AUROC = {mean_auc:.3f} ± {std_auc:.3f}"


def summarize_auc(auc_summary: pd.DataFrame):
    """Print AUROC summary for quick checking."""
    print("\n[AUROC summary]")
    keep_cols = ["model_name", "task_key", "mean_auc", "std_auc"]
    print(auc_summary[keep_cols].to_string(index=False))


# =========================
# 3. Main plotting function
# =========================

def plot_fig2c_roc(
    excel_path: str,
    output_dir: str,
    use_chinese_titles: bool = False,
    show_std_band: bool = True,
    show_chance_line: bool = True,
    dpi: int = 600,
):
    set_matplotlib_style()

    roc_mean, auc_summary = read_excel_data(excel_path)
    summarize_auc(auc_summary)

    ordered_models = get_ordered_models(roc_mean)
    print("\n[Detected models]")
    for m in ordered_models:
        print(f"  - {m}")

    os.makedirs(output_dir, exist_ok=True)

    # 三联图：t3, t4, t5
    fig, axes = plt.subplots(
        1, 3,
        figsize=(10.8, 3.3),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    panel_labels = ["a", "b", "c"]

    for ax, task_key, panel_label in zip(axes, TASK_ORDER, panel_labels):
        task_df = roc_mean[roc_mean["task_key"] == task_key].copy()

        if task_df.empty:
            ax.text(
                0.5, 0.5,
                f"No data for {task_key}",
                ha="center", va="center",
                transform=ax.transAxes
            )
            continue

        if show_chance_line:
            ax.plot(
                [0, 1], [0, 1],
                linestyle=":",
                linewidth=1.0,
                color="#9CA3AF",
                label="Chance",
                zorder=1
            )

        for model_name in ordered_models:
            sub = task_df[task_df["model_name"] == model_name].sort_values("mean_fpr")

            if sub.empty:
                continue

            x = sub["mean_fpr"].to_numpy(dtype=float)
            y = sub["mean_tpr"].to_numpy(dtype=float)
            sd = sub["std_tpr"].to_numpy(dtype=float)

            # 防止异常值出界
            y = np.clip(y, 0.0, 1.0)
            sd = np.nan_to_num(sd, nan=0.0)
            lower = np.clip(y - sd, 0.0, 1.0)
            upper = np.clip(y + sd, 0.0, 1.0)

            color = MODEL_COLOR_MAP.get(model_name, "#000000")
            linestyle = MODEL_LINESTYLE_MAP.get(model_name, "-")
            model_label = MODEL_LABEL_MAP.get(model_name, model_name)
            auc_label = get_auc_label(auc_summary, model_name, task_key)

            ax.plot(
                x, y,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=f"{model_label}\n{auc_label}",
                zorder=3
            )

            if show_std_band:
                ax.fill_between(
                    x, lower, upper,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                    zorder=2
                )

        title_map = TASK_TITLE_CN_MAP if use_chinese_titles else TASK_TITLE_MAP
        ax.set_title(title_map.get(task_key, task_key), pad=6)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

        ax.set_xlabel("False positive rate")
        if ax is axes[0]:
            ax.set_ylabel("True positive rate")

        ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.25)

        # panel label
        ax.text(
            -0.14, 1.08,
            panel_label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left"
        )

        # 图例放右下角，尽量不遮挡 ROC 主区域
        ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#D1D5DB",
            handlelength=1.8,
            borderpad=0.35,
            labelspacing=0.35
        )

    png_path = os.path.join(output_dir, "Fig2C_ROC_curves.png")
    pdf_path = os.path.join(output_dir, "Fig2C_ROC_curves.pdf")
    svg_path = os.path.join(output_dir, "Fig2C_ROC_curves.svg")

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    plt.close(fig)

    print("\n[Saved]")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print(f"  {svg_path}")


# =========================
# 4. CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Plot Fig. 2C ROC curves from fig2c_roc_curve_data.xlsx"
    )

    parser.add_argument(
        "--excel_path",
        type=str,
        default=r"xx_path",
        help="Path to fig2c_roc_curve_data.xlsx"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"xx_path",
        help="Directory to save output figures"
    )

    parser.add_argument(
        "--chinese_titles",
        action="store_true",
        help="Use Chinese task titles"
    )

    parser.add_argument(
        "--no_std_band",
        action="store_true",
        help="Do not show ±1 SD shaded band"
    )

    parser.add_argument(
        "--no_chance_line",
        action="store_true",
        help="Do not show diagonal chance line"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG output DPI"
    )

    args = parser.parse_args()

    plot_fig2c_roc(
        excel_path=args.excel_path,
        output_dir=args.output_dir,
        use_chinese_titles=args.chinese_titles,
        show_std_band=not args.no_std_band,
        show_chance_line=not args.no_chance_line,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
