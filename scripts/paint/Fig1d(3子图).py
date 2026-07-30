# -*- coding: utf-8 -*-
"""
Figure 2D / Fig1d: Minority-class performance in binary tasks.

更新点：
1. 不再硬编码 t1–t5 MTL baseline 的模型名。
2. Excel 读取模型名 excel_model_name 与图例显示 legend_label 分开配置。
3. 指标列名 mean_col / sd_col 与子图标题 panel_title 分开配置。
4. 默认从 RESULT2_Table_Fig2A_Fig2B_completed.xlsx 的 Fig2D_data sheet 读取数据。

图注SEM：Error bars indicate the standard error of the mean across five fold models evaluated on the holdout cohort. Full standard deviations are reported in Supplementary Table X.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================
# 1. 路径设置
# =========================
excel_path = Path(
    r"xx_path"
)

save_dir = excel_path.parent
save_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 任务与模型配置
# =========================
TASK_ORDER = ["t3", "t4", "t5"]
N_FOLDS = 5
ERROR_BAR_MODE = "sem"  # 可选: "sd", "sem", "none"

# excel_model_name：Fig2D_data 中“模型”列的取值
# legend_label：图例显示名称，可以与 excel_model_name 不一致
MODEL_SPECS = [
    {
        "excel_model_name": "Single-task model",
        "legend_label": "Single-task model",
        "color": "#BDBDBD",
    },
    {
        "excel_model_name": "Shared Bottom",
        "legend_label": "Best generic MTL baseline",
        "color": "#7FA6C7",
    },
    {
        "excel_model_name": "Our method",
        "legend_label": "Our method",
        "color": "#2C7FB8",
    },

    # 示例：新增模型只需要追加：
    # {
    #     "excel_model_name": "MMoE",
    #     "legend_label": "MMoE",
    #     "color": "#8C6BB1",
    # },
]

# mean_col / sd_col：Excel 列名
# panel_title：图中子图标题，可以与 Excel 列名不一致
METRIC_SPECS = [
    {
        "mean_col": "Minority precision mean",
        "sd_col": "Minority precision SD",
        "panel_title": "Minority precision",
    },
    {
        "mean_col": "Minority recall mean",
        "sd_col": "Minority recall SD",
        "panel_title": "Minority recall",
    },
    {
        "mean_col": "Minority F1 mean",
        "sd_col": "Minority F1 SD",
        "panel_title": "Minority F1",
    },
]


# =========================
# 3. 工具函数
# =========================
def convert_errors(errors, mode="sem", n_folds=5):
    """
    将 Excel 中读取的 SD 转换为绘图误差线。
    mode:
        - "sd": 使用 SD
        - "sem": 使用 SEM = SD / sqrt(n)
        - "none": 不显示误差线
    """
    errors = np.asarray(errors, dtype=float)

    if mode == "sd":
        return errors
    elif mode == "sem":
        return errors / np.sqrt(n_folds)
    elif mode == "none":
        return np.zeros_like(errors)
    else:
        raise ValueError(f"Unsupported ERROR_BAR_MODE: {mode}")
    
def add_value_labels(ax, bars, values=None, errors=None, offset=0.035):
    if values is None:
        values = [bar.get_height() for bar in bars]
    if errors is None:
        errors = [0.0 for _ in bars]

    for bar, value, err in zip(bars, values, errors):
        if np.isnan(value):
            continue
        err = 0.0 if np.isnan(err) else err
        label_x = bar.get_x() + bar.get_width() / 2
        label_y = value + err + offset
        ax.text(
            label_x,
            label_y,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#222222",
            clip_on=False,
        )


def read_fig2d_data(excel_path):
    if not excel_path.exists():
        raise FileNotFoundError(f"未找到 Excel 文件：{excel_path}")

    df = pd.read_excel(excel_path, sheet_name="Fig2D_data")
    required_cols = {"任务", "模型"}
    for spec in METRIC_SPECS:
        required_cols.add(spec["mean_col"])
        required_cols.add(spec["sd_col"])

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            "Fig2D_data 缺少以下列：\n"
            + "\n".join(sorted(missing))
            + "\n\n当前可用列名：\n"
            + "\n".join(map(str, df.columns.tolist()))
        )

    df["任务"] = df["任务"].astype(str)
    df["模型"] = df["模型"].astype(str)
    return df


def get_metric_values(df, metric_spec, model_name):
    values = []
    errors = []
    for task in TASK_ORDER:
        sub = df[(df["任务"] == task) & (df["模型"] == model_name)]
        if sub.empty:
            values.append(np.nan)
            errors.append(np.nan)
            continue
        values.append(pd.to_numeric(sub[metric_spec["mean_col"]], errors="coerce").iloc[0])
        errors.append(pd.to_numeric(sub[metric_spec["sd_col"]], errors="coerce").iloc[0])
    return np.array(values, dtype=float), np.array(errors, dtype=float)


# =========================
# 4. 主绘图函数
# =========================

def plot_figure2d():
    df = read_fig2d_data(excel_path)

    x = np.arange(len(TASK_ORDER))
    n_models = len(MODEL_SPECS)
    group_width = 0.72
    width = min(0.22, group_width / max(n_models, 1))
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * width

    fig, axes = plt.subplots(1, len(METRIC_SPECS), figsize=(15.5, 4.8), sharey=True)
    if len(METRIC_SPECS) == 1:
        axes = [axes]

    for ax, metric_spec in zip(axes, METRIC_SPECS):
        for i, model_spec in enumerate(MODEL_SPECS):
            model_name = model_spec["excel_model_name"]
            values, errors_sd = get_metric_values(df, metric_spec, model_name)

            # Excel 中读取的是 SD；根据 ERROR_BAR_MODE 转换为 SD / SEM / none
            errors_for_plot = convert_errors(
                errors_sd,
                mode=ERROR_BAR_MODE,
                n_folds=N_FOLDS,
            )
            errors_for_plot = np.nan_to_num(errors_for_plot, nan=0.0)

            bar_pos = x + offsets[i]

            bars = ax.bar(
                bar_pos,
                values,
                width=width,
                yerr=errors_for_plot,
                capsize=4,
                color=model_spec["color"],
                edgecolor="none",
                linewidth=0,
                error_kw={"elinewidth": 1.3, "ecolor": "#555555", "capthick": 1.3},
                label=model_spec["legend_label"],
            )

            add_value_labels(
                ax=ax,
                bars=bars,
                values=values,
                errors=errors_for_plot,
                offset=0.035,
            )

        ax.set_title(metric_spec["panel_title"], fontsize=14, pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(TASK_ORDER, fontsize=12)
        ax.set_xlabel("Task", fontsize=12)
        ax.set_ylim(0, 1.18)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.28)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)
        ax.tick_params(axis="both", labelsize=11, width=1.1, length=4)

    axes[0].set_ylabel("Score", fontsize=12)

    fig.suptitle("Figure 2D. Minority-class performance in binary tasks", fontsize=16, y=1.02)

    legend_handles = [
        Patch(facecolor=spec["color"], edgecolor="none", label=spec["legend_label"])
        for spec in MODEL_SPECS
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(MODEL_SPECS), 4),
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=1.8,
        columnspacing=2.6,
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.22, wspace=0.18)

    png_path = save_dir / "Figure2D_minority_class_performance_final.png"
    pdf_path = save_dir / "Figure2D_minority_class_performance_final.pdf"
    svg_path = save_dir / "Figure2D_minority_class_performance_final.svg"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved SVG: {svg_path}")

    plt.show()
    return fig, axes


def main():
    plot_figure2d()


if __name__ == "__main__":
    main()

