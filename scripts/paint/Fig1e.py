# -*- coding: utf-8 -*-
"""
Fig. 2E: Minority-Class F1 boxplots across binary tasks.

输入 Excel sheet:
优先读取 Fig2E_data
如果不存在，则尝试读取 Fig2D_fold_values

要求 fold-level 数据，即每个模型每个任务每个 fold 一行。
推荐列名:
    任务 / task_key
    模型 / model
    fold
    Minority F1 / minority_f1

输出:
    Fig2E_minority_f1_boxplots_beautified.png
    Fig2E_minority_f1_boxplots_beautified.pdf
    Fig2E_minority_f1_boxplots_beautified.svg
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 1. 路径设置
# ============================================================
excel_path = Path(
    r"xx_path"
)

save_dir = excel_path.parent
save_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 全局绘图配置
# ============================================================
TASK_ORDER = ["t3", "t4", "t5"]

TASK_TITLE_MAP = {
    "t3": "t3: Exercise ECG interpretation",
    "t4": "t4: Ventilatory function",
    "t5": "t5: Heart-rate reserve",
}

MODEL_ORDER = [
    "Single-task model",
    "Shared Bottom",
    "MMOE",
    "CGC",
    "ADATT",
    "Our method",
]

# x 轴显示名，做换行避免重叠
XTICK_LABEL_MAP = {
    "Single-task model": "Single-task\nmodel",
    "Shared Bottom": "Shared-bottom\nMTL",
    "MMOE": "MMoE",
    "CGC": "CGC",
    "ADATT": "AdaTT",
    "Our method": "Our\nmethod",
}

# legend 显示名
LEGEND_LABEL_MAP = {
    "Single-task model": "Single-task model",
    "Shared Bottom": "Shared-bottom MTL",
    "MMOE": "MMoE",
    "CGC": "CGC",
    "ADATT": "AdaTT",
    "Our method": "Our method",
}

# 与 Fig2C / Fig1C 保持一致的色块
COLOR_MAP = {
    "Single-task model": "#1F77B4",  # 蓝色
    "Shared Bottom": "#8C564B",      # 棕色
    "MMOE": "#FF7F0E",               # 橙色
    "CGC": "#9467BD",                # 紫色
    "ADATT": "#2CA02C",              # 绿色
    "Our method": "#D62728",         # 红色
}

# 箱体透明度
ALPHA_MAP = {
    "Single-task model": 0.88,
    "Shared Bottom": 0.68,
    "MMOE": 0.68,
    "CGC": 0.68,
    "ADATT": 0.68,
    "Our method": 0.90,
}

# ============================================================
# Y-axis settings for Fig. 2E
# ============================================================
# 为提高可读性，不同任务使用不同 y 轴范围：
# t3: 0.0–0.6
# t4/t5: 0.6–1.0
Y_LIM_MAP = {
    "t3": (0.0, 0.6),
    "t4": (0.6, 1.0),
    "t5": (0.6, 1.0),
}

Y_TICKS_MAP = {
    "t3": np.arange(0.0, 0.61, 0.2),
    "t4": np.arange(0.6, 1.01, 0.1),
    "t5": np.arange(0.55, 1.01, 0.1),
}

# 均值标签的偏移量。不同 y 轴范围下，offset 不宜太大
MEAN_LABEL_OFFSET_MAP = {
    "t3": 0.025,
    "t4": 0.015,
    "t5": 0.015,
}

# 防止均值标签超过图框顶部
MEAN_LABEL_TOP_MARGIN_MAP = {
    "t3": 0.025,
    "t4": 0.018,
    "t5": 0.018,
}

# 是否显示底部图例
SHOW_LEGEND = False

# 是否标注均值
ANNOTATE_MEAN_FOR = ["Single-task model", "Our method"]

# y 轴范围
Y_LIM = (0.0, 1.03)

# 图片标题
FIGURE_TITLE = "Fig. 2C: Minority-Class F1 of Different Models on the Holdout Set"


# ============================================================
# 3. 数据读取与清洗
# ============================================================
def _find_column(df, candidates, required=True):
    """
    在 df 中按照候选列名查找列。
    """
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            "找不到所需列，候选列名为：\n"
            + "\n".join(candidates)
            + "\n\n当前 Excel 中可用列名为：\n"
            + "\n".join(map(str, df.columns.tolist()))
        )
    return None


def _read_excel_with_fallback(excel_path: Path):
    """
    优先读取 Fig2E_data。
    如果不存在，则尝试读取 Fig2D_fold_values。
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"未找到 Excel 文件：{excel_path}")

    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names

    if "Fig2E_data" in sheet_names:
        sheet_name = "Fig2E_data"
    elif "Fig2D_fold_values" in sheet_names:
        sheet_name = "Fig2D_fold_values"
    else:
        raise ValueError(
            "Excel 中找不到 Fig2E_data 或 Fig2D_fold_values。\n"
            f"当前可用 sheet 为：{sheet_names}"
        )

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"[INFO] Read sheet: {sheet_name}")
    return df


def normalize_model_name(name):
    """
    将不同写法统一到 MODEL_ORDER 中的标准名称。
    """
    s = str(name).strip()

    mapping = {
        "Single-task": "Single-task model",
        "Single-task model": "Single-task model",

        "Shared Bottom": "Shared Bottom",
        "Shared-bottom": "Shared Bottom",
        "Shared-bottom MTL": "Shared Bottom",
        "shared_bottom": "Shared Bottom",

        "MMOE": "MMOE",
        "MMoE": "MMOE",
        "mmoe": "MMOE",

        "CGC": "CGC",
        "cgc": "CGC",

        "ADATT": "ADATT",
        "AdaTT": "ADATT",
        "adatt": "ADATT",

        "Our method": "Our method",
        "Ours": "Our method",
        "our_method": "Our method",
    }

    return mapping.get(s, s)


def normalize_task_name(name):
    """
    统一任务名。
    """
    s = str(name).strip()
    if s in ["t3", "T3"]:
        return "t3"
    if s in ["t4", "T4"]:
        return "t4"
    if s in ["t5", "T5"]:
        return "t5"
    return s


def read_fig2e_data(excel_path: Path):
    df_raw = _read_excel_with_fallback(excel_path)

    task_col = _find_column(
        df_raw,
        ["任务", "task", "task_key", "Task", "Task key"],
        required=True,
    )
    model_col = _find_column(
        df_raw,
        ["模型", "model", "Model", "model_name", "Model name"],
        required=True,
    )
    fold_col = _find_column(
        df_raw,
        ["fold", "Fold", "折", "fold_id"],
        required=True,
    )
    value_col = _find_column(
        df_raw,
        ["Minority F1", "minority_f1", "Minority F1 mean", "minority_f1_mean"],
        required=True,
    )

    df = pd.DataFrame()
    df["task"] = df_raw[task_col].map(normalize_task_name)
    df["model"] = df_raw[model_col].map(normalize_model_name)
    df["fold"] = pd.to_numeric(df_raw[fold_col], errors="coerce")
    df["minority_f1"] = pd.to_numeric(df_raw[value_col], errors="coerce")

    # 只保留 t3/t4/t5 与指定模型
    df = df[df["task"].isin(TASK_ORDER)].copy()
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df = df.dropna(subset=["fold", "minority_f1"])

    if df.empty:
        raise ValueError("读取后 Fig2E 数据为空，请检查 task/model/fold/Minority F1 列。")

    # 排序
    df["task"] = pd.Categorical(df["task"], categories=TASK_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["task", "model", "fold"]).reset_index(drop=True)

    print("[INFO] Data summary:")
    print(
        df.groupby(["task", "model"], observed=False)["minority_f1"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )

    return df


# ============================================================
# 4. 绘图函数
# ============================================================
def plot_fig2e_boxplot():
    df = read_fig2e_data(excel_path)

    # Fig1C 风格：serif 字体、论文风格
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.8, 5.5),
        sharey=False,   # 不共享 y 轴，确保 b/c 也显示刻度
    )

    panel_labels = ["a", "b", "c"]

    for ax, task in zip(axes, TASK_ORDER):
        sub = df[df["task"] == task].copy()

        positions = np.arange(1, len(MODEL_ORDER) + 1)
        data = []

        for model in MODEL_ORDER:
            vals = sub.loc[sub["model"] == model, "minority_f1"].dropna().values
            data.append(vals)

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showfliers=True,
            whis=(0, 100),
            medianprops=dict(color="#111111", linewidth=1.6),
            whiskerprops=dict(color="#333333", linewidth=1.2),
            capprops=dict(color="#333333", linewidth=1.2),
            boxprops=dict(linewidth=1.2, edgecolor="#333333"),
            flierprops=dict(
                marker="o",
                markersize=3.5,
                markerfacecolor="#333333",
                markeredgecolor="#333333",
                alpha=0.55,
            ),
        )

        for patch, model in zip(bp["boxes"], MODEL_ORDER):
            patch.set_facecolor(COLOR_MAP[model])
            patch.set_alpha(ALPHA_MAP[model])
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.2)

        # 标注 Single-task 与 Our method 的 mean
        for pos, model in zip(positions, MODEL_ORDER):
            vals = sub.loc[sub["model"] == model, "minority_f1"].dropna().values
            if len(vals) == 0:
                continue

            if model in ANNOTATE_MEAN_FOR:
                mean_val = float(np.mean(vals))
                max_val = float(np.max(vals))

                current_ylim = Y_LIM_MAP[task]
                label_offset = MEAN_LABEL_OFFSET_MAP[task]
                top_margin = MEAN_LABEL_TOP_MARGIN_MAP[task]

                label_y = min(
                    max_val + label_offset,
                    current_ylim[1] - top_margin,
                )

                ax.text(
                    pos,
                    label_y,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=11.5,
                    color="#222222",
                    clip_on=False,
                )

        ax.set_title(
            TASK_TITLE_MAP[task],
            fontsize=14.5,
            fontweight="bold",
            pad=10,
        )

        # ax.text(
        #     -0.14,
        #     1.04,
        #     panel_label,
        #     transform=ax.transAxes,
        #     fontsize=15,
        #     fontweight="bold",
        #     ha="left",
        #     va="top",
        # )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [XTICK_LABEL_MAP[m] for m in MODEL_ORDER],
            rotation=28,
            ha="right",
            fontsize=10.5,
        )
        ax.set_xlabel("Models", fontsize=13, fontweight="bold",labelpad=-12)

        # 每个任务独立 y 轴范围
        ax.set_ylim(*Y_LIM_MAP[task])
        ax.set_yticks(Y_TICKS_MAP[task])
        ax.tick_params(axis="y", labelleft=True, labelsize=11)

        if task == "t3":
            ax.set_ylabel("Minority F1-score", fontsize=13, fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.38)
        ax.set_axisbelow(True)

        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.2)
            ax.spines[spine].set_color("#333333")

        ax.tick_params(axis="x", labelsize=10.5, width=1.1, length=4)
        ax.tick_params(axis="y", width=1.1, length=4)

    # # 总标题
    # fig.suptitle(
    #     FIGURE_TITLE,
    #     fontsize=16.5,
    #     fontweight="bold",
    #     y=0.995,
    # )

    # 图例
    if SHOW_LEGEND:
        legend_handles = [
            Patch(
                facecolor=COLOR_MAP[m],
                edgecolor="#333333",
                alpha=ALPHA_MAP[m],
                label=LEGEND_LABEL_MAP[m],
            )
            for m in MODEL_ORDER
        ]

        legend = fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(MODEL_ORDER),
            frameon=True,
            fontsize=11.5,
            bbox_to_anchor=(0.5, 0.01),
            columnspacing=1.45,
            handlelength=1.35,
            handletextpad=0.45,
            borderaxespad=0.0,
        )
        legend.get_frame().set_edgecolor("#333333")
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(1.0)

        bottom = 0.24
    else:
        bottom = 0.18

    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.84,
        bottom=bottom,
        wspace=0.17,
    )

    # 保存
    png_path = save_dir / "Fig2E_minority_f1_boxplots_beautified.png"
    pdf_path = save_dir / "Fig2E_minority_f1_boxplots_beautified.pdf"
    svg_path = save_dir / "Fig2E_minority_f1_boxplots_beautified.svg"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved SVG: {svg_path}")

    plt.show()
    return fig, axes


def main():
    plot_fig2e_boxplot()


if __name__ == "__main__":
    main()
