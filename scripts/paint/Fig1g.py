# -*- coding: utf-8 -*-
"""
Fig. 1g: Holdout Macro-F1 across t1–t5 and mean performance.(柱状图脚本)

输入 Excel sheet:
    Table2

数据格式要求:
    Table2 中包含如下列：
        指标 / 模型 / t1 / t2 / t3 / t4 / t5 / Mean across t1–t5
    Macro-F1 行中每个单元格格式为：
        mean ± SD

输出:
    Fig1g_holdout_macro_f1_2x3.png
    Fig1g_holdout_macro_f1_2x3.pdf
    Fig1g_holdout_macro_f1_2x3.svg
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 1. 路径设置
# ============================================================
# 按你的本地路径设置；如果脚本和 Excel 放在同一文件夹，也会自动 fallback。
excel_path = Path(
    r"xx_path"
)



if not excel_path.exists():
    raise FileNotFoundError(
        "未找到 Excel 文件。请检查 excel_path 是否正确，或将 Excel 与本脚本放在同一文件夹。\n"
        f"Current excel_path: {excel_path}"
    )

save_dir = excel_path.parent
save_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 全局绘图配置：复用 Fig1e / Fig2E 的论文版式
# ============================================================
TASK_ORDER = ["t1", "t2", "t3", "t4", "t5", "Mean across t1–t5"]

TASK_TITLE_MAP = {
    "t1": "t1: Weber functional class",
    "t2": "t2: Exercise capacity",
    "t3": "t3: Exercise ECG response",
    "t4": "t4: Breathing reserve",
    "t5": "t5: Heart-rate reserve",
    "Mean across t1–t5": "Mean across t1–t5",
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

# 与 Fig1e 保持一致的色块
COLOR_MAP = {
    "Single-task model": "#1F77B4",  # 蓝色
    "Shared Bottom": "#8C564B",      # 棕色
    "MMOE": "#FF7F0E",               # 橙色
    "CGC": "#9467BD",                # 紫色
    "ADATT": "#2CA02C",              # 绿色
    "Our method": "#D62728",         # 红色
}

# 柱体透明度，复用 Fig1e 的模型层次感
ALPHA_MAP = {
    "Single-task model": 0.88,
    "Shared Bottom": 0.68,
    "MMOE": 0.68,
    "CGC": 0.68,
    "ADATT": 0.68,
    "Our method": 0.90,
}

# 是否显示底部图例。Fig1e 默认为 False，这里保持一致。
SHOW_LEGEND = False

# 是否标注均值。与 Fig1e 一致，仅标注 Single-task 与 Our method。
ANNOTATE_MEAN_FOR = ["Single-task model", "Our method"]

# 图片标题，默认不显示；如需总标题，把下方 suptitle 代码取消注释。
FIGURE_TITLE = "Holdout Macro-F1 of Different Models Across CPET Interpretation Tasks"


# ============================================================
# 3. 数据读取与清洗
# ============================================================
def normalize_model_name(name):
    """将不同模型写法统一到 MODEL_ORDER 中的标准名称。"""
    s = str(name).strip()

    mapping = {
        "Single-task": "Single-task model",
        "Single-task model": "Single-task model",
        "Single task model": "Single-task model",

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


def _find_column(df, candidates, required=True):
    """在 df 中按照候选列名查找列。"""
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


def _parse_mean_sd(value):
    """
    解析 Table2 中的 'mean ± SD' 字符串。
    返回：mean, sd。
    """
    if pd.isna(value):
        return np.nan, np.nan

    s = str(value).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("＋/－", "±").replace("+/-", "±")

    # 标准格式：0.8388 ± 0.0093
    m = re.search(r"([-+]?\d*\.?\d+)\s*±\s*([-+]?\d*\.?\d+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))

    # 如果只有一个数，则认为 SD 缺失。
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if m:
        return float(m.group(1)), np.nan

    return np.nan, np.nan


def read_fig1g_data(excel_path: Path):
    """从 Table2 中读取 Macro-F1 的 mean±SD，并转成长表。"""
    df_raw = pd.read_excel(excel_path, sheet_name="Table2")

    metric_col = _find_column(df_raw, ["指标", "metric", "Metric"], required=True)
    model_col = _find_column(df_raw, ["模型", "model", "Model"], required=True)

    # Table2 中指标列通常是合并单元格，读取后需要向下填充。
    df_raw[metric_col] = df_raw[metric_col].ffill()
    df_raw[model_col] = df_raw[model_col].map(normalize_model_name)

    sub = df_raw[
        (df_raw[metric_col].astype(str).str.strip() == "Macro-F1")
        & (df_raw[model_col].isin(MODEL_ORDER))
    ].copy()

    if sub.empty:
        raise ValueError(
            "Table2 中未找到 Macro-F1 行。请确认 Table2 的 指标 列中包含 Macro-F1，且模型名称可被识别。"
        )

    records = []
    for _, row in sub.iterrows():
        model = row[model_col]
        for task in TASK_ORDER:
            # 兼容 Excel 中短横线/长横线写法差异。
            col = task
            if col not in sub.columns:
                alt_cols = [c for c in sub.columns if str(c).replace("–", "-") == task.replace("–", "-")]
                if alt_cols:
                    col = alt_cols[0]
                else:
                    raise ValueError(f"Table2 中找不到任务列：{task}")

            mean_val, sd_val = _parse_mean_sd(row[col])
            records.append(
                {
                    "task": task,
                    "model": model,
                    "mean": mean_val,
                    "sd": sd_val,
                }
            )

    df = pd.DataFrame(records)
    df = df.dropna(subset=["mean"])
    df["task"] = pd.Categorical(df["task"], categories=TASK_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["task", "model"]).reset_index(drop=True)

    print("[INFO] Read sheet: Table2")
    print("[INFO] Macro-F1 summary:")
    print(df.pivot(index="model", columns="task", values="mean").round(4))

    return df


# ============================================================
# 4. 绘图辅助函数
# ============================================================
def _nice_ylim(values, errors):
    """根据 mean±SD 自动生成适合论文图的 y 轴范围。"""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    errors = np.nan_to_num(errors, nan=0.0)

    low = float(np.nanmin(values - errors))
    high = float(np.nanmax(values + errors))
    span = max(high - low, 0.08)
    pad = max(span * 0.18, 0.018)

    ymin = max(0.0, np.floor((low - pad) / 0.05) * 0.05)
    ymax = min(1.0, np.ceil((high + pad) / 0.05) * 0.05)

    # 避免范围太窄导致误差线和文字拥挤。
    if ymax - ymin < 0.18:
        center = (ymax + ymin) / 2
        ymin = max(0.0, center - 0.09)
        ymax = min(1.0, center + 0.09)
        ymin = np.floor(ymin / 0.05) * 0.05
        ymax = np.ceil(ymax / 0.05) * 0.05

    return float(ymin), float(ymax)


def _nice_yticks(ymin, ymax):
    """根据 y 轴跨度选择 0.05 或 0.10 的刻度间隔。"""
    step = 0.05 if (ymax - ymin) <= 0.30 else 0.10
    ticks = np.arange(ymin, ymax + step / 2, step)
    return np.round(ticks, 2)


# ============================================================
# 5. 绘图函数
# ============================================================
def plot_fig1g_holdout_macro_f1():
    df = read_fig1g_data(excel_path)

    # Fig1e 风格：serif 字体、论文风格。
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.8, 9.6),
        sharey=False,
    )

    axes = axes.flatten()
    positions = np.arange(1, len(MODEL_ORDER) + 1)

    for idx, (ax, task) in enumerate(zip(axes, TASK_ORDER)):
        sub = df[df["task"] == task].copy()
        sub = sub.set_index("model").reindex(MODEL_ORDER).reset_index()

        means = sub["mean"].astype(float).values
        sds = sub["sd"].astype(float).values
        sds = np.nan_to_num(sds, nan=0.0)

        bars = ax.bar(
            positions,
            means,
            yerr=sds,
            width=0.52,
            capsize=4,
            color=[COLOR_MAP[m] for m in MODEL_ORDER],
            alpha=0.86,
            edgecolor="#333333",
            linewidth=1.2,
            error_kw=dict(
                ecolor="#333333",
                elinewidth=1.2,
                capthick=1.2,
            ),
        )

        for bar, model in zip(bars, MODEL_ORDER):
            bar.set_alpha(ALPHA_MAP[model])
            bar.set_edgecolor("#333333")
            bar.set_linewidth(1.2)

        ymin, ymax = _nice_ylim(means, sds)
        yticks = _nice_yticks(ymin, ymax)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(yticks)

        # 标注 Single-task 与 Our method 的 mean。
        label_offset = (ymax - ymin) * 0.035
        top_margin = (ymax - ymin) * 0.035
        for pos, model, mean_val, sd_val in zip(positions, MODEL_ORDER, means, sds):
            if model not in ANNOTATE_MEAN_FOR:
                continue
            label_y = min(mean_val + sd_val + label_offset, ymax - top_margin)
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

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [XTICK_LABEL_MAP[m] for m in MODEL_ORDER],
            rotation=28,
            ha="right",
            fontsize=10.5,
        )
        ax.set_xlabel("Models", fontsize=13, fontweight="bold", labelpad=-12)

        # 2 行 3 列时，左侧两张图显示 y 轴标题，避免过度重复。
        if idx in [0, 3]:
            ax.set_ylabel("Holdout Macro-F1", fontsize=13, fontweight="bold")
        else:
            ax.set_ylabel("")

        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.38)
        ax.set_axisbelow(True)

        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.2)
            ax.spines[spine].set_color("#333333")

        ax.tick_params(axis="x", labelsize=10.5, width=1.1, length=4)
        ax.tick_params(axis="y", labelleft=True, labelsize=11, width=1.1, length=4)

    # 如需总标题，取消注释下面代码。
    # fig.suptitle(
    #     FIGURE_TITLE,
    #     fontsize=16.5,
    #     fontweight="bold",
    #     y=0.995,
    # )

    # 图例：默认关闭，与 Fig1e 保持一致。
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
        bottom = 0.14
    else:
        bottom = 0.10

    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.94,
        bottom=bottom,
        wspace=0.17,
        hspace=0.44,
    )

    # 保存
    png_path = save_dir / "Fig1g_holdout_macro_f1_2x3.png"
    pdf_path = save_dir / "Fig1g_holdout_macro_f1_2x3.pdf"
    svg_path = save_dir / "Fig1g_holdout_macro_f1_2x3.svg"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved SVG: {svg_path}")

    plt.show()
    return fig, axes


def main():
    plot_fig1g_holdout_macro_f1()


if __name__ == "__main__":
    main()

