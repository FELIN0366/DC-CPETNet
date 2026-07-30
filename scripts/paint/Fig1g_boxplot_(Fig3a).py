# -*- coding: utf-8 -*-
"""
Fig. 1g: Holdout Macro-F1 boxplots across t1–t5 and mean performance.

输入 Excel sheet:
    Fig1g_data

数据格式要求:
    Fig1g_data 中每一行对应一个 model × task × fold 的 fold-level 结果，至少包含：
        任务 / task
        模型 / model
        fold
        Holdout Macro-F1 / macro_f1

输出:
    Fig1g_holdout_macro_f1_boxplots_2x3.png
    Fig1g_holdout_macro_f1_boxplots_2x3.pdf
    Fig1g_holdout_macro_f1_boxplots_2x3.svg

注意：
    这是箱型图脚本。不会读取 Table2 的 mean±SD，也不会使用 ax.bar。
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


def _excel_has_sheet(path: Path, sheet_name: str) -> bool:
    try:
        return sheet_name in pd.ExcelFile(path).sheet_names
    except Exception:
        return False


fallback_paths = [
    Path(__file__).resolve().with_name("RESULT2_Table_Fig2A_Fig2B_completed.xlsx"),
    Path(__file__).resolve().with_name("RESULT2_Table_Fig2A_Fig2B_completed(2).xlsx"),
    Path(__file__).resolve().with_name("RESULT2_Table_Fig2A_Fig2B_completed(1).xlsx"),
    Path(r"xx_path"),
    Path(r"xx_path"),
]

# 优先选择包含 Fig1g_data 的 Excel，避免误读只有 Table2 的旧版文件。
if not excel_path.exists() or not _excel_has_sheet(excel_path, "Fig1g_data"):
    for p in fallback_paths:
        if p.exists() and _excel_has_sheet(p, "Fig1g_data"):
            print(f"[INFO] Use fallback Excel path with Fig1g_data: {p}")
            excel_path = p
            break

if not excel_path.exists():
    raise FileNotFoundError(
        "未找到 Excel 文件。请检查 excel_path 是否正确，或将 Excel 与本脚本放在同一文件夹。\n"
        f"Current excel_path: {excel_path}"
    )

if not _excel_has_sheet(excel_path, "Fig1g_data"):
    raise ValueError(
        "当前 Excel 中找不到 Fig1g_data。\n"
        "箱型图需要 fold-level 原始值，不能使用 Table2 或 Full_holdout_metrics 的 mean±SD 汇总值反推。"
    )

save_dir = excel_path.parent
save_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 全局绘图配置：复用 Fig1e / Fig2E 的论文版式
# ============================================================
TASK_ORDER = ["Mean across t1–t5","t1", "t2", "t3", "t4", "t5", ]

TASK_TITLE_MAP = {
    "t1": "t1: Weber functional class",
    "t2": "t2: Exercise capacity",
    "t3": "t3: Exercise ECG response",
    "t4": "t4: Breathing reserve",
    "t5": "t5: Heart-rate reserve",
    "Mean across t1–t5": "Mean across t1–t5",
}

MODEL_ORDER = [
    "Our method",
    "Shared Bottom",
    "MMOE",
    "CGC",
    "ADATT",
    "Single-task model",
]

XTICK_LABEL_MAP = {
    "Single-task model": "Single-task\nmodel",
    "Shared Bottom": "Shared-bottom\nMTL",
    "MMOE": "MMoE",
    "CGC": "CGC",
    "ADATT": "AdaTT",
    "Our method": "DC-CPETNet",
}

LEGEND_LABEL_MAP = {
    "Single-task model": "Single-task model",
    "Shared Bottom": "Shared-bottom MTL",
    "MMOE": "MMoE",
    "CGC": "CGC",
    "ADATT": "AdaTT",
    "Our method": "Our method",
}

COLOR_MAP = {
    "Single-task model": "#1F77B4",
    "Shared Bottom": "#8C564B",
    "MMOE": "#FF7F0E",
    "CGC": "#9467BD",
    "ADATT": "#2CA02C",
    "Our method": "#D62728",
}

ALPHA_MAP = {
    "Single-task model": 0.88,
    "Shared Bottom": 0.68,
    "MMOE": 0.68,
    "CGC": 0.68,
    "ADATT": 0.68,
    "Our method": 0.90,
}

SHOW_LEGEND = False
ANNOTATE_MEAN_FOR = MODEL_ORDER.copy()  # 标注每个模型的 fold-level mean

# 对 t1–t5 + mean 统一使用自动 y 轴；若希望固定某个范围，可在这里指定。
# 例如："t3": (0.60, 0.92)
Y_LIM_MAP = {
    "t1": (0.70, 0.95),
    "t2": (0.70, 0.90),
    "t3": (0.40, 0.75),
    "t4": (0.65, 0.85),
    "t5": (0.5, 1.00),
    "Mean across t1–t5": (0.65, 0.90),
}

# ============================================================
# 3. 数据读取与清洗
# ============================================================
def normalize_model_name(name):
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


def normalize_task_name(name):
    s = str(name).strip()
    s_norm = s.replace("-", "–")
    if s_norm in ["t1", "T1"]:
        return "t1"
    if s_norm in ["t2", "T2"]:
        return "t2"
    if s_norm in ["t3", "T3"]:
        return "t3"
    if s_norm in ["t4", "T4"]:
        return "t4"
    if s_norm in ["t5", "T5"]:
        return "t5"
    if s_norm in ["Mean across t1–t5", "Mean across t1-t5", "mean across t1–t5", "mean across t1-t5"]:
        return "Mean across t1–t5"
    return s


def _find_column(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    # 兼容列名两端空格
    stripped_map = {str(c).strip(): c for c in df.columns}
    for c in candidates:
        if c in stripped_map:
            return stripped_map[c]
    if required:
        raise ValueError(
            "找不到所需列，候选列名为：\n"
            + "\n".join(candidates)
            + "\n\n当前 Excel 中可用列名为：\n"
            + "\n".join(map(str, df.columns.tolist()))
        )
    return None


def read_fig1g_data(excel_path: Path):
    """从 Fig1g_data 中读取 fold-level Holdout Macro-F1。"""
    df_raw = pd.read_excel(excel_path, sheet_name="Fig1g_data")

    task_col = _find_column(df_raw, ["任务", "task", "Task", "task_key"], required=True)
    model_col = _find_column(df_raw, ["模型", "model", "Model", "model_name"], required=True)
    fold_col = _find_column(df_raw, ["fold", "Fold", "折", "fold_id"], required=True)
    value_col = _find_column(
        df_raw,
        ["Holdout Macro-F1", "Macro-F1", "macro_f1", "macro-f1", "holdout_macro_f1", "value"],
        required=True,
    )

    df = pd.DataFrame()
    df["task"] = df_raw[task_col].map(normalize_task_name)
    df["model"] = df_raw[model_col].map(normalize_model_name)
    df["fold"] = pd.to_numeric(df_raw[fold_col], errors="coerce")
    df["macro_f1"] = pd.to_numeric(df_raw[value_col], errors="coerce")

    df = df[df["task"].isin(TASK_ORDER)].copy()
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df = df.dropna(subset=["fold", "macro_f1"])

    if df.empty:
        raise ValueError("读取后 Fig1g_data 为空，请检查 task/model/fold/Holdout Macro-F1 列。")

    df["task"] = pd.Categorical(df["task"], categories=TASK_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["task", "model", "fold"]).reset_index(drop=True)

    print("[INFO] Read sheet: Fig1g_data")
    print("[INFO] Fold-level data summary:")
    print(
        df.groupby(["task", "model"], observed=False)["macro_f1"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
    )
    return df


# ============================================================
# 4. 绘图辅助函数
# ============================================================
def _nice_ylim_for_boxplot(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0, 1.0

    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, 0.08)
    pad = max(span * 0.22, 0.018)

    ymin = max(0.0, np.floor((low - pad) / 0.05) * 0.05)
    ymax = min(1.0, np.ceil((high + pad) / 0.05) * 0.05)

    if ymax - ymin < 0.18:
        center = (ymax + ymin) / 2
        ymin = max(0.0, center - 0.09)
        ymax = min(1.0, center + 0.09)
        ymin = np.floor(ymin / 0.05) * 0.05
        ymax = np.ceil(ymax / 0.05) * 0.05

    return float(ymin), float(ymax)


def _nice_yticks(ymin, ymax):
    step = 0.05 if (ymax - ymin) <= 0.30 else 0.10
    ticks = np.arange(ymin, ymax + step / 2, step)
    return np.round(ticks, 2)


# ============================================================
# 5. 绘图函数
# ============================================================
def plot_fig1g_holdout_macro_f1_boxplots():
    df = read_fig1g_data(excel_path)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16.8, 10.5),   # 增加整体高度，仅拉高子图
        sharey=False,
    )
    axes = axes.flatten()
    positions = np.arange(1, len(MODEL_ORDER) + 1)

    for idx, (ax, task) in enumerate(zip(axes, TASK_ORDER)):
        sub = df[df["task"] == task].copy()

        data = []
        task_values = []
        for model in MODEL_ORDER:
            vals = sub.loc[sub["model"] == model, "macro_f1"].dropna().to_numpy(dtype=float)
            data.append(vals)
            task_values.extend(vals.tolist())

        # 真正的箱型图：这里使用 ax.boxplot，不使用 ax.bar。
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

        if task in Y_LIM_MAP:
            ymin, ymax = Y_LIM_MAP[task]
        else:
            ymin, ymax = _nice_ylim_for_boxplot(task_values)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(_nice_yticks(ymin, ymax))

        # 标注每个模型的 fold-level mean。
        label_offset = (ymax - ymin) * 0.035
        top_margin = (ymax - ymin) * 0.035
        for pos, model, vals in zip(positions, MODEL_ORDER, data):
            if model not in ANNOTATE_MEAN_FOR or len(vals) == 0:
                continue
            mean_val = float(np.mean(vals))
            max_val = float(np.max(vals))
            label_y = min(max_val + label_offset, ymax - top_margin)
            is_our_method = (model == "Our method")
            ax.text(
                pos,
                label_y,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11.0 if is_our_method else 10.5,
                fontweight="bold" if is_our_method else "normal",
                color="#111111" if is_our_method else "#222222",
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
        # 仅加粗 Our method 的 x 轴标签。
        for tick_label, model in zip(ax.get_xticklabels(), MODEL_ORDER):
            if model == "Our method":
                tick_label.set_fontweight("bold")
                tick_label.set_color("#111111")
        ax.set_xlabel("Models", fontsize=13, fontweight="bold", labelpad=-12)

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
        hspace=0.35,
    )

    png_path = save_dir / "Fig1g_holdout_macro_f1_boxplots_2x3_labeled.png"
    pdf_path = save_dir / "Fig1g_holdout_macro_f1_boxplots_2x3_labeled.pdf"
    svg_path = save_dir / "Fig1g_holdout_macro_f1_boxplots_2x3_labeled.svg"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved SVG: {svg_path}")

    plt.show()
    return fig, axes


def main():
    plot_fig1g_holdout_macro_f1_boxplots()


if __name__ == "__main__":
    main()

