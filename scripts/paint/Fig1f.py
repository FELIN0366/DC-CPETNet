# -*- coding: utf-8 -*-
"""
Figure2F grouped bar plot for task-level holdout Accuracy.

输入：
    RESULT2_Table_Fig2A_Fig2B_completed.xlsx
    sheet = Fig2F_data

说明：
1. Fig2F_data 的结构与 Fig2A_data 相同，但指标由 Macro-F1 改为 Accuracy。
2. excel_prefix 与 legend_label 分开配置：
   - excel_prefix：对应 Excel 中列名前缀，例如 "Shared Bottom mean" / "Shared Bottom SD"
   - legend_label：图例显示名称，可与 Excel 列名前缀不同。
3. 默认固定按照 t1, t2, t3, t4, t5, Mean across t1–t5 排序。
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =========================
# 1. 路径设置
# =========================
excel_path = Path(
    r"xx_path"
)


# 图片输出目录：默认衔接 Excel 所在目录
out_dir = excel_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

# 如果中文乱码，可取消注释
# plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "SimHei"]
# plt.rcParams["axes.unicode_minus"] = False


# =========================
# 2. Figure2F 配置
# =========================
SHEET_NAME = "Fig2F_data"

# 建议 Fig2F 直接按照 t1–t5 顺序展示
TASK_ORDER = ["t1", "t2", "t3", "t4", "t5", "Mean across t1–t5"]

# y 轴范围：Accuracy 通常较高，但为了和 Fig2A 保持视觉一致，默认 0.48–1.00
# 如果你希望突出高值区域，可改成 (0.60, 1.00) 或 (0.70, 1.00)
Y_LIMIT = (0.48, 1.00)


# =========================
# 3. 模型配置
# =========================
# excel_prefix：必须严格对应 Fig2F_data 中的列名前缀
# legend_label：图中图例显示名称，可与 excel_prefix 不一致
MODEL_SPECS = [
    {
        "excel_prefix": "Single-task model",
        "legend_label": "Single-task model",
        "color": "#B8B8B8",
    },
    {
        "excel_prefix": "Shared Bottom",
        "legend_label": "Best MTL baseline",
        "color": "#7FA6C7",
    },
    {
        "excel_prefix": "Our method",
        "legend_label": "Our method",
        "color": "#2C7FB8",
    },

    # 示例：后续新增模型只需要追加：
    # {
    #     "excel_prefix": "MMOE",
    #     "legend_label": "MMoE",
    #     "color": "#8C6BB1",
    # },
    # {
    #     "excel_prefix": "CGC",
    #     "legend_label": "CGC",
    #     "color": "#41AB5D",
    # },
]


# =========================
# 4. 读取 Fig2F 数据并固定排序
# =========================
df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)
df["任务"] = df["任务"].astype(str)

missing_tasks = [t for t in TASK_ORDER if t not in set(df["任务"])]
if missing_tasks:
    raise ValueError(
        f"{SHEET_NAME} 中缺少以下任务行：\n"
        + "\n".join(missing_tasks)
        + "\n\n当前已有任务为：\n"
        + "\n".join(df["任务"].tolist())
    )

df = (
    df.set_index("任务")
      .loc[TASK_ORDER]
      .reset_index()
)

tasks = df["任务"].tolist()


def get_model_values(df, excel_prefix):
    mean_col = f"{excel_prefix} mean"
    sd_col = f"{excel_prefix} SD"

    missing_cols = [c for c in [mean_col, sd_col] if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"{SHEET_NAME} 中缺少以下列：\n"
            + "\n".join(missing_cols)
            + f"\n\n当前 {SHEET_NAME} 可用列名为：\n"
            + "\n".join(map(str, df.columns.tolist()))
            + "\n\n请检查 MODEL_SPECS 中 excel_prefix 是否与 Excel 列名前缀完全一致。"
        )

    mean = df[mean_col].to_numpy(dtype=float)
    sd = df[sd_col].to_numpy(dtype=float)
    return mean, sd


model_data = []
for spec in MODEL_SPECS:
    mean, sd = get_model_values(df, spec["excel_prefix"])
    model_data.append({
        "excel_prefix": spec["excel_prefix"],
        "legend_label": spec["legend_label"],
        "color": spec.get("color", None),
        "mean": mean,
        "sd": sd,
    })


# =========================
# 5. 基础参数
# =========================
x = np.arange(len(tasks))
n_models = len(model_data)

group_width = 0.72
bar_width = min(0.24, group_width / max(n_models, 1))
offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

fig_width = max(9.2, 1.15 * len(tasks) + 1.0)
fig, ax = plt.subplots(figsize=(fig_width, 5.2))

err_kw = dict(
    ecolor="0.25",
    elinewidth=0.9,
    capsize=2.5,
    capthick=0.9,
)


# =========================
# 6. 绘制 grouped bar
# =========================
all_bars = []

for i, item in enumerate(model_data):
    yerr = np.nan_to_num(item["sd"], nan=0.0)

    bars = ax.bar(
        x + offsets[i],
        item["mean"],
        width=bar_width,
        yerr=yerr,
        color=item["color"],
        edgecolor="none",
        error_kw=err_kw,
        label=item["legend_label"],
    )
    all_bars.append((bars, item["mean"], item["sd"]))


# =========================
# 7. 数值标注
# =========================
def annotate_bars(bars, values, sds, dy=0.010):
    for bar, val, sd in zip(bars, values, sds):
        if np.isnan(val):
            continue
        height = bar.get_height()
        sd_for_text = 0 if np.isnan(sd) else sd
        y_text = height + sd_for_text + dy
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


for bars, values, sds in all_bars:
    annotate_bars(bars, values, sds)


# =========================
# 8. 坐标轴与风格
# =========================
ax.set_xticks(x)
ax.set_xticklabels(tasks)

ax.set_ylabel("Holdout Accuracy",fontweight="bold")
ax.set_xlabel("Task",fontweight="bold")
# ax.set_title("Figure 2F. Task-level holdout Accuracy comparison")

if Y_LIMIT is not None:
    ax.set_ylim(*Y_LIMIT)

ax.grid(axis="y", alpha=0.22, linewidth=0.8)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

if "Mean across t1–t5" in tasks:
    mean_idx = tasks.index("Mean across t1–t5")
    ax.axvline(mean_idx - 0.5, color="0.75", linestyle="--", linewidth=0.8)

ax.legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.02, 1.00),
)

fig.tight_layout()


# =========================
# 9. 保存
# =========================
out_png = out_dir / "Figure2F_task_level_holdout_accuracy_grouped_bar.png"
out_svg = out_dir / "Figure2F_task_level_holdout_accuracy_grouped_bar.svg"
out_pdf = out_dir / "Figure2F_task_level_holdout_accuracy_grouped_bar.pdf"

fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")

plt.show()

print(f"已保存：{out_png}")
print(f"已保存：{out_svg}")
print(f"已保存：{out_pdf}")

