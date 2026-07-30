# -*- coding: utf-8 -*-
"""
Result 1.7 subgroup analysis data-preparation script

用途
----
1. 读取 Full / Age / Sex / T6 subgroup 的 MTL-style holdout 结果 Excel。
2. 输出当前正文主图、补充 task-wise 图和补充表所需的 4 个 sheet：
   - Fig_subgroup_main_mean_data：正文主图数据，仅包含 Mean across t1–t5 的 Accuracy / Macro-F1。
   - Fig_subgroup_task_profile_data：补充 task-wise 图数据，保留 t1–t5 + Mean across t1–t5。
   - Per_task_holdout_metrics：逐任务完整性能，保留原逻辑。
   - TableS_all_ablation：补充表矩阵，保留原逻辑。

主图最终版式
------------
Figure X | Task-wise robustness across age, sex, and disease-mechanism subgroups.
- a: Age subgroup，组别 6 个：Age <30, Age 30-39, Age 40-49, Age 50-59, Age 60-69, Age >=70
- b: Sex subgroup，组别 2 个：Female, Male
- c: T6 disease-mechanism subgroup，组别 4 个：CMP, HTx, IHD, PVD
- 每个子图均使用同样的两个指标：Accuracy 和 Macro-F1
- 横轴统一为：t1, t2, t3, t4, t5, Mean across t1-t5
- Full: Overall holdout cohort 作为黑色虚线参考线，在三个 panel 中重复提供。

输入 Excel 要求
--------------
每个结果文件包含 holdout_fold1、holdout_fold2、... sheet。
每个 holdout_fold sheet 至少包含 task_key/task 列，以及 accuracy / precision / recall /
macro_f1 / auc(or auroc) / auprc 等指标列。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. 基础配置
# ============================================================

TASK_ORDER = ["t1", "t2", "t3", "t4", "t5"]
TASK_MEAN_LABEL = "Mean across t1–t5"
TASK_ORDER_WITH_MEAN = TASK_ORDER + [TASK_MEAN_LABEL]

TASK_CLINICAL_NAME = {
    "t1": "CPET functional class",
    "t2": "Exercise capacity",
    "t3": "exercise ECG interpretation",
    "t4": "ventilatory function",
    "t5": "heart-rate reserve",
    TASK_MEAN_LABEL: "Mean across t1–t5",
}

# 这里按你最新叙述：t1/t2 为四分类，t3-t5 为二分类。
TASK_TYPE = {
    "t1": "4-class",
    "t2": "4-class",
    "t3": "binary",
    "t4": "binary",
    "t5": "binary",
}

TASK_WEIGHTS = {t: 1.0 for t in TASK_ORDER}
RANKING_TASKS = ["t3", "t4", "t5"]
TREAT_ZERO_AUPRC_AS_MISSING = True

REFERENCE_MODEL_VARIANT = "Overall holdout cohort"

# 主图只用 Accuracy + Macro-F1，避免 Age/Sex/T6 挑指标嫌疑。
MAIN_FIG_METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
]

# TableS 继续保留 Accuracy / Macro-F1 / Macro-recall。
METRIC_SPECS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("recall", "Macro-recall"),
]

MAIN_EXPERIMENTS = [
    "Full",
    "E0", "E1", "E2", "E3", "E00", "E4",   # Age: <30, 30-39, 40-49, 50-59, 60-69, >=70
    "E5", "E6",                             # Sex
    "E7", "E8", "E9", "E10",               # T6
]
ALL_EXPERIMENTS = MAIN_EXPERIMENTS

# 用于主图分组、排序、颜色、线型。
SUBGROUP_META = {
    "Full": {
        "panel": "Reference",
        "panel_letter": "ref",
        "subgroup_type": "Full",
        "subgroup_label": "Full cohort",
        "subgroup_order": 0,
        "line_color": "#222222",
        "line_style": "--",
        "marker": "o",
        "is_reference": True,
    },
    "E0": {
        "panel": "Age",
        "panel_letter": "a",
        "subgroup_type": "Age",
        "subgroup_label": "Age <30",
        "subgroup_order": 1,
        "line_color": "#BBD3E9",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E00": {
        "panel": "Age",
        "panel_letter": "a",
        "subgroup_type": "Age",
        "subgroup_label": "Age 60-69",
        "subgroup_order": 5,
        "line_color": "#2181DA",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E1": {
        "panel": "Age",
        "panel_letter": "a",
        "subgroup_type": "Age",
        "subgroup_label": "Age 30-39",
        "subgroup_order": 2,
        "line_color": "#A6C8E8",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E2": {
        "panel": "Age",
        "panel_letter": "a",
        "subgroup_type": "Age",
        "subgroup_label": "Age 40-49",
        "subgroup_order": 3,
        "line_color": "#6FA8DC",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E3": {
        "panel": "Age",
        "panel_letter": "a",
        "subgroup_type": "Age",
        "subgroup_label": "Age 50-59",
        "subgroup_order": 4,
        "line_color": "#2F75B5",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E4": {
        "panel": "Age",
        "panel_letter": "a",
        "subgroup_type": "Age",
        "subgroup_label": "Age >=70",
        "subgroup_order": 6,
        "line_color": "#1F4E79",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E5": {
        "panel": "Sex",
        "panel_letter": "b",
        "subgroup_type": "Sex",
        "subgroup_label": "Female",
        "subgroup_order": 1,
        "line_color": "#9B59B6",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E6": {
        "panel": "Sex",
        "panel_letter": "b",
        "subgroup_type": "Sex",
        "subgroup_label": "Male",
        "subgroup_order": 2,
        "line_color": "#2CA6A4",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E7": {
        "panel": "T6 disease mechanism",
        "panel_letter": "c",
        "subgroup_type": "T6",
        "subgroup_label": "CMP",
        "subgroup_order": 1,
        "line_color": "#2C7FB8",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E8": {
        "panel": "T6 disease mechanism",
        "panel_letter": "c",
        "subgroup_type": "T6",
        "subgroup_label": "HTx",
        "subgroup_order": 2,
        "line_color": "#70AD47",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E9": {
        "panel": "T6 disease mechanism",
        "panel_letter": "c",
        "subgroup_type": "T6",
        "subgroup_label": "IHD",
        "subgroup_order": 3,
        "line_color": "#F4B183",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
    "E10": {
        "panel": "T6 disease mechanism",
        "panel_letter": "c",
        "subgroup_type": "T6",
        "subgroup_label": "PVD",
        "subgroup_order": 4,
        "line_color": "#C55A9E",
        "line_style": "-",
        "marker": "o",
        "is_reference": False,
    },
}

PANEL_ORDER = {
    "Age": 1,
    "Sex": 2,
    "T6 disease mechanism": 3,
}

METRIC_PANEL_ORDER = {
    "Accuracy": 1,
    "Macro-F1": 2,
}

RECOMMENDED_YLIM = {
    "Accuracy": (0.65, 0.95),
    "Macro-F1": (0.45, 0.90),
}

# 正文主图仅展示 Mean across t1–t5，因此 y 轴可以适当收窄。
RECOMMENDED_MAIN_YLIM = {
    "Accuracy": (0.78, 0.90),
    "Macro-F1": (0.68, 0.83),
}

MAIN_MEAN_PANEL_META = {
    "Age": {
        "panel": "Age subgroup",
        "panel_letter": "a",
        "panel_order": 1,
        "subgroup_experiments": ["E0", "E1", "E2", "E3", "E00", "E4"],
    },
    "Sex": {
        "panel": "Sex subgroup",
        "panel_letter": "b",
        "panel_order": 2,
        "subgroup_experiments": ["E5", "E6"],
    },
    "T6 disease mechanism": {
        "panel": "Disease-mechanism subgroup",
        "panel_letter": "c",
        "panel_order": 3,
        "subgroup_experiments": ["E7", "E8", "E9", "E10"],
    },
}


# ============================================================
# 0.1 输入文件配置
# ============================================================

ABLATION_RESULT_FILES = [
    {
        "experiment": "Full",
        "model_variant": "Overall holdout cohort",
        "path": r"xx_path",
    },
    {
        "experiment": "E0",
        "model_variant": "Age <30",
        "path": r"xx_path",
    },
    {
        "experiment": "E00",
        "model_variant": "Age 60-69",
        "path": r"xx_path",
    },
    {
        "experiment": "E1",
        "model_variant": "Age 30-39",
        "path": r"xx_path",
    },
    {
        "experiment": "E2",
        "model_variant": "Age 40-49",
        "path": r"xx_path",
    },
    {
        "experiment": "E3",
        "model_variant": "Age 50-59",
        "path": r"xx_path",
    },
    {
        "experiment": "E4",
        "model_variant": "Age >=70",
        "path": r"xx_path",
    },
    {
        "experiment": "E5",
        "model_variant": "Female",
        "path": r"xx_path",
    },
    {
        "experiment": "E6",
        "model_variant": "Male",
        "path": r"xx_path",
    },
    {
        "experiment": "E7",
        "model_variant": "CMP",
        "path": r"xx_path",
    },
    {
        "experiment": "E8",
        "model_variant": "HTx",
        "path": r"xx_path",
    },
    {
        "experiment": "E9",
        "model_variant": "IHD",
        "path": r"xx_path",
    },
    {
        "experiment": "E10",
        "model_variant": "PVD",
        "path": r"xx_path",
    },
]

# ============================================================
# 1. 通用工具函数
# ============================================================

def sample_std(values):
    vals = [float(v) for v in values if pd.notna(v)]
    if len(vals) < 2:
        return np.nan
    return float(np.std(vals, ddof=1))


def safe_mean(values):
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return np.nan
    return float(np.mean(vals))


def fmt_mean_sd(mean, sd, digits=4):
    if pd.isna(mean):
        return ""
    if pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def normalize_metric_name(name):
    if name is None or pd.isna(name):
        return None

    s = str(name).strip().lower()
    s = s.replace("（二分类）", "").replace("(二分类)", "").strip()
    s = s.replace(" ", "_")

    mapping = {
        "acc": "accuracy",
        "accuracy": "accuracy",
        "precision": "precision",
        "macro_precision": "precision",
        "recall": "recall",
        "macro_recall": "recall",
        "f1": "f1",
        "f1-score": "f1",
        "f1_score": "f1",
        "macro-f1": "macro_f1",
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
        "weighted-f1": "weighted_f1",
        "auc": "auroc",
        "auroc": "auroc",
        "roc_auc": "auroc",
        "auprc": "auprc",
        "pr_auc": "auprc",
        "loss": "loss",
        "threshold": "threshold",
        "minority_f1": "minority_f1",
        "minority_recall": "minority_recall",
        "minority_precision": "minority_precision",
        "pred_minor_rate": "pred_minor_rate",
        "true_minor_rate": "true_minor_rate",
    }

    return mapping.get(s, s)


def make_display_label(experiment, model_variant):
    if experiment == "Full":
        return f"Full: {model_variant}"
    return f"{experiment}: {model_variant}"


def parse_cli_result_files(result_file_args):
    """
    命令行格式：
      --result_file "E1|Age <40|xx_path"
    """
    specs = []
    for raw in result_file_args or []:
        parts = raw.split("|", 2)
        if len(parts) != 3:
            raise ValueError(
                f"--result_file 格式错误：{raw}\n"
                f"正确格式：--result_file \"E1|Age <40|xx_path\""
            )
        experiment, model_variant, path = [p.strip() for p in parts]
        specs.append({"experiment": experiment, "model_variant": model_variant, "path": path})
    return specs


def normalize_result_specs(raw_specs):
    out = []
    seen_experiments = set()
    seen_labels = set()

    for idx, item in enumerate(raw_specs, start=1):
        experiment = str(item.get("experiment", "")).strip()
        model_variant = str(item.get("model_variant", "")).strip()
        path = str(item.get("path", "")).strip()

        if not experiment:
            raise ValueError(f"第 {idx} 项缺少 experiment。")
        if not model_variant:
            raise ValueError(f"第 {idx} 项缺少 model_variant。")
        if not path:
            raise ValueError(f"第 {idx} 项缺少 path。")

        label = make_display_label(experiment, model_variant)

        if experiment in seen_experiments:
            raise ValueError(f"存在重复 experiment：{experiment}")
        if label in seen_labels:
            raise ValueError(f"存在重复 display label：{label}")

        seen_experiments.add(experiment)
        seen_labels.add(label)

        out.append({
            "experiment": experiment,
            "model_variant": model_variant,
            "display_label": label,
            "path": Path(path),
        })

    if "Full" not in seen_experiments:
        raise ValueError("必须提供 Full / Overall holdout cohort 结果，作为参考线。")

    return out


def filter_specs_by_experiments(specs, experiments):
    exp_set = set(experiments)
    out = [s for s in specs if s["experiment"] in exp_set]
    order_map = {e: i for i, e in enumerate(experiments)}
    out = sorted(out, key=lambda s: order_map.get(s["experiment"], 999))

    missing = [e for e in experiments if e not in {s["experiment"] for s in out}]
    if missing:
        print("[警告] 以下实验未在输入文件中找到，将不会输出：", ", ".join(missing))
    return out


# ============================================================
# 2. 读取 MTL-style holdout_fold 数据
# ============================================================

def read_mtl_style_result(xlsx_path, spec):
    xls = pd.ExcelFile(xlsx_path)
    records = []

    fold_sheets = []
    for s in xls.sheet_names:
        m = re.match(r"holdout_fold\s*(\d+)$", str(s), flags=re.IGNORECASE)
        if m:
            fold_sheets.append((int(m.group(1)), s))
    fold_sheets = sorted(fold_sheets, key=lambda x: x[0])

    if not fold_sheets:
        raise ValueError(f"{xlsx_path} 中未找到 holdout_fold1~holdout_fold5 sheet。")

    metric_candidates = [
        "accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1",
        "auroc", "auprc", "loss", "threshold",
        "minority_f1", "minority_recall", "minority_precision",
        "pred_minor_rate", "true_minor_rate",
    ]

    for fold, sheet in fold_sheets:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)

        rename_dict = {}
        for c in df.columns:
            c_str = str(c).strip()
            if c_str in ["task_key", "task"]:
                rename_dict[c] = "task"
            elif c_str == "task_name":
                rename_dict[c] = "task_name"
            else:
                rename_dict[c] = normalize_metric_name(c)
        df = df.rename(columns=rename_dict)

        if "task" not in df.columns:
            raise ValueError(f"{xlsx_path} 的 {sheet} 中未找到 task_key/task 列。")

        for _, row in df.iterrows():
            task = str(row.get("task")).strip()
            if task not in TASK_ORDER:
                continue

            for metric in metric_candidates:
                if metric not in df.columns or pd.isna(row.get(metric)):
                    continue

                val = row.get(metric)
                if isinstance(val, str) and val.strip().upper() in ["N/A", "NA", ""]:
                    continue

                try:
                    val = float(val)
                except Exception:
                    continue

                if metric == "auprc" and TREAT_ZERO_AUPRC_AS_MISSING and np.isclose(val, 0.0):
                    continue

                records.append({
                    "experiment": spec["experiment"],
                    "model_variant": spec["model_variant"],
                    "display_label": spec["display_label"],
                    "task": task,
                    "fold": fold,
                    "metric": metric,
                    "value": val,
                    "source_file": str(xlsx_path),
                })

    if not records:
        raise ValueError(f"未能从 {xlsx_path} 中读取到 holdout fold 数据。")

    return pd.DataFrame(records)


# ============================================================
# 3. 汇总与取值函数
# ============================================================

def summarize_long(df_long):
    return (
        df_long
        .groupby(["experiment", "model_variant", "display_label", "task", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            sd=("value", lambda x: float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan),
            min=("value", "min"),
            max=("value", "max"),
            n=("value", "count"),
        )
    )


def get_fold_metric_value(df_long, display_label, task, metric, fold):
    sub = df_long[
        (df_long["display_label"] == display_label)
        & (df_long["task"] == task)
        & (df_long["metric"] == metric)
        & (df_long["fold"] == fold)
    ]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["value"])


def get_summary_value(summary, display_label, task, metric, stat="mean"):
    sub = summary[
        (summary["display_label"] == display_label)
        & (summary["task"] == task)
        & (summary["metric"] == metric)
    ]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0][stat])


def fold_task_average(df_long, display_label, fold, tasks, metric="macro_f1", weights=None, require_all=True):
    vals = []
    ws = []
    for task in tasks:
        val = get_fold_metric_value(df_long, display_label, task, metric, fold)
        if pd.notna(val):
            vals.append(float(val))
            ws.append(float(weights.get(task, 1.0)) if weights else 1.0)

    if require_all and len(vals) != len(tasks):
        return np.nan
    if not vals:
        return np.nan
    if weights:
        return float(np.average(vals, weights=ws))
    return float(np.mean(vals))


def aggregate_fold_values(values):
    vals = [v for v in values if pd.notna(v)]
    return safe_mean(vals), sample_std(vals)


def get_task_or_mean_metric(df_long, summary, display_label, task_or_mean, metric, weighted=False):
    if task_or_mean in TASK_ORDER:
        return (
            get_summary_value(summary, display_label, task_or_mean, metric, "mean"),
            get_summary_value(summary, display_label, task_or_mean, metric, "sd"),
        )

    folds = sorted(df_long["fold"].dropna().unique())
    fold_vals = [
        fold_task_average(
            df_long,
            display_label,
            fold,
            TASK_ORDER,
            metric=metric,
            weights=TASK_WEIGHTS if weighted else None,
        )
        for fold in folds
    ]
    return aggregate_fold_values(fold_vals)


def get_task_or_mean_fold_values(df_long, display_label, task_or_mean, metric):
    folds = sorted(df_long["fold"].dropna().unique())
    values = []
    for fold in folds:
        if task_or_mean in TASK_ORDER:
            values.append(get_fold_metric_value(df_long, display_label, task_or_mean, metric, fold))
        else:
            values.append(fold_task_average(df_long, display_label, fold, TASK_ORDER, metric=metric))
    return folds, values


# ============================================================
# 4. Per_task_holdout_metrics：保留原逻辑
# ============================================================

def build_per_task_holdout_metrics(summary, main_specs):
    rows = []

    reference_label = [
        s["display_label"]
        for s in main_specs
        if s["model_variant"] == REFERENCE_MODEL_VARIANT
    ]
    if not reference_label:
        reference_label = [s["display_label"] for s in main_specs if s["experiment"] == "Full"]
    if not reference_label:
        raise ValueError("正文主实验中未找到 Full / Overall holdout cohort。")
    reference_label = reference_label[0]

    metrics = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("macro_f1", "Macro-F1"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
    ]
    delta_metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
    ]

    for spec in main_specs:
        label = spec["display_label"]
        combined = make_display_label(spec["experiment"], spec["model_variant"])
        meta = SUBGROUP_META.get(spec["experiment"], {})

        for task in TASK_ORDER:
            row = {
                "Experiment / model variant": combined,
                "Subgroup type": meta.get("subgroup_type", ""),
                "Subgroup label": meta.get("subgroup_label", spec["model_variant"]),
                "Task": task,
                "Clinical meaning": TASK_CLINICAL_NAME[task],
                "Task type": TASK_TYPE[task],
            }

            for metric_key, metric_label in metrics:
                mean_val = get_summary_value(summary, label, task, metric_key, "mean")
                sd_val = get_summary_value(summary, label, task, metric_key, "sd")
                row[metric_label] = fmt_mean_sd(mean_val, sd_val)

            for metric_key, metric_label in delta_metrics:
                val = get_summary_value(summary, label, task, metric_key, "mean")
                ref_val = get_summary_value(summary, reference_label, task, metric_key, "mean")
                row[f"Δ {metric_label}"] = val - ref_val if pd.notna(val) and pd.notna(ref_val) else np.nan

            rows.append(row)

    out = pd.DataFrame(rows)
    desired_cols = [
        "Experiment / model variant", "Subgroup type", "Subgroup label",
        "Task", "Clinical meaning", "Task type",
        "Accuracy", "Precision", "Recall", "Macro-F1", "AUROC", "AUPRC",
        "Δ Accuracy", "Δ Macro-F1", "Δ AUROC", "Δ AUPRC",
    ]
    for col in desired_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[desired_cols]


# ============================================================
# 5. 主图数据：一个大 sheet 囊括所有内容
# ============================================================

def build_fig_subgroup_task_profile_data(df_long, summary, main_specs):
    """
    输出长表，直接服务最终主图：
      - 3 个 panel：Age / Sex / T6 disease mechanism
      - 2 个 metric panel：Accuracy / Macro-F1
      - x 轴：t1-t5 + Mean across t1-t5
      - Full cohort 作为每个 panel 的 dashed reference line
    """
    spec_by_exp = {s["experiment"]: s for s in main_specs}
    if "Full" not in spec_by_exp:
        raise ValueError("main_specs 中缺少 Full。")

    panel_to_experiments = {
        "Age": ["Full", "E0", "E1", "E2", "E3", "E00", "E4"],
        "Sex": ["Full", "E5", "E6"],
        "T6 disease mechanism": ["Full", "E7", "E8", "E9", "E10"],
    }

    rows = []

    for panel_name, experiments in panel_to_experiments.items():
        for metric_key, metric_label in MAIN_FIG_METRICS:
            y_min, y_max = RECOMMENDED_YLIM[metric_label]

            for exp in experiments:
                if exp not in spec_by_exp:
                    print(f"[警告] {panel_name} panel 缺少 {exp}，跳过。")
                    continue

                spec = spec_by_exp[exp]
                label = spec["display_label"]
                meta = SUBGROUP_META.get(exp, {})

                # Full 在每个 panel 里作为 reference，panel 字段要改成当前 panel。
                if exp == "Full":
                    panel_letter = {"Age": "a", "Sex": "b", "T6 disease mechanism": "c"}[panel_name]
                    subgroup_type = panel_name
                    subgroup_label = "Full cohort"
                    subgroup_order = 0
                    line_color = "#222222"
                    line_style = "--"
                    marker = "o"
                    is_reference = True
                else:
                    panel_letter = meta.get("panel_letter", "")
                    subgroup_type = meta.get("subgroup_type", panel_name)
                    subgroup_label = meta.get("subgroup_label", spec["model_variant"])
                    subgroup_order = meta.get("subgroup_order", 999)
                    line_color = meta.get("line_color", "#666666")
                    line_style = meta.get("line_style", "-")
                    marker = meta.get("marker", "o")
                    is_reference = bool(meta.get("is_reference", False))

                for task_order, task in enumerate(TASK_ORDER_WITH_MEAN, start=1):
                    mean_val, sd_val = get_task_or_mean_metric(df_long, summary, label, task, metric_key)
                    folds, fold_values = get_task_or_mean_fold_values(df_long, label, task, metric_key)

                    row = {
                        "figure": "FigX_subgroup_task_profile",
                        "panel_letter": panel_letter,
                        "panel": panel_name,
                        "panel_order": PANEL_ORDER[panel_name],
                        "metric": metric_label,
                        "metric_key": metric_key,
                        "metric_panel_order": METRIC_PANEL_ORDER[metric_label],
                        "subgroup_type": subgroup_type,
                        "experiment": exp,
                        "model_variant": spec["model_variant"],
                        "display_label": make_display_label(exp, spec["model_variant"]),
                        "subgroup_label": subgroup_label,
                        "subgroup_order": subgroup_order,
                        "task": task,
                        "task_order": task_order,
                        "clinical_meaning": TASK_CLINICAL_NAME[task],
                        "task_type": TASK_TYPE.get(task, "-"),
                        "mean": mean_val,
                        "sd": sd_val,
                        "mean±sd": fmt_mean_sd(mean_val, sd_val),
                        "errorbar_in_main": bool(task == TASK_MEAN_LABEL),
                        "is_reference": is_reference,
                        "line_color": line_color,
                        "line_style": line_style,
                        "marker": marker,
                        "recommended_ymin": y_min,
                        "recommended_ymax": y_max,
                        "x_axis_group": "task-wise" if task in TASK_ORDER else "summary",
                        "notes": "Full cohort dashed reference" if is_reference else "subgroup line",
                    }

                    for fold, val in zip(folds, fold_values):
                        row[f"fold{int(fold)}"] = val

                    rows.append(row)

    out = pd.DataFrame(rows)

    sort_cols = ["panel_order", "metric_panel_order", "subgroup_order", "task_order"]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    # 列顺序固定，方便后续画图脚本读取。
    preferred_cols = [
        "figure", "panel_letter", "panel", "panel_order",
        "metric", "metric_key", "metric_panel_order",
        "subgroup_type", "experiment", "model_variant", "display_label",
        "subgroup_label", "subgroup_order",
        "task", "task_order", "clinical_meaning", "task_type",
        "mean", "sd", "mean±sd",
        "fold1", "fold2", "fold3", "fold4", "fold5",
        "errorbar_in_main", "is_reference",
        "line_color", "line_style", "marker",
        "recommended_ymin", "recommended_ymax",
        "x_axis_group", "notes",
    ]
    for col in preferred_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[preferred_cols]



# ============================================================
# 6. 正文主图数据：Mean across t1–t5
# ============================================================

def build_fig_subgroup_main_mean_data(df_long, summary, main_specs):
    """
    输出正文主图 long table：
      - 3 个 panel：Age subgroup / Sex subgroup / Disease-mechanism subgroup
      - 2 行指标：Mean Accuracy across t1–t5 / Mean Macro-F1 across t1–t5
      - 每个点为 subgroup 的 t1–t5 mean，误差线为 fold-specific holdout 的 SD
      - Full cohort 不作为彩色点参与比较，而是作为每个 panel 的 reference_value 横向虚线
    """
    spec_by_exp = {s["experiment"]: s for s in main_specs}
    if "Full" not in spec_by_exp:
        raise ValueError("main_specs 中缺少 Full，无法生成 reference_value。")

    full_spec = spec_by_exp["Full"]
    full_label = full_spec["display_label"]

    rows = []

    for _panel_key, panel_cfg in MAIN_MEAN_PANEL_META.items():
        panel_name = panel_cfg["panel"]
        panel_letter = panel_cfg["panel_letter"]
        panel_order = panel_cfg["panel_order"]

        for metric_key, metric_label in MAIN_FIG_METRICS:
            y_min, y_max = RECOMMENDED_MAIN_YLIM[metric_label]

            ref_mean, ref_sd = get_task_or_mean_metric(
                df_long,
                summary,
                full_label,
                TASK_MEAN_LABEL,
                metric_key,
            )
            ref_folds, ref_fold_values = get_task_or_mean_fold_values(
                df_long,
                full_label,
                TASK_MEAN_LABEL,
                metric_key,
            )

            for exp in panel_cfg["subgroup_experiments"]:
                if exp not in spec_by_exp:
                    print(f"[警告] {panel_name} 缺少 {exp}，跳过。")
                    continue

                spec = spec_by_exp[exp]
                label = spec["display_label"]
                meta = SUBGROUP_META.get(exp, {})

                mean_val, sd_val = get_task_or_mean_metric(
                    df_long,
                    summary,
                    label,
                    TASK_MEAN_LABEL,
                    metric_key,
                )
                folds, fold_values = get_task_or_mean_fold_values(
                    df_long,
                    label,
                    TASK_MEAN_LABEL,
                    metric_key,
                )

                row = {
                    "figure": "FigX_subgroup_main_mean",
                    "panel_letter": panel_letter,
                    "panel": panel_name,
                    "panel_key": _panel_key,
                    "panel_order": panel_order,
                    "metric": metric_label,
                    "metric_key": metric_key,
                    "metric_panel_order": METRIC_PANEL_ORDER[metric_label],
                    "subgroup_type": meta.get("subgroup_type", panel_name),
                    "experiment": exp,
                    "model_variant": spec["model_variant"],
                    "display_label": make_display_label(exp, spec["model_variant"]),
                    "subgroup_label": meta.get("subgroup_label", spec["model_variant"]),
                    "subgroup_order": meta.get("subgroup_order", 999),
                    "task": TASK_MEAN_LABEL,
                    "task_order": len(TASK_ORDER_WITH_MEAN),
                    "clinical_meaning": TASK_CLINICAL_NAME[TASK_MEAN_LABEL],
                    "task_type": "-",
                    "mean": mean_val,
                    "sd": sd_val,
                    "mean±sd": fmt_mean_sd(mean_val, sd_val),
                    "reference_label": "Full cohort",
                    "reference_experiment": "Full",
                    "reference_value": ref_mean,
                    "reference_mean": ref_mean,
                    "reference_sd": ref_sd,
                    "reference_mean±sd": fmt_mean_sd(ref_mean, ref_sd),
                    "errorbar_in_main": True,
                    "is_reference": False,
                    "plot_color": meta.get("line_color", "#666666"),
                    "line_color": meta.get("line_color", "#666666"),
                    "line_style": meta.get("line_style", "-"),
                    "marker": meta.get("marker", "o"),
                    "recommended_ymin": y_min,
                    "recommended_ymax": y_max,
                    "notes": "Main-text point-range plot; full cohort shown as horizontal dashed reference line.",
                }

                for fold, val in zip(folds, fold_values):
                    row[f"fold{int(fold)}"] = val

                for fold, val in zip(ref_folds, ref_fold_values):
                    row[f"reference_fold{int(fold)}"] = val

                rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = ["panel_order", "metric_panel_order", "subgroup_order"]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    preferred_cols = [
        "figure", "panel_letter", "panel", "panel_key", "panel_order",
        "metric", "metric_key", "metric_panel_order",
        "subgroup_type", "experiment", "model_variant", "display_label",
        "subgroup_label", "subgroup_order",
        "task", "task_order", "clinical_meaning", "task_type",
        "mean", "sd", "mean±sd",
        "fold1", "fold2", "fold3", "fold4", "fold5",
        "reference_label", "reference_experiment", "reference_value",
        "reference_mean", "reference_sd", "reference_mean±sd",
        "reference_fold1", "reference_fold2", "reference_fold3", "reference_fold4", "reference_fold5",
        "errorbar_in_main", "is_reference",
        "plot_color", "line_color", "line_style", "marker",
        "recommended_ymin", "recommended_ymax", "notes",
    ]
    for col in preferred_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out[preferred_cols]


# ============================================================
# 7. TableS all ablation：保留原逻辑
# ============================================================

def build_tableS_all_ablation_matrix(summary, df_long, all_specs):
    labels = [s["display_label"] for s in all_specs]

    header_row_1 = [
        "Metric", "Model variant",
        "t1", "t2", "t3", "t4", "t5", TASK_MEAN_LABEL,
    ]
    header_row_2 = [
        "", "",
        TASK_CLINICAL_NAME["t1"], TASK_CLINICAL_NAME["t2"], TASK_CLINICAL_NAME["t3"],
        TASK_CLINICAL_NAME["t4"], TASK_CLINICAL_NAME["t5"], TASK_CLINICAL_NAME[TASK_MEAN_LABEL],
    ]

    rows = [header_row_1, header_row_2]
    bold_positions = set()
    best_map = {}

    for metric_key, _metric_label in METRIC_SPECS:
        for task in TASK_ORDER_WITH_MEAN:
            label_to_mean = {}
            for label in labels:
                mean_val, _ = get_task_or_mean_metric(df_long, summary, label, task, metric_key, weighted=False)
                label_to_mean[label] = mean_val

            valid_vals = [v for v in label_to_mean.values() if pd.notna(v)]
            if not valid_vals:
                best_map[(metric_key, task)] = set()
                continue

            best_val = max(valid_vals)
            best_map[(metric_key, task)] = {
                label for label, val in label_to_mean.items()
                if pd.notna(val) and np.isclose(val, best_val, atol=1e-12)
            }

    for metric_key, metric_label in METRIC_SPECS:
        for mi, spec in enumerate(all_specs):
            label = spec["display_label"]
            current_row_idx = len(rows)
            row = [metric_label if mi == 0 else "", make_display_label(spec["experiment"], spec["model_variant"])]

            for col_idx, task in enumerate(TASK_ORDER_WITH_MEAN, start=2):
                mean_val, sd_val = get_task_or_mean_metric(df_long, summary, label, task, metric_key)
                row.append(fmt_mean_sd(mean_val, sd_val))

                if label in best_map.get((metric_key, task), set()) and row[-1] != "":
                    bold_positions.add((current_row_idx, col_idx))
            rows.append(row)

    display_df = pd.DataFrame(rows)
    return display_df, rows, bold_positions


# ============================================================
# 7. Excel 写入与格式
# ============================================================

def write_table_matrix_formatted_sheet(workbook, worksheet, table_rows, bold_positions, n_models):
    header_fmt = workbook.add_format({
        "bold": True,
        "font_color": "black",
        "bg_color": "#D9E2F3",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "text_wrap": True,
    })
    subheader_fmt = workbook.add_format({
        "bold": True,
        "font_color": "black",
        "bg_color": "#EDEDED",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "text_wrap": True,
    })
    metric_fmt = workbook.add_format({
        "bold": True,
        "font_color": "black",
        "bg_color": "#E2F0D9",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "text_wrap": True,
    })
    model_fmt = workbook.add_format({
        "border": 1,
        "align": "left",
        "valign": "vcenter",
        "text_wrap": True,
    })
    body_fmt = workbook.add_format({
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "text_wrap": True,
    })
    body_bold_fmt = workbook.add_format({
        "bold": True,
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "text_wrap": True,
    })

    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(2, 2)
    worksheet.set_row(0, 24)
    worksheet.set_row(1, 42)
    worksheet.set_column(0, 0, 14)
    worksheet.set_column(1, 1, 38)
    worksheet.set_column(2, 6, 22)
    worksheet.set_column(7, 7, 26)

    for c, value in enumerate(table_rows[0]):
        worksheet.write(0, c, value, header_fmt)
    for c, value in enumerate(table_rows[1]):
        worksheet.write(1, c, value, subheader_fmt)

    row_idx = 2
    while row_idx < len(table_rows):
        metric_label = table_rows[row_idx][0]
        block_start = row_idx
        block_end = min(row_idx + n_models - 1, len(table_rows) - 1)

        if n_models > 1:
            worksheet.merge_range(block_start, 0, block_end, 0, metric_label, metric_fmt)
        else:
            worksheet.write(block_start, 0, metric_label, metric_fmt)

        for r in range(block_start, block_end + 1):
            worksheet.set_row(r, 26)
            worksheet.write(r, 1, table_rows[r][1], model_fmt)
            for c in range(2, len(table_rows[r])):
                fmt = body_bold_fmt if (r, c) in bold_positions else body_fmt
                worksheet.write(r, c, table_rows[r][c], fmt)

        row_idx += n_models


def autosize_worksheet(ws, df, max_width=46):
    for i, col in enumerate(df.columns):
        vals = df[col].head(200).fillna("").astype(str).tolist()
        max_len = max([len(str(col))] + [len(v) for v in vals])
        ws.set_column(i, i, min(max_len + 2, max_width))


def write_output_excel(output_path, fig_task_profile_data, fig_main_mean_data, per_task, tableS_rows, bold_positions, all_specs):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # 1) TableS_all_ablation：保留原格式化矩阵逻辑。
        ws_tableS = workbook.add_worksheet("TableS_all_ablation")
        writer.sheets["TableS_all_ablation"] = ws_tableS
        write_table_matrix_formatted_sheet(workbook, ws_tableS, tableS_rows, bold_positions, n_models=len(all_specs))

        # 2) Per_task_holdout_metrics：保留原逻辑。
        per_task.to_excel(writer, sheet_name="Per_task_holdout_metrics", index=False)

        # 3) 原 task-wise sheet：保留，用于 Supplementary task-wise profile。
        fig_task_profile_data.to_excel(writer, sheet_name="Fig_subgroup_task_profile_data", index=False)

        # 4) 新正文主图 sheet：Mean across t1–t5 point-range plot。
        fig_main_mean_data.to_excel(writer, sheet_name="Fig_subgroup_main_mean_data", index=False)

        header_fmt = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F4E79",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        })
        number_fmt = workbook.add_format({"num_format": "0.0000"})
        bool_fmt = workbook.add_format({"align": "center"})

        for sheet_name, df in [
            ("Per_task_holdout_metrics", per_task),
            ("Fig_subgroup_task_profile_data", fig_task_profile_data),
            ("Fig_subgroup_main_mean_data", fig_main_mean_data),
        ]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(df), max(len(df.columns) - 1, 0))

            for col_num, value in enumerate(df.columns.values):
                ws.write(0, col_num, value, header_fmt)

            autosize_worksheet(ws, df)

            # 数字列格式
            for i, col in enumerate(df.columns):
                if (
                    col in [
                        "mean", "sd", "reference_value", "reference_mean", "reference_sd",
                        "recommended_ymin", "recommended_ymax",
                    ]
                    or re.match(r"fold\d+", str(col))
                    or re.match(r"reference_fold\d+", str(col))
                ):
                    ws.set_column(i, i, 12, number_fmt)
                if col in ["errorbar_in_main", "is_reference"]:
                    ws.set_column(i, i, 15, bool_fmt)

        # Fig sheets 适当加宽关键列
        for sheet_name, df in [
            ("Fig_subgroup_task_profile_data", fig_task_profile_data),
            ("Fig_subgroup_main_mean_data", fig_main_mean_data),
        ]:
            ws_fig = writer.sheets[sheet_name]
            for col_name in ["display_label", "clinical_meaning", "notes", "reference_mean±sd"]:
                if col_name in df.columns:
                    col_idx = df.columns.get_loc(col_name)
                    ws_fig.set_column(col_idx, col_idx, 34)



# ============================================================
# 8. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="整理 Result 1.7 亚组分析结果，生成 Fig_subgroup_main_mean_data + Fig_subgroup_task_profile_data + Per_task_holdout_metrics + TableS_all_ablation。"
    )
    parser.add_argument(
        "--result_file",
        action="append",
        default=None,
        help=(
            "可重复传入，格式：Experiment|Model variant|文件路径，"
            "例如：E1|Age <40|xx_path"
            "若不传则使用脚本顶部 ABLATION_RESULT_FILES。"
        ),
    )
    parser.add_argument(
        "--out",
        default=r"xx_path",
        help="输出 Excel 文件路径。",
    )
    args = parser.parse_args()

    raw_specs = parse_cli_result_files(args.result_file) if args.result_file else ABLATION_RESULT_FILES
    specs = normalize_result_specs(raw_specs)

    for spec in specs:
        if not spec["path"].exists():
            raise FileNotFoundError(
                f"未找到输入文件：{spec['experiment']} | {spec['model_variant']} | {spec['path']}"
            )

    main_specs = filter_specs_by_experiments(specs, MAIN_EXPERIMENTS)
    all_specs = filter_specs_by_experiments(specs, ALL_EXPERIMENTS)

    print("[读取] 读取 MTL-style holdout 数据...")
    df_list = []
    for idx, spec in enumerate(specs, start=1):
        print(f"  [{idx}] {spec['experiment']} | {spec['model_variant']} | {spec['path']}")
        df_list.append(read_mtl_style_result(spec["path"], spec))

    df_long = pd.concat(df_list, ignore_index=True)
    summary = summarize_long(df_long)

    print("[生成] 保留原 task-wise sheet：Fig_subgroup_task_profile_data...")
    fig_task_profile_data = build_fig_subgroup_task_profile_data(df_long, summary, main_specs)

    print("[生成] 新正文主图 sheet：Fig_subgroup_main_mean_data...")
    fig_main_mean_data = build_fig_subgroup_main_mean_data(df_long, summary, main_specs)

    print("[生成] Per_task_holdout_metrics...")
    per_task = build_per_task_holdout_metrics(summary, main_specs)

    print("[生成] TableS_all_ablation...")
    _tableS_matrix_df, tableS_rows, bold_positions = build_tableS_all_ablation_matrix(summary, df_long, all_specs)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[写入] 写入 Excel...")
    write_output_excel(output_path, fig_task_profile_data, fig_main_mean_data, per_task, tableS_rows, bold_positions, all_specs)

    print("[完成]")
    print(f"输出 Excel：{output_path.resolve()}")
    print("\n输出 Sheets:")
    print("  - Fig_subgroup_main_mean_data")
    print("  - Fig_subgroup_task_profile_data")
    print("  - Per_task_holdout_metrics")
    print("  - TableS_all_ablation")


if __name__ == "__main__":
    main()

