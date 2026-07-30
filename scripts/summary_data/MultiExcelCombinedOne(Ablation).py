# -*- coding: utf-8 -*-
"""
Result5 框架级消融实验整理脚本

用途：
1. 读取多个 MTL-style 结果 Excel（每个文件包含 holdout_fold1~holdout_fold5 sheet）。
2. 生成 Result5 所需的表格：
   - Table5_performance_main：框架级消融实验 holdout 总体性能对比表
   - Fig5a_overall_loss：总体性能损失图数据
   - Fig5b_delta_heatmap / Fig5b_delta_heatmap_long：任务层面 ΔMacro-F1 heatmap 数据
   - Fig5c_auprc_t3_t5：t3–t5 AUPRC 图数据
   - Fig5d_auroc_t3_t5：t3–t5 AUROC 图数据
   - Per_task_holdout_metrics：逐任务完整性能
   - TableS_all_ablation：补充表全部消融实验
   - README：说明文档

Delta 定义：
    Δ = Ablation variant - Proposed framework

输入 Excel 要求：
- 每个结果文件包含 holdout_fold1、holdout_fold2、... sheet。
- 每个 holdout_fold sheet 至少包含 task_key/task 列，以及 accuracy / precision / recall /
  macro_f1 / auc(or auroc) / auprc 等指标列。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. 基础配置：后续主要修改这里
# ============================================================

TASK_ORDER = ["t1", "t2", "t3", "t4", "t5"]
TASK_ORDER_WITH_MEAN = TASK_ORDER + ["Mean across t1–t5"]

TASK_CLINICAL_NAME = {
    "t1": "CPET functional class",
    "t2": "Exercise capacity",
    "t3": "exercise ECG interpretation",
    "t4": "ventilatory function",
    "t5": "heart-rate reserve",
    "Mean across t1–t5": "Mean across t1–t5",
}

TASK_TYPE = {
    "t1": "3-class",
    "t2": "3-class",
    "t3": "binary",
    "t4": "binary",
    "t5": "binary",
}

TASK_WEIGHTS = {
    "t1": 1.0,
    "t2": 1.0,
    "t3": 1.0,
    "t4": 1.0,
    "t5": 1.0,
}

# AUROC / AUPRC 主要用于二分类任务 t3–t5
RANKING_TASKS = ["t3", "t4", "t5"]

# 旧评估脚本中 AUPRC 可能被错误写为 0。
# 若确实要保留 0，请改为 False。
TREAT_ZERO_AUPRC_AS_MISSING = True

REFERENCE_MODEL_VARIANT = "Proposed framework"

# 正文 Table5 / Fig5a / Fig5b / Fig5c / Fig5d 使用这些模型
MAIN_EXPERIMENTS = ["Full", "E1", "E2", "E3", "E4"]

# 补充表使用全部实验
ALL_EXPERIMENTS = ["Full", "E1", "E2", "E3", "E4", "S1", "S2", "S3", "S4", "S5", "S6"]

METRIC_SPECS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("recall", "Macro-recall"),
]


# ============================================================
# 0.1 输入文件配置
# ============================================================

ABLATION_RESULT_FILES = [
    {
        "experiment": "Full",
        "model_variant": "Proposed framework",
        "path": r"xx_path",
    },
    {
        "experiment": "E1",
        "model_variant": "w/o T6 contextualization",
        "path": r"xx_path",
    },
    {
        "experiment": "E2",
        "model_variant": "w/o dual-trunk routing",
        "path": r"xx_path",
    },
    {
        "experiment": "E3",
        "model_variant": "w/o Beta expert decomposition",
        "path": r"xx_path",
    },
    {
        "experiment": "E4",
        "model_variant": "w/o nine-panel prior chain",
        "path": r"xx_path",
    },
    {
        "experiment": "S1",
        "model_variant": "w/o t3-private expert",
        "path": r"xx_path",
    },
    {
        "experiment": "S2",
        "model_variant": "w/o group_245 expert",
        "path": r"xx_path",
    },
    {
        "experiment": "S3",
        "model_variant": "w/o Beta gates",
        "path": r"xx_path",
    },
    {
        "experiment": "S4",
        "model_variant": "w/o PMGT",
        "path": r"xx_path",
    },
    {
        "experiment": "S5",
        "model_variant": "w/o Alpha multi-scale residual CNN",
        "path": r"xx_path",
    },
    {
        "experiment": "S6",
        "model_variant": "w/o student Alpha prior",
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
      --result_file "E2|w/o dual-trunk routing|xx_path"
    """
    specs = []

    for raw in result_file_args or []:
        parts = raw.split("|", 2)
        if len(parts) != 3:
            raise ValueError(
                f"--result_file 格式错误：{raw}\n"
                f"正确格式：--result_file \"E2|w/o dual-trunk routing|xx_path\""
            )

        experiment, model_variant, path = [p.strip() for p in parts]

        specs.append({
            "experiment": experiment,
            "model_variant": model_variant,
            "path": path,
        })

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
        raise ValueError("必须提供 Full / Proposed framework 结果，作为 delta 参考。")

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
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "weighted_f1",
        "auroc",
        "auprc",
        "loss",
        "threshold",
        "minority_f1",
        "minority_recall",
        "minority_precision",
        "pred_minor_rate",
        "true_minor_rate",
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
        .groupby(
            ["experiment", "model_variant", "display_label", "task", "metric"],
            as_index=False,
        )
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


def fold_task_average(
    df_long,
    display_label,
    fold,
    tasks,
    metric="macro_f1",
    weights=None,
    require_all=True,
):
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
    fold_vals = []

    for fold in folds:
        fold_vals.append(
            fold_task_average(
                df_long,
                display_label,
                fold,
                TASK_ORDER,
                metric=metric,
                weights=TASK_WEIGHTS if weighted else None,
            )
        )

    return aggregate_fold_values(fold_vals)


# ============================================================
# 4. Table5：holdout 总体性能对比表
# ============================================================

def build_table5b_performance_main(df_long, summary, main_specs):
    """
    Table 5：框架级消融实验 holdout 总体指标摘要表。
    """
    reference_label = [
        s["display_label"]
        for s in main_specs
        if s["model_variant"] == REFERENCE_MODEL_VARIANT
    ]

    if not reference_label:
        reference_label = [s["display_label"] for s in main_specs if s["experiment"] == "Full"]

    if not reference_label:
        raise ValueError("正文主实验中未找到 Proposed framework / Full。")

    reference_label = reference_label[0]
    folds = sorted(df_long["fold"].dropna().unique())

    def fold_mean_values(label, metric, tasks=TASK_ORDER, weights=None, require_all=True):
        return [
            fold_task_average(
                df_long,
                label,
                fold,
                tasks,
                metric=metric,
                weights=weights,
                require_all=require_all,
            )
            for fold in folds
        ]

    ref = {
        "accuracy": aggregate_fold_values(fold_mean_values(reference_label, "accuracy"))[0],
        "precision": aggregate_fold_values(fold_mean_values(reference_label, "precision"))[0],
        "recall": aggregate_fold_values(fold_mean_values(reference_label, "recall"))[0],
        "macro_f1": aggregate_fold_values(fold_mean_values(reference_label, "macro_f1"))[0],
    }

    rows = []

    for spec in main_specs:
        label = spec["display_label"]

        mean_acc, mean_acc_sd = aggregate_fold_values(fold_mean_values(label, "accuracy"))
        mean_prec, mean_prec_sd = aggregate_fold_values(fold_mean_values(label, "precision"))
        mean_rec, mean_rec_sd = aggregate_fold_values(fold_mean_values(label, "recall"))
        mean_macro_f1, mean_macro_f1_sd = aggregate_fold_values(fold_mean_values(label, "macro_f1"))

        t1_macro_f1, t1_macro_f1_sd = aggregate_fold_values([
            get_fold_metric_value(df_long, label, "t1", "macro_f1", fold)
            for fold in folds
        ])

        beta_macro_f1, beta_macro_f1_sd = aggregate_fold_values(
            fold_mean_values(label, "macro_f1", tasks=["t2", "t3", "t4", "t5"])
        )

        weighted_macro_f1, weighted_macro_f1_sd = aggregate_fold_values(
            fold_mean_values(label, "macro_f1", tasks=TASK_ORDER, weights=TASK_WEIGHTS)
        )

        auroc_mean, auroc_sd = aggregate_fold_values(
            fold_mean_values(label, "auroc", tasks=RANKING_TASKS, require_all=True)
        )

        auprc_mean, auprc_sd = aggregate_fold_values(
            fold_mean_values(label, "auprc", tasks=RANKING_TASKS, require_all=True)
        )

        rows.append({
            "Model variant": spec["model_variant"],
            "Experiment": spec["experiment"],
            "t1–t5 mean Accuracy": fmt_mean_sd(mean_acc, mean_acc_sd),
            "Δ mean Accuracy": mean_acc - ref["accuracy"] if pd.notna(mean_acc) and pd.notna(ref["accuracy"]) else np.nan,
            "t1–t5 mean Precision": fmt_mean_sd(mean_prec, mean_prec_sd),
            "Δ mean Precision": mean_prec - ref["precision"] if pd.notna(mean_prec) and pd.notna(ref["precision"]) else np.nan,
            "t1–t5 mean Recall": fmt_mean_sd(mean_rec, mean_rec_sd),
            "Δ mean Recall": mean_rec - ref["recall"] if pd.notna(mean_rec) and pd.notna(ref["recall"]) else np.nan,
            "t1–t5 mean Macro-F1": fmt_mean_sd(mean_macro_f1, mean_macro_f1_sd),
            "Δ mean Macro-F1": mean_macro_f1 - ref["macro_f1"] if pd.notna(mean_macro_f1) and pd.notna(ref["macro_f1"]) else np.nan,
            "t1 Macro-F1": fmt_mean_sd(t1_macro_f1, t1_macro_f1_sd),
            "t2–t5 mean Macro-F1": fmt_mean_sd(beta_macro_f1, beta_macro_f1_sd),
            "t1–t5 weighted Macro-F1": fmt_mean_sd(weighted_macro_f1, weighted_macro_f1_sd),
            "t3–t5 mean AUROC": fmt_mean_sd(auroc_mean, auroc_sd),
            "t3–t5 mean AUPRC": fmt_mean_sd(auprc_mean, auprc_sd),
        })

    desired_cols = [
        "Model variant",
        "Experiment",
        "t1–t5 mean Accuracy",
        "Δ mean Accuracy",
        "t1–t5 mean Precision",
        "Δ mean Precision",
        "t1–t5 mean Recall",
        "Δ mean Recall",
        "t1–t5 mean Macro-F1",
        "Δ mean Macro-F1",
        "t1 Macro-F1",
        "t2–t5 mean Macro-F1",
        "t1–t5 weighted Macro-F1",
        "t3–t5 mean AUROC",
        "t3–t5 mean AUPRC",
    ]

    return pd.DataFrame(rows)[desired_cols]


# ============================================================
# 5. 逐任务完整性能
# ============================================================

def build_per_task_holdout_metrics(summary, main_specs):
    """
    框架级消融实验在 holdout 队列上的逐任务完整性能。
    """
    rows = []

    reference_label = [
        s["display_label"]
        for s in main_specs
        if s["model_variant"] == REFERENCE_MODEL_VARIANT
    ]

    if not reference_label:
        reference_label = [s["display_label"] for s in main_specs if s["experiment"] == "Full"]

    if not reference_label:
        raise ValueError("正文主实验中未找到 Proposed framework / Full。")

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

        for task in TASK_ORDER:
            row = {
                "Experiment / model variant": combined,
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
        "Experiment / model variant",
        "Task",
        "Clinical meaning",
        "Task type",
        "Accuracy",
        "Precision",
        "Recall",
        "Macro-F1",
        "AUROC",
        "AUPRC",
        "Δ Accuracy",
        "Δ Macro-F1",
        "Δ AUROC",
        "Δ AUPRC",
    ]

    for col in desired_cols:
        if col not in out.columns:
            out[col] = np.nan

    return out[desired_cols]


# ============================================================
# 6. Fig5a / Fig5b / Fig5c / Fig5d 数据
# ============================================================

def build_fig5b_delta_heatmap(df_long, main_specs):
    """
    Figure 5b：任务层面 ΔMacro-F1 heatmap 数据。
    """
    reference_specs = [
        s for s in main_specs
        if s["model_variant"] == REFERENCE_MODEL_VARIANT or s["experiment"] == "Full"
    ]

    if not reference_specs:
        raise ValueError("main_specs 中未找到 Full / Proposed framework。")

    ref_label = reference_specs[0]["display_label"]
    compare_specs = [s for s in main_specs if s["display_label"] != ref_label]
    folds = sorted(df_long["fold"].dropna().unique())

    wide_rows = []

    for task in TASK_ORDER:
        row = {
            "Task": task,
            "Clinical meaning": TASK_CLINICAL_NAME[task],
        }

        for spec in compare_specs:
            label = spec["display_label"]
            deltas = []

            for fold in folds:
                model_val = get_fold_metric_value(df_long, label, task, "macro_f1", fold)
                ref_val = get_fold_metric_value(df_long, ref_label, task, "macro_f1", fold)

                if pd.notna(model_val) and pd.notna(ref_val):
                    deltas.append(model_val - ref_val)

            col_name = f"Δ({spec['model_variant']}–Proposed)"
            row[f"{col_name} mean"] = safe_mean(deltas)
            row[f"{col_name} SD"] = sample_std(deltas)

        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)

    long_rows = []

    for task in TASK_ORDER:
        for spec in compare_specs:
            label = spec["display_label"]
            deltas = []

            for fold in folds:
                model_val = get_fold_metric_value(df_long, label, task, "macro_f1", fold)
                ref_val = get_fold_metric_value(df_long, ref_label, task, "macro_f1", fold)

                if pd.notna(model_val) and pd.notna(ref_val):
                    deltas.append(model_val - ref_val)

            long_rows.append({
                "Model variant": spec["model_variant"],
                "Experiment": spec["experiment"],
                "Task": task,
                "Clinical meaning": TASK_CLINICAL_NAME[task],
                "Delta Macro-F1 mean": safe_mean(deltas),
                "Delta Macro-F1 SD": sample_std(deltas),
                "n folds": len([d for d in deltas if pd.notna(d)]),
            })

    long_df = pd.DataFrame(long_rows)

    return wide_df, long_df


def build_fig5a_overall_loss(df_long, main_specs):
    """
    Figure 5a：总体性能损失图数据。
    """
    reference_specs = [
        s for s in main_specs
        if s["model_variant"] == REFERENCE_MODEL_VARIANT or s["experiment"] == "Full"
    ]

    if not reference_specs:
        raise ValueError("main_specs 中未找到 Full / Proposed framework。")

    ref_label = reference_specs[0]["display_label"]
    compare_specs = [s for s in main_specs if s["display_label"] != ref_label]
    folds = sorted(df_long["fold"].dropna().unique())

    rows = []

    for spec in compare_specs:
        label = spec["display_label"]
        delta_macro_f1 = []
        delta_accuracy = []

        for fold in folds:
            model_mf1 = fold_task_average(df_long, label, fold, TASK_ORDER, metric="macro_f1")
            ref_mf1 = fold_task_average(df_long, ref_label, fold, TASK_ORDER, metric="macro_f1")

            model_acc = fold_task_average(df_long, label, fold, TASK_ORDER, metric="accuracy")
            ref_acc = fold_task_average(df_long, ref_label, fold, TASK_ORDER, metric="accuracy")

            if pd.notna(model_mf1) and pd.notna(ref_mf1):
                delta_macro_f1.append(model_mf1 - ref_mf1)

            if pd.notna(model_acc) and pd.notna(ref_acc):
                delta_accuracy.append(model_acc - ref_acc)

        mf1_mean, mf1_sd = aggregate_fold_values(delta_macro_f1)
        acc_mean, acc_sd = aggregate_fold_values(delta_accuracy)

        rows.append({
            "Model variant": spec["model_variant"],
            "Experiment": spec["experiment"],
            "Δ t1–t5 mean Macro-F1 mean": mf1_mean,
            "Δ t1–t5 mean Macro-F1 SD": mf1_sd,
            "Δ t1–t5 mean Accuracy mean": acc_mean,
            "Δ t1–t5 mean Accuracy SD": acc_sd,
            "n folds Macro-F1": len([v for v in delta_macro_f1 if pd.notna(v)]),
            "n folds Accuracy": len([v for v in delta_accuracy if pd.notna(v)]),
        })

    return pd.DataFrame(rows)


def build_fig5c_auprc_data(summary, main_specs):
    """
    Figure 5c：t3–t5 AUPRC 图数据。
    """
    rows = []

    for task in RANKING_TASKS:
        for spec in main_specs:
            label = spec["display_label"]

            mean_val = get_summary_value(summary, label, task, "auprc", "mean")
            sd_val = get_summary_value(summary, label, task, "auprc", "sd")
            n_val = get_summary_value(summary, label, task, "auprc", "n")

            rows.append({
                "Task": task,
                "Clinical meaning": TASK_CLINICAL_NAME[task],
                "Model variant": spec["model_variant"],
                "Experiment": spec["experiment"],
                "AUPRC mean": mean_val,
                "AUPRC SD": sd_val,
                "n folds": int(n_val) if pd.notna(n_val) else np.nan,
                "Random baseline": "",
            })

    out = pd.DataFrame(rows)

    out["Task"] = pd.Categorical(out["Task"], categories=RANKING_TASKS, ordered=True)
    out["Experiment"] = pd.Categorical(
        out["Experiment"],
        categories=[s["experiment"] for s in main_specs],
        ordered=True,
    )

    out = out.sort_values(["Task", "Experiment"]).reset_index(drop=True)
    out["Task"] = out["Task"].astype(str)
    out["Experiment"] = out["Experiment"].astype(str)

    return out


def build_fig5d_auroc_data(summary, main_specs):
    """
    Figure 5d：t3–t5 AUROC 图数据。

    输出 long table，用于绘制 3 个并列小柱状图：
        d(i) t3 AUROC
        d(ii) t4 AUROC
        d(iii) t5 AUROC

    Random baseline AUROC = 0.5。
    """
    rows = []

    for task in RANKING_TASKS:
        for spec in main_specs:
            label = spec["display_label"]

            mean_val = get_summary_value(summary, label, task, "auroc", "mean")
            sd_val = get_summary_value(summary, label, task, "auroc", "sd")
            n_val = get_summary_value(summary, label, task, "auroc", "n")

            rows.append({
                "Task": task,
                "Clinical meaning": TASK_CLINICAL_NAME[task],
                "Model variant": spec["model_variant"],
                "Experiment": spec["experiment"],
                "AUROC mean": mean_val,
                "AUROC SD": sd_val,
                "n folds": int(n_val) if pd.notna(n_val) else np.nan,
                "Random baseline": 0.5,
            })

    out = pd.DataFrame(rows)

    out["Task"] = pd.Categorical(out["Task"], categories=RANKING_TASKS, ordered=True)
    out["Experiment"] = pd.Categorical(
        out["Experiment"],
        categories=[s["experiment"] for s in main_specs],
        ordered=True,
    )

    out = out.sort_values(["Task", "Experiment"]).reset_index(drop=True)
    out["Task"] = out["Task"].astype(str)
    out["Experiment"] = out["Experiment"].astype(str)

    return out


# ============================================================
# 7. TableS all ablation
# ============================================================

def build_tableS_all_ablation_data(summary, df_long, all_specs):
    rows = []

    for task in TASK_ORDER:
        row = {
            "Task": task,
            "Clinical meaning": TASK_CLINICAL_NAME[task],
            "Task type": TASK_TYPE[task],
        }

        for spec in all_specs:
            label = spec["display_label"]

            for metric_key, metric_label in METRIC_SPECS:
                mean_val = get_summary_value(summary, label, task, metric_key, "mean")
                sd_val = get_summary_value(summary, label, task, metric_key, "sd")
                row[f"{spec['experiment']} {metric_label}"] = fmt_mean_sd(mean_val, sd_val)

        rows.append(row)

    mean_row = {
        "Task": "Mean across t1–t5",
        "Clinical meaning": TASK_CLINICAL_NAME["Mean across t1–t5"],
        "Task type": "-",
    }

    for spec in all_specs:
        label = spec["display_label"]

        for metric_key, metric_label in METRIC_SPECS:
            mean_val, sd_val = get_task_or_mean_metric(
                df_long,
                summary,
                label,
                "Mean across t1–t5",
                metric_key,
            )
            mean_row[f"{spec['experiment']} {metric_label}"] = fmt_mean_sd(mean_val, sd_val)

    rows.append(mean_row)

    return pd.DataFrame(rows)


def build_tableS_all_ablation_matrix(summary, df_long, all_specs):
    labels = [s["display_label"] for s in all_specs]

    header_row_1 = [
        "Metric",
        "Model variant",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "Mean across t1–t5",
    ]

    header_row_2 = [
        "",
        "",
        TASK_CLINICAL_NAME["t1"],
        TASK_CLINICAL_NAME["t2"],
        TASK_CLINICAL_NAME["t3"],
        TASK_CLINICAL_NAME["t4"],
        TASK_CLINICAL_NAME["t5"],
        TASK_CLINICAL_NAME["Mean across t1–t5"],
    ]

    rows = [header_row_1, header_row_2]
    bold_positions = set()

    best_map = {}

    for metric_key, _metric_label in METRIC_SPECS:
        for task in TASK_ORDER_WITH_MEAN:
            label_to_mean = {}

            for label in labels:
                mean_val, _ = get_task_or_mean_metric(
                    df_long,
                    summary,
                    label,
                    task,
                    metric_key,
                    weighted=False,
                )
                label_to_mean[label] = mean_val

            valid_vals = [v for v in label_to_mean.values() if pd.notna(v)]

            if not valid_vals:
                best_map[(metric_key, task)] = set()
                continue

            best_val = max(valid_vals)

            best_map[(metric_key, task)] = {
                label
                for label, val in label_to_mean.items()
                if pd.notna(val) and np.isclose(val, best_val, atol=1e-12)
            }

    for metric_key, metric_label in METRIC_SPECS:
        for mi, spec in enumerate(all_specs):
            label = spec["display_label"]
            current_row_idx = len(rows)

            row = [
                metric_label if mi == 0 else "",
                make_display_label(spec["experiment"], spec["model_variant"]),
            ]

            for col_idx, task in enumerate(TASK_ORDER_WITH_MEAN, start=2):
                mean_val, sd_val = get_task_or_mean_metric(
                    df_long,
                    summary,
                    label,
                    task,
                    metric_key,
                )

                row.append(fmt_mean_sd(mean_val, sd_val))

                if label in best_map.get((metric_key, task), set()) and row[-1] != "":
                    bold_positions.add((current_row_idx, col_idx))

            rows.append(row)

    display_df = pd.DataFrame(rows)

    return display_df, rows, bold_positions


# ============================================================
# 8. README
# ============================================================

def build_readme(specs, main_specs, all_specs):
    rows = [
        ["项目", "说明"],
        [
            "脚本用途",
            "Result5 框架级消融实验整理：Table 5 总体指标摘要、Figure 5a 总体性能损失、Figure 5b task-wise ΔMacro-F1 heatmap、Figure 5c t3–t5 AUPRC、Figure 5d t3–t5 AUROC、TableS_all_ablation。",
        ],
        [
            "Delta 定义",
            "Δ = Ablation variant - Proposed framework；Fig5a/Fig5b 使用 fold-paired delta 后计算 mean / SD。",
        ],
        [
            "正文实验",
            " -> ".join([f"{s['experiment']}:{s['model_variant']}" for s in main_specs]),
        ],
        [
            "表4实验",
            " -> ".join([f"{s['experiment']}:{s['model_variant']}" for s in all_specs]),
        ],
        [
            "weighted Macro-F1",
            f"TASK_WEIGHTS = {TASK_WEIGHTS}; 默认等权，若任务权重需调整请修改脚本顶部 TASK_WEIGHTS。",
        ],
        [
            "Mean AUROC/AUPRC",
            f"Table5b 中 Mean AUROC/AUPRC 仅基于二分类任务 {RANKING_TASKS} 计算；t1/t2 当前为 N/A。",
        ],
        [
            "AUPRC=0 处理",
            f"TREAT_ZERO_AUPRC_AS_MISSING={TREAT_ZERO_AUPRC_AS_MISSING}；若旧结果文件把 AUPRC 错误写成 0，会被视为缺失。",
        ],
        ["输入格式", "每个 mtl_eval_*.xlsx 需要包含 holdout_fold1~holdout_fold5 sheets。"],
        ["Table 5 Sheet", "Table5_performance_main"],
        ["Figure 5a Sheet", "Fig5a_overall_loss"],
        ["Figure 5b Sheet", "Fig5b_delta_heatmap / Fig5b_delta_heatmap_long"],
        ["Figure 5c Sheet", "Fig5c_auprc_t3_t5；Random baseline AUPRC = positive prevalence。"],
        ["Figure 5d Sheet", "Fig5d_auroc_t3_t5；Random baseline AUROC = 0.5。"],
        ["逐任务完整指标 Sheet", "Per_task_holdout_metrics"],
        ["TableS Sheet", "TableS_all_ablation / TableS_all_ablation_data"],
    ]

    for s in specs:
        rows.append([
            f"输入文件 {s['experiment']}",
            f"{s['model_variant']} | {s['path']}",
        ])

    return pd.DataFrame(rows[1:], columns=rows[0])


# ============================================================
# 9. Excel 写入与格式
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


def write_output_excel(
    output_path,
    table5,
    fig5a,
    fig5b_wide,
    fig5b_long,
    fig5c,
    fig5d,
    per_task,
    tableS_data,
    tableS_matrix_df,
    tableS_rows,
    bold_positions,
    all_specs,
    readme,
):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        ws_tableS = workbook.add_worksheet("TableS_all_ablation")
        writer.sheets["TableS_all_ablation"] = ws_tableS

        write_table_matrix_formatted_sheet(
            workbook,
            ws_tableS,
            tableS_rows,
            bold_positions,
            n_models=len(all_specs),
        )

        table5.to_excel(writer, sheet_name="Table5_performance_main", index=False)
        fig5a.to_excel(writer, sheet_name="Fig5a_overall_loss", index=False)
        fig5b_wide.to_excel(writer, sheet_name="Fig5b_delta_heatmap", index=False)
        fig5b_long.to_excel(writer, sheet_name="Fig5b_delta_heatmap_long", index=False)
        fig5c.to_excel(writer, sheet_name="Fig5c_auprc_t3_t5", index=False)
        fig5d.to_excel(writer, sheet_name="Fig5d_auroc_t3_t5", index=False)
        per_task.to_excel(writer, sheet_name="Per_task_holdout_metrics", index=False)
        tableS_data.to_excel(writer, sheet_name="TableS_all_ablation_data", index=False)
        tableS_matrix_df.to_excel(
            writer,
            sheet_name="TableS_all_ablation_matrix",
            index=False,
            header=False,
        )
        readme.to_excel(writer, sheet_name="README", index=False)

        header_fmt = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F4E79",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        })

        neg_fmt = workbook.add_format({
            "font_color": "#C00000",
            "num_format": "+0.0000;-0.0000;0.0000",
        })

        pos_fmt = workbook.add_format({
            "font_color": "#0070C0",
            "num_format": "+0.0000;-0.0000;0.0000",
        })

        sheet_df_pairs = [
            ("Table5_performance_main", table5),
            ("Fig5a_overall_loss", fig5a),
            ("Fig5b_delta_heatmap", fig5b_wide),
            ("Fig5b_delta_heatmap_long", fig5b_long),
            ("Fig5c_auprc_t3_t5", fig5c),
            ("Fig5d_auroc_t3_t5", fig5d),
            ("Per_task_holdout_metrics", per_task),
            ("TableS_all_ablation_data", tableS_data),
            ("TableS_all_ablation_matrix", tableS_matrix_df),
            ("README", readme),
        ]

        for sheet_name, df in sheet_df_pairs:
            ws = writer.sheets[sheet_name]

            if sheet_name != "TableS_all_ablation_matrix":
                ws.freeze_panes(1, 0)

                for col_num, value in enumerate(df.columns.values):
                    ws.write(0, col_num, value, header_fmt)

            for i, col in enumerate(df.columns):
                vals = df[col].head(150).fillna("").astype(str).tolist()
                max_len = max([len(str(col))] + [len(v) for v in vals])
                ws.set_column(i, i, min(max_len + 2, 46))

        delta_sheets = {
            "Table5_performance_main": table5,
            "Fig5a_overall_loss": fig5a,
            "Fig5b_delta_heatmap": fig5b_wide,
            "Fig5b_delta_heatmap_long": fig5b_long,
            "Per_task_holdout_metrics": per_task,
        }

        for sheet_name, df in delta_sheets.items():
            ws = writer.sheets[sheet_name]

            for c, col in enumerate(df.columns):
                if str(col).startswith("Δ") or "Delta" in str(col):
                    ws.conditional_format(
                        1,
                        c,
                        len(df),
                        c,
                        {
                            "type": "cell",
                            "criteria": ">",
                            "value": 0,
                            "format": pos_fmt,
                        },
                    )
                    ws.conditional_format(
                        1,
                        c,
                        len(df),
                        c,
                        {
                            "type": "cell",
                            "criteria": "<",
                            "value": 0,
                            "format": neg_fmt,
                        },
                    )

        writer.sheets["README"].set_column(0, 0, 28)
        writer.sheets["README"].set_column(1, 1, 120)


# ============================================================
# 10. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="整理 Result5 框架级消融实验结果，生成 Table5 / Fig5a / Fig5b / Fig5c / Fig5d / Per-task metrics / TableS all ablation。"
    )

    parser.add_argument(
        "--result_file",
        action="append",
        default=None,
        help=(
            "可重复传入，格式：Experiment|Model variant|文件路径，"
            "例如：E2|w/o dual-trunk routing|xx_path"
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

    print("[生成] Table 5 总体指标摘要 Table5_performance_main...")
    table5b = build_table5b_performance_main(df_long, summary, main_specs)

    print("[生成] Figure 5a 总体性能损失 Fig5a_overall_loss...")
    fig5a = build_fig5a_overall_loss(df_long, main_specs)

    print("[生成] Figure 5b 任务层面 ΔMacro-F1 heatmap...")
    fig5b_wide, fig5b_long = build_fig5b_delta_heatmap(df_long, main_specs)

    print("[生成] Figure 5c t3–t5 AUPRC 数据 Fig5c_auprc_t3_t5...")
    fig5c = build_fig5c_auprc_data(summary, main_specs)

    print("[生成] Figure 5d t3–t5 AUROC 数据 Fig5d_auroc_t3_t5...")
    fig5d = build_fig5d_auroc_data(summary, main_specs)

    print("[生成] Per_task_holdout_metrics...")
    per_task = build_per_task_holdout_metrics(summary, main_specs)

    print("[生成] TableS_all_ablation...")
    tableS_data = build_tableS_all_ablation_data(summary, df_long, all_specs)
    tableS_matrix_df, tableS_rows, bold_positions = build_tableS_all_ablation_matrix(
        summary,
        df_long,
        all_specs,
    )

    readme = build_readme(specs, main_specs, all_specs)

    output_path = Path(args.out)

    print("[写入] 写入 Excel...")

    write_output_excel(
        output_path,
        table5b,
        fig5a,
        fig5b_wide,
        fig5b_long,
        fig5c,
        fig5d,
        per_task,
        tableS_data,
        tableS_matrix_df,
        tableS_rows,
        bold_positions,
        all_specs,
        readme,
    )

    print("[完成]")
    print(f"输出 Excel：{output_path.resolve()}")
    print("\n输出 Sheets:")
    print("  - Table5_performance_main")
    print("  - Fig5a_overall_loss")
    print("  - Fig5b_delta_heatmap")
    print("  - Fig5b_delta_heatmap_long")
    print("  - Fig5c_auprc_t3_t5")
    print("  - Fig5d_auroc_t3_t5")
    print("  - Per_task_holdout_metrics")
    print("  - TableS_all_ablation")
    print("  - TableS_all_ablation_data")
    print("  - TableS_all_ablation_matrix")
    print("  - README")


if __name__ == "__main__":
    main()
