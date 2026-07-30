# -*- coding: utf-8 -*-
"""
整理 Single-task 结果文件与多个 MTL-style 结果文件，生成用于
Table 2、Figure 2A、Figure 2B、Figure 2D/Figure 2E/Figure 2F，
以及 Fig1g Holdout Macro-F1 箱型图所需 fold-level 数据 Excel。

本版更新：
1. 支持 5 个 Single-task 独立 Excel 文件 + 多个 MTL 文件（通过 SINGLE_TASK_FILES / MTL_RESULT_FILES 或命令行配置）。
2. Table2 直接输出为 Excel 可编辑格式化表格，不再生成图片版。
3. Table2 按照用户给定的样式重排：
   - 行按 Accuracy / Macro-F1 / Macro-recall 分块；
   - 每块内列出所有模型；
   - 列为 t1~t5 与 Mean across t1–t5；
   - 对每个指标 × 每个任务（含平均表现）下的最优模型数值加粗。
4. Table2 中的 Delta 改为“所有模型 - Ours”；其他 sheet 中 Delta 逻辑保持不变。
"""

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 基础配置
# ============================================================

TASK_ORDER = ["t1", "t2", "t3", "t4", "t5"]
TASK_ORDER_WITH_MEAN = TASK_ORDER + ["Mean across t1–t5"]

TASK_CLINICAL_NAME = {
    "t1": "Weber functional class",
    "t2": "Exercise capacity",
    "t3": "Exercise ECG response",
    "t4": "Breathing reserve",
    "t5": "Heart rate reserve",
    "Mean across t1–t5": "平均表现",
}

TASK_TYPE = {
    "t1": "3-class",
    "t2": "3-class",
    "t3": "binary",
    "t4": "binary",
    "t5": "binary",
}

METRIC_SPECS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("recall", "Macro-recall"),
]

SINGLE_MODEL_NAME = "Single-task model"
DEFAULT_OURS_MODEL_NAME = "Our method"

# ----------------------------------------------------------------
# 你后续主要改这里：5 个 Single-task 独立 Excel 文件。
# 每一项必须明确 task，避免脚本从文件名猜错任务。
# 命令行也可用 --single_file "t1=xx_path" 覆盖。
# ----------------------------------------------------------------
SINGLE_TASK_FILES = [
    {"task": "t1", "path": r"xx_path"},
    {"task": "t2", "path": r"xx_path"},
    {"task": "t3", "path": r"xx_path"},
    {"task": "t4", "path": r"xx_path"},
    {"task": "t5", "path": r"xx_path"},
]

# 可选：如果你仍然有旧版合并后的 RESULT2.xlsx，可以在命令行传 --single_combined 使用。
# 但当前推荐使用 SINGLE_TASK_FILES / --single_file。

# ----------------------------------------------------------------
# 你后续主要改这里：一个 list 同时写 MTL 文件地址和模型名。
# 顺序会直接决定输出 Excel 中模型列的顺序。
# ----------------------------------------------------------------
MTL_RESULT_FILES = [
    {
        "name": "Shared Bottom",
        "path": r"xx_path"
    },
    {
        "name": "MMOE",
        "path": r"xx_path"
    },
    {
        "name": "CGC",
        "path": r"xx_path"
    },
    {
        "name": "ADATT",
        "path": r"xx_path"
    },
    {
        "name": "Our method",
        "path": r"xx_path"
    },

]

# Fig2A 任务排序默认按最后一个 MTL 模型的 Macro-F1 从高到低排序。
PRIMARY_MODEL_NAME = None

# Table2 中作为 Delta 参照的模型名
TABLE2_DELTA_REFERENCE_MODEL = DEFAULT_OURS_MODEL_NAME

# Table2 Excel 格式
TABLE2_FONT_SIZE = 10


# ============================================================
# 1. 通用工具函数
# ============================================================

def sample_std(values):
    values = [v for v in values if pd.notna(v)]
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1))


def safe_mean(values):
    values = [v for v in values if pd.notna(v)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


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
        "auc": "auc",
        "auroc": "auc",
        "auprc": "auprc",
        "loss": "loss",
        "threshold": "threshold",
        "minority_f1": "minority_f1",
        "minority recall": "minority_recall",
        "minority_recall": "minority_recall",
        "minority precision": "minority_precision",
        "minority_precision": "minority_precision",
        "pred_minor_rate": "pred_minor_rate",
        "true_minor_rate": "true_minor_rate",
        "minority_tp": "minority_tp",
        "minority_fn": "minority_fn",
        "majority_fp": "majority_fp",
        "majority_tn": "majority_tn",
    }
    s = s.replace("（二分类）", "").replace("(二分类)", "").strip()
    s = s.replace(" ", "_")
    return mapping.get(s, s)


def find_task_from_sheet_name(sheet_name):
    m = re.search(r"t\s*([1-5])", sheet_name, flags=re.IGNORECASE)
    if not m:
        return None
    return f"t{m.group(1)}"


def normalize_mtl_specs(mtl_specs):
    out = []
    seen_names = set()

    for idx, item in enumerate(mtl_specs, start=1):
        if isinstance(item, dict):
            name = item.get("name")
            path = item.get("path")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            name, path = item
        else:
            raise ValueError(
                f"MTL_RESULT_FILES 第 {idx} 项格式错误，应为 "
                f"{{'name': 模型名, 'path': 文件路径}} 或 (模型名, 文件路径)。"
            )

        name = str(name).strip() if name is not None else ""
        path = str(path).strip() if path is not None else ""

        if not name:
            raise ValueError(f"MTL_RESULT_FILES 第 {idx} 项缺少 name。")
        if not path:
            raise ValueError(f"MTL_RESULT_FILES 第 {idx} 项缺少 path。")
        if name == SINGLE_MODEL_NAME:
            raise ValueError(f"MTL 模型名不能与 {SINGLE_MODEL_NAME!r} 重名：{name}")
        if name in seen_names:
            raise ValueError(f"MTL_RESULT_FILES 中存在重复模型名：{name}")

        seen_names.add(name)
        out.append({"name": name, "path": Path(path)})

    if not out:
        raise ValueError("MTL_RESULT_FILES 不能为空；至少需要提供一个 MTL-style 结果文件。")

    return out


def build_model_order(mtl_specs):
    return [SINGLE_MODEL_NAME] + [item["name"] for item in mtl_specs]


def get_primary_model_name(model_order):
    if PRIMARY_MODEL_NAME is not None:
        if PRIMARY_MODEL_NAME not in model_order:
            raise ValueError(
                f"PRIMARY_MODEL_NAME={PRIMARY_MODEL_NAME!r} 不在 MODEL_ORDER 中：{model_order}"
            )
        return PRIMARY_MODEL_NAME
    return model_order[-1]


def get_table2_delta_reference_model(model_order):
    if TABLE2_DELTA_REFERENCE_MODEL not in model_order:
        raise ValueError(
            f"TABLE2_DELTA_REFERENCE_MODEL={TABLE2_DELTA_REFERENCE_MODEL!r} 不在 MODEL_ORDER 中：{model_order}"
        )
    return TABLE2_DELTA_REFERENCE_MODEL


def parse_cli_mtl_files(mtl_file_args):
    specs = []
    for raw in mtl_file_args or []:
        if "=" not in raw:
            raise ValueError(
                f"--mtl_file 参数格式错误：{raw}\n"
                f"正确格式示例：--mtl_file \"Our method=xx_path\""
            )
        name, path = raw.split("=", 1)
        specs.append({"name": name.strip(), "path": path.strip()})
    return specs


def parse_cli_single_files(single_file_args):
    """
    解析命令行中的 --single_file 参数。
    格式：t1=文件路径，可重复传入 5 次。
    """
    specs = []
    for raw in single_file_args or []:
        if "=" not in raw:
            raise ValueError(
                f"--single_file 参数格式错误：{raw}\n"
                f"正确格式示例：--single_file \"t1=xx_path\""
            )
        task, path = raw.split("=", 1)
        specs.append({"task": task.strip().lower(), "path": path.strip()})
    return specs


def normalize_single_task_specs(single_specs):
    """
    校验并标准化 SINGLE_TASK_FILES。
    支持 dict: {"task": "t1", "path": ...}
    也兼容 tuple/list: ("t1", path)
    """
    out = []
    seen_tasks = set()

    for idx, item in enumerate(single_specs or [], start=1):
        if isinstance(item, dict):
            task = item.get("task")
            path = item.get("path")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            task, path = item
        else:
            raise ValueError(
                f"SINGLE_TASK_FILES 第 {idx} 项格式错误，应为 "
                f"{{'task': 't1', 'path': 文件路径}} 或 ('t1', 文件路径)。"
            )

        task = str(task).strip().lower() if task is not None else ""
        path = str(path).strip() if path is not None else ""

        if task not in TASK_ORDER:
            raise ValueError(f"SINGLE_TASK_FILES 第 {idx} 项 task={task!r} 非法，应为 t1~t5。")
        if not path:
            raise ValueError(f"SINGLE_TASK_FILES 第 {idx} 项缺少 path。")
        if task in seen_tasks:
            raise ValueError(f"SINGLE_TASK_FILES 中存在重复任务：{task}")

        seen_tasks.add(task)
        out.append({"task": task, "path": Path(path)})

    missing = [t for t in TASK_ORDER if t not in seen_tasks]
    if missing:
        raise ValueError(
            "Single-task 独立文件未配置完整，缺少：" + ", ".join(missing) +
            "。请在 SINGLE_TASK_FILES 中补齐，或用 --single_file t1=... 重复传入。"
        )

    return sorted(out, key=lambda x: TASK_ORDER.index(x["task"]))


def _read_single_task_dataframe(df, task, source_desc, model_name=SINGLE_MODEL_NAME):
    """
    从一个已经读入的 sheet dataframe 中抽取 fold-level 指标。
    适用于单任务独立 Excel：只要 sheet 中存在 Fold 列和指标列即可。
    """
    col_map = {}
    for c in df.columns:
        cn = normalize_metric_name(c)
        if cn is not None:
            col_map[c] = cn
    df = df.rename(columns=col_map)

    if "Fold" not in df.columns:
        possible_fold_cols = [c for c in df.columns if str(c).lower() == "fold"]
        if possible_fold_cols:
            df = df.rename(columns={possible_fold_cols[0]: "Fold"})

    if "Fold" not in df.columns:
        return []

    metric_candidates = [
        "accuracy", "precision", "recall", "f1", "macro_f1", "auc", "auprc",
        # Fig2D 需要的二分类少数类指标。原脚本没有读取这些列，
        # 会导致 Single-task model 在 Fig2D_data 中全部为空。
        "minority_f1", "minority_recall", "minority_precision",
        "pred_minor_rate", "true_minor_rate",
        "minority_tp", "minority_fn", "majority_fp", "majority_tn",
    ]
    if not any(m in df.columns for m in metric_candidates):
        return []

    fold_df = df[df["Fold"].astype(str).str.match(r"^\d+$", na=False)].copy()
    if fold_df.empty:
        return []

    fold_df["fold"] = fold_df["Fold"].astype(int)
    records = []
    for _, row in fold_df.iterrows():
        fold = int(row["fold"])
        for metric in metric_candidates:
            if metric in fold_df.columns and pd.notna(row.get(metric)):
                try:
                    value = float(row[metric])
                except Exception:
                    continue
                records.append({
                    "model": model_name,
                    "task": task,
                    "fold": fold,
                    "metric": metric,
                    "value": value,
                    "source": source_desc,
                })
    return records


def read_single_task_file(single_path, task, model_name=SINGLE_MODEL_NAME):
    """
    读取一个任务对应的单独 Single-task Excel 文件。

    关键修复：
    1. 优先只读取 Holdout_Test sheet，避免把 KFold_Summary 的验证集结果
       和 Holdout_Test 的外部 holdout 结果混在一起。
    2. _read_single_task_dataframe 已包含 minority_* 指标，因此 Fig2D_data
       可以正常填充 Single-task model 的少数类 precision/recall/F1。
    """
    xls = pd.ExcelFile(single_path)
    records = []

    preferred_sheet_names = {"holdout_test", "holdout test", "holdout"}
    selected_sheets = [
        s for s in xls.sheet_names
        if str(s).strip().lower() in preferred_sheet_names
    ]

    # 如果没有 Holdout_Test，则退回遍历所有 sheet，但会保留兼容性。
    # 推荐你的 single-task Excel 统一保留 Holdout_Test sheet。
    if not selected_sheets:
        selected_sheets = xls.sheet_names

    for sheet in selected_sheets:
        df = pd.read_excel(single_path, sheet_name=sheet)
        records.extend(
            _read_single_task_dataframe(
                df,
                task=task,
                source_desc=f"{single_path} | sheet={sheet}",
                model_name=model_name,
            )
        )

    if not records:
        raise ValueError(
            f"未能从 Single-task 文件中读取到 {task} 的 holdout fold-level 数据：{single_path}\n"
            f"要求：Holdout_Test sheet 或至少一个 sheet 同时包含 Fold 列与 "
            f"accuracy/macro_f1/recall/minority_f1 等指标列。"
        )

    return pd.DataFrame(records)

def read_single_task_files(single_specs, model_name=SINGLE_MODEL_NAME):
    """读取 5 个独立 Single-task Excel，并合并为 fold-level long dataframe。"""
    dfs = []
    for spec in single_specs:
        print(f"    - 读取 {spec['task']}：{spec['path']}")
        dfs.append(read_single_task_file(spec["path"], spec["task"], model_name=model_name))
    out = pd.concat(dfs, ignore_index=True)
    # source 仅用于调试，不进入后续 groupby；保留也不影响。
    return out.drop(columns=["source"], errors="ignore")


# ============================================================
# 2. 读取 Single-task model 数据：RESULT2.xlsx
# ============================================================

def read_single_task_result2(result2_path, model_name=SINGLE_MODEL_NAME):
    xls = pd.ExcelFile(result2_path)
    records = []

    for sheet in xls.sheet_names:
        task = find_task_from_sheet_name(sheet)
        if task is None or "singletask" not in sheet.lower().replace(" ", ""):
            continue

        df = pd.read_excel(result2_path, sheet_name=sheet)

        col_map = {}
        for c in df.columns:
            cn = normalize_metric_name(c)
            if cn is not None:
                col_map[c] = cn
        df = df.rename(columns=col_map)

        if "Fold" not in df.columns:
            possible_fold_cols = [c for c in df.columns if str(c).lower() == "fold"]
            if possible_fold_cols:
                df = df.rename(columns={possible_fold_cols[0]: "Fold"})

        if "Fold" not in df.columns:
            raise ValueError(f"Single-task sheet {sheet} 中未找到 Fold 列。")

        fold_df = df[df["Fold"].astype(str).str.match(r"^\d+$", na=False)].copy()
        fold_df["fold"] = fold_df["Fold"].astype(int)

        for _, row in fold_df.iterrows():
            fold = int(row["fold"])
            for metric in [
                "accuracy", "precision", "recall", "f1", "macro_f1", "auc", "auprc",
                "minority_f1", "minority_recall", "minority_precision",
                "pred_minor_rate", "true_minor_rate",
                "minority_tp", "minority_fn", "majority_fp", "majority_tn",
            ]:
                if metric in fold_df.columns and pd.notna(row.get(metric)):
                    records.append({
                        "model": model_name,
                        "task": task,
                        "fold": fold,
                        "metric": metric,
                        "value": float(row[metric]),
                    })

    if not records:
        raise ValueError(f"未能从 {result2_path} 中读取到 SingleTask T1~T5 数据。")

    return pd.DataFrame(records)


# ============================================================
# 3. 读取 MTL-style 数据：mtl_eval_*.xlsx
# ============================================================

def read_mtl_style_result(xlsx_path, model_name):
    xls = pd.ExcelFile(xlsx_path)
    records = []

    fold_sheets = []
    for s in xls.sheet_names:
        m = re.match(r"holdout_fold\s*(\d+)$", s, flags=re.IGNORECASE)
        if m:
            fold_sheets.append((int(m.group(1)), s))
    fold_sheets = sorted(fold_sheets, key=lambda x: x[0])

    if not fold_sheets:
        raise ValueError(f"{xlsx_path} 中未找到 holdout_fold1~holdout_fold5 sheet。")

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

            for metric in [
                "accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1",
                "auc", "auprc", "loss", "threshold", "minority_f1", "minority_recall",
                "minority_precision", "pred_minor_rate", "true_minor_rate", "minority_tp",
                "minority_fn", "majority_fp", "majority_tn",
            ]:
                if metric in df.columns and pd.notna(row.get(metric)):
                    val = row.get(metric)
                    if isinstance(val, str) and val.upper() in ["N/A", "NA", ""]:
                        continue
                    try:
                        val = float(val)
                    except Exception:
                        continue
                    records.append({
                        "model": model_name,
                        "task": task,
                        "fold": fold,
                        "metric": metric,
                        "value": val,
                    })

    if not records:
        raise ValueError(f"未能从 {xlsx_path} 中读取到 holdout fold 数据。")

    return pd.DataFrame(records)


# ============================================================
# 4. 汇总 fold-level long dataframe
# ============================================================

def summarize_long(df_long):
    summary = (
        df_long
        .groupby(["model", "task", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            sd=("value", lambda x: float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan),
            min=("value", "min"),
            max=("value", "max"),
            n=("value", "count"),
        )
    )
    return summary


def get_summary_value(summary, model, task, metric, stat="mean"):
    sub = summary[
        (summary["model"] == model) &
        (summary["task"] == task) &
        (summary["metric"] == metric)
    ]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0][stat])


def get_fold_metric_value(df_long, model, task, metric, fold):
    sub = df_long[
        (df_long["model"] == model) &
        (df_long["task"] == task) &
        (df_long["metric"] == metric) &
        (df_long["fold"] == fold)
    ]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["value"])


def get_model_task_mean_by_fold(df_long, model, fold, metric="macro_f1"):
    vals = []
    for task in TASK_ORDER:
        val = get_fold_metric_value(df_long, model, task, metric, fold)
        if pd.notna(val):
            vals.append(val)
    if len(vals) != len(TASK_ORDER):
        return np.nan
    return float(np.mean(vals))


def get_metric_mean_sd_for_table(summary, model, task_or_mean, metric):
    if task_or_mean != "Mean across t1–t5":
        mean_val = get_summary_value(summary, model, task_or_mean, metric, "mean")
        sd_val = get_summary_value(summary, model, task_or_mean, metric, "sd")
        return mean_val, sd_val

    means = [get_summary_value(summary, model, t, metric, "mean") for t in TASK_ORDER]
    mean_val = safe_mean(means)
    sd_val = sample_std(means)
    return mean_val, sd_val


# ============================================================
# 5. 生成 Table2 数据与格式化矩阵
# ============================================================

def build_table2_data(summary, model_order, delta_reference_model):
    """
    生成 Table2_data：保留原始数据表，且 Delta 改为 所有模型 - Ours。
    """
    rows = []
    delta_models = [m for m in model_order if m != delta_reference_model]

    for task in TASK_ORDER:
        row = {
            "任务": task,
            "临床含义": TASK_CLINICAL_NAME[task],
            "任务类型": TASK_TYPE[task],
        }

        for model in model_order:
            acc_mean, acc_sd = get_metric_mean_sd_for_table(summary, model, task, "accuracy")
            mf1_mean, mf1_sd = get_metric_mean_sd_for_table(summary, model, task, "macro_f1")
            rec_mean, rec_sd = get_metric_mean_sd_for_table(summary, model, task, "recall")
            row[f"{model} Accuracy"] = fmt_mean_sd(acc_mean, acc_sd)
            row[f"{model} Macro-F1"] = fmt_mean_sd(mf1_mean, mf1_sd)
            row[f"{model} Macro-recall"] = fmt_mean_sd(rec_mean, rec_sd)

        ours_mf1 = get_summary_value(summary, delta_reference_model, task, "macro_f1", "mean")
        for model in delta_models:
            model_mf1 = get_summary_value(summary, model, task, "macro_f1", "mean")
            row[f"Δ({model}–{delta_reference_model}) Macro-F1"] = model_mf1 - ours_mf1

        values = {model: get_summary_value(summary, model, task, "macro_f1", "mean") for model in model_order}
        valid_values = {k: v for k, v in values.items() if pd.notna(v)}
        row["Best model by Macro-F1"] = max(valid_values, key=valid_values.get) if valid_values else ""
        rows.append(row)

    mean_row = {
        "任务": "Mean across t1–t5",
        "临床含义": "平均表现",
        "任务类型": "-",
    }
    for model in model_order:
        for metric, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("recall", "Macro-recall")]:
            mean_val, sd_val = get_metric_mean_sd_for_table(summary, model, "Mean across t1–t5", metric)
            mean_row[f"{model} {label}"] = fmt_mean_sd(mean_val, sd_val)

    ours_mean, _ = get_metric_mean_sd_for_table(summary, delta_reference_model, "Mean across t1–t5", "macro_f1")
    for model in delta_models:
        model_mean, _ = get_metric_mean_sd_for_table(summary, model, "Mean across t1–t5", "macro_f1")
        mean_row[f"Δ({model}–{delta_reference_model}) Macro-F1"] = model_mean - ours_mean

    mean_values = {model: get_metric_mean_sd_for_table(summary, model, "Mean across t1–t5", "macro_f1")[0] for model in model_order}
    valid_mean_values = {k: v for k, v in mean_values.items() if pd.notna(v)}
    mean_row["Best model by Macro-F1"] = max(valid_mean_values, key=valid_mean_values.get) if valid_mean_values else ""
    rows.append(mean_row)

    return pd.DataFrame(rows)


def build_table2_matrix(summary, model_order):
    """
    生成 Table2 Excel 格式化表格所需矩阵，以及需要加粗的单元格位置。
    返回：
        display_df: 供检查/写入 Excel 的矩阵表
        rows: 纯二维字符串列表
        bold_positions: 需要 bold 的 (row_idx, col_idx) 集合（针对 rows）
    """
    header_row_1 = ["指标", "模型", "t1", "t2", "t3", "t4", "t5", "Mean across t1–t5"]
    header_row_2 = [
        "", "",
        TASK_CLINICAL_NAME["t1"],
        TASK_CLINICAL_NAME["t2"],
        TASK_CLINICAL_NAME["t3"],
        TASK_CLINICAL_NAME["t4"],
        TASK_CLINICAL_NAME["t5"],
        TASK_CLINICAL_NAME["Mean across t1–t5"],
    ]

    rows = [header_row_1, header_row_2]
    bold_positions = set()

    # 先计算每个 指标 × 列任务 下的最佳均值
    best_map = {}
    for metric_key, metric_label in METRIC_SPECS:
        for task in TASK_ORDER_WITH_MEAN:
            model_to_mean = {}
            for model in model_order:
                mean_val, _ = get_metric_mean_sd_for_table(summary, model, task, metric_key)
                model_to_mean[model] = mean_val
            valid_vals = [v for v in model_to_mean.values() if pd.notna(v)]
            if not valid_vals:
                best_map[(metric_key, task)] = set()
                continue
            best_val = max(valid_vals)
            best_models = {m for m, v in model_to_mean.items() if pd.notna(v) and np.isclose(v, best_val, atol=1e-12)}
            best_map[(metric_key, task)] = best_models

    for metric_key, metric_label in METRIC_SPECS:
        for mi, model in enumerate(model_order):
            row = [metric_label if mi == 0 else "", model]
            current_row_idx = len(rows)

            for task_col_idx, task in enumerate(TASK_ORDER_WITH_MEAN, start=2):
                mean_val, sd_val = get_metric_mean_sd_for_table(summary, model, task, metric_key)
                row.append(fmt_mean_sd(mean_val, sd_val))
                if model in best_map.get((metric_key, task), set()) and row[-1] != "":
                    bold_positions.add((current_row_idx, task_col_idx))

            rows.append(row)

    display_df = pd.DataFrame(rows)
    return display_df, rows, bold_positions



# ============================================================
# 6. 生成 Fig2A 数据
# ============================================================

def build_fig2a_data(summary, df_long, model_order, primary_model):
    rows = []
    for task in TASK_ORDER:
        row = {"任务": task, "临床含义": TASK_CLINICAL_NAME[task]}
        for model in model_order:
            row[f"{model} mean"] = get_summary_value(summary, model, task, "macro_f1", "mean")
            row[f"{model} SD"] = get_summary_value(summary, model, task, "macro_f1", "sd")
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = f"{primary_model} mean"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

    mean_row = {"任务": "Mean across t1–t5", "临床含义": "平均表现"}
    for model in model_order:
        fold_means = []
        for fold in sorted(df_long["fold"].dropna().unique()):
            mean_val = get_model_task_mean_by_fold(df_long, model, fold, metric="macro_f1")
            if pd.notna(mean_val):
                fold_means.append(mean_val)
        mean_row[f"{model} mean"] = safe_mean(fold_means)
        mean_row[f"{model} SD"] = sample_std(fold_means)

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.insert(0, "排序", range(1, len(df) + 1))
    return df


# ============================================================
# 6b. 生成 Fig2F 数据：Accuracy 版本的 Fig2A
# ============================================================

def build_fig2f_data(summary, df_long, model_order, primary_model):
    """
    Fig2F:
    任务级 holdout Accuracy 数据。

    结构与 Fig2A_data 完全一致：
    - 行：t1~t5 + Mean across t1–t5
    - 列：每个模型的 mean / SD

    与 Fig2A 的区别：
    - Fig2A 使用 macro_f1
    - Fig2F 使用 accuracy
    """
    rows = []
    metric = "accuracy"

    for task in TASK_ORDER:
        row = {"任务": task, "临床含义": TASK_CLINICAL_NAME[task]}
        for model in model_order:
            row[f"{model} mean"] = get_summary_value(summary, model, task, metric, "mean")
            row[f"{model} SD"] = get_summary_value(summary, model, task, metric, "sd")
        rows.append(row)

    df = pd.DataFrame(rows)

    # 与 Fig2A 保持一致：默认按 primary_model 的 Accuracy 排序。
    # 如果后续画图希望固定 t1~t5 顺序，可在绘图脚本中使用 TASK_ORDER 强制排序。
    sort_col = f"{primary_model} mean"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

    # Mean across t1–t5：按 fold 先求每个 fold 的五任务 Accuracy 平均，再算 mean ± sample SD
    mean_row = {"任务": "Mean across t1–t5", "临床含义": "平均表现"}
    for model in model_order:
        fold_means = []
        for fold in sorted(df_long["fold"].dropna().unique()):
            mean_val = get_model_task_mean_by_fold(df_long, model, fold, metric=metric)
            if pd.notna(mean_val):
                fold_means.append(mean_val)
        mean_row[f"{model} mean"] = safe_mean(fold_means)
        mean_row[f"{model} SD"] = sample_std(fold_means)

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df.insert(0, "排序", range(1, len(df) + 1))
    return df



# ============================================================
# 6c. 生成 Fig1g 数据：Holdout Macro-F1 箱型图 fold-level 长表
# ============================================================

def build_fig1g_data(df_long, model_order):
    """
    生成 Fig1g_data：用于 2×3 箱型图的 fold-level Holdout Macro-F1 长表。

    图中 6 个子图：
    - t1, t2, t3, t4, t5
    - Mean across t1–t5

    关键点：
    - 箱型图必须使用 fold-level 原始值，不能使用 Full_holdout_metrics 中的 mean / sd / min / max 反推。
    - t1~t5：直接取每个 model × task × fold 的 macro_f1。
    - Mean across t1–t5：先在同一个 model × fold 内对 t1~t5 的 macro_f1 求平均，
      再把 5 个 fold 的平均值作为箱型图数据。
    """
    metric = "macro_f1"

    base = df_long[
        (df_long["task"].isin(TASK_ORDER)) &
        (df_long["metric"] == metric)
    ].copy()

    rows = []

    # t1~t5 的 fold-level Macro-F1
    for _, row in base.iterrows():
        task = str(row["task"]).strip()
        model = str(row["model"]).strip()
        fold = int(row["fold"])
        value = float(row["value"])

        rows.append({
            "任务": task,
            "临床含义": TASK_CLINICAL_NAME.get(task, ""),
            "模型": model,
            "模型显示顺序": model_order.index(model) + 1 if model in model_order else np.nan,
            "fold": fold,
            "Holdout Macro-F1": value,
        })

    # Mean across t1–t5 的 fold-level Macro-F1
    folds = sorted(base["fold"].dropna().unique())
    for model in model_order:
        for fold in folds:
            vals = []
            for task in TASK_ORDER:
                val = get_fold_metric_value(df_long, model, task, metric, fold)
                if pd.notna(val):
                    vals.append(val)

            # 为了避免某个任务缺失时平均值被高估/低估，这里要求 5 个任务齐全
            if len(vals) != len(TASK_ORDER):
                continue

            rows.append({
                "任务": "Mean across t1–t5",
                "临床含义": TASK_CLINICAL_NAME["Mean across t1–t5"],
                "模型": model,
                "模型显示顺序": model_order.index(model) + 1 if model in model_order else np.nan,
                "fold": int(fold),
                "Holdout Macro-F1": float(np.mean(vals)),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "任务", "临床含义", "模型", "模型显示顺序", "fold", "Holdout Macro-F1"
        ])

    task_order_with_mean = TASK_ORDER + ["Mean across t1–t5"]
    out["任务"] = pd.Categorical(out["任务"], categories=task_order_with_mean, ordered=True)
    out["模型"] = pd.Categorical(out["模型"], categories=model_order, ordered=True)
    out = out.sort_values(["任务", "模型", "fold"]).reset_index(drop=True)
    out["任务"] = out["任务"].astype(str)
    out["模型"] = out["模型"].astype(str)

    return out


def build_fig1g_summary(fig1g_data):
    """
    生成 Fig1g_summary：
    用于核对箱型图数据的 mean / SD / SEM / median / IQR。
    """
    if fig1g_data.empty:
        return pd.DataFrame(columns=[
            "任务", "模型", "n", "mean", "sd", "sem", "median", "q1", "q3", "min", "max"
        ])

    rows = []
    for (task, model), g in fig1g_data.groupby(["任务", "模型"], sort=False):
        vals = pd.to_numeric(g["Holdout Macro-F1"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        sd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
        rows.append({
            "任务": task,
            "模型": model,
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "sd": sd,
            "sem": sd / np.sqrt(len(vals)) if len(vals) >= 2 else np.nan,
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. 生成 Fig2B 数据
# ============================================================

def build_fig2b_data(df_long, model_order):
    rows = []
    mtl_models = [m for m in model_order if m != SINGLE_MODEL_NAME]
    task_list = TASK_ORDER + ["Mean across t1–t5"]
    folds = sorted(df_long["fold"].dropna().unique())

    for task in task_list:
        row = {"任务": task, "临床含义": TASK_CLINICAL_NAME.get(task, "平均表现")}
        for model in mtl_models:
            deltas = []
            if task != "Mean across t1–t5":
                for fold in folds:
                    s = get_fold_metric_value(df_long, SINGLE_MODEL_NAME, task, "macro_f1", fold)
                    m = get_fold_metric_value(df_long, model, task, "macro_f1", fold)
                    if pd.notna(s) and pd.notna(m):
                        deltas.append(m - s)
            else:
                for fold in folds:
                    s = get_model_task_mean_by_fold(df_long, SINGLE_MODEL_NAME, fold, metric="macro_f1")
                    m = get_model_task_mean_by_fold(df_long, model, fold, metric="macro_f1")
                    if pd.notna(s) and pd.notna(m):
                        deltas.append(m - s)
            row[f"Δ({model}–Single) mean"] = safe_mean(deltas)
            row[f"Δ({model}–Single) SD"] = sample_std(deltas)
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 8. 生成 Fig2D 数据
# ============================================================

def build_fig2d_data(summary, model_order):
    rows = []
    for task in ["t3", "t4", "t5"]:
        for model in model_order:
            row = {
                "任务": task,
                "临床含义": TASK_CLINICAL_NAME[task],
                "模型": model,
                "Minority precision mean": get_summary_value(summary, model, task, "minority_precision", "mean"),
                "Minority precision SD": get_summary_value(summary, model, task, "minority_precision", "sd"),
                "Minority recall mean": get_summary_value(summary, model, task, "minority_recall", "mean"),
                "Minority recall SD": get_summary_value(summary, model, task, "minority_recall", "sd"),
                "Minority F1 mean": get_summary_value(summary, model, task, "minority_f1", "mean"),
                "Minority F1 SD": get_summary_value(summary, model, task, "minority_f1", "sd"),
                "True minority rate mean": get_summary_value(summary, model, task, "true_minor_rate", "mean"),
                "Pred minority rate mean": get_summary_value(summary, model, task, "pred_minor_rate", "mean"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 8b. 生成 Fig2E 箱型图数据：fold-level minority F1 长表
# ============================================================

def build_fig2e_data(df_long, model_order):
    """
    生成 Fig2E_data：用于箱型图的 fold-level 长表。

    为什么需要单独保存 Fig2E_data：
    - 箱型图必须使用 fold-level 原始值，不能用 mean±SD 反推。
    - 每一行对应一个 task × model × fold。
    - 主绘图指标为 Minority F1；同时保留 minority precision/recall、pred/true minority rate，方便补充表或排查。
    """
    binary_tasks = ["t3", "t4", "t5"]
    metric_keep = [
        "minority_f1",
        "minority_precision",
        "minority_recall",
        "pred_minor_rate",
        "true_minor_rate",
        "macro_f1",
        "auc",
        "auprc",
    ]

    sub = df_long[
        (df_long["task"].isin(binary_tasks)) &
        (df_long["metric"].isin(metric_keep))
    ].copy()

    if sub.empty:
        return pd.DataFrame(columns=[
            "任务", "临床含义", "任务类型", "模型", "模型显示顺序", "fold",
            "Minority F1", "Minority precision", "Minority recall",
            "Pred minority rate", "True minority rate", "Macro-F1", "AUROC", "AUPRC"
        ])

    pivot = (
        sub.pivot_table(
            index=["task", "model", "fold"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns.name = None

    rename_map = {
        "task": "任务",
        "model": "模型",
        "minority_f1": "Minority F1",
        "minority_precision": "Minority precision",
        "minority_recall": "Minority recall",
        "pred_minor_rate": "Pred minority rate",
        "true_minor_rate": "True minority rate",
        "macro_f1": "Macro-F1",
        "auc": "AUROC",
        "auprc": "AUPRC",
    }
    pivot = pivot.rename(columns=rename_map)

    pivot["临床含义"] = pivot["任务"].map(TASK_CLINICAL_NAME)
    pivot["任务类型"] = pivot["任务"].map(TASK_TYPE)
    model_order_map = {m: i + 1 for i, m in enumerate(model_order)}
    pivot["模型显示顺序"] = pivot["模型"].map(model_order_map)

    # 保证核心列存在，避免某些模型缺少个别指标时报错。
    expected_metric_cols = [
        "Minority F1", "Minority precision", "Minority recall",
        "Pred minority rate", "True minority rate", "Macro-F1", "AUROC", "AUPRC"
    ]
    for col in expected_metric_cols:
        if col not in pivot.columns:
            pivot[col] = np.nan

    out_cols = [
        "任务", "临床含义", "任务类型", "模型", "模型显示顺序", "fold",
        "Minority F1", "Minority precision", "Minority recall",
        "Pred minority rate", "True minority rate", "Macro-F1", "AUROC", "AUPRC",
    ]
    pivot = pivot[out_cols]
    pivot["任务"] = pd.Categorical(pivot["任务"], categories=binary_tasks, ordered=True)
    pivot["模型"] = pd.Categorical(pivot["模型"], categories=model_order, ordered=True)
    pivot = pivot.sort_values(["任务", "模型", "fold"]).reset_index(drop=True)
    pivot["任务"] = pivot["任务"].astype(str)
    pivot["模型"] = pivot["模型"].astype(str)
    return pivot


def build_fig2e_summary(fig2e_data):
    """生成 Fig2E_summary：箱型图数据的核对汇总。"""
    if fig2e_data.empty:
        return pd.DataFrame(columns=[
            "任务", "模型", "n", "mean", "sd", "sem", "median", "q1", "q3", "min", "max"
        ])

    rows = []
    for (task, model), g in fig2e_data.groupby(["任务", "模型"], sort=False):
        vals = pd.to_numeric(g["Minority F1"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        sd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
        rows.append({
            "任务": task,
            "模型": model,
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "sd": sd,
            "sem": sd / np.sqrt(len(vals)) if len(vals) >= 2 else np.nan,
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    return pd.DataFrame(rows)


# ============================================================
# 9. 生成 Full_holdout_metrics
# ============================================================

def build_full_holdout_metrics(summary, model_order):
    out = summary.copy()
    out["临床含义"] = out["task"].map(TASK_CLINICAL_NAME)
    out["任务类型"] = out["task"].map(TASK_TYPE)
    out["model"] = pd.Categorical(out["model"], categories=model_order, ordered=True)
    out["task"] = pd.Categorical(out["task"], categories=TASK_ORDER, ordered=True)
    out = out[["model", "task", "临床含义", "任务类型", "metric", "mean", "sd", "min", "max", "n"]].sort_values(["model", "task", "metric"])
    out["model"] = out["model"].astype(str)
    out["task"] = out["task"].astype(str)
    return out


# ============================================================
# 10. README
# ============================================================

def build_readme(single_source_rows, mtl_specs, model_order, primary_model, delta_reference_model):
    rows = [["项目", "说明"]]

    for label, desc in single_source_rows:
        rows.append([label, desc])

    for idx, spec in enumerate(mtl_specs, start=1):
        rows.append([f"输入文件：MTL {idx}", f"{spec['name']} | {spec['path']}"])

    rows.extend([
        ["模型顺序", " -> ".join(model_order)],
        ["Fig2A 排序依据", f"按 {primary_model} 的 Macro-F1 mean 从高到低排序"],
        ["Table2 Delta 参考模型", delta_reference_model],
        ["Single-task model 来源", "优先读取 5 个独立 Single-task Excel；每个文件对应 t1~t5 中一个任务"],
        ["MTL-style 结果来源", "每个 mtl_eval_*.xlsx 中 holdout_fold1~5 sheets"],
        ["Table2", "Excel 可编辑格式化表格，显示 Accuracy / Macro-F1 / Macro-recall，且每列最优值加粗"],
        ["Table2_data", f"Table2 原始数据表；其中 Delta 定义为 Δ(模型–{delta_reference_model})"],
        ["Table2_matrix", "Table2 格式化表格对应的二维矩阵文本，用于排查/核对"],
        ["Fig2A_data", "任务级 holdout Macro-F1 forest/dot plot 数据"],
        ["Fig2F_data", "任务级 holdout Accuracy grouped bar/forest plot 数据；结构与 Fig2A_data 相同，但指标由 Macro-F1 改为 Accuracy"],
        ["Fig1g_data", "用于 Fig1g 2×3 箱型图的 fold-level Holdout Macro-F1 长表；包含 t1~t5 和 Mean across t1–t5"],
        ["Fig1g_summary", "Fig1g_data 的 mean / SD / SEM / median / IQR 核对汇总"],
        ["Fig2B_data", "所有 MTL 模型相对于 Single-task model 的 ΔMacro-F1 增益数据"],
        ["Fig2D_data", "t3–t5 少数类 precision / recall / F1 汇总数据；若 Single-task 缺少少数类指标则为空"],
        ["Fig2E_data", "用于 Fig2E 箱型图的 fold-level minority F1 长表；每行对应 task × model × fold"],
        ["Fig2E_summary", "Fig2E_data 的 mean / SD / SEM / median / IQR 核对汇总"],
        ["Full_holdout_metrics", "所有模型、任务、指标的 mean / SD / min / max 长表"],
        ["SD 计算方式", "基于 fold-level 结果计算 sample standard deviation, ddof=1"],
        ["Fig2C 说明", "ROC 曲线需要样本级 y_true 与 y_score/probability；summary Excel 只能提供 AUROC 均值与 SD"],
    ])

    return pd.DataFrame(rows[1:], columns=rows[0])

# ============================================================
# 11. Excel 输出和格式
# ============================================================

def write_table2_formatted_sheet(workbook, worksheet, table2_rows, bold_positions, model_order):
    """把 Table2 直接写成 Excel 可编辑格式化表格，不生成图片。"""
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
    blank_metric_fmt = workbook.add_format({
        "bg_color": "#E2F0D9",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
    })

    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(2, 2)
    worksheet.set_row(0, 24)
    worksheet.set_row(1, 42)
    worksheet.set_column(0, 0, 13)
    worksheet.set_column(1, 1, 24)
    worksheet.set_column(2, 6, 22)
    worksheet.set_column(7, 7, 26)

    # 表头两行
    for c, value in enumerate(table2_rows[0]):
        worksheet.write(0, c, value, header_fmt)
    for c, value in enumerate(table2_rows[1]):
        worksheet.write(1, c, value, subheader_fmt)

    n_models = len(model_order)
    row_idx = 2
    while row_idx < len(table2_rows):
        metric_label = table2_rows[row_idx][0]
        block_start = row_idx
        block_end = row_idx + n_models - 1

        # 合并指标列，视觉上接近用户截图中的分块表格
        if n_models > 1:
            worksheet.merge_range(block_start, 0, block_end, 0, metric_label, metric_fmt)
        else:
            worksheet.write(block_start, 0, metric_label, metric_fmt)

        for r in range(block_start, block_end + 1):
            excel_r = r
            worksheet.set_row(excel_r, 26)
            # 指标列已合并，非首行不用重复写
            if r != block_start and n_models <= 1:
                worksheet.write(excel_r, 0, "", blank_metric_fmt)
            worksheet.write(excel_r, 1, table2_rows[r][1], model_fmt)
            for c in range(2, len(table2_rows[r])):
                fmt = body_bold_fmt if (r, c) in bold_positions else body_fmt
                worksheet.write(excel_r, c, table2_rows[r][c], fmt)

        row_idx += n_models


def write_output_excel(output_path, table2_data, table2_matrix, table2_rows, table2_bold_positions, model_order, fig2a, fig2f, fig1g, fig1g_summary, fig2b, fig2d, fig2e, fig2e_summary, full_metrics, readme):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # Table2：直接写成 Excel 格式化表格，不插入图片
        ws_table2 = workbook.add_worksheet("Table2")
        writer.sheets["Table2"] = ws_table2
        write_table2_formatted_sheet(workbook, ws_table2, table2_rows, table2_bold_positions, model_order)

        table2_data.to_excel(writer, sheet_name="Table2_data", index=False)
        table2_matrix.to_excel(writer, sheet_name="Table2_matrix", index=False, header=False)
        fig2a.to_excel(writer, sheet_name="Fig2A_data", index=False)
        fig2f.to_excel(writer, sheet_name="Fig2F_data", index=False)
        fig1g.to_excel(writer, sheet_name="Fig1g_data", index=False)
        fig1g_summary.to_excel(writer, sheet_name="Fig1g_summary", index=False)
        fig2b.to_excel(writer, sheet_name="Fig2B_data", index=False)
        fig2d.to_excel(writer, sheet_name="Fig2D_data", index=False)
        fig2e.to_excel(writer, sheet_name="Fig2E_data", index=False)
        fig2e_summary.to_excel(writer, sheet_name="Fig2E_summary", index=False)
        full_metrics.to_excel(writer, sheet_name="Full_holdout_metrics", index=False)
        readme.to_excel(writer, sheet_name="README", index=False)

        header_fmt = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F4E79",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        })
        neg_fmt = workbook.add_format({"font_color": "#C00000", "num_format": "+0.0000;-0.0000;0.0000"})
        pos_fmt = workbook.add_format({"font_color": "#0070C0", "num_format": "+0.0000;-0.0000;0.0000"})

        for sheet_name, df in [
            ("Table2_data", table2_data),
            ("Table2_matrix", table2_matrix),
            ("Fig2A_data", fig2a),
            ("Fig2F_data", fig2f),
            ("Fig1g_data", fig1g),
            ("Fig1g_summary", fig1g_summary),
            ("Fig2B_data", fig2b),
            ("Fig2D_data", fig2d),
            ("Fig2E_data", fig2e),
            ("Fig2E_summary", fig2e_summary),
            ("Full_holdout_metrics", full_metrics),
            ("README", readme),
        ]:
            ws = writer.sheets[sheet_name]
            if sheet_name != "Table2_matrix":
                ws.freeze_panes(1, 0)
                for col_num, value in enumerate(df.columns.values):
                    ws.write(0, col_num, value, header_fmt)
            for i, col in enumerate(df.columns):
                max_len = max([len(str(col))] + [len(str(x)) for x in df[col].head(150).fillna("").tolist()])
                ws.set_column(i, i, min(max_len + 2, 42))

        # Table2_data：突出 Δ
        ws = writer.sheets["Table2_data"]
        delta_cols = [i for i, c in enumerate(table2_data.columns) if str(c).startswith("Δ")]
        for c in delta_cols:
            ws.conditional_format(1, c, len(table2_data), c, {"type": "cell", "criteria": ">", "value": 0, "format": pos_fmt})
            ws.conditional_format(1, c, len(table2_data), c, {"type": "cell", "criteria": "<", "value": 0, "format": neg_fmt})

        # Fig2B：保持原逻辑，仍是 MTL 模型 - Single-task model
        ws = writer.sheets["Fig2B_data"]
        for c, col in enumerate(fig2b.columns):
            if str(col).startswith("Δ"):
                ws.conditional_format(1, c, len(fig2b), c, {"type": "cell", "criteria": ">", "value": 0, "format": pos_fmt})
                ws.conditional_format(1, c, len(fig2b), c, {"type": "cell", "criteria": "<", "value": 0, "format": neg_fmt})

        writer.sheets["README"].set_column(0, 0, 28)
        writer.sheets["README"].set_column(1, 1, 120)


# ============================================================
# 12. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "整理 Single-task 与多个 MTL-style 结果文件，"
            "生成 Table2/Fig2A/Fig2B/Fig2D/Fig2E/Fig2F 数据 Excel。"
        )
    )
    parser.add_argument("--single_file", action="append", default=None, help="可重复传入 5 次，格式：t1=文件路径。若不传，则使用脚本顶部 SINGLE_TASK_FILES 列表。")
    parser.add_argument("--single_combined", default=None, help="可选：旧版合并后的 RESULT2.xlsx。如果使用 5 个独立 single 文件，不需要传这个参数。")
    parser.add_argument("--mtl_file", action="append", default=None, help="可重复传入，格式：模型名=文件路径。如果不传，则使用脚本顶部 MTL_RESULT_FILES 列表。")
    parser.add_argument("--out", default=r"xx_path", help="输出 Excel 文件名")
    args = parser.parse_args()

    output_path = Path(args.out)

    # Single-task 输入：优先命令行 --single_file，其次脚本顶部 SINGLE_TASK_FILES，最后才兼容旧版 --single_combined。
    raw_single_specs = parse_cli_single_files(args.single_file) if args.single_file else SINGLE_TASK_FILES
    use_separate_single_files = bool(raw_single_specs)

    if use_separate_single_files:
        single_specs = normalize_single_task_specs(raw_single_specs)
        single_source_rows = [(f"输入文件：Single-task {spec['task']}", str(spec["path"])) for spec in single_specs]
    else:
        if not args.single_combined:
            raise ValueError(
                "当前脚本默认按 5 个独立 Single-task Excel 读取。\n"
                "请在脚本顶部 SINGLE_TASK_FILES 中配置 t1~t5 文件，或用 --single_file t1=... 重复传入；\n"
                "如果你仍要读取旧版合并 RESULT2.xlsx，请传 --single_combined 路径。"
            )
        single_combined_path = Path(args.single_combined)
        single_source_rows = [("输入文件：Single-task combined", str(single_combined_path))]

    raw_mtl_specs = parse_cli_mtl_files(args.mtl_file) if args.mtl_file else MTL_RESULT_FILES
    mtl_specs = normalize_mtl_specs(raw_mtl_specs)
    model_order = build_model_order(mtl_specs)
    primary_model = get_primary_model_name(model_order)
    delta_reference_model = get_table2_delta_reference_model(model_order)

    # 文件存在性检查
    if use_separate_single_files:
        for spec in single_specs:
            if not spec["path"].exists():
                raise FileNotFoundError(f"未找到 Single-task 输入文件：{spec['task']} | {spec['path']}")
    else:
        if not single_combined_path.exists():
            raise FileNotFoundError(f"未找到 Single-task combined 输入文件：{single_combined_path}")

    for spec in mtl_specs:
        if not spec["path"].exists():
            raise FileNotFoundError(f"未找到 MTL 输入文件：{spec['name']} | {spec['path']}")

    print("[1] 读取 Single-task model 数据...")
    if use_separate_single_files:
        df_single = read_single_task_files(single_specs, model_name=SINGLE_MODEL_NAME)
    else:
        df_single = read_single_task_result2(single_combined_path, model_name=SINGLE_MODEL_NAME)
    df_list = [df_single]

    for idx, spec in enumerate(mtl_specs, start=1):
        print(f"[{idx + 1}] 读取 MTL-style 数据：{spec['name']} ...")
        df_list.append(read_mtl_style_result(spec["path"], spec["name"]))

    print("[合并] 合并 fold-level 数据...")
    df_long = pd.concat(df_list, ignore_index=True)
    summary = summarize_long(df_long)

    print("[生成] 生成 Table2 / Fig2A / Fig2B / Fig2D / Fig2E / Fig2F 数据...")
    table2_data = build_table2_data(summary, model_order, delta_reference_model)
    table2_matrix, table2_rows, table2_bold_positions = build_table2_matrix(summary, model_order)
    fig2a = build_fig2a_data(summary, df_long, model_order, primary_model)
    fig2f = build_fig2f_data(summary, df_long, model_order, primary_model)
    fig1g = build_fig1g_data(df_long, model_order)
    fig1g_summary = build_fig1g_summary(fig1g)
    fig2b = build_fig2b_data(df_long, model_order)
    fig2d = build_fig2d_data(summary, model_order)
    fig2e = build_fig2e_data(df_long, model_order)
    fig2e_summary = build_fig2e_summary(fig2e)
    full_metrics = build_full_holdout_metrics(summary, model_order)

    readme = build_readme(single_source_rows, mtl_specs, model_order, primary_model, delta_reference_model)

    print("[写入] 写入 Excel...")
    write_output_excel(
        output_path,
        table2_data,
        table2_matrix,
        table2_rows,
        table2_bold_positions,
        model_order,
        fig2a,
        fig2f,
        fig1g,
        fig1g_summary,
        fig2b,
        fig2d,
        fig2e,
        fig2e_summary,
        full_metrics,
        readme,
    )

    print("[完成]")
    print(f"输出 Excel：{output_path.resolve()}")
    print("\n模型顺序：")
    for model in model_order:
        print(f"  - {model}")
    print("\nSheets:")
    print("  - Table2 (Excel格式化表格)")
    print("  - Table2_data")
    print("  - Table2_matrix")
    print("  - Fig2A_data")
    print("  - Fig2F_data")
    print("  - Fig1g_data")
    print("  - Fig1g_summary")
    print("  - Fig2B_data")
    print("  - Fig2D_data")
    print("  - Fig2E_data")
    print("  - Fig2E_summary")
    print("  - Full_holdout_metrics")
    print("  - README")


if __name__ == "__main__":
    main()

