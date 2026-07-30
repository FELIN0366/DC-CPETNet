#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate Fold ROC Data for Figure 2C
用途：运行MTL五折后会生成results/fig2c_roc_raw/*.json，扫描 results/fig2c_roc_raw/*.json，聚合五折 ROC 数据，生成 Excel。
- 读取各 fold 的 ROC 原始数据 (从 holdout evaluation 导出)
- 计算五折平均 ROC 曲线
- 输出 Excel (5 sheets)

使用方法：
    python scripts/aggregate_fig2c_roc_excel.py

后续运行：Fig2C.py会读取这个 Excel 来绘制最终的 ROC 曲线。

创建日期: 2026-05-24
"""

import sys
import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# =============================================================================
# 任务配置 (固定)
# =============================================================================

BINARY_TASKS = ["t3", "t4", "t5"]

TASK_NAME_MAP = {
    "t3": "标准心电运动负荷试验",
    "t4": "运动中换气肺功能",
    "t5": "心率储备",
}

EXPECTED_FOLDS = [1, 2, 3, 4, 5]

# =============================================================================
# 输出模型名配置
# =============================================================================
# 这里用于重新定义导出 Excel 中的 model_name。
# - None：沿用 JSON / sample_scores 中原始 model_name
# - 字符串：强制把 sample_scores、roc_points_fold、roc_points_mean、auc_summary 中的 model_name 全部改成该名称
#
# 例如：
# OUTPUT_MODEL_NAME = "Single-task model"
# OUTPUT_MODEL_NAME = "Shared Bottom"
# OUTPUT_MODEL_NAME = "Our method"


# =============================================================================
# ROC Curve Interpolation
# =============================================================================

def interpolate_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    n_points: int = 101
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 ROC 曲线插值到固定 FPR 网格

    Args:
        fpr: FPR 点
        tpr: TPR 点
        n_points: 插值点数 (默认 101: 0.00, 0.01, ..., 1.00)

    Returns:
        (fpr_grid, tpr_interp)
    """
    fpr_grid = np.linspace(0.0, 1.0, n_points)

    # 处理重复 FPR 值 (保留最大 TPR)
    unique_fpr = []
    unique_tpr = []
    for i in range(len(fpr)):
        if i == 0 or fpr[i] > fpr[i-1]:
            unique_fpr.append(fpr[i])
            unique_tpr.append(tpr[i])
        elif fpr[i] == fpr[i-1]:
            unique_tpr[-1] = max(unique_tpr[-1], tpr[i])

    unique_fpr = np.array(unique_fpr)
    unique_tpr = np.array(unique_tpr)

    # 插值
    tpr_interp = np.interp(fpr_grid, unique_fpr, unique_tpr)

    # 确保 (0, 0) 和 (1, 1) 端点
    tpr_interp[0] = 0.0
    tpr_interp[-1] = 1.0

    return fpr_grid, tpr_interp


def compute_mean_roc(
    fpr_tpr_list: List[Tuple[np.ndarray, np.ndarray]],
    n_points: int = 101
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算五折平均 ROC 曲线

    Args:
        fpr_tpr_list: 各 fold 的 (fpr, tpr) 列表
        n_points: 插值网格大小

    Returns:
        (fpr_grid, mean_tpr, std_tpr)
    """
    fpr_grid = np.linspace(0.0, 1.0, n_points)

    all_tpr_interp = []
    for fpr, tpr in fpr_tpr_list:
        _, tpr_interp = interpolate_roc_curve(fpr, tpr, n_points)
        all_tpr_interp.append(tpr_interp)

    all_tpr_interp = np.array(all_tpr_interp)  # shape: [n_folds, n_points]

    mean_tpr = np.mean(all_tpr_interp, axis=0)
    std_tpr = np.std(all_tpr_interp, axis=0, ddof=1)  # sample std

    return fpr_grid, mean_tpr, std_tpr


# =============================================================================
# Model name utilities
# =============================================================================

def resolve_output_model_name(
    model_type: str,
    sample_scores_df: pd.DataFrame,
    model_name_override: Optional[str] = None
) -> str:
    """
    解析最终写入 Excel 的 model_name。

    优先级：
    1. 如果传入 model_name_override，则直接使用该名称；
    2. 否则尝试从 sample_scores_df 中根据 model_type 读取原始 model_name；
    3. 如果读取失败，则退回 model_type。
    """
    if model_name_override is not None and str(model_name_override).strip() != "":
        return str(model_name_override).strip()

    if (
        sample_scores_df is not None
        and len(sample_scores_df) > 0
        and "model_type" in sample_scores_df.columns
        and "model_name" in sample_scores_df.columns
    ):
        sub = sample_scores_df[sample_scores_df["model_type"] == model_type]
        if not sub.empty and pd.notna(sub["model_name"].iloc[0]):
            return str(sub["model_name"].iloc[0])

    return str(model_type)


def apply_model_name_override_to_raw_tables(
    sample_scores_df: pd.DataFrame,
    roc_points_fold_df: pd.DataFrame,
    model_name_override: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    如果用户指定 model_name_override，则同步修改原始 sheet 中的 model_name，
    保证 sample_scores / roc_points_fold / roc_points_mean / auc_summary 口径一致。
    """
    if model_name_override is None or str(model_name_override).strip() == "":
        return sample_scores_df, roc_points_fold_df

    out_name = str(model_name_override).strip()

    sample_scores_df = sample_scores_df.copy()
    roc_points_fold_df = roc_points_fold_df.copy()

    if len(sample_scores_df) > 0:
        sample_scores_df["model_name"] = out_name

    if len(roc_points_fold_df) > 0:
        if "model_name" in roc_points_fold_df.columns:
            roc_points_fold_df["model_name"] = out_name
        else:
            roc_points_fold_df.insert(0, "model_name", out_name)

    return sample_scores_df, roc_points_fold_df


# =============================================================================
# Main Workflow
# =============================================================================

def aggregate_fold_roc_data(
    input_dir: str,
    output_excel: str,
    model_name_override: Optional[str] = None
):
    """
    聚合五折 ROC 数据，生成 Excel

    Args:
        input_dir: ROC JSON 文件目录 (如 results/fig2c_roc_raw)
        output_excel: 输出 Excel 文件路径 (如 figures/fig2c_roc_curve_data.xlsx)
        model_name_override: 可选；强制重新定义输出 Excel 中的 model_name
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    # 扫描 JSON 文件
    json_files = sorted(input_path.glob("*_roc_data.json"))

    if not json_files:
        raise FileNotFoundError(f"输入目录中没有 JSON 文件: {input_dir}")

    # 提取 fold 编号
    fold_to_file = {}
    for f in json_files:
        match = re.search(r"fold(\d+)", f.name)
        if match:
            fold_num = int(match.group(1))
            fold_to_file[fold_num] = f

    # 检查 fold 完整性
    missing_folds = [f for f in EXPECTED_FOLDS if f not in fold_to_file]

    if missing_folds:
        print(f"\n[Warning] Missing folds: {missing_folds}")
        print(f"  已找到: {list(fold_to_file.keys())}")
        print(f"  将继续处理可用 fold...")
    else:
        print(f"\n[Fold Discovery] 找到全部 fold: {list(fold_to_file.keys())}")

    # 收集数据
    all_sample_scores = []
    all_roc_points_fold = []
    all_run_info = []

    # AUC 汇总数据结构: {model_type: {task_key: {fold: auc}}}
    auc_by_model_task_fold = {}

    # FPR/TPR 数据结构: {model_type: {task_key: [(fpr, tpr), ...]}}
    fpr_tpr_by_model_task = {}

    for fold in EXPECTED_FOLDS:
        if fold not in fold_to_file:
            continue

        json_path = fold_to_file[fold]

        print(f"\n[Fold {fold}] 加载: {json_path.name}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 收集 sample_scores
        if "sample_scores" in data:
            all_sample_scores.extend(data["sample_scores"])
            print(f"  sample_scores: {len(data['sample_scores'])} 条记录")

        # 收集 roc_points_fold
        if "roc_points_fold" in data:
            all_roc_points_fold.extend(data["roc_points_fold"])
            print(f"  roc_points_fold: {len(data['roc_points_fold'])} 条记录")

            # 提取 AUC 和 FPR/TPR
            for point in data["roc_points_fold"]:
                model_type = point.get("model_type", "unknown")
                task_key = point.get("task_key")

                if model_type not in auc_by_model_task_fold:
                    auc_by_model_task_fold[model_type] = {}
                    fpr_tpr_by_model_task[model_type] = {}

                if task_key not in auc_by_model_task_fold[model_type]:
                    auc_by_model_task_fold[model_type][task_key] = {}
                    fpr_tpr_by_model_task[model_type][task_key] = []

                # 存储 AUC (每个 fold 只存一次)
                if fold not in auc_by_model_task_fold[model_type][task_key]:
                    auc_by_model_task_fold[model_type][task_key][fold] = point.get("auc")

        # 收集 run_info
        if "run_info" in data:
            all_run_info.append(data["run_info"])

    # 构建 DataFrame
    print(f"\n{'='*60}")
    print("[DataFrame 构建]")
    print(f"{'='*60}")

    # Sheet 1: sample_scores
    sample_scores_df = pd.DataFrame(all_sample_scores)
    print(f"  sample_scores: {len(sample_scores_df)} 行")

    # Sheet 2: roc_points_fold
    roc_points_fold_df = pd.DataFrame(all_roc_points_fold)
    print(f"  roc_points_fold: {len(roc_points_fold_df)} 行")

    # 如果用户指定了输出模型名，则同步覆盖原始表中的 model_name
    sample_scores_df, roc_points_fold_df = apply_model_name_override_to_raw_tables(
        sample_scores_df,
        roc_points_fold_df,
        model_name_override=model_name_override
    )
    if model_name_override is not None and str(model_name_override).strip() != "":
        print(f"  [Model Name Override] model_name 已统一重定义为: {model_name_override}")

    # Sheet 3: roc_points_mean (五折平均 ROC)
    roc_points_mean_rows = []

    for model_type in auc_by_model_task_fold:
        for task_key in auc_by_model_task_fold[model_type]:
            # 获取各 fold 的 (fpr, tpr) 数据
            fpr_tpr_list = []

            # 从 roc_points_fold_df 中提取
            mask = (roc_points_fold_df["model_type"] == model_type) & (roc_points_fold_df["task_key"] == task_key)

            for fold in EXPECTED_FOLDS:
                if fold in auc_by_model_task_fold[model_type][task_key]:
                    fold_mask = mask & (roc_points_fold_df["fold"] == fold)
                    if fold_mask.any():
                        fpr_arr = roc_points_fold_df[fold_mask]["fpr"].values
                        tpr_arr = roc_points_fold_df[fold_mask]["tpr"].values
                        fpr_tpr_list.append((fpr_arr, tpr_arr))

            if len(fpr_tpr_list) == 0:
                continue

            # 计算平均 ROC
            fpr_grid, mean_tpr, std_tpr = compute_mean_roc(fpr_tpr_list, n_points=101)

            # 计算平均 AUC
            fold_aucs = [auc_by_model_task_fold[model_type][task_key].get(f) for f in EXPECTED_FOLDS if f in auc_by_model_task_fold[model_type][task_key]]
            fold_aucs_valid = [a for a in fold_aucs if a is not None and not np.isnan(a)]

            mean_auc = np.mean(fold_aucs_valid) if fold_aucs_valid else np.nan
            std_auc = np.std(fold_aucs_valid, ddof=1) if len(fold_aucs_valid) >= 2 else np.nan

            # 获取模型名称和任务名称
            model_name = resolve_output_model_name(
                model_type,
                sample_scores_df,
                model_name_override=model_name_override
            )
            task_name = TASK_NAME_MAP.get(task_key, task_key)
            positive_class = sample_scores_df[sample_scores_df["task_key"] == task_key]["positive_class"].iloc[0] if len(sample_scores_df) > 0 else None

            # 构建行
            for i in range(len(fpr_grid)):
                roc_points_mean_rows.append({
                    "model_name": model_name,
                    "model_type": model_type,
                    "task_key": task_key,
                    "task_name": task_name,
                    "mean_fpr": float(fpr_grid[i]),
                    "mean_tpr": float(mean_tpr[i]),
                    "std_tpr": float(std_tpr[i]),
                    "mean_auc": float(mean_auc),
                    "std_auc": float(std_auc),
                    "positive_class": positive_class,
                })

    roc_points_mean_df = pd.DataFrame(roc_points_mean_rows)
    print(f"  roc_points_mean: {len(roc_points_mean_df)} 行")

    # Sheet 4: auc_summary
    auc_summary_rows = []

    for model_type in auc_by_model_task_fold:
        model_name = resolve_output_model_name(
            model_type,
            sample_scores_df,
            model_name_override=model_name_override
        )

        for task_key in auc_by_model_task_fold[model_type]:
            task_name = TASK_NAME_MAP.get(task_key, task_key)
            positive_class = sample_scores_df[sample_scores_df["task_key"] == task_key]["positive_class"].iloc[0] if len(sample_scores_df) > 0 else None

            row = {
                "model_name": model_name,
                "model_type": model_type,
                "task_key": task_key,
                "task_name": task_name,
                "positive_class": positive_class,
            }

            fold_aucs = []
            for fold in EXPECTED_FOLDS:
                auc_val = auc_by_model_task_fold[model_type][task_key].get(fold)
                row[f"auc_fold{fold}"] = auc_val
                if auc_val is not None and not np.isnan(auc_val):
                    fold_aucs.append(auc_val)

            row["mean_auc"] = np.mean(fold_aucs) if fold_aucs else np.nan
            row["std_auc"] = np.std(fold_aucs, ddof=1) if len(fold_aucs) >= 2 else np.nan
            row["n_folds"] = len(fold_aucs)

            auc_summary_rows.append(row)

    auc_summary_df = pd.DataFrame(auc_summary_rows)
    print(f"  auc_summary: {len(auc_summary_df)} 行")

    # Sheet 5: run_info
    run_info_rows = []

    # 脚本信息
    run_info_rows.append({"item": "script", "value": "aggregate_fig2c_roc_excel.py"})
    run_info_rows.append({"item": "export_time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    run_info_rows.append({"item": "input_dir", "value": input_dir})
    run_info_rows.append({"item": "output_excel", "value": output_excel})
    run_info_rows.append({"item": "expected_folds", "value": str(EXPECTED_FOLDS)})
    run_info_rows.append({"item": "found_folds", "value": str(list(fold_to_file.keys()))})
    run_info_rows.append({"item": "missing_folds", "value": str(missing_folds)})
    run_info_rows.append({"item": "model_name_override", "value": str(model_name_override)})

    # 各 fold 的 run_info
    for info in all_run_info:
        for key, value in info.items():
            run_info_rows.append({"item": f"fold{info.get('fold', '?')}_{key}", "value": str(value)})

    run_info_df = pd.DataFrame(run_info_rows)
    print(f"  run_info: {len(run_info_df)} 行")

    # 导出到 Excel
    print(f"\n{'='*60}")
    print("[Excel 导出]")
    print(f"{'='*60}")

    output_dir = os.path.dirname(output_excel)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        sample_scores_df.to_excel(writer, sheet_name="sample_scores", index=False)
        roc_points_fold_df.to_excel(writer, sheet_name="roc_points_fold", index=False)
        roc_points_mean_df.to_excel(writer, sheet_name="roc_points_mean", index=False)
        auc_summary_df.to_excel(writer, sheet_name="auc_summary", index=False)
        run_info_df.to_excel(writer, sheet_name="run_info", index=False)

        # 格式化
        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F4E79",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        })

        for sheet_name in ["sample_scores", "roc_points_fold", "roc_points_mean", "auc_summary", "run_info"]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes(1, 0)

    print(f"\n[完成] 输出: {output_excel}")
    print(f"  Sheets: sample_scores, roc_points_fold, roc_points_mean, auc_summary, run_info")
    print(f"  任务: t3, t4, t5")
    print(f"  Folds: {list(fold_to_file.keys())}")

    # 打印 AUC 汇总
    print(f"\n[AUC Summary]")
    for model_type in auc_by_model_task_fold:
        for task_key in auc_by_model_task_fold[model_type]:
            fold_aucs = auc_by_model_task_fold[model_type][task_key]
            mean_auc = np.mean([v for v in fold_aucs.values() if v is not None])
            std_auc = np.std([v for v in fold_aucs.values() if v is not None], ddof=1)
            print(f"  [{model_type}] {task_key}: mean_auc={mean_auc:.4f} +/- {std_auc:.4f}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="聚合五折 ROC 数据生成 Excel")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"xx_path",
        help="ROC JSON 文件目录"
    )
    parser.add_argument(
        "--output_excel",
        type=str,
        default=r"xx_path",
        help="输出 Excel 文件路径"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="MMOE",
        help=(
            "可选：重新定义输出 Excel 中的 model_name。"
            "例如：--model_name \"Shared Bottom\" 或 --model_name \"Single-task model\""
        )
    )

    args = parser.parse_args()

    # 切换到项目目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    aggregate_fold_roc_data(
        input_dir=args.input_dir,
        output_excel=args.output_excel,
        model_name_override=args.model_name
    )


if __name__ == "__main__":
    main()
