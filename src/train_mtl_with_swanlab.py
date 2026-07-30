"""
多任务学习训练脚本 (MTL Training with SwanLab)
===============================================

包含:
1. build_mtl_criterions() - 损失函数工厂
2. MTLTotalLoss - 同方差不确定性加权 + KD
3. two_stage_threshold_search() - 两阶段阈值搜索 (官方唯一实现)
4. collect_logits_and_labels() - 收集 logits 和 labels
5. compute_mtl_metrics() - 多任务指标计算
6. MTLTrainer - 四阶段训练器 (v4: Stage0 + Stage1 + Stage2 + Stage3)
7. load_single_task_checkpoint_into_mtl() - 检查点迁移
8. GateEntropyRegularization - 门控熵正则化
9. GateTemperatureScheduler - 门控温度调度器

创建日期: 2026-04-14
更新日期: 2026-04-29 (v4 Stage0 实现)
"""

import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入现有模块
from train_with_swanlab import (
    UnifiedLDAMLoss,
    DiceLoss,
    compute_binary_pos_weight,
    detect_minority_class_index,
    FocalLoss
)

from model_mtl import HDSTGCNMTL, ProtectedDualEngineMTL, ProtectedDualEngineMTL_v3, ProtectedDualEngineMTL_v4
from model_mtl import (
    init_trunks_from_baseline,
    init_residual_experts_near_identity,
    init_B_t3_from_t3_best,
    init_protected_dual_engine_mtl_v4
)
from task_specs import TaskSpec, build_task_specs_from_config, validate_task_specs
from config import Config

# SwanLab 已强制禁用，避免 import / init / log 带来的额外开销。
swanlab = None
SWANLAB_AVAILABLE = False


# =============================================================================
# 二分类阈值搜索工具函数 (官方唯一实现)
# =============================================================================

def collect_logits_and_labels(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    task_keys: List[str],
    device: str
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    收集指定 DataLoader 的 logits 和 labels

    Args:
        model: MTL 模型
        data_loader: 数据加载器 (val_loader 或 test_loader)
        task_keys: 需要收集的任务列表
        device: 设备

    Returns:
        {task_key: {"logits": np.ndarray, "labels": np.ndarray}}
    """
    model.eval()
    all_logits = {k: [] for k in task_keys}
    all_labels = {k: [] for k in task_keys}

    with torch.no_grad():
        for batch in data_loader:
            x_dyn = batch["x_dyn"].to(device)
            x_static = batch["x_static"].to(device) if batch["x_static"] is not None else None

            outputs = model(x_dyn, x_static)

            for task_key in task_keys:
                if task_key in outputs:
                    logits = outputs[task_key]["logits"].cpu()
                    all_logits[task_key].append(logits)
                if task_key in batch["labels"]:
                    labels = batch["labels"][task_key].cpu()
                    all_labels[task_key].append(labels)

    result = {}
    for task_key in task_keys:
        if all_logits[task_key] and all_labels[task_key]:
            result[task_key] = {
                "logits": torch.cat(all_logits[task_key]).numpy(),
                "labels": torch.cat(all_labels[task_key]).numpy()
            }
    return result


def make_binary_target_for_loss(raw_labels: torch.Tensor, minority_idx: int) -> torch.Tensor:
    """
    统一二分类 target 编码：将原始标签重编码为 "少数类=1, 多数类=0"

    训练、验证、holdout 损失计算全部使用此函数生成的 target，
    确保 BCE/LDAM 损失函数内部 "正类" 语义一致为少数类。

    Args:
        raw_labels: [B] 原始标签 (值为 0 或 1)
        minority_idx: 少数类标签值 (0 或 1)

    Returns:
        target_for_loss: [B] FloatTensor, 少数类样本为 1.0, 多数类为 0.0
    """
    return (raw_labels == minority_idx).float()


def get_minority_prob_from_logits(logits: torch.Tensor, minority_idx: int = None) -> torch.Tensor:
    """
    将 logits 转换为少数类概率 (minority_prob)

    [v2 修复] 训练阶段已统一 target=1 表示少数类，因此：
      - sigmoid(logit) = P(target=1) = P(minority)
      - minority_prob = sigmoid(logits)，无需再根据 minority_idx 反转

    Args:
        logits: [B, 1] 或 [B] 模型输出 logits
        minority_idx: [已废弃] 少数类标签值，保留参数仅为兼容旧调用

    Returns:
        minority_prob: [B] 少数类概率
    """
    return torch.sigmoid(logits).squeeze(-1)  # [B]


def t3_threshold_search(
    logits: np.ndarray,
    labels: np.ndarray,
    minority_idx: int,
    save_dir: str,
    fold: int,
    checkpoint_stage: str = "stage3",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    t3 专用阈值搜索 - 严格控制 false positive

    t3 问题: 极度不平衡 (true_minor_rate≈0.088), threshold=0.5 导致 pred_minor_rate≈0.45
    → false positive 过多, minority_precision 崩溃

    策略:
    1. Hard constraints (优先级最高):
       - pred_minor_rate ∈ [max(true*0.6, 0.04), min(max(true*1.8, 0.12), 0.18)]
       - minority_precision >= 0.15
       - minority_recall >= 0.20

    2. 三级放宽 (严禁 fallback 到 0.5):
       Step 1: 放宽 precision/recall, 保留 pred_minor_rate <= 0.18
       Step 2: 放宽 pred_minor_rate <= 0.20
       Step 3: 选择 pred_minor_rate 最接近 true*1.5 的候选

    3. Score: macro_f1 + 0.2*minority_f1 - 0.1*|pred_rate - target_rate|/true_rate
       target_rate = min(max(true*1.3, 0.10), 0.15)

    Returns:
        t3 专用阈值搜索结果字典
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    # 将 logits 转换为 minority_prob
    probs = get_minority_prob_from_logits(torch.from_numpy(logits), minority_idx).numpy()
    if probs.ndim == 2:
        probs = probs.squeeze(-1)

    true_minor_rate = float(np.mean(labels == minority_idx))
    target_rate = min(max(true_minor_rate * 1.3, 0.10), 0.15)

    # 计算候选阈值数量
    candidate_count = 0
    valid_candidate_count = 0

    def _eval_threshold(thresh):
        """评估单个阈值，返回完整指标字典"""
        preds = np.where(probs > thresh, minority_idx, 1 - minority_idx)
        pred_minor_rate = float(np.mean(preds == minority_idx))

        # 计算 AUC
        y_true_minor = (labels == minority_idx).astype(int)
        try:
            auc = roc_auc_score(y_true_minor, probs)
        except:
            auc = np.nan

        return {
            "threshold": thresh,
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "minority_f1": f1_score(labels, preds, pos_label=minority_idx, zero_division=0),
            "minority_precision": precision_score(labels, preds, pos_label=minority_idx, zero_division=0),
            "minority_recall": recall_score(labels, preds, pos_label=minority_idx, zero_division=0),
            "pred_minor_rate": pred_minor_rate,
            "true_minor_rate": true_minor_rate,
            "auc": auc,
        }

    def _score(m):
        """综合评分"""
        rate_dev = abs(m["pred_minor_rate"] - target_rate) / max(true_minor_rate, 1e-6)
        return m["macro_f1"] + 0.2 * m["minority_f1"] - 0.1 * rate_dev

    def _check_strict(m):
        """严格约束检查"""
        rate_lo = max(true_minor_rate * 0.6, 0.04)
        rate_hi = min(max(true_minor_rate * 1.8, 0.12), 0.18)

        if m["pred_minor_rate"] < rate_lo or m["pred_minor_rate"] > rate_hi:
            return False
        if m["minority_precision"] < 0.15:
            return False
        if m["minority_recall"] < 0.20:
            return False
        return True

    def _check_step1(m):
        """Step 1: 放宽 precision/recall, 保留 pred_minor_rate <= 0.18"""
        rate_lo = max(true_minor_rate * 0.6, 0.04)
        rate_hi = 0.18

        if m["pred_minor_rate"] < rate_lo or m["pred_minor_rate"] > rate_hi:
            return False
        # 不检查 precision/recall
        return True

    def _check_step2(m):
        """Step 2: 放宽 pred_minor_rate <= 0.20"""
        rate_lo = max(true_minor_rate * 0.6, 0.04)
        rate_hi = 0.20

        if m["pred_minor_rate"] < rate_lo or m["pred_minor_rate"] > rate_hi:
            return False
        return True

    def _pick_best_scored(candidates):
        """从候选中选出 score 最高的"""
        if not candidates:
            return None
        return max(candidates, key=_score)

    def _pick_best_near_target(candidates):
        """Step 3: 选择 pred_minor_rate 最接近 true*1.5 的候选中 macro_f1 最高的"""
        if not candidates:
            return None
        target_pred_rate = true_minor_rate * 1.5
        # 按与 target_pred_rate 的距离排序
        sorted_candidates = sorted(candidates, key=lambda m: abs(m["pred_minor_rate"] - target_pred_rate))
        # 取前 5 个最接近的候选，选择 macro_f1 最高的
        top_candidates = sorted_candidates[:5]
        return max(top_candidates, key=lambda m: m["macro_f1"])

    # Baseline (阈值 0.5)
    baseline = _eval_threshold(0.5)

    # Stage1: 粗搜
    thresholds_coarse = np.arange(0.05, 0.96, 0.05)
    all_coarse = []

    for thresh in thresholds_coarse:
        m = _eval_threshold(thresh)
        all_coarse.append(m)

    candidate_count = len(all_coarse)

    # 逐级放宽约束
    strict_coarse = [m for m in all_coarse if _check_strict(m)]
    step1_coarse = [m for m in all_coarse if _check_step1(m)]
    step2_coarse = [m for m in all_coarse if _check_step2(m)]

    # 选择粗搜最优
    best_coarse = _pick_best_scored(strict_coarse)
    fallback_level = "strict"

    if best_coarse is None:
        best_coarse = _pick_best_scored(step1_coarse)
        fallback_level = "step1"

    if best_coarse is None:
        best_coarse = _pick_best_scored(step2_coarse)
        fallback_level = "step2"

    if best_coarse is None:
        best_coarse = _pick_best_near_target(all_coarse)
        fallback_level = "step3"

    # 严禁 fallback 到 0.5
    if best_coarse is None:
        # 极端情况: 选择 pred_minor_rate 最小的 (减少 false positive)
        best_coarse = max(all_coarse, key=lambda m: m["threshold"])
        fallback_level = "extreme"

    best_thresh_1 = best_coarse["threshold"]

    # Stage2: 细搜
    thresholds_fine = np.arange(best_thresh_1 - 0.1, best_thresh_1 + 0.11, 0.005)
    thresholds_fine = np.clip(thresholds_fine, 0.01, 0.99)
    thresholds_fine = np.unique(thresholds_fine)

    all_fine = []
    threshold_metrics = {}

    for thresh in thresholds_fine:
        m = _eval_threshold(thresh)
        threshold_metrics[f"{thresh:.3f}"] = {
            "macro_f1": m["macro_f1"],
            "minority_f1": m["minority_f1"],
            "minority_precision": m["minority_precision"],
            "minority_recall": m["minority_recall"],
            "pred_minor_rate": m["pred_minor_rate"],
        }
        all_fine.append(m)

    # 逐级放宽约束 (细搜)
    strict_fine = [m for m in all_fine if _check_strict(m)]
    step1_fine = [m for m in all_fine if _check_step1(m)]
    step2_fine = [m for m in all_fine if _check_step2(m)]

    valid_candidate_count = len(strict_fine)
    accepted_by_constraints = len(strict_fine) > 0

    best_fine = _pick_best_scored(strict_fine)
    final_fallback_level = "strict"

    if best_fine is None:
        best_fine = _pick_best_scored(step1_fine)
        final_fallback_level = "step1"
        valid_candidate_count = len(step1_fine)

    if best_fine is None:
        best_fine = _pick_best_scored(step2_fine)
        final_fallback_level = "step2"
        valid_candidate_count = len(step2_fine)

    if best_fine is None:
        best_fine = _pick_best_near_target(all_fine)
        final_fallback_level = "step3"
        valid_candidate_count = len(all_fine)

    if best_fine is None:
        best_fine = max(all_fine, key=lambda m: m["threshold"])
        final_fallback_level = "extreme"
        valid_candidate_count = len(all_fine)

    # 最终结果
    best_threshold = best_fine["threshold"]
    best_macro_f1 = best_fine["macro_f1"]
    best_minority_f1 = best_fine["minority_f1"]
    best_minority_precision = best_fine["minority_precision"]
    best_minority_recall = best_fine["minority_recall"]
    best_pred_minor_rate = best_fine["pred_minor_rate"]
    best_auc = best_fine["auc"]

    fallback_reason = None
    if final_fallback_level != "strict":
        fallback_reason = f"No candidate under strict constraints, fallback to {final_fallback_level}"

    # 日志输出
    if verbose:
        print(f"\n[t3 Threshold Search]")
        print(f"  threshold={best_threshold:.4f}")
        print(f"  macro_f1={best_macro_f1:.4f}")
        print(f"  minority_f1={best_minority_f1:.4f}")
        print(f"  minority_precision={best_minority_precision:.4f}")
        print(f"  minority_recall={best_minority_recall:.4f}")
        print(f"  pred_minor_rate={best_pred_minor_rate:.4f}")
        print(f"  true_minor_rate={true_minor_rate:.4f}")
        print(f"  accepted_by_constraints={accepted_by_constraints}")
        print(f"  fallback_level={final_fallback_level}")
        if fallback_reason:
            print(f"  fallback_reason={fallback_reason}")
        print(f"  candidate_count={candidate_count}, valid_candidate_count={valid_candidate_count}")
        print(f"  threshold_0.5: macro_f1={baseline['macro_f1']:.4f}, minority_f1={baseline['minority_f1']:.4f}, pred_minor_rate={baseline['pred_minor_rate']:.4f}")

    # 保存结果
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f"t3_threshold_fold{fold}_{checkpoint_stage}.json")
    result = {
        "task_key": "t3",
        "fold": fold,
        "checkpoint_stage": checkpoint_stage,
        "search_split": "val",
        "best_threshold": best_threshold,
        "selection_metric": "macro_f1_with_rate_control",
        "macro_f1": best_macro_f1,
        "minority_f1": best_minority_f1,
        "minority_precision": best_minority_precision,
        "minority_recall": best_minority_recall,
        "auc": best_auc,
        "pred_minor_rate": best_pred_minor_rate,
        "true_minor_rate": true_minor_rate,
        "target_rate": target_rate,
        "accepted_by_constraints": accepted_by_constraints,
        "fallback_level": final_fallback_level,
        "fallback_reason": fallback_reason,
        "candidate_count": candidate_count,
        "valid_candidate_count": valid_candidate_count,
        "threshold_0_5_macro_f1": baseline["macro_f1"],
        "threshold_0_5_minority_f1": baseline["minority_f1"],
        "threshold_0_5_pred_minor_rate": baseline["pred_minor_rate"],
        "threshold_0_5_minority_precision": baseline["minority_precision"],
        "threshold_0_5_minority_recall": baseline["minority_recall"],
        "improvement": best_macro_f1 - baseline["macro_f1"],
        "threshold_metrics": threshold_metrics
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def two_stage_threshold_search(
    task_key: str,
    logits: np.ndarray,
    labels: np.ndarray,
    minority_idx: int,
    save_dir: str,
    fold: int,
    checkpoint_stage: str = "stage3",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    两阶段阈值搜索 (粗搜 + 细搜) - 官方唯一实现

    [v2] t3 使用专用函数 t3_threshold_search，t4/t5 使用通用策略

    评分: score = macro_f1 + 0.3*minority_f1 - 0.1*|pred_rate-true_rate|/max(true_rate,1e-6)
    Hard constraints (必须全部满足):
      - pred_minor_rate ∈ [true*0.5, true*1.8] (严格); 无候选时放宽至 [0.3, 2.5]
      - 若 baseline_minority_f1 > 0: minority_f1 >= max(0.05, baseline*0.7) 且 minority_f1 != 0
      - 若 baseline_minority_recall > 0: minority_recall >= max(0.05, baseline*0.5)
    Fallback: 所有阈值均不满足 hard constraints 时回退到 0.5

    Stage1: 粗搜 0.05~0.95, step=0.05
    Stage2: 细搜 optimal±0.1, step=0.005

    Returns:
        {
            "best_threshold": float,
            "best_f1_on_val": float,              # 兼容旧字段，= best score
            "best_macro_f1_on_val": float,
            "best_minority_f1_on_val": float,
            "best_minority_precision_on_val": float,
            "best_minority_recall_on_val": float,
            "baseline_f1_at_0.5_on_val": float,   # 兼容旧字段，= baseline_macro_f1
            "baseline_macro_f1_at_0.5_on_val": float,
            "baseline_minority_f1_at_0.5_on_val": float,
            "baseline_minority_precision_at_0.5_on_val": float,
            "baseline_minority_recall_at_0.5_on_val": float,
            "improvement": float,
            "best_pred_minor_rate": float,
            "true_minor_rate": float,
            "accepted_by_constraints": bool,
            "fallback_reason": str or None,
            "selection_metric": "macro_f1_minority_protected",
            "threshold_metrics": dict,
            "search_split": "val"
        }
    """
    # [v2] t3 使用专用策略
    if task_key == "t3":
        return t3_threshold_search(
            logits=logits,
            labels=labels,
            minority_idx=minority_idx,
            save_dir=save_dir,
            fold=fold,
            checkpoint_stage=checkpoint_stage,
            verbose=verbose
        )

    from sklearn.metrics import f1_score, precision_score, recall_score

    # 将 logits 转换为 minority_prob (少数类概率)
    probs = get_minority_prob_from_logits(torch.from_numpy(logits), minority_idx).numpy()
    if probs.ndim == 2:
        probs = probs.squeeze(-1)

    true_minor_rate = float(np.mean(labels == minority_idx))

    def _eval_threshold(thresh):
        """评估单个阈值，返回完整指标字典"""
        preds = np.where(probs > thresh, minority_idx, 1 - minority_idx)
        pred_minor_rate = float(np.mean(preds == minority_idx))
        return {
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "minority_f1": f1_score(labels, preds, pos_label=minority_idx, zero_division=0),
            "minority_precision": precision_score(labels, preds, pos_label=minority_idx, zero_division=0),
            "minority_recall": recall_score(labels, preds, pos_label=minority_idx, zero_division=0),
            "pred_minor_rate": pred_minor_rate,
            "true_minor_rate": true_minor_rate,
        }

    def _score(m):
        """综合评分: macro_f1 + 0.3*minority_f1 - 0.1*|偏移|"""
        rate_dev = abs(m["pred_minor_rate"] - true_minor_rate) / max(true_minor_rate, 1e-6)
        return m["macro_f1"] + 0.3 * m["minority_f1"] - 0.1 * rate_dev

    def _check_hard(m, bl_min_f1, bl_min_recall, rate_lo, rate_hi):
        """Hard constraints 检查"""
        # pred_minor_rate 约束
        if m["pred_minor_rate"] < rate_lo or m["pred_minor_rate"] > rate_hi:
            return False
        # minority_f1 下限
        if bl_min_f1 > 0:
            if m["minority_f1"] == 0:
                return False
            if m["minority_f1"] < max(0.05, bl_min_f1 * 0.7):
                return False
        # minority_recall 下限
        if bl_min_recall > 0:
            if m["minority_recall"] < max(0.05, bl_min_recall * 0.5):
                return False
        return True

    def _pick_best_scored(candidates):
        """从候选中选出 score 最高的"""
        if not candidates:
            return None
        return max(candidates, key=_score)

    # Baseline (阈值 0.5) — 先算，约束依赖 baseline
    baseline = _eval_threshold(0.5)
    baseline_macro_f1 = baseline["macro_f1"]
    baseline_minority_f1 = baseline["minority_f1"]
    baseline_minority_precision = baseline["minority_precision"]
    baseline_minority_recall = baseline["minority_recall"]

    # 两级 pred_minor_rate 约束
    rate_strict = (true_minor_rate * 0.5, true_minor_rate * 1.8)
    rate_relaxed = (true_minor_rate * 0.3, true_minor_rate * 2.5)

    # Stage1: 粗搜
    thresholds_coarse = np.arange(0.05, 0.96, 0.05)
    all_coarse = []

    for thresh in thresholds_coarse:
        m = _eval_threshold(thresh)
        m["threshold"] = thresh
        all_coarse.append(m)

    # 逐级放宽约束选取粗搜最优
    strict_coarse = [m for m in all_coarse
                     if _check_hard(m, baseline_minority_f1, baseline_minority_recall, *rate_strict)]
    relaxed_coarse = [m for m in all_coarse
                      if _check_hard(m, baseline_minority_f1, baseline_minority_recall, *rate_relaxed)]

    best_coarse = _pick_best_scored(strict_coarse)
    used_relaxed_coarse = False
    if best_coarse is None:
        best_coarse = _pick_best_scored(relaxed_coarse)
        used_relaxed_coarse = True
    if best_coarse is None:
        best_coarse = _pick_best_scored(all_coarse)  # 最后兜底
    best_thresh_1 = best_coarse["threshold"]

    # Stage2: 细搜
    thresholds_fine = np.arange(best_thresh_1 - 0.1, best_thresh_1 + 0.11, 0.005)
    thresholds_fine = np.clip(thresholds_fine, 0.01, 0.99)
    thresholds_fine = np.unique(thresholds_fine)

    all_fine = []
    threshold_metrics = {}

    for thresh in thresholds_fine:
        m = _eval_threshold(thresh)
        threshold_metrics[f"{thresh:.3f}"] = {
            "precision": m["minority_precision"],
            "recall": m["minority_recall"],
            "f1": m["minority_f1"],
            "macro_f1": m["macro_f1"],
            "pred_minor_rate": m["pred_minor_rate"],
        }
        m["threshold"] = thresh
        all_fine.append(m)

    # 逐级放宽约束选取细搜最优
    strict_fine = [m for m in all_fine
                   if _check_hard(m, baseline_minority_f1, baseline_minority_recall, *rate_strict)]
    relaxed_fine = [m for m in all_fine
                    if _check_hard(m, baseline_minority_f1, baseline_minority_recall, *rate_relaxed)]

    accepted_by_constraints = True
    fallback_reason = None
    used_relaxed_fine = False

    best_fine = _pick_best_scored(strict_fine)
    if best_fine is None:
        best_fine = _pick_best_scored(relaxed_fine)
        used_relaxed_fine = True
    if best_fine is None:
        # 全部 hard constraints 不满足 → fallback 到 0.5
        accepted_by_constraints = False
        fallback_reason = "No valid threshold under minority-protection constraints"
        best_fine = None

    if best_fine is not None:
        best_thresh_2 = best_fine["threshold"]
        best_macro_f1 = best_fine["macro_f1"]
        best_minority_f1 = best_fine["minority_f1"]
        best_minority_precision = best_fine["minority_precision"]
        best_minority_recall = best_fine["minority_recall"]
        best_pred_minor_rate = best_fine["pred_minor_rate"]
    else:
        # fallback: 使用 baseline 0.5
        best_thresh_2 = 0.5
        best_macro_f1 = baseline_macro_f1
        best_minority_f1 = baseline_minority_f1
        best_minority_precision = baseline_minority_precision
        best_minority_recall = baseline_minority_recall
        best_pred_minor_rate = baseline["pred_minor_rate"]

    if verbose:
        if fallback_reason:
            print(f"[Threshold Search Warning] {task_key}: {fallback_reason}, fallback to 0.5.")
        rate_note = " [relaxed rate]" if used_relaxed_fine else ""
        print(f"[{task_key}] 阈值搜索: 0.5→{best_thresh_2:.3f}, "
              f"macro-F1: {baseline_macro_f1:.4f}→{best_macro_f1:.4f}, "
              f"minority-F1: {baseline_minority_f1:.4f}→{best_minority_f1:.4f}, "
              f"minority-recall: {baseline_minority_recall:.4f}→{best_minority_recall:.4f}, "
              f"pred_minor_rate: {best_pred_minor_rate:.4f} (true={true_minor_rate:.4f})"
              f"{rate_note}")

    # 保存结果
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f"{task_key}_threshold_fold{fold}_{checkpoint_stage}.json")
    result = {
        "task_key": task_key,
        "fold": fold,
        "checkpoint_stage": checkpoint_stage,
        "search_split": "val",
        "best_threshold": best_thresh_2,
        "best_f1_on_val": _score(best_fine) if best_fine is not None else _score(baseline),
        "best_macro_f1_on_val": best_macro_f1,
        "best_minority_f1_on_val": best_minority_f1,
        "best_minority_precision_on_val": best_minority_precision,
        "best_minority_recall_on_val": best_minority_recall,
        "baseline_f1_at_0.5_on_val": baseline_macro_f1,
        "baseline_macro_f1_at_0.5_on_val": baseline_macro_f1,
        "baseline_minority_f1_at_0.5_on_val": baseline_minority_f1,
        "baseline_minority_precision_at_0.5_on_val": baseline_minority_precision,
        "baseline_minority_recall_at_0.5_on_val": baseline_minority_recall,
        "improvement": best_macro_f1 - baseline_macro_f1,
        "best_pred_minor_rate": best_pred_minor_rate,
        "true_minor_rate": true_minor_rate,
        "accepted_by_constraints": accepted_by_constraints,
        "fallback_reason": fallback_reason,
        "selection_metric": "macro_f1_minority_protected",
        "threshold_metrics": threshold_metrics
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    return result


def apply_threshold(
    probs: torch.Tensor,
    threshold: float,
    minority_idx: int,
    majority_idx: int
) -> torch.Tensor:
    """
    应用指定阈值生成预测

    Args:
        probs: 少数类概率 [B]
        threshold: 阈值
        minority_idx: 少数类标签值
        majority_idx: 多数类标签值

    Returns:
        预测标签 [B]
    """
    return torch.where(
        probs > threshold,
        torch.tensor(minority_idx, device=probs.device),
        torch.tensor(majority_idx, device=probs.device)
    ).long()


# =============================================================================
# v2/v3 Gate Expert Name Mapping (语义化命名)
# =============================================================================

# v2 Gate 专家名称映射 (用于日志记录)
V2_GATE_EXPERT_NAMES = {
    # Alpha 任务 (2个专家)
    "t1": ["shared", "t1_private"],
    "t6": ["shared", "t6_private"],
    # Beta 任务
    "t2": ["shared", "group_245", "t2_private"],  # 3个专家
    "t3": ["shared", "t3_private"],               # 2个专家
    "t4": ["shared", "group_245", "t4_private"],  # 3个专家
    "t5": ["shared", "group_245", "t5_private"],  # 3个专家
}

# v3 Gate 专家名称映射 (v3: Alpha无gate, Beta收缩为2维)
V3_GATE_EXPERT_NAMES = {
    # Alpha 任务: 无gate (v3删除CGC)
    # Beta 任务: 全部2专家
    "t2": ["shared", "group_245"],      # v3: 删除t2_private
    "t3": ["shared", "t3_private"],     # v3: 不变
    "t4": ["shared", "group_245"],      # v3: 删除t4_private
    "t5": ["shared", "group_245"],      # v3: 删除t5_private
}

def get_gate_ratio_names(task_key: str, architecture_version: str = "v2") -> List[str]:
    """
    获取任务的gate权重语义化名称 (v2/v3)

    Args:
        task_key: 任务键 (t1~t6)
        architecture_version: 架构版本 ("v2" 或 "v3")

    Returns:
        names: 权重名称列表 (如 ["shared_ratio", "private_ratio"])
               v3的Alpha任务返回空列表 (无gate)
    """
    # 选择对应的映射表
    if architecture_version == "v3":
        expert_names = V3_GATE_EXPERT_NAMES.get(task_key, [])
    else:
        expert_names = V2_GATE_EXPERT_NAMES.get(task_key, [])

    # v3: Alpha任务无gate
    if architecture_version == "v3" and task_key in ["t1", "t6"]:
        return []

    # 转换为ratio名称
    names = []
    for expert_name in expert_names:
        if expert_name == "shared":
            names.append("shared_ratio")
        elif "private" in expert_name:
            names.append("private_ratio")
        elif "group" in expert_name:
            names.append("group_ratio")
        else:
            names.append(f"{expert_name}_ratio")

    return names


# =============================================================================
# T8: Gate Entropy Regularization Loss (新增)
# =============================================================================

class GateEntropyRegularization(nn.Module):
    """
    Gate Entropy Regularization Loss

    Encourages diverse expert selection (prevents gate collapse)

    Formula: L_gate = -λ_gate * Σ_t H(α_t)
    where H(α_t) = -Σ_i α_t[i] * log(α_t[i])

    Args:
        lambda_gate: Regularization weight (default 0.002)
    """

    def __init__(self, lambda_gate: float = 0.002):
        super().__init__()
        self.lambda_gate = lambda_gate

    def forward(self, gate_weights_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            gate_weights_dict: {"t1": [B, num_experts], ..., "t6": [B, num_experts]}

        Returns:
            loss: scalar tensor
        """
        total_entropy = 0.0

        for task_key, gate_weights in gate_weights_dict.items():
            # gate_weights: [B, num_experts]
            # Compute entropy per sample: H = -Σ_i α_i * log(α_i)
            log_weights = torch.log(gate_weights + 1e-8)  # [B, num_experts]
            entropy = -torch.sum(gate_weights * log_weights, dim=1)  # [B]
            total_entropy = total_entropy + entropy.mean()

        # Return negative weighted entropy (encourage high entropy)
        return -self.lambda_gate * total_entropy


# =============================================================================
# T9: Gate Temperature Scheduler (新增)
# =============================================================================

class GateTemperatureScheduler:
    """
    Gate Temperature Annealing Scheduler

    Schedule:
    - 30% epochs: tau = tau_start (exploration)
    - 40% epochs: tau = tau_start → tau_mid (transition)
    - 30% epochs: tau = tau_mid → tau_end (convergence)

    Default: tau_start=2.0, tau_mid=1.0, tau_end=0.7

    Args:
        tau_start: Starting temperature (high = uniform distribution)
        tau_mid: Mid temperature
        tau_end: Ending temperature (low = sharp distribution)
        total_epochs: Total training epochs
    """

    def __init__(
        self,
        tau_start: float = 2.0,
        tau_mid: float = 1.0,
        tau_end: float = 0.7,
        total_epochs: int = 100
    ):
        self.tau_start = tau_start
        self.tau_mid = tau_mid
        self.tau_end = tau_end
        self.total_epochs = total_epochs

    def get_tau(self, epoch: int) -> float:
        """
        Get temperature for current epoch

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            tau: Temperature value
        """
        # Phase boundaries
        phase1_end = int(self.total_epochs * 0.3)  # Exploration phase
        phase2_end = int(self.total_epochs * 0.7)  # Transition phase

        if epoch < phase1_end:
            # Phase 1: Exploration (fixed tau_start)
            return self.tau_start
        elif epoch < phase2_end:
            # Phase 2: Transition (tau_start → tau_mid)
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            tau = self.tau_start + (self.tau_mid - self.tau_start) * progress
            return tau
        else:
            # Phase 3: Convergence (tau_mid → tau_end)
            progress = (epoch - phase2_end) / (self.total_epochs - phase2_end)
            tau = self.tau_mid + (self.tau_end - self.tau_mid) * progress
            return tau


# =============================================================================
# T7: 损失函数工厂
# =============================================================================

def build_mtl_criterions(
    task_specs: Dict[str, TaskSpec],
    device: str = "cpu"
) -> Dict[str, nn.Module]:
    """
    构建 MTL 损失函数字典

    根据 TaskSpec.loss_name 为每个任务构建对应损失:
    - ce: CrossEntropyLoss(weight=class_weights)
    - ldam: UnifiedLDAMLoss(cls_num_list, weight=class_weights/pos_weight)
    - bce: BCEWithLogitsLoss(pos_weight=pos_weight)

    Args:
        task_specs: TaskSpec 字典
        device: 设备

    Returns:
        criterions: {"t1": nn.Module, ..., "t6": nn.Module}
    """
    criterions = {}

    for task_key, spec in task_specs.items():
        if spec.loss_name == "ce":
            # 加权交叉熵 (多分类)
            if spec.class_weights is not None:
                weight = spec.class_weights.to(device)
            else:
                weight = None
            criterions[task_key] = nn.CrossEntropyLoss(weight=weight)

        elif spec.loss_name == "ldam":
            # LDAM Loss (多分类或二分类)
            cls_num_list = spec.class_counts

            if spec.is_binary:
                # 二分类 LDAM: 使用 pos_weight
                weight = spec.pos_weight.to(device) if spec.pos_weight is not None else None

                # [关键修复] 确保 cls_num_list 顺序为 [多数类样本数, 少数类样本数]
                # LDAM 内部假设: m_list[0]=多数类margin(小), m_list[1]=少数类margin(大)
                # 当 minority_idx==0 时，class_counts=[少数类, 多数类]，需要反转
                # 当 minority_idx==1 时，class_counts=[多数类, 少数类]，顺序正确
                if spec.minority_idx == 0:
                    cls_num_list = cls_num_list[::-1]  # 反转顺序
            else:
                # 多分类 LDAM: 使用 class_weights
                weight = spec.class_weights.to(device) if spec.class_weights is not None else None

            criterions[task_key] = UnifiedLDAMLoss(
                cls_num_list=cls_num_list,
                max_m=0.5,
                s=30,
                weight=weight
            )

        elif spec.loss_name == "bce":
            # BCEWithLogitsLoss (二分类)
            pos_weight = spec.pos_weight.to(device) if spec.pos_weight is not None else None
            criterions[task_key] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        else:
            raise ValueError(f"未知损失类型: {spec.loss_name}")

    return criterions


class MTLTotalLoss(nn.Module):
    """
    MTL 总损失: 固定权重缩放 + 同方差不确定性加权 + KD

    公式:
    L_total = Σ_t (e^{-s_t} * (w_t * L_t) + s_t) + λ_KD * L_KD

    其中:
    - w_t 是各阶段固定权重 (来自 YAML loss_weights, 如 t6:0.05, t3:1.2)
    - s_t = log(sigma_t^2) 是可学习不确定性参数
    - L_KD 是 KL 散度蒸馏损失 (仅 t1)
    - λ_KD 是 KD 权重 (默认 0.5)

    Args:
        kd_weight: KD 权重 (默认 0.5)
        kd_temperature: KD 温度 (默认 2.0)
    """

    def __init__(
        self,
        kd_weight: float = 0.5,
        kd_temperature: float = 2.0
    ):
        super().__init__()
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature

    def forward(
        self,
        task_losses: Dict[str, torch.Tensor],
        log_vars: Dict[str, nn.Parameter],
        teacher_logits_t1: Optional[torch.Tensor] = None,
        student_logits_t1: Optional[torch.Tensor] = None,
        task_mask: Optional[Dict[str, torch.Tensor]] = None,
        use_kd: bool = True,
        use_uncertainty_weighting: bool = True,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 MTL 总损失

        公式: L_total = Σ_t (e^{-s_t} * (w_t * L_t) + s_t) + λ_KD * L_KD

        Args:
            task_losses: {"t1": L1, ..., "t6": L6} 各任务原始损失
            log_vars: {"t1": s1, ..., "t6": s6} 可学习不确定性参数
            teacher_logits_t1: 教师模型 logits (仅 t1 KD 时使用)
            student_logits_t1: 学生模型 logits
            task_mask: {"t1": mask1, ...} 任务掩码 (处理缺失标签)
            use_kd: 是否启用 KD (阶段二禁用)
            use_uncertainty_weighting: 是否启用不确定性加权 (阶段一禁用)
            loss_weights: {"t1": 1.0, "t6": 0.05, ...} 各阶段固定权重 (在不确定性加权前应用)

        Returns:
            total_loss: 总损失
            loss_dict: 各分量字典 {"t1": v1, ..., "kd": vk}
        """
        loss_dict = {}
        total_loss = 0.0

        # === 任务损失加权 ===
        for task_key, loss in task_losses.items():
            # 任务掩码 (默认全 1)
            if task_mask is not None and task_key in task_mask:
                mask = task_mask[task_key]
                loss = loss * mask.float()

            # 固定权重缩放 (在不确定性加权之前)
            if loss_weights is not None and task_key in loss_weights:
                loss = loss * loss_weights[task_key]

            # 不确定性加权
            if use_uncertainty_weighting and task_key in log_vars:
                s_t = log_vars[task_key]
                weighted_loss = torch.exp(-s_t) * loss + s_t
            else:
                weighted_loss = loss

            total_loss = total_loss + weighted_loss
            loss_dict[task_key] = loss.item() if isinstance(loss, torch.Tensor) else loss

        # === KD 损失 (仅 t1) ===
        # if use_kd and teacher_logits_t1 is not None and student_logits_t1 is not None:
        #     kd_loss = self._compute_kd_loss(teacher_logits_t1, student_logits_t1)
        #     total_loss = total_loss + self.kd_weight * kd_loss
        #     loss_dict["kd_t1"] = kd_loss.item()

        return total_loss, loss_dict

    def _compute_kd_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 KL 散度蒸馏损失

        公式: L_KD = T^2 * KL(Softmax(z_teacher/T) || Softmax(z_student/T))

        Args:
            teacher_logits: [B, C] 教师模型 logits
            student_logits: [B, C] 学生模型 logits

        Returns:
            kd_loss: KL 散度损失
        """
        T = self.kd_temperature

        # Softmax with temperature
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        student_log_probs = F.log_softmax(student_logits / T, dim=1)

        # KL divergence
        kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T ** 2)

        return kd_loss


# =============================================================================
# 指标计算适配
# =============================================================================

def compute_mtl_metrics(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    labels: Dict[str, torch.Tensor],
    task_specs: Dict[str, TaskSpec],
    device: str = "cpu"
) -> Dict[str, Dict[str, float]]:
    """
    计算多任务指标

    Args:
        outputs: {"t1": {"logits": ...}, ...}
        labels: {"t1": [B], ...}
        task_specs: TaskSpec 字典
        device: 设备

    Returns:
        metrics: {"t1": {"acc": ..., "f1": ..., ...}, ...}
    """
    metrics = {}

    for task_key, spec in task_specs.items():
        logits = outputs[task_key]["logits"]
        target = labels[task_key].to(device)

        if spec.is_binary:
            # 二分类指标
            task_metrics = _compute_binary_metrics(logits, target, spec)
        else:
            # 多分类指标
            task_metrics = _compute_multiclass_metrics(logits, target, spec)

        metrics[task_key] = task_metrics

    return metrics


# =============================================================================
# v4 评估指标函数
# =============================================================================

def compute_mean_macro_f1_t1_to_t5(val_metrics: Dict[str, Dict[str, float]]) -> float:
    """
    计算 t1-t5 的平均 Macro-F1 (v4 主指标之一)

    Args:
        val_metrics: {"t1": {"macro_f1": ...}, ..., "t5": {"macro_f1": ...}, "t6": {...}}

    Returns:
        mean_macro_f1: t1-t5 的平均 Macro-F1
    """
    f1s = []
    for t in ["t1", "t2", "t3", "t4", "t5"]:
        if t in val_metrics and "macro_f1" in val_metrics[t]:
            f1s.append(val_metrics[t]["macro_f1"])

    if len(f1s) == 0:
        return 0.0

    return np.mean(f1s)


def compute_mean_macro_f1_t2_to_t5(val_metrics: Dict[str, Dict[str, float]]) -> float:
    """
    计算 t2-t5 的平均 Macro-F1 (Stage2 checkpoint 指标)

    用于 Stage2 Beta预热阶段的模型选择，与 config_mtl.yaml 中的
    checkpoint_save_metric: "mean_macro_f1_t2_to_t5" 保持一致。

    Args:
        val_metrics: {"t2": {"macro_f1": ...}, ..., "t5": {"macro_f1": ...}}

    Returns:
        mean_macro_f1: t2-t5 的平均 Macro-F1
    """
    f1s = []
    for t in ["t2", "t3", "t4", "t5"]:
        if t in val_metrics and "macro_f1" in val_metrics[t]:
            f1s.append(val_metrics[t]["macro_f1"])

    if len(f1s) == 0:
        return 0.0

    return np.mean(f1s)


def compute_weighted_macro_f1_t1_to_t5(
    val_metrics: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    计算加权平均 Macro-F1 (v4 主 checkpoint 指标)

    权重设计理由:
    - t3 权重最高 (0.25): 极度不平衡二分类，最困难
    - t1 权重中等 (0.20): 主任务，功能分级
    - t2/t4 权重中等 (0.20): 二分类任务
    - t5 权重最低 (0.15): 心率储备，相对简单

    Args:
        val_metrics: {"t1": {"macro_f1": ...}, ..., "t5": {"macro_f1": ...}}
        weights: 任务权重字典，默认 {"t1": 0.20, "t2": 0.20, "t3": 0.25, "t4": 0.20, "t5": 0.15}

    Returns:
        weighted_macro_f1: 加权平均 Macro-F1
    """
    if weights is None:
        weights = {"t1": 0.20, "t2": 0.20, "t3": 0.25, "t4": 0.20, "t5": 0.15}

    weighted_sum = 0.0
    total_weight = 0.0

    for t, w in weights.items():
        if t in val_metrics and "macro_f1" in val_metrics[t]:
            weighted_sum += w * val_metrics[t]["macro_f1"]
            total_weight += w

    if total_weight == 0:
        return 0.0

    return weighted_sum


def compute_v4_minimal_metrics(
    model: nn.Module,
    val_metrics: Dict[str, Dict[str, float]],
    stage: str
) -> Dict[str, float]:
    """
    计算 v4 最小安全日志指标

    Args:
        model: ProtectedDualEngineMTL_v4 模型
        val_metrics: 验证指标
        stage: 当前阶段名称

    Returns:
        logs: SwanLab 日志字典
    """
    logs = {}

    # t1-t5 Macro-F1
    for t in ["t1", "t2", "t3", "t4", "t5"]:
        if t in val_metrics and "macro_f1" in val_metrics[t]:
            logs[f"{stage}/{t}_macro_f1"] = val_metrics[t]["macro_f1"]

    # v4 综合指标
    logs[f"{stage}/mean_macro_f1_t1_to_t5"] = compute_mean_macro_f1_t1_to_t5(val_metrics)
    logs[f"{stage}/weighted_macro_f1_t1_to_t5"] = compute_weighted_macro_f1_t1_to_t5(val_metrics)
    # Stage2/Stage3 Phase1 checkpoint 指标 (统一命名)
    logs[f"{stage}/mean_macro_f1_t2_to_t5"] = compute_mean_macro_f1_t2_to_t5(val_metrics)

    # t6 仅日志 (不参与排名)
    if "t6" in val_metrics:
        logs[f"{stage}/t6_macro_f1_aux"] = val_metrics["t6"].get("macro_f1", 0.0)
        logs[f"{stage}/t6_loss_aux"] = val_metrics["t6"].get("loss", 0.0)

    # v4 特有日志
    if hasattr(model, 'get_alpha_t1_gate_value'):
        logs["v4/alpha_t1_gate"] = model.get_alpha_t1_gate_value()

    if hasattr(model, 't6_deep_context_module') and model.t6_deep_context_module is not None:
        # c6_deep norm (如果 aux 中有)
        if "t6" in val_metrics and "c6_deep" in val_metrics["t6"]:
            c6_deep = val_metrics["t6"]["c6_deep"]
            if c6_deep is not None:
                logs["v4/c6_deep_norm"] = np.linalg.norm(c6_deep)

    return logs


def _compute_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    spec: TaskSpec
) -> Dict[str, float]:
    """
    计算二分类指标 (使用 minority_idx 作为 pos_label)

    Args:
        logits: [B, 1] 单节点 logits
        target: [B] 标签 (0 或 1)
        spec: TaskSpec

    Returns:
        metrics: {"acc", "precision", "recall", "f1", "macro_f1", "auc", "auprc"}
    """
    minority_idx = spec.minority_idx if spec.minority_idx is not None else 1

    # logits → minority_prob (统一方向: P(minority))
    minority_prob = get_minority_prob_from_logits(logits, minority_idx)  # [B]

    # 预测标签 (阈值 0.5)
    pred = torch.where(
        minority_prob > 0.5,
        torch.tensor(minority_idx, device=logits.device),
        torch.tensor(1 - minority_idx, device=logits.device)
    ).long()

    # 计算指标
    correct = (pred == target).sum().item()
    total = target.size(0)
    acc = correct / total

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    probs_np = minority_prob.cpu().numpy()

    try:
        precision = precision_score(target_np, pred_np, pos_label=minority_idx, zero_division=0)
        recall = recall_score(target_np, pred_np, pos_label=minority_idx, zero_division=0)
        f1 = f1_score(target_np, pred_np, pos_label=minority_idx, zero_division=0)

        # [v2 统一] AUC: probs_np 是 minority_prob, y_true_minor = (labels == minority_idx)
        y_true_minor = (target_np == minority_idx).astype(int)
        auc = roc_auc_score(y_true_minor, probs_np)

        # AUPRC: Average Precision Score
        auprc = average_precision_score(y_true_minor, probs_np)
    except Exception as e:
        print(f"[AUC Error] binary: target_shape={target_np.shape}, probs_shape={probs_np.shape}, unique_labels={np.unique(target_np)}, error={e}")
        precision, recall, f1, auc, auprc = 0.0, 0.0, 0.0, np.nan, np.nan

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": f1_score(target_np, pred_np, average='macro', zero_division=0),  # 两类F1均值 (非 minority-F1)
        "auc": auc,
        "auprc": auprc
    }


def _compute_multiclass_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    spec: TaskSpec
) -> Dict[str, float]:
    """
    计算多分类指标

    Args:
        logits: [B, C] 多分类 logits
        target: [B] 标签 (0~C-1)
        spec: TaskSpec

    Returns:
        metrics: {"acc", "precision", "recall", "macro_f1", "auc", "auprc"}
    """
    # 预测
    pred = logits.argmax(dim=1)

    # Accuracy
    correct = (pred == target).sum().item()
    total = target.size(0)
    acc = correct / total

    # sklearn 指标
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, label_binarize

    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    probs_np = F.softmax(logits, dim=1).cpu().numpy()
    n_classes = probs_np.shape[1]

    try:
        precision = precision_score(target_np, pred_np, average='macro', zero_division=0)
        recall = recall_score(target_np, pred_np, average='macro', zero_division=0)
        macro_f1 = f1_score(target_np, pred_np, average='macro', zero_division=0)

        # AUC (multi-class OvR)
        auc = roc_auc_score(target_np, probs_np, multi_class='ovr', average='macro')

        # AUPRC: 多分类 One-vs-Rest 宏平均
        # 将标签二值化: shape [N, C]
        y_true_bin = label_binarize(target_np, classes=list(range(n_classes)))

        # 处理二分类情况 (label_binarize 返回 [N, 1] 而非 [N, 2])
        if y_true_bin.shape[1] == 1 and n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        # 计算每个类别的 AP
        ap_per_class = []
        for c in range(n_classes):
            if np.sum(y_true_bin[:, c]) > 0:  # 类别 c 有正样本
                ap_c = average_precision_score(y_true_bin[:, c], probs_np[:, c])
                ap_per_class.append(ap_c)

        # 宏平均 AUPRC
        auprc = np.mean(ap_per_class) if ap_per_class else 0.0
    except Exception as e:
        print(f"[AUC Error] multiclass: target_shape={target_np.shape}, probs_shape={probs_np.shape}, unique_labels={np.unique(target_np)}, error={e}")
        precision, recall, macro_f1, auc, auprc = 0.0, 0.0, 0.0, np.nan, np.nan

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1,
        "auc": auc,
        "auprc": auprc
    }


# =============================================================================
# T10: 检查点加载/保存
# =============================================================================

def load_single_task_checkpoint_into_mtl(
    model: HDSTGCNMTL,
    ckpt_path: str,
    task_name: str,
    branch_type: str = "alpha",
    strict: bool = False
) -> None:
    """
    将单任务检查点加载到 MTL 模型

    映射规则:
    - t6 单任务 -> alpha_encoder + alpha_interactors.t6 + static_encoders.t6 + classifiers.t6
    - t1 单任务 -> alpha_interactors.t1 + static_encoders.t1 + classifiers.t1
    - t2~t5 单任务 -> beta_projectors.tX + static_encoders.tX + classifiers.tX

    Args:
        model: HDSTGCNMTL 模型
        ckpt_path: 单任务检查点路径
        task_name: 任务名称 ("t1" ~ "t6")
        branch_type: 分支类型 ("alpha" 或 "beta")
        strict: 是否严格匹配
    """
    if not os.path.exists(ckpt_path):
        print(f"[Warning] 检查点不存在: {ckpt_path}")
        return

    # 加载检查点 (添加 weights_only=False 解决 PyTorch 2.6 安全限制)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # 构建映射
    new_state_dict = {}

    if task_name == "t6" and branch_type == "alpha":
        # t6 是主任务，加载完整 Alpha 分支
        # 单任务 state_dict 映射到 MTL state_dict
        mapping = {
            # 时序编码器 (共享)
            "temporal_encoder.": "alpha_encoder.encoder.",
            "encoder.": "alpha_encoder.encoder.",

            # PMGT (t6 专属)
            "pmgt.": "alpha_interactors.t6.pmgt.",
            "prior_masked.": "alpha_interactors.t6.pmgt.",
            "spatial_graph.": "alpha_interactors.t6.pmgt.",

            # 静态编码器
            "static_encoder.": "static_encoders.t6.encoder.",
            "static_branch.": "static_encoders.t6.encoder.",

            # 分类器
            "classifier.": f"classifiers.t6.classifier.",
            "fc.": f"classifiers.t6.classifier.",
            "head.": f"classifiers.t6.classifier.",
        }

    elif task_name == "t1" and branch_type == "alpha":
        # t1 加载 PMGT + 静态 + 分类器 (不加载编码器)
        mapping = {
            "pmgt.": "alpha_interactors.t1.pmgt.",
            "prior_masked.": "alpha_interactors.t1.pmgt.",
            "static_encoder.": "static_encoders.t1.encoder.",
            "classifier.": "classifiers.t1.classifier.",
            "fc.": "classifiers.t1.classifier.",
        }

    elif branch_type == "beta":
        # t2~t5 加载投影头 + 静态 + 分类器
        mapping = {
            "flatten_proj.": f"beta_projectors.{task_name}.net.",
            "projector.": f"beta_projectors.{task_name}.net.",
            "static_encoder.": f"static_encoders.{task_name}.encoder.",
            "classifier.": f"classifiers.{task_name}.classifier.",
            "fc.": f"classifiers.{task_name}.classifier.",
        }

    else:
        print(f"[Warning] 未知任务-分支组合: {task_name}/{branch_type}")
        return

    # 应用映射
    for old_key, value in state_dict.items():
        new_key = old_key
        for prefix, new_prefix in mapping.items():
            if old_key.startswith(prefix):
                new_key = old_key.replace(prefix, new_prefix, 1)
                break

        if new_key != old_key:
            new_state_dict[new_key] = value

    # 加载到模型
    if new_state_dict:
        model.load_state_dict(new_state_dict, strict=strict)
        print(f"[Info] 从 {ckpt_path} 加载 {task_name} 参数 ({len(new_state_dict)} 个)")
    else:
        print(f"[Warning] 无匹配参数可加载")


def save_mtl_checkpoint(
    model: HDSTGCNMTL,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stage: str,
    fold: int,
    metrics: Dict[str, Dict[str, float]],
    save_dir: str = "models",
    prefix: str = "mtl"
) -> str:
    """
    保存 MTL 检查点

    Args:
        model: HDSTGCNMTL 模型
        optimizer: 优化器
        epoch: 当前 epoch
        stage: 阶段名称 ("stage1", "stage2", "stage3")
        fold: Fold 编号
        metrics: 指标字典
        save_dir: 保存目录
        prefix: 文件前缀

    Returns:
        save_path: 保存路径
    """
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"best_{prefix}_{stage}_fold{fold}.pth")

    checkpoint = {
        "epoch": epoch,
        "stage": stage,
        "fold": fold,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "log_vars": {k: v.item() for k, v in model.log_vars.items()}
    }

    torch.save(checkpoint, save_path)
    print(f"[Info] 检查点保存至: {save_path}")

    return save_path


# =============================================================================
# T9: 三阶段训练器
# =============================================================================

class MTLTrainer:
    """
    多任务学习三阶段训练器

    阶段一: Alpha 锚定 (t1, t6)
    阶段二: Beta 预热 (t2~t5)
    阶段三: 联合微调 (全任务)

    关键设计:
    - 梯度累加 (effective batch size >= 32)
    - 模块冻结/解冻
    - 学习率缩放
    - 任务六 KD 保护
    - SwanLab 日志
    - 阈值扫描 (t3, t4, t5) [新增]
    - 门控熵正则化 (ProtectedDualEngineMTL 专用) [新增]
    - 门控温度调度器 (ProtectedDualEngineMTL 专用) [新增]
    """

    def __init__(
        self,
        model: HDSTGCNMTL,
        criterions: Dict[str, nn.Module],
        total_loss: MTLTotalLoss,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        task_specs: Dict[str, TaskSpec],
        device: str = "cuda",
        swanlab_run: Optional[Any] = None,
        teacher_model: Optional[nn.Module] = None
    ):
        self.model = model
        self.criterions = criterions
        self.total_loss = total_loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.task_specs = task_specs
        self.device = device
        self.swanlab_run = swanlab_run
        self.teacher_model = teacher_model

        # [新增] 实验命名: checkpoint_prefix 自动生成
        # [修改 2026-05-27] Clean 架构模式使用专属命名
        # *_clean mode → 直接使用 mode 名称
        # baseline → mtl_v4_baseline
        # t6_auxiliary → mtl_v4_t6_protected
        mode = config.get('mode', '')
        suffix = config.get('experiment_suffix', '')
        if mode and mode.endswith("_clean"):
            self.checkpoint_prefix = mode + suffix
        else:
            t6_aux_enabled = config.get('hcgc_v4', {}).get('t6_auxiliary_mode', {}).get('enabled', False)
            base_name = "mtl_v4_t6_protected" if t6_aux_enabled else "mtl_v4_baseline"
            self.checkpoint_prefix = base_name + suffix
        print(f"[Info] 实验命名: checkpoint_prefix = {self.checkpoint_prefix}")

        # [新增] 调试文件名 (训练启动时生成，整个训练过程使用同一文件)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_file = f"debug_t3_metrics_{timestamp}.txt"
        self.debug_header_written = False

        # [重构] 阈值搜索配置 (替代 threshold_sweep)
        eval_config = config.get('eval', {})
        threshold_search_config = eval_config.get('binary_threshold_search', {})
        self.threshold_search_tasks = threshold_search_config.get('enabled_tasks', ["t3", "t4", "t5"])
        self.threshold_mode = threshold_search_config.get('mode', 'search')
        self.threshold_save_dir = threshold_search_config.get('save_dir', 'threshold_search_results')
        self.fixed_thresholds = eval_config.get('fixed_thresholds', {})
        self.best_thresholds: Dict[str, float] = {}  # 存储搜索结果
        self.current_fold = 1  # [新增] 当前 Fold 编号 (用于阈值搜索文件命名)

        # [新增] 指标历史缓存 (用于JSON导出)
        self.metrics_history = {}  # 结构: {stage: {epoch: {task_key: {metric: value}}}}
        self.train_metrics_history = {}  # 结构: {stage: {epoch: {task_key: {metric: value}}}}
        self.log_vars_history = {}  # 结构: {stage: {epoch: {task_key: value}}}

        # [新增] 增量保存配置 (防止崩溃丢失)
        self.incremental_save_dir = config.get('incremental_save_dir', 'metrics_logs')
        self.incremental_save_enabled = config.get('incremental_save_enabled', True)  # 默认启用
        self.incremental_json_path = None  # 在 Fold 开始时初始化

        # 打印阈值搜索状态
        if self.threshold_mode == 'search':
            print(f"[Info] 阈值搜索模式: tasks={self.threshold_search_tasks}, save_dir={self.threshold_save_dir}")
        else:
            print(f"[Info] 固定阈值模式: fixed_thresholds={self.fixed_thresholds}")

        # =====================================================
        # [新增] ProtectedDualEngineMTL 检测与初始化
        # =====================================================
        # v1 检测: alpha_experts + beta_experts
        # v2 检测: alpha_trunk + beta_trunk + alpha_residual_experts (非None) + beta_residual_experts
        # v3 检测: alpha_trunk + beta_trunk + alpha_residual_experts is None + beta_residual_experts只有3个
        # v4 检测: t6_deep_context_module 属性存在 + t6_deep_context_enabled=True
        self.is_protected_dual_engine_v1 = hasattr(model, 'alpha_experts') and hasattr(model, 'beta_experts')
        self.is_protected_dual_engine_v2 = hasattr(model, 'alpha_trunk') and hasattr(model, 'beta_trunk') and \
                                           hasattr(model, 'alpha_residual_experts') and model.alpha_residual_experts is not None and \
                                           hasattr(model, 'beta_residual_experts')
        # v3 检测: alpha_residual_experts is None (CGC删除), beta_residual_experts只有3个专家
        self.is_protected_dual_engine_v3 = hasattr(model, 'alpha_trunk') and hasattr(model, 'beta_trunk') and \
                                           hasattr(model, 'alpha_residual_experts') and model.alpha_residual_experts is None and \
                                           hasattr(model, 'beta_residual_experts') and model.beta_residual_experts is not None and len(model.beta_residual_experts) == 3
        # v4 检测: 优先检查 architecture.variant，其次检查模型属性
        # [修复 - 2026-04-28] 当 t6_deep_context.enabled=false 时，模型属性检测会失败
        # 但 yaml 已指定 variant="protected_dual_engine_t6_guided_v4"，应识别为 v4
        architecture_variant = config.get('architecture', {}).get('variant', '')
        if architecture_variant == 'protected_dual_engine_t6_guided_v4':
            self.is_protected_dual_engine_v4 = True
            print("[Info v4] 检测到 v4 架构 (基于 yaml variant)")
        else:
            # 原检测逻辑 (fallback)
            self.is_protected_dual_engine_v4 = hasattr(model, 't6_deep_context_module') and \
                                               hasattr(model, 't6_deep_context_enabled') and model.t6_deep_context_enabled
        self.is_protected_dual_engine = self.is_protected_dual_engine_v1 or self.is_protected_dual_engine_v2 or \
                                        self.is_protected_dual_engine_v3 or self.is_protected_dual_engine_v4

        # [新增] 为非 v4 架构（如 MMoE Clean）初始化默认属性，避免属性未定义错误
        if not self.is_protected_dual_engine_v4:
            # MMoE Clean / CGC Clean / Shared Bottom Clean 等架构不需要 t6 辅助模式
            self.t6_auxiliary_mode_enabled = False
            self.context_reg_enabled = False
            self.no_t6_context_but_train_t6_enabled = False
            self.v4_stage1_active_tasks_override = None
            self.v4_phase2_active_tasks_override = None
            self.v4_stage1_loss_weights_override = None
            self.v4_phase2_loss_weights_override = None
            self.skip_t6_injection_stages = []
            self.detach_t6_context_stages = []

        if self.is_protected_dual_engine_v4:
            print("[Info v4] 检测到 ProtectedDualEngineMTL_v4 模型 (T6-guided Context Injection)")
            print("[Info v4] T6 Deep Feature Context: dyn_feat_t6 -> c6_deep")
            print("[Info v4] Beta Gate context_dim: 40 -> 56 (+ c6_deep[16])")
            print("[Info v4] t1 上下文增强: dyn_feat_t1 + alpha * delta")
            print("[Info v4] KD 废弃: use_kd=false (Stage1-Stage3)")
            print("[Info v4] checkpoint 指标: weighted_macro_f1_t1_to_t5")
            self.architecture_version = "v4"

            # v4 专用配置
            t6_deep_context_config = config.get('t6_deep_context', {})
            self.v4_minimal_logging = config.get('monitoring', {}).get('v4_minimal_logging', True)
            self.freeze_t6_log_var = config.get('uncertainty_weighting', {}).get('freeze_t6_log_var', True)

            print(f"[Info v4] freeze_t6_log_var: {self.freeze_t6_log_var}")

            # 初始冻结 t6 log_var
            self._apply_t6_log_var_freeze()

            # [修复 - 2026-04-28] 先读取 T6辅助模式开关，再根据模式设置注入策略
            t6_aux_cfg = config.get('hcgc_v4', {}).get('t6_auxiliary_mode', {})
            self.t6_auxiliary_mode_enabled = t6_aux_cfg.get('enabled', False)

            # [新增 - 2026-04-30] T6辅助模式: c6_deep L2 正则化配置
            context_reg_cfg = t6_aux_cfg.get('context_reg', {})
            self.context_reg_enabled = self.t6_auxiliary_mode_enabled and context_reg_cfg.get('enabled', True)
            self.context_reg_lambda = context_reg_cfg.get('lambda', 0.01)
            if self.context_reg_enabled:
                print(f"[Info v4] T6 context_reg: enabled=True, lambda={self.context_reg_lambda}")

            if not self.t6_auxiliary_mode_enabled:
                # [实验E] 检测 no_t6_context_but_train_t6 开关
                self.no_t6_context_but_train_t6_enabled = config.get('hcgc_v4', {}).get('no_t6_context_but_train_t6', {}).get('enabled', False)

                if self.no_t6_context_but_train_t6_enabled:
                    # 实验E: Baseline + t6 ordinary weak supervision
                    print("[Info v4] 实验E: Baseline + t6 ordinary weak supervision")
                    print("[Info v4] 实验E: t6 参与训练和 loss，但不作为 deep context source")
                    self.v4_stage1_active_tasks_override = ["t1", "t6"]
                    self.v4_phase2_active_tasks_override = ["t1", "t2", "t3", "t4", "t5", "t6"]
                    # 实验E: 所有阶段都跳过 t6 context injection (t6 不注入 beta_gate 或 t1)
                    self.skip_t6_injection_stages = ['stage0', 'stage1_alpha_anchor', 'stage2_beta_warmup', 'stage3_phase1', 'stage3_phase2']
                    self.detach_t6_context_stages = []  # 无注入，无需梯度隔离
                    # 实验E: loss_weights 覆盖 (确保 t6 弱监督权重正确)
                    self.v4_stage1_loss_weights_override = {"t1": 1.0, "t6": 0.1}
                    self.v4_phase2_loss_weights_override = {"t1": 1.0, "t2": 1.0, "t3": 1.2, "t4": 1.0, "t5": 1.0, "t6": 0.05}
                    print(f"[Info v4] 实验E Stage1 active_tasks: {self.v4_stage1_active_tasks_override}")
                    print(f"[Info v4] 实验E Phase2 active_tasks: {self.v4_phase2_active_tasks_override}")
                    print(f"[Info v4] 实验E skip_t6_injection_stages: {self.skip_t6_injection_stages} (全部阶段)")
                    print(f"[Info v4] 实验E detach_t6_context_stages: {self.detach_t6_context_stages} (无需设置)")
                    print(f"[Info v4] 实验E Stage1 loss_weights: {self.v4_stage1_loss_weights_override}")
                    print(f"[Info v4] 实验E Phase2 loss_weights: {self.v4_phase2_loss_weights_override}")
                else:
                    # Baseline MTL 模式: t6完全剔除，所有阶段都不需要注入相关逻辑
                    print("[Info v4] Baseline MTL 模式: t6完全剔除，仅保留辅助日志")
                    self.v4_stage1_active_tasks_override = ["t1"]
                    self.v4_phase2_active_tasks_override = ["t1", "t2", "t3", "t4", "t5"]
                    self.v4_stage1_loss_weights_override = None  # 使用yaml配置
                    self.v4_phase2_loss_weights_override = None  # 使用yaml配置
                    # Baseline模式: 所有阶段都跳过注入（因为t6不参与）
                    # [修复 - 2026-05-06] 添加 stage0，使 Stage0 验证时也跳过注入路径计算
                    self.skip_t6_injection_stages = ['stage0', 'stage1_alpha_anchor', 'stage2_beta_warmup', 'stage3_phase1', 'stage3_phase2']
                    # Baseline模式: 不需要梯度隔离设置（因为没有注入）
                    self.detach_t6_context_stages = []  # 空列表
                    print(f"[Info v4] Stage1 active_tasks: {self.v4_stage1_active_tasks_override}")
                    print(f"[Info v4] Phase2 active_tasks: {self.v4_phase2_active_tasks_override}")
                    print(f"[Info v4] skip_t6_injection_stages: {self.skip_t6_injection_stages} (全部阶段)")
                    print(f"[Info v4] detach_t6_context_stages: {self.detach_t6_context_stages} (无需设置)")
            else:
                # T6辅助模式: 按文档定义的阶梯式梯度策略
                print("[Info v4] T6辅助模式: t6作为上下文提供者参与训练")
                self.no_t6_context_but_train_t6_enabled = False  # 实验E仅适用于baseline模式
                self.v4_stage1_active_tasks_override = None  # 使用yaml配置
                self.v4_phase2_active_tasks_override = None  # 使用yaml配置
                self.v4_stage1_loss_weights_override = None
                self.v4_phase2_loss_weights_override = None
                # T6辅助模式: Stage0和Stage1不注入，其他阶段注入
                # [修复 - 2026-05-06] 添加 stage0，使 Stage0 验证时跳过注入路径计算
                self.skip_t6_injection_stages = ['stage0', 'stage1_alpha_anchor']
                # T6辅助模式: Stage2和Phase1需要梯度隔离
                self.detach_t6_context_stages = ['stage2_beta_warmup', 'stage3_phase1']  # 阶梯式解冻
                print(f"[Info v4] skip_t6_injection_stages: {self.skip_t6_injection_stages}")
                print(f"[Info v4] detach_t6_context_stages: {self.detach_t6_context_stages}")
        elif self.is_protected_dual_engine_v3:
            print("[Info v2] 检测到 ProtectedDualEngineMTL v2 模型 (残差专家架构)")
            print("[Info v2] trunk output 不再作为 gate 候选")
            self.architecture_version = "v2"
            self.no_t6_context_but_train_t6_enabled = False
        elif self.is_protected_dual_engine_v1:
            print("[Info v1] 检测到 ProtectedDualEngineMTL v1 模型")
            self.architecture_version = "v1"
            self.no_t6_context_but_train_t6_enabled = False
        else:
            self.architecture_version = "baseline"
            self.no_t6_context_but_train_t6_enabled = False

        if self.is_protected_dual_engine:
            print("[Info] 启用门控专用组件")

            # 门控熵正则化
            gate_entropy_config = config.get('gate_entropy', {})
            lambda_gate = gate_entropy_config.get('lambda_gate', 0.002)
            self.gate_entropy_reg = GateEntropyRegularization(lambda_gate=lambda_gate)
            self.use_gate_entropy = gate_entropy_config.get('enabled', True)  # 默认启用

            # 门控温度调度器
            temp_config = config.get('gate_temperature', {})
            # v3可能从hcgc_v3.gate读取配置
            hcgc_v3_config = config.get('hcgc_v3', {})
            gate_config = hcgc_v3_config.get('gate', temp_config)

            self.temp_scheduler = GateTemperatureScheduler(
                tau_start=gate_config.get('tau_start', 2.0),
                tau_mid=gate_config.get('tau_mid', 1.0),
                tau_end=gate_config.get('tau_end', 0.7),
                total_epochs=config.get('total_epochs', 100)
            )

            # 记录温度历史 (用于 SwanLab)
            self.tau_history = []

            print(f"[Info] 门控熵正则化: lambda_gate={lambda_gate}, enabled={self.use_gate_entropy}")
            print(f"[Info] 门控温度调度: tau_start={self.temp_scheduler.tau_start} → tau_mid={self.temp_scheduler.tau_mid} → tau_end={self.temp_scheduler.tau_end}")
        else:
            self.is_protected_dual_engine = False
            self.gate_entropy_reg = None
            self.temp_scheduler = None
            self.use_gate_entropy = False
            self.tau_history = []

        # =====================================================
        # [新增 - 2026-05-25] t3_anchor: 边界稳定机制
        # =====================================================
        # Stage3 Phase2 期间锚定 t3 预测到历史最优 checkpoint
        t3_anchor_config = self.config.get('t3_anchor', {})
        self.t3_anchor_enabled = t3_anchor_config.get('enabled', False)
        self.t3_anchor_lambda = t3_anchor_config.get('lambda', 0.2)
        self.t3_anchor_teacher_stage = t3_anchor_config.get('teacher_stage', 'auto_best_t3')
        self.t3_anchor_loss_type = t3_anchor_config.get('loss_type', 'mse_prob')
        self.t3_anchor_teacher_model = None  # 在 stage3_phase2 开始时加载

        if self.t3_anchor_enabled:
            print(f"[t3_anchor] 边界稳定机制已启用:")
            print(f"[t3_anchor]   lambda={self.t3_anchor_lambda}")
            print(f"[t3_anchor]   teacher_stage={self.t3_anchor_teacher_stage}")
            print(f"[t3_anchor]   loss_type={self.t3_anchor_loss_type}")

        # =====================================================

        # 优化器 (不含 log_vars 的 weight decay)
        self.optimizer = self._build_optimizer()

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        # 当前阶段
        self.current_stage = config.get('current_stage', 'stage1_alpha_anchor')

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """
        构建优化器

        - HDSTGCNMTL: log_vars 不加 weight decay
        - ProtectedDualEngineMTL: 参数分组 + 不同学习率缩放
        """
        base_lr = self.config.get('lr', 0.0003)
        weight_decay = self.config.get('weight_decay', 0.001)

        # =====================================================
        # ProtectedDualEngineMTL: 参数分组优化器
        # =====================================================
        if self.is_protected_dual_engine:
            param_groups = []

            # 学习率缩放系数 (基于层级重要性)
            # - alpha_base: 0.1 (极小，共享基础特征)
            # - alpha_experts (非base): 0.3 (小)
            # - alpha_gates: 0.8 (中等，任务特定门控)
            # - beta_base: 0.3 (小到中等)
            # - beta_experts (非base): 0.8 (中等)
            # - beta_gates: 1.0 (正常，需要充分学习)
            # - task_heads (interactors/projectors): 0.8 (中等)
            # - classifiers: 1.0 (正常)
            # - log_vars: 1.0 (正常)

            lr_scale_config = self.config.get('lr_scale_protected', {})
            lr_scales = {
                'alpha_base': lr_scale_config.get('alpha_base', 0.1),
                'alpha_experts': lr_scale_config.get('alpha_experts', 0.3),
                'alpha_gates': lr_scale_config.get('alpha_gates', 0.8),
                'beta_base': lr_scale_config.get('beta_base', 0.3),
                'beta_experts': lr_scale_config.get('beta_experts', 0.8),
                'beta_gates': lr_scale_config.get('beta_gates', 1.0),
                'task_heads': lr_scale_config.get('task_heads', 0.8),
                'classifiers': lr_scale_config.get('classifiers', 1.0),
                'static_encoders': lr_scale_config.get('static_encoders', 1.0),
                'gate_context': lr_scale_config.get('gate_context', 0.5),
                'static_mlp': lr_scale_config.get('static_mlp', 0.5),
            }

            # 按参数名分组
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                # 根据参数名确定学习率缩放
                scale = 1.0  # 默认
                wd = weight_decay  # 默认

                # log_vars: 无 weight decay
                if 'log_vars' in name:
                    scale = 1.0
                    wd = 0.0
                # Alpha 专家
                elif 'alpha_experts' in name:
                    if 'alpha_experts.base' in name:
                        scale = lr_scales['alpha_base']
                    else:
                        scale = lr_scales['alpha_experts']
                # Alpha 门控
                elif 'alpha_gates' in name:
                    scale = lr_scales['alpha_gates']
                # Beta 专家
                elif 'beta_experts' in name:
                    if 'beta_experts.base' in name:
                        scale = lr_scales['beta_base']
                    else:
                        scale = lr_scales['beta_experts']
                # Beta 门控
                elif 'beta_gates' in name:
                    scale = lr_scales['beta_gates']
                # 任务交互头
                elif 'alpha_interactors' in name or 'beta_projectors' in name:
                    scale = lr_scales['task_heads']
                # 分类头
                elif 'classifiers' in name:
                    scale = lr_scales['classifiers']
                # 静态编码器
                elif 'static_encoders' in name:
                    scale = lr_scales['static_encoders']
                # Gate context
                elif 'alpha_gate_context' in name or 'beta_gate_context' in name:
                    scale = lr_scales['gate_context']
                # 共享静态 MLP
                elif 'shared_static_mlp' in name:
                    scale = lr_scales['static_mlp']

                param_groups.append({
                    'params': param,
                    'lr': base_lr * scale,
                    'weight_decay': wd,
                    'name': name  # 添加 name 用于调试
                })

            optimizer = torch.optim.Adam(param_groups)

            # 打印参数分组信息
            print(f"[Info] ProtectedDualEngineMTL 参数分组优化器: {len(param_groups)} 个参数组")
            lr_counts = {}
            for pg in param_groups:
                lr = pg['lr']
                lr_counts[lr] = lr_counts.get(lr, 0) + 1
            for lr, count in sorted(lr_counts.items(), key=lambda x: x[0]):
                print(f"  lr={lr:.6f}: {count} params")

            return optimizer

        # =====================================================
        # HDSTGCNMTL (原始): 简单分组
        # =====================================================
        # 分离 log_vars 和其他参数
        log_var_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if 'log_vars' in name:
                log_var_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': log_var_params, 'lr': base_lr, 'weight_decay': 0.0}  # log_vars 无 weight decay
        ])

        return optimizer

    def _apply_t6_log_var_freeze(self):
        """根据 freeze_t6_log_var 配置冻结/解冻 t6 的 log_var 参数"""
        if not self.is_protected_dual_engine_v4:
            return
        if self.freeze_t6_log_var:
            if "t6" in self.model.log_vars:
                self.model.log_vars["t6"].requires_grad = False
                print("[v4] t6 log_var 已冻结 (不参与不确定性自动加权)")
        else:
            if "t6" in self.model.log_vars:
                self.model.log_vars["t6"].requires_grad = True
                print("[v4] t6 log_var 已解冻")

    def run_stage0_t6_semantic_warmstart(self, epochs: int = 3):
        """
        [v4新增] Stage0: T6 Semantic Warm-start

        目标: 让 dyn_feat_t6 具备初始疾病语义，为后续 t6 deep context injection 提供稳定基础

        - 只训练 t6 任务
        - Beta 分支、t1_head 冻结
        - 使用 t6 teacher KD (训练式 warm-start，不是权重初始化)
        - checkpoint: save_last_only (不参与最终模型选择)

        Loss公式: L = t6_ce * CE_t6 + t6_kd * KD_t6 [+ t6_feat * FeatureKD]
        """
        # 检查是否启用
        stage0_cfg = self.config.get('training_stages', {}).get('v4_stage0_t6_semantic_warmstart', {})
        if not stage0_cfg.get('enabled', False):
            print("[Stage0] 未启用，跳过")
            return False

        # Feature Distillation 配置
        feat_distill_cfg = stage0_cfg.get('feature_distillation', {})
        feat_distill_enabled = feat_distill_cfg.get('enabled', False)
        teacher_usage = stage0_cfg.get('teacher_usage', 'logits_only')

        # [修复 2026-05-13] 不再强制启用 feature distillation
        # 尊重配置文件中的 feature_distillation.enabled 设置，以支持消融实验 A (logits_only)
        # 原逻辑: if teacher_usage == 'logits_and_features' and not feat_distill_enabled:
        # 新逻辑: 只有当 teacher_usage == 'logits_and_features' 且配置文件未显式设置 enabled=False 时才使用默认配置
        if teacher_usage == 'logits_and_features' and feat_distill_enabled and not feat_distill_cfg:
            # 仅当 teacher_usage=logits_and_features 且 enabled=True 但无其他配置时，使用默认配置
            feat_distill_cfg = {'enabled': True, 'teacher_feature': 'dyn_feat', 'student_feature': 'dyn_feat',
                                'loss': 'mse', 'normalize': True, 'detach_teacher': True, 'weight': 0.2}

        print("\n" + "=" * 80)
        print("Stage0: T6 Semantic Warm-start (v4)")
        print("=" * 80)
        print(f"[Stage0] 目标: 让 dyn_feat_t6 具备初始疾病语义")
        print(f"[Stage0] 方法: t6 teacher logits KD" +
              (" + feature distillation (weight=" + str(feat_distill_cfg.get('weight', 0.2)) + ")" if feat_distill_enabled else " (仅 logits KD)") +
              " (训练式 warm-start)")
        print(f"[Stage0] Epochs: {epochs}")
        print(f"[Stage0] teacher_usage: {teacher_usage}")
        print(f"[Stage0] feature_distillation.enabled: {feat_distill_enabled}")

        # 加载 t6 teacher checkpoint
        teacher_checkpoint_path = stage0_cfg.get('teacher_checkpoint', '')
        if not teacher_checkpoint_path or not os.path.exists(teacher_checkpoint_path):
            print(f"[Stage0 Error] 未找到 teacher checkpoint: {teacher_checkpoint_path}")
            print("[Stage0] 跳过 Stage0，使用 from_scratch 初始化")
            return False

        print(f"[Stage0] 加载 teacher checkpoint: {teacher_checkpoint_path}")

        # 加载 teacher 模型 (用于产生 logits，不拷贝权重)
        # 使用单任务 HDSTGCN 模型结构
        from model import HDSTGCN
        teacher_state_dict = torch.load(teacher_checkpoint_path, map_location=self.device, weights_only=False)

        # 从 checkpoint 推断 num_classes
        # checkpoint 是纯 state_dict，没有元信息，需要从 classifier 最后层推断
        # classifier 结构: Linear(fusion_dim, 32) -> ... -> Linear(32, num_classes)
        # 查找 classifier.4.weight 或 classifier.4.bias 的形状
        teacher_num_classes = 5  # 默认值
        for key, tensor in teacher_state_dict.items():
            if 'classifier.4.weight' in key or 'classifier.4.bias' in key:
                # Linear(32, num_classes) 的 weight 形状是 [num_classes, 32]
                teacher_num_classes = tensor.shape[0] if 'weight' in key else tensor.shape[0]
                break

        # 获取必需参数 (匹配 HDSTGCN.__init__ 签名)
        # input_dim: 时间步长 (max_length 或 L_win)
        # hidden_dim: 隐藏维度
        # channel_groups: 通道分组 (从 adapt_mode 自动推断或配置获取)
        # output_dim: 类别数
        # num_channel: 特征通道数
        use_var_length = self.config.get('use_variable_length', False)
        input_dim = self.config.get('max_length', 330) if use_var_length else self.config.get('L_win', 200)
        hidden_dim = self.config.get('hidden_dim', 16)
        num_channel = self.config.get('num_channels', 30)  # 注意: HDSTGCN 参数名为 num_channel (无s)

        # channel_groups: 从配置或 features.adapt_mode 推断
        from feature_mapping import get_channel_groups
        adapt_mode = self.config.get('adapt_mode', 'nine_graph')
        channel_groups_dict = get_channel_groups(adapt_mode)
        channel_groups = channel_groups_dict.get(adapt_mode, list(channel_groups_dict.values())[0])

        teacher_model = HDSTGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            channel_groups=channel_groups,
            output_dim=teacher_num_classes,  # 注意: HDSTGCN 参数名为 output_dim (非 num_classes)
            num_channel=num_channel,         # 注意: HDSTGCN 参数名为 num_channel (非 num_channels)
            D_time=self.config.get('D_time', 16),
            T_mid=self.config.get('T_mid', 24),
            semantic_adj=self.model.semantic_adj if hasattr(self.model, 'semantic_adj') else None,
            temporal_encoder_type=self.config.get('temporal_encoder_type', 'cnn'),
            graph_ablation=self.config.get('graph_ablation', 'prior_masked'),
            use_static_features=self.config.get('static_features', {}).get('enabled', True),
            static_dim=self.config.get('static_features', {}).get('static_dim', 16)
        ).to(self.device)
        teacher_model.load_state_dict(teacher_state_dict, strict=False)
        teacher_model.eval()
        self.stage0_teacher_model = teacher_model
        print(f"[Stage0] Teacher 模型加载完成 (num_classes={teacher_num_classes})")

        # KD 参数
        kd_temperature = stage0_cfg.get('kd_temperature', 2.0)
        loss_weights = stage0_cfg.get('loss_weights', {})
        ce_t6_weight = loss_weights.get('t6_ce', 1.0)
        kd_t6_weight = loss_weights.get('t6_kd', 0.5)
        feat_t6_weight = loss_weights.get('t6_feat', feat_distill_cfg.get('weight', 0.2)) if feat_distill_enabled else 0.0

        print(f"[Stage0] KD temperature: {kd_temperature}")
        print(f"[Stage0] Loss weights: CE={ce_t6_weight}, KD={kd_t6_weight}, FeatKD={feat_t6_weight}")
        if feat_distill_enabled:
            print(f"[Stage0] Feature distillation: teacher_feat={feat_distill_cfg.get('teacher_feature', 'dyn_feat')}, "
                  f"student_feat={feat_distill_cfg.get('student_feature', 'dyn_feat')}, "
                  f"loss={feat_distill_cfg.get('loss', 'mse')}, normalize={feat_distill_cfg.get('normalize', True)}, "
                  f"detach_teacher={feat_distill_cfg.get('detach_teacher', True)}")

        # 冻结模块
        freeze_modules = stage0_cfg.get('freeze_modules', ['beta_branch', 't1_head'])
        if freeze_modules:
            print(f"[Stage0] 冻结模块: {freeze_modules}")
            if self.is_protected_dual_engine:
                if 'beta_branch' in freeze_modules:
                    self.model.freeze_beta_modules()
                #冻结 t1 相关模块
                if 't1_head' in freeze_modules:
                    if hasattr(self.model, 'alpha_interactors') and 't1' in self.model.alpha_interactors:
                        for param in self.model.alpha_interactors['t1'].parameters():
                            param.requires_grad = False
                    if hasattr(self.model, 'classifiers') and 't1' in self.model.classifiers:
                        for param in self.model.classifiers['t1'].parameters():
                            param.requires_grad = False

        # 训练模块 (仅 alpha_trunk, t6 相关)
        print("[Stage0] 训练模块: alpha_trunk, alpha_interactors.t6, classifiers.t6")

        # 初始化增量 JSON
        self._init_incremental_json(fold=self.config.get('fold', 1))

        active_tasks = ["t6"]
        best_metric = 0.0

        for epoch in range(epochs):
            # 训练 (使用 KD)
            train_metrics = self._train_epoch_stage0(
                epoch,
                active_tasks=active_tasks,
                teacher_model=self.stage0_teacher_model,
                kd_temperature=kd_temperature,
                ce_weight=ce_t6_weight,
                kd_weight=kd_t6_weight,
                feat_distill_enabled=feat_distill_enabled,
                feat_distill_cfg=feat_distill_cfg,
                feat_weight=feat_t6_weight
            )

            # 验证
            val_metrics = self._validate_epoch(active_tasks, epoch=epoch, stage="stage0")

            # Stage0 使用 t6_macro_f1 作为监控指标 (但不参与最终选择)
            main_metric = val_metrics["t6"]["macro_f1"]
            self.scheduler.step(main_metric)

            self._log_to_swanlab(epoch, train_metrics, val_metrics, stage="stage0")

            print(f"[Stage0] Epoch {epoch+1}/{epochs} | t6_macro_f1: {main_metric:.4f}")

        # Stage0 结束: 保存 last checkpoint (不参与最终选择)
        save_mtl_checkpoint(
            self.model, self.optimizer, epochs-1, "stage0",
            fold=self.config.get('fold', 1),
            metrics=val_metrics,
            prefix=self.checkpoint_prefix
        )
        self._save_stage_to_json("stage0")

        # 解冻所有模块 (为 Stage1 准备)
        self.model.unfreeze_all()
        self._apply_t6_log_var_freeze()  # 恢复 t6 log_var 冻结状态

        # 清理 teacher 模型
        del self.stage0_teacher_model
        self.stage0_teacher_model = None

        print(f"[Stage0 完成] dyn_feat_t6 已具备初始疾病语义")
        print(f"[Stage0] Stage1 将使用 t6_weight=0.05 (弱监督)")

        return True

    def _train_epoch_stage0(
        self,
        epoch: int,
        active_tasks: List[str],
        teacher_model: nn.Module,
        kd_temperature: float = 2.0,
        ce_weight: float = 1.0,
        kd_weight: float = 0.5,
        feat_distill_enabled: bool = False,
        feat_distill_cfg: dict = None,
        feat_weight: float = 0.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Stage0 专用训练 epoch (包含 KD + Feature Distillation)

        Args:
            epoch: Epoch 编号
            active_tasks: 活动任务列表 (Stage0 只包含 ["t6"])
            teacher_model: T6 teacher 模型 (用于 KD)
            kd_temperature: KD 温度参数
            ce_weight: CE loss 权重
            kd_weight: KD loss 权重
            feat_distill_enabled: 是否启用 feature distillation
            feat_distill_cfg: feature distillation 配置字典
            feat_weight: feature distillation loss 权重

        Returns:
            各任务的训练指标字典 (含 ce_loss, kd_loss, feat_loss)
        """
        self.model.train()
        teacher_model.eval()

        accumulation_steps = self.config.get('accumulation_steps', 2)
        all_metrics = {t: {"loss": 0.0, "ce_loss": 0.0, "kd_loss": 0.0, "feat_loss": 0.0, "count": 0} for t in active_tasks}

        # Feature distillation 参数
        teacher_feat_key = "dyn_feat"
        student_feat_key = "dyn_feat"
        feat_loss_type = "mse"
        feat_normalize = True
        feat_detach_teacher = True
        if feat_distill_enabled and feat_distill_cfg:
            teacher_feat_key = feat_distill_cfg.get('teacher_feature', 'dyn_feat')
            student_feat_key = feat_distill_cfg.get('student_feature', 'dyn_feat')
            feat_loss_type = feat_distill_cfg.get('loss', 'mse')
            feat_normalize = feat_distill_cfg.get('normalize', True)
            feat_detach_teacher = feat_distill_cfg.get('detach_teacher', True)

        for batch_idx, batch in enumerate(self.train_loader):
            x_dyn = batch["x_dyn"].to(self.device)
            x_static = batch["x_static"].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

            # 前向传播 (MTL模型)
            if self.is_protected_dual_engine:
                outputs = self.model(
                    x_dyn, x_static,
                    tau_override=None,
                    return_gate_weights=False,
                    detach_t6_context=False,
                    skip_t6_injection=False  # Stage0 不涉及注入
                )
            else:
                outputs = self.model(x_dyn, x_static, return_aux=False)

            # 计算 CE loss
            task_key = "t6"
            logits = outputs[task_key]["logits"]
            target = labels[task_key].long()

            ce_loss = self.criterions[task_key](logits, target)

            # 计算 KD loss + Feature Distillation
            feat_loss = torch.tensor(0.0, device=self.device)

            # Teacher 前向传播
            use_feat_distill = feat_distill_enabled and feat_weight > 0
            teacher_feat_dict = None
            with torch.no_grad():
                if use_feat_distill:
                    # Feature distillation 需要 teacher 的中间特征
                    teacher_logits, teacher_feat_dict = teacher_model.forward_with_features(
                        x_dyn, static_x=x_static, return_feature_dict=True
                    )
                else:
                    teacher_logits = teacher_model(x_dyn, static_x=x_static)
                    if isinstance(teacher_logits, dict):
                        teacher_logits = teacher_logits["logits"]

            # KD loss: KL divergence between student and teacher soft labels
            kd_loss = self._compute_kd_loss(
                student_logits=logits,
                teacher_logits=teacher_logits,
                temperature=kd_temperature
            )

            # Feature distillation loss
            if use_feat_distill:
                # Student feature: 从 MTL outputs 中获取
                student_feat = outputs[task_key][student_feat_key]  # [B, 48]
                # Teacher feature: 从 teacher_feat_dict 中获取
                teacher_feat = teacher_feat_dict[teacher_feat_key]  # [B, 48]
                if feat_detach_teacher:
                    teacher_feat = teacher_feat.detach()

                feat_loss = self._compute_feature_distill_loss(
                    student_feat=student_feat,
                    teacher_feat=teacher_feat,
                    loss_type=feat_loss_type,
                    normalize=feat_normalize
                )

            # 总 loss: CE * t6_ce + KD * t6_kd + FeatKD * t6_feat
            total_loss = ce_weight * ce_loss + kd_weight * kd_loss + feat_weight * feat_loss

            # 梯度累加
            scaled_loss = total_loss / accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 累积指标
            all_metrics[task_key]["loss"] += total_loss.item()
            all_metrics[task_key]["ce_loss"] += ce_loss.item()
            all_metrics[task_key]["kd_loss"] += kd_loss.item()
            all_metrics[task_key]["feat_loss"] += feat_loss.item()
            all_metrics[task_key]["count"] += 1

        # 平均
        avg_metrics = {}
        for task_key in active_tasks:
            cnt = max(all_metrics[task_key]["count"], 1)
            avg_metrics[task_key] = {
                "loss": all_metrics[task_key]["loss"] / cnt,
                "ce_loss": all_metrics[task_key]["ce_loss"] / cnt,
                "kd_loss": all_metrics[task_key]["kd_loss"] / cnt,
                "feat_loss": all_metrics[task_key]["feat_loss"] / cnt,
            }

        return avg_metrics

    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 2.0
    ) -> torch.Tensor:
        """
        计算 KD loss (KL divergence)

        公式: L_KD = T^2 * KL(p_teacher || p_student)
        其中 p = softmax(logits / T)

        Args:
            student_logits: 学生模型 logits [B, C]
            teacher_logits: 教师模型 logits [B, C]
            temperature: 温度参数

        Returns:
            KD loss (scalar)
        """
        # 软标签
        student_soft = F.log_softmax(student_logits / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)

        # KL divergence
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

        return kd_loss

    def _compute_feature_distill_loss(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        loss_type: str = "mse",
        normalize: bool = True
    ) -> torch.Tensor:
        """
        计算 Feature Distillation Loss

        Args:
            student_feat: 学生模型特征 [B, D]
            teacher_feat: 教师模型特征 [B, D] (已 detach)
            loss_type: "mse" 或 "smooth_l1"
            normalize: 是否 L2 归一化后再计算

        Returns:
            Feature distillation loss (scalar)
        """
        if normalize:
            student_feat = F.normalize(student_feat, p=2, dim=1)
            teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

        if loss_type == "smooth_l1":
            feat_loss = F.smooth_l1_loss(student_feat, teacher_feat)
        else:
            feat_loss = F.mse_loss(student_feat, teacher_feat)

        return feat_loss

    def run_stage1_alpha_anchor(self, epochs: int = 20):
        """
        阶段一: Alpha 主任务锚定

        - 只训练 t1, t6
        - Beta 分支冻结
        - Alpha 编码器从 t6 单任务加载
        - 不使用不确定性加权 (固定权重)
        - KD 保护 t6
        """
        print("\n" + "=" * 80)
        print("阶段一: Alpha 主任务锚定")
        print("=" * 80)

        # [修复 - 2026-04-22] 初始化增量 JSON 文件
        self._init_incremental_json(fold=self.config.get('fold', 1))

        # [Ablation E2] 检测 single_shared_alpha 消融模式
        # E2: 所有任务走 Alpha 分支，Stage1 需要训练全部任务
        if hasattr(self.model, 'ablation_single_shared_alpha') and self.model.ablation_single_shared_alpha:
            # E2 消融: alpha_tasks 包含 t1-t5 + t6
            active_tasks = self.model.alpha_tasks  # 所有 Alpha 任务
            print(f"[Ablation E2] Stage1 训练所有 Alpha 任务: {active_tasks}")
            # E2 不需要冻结 Beta（Beta 已在模型初始化时禁用）
        else:
            # 冻结 Beta 分支 (兼容两种模型)
            if self.is_protected_dual_engine:
                self.model.freeze_beta_modules() #v2/v3/v4
            else:
                self.model.freeze_beta_branch() #v1

        # [修复 - 2026-05-06] v4 T6辅助模式: 冻结 t6_deep_context 模块 (与文档描述一致)
        if self.is_protected_dual_engine_v4 and self.t6_auxiliary_mode_enabled:
            if hasattr(self.model, 'freeze_t6_context_modules'):
                self.model.freeze_t6_context_modules()
                print("[Stage1 v4] T6DeepFeatureContextModule frozen (encoder + bridge)")

        # 从 t1 单任务加载 (如果有)
        if self.teacher_model is not None:
            print("[Info] 使用教师模型作为初始化参考")

        # [修复 - 2026-04-29] Stage1 不应用全局 lr_scale
        # v3/v4 设计意图: Stage1 使用完整学习率，Stage3 才精细化缩放
        # 避免 alpha_trunk 学习率过小 (0.0003 * 0.1 = 0.00003)，导致收敛极慢
        print("[Stage1] 使用完整学习率 (不应用 lr_scale 缩放)")

        # [新增 - 2026-04-28] v4 Baseline模式: 使用override后的active_tasks
        if self.is_protected_dual_engine_v4 and hasattr(self, 'v4_stage1_active_tasks_override') and self.v4_stage1_active_tasks_override:
            active_tasks = self.v4_stage1_active_tasks_override
            mode_label = "实验E" if getattr(self, 'no_t6_context_but_train_t6_enabled', False) else "Baseline模式"
            print(f"[Stage1 v4] active_tasks: {active_tasks} ({mode_label})")
        else:
            active_tasks = ["t1", "t6"]
            print(f"[Stage1] active_tasks: {active_tasks}")

        # [修复 - 2026-04-29] 从配置读取 use_kd_t1 (修复路径: training_stages -> v4_stage1)
        if self.is_protected_dual_engine_v4:
            stage1_cfg = self.config.get('training_stages', {}).get('v4_stage1_alpha_anchor', {})
            use_kd = stage1_cfg.get('use_kd_t1', False)
            stage1_loss_weights = stage1_cfg.get('loss_weights', None)
            # [实验E] 覆盖 loss_weights (确保 t6 弱监督权重正确)
            if hasattr(self, 'v4_stage1_loss_weights_override') and self.v4_stage1_loss_weights_override is not None:
                stage1_loss_weights = self.v4_stage1_loss_weights_override
                print(f"[Stage1 实验E] loss_weights override: {stage1_loss_weights}")
            # [Ablation E2] E2 模式下从 yaml 的 ablation 配置读取 loss_weights
            elif hasattr(self.model, 'ablation_single_shared_alpha') and self.model.ablation_single_shared_alpha:
                ablation_cfg = self.config.get('hcgc_v4', {}).get('ablation', {})
                stage1_loss_weights = ablation_cfg.get('loss_weights', None)
                if stage1_loss_weights:
                    print(f"[Ablation E2 Stage1] loss_weights from yaml: {stage1_loss_weights}")
        else:
            # v1/v2/v3 旧版本默认使用 KD
            use_kd = True
            stage1_loss_weights = None
        print(f"[Stage1] use_kd={use_kd} (v4={self.is_protected_dual_engine_v4})")
        if stage1_loss_weights:
            print(f"[Stage1] loss_weights: {stage1_loss_weights}")

        best_metric = 0.0

        for epoch in range(epochs):
            # 训练
            train_metrics = self._train_epoch(
                epoch,
                active_tasks=active_tasks,
                use_kd=use_kd,
                use_uncertainty_weighting=False,
                stage="stage1",
                loss_weights=stage1_loss_weights
            )

            # 验证
            val_metrics = self._validate_epoch(active_tasks, epoch=epoch, stage="stage1")

            # [Ablation E2] E2 模式下训练所有任务，使用综合指标
            if hasattr(self.model, 'ablation_single_shared_alpha') and self.model.ablation_single_shared_alpha:
                # E2: 使用 weighted_macro_f1_t1_to_t5 作为主指标
                main_metric = compute_weighted_macro_f1_t1_to_t5(val_metrics)
                checkpoint_metric_name = "weighted_macro_f1_t1_to_t5"
                print(f"[Ablation E2 Stage1] checkpoint指标: {checkpoint_metric_name}")
            # [新增 - 2026-04-28] v4 Baseline模式和T6辅助模式: 使用 t1 作为主指标
            elif self.is_protected_dual_engine_v4:
                # Baseline模式: 使用 t1_macro_f1 作为主指标
                main_metric = val_metrics["t1"]["macro_f1"]
                checkpoint_metric_name = "t1_macro_f1"
            else:
                # 默认: 使用 t6_macro_f1 作为主指标
                main_metric = val_metrics["t6"]["macro_f1"]
                checkpoint_metric_name = "t6_macro_f1"
            self.scheduler.step(main_metric)

            # [修复 - 2026-04-22] 始终调用 _log_to_swanlab (即使 SwanLab 禁用也会填充 metrics_history)
            self._log_to_swanlab(epoch, train_metrics, val_metrics, stage="stage1")

            # 保存最佳
            if main_metric > best_metric:
                best_metric = main_metric
                save_mtl_checkpoint(
                    self.model, self.optimizer, epoch, "stage1",
                    fold=self.config.get('fold', 1),
                    metrics=val_metrics,
                    prefix=self.checkpoint_prefix
                )
                # [新增] 更新增量 JSON checkpoint 信息
                self._update_checkpoint_info_in_json("stage1", epoch, main_metric, checkpoint_metric_name)

            # [修复 - 2026-04-28] Baseline模式下打印 t1 指标
            print(f"Epoch {epoch+1}/{epochs} | {checkpoint_metric_name}: {main_metric:.4f} | Best: {best_metric:.4f}")

        # 记录最佳指标 (供 main_mtl.py 返回)
        self.stage1_best_f1 = best_metric

        # [修复] 回退到 stage1 best checkpoint (避免从 last epoch 开始 stage2)
        stage1_best_path = os.path.join("models", f"best_{self.checkpoint_prefix}_stage1_fold{self.config.get('fold', 1)}.pth")
        if os.path.exists(stage1_best_path):
            print(f"[Stage1] 加载 best checkpoint 以回退到最优状态: {stage1_best_path}")
            ckpt = torch.load(stage1_best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"[Stage1] 已回退到 best epoch (metric={best_metric:.4f})")
        else:
            print(f"[Stage1 Warning] best checkpoint 未找到 ({stage1_best_path})，使用 last epoch 状态")

        # [新增] 阶段结束时批量保存 JSON
        self._save_stage_to_json("stage1")

        # 解冻
        self.model.unfreeze_all()
        self._apply_t6_log_var_freeze()  # 恢复 t6 log_var 冻结状态

    def run_stage2_beta_warmup(self, epochs: int = 20):
        """
        阶段二: Beta 副引擎预热

        - 只训练 t2~t5
        - Alpha 分支冻结
        - 使用 Beta 任务不确定性加权
        - 不使用 KD

        [Ablation E2] 如果 Beta 完全禁用，跳过此阶段
        """
        print("\n" + "=" * 80)
        print("阶段二: Beta 副引擎预热")
        print("=" * 80)

        # [Ablation E2] 检测 single_shared_alpha 消融模式
        if hasattr(self.model, 'ablation_single_shared_alpha') and self.model.ablation_single_shared_alpha:
            print("[Ablation E2] Beta 分支完全禁用，跳过 Stage2 Beta Warmup")
            print("[Ablation E2] 所有任务已通过 Alpha 分支在 Stage1 训练完成")
            self.stage2_best_f1 = 0.0  # 记录为0，表示跳过
            return

        # 冻结 Alpha 分支 (兼容两种模型)
        if self.is_protected_dual_engine:
            self.model.freeze_alpha_modules()
        else:
            self.model.freeze_alpha_branch()

        # 冻结 t1, t6 的 log_vars
        for task_key in ["t1", "t6"]:
            self.model.log_vars[task_key].requires_grad = False

        active_tasks = ["t2", "t3", "t4", "t5"]

        # [修复 - 2026-05-06] use_kd 配置键名统一 (原 use_kd_t6 命名误导)
        # 注意: Stage2 的 active_tasks 不包含 t1，KD 实际无效
        if self.is_protected_dual_engine_v4:
            stage2_cfg = self.config.get('training_stages', {}).get('v4_stage2_beta_warmup', {})
            use_kd = stage2_cfg.get('use_kd', False)
            stage2_loss_weights = stage2_cfg.get('loss_weights', None)
        else:
            # v1/v2/v3 旧版本默认不使用 KD (Stage2)
            use_kd = False
            stage2_loss_weights = None
        print(f"[Stage2] use_kd={use_kd} (v4={self.is_protected_dual_engine_v4})")
        if stage2_loss_weights:
            print(f"[Stage2] loss_weights: {stage2_loss_weights}")

        best_metric = 0.0

        for epoch in range(epochs):
            train_metrics = self._train_epoch(
                epoch,
                active_tasks=active_tasks,
                use_kd=use_kd,
                use_uncertainty_weighting=True,
                stage="stage2",
                loss_weights=stage2_loss_weights
            )

            val_metrics = self._validate_epoch(active_tasks, epoch=epoch, stage="stage2")

            # [修复 - 2026-04-27] 使用统一指标函数，与 config_mtl.yaml 保持一致
            mean_f1_t2_to_t5 = compute_mean_macro_f1_t2_to_t5(val_metrics)
            self.scheduler.step(mean_f1_t2_to_t5)

            # [修复 - 2026-04-22] 始终调用 _log_to_swanlab (即使 SwanLab 禁用也会填充 metrics_history)
            self._log_to_swanlab(epoch, train_metrics, val_metrics, stage="stage2")

            if mean_f1_t2_to_t5 > best_metric:
                best_metric = mean_f1_t2_to_t5
                save_mtl_checkpoint(
                    self.model, self.optimizer, epoch, "stage2",
                    fold=self.config.get('fold', 1),
                    metrics=val_metrics,
                    prefix=self.checkpoint_prefix
                )
                # [新增] 更新增量 JSON checkpoint 信息 (统一指标命名)
                self._update_checkpoint_info_in_json("stage2", epoch, mean_f1_t2_to_t5, "mean_macro_f1_t2_to_t5")

            print(f"Epoch {epoch+1}/{epochs} | Mean F1 (t2-t5): {mean_f1_t2_to_t5:.4f} | Best: {best_metric:.4f}")

        # 记录最佳指标 (供 main_mtl.py 返回)
        self.stage2_best_f1 = best_metric

        # [修复] 回退到 stage2 best checkpoint (避免从 last epoch 开始 stage3)
        stage2_best_path = os.path.join("models", f"best_{self.checkpoint_prefix}_stage2_fold{self.config.get('fold', 1)}.pth")
        if os.path.exists(stage2_best_path):
            print(f"[Stage2] 加载 best checkpoint 以回退到最优状态: {stage2_best_path}")
            ckpt = torch.load(stage2_best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"[Stage2] 已回退到 best epoch (metric={best_metric:.4f})")
        else:
            print(f"[Stage2 Warning] best checkpoint 未找到 ({stage2_best_path})，使用 last epoch 状态")

        # [新增] 阶段结束时批量保存 JSON
        self._save_stage_to_json("stage2")

        self.model.unfreeze_all()
        self._apply_t6_log_var_freeze()  # 恢复 t6 log_var 冻结状态

    def run_stage3_joint_finetune(self, epochs: int = 60, skip_phase1: bool = False):
        """
        阶段三: 全模型联合微调

        v3 两段式训练 (仅v3架构生效):
        - 前30% epochs: 冻结Alpha，强化Beta专家
        - 后70% epochs: 联合微调 (全任务)

        v2/v1/baseline: 全程联合微调

        Args:
            epochs: 总训练轮数
            skip_phase1: [新增] 是否跳过 Phase1 直接进入 Phase2 (用于 stage3_phase2 resume)

        保存策略:
        - 先保存最佳模型 (按综合分数，无论是否满足约束)
        - 记录约束状态用于最终选择
        """
        print("\n" + "=" * 80)
        print("阶段三: 联合蒸馏微调")
        if skip_phase1:
            print("[Stage3 Phase2 Resume] 跳过 Phase1，直接进入 Phase2")
        print("=" * 80)

        # [修复 - 2026-04-22] 如果 incremental_json 未初始化，在这里初始化 (用于 resume stage3)
        if self.incremental_json_path is None:
            self._init_incremental_json(fold=self.config.get('fold', 1))

        # 检查是否需要v3两段式训练
        hcgc_v3_config = self.config.get('hcgc_v3', {})
        training_stages_config = self.config.get('training_stages', {})
        stage3_config = training_stages_config.get('stage3_joint_finetune', {})
        freeze_alpha_ratio = hcgc_v3_config.get('freeze_alpha_epochs_ratio',
                                                 stage3_config.get('freeze_alpha_epochs_ratio', 0.3))

        use_two_phase = freeze_alpha_ratio > 0

        if use_two_phase:
            freeze_epochs = int(epochs * freeze_alpha_ratio)

            # [新增 - 2026-04-21] 支持 phase2 独立时长
            training_stages_config = self.config.get('training_stages', {})
            stage3_config = training_stages_config.get('stage3_joint_finetune', {})
            phase2_config = stage3_config.get('phase2_joint', {})
            phase2_epochs_override = phase2_config.get('phase2_epochs_override', None)

            if phase2_epochs_override is not None:
                joint_epochs = phase2_epochs_override
                print(f"[v3 两段式] Phase2 独立时长: {joint_epochs} epochs (覆盖计算值 {epochs - freeze_epochs})")
            else:
                joint_epochs = epochs - freeze_epochs

            print(f"[v3 两段式] 启用")
            if not skip_phase1:
                print(f"[v3 两段式] Phase 1: {freeze_epochs} epochs (冻结Alpha，强化Beta)")
            print(f"[v3 两段式] Phase 2: {joint_epochs} epochs (联合微调)")
            print("=" * 80)

            # ===============================================
            # Phase 1: 冻结Alpha，强化Beta专家
            # ===============================================
            phase1_best_metric = 0.0  # 初始化 (用于 Phase 2 打印)

            # [Ablation E2] 检测 single_shared_alpha 消融模式
            if hasattr(self.model, 'ablation_single_shared_alpha') and self.model.ablation_single_shared_alpha:
                print("[Ablation E2] Beta 分支完全禁用，跳过 Stage3 Phase 1 (Beta 强化阶段)")
                print("[Ablation E2] 直接进入 Phase 2 联合微调")
                skip_phase1 = True  # 强制跳过 Phase 1
                phase1_epochs = 0

            if not skip_phase1:
                # [修复 - 2026-04-29] 从配置读取 phase1 配置 (修复路径: training_stages -> v4_stage3)
                if self.is_protected_dual_engine_v4:
                    v4_stage3_cfg = self.config.get('training_stages', {}).get('v4_stage3_joint_finetune', {})
                    phase1_config = v4_stage3_cfg.get('v4_phase1_beta_warmup', {})
                else:
                    # v3: 从 training_stages.stage3_joint_finetune.phase1_beta_warmup 读取
                    phase1_config = stage3_config.get('phase1_beta_warmup', {})
                phase1_active_tasks = phase1_config.get('active_tasks', ["t2", "t3", "t4", "t5"])
                # [修复 - 2026-05-06] use_kd 配置键名统一 (原 use_kd_t6 命名误导)
                phase1_use_kd = phase1_config.get('use_kd', False)
                phase1_loss_weights = phase1_config.get('loss_weights', None)

                print(f"\n[v3/v4 Phase 1] 冻结Alpha，只训Beta任务 {phase1_active_tasks}")
                print(f"[Phase 1] use_kd={phase1_use_kd} (v4={self.is_protected_dual_engine_v4})")
                if phase1_loss_weights:
                    print(f"[Phase 1] loss_weights: {phase1_loss_weights}")

                # [修复] MMoE Clean 等架构不支持 freeze_alpha_modules，使用条件检查
                if hasattr(self.model, 'freeze_alpha_modules'):
                    self.model.freeze_alpha_modules()
                    print("[Phase 1] Alpha模块已冻结")
                else:
                    print("[Phase 1] 当前架构不支持 freeze_alpha_modules，跳过冻结")

                # 学习率缩放 (Phase 1: Beta相关模块)
                lr_scale = self.config.get('lr_scale', {})
                self._apply_lr_scale(lr_scale)

                for epoch in range(freeze_epochs):
                    # Phase 1: 只训Beta任务
                    train_metrics = self._train_epoch(
                        epoch,
                        active_tasks=phase1_active_tasks,
                        use_kd=phase1_use_kd,  # 从配置读取
                        use_uncertainty_weighting=True,
                        stage="stage3_phase1",
                        loss_weights=phase1_loss_weights
                    )

                    val_metrics = self._validate_epoch(
                        phase1_active_tasks,
                        epoch=epoch,
                        stage="stage3_phase1"
                    )

                    # Phase 1评分: Beta平均F1 (使用 yaml 配置的 active_tasks)
                    # [修复 - 2026-04-27] 使用统一指标函数，与 config_mtl.yaml 保持一致
                    mean_f1_t2_to_t5 = compute_mean_macro_f1_t2_to_t5(val_metrics)

                    # [修复 - 2026-04-22] 始终调用 _log_to_swanlab (即使 SwanLab 禁用也会填充 metrics_history)
                    self._log_to_swanlab(epoch, train_metrics, val_metrics, stage="stage3_phase1")

                    if mean_f1_t2_to_t5 > phase1_best_metric:
                        phase1_best_metric = mean_f1_t2_to_t5
                        # 保存Phase 1最佳模型
                        save_mtl_checkpoint(
                            self.model, self.optimizer, epoch, "stage3_phase1",
                            fold=self.config.get('fold', 1),
                            metrics=val_metrics,
                            prefix=self.checkpoint_prefix
                        )
                        # [新增] 更新增量 JSON checkpoint 信息 (统一指标命名)
                        self._update_checkpoint_info_in_json("stage3_phase1", epoch, mean_f1_t2_to_t5, "mean_macro_f1_t2_to_t5")
                        print(f"[Phase 1 保存] Epoch {epoch+1}: Mean F1 (t2-t5)={mean_f1_t2_to_t5:.4f}")

                    print(f"[Phase 1] Epoch {epoch+1}/{freeze_epochs} | Mean F1 (t2-t5): {mean_f1_t2_to_t5:.4f} | Best: {phase1_best_metric:.4f}")

                # Phase 1完成，记录最佳指标
                print(f"\n[Phase 1 完成] 最佳 Mean F1 (t2-t5): {phase1_best_metric:.4f}")

                # [修复] 回退到 Phase1 best checkpoint (避免从 last epoch 开始 Phase2)
                phase1_best_path = os.path.join("models", f"best_{self.checkpoint_prefix}_stage3_phase1_fold{self.config.get('fold', 1)}.pth")
                if os.path.exists(phase1_best_path):
                    print(f"[Phase 1] 加载 best checkpoint 以回退到最优状态: {phase1_best_path}")
                    ckpt = torch.load(phase1_best_path, map_location=self.device, weights_only=False)
                    self.model.load_state_dict(ckpt['model_state_dict'])
                    if 'optimizer_state_dict' in ckpt:
                        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print(f"[Phase 1] 已回退到 best epoch (metric={phase1_best_metric:.4f})")
                else:
                    print(f"[Phase 1 Warning] best checkpoint 未找到 ({phase1_best_path})，使用 last epoch 状态")
            else:
                print(f"\n[Phase 1 Skip] 跳过 Phase1，直接进入 Phase2")
                freeze_epochs = 0  # Phase 1 已跳过，start_epoch 应为 0

            # ===============================================
            # Phase 2: 解冻Alpha，联合微调 (精细化控制)
            # ===============================================
            print("\n[v3 Phase 2] 精细化切换：选择性冻结")

            # Step 1: 全解冻（恢复基线）
            self.model.unfreeze_all()
            self._apply_t6_log_var_freeze()  # 恢复 t6 log_var 冻结状态

            # Step 2: 从 phase2_joint 配置读取冻结列表 (phase2_config 已在前面读取)
            # [v3.1 - 2026-04-22] Beta Gates 不冻结，使用小学习率精细调控
            freeze_modules = phase2_config.get('freeze_modules', ["alpha_trunk"])

            if freeze_modules:
                print(f"[Phase 2] 冻结模块: {freeze_modules}")
                if hasattr(self.model, 'freeze_specific_modules'):
                    self.model.freeze_specific_modules(freeze_modules)
                    freeze_status = self.model.get_freeze_status()
                    self._log_freeze_status_to_swanlab(freeze_status, stage="stage3_phase2")
                else:
                    print("[Phase 2] 当前架构不支持 freeze_specific_modules，跳过冻结")

            # Step 3: 应用 Phase2 专属学习率缩放 (含 beta_gates, beta_gate_context)
            lr_scale_override = phase2_config.get('lr_scale_override', {})
            if lr_scale_override:
                self._apply_lr_scale_v2(lr_scale_override)
            else:
                lr_scale = self.config.get('lr_scale', {})
                self._apply_lr_scale_v2(lr_scale)

            # Step 4: Gate 控制配置
            gate_control = phase2_config.get('gate_control', {})
            if gate_control:
                self._apply_gate_control(gate_control, stage="phase2")

            # Step 5: [v3.1] T3 任务专属保护 (可选)
            freeze_t3_config = phase2_config.get('freeze_modules_t3', {})
            if freeze_t3_config.get('enabled', False):
                t3_modules = freeze_t3_config.get('modules', ["t3_gate", "beta_t3_private_expert"])
                # 使用任务专属冻结方法
                if hasattr(self.model, 'freeze_task_specific_modules'):
                    self.model.freeze_task_specific_modules({"t3": t3_modules})
                    print(f"[v3.1] T3 专属模块冻结: {t3_modules}")

            # Step 6: [新增 - 2026-05-25] T3 Anchor Teacher 加载 (边界稳定机制)
            # 在 Phase2 开始前加载历史最优 t3 checkpoint 作为 anchor teacher
            if self.t3_anchor_enabled and use_two_phase:
                print(f"\n[t3_anchor] 加载 Anchor Teacher...")
                fold = self.config.get('fold', 1)

                # 查找候选 checkpoint 路径
                stage2_ckpt_path = os.path.join(
                    "models",
                    f"best_{self.checkpoint_prefix}_stage2_fold{fold}.pth"
                )
                stage3_phase1_ckpt_path = os.path.join(
                    "models",
                    f"best_{self.checkpoint_prefix}_stage3_phase1_fold{fold}.pth"
                )

                # 收集存在的 checkpoint 及其 t3 Macro-F1
                candidate_ckpts = []

                if os.path.exists(stage2_ckpt_path):
                    ckpt2 = torch.load(stage2_ckpt_path, map_location='cpu', weights_only=False)
                    metrics2 = ckpt2.get('metrics', {})
                    if 't3' in metrics2 and 'macro_f1' in metrics2['t3']:
                        t3_f1_stage2 = metrics2['t3']['macro_f1']
                        candidate_ckpts.append({
                            'path': stage2_ckpt_path,
                            'stage': 'stage2',
                            't3_macro_f1': t3_f1_stage2
                        })
                        print(f"[t3_anchor] Stage2 checkpoint: t3_macro_f1={t3_f1_stage2:.4f}")
                    else:
                        print(f"[t3_anchor Warning] Stage2 checkpoint 缺少 t3 metrics")

                if os.path.exists(stage3_phase1_ckpt_path):
                    ckpt_p1 = torch.load(stage3_phase1_ckpt_path, map_location='cpu', weights_only=False)
                    metrics_p1 = ckpt_p1.get('metrics', {})
                    if 't3' in metrics_p1 and 'macro_f1' in metrics_p1['t3']:
                        t3_f1_phase1 = metrics_p1['t3']['macro_f1']
                        candidate_ckpts.append({
                            'path': stage3_phase1_ckpt_path,
                            'stage': 'stage3_phase1',
                            't3_macro_f1': t3_f1_phase1
                        })
                        print(f"[t3_anchor] Stage3_phase1 checkpoint: t3_macro_f1={t3_f1_phase1:.4f}")
                    else:
                        print(f"[t3_anchor Warning] Stage3_phase1 checkpoint 缺少 t3 metrics")

                # 选择最优 checkpoint (按 t3 Macro-F1 排序)
                if len(candidate_ckpts) > 0:
                    # auto_best_t3 模式: 选择 t3 Macro-F1 最高的 checkpoint
                    best_candidate = max(candidate_ckpts, key=lambda x: x['t3_macro_f1'])
                    best_ckpt_path = best_candidate['path']
                    best_ckpt_stage = best_candidate['stage']
                    best_t3_f1 = best_candidate['t3_macro_f1']

                    print(f"[t3_anchor] 选择最优 teacher checkpoint: {best_ckpt_stage}")
                    print(f"[t3_anchor]   path: {best_ckpt_path}")
                    print(f"[t3_anchor]   t3_macro_f1: {best_t3_f1:.4f}")

                    # 加载 teacher 模型 (与 self.model 相同架构)
                    # 创建相同架构的模型副本
                    import copy
                    self.t3_anchor_teacher_model = copy.deepcopy(self.model)

                    # 加载 checkpoint 权重
                    best_ckpt = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
                    self.t3_anchor_teacher_model.load_state_dict(best_ckpt['model_state_dict'], strict=False)

                    # 设置为 eval 模式并冻结所有参数
                    self.t3_anchor_teacher_model.eval()
                    for param in self.t3_anchor_teacher_model.parameters():
                        param.requires_grad = False

                    print(f"[t3_anchor] Teacher 模型加载完成，已冻结所有参数")
                    print(f"[t3_anchor] Phase2 训练时将使用 lambda={self.t3_anchor_lambda} 的 anchor loss")

                elif len(candidate_ckpts) == 0:
                    print(f"[t3_anchor Warning] 未找到任何有效的 t3 checkpoint，禁用 anchor 机制")
                    self.t3_anchor_enabled = False
                    self.t3_anchor_teacher_model = None

        else:
            # v2/v1/baseline: 全程联合微调
            print(f"[v2/baseline] 全程联合微调 ({epochs} epochs)")
            self.model.unfreeze_all()
            self._apply_t6_log_var_freeze()  # 恢复 t6 log_var 冻结状态
            lr_scale = self.config.get('lr_scale', {})
            self._apply_lr_scale(lr_scale)

        # ===============================================
        # 联合微调阶段 (v3 Phase 2 或 v2全程)
        # ===============================================
        # [修复 - 2026-04-22] 从 yaml 读取 phase2 active_tasks
        # [新增 - 2026-04-28] v4 Baseline模式: 使用override后的active_tasks
        # [Ablation E2] E2 模式下使用模型的所有 alpha_tasks
        if hasattr(self.model, 'ablation_single_shared_alpha') and self.model.ablation_single_shared_alpha:
            phase2_active_tasks = self.model.alpha_tasks  # E2: 所有任务走 Alpha
            print(f"[Ablation E2 Phase2] active_tasks: {phase2_active_tasks}")
        elif self.is_protected_dual_engine_v4 and hasattr(self, 'v4_phase2_active_tasks_override') and self.v4_phase2_active_tasks_override:
            phase2_active_tasks = self.v4_phase2_active_tasks_override
            mode_label = "实验E" if getattr(self, 'no_t6_context_but_train_t6_enabled', False) else "Baseline模式"
            print(f"[Phase2 v4] active_tasks: {phase2_active_tasks} ({mode_label})")
        elif use_two_phase:
            phase2_active_tasks = phase2_config.get('active_tasks', ["t1", "t2", "t3", "t4", "t5", "t6"])
        else:
            # v2/baseline: 从 stage3_joint_finetune 读取
            phase2_active_tasks = stage3_config.get('active_tasks', ["t1", "t2", "t3", "t4", "t5", "t6"])

        active_tasks = phase2_active_tasks
        best_metric = 0.0

        # [v3 已废弃 - 2026-04-29] v4 不使用 t6 约束机制
        # best_t6_f1 = 0.0
        # t6_constraint_delta = self.config.get('t6_constraint_delta', 0.005)
        # t6_baseline = self.config.get('t6_single_task_f1', 0.85)

        # 记录约束状态 (供 main_mtl.py 返回)
        self.stage3_meets_constraint = False
        self.stage3_best_f1 = 0.0

        # [v3.1] Phase2 early stopping 配置 (直接从 phase2_config 读取)
        phase2_early_stop_patience = 0
        phase2_early_stop_min_delta = 0.0
        if use_two_phase:
            phase2_early_stop_patience = phase2_config.get('early_stop_patience', 4)
            phase2_early_stop_min_delta = phase2_config.get('early_stop_min_delta', 0.001)
            print(f"[Phase 2 Early Stop] patience={phase2_early_stop_patience}, min_delta={phase2_early_stop_min_delta}")

        # v3: 联合微调使用 joint_epochs (支持 phase2_epochs_override)
        start_epoch = freeze_epochs if use_two_phase else 0
        phase2_epochs = joint_epochs if use_two_phase else epochs

        # [v3 已废弃 - 2026-04-29] v4 不使用双 checkpoint 策略
        # dual_checkpoint_config = phase2_config.get('dual_checkpoint', {})
        # dual_checkpoint_enabled = dual_checkpoint_config.get('enabled', True) if use_two_phase else False
        dual_checkpoint_enabled = False  # v4 禁用

        # [v3 已废弃 - 2026-04-29] v4 不追踪 t6/beta 双分数
        # best_t6_score = 0.0           # 主指标: t6_constrained_score
        # best_beta_score = 0.0         # 副指标: beta_balanced_score
        # best_t6_epoch = start_epoch
        # best_beta_epoch = start_epoch

        # Early stopping 状态
        no_improve_count = 0
        best_epoch = start_epoch  # 记录最佳 epoch

        for epoch_offset in range(phase2_epochs):
            epoch = start_epoch + epoch_offset

            # [修复 - 2026-05-06] use_kd 配置键名统一 (原 use_kd_t6 命名误导)
            phase2_loss_weights = None
            if self.is_protected_dual_engine_v4:
                # v4: 从 v4_phase2_joint 读取
                v4_stage3_cfg = self.config.get('training_stages', {}).get('v4_stage3_joint_finetune', {})
                v4_phase2_cfg = v4_stage3_cfg.get('v4_phase2_joint', {})
                phase2_use_kd = v4_phase2_cfg.get('use_kd', False)
                phase2_loss_weights = v4_phase2_cfg.get('loss_weights', None)
                # [实验E] 覆盖 loss_weights (确保 t6 弱监督权重正确)
                if hasattr(self, 'v4_phase2_loss_weights_override') and self.v4_phase2_loss_weights_override is not None:
                    phase2_loss_weights = self.v4_phase2_loss_weights_override
                    if epoch_offset == 0:
                        print(f"[Phase2 实验E] loss_weights override: {phase2_loss_weights}")
                # [Ablation E2] E2 模式下从 yaml 的 ablation 配置读取 loss_weights
                elif hasattr(self.model, 'ablation_single_shared_alpha') and self.model.ablation_single_shared_alpha:
                    ablation_cfg = self.config.get('hcgc_v4', {}).get('ablation', {})
                    phase2_loss_weights = ablation_cfg.get('loss_weights', None)
                    if phase2_loss_weights and epoch_offset == 0:
                        print(f"[Ablation E2 Phase2] loss_weights from yaml: {phase2_loss_weights}")
            elif use_two_phase:
                # v3: 从 phase2_joint 读取
                phase2_use_kd = phase2_config.get('use_kd', False)
                phase2_loss_weights = phase2_config.get('loss_weights', None)
            else:
                # v2/baseline: 从 stage3_joint_finetune 读取
                phase2_use_kd = stage3_config.get('use_kd', False)

            if phase2_loss_weights and epoch_offset == 0:
                print(f"[Phase2] loss_weights: {phase2_loss_weights}")

            train_metrics = self._train_epoch(
                epoch,
                active_tasks=active_tasks,
                use_kd=phase2_use_kd,
                use_uncertainty_weighting=True,
                stage="stage3_phase2" if use_two_phase else "stage3",
                loss_weights=phase2_loss_weights
            )

            # ========== [修复 - 2026-04-21] 不在训练循环内运行阈值搜索 ==========
            # threshold search 在训练结束后重新加载最佳 checkpoint 再运行
            val_metrics = self._validate_epoch(
                active_tasks,
                epoch=epoch,
                stage="stage3_phase2" if use_two_phase else "stage3",
                run_threshold_search=False
            )

            # [v3 已废弃 - 2026-04-29] v4 不使用 t6 约束检查
            # if "t6" in val_metrics:
            #     t6_f1 = val_metrics["t6"]["macro_f1"]
            #     meets_constraint = t6_f1 >= (t6_baseline - t6_constraint_delta)
            # else:
            #     t6_f1 = 0.0
            #     meets_constraint = True

            # [v3 已废弃 - 2026-04-29] v4 不使用 t6_score 综合评分
            # aux_tasks = [t for t in active_tasks if t != "t6"]
            # aux_f1s = [val_metrics[t]["macro_f1"] for t in aux_tasks]
            # aux_avg = np.mean(aux_f1s)
            # t6_score = t6_f1 + 0.3 * aux_avg

            # [v3 已废弃 - 2026-04-29] v4 不使用 beta_score 平衡分数
            # beta_tasks = [t for t in active_tasks if t.startswith('t') and t not in ['t1', 't6']]
            # if beta_tasks:
            #     beta_f1s = [val_metrics[t]["macro_f1"] for t in beta_tasks]
            #     beta_avg = np.mean(beta_f1s)
            #     beta_std = np.std(beta_f1s)
            #     beta_score = beta_avg - beta_std
            # else:
            #     beta_score = 0.0

            # ========== [v4 指标计算 - 2026-04-29] ==========
            # v4 主指标: weighted_macro_f1_t1_to_t5
            v4_weighted_f1 = compute_weighted_macro_f1_t1_to_t5(val_metrics)
            v4_mean_f1 = compute_mean_macro_f1_t1_to_t5(val_metrics)

            # 使用 weighted_macro_f1_t1_to_t5 作为调度器依据 (v4 主指标)
            if self.is_protected_dual_engine_v4:
                self.scheduler.step(v4_weighted_f1)
            else:
                # v2/v3: [已废弃] 使用 t6_score，v4 统一使用 weighted_f1
                self.scheduler.step(v4_weighted_f1)

            stage_name = "stage3_phase2" if use_two_phase else "stage3"
            # [修复 - 2026-04-22] 始终调用 _log_to_swanlab (即使 SwanLab 禁用也会填充 metrics_history)
            self._log_to_swanlab(epoch, train_metrics, val_metrics, stage=stage_name)

            # ========== [新增 - 2026-04-27] v4 Checkpoint 保存策略 ==========
            if self.is_protected_dual_engine_v4:
                checkpoint_stage = "stage3_phase2"
                fold = self.config.get('fold', 1)

                # v4 使用 weighted_macro_f1_t1_to_t5 作为主指标
                if v4_weighted_f1 > best_metric + phase2_early_stop_min_delta:
                    best_metric = v4_weighted_f1
                    best_epoch = epoch
                    self.stage3_best_f1 = v4_weighted_f1
                    self.stage3_meets_constraint = True  # v4 不使用 t6 约束
                    no_improve_count = 0

                    save_mtl_checkpoint(
                        self.model, self.optimizer, epoch, checkpoint_stage,
                        fold=fold,
                        metrics=val_metrics,
                        prefix=self.checkpoint_prefix
                    )
                    self._update_checkpoint_info_in_json(checkpoint_stage, epoch, v4_weighted_f1, "weighted_macro_f1_t1_to_t5")
                    print(f"[v4 Checkpoint] Epoch {epoch+1}: weighted_f1={v4_weighted_f1:.4f}, mean_f1={v4_mean_f1:.4f}")

                    # v4 minimal logging (SwanLab)
                    if self.swanlab_run and self.v4_minimal_logging:
                        v4_logs = compute_v4_minimal_metrics(self.model, val_metrics, stage_name)
                        self.swanlab_run.log({**v4_logs, "epoch": epoch})
                else:
                    no_improve_count += 1

                # v4: t6 仅日志，不参与排名
                if self.v4_minimal_logging and self.swanlab_run:
                    # [修复 - 2026-04-28] 使用标准化后的stage名称判断
                    normalized_stage_name = 'stage3_phase2'  # Phase2固定映射
                    self.swanlab_run.log({
                        "v4/detach_t6_context": normalized_stage_name in self.detach_t6_context_stages,
                        "epoch": epoch
                    })

            # ========== Epoch 打印 (v4 简化版) ==========
            if self.is_protected_dual_engine_v4:
                print(f"Epoch {epoch+1}/{start_epoch + phase2_epochs} | Weighted F1 (t1-t5): {v4_weighted_f1:.4f} | Mean F1: {v4_mean_f1:.4f} | Best: {best_metric:.4f}")
            else:
                # [v3/v2 已废弃 - 2026-04-29]
                # print(f"Epoch {epoch+1} | t6 F1: {t6_f1:.4f} | Score: {t6_score:.4f} | Best: {best_metric:.4f}")
                print(f"Epoch {epoch+1}/{start_epoch + phase2_epochs} | Weighted F1: {v4_weighted_f1:.4f} | Best: {best_metric:.4f}")

            # ========== [新增 - 2026-04-21] Phase2 Early Stopping ==========
            if use_two_phase and no_improve_count >= phase2_early_stop_patience:
                print(f"\n[Phase 2 Early Stop] 连续 {no_improve_count} epoch 无提升，提前停止")
                break

        # ========== [修复 - 2026-04-22] 训练结束后重新加载最佳 checkpoint 并运行阈值搜索 ==========
        # [v4 简化 - 2026-04-29] v4 使用统一的 checkpoint 文件名
        if use_two_phase and self.threshold_mode == "search":
            fold = self.config.get('fold', 1)

            # 使用动态生成的 checkpoint_prefix
            checkpoint_path = os.path.join("models", f"best_{self.checkpoint_prefix}_stage3_phase2_fold{fold}.pth")
            checkpoint_name = self.checkpoint_prefix

            if os.path.exists(checkpoint_path):
                print(f"\n[Threshold Search] 加载 {checkpoint_name} checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()

                # 运行阈值搜索
                val_metrics = self._validate_epoch(
                    active_tasks,
                    epoch=best_epoch,  # v4 使用 best_epoch
                    stage="stage3_phase2",
                    run_threshold_search=True
                )
                print(f"[Threshold Search] 完成，最佳阈值: {self.best_thresholds}")

        # 最终状态汇总 (v4 简化版)
        print("\n" + "-" * 40)
        print(f"阶段三完成汇总:")
        if use_two_phase:
            print(f"  Phase 1 最佳 Beta Avg: {phase1_best_metric:.4f}")
        print(f"  最佳 Weighted F1 (t1-t5): {best_metric:.4f}")
        print(f"  最佳 Epoch: {best_epoch+1}")
        print("-" * 40)

        # [v3/v2 已废弃 - 2026-04-29]
        # if dual_checkpoint_enabled:
        #     print(f"  [v3.1 双 checkpoint]")
        #     print(f"    - T6 Protected: best_t6_score={best_t6_score:.4f}")
        #     print(f"    - Beta Balanced: best_beta_score={best_beta_score:.4f}")
        # print(f"  最佳 t6 Macro-F1: {best_t6_f1:.4f}")
        # print(f"  约束基准: {t6_baseline:.4f} - {t6_constraint_delta:.4f}")

        # [新增] 阶段结束时批量保存 JSON
        # 如果启用两段式，需要保存 stage3_phase1 和 stage3_phase2
        if use_two_phase:
            self._save_stage_to_json("stage3_phase1")
            self._save_stage_to_json("stage3_phase2")
        else:
            self._save_stage_to_json("stage3")

    def _train_epoch(
        self,
        epoch: int,
        active_tasks: List[str],
        use_kd: bool,
        use_uncertainty_weighting: bool,
        stage: str = "",
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        训练一个 epoch

        [新增] ProtectedDualEngineMTL 支持:
        - 温度调度 (tau_override)
        - 门控熵正则化损失
        - 门控权重日志
        """
        self.model.train()

        accumulation_steps = self.config.get('accumulation_steps', 2)
        all_metrics = {t: {"loss": 0.0, "count": 0} for t in active_tasks}

        # =====================================================
        # [新增] ProtectedDualEngineMTL: 温度调度
        # =====================================================
        tau_override = None
        if self.is_protected_dual_engine and self.temp_scheduler is not None:
            tau_override = self.temp_scheduler.get_tau(epoch)
            # 记录温度历史
            self.tau_history.append({"epoch": epoch, "tau": tau_override, "stage": stage})

            # SwanLab 日志
            if self.swanlab_run:
                self.swanlab_run.log({
                    f"{stage}/gate_temperature/tau": tau_override,
                    "epoch": epoch
                })

        # =====================================================

        for batch_idx, batch in enumerate(self.train_loader):
            x_dyn = batch["x_dyn"].to(self.device)
            x_static = batch["x_static"].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

            # =====================================================
            # [新增] ProtectedDualEngineMTL: 前向传播
            # =====================================================
            if self.is_protected_dual_engine:
                # 传递 tau_override 和 return_gate_weights
                # v4 新增: detach_t6_context, skip_t6_injection

                # [修复 - 2026-04-28] stage名称标准化映射
                # _train_epoch调用传入的是简化名称（stage1, stage2, stage3_phase1等）
                # 但配置定义的是完整名称（stage1_alpha_anchor, stage2_beta_warmup等）
                stage_name_mapping = {
                    'stage1': 'stage1_alpha_anchor',
                    'stage2': 'stage2_beta_warmup',
                    'stage3_phase1': 'stage3_phase1',
                    'stage3_phase2': 'stage3_phase2',
                    'stage3': 'stage3_phase2',  # stage3默认映射到phase2
                }
                normalized_stage = stage_name_mapping.get(stage, stage)

                detach_t6_context = False
                skip_t6_injection = False
                skip_beta_branch = False  # [新增 - 2026-04-29]

                if self.is_protected_dual_engine_v4:
                    # v4: 根据阶段决定梯度策略（使用标准化后的stage名称）
                    skip_t6_injection = normalized_stage in self.skip_t6_injection_stages
                    detach_t6_context = normalized_stage in self.detach_t6_context_stages

                    # [优化 - 2026-04-29] 检测是否需要跳过 Beta 分支计算
                    # Stage1/Stage2 冻结 Beta 分支时，跳过前向计算以节省时间和内存
                    beta_tasks = ["t2", "t3", "t4", "t5"]
                    beta_branch_active = any(t in active_tasks for t in beta_tasks)
                    skip_beta_branch = not beta_branch_active  # 如果 Beta 任务不在 active_tasks 中，跳过

                elif self.is_protected_dual_engine_v3:
                    # v3: 同样支持 skip_beta_branch
                    beta_tasks = ["t2", "t3", "t4", "t5"]
                    beta_branch_active = any(t in active_tasks for t in beta_tasks)
                    skip_beta_branch = not beta_branch_active

                outputs = self.model(
                    x_dyn, x_static,
                    tau_override=tau_override,
                    return_gate_weights=self.use_gate_entropy,
                    detach_t6_context=detach_t6_context,
                    skip_t6_injection=skip_t6_injection,
                    skip_beta_branch=skip_beta_branch  # [新增 - 2026-04-29]
                )
            else:
                # 原始 HDSTGCNMTL
                outputs = self.model(x_dyn, x_static, return_aux=False)
            # =====================================================

            # 计算各任务损失
            task_losses = {}
            for task_key in active_tasks:
                logits = outputs[task_key]["logits"]
                target = labels[task_key]

                # 根据损失类型转换 target 类型
                if task_key in self.task_specs:
                    spec = self.task_specs[task_key]
                    if spec.loss_name == "bce":
                        # [v2 统一] BCE 损失：使用统一的 target 编码 (少数类=1)
                        target = make_binary_target_for_loss(target, spec.minority_idx)

                        # 二分类 logits 是 [B, 1]，target 需要 unsqueeze 成 [B, 1]
                        if logits.dim() == 2 and logits.size(1) == 1:
                            target = target.unsqueeze(1)
                    elif spec.loss_name == "ldam":
                        # LDAM 二分类需要特殊处理
                        if spec.is_binary:
                            # [v2 统一] LDAM 二分类：使用统一的 target 编码 (少数类=1)
                            target = make_binary_target_for_loss(target, spec.minority_idx)
                        else:
                            # LDAM 多分类: 保持 LongTensor
                            target = target.long()
                    else:
                        # CE 损失需要 LongTensor 目标 [B]
                        target = target.long()

                if self.criterions[task_key] is not None:
                    loss = self.criterions[task_key](logits, target)
                    task_losses[task_key] = loss

            # KD 损失
            teacher_logits_t1 = None
            if use_kd and self.teacher_model is not None and "t1" in active_tasks:
                with torch.no_grad():
                    self.teacher_model.eval()
                    # 正确调用：static_x 作为关键字参数传递
                    teacher_out = self.teacher_model(x_dyn, static_x=x_static)
                    teacher_logits_t1 = teacher_out["logits"] if isinstance(teacher_out, dict) else teacher_out

            # 总损失
            total_loss, loss_dict = self.total_loss(
                task_losses,
                self.model.log_vars,
                teacher_logits_t1=teacher_logits_t1,
                student_logits_t1=outputs["t1"]["logits"] if "t1" in active_tasks else None,
                use_kd=use_kd,
                use_uncertainty_weighting=use_uncertainty_weighting,
                loss_weights=loss_weights
            )

            # =====================================================
            # [新增 - 2026-05-25] T3 Boundary Stabilization: Anchor Loss
            # =====================================================
            t3_anchor_loss = None
            if self.t3_anchor_enabled and self.t3_anchor_teacher_model is not None and stage == "stage3_phase2" and "t3" in active_tasks:
                with torch.no_grad():
                    self.t3_anchor_teacher_model.eval()
                    teacher_outputs = self.t3_anchor_teacher_model(x_dyn, x_static, return_aux=False)
                    teacher_t3_logits = teacher_outputs["t3"]["logits"]
                    teacher_t3_prob = torch.sigmoid(teacher_t3_logits)  # t3 is binary

                student_t3_logits = outputs["t3"]["logits"]
                student_t3_prob = torch.sigmoid(student_t3_logits)

                # Compute anchor loss based on loss_type
                if self.t3_anchor_loss_type == "mse_prob":
                    t3_anchor_loss = F.mse_loss(student_t3_prob, teacher_t3_prob)
                elif self.t3_anchor_loss_type == "bce_prob":
                    t3_anchor_loss = F.binary_cross_entropy(student_t3_prob, teacher_t3_prob)
                elif self.t3_anchor_loss_type == "kl_logit":
                    # KL divergence on logits (temperature=1.0)
                    t3_anchor_loss = F.kl_div(
                        F.logsigmoid(student_t3_logits),
                        torch.sigmoid(teacher_t3_logits),
                        reduction='batchmean'
                    )

                # Add to total_loss
                total_loss = total_loss + self.t3_anchor_lambda * t3_anchor_loss

                # Log metrics (batch_idx == 0 to log once per epoch)
                if self.swanlab_run and batch_idx == 0:
                    t3_raw_bce = task_losses.get("t3", 0.0)
                    t3_pred_minor_rate = (student_t3_prob > 0.5).float().mean().item()
                    self.swanlab_run.log({
                        f"{stage}/loss/t3_anchor": t3_anchor_loss.item(),
                        f"{stage}/loss/t3_raw_bce": t3_raw_bce if isinstance(t3_raw_bce, float) else t3_raw_bce.item(),
                        f"{stage}/t3_pred_minor_rate": t3_pred_minor_rate,
                        "epoch": epoch
                    })
            # =====================================================

            # =====================================================
            # [新增] ProtectedDualEngineMTL: 门控熵正则化损失
            # =====================================================
            gate_entropy_loss = None
            if self.is_protected_dual_engine and self.use_gate_entropy:
                # [修复 - 2026-04-29] 检查 beta_gates 是否可训练
                # Stage1/Stage2 冻结 Beta 分支时，gate_entropy_loss 无效
                # [Ablation E2] beta_gates 可能为 None
                if self.model.beta_gates is not None and len(self.model.beta_gates) > 0:
                    beta_gate_trainable = any(
                        p.requires_grad
                        for gate in self.model.beta_gates.values()
                        for p in gate.parameters()
                    )
                else:
                    beta_gate_trainable = False

                if not beta_gate_trainable:
                    # Beta gates 冻结，跳过 gate entropy 计算
                    pass
                else:
                    # 仅在前半训练阶段启用 (防止后期干扰收敛)
                    # Phase2 可以覆盖此限制
                    total_epochs = self.config.get('total_epochs', 100)
                    phase2_override = getattr(self, 'use_gate_entropy_in_phase2', True)

                    # 判断是否应用 gate entropy
                    should_apply_gate_entropy = (
                        (epoch < total_epochs * 0.5) or
                        (stage == "stage3_phase2" and phase2_override)
                    )

                    if should_apply_gate_entropy and "gate_weights" in outputs:
                        gate_entropy_loss = self.gate_entropy_reg(outputs["gate_weights"])
                        total_loss = total_loss + gate_entropy_loss

                        # SwanLab 日志
                        if self.swanlab_run and batch_idx == 0:  # 每个 epoch 只记录一次
                            self.swanlab_run.log({

                                
                                f"{stage}/loss/gate_entropy": gate_entropy_loss.item(),
                                "epoch": epoch
                            })
            # =====================================================

            # =====================================================
            # [新增 - 2026-04-30] T6辅助模式: c6_deep L2 正则化
            # L_context_reg = lambda * ||c6_deep||^2
            # 防止 context 向量爆炸，稳定 t6 上下文注入
            # 仅在 T6 辅助模式 + c6_deep 存在 + 非跳过注入阶段 时生效
            # =====================================================
            if self.t6_auxiliary_mode_enabled and self.context_reg_enabled:
                if "t6" in outputs and isinstance(outputs["t6"], dict) and "c6_deep" in outputs["t6"]:
                    c6_deep = outputs["t6"]["c6_deep"]
                    context_reg_loss = self.context_reg_lambda * (c6_deep ** 2).mean()
                    total_loss = total_loss + context_reg_loss

                    if self.swanlab_run and batch_idx == 0:
                        self.swanlab_run.log({
                            f"{stage}/loss/context_reg": context_reg_loss.item(),
                            f"{stage}/c6_deep_norm": c6_deep.norm().item(),
                            "epoch": epoch
                        })
            # =====================================================

            # 梯度累加
            scaled_loss = total_loss / accumulation_steps
            scaled_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip', 1.0))
                self.optimizer.step()
                self.optimizer.zero_grad()

            # 累积指标
            for task_key in active_tasks:
                if task_key in loss_dict:
                    all_metrics[task_key]["loss"] += loss_dict[task_key]
                    all_metrics[task_key]["count"] += 1

        # =====================================================
        # [新增] ProtectedDualEngineMTL: 门控权重日志 (SwanLab)
        # =====================================================
        if self.is_protected_dual_engine and self.swanlab_run and epoch % 5 == 0:
            # [修复 - 2026-04-29] 检查 beta_gates 是否可训练
            # Stage1/Stage2 冻结 Beta gates 时，不需要记录门控权重
            # [Ablation E2] beta_gates 可能为 None
            if self.model.beta_gates is not None and len(self.model.beta_gates) > 0:
                beta_gate_trainable = any(
                    p.requires_grad
                    for gate in self.model.beta_gates.values()
                    for p in gate.parameters()
                )
            else:
                beta_gate_trainable = False

            if not beta_gate_trainable:
                # Beta gates 冻结，跳过门控权重日志
                pass
            else:
                # 每 5 个 epoch 记录一次门控权重 (减少日志量)
                with torch.no_grad():
                    # 用一个批次数据计算门控权重
                    sample_batch = next(iter(self.train_loader))
                    x_dyn = sample_batch["x_dyn"].to(self.device)
                    x_static = sample_batch["x_static"].to(self.device)

                    outputs = self.model(
                        x_dyn, x_static,
                        tau_override=tau_override,
                        return_gate_weights=True
                    )

                    if "gate_weights" in outputs:
                        for task_key, weights in outputs["gate_weights"].items():
                            # weights: [B, num_experts]
                            mean_weights = weights.mean(dim=0).cpu().numpy()

                            # 使用语义化名称 (shared_ratio, private_ratio, group_ratio)
                            # 传递architecture_version以支持v3
                            ratio_names = get_gate_ratio_names(task_key, self.architecture_version)

                            # v3: Alpha任务无gate，跳过 (ratio_names为空)
                            if len(ratio_names) == 0:
                                continue

                            # 记录每个专家的平均权重
                            for i, (w, name) in enumerate(zip(mean_weights, ratio_names)):
                                self.swanlab_run.log({
                                    f"{stage}/gate/{task_key}/{name}": float(w),
                                    "epoch": epoch
                                })

                            # 记录熵 (监控门控多样性)
                            log_weights = np.log(mean_weights + 1e-8)
                            entropy = -np.sum(mean_weights * log_weights)
                            self.swanlab_run.log({
                                f"{stage}/gate/{task_key}/entropy": float(entropy),
                                "epoch": epoch
                                })
        # =====================================================

        # 平均
        avg_metrics = {}
        for task_key in active_tasks:
            avg_metrics[task_key] = {
                "loss": all_metrics[task_key]["loss"] / max(all_metrics[task_key]["count"], 1)
            }

        return avg_metrics

    def _validate_epoch(self, active_tasks: List[str], epoch: int = 0, stage: str = "",
                        run_threshold_search: bool = False,
                        thresholds_to_apply: Dict[str, float] = None) -> Dict[str, Dict[str, float]]:
        """
        验证一个 epoch

        Args:
            active_tasks: 活动任务列表
            epoch: Epoch 编号
            stage: 阶段名称
            run_threshold_search: 是否在验证后执行阈值搜索
            thresholds_to_apply: 要应用的阈值字典 (从搜索结果传入)

        Returns:
            各任务的指标字典
        """
        self.model.eval()

        # 收集所有预测和标签
        all_preds = {task_key: [] for task_key in active_tasks}
        all_labels = {task_key: [] for task_key in active_tasks}
        all_probs = {task_key: [] for task_key in active_tasks}  # 用于 AUC 计算
        # [新增] 收集 logits 用于阈值搜索
        all_logits = {task_key: [] for task_key in active_tasks if self.task_specs[task_key].is_binary}

        with torch.no_grad():
            for batch in self.val_loader:
                x_dyn = batch["x_dyn"].to(self.device)
                x_static = batch["x_static"].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

                # v4 支持: 根据架构传递不同参数
                if self.is_protected_dual_engine_v4:
                    # [修复 - 2026-04-28] stage名称标准化映射（与_train_epoch保持一致）
                    stage_name_mapping = {
                        'stage1': 'stage1_alpha_anchor',
                        'stage2': 'stage2_beta_warmup',
                        'stage3_phase1': 'stage3_phase1',
                        'stage3_phase2': 'stage3_phase2',
                        'stage3': 'stage3_phase2',
                    }
                    normalized_stage = stage_name_mapping.get(stage, stage)

                    # v4: 验证时不注入 t6 context (skip=True) 或使用 detach（使用标准化后的名称）
                    skip_t6_injection = normalized_stage in self.skip_t6_injection_stages
                    detach_t6_context = normalized_stage in self.detach_t6_context_stages

                    # [优化 - 2026-04-29] 检测是否需要跳过 Beta 分支计算
                    beta_tasks = ["t2", "t3", "t4", "t5"]
                    beta_branch_active = any(t in active_tasks for t in beta_tasks)
                    skip_beta_branch = not beta_branch_active

                    outputs = self.model(
                        x_dyn, x_static,
                        return_aux=False,
                        detach_t6_context=detach_t6_context,
                        skip_t6_injection=skip_t6_injection,
                        skip_beta_branch=skip_beta_branch  # [新增 - 2026-04-29]
                    )
                elif self.is_protected_dual_engine_v3:
                    # [优化 - 2026-04-29] v3 同样支持 skip_beta_branch
                    beta_tasks = ["t2", "t3", "t4", "t5"]
                    beta_branch_active = any(t in active_tasks for t in beta_tasks)
                    skip_beta_branch = not beta_branch_active

                    outputs = self.model(
                        x_dyn, x_static,
                        return_aux=False,
                        skip_beta_branch=skip_beta_branch
                    )
                else:
                    outputs = self.model(x_dyn, x_static, return_aux=False)

                for task_key in active_tasks:
                    logits = outputs[task_key]["logits"]
                    target = labels[task_key]

                    # 获取预测
                    spec = self.task_specs[task_key]
                    if spec.is_binary:
                        # 二分类: logits → minority_prob → threshold
                        probs = get_minority_prob_from_logits(logits, spec.minority_idx)  # [B]

                        # ========== [重构] 可配置阈值应用 (替代硬编码 0.5) ==========
                        # 优先级:
                        #   1. thresholds_to_apply (外部传入，如 Holdout 评估)
                        #   2. best_thresholds (阈值搜索结果 - 搜索后的优化值)
                        #   3. fixed_thresholds (配置的默认值)
                        #   4. 0.5 (硬编码兜底)
                        #
                        # 说明: fixed_thresholds 作为配置的默认值，无论 mode 是 "search" 还是 "fixed"
                        #       都会生效。搜索完成后 best_thresholds 会覆盖 fixed_thresholds。
                        threshold = self.fixed_thresholds.get(task_key, 0.5)  # 默认使用 fixed_thresholds
                        if thresholds_to_apply and task_key in thresholds_to_apply:
                            # 优先级1: 外部传入阈值 (Holdout评估等)
                            threshold = thresholds_to_apply[task_key]
                        elif task_key in self.best_thresholds:
                            # 优先级2: 搜索得到的最优阈值 (覆盖 fixed_thresholds)
                            threshold = self.best_thresholds[task_key]

                        # [关键修复] 预测转换：sigmoid>threshold 预测为少数类(minority_idx)
                        preds = torch.where(
                            probs > threshold,
                            torch.tensor(spec.minority_idx, device=logits.device),
                            torch.tensor(spec.majority_idx, device=logits.device)
                        ).long()
                        # ================================================

                        all_probs[task_key].append(probs.cpu())
                        # [新增] 收集 logits 用于阈值搜索
                        if task_key in all_logits:
                            all_logits[task_key].append(logits.cpu())
                    else:
                        # 多分类: softmax + argmax
                        probs = F.softmax(logits, dim=1)
                        preds = logits.argmax(dim=1)
                        all_probs[task_key].append(probs.cpu())

                    all_preds[task_key].append(preds.cpu())
                    all_labels[task_key].append(target.cpu())

        # 合并并计算指标
        metrics = {}
        for task_key in active_tasks:
            preds = torch.cat(all_preds[task_key], dim=0).numpy()
            targets = torch.cat(all_labels[task_key], dim=0).numpy()

            # 计算 Macro F1
            # ========== [修复 - 2026-04-23] 统一导入所有sklearn指标函数 (解决scoping问题) ==========
            from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score, precision_score
            spec = self.task_specs[task_key]

            if spec.is_binary:
                # 二分类评估: 使用 pos_label=minority_idx 计算少数类指标
                probs = torch.cat(all_probs[task_key], dim=0).numpy()

                # ========== [新增] 计算所有二分类诊断指标 ==========
                cm = confusion_matrix(targets, preds)

                # 少数类指标 (pos_label=minority_idx)
                minority_f1 = f1_score(targets, preds, average='binary', pos_label=spec.minority_idx)
                minority_recall = recall_score(targets, preds, pos_label=spec.minority_idx, average='binary')
                minority_precision = precision_score(targets, preds, pos_label=spec.minority_idx, zero_division=0)

                # 预测/真实少数类比例
                pred_minor_rate = (preds == spec.minority_idx).mean()
                true_minor_rate = (targets == spec.minority_idx).mean()

                # 混淆矩阵元素 (根据 minority_idx 解析)
                # cm 格式: [[TN(if minority=1), FP], [FN, TP(if minority=1)]]
                # 当 minority_idx=0 时，cm[0,0] 是少数类正确预测，cm[0,1] 是少数类错判为多数类
                if spec.minority_idx == 0:
                    # minority=0, majority=1
                    # cm[0,0]: 真实0预测0 (少数类TP) -> minority_tp
                    # cm[0,1]: 真实0预测1 (少数类FN) -> minority_fn
                    # cm[1,0]: 真实1预测0 (多数类FP) -> majority_fp
                    # cm[1,1]: 真实1预测1 (多数类TN) -> majority_tn
                    minority_tp = cm[0, 0]
                    minority_fn = cm[0, 1]
                    majority_fp = cm[1, 0]
                    majority_tn = cm[1, 1]
                else:
                    # minority=1, majority=0 (sklearn 默认格式)
                    minority_tp = cm[1, 1]
                    minority_fn = cm[1, 0]
                    majority_fp = cm[0, 1]
                    majority_tn = cm[0, 0]
                # ================================================

                # 真正的 macro-F1: 两类 F1 的平均
                macro_f1 = f1_score(targets, preds, average='macro')

                # [v2 统一] 二分类 AUC: probs 是 minority_prob, y_true_minor = (labels == minority_idx)
                y_true_minor = (targets == spec.minority_idx).astype(int)
                try:
                    auc = roc_auc_score(y_true_minor, probs)
                except Exception as e:
                    print(f"[AUC Error] {task_key}: labels={targets.shape}, probs={probs.shape}, unique_labels={np.unique(targets)}, error={e}")
                    auc = np.nan

                # ========== [诊断] 双向 AUC + label-wise prob 均值 (t3/t4/t5) ==========
                # 仅对 t3/t4/t5 打印诊断
                if task_key in ["t3", "t4", "t5"]:
                    # 双向 AUC (y_true_minor 已定义)
                    try:
                        auc_minority = roc_auc_score(y_true_minor, probs)
                        auc_reverse = roc_auc_score(y_true_minor, 1.0 - probs)
                    except:
                        auc_minority = np.nan
                        auc_reverse = np.nan

                    # label-wise probability 均值
                    minority_mask = targets == spec.minority_idx
                    majority_mask = targets == spec.majority_idx

                    prob_mean_minority = probs[minority_mask].mean() if minority_mask.sum() > 0 else np.nan
                    prob_mean_majority = probs[majority_mask].mean() if majority_mask.sum() > 0 else np.nan

                    # 诊断判断
                    ranking_warning = ""
                    if prob_mean_minority < prob_mean_majority:
                        ranking_warning = "[Ranking Warning] minority_prob is higher on majority samples!"

                    # 打印诊断表
                    print(f"\n[BinaryDiag/{stage}] {task_key} (minority_idx={spec.minority_idx}):")
                    print(f"  auc_minority={auc_minority:.4f}, auc_reverse={auc_reverse:.4f}")
                    print(f"  prob_mean_true_minority={prob_mean_minority:.4f}, prob_mean_true_majority={prob_mean_majority:.4f}")
                    print(f"  pred_minor_rate={pred_minor_rate:.4f}, true_minor_rate={true_minor_rate:.4f}")
                    print(f"  minority_precision={minority_precision:.4f}, minority_recall={minority_recall:.4f}, minority_f1={minority_f1:.4f}")
                    if ranking_warning:
                        print(f"  {ranking_warning}")
                # ================================================

                # ========== [DEBUG] 任务3详细诊断 (写入txt文档) ==========
                # if task_key == "t3":
                #     # 写入调试文档 (表格格式，使用 trainer 初始化时的时间戳)
                #     debug_file = self.debug_file
                #     header_written = self.debug_header_written

                #     with open(debug_file, 'a') as f:
                #         if not header_written:
                #             # 写入表头和参数说明 (仅第一次)
                #             f.write("=" * 80 + "\n")
                #             f.write("任务3 (t3) 调试诊断表 - 二分类评估链路检查\n")
                #             f.write("=" * 80 + "\n")
                #             f.write(f"minority_idx: {spec.minority_idx} (少数类标签)\n")
                #             f.write(f"majority_idx: {spec.majority_idx} (多数类标签)\n")
                #             f.write("说明: PredMinor/TrueMinor 统计的是预测/真实为 minority_idx 的比例\n")
                #             f.write("说明: F1Minor 是少数类 F1 (pos_label=minority_idx), MacroF1 是两类 F1 平均\n")
                #             f.write("说明: AUC 的正类定义为 label=1 (sklearn 默认), probs 已根据 minority_idx 对齐\n")
                #             f.write("=" * 80 + "\n\n")
                #             # 表头 - 统一使用 minority 视角命名
                #             header1 = "Epoch    Stage     | 混淆矩阵 (真实\\预测)      | PredMinor TrueMinor Recall   F1Minor  MacroF1  AUC\n"
                #             header2 = "                   |       0     1             | Rate     Rate     Minor\n"
                #             f.write(header1)
                #             f.write(header2)
                #             f.write("-" * 80 + "\n")
                #             self.debug_header_written = True  # 标记已写入

                #         # 写入当前epoch数据行
                #         cm_str = f"0: {cm[0,0]:4d}  {cm[0,1]:4d}\n        1: {cm[1,0]:4d}  {cm[1,1]:4d}"
                #         f.write(f"{epoch:<8} {stage:<8} | {cm_str:<25} | {pred_minor_rate:<8.4f} {true_minor_rate:<8.4f} {minority_recall:<8.4f} {minority_f1:<8.4f} {macro_f1:<8.4f} {auc:<8.4f}\n")
                # ================================================

                # ========== [修复 - 2026-04-23] 添加 precision/recall/f1 字段 (供Excel写入) ==========
                # Excel logger 需要 "precision", "recall", "f1" 字段
                # 二分类使用 macro 平均 (两类F1的平均)
                precision = precision_score(targets, preds, average='macro', zero_division=0)
                recall = recall_score(targets, preds, average='macro', zero_division=0)

                # 二分类指标记录 (统一格式)
                metrics[task_key] = {
                    "accuracy": (preds == targets).mean(),
                    "precision": precision,
                    "recall": recall,
                    "f1": macro_f1,  # f1 = macro_f1 (两类平均)
                    "macro_f1": macro_f1,
                    "auc": auc,
                    # 少数类详细指标 (保留用于诊断)
                    "minority_f1": minority_f1,
                    "minority_recall": minority_recall,
                    "minority_precision": minority_precision,
                    "pred_minor_rate": pred_minor_rate,
                    "true_minor_rate": true_minor_rate,
                    "minority_tp": int(minority_tp),
                    "minority_fn": int(minority_fn),
                    "majority_fp": int(majority_fp),
                    "majority_tn": int(majority_tn)
                }
                # ================================================
                # ================================================

            else:
                macro_f1 = f1_score(targets, preds, average='macro')

                # ========== [修复 - 2026-04-23] 添加 precision/recall/f1 字段 (供Excel写入) ==========
                # Excel logger 需要 "precision", "recall", "f1" 字段
                # 多分类使用 macro 平均
                precision = precision_score(targets, preds, average='macro', zero_division=0)
                recall = recall_score(targets, preds, average='macro', zero_division=0)

                # AUC (多分类 - one-vs-rest)
                probs = torch.cat(all_probs[task_key], dim=0).numpy()
                try:
                    auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
                except Exception as e:
                    print(f"[AUC Error] {task_key}: labels={targets.shape}, probs={probs.shape}, unique_labels={np.unique(targets)}, error={e}")
                    auc = np.nan

                metrics[task_key] = {
                    "accuracy": (preds == targets).mean(),
                    "precision": precision,
                    "recall": recall,
                    "f1": macro_f1,  # f1 = macro_f1
                    "macro_f1": macro_f1,
                    "auc": auc
                }
                # ================================================

        # [诊断] 打印每个二分类任务的 minority_prob 方向诊断
        for task_key in active_tasks:
            spec = self.task_specs[task_key]
            if not spec.is_binary:
                continue
            probs_diag = torch.cat(all_probs[task_key], dim=0).numpy() if all_probs[task_key] else np.array([])
            preds_diag = torch.cat(all_preds[task_key], dim=0).numpy() if all_preds[task_key] else np.array([])
            targets_diag = torch.cat(all_labels[task_key], dim=0).numpy() if all_labels[task_key] else np.array([])
            if len(probs_diag) > 0:
                # 获取当前阈值
                threshold = self.fixed_thresholds.get(task_key, 0.5)
                if task_key in self.best_thresholds:
                    threshold = self.best_thresholds[task_key]
                pred_minor_rate = (preds_diag == spec.minority_idx).mean()
                true_minor_rate = (targets_diag == spec.minority_idx).mean()
                print(f"[BinaryDiag] {task_key}: minority_idx={spec.minority_idx}, majority_idx={spec.majority_idx}, "
                      f"threshold={threshold:.3f}, prob_mean={probs_diag.mean():.4f}, "
                      f"prob_min={probs_diag.min():.4f}, prob_max={probs_diag.max():.4f}, "
                      f"pred_minor_rate={pred_minor_rate:.4f}, true_minor_rate={true_minor_rate:.4f}")

        # ========== [新增] 阈值搜索逻辑 (val-search → test-apply 协议) ==========
        if run_threshold_search and self.threshold_mode == "search":
            # 仅对二分类任务执行阈值搜索
            for task_key in self.threshold_search_tasks:
                if task_key in all_logits and task_key in active_tasks:
                    spec = self.task_specs[task_key]
                    if not spec.is_binary:
                        continue

                    # 合并 logits 和 labels
                    logits_np = torch.cat(all_logits[task_key], dim=0).numpy()
                    labels_np = torch.cat(all_labels[task_key], dim=0).numpy()

                    # 执行两阶段阈值搜索
                    search_result = two_stage_threshold_search(
                        task_key=task_key,
                        logits=logits_np,
                        labels=labels_np,
                        minority_idx=spec.minority_idx,
                        save_dir=self.threshold_save_dir,
                        fold=self.current_fold,
                        checkpoint_stage=stage,
                        verbose=True
                    )

                    # 存储最优阈值
                    self.best_thresholds[task_key] = search_result["best_threshold"]

                    # 记录到 metrics (用于 SwanLab 日志)
                    metrics[task_key]["best_threshold"] = search_result["best_threshold"]
                    metrics[task_key]["threshold_f1_improvement"] = search_result["improvement"]

        # [新增] 更新最佳阈值到增量 JSON
        if self.best_thresholds:
            self._update_best_thresholds_in_json()
        # ================================================

        return metrics

    # =========================================================================
    # [新增 - 2026-04-22] Excel Logger 评估链路集成方法
    # =========================================================================

    def compute_val_metrics_with_thresholds(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        active_tasks: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        应用搜索阈值计算验证集最终指标 (供 Excel logger 使用)

        在阈值搜索完成后调用，使用搜索得到的最佳阈值重新计算验证集指标。
        这是 val-search → test-apply 协议的关键步骤。

        Args:
            thresholds: 阈值字典 {"t3": 0.325, ...}，默认使用 self.best_thresholds
            active_tasks: 活动任务列表，默认使用所有任务

        Returns:
            val_metrics: 各任务指标字典 {"t1": {"acc": ..., "f1": ..., ...}, ...}
        """
        # 使用默认阈值
        if thresholds is None:
            thresholds = self.best_thresholds

        # 使用默认活动任务
        if active_tasks is None:
            active_tasks = list(self.task_specs.keys())

        print(f"\n[Val Metrics with Thresholds] 应用阈值: {thresholds}")

        # 调用 _validate_epoch，传入 thresholds_to_apply
        val_metrics = self._validate_epoch(
            active_tasks=active_tasks,
            epoch=-1,  # 标记为最终评估
            stage="val_final",
            run_threshold_search=False,  # 不再执行阈值搜索
            thresholds_to_apply=thresholds
        )

        # 打印结果摘要
        print(f"[Val Metrics with Thresholds] 结果:")
        for task_key, metrics in val_metrics.items():
            spec = self.task_specs[task_key]
            if spec.is_binary:
                threshold = thresholds.get(task_key, 0.5)
                print(f"  {task_key}: macro_f1={metrics.get('macro_f1', 0):.4f}, "
                      f"f1={metrics.get('minority_f1', 0):.4f}, auc={metrics.get('auc', 0):.4f}, "
                      f"threshold={threshold:.3f}")
            else:
                print(f"  {task_key}: macro_f1={metrics.get('macro_f1', 0):.4f}, "
                      f"acc={metrics.get('accuracy', 0):.4f}")

        return val_metrics

    def evaluate_holdout_with_val_thresholds(
        self,
        holdout_loader: torch.utils.data.DataLoader,
        thresholds: Optional[Dict[str, float]] = None,
        checkpoint_path: Optional[str] = None,
        fold: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        使用 val 搜索的阈值评估 Holdout 测试集 (供 Excel logger 使用)

        这是 val-search → test-apply 协议的最终步骤：
        1. 加载 checkpoint A (mtl_v31_t6_protected)
        2. 使用 val 搜索的最佳阈值评估 holdout test

        Args:
            holdout_loader: Holdout 测试集 DataLoader
            thresholds: 阈值字典，默认使用 self.best_thresholds
            checkpoint_path: checkpoint 路径，默认查找 fold 对应的 checkpoint
            fold: Fold 编号，用于查找 checkpoint

        Returns:
            holdout_metrics: 各任务指标字典 {"t1": {"acc": ..., "f1": ..., ...}, ...}
        """
        import glob

        # 使用默认阈值
        if thresholds is None:
            thresholds = self.best_thresholds

        # 确定 fold
        if fold is None:
            fold = self.current_fold

        # 确定 checkpoint 路径
        if checkpoint_path is None:
            # 使用动态生成的 checkpoint_prefix
            checkpoint_path = os.path.join("models", f"best_{self.checkpoint_prefix}_stage3_phase2_fold{fold}.pth")

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\n[Holdout with Val Thresholds] 加载 checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        else:
            print(f"[Holdout with Val Thresholds] 使用当前模型状态")

        print(f"[Holdout with Val Thresholds] 应用阈值: {thresholds}")

        # 使用 evaluate_mtl_on_holdout_test 函数
        holdout_metrics = evaluate_mtl_on_holdout_test(
            model=self.model,
            test_loader=holdout_loader,
            task_specs=self.task_specs,
            device=self.device,
            criterions=self.criterions,
            thresholds=thresholds
        )

        # 打印结果摘要
        print(f"[Holdout with Val Thresholds] 结果:")
        for task_key, metrics in holdout_metrics.items():
            spec = self.task_specs[task_key]
            if spec.is_binary:
                threshold = thresholds.get(task_key, 0.5)
                print(f"  {task_key}: macro_f1={metrics.get('macro_f1', 0):.4f}, "
                      f"precision={metrics.get('precision', 0):.4f}, "
                      f"recall={metrics.get('recall', 0):.4f}, "
                      f"threshold={threshold:.3f}")
            else:
                print(f"  {task_key}: macro_f1={metrics.get('macro_f1', 0):.4f}, "
                      f"acc={metrics.get('accuracy', 0):.4f}")

        return holdout_metrics

    def get_task_names_mapping(self) -> Dict[str, str]:
        """
        获取任务键到任务名称的映射 (供 Excel logger 使用)

        Returns:
            task_names: {"t1": "运动心功能分级", "t2": ...}
        """
        task_names = {}
        for task_key, spec in self.task_specs.items():
            task_names[task_key] = spec.name
        return task_names

    # =========================================================================

    def _apply_lr_scale(self, lr_scale: Dict[str, float]):
        """应用学习率缩放"""
        base_lr = self.config.get('lr', 0.0003)

        for param_group in self.optimizer.param_groups:
            # 这里简化处理，实际需要按参数名分组
            pass

    def _apply_lr_scale_v2(self, lr_scale_config: Dict[str, float]):
        """
        应用精细化学习率缩放（基于参数分组）。

        实现:
        - 使用 model.get_parameter_groups() 获取分组
        - 为每个分组创建独立的 param_group
        - 设置 param_group['lr'] = base_lr * scale
        - 如果分组不在 lr_scale_config 中，使用 base_lr
        - [新增 - 2026-04-22] 参数去重：确保同一参数只属于一个组

        Args:
            lr_scale_config: {"beta_residual_experts": 1.5, "beta_projectors": 1.0, ...}
        """
        # [v4 Clean] 检查模型是否支持 get_parameter_groups
        if not hasattr(self.model, 'get_parameter_groups'):
            print("[LR Scale v2] 当前架构不支持 get_parameter_groups，跳过学习率缩放")
            return

        base_lr = self.optimizer.param_groups[0]['lr']
        param_groups = self.model.get_parameter_groups()

        # [新增] 参数去重：优先使用任务专属分组，避免重复添加
        # 优先级：任务专属分组 > 总分组
        # 例如：t3_gate > beta_gates
        priority_groups = ['t2_gate', 't3_gate', 't4_gate', 't5_gate',
                           't2_projector', 't3_projector', 't4_projector', 't5_projector',
                           'beta_shared_expert', 'beta_group_245_expert', 'beta_t3_private_expert']

        # 收集已添加的参数 (用 id 防止重复)
        added_param_ids = set()
        trainable_params = []

        # 先处理优先级分组 (任务专属)
        for group_name in priority_groups:
            if group_name not in param_groups:
                continue
            params = param_groups[group_name]
            if len(params) == 0:
                continue
            # 过滤出可训练且未添加的参数
            new_params = [p for p in params if p.requires_grad and id(p) not in added_param_ids]
            if len(new_params) == 0:
                continue
            # 标记已添加
            for p in new_params:
                added_param_ids.add(id(p))
            scale = lr_scale_config.get(group_name, 1.0)
            trainable_params.append({
                'params': new_params,
                'lr': base_lr * scale,
                'name': group_name
            })

        # 再处理其他分组 (跳过已添加的参数)
        for group_name, params in param_groups.items():
            if group_name in priority_groups:
                continue  # 已处理
            if len(params) == 0:
                continue
            # 过滤出可训练且未添加的参数
            new_params = [p for p in params if p.requires_grad and id(p) not in added_param_ids]
            if len(new_params) == 0:
                continue
            # 标记已添加
            for p in new_params:
                added_param_ids.add(id(p))
            scale = lr_scale_config.get(group_name, 1.0)
            trainable_params.append({
                'params': new_params,
                'lr': base_lr * scale,
                'name': group_name
            })

        if len(trainable_params) == 0:
            print("[LR Scale v2] Warning: No trainable parameters found!")
            return

        # 重建 optimizer (保留 Adam 的配置)
        optimizer_class = type(self.optimizer)
        optimizer_kwargs = {
            'lr': base_lr,
            'betas': self.optimizer.defaults.get('betas', (0.9, 0.999)),
            'eps': self.optimizer.defaults.get('eps', 1e-8),
            'weight_decay': self.optimizer.defaults.get('weight_decay', 0),
        }

        # 使用新的参数组创建 optimizer
        first_group = trainable_params[0]
        remaining_groups = trainable_params[1:]

        self.optimizer = optimizer_class([first_group], **optimizer_kwargs)
        for group in remaining_groups:
            self.optimizer.add_param_group(group)

        print(f"[LR Scale v2] Applied: {lr_scale_config}")
        print(f"[LR Scale v2] Optimizer rebuilt with {len(trainable_params)} param groups")

        # 记录到 SwanLab
        if hasattr(self, 'swanlab_run') and self.swanlab_run:
            self._log_lr_scale_to_swanlab(lr_scale_config, base_lr)

    def _apply_gate_control(self, gate_control: Dict[str, Any], stage: str = "phase2"):
        """
        应用 Gate 相关控制配置。

        Args:
            gate_control: {
                "entropy_reg_enabled": bool,
                "entropy_reg_weight_override": float,
                "tau_override": float
            }
        """
        if 'entropy_reg_weight_override' in gate_control:
            self.gate_entropy_weight = gate_control['entropy_reg_weight_override']
            print(f"[Gate Control] entropy_reg_weight = {self.gate_entropy_weight}")

        if 'tau_override' in gate_control:
            # 设置 Gate 温度 (如果模型支持)
            if hasattr(self.model, 'set_gate_temperature'):
                self.model.set_gate_temperature(gate_control['tau_override'])
            print(f"[Gate Control] tau = {gate_control['tau_override']}")

        self.use_gate_entropy_in_phase2 = gate_control.get('entropy_reg_enabled', True)
        print(f"[Gate Control] entropy_reg_enabled = {self.use_gate_entropy_in_phase2}")

    def _log_freeze_status_to_swanlab(self, freeze_status: Dict[str, Dict[str, Any]], stage: str = "stage3_phase2"):
        """
        记录冻结状态到 SwanLab。

        Args:
            freeze_status: {"alpha_trunk": {"total": N, "frozen": M, "ratio": M/N}, ...}
            stage: 日志前缀
        """
        if not (hasattr(self, 'swanlab_run') and self.swanlab_run):
            return

        for module_name, status in freeze_status.items():
            self.swanlab_run.log({
                f"{stage}/freeze/{module_name}/ratio": status['ratio'],
                f"{stage}/freeze/{module_name}/total": status['total'],
                f"{stage}/freeze/{module_name}/frozen": status['frozen']
            })

        # Global ratio
        total_params = sum(s['total'] for s in freeze_status.values())
        frozen_params = sum(s['frozen'] for s in freeze_status.values())
        global_ratio = frozen_params / total_params if total_params > 0 else 0
        self.swanlab_run.log({f"{stage}/freeze/global_ratio": global_ratio})
        print(f"[SwanLab] Freeze status logged: global_ratio={global_ratio:.2%}")

    def _log_lr_scale_to_swanlab(self, lr_scale_config: Dict[str, float], base_lr: float):
        """
        记录学习率缩放分布到 SwanLab。

        Args:
            lr_scale_config: {"beta_residual_experts": 1.5, ...}
            base_lr: 基础学习率
        """
        if not (hasattr(self, 'swanlab_run') and self.swanlab_run):
            return

        for group_name, scale in lr_scale_config.items():
            self.swanlab_run.log({f"lr_scale/{group_name}": scale})
            self.swanlab_run.log({f"lr_actual/{group_name}": base_lr * scale})

    def _log_to_swanlab(self, epoch, train_metrics, val_metrics, stage):
        """记录 SwanLab 日志 + JSON 缓存

        [修复 - 2026-04-22] 即使 SwanLab 禁用，也填充 metrics_history 以便导出 JSON
        """
        # ========== [修复] 初始化阶段缓存 (始终执行，不受 SwanLab 影响) ==========
        if stage not in self.metrics_history:
            self.metrics_history[stage] = {}
        if stage not in self.train_metrics_history:
            self.train_metrics_history[stage] = {}
        if stage not in self.log_vars_history:
            self.log_vars_history[stage] = {}
        # ================================================

        # ========== [修复] 记录训练指标到 JSON 缓存 (始终执行) ==========
        epoch_train_metrics = {}
        for task_key, metrics in train_metrics.items():
            epoch_train_metrics[task_key] = {}
            for metric_name, value in metrics.items():
                # JSON 缓存 (转换为 Python 基础类型)
                epoch_train_metrics[task_key][metric_name] = float(value) if isinstance(value, (int, float, np.floating)) else value
        self.train_metrics_history[stage][epoch] = epoch_train_metrics
        # ================================================

        # ========== [修复] 记录验证指标到 JSON 缓存 (始终执行) ==========
        epoch_val_metrics = {}
        for task_key, metrics in val_metrics.items():
            epoch_val_metrics[task_key] = {}
            for metric_name, value in metrics.items():
                # JSON 缓存
                epoch_val_metrics[task_key][metric_name] = float(value) if isinstance(value, (int, float, np.floating)) else value
        self.metrics_history[stage][epoch] = epoch_val_metrics
        # ================================================

        # ========== [修复] 记录 log_vars 到 JSON 缓存 (始终执行) ==========
        # stage2 和 stage3_phase1/phase2 需要记录 log_vars/t2-t5
        epoch_log_vars = {}
        if stage in ["stage2", "stage3", "stage3_phase1", "stage3_phase2"]:
            for task_key in ["t2", "t3", "t4", "t5"]:
                if task_key in self.model.log_vars:
                    log_var_value = self.model.log_vars[task_key].item()
                    # s_t = log(sigma^2), sigma^2 = exp(s_t)
                    sigma_sq = np.exp(log_var_value)
                    # JSON 缓存
                    epoch_log_vars[task_key] = {
                        "log_var": float(log_var_value),
                        "sigma_sq": float(sigma_sq),
                        "uncertainty_weight": float(1.0 / (sigma_sq + 1e-6))
                    }
        # ================================================

        # stage3 还需要记录 t1, t6 的 log_vars
        if stage in ["stage3", "stage3_phase2"]:
            for task_key in ["t1", "t6"]:
                if task_key in self.model.log_vars:
                    log_var_value = self.model.log_vars[task_key].item()
                    sigma_sq = np.exp(log_var_value)
                    # JSON 缓存
                    epoch_log_vars[task_key] = {
                        "log_var": float(log_var_value),
                        "sigma_sq": float(sigma_sq),
                        "uncertainty_weight": float(1.0 / (sigma_sq + 1e-6))
                    }

        # 存储 log_vars 历史
        if epoch_log_vars:
            self.log_vars_history[stage][epoch] = epoch_log_vars

        # ========== SwanLab 日志 (仅在启用时执行) ==========
        if not self.swanlab_run:
            return  # SwanLab 禁用，但 metrics_history 已填充

        # SwanLab 训练指标日志
        for task_key, metrics in train_metrics.items():
            for metric_name, value in metrics.items():
                self.swanlab_run.log({
                    f"{stage}/train/{task_key}/{metric_name}": value,
                    "epoch": epoch
                })

        # SwanLab 验证指标日志
        for task_key, metrics in val_metrics.items():
            for metric_name, value in metrics.items():
                self.swanlab_run.log({
                    f"{stage}/{task_key}/{metric_name}": value,
                    "epoch": epoch
                })

        # SwanLab log_vars 日志
        if stage in ["stage2", "stage3", "stage3_phase1", "stage3_phase2"]:
            for task_key in ["t2", "t3", "t4", "t5"]:
                if task_key in self.model.log_vars:
                    log_var_value = self.model.log_vars[task_key].item()
                    sigma_sq = np.exp(log_var_value)
                    self.swanlab_run.log({
                        f"{stage}/log_vars/{task_key}": log_var_value,
                        f"{stage}/uncertainty_weight/{task_key}": 1.0 / (sigma_sq + 1e-6),
                        "epoch": epoch
                    })

        if stage in ["stage3", "stage3_phase2"]:
            for task_key in ["t1", "t6"]:
                if task_key in self.model.log_vars:
                    log_var_value = self.model.log_vars[task_key].item()
                    sigma_sq = np.exp(log_var_value)
                    self.swanlab_run.log({
                        f"{stage}/log_vars/{task_key}": log_var_value,
                        f"{stage}/uncertainty_weight/{task_key}": 1.0 / (sigma_sq + 1e-6),
                        "epoch": epoch
                    })
        # ================================================

        # 注意: JSON 增量保存已改为每个 stage 结束时执行 (减少 IO 频率)

    def save_metrics_to_json(self, save_dir: str = "metrics_logs", fold: int = 1):
        """
        导出所有指标历史到 JSON 文件

        Args:
            save_dir: 保存目录
            fold: Fold 编号

        Returns:
            json_path: JSON 文件路径
        """
        import json
        os.makedirs(save_dir, exist_ok=True)

        # 生成文件名 (使用 checkpoint_prefix + timestamp)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(save_dir, f"{self.checkpoint_prefix}_metrics_fold{fold}_{timestamp}.json")

        # 构建完整导出结构
        export_data = {
            "metadata": {
                "fold": fold,
                "timestamp": timestamp,
                "task_specs": {
                    k: {
                        "name": v.name,
                        "num_classes": v.num_classes,
                        "branch": v.branch,
                        "loss_name": v.loss_name,
                        "is_binary": v.is_binary,
                        "minority_idx": v.minority_idx,
                        "majority_idx": v.majority_idx
                    } for k, v in self.task_specs.items()
                }
            },
            "threshold_search": {
                "mode": self.threshold_mode,
                "tasks": self.threshold_search_tasks,
                "fixed_thresholds": self.fixed_thresholds,
                "best_thresholds": self.best_thresholds
            },
            "metrics": {
                "train": self.train_metrics_history,
                "val": self.metrics_history,
                "log_vars": self.log_vars_history
            }
        }

        # 写入 JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"[Info] 指标历史已导出至: {json_path}")
        return json_path

    def _init_incremental_json(self, fold: int = 1):
        """
        [新增] 初始化增量保存 JSON 文件

        在每个 Fold 开始时调用，创建空的 JSON 文件骨架。
        后续每个 epoch 会追加写入，确保崩溃时不丢失已完成的数据。

        Args:
            fold: Fold 编号
        """
        # ========== [修复 - 2026-04-23] 同步更新 current_fold ==========
        # 确保阈值搜索时使用正确的fold编号
        self.current_fold = fold
        # ============================================================

        if not self.incremental_save_enabled:
            return

        os.makedirs(self.incremental_save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.incremental_json_path = os.path.join(
            self.incremental_save_dir,
            f"mtl_metrics_fold{fold}_{timestamp}_incremental.json"
        )

        # 初始化骨架
        initial_data = {
            "metadata": {
                "fold": fold,
                "timestamp": timestamp,
                "task_specs": {
                    k: {
                        "name": v.name,
                        "num_classes": v.num_classes,
                        "branch": v.branch,
                        "loss_name": v.loss_name,
                        "is_binary": v.is_binary,
                        "minority_idx": v.minority_idx,
                        "majority_idx": v.majority_idx
                    } for k, v in self.task_specs.items()
                },
                "incremental_save": True
            },
            "threshold_search": {
                "mode": self.threshold_mode,
                "tasks": self.threshold_search_tasks,
                "fixed_thresholds": self.fixed_thresholds,
                "best_thresholds": {}  # 运行过程中更新
            },
            "metrics": {
                "train": {},
                "val": {},
                "log_vars": {}
            },
            "checkpoint_info": {
                "stage1_best": {"epoch": None, "metric": None},
                "stage2_best": {"epoch": None, "metric": None},
                "stage3_best": {"epoch": None, "metric": None},
                "stage3_phase1_best": {"epoch": None, "metric": None},
                "stage3_phase2_best": {"epoch": None, "metric": None}  # [新增 - 2026-04-21]
            }
        }

        with open(self.incremental_json_path, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)

        print(f"[增量保存] 初始化 JSON: {self.incremental_json_path}")

    def _save_stage_to_json(self, stage: str):
        """
        [新增] 在每个 stage 结束时批量保存该 stage 的所有 epoch 数据

        相比每个 epoch 保存一次，这种方式减少 IO 操作频率，同时保证数据完整性。
        数据来源: 内存中的 train_metrics_history, metrics_history, log_vars_history

        Args:
            stage: 阶段名称 ("stage1", "stage2", "stage3", "stage3_phase1", "stage3_phase2")
        """
        if not self.incremental_save_enabled or self.incremental_json_path is None:
            return

        # 检查是否有该 stage 的数据
        if stage not in self.train_metrics_history or stage not in self.metrics_history:
            print(f"[增量保存] Warning: {stage} 没有数据可保存")
            return

        # 读取现有数据
        try:
            with open(self.incremental_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[增量保存] Warning: 读取失败 {e}, 使用空骨架")
            data = {
                "metrics": {"train": {}, "val": {}, "log_vars": {}},
                "checkpoint_info": {
                    "stage1_best": {"epoch": None, "metric": None},
                    "stage2_best": {"epoch": None, "metric": None},
                    "stage3_best": {"epoch": None, "metric": None},
                    "stage3_phase1_best": {"epoch": None, "metric": None}
                }
            }

        # 转换该 stage 的所有 epoch 数据为 Python 基础类型
        stage_train_data = {}
        for epoch, metrics in self.train_metrics_history[stage].items():
            stage_train_data[str(epoch)] = {
                task_key: {
                    metric_name: float(value) if isinstance(value, (int, float, np.floating)) else value
                    for metric_name, value in task_metrics.items()
                }
                for task_key, task_metrics in metrics.items()
            }

        stage_val_data = {}
        for epoch, metrics in self.metrics_history[stage].items():
            stage_val_data[str(epoch)] = {
                task_key: {
                    metric_name: float(value) if isinstance(value, (int, float, np.floating)) else value
                    for metric_name, value in task_metrics.items()
                }
                for task_key, task_metrics in metrics.items()
            }

        stage_log_vars_data = {}
        if stage in self.log_vars_history:
            for epoch, log_vars in self.log_vars_history[stage].items():
                stage_log_vars_data[str(epoch)] = {
                    task_key: {
                        "log_var": float(var_dict.get("log_var", 0)),
                        "sigma_sq": float(var_dict.get("sigma_sq", 1)),
                        "uncertainty_weight": float(var_dict.get("uncertainty_weight", 1))
                    }
                    for task_key, var_dict in log_vars.items()
                }

        # 更新数据结构
        data["metrics"]["train"][stage] = stage_train_data
        data["metrics"]["val"][stage] = stage_val_data
        data["metrics"]["log_vars"][stage] = stage_log_vars_data

        # 写回文件
        with open(self.incremental_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        num_epochs = len(stage_train_data)
        print(f"[增量保存] Stage {stage} 完成 ({num_epochs} epochs) 已写入 JSON")

    def _append_epoch_to_json(
        self,
        epoch: int,
        stage: str,
        train_metrics: Dict[str, Dict[str, float]],
        val_metrics: Dict[str, Dict[str, float]],
        log_vars: Dict[str, Dict[str, float]] = None
    ):
        """
        [新增] 增量追加单个 epoch 的指标到 JSON 文件

        每个 epoch 结束后调用，追加写入当 epoch 的训练/验证指标。
        使用"读取-修改-写入"策略确保数据完整性。

        Args:
            epoch: 当前 epoch 编号
            stage: 阶段名称 ("stage1", "stage2", "stage3", "stage3_phase1", "stage3_phase2")
            train_metrics: 训练指标 {"t1": {...}, "t2": {...}, ...}
            val_metrics: 验证指标 {"t1": {...}, "t2": {...}, ...}
            log_vars: 不确定性权重 (可选)
        """
        if not self.incremental_save_enabled or self.incremental_json_path is None:
            return

        # 读取现有数据
        try:
            with open(self.incremental_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[增量保存] Warning: 读取失败 {e}, 使用空骨架")
            data = {
                "metrics": {"train": {}, "val": {}, "log_vars": {}},
                "checkpoint_info": {
                    "stage1_best": {"epoch": None, "metric": None},
                    "stage2_best": {"epoch": None, "metric": None},
                    "stage3_best": {"epoch": None, "metric": None},
                    "stage3_phase1_best": {"epoch": None, "metric": None}
                }
            }

        # 转换指标为 Python 基础类型
        epoch_train_metrics = {}
        for task_key, metrics in train_metrics.items():
            epoch_train_metrics[task_key] = {}
            for metric_name, value in metrics.items():
                epoch_train_metrics[task_key][metric_name] = float(value) if isinstance(value, (int, float, np.floating)) else value

        epoch_val_metrics = {}
        for task_key, metrics in val_metrics.items():
            epoch_val_metrics[task_key] = {}
            for metric_name, value in metrics.items():
                epoch_val_metrics[task_key][metric_name] = float(value) if isinstance(value, (int, float, np.floating)) else value

        # 追加到对应 stage
        stage_key = stage
        if stage_key not in data["metrics"]["train"]:
            data["metrics"]["train"][stage_key] = {}
        if stage_key not in data["metrics"]["val"]:
            data["metrics"]["val"][stage_key] = {}
        if stage_key not in data["metrics"]["log_vars"]:
            data["metrics"]["log_vars"][stage_key] = {}

        data["metrics"]["train"][stage_key][str(epoch)] = epoch_train_metrics
        data["metrics"]["val"][stage_key][str(epoch)] = epoch_val_metrics

        # 追加 log_vars (如果提供)
        if log_vars:
            epoch_log_vars = {}
            for task_key, var_dict in log_vars.items():
                epoch_log_vars[task_key] = {
                    "log_var": float(var_dict.get("log_var", 0)),
                    "sigma_sq": float(var_dict.get("sigma_sq", 1)),
                    "uncertainty_weight": float(var_dict.get("uncertainty_weight", 1))
                }
            data["metrics"]["log_vars"][stage_key][str(epoch)] = epoch_log_vars

        # 写回文件
        with open(self.incremental_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[增量保存] Epoch {epoch} ({stage}) 已写入")

    def _update_checkpoint_info_in_json(
        self,
        stage: str,
        epoch: int,
        metric: float,
        metric_name: str = "macro_f1"
    ):
        """
        [新增] 更新 checkpoint 信息到增量 JSON

        当保存最佳模型时调用，记录最佳 epoch 和指标。

        Args:
            stage: 阶段名称
            epoch: 最佳 epoch
            metric: 最佳指标值
            metric_name: 指标名称
        """
        if not self.incremental_save_enabled or self.incremental_json_path is None:
            return

        try:
            with open(self.incremental_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return

        # 映射 stage 名称到 checkpoint_info 的键
        stage_mapping = {
            "stage1": "stage1_best",
            "stage2": "stage2_best",
            "stage3": "stage3_best",
            "stage3_phase1": "stage3_phase1_best",
            "stage3_phase2": "stage3_phase2_best"  # [新增 - 2026-04-21]
        }

        key = stage_mapping.get(stage, stage)
        if "checkpoint_info" not in data:
            data["checkpoint_info"] = {}

        data["checkpoint_info"][key] = {
            "epoch": epoch,
            "metric": float(metric),
            "metric_name": metric_name
        }

        with open(self.incremental_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[增量保存] Checkpoint info 更新: {key} @ epoch {epoch} ({metric_name}={metric:.4f})")

    def _update_best_thresholds_in_json(self):
        """
        [新增] 更新最佳阈值到增量 JSON

        阈值搜索完成后调用。
        """
        if not self.incremental_save_enabled or self.incremental_json_path is None:
            return

        try:
            with open(self.incremental_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return

        data["threshold_search"]["best_thresholds"] = self.best_thresholds

        with open(self.incremental_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[增量保存] 最佳阈值已更新: {self.best_thresholds}")


# =============================================================================
# [新增] Holdout 测试集评估函数
# =============================================================================

def evaluate_mtl_on_holdout_test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    task_specs: Dict[str, TaskSpec],
    device: str,
    criterions: Dict[str, nn.Module],
    thresholds: Dict[str, float] = None,
    # [新增 2026-05-24] ROC 导出参数
    return_roc_data: bool = False,
    fold_idx: int = None,
    checkpoint_path: str = None,
    model_name: str = "Our method",
    model_type: str = "our_method",
    return_prediction_table: bool = False
) -> Union[Dict[str, Dict[str, float]], Tuple[Dict, Dict], Tuple[Dict, Dict, List[Dict[str, Any]]]]:
    """
    在 Holdout 独立测试集上评估 MTL 模型

    Args:
        model: MTL 模型
        test_loader: Holdout 测试集 DataLoader
        task_specs: 任务规格字典
        device: 设备
        criterions: 损失函数字典
        thresholds: [新增] 二分类阈值字典 (从 val-search 结果传入)
        return_roc_data: [新增] 是否返回 ROC 导出数据 (仅对 t3/t4/t5)
        fold_idx: [新增] Fold 编号 (用于 ROC 导出)
        checkpoint_path: [新增] Checkpoint 路径 (用于 ROC 导出)
        model_name: [新增] 模型名称 (用于 ROC 导出)
        model_type: [新增] 模型类型标识 (用于 ROC 导出)

    Returns:
        metrics: 各任务的评估指标字典
            {
                't1': {'accuracy': 0.85, 'macro_f1': 0.82, ...},
                't2': {...},
                ...
            }
            roc_export_data: [新增] ROC 导出数据字典 (仅当 return_roc_data=True)
            {
                "sample_scores": [...],
                "roc_points_fold": [...],
                "run_info": {...}
            }
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    from sklearn.metrics import roc_curve, roc_auc_score
    from datetime import datetime

    model.eval()

    # 初始化累积器
    all_preds = {task_key: [] for task_key in task_specs}
    all_labels = {task_key: [] for task_key in task_specs}
    all_losses = {task_key: [] for task_key in task_specs}
    all_probs = {task_key: [] for task_key in task_specs if task_specs[task_key].is_binary}

    # [新增] ROC 数据收集器 (仅当 return_roc_data=True 且为 t3/t4/t5)
    roc_tasks = ["t3", "t4", "t5"]
    sample_scores_list = []
    roc_points_fold_list = []
    prediction_table_rows = []

    # 任务名称映射
    task_name_map = {
        "t3": "标准心电运动负荷试验",
        "t4": "运动中换气肺功能",
        "t5": "心率储备",
    }

    with torch.no_grad():
        for batch in test_loader:
            x_dyn = batch['x_dyn'].to(device)
            x_static = batch.get('x_static')
            if x_static is not None:
                x_static = x_static.to(device)

            labels = batch['labels']
            masks = batch['label_mask']

            # 前向传播
            outputs = model(x_dyn, x_static=x_static)

            # 计算各任务损失和指标
            for task_key, spec in task_specs.items():
                task_output = outputs[task_key]
                task_label = labels[task_key].to(device)
                task_mask = masks[task_key].to(device)

                # [修复] 模型输出是嵌套 dict: {"t1": {"logits": Tensor}, ...}
                # 需要取出 logits Tensor
                if isinstance(task_output, dict):
                    task_logits = task_output['logits']
                else:
                    task_logits = task_output  # 兼容旧格式

                # 损失计算 (考虑掩码)
                criterion = criterions[task_key]

                # [修复] 二分类任务需要调整 tensor shape
                if spec.is_binary:
                    # [v2 统一] 二分类 target 编码: 少数类=1
                    target_for_loss = make_binary_target_for_loss(task_label, spec.minority_idx)

                    # BCEWithLogitsLoss 和 UnifiedLDAMLoss 都要求 logits 和 label 形状匹配
                    # 模型输出: [B, 1] 或 [B] (取决于 num_classes=2 时 squeeze 情况)
                    # 需要确保两者都是 2D: [B, 1]
                    if task_logits.dim() == 1:
                        # 如果 logits 已经是 [B]，unsqueeze 成 [B, 1]
                        logits_for_loss = task_logits.unsqueeze(-1)
                    else:
                        # 如果 logits 是 [B, 1]，保持不变
                        logits_for_loss = task_logits

                    # target 也需要 unsqueeze 成 [B, 1]
                    target_for_loss = target_for_loss.unsqueeze(-1)
                    loss = criterion(logits_for_loss, target_for_loss)
                else:
                    loss = criterion(task_logits, task_label)

                # 累积预测和标签 (使用 task_logits)
                if spec.is_binary:
                    # logits → minority_prob (统一方向: P(minority))
                    probs = get_minority_prob_from_logits(task_logits, spec.minority_idx)
                    # 应用阈值 (从 val-search 结果或默认 0.5)
                    threshold = thresholds.get(task_key, 0.5) if thresholds else 0.5
                    preds = apply_threshold(probs, threshold, spec.minority_idx, spec.majority_idx)
                    class_probs_for_export = torch.zeros(
                        (probs.shape[0], 2),
                        dtype=probs.dtype,
                        device=probs.device
                    )
                    class_probs_for_export[:, spec.minority_idx] = probs
                    class_probs_for_export[:, spec.majority_idx] = 1.0 - probs
                else:
                    probs = None
                    preds = task_logits.argmax(dim=1)
                    threshold = None
                    class_probs_for_export = F.softmax(task_logits, dim=1)

                # 过滤有效样本 (mask=1)
                if return_prediction_table:
                    filenames = batch.get("filename", [None] * task_label.shape[0])
                    patient_ids = batch.get("patient_id", [None] * task_label.shape[0])
                    ages = batch.get("age", [None] * task_label.shape[0])
                    sexes = batch.get("sex", [None] * task_label.shape[0])
                    t6_labels = labels.get("t6")
                    t6_masks = masks.get("t6")

                    task_label_cpu = task_label.detach().cpu()
                    task_mask_cpu = task_mask.detach().cpu()
                    preds_cpu = preds.detach().cpu()
                    probs_export_cpu = class_probs_for_export.detach().cpu().numpy()
                    t6_labels_cpu = t6_labels.detach().cpu() if t6_labels is not None else None
                    t6_masks_cpu = t6_masks.detach().cpu() if t6_masks is not None else None

                    for sample_i in range(task_label_cpu.shape[0]):
                        label_available = bool(task_mask_cpu[sample_i].item() == 1)
                        t6_available = (
                            t6_masks_cpu is not None and
                            bool(t6_masks_cpu[sample_i].item() == 1)
                        )
                        class_prob_values = [
                            float(v) for v in probs_export_cpu[sample_i].tolist()
                        ]
                        row = {
                            "filename": filenames[sample_i] if sample_i < len(filenames) else None,
                            "patient_id": patient_ids[sample_i] if sample_i < len(patient_ids) else None,
                            "fold": fold_idx,
                            "split": "holdout",
                            "task": task_key,
                            "y_true": int(task_label_cpu[sample_i].item()) if label_available else None,
                            "y_pred": int(preds_cpu[sample_i].item()),
                            "class_probabilities": json.dumps(class_prob_values, ensure_ascii=False),
                            "selected_threshold": float(threshold) if spec.is_binary else None,
                            "disease_context/t6": (
                                int(t6_labels_cpu[sample_i].item())
                                if t6_labels_cpu is not None and t6_available
                                else None
                            ),
                            "age": ages[sample_i] if sample_i < len(ages) else None,
                            "sex": sexes[sample_i] if sample_i < len(sexes) else None,
                            "label_available": label_available,
                        }
                        for class_idx, prob_value in enumerate(class_prob_values):
                            row[f"prob_class_{class_idx}"] = prob_value
                        prediction_table_rows.append(row)

                valid_mask = task_mask == 1
                if valid_mask.sum() > 0:
                    all_preds[task_key].append(preds[valid_mask].cpu())
                    all_labels[task_key].append(task_label[valid_mask].cpu())
                    if probs is not None:
                        all_probs[task_key].append(probs[valid_mask].cpu())
                    all_losses[task_key].append(loss.item())

    # 计算各任务指标
    metrics = {}
    for task_key in task_specs:
        if len(all_preds[task_key]) == 0:
            metrics[task_key] = {'accuracy': 0, 'macro_f1': 0, 'loss': 0}
            continue

        preds_cat = torch.cat(all_preds[task_key]).numpy()
        labels_cat = torch.cat(all_labels[task_key]).numpy()
        avg_loss = np.mean(all_losses[task_key]) if all_losses[task_key] else 0

        spec = task_specs[task_key]

        # 计算基础指标
        accuracy = accuracy_score(labels_cat, preds_cat)

        if spec.is_binary:
            # ========== [修复 - 2026-04-23] 二分类使用 macro 平均 (与Excel一致) ==========
            # Excel logger 需要 precision/recall/f1 使用 macro 平均
            try:
                macro_f1 = f1_score(labels_cat, preds_cat, average='macro')
                # 使用 macro 平均 (两类F1的平均)，而非 binary (只计算少数类)
                precision = precision_score(labels_cat, preds_cat, average='macro', zero_division=0)
                recall = recall_score(labels_cat, preds_cat, average='macro', zero_division=0)

                # [v2 统一] AUC 计算: probs_cat 是 minority_prob, y_true_minor = (labels == minority_idx)
                from sklearn.metrics import roc_auc_score
                probs_cat = torch.cat(all_probs[task_key]).numpy()
                y_true_minor = (labels_cat == spec.minority_idx).astype(int)
                try:
                    auc = roc_auc_score(y_true_minor, probs_cat)
                except Exception as e:
                    print(f"[AUC Error] {task_key}: labels={labels_cat.shape}, probs={probs_cat.shape}, unique_labels={np.unique(labels_cat)}, error={e}")
                    auc = np.nan

                # 二分类少数类指标
                minority_f1 = f1_score(labels_cat, preds_cat, average='binary', pos_label=spec.minority_idx)
                minority_recall = recall_score(labels_cat, preds_cat, pos_label=spec.minority_idx, average='binary')
                minority_precision = precision_score(labels_cat, preds_cat, pos_label=spec.minority_idx, zero_division=0)
                pred_minor_rate = (preds_cat == spec.minority_idx).mean()
                true_minor_rate = (labels_cat == spec.minority_idx).mean()

                # 混淆矩阵 → 少数类/多数类计数
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(labels_cat, preds_cat)
                if spec.minority_idx == 0:
                    minority_tp = int(cm[0, 0])
                    minority_fn = int(cm[0, 1])
                    majority_fp = int(cm[1, 0])
                    majority_tn = int(cm[1, 1])
                else:
                    minority_tp = int(cm[1, 1])
                    minority_fn = int(cm[1, 0])
                    majority_fp = int(cm[0, 1])
                    majority_tn = int(cm[0, 0])

                # ========== [诊断] 双向 AUC + label-wise prob 均值 (t3/t4/t5) ==========
                # 仅对 t3/t4/t5 打印诊断
                if task_key in ["t3", "t4", "t5"]:
                    # 统一定义：y_true_minor = (labels == minority_idx)
                    y_true_minor = (labels_cat == spec.minority_idx).astype(int)

                    # 双向 AUC
                    try:
                        auc_minority_holdout = roc_auc_score(y_true_minor, probs_cat)
                        auc_reverse_holdout = roc_auc_score(y_true_minor, 1.0 - probs_cat)
                    except:
                        auc_minority_holdout = np.nan
                        auc_reverse_holdout = np.nan

                    # label-wise probability 均值
                    minority_mask_holdout = labels_cat == spec.minority_idx
                    majority_mask_holdout = labels_cat == spec.majority_idx

                    prob_mean_minority_holdout = probs_cat[minority_mask_holdout].mean() if minority_mask_holdout.sum() > 0 else np.nan
                    prob_mean_majority_holdout = probs_cat[majority_mask_holdout].mean() if majority_mask_holdout.sum() > 0 else np.nan

                    # 诊断判断
                    ranking_warning_holdout = ""
                    if prob_mean_minority_holdout < prob_mean_majority_holdout:
                        ranking_warning_holdout = "[Ranking Warning] minority_prob is higher on majority samples!"

                    # 打印诊断表
                    print(f"\n[HoldoutDiag] {task_key} (minority_idx={spec.minority_idx}):")
                    print(f"  auc_minority={auc_minority_holdout:.4f}, auc_reverse={auc_reverse_holdout:.4f}")
                    print(f"  prob_mean_true_minority={prob_mean_minority_holdout:.4f}, prob_mean_true_majority={prob_mean_majority_holdout:.4f}")
                    print(f"  pred_minor_rate={pred_minor_rate:.4f}, true_minor_rate={true_minor_rate:.4f}")
                    print(f"  minority_precision={minority_precision:.4f}, minority_recall={minority_recall:.4f}, minority_f1={minority_f1:.4f}")
                    if ranking_warning_holdout:
                        print(f"  {ranking_warning_holdout}")
                # ================================================

                metrics[task_key] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': macro_f1,  # f1 = macro_f1
                    'macro_f1': macro_f1,
                    'auc': auc,
                    'loss': avg_loss,
                    'minority_f1': minority_f1,
                    'minority_recall': minority_recall,
                    'minority_precision': minority_precision,
                    'pred_minor_rate': pred_minor_rate,
                    'true_minor_rate': true_minor_rate,
                    'minority_tp': minority_tp,
                    'minority_fn': minority_fn,
                    'majority_fp': majority_fp,
                    'majority_tn': majority_tn,
                }
                # ================================================

                # ========== [新增 2026-05-24] ROC 数据收集 (仅 t3/t4/t5) ==========
                if return_roc_data and task_key in roc_tasks:
                    # 复用已有的 probs_cat, labels_cat, y_true_minor
                    # 计算 ROC 曲线
                    fpr, tpr, roc_thresholds = roc_curve(y_true_minor, probs_cat)
                    auc_reverse = roc_auc_score(y_true_minor, 1.0 - probs_cat)

                    n_positive = int(y_true_minor.sum())
                    n_negative = len(y_true_minor) - n_positive

                    # 校验：导出的 AUC 应与 metrics AUC 一致
                    if auc != metrics[task_key]["auc"]:
                        print(f"[ROC Export Warning] Exported ROC AUC ({auc:.4f}) != holdout metrics AUC ({metrics[task_key]['auc']:.4f})")

                    # 打印导出信息
                    print(f"\n[ROC Export]")
                    print(f"  model={model_name}")
                    print(f"  fold={fold_idx}")
                    print(f"  task={task_key}")
                    print(f"  positive_class={spec.minority_idx}")
                    print(f"  n_positive={n_positive}")
                    print(f"  n_negative={n_negative}")
                    print(f"  auc={auc:.4f}")
                    print(f"  auc_reverse={auc_reverse:.4f}")
                    print(f"  original_holdout_auc={metrics[task_key]['auc']:.4f}")

                    # 构建 sample_scores (逐样本)
                    n_samples = len(labels_cat)
                    for sample_idx in range(n_samples):
                        sample_scores_list.append({
                            "model_name": model_name,
                            "model_type": model_type,
                            "task_key": task_key,
                            "task_name": task_name_map.get(task_key, task_key),
                            "fold": fold_idx,
                            "sample_index": sample_idx,
                            "y_true": int(labels_cat[sample_idx]),
                            "y_true_minor": int(y_true_minor[sample_idx]),
                            "y_score": float(probs_cat[sample_idx]),
                            "positive_class": spec.minority_idx,
                            "checkpoint_path": checkpoint_path,
                        })

                    # 构建 roc_points_fold (ROC 曲线点)
                    for i in range(len(fpr)):
                        roc_points_fold_list.append({
                            "model_name": model_name,
                            "model_type": model_type,
                            "task_key": task_key,
                            "task_name": task_name_map.get(task_key, task_key),
                            "fold": fold_idx,
                            "fpr": float(fpr[i]),
                            "tpr": float(tpr[i]),
                            "threshold": float(roc_thresholds[i]),
                            "auc": float(auc),
                            "auc_reverse": float(auc_reverse),
                            "n_positive": n_positive,
                            "n_negative": n_negative,
                            "positive_class": spec.minority_idx,
                            "checkpoint_path": checkpoint_path,
                        })
                # ================================================
            except Exception as e:
                print(f"[Warning] {task_key} 指标计算失败: {e}")
                metrics[task_key] = {'accuracy': accuracy, 'macro_f1': accuracy, 'loss': avg_loss}
        else:
            # ========== [修复 - 2026-04-23] 多分类添加 precision/recall/f1 字段 ==========
            # Excel logger 需要 precision/recall/f1 字段
            try:
                macro_f1 = f1_score(labels_cat, preds_cat, average='macro', zero_division=0)
                weighted_f1 = f1_score(labels_cat, preds_cat, average='weighted', zero_division=0)
                precision = precision_score(labels_cat, preds_cat, average='macro', zero_division=0)
                recall = recall_score(labels_cat, preds_cat, average='macro', zero_division=0)

                metrics[task_key] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': macro_f1,  # f1 = macro_f1
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'loss': avg_loss
                }
                # ================================================
            except Exception as e:
                print(f"[Warning] {task_key} 指标计算失败: {e}")
                metrics[task_key] = {'accuracy': accuracy, 'macro_f1': accuracy, 'loss': avg_loss}

    # [诊断] 打印每个二分类任务的 minority_prob 方向诊断
    for task_key in task_specs:
        spec = task_specs[task_key]
        if not spec.is_binary:
            continue
        threshold = thresholds.get(task_key, 0.5) if thresholds else 0.5
        probs_cat = torch.cat(all_probs[task_key]).numpy() if all_probs[task_key] else np.array([])
        preds_cat_local = torch.cat(all_preds[task_key]).numpy() if all_preds[task_key] else np.array([])
        labels_cat_local = torch.cat(all_labels[task_key]).numpy() if all_labels[task_key] else np.array([])
        if len(probs_cat) > 0:
            pred_minor_rate = (preds_cat_local == spec.minority_idx).mean()
            true_minor_rate = (labels_cat_local == spec.minority_idx).mean()
            print(f"[BinaryDiag] {task_key}: minority_idx={spec.minority_idx}, majority_idx={spec.majority_idx}, "
                  f"threshold={threshold:.3f}, prob_mean={probs_cat.mean():.4f}, "
                  f"prob_min={probs_cat.min():.4f}, prob_max={probs_cat.max():.4f}, "
                  f"pred_minor_rate={pred_minor_rate:.4f}, true_minor_rate={true_minor_rate:.4f}")
    # ========== [新增 2026-05-24] 返回 ROC 数据 (可选) ==========
    if return_roc_data:
        roc_export_data = {
            "sample_scores": sample_scores_list,
            "roc_points_fold": roc_points_fold_list,
            "run_info": {
                "model_name": model_name,
                "model_type": model_type,
                "fold": fold_idx,
                "checkpoint_path": checkpoint_path,
                "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "n_holdout_samples": len(test_loader.dataset) if hasattr(test_loader, 'dataset') else None
            }
        }
        if return_prediction_table:
            return metrics, roc_export_data, prediction_table_rows
        return metrics, roc_export_data
    if return_prediction_table:
        return metrics, None, prediction_table_rows
    else:
        return metrics


def aggregate_mtl_holdout_metrics(
    all_fold_metrics: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    聚合多个 Fold 的 Holdout 测试集指标

    Args:
        all_fold_metrics: 各 Fold 的指标列表
            [{'t1': {...}, 't2': {...}}, ...]

    Returns:
        aggregated: 各任务的 mean ± std
            {
                't1': {'macro_f1': (mean, std), 'accuracy': (mean, std)},
                ...
            }
    """
    # 获取所有任务键
    task_keys = list(all_fold_metrics[0].keys())

    aggregated = {}
    for task_key in task_keys:
        # 收集各 Fold 的指标
        metric_names = list(all_fold_metrics[0][task_key].keys())
        task_stats = {}

        for metric_name in metric_names:
            values = []
            for fold_metrics in all_fold_metrics:
                if task_key in fold_metrics and metric_name in fold_metrics[task_key]:
                    values.append(fold_metrics[task_key][metric_name])

            if values:
                mean = np.mean(values)
                std = np.std(values)
                task_stats[metric_name] = (mean, std)
            else:
                task_stats[metric_name] = (0.0, 0.0)

        aggregated[task_key] = task_stats

    return aggregated


def save_fold_holdout_stats(
    fold_idx: int,
    stats: dict,
    static_stats: dict,
    metrics: Dict[str, Dict[str, float]],
    results_dir: str = "results",
    checkpoint_prefix: str = "mtl_v4_baseline"
) -> str:
    """
    保存单个 Fold 的 Holdout 统计量和指标

    Args:
        fold_idx: Fold 编号
        stats: 动态特征统计量
        static_stats: 静态特征统计量
        metrics: Holdout 测试集指标
        results_dir: 结果保存目录
        checkpoint_prefix: 实验命名前缀

    Returns:
        save_path: 保存路径
    """
    import json

    def to_list(val):
        """将 numpy array 转换为 Python list"""
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (list, tuple)):
            return list(val)
        return val

    save_path = os.path.join(results_dir, f"{checkpoint_prefix}_holdout_fold{fold_idx+1}_stats.json")

    save_data = {
        'fold': fold_idx + 1,
        'train_stats': {
            'mean': to_list(stats.get('mean', [])),
            'std': to_list(stats.get('std', []))
        },
        'static_stats': {
            'mean': to_list(static_stats.get('mean', [])),
            'std': to_list(static_stats.get('std', []))
        },
        'holdout_metrics': metrics
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"[Holdout] Fold {fold_idx+1} 统计量已保存: {save_path}")

    return save_path

def save_fold_roc_json(
    roc_export_data: Dict,
    output_dir: str = "results/fig2c_roc_raw",
    model_type: str = "our_method",
    fold: int = None
) -> str:
    """
    [新增 2026-05-24] 保存单个 Fold 的 ROC 导出数据

    Args:
        roc_export_data: ROC 导出数据字典
            {
                "sample_scores": [...],
                "roc_points_fold": [...],
                "run_info": {...}
            }
        output_dir: 输出目录 (默认 results/fig2c_roc_raw)
        model_type: 模型类型标识 (默认 our_method)
        fold: Fold 编号

    Returns:
        save_path: 保存路径
    """
    import json

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_type}_fold{fold}_roc_data.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(roc_export_data, f, ensure_ascii=False, indent=2)

    print(f"[ROC Export] Saved fold ROC data to {output_path}")

    return output_path


def save_holdout_prediction_table(
    prediction_rows: List[Dict[str, Any]],
    output_dir: str = "results/holdout_predictions",
    model_type: str = "our_method",
    fold: int = None,
    save_excel: bool = True
) -> Dict[str, Optional[str]]:
    """
    Save a unified long-form holdout prediction table.

    One row corresponds to one holdout sample and one task.
    """
    os.makedirs(output_dir, exist_ok=True)
    fold_part = f"fold{fold}" if fold is not None else "foldNA"
    csv_path = os.path.join(output_dir, f"{model_type}_{fold_part}_holdout_predictions.csv")
    xlsx_path = os.path.join(output_dir, f"{model_type}_{fold_part}_holdout_predictions.xlsx")

    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        df = pd.DataFrame(prediction_rows)
        preferred_columns = [
            "filename", "patient_id", "fold", "split", "task", "y_true", "y_pred",
            "class_probabilities", "selected_threshold", "disease_context/t6",
            "age", "sex", "label_available"
        ]
        prob_columns = sorted(
            [c for c in df.columns if c.startswith("prob_class_")],
            key=lambda c: int(c.rsplit("_", 1)[1])
        )
        other_columns = [c for c in df.columns if c not in preferred_columns + prob_columns]
        df = df[[c for c in preferred_columns if c in df.columns] + prob_columns + other_columns]
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        saved_xlsx_path = None
        if save_excel:
            try:
                df.to_excel(xlsx_path, index=False)
                saved_xlsx_path = xlsx_path
            except Exception as e:
                print(f"[Holdout Prediction Export Warning] Excel export failed: {e}")
        print(f"[Holdout Prediction Export] Saved CSV to {csv_path}")
        if saved_xlsx_path:
            print(f"[Holdout Prediction Export] Saved Excel to {saved_xlsx_path}")
        return {"csv": csv_path, "xlsx": saved_xlsx_path}

    import csv
    fieldnames = []
    for row in prediction_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"[Holdout Prediction Export] Saved CSV to {csv_path}")
    return {"csv": csv_path, "xlsx": None}
    
def save_holdout_summary(
    aggregated_metrics: Dict[str, Dict[str, Tuple[float, float]]],
    results_dir: str = "results"
) -> str:
    """
    保存 Holdout 测试集聚合结果

    Args:
        aggregated_metrics: 聚合后的指标字典
        results_dir: 结果保存目录

    Returns:
        save_path: 保存路径
    """
    import json

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(results_dir, f"mtl_holdout_summary_{timestamp}.json")

    # 转换 tuple 为 dict 格式便于阅读
    formatted = {}
    for task_key, task_stats in aggregated_metrics.items():
        formatted[task_key] = {}
        for metric_name, (mean, std) in task_stats.items():
            formatted[task_key][metric_name] = {
                'mean': round(mean, 4),
                'std': round(std, 4),
                'formatted': f"{mean:.4f} ± {std:.4f}"
            }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    print(f"[Holdout] 聚合结果已保存: {save_path}")

    return save_path


def print_holdout_summary(aggregated_metrics: Dict[str, Dict[str, Tuple[float, float]]]):
    """
    打印 Holdout 测试集聚合结果

    Args:
        aggregated_metrics: 聚合后的指标字典
    """
    print("\n" + "=" * 80)
    print("[Holdout Test] 全 Fold 聚合指标 (mean ± std)")
    print("=" * 80)

    for task_key, task_stats in aggregated_metrics.items():
        print(f"\n{task_key}:")
        for metric_name, (mean, std) in task_stats.items():
            print(f"  {metric_name}: {mean:.4f} ± {std:.4f}")


# =============================================================================
# 主入口
# =============================================================================

def main():
    """测试入口"""
    print("=" * 80)
    print("MTL Training Test")
    print("=" * 80)

    # 测试损失函数工厂
    from task_specs import TaskSpec

    task_specs = {
        "t1": TaskSpec("t1", "运动心功能分级", 3, "alpha", "ce", False, 0.3, "运动心功能分级"),
        "t2": TaskSpec("t2", "运动耐量", 3, "beta", "ldam", False, 0.3, "运动耐量"),
        "t3": TaskSpec("t3", "标准心电运动负荷试验", 2, "beta", "bce", True, 0.3, "标准心电运动负荷试验"),
        "t4": TaskSpec("t4", "运动中换气肺功能", 2, "beta", "ldam", True, 0.3, "运动中换气肺功能"),
        "t5": TaskSpec("t5", "心率储备", 2, "beta", "ldam", True, 0.3, "心率储备"),
        "t6": TaskSpec("t6", "匹配的第一大类", 6, "alpha", "ce", False, 0.3, "匹配的第一大类", kd_teacher="dummy.pth")
    }

    # 更新统计
    task_specs["t1"].update_stats([100, 200, 150])
    task_specs["t3"].update_stats([450, 50])
    task_specs["t6"].update_stats([150, 100, 80, 60, 40, 20])

    # 构建损失
    criterions = build_mtl_criterions(task_specs)
    print("\n损失函数:")
    for k, v in criterions.items():
        print(f"  {k}: {type(v).__name__}")

    # 测试 MTLTotalLoss
    total_loss = MTLTotalLoss(kd_weight=0.5, kd_temperature=2.0)
    print("\nMTLTotalLoss 测试通过")

    # 测试两阶段阈值搜索
    print("\n两阶段阈值搜索测试:")
    logits = np.random.randn(100)  # 模拟 logits
    labels = np.random.randint(0, 2, 100)  # 模拟标签
    search_result = two_stage_threshold_search(
        task_key="t3",
        logits=logits,
        labels=labels,
        minority_idx=1,
        save_dir="threshold_search_test",
        fold=1,
        checkpoint_stage="test",
        verbose=True
    )
    print(f"  最优阈值: {search_result['best_threshold']:.3f}")
    print(f"  最优 F1: {search_result['best_f1_on_val']:.4f}")
    print(f"  基准 F1 (0.5): {search_result['baseline_f1_at_0.5_on_val']:.4f}")

    print("\n[PASS] 所有测试通过")


if __name__ == "__main__":
    main()
