"""
MTL 入口脚本 - 多任务学习训练
============================

三阶段训练流程:
- 阶段一: Alpha 锚定 (t1, t6) - 20 epochs
- 阶段二: Beta 预热 (t2~t5) - 20 epochs
- 阶段三: 联合微调 (全任务) - 60 epochs

使用方法:
    # 单次训练 (5-fold)
    python src/main_mtl.py --holdout_enabled --fold 0

    # 启用 Holdout 模式
    python src/main_mtl.py --run_all_folds --holdout_enabled

    # 恢复训练
    python src/main_mtl.py --resume --stage stage3 --checkpoint models/best_mtl_stage2_fold1.pth

    # Baseline模式
    python src/main_mtl.py --mode baseline --holdout_enabled --fold 0

    # T6 Auxiliary模式
    python src/main_mtl.py --mode t6_auxiliary --holdout_enabled --fold 0

    # 实验B: Ablation B (feature KD)
    python src/main_mtl.py --mode t6_auxiliary --holdout_enabled --run_all_folds --set mtl.experiment_suffix=ablation_B_feat02
    # 消融B 变体1
    python src/main_mtl.py --mode t6_auxiliary --holdout_enabled --run_all_folds --config configs/config_mtl_ablation_B_feat02.yaml --set mtl.experiment_suffix=_v1

    # 实验E: Baseline + t6弱监督 (无context injection)
    python src/main_mtl.py --mode experiment_E --holdout_enabled --fold 0

    # 配置覆盖
    python src/main_mtl.py --mode baseline --set mtl.experiment_suffix=_test

    # Holdout-only: 跳过训练，直接评估测试集
    python src/main_mtl.py --mode baseline  --holdout_enabled --fold 0 --resume --stage holdout

    # Holdout-only:模型可解释性分析
    python src/main_mtl.py --mode t6_auxiliary --holdout_enabled --fold 4 --checkpoint "models/best_mtl_v4_t6_protectedablation_B_feat02_stage3_phase2_fold5.pth" --resume --stage holdout/
    --interpret_enabled [--interpret_save_attr,--interpret_context_counterfactual,--interpret_save_intermediates]

创建日期: 2026-04-14
更新日期: 2026-04-20 (MTL专用Holdout划分文件生成)
"""

import os
import sys
import argparse
import yaml
import json
import glob
import re
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入自定义模块
from config import Config
from task_specs import build_task_specs_from_config, validate_task_specs, TaskSpec
from model_mtl import HDSTGCNMTL, ProtectedDualEngineMTL, ProtectedDualEngineMTL_v3, ProtectedDualEngineMTL_v4
from model_mtl import (
    init_trunks_from_baseline,
    init_residual_experts_near_identity,
    init_B_t3_from_t3_best,
    init_protected_dual_engine_mtl_v4
)
from dataset_mtl import create_mtl_dataloaders, create_mtl_holdout_test_loader, load_mtl_labels, MTL_LABEL_COLUMNS
from dataset_new import preload_all_data
from train_mtl_with_swanlab import (
    build_mtl_criterions,
    MTLTotalLoss,
    MTLTrainer,
    load_single_task_checkpoint_into_mtl,
    save_mtl_checkpoint,
    evaluate_mtl_on_holdout_test,
    aggregate_mtl_holdout_metrics,
    save_fold_holdout_stats,
    save_holdout_summary,
    print_holdout_summary,
    save_holdout_prediction_table,
    save_fold_roc_json,  # [新增 2026-05-24]
)
from feature_mapping import get_nine_graph_adjacency, get_nine_graph_config
from mtl_excel_logger_v2 import create_mtl_excel_logger_v2, MTLExcelLoggerV2

# [新增 2026-06-08] Clinical Interpretation 辅助函数

TASKS = ["t1", "t2", "t3", "t4", "t5"]
ALL_TASKS = ["t1", "t2", "t3", "t4", "t5", "t6"]


def to_numpy_interpret(x):
    """Convert tensor/dict/list to numpy for interpretation output"""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_numpy_interpret(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_numpy_interpret(v) for v in x]
    return x


def to_jsonable_interpret(x):
    """Convert to JSON-serializable format"""
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable_interpret(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable_interpret(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def move_batch_to_device(batch, device):
    """Move batch to device"""
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(v.to(device) if torch.is_tensor(v) else v for v in batch)
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def get_from_batch(batch: Dict, candidates: List[str], required=True):
    """Get value from batch by candidate keys"""
    for k in candidates:
        if k in batch:
            return batch[k]
    if required:
        raise KeyError(f"batch 中找不到字段，候选名={candidates}，实际 keys={list(batch.keys())}")
    return None


def infer_logits_interpret(outputs: Dict, task: str):
    """Extract logits from outputs for a task"""
    if task not in outputs:
        raise KeyError(f"outputs 中找不到任务 {task}，实际 keys={list(outputs.keys())}")
    x = outputs[task]
    if isinstance(x, dict):
        for k in ["logits", "pred", "output"]:
            if k in x:
                return x[k]
        raise KeyError(f"outputs[{task}] 是 dict，但找不到 logits/pred/output，keys={list(x.keys())}")
    if torch.is_tensor(x):
        return x
    raise TypeError(f"outputs[{task}] 格式无法识别：{type(x)}")


def select_target_score(logits: torch.Tensor, labels=None, target_mode="pred"):
    """Select target logit for gradient computation"""
    if logits.ndim == 2 and logits.size(1) > 1:
        if target_mode == "label" and labels is not None:
            target = labels.long()
        else:
            target = logits.argmax(dim=1)
        return logits[torch.arange(logits.size(0), device=logits.device), target].sum()
    return logits.reshape(logits.size(0), -1)[:, 0].sum()


def save_npy_interpret(path: Path, arr):
    """Save numpy array"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def save_json_interpret(path: Path, obj):
    """Save JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable_interpret(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def compute_grad_x_input_for_batch(model, batch, task, device):
    """
    Compute gradient x input attribution for a batch.

    数据已经降采样为200个时间点，无需额外插值。
    返回: [B, 200, C] attribution
    """
    model.eval()
    batch = move_batch_to_device(batch, device)
    x_dyn = get_from_batch(batch, ["x_dyn", "dynamic_x", "x_dynamic", "x", "features"])
    labels = get_from_batch(batch, [f"label_{task}", task, f"y_{task}"], required=False)

    x_dyn = x_dyn.detach().clone().requires_grad_(True)
    batch2 = dict(batch)
    for key in ["x_dyn", "dynamic_x", "x_dynamic", "x", "features"]:
        if key in batch2:
            batch2[key] = x_dyn
            break

    # 调用模型 (return_intermediates=True)
    outputs = model(
        x_dyn,
        batch2.get("x_static", batch2.get("static", None)),
        lengths=None,
        return_intermediates=True,
        context_mode="normal"
    )
    logits = infer_logits_interpret(outputs, task)
    score = select_target_score(logits, labels=labels, target_mode="pred")

    model.zero_grad(set_to_none=True)
    score.backward()

    # Gradient x Input attribution: [B, 200, 30]
    attr = (x_dyn.grad * x_dyn).abs().detach().cpu().numpy()

    # 每个样本归一化到 [0, 1] (按99分位数)
    for i in range(attr.shape[0]):
        denom = np.percentile(attr[i], 99) + 1e-8
        attr[i] = np.clip(attr[i] / denom, 0, 1)

    return attr, to_numpy_interpret(outputs)


@torch.no_grad()
def collect_intermediates(model, loader, device, output_dir):
    """
    Collect intermediates from all batches (dataloader已管理batch)
    必须生成 manifest.json，即使某些字段为 None

    Args:
        model: 模型
        loader: DataLoader
        device: 设备
        output_dir: 输出目录（绝对路径）

    Returns:
        manifest: 保存的所有数组信息
    """
    print(f"[Interpret] save_intermediates=True")
    print(f"[Interpret] intermediates output_dir={output_dir}")

    collected = {
        "t1_pmgt_attn": [],
        "t6_pmgt_attn": [],
        "t1_pmgt_attn_scores": [],
        "t6_pmgt_attn_scores": [],
        "c6_deep": [],
        "dyn_feat_t6": [],
        "logits_t1": [],
        "logits_t2": [],
        "logits_t3": [],
        "logits_t4": [],
        "logits_t5": [],
        "logits_t6": [],
    }

    beta_gate_weights = {
        "t2": [],
        "t3": [],
        "t4": [],
        "t5": [],
    }

    warnings = []
    batch_keys_record = None
    n_batches_processed = 0

    model.eval()

    for bidx, batch in enumerate(loader):
        n_batches_processed = bidx + 1

        if batch_keys_record is None:
            if isinstance(batch, dict):
                batch_keys_record = list(batch.keys())
            else:
                batch_keys_record = str(type(batch))

        batch = move_batch_to_device(batch, device)
        outputs = model(
            get_from_batch(batch, ["x_dyn", "dynamic_x", "x_dynamic", "x", "features"]),
            batch.get("x_static", batch.get("static", None)),
            lengths=None,
            return_intermediates=True,
            context_mode="normal"
        )

        if bidx == 0:
            print(f"[Interpret] return_intermediates=True forward OK")
            print(f"[Interpret] outputs keys={list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}")

        if not isinstance(outputs, dict):
            warnings.append(f"batch {bidx}: model output is not dict (type={type(outputs)})")
            continue

        inter = outputs.get("intermediates", None)
        if inter is None:
            warnings.append(f"batch {bidx}: outputs['intermediates'] not found")
            if bidx == 0:
                print(f"[Interpret] WARNING: outputs['intermediates'] not found")
        else:
            if bidx == 0:
                print(f"[Interpret] intermediates keys={list(inter.keys())}")

            # 保存 PMGT attention
            for key in ["t1_pmgt_attn", "t6_pmgt_attn", "t1_pmgt_attn_scores", "t6_pmgt_attn_scores"]:
                value = inter.get(key, None)
                if value is not None and torch.is_tensor(value):
                    collected[key].append(value.detach().cpu())
                elif bidx == 0:
                    warnings.append(f"intermediates['{key}'] is None or not tensor")

            # 保存 c6_deep 和 dyn_feat_t6
            for key in ["c6_deep", "dyn_feat_t6"]:
                value = inter.get(key, None)
                if value is not None and torch.is_tensor(value):
                    collected[key].append(value.detach().cpu())
                elif bidx == 0:
                    warnings.append(f"intermediates['{key}'] is None or not tensor")

            # 保存 beta_gate_weights
            bg = inter.get("beta_gate_weights", None)
            if bg is not None:
                if isinstance(bg, dict):
                    for task in ["t2", "t3", "t4", "t5"]:
                        if task in bg and bg[task] is not None and torch.is_tensor(bg[task]):
                            beta_gate_weights[task].append(bg[task].detach().cpu())
                        elif bidx == 0:
                            warnings.append(f"beta_gate_weights['{task}'] is None or not tensor")
                elif torch.is_tensor(bg):
                    # 如果是 tensor，可能是所有任务的权重合并
                    warnings.append("beta_gate_weights is tensor, not dict (format unexpected)")
                else:
                    warnings.append(f"beta_gate_weights type={type(bg)}")
            elif bidx == 0:
                warnings.append("beta_gate_weights not found in intermediates")

        # 保存 logits
        for task in ALL_TASKS:
            try:
                logits = infer_logits_interpret(outputs, task)
                if logits is not None and torch.is_tensor(logits):
                    collected[f"logits_{task}"].append(logits.detach().cpu())
            except Exception as e:
                if bidx == 0:
                    warnings.append(f"logits_{task} extraction failed: {e}")

    # 构建 manifest
    manifest = {
        "batch_keys": batch_keys_record,
        "warnings": warnings,
        "saved_arrays": {},
        "n_batches_processed": n_batches_processed,
    }

    # 保存收集的数组
    output_dir_path = Path(output_dir)
    for key, values in collected.items():
        if len(values) > 0:
            arr = torch.cat(values, dim=0).numpy()
            np.save(str(output_dir_path / f"{key}.npy"), arr)
            manifest["saved_arrays"][key] = list(arr.shape)
        else:
            manifest["saved_arrays"][key] = None

    # 保存 beta_gate_weights
    for task, values in beta_gate_weights.items():
        out_key = f"beta_gate_weights_{task}"
        if len(values) > 0:
            arr = torch.cat(values, dim=0).numpy()
            np.save(str(output_dir_path / f"{out_key}.npy"), arr)
            manifest["saved_arrays"][out_key] = list(arr.shape)
        else:
            manifest["saved_arrays"][out_key] = None

    # 保存 manifest.json
    manifest_path = output_dir_path / "manifest.json"
    save_json_interpret(manifest_path, manifest)
    print(f"[Interpret] saved intermediates manifest to {manifest_path}")

    # 检查是否成功保存
    if not manifest_path.exists():
        raise RuntimeError(f"[Interpret] Failed to save intermediates/manifest.json to {manifest_path}")

    return manifest


@torch.no_grad()
def run_context_counterfactual(model, loader, device, context_modes="normal,zero,shuffle", thresholds=None, task_specs=None):
    """
    Run context counterfactual comparison (dataloader已管理batch)

    [新增 2026-06-10] 方案A支持：保存 t1-t5 的 projector/fused feature 用于散点图可视化

    保存内容：
    1. logits (原有): logits_{mode}_{task}.npy，所有 t1-t5
    2. projector feature (新增): t{N}_projector_{mode}.npy，t1-t5，shape [N_samples, 48]
       - 注意：t1_projector 实际上是 t6-guided 后的 dyn_feat，不是 Beta FlattenProjector 输出
       - t2-t5 是 Beta 任务的 FlattenProjector 输出
    3. fused feature (新增): t{N}_fused_{mode}.npy，t1-t5，shape [N_samples, 64]
    4. t1 特殊中间表征 (新增): t1_base_{mode}.npy, t1_context_delta_{mode}.npy
    5. labels (新增): labels_t{task}.npy，所有 t1-t6
    6. sample_ids (新增): sample_ids.npy 或 sample_ids.txt，用于核对样本顺序

    注意：
    - t1/t6 是 Alpha 任务，没有 Beta FlattenProjector
    - t1_projector_feat = t6-guided dyn_feat (用于散点图，命名兼容 t2-t5)
    - t1_base_feat = 注入前的 PMGT 输出
    - t1_context_delta = t6-to-t1 bridge 注入的增量
    """
    modes = [m.strip() for m in context_modes.split(",") if m.strip()]

    # 存储结构：logits
    logits_store = {m: {t: [] for t in TASKS} for m in modes}

    # 存储结构：projector (t1-t5)
    # t1 的 projector 实际上是 t6-guided feature
    projector_store = {m: {t: [] for t in ["t1", "t2", "t3", "t4", "t5"]} for m in modes}

    # 存储结构：fused (t1-t5)
    fused_store = {m: {t: [] for t in ["t1", "t2", "t3", "t4", "t5"]} for m in modes}

    # 存储结构：t1 特殊中间表征
    t1_base_store = {m: [] for m in modes}  # 注入前的 base feature
    t1_delta_store = {m: [] for m in modes}  # t6-to-t1 bridge 注入的增量

    # 存储标签和 sample_ids
    labels_store = {t: [] for t in ["t1", "t2", "t3", "t4", "t5", "t6"]}
    sample_ids_store = []

    for bidx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        # 收集标签（每个 batch 只收集一次，与 mode 无关）
        labels_dict = batch.get("labels", {})
        for task in ["t1", "t2", "t3", "t4", "t5", "t6"]:
            if task in labels_dict:
                labels_store[task].append(to_numpy_interpret(labels_dict[task]))

        # 收集 sample_ids（如果 batch 中有）
        if "sample_id" in batch:
            sample_ids_store.extend(batch["sample_id"])
        elif "filename" in batch:
            sample_ids_store.extend(batch["filename"])

        # 遍历三种 context_mode
        for mode in modes:
            outputs = model(
                get_from_batch(batch, ["x_dyn", "dynamic_x", "x_dynamic", "x", "features"]),
                batch.get("x_static", batch.get("static", None)),
                lengths=None,
                return_intermediates=True,
                context_mode=mode
            )

            # 收集 logits（所有 t1-t5）
            for task in TASKS:
                logits_store[mode][task].append(to_numpy_interpret(infer_logits_interpret(outputs, task)))

            # 收集 intermediates（包含 t1 特殊表征）
            inter = outputs.get("intermediates", {})

            # 收集 t1 中间表征（从 intermediates）
            t1_base = inter.get("t1_base_feat", None)
            t1_projector = inter.get("t1_projector_feat", None)  # guided feature
            t1_fused = inter.get("t1_fused_feat", None)
            t1_delta = inter.get("t1_context_delta", None)

            if t1_base is not None:
                t1_base_store[mode].append(to_numpy_interpret(t1_base))
            if t1_projector is not None:
                projector_store[mode]["t1"].append(to_numpy_interpret(t1_projector))
            if t1_fused is not None:
                fused_store[mode]["t1"].append(to_numpy_interpret(t1_fused))
            if t1_delta is not None:
                t1_delta_store[mode].append(to_numpy_interpret(t1_delta))

            # 收集 t2-t5 projector (dyn_feat) 和 fused feature（从 outputs）
            for task in ["t2", "t3", "t4", "t5"]:
                if task in outputs and isinstance(outputs[task], dict):
                    # projector feature = dyn_feat (FlattenProjector 输出)
                    dyn_feat = outputs[task].get("dyn_feat", None)
                    if dyn_feat is not None:
                        projector_store[mode][task].append(to_numpy_interpret(dyn_feat))

                    # fused feature = torch.cat([dyn_feat, static_feat], dim=1)
                    fused_feat = outputs[task].get("fused_feat", None)
                    if fused_feat is not None:
                        fused_store[mode][task].append(to_numpy_interpret(fused_feat))

    # 保存文件
    manifest = {"modes": modes, "tasks": TASKS, "alpha_tasks": ["t1", "t6"], "beta_tasks": ["t2", "t3", "t4", "t5"]}
    delta_summary = {}

    # 1. 保存 logits（原有逻辑）
    for mode in modes:
        manifest[mode] = {}
        for task in TASKS:
            arr = np.concatenate(logits_store[mode][task], axis=0)
            save_npy_interpret(Path(f"logits_{mode}_{task}.npy"), arr)
            manifest[mode][f"logits_{task}"] = list(arr.shape)

    # 2. 保存 t1 base feature（新增，注入前的 PMGT 输出）
    for mode in modes:
        if len(t1_base_store[mode]) > 0:
            arr = np.concatenate(t1_base_store[mode], axis=0)
            save_npy_interpret(Path(f"t1_base_{mode}.npy"), arr)
            manifest[mode]["t1_base"] = list(arr.shape)
            print(f"[Interpret] Saved t1_base_{mode}.npy shape={arr.shape}")
        else:
            manifest[mode]["t1_base"] = None
            print(f"[Interpret] WARNING: t1_base_{mode} is empty (t6 injection may be disabled)")

    # 3. 保存 t1 context delta（新增，t6-to-t1 bridge 注入的增量）
    for mode in modes:
        if len(t1_delta_store[mode]) > 0:
            arr = np.concatenate(t1_delta_store[mode], axis=0)
            save_npy_interpret(Path(f"t1_context_delta_{mode}.npy"), arr)
            manifest[mode]["t1_context_delta"] = list(arr.shape)
            print(f"[Interpret] Saved t1_context_delta_{mode}.npy shape={arr.shape}")
        else:
            manifest[mode]["t1_context_delta"] = None

    # 4. 保存 projector feature（新增，t1-t5）
    # t1 的 projector 实际上是 t6-guided feature
    for mode in modes:
        for task in ["t1", "t2", "t3", "t4", "t5"]:
            if len(projector_store[mode][task]) > 0:
                arr = np.concatenate(projector_store[mode][task], axis=0)
                save_npy_interpret(Path(f"t{task[1:] if task != 't1' else '1'}_projector_{mode}.npy"), arr)
                manifest[mode][f"projector_{task}"] = list(arr.shape)
                print(f"[Interpret] Saved t{task[1:] if task != 't1' else '1'}_projector_{mode}.npy shape={arr.shape}")
            else:
                manifest[mode][f"projector_{task}"] = None
                if task == "t1":
                    print(f"[Interpret] WARNING: t1_projector_{mode} is empty (t6 injection may be disabled)")
                else:
                    print(f"[Interpret] WARNING: t{task[1:]}_projector_{mode} is empty")

    # 5. 保存 fused feature（新增，t1-t5）
    for mode in modes:
        for task in ["t1", "t2", "t3", "t4", "t5"]:
            if len(fused_store[mode][task]) > 0:
                arr = np.concatenate(fused_store[mode][task], axis=0)
                save_npy_interpret(Path(f"t{task[1:] if task != 't1' else '1'}_fused_{mode}.npy"), arr)
                manifest[mode][f"fused_{task}"] = list(arr.shape)
                print(f"[Interpret] Saved t{task[1:] if task != 't1' else '1'}_fused_{mode}.npy shape={arr.shape}")
            else:
                manifest[mode][f"fused_{task}"] = None

    # 6. 保存 labels（新增，t1-t6）
    for task in ["t1", "t2", "t3", "t4", "t5", "t6"]:
        if len(labels_store[task]) > 0:
            arr = np.concatenate(labels_store[task], axis=0)
            save_npy_interpret(Path(f"labels_{task}.npy"), arr)
            manifest[f"labels_{task}"] = list(arr.shape)
            print(f"[Interpret] Saved labels_{task}.npy shape={arr.shape}")
        else:
            manifest[f"labels_{task}"] = None

    # 7. 保存 sample_ids（新增）
    if len(sample_ids_store) > 0:
        # 如果是 list，转为 numpy array
        if isinstance(sample_ids_store[0], str):
            # 字符串列表，保存为文本文件
            with open("sample_ids.txt", "w", encoding="utf-8") as f:
                for sid in sample_ids_store:
                    f.write(sid + "\n")
            manifest["sample_ids_file"] = "sample_ids.txt"
            manifest["n_samples"] = len(sample_ids_store)
        else:
            arr = np.array(sample_ids_store)
            save_npy_interpret(Path("sample_ids.npy"), arr)
            manifest["sample_ids"] = list(arr.shape)
        print(f"[Interpret] Saved sample_ids with {len(sample_ids_store)} samples")
    else:
        manifest["sample_ids"] = None
        print(f"[Interpret] WARNING: sample_ids not found in batch")

    # 8. 计算 delta summary（logits 级别）
    for task in TASKS:
        if "normal" in modes and "zero" in modes:
            normal = np.concatenate(logits_store["normal"][task], axis=0)
            zero = np.concatenate(logits_store["zero"][task], axis=0)
            delta_summary[task] = {
                "delta_normal_minus_zero_abs_mean": float(np.mean(np.abs(normal - zero))),
                "delta_normal_minus_zero_abs_median": float(np.median(np.abs(normal - zero))),
            }
        if "normal" in modes and "shuffle" in modes:
            normal = np.concatenate(logits_store["normal"][task], axis=0)
            shuffle = np.concatenate(logits_store["shuffle"][task], axis=0)
            if task not in delta_summary:
                delta_summary[task] = {}
            delta_summary[task]["delta_normal_minus_shuffle_abs_mean"] = float(np.mean(np.abs(normal - shuffle)))
            delta_summary[task]["delta_normal_minus_shuffle_abs_median"] = float(np.median(np.abs(normal - shuffle)))

    save_json_interpret(Path("manifest.json"), manifest)
    save_json_interpret(Path("delta_summary.json"), delta_summary)

    print(f"[Interpret] Context counterfactual completed")
    print(f"[Interpret] Total samples: {manifest.get('n_samples', 'unknown')}")

    return delta_summary


def run_holdout_interpretation(
    model,
    holdout_loader,
    config,
    mtl_config,
    device,
    fold_idx,
    output_dir,
    thresholds,
    task_specs,
    checkpoint_path,
    mode,
    interpret_save_intermediates=False,
    interpret_save_attr=False,
    interpret_context_counterfactual=False,
    interpret_context_modes="normal,zero,shuffle"
):
    """
    使用 main_mtl.py 已经构建好的 model / holdout_loader / config / thresholds，
    在完全相同 holdout 样本上执行 clinical interpretation。

    数据已经通过 dataloader 完成 batch 管理，所有序列降采样为200个时间点。
    """
    import shutil
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存 run_meta.json
    run_meta = {
        "mode": mode,
        "stage": "holdout",
        "fold": fold_idx + 1,
        "checkpoint": checkpoint_path,
        "split": "holdout",
        "n_samples": len(holdout_loader.dataset),
        "n_time_points": 200,  # 固定降采样时间点数
        "thresholds": thresholds,
        "interpret_save_intermediates": interpret_save_intermediates,
        "interpret_save_attr": interpret_save_attr,
        "interpret_context_counterfactual": interpret_context_counterfactual,
        "interpret_context_modes": interpret_context_modes,
        "batch_keys": list(holdout_loader.dataset[0].keys()) if hasattr(holdout_loader.dataset, '__getitem__') else [],
        "model_signature": {
            "has_t6_deep_context_module": hasattr(model, 't6_deep_context_module') and model.t6_deep_context_module is not None,
            "t6_deep_context_enabled": getattr(model, 't6_deep_context_enabled', False),
            "beta_gate_context_dim": getattr(model, 'beta_gate_context', None) and hasattr(model.beta_gate_context, 'output_dim') and model.beta_gate_context.output_dim if model.beta_gate_context else 0,
        }
    }
    save_json_interpret(output_path / "run_meta.json", run_meta)
    print(f"[Interpretation] Saved run_meta.json")

    # 1. Variable-time attribution (200个固定时间点)
    if interpret_save_attr:
        print(f"[Interpretation] Computing variable-time attribution...")
        attr_dir = (output_path / "variable_time_attr").resolve()  # 转为绝对路径
        attr_dir.mkdir(exist_ok=True)
        # 切换工作目录到 attr_dir
        old_cwd = os.getcwd()
        os.chdir(str(attr_dir))

        task_values = {t: [] for t in TASKS}
        for bidx, batch in enumerate(holdout_loader):
            for task in TASKS:
                attr, _ = compute_grad_x_input_for_batch(model, batch, task=task, device=device)
                task_values[task].append(attr)

        summary = {}
        for task in TASKS:
            arr = np.concatenate(task_values[task], axis=0)  # [N, 200, C]
            save_npy_interpret(attr_dir / f"{task}_all_samples.npy", arr)
            mean_arr = arr.mean(axis=0)
            save_npy_interpret(attr_dir / f"{task}_mean.npy", mean_arr)
            summary[task] = {
                "n_samples": int(arr.shape[0]),
                "shape_all": list(arr.shape),
                "shape_mean": list(mean_arr.shape),
                "top_channels_by_mean_attr": np.argsort(-mean_arr.mean(axis=0))[:10].tolist(),
            }
        save_json_interpret(attr_dir / "summary.json", summary)

        # [新增] 生成特征名称映射文件
        nine_config = get_nine_graph_config()
        feature_names = nine_config['feature_names']
        channel_mapping = {
            "description": "VarIndex 到特征名称的映射 (30个CPET特征)",
            "adapt_mode": "nine_graph",
            "n_features": len(feature_names),
            "mapping": {str(i): name for i, name in enumerate(feature_names)},
            "top_channels_interpretation": {
                task: {
                    "indices": summary[task]["top_channels_by_mean_attr"],
                    "names": [feature_names[i] for i in summary[task]["top_channels_by_mean_attr"]]
                }
                for task in TASKS
            }
        }
        save_json_interpret(attr_dir / "channel_name_mapping.json", channel_mapping)
        print(f"[Interpretation] Saved channel_name_mapping.json with {len(feature_names)} feature names")

        os.chdir(old_cwd)
        print(f"[Interpretation] Variable-time attribution completed")

    # 2. Collect intermediates
    if interpret_save_intermediates:
        print(f"[Interpretation] Collecting intermediates...")
        inter_dir = (output_path / "intermediates").resolve()  # 转为绝对路径
        inter_dir.mkdir(parents=True, exist_ok=True)

        manifest = collect_intermediates(model, holdout_loader, device, inter_dir)
        print(f"[Interpretation] Intermediates collection completed")

    # 3. Context counterfactual
    if interpret_context_counterfactual:
        print(f"[Interpretation] Running context counterfactual...")
        cf_dir = (output_path / "context_counterfactual").resolve()  # 转为绝对路径
        cf_dir.mkdir(exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(str(cf_dir))

        delta_summary = run_context_counterfactual(
            model, holdout_loader, device,
            context_modes=interpret_context_modes,
            thresholds=thresholds,
            task_specs=task_specs
        )
        os.chdir(old_cwd)
        print(f"[Interpretation] Context counterfactual completed")
        print(f"[Interpretation] Delta summary: {delta_summary}")

    # [新增 2026-06-10] 更新 run_meta.json，包含新增文件的 shape
    # 从各子目录读取 manifest.json，更新 run_meta
    run_meta_updated = run_meta.copy()

    # 更新 variable_time_attr 的 shape 信息
    if interpret_save_attr:
        attr_manifest_path = output_path / "variable_time_attr" / "summary.json"
        if attr_manifest_path.exists():
            with open(attr_manifest_path, 'r', encoding='utf-8') as f:
                attr_manifest = json.load(f)
            run_meta_updated["variable_time_attr_shapes"] = {
                task: attr_manifest.get(task, {}).get("shape_all")
                for task in TASKS
            }

    # 更新 intermediates 的 shape 信息
    if interpret_save_intermediates:
        inter_manifest_path = output_path / "intermediates" / "manifest.json"
        if inter_manifest_path.exists():
            with open(inter_manifest_path, 'r', encoding='utf-8') as f:
                inter_manifest = json.load(f)
            run_meta_updated["intermediates_shapes"] = inter_manifest.get("saved_arrays", {})

    # 更新 context_counterfactual 的 shape 信息（包含新增的 projector/fused）
    if interpret_context_counterfactual:
        cf_manifest_path = output_path / "context_counterfactual" / "manifest.json"
        if cf_manifest_path.exists():
            with open(cf_manifest_path, 'r', encoding='utf-8') as f:
                cf_manifest = json.load(f)
            run_meta_updated["context_counterfactual_shapes"] = cf_manifest

            # 提取新增的 projector/fused feature shape（方案A专用）
            projector_shapes = {}
            fused_shapes = {}
            for mode in interpret_context_modes.split(","):
                mode = mode.strip()
                if mode in cf_manifest:
                    for task in ["t2", "t3", "t4", "t5"]:
                        projector_key = f"projector_{task}"
                        fused_key = f"fused_{task}"
                        if projector_key in cf_manifest[mode]:
                            projector_shapes[f"{task}_{mode}"] = cf_manifest[mode][projector_key]
                        if fused_key in cf_manifest[mode]:
                            fused_shapes[f"{task}_{mode}"] = cf_manifest[mode][fused_key]

            run_meta_updated["projector_feature_shapes"] = projector_shapes
            run_meta_updated["fused_feature_shapes"] = fused_shapes

            # 提取 labels shape
            labels_shapes = {}
            for task in ["t2", "t3", "t4", "t5", "t6"]:
                labels_key = f"labels_{task}"
                if labels_key in cf_manifest:
                    labels_shapes[task] = cf_manifest[labels_key]
            run_meta_updated["labels_shapes"] = labels_shapes

    # 重新保存 run_meta.json（包含更新的 shape 信息）
    save_json_interpret(output_path / "run_meta.json", run_meta_updated)
    print(f"[Interpretation] Updated run_meta.json with feature shapes")

    print(f"[Interpretation] All interpretation tasks completed")


def load_holdout_split_info_mtl(
    config_path: str = None,
    results_dir: str = "results",
    auto_find_latest: bool = True,
    strict_mtl_mode: bool = True  # [新增] 严格模式：禁止使用单任务划分文件
) -> dict:
    """
    加载 MTL 专用的 Holdout 划分信息 (holdout_split_info_mtl.json)

    Args:
        config_path: 配置文件中指定的划分文件路径
        results_dir: 结果目录
        auto_find_latest: 是否自动查找最新划分文件
        strict_mtl_mode: [新增] 严格模式，禁止使用单任务划分文件 (默认 True)

    Returns:
        split_info: 完整划分信息字典，必须包含 dev_filenames/test_filenames
    """
    # 优先查找 MTL 专用划分文件
    mtl_split_file = os.path.join(results_dir, "holdout_split_info_mtl.json")

    if config_path and config_path != "auto":
        split_file = config_path
    elif os.path.exists(mtl_split_file):
        split_file = mtl_split_file
        print(f"[Holdout MTL] 使用 MTL 专用划分文件: {split_file}")
    elif auto_find_latest:
        # 尝试查找其他划分文件 (MTL 专用)
        pattern = os.path.join(results_dir, "holdout_split_info_mtl*.json")
        split_files = glob.glob(pattern)

        if not split_files:
            # ========== [修复] 严格模式下禁止使用单任务划分文件 ==========
            if strict_mtl_mode:
                raise FileNotFoundError(
                    f"\n[Holdout MTL Error] 未找到 MTL 专用划分文件!\n"
                    f"  - 搜索路径: {results_dir}/holdout_split_info_mtl*.json\n"
                    f"  - 解决方案:\n"
                    f"    1. 运行 'python src/main_mtl.py --holdout_enabled --run_all_folds' 生成 MTL 专用划分文件\n"
                    f"    2. 或在 configs/config_mtl.yaml 中设置 mtl.holdout.enabled=false 禁用 Holdout 模式\n"
                    f"  - 注意: 单任务划分文件 (holdout_split_info.json) 与 MTL 样本数不一致，会导致索引错误\n"
                )
            # else:
            #     # [兼容旧逻辑] 仅在非严格模式下回退查找单任务划分文件
            #     pattern = os.path.join(results_dir, "holdout_split_info.json")
            #     split_files = glob.glob(pattern)
            #     if split_files:
            #         print(f"[Holdout MTL Warning] 未找到 MTL 专用划分文件，使用单任务划分文件")
            #         print(f"  [警告] 这可能导致边界检查失败，建议运行 MTL Holdout 刭分生成")
            #     else:
            #         # 尝试查找其他划分文件
            #         pattern = os.path.join(results_dir, "holdout_split_info*.json")
            #         split_files = glob.glob(pattern)
            #         # 排除特定任务的划分文件
            #         split_files = [f for f in split_files if 'class_' not in f and '_wo_' not in f and '_w_' not in f]
            # ============================================================

        if not split_files:
            raise FileNotFoundError(f"未找到 MTL Holdout 划分文件: 请先运行 MTL Holdout 模式生成划分文件")

        # 按修改时间排序，选择最新的
        split_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        split_file = split_files[0]
        print(f"[Holdout MTL] 自动选择划分文件: {split_file}")
    else:
        raise ValueError("必须指定划分文件路径或启用 auto_find_latest")

    with open(split_file, 'r', encoding='utf-8') as f:
        split_info = json.load(f)

    dev_filenames = split_info.get('dev_filenames', [])
    test_filenames = split_info.get('test_filenames', [])

    if not dev_filenames or not test_filenames:
        raise ValueError(
            f"划分文件缺少 dev_filenames 或 test_filenames: {split_file}\n"
            f"[Holdout MTL] 当前版本不再读取 dev_indices/test_indices，请重新生成基于 filenames 的划分文件。"
        )

    print(f"[Holdout MTL] Dev_Set: {len(dev_filenames)} 样本, Test_Set: {len(test_filenames)} 样本")

    # 检查是否是 MTL 专用划分文件
    mode = split_info.get('mode', 'unknown')
    if mode != 'MTL':
        if strict_mtl_mode:
            raise ValueError(
                f"\n[Holdout MTL Error] 划分文件不是 MTL 专用文件!\n"
                f"  - 文件: {split_file}\n"
                f"  - mode: {mode}\n"
                f"  - 解决方案: 运行 'python src/main_mtl.py --holdout_enabled --run_all_folds' 生成 MTL 专用划分文件\n"
            )
        else:
            print(f"[Holdout MTL Warning] 划分文件 mode={mode}, 不是 MTL 专用文件")
            print(f"  [警告] 建议重新运行 MTL Holdout 训练生成专用划分文件")

    split_info['_split_file'] = split_file
    return split_info


def map_holdout_split_filenames_to_indices(
    split_info: dict,
    loaded_filenames: List[str]
) -> Tuple[List[int], List[int]]:
    """
    Map filename-based MTL holdout split fields to current in-memory indices.

    The JSON split file treats dev_filenames/test_filenames as the only
    persistent split authority. dev_indices/test_indices are intentionally not
    read from disk because their meaning depends on the current loaded dataset.
    """
    dev_filenames_raw = split_info.get('dev_filenames', [])
    test_filenames_raw = split_info.get('test_filenames', [])

    if not isinstance(dev_filenames_raw, list) or not isinstance(test_filenames_raw, list):
        raise ValueError("[Holdout MTL] dev_filenames/test_filenames 必须是列表")

    dev_filenames = [str(name).strip() for name in dev_filenames_raw if str(name).strip()]
    test_filenames = [str(name).strip() for name in test_filenames_raw if str(name).strip()]

    if not dev_filenames or not test_filenames:
        raise ValueError("[Holdout MTL] dev_filenames/test_filenames 不能为空")

    loaded_filename_to_idx = {}
    duplicate_loaded = []
    for idx, filename in enumerate(loaded_filenames):
        key = str(filename).strip()
        if key in loaded_filename_to_idx:
            duplicate_loaded.append(key)
        else:
            loaded_filename_to_idx[key] = idx

    if duplicate_loaded:
        preview = duplicate_loaded[:10]
        raise ValueError(
            f"[Holdout MTL] 当前加载数据中存在重复 filename，无法唯一映射: {preview}"
        )

    overlap = sorted(set(dev_filenames) & set(test_filenames))
    if overlap:
        raise ValueError(
            f"[Holdout MTL] dev_filenames 与 test_filenames 存在重叠，示例: {overlap[:10]}"
        )

    def map_one_split(split_name: str, filenames: List[str]) -> List[int]:
        missing = [name for name in filenames if name not in loaded_filename_to_idx]
        if missing:
            split_file = split_info.get('_split_file', 'unknown')
            raise ValueError(
                f"[Holdout MTL] {split_name} 中有 {len(missing)} 个 filename "
                f"无法在当前加载成功的数据中找到。\n"
                f"  - split_file: {split_file}\n"
                f"  - missing examples: {missing[:20]}\n"
                f"请检查 data_root、label_file、数据读取筛选规则是否与生成 split 时一致。"
            )
        return [loaded_filename_to_idx[name] for name in filenames]

    dev_indices = map_one_split("dev_filenames", dev_filenames)
    test_indices = map_one_split("test_filenames", test_filenames)

    print(
        f"[Holdout MTL] 已按 filename 映射到当前数据索引: "
        f"Dev_Set={len(dev_indices)}, Test_Set={len(test_indices)}, "
        f"Loaded={len(loaded_filenames)}"
    )

    return dev_indices, test_indices


def create_mtl_holdout_split(
    config: Config,
    mtl_config: dict,
    holdout_ratio: float = 0.2,
    holdout_seed: int = 42,
    output_dir: str = "results",
    force_regenerate: bool = False
) -> dict:
    """
    创建 MTL 专用的 Holdout 划分文件

    参考 train_with_swanlab.py 中单任务模式的实现，
    为 MTL 模式生成专用的划分文件 (holdout_split_info_mtl.json)

    Args:
        config: 基础 Config (用于数据路径)
        mtl_config: MTL 配置字典
        holdout_ratio: Holdout 测试集比例 (默认 0.2)
        holdout_seed: Holdout 划分随机种子 (默认 42)
        output_dir: 输出目录 (默认 results)
        force_regenerate: 是否强制重新生成 (默认 False，已存在则跳过)

    Returns:
        split_info: 完整划分信息字典，仅持久化 dev_filenames/test_filenames
    """
    print("\n" + "="*80)
    print("[MTL Holdout] 生成 MTL 专用划分文件")
    print("="*80)

    # 检查是否已存在划分文件
    split_file = os.path.join(output_dir, "holdout_split_info_mtl.json")
    if os.path.exists(split_file) and not force_regenerate:
        print(f"[MTL Holdout] 划分文件已存在: {split_file}")
        print(f"  使用现有划分文件 (如需重新生成，请删除该文件或设置 force_regenerate=True)")

        # 加载现有划分文件
        with open(split_file, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        dev_filenames = split_info.get('dev_filenames', [])
        test_filenames = split_info.get('test_filenames', [])
        if not dev_filenames or not test_filenames:
            raise ValueError(
                f"划分文件缺少 dev_filenames 或 test_filenames: {split_file}"
            )
        print(f"[MTL Holdout] Dev_Set: {len(dev_filenames)} 样本, Test_Set: {len(test_filenames)} 样本")
        split_info['_split_file'] = split_file
        return split_info

    # ========== Step 1: 加载 MTL 标签和数据文件 ==========
    print("\n[Step 1] 加载 MTL 标签和数据文件...")

    label_file = config.data.label_file
    data_root = config.data.data_root

    # 加载 MTL 标签
    filename_to_labels, task_label_mappings, task_class_counts = load_mtl_labels(
        label_file,
        min_label_freq=50
    )

    print(f"  - MTL 任务数: {len(task_label_mappings)}")
    for task_key, mapping in task_label_mappings.items():
        print(f"    {task_key}: {len(mapping)} 类")

    # 扫描数据文件
    data_files = sorted([f for f in os.listdir(data_root) if f.endswith('.xlsx')])

    print(f"  - 数据文件总数: {len(data_files)}")

    # ========== Step 2: 匹配文件名与标签 (生成 filename 候选列表) ==========
    print("\n[Step 2] 匹配文件名与 MTL 标签...")

    valid_filenames = []

    for filename in data_files:  # data_files 已排序，与 dataset_new.py 顺序一致
        # MTL 模式需要至少一个任务有标签
        label_data = filename_to_labels.get(filename)
        if label_data is None:
            continue

        valid_filenames.append(filename)

    n_samples = len(valid_filenames)
    print(f"  - 有效样本数: {n_samples}")

    if n_samples != len(filename_to_labels):
        label_filenames = set(filename_to_labels.keys())
        data_filenames = set(data_files)
        labels_without_data = sorted(label_filenames - data_filenames)
        data_without_labels = sorted(data_filenames - label_filenames)
        stem_to_data_filename = {
            os.path.splitext(name)[0]: name
            for name in data_files
        }

        print("\n[Step 2 诊断] MTL 标签样本数与有效样本数不一致")
        print(f"  - 标签表样本数: {len(filename_to_labels)}")
        print(f"  - 有效样本数: {n_samples}")
        print(f"  - 标签表有、但数据目录无精确同名 .xlsx 的样本数: {len(labels_without_data)}")
        print(f"  - 数据目录有、但 MTL 标签表无精确同名记录的文件数: {len(data_without_labels)}")

        if labels_without_data:
            print("  - 无效样本明细（标签表存在，但未进入有效样本；按精确文件名匹配）:")
            max_report = 50
            for bad_filename in labels_without_data[:max_report]:
                task_labels = filename_to_labels.get(bad_filename, {})
                label_preview = ", ".join(
                    f"{task_key}={label_value}"
                    for task_key, label_value in sorted(task_labels.items())
                )
                same_stem = stem_to_data_filename.get(os.path.splitext(bad_filename)[0])
                same_stem_msg = f", same_stem_data_file={same_stem}" if same_stem else ""
                print(f"    - filename={bad_filename}, labels={{ {label_preview} }}{same_stem_msg}")
            if len(labels_without_data) > max_report:
                print(f"    ... 还有 {len(labels_without_data) - max_report} 个未显示")

    if n_samples == 0:
        raise ValueError("没有找到有效的 MTL 样本!")

    # ========== Step 3: Holdout 划分 (使用内部索引) ==========
    print("\n[Step 3] 执行 Holdout 划分...")

    n_test = int(n_samples * holdout_ratio)
    n_dev = n_samples - n_test

    # 随机打乱内部索引 (0 到 n_samples-1)
    np.random.seed(holdout_seed)
    all_internal_indices = np.arange(n_samples)
    shuffled_internal_indices = np.random.permutation(all_internal_indices)

    # 划分 Dev_Set 和 Test_Set (内部索引)
    test_internal_indices = shuffled_internal_indices[:n_test].tolist()
    dev_internal_indices = shuffled_internal_indices[n_test:].tolist()

    print(f"  - Dev_Set: {n_dev} 样本 ({n_dev/n_samples*100:.1f}%)")
    print(f"  - Test_Set: {n_test} 样本 ({n_test/n_samples*100:.1f}%)")
    print(f"  - 最大 dev_idx: {max(dev_internal_indices)}, 最大 test_idx: {max(test_internal_indices)}")

    # ========== Step 4: 统计各任务的类别分布 ==========
    print("\n[Step 4] 统计各任务类别分布...")

    class_distribution = {}

    for task_key, mapping in task_label_mappings.items():
        dev_counts = {}
        test_counts = {}

        # 使用内部索引获取文件名，再从文件名获取标签
        for internal_idx in dev_internal_indices:
            filename = valid_filenames[internal_idx]
            label_data = filename_to_labels.get(filename)
            if label_data and task_key in label_data:
                label_name = label_data[task_key]
                dev_counts[label_name] = dev_counts.get(label_name, 0) + 1

        for internal_idx in test_internal_indices:
            filename = valid_filenames[internal_idx]
            label_data = filename_to_labels.get(filename)
            if label_data and task_key in label_data:
                label_name = label_data[task_key]
                test_counts[label_name] = test_counts.get(label_name, 0) + 1

        class_distribution[task_key] = {
            'dev': dev_counts,
            'test': test_counts
        }

        print(f"  {task_key}:")
        print(f"    Dev: {dev_counts}")
        print(f"    Test: {test_counts}")

    # ========== Step 5: 保存划分信息 ==========
    print("\n[Step 5] 保存 MTL 专用划分信息...")

    os.makedirs(output_dir, exist_ok=True)

    dev_filenames = [valid_filenames[i] for i in dev_internal_indices]
    test_filenames = [valid_filenames[i] for i in test_internal_indices]

    split_info = {
        "schema_version": "mtl_holdout_filename_only_v1",
        "holdout_enabled": True,
        "holdout_ratio": holdout_ratio,
        "holdout_seed": holdout_seed,
        "kfold_seed": mtl_config.get('training', {}).get('random_seed', 3407),
        "n_samples": n_samples,
        "n_dev": n_dev,
        "n_test": n_test,
        "split_unit": "filename",
        "indices_persisted": False,
        "dev_filenames": dev_filenames,
        "test_filenames": test_filenames,
        "label_mapping": task_label_mappings,
        "class_distribution": class_distribution,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "MTL",  # 标识为 MTL 专用划分文件
        "tasks": list(task_label_mappings.keys())
    }

    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"  输出文件: {split_file}")

    # ========== Step 6: 打印总结 ==========
    print("\n" + "="*80)
    print("[MTL Holdout] 划分完成")
    print("="*80)
    print(f"总样本数: {n_samples}")
    print(f"Dev_Set: {n_dev} 样本 (用于 K-Fold)")
    print(f"Test_Set: {n_test} 样本 (独立测试集)")
    print("划分单位: filename")
    print(f"输出文件: {split_file}")
    print("="*80 + "\n")

    split_info['_split_file'] = split_file
    return split_info


def deep_merge_configs(base_config: dict, override_config: dict) -> dict:
    """
    深度合并两个配置字典

    Args:
        base_config: 基础配置字典
        override_config: 覆盖配置字典

    Returns:
        merged_config: 合并后的配置字典

    规则:
        - override_config 中的值覆盖 base_config
        - 对于字典类型，进行递归深度合并
        - 对于列表类型，直接替换 (不合并)
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # 递归深度合并字典
            merged[key] = deep_merge_configs(merged[key], value)
        else:
            # 直接覆盖 (包括列表、字符串、数值等)
            merged[key] = value

    return merged


def load_mtl_config_with_mode(mode: str = "baseline", config_dir: str = None, experiment_suffix: str = None) -> dict:
    """
    根据 mode 加载并合并 MTL 配置

    Args:
        mode: 运行模式
            - "baseline": Baseline 模式 (t6 完全剔除)
            - "t6_auxiliary": T6 Auxiliary 模式 (t6 上下文辅助)
            - "experiment_E": 实验E (Baseline + t6 弱监督, 无 context injection)
        config_dir: 配置文件目录 (默认 configs/)
        experiment_suffix: [新增] 实验后缀，用于自动加载消融配置文件
            - "ablation_A_logitsonly": 加载 config_mtl_ablation_A_logitsonly.yaml
            - "ablation_B_feat02": 加载 config_mtl_ablation_B_feat02.yaml
            - "ablation_C_feat01": 加载 config_mtl_ablation_C_feat01.yaml
            - "ablation_D_feat03": 加载 config_mtl_ablation_D_feat03.yaml

    Returns:
        mtl_config: 合并后的完整配置字典

    Raises:
        ValueError: mode 参数无效
        FileNotFoundError: 配置文件不存在
    """
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')

    # 验证 mode 参数
    # [修改 2026-05-27] 添加 CGC, SharedBottom, AdaTT clean 模式
    valid_modes = ["baseline", "t6_auxiliary", "experiment_E", "mmoe_clean", "cgc_clean", "shared_bottom_clean", "adatt_clean"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    # 构建配置文件路径
    base_config_path = os.path.join(config_dir, 'config_mtl_base.yaml')
    # Clean 架构模式: 使用 configs/clean/ 目录下的专用配置文件
    if mode.endswith("_clean"):
        mode_config_path = os.path.join(config_dir, f'clean/{mode}.yaml')
    # 实验E: 使用 baseline 基础配置 + experiment_E 专属覆盖
    elif mode == "experiment_E":
        mode_config_path = os.path.join(config_dir, 'config_mtl_experiment_E.yaml')
    else:
        mode_config_path = os.path.join(config_dir, f'config_mtl_{mode}.yaml')

    # 检查文件存在性
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
    if not os.path.exists(mode_config_path):
        raise FileNotFoundError(f"Mode config file not found: {mode_config_path}")

    # 加载配置文件
    print(f"\n[Config] Loading mode: {mode}")
    print(f"[Config] Base config: {base_config_path}")
    print(f"[Config] Mode config: {mode_config_path}")

    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    with open(mode_config_path, 'r', encoding='utf-8') as f:
        mode_config = yaml.safe_load(f)

    # 深度合并配置
    merged_config = deep_merge_configs(base_config, mode_config)

    # [新增] 检查是否需要加载消融配置文件
    ablation_config_path = None
    if experiment_suffix and experiment_suffix.startswith("ablation_"):
        ablation_config_name = f"config_mtl_{experiment_suffix}.yaml"
        ablation_config_path = os.path.join(config_dir, ablation_config_name)

        if os.path.exists(ablation_config_path):
            print(f"[Config] Ablation config detected: {ablation_config_path}")
            with open(ablation_config_path, 'r', encoding='utf-8') as f:
                ablation_config = yaml.safe_load(f)

            # 深度合并消融配置
            merged_config = deep_merge_configs(merged_config, ablation_config)
            print(f"[Config] Ablation config merged: {experiment_suffix}")
        else:
            print(f"[Config WARNING] Ablation config file not found: {ablation_config_path}")
            print(f"[Config] Will use --set overrides instead")

    # 验证关键配置
    t6_aux_enabled = merged_config.get('mtl', {}).get('hcgc_v4', {}).get('t6_auxiliary_mode', {}).get('enabled', False)
    print(f"[Config] t6_auxiliary_mode.enabled: {t6_aux_enabled}")

    # 验证配置与模式一致性
    expected_enabled = (mode == "t6_auxiliary")
    if t6_aux_enabled != expected_enabled:
        print(f"[Config WARNING] t6_auxiliary_mode.enabled ({t6_aux_enabled}) does not match mode ({mode})")
    else:
        print(f"[Config] Mode verification: PASS")

    # [新增] 打印 Stage0 feature_distillation 配置
    stage0_cfg = merged_config.get('mtl', {}).get('training_stages', {}).get('v4_stage0_t6_semantic_warmstart', {})
    feat_distill = stage0_cfg.get('feature_distillation', {})
    if feat_distill:
        print(f"[Config] Stage0 feature_distillation: enabled={feat_distill.get('enabled', False)}, weight={feat_distill.get('weight', 0.0)}")
    loss_weights = stage0_cfg.get('loss_weights', {})
    if loss_weights:
        print(f"[Config] Stage0 loss_weights: t6_ce={loss_weights.get('t6_ce', 1.0)}, t6_kd={loss_weights.get('t6_kd', 0.5)}, t6_feat={loss_weights.get('t6_feat', 0.0)}")

    return merged_config


def update_nested_config(config: dict, key_path: str, value: any) -> dict:
    """
    更新嵌套配置字典的指定键值

    Args:
        config: 配置字典
        key_path: 键路径 (如 "mtl.ablation.variant")
        value: 新值

    Returns:
        config: 更新后的配置字典

    示例:
        config = {"mtl": {"ablation": {"variant": "baseline"}}}
        update_nested_config(config, "mtl.ablation.variant", "t3_adapter")
        # -> {"mtl": {"ablation": {"variant": "t3_adapter"}}}
    """
    keys = key_path.split('.')
    current = config

    # 遍历到目标键的父级
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # 更新目标键
    final_key = keys[-1]

    # 类型推断 (尝试将字符串转换为合适类型)
    if isinstance(value, str):
        # 尝试转换为 bool
        if value.lower() in ['true', 'yes', '1']:
            value = True
        elif value.lower() in ['false', 'no', '0']:
            value = False
        # 尝试转换为 int
        elif value.isdigit():
            value = int(value)
        # 尝试转换为 float
        elif '.' in value and all(c.isdigit() or c == '.' or c == '-' for c in value):
            try:
                value = float(value)
            except ValueError:
                pass

    current[final_key] = value
    return config


def apply_config_overrides(config: dict, overrides: list[str]) -> dict:
    """
    应用多个配置覆盖

    Args:
        config: 配置字典
        overrides: 覆盖列表 (如 ["mtl.ablation.variant=t3_adapter", "training.batch_size=16"])

    Returns:
        config: 更新后的配置字典
    """
    for override in overrides:
        if '=' not in override:
            print(f"[Warning] 无效的覆盖格式: {override} (应为 key=value)")
            continue

        key_path, value = override.split('=', 1)
        config = update_nested_config(config, key_path.strip(), value.strip())
        print(f"[Config Override] {key_path} = {value}")

    return config


def find_best_single_task_checkpoint(
    model_dir: str = "models",
    pattern: str = "best_HDSTGCN*fold*.pth"
) -> tuple:
    """
    自动扫描并选择 macro_f1 最高的单任务检查点

    Args:
        model_dir: 模型目录
        pattern: 文件匹配模式

    Returns:
        best_path: 最佳检查点路径
        best_f1: 最佳 macro_f1
        best_fold: 最佳 fold 编号
    """
    import glob

    best_path = None
    best_f1 = 0.0
    best_fold = None

    checkpoint_files = glob.glob(os.path.join(model_dir, pattern))
    print(f"\n[Teacher] 扫描检查点: {len(checkpoint_files)} 个文件")

    for ckpt_path in checkpoint_files:
        filename = os.path.basename(ckpt_path)
        fold_match = filename.split('fold')[-1].split('.')[0] if 'fold' in filename else None

        if fold_match is None:
            continue

        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            metrics = ckpt.get('metrics', {})
            macro_f1 = metrics.get('macro_f1', metrics.get('val_macro_f1', None))

            if macro_f1 is not None and macro_f1 > 0:
                print(f"  fold{fold_match}: macro_f1={macro_f1:.4f}")
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_path = ckpt_path
                    best_fold = int(fold_match)
            else:
                # 没有 metrics 字段，但仍可以作为候选
                print(f"  fold{fold_match}: macro_f1=N/A (无metrics)")
                if best_path is None:
                    # 选择第一个找到的有效文件作为默认
                    best_path = ckpt_path
                    best_fold = int(fold_match)
                    print(f"  → 选择为默认教师模型 (无metrics时按fold顺序)")

        except Exception as e:
            print(f"  fold{fold_match}: 加载失败 - {e}")
            continue

    return best_path, best_f1, best_fold


def build_teacher_model(config: Config, checkpoint_path: str, device: str) -> torch.nn.Module:
    """
    构建 t6 教师模型 (单任务 HDSTGCN)

    Args:
        config: 单任务 Config
        checkpoint_path: 单任务检查点路径
        device: 设备

    Returns:
        teacher_model: 教师模型 (eval 模式)
    """
    from model import HDSTGCN
    from feature_mapping import create_adjacency_matrix, get_nine_graph_config

    # 确定是否使用变长模式
    use_var_length = getattr(config.model, 'use_variable_length', False)

    # 确定是否使用静态特征
    use_static = False
    static_dim = 16
    static_ablation = "full"
    num_static_features = 5
    if hasattr(config.model, 'static_features') and config.model.static_features is not None:
        use_static = config.model.static_features.enabled
        static_dim = config.model.static_features.static_dim
        static_ablation = config.model.static_features.ablation
        num_static_features = getattr(config.model.static_features, 'num_features', 5)

    # 获取时序编码器参数
    temporal_encoder_type = getattr(config.model, 'temporal_encoder', None)
    if temporal_encoder_type is not None:
        temporal_encoder_type = getattr(temporal_encoder_type, 'type', 'gru')
    else:
        temporal_encoder_type = 'gru'

    T_mid = 24
    temporal_encoder_cfg = getattr(config.model, 'temporal_encoder', None)

    # 先验门控参数
    gamma_init = 1.0
    gamma_min = 0.1
    if hasattr(config.model, 'prior_gate') and config.model.prior_gate is not None:
        gamma_init = getattr(config.model.prior_gate, 'gamma_init', 1.0)
        gamma_min = getattr(config.model.prior_gate, 'gamma_min', 0.1)

    # 通道注意力参数
    use_channel_attention = False
    channel_attention_init = 1.0
    if hasattr(config.model, 'channel_attention') and config.model.channel_attention is not None:
        use_channel_attention = getattr(config.model.channel_attention, 'enabled', False)
        channel_attention_init = getattr(config.model.channel_attention, 'init_value', 1.0)

    # 获取邻接矩阵
    semantic_adj = None
    attention_weights_matrix = None
    adapt_mode = getattr(config.features, 'adapt_mode', 'medical')

    if adapt_mode == "nine_graph":
        nine_config = get_nine_graph_config()
        semantic_adj = nine_config['adjacency']
        attention_weights_matrix = nine_config['attention_weights']
    else:
        semantic_adj = create_adjacency_matrix()

    # Flatten MLP 配置
    flatten_mlp_config = getattr(config.model, 'flatten_mlp', None)

    # Pooling_only 配置
    pooling_only_config = getattr(config.model, 'pooling_only', None)

    # ========== 加载检查点，推断 output_dim ==========
    if not os.path.exists(checkpoint_path):
        print(f"[Warning] 教师模型不存在: {checkpoint_path}")
        # 构建默认模型 (4分类)，并移动到 device
        model = HDSTGCN(
            input_dim=config.data.max_length if use_var_length else config.data.L_win,
            hidden_dim=config.model.hidden_dim,
            output_dim=4,  # 默认 4 分类
            channel_groups=config.features.channel_groups,
            num_channel=config.features.num_channels,
            D_time=config.model.D_time,
            dropout=config.model.dropout,
            semantic_adj=semantic_adj,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features,
            static_ablation=static_ablation,
            graph_ablation=config.model.graph_ablation,
            temporal_encoder_type=temporal_encoder_type,
            T_mid=T_mid,
            temporal_encoder_cfg=temporal_encoder_cfg,
            gamma_init=gamma_init,
            gamma_min=gamma_min,
            use_channel_attention=use_channel_attention,
            channel_attention_init=channel_attention_init,
            attention_weights=attention_weights_matrix,
            flatten_mlp_config=flatten_mlp_config,
            pooling_only_config=pooling_only_config,
            is_binary=False
        ).to(device)  # 移动到指定设备
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # 从检查点推断 output_dim (classifier 最后一层)
    classifier_weight = state_dict.get('classifier.4.weight', None)
    if classifier_weight is not None:
        inferred_output_dim = classifier_weight.shape[0]
        print(f"[Teacher] 从检查点推断 output_dim={inferred_output_dim}")
    else:
        inferred_output_dim = 4  # 默认值
        print(f"[Teacher] 无法推断 output_dim，使用默认值 {inferred_output_dim}")

    # 判断是否是二分类 (output_dim=2 的情况)
    is_binary = (inferred_output_dim == 2)

    # 构建模型以匹配检查点结构，并移动到 device
    model = HDSTGCN(
        input_dim=config.data.max_length if use_var_length else config.data.L_win,
        hidden_dim=config.model.hidden_dim,
        output_dim=inferred_output_dim,  # 使用推断的 output_dim
        channel_groups=config.features.channel_groups,
        num_channel=config.features.num_channels,
        D_time=config.model.D_time,
        dropout=config.model.dropout,
        semantic_adj=semantic_adj,
        # 静态特征参数
        use_static_features=use_static,
        static_dim=static_dim,
        num_static_features=num_static_features,
        static_ablation=static_ablation,
        graph_ablation=config.model.graph_ablation,
        # 时序编码器参数
        temporal_encoder_type=temporal_encoder_type,
        T_mid=T_mid,
        temporal_encoder_cfg=temporal_encoder_cfg,
        # 先验门控参数
        gamma_init=gamma_init,
        gamma_min=gamma_min,
        # 通道注意力参数
        use_channel_attention=use_channel_attention,
        channel_attention_init=channel_attention_init,
        # 注意力权重预设
        attention_weights=attention_weights_matrix,
        # Flatten MLP 配置
        flatten_mlp_config=flatten_mlp_config,
        # Pooling_only 配置
        pooling_only_config=pooling_only_config,
        # 二分类模式
        is_binary=is_binary
    ).to(device)  # 移动到指定设备

    # 加载参数 (strict=False 允许缺失的参数)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"[Teacher] 缺失的参数 (将使用默认值): {missing_keys}")
    if unexpected_keys:
        print(f"[Teacher] 未预期的参数 (将被忽略): {unexpected_keys}")

    print(f"[Teacher] 加载教师模型: {checkpoint_path}")
    print(f"[Teacher] 输出维度: {inferred_output_dim} (is_binary={is_binary})")

    # 设置为 eval 模式，冻结参数
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def run_mtl_training(
    config: Config,
    mtl_config: dict,
    fold_idx: int = 0,
    n_folds: int = 5,
    device: str = 'cuda',
    resume_stage: str = None,
    resume_checkpoint: str = None,
    checkpoint_stage2: str = None,
    checkpoint_phase1: str = None,  # [新增] Stage3 Phase1 checkpoint
    disable_swanlab: bool = False,
    # [新增] Holdout 参数
    dev_indices: List[int] = None,
    test_indices: List[int] = None,
    holdout_enabled: bool = False,
    all_data_cache: dict = None,
    # [新增 2026-05-27] 运行模式 (用于 Clean 架构命名)
    mode: str = "baseline",
    # [新增 2026-06-08] Clinical Interpretation 参数
    interpret_enabled: bool = False,
    interpret_out_dir: str = None,
    interpret_save_intermediates: bool = False,
    interpret_save_attr: bool = False,
    interpret_context_counterfactual: bool = False,
    interpret_context_modes: str = "normal,zero,shuffle"
):
    """
    执行 MTL 三阶段训练 (支持 Holdout 模式)

    Args:
        config: 基础 Config (用于数据加载)
        mtl_config: MTL 配置字典
        fold_idx: Fold 编号
        n_folds: 总 Fold 数
        device: 设备
        resume_stage: 恢复阶段 ("stage1", "stage2", "stage3", "stage3_phase2", "holdout")
        resume_checkpoint: 恢复检查点路径
        checkpoint_stage2: 阶段2检查点路径 (从阶段3恢复时使用)
        checkpoint_phase1: [新增] Stage3 Phase1 检查点路径 (从 stage3_phase2 恢复时使用)
        disable_swanlab: 是否禁用 SwanLab
        dev_indices: [新增] 预划分的 Dev_Set 索引列表
        test_indices: [新增] 预划分的 Test_Set 索引列表
        holdout_enabled: [新增] 是否启用 Holdout 测试集评估
        all_data_cache: [新增] 预加载的数据缓存 (所有 Fold 共享)
        mode: [新增 2026-05-27] 运行模式 (用于 Clean 架构命名)
    """
    print("\n" + "=" * 80)
    print("HDSTGCN MTL 三阶段训练")
    print("=" * 80)
    print(f"Fold: {fold_idx + 1}/{n_folds}")
    print(f"Device: {device}")
    print("=" * 80 + "\n")

    # ========== 1. 加载多任务标签统计 ==========
    label_file = config.data.label_file
    filename_to_labels, task_label_mappings, task_class_counts = load_mtl_labels(
        label_file,
        min_label_freq=50
    )

    # ========== 2. 构建 TaskSpec ==========
    task_specs = build_task_specs_from_config(
        mtl_config,
        dataset_stats=task_class_counts,
        device=device
    )

    # 验证 (传入 config 以支持消融模式检测)
    if not validate_task_specs(task_specs, mtl_config):
        raise ValueError("TaskSpec 验证失败")

    print("\n[TaskSpec] 任务配置:")
    for task_key, spec in task_specs.items():
        print(f"  {spec}")

    # ========== 3. 创建 DataLoader (支持 Holdout) ==========
    batch_size = mtl_config.get('training', {}).get('batch_size', 16)

    train_loader, test_loader, dataset = create_mtl_dataloaders(
        config=config,
        fold_idx=fold_idx,
        n_folds=n_folds,
        batch_size=batch_size,
        use_variable_length=False,  # 固定长度 200 帧
        task_keys=list(MTL_LABEL_COLUMNS.keys()),
        num_workers=4,
        # [新增] Holdout 参数
        dev_indices=dev_indices,
        test_indices=test_indices if holdout_enabled else None,
        use_holdout_test=False,
        strict_no_filter=holdout_enabled
    )

    print(f"\n[DataLoader] 训练集: {len(train_loader.dataset)}, 测试集: {len(test_loader.dataset)}")

    # ========== 4. 创建模型 ==========
    # 获取九图邻接矩阵
    semantic_adj = get_nine_graph_adjacency()

    # 检测架构版本
    architecture_cfg = mtl_config.get('mtl', {}).get('architecture', {})
    variant = architecture_cfg.get('variant', 'baseline')

    print(f"\n[Architecture] 使用架构变体: {variant}")

    t1_ckpt = None

    if variant == "protected_dual_engine_t6_guided_v4":
        # v4 架构: ProtectedDualEngineMTL_v4 (T6-guided Context Injection)
        print("[v4] T6角色: 疾病上下文提供者 (不是锚定保护任务)")
        print("[v4] dyn_feat_t6: 压缩为 c6_deep，注入 t1-t5")
        print("[v4] Beta Gate context_dim: 40 -> 56 (+ c6_deep[16])")
        print("[v4] KD废弃: use_kd_t6=False")
        print("[v4] checkpoint指标: weighted_macro_f1_t1_to_t5")

        # [v4 Config-Driven] 构建完整配置字典
        hcgc_v4_cfg = mtl_config.get('mtl', {}).get('hcgc_v4', {})
        alpha_cfg = mtl_config.get('mtl', {}).get('alpha', {})
        beta_cfg = mtl_config.get('mtl', {}).get('beta', {})
        # [修复 2026-05-07] 正确的配置路径: mtl.hcgc_v4.t6_auxiliary_mode
        t6_auxiliary_mode_cfg = hcgc_v4_cfg.get('t6_auxiliary_mode', {})

        full_config = {
            'hcgc_v4': hcgc_v4_cfg,
            'alpha': alpha_cfg,
            'beta': beta_cfg,
            't6_auxiliary_mode': t6_auxiliary_mode_cfg,  # 传递 t6_auxiliary_mode 配置
        }

        model = ProtectedDualEngineMTL_v4(
            task_specs=task_specs,
            num_channels=30,  # nine_graph 模式
            D_time=16,
            T_mid=24,
            semantic_adj=semantic_adj,
            device=device,
            config=full_config  # 传递完整配置（替换 t6_deep_context_config）
        )

    elif variant == "protected_dual_engine_asymmetric_v3":
        # v3 架构: ProtectedDualEngineMTL_v3 (简化版)
        print("[v3] Alpha删除CGC: trunk输出直接传递")
        print("[v3] Beta收缩为三专家: shared, group_245, t3_private")
        print("[v3] Beta gates改为2维: t2/t4/t5只有[shared, group_245]")

        # [v3 Config-Driven] 构建完整配置字典
        hcgc_v3_cfg = mtl_config.get('mtl', {}).get('hcgc_v4', {})  # v3 继承 v4 配置结构
        alpha_cfg = mtl_config.get('mtl', {}).get('alpha', {})
        beta_cfg = mtl_config.get('mtl', {}).get('beta', {})

        full_config = {
            'hcgc_v4': hcgc_v3_cfg,
            'alpha': alpha_cfg,
            'beta': beta_cfg,
        }

        model = ProtectedDualEngineMTL_v3(
            task_specs=task_specs,
            num_channels=30,  # nine_graph 模式
            D_time=16,
            T_mid=24,
            semantic_adj=semantic_adj,
            device=device,
            config=full_config  # 传递完整配置
        )

    elif variant == "protected_dual_engine_hcgc_ple_v2":
        # v2 架构: ProtectedDualEngineMTL
        print("[v2] trunk output 不再作为 gate 候选")
        print("[v2] 专家改为残差变换 (H_trunk + delta)")

        model = ProtectedDualEngineMTL(
            task_specs=task_specs,
            num_channels=30,  # nine_graph 模式
            D_time=16,
            semantic_adj=semantic_adj,
            config=mtl_config
        )

    elif variant == "protected_dual_engine_hcgc_ple":
        # v1 架构: ProtectedDualEngineMTL (旧版)
        model = ProtectedDualEngineMTL(
            task_specs=task_specs,
            num_channels=30,
            D_time=16,
            semantic_adj=semantic_adj,
            config=mtl_config
        )

    else:
        # baseline 架构: HDSTGCNMTL
        model = HDSTGCNMTL(
            task_specs=task_specs,
            num_channels=30,  # nine_graph 模式
            D_time=16,
            semantic_adj=semantic_adj,
            config=mtl_config
        )

    model = model.to(device)

    # 打印参数量
    counts = model.get_num_parameters()
    print(f"\n[Model] 参数量:")
    for key, count in counts.items():
        print(f"  {key}: {count}")

    # ========== 5. 模型初始化 (v2/v3/v4 特有流程) ==========
    if variant == "protected_dual_engine_t6_guided_v4":
        # v4 初始化流程: 支持 checkpoint_inheritance 或 from_scratch_structured
        policy = mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('initialization', {}).get('policy', 'from_scratch_structured')
        print(f"\n[v4 Init] 执行 v4 初始化流程 (policy={policy})...")

        # 从配置获取初始化参数
        hcgc_v4_cfg = mtl_config.get('mtl', {}).get('hcgc_v4', {})
        checkpoints_cfg = mtl_config.get('mtl', {}).get('checkpoints', {})

        # 构建完整配置（包含 t1_checkpoint 路径）
        full_init_config = {
            **hcgc_v4_cfg,
            'checkpoints': checkpoints_cfg,  # 传递 checkpoints 块
            't1_checkpoint_path': checkpoints_cfg.get('t1_checkpoint', None)  # 直接传递路径
        }

        # 调用 init_protected_dual_engine_mtl_v4
        init_protected_dual_engine_mtl_v4(
            model=model,
            config=full_init_config,
            task_specs=task_specs,
            device=device
        )

        # v4 也需要教师模型用于 KD
        checkpoints_cfg = mtl_config.get('mtl', {}).get('checkpoints', {})
        # t6_ckpt_config = checkpoints_cfg.get('t6_checkpoint', 'auto')
        # t6_best_f1 = checkpoints_cfg.get('t6_single_task_best_f1', None)

        # if t6_ckpt_config == 'auto' or t6_ckpt_config is None:
        #     print("\n[Teacher v4] 自动选择最优教师模型...")
        #     t6_ckpt, t6_best_f1_auto, best_fold = find_best_single_task_checkpoint()
        #     if t6_ckpt:
        #         t6_best_f1 = t6_best_f1_auto
        #         print(f"[Teacher v4] 最佳模型: fold{best_fold}, macro_f1={t6_best_f1_auto:.4f}")
        #     else:
        #         t6_ckpt = None
        #         t6_best_f1 = 0.85
        # else:
        #     t6_ckpt = t6_ckpt_config
        #     if t6_best_f1 is None:
        #         try:
        #             ckpt = torch.load(t6_ckpt, map_location='cpu', weights_only=False)
        #             metrics = ckpt.get('metrics', {})
        #             t6_best_f1 = metrics.get('macro_f1', 0.85)
        #         except:
        #             t6_best_f1 = 0.85

    # elif variant == "protected_dual_engine_asymmetric_v3":
    #     # v3 初始化流程
    #     print("\n[v3 Init] 执行 v3 初始化流程...")

    #     # 从配置获取初始化参数
    #     hcgc_v3_cfg = mtl_config.get('mtl', {}).get('hcgc_v3', {})
    #     init_cfg = hcgc_v3_cfg.get('initialization', {})

    #     baseline_path = init_cfg.get('baseline_checkpoint', None)
    #     t3_path = init_cfg.get('t3_best_checkpoint', None)
    #     identity_strength = init_cfg.get('identity_strength', 0.1)

    #     # Step 1: trunk 从 baseline 初始化
    #     if baseline_path and os.path.exists(baseline_path):
    #         init_trunks_from_baseline(model, baseline_path, device=device)
    #     else:
    #         print("[v3 Init Warning] 未找到 baseline checkpoint, trunk 使用随机初始化")

    #     # Step 2: Beta残差专家接近 identity 初始化 (v3: 只有3个专家)
    #     for expert_name in ["shared", "group_245", "t3_private"]:
    #         expert = model.beta_residual_experts[expert_name]
    #         for module in expert.modules():
    #             # 只对Linear层进行near-identity初始化
    #             if isinstance(module, nn.Linear):
    #                 nn.init.normal_(module.weight, mean=0.0, std=identity_strength)
    #                 if module.bias is not None:
    #                     nn.init.zeros_(module.bias)
    #         print(f"[v3 Init] beta_residual_experts['{expert_name}']: near-identity init (std={identity_strength})")

    #     # Step 3: E_t3 增强初始化 (Xavier - 只对Linear层)
    #     if t3_path and os.path.exists(t3_path):
    #         print(f"[v3 Init] T3 checkpoint found: {t3_path}")
    #         print("[v3 Init] Using enhanced Xavier initialization for E_t3")
    #         expert = model.beta_residual_experts["t3_private"]
    #         for module in expert.modules():
    #             # Xavier初始化只适用于Linear层(>=2维权重)
    #             if isinstance(module, nn.Linear):
    #                 nn.init.xavier_uniform_(module.weight)
    #                 if module.bias is not None:
    #                     nn.init.zeros_(module.bias)
    #     else:
    #         print("[v3 Init Warning] 未找到 t3 checkpoint, E_t3 使用标准初始化")

    #     # Step 4: 头部初始化
    #     if baseline_path and os.path.exists(baseline_path):
    #         init_heads_from_baseline(model, baseline_path, task_specs, device=device)

    #     print("[v3 Init] 初始化完成")

    #     # v3 也需要教师模型用于 KD
    #     checkpoints_cfg = mtl_config.get('mtl', {}).get('checkpoints', {})
    #     t6_ckpt_config = checkpoints_cfg.get('t6_checkpoint', 'auto')
    #     t6_best_f1 = checkpoints_cfg.get('t6_single_task_best_f1', None)

    #     if t6_ckpt_config == 'auto' or t6_ckpt_config is None:
    #         print("\n[Teacher v3] 自动选择最优教师模型...")
    #         t6_ckpt, t6_best_f1_auto, best_fold = find_best_single_task_checkpoint()
    #         if t6_ckpt:
    #             t6_best_f1 = t6_best_f1_auto
    #             print(f"[Teacher v3] 最佳模型: fold{best_fold}, macro_f1={t6_best_f1_auto:.4f}")
    #         else:
    #             t6_ckpt = None
    #             t6_best_f1 = 0.85
    #     else:
    #         t6_ckpt = t6_ckpt_config
    #         if t6_best_f1 is None:
    #             try:
    #                 ckpt = torch.load(t6_ckpt, map_location='cpu', weights_only=False)
    #                 metrics = ckpt.get('metrics', {})
    #                 t6_best_f1 = metrics.get('macro_f1', 0.85)
    #             except:
    #                 t6_best_f1 = 0.85

    # elif variant == "protected_dual_engine_hcgc_ple_v2":
    #     # v2 初始化流程
    #     print("\n[v2 Init] 执行 v2 初始化流程...")

    #     # 从配置获取初始化参数
    #     hcgc_v2_cfg = mtl_config.get('mtl', {}).get('hcgc_v2', {})
    #     init_cfg = hcgc_v2_cfg.get('initialization', {})

    #     baseline_path = init_cfg.get('baseline_checkpoint', None)
    #     t3_path = init_cfg.get('t3_best_checkpoint', None)
    #     identity_strength = init_cfg.get('identity_strength', 0.1)

    #     # Step 1: trunk 从 baseline 初始化
    #     if baseline_path and os.path.exists(baseline_path):
    #         init_trunks_from_baseline(model, baseline_path, device=device)
    #     else:
    #         print("[v2 Init Warning] 未找到 baseline checkpoint, trunk 使用随机初始化")

    #     # Step 2: 残差专家接近 identity 初始化
    #     init_residual_experts_near_identity(model, identity_strength=identity_strength, device=device)

    #     # Step 3: E_t3 增强初始化
    #     if t3_path and os.path.exists(t3_path):
    #         init_B_t3_from_t3_best(model, t3_path, device=device)
    #     else:
    #         print("[v2 Init Warning] 未找到 t3 checkpoint, E_t3 使用标准初始化")

    #     # Step 4: 头部初始化
    #     if baseline_path and os.path.exists(baseline_path):
    #         init_heads_from_baseline(model, baseline_path, task_specs, device=device)

    #     print("[v2 Init] 初始化完成")

    #     # v2 也需要教师模型用于 KD
    #     checkpoints_cfg = mtl_config.get('mtl', {}).get('checkpoints', {})
    #     t6_ckpt_config = checkpoints_cfg.get('t6_checkpoint', 'auto')
    #     t6_best_f1 = checkpoints_cfg.get('t6_single_task_best_f1', None)

    #     if t6_ckpt_config == 'auto' or t6_ckpt_config is None:
    #         print("\n[Teacher v2] 自动选择最优教师模型...")
    #         t6_ckpt, t6_best_f1_auto, best_fold = find_best_single_task_checkpoint()
    #         if t6_ckpt:
    #             t6_best_f1 = t6_best_f1_auto
    #             print(f"[Teacher v2] 最佳模型: fold{best_fold}, macro_f1={t6_best_f1_auto:.4f}")
    #         else:
    #             t6_ckpt = None
    #             t6_best_f1 = 0.85
    #     else:
    #         t6_ckpt = t6_ckpt_config
    #         if t6_best_f1 is None:
    #             try:
    #                 ckpt = torch.load(t6_ckpt, map_location='cpu', weights_only=False)
    #                 metrics = ckpt.get('metrics', {})
    #                 t6_best_f1 = metrics.get('macro_f1', 0.85)
    #             except:
    #                 t6_best_f1 = 0.85

    # else:
    #     # baseline/v1 初始化流程 (原有的单任务预训练加载)
    #     checkpoints_cfg = mtl_config.get('mtl', {}).get('checkpoints', {})

    #     # ========== 5.1 自动选择最优教师模型 ==========
    #     t6_ckpt_config = checkpoints_cfg.get('t6_checkpoint', 'auto')
    #     t6_best_f1 = checkpoints_cfg.get('t6_single_task_best_f1', None)

    #     if t6_ckpt_config == 'auto' or t6_ckpt_config is None:
    #         # 自动扫描选择最优 fold
    #         print("\n[Teacher] 自动选择最优教师模型...")
    #         t6_ckpt, t6_best_f1_auto, best_fold = find_best_single_task_checkpoint()

    #         if t6_ckpt is None:
    #             print("[Warning] 未找到单任务检查点，请先运行单任务训练")
    #             print("[Warning] 将不使用教师模型蒸馏")
    #             t6_ckpt = None
    #             t6_best_f1 = 0.85  # 默认值
    #         else:
    #             print(f"[Teacher] 最佳模型: fold{best_fold}, macro_f1={t6_best_f1_auto:.4f}")
    #             print(f"[Teacher] 路径: {t6_ckpt}")
    #             t6_best_f1 = t6_best_f1_auto
    #     else:
    #         t6_ckpt = t6_ckpt_config
    #         if t6_best_f1 is None:
    #             # 尝试从检查点读取
    #             try:
    #                 ckpt = torch.load(t6_ckpt, map_location='cpu', weights_only=False)
    #                 metrics = ckpt.get('metrics', {})
    #                 t6_best_f1 = metrics.get('macro_f1', metrics.get('val_macro_f1', 0.85))
    #             except:
    #                 t6_best_f1 = 0.85

        # ========== 5.2 加载单任务预训练权重 ==========
        t1_ckpt = None

        if checkpoints_cfg.get('load_single_task_pretrained', True):
            # [实验E] t6 PMGT 从 t6 checkpoint 初始化 (弱监督任务需要合理的 t6 特征)
            no_t6_context_but_train_t6 = mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('no_t6_context_but_train_t6', {}).get('enabled', False)
            if no_t6_context_but_train_t6:
                t6_ckpt = checkpoints_cfg.get('t6_checkpoint')
                if t6_ckpt and t6_ckpt != 'auto' and os.path.exists(t6_ckpt):
                    load_single_task_checkpoint_into_mtl(
                        model=model,
                        ckpt_path=t6_ckpt,
                        task_name="t6",
                        branch_type="alpha",
                        strict=False
                    )
                    print("[实验E] alpha_interactors['t6'] PMGT initialized from t6 checkpoint")
                else:
                    print(f"[实验E Warning] t6_checkpoint not found: {t6_ckpt}, alpha_interactors['t6'] using default init")

            # t1 单任务 (如果有)
            t1_ckpt = checkpoints_cfg.get('t1_checkpoint')
            if t1_ckpt and t1_ckpt != 'auto' and os.path.exists(t1_ckpt):
                load_single_task_checkpoint_into_mtl(
                    model=model,
                    ckpt_path=t1_ckpt,
                    task_name="t1",
                    branch_type="alpha",
                    strict=False
                )

    # ========== 5.5 实验E 验证和日志 ==========
    no_t6_context_but_train_t6 = mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('no_t6_context_but_train_t6', {}).get('enabled', False)
    if no_t6_context_but_train_t6:
        print("\n" + "=" * 60)
        print("[Experiment E] no_t6_context_but_train_t6 enabled")
        print("[Experiment E] base mode = baseline")
        print("[Experiment E] Stage0 disabled")
        print("[Experiment E] feature_distillation disabled")
        print("[Experiment E] t6_context.inject=false for all stages")
        print("[Experiment E] context_reg disabled")
        print("[Experiment E] beta_gate context_dim=40")
        print("[Experiment E] c6_deep disabled/not used")
        print("[Experiment E] t6_to_t1_adapter disabled/not used")
        print("[Experiment E] Stage1 active_tasks=[\"t1\",\"t6\"]")
        print("[Experiment E] Stage2 active_tasks=[\"t2\",\"t3\",\"t4\",\"t5\"]")
        print("[Experiment E] Stage3 Phase1 active_tasks=[\"t2\",\"t3\",\"t4\",\"t5\"]")
        print("[Experiment E] Stage3 Phase2 active_tasks=[\"t1\",\"t2\",\"t3\",\"t4\",\"t5\",\"t6\"]")
        print("[Experiment E] Stage1 loss_weights: t1=1.0, t6=0.1")
        print("[Experiment E] Phase2 loss_weights: t1=1.0,t2=1.0,t3=1.2,t4=1.0,t5=1.0,t6=0.05")
        print("[Experiment E] checkpoint metric=weighted_macro_f1_t1_to_t5")
        print("[Experiment E] t6 is auxiliary loss only, not context source")
        print("=" * 60)

        # [实验E 安全检查] 验证 beta_gate context_dim == 40
        if variant == "protected_dual_engine_t6_guided_v4" and hasattr(model, 'beta_gates'):
            for gate_key in ["t2", "t3", "t4", "t5"]:
                gate = model.beta_gates[gate_key]
                # TaskSpecificGate 的 context_dim 存储在构造函数中
                if hasattr(gate, 'context_dim'):
                    actual_dim = gate.context_dim
                elif hasattr(gate, 'context_projector') and hasattr(gate.context_projector, 'in_features'):
                    actual_dim = gate.context_projector.in_features
                else:
                    # 从权重推断
                    for name, param in gate.named_parameters():
                        if 'context' in name and param.dim() >= 2:
                            actual_dim = param.shape[-1] if param.shape[0] < param.shape[-1] else param.shape[0]
                            break
                    else:
                        actual_dim = -1

                print(f"[Experiment E] beta_gates['{gate_key}'].context_dim = {actual_dim}")
                if actual_dim == 56:
                    raise RuntimeError(
                        f"\n[Experiment E FATAL] beta_gates['{gate_key}'].context_dim = 56!\n"
                        f"  实验E 要求 beta_gate context_dim = 40 (无 c6_deep 注入)\n"
                        f"  请检查 t6_auxiliary_mode.enabled 是否被错误设置为 true"
                    )
                assert actual_dim == 40, f"beta_gates['{gate_key}'].context_dim = {actual_dim}, expected 40"

            print("[Experiment E] beta_gate context_dim validation: PASS (all gates = 40)")
        print()

    # ========== 6. 创建教师模型 (KD) ==========
    teacher_model = None
    if t1_ckpt and os.path.exists(t1_ckpt):
        teacher_model = build_teacher_model(config, t1_ckpt, device)

    # ========== 7. 创建损失函数 ==========
    criterions = build_mtl_criterions(task_specs, device=device)

    total_loss = MTLTotalLoss(
        kd_weight=0.5,
        kd_temperature=2.0
    )

    # ========== 8. SwanLab 初始化 ==========
    swanlab_run = None
    disable_swanlab = True
    # SwanLab logging is disabled for faster local runs.

    # ========== 9. 创建训练器 ==========
    trainer_config = {
        'lr': mtl_config.get('training', {}).get('lr', 0.0003),
        'weight_decay': mtl_config.get('training', {}).get('weight_decay', 0.001),
        'gradient_clip': mtl_config.get('training', {}).get('gradient_clip', 1.0),
        'accumulation_steps': mtl_config.get('training', {}).get('accumulation_steps', 2),
        'fold': fold_idx + 1,
        # 't6_single_task_f1': t6_best_f1 if t6_best_f1 is not None else 0.85,  # 使用真实值
        't6_constraint_delta': 0.005,
        # [P0 修复] 添加完整的 training_stages 配置
        'training_stages': mtl_config.get('mtl', {}).get('training_stages', {}),
        'lr_scale': mtl_config.get('mtl', {}).get('lr_scale', {}),
        # [修复 - 2026-04-28] 添加架构相关配置 (用于 v4 Baseline 模式检测)
        'architecture': mtl_config.get('mtl', {}).get('architecture', {}),
        'hcgc_v4': mtl_config.get('mtl', {}).get('hcgc_v4', {}),
        't6_deep_context': mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('t6_deep_context', {}),
        'monitoring': mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('monitoring', {}),
        'uncertainty_weighting': mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('uncertainty_weighting', {}),
        # [新增] 实验命名后缀
        'experiment_suffix': mtl_config.get('mtl', {}).get('experiment_suffix', ''),
        # [新增 2026-05-27] 运行模式 (用于 Clean 架构命名)
        'mode': mode,
    }

    trainer = MTLTrainer(
        model=model,
        criterions=criterions,
        total_loss=total_loss,
        train_loader=train_loader,
        val_loader=test_loader,
        config=trainer_config,
        task_specs=task_specs,
        device=device,
        swanlab_run=swanlab_run,
        teacher_model=teacher_model
    )

    # [新增] 初始化增量保存 JSON (防止崩溃丢失)
    trainer._init_incremental_json(fold=fold_idx + 1)

    # ========== 9.5 加载恢复检查点 (如果指定) ==========
    # 加载恢复检查点 (根据 checkpoint 内部的 stage 字段判断阶段)
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        ckpt_stage = ckpt.get('stage', 'unknown')

        print(f"\n[Resume] 加载检查点: {resume_checkpoint}")
        print(f"[Resume] 检查点阶段: {ckpt_stage}")

        model_state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(model_state, strict=False)
        print(f"[Resume] 模型状态已加载")

        if 'metrics' in ckpt:
            print(f"[Resume] 检查点指标: {ckpt['metrics']}")

    elif resume_checkpoint is not None:
        print(f"[Warning] 检查点不存在: {resume_checkpoint}")

    # 加载阶段2检查点 (Beta分支，从阶段3恢复时使用)
    if checkpoint_stage2 is not None and os.path.exists(checkpoint_stage2):
        print(f"\n[Resume] 加载阶段2检查点: {checkpoint_stage2}")
        ckpt2 = torch.load(checkpoint_stage2, map_location=device, weights_only=False)

        model_state2 = ckpt2.get('model_state_dict', ckpt2.get('state_dict', ckpt2))
        # 只加载Beta分支相关参数 (beta_encoder, beta_projectors, beta classifiers等)
        beta_keys = ['beta_encoder', 'beta_projectors', 'static_encoders.t2', 'static_encoders.t3',
                     'static_encoders.t4', 'static_encoders.t5', 'classifiers.t2', 'classifiers.t3',
                     'classifiers.t4', 'classifiers.t5', 'log_vars.t2', 'log_vars.t3', 'log_vars.t4', 'log_vars.t5']

        filtered_state = {}
        for key, value in model_state2.items():
            # 检查是否是Beta分支参数
            if any(beta_key in key for beta_key in beta_keys):
                filtered_state[key] = value

        if filtered_state:
            model.load_state_dict(filtered_state, strict=False)
            print(f"[Resume] 阶段2 Beta分支参数已加载 ({len(filtered_state)} 个)")
        else:
            # 如果没有过滤到参数，直接全量加载（阶段2检查点应包含完整模型）
            model.load_state_dict(model_state2, strict=False)
            print(f"[Resume] 阶段2完整模型状态已加载")

        # 加载 optimizer 状态
        if 'optimizer_state_dict' in ckpt2:
            trainer.optimizer.load_state_dict(ckpt2['optimizer_state_dict'])
            print(f"[Resume] 阶段2 Optimizer 状态已加载")

        if 'metrics' in ckpt2:
            print(f"[Resume] 阶段2指标: {ckpt2['metrics']}")
        if 'stage' in ckpt2:
            print(f"[Resume] 阶段2来自: {ckpt2['stage']}")

    elif checkpoint_stage2 is not None:
        print(f"[Warning] 阶段2检查点不存在: {checkpoint_stage2}")

    # ========== 10. 执行四阶段训练 (v4: Stage0 + Stage1 + Stage2 + Stage3) ==========
    stages_cfg = mtl_config.get('mtl', {}).get('training_stages', {})

    # ========== [新增] holdout 模式: 跳过全部训练，直接进入评估 ==========
    if resume_stage == "holdout":
        print("\n" + "=" * 80)
        print("[Holdout-Only Mode] 跳过全部训练阶段，直接进行 Holdout 测试集评估")
        print("=" * 80)

        # Holdout-only 模式只使用用户通过 --checkpoint 指定的模型。
        # 该 checkpoint 已在上方通用 resume 逻辑中加载，这里不再构造或加载默认 models/best_* 路径。
        checkpoint_path = resume_checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[Holdout-Only] 使用已加载的用户指定 checkpoint: {checkpoint_path}")
        else:
            print(f"[Holdout-Only] Warning: 未找到用户指定 checkpoint: {checkpoint_path}，使用当前模型状态")

    else:
        # 正常训练流程
        # [v4新增] Stage0: T6 Semantic Warmstart (可选)
        # 仅在 v4架构 + T6辅助模式 + enabled=true 时执行
        if variant == "protected_dual_engine_t6_guided_v4":
            stage0_cfg = stages_cfg.get('v4_stage0_t6_semantic_warmstart', {})
            t6_aux_mode = mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('t6_auxiliary_mode', {}).get('enabled', False)

            if stage0_cfg.get('enabled', False) and t6_aux_mode:
                # T6辅助模式 + Stage0启用: 执行 warmstart
                stage0_epochs = stage0_cfg.get('epochs', 3)
                stage0_executed = trainer.run_stage0_t6_semantic_warmstart(epochs=stage0_epochs)
                if stage0_executed:
                    print("[v4] Stage0 完成，Stage1 将使用 t6_weight=0.05 (弱监督)")
                else:
                    print("[v4] Stage0 未执行，Stage1 将使用 t6_weight=0.1 (标准监督)")
            else:
                print("[v4] Stage0 未启用，直接进入 Stage1")

        # Stage1: Alpha 锚定
        if resume_stage is None or resume_stage == "stage1":
            # 阶段一: Alpha 锚定
            stage1_epochs = stages_cfg.get('v4_stage1_alpha_anchor', {}).get('epochs', 20) if variant == "protected_dual_engine_t6_guided_v4" else stages_cfg.get('stage1_alpha_anchor', {}).get('epochs', 20)
            trainer.run_stage1_alpha_anchor(epochs=stage1_epochs)

        if resume_stage is None or resume_stage in ["stage1", "stage2"]:
            # 阶段二: Beta 预热
            stage2_key = 'v4_stage2_beta_warmup' if variant == "protected_dual_engine_t6_guided_v4" else 'stage2_beta_warmup'
            stage2_epochs = stages_cfg.get(stage2_key, {}).get('epochs', 20)
            trainer.run_stage2_beta_warmup(epochs=stage2_epochs)

        # ========== [新增] stage3_phase2 支持 - 跳过 phase1 直接进入 phase2 ==========
        if resume_stage == "stage3_phase2":
            # 加载 Phase1 checkpoint (如果提供了 checkpoint_phase1)
            if checkpoint_phase1 and os.path.exists(checkpoint_phase1):
                print(f"\n[Stage3 Phase2 Resume] 加载 Phase1 checkpoint: {checkpoint_phase1}")
                checkpoint_data = torch.load(checkpoint_phase1, map_location=device, weights_only=False)
                trainer.model.load_state_dict(checkpoint_data['model_state_dict'])
                trainer.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                print("[Stage3 Phase2 Resume] 模型加载完成，跳过 Phase1")
            else:
                print(f"[Warning] stage3_phase2 需要提供 --checkpoint_phase1 参数")

            # 直接执行 Phase2 (跳过 Phase1)
            stage3_key = 'v4_stage3_joint_finetune' if variant == "protected_dual_engine_t6_guided_v4" else 'stage3_joint_finetune'
            stage3_epochs = stages_cfg.get(stage3_key, {}).get('epochs', 60)
            trainer.run_stage3_joint_finetune(epochs=stage3_epochs, skip_phase1=True)

        elif resume_stage is None or resume_stage in ["stage1", "stage2", "stage3"]:
            # 阶段三: 联合微调 (完整流程: Phase1 + Phase2)
            stage3_key = 'v4_stage3_joint_finetune' if variant == "protected_dual_engine_t6_guided_v4" else 'stage3_joint_finetune'
            stage3_epochs = stages_cfg.get(stage3_key, {}).get('epochs', 60)
            trainer.run_stage3_joint_finetune(epochs=stage3_epochs)

    # ========== [新增] Holdout 测试集评估 ==========
    holdout_metrics = None
    if resume_stage == "holdout":
        holdout_enabled = True  # holdout-only 模式强制启用
    if holdout_enabled and test_indices is not None:
        print(f"\n{'='*80}")
        if resume_stage == "holdout":
            print(f"[Holdout Test] Fold {fold_idx+1} Holdout-only 评估模式")
        else:
            print(f"[Holdout Test] Fold {fold_idx+1} 训练完成，开始评估独立测试集")
        print("="*80)

        # 使用训练集统计量 (从 dataset 获取)
        fold_stats = dataset.stats
        fold_static_stats = dataset.static_stats

        # 创建 Holdout 测试集 DataLoader
        holdout_loader = create_mtl_holdout_test_loader(
            config=config,
            test_indices=test_indices,
            dev_indices=dev_indices,
            fold_stats=fold_stats,
            fold_static_stats=fold_static_stats,
            batch_size=batch_size,
            use_variable_length=False,
            task_keys=list(MTL_LABEL_COLUMNS.keys()),
            num_workers=4,
            all_data_cache=all_data_cache,
            strict_no_filter=holdout_enabled
        )

        if resume_stage == "holdout":
            checkpoint_path = resume_checkpoint
        else:
            # ========== [修复 - 2026-04-23] 使用动态 checkpoint_prefix ==========
            # 根据 t6_auxiliary_mode 自动生成 checkpoint_prefix
            t6_aux_enabled = mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('t6_auxiliary_mode', {}).get('enabled', False)
            base_name = "mtl_v4_t6_protected" if t6_aux_enabled else "mtl_v4_baseline"
            suffix = mtl_config.get('mtl', {}).get('experiment_suffix', '')
            checkpoint_prefix = base_name + suffix
            checkpoint_path = os.path.join("models", f"best_{checkpoint_prefix}_stage3_phase2_fold{fold_idx+1}.pth")

        if resume_stage != "holdout" and checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
            print(f"[Holdout Test] 加载checkpoint: {checkpoint_path}")
        elif resume_stage == "holdout":
            print(f"[Holdout Test] 使用用户指定 checkpoint: {checkpoint_path}")
        else:
            print(f"[Holdout Test] Warning: 未找到 checkpoint {checkpoint_path}，使用当前模型状态")
        # ============================================================

        # ========== [修复] 加载阈值搜索结果并传入评估函数 ==========
        thresholds_to_apply = {}
        threshold_search_dir = mtl_config.get('mtl', {}).get('threshold_search', {}).get('save_dir', 'threshold_search_results')

        for task_key in ["t3", "t4", "t5"]:
            # 尝试加载各阶段的阈值文件 (优先使用 stage3 或 stage3_phase2)
            for stage_suffix in ["stage3_phase2", "stage3", "stage3_phase1"]:
                threshold_file = os.path.join(
                    threshold_search_dir,
                    f"{task_key}_threshold_fold{fold_idx+1}_{stage_suffix}.json"
                )
                if os.path.exists(threshold_file):
                    with open(threshold_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        thresholds_to_apply[task_key] = data.get("best_threshold", 0.5)
                    print(f"[Holdout Test] 加载阈值: {task_key}={thresholds_to_apply[task_key]:.3f} (from {stage_suffix})")
                    break  # 找到第一个有效的就停止

        if thresholds_to_apply:
            print(f"[Holdout Test] 应用阈值搜索结果: {thresholds_to_apply}")
            # 回填 trainer.best_thresholds，使 final_result['thresholds'] 和 val_with_threshold 有值
            trainer.best_thresholds.update(thresholds_to_apply)
        else:
            print(f"[Holdout Test] 未找到阈值搜索结果，使用默认阈值 0.5")
        # ============================================================

        # 评估
        # [新增 2026-05-24] ROC 导出参数
        holdout_model_type = mode if mode.endswith("_clean") else "our_method"
        holdout_result = evaluate_mtl_on_holdout_test(
            model=model,
            test_loader=holdout_loader,
            task_specs=task_specs,
            device=device,
            criterions=criterions,
            thresholds=thresholds_to_apply,
            return_roc_data=True,
            fold_idx=fold_idx + 1,
            checkpoint_path=checkpoint_path,
            model_name="Our method",
            model_type=holdout_model_type,
            return_prediction_table=True,
        )

        # 处理返回值 (可能返回 tuple 或 dict)
        prediction_table_rows = None
        if isinstance(holdout_result, tuple):
            if len(holdout_result) == 3:
                holdout_metrics, roc_export_data, prediction_table_rows = holdout_result
            else:
                holdout_metrics, roc_export_data = holdout_result
        else:
            holdout_metrics = holdout_result
            roc_export_data = None

        print(f"\n[Holdout Test] Fold {fold_idx+1} 测试集指标:")
        for task_key, metrics in holdout_metrics.items():
            print(f"  {task_key}: {metrics}")
        
                # [新增 2026-05-24] 保存 ROC JSON
        if roc_export_data is not None:
            save_fold_roc_json(
                roc_export_data,
                output_dir="results/fig2c_roc_raw",
                model_type=holdout_model_type,
                fold=fold_idx + 1
            )


        # 保存 Fold 统计量 (供后续聚合)
        if prediction_table_rows is not None:
            save_holdout_prediction_table(
                prediction_table_rows,
                output_dir="results/holdout_predictions",
                model_type=holdout_model_type,
                fold=fold_idx + 1
            )

        holdout_cfg = mtl_config.get('mtl', {}).get('holdout', {})
        if holdout_cfg.get('save_fold_stats', True):
            # [修复 - 2026-05-06] 使用动态 checkpoint_prefix
            # [修改 2026-05-27] Clean 架构模式使用专属命名
            suffix = mtl_config.get('mtl', {}).get('experiment_suffix', '')
            if mode.endswith("_clean"):
                stats_checkpoint_prefix = mode + suffix
            else:
                t6_aux_enabled = mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('t6_auxiliary_mode', {}).get('enabled', False)
                base_name = "mtl_v4_t6_protected" if t6_aux_enabled else "mtl_v4_baseline"
                stats_checkpoint_prefix = base_name + suffix

            save_fold_holdout_stats(
                fold_idx=fold_idx,
                stats=fold_stats,
                static_stats=fold_static_stats,
                metrics=holdout_metrics,
                results_dir=holdout_cfg.get('results_dir', 'results'),
                checkpoint_prefix=stats_checkpoint_prefix
            )

        # ========== [新增 2026-06-08] Clinical Interpretation ==========
        if interpret_enabled and holdout_loader is not None:
            print(f"\n{'='*80}")
            print(f"[Interpretation] Fold {fold_idx+1} Clinical Interpretation")
            print("="*80)

            # 确定输出目录
            if interpret_out_dir is None:
                results_dir = holdout_cfg.get('results_dir', 'results')
                interpret_out_dir = os.path.join(results_dir, 'interpretation', f'fold{fold_idx+1}')

            os.makedirs(interpret_out_dir, exist_ok=True)
            print(f"[Interpretation] Output directory: {interpret_out_dir}")

            # 执行解释性推理
            run_holdout_interpretation(
                model=model,
                holdout_loader=holdout_loader,
                config=config,
                mtl_config=mtl_config,
                device=device,
                fold_idx=fold_idx,
                output_dir=interpret_out_dir,
                thresholds=thresholds_to_apply,
                task_specs=task_specs,
                checkpoint_path=checkpoint_path,
                mode=mode,
                interpret_save_intermediates=interpret_save_intermediates,
                interpret_save_attr=interpret_save_attr,
                interpret_context_counterfactual=interpret_context_counterfactual,
                interpret_context_modes=interpret_context_modes
            )
            print(f"[Interpretation] Completed. Results saved to: {interpret_out_dir}")

    # ========== 11. 清理 ==========
    # 导出指标历史到 JSON
    trainer.save_metrics_to_json(save_dir="metrics_logs", fold=fold_idx + 1)

    # SwanLab logging is disabled; no finish call is needed.

    # ========== 12. 返回结果 ==========
    # [新增 - 2026-04-22] 收集 Excel logger 需要的数据
    val_metrics_with_thresholds = None
    if hasattr(trainer, 'compute_val_metrics_with_thresholds'):
        # 始终计算 val 指标 (即使 best_thresholds 为空，多分类任务仍可正确计算)
        val_metrics_with_thresholds = trainer.compute_val_metrics_with_thresholds(
            thresholds=trainer.best_thresholds if hasattr(trainer, 'best_thresholds') else {}
        )

    # 获取任务名称映射
    task_names = trainer.get_task_names_mapping() if hasattr(trainer, 'get_task_names_mapping') else {}

    final_result = {
        'fold': fold_idx + 1,
        # 't6_best_f1': t6_best_f1,
        'stage1_best_f1': trainer.stage1_best_f1 if hasattr(trainer, 'stage1_best_f1') else None,
        'stage2_best_f1': trainer.stage2_best_f1 if hasattr(trainer, 'stage2_best_f1') else None,
        'stage3_best_f1': trainer.stage3_best_f1 if hasattr(trainer, 'stage3_best_f1') else None,
        'stage3_meets_constraint': trainer.stage3_meets_constraint if hasattr(trainer, 'stage3_meets_constraint') else None,
        'holdout_metrics': holdout_metrics,  # Holdout 测试集指标
        # [新增 - 2026-04-22] Excel logger 数据
        'val_metrics': val_metrics_with_thresholds,
        'thresholds': trainer.best_thresholds if hasattr(trainer, 'best_thresholds') else {},
        'task_names': task_names,
        'trainer': trainer  # [新增] 返回 trainer 实例，用于后续评估
    }

    # SwanLab logging is disabled; no DataPorter cleanup is needed.

    print("\n" + "=" * 80)
    print("MTL 训练完成")
    print("=" * 80)
    print(f"Fold {fold_idx + 1} 结果:")
    for k, v in final_result.items():
        if k not in['holdout_metrics', 'val_metrics']:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}:")
            if v is not None:
                for task_key, metrics in v.items():
                    print(f"    {task_key}: {metrics}")

    return final_result


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="HDSTGCN MTL 三阶段训练")

    # [修改 2026-05-07] 运行模式参数 (替代 --mtl_config)
    # [修改 2026-05-27] 添加 CGC, SharedBottom, AdaTT clean 模式
    parser.add_argument("--mode", type=str, default="t6_auxiliary",
                        choices=["baseline", "t6_auxiliary", "experiment_E", "mmoe_clean", "cgc_clean", "shared_bottom_clean", "adatt_clean"],
                        help="MTL 运行模式: baseline (t6剔除) | t6_auxiliary (t6上下文辅助) | experiment_E (baseline+t6弱监督) | *_clean (MMoE/CGC/SharedBottom/AdaTT基线架构)")

    # 配置路径
    parser.add_argument("--config", type=str, default=None,
                        help="基础配置文件路径 (默认 configs/config.yaml)")

    # Fold 参数
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold 编号 (0 to n_folds-1)")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="总 Fold 数")
    parser.add_argument("--run_all_folds", action="store_true",
                        help="自动运行所有 n_folds (遍历 fold 0 到 n_folds-1)")
    parser.add_argument("--start_fold", type=int, default=1,
                        help="从第几个 fold 开始训练 (1-based, 用于 resume 场景, 默认 1)")

    # 恢复训练
    parser.add_argument("--resume", action="store_true",
                        help="恢复训练")
    parser.add_argument("--stage", type=str, default=None,
                        choices=["stage1", "stage2", "stage3", "stage3_phase2", "holdout"],
                        help="恢复阶段 (stage3_phase2 跳过 phase1 直接进入 phase2; holdout 跳过训练直接评估测试集)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="恢复检查点路径")
    parser.add_argument("--checkpoint_stage2", type=str, default=None,
                        help="阶段2检查点路径 (从阶段3恢复时使用)")
    parser.add_argument("--checkpoint_phase1", type=str, default=None,
                        help="[新增] Stage3 Phase1 检查点路径 (从 stage3_phase2 恢复时使用)")

    # 其他参数
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU 设备号")
    parser.add_argument("--disable_swanlab", action="store_true",
                        help="禁用 SwanLab 日志")

    # [新增] Holdout 参数
    parser.add_argument("--holdout_enabled", action="store_true",
                        help="启用 Holdout 测试集模式")
    parser.add_argument("--holdout_split_file", type=str, default="auto",
                        help="Holdout 划分文件路径 (auto=自动查找)")

    # [新增] 配置覆盖参数 (--set key=value)
    parser.add_argument("--set", type=str, nargs='+', default=[],
                        help="覆盖配置参数 (如 --set mtl.ablation.variant=t3_adapter training.batch_size=16)")

    # 亚组分析：传入不同的holdout划分文件以评估不同亚组的性能
    parser.add_argument("--split_file", type=str, default="auto",
                        help="Holdout 划分文件路径 (auto=自动查找)")

    # [新增 2026-06-08] Clinical Interpretation 参数
    parser.add_argument("--interpret_enabled", action="store_true",
                        help="启用 Clinical Interpretation 模式 (在 holdout evaluation 后执行解释性推理)")
    parser.add_argument("--interpret_out_dir", type=str, default=None,
                        help="解释结果输出目录 (默认: <results_dir>/interpretation/fold{fold})")
    parser.add_argument("--interpret_save_intermediates", action="store_true",
                        help="保存中间变量 (PMGT attention, c6_deep, beta_gate_weights)")
    parser.add_argument("--interpret_save_attr", action="store_true",
                        help="保存变量-时间 attribution")
    parser.add_argument("--interpret_context_counterfactual", action="store_true",
                        help="执行 context counterfactual (normal/zero/shuffle)")
    parser.add_argument("--interpret_context_modes", type=str, default="normal,zero,shuffle",
                        help="Context counterfactual 模式列表 (逗号分隔)")
    args = parser.parse_args()

    # 设置设备
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # [新增 2026-05-15] 从 --config 参数中提取 experiment_suffix (消融配置文件支持)
    experiment_suffix = None
    if args.config:
        config_basename = os.path.basename(args.config)
        # 检查是否是消融配置文件 (格式: config_mtl_ablation_*.yaml)
        if config_basename.startswith("config_mtl_ablation_"):
            # 提取 ablation_* 部分
            match = re.match(r"config_mtl_ablation_(.+)\.yaml", config_basename)
            if match:
                experiment_suffix = f"ablation_{match.group(1)}"
                print(f"[Config] Detected experiment_suffix from --config: {experiment_suffix}")

    # 加载基础配置 (消融配置文件不包含 data/features, 使用默认 config.yaml)
    base_config_path = args.config
    if experiment_suffix:
        # 消融实验使用默认基础配置
        base_config_path = None
        print(f"[Config] Ablation mode: using default base config for data/features")

    base_config = Config.load(base_config_path)

    # [新增] 从 --set 参数中提取 experiment_suffix (覆盖 --config 提取的)
    for override in args.set:
        if override.startswith("mtl.experiment_suffix="):
            experiment_suffix = override.split("=", 1)[1].strip()
            print(f"[Config] Detected experiment_suffix from --set: {experiment_suffix}")
            break

    # [修改 2026-05-13] 加载 MTL 配置 - 支持消融配置文件自动加载
    mtl_config = load_mtl_config_with_mode(mode=args.mode, experiment_suffix=experiment_suffix)

    # [新增] 应用配置覆盖
    if args.set:
        mtl_config = apply_config_overrides(mtl_config, args.set)

    # ========== [新增] Holdout 模式初始化 (MTL 专用) ==========
    # 从配置文件读取 Holdout 设置
    holdout_cfg = mtl_config.get('mtl', {}).get('holdout', {})
    holdout_enabled = holdout_cfg.get('enabled', False) or args.holdout_enabled

    dev_indices = None
    test_indices = None
    all_data_cache = None

    if holdout_enabled:
        print("\n[Holdout MTL] 启用 Holdout 测试集模式 (MTL 专用)")

        # 检查划分文件是否存在，不存在则自动生成
        split_file_path = holdout_cfg.get('split_file', 'auto')
        results_dir = holdout_cfg.get('results_dir', 'results')
        mtl_split_file = os.path.join(results_dir, "holdout_split_info_mtl.json")

        # 获取划分参数
        holdout_ratio = holdout_cfg.get('ratio', 0.2)
        holdout_seed = holdout_cfg.get('seed', 42)

        if split_file_path != 'auto' and os.path.exists(split_file_path):
            # 指定了划分文件路径
            split_info = load_holdout_split_info_mtl(
                config_path=split_file_path,
                results_dir=results_dir,
                auto_find_latest=False
            )
        elif os.path.exists(mtl_split_file):
            # MTL 专用划分文件已存在
            split_info = load_holdout_split_info_mtl(
                config_path='auto',
                results_dir=results_dir,
                auto_find_latest=True
            )
        else:
            # 自动生成 MTL 专用划分文件
            print("\n[Holdout MTL] 划分文件不存在，自动生成 MTL 专用划分文件...")
            split_info = create_mtl_holdout_split(
                config=base_config,
                mtl_config=mtl_config,
                holdout_ratio=holdout_ratio,
                holdout_seed=holdout_seed,
                output_dir=results_dir,
                force_regenerate=False
            )

        # 预加载数据缓存 (所有 Fold 共享，避免重复加载)
        all_data_cache = preload_all_data(
            config=base_config,
            use_variable_length=False,
            use_static_features=True,
            strict_no_filter=True
        )
        loaded_filenames = all_data_cache.get('filenames', [])
        print(f"[Holdout MTL] 预加载数据缓存完成: {len(all_data_cache.get('raw_datalist', []))} 样本")

        dev_indices, test_indices = map_holdout_split_filenames_to_indices(
            split_info=split_info,
            loaded_filenames=loaded_filenames
        )

    # ========== 执行训练 ==========

    # ========== [新增 - 2026-04-22] Excel Logger 初始化 ==========
    excel_logger = None

    # [修复 - 2026-05-06] 使用动态 checkpoint_prefix
    t6_aux_enabled = mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('t6_auxiliary_mode', {}).get('enabled', False)
    base_name = "mtl_v4_t6_protected" if t6_aux_enabled else "mtl_v4_baseline"
    suffix = mtl_config.get('mtl', {}).get('experiment_suffix', '')
    excel_checkpoint_name = base_name + suffix

    # 检查是否启用 Excel logger (从配置读取)
    excel_cfg = mtl_config.get('mtl', {}).get('excel_logger', {})
    excel_enabled = excel_cfg.get('enabled', True)  # 默认启用

    if args.run_all_folds and excel_enabled:
        # 构建配置快照（仅用于基本信息摘要）
        config_snapshot = {
            "tasks": mtl_config.get('mtl', {}).get('tasks', {}),
            "training": mtl_config.get('training', {}),
            "data": {
                "test_ratio": holdout_cfg.get('ratio', 0.2) if holdout_enabled else 0,
                "n_folds": args.n_folds,
                "holdout_enabled": holdout_enabled
            },
            "mtl": mtl_config.get('mtl', {})
        }

        # 读取完整的配置文件内容 (用于 Excel logger)
        # [修复 2026-05-07] 使用分离配置文件模式
        # [修改 2026-05-27] Clean 架构模式使用 configs/clean/ 目录
        base_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_mtl_base.yaml')
        if args.mode.endswith("_clean"):
            _mode_cfg_name = f'clean/{args.mode}.yaml'
        elif args.mode == 'experiment_E':
            _mode_cfg_name = 'config_mtl_experiment_E.yaml'
        else:
            _mode_cfg_name = f'config_mtl_{args.mode}.yaml'
        mode_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', _mode_cfg_name)
        raw_yaml_content = None
        try:
            # 合并两个配置文件的内容
            with open(base_config_path, 'r', encoding='utf-8') as f:
                base_yaml = f.read()
            with open(mode_config_path, 'r', encoding='utf-8') as f:
                mode_yaml = f.read()
            raw_yaml_content = f"# Merged config for mode={args.mode}\n\n# === Base Config ===\n{base_yaml}\n\n# === Mode Config ===\n{mode_yaml}"
            print(f"[Excel Logger V2] 已读取配置文件: base + {args.mode}")
        except Exception as e:
            print(f"[Excel Logger V2 Warning] 读取配置文件失败: {e}，将使用字典格式保存")

        # Excel 文件路径
        results_dir = holdout_cfg.get('results_dir', 'results')
        excel_path = os.path.join(results_dir, f"mtl_eval_{excel_checkpoint_name}.xlsx")

        try:
            # Resume 时加载已有 Excel 文件而非创建新文件
            _start_fold = max(1, min(args.start_fold, args.n_folds))
            _is_resume = args.run_all_folds and _start_fold > 1
            excel_logger = create_mtl_excel_logger_v2(
                excel_path=excel_path,
                checkpoint_name=excel_checkpoint_name,
                config_dict=config_snapshot,
                raw_yaml_content=raw_yaml_content,  # 传递完整 YAML 内容
                n_folds=args.n_folds,
                force_new=not _is_resume  # Resume 时加载已有文件
            )
            # Resume: 从已有 Excel sheet 中恢复已完成 fold 的数据
            if _is_resume:
                loaded_folds = excel_logger.load_existing_fold_data()
                print(f"[Excel Logger V2] Resume: 已加载 {len(loaded_folds)} 个已有 fold 数据")
            print(f"\n[Excel Logger V2] 初始化完成: {excel_path}")
            if raw_yaml_content:
                print(f"[Excel Logger V2] config_snapshot 已写入完整 YAML 内容 ({len(raw_yaml_content)} 字符)")
            else:
                print(f"[Excel Logger V2] config_snapshot 已写入 (字典格式)")
        except Exception as e:
            print(f"[Excel Logger V2 Warning] 初始化失败: {e}")
            excel_logger = None
    # =========================================================================

    if args.run_all_folds:
        # 校验 start_fold
        start_fold = max(1, min(args.start_fold, args.n_folds))
        is_resume = start_fold > 1

        # 自动运行所有 Fold
        print("\n" + "=" * 80)
        if is_resume:
            print(f"MTL 全 Fold 训练模式 (RESUME): 从 Fold {start_fold} 开始，共 {args.n_folds} 个 Fold")
        else:
            print(f"MTL 全 Fold 训练模式: 将运行 {args.n_folds} 个 Fold")
        if holdout_enabled:
            print(f"[Holdout] 每个 Fold 结束后将在独立测试集 ({len(test_indices)} 样本) 上评估")
        print("=" * 80)

        all_fold_results = []
        all_holdout_metrics = []  # [新增] 收集各 Fold Holdout 指标

        # ============ Resume: 加载已完成 fold 的结果 ============
        if is_resume:
            print(f"\n[Resume] 加载 Fold 1~{start_fold - 1} 的已有结果...")
            checkpoint_prefix_resume = excel_checkpoint_name  # e.g. mtl_v4_t6_protectedablation_D_feat03
            for prev_fold_idx in range(start_fold - 1):
                fold_num = prev_fold_idx + 1
                prev_result = None

                # 从 holdout stats JSON 加载
                holdout_stats_path = os.path.join(
                    holdout_cfg.get('results_dir', 'results'),
                    f"{checkpoint_prefix_resume}_holdout_fold{fold_num}_stats.json"
                )
                if os.path.exists(holdout_stats_path):
                    try:
                        with open(holdout_stats_path, 'r', encoding='utf-8') as f:
                            prev_data = json.load(f)
                        prev_holdout_metrics = prev_data.get('holdout_metrics', {})

                        # 从阈值搜索 JSON 加载阈值
                        prev_thresholds = {}
                        threshold_search_dir = mtl_config.get('mtl', {}).get('eval', {}).get(
                            'binary_threshold_search', {}).get('save_dir', 'threshold_search_results')
                        for task_key in ['t3', 't4', 't5']:
                            tf = os.path.join(
                                threshold_search_dir,
                                f"{task_key}_threshold_fold{fold_num}_stage3_phase2.json"
                            )
                            if os.path.exists(tf):
                                with open(tf, 'r', encoding='utf-8') as f:
                                    td = json.load(f)
                                prev_thresholds[task_key] = {
                                    "best_threshold": td.get("best_threshold", 0.5),
                                    "best_f1": td.get("best_f1_on_val", 0),
                                    "baseline_f1": td.get("baseline_f1_at_0.5_on_val", 0),
                                    "improvement": td.get("improvement", 0)
                                }

                        # 从 metrics_logs 中的最新 incremental JSON 加载 val_metrics
                        prev_val_metrics = {}
                        metrics_dir = "metrics_logs"
                        # 查找该 fold 最新的 incremental metrics JSON
                        import glob as _glob
                        pattern = os.path.join(metrics_dir, f"mtl_metrics_fold{fold_num}_*_incremental.json")
                        incremental_files = sorted(_glob.glob(pattern))
                        if incremental_files:
                            try:
                                with open(incremental_files[-1], 'r', encoding='utf-8') as f:
                                    metrics_data = json.load(f)
                                # 从 incremental JSON 提取最终 val 指标
                                if isinstance(metrics_data, dict):
                                    final_stage = metrics_data.get('stage3_phase2', metrics_data.get('stage3', {}))
                                    if isinstance(final_stage, dict):
                                        for task_key, task_data in final_stage.items():
                                            if task_key.startswith('t') and isinstance(task_data, dict):
                                                val_metrics_raw = task_data.get('val_metrics', {})
                                                if val_metrics_raw:
                                                    prev_val_metrics[task_key] = val_metrics_raw
                            except Exception as e:
                                print(f"[Resume Warning] 读取 fold{fold_num} metrics 失败: {e}")

                        # 构造与 run_mtl_training 返回格式兼容的字典
                        prev_result = {
                            'fold': fold_num,
                            'holdout_metrics': prev_holdout_metrics,
                            'val_metrics': prev_val_metrics if prev_val_metrics else prev_holdout_metrics,
                            'thresholds': prev_thresholds,
                            'task_names': {},  # 后面从最新 fold 获取
                            'trainer': None
                        }

                        print(f"[Resume] Fold {fold_num}: holdout_metrics 已加载 (tasks: {list(prev_holdout_metrics.keys())})")
                    except Exception as e:
                        print(f"[Resume Warning] 加载 Fold {fold_num} 结果失败: {e}")
                else:
                    print(f"[Resume Warning] Fold {fold_num} 的 holdout stats 文件不存在: {holdout_stats_path}")

                if prev_result is not None:
                    all_fold_results.append({
                        'fold': fold_num,
                        'result': prev_result
                    })
                    if prev_result.get('holdout_metrics'):
                        all_holdout_metrics.append(prev_result['holdout_metrics'])

            print(f"[Resume] 已加载 {len(all_fold_results)} 个已完成 fold 的结果")

        for fold_idx in range(start_fold - 1, args.n_folds):
            print(f"\n{'=' * 80}")
            print(f"开始训练 Fold {fold_idx + 1}/{args.n_folds}")
            print("=" * 80)

            fold_result = run_mtl_training(
                config=base_config,
                mtl_config=mtl_config,
                fold_idx=fold_idx,
                n_folds=args.n_folds,
                device=device,
                resume_stage=None,  # 全 Fold 模式不支持恢复
                resume_checkpoint=None,
                checkpoint_stage2=None,
                disable_swanlab=args.disable_swanlab,
                # [新增] Holdout 参数
                dev_indices=dev_indices,
                test_indices=test_indices,
                holdout_enabled=holdout_enabled,
                all_data_cache=all_data_cache,
                # [新增 2026-05-27] 运行模式
                mode=args.mode,
                # [新增 2026-06-08] Clinical Interpretation 参数
                interpret_enabled=args.interpret_enabled,
                interpret_out_dir=args.interpret_out_dir,
                interpret_save_intermediates=args.interpret_save_intermediates,
                interpret_save_attr=args.interpret_save_attr,
                interpret_context_counterfactual=args.interpret_context_counterfactual,
                interpret_context_modes=args.interpret_context_modes
            )
            all_fold_results.append({
                'fold': fold_idx + 1,
                'result': fold_result
            })

            # 收集 Holdout 指标
            if fold_result.get('holdout_metrics'):
                all_holdout_metrics.append(fold_result['holdout_metrics'])

            # ========== [新增 - 2026-04-22] Excel Logger V2 写入 (每个 fold 3 个 sheet) ==========
            if excel_logger is not None:
                try:
                    # 获取数据
                    val_metrics = fold_result.get('val_metrics')
                    holdout_metrics = fold_result.get('holdout_metrics')
                    thresholds_simple = fold_result.get('thresholds', {})
                    task_names = fold_result.get('task_names', {})

                    # 构建完整的阈值信息（包含 best_f1, baseline_f1, improvement）
                    thresholds_full = {}
                    threshold_search_dir = mtl_config.get('mtl', {}).get('eval', {}).get(
                        'binary_threshold_search', {}).get('save_dir', 'threshold_search_results')

                    for task_key in ['t3', 't4', 't5']:
                        # 尝试从阈值搜索 JSON 文件加载完整信息
                        threshold_file = os.path.join(
                            threshold_search_dir,
                            f"{task_key}_threshold_fold{fold_idx+1}_stage3_phase2.json"
                        )
                        if os.path.exists(threshold_file):
                            with open(threshold_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                thresholds_full[task_key] = {
                                    "best_threshold": data.get("best_threshold", 0.5),
                                    "best_f1": data.get("best_f1_on_val", 0),
                                    "baseline_f1": data.get("baseline_f1_at_0.5_on_val", 0),
                                    "improvement": data.get("improvement", 0)
                                }
                        elif task_key in thresholds_simple:
                            # 如果文件不存在，使用简化版阈值
                            thresholds_full[task_key] = {
                                "best_threshold": thresholds_simple[task_key],
                                "best_f1": 0,
                                "baseline_f1": 0,
                                "improvement": 0
                            }

                    # 一行调用写入 3 个 sheet（val_fold{N}, holdout_fold{N}, threshold_fold{N}）
                    if val_metrics and holdout_metrics:
                        excel_logger.write_fold_results(
                            fold=fold_idx + 1,
                            val_metrics=val_metrics,
                            holdout_metrics=holdout_metrics,
                            thresholds=thresholds_full,
                            task_names=task_names
                        )
                        print(f"[Excel Logger V2] Fold {fold_idx + 1} 已写入 3 个 sheet 并保存")
                    else:
                        print(f"[Excel Logger V2 Warning] Fold {fold_idx + 1} 数据不完整，跳过写入")

                except Exception as e:
                    print(f"[Excel Logger V2 Warning] Fold {fold_idx + 1} 写入失败: {e}")
            # ============================================================

        # 打印汇总
        print("\n" + "=" * 80)
        print("全 Fold 训练完成汇总")
        print("=" * 80)
        for fr in all_fold_results:
            print(f"  Fold {fr['fold']}: {fr['result']}")

        # [新增] 聚合 Holdout 指标
        if holdout_enabled and all_holdout_metrics:
            aggregated_metrics = aggregate_mtl_holdout_metrics(all_holdout_metrics)
            print_holdout_summary(aggregated_metrics)

            # 保存聚合结果
            save_holdout_summary(
                aggregated_metrics,
                results_dir=holdout_cfg.get('results_dir', 'results')
            )

        # ========== [新增 - 2026-04-22] Excel Logger V2 汇总 (4 个 sheet) ==========
        if excel_logger is not None:
            try:
                # 获取任务名称映射（从最后一个 fold）
                task_names = {}
                if all_fold_results:
                    last_fold_result = all_fold_results[-1]['result']
                    task_names = last_fold_result.get('task_names', {})

                # 写入汇总 sheet（val_summary, holdout_summary, threshold_summary）
                excel_logger.write_summary_sheets(task_names=task_names)

                print(f"[Excel Logger V2] 汇总完成：已写入 4 个汇总 sheet")
                print(f"[Excel Logger V2] Excel 文件保存至: {excel_logger.excel_path}")
                print(f"[Excel Logger V2] 共 {excel_logger.completed_folds} 个 fold，{len(excel_logger.wb.sheetnames)} 个 sheet")
            except Exception as e:
                print(f"[Excel Logger V2 Warning] 汇总写入失败: {e}")
        # ===========================================================

    else:
        # 单次训练 (指定 fold)
        fold_result = run_mtl_training(
            config=base_config,
            mtl_config=mtl_config,
            fold_idx=args.fold,
            n_folds=args.n_folds,
            device=device,
            resume_stage=args.stage if args.resume else None,
            resume_checkpoint=args.checkpoint,
            checkpoint_stage2=args.checkpoint_stage2,
            checkpoint_phase1=args.checkpoint_phase1,  # [新增]
            disable_swanlab=args.disable_swanlab,
            # [新增] Holdout 参数
            dev_indices=dev_indices,
            test_indices=test_indices,
            holdout_enabled=holdout_enabled,
            all_data_cache=all_data_cache,
            # [新增 2026-05-27] 运行模式
            mode=args.mode,
            # [新增 2026-06-08] Clinical Interpretation 参数
            interpret_enabled=args.interpret_enabled,
            interpret_out_dir=args.interpret_out_dir,
            interpret_save_intermediates=args.interpret_save_intermediates,
            interpret_save_attr=args.interpret_save_attr,
            interpret_context_counterfactual=args.interpret_context_counterfactual,
            interpret_context_modes=args.interpret_context_modes
        )

        # [新增 - 2026-05-06] 单 fold 模式也写入 Excel
        excel_cfg = mtl_config.get('mtl', {}).get('excel_logger', {})
        excel_enabled = excel_cfg.get('enabled', True)

        if excel_enabled and fold_result is not None:
            try:
                # [修复 - 2026-05-06] 使用动态 checkpoint_prefix
                # [修改 2026-05-27] Clean 架构模式使用专属命名
                suffix = mtl_config.get('mtl', {}).get('experiment_suffix', '')
                if args.mode.endswith("_clean"):
                    excel_checkpoint_name = args.mode + suffix
                else:
                    t6_aux_enabled = mtl_config.get('mtl', {}).get('hcgc_v4', {}).get('t6_auxiliary_mode', {}).get('enabled', False)
                    base_name = "mtl_v4_t6_protected" if t6_aux_enabled else "mtl_v4_baseline"
                    excel_checkpoint_name = base_name + suffix

                results_dir = holdout_cfg.get('results_dir', 'results')
                excel_path = os.path.join(results_dir, f"mtl_eval_{excel_checkpoint_name}.xlsx")

                # 读取完整 YAML (分离配置模式)
                raw_yaml_content = None
                try:
                    base_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_mtl_base.yaml')
                    # [修改 2026-05-27] Clean 架构模式使用 configs/clean/ 目录
                    if args.mode.endswith("_clean"):
                        _mode_cfg_name2 = f'clean/{args.mode}.yaml'
                    elif args.mode == 'experiment_E':
                        _mode_cfg_name2 = 'config_mtl_experiment_E.yaml'
                    else:
                        _mode_cfg_name2 = f'config_mtl_{args.mode}.yaml'
                    mode_path = os.path.join(os.path.dirname(__file__), '..', 'configs', _mode_cfg_name2)
                    with open(base_path, 'r', encoding='utf-8') as f:
                        base_yaml = f.read()
                    with open(mode_path, 'r', encoding='utf-8') as f:
                        mode_yaml = f.read()
                    raw_yaml_content = f"# Merged config for mode={args.mode}\n\n# === Base Config ===\n{base_yaml}\n\n# === Mode Config ===\n{mode_yaml}"
                except Exception:
                    pass

                config_snapshot = {
                    "tasks": mtl_config.get('mtl', {}).get('tasks', {}),
                    "training": mtl_config.get('training', {}),
                    "data": {
                        "test_ratio": holdout_cfg.get('ratio', 0.2) if holdout_enabled else 0,
                        "n_folds": args.n_folds,
                        "holdout_enabled": holdout_enabled
                    },
                    "mtl": mtl_config.get('mtl', {})
                }

                single_excel_logger = create_mtl_excel_logger_v2(
                    excel_path=excel_path,
                    checkpoint_name=excel_checkpoint_name,
                    config_dict=config_snapshot,
                    raw_yaml_content=raw_yaml_content,
                    n_folds=1,
                    force_new=True
                )

                val_metrics = fold_result.get('val_metrics')
                holdout_metrics = fold_result.get('holdout_metrics')
                thresholds_simple = fold_result.get('thresholds', {})
                task_names = fold_result.get('task_names', {})

                # 构建完整阈值信息
                thresholds_full = {}
                threshold_search_dir = mtl_config.get('mtl', {}).get('eval', {}).get(
                    'binary_threshold_search', {}).get('save_dir', 'threshold_search_results')
                for task_key in ['t3', 't4', 't5']:
                    threshold_file = os.path.join(
                        threshold_search_dir,
                        f"{task_key}_threshold_fold{args.fold+1}_stage3_phase2.json"
                    )
                    if os.path.exists(threshold_file):
                        with open(threshold_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            thresholds_full[task_key] = {
                                "best_threshold": data.get("best_threshold", 0.5),
                                "best_f1": data.get("best_f1_on_val", 0),
                                "baseline_f1": data.get("baseline_f1_at_0.5_on_val", 0),
                                "improvement": data.get("improvement", 0)
                            }
                    elif task_key in thresholds_simple:
                        thresholds_full[task_key] = {
                            "best_threshold": thresholds_simple[task_key],
                            "best_f1": 0,
                            "baseline_f1": 0,
                            "improvement": 0
                        }

                if val_metrics and holdout_metrics:
                    single_excel_logger.write_fold_results(
                        fold=args.fold + 1,
                        val_metrics=val_metrics,
                        holdout_metrics=holdout_metrics,
                        thresholds=thresholds_full,
                        task_names=task_names
                    )
                    single_excel_logger.write_summary_sheets(task_names=task_names)
                    print(f"\n[Excel Logger V2] 单 Fold Excel 已保存: {excel_path}")
                else:
                    print(f"[Excel Logger V2 Warning] 数据不完整，跳过写入")
            except Exception as e:
                print(f"[Excel Logger V2 Warning] 单 Fold Excel 写入失败: {e}")


# SwanLab 已强制禁用，避免 import / init / log 带来的额外开销。
SWANLAB_AVAILABLE = False


if __name__ == "__main__":
    main()
