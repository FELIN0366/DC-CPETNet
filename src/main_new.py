"""
主训练脚本 - 整合所有模块
支持训练、推理、K折交叉验证

配置来源: configs/config.yaml (唯一入口)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 添加src到路径
sys.path.insert(0, os.path.dirname(__file__))
os.environ['NO_PROXY'] = 'swanlog.com'

from torch.utils.data import WeightedRandomSampler

from config import Config
from feature_mapping import create_adjacency_matrix, get_feature_list, get_nine_graph_config
from model import HDSTGCN
from baselines import ResNet1D, LSTMNet, MedNet, STFinalNet as BaselineSTFinalNet, CNNGAF, KESTNet
from dataset_new import CPETDatasetNew, CPETDatasetNewKFold, collate_fn_variable_length
from train_with_swanlab import train_with_swanlab, train_kfold_with_swanlab, build_optimizer_with_gamma_lr
from train_with_swanlab import BINARY_MINORITY_METRIC_TASKS, MINORITY_IDX_MAP, compute_binary_minority_metrics

# =============================================================================
# [新增] ROC 导出配置 (Figure 2C)
# =============================================================================
# 仅对以下三个二分类任务导出 ROC 数据
ROC_EXPORT_TASKS = [
    "标准心电运动负荷试验",
    "运动中换气肺功能",
    "心率储备",
]

# 任务名称到论文编号的映射
TARGET_TO_TASK_KEY = {
    "标准心电运动负荷试验": "t3",
    "运动中换气肺功能": "t4",
    "心率储备": "t5",
}

# 模型注册表
MODEL_REGISTRY = {
    "HDSTGCN": HDSTGCN,
    "STFinalNet": BaselineSTFinalNet,
    "resnet": ResNet1D,
    "lstm": LSTMNet,
    "mednet": MedNet,
    "CNNGAF": CNNGAF,
    "KESTNet": KESTNet,
}


def get_model_class(model_name: str):
    """根据名称获取模型类"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"模型 '{model_name}' 未注册。可选: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def set_seed(seed: int):
    """设置随机种子"""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_weighted_sampler(dataset, label_mapping: list):
    """
    [已废弃] 为长尾数据集创建 WeightedRandomSampler

    ⚠️ 废弃原因：
    1. 在极度不平衡二分类场景（阳性率 < 20%）导致严重过拟合
    2. 让模型反复看到极少数阳性样本，死记硬背噪声纹理而非学习有效特征
    3. 在任务3（心电图，阳性率9.8%）和任务5（心率储备用尽，阳性率19.5%）
       中导致性能崩塌（少数类 Recall=0）

    推荐替代方案：
    - Dice Loss: 直接优化 F1-Score 重叠度（config.yaml: loss.type: "Dice")
    - LDAM Loss: 决策边界重塑（config.yaml: loss.type: "LDAM"）
    - BCEWithLogitsLoss + pos_weight: 基线方案

    核心机制（历史记录）:
    - 动态采样：在每个 Epoch 组装 Batch 时，按照样本所属类别的权重进行有放回抽样
    - 权重计算：类别频次的倒数
    - 隔离性：不改变底层 Dataset 的实际大小和原有数据结构

    Args:
        dataset: 数据集对象 (CPETDatasetSubset 或类似)
        label_mapping: 标签名称列表 (config.part_actions)

    Returns:
        WeightedRandomSampler 实例
    """
    # 1. 提取标签列表
    # dataset.labellist 返回子集的标签列表 (字符串形式)
    label_list = dataset.labellist

    # 2. 创建标签到索引的映射 (label_mapping 是标签名称列表)
    label_to_idx = {label: idx for idx, label in enumerate(label_mapping)}

    # 3. 将标签转换为索引
    target_list = torch.tensor([label_to_idx[label] for label in label_list], dtype=torch.long)

    # 4. 统计各类别频次
    class_count = torch.bincount(target_list)

    # 5. 计算类别权重 (频次的倒数)
    # 增加小常数 1e-6 防止除零异常
    class_weights = 1.0 / (class_count.float() + 1e-6)

    # 6. 为每个样本映射权重
    sample_weights = class_weights[target_list]

    # 7. 创建 WeightedRandomSampler
    # replacement=True 必须开启，以支持有放回抽样
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )

    return sampler


# =============================================================================
# [新增] ROC 导出辅助函数 (Figure 2C)
# =============================================================================

def is_binary_roc_export_task(config, n_classes, is_multilabel=False):
    """
    判断当前任务是否需要导出 ROC 数据

    Args:
        config: Config 对象
        n_classes: 类别数
        is_multilabel: 是否为多标签模式

    Returns:
        bool: 是否需要导出 ROC
    """
    target_col_name = getattr(config.features, "target_col_name", "")
    return (
        target_col_name in ROC_EXPORT_TASKS
        and n_classes == 2
        and not is_multilabel
    )


def compute_single_task_fold_roc_data(
    all_labels,
    all_probs,
    minority_idx,
    fold_num,
    target_col_name,
    model_path,
    model_name="Single-task model",
    model_type="single_task"
):
    """
    计算单个 fold 的 ROC 数据

    Args:
        all_labels: 真实标签 (numpy array)
        all_probs: 模型输出概率 (numpy array, shape: [N, 2])
        minority_idx: 少数类索引
        fold_num: Fold 编号
        target_col_name: 任务名称
        model_path: 模型路径
        model_name: 模型名称 (用于 Excel)
        model_type: 模型类型 (用于 Excel)

    Returns:
        sample_df: 逐样本概率 DataFrame
        roc_df: ROC 曲线点 DataFrame
        auc: ROC AUC 值
        auc_reverse: 反向 AUC (诊断字段)
        n_positive: 阳性样本数
        n_negative: 阴性样本数
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    task_key = TARGET_TO_TASK_KEY[target_col_name]

    y_true = np.asarray(all_labels)
    y_score = np.asarray(all_probs)[:, minority_idx]  # 少数类概率
    y_true_minor = (y_true == minority_idx).astype(int)

    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true_minor, y_score)
    auc = roc_auc_score(y_true_minor, y_score)
    auc_reverse = roc_auc_score(y_true_minor, 1.0 - y_score)  # 诊断字段

    n_positive = int(y_true_minor.sum())
    n_negative = len(y_true_minor) - n_positive

    # 构建逐样本 DataFrame
    sample_df = pd.DataFrame({
        "model_name": model_name,
        "model_type": model_type,
        "task_key": task_key,
        "task_name": target_col_name,
        "target_col_name": target_col_name,
        "fold": fold_num,
        "sample_index": np.arange(len(y_true)),
        "y_true": y_true,
        "y_true_minor": y_true_minor,
        "y_score": y_score,
        "positive_class": minority_idx,
        "checkpoint_path": model_path,
    })

    # 构建 ROC 点 DataFrame
    roc_df = pd.DataFrame({
        "model_name": model_name,
        "model_type": model_type,
        "task_key": task_key,
        "task_name": target_col_name,
        "target_col_name": target_col_name,
        "fold": fold_num,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
        "auc": auc,
        "auc_reverse": auc_reverse,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_class": minority_idx,
        "checkpoint_path": model_path,
    })

    return sample_df, roc_df, auc, auc_reverse, n_positive, n_negative


def compute_mean_roc_from_fold_points(roc_points_fold_df, model_type, task_key, n_points=101):
    """
    从各 fold 的 ROC 点计算平均 ROC 曲线

    Args:
        roc_points_fold_df: 各 fold ROC 点 DataFrame
        model_type: 模型类型
        task_key: 任务编号
        n_points: 统一横轴点数

    Returns:
        mean_df: 平均 ROC DataFrame
    """
    mean_fpr = np.linspace(0.0, 1.0, n_points)
    interp_tprs = []
    fold_aucs = []

    # 筛选当前任务的数据
    sub = roc_points_fold_df[
        (roc_points_fold_df["model_type"] == model_type)
        & (roc_points_fold_df["task_key"] == task_key)
    ]

    for fold in sorted(sub["fold"].unique()):
        fold_df = sub[sub["fold"] == fold].sort_values("fpr")
        fpr = fold_df["fpr"].to_numpy()
        tpr = fold_df["tpr"].to_numpy()
        auc = fold_df["auc"].iloc[0]

        # 插值到统一横轴
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0

        interp_tprs.append(interp_tpr)
        fold_aucs.append(auc)

    interp_tprs = np.asarray(interp_tprs)
    fold_aucs = np.asarray(fold_aucs)

    # 提取第一行的元信息
    first = sub.iloc[0]

    mean_df = pd.DataFrame({
        "model_name": first["model_name"],
        "model_type": model_type,
        "task_key": task_key,
        "task_name": first["task_name"],
        "target_col_name": first["target_col_name"],
        "mean_fpr": mean_fpr,
        "mean_tpr": interp_tprs.mean(axis=0),
        "std_tpr": interp_tprs.std(axis=0, ddof=1),
        "mean_auc": fold_aucs.mean(),
        "std_auc": fold_aucs.std(ddof=1),
        "positive_class": first["positive_class"],
    })

    return mean_df


def export_single_task_roc_excel(
    output_excel,
    sample_scores_df,
    roc_points_fold_df,
    roc_points_mean_df,
    auc_summary_df,
    run_info_df
):
    """
    导出 ROC 数据到 Excel

    Args:
        output_excel: 输出路径
        sample_scores_df: 逐样本分数 DataFrame
        roc_points_fold_df: 各 fold ROC 点 DataFrame
        roc_points_mean_df: 平均 ROC DataFrame
        auc_summary_df: AUC 汇总 DataFrame
        run_info_df: 运行信息 DataFrame

    Returns:
        output_excel: 实际输出路径
    """
    import os

    os.makedirs(os.path.dirname(output_excel), exist_ok=True)

    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        sample_scores_df.to_excel(writer, sheet_name="sample_scores", index=False)
        roc_points_fold_df.to_excel(writer, sheet_name="roc_points_fold", index=False)
        roc_points_mean_df.to_excel(writer, sheet_name="roc_points_mean", index=False)
        auc_summary_df.to_excel(writer, sheet_name="auc_summary", index=False)
        run_info_df.to_excel(writer, sheet_name="run_info", index=False)

    return output_excel


def main():
    """主函数"""
    print("=" * 80)
    print("CPET疾病分类 - 主训练脚本")
    print("=" * 80)

    # =========================================================================
    # 加载配置 (唯一入口)
    # =========================================================================
    config = Config.load()
    config.runtime.disable_swanlab = True
    config.print_config()

    # SwanLab 已强制禁用
    os.environ['DISABLE_SWANLAB'] = 'true'

    # 加载 SwanLab 环境配置
    swanlab_env_file = os.path.join(os.path.dirname(__file__), '..', 'swanlab.env')
    if os.path.exists(swanlab_env_file):
        with open(swanlab_env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

    # 设置随机种子
    set_seed(config.training.random_seed)

    # 设置设备
    device = torch.device(f"cuda:{config.runtime.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取模型类
    ModelClass = get_model_class(config.model.name)
    print(f"模型架构: {ModelClass.__name__}")

    # =========================================================================
    # 根据运行模式执行
    # =========================================================================
    if config.runtime.mode == "kfold":
        _run_kfold(config, ModelClass, device)

    elif config.runtime.mode == "train":
        _run_train(config, ModelClass, device)

    elif config.runtime.mode == "inference":
        _run_inference(config, ModelClass, device)


def _run_train(config: Config, ModelClass, device):
    """训练模式"""
    print("\n" + "=" * 80)
    print("训练模式")
    print("=" * 80)

    # 确定是否使用变长模式
    use_var_length = getattr(config.model, 'use_variable_length', False) and config.model.name == "HDSTGCN"

    # 确定是否使用静态特征
    use_static = False
    static_dim = 16
    static_ablation = "full"
    if hasattr(config.model, 'static_features') and config.model.static_features is not None:
        use_static = config.model.static_features.enabled
        static_dim = config.model.static_features.static_dim
        static_ablation = config.model.static_features.ablation

    # [新增] 确定是否多标签模式
    is_multilabel = False
    if hasattr(config, 'task') and config.task is not None:
        is_multilabel = (config.task.mode == "multi_label")
        if is_multilabel:
            print("[任务模式] 多标签分类")
    else:
        print("[任务模式] 单标签分类")

    # 创建单一数据集实例 (解决数据泄露问题)
    print("\n加载数据集...")
    full_dataset = CPETDatasetNew(
        config,
        test_ratio=config.data.test_ratio,
        feature_indices=config.features.channels,
        use_variable_length=use_var_length,
        max_length=config.data.max_length,
        use_static_features=use_static
    )

    # [新增] 同步多标签状态 (由数据集设置)
    is_multilabel = getattr(config, 'is_multilabel', False)
    if is_multilabel:
        print(f"[多标签] 标签数: {full_dataset.n_classes}")

    # 更新 num_static_features (动态计算)
    if use_static and hasattr(full_dataset, 'num_static_features'):
        if hasattr(config.model, 'static_features') and config.model.static_features:
            config.model.static_features.num_features = full_dataset.num_static_features
            print(f"[训练] 静态特征数: {full_dataset.num_static_features}")

    # [新增] 同步 Known-T6 Context 信息
    if hasattr(config, 'known_t6_context') and config.known_t6_context.enabled:
        if hasattr(full_dataset, 't6_n_classes') and full_dataset.t6_n_classes > 0:
            config.t6_n_classes = full_dataset.t6_n_classes  # 传递给 model
            print(f"[Known-T6 Context] t6_n_classes={full_dataset.t6_n_classes}")

    # 获取训练/测试子集
    train_dataset = full_dataset.get_split("train")
    val_dataset = full_dataset.get_split("test")

    # 更新类别信息
    config.update_with_dataset(train_dataset)

    # [新增] 创建 WeightedRandomSampler (如果启用且为单标签模式)
    train_sampler = None
    use_weighted_sampler = config.sampler.enabled and not is_multilabel
    drop_last_flag = config.sampler.drop_last if use_weighted_sampler else False

    if use_weighted_sampler:
        print("\n[WeightedRandomSampler] 正在创建采样器...")
        train_sampler = create_weighted_sampler(train_dataset, config.part_actions)
        print(f"[WeightedRandomSampler] 已启用 (replacement=True, drop_last={drop_last_flag})")
        # 打印类别分布信息
        label_list = train_dataset.labellist
        from collections import Counter
        label_counts = Counter(label_list)
        print(f"[WeightedRandomSampler] 类别分布:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            weight = 1.0 / (count + 1e-6)
            print(f"  {label}: {count} 样本, 权重={weight:.4f}")

    # 创建数据加载器
    if use_var_length:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.training.batch_size,
            shuffle=False if train_sampler else True,  # 使用 sampler 时必须 shuffle=False
            sampler=train_sampler,
            num_workers=0,
            collate_fn=collate_fn_variable_length,
            drop_last=drop_last_flag
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1,
            shuffle=False, num_workers=0,
            collate_fn=collate_fn_variable_length
        )
        print(f"\n数据加载器: 变长模式")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.training.batch_size,
            shuffle=False if train_sampler else True,  # 使用 sampler 时必须 shuffle=False
            sampler=train_sampler,
            num_workers=0,
            drop_last=drop_last_flag
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1,
            shuffle=False, num_workers=0
        )
        print(f"\n数据加载器: 固定长度模式")

    if use_static:
        print(f"静态特征: 已启用 (dim={static_dim}, ablation={static_ablation})")

    print(f"\n数据集信息:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    if is_multilabel:
        print(f"  标签数: {config.n_class}")
    else:
        print(f"  类别数: {config.n_class}")
    print(f"  特征数: {config.features.num_channels}")

    # 创建模型
    model = _create_model(config, ModelClass, device, use_var_length)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数: {total_params:,} (可训练: {trainable_params:,})")

    # 获取邻接矩阵 (HDSTGCN 不需要)
    # [修复] 传递 optional_keys 以支持 vco2 和 o2pulse 特征
    optional_keys = []
    if hasattr(config.features, 'o2pulse_enabled') and config.features.o2pulse_enabled:
        optional_keys.append('o2pulse')
    if hasattr(config.features, 'vco2_enabled') and config.features.vco2_enabled:
        optional_keys.append('vco2')
    semantic_adj = create_adjacency_matrix(config.features.adapt_mode, optional_keys if optional_keys else None)
    adj = torch.from_numpy(semantic_adj).float()

    # 创建优化器 (支持 gamma 参数学习率分离)
    gamma_lr_scale = 0.3
    if hasattr(config.model, 'prior_gate') and config.model.prior_gate is not None:
        gamma_lr_scale = config.model.prior_gate.gamma_lr_scale

    if config.model.name == "HDSTGCN" and config.model.graph_ablation == "prior_masked":
        optimizer = build_optimizer_with_gamma_lr(
            model,
            base_lr=config.training.lr,
            weight_decay=config.training.weight_decay,
            gamma_lr_scale=gamma_lr_scale
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay
        )

    # 训练
    best_acc, cm_or_metrics, report, _ = train_with_swanlab(
        model, train_loader, val_loader, optimizer,
        device, config, adj, val_dataset=val_dataset
    )

    # [修改] 根据模式打印最终结果
    if is_multilabel:
        print(f"\n最终 Macro-F1: {best_acc:.4f}")
        if cm_or_metrics is not None:
            print(f"最终评估指标:")
            print(f"  Macro-F1: {cm_or_metrics.get('macro_f1', 0):.4f}")
            print(f"  Micro-F1: {cm_or_metrics.get('micro_f1', 0):.4f}")
            print(f"  mAP: {cm_or_metrics.get('mAP', 0):.4f}")
    else:
        print(f"\n最终验证准确率: {best_acc:.4f}")
        if cm_or_metrics is not None:
            print(f"最终评估指标:")
            print(f"  ACC:      {cm_or_metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {cm_or_metrics.get('precision', 0):.4f}")
            print(f"  Recall:   {cm_or_metrics.get('recall', 0):.4f}")
            print(f"  F1-score: {cm_or_metrics.get('f1_score', 0):.4f}")
            print(f"  AUROC:    {cm_or_metrics.get('auroc', 0):.4f}")
            print(f"  AUPRC:    {cm_or_metrics.get('auprc', 0):.4f}")


def _run_kfold(config: Config, ModelClass, device):
    """K折交叉验证模式"""
    print("\n" + "=" * 80)
    print("K折交叉验证模式")
    print("=" * 80)

    # 确定是否多标签模式
    is_multilabel = False
    if hasattr(config, 'task') and config.task is not None:
        is_multilabel = (config.task.mode == "multi_label")
        if is_multilabel:
            print("[K-Fold] 多标签分类模式")

    # =========================================================================
    # [新增] 检查是否跳过 K-Fold 训练，直接评估测试集
    # =========================================================================
    skip_kfold = getattr(config.runtime, 'skip_kfold', False) or getattr(config.runtime, 'eval_only', False)

    if skip_kfold:
        print("\n[跳过 K-Fold] 直接进行测试集评估模式")
        test_results = _run_holdout_test_only(config, ModelClass, device, is_multilabel)
        if test_results:
            print("\n测试集评估完成!")
        return

    # =========================================================================
    # 正常执行 K 折验证
    # =========================================================================
    kfold_results, test_results = train_kfold_with_swanlab(
        ModelClass, config, n_folds=config.runtime.n_folds, device=device
    )

    # 保存结果
    import json
    suffix = "_multilabel" if is_multilabel else ""
    exp_suffix = config.exp_suffix  # 获取实验后缀
    result_file = os.path.join(
        config.data.output_root,
        f"kfold_results_{config.model.name}_{config.features.adapt_mode}{suffix}{exp_suffix}.json"
    )

    with open(result_file, 'w', encoding='utf-8') as f:
        results_json = []
        for r in kfold_results:
            if is_multilabel:
                results_json.append({
                    'fold': r['fold'],
                    'macro_f1': float(r['macro_f1']),
                    'metrics': r.get('metrics', {}),
                    'report': r['report']
                })
            else:
                # [修改] 现在返回完整的 metrics 字典
                m = r.get('metrics', {})
                result_entry = {
                    'fold': r['fold'],
                    'accuracy': float(m.get('accuracy', 0)),
                    'precision': float(m.get('precision', 0)),
                    'recall': float(m.get('recall', 0)),
                    'f1_score': float(m.get('f1_score', 0)),
                    'auroc': float(m.get('auroc', 0)),
                    'auprc': float(m.get('auprc', 0)),
                    'report': r['report']
                }

                # [新增] 添加 minority-class 指标 (如果存在)
                if 'minority_f1_full' in m:
                    result_entry['minority_f1'] = float(m.get('minority_f1_full', 0))
                    result_entry['minority_recall'] = float(m.get('minority_recall_full', 0))
                    result_entry['minority_precision'] = float(m.get('minority_precision_full', 0))
                    result_entry['pred_minor_rate'] = float(m.get('pred_minor_rate', 0))
                    result_entry['true_minor_rate'] = float(m.get('true_minor_rate', 0))
                    result_entry['minority_tp'] = int(m.get('minority_tp', 0))
                    result_entry['minority_fn'] = int(m.get('minority_fn', 0))
                    result_entry['majority_fp'] = int(m.get('majority_fp', 0))
                    result_entry['majority_tn'] = int(m.get('majority_tn', 0))

                results_json.append(result_entry)
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {result_file}")


def _run_inference(config: Config, ModelClass, device):
    """推理模式"""
    print("\n" + "=" * 80)
    print("推理模式")
    print("=" * 80)

    # [新增] 确定是否多标签模式
    is_multilabel = False
    if hasattr(config, 'task') and config.task is not None:
        is_multilabel = (config.task.mode == "multi_label")

    # 检查模型路径
    if config.runtime.model_path is None:
        suffix = "_multilabel" if is_multilabel else ""
        exp_suffix = config.exp_suffix  # 获取实验后缀
        model_path = os.path.join(
            config.data.output_root, "..", "models",
            f"best_{config.model.name}_{config.dataset}_{config.features.adapt_mode}{suffix}{exp_suffix}.pth"
        )
    else:
        model_path = config.runtime.model_path

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return

    print(f"加载模型: {model_path}")

    # 创建单一数据集实例
    use_var_length = getattr(config.model, 'use_variable_length', False) and config.model.name == "HDSTGCN"
    use_static = False
    if hasattr(config.model, 'static_features') and config.model.static_features is not None:
        use_static = config.model.static_features.enabled

    full_dataset = CPETDatasetNew(
        config,
        test_ratio=config.data.test_ratio,
        feature_indices=config.features.channels,
        use_variable_length=use_var_length,
        max_length=config.data.max_length,
        use_static_features=use_static
    )

    # [新增] 同步多标签状态
    is_multilabel = getattr(config, 'is_multilabel', False)

    # 更新 num_static_features (动态计算)
    if use_static and hasattr(full_dataset, 'num_static_features'):
        if hasattr(config.model, 'static_features') and config.model.static_features:
            config.model.static_features.num_features = full_dataset.num_static_features

    # 获取测试子集
    test_dataset = full_dataset.get_split("test")

    config.update_with_dataset(test_dataset)

    # 创建数据加载器
    if use_var_length:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            shuffle=False, num_workers=0,
            collate_fn=collate_fn_variable_length
        )
    else:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            shuffle=False, num_workers=0
        )

    # 创建模型
    model = _create_model(config, ModelClass, device, use_var_length)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # 邻接矩阵
    if config.model.name != "HDSTGCN":
        adj = torch.from_numpy(create_adjacency_matrix(config.features.adapt_mode)).float().to(device)
    else:
        adj = None

    # 推理
    all_preds = []
    all_labels = []

    print("\n开始推理...")
    with torch.no_grad():
        for batch in test_loader:
            num_elements = len(batch)

            # 处理静态特征
            if use_var_length and use_static and num_elements == 4:
                data, lengths, static_data, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                static_data = static_data.to(device)
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data)
            elif use_var_length and num_elements == 3:
                data, lengths, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None)
            else:
                if num_elements == 3 and use_static:
                    data, static_data, labels = batch
                    static_data = static_data.to(device)
                else:
                    data, labels = batch
                    static_data = None
                data = data.to(device)
                labels = labels.to(device)

                if adj is not None:
                    outputs = model(data, adj, static_x=static_data)
                else:
                    outputs = model(data, static_x=static_data)

            if is_multilabel:
                # 多标签模式: sigmoid 输出
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())
            else:
                # 单标签模式: argmax
                pred = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(pred)
                all_labels.extend(labels.cpu().numpy())

    # [修改] 根据模式计算评估指标
    if is_multilabel:
        from train_with_swanlab import compute_multilabel_metrics, print_multilabel_results

        all_probs = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 获取标签名称
        label_names = config.part_actions if hasattr(config, 'part_actions') else None

        # 获取共现矩阵
        co_matrix = getattr(config, 'co_occurrence_matrix', None)

        # 打印详细的多标签结果矩阵
        print_multilabel_results(
            all_probs, all_labels,
            label_names=label_names,
            co_occurrence_matrix=co_matrix,
            threshold=0.5
        )

        # 计算并打印汇总指标
        metrics = compute_multilabel_metrics(all_probs, all_labels)

        print(f"\n多标签评估结果汇总:")
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  Micro-F1: {metrics['micro_f1']:.4f}")
        print(f"  mAP: {metrics['mAP']:.4f}")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
        print(f"  Jaccard: {metrics['jaccard']:.4f}")
    else:
        # 计算准确率
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"\n测试准确率: {accuracy:.4f}")

        # 打印分类报告
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n分类报告:")
        print(classification_report(
            all_labels, all_preds,
            target_names=config.part_actions,
            zero_division=0
        ))

        print("\n混淆矩阵:")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)


def _create_model(config: Config, ModelClass, device, use_var_length: bool):
    """创建模型"""
    # 获取静态特征配置
    use_static = False
    static_dim = 16
    static_ablation = "full"
    num_static_features = 5
    if hasattr(config.model, 'static_features') and config.model.static_features is not None:
        use_static = config.model.static_features.enabled
        static_dim = config.model.static_features.static_dim
        static_ablation = config.model.static_features.ablation
        num_static_features = config.model.static_features.num_features

    # 获取时序编码器配置
    temporal_encoder_type = "gru"
    T_mid = 24
    temporal_encoder_cfg = None  # 新增: 完整的时序编码器配置对象
    if hasattr(config.model, 'temporal_encoder') and config.model.temporal_encoder is not None:
        temporal_encoder_type = config.model.temporal_encoder.type
        T_mid = config.model.temporal_encoder.T_mid
        temporal_encoder_cfg = config.model.temporal_encoder  # 传递完整配置对象

    # 获取先验门控配置
    gamma_init = 1.0
    gamma_min = 0.1
    gamma_lr_scale = 0.3
    if hasattr(config.model, 'prior_gate') and config.model.prior_gate is not None:
        gamma_init = config.model.prior_gate.gamma_init
        gamma_min = config.model.prior_gate.gamma_min
        gamma_lr_scale = config.model.prior_gate.gamma_lr_scale

    # [新增] 获取通道注意力配置
    use_channel_attention = False
    channel_attention_init = 1.0
    if hasattr(config.model, 'channel_attention') and config.model.channel_attention is not None:
        use_channel_attention = config.model.channel_attention.enabled
        channel_attention_init = config.model.channel_attention.init_value

    # [新增] 获取 Flatten MLP 配置 (仅 flatten_only 模式生效)
    flatten_mlp_config = None
    if hasattr(config.model, 'flatten_mlp') and config.model.flatten_mlp is not None:
        flatten_mlp_config = config.model.flatten_mlp

    # [新增] 获取 Pooling_only 配置 (仅 pooling_only 模式生效)
    pooling_only_config = None
    if hasattr(config.model, 'pooling_only') and config.model.pooling_only is not None:
        pooling_only_config = config.model.pooling_only

    # [新增] 获取 Known-T6 Context 配置
    use_known_t6_context = False
    t6_n_classes = 0
    if hasattr(config, 'known_t6_context') and config.known_t6_context is not None:
        use_known_t6_context = config.known_t6_context.enabled
        # t6_n_classes 需要从数据集获取，或从 config 传递
        t6_n_classes = getattr(config, 't6_n_classes', 0)

    if config.model.name == "HDSTGCN":
        print(f"\n创建 HDSTGCN 模型:")
        print(f"  时序编码维度: {config.model.D_time}")
        print(f"  时序编码器: {temporal_encoder_type}" + (f" (T_mid={T_mid})" if temporal_encoder_type == "cnn" else ""))
        # 新增: 打印可插拔模块配置
        if temporal_encoder_type == "cnn" and temporal_encoder_cfg is not None:
            use_multiscale = getattr(temporal_encoder_cfg, 'use_multiscale', False)
            use_residual = getattr(temporal_encoder_cfg, 'use_residual', False)
            if use_multiscale or use_residual:
                print(f"  可插拔模块: multiscale={use_multiscale}, residual={use_residual}")
        print(f"  Dropout: {config.model.dropout}")
        print(f"  变长模式: {use_var_length}")
        print(f"  图模式: {config.model.graph_ablation}")
        if config.model.graph_ablation == "prior_masked":
            print(f"  先验门控: gamma_init={gamma_init}, gamma_min={gamma_min}, lr_scale={gamma_lr_scale}")
            # [新增] 检查核心斜率权重预设是否启用
            use_attn_weights = False
            if hasattr(config, 'nine_graph') and hasattr(config.nine_graph, 'attention_weights'):
                use_attn_weights = config.nine_graph.attention_weights.enabled
            print(f"  核心斜率权重预设: {use_attn_weights}")
        elif config.model.graph_ablation == "flatten_only":
            # [新增] 打印 Flatten MLP 配置
            if flatten_mlp_config is not None:
                use_two_layer = flatten_mlp_config.use_two_layer
                print(f"  Flatten MLP: use_two_layer={use_two_layer}, dropout={flatten_mlp_config.dropout}")
            else:
                print(f"  Flatten MLP: 单层 MLP (480→48)")
        elif config.model.graph_ablation == "pooling_only":
            # [新增] 打印 Pooling_only 配置
            if pooling_only_config is not None:
                pooling_type = pooling_only_config.pooling_type
                mlp_layers = pooling_only_config.mlp_layers
                print(f"  Pooling_only: pooling_type={pooling_type}, mlp_layers={mlp_layers}, dropout={pooling_only_config.dropout}")
            else:
                print(f"  Pooling_only: avg pooling, 单层 MLP (30→48)")
        if use_static:
            # [增强] 显示消融模式对分类器维度的影响
            fusion_dim_map = {
                "full": "64 (48动态 + 16静态)",
                "cpet_only": "48 (仅动态特征)",
                "static_only": "16 (仅静态特征)"
            }
            fusion_info = fusion_dim_map.get(static_ablation, "未知")
            print(f"  静态特征: 已启用 (dim={static_dim}, ablation={static_ablation})")
            print(f"  分类器输入维度: {fusion_info}")
        if use_channel_attention:
            print(f"  通道注意力: 已启用 (init_value={channel_attention_init})")
        # [新增] 打印 Known-T6 Context 配置
        if use_known_t6_context and t6_n_classes > 0:
            print(f"  Known-T6 Context: 已启用 (t6_n_classes={t6_n_classes})")

        # 获取邻接矩阵作为医学先验
        # [修复] 传递 optional_keys 以支持 vco2 和 o2pulse 特征
        optional_keys = []
        if hasattr(config.features, 'o2pulse_enabled') and config.features.o2pulse_enabled:
            optional_keys.append('o2pulse')
        if hasattr(config.features, 'vco2_enabled') and config.features.vco2_enabled:
            optional_keys.append('vco2')
        semantic_adj = create_adjacency_matrix(config.features.adapt_mode, optional_keys if optional_keys else None)

        # [新增] 获取核心斜率注意力权重预设 (仅 prior_masked 模式生效)
        attention_weights_matrix = None
        if config.model.graph_ablation == "prior_masked":
            use_attn_weights = False
            if hasattr(config, 'nine_graph') and hasattr(config.nine_graph, 'attention_weights'):
                use_attn_weights = config.nine_graph.attention_weights.enabled
            if use_attn_weights and config.features.adapt_mode == "nine_graph":
                # 调用 get_nine_graph_config 获取完整的九图配置（包含权重矩阵）
                nine_config = get_nine_graph_config(optional_keys if optional_keys else None)
                attention_weights_matrix = nine_config['attention_weights']

        model = ModelClass(
            input_dim=config.data.max_length if use_var_length else config.data.L_win,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.n_class,
            channel_groups=config.features.channel_groups,
            num_channel=config.features.num_channels,
            D_time=config.model.D_time,
            dropout=config.model.dropout,
            semantic_adj=semantic_adj,  # <--- 传入医学先验邻接矩阵
            # 静态特征参数
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features,
            static_ablation=static_ablation,
            graph_ablation=config.model.graph_ablation,
            # 时序编码器参数
            temporal_encoder_type=temporal_encoder_type,
            T_mid=T_mid,
            temporal_encoder_cfg=temporal_encoder_cfg,  # 新增: 传递完整配置对象
            # 先验门控参数
            gamma_init=gamma_init,
            gamma_min=gamma_min,
            # [新增] 通道注意力参数
            use_channel_attention=use_channel_attention,
            channel_attention_init=channel_attention_init,
            # [新增] 核心斜率注意力权重预设
            attention_weights=attention_weights_matrix,
            # [新增] Flatten MLP 配置 (仅 flatten_only 模式生效)
            flatten_mlp_config=flatten_mlp_config,
            # [新增] Pooling_only 配置 (仅 pooling_only 模式生效)
            pooling_only_config=pooling_only_config,
            # [新增] Known-T6 Context 参数
            use_known_t6_context=use_known_t6_context,
            t6_n_classes=t6_n_classes
        ).to(device)

    elif config.model.name == "STFinalNet":
        print(f"\n创建 STFinalNet 模型:")
        print(f"  消融模式: {config.model.ablation}")
        print(f"  变量嵌入: {config.model.use_var_embedding} (dim={config.model.var_embed_dim})")
        print(f"  动态图: {config.model.use_dynamic_graph}")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim})")

        # [修复] 传递 optional_keys 以支持 vco2 和 o2pulse 特征
        optional_keys = []
        if hasattr(config.features, 'o2pulse_enabled') and config.features.o2pulse_enabled:
            optional_keys.append('o2pulse')
        if hasattr(config.features, 'vco2_enabled') and config.features.vco2_enabled:
            optional_keys.append('vco2')
        semantic_adj = create_adjacency_matrix(config.features.adapt_mode, optional_keys if optional_keys else None)

        model = ModelClass(
            input_dim=config.data.L_win,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.n_class,
            channel_groups=config.features.channel_groups,
            num_channel=config.features.num_channels,
            ablation=config.model.ablation,
            use_var_embedding=config.model.use_var_embedding,
            use_dynamic_graph=config.model.use_dynamic_graph,
            var_embed_dim=config.model.var_embed_dim,
            semantic_adj=semantic_adj,
            use_variable_length=False,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    elif config.model.name == "lstm":
        print(f"\n创建 LSTMNet 模型:")
        print(f"  Hidden dim: {config.model.hidden_dim}")
        print(f"  Num layers: {config.model.num_layers or 2}")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim})")

        model = ModelClass(
            input_dim=config.data.L_win,
            output_dim=config.n_class,
            num_channel=config.features.num_channels,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers or 2,
            use_variable_length=False,  # LSTM uses pack_padded_sequence
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    elif config.model.name == "resnet":
        print(f"\n创建 ResNet1D 模型:")
        print(f"  Hidden dim: {config.model.hidden_dim}")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim})")

        model = ModelClass(
            input_dim=config.data.L_win,
            output_dim=config.n_class,
            num_channel=config.features.num_channels,
            hidden_dim=config.model.hidden_dim,
            use_variable_length=False,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    elif config.model.name == "mednet":
        print(f"\n创建 MedNet 模型:")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim})")

        model = ModelClass(
            input_dim=config.data.L_win,
            hidden_dim=config.model.hidden_dim,
            channel_groups=config.features.channel_groups,
            output_dim=config.n_class,
            num_channel=config.features.num_channels,
            use_variable_length=False,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    elif config.model.name == "CNNGAF":
        # 获取 CNN-GAF 特定配置
        image_size = getattr(config.model, 'image_size', 64)
        cnn_channels = getattr(config.model, 'cnn_channels', [16, 32])
        attention_dim = getattr(config.model, 'attention_dim', 16)

        print(f"\n创建 CNNGAF 模型:")
        print(f"  GADF 图像大小: {image_size}x{image_size}")
        print(f"  CNN 通道: {cnn_channels}")
        print(f"  注意力维度: {attention_dim}")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim})")

        model = ModelClass(
            input_dim=config.data.L_win,
            output_dim=config.n_class,
            num_channel=config.features.num_channels,
            image_size=image_size,
            cnn_channels=cnn_channels,
            attention_dim=attention_dim,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    elif config.model.name == "KESTNet":
        # 获取 KESTNet 特定配置
        num_gcn_layers = getattr(config.model, 'num_gcn_layers', 2)
        num_transformer_layers = getattr(config.model, 'num_transformer_layers', 2)
        num_cst_layers = getattr(config.model, 'num_cst_layers', 2)
        num_heads = getattr(config.model, 'num_heads', 4)

        print(f"\n创建 KESTNet 模型:")
        print(f"  Hidden dim: {config.model.hidden_dim}")
        print(f"  GCN layers: {num_gcn_layers}")
        print(f"  Transformer layers: {num_transformer_layers}")
        print(f"  CST iterations: {num_cst_layers}")
        print(f"  Attention heads: {num_heads}")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim})")

        # 获取邻接矩阵作为医学先验
        optional_keys = []
        if hasattr(config.features, 'o2pulse_enabled') and config.features.o2pulse_enabled:
            optional_keys.append('o2pulse')
        if hasattr(config.features, 'vco2_enabled') and config.features.vco2_enabled:
            optional_keys.append('vco2')
        semantic_adj = create_adjacency_matrix(config.features.adapt_mode, optional_keys if optional_keys else None)

        # 获取系统划分索引 (nine_graph 模式的 4 个子系统)
        system_channel_indices = config.features.channel_groups if config.features.adapt_mode == "nine_graph" else None

        model = ModelClass(
            input_dim=config.data.max_length,
            num_channel=config.features.num_channels,
            output_dim=config.n_class,
            hidden_dim=config.model.hidden_dim,
            num_gcn_layers=num_gcn_layers,
            num_transformer_layers=num_transformer_layers,
            num_cst_layers=num_cst_layers,
            num_heads=num_heads,
            dropout=config.model.dropout,
            semantic_adj=semantic_adj,
            system_channel_indices=system_channel_indices,
            use_variable_length=True,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    else:
        raise ValueError(f"未知模型: {config.model.name}")

    return model


def _create_model_with_n_classes(config, ModelClass, device, n_classes, use_variable_length):
    """
    创建指定类别数的模型 (用于测试集评估时动态设置类别数)

    Args:
        config: Config 配置对象
        ModelClass: 模型类
        device: 设备
        n_classes: 类别数
        use_variable_length: 是否使用变长序列

    Returns:
        model: 创建的模型
    """
    from feature_mapping import create_adjacency_matrix

    # 获取静态特征配置
    use_static = False
    static_dim = 16
    num_static_features = 5
    if hasattr(config.model, 'static_features') and config.model.static_features is not None:
        use_static = config.model.static_features.enabled
        static_dim = config.model.static_features.static_dim
        num_static_features = config.model.static_features.num_features

    # 获取时序编码器配置
    temporal_encoder_type = "gru"
    T_mid = 24
    temporal_encoder_cfg = None
    if hasattr(config.model, 'temporal_encoder') and config.model.temporal_encoder is not None:
        temporal_encoder_type = config.model.temporal_encoder.type
        T_mid = config.model.temporal_encoder.T_mid
        temporal_encoder_cfg = config.model.temporal_encoder

    # 获取先验门控配置
    gamma_init = 1.0
    gamma_min = 0.1
    if hasattr(config.model, 'prior_gate') and config.model.prior_gate is not None:
        gamma_init = config.model.prior_gate.gamma_init
        gamma_min = config.model.prior_gate.gamma_min

    # 获取邻接矩阵
    optional_keys = []
    if hasattr(config.features, 'o2pulse_enabled') and config.features.o2pulse_enabled:
        optional_keys.append('o2pulse')
    if hasattr(config.features, 'vco2_enabled') and config.features.vco2_enabled:
        optional_keys.append('vco2')
    semantic_adj = create_adjacency_matrix(config.features.adapt_mode, optional_keys if optional_keys else None)

    # [新增] 获取核心斜率注意力权重预设 (仅 prior_masked 模式生效)
    attention_weights_matrix = None
    if config.model.name == "HDSTGCN" and config.model.graph_ablation == "prior_masked":
        use_attn_weights = False
        if hasattr(config, 'nine_graph') and hasattr(config.nine_graph, 'attention_weights'):
            use_attn_weights = config.nine_graph.attention_weights.enabled
        if use_attn_weights and config.features.adapt_mode == "nine_graph":
            from feature_mapping import get_nine_graph_config
            nine_config = get_nine_graph_config(optional_keys if optional_keys else None)
            attention_weights_matrix = nine_config['attention_weights']

    # [新增] 获取 Known-T6 Context 配置
    use_known_t6_context = False
    t6_n_classes = 0
    if hasattr(config, 'known_t6_context') and config.known_t6_context is not None:
        use_known_t6_context = config.known_t6_context.enabled
        t6_n_classes = getattr(config, 't6_n_classes', 0)

    model = None

    if config.model.name == "HDSTGCN":
        print(f"\n创建 {ModelClass.__name__} 模型 (指定类别数):")
        print(f"  类别数: {n_classes}")
        print(f"  时序编码维度: {config.model.D_time}")
        print(f"  时序编码器: {temporal_encoder_type}" + (f" (T_mid={T_mid})" if temporal_encoder_type == "cnn" else ""))
        print(f"  变长模式: {use_variable_length}")

        model = ModelClass(
            input_dim=config.data.max_length if use_variable_length else config.data.L_win,
            hidden_dim=config.model.hidden_dim,
            output_dim=n_classes,  # [关键] 使用传入的类别数
            channel_groups=config.features.channel_groups,
            num_channel=config.features.num_channels,
            D_time=config.model.D_time,
            dropout=config.model.dropout,
            semantic_adj=semantic_adj,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features,
            use_variable_length=use_variable_length,
            graph_ablation=config.model.graph_ablation,
            temporal_encoder_type=temporal_encoder_type,
            T_mid=T_mid,
            temporal_encoder_cfg=temporal_encoder_cfg,
            gamma_init=gamma_init,
            gamma_min=gamma_min,
            attention_weights=attention_weights_matrix,  # [新增] 核心斜率权重预设
            # [新增] Known-T6 Context 参数
            use_known_t6_context=use_known_t6_context,
            t6_n_classes=t6_n_classes
        ).to(device)

    elif config.model.name == "STFinalNet":
        print(f"\n创建 STFinalNet 模型 (指定类别数):")
        print(f"  类别数: {n_classes}")
        print(f"  消融模式: {config.model.ablation}")

        model = ModelClass(
            input_dim=config.data.L_win,
            hidden_dim=config.model.hidden_dim,
            output_dim=n_classes,
            channel_groups=config.features.channel_groups,
            num_channel=config.features.num_channels,
            ablation=config.model.ablation,
            use_var_embedding=config.model.use_var_embedding,
            use_dynamic_graph=config.model.use_dynamic_graph,
            var_embed_dim=config.model.var_embed_dim,
            semantic_adj=semantic_adj,
            use_variable_length=False,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    elif config.model.name == "lstm":
        print(f"\n创建 LSTMNet 模型 (指定类别数):")
        print(f"  类别数: {n_classes}")
        print(f"  Hidden dim: {config.model.hidden_dim}")

        model = ModelClass(
            input_dim=config.data.L_win,
            output_dim=n_classes,
            num_channel=config.features.num_channels,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers or 2,
            use_variable_length=False,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    elif config.model.name == "resnet":
        print(f"\n创建 ResNet1D 模型 (指定类别数):")
        print(f"  类别数: {n_classes}")
        print(f"  Hidden dim: {config.model.hidden_dim}")

        model = ModelClass(
            input_dim=config.data.L_win,
            output_dim=n_classes,
            num_channel=config.features.num_channels,
            hidden_dim=config.model.hidden_dim,
            use_variable_length=False,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    elif config.model.name == "mednet":
        print(f"\n创建 MedNet 模型 (指定类别数):")
        print(f"  类别数: {n_classes}")

        model = ModelClass(
            input_dim=config.data.L_win,
            hidden_dim=config.model.hidden_dim,
            channel_groups=config.features.channel_groups,
            output_dim=n_classes,
            num_channel=config.features.num_channels,
            use_variable_length=False,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    elif config.model.name == "CNNGAF":
        # 获取 CNNGAF 特定配置
        image_size = getattr(config.model, 'image_size', 64)
        cnn_channels = getattr(config.model, 'cnn_channels', [16, 32])
        attention_dim = getattr(config.model, 'attention_dim', 16)

        print(f"\n创建 CNNGAF 模型 (指定类别数):")
        print(f"  类别数: {n_classes}")
        print(f"  GADF 图像大小: {image_size}x{image_size}")
        print(f"  CNN 通道: {cnn_channels}")
        print(f"  注意力维度: {attention_dim}")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim})")

        model = ModelClass(
            input_dim=config.data.L_win,  # CNNGAF 使用固定长度
            output_dim=n_classes,
            num_channel=config.features.num_channels,
            image_size=image_size,
            cnn_channels=cnn_channels,
            attention_dim=attention_dim,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features,
            dropout=config.model.dropout
        ).to(device)

    elif config.model.name == "KESTNet":
        # 获取 KESTNet 特定配置
        num_gcn_layers = getattr(config.model, 'num_gcn_layers', 2)
        num_transformer_layers = getattr(config.model, 'num_transformer_layers', 2)
        num_cst_layers = getattr(config.model, 'num_cst_layers', 2)
        num_heads = getattr(config.model, 'num_heads', 4)

        print(f"\n创建 KESTNet 模型 (指定类别数):")
        print(f"  类别数: {n_classes}")
        print(f"  Hidden dim: {config.model.hidden_dim}")
        print(f"  GCN layers: {num_gcn_layers}")
        print(f"  Transformer layers: {num_transformer_layers}")
        print(f"  CST iterations: {num_cst_layers}")
        print(f"  Attention heads: {num_heads}")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim})")

        # 获取系统划分索引 (nine_graph 模式的 4 个子系统)
        system_channel_indices = config.features.channel_groups if config.features.adapt_mode == "nine_graph" else None

        model = ModelClass(
            input_dim=config.data.max_length,
            num_channel=config.features.num_channels,
            output_dim=n_classes,
            hidden_dim=config.model.hidden_dim,
            num_gcn_layers=num_gcn_layers,
            num_transformer_layers=num_transformer_layers,
            num_cst_layers=num_cst_layers,
            num_heads=num_heads,
            dropout=config.model.dropout,
            semantic_adj=semantic_adj,
            system_channel_indices=system_channel_indices,
            use_variable_length=True,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=num_static_features
        ).to(device)

    else:
        raise ValueError(f"未知模型: {config.model.name}")

    return model


def _run_holdout_test_only(config: Config, ModelClass, device, is_multilabel=False):
    """
    跳过 K-Fold 训练，直接进行测试集评估

    前提条件:
        1. 已完成 K-Fold 训练
        2. 存在模型文件: models/best_xxx_foldN.pth
        3. 存在统计量文件: results/kfold_train_stats_xxx.json
        4. 存在划分文件: results/holdout_split_info.json

    [新增] 支持评估所有 Fold 并输出均值±标准差
    """
    import json
    from datetime import datetime
    from collections import Counter
    from sklearn.metrics import confusion_matrix
    from dataset_new import preload_all_data_for_kfold, CPETDatasetNewKFold
    from train_with_swanlab import (
        _create_model_for_kfold,
        compute_classification_metrics,
        _find_holdout_split_file,
        _resolve_holdout_indices_from_split,
    )

    output_dir = config.data.output_root
    model_name = config.model.name
    adapt_mode = config.features.adapt_mode
    suffix = "_multilabel" if is_multilabel else ""
    exp_suffix = config.exp_suffix  # 获取实验后缀
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    print("\n" + "="*80)
    print("【断点恢复 - 直接评估测试集】")
    print("="*80)

    # =========================================================================
    # 1. 查找并加载统计量文件
    # =========================================================================
    stats_file = os.path.join(output_dir, f"kfold_train_stats_{model_name}_{adapt_mode}{suffix}{exp_suffix}.json")
    if not os.path.exists(stats_file):
        print(f"\n错误: 未找到统计量文件: {stats_file}")
        print("请先运行完整的 K-Fold 训练")
        return None

    print(f"\n加载训练集统计量: {stats_file}")
    with open(stats_file, 'r', encoding='utf-8') as f:
        kfold_stats = json.load(f)

    best_fold = kfold_stats.get('best_fold', 1)
    n_folds = kfold_stats.get('n_folds', 5)
    all_fold_stats = kfold_stats.get('all_fold_stats', {})  # {fold_num: {'stats': ..., 'static_stats': ...}}

    print(f"  - 最佳 Fold: {best_fold} (Macro-F1: {kfold_stats.get('best_macro_f1', 'N/A'):.4f})")
    print(f"  - Fold 总数: {n_folds}")
    print(f"  - 所有 Fold 统计量: {list(all_fold_stats.keys()) if all_fold_stats else '未保存 (回退最佳Fold)'}")

    # =========================================================================
    # 2. 查找 Holdout 划分文件
    # =========================================================================
    split_file = _find_holdout_split_file(config, output_dir)
    if not os.path.exists(split_file):
        print(f"\n错误: 未找到划分文件: {split_file}")
        print("请先运行完整的 K-Fold 训练")
        return None

    print(f"\n加载划分信息: {split_file}")
    with open(split_file, 'r', encoding='utf-8') as f:
        split_info = json.load(f)

    test_indices = split_info.get('test_indices')
    dev_indices = split_info.get('dev_indices')
    needs_split_reconstruction = dev_indices is None or test_indices is None
    if needs_split_reconstruction:
        test_indices = []
        dev_indices = []
    print(f"  - Dev_Set: {len(dev_indices)} 样本")
    print(f"  - Test_Set: {len(test_indices)} 样本")

    # =========================================================================
    # 3. 预加载数据
    # =========================================================================
    print("\n预加载数据...")
    use_variable_length = getattr(config.model, 'use_variable_length', False)
    use_static = False
    if hasattr(config.model, 'static_features') and config.model.static_features is not None:
        use_static = config.model.static_features.enabled

    data_cache = preload_all_data_for_kfold(
        config,
        use_variable_length=use_variable_length,
        max_length=config.data.max_length,
        use_static_features=use_static,
        feature_indices=config.features.channels
    )

    if needs_split_reconstruction:
        dev_indices, test_indices = _resolve_holdout_indices_from_split(
            split_info,
            data_cache.get('filenames', []),
            split_file
        )
        print(f"  - Dev_Set: {len(dev_indices)} samples")
        print(f"  - Test_Set: {len(test_indices)} samples")

    random_seed = getattr(config.training, 'random_seed', 3407)
    dataset_name = getattr(config, 'dataset', 'CPET_New')

    # =========================================================================
    # 4. 决定评估策略
    # =========================================================================
    evaluate_all_folds = bool(all_fold_stats)  # 有所有 Fold 统计量才评估全部

    if evaluate_all_folds:
        print("\n【评估所有 Fold 在测试集上】")
        all_fold_test_results = []

        for fold_num in range(1, n_folds + 1):
            print(f"\n--- Fold {fold_num} ---")

            # 获取该 Fold 的统计量
            fold_stat = all_fold_stats.get(str(fold_num), all_fold_stats.get(fold_num, {}))
            train_stats = fold_stat.get('stats', {})
            train_static_stats = fold_stat.get('static_stats', {})

            if not train_stats:
                print(f"  [警告] Fold {fold_num} 缺少统计量，跳过")
                continue

            # 模型路径
            model_path = os.path.join(
                models_dir,
                f"best_{model_name}_{dataset_name}_{adapt_mode}_fold{fold_num}{suffix}{exp_suffix}.pth"
            )

            if not os.path.exists(model_path):
                print(f"  [警告] 未找到模型文件: {model_path}")
                continue

            # 创建测试集数据集
            stats_for_dataset = {}
            for key in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']:
                if key in train_stats:
                    stats_for_dataset[key] = np.array(train_stats[key])

            static_stats_for_dataset = None
            if train_static_stats:
                static_stats_for_dataset = {
                    'mean': np.array(train_static_stats['mean']),
                    'std': np.array(train_static_stats['std'])
                }

            test_dataset = CPETDatasetNewKFold(
                config, fold_idx=0, n_folds=1,
                phase="train", random_seed=random_seed,
                feature_indices=config.features.channels,
                use_variable_length=use_variable_length,
                max_length=config.data.max_length,
                use_static_features=use_static,
                dev_indices=dev_indices,
                test_indices=test_indices,
                all_data_cache=data_cache,
                use_holdout_test=True,
                train_stats=stats_for_dataset,
                train_static_stats=static_stats_for_dataset
            )

            n_classes = len(test_dataset.label_mapping)

            # 创建 DataLoader
            if use_variable_length:
                from dataset_new import collate_fn_variable_length
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=1,
                    shuffle=False, num_workers=0,
                    collate_fn=collate_fn_variable_length
                )
            else:
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=1,
                    shuffle=False, num_workers=0
                )

            # 加载模型
            model = _create_model_with_n_classes(config, ModelClass, device, n_classes, use_variable_length)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            model.to(device)
            model.eval()

            # 推理
            all_preds, all_labels, all_probs = [], [], []

            # [新增] 检测二分类模式
            is_binary = (n_classes == 2) and not getattr(config, 'is_multilabel', False)
            minority_idx = None
            if is_binary:
                # 从 test_dataset 检测少数类索引
                from collections import Counter
                label_counts = Counter(test_dataset.labellist)
                sorted_labels = sorted(test_dataset.label_mapping.items(), key=lambda x: x[1])
                samples_per_cls = [label_counts.get(name, 0) for name, _ in sorted_labels]
                if samples_per_cls[0] < samples_per_cls[1]:
                    minority_idx = 0
                else:
                    minority_idx = 1
                print(f"  [二分类模式] 少数类索引={minority_idx} ({sorted_labels[minority_idx][0]})")

            with torch.no_grad():
                for batch in test_loader:
                    num_elements = len(batch)

                    if use_variable_length and use_static and num_elements == 4:
                        data, lengths, static_data, labels = batch
                        data, lengths, static_data, labels = data.to(device), lengths.to(device), static_data.to(device), labels.to(device)
                        outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data)
                    elif use_variable_length and num_elements == 3:
                        data, lengths, labels = batch
                        data, lengths, labels = data.to(device), lengths.to(device), labels.to(device)
                        outputs = model(data, lengths=lengths, prior_adj=None)
                    else:
                        if num_elements == 3 and use_static:
                            data, static_data, labels = batch
                            static_data = static_data.to(device)
                        else:
                            data, labels = batch
                            static_data = None
                        data, labels = data.to(device), labels.to(device)
                        outputs = model(data, static_x=static_data)

                    # [修复] 二分类和多分类的推理逻辑不同
                    if is_binary:
                        # 二分类: sigmoid + 阈值
                        probs_positive = torch.sigmoid(outputs)  # [B, 1] - 少数类概率

                        # 概率伪装为 [B, 2]，用于 AUROC/AUPRC 计算
                        if minority_idx == 1:
                            probs = torch.cat([1 - probs_positive, probs_positive], dim=1)  # [B, 2]
                        else:
                            probs = torch.cat([probs_positive, 1 - probs_positive], dim=1)  # [B, 2]

                        # 预测标签: sigmoid > 0.5 → 少数类, 否则 → 多数类
                        preds = torch.where(probs_positive.squeeze(1) > 0.5, minority_idx, 1 - minority_idx).long()
                    else:
                        # 多分类: softmax + argmax
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(outputs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            all_probs = np.array(all_probs)

            # 计算指标
            # [修复] 二分类模式必须传递 minority_idx
            metrics = compute_classification_metrics(
                all_preds, all_labels, all_probs,
                average='macro',
                minority_idx=minority_idx if is_binary else None
            )

            all_fold_test_results.append({
                'fold': fold_num,
                'metrics': metrics
            })

            print(f"  Macro-F1: {metrics['macro_f1']:.4f}, Acc: {metrics['accuracy']:.4f}, AUROC: {metrics['auroc']:.4f}")

        # =========================================================================
        # 5. 输出汇总结果 (均值±标准差)
        # =========================================================================
        if all_fold_test_results:
            print("\n" + "="*80)
            print(f"【测试集评估结果汇总】({len(all_fold_test_results)} Folds)")
            print("="*80)

            # 收集各指标
            accs = [r['metrics']['accuracy'] for r in all_fold_test_results]
            precisions = [r['metrics']['precision'] for r in all_fold_test_results]
            recalls = [r['metrics']['recall'] for r in all_fold_test_results]
            f1s = [r['metrics']['f1_score'] for r in all_fold_test_results]
            macro_f1s = [r['metrics']['macro_f1'] for r in all_fold_test_results]
            aurocs = [r['metrics']['auroc'] for r in all_fold_test_results]
            auprcs = [r['metrics']['auprc'] for r in all_fold_test_results]

             # [新增] 收集 minority-class 指标 (如果为二分类任务)
            target_col_name = getattr(config.features, 'target_col_name', '')
            is_minority_task = target_col_name in BINARY_MINORITY_METRIC_TASKS and is_binary

            minority_f1s = []
            minority_recalls = []
            minority_precisions = []
            pred_minor_rates = []
            true_minor_rates = []
            minority_tps = []
            minority_fns = []
            majority_fps = []
            majority_tns = []

            if is_minority_task:
                for r in all_fold_test_results:
                    m = r['metrics']
                    if 'minority_f1_full' in m:
                        minority_f1s.append(m.get('minority_f1_full', 0))
                        minority_recalls.append(m.get('minority_recall_full', 0))
                        minority_precisions.append(m.get('minority_precision_full', 0))
                        pred_minor_rates.append(m.get('pred_minor_rate', 0))
                        true_minor_rates.append(m.get('true_minor_rate', 0))
                        minority_tps.append(m.get('minority_tp', 0))
                        minority_fns.append(m.get('minority_fn', 0))
                        majority_fps.append(m.get('majority_fp', 0))
                        majority_tns.append(m.get('majority_tn', 0))

            print(f"{'指标':<15} {'Mean':<15} {'Std':<15}")
            print("-" * 50)
            print(f"{'Accuracy':<15} {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            print(f"{'Precision':<15} {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
            print(f"{'Recall':<15} {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
            print(f"{'F1-score':<15} {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"{'Macro-F1':<15} {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
            print(f"{'AUROC':<15} {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
            print(f"{'AUPRC':<15} {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
            print("-" * 50)

            # [新增] 打印 minority-class 指标汇总
            if is_minority_task and minority_f1s:
                print("\n【Minority-Class Metrics汇总】")
                print("-" * 50)
                minority_idx = MINORITY_IDX_MAP.get(target_col_name, None)
                print(f"  target_col_name: {target_col_name}, minority_idx: {minority_idx}")
                print(f"  minority_f1:        {np.mean(minority_f1s):.4f} ± {np.std(minority_f1s):.4f}")
                print(f"  minority_recall:    {np.mean(minority_recalls):.4f} ± {np.std(minority_recalls):.4f}")
                print(f"  minority_precision: {np.mean(minority_precisions):.4f} ± {np.std(minority_precisions):.4f}")
                print(f"  true_minor_rate:    {np.mean(true_minor_rates):.4f} ± {np.std(true_minor_rates):.4f}")
                print(f"  pred_minor_rate:    {np.mean(pred_minor_rates):.4f} ± {np.std(pred_minor_rates):.4f}")
                print(f"  minority_tp:        {np.mean(minority_tps):.1f} ± {np.std(minority_tps):.1f}")
                print(f"  minority_fn:        {np.mean(minority_fns):.1f} ± {np.std(minority_fns):.1f}")
                print(f"  majority_fp:        {np.mean(majority_fps):.1f} ± {np.std(majority_fps):.1f}")
                print(f"  majority_tn:        {np.mean(majority_tns):.1f} ± {np.std(majority_tns):.1f}")
                print("-" * 50)

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(output_dir, f"holdout_test_all_folds_{model_name}_{adapt_mode}_{timestamp}.json")

            result_data = {
                'n_folds': len(all_fold_test_results),
                'n_test_samples': len(test_indices),
                'metrics_mean': {
                    'accuracy': float(np.mean(accs)),
                    'precision': float(np.mean(precisions)),
                    'recall': float(np.mean(recalls)),
                    'f1_score': float(np.mean(f1s)),
                    'macro_f1': float(np.mean(macro_f1s)),
                    'auroc': float(np.mean(aurocs)),
                    'auprc': float(np.mean(auprcs))
                },
                'metrics_std': {
                    'accuracy': float(np.std(accs)),
                    'precision': float(np.std(precisions)),
                    'recall': float(np.std(recalls)),
                    'f1_score': float(np.std(f1s)),
                    'macro_f1': float(np.std(macro_f1s)),
                    'auroc': float(np.std(aurocs)),
                    'auprc': float(np.std(auprcs))
                },
                'fold_results': [
                    {'fold': r['fold'], 'metrics': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in r['metrics'].items()}}

                    for r in all_fold_test_results
                ]
            }

             # [新增] 添加 minority-class 指标汇总
            if is_minority_task and minority_f1s:
                result_data['minority_metrics_mean'] = {
                    'minority_f1': float(np.mean(minority_f1s)),
                    'minority_recall': float(np.mean(minority_recalls)),
                    'minority_precision': float(np.mean(minority_precisions)),
                    'pred_minor_rate': float(np.mean(pred_minor_rates)),
                    'true_minor_rate': float(np.mean(true_minor_rates)),
                }
                result_data['minority_metrics_std'] = {
                    'minority_f1': float(np.std(minority_f1s)),
                    'minority_recall': float(np.std(minority_recalls)),
                    'minority_precision': float(np.std(minority_precisions)),
                    'pred_minor_rate': float(np.std(pred_minor_rates)),
                    'true_minor_rate': float(np.std(true_minor_rates)),
                }
                result_data['minority_counts_mean'] = {
                    'minority_tp': float(np.mean(minority_tps)),
                    'minority_fn': float(np.mean(minority_fns)),
                    'majority_fp': float(np.mean(majority_fps)),
                    'majority_tn': float(np.mean(majority_tns)),
                }
                result_data['minority_counts_std'] = {
                    'minority_tp': float(np.std(minority_tps)),
                    'minority_fn': float(np.std(minority_fns)),
                    'majority_fp': float(np.std(majority_fps)),
                    'majority_tn': float(np.std(majority_tns)),
                }

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            print(f"\n[已保存] 所有 Fold 测试集结果 → {result_file}")

            return result_data

        else:
            print("\n[警告] 未收集到任何 Fold 的测试集评估结果")
            return None

    else:
        # =========================================================================
        # 回退模式: 只评估最佳 Fold (兼容旧统计量文件)
        # =========================================================================
        print("\n【回退模式: 只评估最佳 Fold】")
        print("提示: 重新运行完整 K-Fold 训练可保存所有 Fold 统计量")

        train_stats = kfold_stats.get('train_stats', {})
        train_static_stats = kfold_stats.get('train_static_stats', {})

        # 创建测试集数据集
        stats_for_dataset = {}
        for key in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']:
            if key in train_stats:
                stats_for_dataset[key] = np.array(train_stats[key])

        static_stats_for_dataset = None
        if train_static_stats:
            static_stats_for_dataset = {
                'mean': np.array(train_static_stats['mean']),
                'std': np.array(train_static_stats['std'])
            }

        test_dataset = CPETDatasetNewKFold(
            config, fold_idx=0, n_folds=1,
            phase="train", random_seed=random_seed,
            feature_indices=config.features.channels,
            use_variable_length=use_variable_length,
            max_length=config.data.max_length,
            use_static_features=use_static,
            dev_indices=dev_indices,
            test_indices=test_indices,
            all_data_cache=data_cache,
            use_holdout_test=True,
            train_stats=stats_for_dataset,
            train_static_stats=static_stats_for_dataset
        )

        n_classes = len(test_dataset.label_mapping)
        print(f"  - 实际类别数: {n_classes}")

        # 创建 DataLoader
        if use_variable_length:
            from dataset_new import collate_fn_variable_length
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1,
                shuffle=False, num_workers=0,
                collate_fn=collate_fn_variable_length
            )
        else:
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1,
                shuffle=False, num_workers=0
            )

        # 加载模型
        model_path = os.path.join(
            models_dir,
            f"best_{model_name}_{dataset_name}_{adapt_mode}_fold{best_fold}{suffix}{exp_suffix}.pth"
        )

        if not os.path.exists(model_path):
            print(f"\n错误: 未找到模型文件: {model_path}")
            return None

        print(f"\n加载模型: {model_path}")

        model = _create_model_with_n_classes(config, ModelClass, device, n_classes, use_variable_length)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()

        # 推理
        print("\n正在进行推理...")
        all_preds, all_labels, all_probs = [], [], []

        # [新增] 检测二分类模式
        is_binary = (n_classes == 2) and not getattr(config, 'is_multilabel', False)
        minority_idx = None
        if is_binary:
            # 从 test_dataset 检测少数类索引
            from collections import Counter
            label_counts = Counter(test_dataset.labellist)
            sorted_labels = sorted(test_dataset.label_mapping.items(), key=lambda x: x[1])
            samples_per_cls = [label_counts.get(name, 0) for name, _ in sorted_labels]
            if samples_per_cls[0] < samples_per_cls[1]:
                minority_idx = 0
            else:
                minority_idx = 1
            print(f"  [二分类模式] 少数类索引={minority_idx} ({sorted_labels[minority_idx][0]})")

        with torch.no_grad():
            for batch in test_loader:
                num_elements = len(batch)

                if use_variable_length and use_static and num_elements == 4:
                    data, lengths, static_data, labels = batch
                    data, lengths, static_data, labels = data.to(device), lengths.to(device), static_data.to(device), labels.to(device)
                    outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data)
                elif use_variable_length and num_elements == 3:
                    data, lengths, labels = batch
                    data, lengths, labels = data.to(device), lengths.to(device), labels.to(device)
                    outputs = model(data, lengths=lengths, prior_adj=None)
                else:
                    if num_elements == 3 and use_static:
                        data, static_data, labels = batch
                        static_data = static_data.to(device)
                    else:
                        data, labels = batch
                        static_data = None
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data, static_x=static_data)

                # [修复] 二分类和多分类的推理逻辑不同
                if is_binary:
                    # 二分类: sigmoid + 阈值
                    probs_positive = torch.sigmoid(outputs)  # [B, 1] - 少数类概率

                    # 概率伪装为 [B, 2]，用于 AUROC/AUPRC 计算
                    if minority_idx == 1:
                        probs = torch.cat([1 - probs_positive, probs_positive], dim=1)  # [B, 2]
                    else:
                        probs = torch.cat([probs_positive, 1 - probs_positive], dim=1)  # [B, 2]

                    # 预测标签: sigmoid > 0.5 → 少数类, 否则 → 多数类
                    preds = torch.where(probs_positive.squeeze(1) > 0.5, minority_idx, 1 - minority_idx).long()
                else:
                    # 多分类: softmax + argmax
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算指标
        # [修复] 二分类模式必须传递 minority_idx
        metrics = compute_classification_metrics(
            all_preds, all_labels, all_probs,
            average='macro',
            minority_idx=minority_idx if is_binary else None
        )

        print("\n" + "="*80)
        print("【测试集评估结果】")
        print("="*80)
        print(f"使用模型: Fold {best_fold}")
        print("-" * 50)
        print(f"{'指标':<15} {'值':<15}")
        print("-" * 50)
        print(f"{'Accuracy':<15} {metrics['accuracy']:.4f}")
        print(f"{'Precision':<15} {metrics['precision']:.4f}")
        print(f"{'Recall':<15} {metrics['recall']:.4f}")
        print(f"{'F1-score':<15} {metrics['f1_score']:.4f}")
        print(f"{'AUROC':<15} {metrics['auroc']:.4f}")
        print(f"{'AUPRC':<15} {metrics['auprc']:.4f}")
        print("-" * 50)

        # 混淆矩阵
        print("\n【混淆矩阵】")
        cm = confusion_matrix(all_labels, all_preds)

        label_mapping_reversed = {v: k for k, v in test_dataset.label_mapping.items()}
        # 按标签名排序，获取排序后的索引列表
        sorted_indices = sorted(label_mapping_reversed.keys(), key=lambda x: label_mapping_reversed[x])

        print(f"{'':>18}", end="")
        for idx in sorted_indices:
            label_name = label_mapping_reversed[idx]
            print(f"{label_name[:8]:>10}", end="")
        print()

        for i, idx in enumerate(sorted_indices):
            label_name = label_mapping_reversed[idx]
            print(f"{label_name[:16]:>18}", end="")
            for j in range(len(sorted_indices)):
                print(f"{cm[i, j]:>10}", end="")
            print()

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"holdout_test_results_{model_name}_{adapt_mode}_{timestamp}.json")

        result_data = {
            'best_fold': best_fold,
            'n_test_samples': len(test_indices),
            'model_path': model_path,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'confusion_matrix': [[int(x) for x in row] for row in cm.tolist()],
            'label_mapping': label_mapping_reversed
        }

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\n[已保存] 测试集结果 → {result_file}")

        return result_data


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
