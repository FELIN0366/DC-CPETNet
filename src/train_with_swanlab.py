"""
集成SwanLab的训练脚本
替换TensorBoard，提供更好的实验跟踪
** 已适配统一配置系统 (Config) **
"""

from collections import Counter, defaultdict
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import WeightedRandomSampler
from model import write_debug_log
import warnings
warnings.filterwarnings("ignore")

# SwanLab 已强制禁用，避免 import / init / log 带来的额外开销。
DISABLE_SWANLAB = True

# 全局标志: 置为 True 后所有 _safe_swanlab_call 都会直接返回
_swancab_network_error = True
SWANLAB_AVAILABLE = False


class MockSwanLab:
    def init(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


swanlab = MockSwanLab()


def _safe_swanlab_call(func_name, *args, **kwargs):
    """
    安全调用 SwanLab API，网络错误时自动降级

    Args:
        func_name: SwanLab 方法名 (init, log, finish, login)
        *args, **kwargs: 传递给方法的参数

    Returns:
        True: 调用成功或已禁用
        False: 调用失败
    """
    global _swancab_network_error

    # 如果之前已经发生网络错误，直接跳过
    if _swancab_network_error:
        return False

    if not SWANLAB_AVAILABLE:
        return False

    try:
        func = getattr(swanlab, func_name)
        func(*args, **kwargs)
        return True
    except Exception as e:
        error_msg = str(e)
        # 检测网络相关错误
        if any(keyword in error_msg for keyword in [
            'Max retries exceeded', 'too many 500 error', 'ConnectionError',
            'TimeoutError', 'NetworkError', 'HTTPSConnectionPool',
            'api.swanlab.cn', 'ResponseError'
        ]):
            print(f"\n[警告] SwanLab 网络错误: {error_msg[:100]}...")
            print("[信息] 已自动禁用 SwanLab 日志，实验将继续运行")
            _swancab_network_error = True
        else:
            # 非网络错误，打印但不禁用
            print(f"\n[警告] SwanLab {func_name} 调用失败: {error_msg[:100]}...")
        return False


def _get_config_value(config, attr_path, default=None):
    """
    从 Config 对象或旧版 args 对象获取属性值

    Args:
        config: Config 对象或 args 对象
        attr_path: 属性路径，如 'training.gradient_clip' 或 'gradient_clip'
        default: 默认值

    Returns:
        属性值
    """
    if config is None:
        return default

    # 尝试按路径访问 (Config 对象)
    parts = attr_path.split('.')
    obj = config
    try:
        for part in parts:
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        pass

    # 尝试直接访问 (旧版 args 对象)
    try:
        return getattr(config, attr_path.replace('.', '_'))
    except AttributeError:
        return default


def _workspace_results_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def _find_holdout_split_file(config=None, output_dir=None):
    candidates = [
        os.path.join(_workspace_results_dir(), 'holdout_split_info.json'),
    ]
    if output_dir:
        candidates.append(os.path.join(output_dir, 'holdout_split_info.json'))

    seen = set()
    for path in candidates:
        norm_path = os.path.abspath(path)
        if norm_path in seen:
            continue
        seen.add(norm_path)
        if os.path.exists(norm_path):
            return norm_path

    return candidates[0]


def _resolve_holdout_indices_from_split(split_info, filenames, split_file):
    if 'dev_indices' in split_info and 'test_indices' in split_info:
        return list(split_info['dev_indices']), list(split_info['test_indices'])

    if 'dev_filenames' not in split_info or 'test_filenames' not in split_info:
        raise KeyError(
            f"{split_file} must contain dev_indices/test_indices or "
            "dev_filenames/test_filenames"
        )

    filename_to_index = {}
    duplicates = set()
    for idx, filename in enumerate(filenames):
        if filename in filename_to_index:
            duplicates.add(filename)
        filename_to_index[filename] = idx

    if duplicates:
        duplicate_preview = sorted(duplicates)[:5]
        raise ValueError(
            f"Duplicate filenames in preloaded data prevent deterministic split "
            f"reconstruction: {duplicate_preview}"
        )

    missing = [
        filename
        for filename in split_info['dev_filenames'] + split_info['test_filenames']
        if filename not in filename_to_index
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} filenames from {split_file} are absent after preload; "
            f"examples: {missing[:5]}"
        )

    dev_indices = [filename_to_index[filename] for filename in split_info['dev_filenames']]
    test_indices = [filename_to_index[filename] for filename in split_info['test_filenames']]
    return dev_indices, test_indices


def create_weighted_sampler(dataset, label_mapping: list):
    """
    为长尾数据集创建 WeightedRandomSampler

    核心机制:
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

    # 2. 创建标签到索引的映射
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


def build_optimizer_with_gamma_lr(model, base_lr, weight_decay, gamma_lr_scale=0.3):
    """
    构建优化器，为 gamma 参数使用独立的学习率

    Args:
        model: 模型
        base_lr: 基础学习率
        weight_decay: 权重衰减
        gamma_lr_scale: gamma 学习率缩放因子

    Returns:
        optimizer: AdamW 优化器
    """
    gamma_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'gamma_raw' in name or 'gamma' in name:
            gamma_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': other_params, 'lr': base_lr},
        {'params': gamma_params, 'lr': base_lr * gamma_lr_scale}
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)

    if len(gamma_params) > 0:
        print(f"[优化器] gamma 参数学习率: {base_lr * gamma_lr_scale:.6f} (缩放因子: {gamma_lr_scale})")

    return optimizer


def train_epoch(model, train_loader, optimizer, criterion, device, adj=None, epoch=0, config=None,
                is_binary=False, minority_idx=None):
    """
    训练一个epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数 (可以是 CombinedLoss 或普通 Loss)
        device: 设备
        adj: 邻接矩阵
        epoch: 当前epoch
        config: Config 配置对象 (或旧版 args)
        is_binary: 是否为二分类模式
        minority_idx: 二分类模式下的少数类索引

    Returns:
        mean_loss: 平均损失
        accuracy: 准确率
        loss_dict_avg: 平均损失分解 (如果使用 CombinedLoss)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 检测是否使用变长序列模式
    use_variable_length = _get_config_value(config, 'model.use_variable_length', False)
    use_static = _get_config_value(config, 'model.static_features.enabled', False)
    gradient_clip = _get_config_value(config, 'training.gradient_clip', 1.0)

    # 获取模型名称，用于区分 HDSTGCN 和 STFinalNet
    model_name = _get_config_value(config, 'model.name', 'HDSTGCN')
    is_hdstgcn = model_name == 'HDSTGCN'

    # 检测是否使用 CombinedLoss
    use_combined_loss = isinstance(criterion, CombinedLoss)

    # 累计损失分解
    loss_dict_sum = {'loss_ce': 0.0, 'loss_supcon': 0.0, 'loss_total': 0.0}

    for batch_idx, batch in enumerate(train_loader):
        # 智能解包: 根据返回元素数量判断数据格式
        num_elements = len(batch)

        # [新增] Known-T6 Context 模式处理
        t6_context = None  # 默认为 None

        if use_variable_length and use_static and num_elements == 5:
            # 变长模式 + 静态特征 + t6: (data, lengths, static_data, t6_context, labels)
            data, lengths, static_data, t6_context, labels = batch
            data = data.to(device)
            lengths = lengths.to(device)
            static_data = static_data.to(device)
            t6_context = t6_context.to(device)  # [新增]
            labels = labels.to(device)
        elif use_variable_length and use_static and num_elements == 4:
            # 变长模式 + 静态特征: (data, lengths, static_data, labels)
            data, lengths, static_data, labels = batch
            data = data.to(device)
            lengths = lengths.to(device)
            static_data = static_data.to(device)
            labels = labels.to(device)
        elif use_variable_length and num_elements == 4:
            # 变长模式 + t6: (data, lengths, t6_context, labels)
            data, lengths, t6_context, labels = batch
            data = data.to(device)
            lengths = lengths.to(device)
            t6_context = t6_context.to(device)  # [新增]
            labels = labels.to(device)
            static_data = None
        elif use_variable_length and num_elements == 3:
            # 变长模式: (data, lengths, labels)
            data, lengths, labels = batch
            data = data.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            static_data = None
        elif num_elements == 4 and use_static:
            # 固定长度 + 静态 + t6: (data, static_data, t6_context, labels)
            data, static_data, t6_context, labels = batch
            data = data.to(device)
            static_data = static_data.to(device)
            t6_context = t6_context.to(device)  # [新增]
            labels = labels.to(device)
        elif num_elements == 3 and use_static:
            # 固定长度 + 静态: (data, static_data, labels)
            data, static_data, labels = batch
            data = data.to(device)
            static_data = static_data.to(device)
            labels = labels.to(device)
        elif num_elements == 3:
            # 固定长度 + t6: (data, t6_context, labels)
            data, t6_context, labels = batch
            data = data.to(device)
            t6_context = t6_context.to(device)  # [新增]
            labels = labels.to(device)
            static_data = None
        else:
            # 固定长度模式: (data, labels)
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)
            static_data = None

        optimizer.zero_grad()

        # 前向传播
        if use_variable_length:
            # HDSTGCN 变长模式: 使用 lengths 参数
            adj_temp = adj.to(device) if adj is not None else None
            if use_combined_loss and hasattr(model, 'forward_with_features'):
                # 使用 forward_with_features 获取特征
                outputs, features = model.forward_with_features(data, lengths=lengths, prior_adj=adj_temp, static_x=static_data, t6_context=t6_context)
            else:
                outputs = model(data, lengths=lengths, prior_adj=adj_temp, static_x=static_data, t6_context=t6_context)
                features = None
        elif adj is not None and not is_hdstgcn:
            # STFinalNet 固定长度模式: 使用邻接矩阵作为位置参数
            # 注意: HDSTGCN 不应该进入此分支，因为邻接矩阵在模型初始化时已内部包含
            adj_temp = adj.to(device)
            if use_combined_loss and hasattr(model, 'forward_with_features'):
                outputs, features = model.forward_with_features(data, adj_temp, static_x=static_data, t6_context=t6_context)
            else:
                outputs = model(data, adj_temp, static_x=static_data, t6_context=t6_context)
                features = None
        else:
            # HDSTGCN 固定长度模式 或 无邻接矩阵模式
            # HDSTGCN 不需要传递 adj（模型内部已包含）和 lengths（固定长度不需要 mask）
            if use_combined_loss and hasattr(model, 'forward_with_features'):
                outputs, features = model.forward_with_features(data, static_x=static_data, t6_context=t6_context)
            else:
                outputs = model(data, static_x=static_data, t6_context=t6_context)
                features = None

        # 计算损失
        if is_binary:
            # 二分类模式: BCEWithLogitsLoss 需要 [B, 1] 输出和 [B, 1] 标签
            # [关键修复] 标签转换: BCE期望正类(少数类)标签=1.0, 负类标签=0.0
            # - 如果 minority_idx == 0: 原标签值 0(少数类) → 需转换为 1.0, 原标签值 1(多数类) → 需转换为 0.0
            # - 如果 minority_idx == 1: 原标签值已经是正确的 (少数类=1, 多数类=0)
            if minority_idx == 0:
                # 标签反转: 0 → 1.0 (少数类变为正类), 1 → 0.0 (多数类变为负类)
                binary_labels = 1.0 - labels.float()
            else:
                # 标签已经是正确的格式
                binary_labels = labels.float()
            loss = criterion(outputs, binary_labels.unsqueeze(1))
        elif use_combined_loss:
            loss, loss_dict = criterion(outputs, labels, features)
            for k, v in loss_dict.items():
                loss_dict_sum[k] += v
        else:
            loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        # 梯度裁剪 (从配置读取)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        # 统计
        running_loss += loss.item()

        # 预测
        if is_binary:
            # 二分类模式: sigmoid + 0.5 阈值
            probs = torch.sigmoid(outputs)  # [B, 1]
            predict_y = (probs > 0.5).long().squeeze(1)
            # 映射预测结果: sigmoid > 0.5 → 少数类, 否则 → 多数类
            predict_y = torch.where(predict_y == 1, minority_idx, 1 - minority_idx).long()
        else:
            predict_y = torch.argmax(outputs, dim=1)

        # label 现在直接就是索引，不需要再 argmax
        label_y = labels

        correct += (predict_y == label_y).sum().item()
        total += labels.size(0)

    mean_loss = running_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0

    # 计算平均损失分解
    loss_dict_avg = {k: v / len(train_loader) for k, v in loss_dict_sum.items()} if use_combined_loss else None

    return mean_loss, accuracy, loss_dict_avg


def validate_epoch(model, val_loader, criterion, device, adj=None, config=None,
                   is_binary=False, minority_idx=None):
    """
    验证一个epoch

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        adj: 邻接矩阵
        config: Config 配置对象 (或旧版 args)
        is_binary: 是否为二分类模式
        minority_idx: 二分类模式下的少数类索引

    Returns:
        mean_loss: 平均损失
        accuracy: 准确率
        all_preds: 所有预测
        all_labels: 所有标签
        all_probs: 所有预测概率 (用于 AUROC/AUPRC 计算)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  # 新增: 收集概率值

    # 检测是否使用变长序列模式
    use_variable_length = _get_config_value(config, 'model.use_variable_length', False)
    use_static = _get_config_value(config, 'model.static_features.enabled', False)

    # 获取模型名称，用于区分 HDSTGCN 和 STFinalNet
    model_name = _get_config_value(config, 'model.name', 'HDSTGCN')
    is_hdstgcn = model_name == 'HDSTGCN'

    with torch.no_grad():
        for batch in val_loader:
            # 智能解包: 根据返回元素数量判断数据格式
            num_elements = len(batch)

            # [新增] Known-T6 Context 模式处理
            t6_context = None  # 默认为 None

            if use_variable_length and use_static and num_elements == 5:
                # 变长模式 + 静态特征 + t6: (data, lengths, static_data, t6_context, labels)
                data, lengths, static_data, t6_context, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                static_data = static_data.to(device)
                t6_context = t6_context.to(device)  # [新增]
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data, t6_context=t6_context)
            elif use_variable_length and use_static and num_elements == 4:
                # 变长模式 + 静态特征: (data, lengths, static_data, labels)
                data, lengths, static_data, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                static_data = static_data.to(device)
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data)
            elif use_variable_length and num_elements == 4:
                # 变长模式 + t6: (data, lengths, t6_context, labels)
                data, lengths, t6_context, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                t6_context = t6_context.to(device)  # [新增]
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, t6_context=t6_context)
            elif use_variable_length and num_elements == 3:
                # 变长模式: (data, lengths, labels)
                data, lengths, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None)
            elif num_elements == 4 and use_static:
                # 固定长度 + 静态 + t6: (data, static_data, t6_context, labels)
                data, static_data, t6_context, labels = batch
                data = data.to(device)
                static_data = static_data.to(device)
                t6_context = t6_context.to(device)  # [新增]
                labels = labels.to(device)
                if adj is not None and not is_hdstgcn:
                    adj_temp = adj.to(device)
                    outputs = model(data, adj_temp, static_x=static_data, t6_context=t6_context)
                else:
                    outputs = model(data, static_x=static_data, t6_context=t6_context)
            elif num_elements == 3 and use_static:
                # 固定长度 + 静态: (data, static_data, labels)
                data, static_data, labels = batch
                data = data.to(device)
                static_data = static_data.to(device)
                labels = labels.to(device)
                if adj is not None and not is_hdstgcn:
                    adj_temp = adj.to(device)
                    outputs = model(data, adj_temp, static_x=static_data)
                else:
                    outputs = model(data, static_x=static_data)
            elif num_elements == 3:
                # 固定长度 + t6: (data, t6_context, labels)
                data, t6_context, labels = batch
                data = data.to(device)
                t6_context = t6_context.to(device)  # [新增]
                labels = labels.to(device)
                outputs = model(data, t6_context=t6_context)
            else:
                # 固定长度模式: (data, labels)
                data, labels = batch
                data = data.to(device)
                labels = labels.to(device)
                static_data = None

                if adj is not None and not is_hdstgcn:
                    # STFinalNet 固定长度模式: 使用邻接矩阵作为位置参数
                    adj_temp = adj.to(device)
                    outputs = model(data, adj_temp, static_x=static_data)
                else:
                    # HDSTGCN 固定长度模式: 不需要传递 adj 和 lengths
                    outputs = model(data, static_x=static_data)

            # 计算损失
            if is_binary:
                # [关键修复] 标签转换: BCE期望正类(少数类)标签=1.0
                if minority_idx == 0:
                    binary_labels = 1.0 - labels.float()
                else:
                    binary_labels = labels.float()
                loss_result = criterion(outputs, binary_labels.unsqueeze(1))
            else:
                loss_result = criterion(outputs, labels)

            # 处理 CombinedLoss 返回的元组
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
            else:
                loss = loss_result

            running_loss += loss.item()

            # 预测与概率
            if is_binary:
                # 二分类模式: sigmoid + 阈值
                probs_positive = torch.sigmoid(outputs)  # [B, 1] - 少数类概率

                # [关键] 概率伪装为 [B, 2]，确保少数类概率在 minority_idx 位置
                # 用于 AUROC/AUPRC 计算 (compute_classification_metrics 需要 [N, C] 格式)
                if minority_idx == 1:
                    # 少数类是索引1，多数类是索引0
                    probs = torch.cat([1 - probs_positive, probs_positive], dim=1)  # [B, 2]
                else:
                    # 少数类是索引0，多数类是索引1 (罕见情况)
                    probs = torch.cat([probs_positive, 1 - probs_positive], dim=1)  # [B, 2]

                # 预测标签: sigmoid > 0.5 → 少数类, 否则 → 多数类
                predict_y = torch.where(probs_positive.squeeze(1) > 0.5, minority_idx, 1 - minority_idx).long()
            else:
                predict_y = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)  # 计算概率值

            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # 收集概率值

    mean_loss = running_loss / len(val_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return mean_loss, accuracy, all_preds, all_labels, np.array(all_probs)  # 新增返回概率


def compute_classification_metrics(all_preds, all_labels, all_probs, average='macro', minority_idx=None):
    """
    计算多分类任务的完整评估指标

    Args:
        all_preds: 预测标签 (numpy array, shape: [N])
        all_labels: 真实标签 (numpy array, shape: [N])
        all_probs: 预测概率 (numpy array, shape: [N, C])
        average: Precision/Recall/F1 的平均策略
                 'macro': 宏平均 (各类别权重相等，关注少数类)
                 'weighted': 加权平均 (按类别样本数加权)
                 注: 如需切换为 weighted，修改此参数即可
        minority_idx: 二分类模式下的少数类索引 (用于指定 pos_label)

    Returns:
        metrics: 包含所有指标的字典
    """
    n_classes = all_probs.shape[1] if len(all_probs.shape) > 1 else len(np.unique(all_labels))

    # 基础指标
    accuracy = np.mean(all_preds == all_labels)

    # Precision, Recall, F1-score
    # 二分类模式: 使用 macro 平均，对两个类别公平对待
    # 注: 之前的 average='binary' 只计算少数类，不是真正的 Macro-F1
    if n_classes == 2:
        # Macro-F1: 两个类别 F1 的平均值 (真正的宏平均)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # [可选] 单独计算少数类指标，用于诊断分析
        pos_label = minority_idx if minority_idx is not None else 1
        minority_precision = precision_score(all_labels, all_preds, average='binary', pos_label=pos_label, zero_division=0)
        minority_recall = recall_score(all_labels, all_preds, average='binary', pos_label=pos_label, zero_division=0)
        minority_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=pos_label, zero_division=0)
    else:
        # 多分类: 使用指定的平均策略
        precision = precision_score(all_labels, all_preds, average=average, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=average, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=average, zero_division=0)

    # AUROC: 区分二分类和多分类
    # sklearn roc_auc_score 对二分类和多分类有不同的 API 要求
    try:
        if n_classes == 2:
            # 二分类: 使用正类概率 (少数类)
            # [关键修复] sklearn 的 roc_auc_score 当标签是 {0,1} 时不支持 pos_label 参数
            # 它默认认为标签值较大的 (1) 是正类。
            # 当 minority_idx=0 时，标签值 0 才是正类，需要翻转标签。
            pos_label = minority_idx if minority_idx is not None else 1

            if pos_label == 1:
                # 正类标签值是 1，sklearn 默认行为正确
                auroc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                # 正类标签值是 0，需要翻转标签让 sklearn 认为翻转后的 1 是正类
                # 翻转后: 原标签 0 (正类) → 1, 原标签 1 (负类) → 0
                flipped_labels = 1 - all_labels
                auroc = roc_auc_score(flipped_labels, all_probs[:, 0])
        else:
            # 多分类: One-vs-Rest 策略，宏平均
            auroc = roc_auc_score(
                all_labels,
                all_probs,
                multi_class='ovr',
                average='macro',
                labels=list(range(n_classes))
            )
    except Exception as e:
        print(f"[警告] AUROC 计算失败: {e}")
        auroc = 0.0

    # AUPRC: 多分类 One-vs-Rest 宏平均
    # sklearn 的 average_precision_score 不直接支持多分类 average 参数
    # 需手动实现: 对每个类别计算 AP，然后取宏平均
    try:
        # 将标签二值化: shape [N, C]
        y_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))

        # 处理二分类情况 (label_binarize 返回 [N, 1] 而非 [N, 2])
        if y_true_bin.shape[1] == 1 and n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        # 计算每个类别的 AP
        ap_per_class = []
        for c in range(n_classes):
            if np.sum(y_true_bin[:, c]) > 0:  # 类别 c 有正样本
                ap_c = average_precision_score(y_true_bin[:, c], all_probs[:, c])
                ap_per_class.append(ap_c)

        # 宏平均 AUPRC
        auprc = np.mean(ap_per_class) if ap_per_class else 0.0
    except Exception as e:
        print(f"[警告] AUPRC 计算失败: {e}")
        auprc = 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'macro_f1': f1,  # 真正的 Macro-F1 (两个类别 F1 平均)
        'auroc': auroc,
        'auprc': auprc
    }

    # 二分类模式: 添加少数类单独指标 (用于诊断分析)
    if n_classes == 2:
        metrics['minority_precision'] = minority_precision
        metrics['minority_recall'] = minority_recall
        metrics['minority_f1'] = minority_f1
        # [新增] 计算完整的 minority-class 指标 (用于 Figure 2D)
        minority_metrics = compute_binary_minority_metrics(all_labels, all_preds, pos_label)
        metrics['minority_f1_full'] = minority_metrics['minority_f1']
        metrics['minority_recall_full'] = minority_metrics['minority_recall']
        metrics['minority_precision_full'] = minority_metrics['minority_precision']
        metrics['pred_minor_rate'] = minority_metrics['pred_minor_rate']
        metrics['true_minor_rate'] = minority_metrics['true_minor_rate']
        metrics['minority_tp'] = minority_metrics['minority_tp']
        metrics['minority_fn'] = minority_metrics['minority_fn']
        metrics['majority_fp'] = minority_metrics['majority_fp']
        metrics['majority_tn'] = minority_metrics['majority_tn']

    return metrics
# =========================================================================
# [新增] 二分类少数类指标计算 - 用于 Figure 2D 绘图
# =========================================================================

# 需要计算 minority-class 指标的二分类任务列表
BINARY_MINORITY_METRIC_TASKS = [
    "标准心电运动负荷试验",
    "运动中换气肺功能",
    "心率储备",
]

# 各任务的少数类索引映射 (与 MTL 保持一致)
MINORITY_IDX_MAP = {
    "标准心电运动负荷试验": 0,  # 阳性类（异常）为少数类
    "运动中换气肺功能": 0,      # 阳性类（异常）为少数类
    "心率储备": 1,              # 阳性类（心率储备用尽）为少数类
}


def compute_binary_minority_metrics(y_true, y_pred, minority_idx):
    """
    计算二分类任务的少数类完整指标

    用于绘制 Figure 2D: binary-task minority-class performance

    Args:
        y_true: 真实标签 (numpy array, shape: [N], 值为 0 或 1)
        y_pred: 预测标签 (numpy array, shape: [N], 值为 0 或 1)
        minority_idx: 少数类索引

    Returns:
        dict: 包含以下指标的字典:
            - minority_f1: 少数类 F1-score
            - minority_recall: 少数类 Recall
            - minority_precision: 少数类 Precision
            - pred_minor_rate: 预测为少数类的比例
            - true_minor_rate: 实际少数类的比例
            - minority_tp: 少数类 True Positive 数
            - minority_fn: 少数类 False Negative 数
            - majority_fp: 多数类 False Positive 数
            - majority_tn: 多数类 True Negative 数
    """
    majority_idx = 1 - minority_idx

    # 计算混淆矩阵元素 (以 minority class 为阳性类)
    minority_tp = int(((y_true == minority_idx) & (y_pred == minority_idx)).sum())
    minority_fn = int(((y_true == minority_idx) & (y_pred == majority_idx)).sum())
    majority_fp = int(((y_true == majority_idx) & (y_pred == minority_idx)).sum())
    majority_tn = int(((y_true == majority_idx) & (y_pred == majority_idx)).sum())

    # 计算少数类 Precision, Recall, F1
    minority_precision = minority_tp / (minority_tp + majority_fp) if (minority_tp + majority_fp) > 0 else 0.0
    minority_recall = minority_tp / (minority_tp + minority_fn) if (minority_tp + minority_fn) > 0 else 0.0

    if minority_precision + minority_recall > 0:
        minority_f1 = 2 * minority_precision * minority_recall / (minority_precision + minority_recall)
    else:
        minority_f1 = 0.0

    # 计算比例
    pred_minor_rate = float((y_pred == minority_idx).mean())
    true_minor_rate = float((y_true == minority_idx).mean())

    return {
        "minority_f1": minority_f1,
        "minority_recall": minority_recall,
        "minority_precision": minority_precision,
        "pred_minor_rate": pred_minor_rate,
        "true_minor_rate": true_minor_rate,
        "minority_tp": minority_tp,
        "minority_fn": minority_fn,
        "majority_fp": majority_fp,
        "majority_tn": majority_tn,
    }


def get_minority_idx_for_task(target_col_name):
    """
    根据任务名称获取少数类索引

    Args:
        target_col_name: 任务名称 (config.features.target_col_name)

    Returns:
        minority_idx: 少数类索引，如果任务不在列表中则返回 None
    """
    return MINORITY_IDX_MAP.get(target_col_name, None)

def extract_error_patient_details(error_samples, label_file, output_dir):
    """
    提取错误分类样本的详细信息并保存为Excel表格

    Args:
        error_samples: 错误样本列表, 每个元素包含 {'filename', 'true_label', 'pred_label', 'true_idx', 'pred_idx'}
        label_file: final_summary_report.xlsx 路径
        output_dir: 输出目录

    Returns:
        error_df: 错误样本详细信息DataFrame
        stats_df: 统计分析DataFrame
    """
    # 读取标签文件 (第二行是列名)
    df = pd.read_excel(label_file, header=1)

    # 关键列索引 (根据之前查看的结果)
    # [76] 提取的单病种
    # [79] 编号
    # [82] 匹配的第一大类
    col_names = df.columns.tolist()

    # 尝试找到正确的列 (兼容不同编码)
    col_single_disease = None
    col_patient_id = None
    col_main_category = None

    for i, c in enumerate(col_names):
        c_str = str(c)
        if '提取的单病种' in c_str or '单病种' in c_str:
            col_single_disease = c
        if '编号' in c_str:
            col_patient_id = c
        if '第一大类' in c_str or '匹配的第一大类' in c_str:
            col_main_category = c

    # 如果没找到，使用索引
    if col_single_disease is None:
        col_single_disease = col_names[76]
    if col_patient_id is None:
        col_patient_id = col_names[79]
    if col_main_category is None:
        col_main_category = col_names[82]

    print(f"\n[信息] 使用列: 单病种='{col_single_disease}', 编号='{col_patient_id}', 第一大类='{col_main_category}'")

    # 提取错误样本信息；导出时只保留匿名样本号，不写出原始个人编号。
    error_details = []

    for anon_idx, sample in enumerate(error_samples):
        filename = sample['filename']
        # 从文件名提取内部匹配编号；该编号不写入导出文件。
        patient_id = filename.split('_')[0] if '_' in filename else filename[:-5]

        # 在DataFrame中查找对应行
        try:
            patient_id_int = int(patient_id)
            row = df[df[col_patient_id] == patient_id_int]
        except:
            row = df[df[col_patient_id].astype(str) == str(patient_id)]

        if len(row) > 0:
            row = row.iloc[0]
            error_details.append({
                '匿名样本编号': f"sample_{anon_idx:06d}",
                '真实标签': sample['true_label'],
                '预测标签': sample['pred_label'],
                '提取的单病种': row[col_single_disease] if col_single_disease in row.index else '',
                '匹配的第一大类': row[col_main_category] if col_main_category in row.index else '',
                '文件名': filename
            })
        else:
            error_details.append({
                '匿名样本编号': f"sample_{anon_idx:06d}",
                '真实标签': sample['true_label'],
                '预测标签': sample['pred_label'],
                '提取的单病种': '(未找到)',
                '匹配的第一大类': '(未找到)',
                '文件名': filename
            })

    # 创建DataFrame
    error_df = pd.DataFrame(error_details)

    # ========== 统计分析 ==========
    stats_results = []

    # 1. 按真实标签统计
    # print("\n" + "="*60)
    # print("【错误分类统计分析】")
    # print("="*60)

    for true_label in error_df['真实标签'].unique():
        subset = error_df[error_df['真实标签'] == true_label]
        n_errors = len(subset)

        # 统计被误判为哪些类别
        pred_counts = subset['预测标签'].value_counts().to_dict()

        # 统计单病种分布
        single_disease_counts = subset['提取的单病种'].value_counts().to_dict()

        stats_results.append({
            '统计类型': '按真实标签',
            '类别': true_label,
            '错误总数': n_errors,
            '误判分布': str(pred_counts),
            '单病种分布': str(single_disease_counts)
        })

        # print(f"\n【{true_label}】错误数: {n_errors}")
        # print(f"  误判分布: {pred_counts}")
        # print(f"  单病种分布: {single_disease_counts}")

    # 2. 按预测标签统计
    # print("\n" + "-"*60)
    for pred_label in error_df['预测标签'].unique():
        subset = error_df[error_df['预测标签'] == pred_label]
        n_errors = len(subset)

        # 统计来自哪些真实标签
        true_counts = subset['真实标签'].value_counts().to_dict()

        # print(f"【被误判为 {pred_label}】错误数: {n_errors}")
        # print(f"  来源分布: {true_counts}")

    # 3. 泵功能衰竭 ↔ 缺血性心脏病 专项分析
    # print("\n" + "="*60)
    # print("【重点关注】泵功能衰竭 ↔ 缺血性心脏病 混淆分析")
    # print("="*60)

    target_pairs = [
        ("泵功能衰竭", "缺血性心脏病"),
        ("缺血性心脏病", "泵功能衰竭")
    ]

    focus_stats = []
    for true_label, pred_label in target_pairs:
        subset = error_df[(error_df['真实标签'] == true_label) &
                          (error_df['预测标签'] == pred_label)]

        if len(subset) > 0:
            single_disease_dist = subset['提取的单病种'].value_counts().to_dict()

            # print(f"\n{true_label} → 误判为 {pred_label} ({len(subset)} 例):")
            # print(f"  单病种分布: {single_disease_dist}")

            # 添加到统计结果
            stats_results.append({
                '统计类型': '重点关注混淆',
                '类别': f'{true_label}→{pred_label}',
                '错误总数': len(subset),
                '误判分布': '-',
                '单病种分布': str(single_disease_dist)
            })

            focus_stats.append({
                '混淆类型': f'{true_label}→{pred_label}',
                '数量': len(subset),
                '单病种分布': single_disease_dist
            })

    stats_df = pd.DataFrame(stats_results)

    # ========== 保存到Excel ==========
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '错误分类样本详情.xlsx')

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        error_df.to_excel(writer, sheet_name='错误样本详情', index=False)
        stats_df.to_excel(writer, sheet_name='统计分析', index=False)

        # 如果有关注的混淆，添加专项sheet
        if focus_stats:
            focus_df = pd.DataFrame([
                {
                    '混淆类型': s['混淆类型'],
                    '数量': s['数量'],
                    '单病种分布': str(s['单病种分布'])
                }
                for s in focus_stats
            ])
            focus_df.to_excel(writer, sheet_name='泵缺血混淆分析', index=False)

    # print(f"\n[已保存] 错误样本详情 → {output_path}")

    return error_df, stats_df


def extract_error_patient_details_kfold(all_error_samples, label_file, output_dir):
    """
    K折交叉验证 - 提取并保存累积的错误样本详情

    Args:
        all_error_samples: 所有折累积的错误样本列表，每个元素包含
                          {'filename', 'true_label', 'pred_label', 'true_idx', 'pred_idx', 'fold'}
        label_file: final_summary_report.xlsx 路径
        output_dir: 输出目录

    Returns:
        error_df: 错误样本详细信息DataFrame
        stats_df: 统计分析DataFrame
    """
    # 读取标签文件 (第二行是列名)
    df = pd.read_excel(label_file, header=1)

    # 关键列索引
    col_names = df.columns.tolist()

    # 尝试找到正确的列 (兼容不同编码)
    col_single_disease = None
    col_patient_id = None
    col_main_category = None

    for i, c in enumerate(col_names):
        c_str = str(c)
        if '提取的单病种' in c_str or '单病种' in c_str:
            col_single_disease = c
        if '编号' in c_str:
            col_patient_id = c
        if '第一大类' in c_str or '匹配的第一大类' in c_str:
            col_main_category = c

    # 如果没找到，使用索引
    if col_single_disease is None and len(col_names) > 76:
        col_single_disease = col_names[76]
    if col_patient_id is None and len(col_names) > 79:
        col_patient_id = col_names[79]
    if col_main_category is None and len(col_names) > 82:
        col_main_category = col_names[82]

    # print(f"[信息] 使用列: 单病种='{col_single_disease}', 编号='{col_patient_id}', 第一大类='{col_main_category}'")

    # 提取错误样本信息；导出时只保留匿名样本号，不写出原始个人编号。
    error_details = []

    for anon_idx, sample in enumerate(all_error_samples):
        filename = sample['filename']
        patient_id = filename.split('_')[0] if '_' in filename else filename[:-5]

        # 在DataFrame中查找对应行
        try:
            patient_id_int = int(patient_id)
            row = df[df[col_patient_id] == patient_id_int]
        except:
            row = df[df[col_patient_id].astype(str) == str(patient_id)]

        if len(row) > 0:
            row = row.iloc[0]
            error_details.append({
                'Fold': sample.get('fold', '-'),
                '匿名样本编号': f"sample_{anon_idx:06d}",
                '真实标签': sample['true_label'],
                '预测标签': sample['pred_label'],
                '提取的单病种': row[col_single_disease] if col_single_disease in row.index else '',
                '匹配的第一大类': row[col_main_category] if col_main_category in row.index else '',
                '文件名': filename
            })
        else:
            error_details.append({
                'Fold': sample.get('fold', '-'),
                '匿名样本编号': f"sample_{anon_idx:06d}",
                '真实标签': sample['true_label'],
                '预测标签': sample['pred_label'],
                '提取的单病种': '(未找到)',
                '匹配的第一大类': '(未找到)',
                '文件名': filename
            })

    # 创建DataFrame
    error_df = pd.DataFrame(error_details)

    # ========== 统计分析 ==========
    stats_results = []

    # print(f"\n[统计] 总错误样本数: {len(error_df)}")

    # 1. 按折统计
    # print("\n【按折统计】")
    for fold_num in sorted(error_df['Fold'].unique()):
        subset = error_df[error_df['Fold'] == fold_num]
        # print(f"  Fold {fold_num}: {len(subset)} 个错误样本")
        stats_results.append({
            '统计类型': '按折统计',
            '类别': f'Fold {fold_num}',
            '错误总数': len(subset),
            '误判分布': '-',
            '单病种分布': '-'
        })

    # 2. 按真实标签统计
    # print("\n【按真实标签统计】")
    for true_label in error_df['真实标签'].unique():
        subset = error_df[error_df['真实标签'] == true_label]
        n_errors = len(subset)
        pred_counts = subset['预测标签'].value_counts().to_dict()

        # print(f"  【{true_label}】: {n_errors} 个错误 → {pred_counts}")
        stats_results.append({
            '统计类型': '按真实标签',
            '类别': true_label,
            '错误总数': n_errors,
            '误判分布': str(pred_counts),
            '单病种分布': '-'
        })

    stats_df = pd.DataFrame(stats_results)

    # ========== 保存到Excel ==========
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '错误分类样本详情.xlsx')

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        error_df.to_excel(writer, sheet_name='错误样本详情', index=False)
        stats_df.to_excel(writer, sheet_name='统计分析', index=False)

    # print(f"\n[已保存] K折错误样本详情 → {output_path}")

    return error_df, stats_df


def train_with_swanlab(model, train_loader, val_loader, optimizer,
                       device, config, adj=None, experiment_name=None, criterion=None,
                       val_dataset=None, save_error_details=True):
    """
    使用SwanLab进行训练和实验跟踪

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        config: Config 配置对象 (或旧版 args)
        adj: 邻接矩阵
        experiment_name: 实验名称
        val_dataset: 验证数据集 (可选, 用于错误分析)
        save_error_details: 是否保存错误样本详情到Excel (默认True，K-Fold模式设为False)
    """
    # 检测是否使用变长序列模式
    use_variable_length = _get_config_value(config, 'model.use_variable_length', False)

    # 从配置获取参数
    epochs = _get_config_value(config, 'training.epochs', 100)
    lr = _get_config_value(config, 'training.lr', 0.0003)
    batch_size = _get_config_value(config, 'training.batch_size', 8)
    num_channels = _get_config_value(config, 'features.num_channels', 22)
    n_class = getattr(config, 'n_class', 4)
    adapt_mode = _get_config_value(config, 'features.adapt_mode', 'full')
    L_win = _get_config_value(config, 'data.L_win', 162)
    random_seed = _get_config_value(config, 'training.random_seed', 3407)
    model_name = _get_config_value(config, 'model.name', 'HDSTGCN')
    ablation = _get_config_value(config, 'model.ablation', 'both')
    output_root = _get_config_value(config, 'data.output_root', './results')
    max_length = _get_config_value(config, 'data.max_length', 330)

    # 调度器参数
    scheduler_factor = _get_config_value(config, 'training.scheduler.factor', 0.5)
    scheduler_patience = _get_config_value(config, 'training.scheduler.patience', 10)
    scheduler_min_lr = _get_config_value(config, 'training.scheduler.min_lr', 1e-6)

    # 早停参数
    early_patience = _get_config_value(config, 'training.early_stopping.patience', 50)

    # 初始化SwanLab (网络错误时自动降级)
    if SWANLAB_AVAILABLE and not _swancab_network_error:
        # 设置API key（如果环境变量中没有）
        swanlab_api_key = os.environ.get('SWANLAB_API_KEY', None)
        if swanlab_api_key:
            _safe_swanlab_call('login', api_key=swanlab_api_key)

        # 使用 Config.to_dict() 获取完整配置字典
        # 这确保所有 config.yaml 中的配置都被记录到 SwanLab
        if hasattr(config, 'to_dict'):
            swanlab_config = config.to_dict(exclude_paths=True)
        else:
            # 兼容旧版 args 对象
            swanlab_config = {
                "ablation": ablation,
                "model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "num_features": num_channels,
                "num_classes": n_class,
                "adapt_mode": adapt_mode,
                "window_length": L_win,
                "random_seed": random_seed,
                "use_variable_length": use_variable_length,
            }
            if use_variable_length:
                swanlab_config["max_length"] = max_length

        _safe_swanlab_call(
            'init',
            project="CPET-Classification",
            experiment_name=experiment_name or f"{model_name}_{adapt_mode}",
            config=swanlab_config
        )

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=scheduler_factor,
    #     patience=scheduler_patience, min_lr=scheduler_min_lr
    # )
    # 将 ReduceLROnPlateau 替换为 CosineAnnealingLR：按照一个余弦曲线在总训练轮数 (epochs) 内平滑地把学习率降到最低值
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,             # T_max 设为总轮数，即在这个轮数内降到最低点
        eta_min=scheduler_min_lr  # 最低学习率，可保持用你配置里的 1e-6
    )
    # 训练循环
    best_acc = 0.0
    best_epoch = 0

    # [修改] 早停变量初始化 - 基于配置的指标保存最佳模型
    # 针对长尾分布问题，Macro-F1 比 Loss 更能反映少数类性能
    # 从配置读取保存指标: "macro_f1", "auprc", 或 "loss"
    save_best_metric = _get_config_value(config, 'training.save_best_metric', 'macro_f1')
    print(f"[最佳模型保存指标] {save_best_metric.upper()}")

    best_macro_f1 = 0.0  # 追踪 Macro-F1 (越大越好)
    best_val_loss = float('inf')  # 保留用于记录
    best_auprc = 0.0  # 追踪 AUPRC
    best_model_state = None  # 保存最佳模型的state_dict
    patience = early_patience  # 容忍度
    counter = 0  # 早停计数器

    # [新增] epoch 级别历史记录 (用于保存到 Excel) - 增加更多指标
    epoch_history = []  # 每个元素: {'epoch': int, 'train_loss': float, 'train_acc': float, 'val_loss': float, 'val_acc': float, 'val_macro_f1': float, 'val_auprc': float}

    save_path = os.path.join(output_root, "../models")
    os.makedirs(save_path, exist_ok=True)

    print("\n" + "="*80)
    print("开始训练")
    if use_variable_length:
        print("[变长序列模式] 已启用")
    print("="*80)

    # [变长模式] 打印长度分布统计
    if use_variable_length:
        try:
            for batch in train_loader:
                if len(batch) == 3:
                    _, lengths, _ = batch
                    print(f"[长度分布] min={lengths.min().item()}, max={lengths.max().item()}, mean={lengths.float().mean().item():.1f}")
                    break
        except Exception as e:
            print(f"[警告] 无法获取长度分布: {e}")

    # ========== 检测多标签模式 ==========
    is_multilabel = getattr(config, 'is_multilabel', False)
    if is_multilabel:
        print("[多标签模式] 已启用")

    # ========== [新增] 检测二分类模式 ==========
    n_classes = len(train_loader.dataset.label_mapping)
    binary_config = getattr(config, 'loss', None)
    if binary_config is None:
        binary_config_obj = None
    else:
        binary_config_obj = getattr(binary_config, 'binary', None)

    # 获取二分类配置参数 (兼容 dataclass 和 dict)
    if binary_config_obj is not None:
        binary_auto_detect = getattr(binary_config_obj, 'auto_detect', True)
        binary_enabled = getattr(binary_config_obj, 'enabled', False)
    else:
        binary_auto_detect = True
        binary_enabled = False

    # [新增] 优先读取 config 中的预设置值 (K-Fold 循环可能已设置)
    is_binary = getattr(config, 'is_binary', False) and not is_multilabel
    minority_idx = getattr(config, 'minority_idx', None)
    majority_idx = getattr(config, 'majority_idx', None)

    # 如果 config 中未设置，则进行检测
    if not is_binary:
        is_binary = (n_classes == 2) and (binary_auto_detect or binary_enabled) and not is_multilabel
        if is_binary:
            minority_idx, majority_idx, samples_per_cls = detect_minority_class_index(train_loader.dataset)
            if minority_idx is None:
                # 检测失败，回退到多分类模式
                is_binary = False
                print("[二分类模式] 检测失败，回退到多分类模式")

    # 打印二分类状态 (无论从哪里检测)
    if is_binary and minority_idx is not None:
        print(f"[二分类模式] 已启用 - 少数类索引={minority_idx}")

    # [修复] 基于 alpha 配置计算类别权重
    # 多标签模式不需要 class_weights
    # 二分类模式使用 BCEWithLogitsLoss，不需要 class_weights
    if not is_multilabel and not is_binary:
        # 获取 alpha 配置
        alpha_config = _get_config_value(config, 'loss.alpha', 'auto')

        if alpha_config is None or alpha_config == "balanced":
            # 均衡权重：所有类别权重为 1.0 (用于 Exp 1: 仅输入端加权)
            class_weights = torch.ones(n_classes, dtype=torch.float).to(device)
            print(f"[权重配置] alpha={alpha_config} → 使用均衡权重 (全为1.0)")
        elif isinstance(alpha_config, list):
            # 手动指定权重
            class_weights = torch.tensor(alpha_config, dtype=torch.float).to(device)
            print(f"[权重配置] alpha=手动指定 → 权重: {alpha_config}")
        else:
            # alpha == "auto" 或其他值：基于样本分布自动计算
            weights = extract_weights_from_trainset(train_loader.dataset)
            class_weights = torch.tensor(weights, dtype=torch.float).to(device)
            print(f"[权重配置] alpha=auto → 基于样本分布计算权重")
    else:
        class_weights = None

    # ========== 构建损失函数 ==========
    loss_type = _get_config_value(config, 'loss.type', 'CrossEntropy')
    supcon_weight = _get_config_value(config, 'loss.supcon_weight', 0.3)
    temperature = _get_config_value(config, 'loss.temperature', 0.07)
    gamma = _get_config_value(config, 'loss.gamma', 1.5)

    # [新增] LCRLoss 参数
    lcr_enabled = _get_config_value(config, 'loss.lcr_enabled', False)
    lcr_lambda_co = _get_config_value(config, 'loss.lcr_lambda_co', 0.1)
    lcr_epsilon = _get_config_value(config, 'loss.lcr_epsilon', 1e-6)

    # [新增] Dice Loss 和 UnifiedLDAMLoss 参数
    dice_smooth = _get_config_value(config, 'loss.dice_smooth', 1.0)
    ldam_max_m = _get_config_value(config, 'loss.ldam_max_m', 0.5)
    ldam_scale_s = _get_config_value(config, 'loss.ldam_scale_s', 30)

    if is_multilabel:
        # 多标签模式
        co_matrix = getattr(config, 'co_occurrence_matrix', None)
        if lcr_enabled and co_matrix is not None:
            criterion = LCRLoss(
                co_occurrence_matrix=co_matrix,
                lambda_co=lcr_lambda_co,
                epsilon=lcr_epsilon
            ).to(device)
            print(f"[损失函数] LCRLoss (lambda_co: {lcr_lambda_co})")
        else:
            criterion = nn.BCEWithLogitsLoss().to(device)
            print("[损失函数] BCEWithLogitsLoss")
    elif is_binary:
        # [新增] 二分类模式: 支持 Dice 和 LDAM
        # 获取类别样本数
        _, _, samples_per_cls = detect_minority_class_index(train_loader.dataset)
        pos_weight_config = getattr(binary_config_obj, 'pos_weight', 'auto') if binary_config_obj is not None else 'auto'

        if loss_type == "Dice":
            criterion = DiceLoss(smooth=dice_smooth).to(device)
            print(f"[损失函数] DiceLoss (smooth={dice_smooth})")
        elif loss_type == "LDAM":
            # 二分类模式: pos_weight 传入 UnifiedLDAMLoss 的 weight 参数
            if pos_weight_config == "auto":
                pos_weight = compute_binary_pos_weight(train_loader.dataset, device)
            else:
                pos_weight = torch.tensor([float(pos_weight_config)], dtype=torch.float).to(device)
                print(f"[二分类权重] 手动配置 pos_weight={pos_weight.item():.4f}")

            # [关键修复] 确保 cls_num_list 顺序为 [多数类样本数, 少数类样本数]
            # LDAM 内部假设: m_list[0]=多数类margin(小), m_list[1]=少数类margin(大)
            # 当 minority_idx==0 时，samples_per_cls=[少数类, 多数类]，需要反转
            # 当 minority_idx==1 时，samples_per_cls=[多数类, 少数类]，顺序正确
            ldam_cls_num_list = samples_per_cls
            if minority_idx == 0:
                ldam_cls_num_list = samples_per_cls[::-1]  # 反转顺序

            criterion = UnifiedLDAMLoss(
                cls_num_list=ldam_cls_num_list,
                max_m=ldam_max_m,
                s=ldam_scale_s,
                weight=pos_weight
            ).to(device)
            print(f"[损失函数] UnifiedLDAMLoss (max_m={ldam_max_m}, s={ldam_scale_s})")
        else:
            # 默认: BCEWithLogitsLoss + pos_weight
            if pos_weight_config == "auto":
                pos_weight = compute_binary_pos_weight(train_loader.dataset, device)
            else:
                pos_weight = torch.tensor([float(pos_weight_config)], dtype=torch.float).to(device)
                print(f"[二分类权重] 手动配置 pos_weight={pos_weight.item():.4f}")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
            print(f"[损失函数] 二分类 BCEWithLogitsLoss (pos_weight={pos_weight.item():.4f})")
    elif loss_type == "CombinedLoss":
        criterion = CombinedLoss(
            ce_weight=1.0,
            supcon_weight=supcon_weight,
            temperature=temperature,
            class_weights=class_weights,
            gamma=gamma,
            loss_type="FocalLoss"  # 使用 Focal 作为基础损失
        )
        print(f"[损失函数] CombinedLoss (SupCon权重: {supcon_weight}, 温度: {temperature})")
    elif loss_type == "FocalLoss":
        criterion = FocalLoss(alpha=class_weights, gamma=gamma).to(device)
        print(f"[损失函数] FocalLoss (gamma: {gamma})")
    elif loss_type == "Dice":
        # [新增] 多分类 Dice Loss
        criterion = DiceLoss(smooth=dice_smooth).to(device)
        print(f"[损失函数] DiceLoss (smooth={dice_smooth})")
    elif loss_type == "LDAM":
        # [新增] 多分类 UnifiedLDAMLoss
        # 需要从 dataset 获取各类别样本数
        label_counts = Counter(train_loader.dataset.labellist)
        sorted_labels = sorted(train_loader.dataset.label_mapping.items(), key=lambda x: x[1])
        cls_num_list = [label_counts.get(name, 0) for name, _ in sorted_labels]
        criterion = UnifiedLDAMLoss(
            cls_num_list=cls_num_list,
            max_m=ldam_max_m,
            s=ldam_scale_s,
            weight=class_weights
        ).to(device)
        print(f"[损失函数] UnifiedLDAMLoss (max_m={ldam_max_m}, s={ldam_scale_s})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        print("[损失函数] CrossEntropyLoss")

    for epoch in range(epochs):
        # ========== 训练/验证 ==========
        if is_multilabel:
            # 多标签模式
            train_loss, train_loss_dict = train_epoch_multilabel(
                model, train_loader, optimizer, criterion, device, adj, epoch, config
            )
            val_loss, val_metrics, val_probs, val_labels = validate_epoch_multilabel(
                model, val_loader, criterion, device, adj, config
            )
            train_acc = 0.0  # 多标签模式不计算训练准确率
            val_acc = val_metrics.get('macro_f1', 0.0)  # 使用 macro_f1 作为主要指标
            val_preds = None  # 多标签模式不使用
        elif is_binary:
            # [新增] 二分类模式
            train_result = train_epoch(
                model, train_loader, optimizer, criterion, device, adj, epoch, config,
                is_binary=True, minority_idx=minority_idx
            )
            # 解包结果 (可能是 2 或 3 个返回值)
            if len(train_result) == 3:
                train_loss, train_acc, train_loss_dict = train_result
            else:
                train_loss, train_acc = train_result
                train_loss_dict = None

            val_loss, val_acc, val_preds, val_labels, val_probs = validate_epoch(
                model, val_loader, criterion, device, adj, config,
                is_binary=True, minority_idx=minority_idx
            )
            # [修改] 二分类模式下使用 minority_idx 作为 pos_label
            val_metrics = compute_classification_metrics(
                np.array(val_preds), np.array(val_labels), val_probs,
                average='macro', minority_idx=minority_idx
            )
            val_macro_f1 = val_metrics['f1_score']
            val_auprc = val_metrics['auprc']
        else:
            # 单标签模式 (多分类)
            train_result = train_epoch(
                model, train_loader, optimizer, criterion, device, adj, epoch, config
            )
            # 解包结果 (可能是 2 或 3 个返回值)
            if len(train_result) == 3:
                train_loss, train_acc, train_loss_dict = train_result
            else:
                train_loss, train_acc = train_result
                train_loss_dict = None

            val_loss, val_acc, val_preds, val_labels, val_probs = validate_epoch(
                model, val_loader, criterion, device, adj, config
            )
            # [修改] 单标签模式下计算完整指标 (Macro-F1, AUPRC)
            val_metrics = compute_classification_metrics(
                np.array(val_preds), np.array(val_labels), val_probs, average='macro'
            )
            val_macro_f1 = val_metrics['f1_score']
            val_auprc = val_metrics['auprc']

        # [新增] 更新学习率
        # scheduler.step(val_loss)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ========== 打印进度 ==========
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if is_multilabel:
                loss_info = f"Train Loss: {train_loss:.4f}"
                if train_loss_dict:
                    loss_info += f" (weight: {train_loss_dict['loss_weight']:.4f}, co: {train_loss_dict['loss_co']:.4f})"
                print(f"Epoch [{epoch+1}/{epochs}] (lr={current_lr:.6f}) - "
                      f"{loss_info} | "
                      f"Val Loss: {val_loss:.4f}, Macro-F1: {val_metrics['macro_f1']:.4f}, "
                      f"Micro-F1: {val_metrics['micro_f1']:.4f}, mAP: {val_metrics['mAP']:.4f}")
            elif is_binary:
                # [修改] 二分类模式打印 - 显示 Macro-F1 和少数类单独指标
                loss_info = f"Train Loss: {train_loss:.4f}"
                print(f"Epoch [{epoch+1}/{epochs}] (lr={current_lr:.6f}) [二分类] - "
                      f"{loss_info}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                      f"Macro-F1: {val_macro_f1:.4f}, "
                      f"少数类(F1): {val_metrics['minority_f1']:.4f}, "
                      f"AUPRC: {val_auprc:.4f}")
            else:
                # [修改] 单标签模式打印完整指标
                loss_info = f"Train Loss: {train_loss:.4f}"
                if train_loss_dict:
                    loss_info += f" (CE: {train_loss_dict['loss_ce']:.4f}, SupCon: {train_loss_dict['loss_supcon']:.4f})"
                print(f"Epoch [{epoch+1}/{epochs}] (lr={current_lr:.6f}) - "
                      f"{loss_info}, Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                      f"Macro-F1: {val_macro_f1:.4f}, AUPRC: {val_auprc:.4f}")

        # ========== 记录到SwanLab ==========
        if SWANLAB_AVAILABLE and not _swancab_network_error:
            log_dict = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1
            }

            if is_multilabel:
                # 多标签指标
                log_dict["val/macro_f1"] = val_metrics['macro_f1']
                log_dict["val/micro_f1"] = val_metrics['micro_f1']
                log_dict["val/mAP"] = val_metrics['mAP']
                log_dict["val/hamming_loss"] = val_metrics['hamming_loss']
                log_dict["val/subset_accuracy"] = val_metrics['subset_accuracy']
                log_dict["val/jaccard"] = val_metrics['jaccard']
                log_dict["val/fallback_count"] = val_metrics.get('fallback_count', 0)

                # LCRLoss 分解
                if train_loss_dict:
                    log_dict["train/loss_weight"] = train_loss_dict['loss_weight']
                    log_dict["train/loss_co"] = train_loss_dict['loss_co']
                    log_dict["train/effective_lambda"] = train_loss_dict.get('effective_lambda', 0)
                    log_dict["train/magnitude_ratio"] = train_loss_dict.get('magnitude_ratio', 1.0)
            else:
                # [修改] 单标签/二分类指标 - 增加完整指标记录
                log_dict["train/accuracy"] = train_acc
                log_dict["val/accuracy"] = val_acc
                log_dict["val/macro_f1"] = val_macro_f1  # 二分类时为少数类F1
                log_dict["val/auprc"] = val_auprc  # 新增
                log_dict["val/auroc"] = val_metrics['auroc']  # 新增
                log_dict["val/precision"] = val_metrics['precision']  # 二分类时为少数类Precision
                log_dict["val/recall"] = val_metrics['recall']  # 二分类时为少数类Recall

                # [修改] 二分类模式额外记录 - 区分 Macro-F1 和少数类单独指标
                if is_binary:
                    log_dict["binary/is_binary"] = 1.0
                    log_dict["binary/minority_idx"] = minority_idx
                    log_dict["binary/f1_minority"] = val_metrics['minority_f1']  # 少数类单独 F1
                    log_dict["binary/precision_minority"] = val_metrics['minority_precision']
                    log_dict["binary/recall_minority"] = val_metrics['minority_recall']
                    log_dict["val/macro_f1"] = val_macro_f1  # 真正的 Macro-F1 (两个类别平均)

                # CombinedLoss 分解
                if train_loss_dict:
                    log_dict["train/loss_ce"] = train_loss_dict['loss_ce']
                    log_dict["train/loss_supcon"] = train_loss_dict['loss_supcon']

                # [新增] 记录 gamma 值 (如果模型有 get_gamma 方法)
                if hasattr(model, 'get_gamma'):
                    gamma_val = model.get_gamma()
                    if gamma_val is not None:
                        log_dict["prior_gate/gamma"] = gamma_val.item()

                # [新增] 记录通道注意力权重 (如果模型有 get_channel_weights 方法)
                if hasattr(model, 'get_channel_weights'):
                    channel_weights = model.get_channel_weights()
                    if channel_weights is not None:
                        # 记录前5个通道权重作为示例
                        for i in range(min(5, len(channel_weights))):
                            log_dict[f"channel_attention/weight_{i}"] = channel_weights[i].item()

            _safe_swanlab_call('log', log_dict)

        # [修改] 记录 epoch 历史到列表 - 增加完整指标
        if is_multilabel:
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'val_macro_f1': float(val_metrics['macro_f1']),
                'val_auprc': float(val_metrics.get('mAP', 0))
            }
        else:
            epoch_record = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'val_macro_f1': float(val_macro_f1),
                'val_auprc': float(val_auprc)
            }
        epoch_history.append(epoch_record)

        # [修改] 保存最佳模型 (基于配置的指标)
        # 支持: "macro_f1" (默认), "auprc", "loss"
        # 针对长尾分布问题，Macro-F1/AUPRC 比 Loss 更能反映少数类性能
        if is_multilabel:
            current_macro_f1 = val_metrics['macro_f1']
            current_auprc = val_metrics.get('mAP', 0)
        else:
            current_macro_f1 = val_macro_f1
            current_auprc = val_auprc

        # 根据配置选择当前指标值
        if save_best_metric == "auprc":
            current_best_metric = current_auprc
            metric_name = "AUPRC"
            is_better = current_best_metric > best_auprc
            best_value = best_auprc
        elif save_best_metric == "loss":
            current_best_metric = val_loss
            metric_name = "Val Loss"
            is_better = current_best_metric < best_val_loss
            best_value = best_val_loss
        else:  # 默认 macro_f1
            current_best_metric = current_macro_f1
            metric_name = "Macro-F1"
            is_better = current_best_metric > best_macro_f1
            best_value = best_macro_f1

        if is_better:
            # 更新所有最佳值
            best_macro_f1 = current_macro_f1
            best_val_loss = val_loss
            best_auprc = current_auprc
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()  # 保存到内存
            counter = 0  # 重置早停计数器

            if (epoch + 1) % 10 == 0:
                print(f"  ✓ 保存最佳模型 (依据: {metric_name}: {current_best_metric:.4f}, "
                      f"Macro-F1: {best_macro_f1:.4f}, AUPRC: {best_auprc:.4f})")
        else:
            # 早停监测 - 基于配置指标无提升
            counter += 1
            if counter >= patience:
                print(f"\n[早停触发] 验证集 {metric_name} 已连续 {patience} 轮未提升。")
                print(f"当前最佳 {metric_name}: {best_value:.4f} (Epoch {best_epoch})")
                print(f"停止训练以防止过拟合。")
                break

    # 训练结束
    print("\n" + "="*80)
    print("训练完成!")
    if is_multilabel:
        print(f"最佳 Macro-F1: {best_macro_f1:.4f}, mAP: {best_auprc:.4f} (Epoch {best_epoch})")
    else:
        print(f"最佳 Macro-F1: {best_macro_f1:.4f}, AUPRC: {best_auprc:.4f}, Acc: {best_acc:.4f} (Epoch {best_epoch})")
        print(f"(对应 Val Loss: {best_val_loss:.4f}, 保存依据: {save_best_metric.upper()})")
    print("="*80)

    # 保存最佳模型到磁盘 (包含 fold 编号)
    dataset = getattr(config, 'dataset', 'CPET_New')
    suffix = "_multilabel" if is_multilabel else ""
    exp_suffix = config.exp_suffix  # 获取实验后缀
    # [修复] 添加 fold 编号到模型路径
    current_fold = getattr(config, '_current_fold', 0) + 1  # 获取当前 fold 编号
    model_path = os.path.join(
        save_path,
        f"best_{model_name}_{dataset}_{adapt_mode}_fold{current_fold}{suffix}{exp_suffix}.pth"
    )
    if best_model_state is not None:
        torch.save(best_model_state, model_path)
        model.load_state_dict(best_model_state)
        print(f"[Fold {current_fold}] 最佳模型已保存: {model_path}")
    else:
        print("警告: 未保存任何最佳模型，使用当前模型进行评估")

    # ========== 最终评估 ==========
    if is_multilabel:
        final_loss, final_metrics, final_probs, final_labels = validate_epoch_multilabel(
            model, val_loader, criterion, device, adj, config
        )

        # 获取标签名称和共现矩阵
        label_names = getattr(config, 'part_actions', None)
        co_matrix = getattr(config, 'co_occurrence_matrix', None)

        # 打印详细的多标签结果矩阵
        print_multilabel_results(
            final_probs, final_labels,
            label_names=label_names,
            co_occurrence_matrix=co_matrix,
            threshold=0.5
        )

        print("\n多标签评估结果汇总:")
        print(f"  Macro-F1: {final_metrics['macro_f1']:.4f}")
        print(f"  Micro-F1: {final_metrics['micro_f1']:.4f}")
        print(f"  mAP: {final_metrics['mAP']:.4f}")
        print(f"  Hamming Loss: {final_metrics['hamming_loss']:.4f}")
        print(f"  Subset Accuracy: {final_metrics['subset_accuracy']:.4f}")
        print(f"  Jaccard: {final_metrics['jaccard']:.4f}")

        cm = None
        report = None

        # 记录最终结果到SwanLab
        if SWANLAB_AVAILABLE and not _swancab_network_error:
            _safe_swanlab_call('log', {
                "final/macro_f1": final_metrics['macro_f1'],
                "final/micro_f1": final_metrics['micro_f1'],
                "final/mAP": final_metrics['mAP'],
                "final/hamming_loss": final_metrics['hamming_loss'],
                "final/subset_accuracy": final_metrics['subset_accuracy']
            })
    else:
        # [修复] 根据是否为二分类模式，传递不同的参数
        if is_binary:
            # 二分类模式
            final_loss, final_acc, final_preds, final_labels, final_probs = validate_epoch(
                model, val_loader, criterion, device, adj, config,
                is_binary=True, minority_idx=minority_idx
            )
        else:
            # 单标签多分类模式
            final_loss, final_acc, final_preds, final_labels, final_probs = validate_epoch(
                model, val_loader, criterion, device, adj, config
            )

        # 计算混淆矩阵和分类报告
        cm = confusion_matrix(final_labels, final_preds)
        part_actions = getattr(config, 'part_actions', [f"Class_{i}" for i in range(n_class)])
        report = classification_report(
            final_labels, final_preds,
            target_names=part_actions,
            zero_division=0
        )

        # 计算完整指标 (ACC, Precision, Recall, F1, AUROC, AUPRC)
        # [修复] 二分类模式必须传递 minority_idx，否则 AUROC 和 minority_f1 会错误使用 pos_label=1
        final_metrics = compute_classification_metrics(
            np.array(final_preds),
            np.array(final_labels),
            final_probs,
            average='macro',
            minority_idx=minority_idx if is_binary else None
        )

        print("\n混淆矩阵:")
        print(cm)
        print("\n分类报告:")
        print(report)

        # 输出完整评估指标汇总
        print("\n" + "="*60)
        print("【评估指标汇总】")
        print("="*60)
        print(f"  ACC (Accuracy):  {final_metrics['accuracy']:.4f}")
        print(f"  Precision:       {final_metrics['precision']:.4f}")
        print(f"  Recall:          {final_metrics['recall']:.4f}")
        print(f"  F1-score:        {final_metrics['f1_score']:.4f}")
        print(f"  AUROC:           {final_metrics['auroc']:.4f}")
        print(f"  AUPRC:           {final_metrics['auprc']:.4f}")
        print("="*60)

        # ========== 输出错误分类的匿名样本编号 ==========
        # [修复] 初始化 error_samples，避免 UnboundLocalError
        error_samples = []
        # if val_dataset is not None and hasattr(val_dataset, 'filenames_list'):
        #     print("\n" + "="*80)
        #     print("错误分类分析")
        #     print("="*80)

        #     filenames = val_dataset.filenames_list

        #     # 获取标签名称映射
        #     idx_to_label = {v: k for k, v in val_dataset.label_mapping.items()}

        #     # 找出所有错误分类的样本
        #     for i, (pred, label) in enumerate(zip(final_preds, final_labels)):
        #         if pred != label:
        #             error_samples.append({
        #                 'filename': filenames[i],
        #                 'true_label': idx_to_label.get(label, f"Class_{label}"),
        #                 'pred_label': idx_to_label.get(pred, f"Class_{pred}"),
        #                 'true_idx': label,
        #                 'pred_idx': pred
        #             })

        #     if error_samples:
        #         print(f"\n共 {len(error_samples)} 个错误分类样本:")

        #         # 按真实标签分组
        #         from collections import defaultdict
        #         errors_by_true_label = defaultdict(list)
        #         for sample in error_samples:
        #             errors_by_true_label[sample['true_label']].append(sample)

        #         # 输出每组错误
        #         for true_label in sorted(errors_by_true_label.keys(),
        #                                 key=lambda x: len(errors_by_true_label[x]),
        #                                 reverse=True):
        #             samples = errors_by_true_label[true_label]
        #             print(f"\n【{true_label}】被错误分类 ({len(samples)} 个):")

        #             # 按预测标签分组
        #             errors_by_pred = defaultdict(list)
        #             for s in samples:
        #                 errors_by_pred[s['pred_label']].append(s['filename'])

        #             for pred_label, fnames in sorted(errors_by_pred.items(),
        #                                             key=lambda x: -len(x[1])):
        #                 print(f"  → 误判为【{pred_label}】: {len(fnames)} 个")
        #                 # 输出匿名样本编号
        #                 for fname in fnames:
        #                     # 仅用于调试时生成匿名样本编号
        #                     patient_id = fname.split('_')[0] if '_' in fname else fname[:-5]
        #                     print(f"      - 患者 {patient_id}")
        #     else:
        #         print("\n所有样本分类正确!")

        #     # 特别关注: 泵功能衰竭 ↔ 缺血性心脏病 的混淆
        #     print("\n" + "-"*60)
        #     print("【重点关注】泵功能衰竭 ↔ 缺血性心脏病 混淆分析:")
        #     print("-"*60)

        #     target_pairs = [
        #         ("泵功能衰竭", "缺血性心脏病"),
        #         ("缺血性心脏病", "泵功能衰竭")
        #     ]

        #     for true_label, pred_label in target_pairs:
        #         # 找到这类错误
        #         pair_errors = [s for s in error_samples
        #                       if s['true_label'] == true_label and s['pred_label'] == pred_label]

        #         if pair_errors:
        #             print(f"\n{true_label} → 误判为 {pred_label} ({len(pair_errors)} 个):")
        #             for s in pair_errors:
        #                 patient_id = s['filename'].split('_')[0] if '_' in s['filename'] else s['filename'][:-5]
        #                 print(f"  - 患者 {patient_id}")
        #         else:
        #             print(f"\n{true_label} → 误判为 {pred_label}: 无")

        # ========== 提取错误样本详细信息并保存为Excel ==========
        # [已注释] 错误分类样本详情功能暂时禁用
        # if error_samples and val_dataset is not None:
        #     # 获取标签文件路径
        #     label_file = getattr(config, 'label_file', None)
        #     if label_file is None and hasattr(config, 'data'):
        #         label_file = getattr(config.data, 'label_file', 'xx_path')

        #     # 输出目录
        #     output_dir = _get_config_value(config, 'data.output_root', './results')

        #     if label_file and os.path.exists(label_file):
        #         if save_error_details:
        #             # 单次训练模式: 直接保存错误样本详情
        #             print("\n" + "="*80)
        #             print("【提取错误样本详细信息】")
        #             print("="*80)

        #             error_df, stats_df = extract_error_patient_details(
        #                 error_samples, label_file, output_dir
        #             )
        #         else:
        #             # K-Fold 模式: 不保存，返回错误样本列表供上层累积
        #             pass
        #     else:
        #         print(f"\n[警告] 标签文件不存在: {label_file}")

        print("="*80)
        # =============================================

        # 记录最终结果到SwanLab
        if SWANLAB_AVAILABLE and not _swancab_network_error:
            _safe_swanlab_call('log', {
                "final/accuracy": final_acc,
                "final/confusion_matrix": cm.tolist()
            })

    # 结束实验
    if SWANLAB_AVAILABLE and not _swancab_network_error:
        _safe_swanlab_call('finish')

    if is_multilabel:
        # 多标签模式: 返回 macro_f1, metrics dict, None, empty error list, epoch_history
        return best_macro_f1, final_metrics, None, [], epoch_history
    else:
        # [修改] 单标签模式: 返回 macro_f1 (模型保存依据), metrics dict, report, error_samples, epoch_history
        # 注意: best_acc 仍保留为 accuracy，但模型保存基于 macro_f1
        return best_macro_f1, final_metrics, report, error_samples, epoch_history


def train_kfold_with_swanlab(model_class, config, n_folds=5, device=None):
    """
    K折交叉验证训练（集成SwanLab）
    ** 已适配统一配置系统 (Config) **
    ** 支持: 变长序列、静态特征、多标签、CNN编码器、gamma学习率分离 **
    ** [新增] Holdout 测试集: 先划分 20% 独立测试集，再进行 K-Fold **

    Args:
        model_class: 模型类
        config: Config 配置对象
        n_folds: 折数
        device: 设备 (可选，默认从配置读取)

    Returns:
        all_results: 所有fold的结果
    """
    from dataset_new import CPETDatasetNewKFold, preload_all_data_for_kfold
    from feature_mapping import create_adjacency_matrix

    # =========================================================================
    # [关键] 在实验开始时立即保存 config.yaml 内容
    # 避免后续修改yaml导致配置与实验名称不匹配
    # =========================================================================
    import yaml
    script_dir = os.path.dirname(__file__)
    config_yaml_path = os.path.join(script_dir, '..', 'configs', 'config.yaml')
    if not os.path.exists(config_yaml_path):
        config_yaml_path = os.path.join(script_dir, 'config.yaml')

    config_yaml_content = ""
    if os.path.exists(config_yaml_path):
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            config_yaml_content = f.read()
        print(f"[配置快照] 已在实验开始时保存 config.yaml 内容")

    # =========================================================================
    # 从配置获取参数
    # =========================================================================
    gpu = _get_config_value(config, 'runtime.gpu', 0)
    if device is None:
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据参数
    random_seed = _get_config_value(config, 'training.random_seed', 3407)
    batch_size = _get_config_value(config, 'training.batch_size', 8)

    # 模型参数
    model_name = _get_config_value(config, 'model.name', 'HDSTGCN')
    use_variable_length = _get_config_value(config, 'model.use_variable_length', False)

    # 确定是否使用静态特征
    use_static = False
    static_dim = 16
    static_ablation = "full"
    num_static_features = 5
    if hasattr(config.model, 'static_features') and config.model.static_features is not None:
        use_static = config.model.static_features.enabled
        static_dim = config.model.static_features.static_dim
        static_ablation = config.model.static_features.ablation
        num_static_features = config.model.static_features.num_features

    # 确定是否多标签模式
    is_multilabel = False
    if hasattr(config, 'task') and config.task is not None:
        is_multilabel = (config.task.mode == "multi_label")
        if is_multilabel:
            print("[K-Fold 任务模式] 多标签分类")
    else:
        print("[K-Fold 任务模式] 单标签分类")

    # 获取邻接矩阵 (HDSTGCN 不需要作为参数传递，模型内部已包含)
    # [修复] 传递 optional_keys 以支持 vco2 和 o2pulse 特征
    optional_keys = []
    if hasattr(config.features, 'o2pulse_enabled') and config.features.o2pulse_enabled:
        optional_keys.append('o2pulse')
    if hasattr(config.features, 'vco2_enabled') and config.features.vco2_enabled:
        optional_keys.append('vco2')
    semantic_adj = create_adjacency_matrix(config.features.adapt_mode, optional_keys if optional_keys else None)
    adj = torch.from_numpy(semantic_adj).float()

    all_results = []

    # =========================================================================
    # [新增] Holdout 测试集配置
    # =========================================================================
    holdout_enabled = False
    holdout_ratio = 0.2
    holdout_seed = 42
    holdout_save = True

    if hasattr(config, 'data') and hasattr(config.data, 'holdout') and config.data.holdout is not None:
        holdout_cfg = config.data.holdout
        holdout_enabled = getattr(holdout_cfg, 'enabled', False)
        holdout_ratio = getattr(holdout_cfg, 'test_ratio', 0.2)
        holdout_seed = getattr(holdout_cfg, 'random_seed', 42)
        holdout_save = getattr(holdout_cfg, 'save_split', True)

    # =========================================================================
    # [新增] 预加载所有数据 + 全局切分
    # =========================================================================
    dev_indices = None
    test_indices = None
    all_data_cache = None

    if holdout_enabled:
        print("\n" + "="*80)
        print("【Holdout 测试集模式】")
        print("="*80)
        print(f"  - 独立测试集比例: {holdout_ratio*100:.0f}%")
        print(f"  - 全局切分随机种子: {holdout_seed}")
        print(f"  - K-Fold 随机种子: {random_seed}")

        # 预加载所有数据 (避免每个 fold 重复加载)
        all_data_cache = preload_all_data_for_kfold(
            config,
            use_variable_length=use_variable_length,
            max_length=config.data.max_length,
            use_static_features=use_static,
            feature_indices=config.features.channels
        )

        # 获取标签列表用于 stratify
        raw_labellist = all_data_cache['raw_labellist']
        label_mapping = all_data_cache['label_mapping']
        is_multilabel_cache = all_data_cache['is_multilabel']

        # 执行全局切分
        n_samples = len(raw_labellist)
        output_dir = _get_config_value(config, 'data.output_root', './results')
        split_file = _find_holdout_split_file(config, output_dir)
        fixed_holdout_split_loaded = False

        if os.path.exists(split_file):
            print(f"\n[Holdout] Loading fixed split: {split_file}")
            with open(split_file, 'r', encoding='utf-8') as f:
                existing_split = json.load(f)
            dev_indices, test_indices = _resolve_holdout_indices_from_split(
                existing_split,
                all_data_cache.get('filenames', []),
                split_file
            )
            fixed_holdout_split_loaded = True
            print("[Holdout] Using existing split (skip re-splitting)")

        if (not fixed_holdout_split_loaded) and is_multilabel_cache:
            # 多标签模式: 使用普通划分
            dev_indices, test_indices = train_test_split(
                range(n_samples),
                test_size=holdout_ratio,
                random_state=holdout_seed
            )
        elif not fixed_holdout_split_loaded:
            # 单标签模式: 使用分层划分
            label_indices = [label_mapping[label] for label in raw_labellist]
            dev_indices, test_indices = train_test_split(
                range(n_samples),
                test_size=holdout_ratio,
                stratify=label_indices,
                random_state=holdout_seed
            )

        print(f"\n[Holdout] Dev_Set: {len(dev_indices)} 样本 (用于 K-Fold)")
        print(f"[Holdout] Test_Set: {len(test_indices)} 样本 (已冻结)")

        # 验证类别分布
        from collections import Counter
        dev_labels = [raw_labellist[i] for i in dev_indices]
        test_labels = [raw_labellist[i] for i in test_indices]
        dev_counts = Counter(dev_labels)
        test_counts = Counter(test_labels)

        print("\n[Holdout] 类别分布验证:")
        print(f"{'类别':<15} {'Dev_Set':<12} {'Test_Set':<12} {'Test比例':<10}")
        print("-" * 55)
        for label in label_mapping.keys():
            dev_n = dev_counts.get(label, 0)
            test_n = test_counts.get(label, 0)
            ratio = test_n / (dev_n + test_n) if (dev_n + test_n) > 0 else 0
            print(f"{label:<15} {dev_n:<12} {test_n:<12} {ratio:.2%}")

        # 保存划分结果
        if holdout_save:
            output_dir = _get_config_value(config, 'data.output_root', './results')
            os.makedirs(output_dir, exist_ok=True)
            exp_suffix = config.exp_suffix  # 获取实验后缀

            split_file = os.path.join(output_dir, f"holdout_split_info{exp_suffix}.json")
            split_info = {
                'holdout_enabled': True,
                'holdout_ratio': holdout_ratio,
                'holdout_seed': holdout_seed,
                'kfold_seed': random_seed,
                'n_samples': n_samples,
                'n_dev': len(dev_indices),
                'n_test': len(test_indices),
                'dev_indices': dev_indices,
                'test_indices': test_indices,
                'label_mapping': {v: k for k, v in label_mapping.items()},
                'class_distribution': {
                    'dev': dict(dev_counts),
                    'test': dict(test_counts)
                }
            }
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)
            print(f"\n[Holdout] 划分结果已保存: {split_file}")

        print("="*80 + "\n")

    # =========================================================================
    # [新增] K折交叉验证指标列表 - 用于计算均值 ± 标准差
    # =========================================================================
    kfold_metrics = {
        'acc': [],      # Accuracy
        'precision': [],# Precision (macro)
        'recall': [],   # Recall (macro)
        'f1': [],       # F1-score (macro)
        'auroc': [],    # AUROC (macro)
        'auprc': [],     # AUPRC (macro)
        # [新增] 少数类指标 (仅二分类任务)
        'minority_f1': [],
        'minority_recall': [],
        'minority_precision': [],
        'pred_minor_rate': [],
        'true_minor_rate': [],
        'minority_tp': [],
        'minority_fn': [],
        'majority_fp': [],
        'majority_tn': [],
    }

    # [新增] 累积所有折的错误样本 (循环外部初始化)
    all_error_samples = []

    # [新增] 累积所有折的 epoch 历史记录
    all_epoch_histories = []  # 每个元素: {'fold': int, 'history': [epoch_records...]}

    # [新增] 保存每个 Fold 的训练集统计量 (用于测试集归一化)
    fold_train_stats = {}  # {fold_number: {'stats': ..., 'static_stats': ...}}

    # [新增] 累积每个 Fold 的测试集评估结果
    all_test_results = []  # 每个元素: {'fold': int, 'metrics': dict}

    # [新增] 预定义变量 (用于每个 Fold 的测试集评估)
    dataset_name = getattr(config, 'dataset', 'CPET_New')
    suffix = "_multilabel" if is_multilabel else ""
    exp_suffix = config.exp_suffix  # 获取实验后缀
    output_dir = _get_config_value(config, 'data.output_root', './results')
    models_dir = os.path.join(output_dir, "../models")  # 与 save_path 一致

    # =========================================================================
    # 遍历每个 Fold
    # =========================================================================
    for fold in range(n_folds):
        print("\n" + "="*80)
        print(f"Fold {fold+1}/{n_folds}")
        print("="*80)

        # [修复] 设置当前 fold 编号到 config，用于模型保存
        config._current_fold = fold

        # 创建数据集
        train_dataset = CPETDatasetNewKFold(
            config, fold_idx=fold, n_folds=n_folds,
            phase="train", random_seed=random_seed,
            feature_indices=config.features.channels,
            use_variable_length=use_variable_length,
            max_length=config.data.max_length,
            use_static_features=use_static,
            dev_indices=dev_indices,        # [新增] 传入 Dev_Set 索引
            test_indices=test_indices,      # [新增] 传入 Test_Set 索引
            all_data_cache=all_data_cache   # [新增] 传入预加载数据
        )

        val_dataset = CPETDatasetNewKFold(
            config, fold_idx=fold, n_folds=n_folds,
            phase="test", random_seed=random_seed,
            feature_indices=config.features.channels,
            use_variable_length=use_variable_length,
            max_length=config.data.max_length,
            use_static_features=use_static,
            dev_indices=dev_indices,
            test_indices=test_indices,
            all_data_cache=all_data_cache
        )

        # [新增] 同步多标签状态 (由数据集设置)
        is_multilabel = getattr(config, 'is_multilabel', False)
        if is_multilabel:
            print(f"[多标签] 标签数: {train_dataset.n_classes}")

        # [新增] 同步 Known-T6 Context 信息
        if hasattr(config, 'known_t6_context') and config.known_t6_context.enabled:
            if hasattr(train_dataset, 't6_n_classes') and train_dataset.t6_n_classes > 0:
                config.t6_n_classes = train_dataset.t6_n_classes  # 传递给 model
                print(f"[Fold {fold+1}] [Known-T6 Context] t6_n_classes={train_dataset.t6_n_classes}")

        # [新增] 检测二分类模式
        n_classes = train_dataset.n_classes
        binary_config_obj = getattr(config.loss, 'binary', None) if hasattr(config, 'loss') else None

        # 获取二分类配置参数 (兼容 dataclass)
        if binary_config_obj is not None:
            binary_auto_detect = getattr(binary_config_obj, 'auto_detect', True)
            binary_enabled = getattr(binary_config_obj, 'enabled', False)
        else:
            binary_auto_detect = True
            binary_enabled = False

        is_binary = (n_classes == 2) and (binary_auto_detect or binary_enabled) and not is_multilabel

        # [新增] 动态检测少数类索引
        minority_idx = None
        if is_binary:
            minority_idx, majority_idx, samples_per_cls = detect_minority_class_index(train_dataset)
            if minority_idx is None:
                is_binary = False
                print(f"[Fold {fold+1}] [二分类模式] 检测失败，回退到多分类模式")
            else:
                print(f"[Fold {fold+1}] [二分类模式] 已启用 - 少数类索引={minority_idx}")
                # [新增] 存储到 config，供 train_with_swanlab() 使用
                config.is_binary = True
                config.minority_idx = minority_idx
                config.majority_idx = majority_idx

        # 更新 num_static_features (动态计算)
        if use_static and hasattr(train_dataset, 'num_static_features'):
            if hasattr(config.model, 'static_features') and config.model.static_features:
                config.model.static_features.num_features = train_dataset.num_static_features
                print(f"[Fold {fold+1}] 静态特征数: {train_dataset.num_static_features}")

        # 更新类别信息
        config.update_with_dataset(train_dataset)

        # [新增] 创建 WeightedRandomSampler (如果启用且为单标签模式)
        train_sampler = None
        use_weighted_sampler = False
        drop_last_flag = False

        if hasattr(config, 'sampler') and config.sampler is not None:
            use_weighted_sampler = config.sampler.enabled and not is_multilabel
            drop_last_flag = config.sampler.drop_last if use_weighted_sampler else False

        if use_weighted_sampler:
            print(f"\n[Fold {fold+1}] [WeightedRandomSampler] 正在创建采样器...")
            train_sampler = create_weighted_sampler(train_dataset, config.part_actions)
            print(f"[Fold {fold+1}] [WeightedRandomSampler] 已启用 (replacement=True, drop_last={drop_last_flag})")
            # 打印类别分布信息
            label_list = train_dataset.labellist
            label_counts = Counter(label_list)
            print(f"[Fold {fold+1}] [WeightedRandomSampler] 类别分布:")
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                weight = 1.0 / (count + 1e-6)
                print(f"  {label}: {count} 样本, 权重={weight:.4f}")

        # 创建数据加载器
        if use_variable_length:
            from dataset_new import collate_fn_variable_length
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size,
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
            print(f"数据加载器: 变长模式")
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size,
                shuffle=False if train_sampler else True,  # 使用 sampler 时必须 shuffle=False
                sampler=train_sampler,
                num_workers=0,
                drop_last=drop_last_flag
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1,
                shuffle=False, num_workers=0
            )
            print(f"数据加载器: 固定长度模式")

        print(f"Fold {fold+1} - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

        # =========================================================================
        # 创建模型 (复用 _create_model 逻辑)
        # =========================================================================
        model = _create_model_for_kfold(config, model_class, device, use_variable_length, is_binary=is_binary)

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")

        # [新增] 打印时序编码器参数量和消融模式
        if hasattr(model, 'temporal_encoder_ablation'):
            temporal_params = sum(p.numel() for p in model.temporal_encoder.parameters())
            print(f"时序编码器: {model.temporal_encoder_ablation} (参数: {temporal_params:,})")

        # =========================================================================
        # 创建优化器 (支持 gamma 参数学习率分离)
        # =========================================================================
        gamma_lr_scale = 0.3
        if hasattr(config.model, 'prior_gate') and config.model.prior_gate is not None:
            gamma_lr_scale = config.model.prior_gate.gamma_lr_scale

        if model_name == "HDSTGCN" and config.model.graph_ablation == "prior_masked":
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

        # =========================================================================
        # 训练
        # =========================================================================
        result = train_with_swanlab(
            model, train_loader, val_loader, optimizer,
            device, config, adj,
            experiment_name=f"{model_name}_fold{fold+1}",
            val_dataset=val_dataset,
            save_error_details=False  # K-Fold 模式: 不在每折中保存错误样本
        )

        # 解包返回值 (单标签模式返回5个值，多标签模式返回5个值)
        # 第一个返回值现在是 best_macro_f1 (模型保存依据)
        if is_multilabel:
            best_macro_f1, final_metrics, _, _, fold_epoch_history = result
            fold_error_samples = []
        else:
            best_macro_f1, final_metrics, report, fold_error_samples, fold_epoch_history = result

        # [新增] 保存当前 fold 的 epoch 历史
        all_epoch_histories.append({
            'fold': fold + 1,
            'history': fold_epoch_history
        })

        # [新增] 保存当前 Fold 的训练集统计量 (用于测试集归一化)
        fold_train_stats[fold + 1] = {
            'stats': train_dataset.stats,  # 动态特征统计量
            'static_stats': getattr(train_dataset, 'static_stats', None)  # 静态特征统计量
        }

        # =========================================================================
        # [新增] 每个 Fold 在测试集上评估
        # =========================================================================
        if holdout_enabled and test_indices is not None:
            print(f"\n[Fold {fold+1}] 在测试集上评估...")

            # 获取当前 Fold 的模型路径 (使用 models_dir 与保存路径一致)
            fold_model_path = os.path.join(
                models_dir,
                f"best_{model_name}_{dataset_name}_{config.features.adapt_mode}_fold{fold+1}{suffix}{exp_suffix}.pth"
            )

            if os.path.exists(fold_model_path):
                # 评估该 Fold 在测试集上
                fold_test_metrics = _evaluate_single_fold_on_testset(
                    config, model_class, device, fold+1,
                    fold_model_path, fold_train_stats[fold+1],
                    test_indices, dev_indices, all_data_cache,
                    use_variable_length, use_static, is_multilabel,
                    is_binary, minority_idx,
                    random_seed
                )

                # 保存测试集结果
                all_test_results.append({
                    'fold': fold + 1,
                    'metrics': fold_test_metrics
                })

                # 打印完整测试集指标
                print(f"  【测试集评估指标】")
                print(f"    - Accuracy:  {fold_test_metrics['accuracy']:.4f}")
                print(f"    - Precision: {fold_test_metrics['precision']:.4f}")
                print(f"    - Recall:    {fold_test_metrics['recall']:.4f}")
                print(f"    - F1-score:  {fold_test_metrics['f1_score']:.4f}")
                print(f"    - Macro-F1:  {fold_test_metrics['macro_f1']:.4f}")
                print(f"    - AUROC:     {fold_test_metrics['auroc']:.4f}")
                print(f"    - AUPRC:     {fold_test_metrics['auprc']:.4f}")
                
                # [新增] 打印 minority-class 指标 (如果为二分类任务)
                target_col_name = _get_config_value(config, 'features.target_col_name', '')
                if target_col_name in BINARY_MINORITY_METRIC_TASKS and is_binary:
                    minority_idx = MINORITY_IDX_MAP.get(target_col_name, None)
                    if minority_idx is not None:
                        print(f"  [Minority Metrics] target_col_name={target_col_name}")
                        print(f"    minority_idx={minority_idx}")
                        print(f"    minority_precision={fold_test_metrics.get('minority_precision_full', 0):.4f}")
                        print(f"    minority_recall={fold_test_metrics.get('minority_recall_full', 0):.4f}")
                        print(f"    minority_f1={fold_test_metrics.get('minority_f1_full', 0):.4f}")
                        print(f"    true_minor_rate={fold_test_metrics.get('true_minor_rate', 0):.4f}")
                        print(f"    pred_minor_rate={fold_test_metrics.get('pred_minor_rate', 0):.4f}")
                        print(f"    TP={fold_test_metrics.get('minority_tp', 0)}, FN={fold_test_metrics.get('minority_fn', 0)}, FP={fold_test_metrics.get('majority_fp', 0)}, TN={fold_test_metrics.get('majority_tn', 0)}")

            else:
                print(f"  [警告] 未找到 Fold {fold+1} 的模型文件")

        # 记录结果
        if is_multilabel:
            all_results.append({
                'fold': fold + 1,
                'macro_f1': best_macro_f1,
                'metrics': final_metrics,
                'report': None
            })
            # 记录多标签指标
            kfold_metrics['acc'].append(best_macro_f1)
            kfold_metrics['f1'].append(final_metrics.get('macro_f1', best_macro_f1))
        else:
            all_results.append({
                'fold': fold + 1,
                'macro_f1': best_macro_f1,  # [修改] 现在记录 macro_f1 作为主要指标
                'accuracy': final_metrics['accuracy'],  # 同时记录 accuracy
                'metrics': final_metrics,
                'report': report
            })
            # 记录单标签完整指标
            kfold_metrics['acc'].append(final_metrics['accuracy'])
            kfold_metrics['precision'].append(final_metrics['precision'])
            kfold_metrics['recall'].append(final_metrics['recall'])
            kfold_metrics['f1'].append(final_metrics['f1_score'])
            kfold_metrics['auroc'].append(final_metrics['auroc'])
            kfold_metrics['auprc'].append(final_metrics['auprc'])

            # [新增] 记录少数类指标 (仅二分类任务)
            if is_binary:
                kfold_metrics['minority_f1'].append(final_metrics.get('minority_f1_full', 0))
                kfold_metrics['minority_recall'].append(final_metrics.get('minority_recall_full', 0))
                kfold_metrics['minority_precision'].append(final_metrics.get('minority_precision_full', 0))
                kfold_metrics['pred_minor_rate'].append(final_metrics.get('pred_minor_rate', 0))
                kfold_metrics['true_minor_rate'].append(final_metrics.get('true_minor_rate', 0))
                kfold_metrics['minority_tp'].append(final_metrics.get('minority_tp', 0))
                kfold_metrics['minority_fn'].append(final_metrics.get('minority_fn', 0))
                kfold_metrics['majority_fp'].append(final_metrics.get('majority_fp', 0))
                kfold_metrics['majority_tn'].append(final_metrics.get('majority_tn', 0))
            else:
                # 多分类任务: 填充 N/A
                kfold_metrics['minority_f1'].append('N/A')
                kfold_metrics['minority_recall'].append('N/A')
                kfold_metrics['minority_precision'].append('N/A')
                kfold_metrics['pred_minor_rate'].append('N/A')
                kfold_metrics['true_minor_rate'].append('N/A')
                kfold_metrics['minority_tp'].append('N/A')
                kfold_metrics['minority_fn'].append('N/A')
                kfold_metrics['majority_fp'].append('N/A')
                kfold_metrics['majority_tn'].append('N/A')

            # 累积错误样本 (添加 Fold 列)
            for sample in fold_error_samples:
                sample['fold'] = fold + 1
                all_error_samples.append(sample)

    # =========================================================================
    # K折循环结束 - 统一保存错误样本
    # [已注释] 错误分类样本详情功能暂时禁用
    # =========================================================================
    # if all_error_samples and not is_multilabel:
    #     label_file = getattr(config, 'label_file', None)
    #     if label_file is None and hasattr(config, 'data'):
        #         label_file = getattr(config.data, 'label_file', 'xx_path')

    #     output_dir = _get_config_value(config, 'data.output_root', './results')

    #     if label_file and os.path.exists(label_file):
    #         # print("\n" + "="*80)
    #         # print("【K折交叉验证 - 统一保存错误样本详情】")
    #         # print("="*80)

    #         # 调用保存函数 (累积了所有折的错误样本)
    #         error_df, stats_df = extract_error_patient_details_kfold(
    #             all_error_samples, label_file, output_dir
    #         )

    # =========================================================================
    # 打印总结 - 完整指标
    # =========================================================================
    print("\n" + "="*80)
    print("K折交叉验证总结")
    print("="*80)

    if is_multilabel:
        macro_f1s = [r['macro_f1'] for r in all_results]
        print(f"平均 Macro-F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
        print(f"最佳 Macro-F1: {np.max(macro_f1s):.4f}")
        print(f"最差 Macro-F1: {np.min(macro_f1s):.4f}")
    else:
        # 完整指标汇总表格
        print("\n【评估指标汇总 (均值 ± 标准差)】")
        print("-" * 50)
        print(f"{'指标':<15} {'均值':<12} {'标准差':<12}")
        print("-" * 50)
        for metric_name in ['acc', 'precision', 'recall', 'f1', 'auroc', 'auprc']:
            values = kfold_metrics[metric_name]
            mean_val = np.mean(values)
            std_val = np.std(values)
            display_name = {
                'acc': 'ACC',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1': 'F1-score',
                'auroc': 'AUROC',
                'auprc': 'AUPRC'
            }[metric_name]
            print(f"{display_name:<15} {mean_val:.4f}       ± {std_val:.4f}")
        print("-" * 50)

        # 最佳/最差折
        print(f"\n最佳准确率: {np.max(kfold_metrics['acc']):.4f} (Fold {np.argmax(kfold_metrics['acc'])+1})")
        print(f"最差准确率: {np.min(kfold_metrics['acc']):.4f} (Fold {np.argmin(kfold_metrics['acc'])+1})")

    # =========================================================================
    # 保存汇总结果到 Excel (多个 sheets)
    # =========================================================================
    output_dir = _get_config_value(config, 'data.output_root', './results')
    os.makedirs(output_dir, exist_ok=True)

    # 构建 KFold 汇总数据
    kfold_data = []
    for r in all_results:
        if is_multilabel:
            kfold_data.append({
                'Fold': r['fold'],
                'Macro-F1': r['macro_f1'],
                'Micro-F1': r['metrics'].get('micro_f1', 0),
                'mAP': r['metrics'].get('mAP', 0)
            })
        else:
            m = r['metrics']
            # [新增] 检查是否为二分类任务且需要 minority 指标
            target_col_name = _get_config_value(config, 'features.target_col_name', '')
            is_minority_task = target_col_name in BINARY_MINORITY_METRIC_TASKS and is_binary

            row_data = {
                'Fold': r['fold'],
                'ACC': m['accuracy'],
                'Precision': m['precision'],
                'Recall': m['recall'],
                'F1-score': m['f1_score'],
                'AUROC': m['auroc'],
                'AUPRC': m['auprc']
            }

            # [新增] 添加 minority-class 指标
            if is_minority_task:
                row_data['minority_f1'] = m.get('minority_f1_full', 0)
                row_data['minority_recall'] = m.get('minority_recall_full', 0)
                row_data['minority_precision'] = m.get('minority_precision_full', 0)
                row_data['pred_minor_rate'] = m.get('pred_minor_rate', 0)
                row_data['true_minor_rate'] = m.get('true_minor_rate', 0)
                row_data['minority_tp'] = m.get('minority_tp', 0)
                row_data['minority_fn'] = m.get('minority_fn', 0)
                row_data['majority_fp'] = m.get('majority_fp', 0)
                row_data['majority_tn'] = m.get('majority_tn', 0)
            else:
                row_data['minority_f1'] = 'N/A'
                row_data['minority_recall'] = 'N/A'
                row_data['minority_precision'] = 'N/A'
                row_data['pred_minor_rate'] = 'N/A'
                row_data['true_minor_rate'] = 'N/A'
                row_data['minority_tp'] = 'N/A'
                row_data['minority_fn'] = 'N/A'
                row_data['majority_fp'] = 'N/A'
                row_data['majority_tn'] = 'N/A'

            kfold_data.append(row_data)


    # 添加汇总行
    if is_multilabel:
        macro_f1s = [r['macro_f1'] for r in all_results]
        kfold_data.append({
            'Fold': 'Mean±Std',
            'Macro-F1': f"{np.mean(macro_f1s):.4f}±{np.std(macro_f1s):.4f}",
            'Micro-F1': '-',
            'mAP': '-'
        })
    else:
        # [新增] 检查是否为二分类任务且需要 minority 指标
        target_col_name = _get_config_value(config, 'features.target_col_name', '')
        is_minority_task = target_col_name in BINARY_MINORITY_METRIC_TASKS

        # 基础汇总行
        summary_row = {
            'Fold': 'Mean±Std',
            'ACC': f"{np.mean(kfold_metrics['acc']):.4f}±{np.std(kfold_metrics['acc']):.4f}",
            'Precision': f"{np.mean(kfold_metrics['precision']):.4f}±{np.std(kfold_metrics['precision']):.4f}",
            'Recall': f"{np.mean(kfold_metrics['recall']):.4f}±{np.std(kfold_metrics['recall']):.4f}",
            'F1-score': f"{np.mean(kfold_metrics['f1']):.4f}±{np.std(kfold_metrics['f1']):.4f}",
            'AUROC': f"{np.mean(kfold_metrics['auroc']):.4f}±{np.std(kfold_metrics['auroc']):.4f}",
            'AUPRC': f"{np.mean(kfold_metrics['auprc']):.4f}±{np.std(kfold_metrics['auprc']):.4f}"
        }

        # [新增] 添加 minority-class 指标汇总
        if is_minority_task and is_binary:
            # 计算 minority 指标的 mean±std
            minority_f1_vals = [v for v in kfold_metrics['minority_f1'] if v != 'N/A']
            minority_recall_vals = [v for v in kfold_metrics['minority_recall'] if v != 'N/A']
            minority_precision_vals = [v for v in kfold_metrics['minority_precision'] if v != 'N/A']
            pred_minor_rate_vals = [v for v in kfold_metrics['pred_minor_rate'] if v != 'N/A']
            true_minor_rate_vals = [v for v in kfold_metrics['true_minor_rate'] if v != 'N/A']
            minority_tp_vals = [v for v in kfold_metrics['minority_tp'] if v != 'N/A']
            minority_fn_vals = [v for v in kfold_metrics['minority_fn'] if v != 'N/A']
            majority_fp_vals = [v for v in kfold_metrics['majority_fp'] if v != 'N/A']
            majority_tn_vals = [v for v in kfold_metrics['majority_tn'] if v != 'N/A']

            if minority_f1_vals:
                summary_row['minority_f1'] = f"{np.mean(minority_f1_vals):.4f}±{np.std(minority_f1_vals):.4f}"
                summary_row['minority_recall'] = f"{np.mean(minority_recall_vals):.4f}±{np.std(minority_recall_vals):.4f}"
                summary_row['minority_precision'] = f"{np.mean(minority_precision_vals):.4f}±{np.std(minority_precision_vals):.4f}"
                summary_row['pred_minor_rate'] = f"{np.mean(pred_minor_rate_vals):.4f}±{np.std(pred_minor_rate_vals):.4f}"
                summary_row['true_minor_rate'] = f"{np.mean(true_minor_rate_vals):.4f}±{np.std(true_minor_rate_vals):.4f}"
                # 计数指标也可以计算 mean±std
                summary_row['minority_tp'] = f"{np.mean(minority_tp_vals):.1f}±{np.std(minority_tp_vals):.1f}"
                summary_row['minority_fn'] = f"{np.mean(minority_fn_vals):.1f}±{np.std(minority_fn_vals):.1f}"
                summary_row['majority_fp'] = f"{np.mean(majority_fp_vals):.1f}±{np.std(majority_fp_vals):.1f}"
                summary_row['majority_tn'] = f"{np.mean(majority_tn_vals):.1f}±{np.std(majority_tn_vals):.1f}"
            else:
                summary_row['minority_f1'] = 'N/A'
                summary_row['minority_recall'] = 'N/A'
                summary_row['minority_precision'] = 'N/A'
                summary_row['pred_minor_rate'] = 'N/A'
                summary_row['true_minor_rate'] = 'N/A'
                summary_row['minority_tp'] = 'N/A'
                summary_row['minority_fn'] = 'N/A'
                summary_row['majority_fp'] = 'N/A'
                summary_row['majority_tn'] = 'N/A'
        else:
            summary_row['minority_f1'] = 'N/A'
            summary_row['minority_recall'] = 'N/A'
            summary_row['minority_precision'] = 'N/A'
            summary_row['pred_minor_rate'] = 'N/A'
            summary_row['true_minor_rate'] = 'N/A'
            summary_row['minority_tp'] = 'N/A'
            summary_row['minority_fn'] = 'N/A'
            summary_row['majority_fp'] = 'N/A'
            summary_row['majority_tn'] = 'N/A'

        kfold_data.append(summary_row)

    # config.yaml 内容已在函数开始时保存 (config_yaml_content)
    # 构建 epoch 历史数据 (包含新增的 Macro-F1 和 AUPRC 字段)
    epoch_history_data = []
    for fold_history in all_epoch_histories:
        fold_num = fold_history['fold']
        for record in fold_history['history']:
            epoch_record = {
                'Fold': fold_num,
                'Epoch': record['epoch'],
                'Train Loss': record['train_loss'],
                'Train Acc': record['train_acc'],
                'Val Loss': record['val_loss'],
                'Val Acc': record['val_acc'],
                'Val Macro-F1': record.get('val_macro_f1', 0.0),  # 新增
                'Val AUPRC': record.get('val_auprc', 0.0)  # 新增
            }
            epoch_history_data.append(epoch_record)

    # 保存到 Excel (多个 sheets)
    suffix = "_multilabel" if is_multilabel else ""
    exp_suffix = config.exp_suffix  # 获取实验后缀
    excel_path = os.path.join(output_dir, f"kfold_summary_metrics_{config.model.name}_{config.features.adapt_mode}{suffix}{exp_suffix}.xlsx")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: KFold 汇总指标
        kfold_df = pd.DataFrame(kfold_data)
        kfold_df.to_excel(writer, sheet_name='KFold_Summary', index=False)

        # Sheet 2: config.yaml 内容
        if config_yaml_content:
            config_lines = config_yaml_content.split('\n')
            config_df = pd.DataFrame({'config.yaml': config_lines})
            config_df.to_excel(writer, sheet_name='Config', index=False)

        # Sheet 3: Epoch 训练历史
        if epoch_history_data:
            epoch_df = pd.DataFrame(epoch_history_data)
            epoch_df.to_excel(writer, sheet_name='Epoch_History', index=False)

    print(f"\n[已保存] K折汇总结果 → {excel_path}")
    print(f"  - Sheet 'KFold_Summary': K折交叉验证汇总指标")
    if config_yaml_content:
        print(f"  - Sheet 'Config': 当前配置文件内容")
    if epoch_history_data:
        print(f"  - Sheet 'Epoch_History': Epoch 级别训练历史")

    # =========================================================================
    # [新增] 保存训练集统计量到文件 (用于独立测试集评估)
    # =========================================================================
    stats_file = os.path.join(output_dir, f"kfold_train_stats_{config.model.name}_{config.features.adapt_mode}{suffix}{exp_suffix}.json")

    # 找到最佳 Fold
    best_fold_idx = np.argmax([r['macro_f1'] for r in all_results])
    best_fold = all_results[best_fold_idx]
    best_fold_stats = fold_train_stats.get(best_fold['fold'], {})

    stats_to_save = {
        'best_fold': best_fold['fold'],
        'best_macro_f1': best_fold['macro_f1'],
        'model_name': config.model.name,
        'adapt_mode': config.features.adapt_mode,
        'n_folds': len(all_results),
        'all_fold_results': [
            {'fold': r['fold'], 'macro_f1': r['macro_f1']}
            for r in all_results
        ]
    }

    # 保存动态特征统计量 (需要转换为可序列化格式)
    # [关键修复] 保存完整的统计量，包括 robust 方法需要的 median, q25, q75
    if best_fold_stats.get('stats') is not None:
        stats = best_fold_stats['stats']
        stats_to_save['train_stats'] = {}
        for key in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']:
            if key in stats:
                val = stats[key]
                stats_to_save['train_stats'][key] = val.tolist() if hasattr(val, 'tolist') else list(val)

    # 保存静态特征统计量
    if best_fold_stats.get('static_stats') is not None:
        static_stats = best_fold_stats['static_stats']
        stats_to_save['train_static_stats'] = {}
        for key in ['mean', 'std']:
            if key in static_stats:
                val = static_stats[key]
                stats_to_save['train_static_stats'][key] = val.tolist() if hasattr(val, 'tolist') else list(static_stats[key])

    # [新增] 保存所有 Fold 的统计量 (用于 skip_kfold 模式评估所有 Fold)
    all_fold_stats_serializable = {}
    for fold_num, fold_stat in fold_train_stats.items():
        fold_stat_dict = {}
        if fold_stat.get('stats') is not None:
            stats = fold_stat['stats']
            fold_stat_dict['stats'] = {}
            for key in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75']:
                if key in stats:
                    val = stats[key]
                    fold_stat_dict['stats'][key] = val.tolist() if hasattr(val, 'tolist') else list(val)
        if fold_stat.get('static_stats') is not None:
            static_stats = fold_stat['static_stats']
            fold_stat_dict['static_stats'] = {}
            for key in ['mean', 'std']:
                if key in static_stats:
                    val = static_stats[key]
                    fold_stat_dict['static_stats'][key] = val.tolist() if hasattr(val, 'tolist') else list(static_stats[key])
        all_fold_stats_serializable[fold_num] = fold_stat_dict

    stats_to_save['all_fold_stats'] = all_fold_stats_serializable

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_to_save, f, indent=2, ensure_ascii=False)

    print(f"[已保存] 训练集统计量 → {stats_file}")
    print(f"  - 最佳 Fold: {best_fold['fold']} (Macro-F1: {best_fold['macro_f1']:.4f})")

    # =========================================================================
    # [新增] 独立测试集评估 (Holdout 模式)
    # =========================================================================
    test_results = None
    if holdout_enabled and test_indices is not None:
        print("\n" + "="*80)
        print("【独立测试集评估】")
        print("="*80)
        print(f"测试集样本数: {len(test_indices)}")

        # =========================================================================
        # [新增] 输出所有 Fold 在测试集上的结果 + 均值±标准差
        # =========================================================================
        if all_test_results:
            print("\n" + "-"*80)
            print("【各 Fold 测试集评估结果】")
            print("-"*80)

            # 收集各指标
            test_accs = []
            test_f1s = []
            test_aurocs = []
            test_auprcs = []
            test_precisions = []
            test_recalls = []
            test_macro_f1s = []
                        # [新增] minority-class 指标
            test_minority_f1s = []
            test_minority_recalls = []
            test_minority_precisions = []
            test_pred_minor_rates = []
            test_true_minor_rates = []
            test_minority_tps = []
            test_minority_fns = []
            test_majority_fps = []
            test_majority_tns = []

            # [新增] 检查是否为需要 minority 指标的任务
            target_col_name = _get_config_value(config, 'features.target_col_name', '')
            is_minority_task = target_col_name in BINARY_MINORITY_METRIC_TASKS and is_binary

            print(f"{'Fold':<6} {'Acc':<8} {'F1':<8} {'Macro-F1':<10} {'AUROC':<8} {'AUPRC':<8}")
            print("-"*60)

            for result in all_test_results:
                m = result['metrics']
                print(f"Fold {result['fold']:<2} {m['accuracy']:.4f}   {m['f1_score']:.4f}   {m['macro_f1']:.4f}     {m['auroc']:.4f}   {m['auprc']:.4f}")

                test_accs.append(m['accuracy'])
                test_f1s.append(m['f1_score'])
                test_macro_f1s.append(m['macro_f1'])
                test_aurocs.append(m['auroc'])
                test_auprcs.append(m['auprc'])
                test_precisions.append(m['precision'])
                test_recalls.append(m['recall'])

                # [新增] 收集 minority-class 指标
                if is_minority_task:
                    test_minority_f1s.append(m.get('minority_f1_full', 0))
                    test_minority_recalls.append(m.get('minority_recall_full', 0))
                    test_minority_precisions.append(m.get('minority_precision_full', 0))
                    test_pred_minor_rates.append(m.get('pred_minor_rate', 0))
                    test_true_minor_rates.append(m.get('true_minor_rate', 0))
                    test_minority_tps.append(m.get('minority_tp', 0))
                    test_minority_fns.append(m.get('minority_fn', 0))
                    test_majority_fps.append(m.get('majority_fp', 0))
                    test_majority_tns.append(m.get('majority_tn', 0))

            # 计算均值±标准差
            print("-"*60)
            print(f"{'Mean':<6} {np.mean(test_accs):.4f}   {np.mean(test_f1s):.4f}   {np.mean(test_macro_f1s):.4f}     {np.mean(test_aurocs):.4f}   {np.mean(test_auprcs):.4f}")
            print(f"{'Std':<6} {np.std(test_accs):.4f}   {np.std(test_f1s):.4f}   {np.std(test_macro_f1s):.4f}     {np.std(test_aurocs):.4f}   {np.std(test_auprcs):.4f}")
            print("-"*60)

            # 完整输出均值±标准差
            print("\n【测试集评估结果汇总 (Mean ± Std)】")
            print("="*50)
            print(f"{'指标':<15} {'值':<20}")
            print("-"*50)
            print(f"{'Accuracy':<15} {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
            print(f"{'Precision':<15} {np.mean(test_precisions):.4f} ± {np.std(test_precisions):.4f}")
            print(f"{'Recall':<15} {np.mean(test_recalls):.4f} ± {np.std(test_recalls):.4f}")
            print(f"{'F1-score':<15} {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
            print(f"{'Macro-F1':<15} {np.mean(test_macro_f1s):.4f} ± {np.std(test_macro_f1s):.4f}")
            print(f"{'AUROC':<15} {np.mean(test_aurocs):.4f} ± {np.std(test_aurocs):.4f}")
            print(f"{'AUPRC':<15} {np.mean(test_auprcs):.4f} ± {np.std(test_auprcs):.4f}")
            print("-"*50)

            # 保存测试集汇总结果
            test_results = {
                'n_test_samples': len(test_indices),
                'n_folds': len(all_test_results),
                'fold_results': all_test_results,
                'mean_std': {
                    'accuracy': {'mean': float(np.mean(test_accs)), 'std': float(np.std(test_accs))},
                    'precision': {'mean': float(np.mean(test_precisions)), 'std': float(np.std(test_precisions))},
                    'recall': {'mean': float(np.mean(test_recalls)), 'std': float(np.std(test_recalls))},
                    'f1_score': {'mean': float(np.mean(test_f1s)), 'std': float(np.std(test_f1s))},
                    'macro_f1': {'mean': float(np.mean(test_macro_f1s)), 'std': float(np.std(test_macro_f1s))},
                    'auroc': {'mean': float(np.mean(test_aurocs)), 'std': float(np.std(test_aurocs))},
                    'auprc': {'mean': float(np.mean(test_auprcs)), 'std': float(np.std(test_auprcs))}
                }
            }

            # 添加测试集汇总到 Excel
            test_summary_data = []
            for result in all_test_results:
                m = result['metrics']
                row_data = {
                    'Fold': result['fold'],
                    'Accuracy': m['accuracy'],
                    'Precision': m['precision'],
                    'Recall': m['recall'],
                    'F1-score': m['f1_score'],
                    'Macro-F1': m['macro_f1'],
                    'AUROC': m['auroc'],
                    'AUPRC': m['auprc']
                }

                # [新增] 添加 minority-class 指标
                if is_minority_task:
                    row_data['minority_f1'] = m.get('minority_f1_full', 0)
                    row_data['minority_recall'] = m.get('minority_recall_full', 0)
                    row_data['minority_precision'] = m.get('minority_precision_full', 0)
                    row_data['pred_minor_rate'] = m.get('pred_minor_rate', 0)
                    row_data['true_minor_rate'] = m.get('true_minor_rate', 0)
                    row_data['minority_tp'] = m.get('minority_tp', 0)
                    row_data['minority_fn'] = m.get('minority_fn', 0)
                    row_data['majority_fp'] = m.get('majority_fp', 0)
                    row_data['majority_tn'] = m.get('majority_tn', 0)
                else:
                    row_data['minority_f1'] = 'N/A'
                    row_data['minority_recall'] = 'N/A'
                    row_data['minority_precision'] = 'N/A'
                    row_data['pred_minor_rate'] = 'N/A'
                    row_data['true_minor_rate'] = 'N/A'
                    row_data['minority_tp'] = 'N/A'
                    row_data['minority_fn'] = 'N/A'
                    row_data['majority_fp'] = 'N/A'
                    row_data['majority_tn'] = 'N/A'

                test_summary_data.append(row_data)

            # 添加汇总行
            mean_row = {
                'Fold': 'Mean',
                'Accuracy': np.mean(test_accs),
                'Precision': np.mean(test_precisions),
                'Recall': np.mean(test_recalls),
                'F1-score': np.mean(test_f1s),
                'Macro-F1': np.mean(test_macro_f1s),
                'AUROC': np.mean(test_aurocs),
                'AUPRC': np.mean(test_auprcs)
            }
            std_row = {
                'Fold': 'Std',
                'Accuracy': np.std(test_accs),
                'Precision': np.std(test_precisions),
                'Recall': np.std(test_recalls),
                'F1-score': np.std(test_f1s),
                'Macro-F1': np.std(test_macro_f1s),
                'AUROC': np.std(test_aurocs),
                'AUPRC': np.std(test_auprcs)
           }

            # [新增] 添加 minority-class 指标汇总
            if is_minority_task and test_minority_f1s:
                mean_row['minority_f1'] = np.mean(test_minority_f1s)
                mean_row['minority_recall'] = np.mean(test_minority_recalls)
                mean_row['minority_precision'] = np.mean(test_minority_precisions)
                mean_row['pred_minor_rate'] = np.mean(test_pred_minor_rates)
                mean_row['true_minor_rate'] = np.mean(test_true_minor_rates)
                mean_row['minority_tp'] = np.mean(test_minority_tps)
                mean_row['minority_fn'] = np.mean(test_minority_fns)
                mean_row['majority_fp'] = np.mean(test_majority_fps)
                mean_row['majority_tn'] = np.mean(test_majority_tns)

                std_row['minority_f1'] = np.std(test_minority_f1s)
                std_row['minority_recall'] = np.std(test_minority_recalls)
                std_row['minority_precision'] = np.std(test_minority_precisions)
                std_row['pred_minor_rate'] = np.std(test_pred_minor_rates)
                std_row['true_minor_rate'] = np.std(test_true_minor_rates)
                std_row['minority_tp'] = np.std(test_minority_tps)
                std_row['minority_fn'] = np.std(test_minority_fns)
                std_row['majority_fp'] = np.std(test_majority_fps)
                std_row['majority_tn'] = np.std(test_majority_tns)
            else:
                mean_row['minority_f1'] = 'N/A'
                mean_row['minority_recall'] = 'N/A'
                mean_row['minority_precision'] = 'N/A'
                mean_row['pred_minor_rate'] = 'N/A'
                mean_row['true_minor_rate'] = 'N/A'
                mean_row['minority_tp'] = 'N/A'
                mean_row['minority_fn'] = 'N/A'
                mean_row['majority_fp'] = 'N/A'
                mean_row['majority_tn'] = 'N/A'

                std_row['minority_f1'] = 'N/A'
                std_row['minority_recall'] = 'N/A'
                std_row['minority_precision'] = 'N/A'
                std_row['pred_minor_rate'] = 'N/A'
                std_row['true_minor_rate'] = 'N/A'
                std_row['minority_tp'] = 'N/A'
                std_row['minority_fn'] = 'N/A'
                std_row['majority_fp'] = 'N/A'
                std_row['majority_tn'] = 'N/A'

            test_summary_data.append(mean_row)
            test_summary_data.append(std_row)

            test_summary_df = pd.DataFrame(test_summary_data)
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                test_summary_df.to_excel(writer, sheet_name='Holdout_Test', index=False)

            print(f"\n[已添加] 测试集结果 → Excel 'Holdout_Test' sheet")

        else:
            print("\n[警告] 未收集到各 Fold 的测试集评估结果")

    return all_results, test_results


def _evaluate_single_fold_on_testset(config, model_class, device, fold_num, model_path, fold_stats,
                                      test_indices, dev_indices, all_data_cache,
                                      use_variable_length, use_static, is_multilabel, is_binary, minority_idx,
                                      random_seed):
    """
    评估单个 Fold 在测试集上的表现

    Args:
        config: Config 配置对象
        model_class: 模型类
        device: 设备
        fold_num: Fold 编号
        model_path: 模型权重路径
        fold_stats: 该 Fold 的训练集统计量 {'stats': ..., 'static_stats': ...}
        test_indices: 测试集索引
        dev_indices: Dev 集索引
        all_data_cache: 预加载的数据缓存
        use_variable_length: 是否使用变长序列
        use_static: 是否使用静态特征
        is_multilabel: 是否多标签模式
        is_binary: 是否二分类模式
        minority_idx: 二分类模式下的少数类索引
        random_seed: 随机种子

    Returns:
        dict: 测试集指标 {'accuracy', 'precision', 'recall', 'f1_score', 'auroc', 'auprc', 'macro_f1'}
    """
    from dataset_new import CPETDatasetNewKFold as CPETDatasetKFold

    train_stats = fold_stats.get('stats', None)
    train_static_stats = fold_stats.get('static_stats', None)

    # 创建测试集数据集
    test_dataset = CPETDatasetKFold(
        config, fold_idx=0, n_folds=1,
        phase="train", random_seed=random_seed,
        feature_indices=config.features.channels,
        use_variable_length=use_variable_length,
        max_length=config.data.max_length,
        use_static_features=use_static,
        dev_indices=dev_indices,
        test_indices=test_indices,
        all_data_cache=all_data_cache,
        use_holdout_test=True,
        train_stats=train_stats,
        train_static_stats=train_static_stats
    )

    # 获取实际类别数
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

    # 创建模型并加载权重
    model = _create_model_with_n_classes_for_kfold(config, model_class, device, n_classes, use_variable_length, is_binary=is_binary)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # 推理
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            num_elements = len(batch)

            # [新增] Known-T6 Context 模式处理
            t6_context = None  # 默认为 None

            if use_variable_length and use_static and num_elements == 5:
                # 变长模式 + 静态特征 + t6: (data, lengths, static_data, t6_context, labels)
                data, lengths, static_data, t6_context, labels = batch
                data, lengths, static_data, t6_context, labels = data.to(device), lengths.to(device), static_data.to(device), t6_context.to(device), labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data, t6_context=t6_context)
            elif use_variable_length and use_static and num_elements == 4:
                # 变长模式 + 静态特征: (data, lengths, static_data, labels)
                data, lengths, static_data, labels = batch
                data, lengths, static_data, labels = data.to(device), lengths.to(device), static_data.to(device), labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data)
            elif use_variable_length and num_elements == 4:
                # 变长模式 + t6: (data, lengths, t6_context, labels)
                data, lengths, t6_context, labels = batch
                data, lengths, t6_context, labels = data.to(device), lengths.to(device), t6_context.to(device), labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, t6_context=t6_context)
            elif use_variable_length and num_elements == 3:
                # 变长模式: (data, lengths, labels)
                data, lengths, labels = batch
                data, lengths, labels = data.to(device), lengths.to(device), labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None)
            elif num_elements == 4 and use_static:
                # 固定长度 + 静态 + t6: (data, static_data, t6_context, labels)
                data, static_data, t6_context, labels = batch
                data, static_data, t6_context, labels = data.to(device), static_data.to(device), t6_context.to(device), labels.to(device)
                outputs = model(data, static_x=static_data, t6_context=t6_context)
            elif num_elements == 3 and use_static:
                # 固定长度 + 静态: (data, static_data, labels)
                data, static_data, labels = batch
                static_data = static_data.to(device)
                data, labels = data.to(device), labels.to(device)
                outputs = model(data, static_x=static_data)
            elif num_elements == 3:
                # 固定长度 + t6: (data, t6_context, labels)
                data, t6_context, labels = batch
                data, t6_context, labels = data.to(device), t6_context.to(device), labels.to(device)
                outputs = model(data, t6_context=t6_context)
            else:
                # 固定长度模式: (data, labels)
                data, labels = batch
                static_data = None
                data, labels = data.to(device), labels.to(device)
                outputs = model(data, static_x=static_data)

            # [修复] 二分类和多分类的推理逻辑不同
            if is_binary:
                # 二分类模式: sigmoid + 阈值
                probs_positive = torch.sigmoid(outputs)  # [B, 1] - 少数类概率

                # 概率伪装为 [B, 2]，用于 AUROC/AUPRC 计算
                if minority_idx == 1:
                    probs = torch.cat([1 - probs_positive, probs_positive], dim=1)  # [B, 2]
                else:
                    # 少数类是索引0
                    probs = torch.cat([probs_positive, 1 - probs_positive], dim=1)  # [B, 2]

                # 预测标签: sigmoid > 0.5 → 少数类, 否则 → 多数类
                preds = torch.where(probs_positive.squeeze(1) > 0.5, minority_idx, 1 - minority_idx).long()
            else:
                # 多分类模式: softmax + argmax
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

    return metrics


def _create_model_with_n_classes_for_kfold(config, ModelClass, device, n_classes, use_var_length: bool, is_binary=False):
    """
    创建指定类别数的模型 (用于测试集评估)

    Args:
        config: Config 配置对象
        ModelClass: 模型类
        device: 设备
        n_classes: 类别数
        use_var_length: 是否使用变长序列
        is_binary: 是否为二分类模式

    Returns:
        model: 创建的模型
    """
    from feature_mapping import create_adjacency_matrix

    # 获取静态特征配置
    use_static = False
    static_dim = 16
    num_static_features = 5
    static_ablation = "full"  # 默认值
    if hasattr(config.model, 'static_features') and config.model.static_features is not None:
        use_static = config.model.static_features.enabled
        static_dim = config.model.static_features.static_dim
        num_static_features = config.model.static_features.num_features
        static_ablation = config.model.static_features.ablation  # [修复] 获取消融模式

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

    # [新增] 获取 Known-T6 Context 配置
    use_known_t6_context = False
    t6_n_classes = 0
    if hasattr(config, 'known_t6_context') and config.known_t6_context is not None:
        use_known_t6_context = config.known_t6_context.enabled
        # t6_n_classes 从 config 动态属性获取 (由数据集同步)
        t6_n_classes = getattr(config, 't6_n_classes', 0)

    model = ModelClass(
        input_dim=config.data.max_length if use_var_length else config.data.L_win,
        hidden_dim=config.model.hidden_dim,
        output_dim=n_classes,  # 使用传入的类别数
        channel_groups=config.features.channel_groups,
        num_channel=config.features.num_channels,
        D_time=config.model.D_time,
        dropout=config.model.dropout,
        semantic_adj=semantic_adj,
        use_static_features=use_static,
        static_dim=static_dim,
        num_static_features=num_static_features,
        static_ablation=static_ablation,  # [修复] 传递消融模式
        use_variable_length=use_var_length,
        graph_ablation=config.model.graph_ablation,
        temporal_encoder_type=temporal_encoder_type,
        T_mid=T_mid,
        temporal_encoder_cfg=temporal_encoder_cfg,
        gamma_init=gamma_init,
        gamma_min=gamma_min,
        is_binary=is_binary,  # [新增] 二分类模式
        # [新增] Known-T6 Context 参数
        use_known_t6_context=use_known_t6_context,
        t6_n_classes=t6_n_classes
    )

    return model


def _create_model_for_kfold(config, ModelClass, device, use_var_length: bool, is_binary=False):
    """
    为 K-Fold 创建模型 (复用 main_new._create_model 逻辑)

    Args:
        config: Config 配置对象
        ModelClass: 模型类
        device: 设备
        use_var_length: 是否使用变长序列
        is_binary: 是否为二分类模式

    Returns:
        model: 创建的模型
    """
    from feature_mapping import create_adjacency_matrix

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

    # 获取通道注意力配置
    use_channel_attention = False
    channel_attention_init = 1.0
    if hasattr(config.model, 'channel_attention') and config.model.channel_attention is not None:
        use_channel_attention = config.model.channel_attention.enabled
        channel_attention_init = config.model.channel_attention.init_value

    # [新增] 获取 Known-T6 Context 配置
    use_known_t6_context = False
    t6_n_classes = 0
    if hasattr(config, 'known_t6_context') and config.known_t6_context is not None:
        use_known_t6_context = config.known_t6_context.enabled
        # t6_n_classes 从 config 动态属性获取 (由数据集同步)
        t6_n_classes = getattr(config, 't6_n_classes', 0)

    if config.model.name == "HDSTGCN":
        print(f"\n创建 HDSTGCN 模型 (K-Fold):")
        print(f"  时序编码维度: {config.model.D_time}")
        print(f"  时序编码器: {temporal_encoder_type}" + (f" (T_mid={T_mid})" if temporal_encoder_type == "cnn" else ""))
        print(f"  Dropout: {config.model.dropout}")
        print(f"  变长模式: {use_var_length}")
        print(f"  图模式: {config.model.graph_ablation}")
        if config.model.graph_ablation == "prior_masked":
            print(f"  先验门控: gamma_init={gamma_init}, gamma_min={gamma_min}")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim}, ablation={static_ablation})")

        # [新增] 打印 Known-T6 Context 配置
        if use_known_t6_context and t6_n_classes > 0:
            print(f"  Known-T6 Context: 已启用 (t6_n_classes={t6_n_classes})")

        # [新增] 打印时序编码器消融模式
        temporal_ablation = getattr(temporal_encoder_cfg, 'ablation', 'full') if temporal_encoder_cfg else 'full'
        if temporal_encoder_type == "cnn" and temporal_ablation != "full":
            print(f"  时序编码器消融: {temporal_ablation}")

        # 获取邻接矩阵作为医学先验
        # [修复] 传递 optional_keys 以支持 vco2 和 o2pulse 特征
        optional_keys = []
        if hasattr(config.features, 'o2pulse_enabled') and config.features.o2pulse_enabled:
            optional_keys.append('o2pulse')
        if hasattr(config.features, 'vco2_enabled') and config.features.vco2_enabled:
            optional_keys.append('vco2')
        semantic_adj = create_adjacency_matrix(config.features.adapt_mode, optional_keys if optional_keys else None)

        model = ModelClass(
            input_dim=config.data.max_length if use_var_length else config.data.L_win,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.n_class,
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
            # [新增] 二分类模式参数
            is_binary=is_binary,
            # [新增] Known-T6 Context 参数
            use_known_t6_context=use_known_t6_context,
            t6_n_classes=t6_n_classes
        ).to(device)

    elif config.model.name == "STFinalNet":
        print(f"\n创建 STFinalNet 模型 (K-Fold):")
        print(f"  消融模式: {config.model.ablation}")
        print(f"  变量嵌入: {config.model.use_var_embedding} (dim={config.model.var_embed_dim})")
        print(f"  动态图: {config.model.use_dynamic_graph}")

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
            semantic_adj=semantic_adj
        ).to(device)

    elif config.model.name == "lstm":
        model = ModelClass(
            input_dim=config.data.L_win,
            output_dim=config.n_class,
            num_channel=config.features.num_channels,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers or 2
        ).to(device)

    elif config.model.name == "resnet":
        model = ModelClass(
            input_dim=config.data.L_win,
            output_dim=config.n_class,
            num_channel=config.features.num_channels,
            hidden_dim=config.model.hidden_dim
        ).to(device)

    elif config.model.name == "mednet":
        model = ModelClass(
            input_dim=config.data.L_win,
            hidden_dim=config.model.hidden_dim,
            channel_groups=config.features.channel_groups,
            output_dim=config.n_class,
            num_channel=config.features.num_channels
        ).to(device)

    elif config.model.name == "CNNGAF":
        # 获取 CNNGAF 特定配置
        image_size = getattr(config.model, 'image_size', 64)
        cnn_channels = getattr(config.model, 'cnn_channels', [16, 32])
        attention_dim = getattr(config.model, 'attention_dim', 16)

        print(f"\n创建 CNNGAF 模型 (K-Fold):")
        print(f"  GADF 图像大小: {image_size}x{image_size}")
        print(f"  CNN 通道: {cnn_channels}")
        print(f"  注意力维度: {attention_dim}")
        if use_static:
            print(f"  静态特征: 已启用 (dim={static_dim})")

        model = ModelClass(
            input_dim=config.data.L_win,  # CNNGAF 使用固定长度
            output_dim=config.n_class,
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

        print(f"\n创建 KESTNet 模型 (K-Fold):")
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
# train_with_swanlab.py

class FocalLoss(nn.Module):
    """
    [已废弃] Focal Loss - 强迫模型拟合难分样本

    ⚠️ 废弃原因：
    1. 在极度不平衡场景（阳性率 < 20%）强迫模型拟合"难分样本"
    2. 这些样本往往是带有严重伪影的脏数据，而非有价值的边界样本
    3. 在任务3（心电图，阳性率9.8%）和任务5（心率储备用尽，阳性率19.5%）
       中导致性能崩塌（验证集性能下降）

    推荐替代方案：
    - Dice Loss: 直接优化 F1-Score 重叠度（config.yaml: loss.type: "Dice")
    - LDAM Loss: 决策边界重塑（config.yaml: loss.type: "LDAM"）
    - BCEWithLogitsLoss + pos_weight: 基线方案

    原理（历史记录）:
    - 通过降低易分类样本的权重，聚焦于难分类样本
    - 公式: L = (1 - p_t)^gamma * CrossEntropy
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', debug_freq=0.01):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha 应当是一个 tensor，对应类别权重
        self.alpha = alpha
        self.debug_freq = debug_freq  # 调试输出频率
        self._debug_step = 0

    def forward(self, inputs, targets):
        # inputs: [N, C] (logits)
        # targets: [N] (class indices)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # [调试] 定期打印 Loss 分布信息
        self._debug_step += 1
        if self.training and (self._debug_step % max(1, int(1/self.debug_freq)) == 0):
            with torch.no_grad():
                probs = F.softmax(inputs, dim=1)
                max_probs = probs.max(dim=1)[0]
                # if self._debug_step % 1000== 0:
                #     print(f"\n[Loss调试 Step {self._debug_step}]")
                #     print(f"  ce_loss: [{ce_loss.min():.4f}, {ce_loss.max():.4f}], mean={ce_loss.mean():.4f}")
                #     print(f"  pt: [{pt.min():.4f}, {pt.max():.4f}], mean={pt.mean():.4f}")
                #     print(f"  logits: mean={inputs.mean():.4f}, std={inputs.std():.4f}, range=[{inputs.min():.2f}, {inputs.max():.2f}]")
                #     print(f"  max_prob: mean={max_probs.mean():.4f}, range=[{max_probs.min():.4f}, {max_probs.max():.4f}]")
                #     # 理论基准：-log(1/num_classes) ≈ 1.609 for 5-class
                #     num_classes = inputs.size(1)
                #     random_baseline = -np.log(1.0 / num_classes)
                #     print(f"  [基准] 5分类随机猜测 Loss ≈ {random_baseline:.3f}")

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss - 直接优化 F1-Score 的重叠度损失

    适用场景:
    - 极度不平衡的二分类任务 (阳性率 < 20%)
    - 医学图像分割、罕见疾病检测

    核心公式:
    Dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    Loss = 1 - Dice

    Args:
        smooth: 平滑系数，防止除零 (默认 1.0)
        reduction: 归约方式，'mean', 'sum', 'none' (默认 'mean')
    """
    def __init__(self, smooth=1.0, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1] (二分类 sigmoid 输出) 或 [B, C] (多分类 softmax 输出)
            targets: [B] (标签索引)

        Returns:
            loss: Dice 损失
        """
        # 二分类模式: logits [B, 1] 或 [B, 2]
        if logits.dim() == 2 and logits.size(1) == 1:
            # [B, 1] 格式 - 二分类 sigmoid 输出
            probs = torch.sigmoid(logits)
            targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()

            intersection = (probs * targets).sum()
            union = probs.sum() + targets.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)

            return 1. - dice
        else:
            # 多分类模式: logits [B, C], targets [B]
            probs = F.softmax(logits, dim=1)
            targets_oh = F.one_hot(targets, logits.size(1)).float()

            intersection = (probs * targets_oh).sum(dim=0)
            union = probs.sum(dim=0) + targets_oh.sum(dim=0)
            dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)

            if self.reduction == 'mean':
                return 1. - dice_per_class.mean()
            elif self.reduction == 'sum':
                return 1. - dice_per_class.sum()
            else:
                return 1. - dice_per_class


class UnifiedLDAMLoss(nn.Module):
    """
    统一版 LDAM Loss，自动路由 [B, 1] 二分类和 [B, C] 多分类任务。

    通过检测 logits.size(1) 自动决定应用基于 Sigmoid 的二分类 Margin 惩罚，
    还是基于 Softmax 的多分类 Margin 惩罚。

    核心思想: 为少数类施加更大的 margin，要求更高的置信度才能预测为该类，
    从而重塑决策边界，改善长尾分类问题。

    Margin 公式:
    m_c = C / sqrt(sqrt(n_c))  # n_c 为类别 c 的样本数

    二分类公式:
    - target=1 (少数类): logits - m1 (要求更高置信度)
    - target=0 (多数类): logits + m0 (放宽要求)

    多分类公式:
    logits' = logits - m_c (仅对正确类别)
    Loss = CrossEntropy(scale * logits')

    Args:
        cls_num_list: 各类别样本数列表 [n_0, n_1, ..., n_C]
        max_m: 最大 margin 上限 (默认 0.5)
        s: 缩放因子 (默认 30)
        weight: 二分类为 pos_weight，多分类为 class_weights
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None):
        super().__init__()
        # 计算基于频率的 margin
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)

        self.s = s  # scale 参数
        self.weight = weight  # 二分类为 pos_weight，多分类为 class_weights

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, 1] (二分类) 或 [B, C] (多分类)，也兼容 [B] (会自动 unsqueeze)
            targets: [B] 或 [B, 1] (标签索引或二分类标签)

        Returns:
            loss: LDAM 损失
        """
        # 确保 m_list 在正确设备上
        self.m_list = self.m_list.to(logits.device)

        # [兼容性修复] 如果 logits 是 1D [B]，unsqueeze 成 [B, 1]
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)

        # [兼容性修复] 如果 targets 是 2D [B, 1]，squeeze 成 [B]
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(-1)

        # === 路由 1：二分类模式 [B, 1] ===
        if logits.size(1) == 1:
            # 确保 targets 形状对齐 [B, 1] 且为 float
            targets = targets.float().view(-1, 1)

            m0 = self.m_list[0]  # 多数类 margin
            m1 = self.m_list[1]  # 少数类 margin

            # target=1 时，logit 减 m1 (要求更高置信度)
            # target=0 时，logit 加 m0 (放宽要求)
            margins = torch.where(targets == 1.0, -m1, m0)
            logits_m = logits + margins

            # 计算缩放后的 BCEWithLogits
            return F.binary_cross_entropy_with_logits(
                logits_m * self.s,
                targets,
                pos_weight=self.weight
            )

        # === 路由 2：多分类模式 [B, C] ===
        else:
            # 确保 targets 为 1D long tensor
            targets = targets.long().view(-1)

            # 构造 one-hot index
            index = torch.zeros_like(logits, dtype=torch.bool)
            index.scatter_(1, targets.view(-1, 1), True)

            # 获取对应 batch 的 margin
            index_float = index.float()
            batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
            batch_m = batch_m.view((-1, 1))

            # 仅在真实类别的 logit 上减去 margin
            x_m = logits - batch_m
            output = torch.where(index, x_m, logits)

            # 计算缩放后的 CrossEntropy
            return F.cross_entropy(
                output * self.s,
                targets,
                weight=self.weight
            )


class SupConLoss(nn.Module):
    """
    有监督对比学习损失 (Supervised Contrastive Loss)

    参考: "Supervised Contrastive Learning" (Khosla et al., 2020)

    核心思想: 拉近同类样本，推远异类样本

    Args:
        temperature: 温度参数 (默认 0.07)
        base_temperature: 基准温度
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] 特征向量 (已归一化)
            labels: [B] 标签

        Returns:
            loss: 对比学习损失
        """
        device = features.device
        batch_size = features.shape[0]

        # 归一化特征
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵: [B, B]
        similarity = torch.matmul(features, features.T) / self.temperature

        # 构建正样本掩码 (同类样本)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 移除对角线 (自身不作为正样本)
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask * (1 - mask_self)  # 正样本掩码 (排除自身)

        # 计算损失
        # 对于每个样本，计算与其正样本的对比损失
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()  # 数值稳定性

        # 分母: exp(logit) 对所有样本求和 (排除自身)
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * (1 - mask_self)  # 排除自身
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # 只计算有正样本的情况
        mask_pos_sum = mask_pos.sum(dim=1)
        valid_mask = mask_pos_sum > 0  # 有正样本的样本

        if valid_mask.sum() == 0:
            # 没有正样本对，返回 0
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 计算每个样本的平均对比损失
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (mask_pos_sum + 1e-8)

        # 只计算有效样本
        loss = - (self.base_temperature / self.temperature) * mean_log_prob[valid_mask].mean()

        return loss


class CombinedLoss(nn.Module):
    """
    组合损失: CrossEntropy/FocalLoss + SupConLoss

    Args:
        ce_weight: CrossEntropy/FocalLoss 权重
        supcon_weight: SupConLoss 权重
        temperature: SupCon 温度参数
        class_weights: 类别权重 (用于 CE/Focal)
        gamma: Focal loss gamma 参数
    """
    def __init__(self, ce_weight=1.0, supcon_weight=0.3, temperature=0.07,
                 class_weights=None, gamma=1.5, loss_type="CombinedLoss"):
        super().__init__()
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        self.loss_type = loss_type

        # CE/Focal Loss
        if loss_type == "FocalLoss":
            self.ce_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        # SupCon Loss
        self.supcon_loss = SupConLoss(temperature=temperature)

    def forward(self, outputs, labels, features=None):
        """
        Args:
            outputs: [B, C] 模型输出 logits
            labels: [B] 标签
            features: [B, D] 特征向量 (用于 SupCon)，如果为 None 则只用 CE

        Returns:
            loss: 组合损失
            loss_dict: 损失分解字典
        """
        # CE/Focal Loss
        loss_ce = self.ce_loss(outputs, labels)

        # SupCon Loss (如果提供了特征)
        if features is not None and self.supcon_weight > 0:
            loss_supcon = self.supcon_loss(features, labels)
        else:
            loss_supcon = torch.tensor(0.0, device=outputs.device)

        # 组合损失
        total_loss = self.ce_weight * loss_ce + self.supcon_weight * loss_supcon

        # 返回损失分解 (用于 SwanLab 记录)
        loss_dict = {
            'loss_ce': loss_ce.item(),
            'loss_supcon': loss_supcon.item() if isinstance(loss_supcon, torch.Tensor) else loss_supcon,
            'loss_total': total_loss.item()
        }

        return total_loss, loss_dict


# =============================================================================
# LCRLoss - 多标签分类损失 (新增)
# =============================================================================

class LCRLoss(nn.Module):
    """
    Label Correlation Regularized Loss (标签共现正则化损失)

    L_LCR = L_weight + λ * L_co

    L_weight: 加权二元交叉熵 (动态频率权重)
    L_co: 共现正则化损失

    Args:
        co_occurrence_matrix: [n_labels, n_labels] 归一化共现矩阵
        lambda_co: 共现正则化权重 (默认 0.1)
        epsilon: 数值稳定性常数 (默认 1e-6)
        auto_balance_lambda: 是否自动平衡 lambda (当量级差异>100x时调整)
    """

    def __init__(self, co_occurrence_matrix=None, lambda_co=0.1, epsilon=1e-6, auto_balance_lambda=True):
        super().__init__()
        self.lambda_co = lambda_co
        self.epsilon = epsilon
        self.auto_balance_lambda = auto_balance_lambda

        # 梯度量级对齐监控
        self.raw_loss_weight_history = []  # 记录最近 N 个 batch 的 loss_weight
        self.raw_loss_co_history = []      # 记录最近 N 个 batch 的 loss_co
        self.history_size = 100            # 历史窗口大小

        # 注册共现矩阵为 buffer (不参与梯度计算，但随模型移动到设备)
        if co_occurrence_matrix is not None:
            self.register_buffer('co_matrix', torch.from_numpy(co_occurrence_matrix).float())
        else:
            self.co_matrix = None

    def compute_dynamic_weights(self, targets):
        """
        动态计算正负样本权重

        Args:
            targets: [B, C] multi-hot 标签矩阵

        Returns:
            pos_weights: [C] 正样本权重
            neg_weights: [C] 负样本权重

        公式:
            q_c^pos = 1 / sqrt(P_c/N + ε)  # P_c: 正样本数
            q_c^neg = 1 / sqrt(N_c/N + ε)  # N_c: 负样本数
        """
        B, C = targets.shape
        pos_count = targets.sum(dim=0).clamp(min=1)  # 避免除零
        neg_count = (B - pos_count).clamp(min=1)

        pos_weights = 1.0 / torch.sqrt(pos_count / B + self.epsilon)
        neg_weights = 1.0 / torch.sqrt(neg_count / B + self.epsilon)

        return pos_weights, neg_weights

    def weighted_bce_loss(self, logits, targets):
        """
        加权二元交叉熵损失

        Args:
            logits: [B, C] 模型输出 (未经过 sigmoid)
            targets: [B, C] multi-hot 标签矩阵

        Returns:
            loss: 加权 BCE 损失
        """
        pos_w, neg_w = self.compute_dynamic_weights(targets)

        # 数值稳定的 sigmoid
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.epsilon, max=1 - self.epsilon)

        # 加权 BCE
        pos_loss = pos_w * targets * torch.log(probs)
        neg_loss = neg_w * (1 - targets) * torch.log(1 - probs)

        return -(pos_loss + neg_loss).mean()

    def co_occurrence_regularization(self, probs):
        """
        共现正则化损失

        L_co = Σ_i Σ_j co_matrix[i,j] * ||ŷ_i - ŷ_j||²

        Args:
            probs: [B, C] sigmoid 后的概率

        Returns:
            loss: 共现正则化损失 (已归一化)
        """
        if self.co_matrix is None:
            return torch.tensor(0.0, device=probs.device)

        B, C = probs.shape

        # 计算标签对之间的差异: [B, C, 1] - [B, 1, C] = [B, C, C]
        diff_sq = (probs.unsqueeze(2) - probs.unsqueeze(1)) ** 2

        # 加权求和: co_matrix [C, C] 广播到 [B, C, C]
        # [修复] 添加归一化, 避免随 batch 和类别数累积
        return (self.co_matrix.unsqueeze(0) * diff_sq).sum() / (B * C * C)

    def forward(self, logits, targets):
        """
        前向传播

        Args:
            logits: [B, C] 模型输出
            targets: [B, C] multi-hot 标签

        Returns:
            total_loss: 总损失
            loss_dict: 损失分解字典
        """
        # 加权 BCE (原始值, 用于监控)
        loss_weight = self.weighted_bce_loss(logits, targets)

        # 共现正则化 (原始值, 用于监控)
        probs = torch.sigmoid(logits)
        loss_co = self.co_occurrence_regularization(probs)

        # 记录原始值用于梯度量级对齐监控
        raw_loss_weight = loss_weight.item()
        raw_loss_co = loss_co.item() if isinstance(loss_co, torch.Tensor) else loss_co

        # 更新历史记录
        self.raw_loss_weight_history.append(raw_loss_weight)
        self.raw_loss_co_history.append(raw_loss_co)
        if len(self.raw_loss_weight_history) > self.history_size:
            self.raw_loss_weight_history.pop(0)
            self.raw_loss_co_history.pop(0)

        # 动态 lambda 调整 (当量级差异 > 100x 时自动调整)
        effective_lambda = self.lambda_co
        magnitude_ratio = 1.0
        if self.auto_balance_lambda and len(self.raw_loss_weight_history) >= 10:
            avg_weight = sum(self.raw_loss_weight_history[-10:]) / 10
            avg_co = sum(self.raw_loss_co_history[-10:]) / 10
            if avg_co > 1e-8:  # 避免除零
                magnitude_ratio = avg_weight / avg_co
                if magnitude_ratio > 100:  # loss_weight 比 loss_co 大两个数量级以上
                    effective_lambda = self.lambda_co * (magnitude_ratio / 10)  # 放大 lambda
                elif magnitude_ratio < 0.01:  # loss_co 比 loss_weight 大两个数量级以上
                    effective_lambda = self.lambda_co * (magnitude_ratio * 10)  # 缩小 lambda

        # 总损失
        total_loss = loss_weight + effective_lambda * loss_co

        loss_dict = {
            'loss_weight': raw_loss_weight,
            'loss_co': raw_loss_co,
            'loss_total': total_loss.item(),
            'effective_lambda': effective_lambda,
            'magnitude_ratio': magnitude_ratio if self.auto_balance_lambda else 1.0
        }

        return total_loss, loss_dict


# =============================================================================
# 多标签评估指标 (新增)
# =============================================================================

def find_optimal_thresholds(probs, labels):
    """
    为每个类别找到最优阈值 (基于验证集 F1)

    Args:
        probs: [N, C] 概率矩阵
        labels: [N, C] 标签矩阵

    Returns:
        thresholds: [C] 每个类别的最优阈值
    """
    from sklearn.metrics import f1_score

    n_classes = probs.shape[1]
    thresholds = []

    for c in range(n_classes):
        best_th, best_f1 = 0.5, 0.0
        for th in np.arange(0.1, 0.9, 0.05):
            preds_c = (probs[:, c] >= th).astype(int)
            f1 = f1_score(labels[:, c], preds_c, zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, th
        thresholds.append(best_th)

    return np.array(thresholds)


def apply_prediction_with_fallback(probs, thresholds):
    """
    应用阈值预测，当所有概率低于阈值时使用 Argmax 回退

    回退逻辑:
    1. 先用阈值生成二值预测
    2. 如果某样本所有预测都为 0 (全低于阈值), 使用 Argmax 选最大概率类别
    3. 保证每个样本至少有一个预测标签

    Args:
        probs: [N, C] 概率矩阵
        thresholds: [C] 每个类别的阈值

    Returns:
        preds_binary: [N, C] 二值预测矩阵
        fallback_count: 使用 Argmax 回退的样本数
    """
    # 1. 阈值二值化
    preds_binary = (probs >= thresholds).astype(int)

    # 2. 检查全零预测
    all_zero_mask = preds_binary.sum(axis=1) == 0
    fallback_count = all_zero_mask.sum()

    # 3. 对全零预测使用 Argmax 回退
    if fallback_count > 0:
        # 找到每个全零样本的最大概率类别
        max_prob_indices = probs[all_zero_mask].argmax(axis=1)
        # 设置该类别为 1
        for i, idx in enumerate(max_prob_indices):
            original_idx = np.where(all_zero_mask)[0][i]
            preds_binary[original_idx, idx] = 1

    return preds_binary, fallback_count


def print_multilabel_results(all_probs, all_labels, label_names=None, co_occurrence_matrix=None, threshold=0.5):
    """
    打印多标签分类结果矩阵

    Args:
        all_probs: [N, C] 预测概率矩阵
        all_labels: [N, C] 真实标签矩阵
        label_names: 标签名称列表
        co_occurrence_matrix: [C, C] 标签共现矩阵
        threshold: 二值化阈值
    """
    from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score, precision_score, recall_score

    n_samples, n_classes = all_labels.shape

    # 生成默认标签名称
    if label_names is None:
        label_names = [f"Class_{i}" for i in range(n_classes)]

    print("\n" + "="*80)
    print("多标签分类结果矩阵")
    print("="*80)

    # 1. 标签频率统计
    print("\n【1】标签频率统计")
    print("-"*60)
    label_counts = all_labels.sum(axis=0).astype(int)
    pred_counts = (all_probs >= threshold).sum(axis=0).astype(int)

    print(f"{'标签名称':<30} {'真实频次':>10} {'预测频次':>10} {'频率':>10}")
    print("-"*60)
    for i, name in enumerate(label_names):
        freq = label_counts[i] / n_samples * 100
        print(f"{name:<30} {label_counts[i]:>10} {pred_counts[i]:>10} {freq:>9.1f}%")
    print("-"*60)
    print(f"{'总计':<30} {label_counts.sum():>10} {pred_counts.sum():>10}")

    # 2. 每个类别的详细指标
    print("\n【2】各类别详细指标")
    print("-"*80)
    preds_binary = (all_probs >= threshold).astype(int)

    # 计算每个类别的 P, R, F1
    precision = precision_score(all_labels, preds_binary, average=None, zero_division=0)
    recall = recall_score(all_labels, preds_binary, average=None, zero_division=0)
    f1 = f1_score(all_labels, preds_binary, average=None, zero_division=0)

    print(f"{'标签名称':<25} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-"*80)
    for i, name in enumerate(label_names):
        support = int(label_counts[i])
        print(f"{name:<25} {precision[i]:>12.4f} {recall[i]:>12.4f} {f1[i]:>12.4f} {support:>10}")
    print("-"*80)

    # 宏平均和微平均
    macro_p = precision_score(all_labels, preds_binary, average='macro', zero_division=0)
    macro_r = recall_score(all_labels, preds_binary, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, preds_binary, average='macro', zero_division=0)
    micro_p = precision_score(all_labels, preds_binary, average='micro', zero_division=0)
    micro_r = recall_score(all_labels, preds_binary, average='micro', zero_division=0)
    micro_f1 = f1_score(all_labels, preds_binary, average='micro', zero_division=0)

    print(f"{'Macro Average':<25} {macro_p:>12.4f} {macro_r:>12.4f} {macro_f1:>12.4f} {n_samples:>10}")
    print(f"{'Micro Average':<25} {micro_p:>12.4f} {micro_r:>12.4f} {micro_f1:>12.4f} {n_samples:>10}")

    # 3. 多标签混淆矩阵
    print("\n【3】多标签混淆矩阵 (每个类别)")
    print("-"*60)
    mcm = multilabel_confusion_matrix(all_labels, preds_binary)

    for i, name in enumerate(label_names):
        tn, fp, mcm_fn, tp = mcm[i].ravel()
        print(f"\n[{name}]")
        print(f"  真负例(TN): {tn:>6}  假正例(FP): {fp:>6}")
        print(f"  假负例(FN): {mcm_fn:>6}  真正例(TP): {tp:>6}")

    # 4. 标签共现矩阵（如果提供）
    if co_occurrence_matrix is not None:
        print("\n【4】标签共现矩阵 P(label_j=1 | label_i=1)")
        print("-"*80)

        # 打印表头
        header = "标签\\标签".ljust(15)
        for name in label_names:
            short_name = name[:8] if len(name) > 8 else name
            header += f"{short_name:>10}"
        print(header)
        print("-"*80)

        for i, name in enumerate(label_names):
            row_name = name[:12] if len(name) > 12 else name
            row = f"{row_name:<15}"
            for j in range(n_classes):
                row += f"{co_occurrence_matrix[i, j]:>10.3f}"
            print(row)

    # 5. 样本级预测分布
    print("\n【5】样本级标签数量分布")
    print("-"*60)
    true_label_counts = all_labels.sum(axis=1).astype(int)
    pred_label_counts = preds_binary.sum(axis=1).astype(int)

    max_labels = max(true_label_counts.max(), pred_label_counts.max()) + 1
    true_dist = np.bincount(true_label_counts, minlength=max_labels)
    pred_dist = np.bincount(pred_label_counts, minlength=max_labels)

    print(f"{'标签数':<10} {'真实样本数':>15} {'预测样本数':>15}")
    print("-"*60)
    for k in range(max_labels):
        if true_dist[k] > 0 or pred_dist[k] > 0:
            print(f"{k:<10} {true_dist[k]:>15} {pred_dist[k]:>15}")

    # 6. 标签组合统计（Top 10）
    print("\n【6】常见标签组合 (Top 10)")
    print("-"*60)

    # 将标签矩阵转换为标签名组合
    from collections import Counter
    true_combos = []
    for i in range(n_samples):
        combo = tuple(label_names[j] for j in range(n_classes) if all_labels[i, j] == 1)
        true_combos.append(combo if combo else ("无标签",))

    combo_counts = Counter(true_combos)
    for combo, count in combo_counts.most_common(10):
        combo_str = " + ".join(combo) if len(combo) <= 3 else f"{combo[0]} + ... ({len(combo)}个标签)"
        print(f"  {combo_str:<50} {count:>5} 次 ({count/n_samples*100:.1f}%)")

    print("\n" + "="*80)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1
    }


def compute_multilabel_metrics(preds, labels, threshold=0.5, adaptive_thresholds=None):
    """
    计算多标签分类指标

    Args:
        preds: [N, C] 模型输出概率 (sigmoid 后)
        labels: [N, C] multi-hot 标签
        threshold: 二值化阈值 (当 adaptive_thresholds 为 None 时使用)
        adaptive_thresholds: [C] 每个类别的最优阈值 (可选)

    Returns:
        metrics: 包含各项指标的字典
            - macro_f1: 宏平均 F1
            - micro_f1: 微平均 F1
            - mAP: 平均精度均值
            - hamming_loss: 汉明损失
            - subset_accuracy: 子集准确率
            - jaccard: Jaccard 相似度
            - fallback_count: 使用 Argmax 回退的样本数
    """
    from sklearn.metrics import f1_score, average_precision_score, hamming_loss as sk_hamming_loss, jaccard_score

    # 使用自适应阈值或固定阈值，并应用回退逻辑
    if adaptive_thresholds is not None:
        preds_binary, fallback_count = apply_prediction_with_fallback(preds, adaptive_thresholds)
    else:
        # 固定阈值也需要回退逻辑
        thresholds = np.full(preds.shape[1], threshold)
        preds_binary, fallback_count = apply_prediction_with_fallback(preds, thresholds)

    metrics = {
        'macro_f1': f1_score(labels, preds_binary, average='macro', zero_division=0),
        'micro_f1': f1_score(labels, preds_binary, average='micro', zero_division=0),
        'hamming_loss': sk_hamming_loss(labels, preds_binary),
        'subset_accuracy': (preds_binary == labels).all(axis=1).mean(),
        'jaccard': jaccard_score(labels, preds_binary, average='samples', zero_division=0),
        'fallback_count': int(fallback_count)
    }

    # mAP (需要概率值，不是二值化后的)
    try:
        metrics['mAP'] = average_precision_score(labels, preds, average='macro')
    except ValueError:
        # 如果某些标签没有正样本，跳过 mAP
        metrics['mAP'] = 0.0

    return metrics


def train_epoch_multilabel(model, train_loader, optimizer, criterion, device, adj=None, epoch=0, config=None):
    """
    多标签训练 epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: LCRLoss 或 BCEWithLogitsLoss
        device: 设备
        adj: 邻接矩阵 (HDSTGCN 不使用)
        epoch: 当前 epoch
        config: Config 配置对象

    Returns:
        mean_loss: 平均损失
        loss_dict_avg: 平均损失分解
    """
    model.train()
    running_loss = 0.0

    use_variable_length = _get_config_value(config, 'model.use_variable_length', False)
    use_static = _get_config_value(config, 'model.static_features.enabled', False)
    gradient_clip = _get_config_value(config, 'training.gradient_clip', 1.0)

    # 获取模型名称，用于区分 HDSTGCN 和 STFinalNet
    model_name = _get_config_value(config, 'model.name', 'HDSTGCN')
    is_hdstgcn = model_name == 'HDSTGCN'

    # 损失分解累计
    loss_dict_sum = {
        'loss_weight': 0.0,
        'loss_co': 0.0,
        'loss_total': 0.0,
        'effective_lambda': 0.0,
        'magnitude_ratio': 0.0
    }
    use_lcr_loss = isinstance(criterion, LCRLoss)

    for batch_idx, batch in enumerate(train_loader):
        num_elements = len(batch)

        # [新增] Known-T6 Context 模式处理
        t6_context = None  # 默认为 None

        if use_variable_length and use_static and num_elements == 5:
            # 变长模式 + 静态特征 + t6: (data, lengths, static_data, t6_context, labels)
            data, lengths, static_data, t6_context, labels = batch
            data = data.to(device)
            lengths = lengths.to(device)
            static_data = static_data.to(device)
            t6_context = t6_context.to(device)  # [新增]
            labels = labels.to(device)
        elif use_variable_length and use_static and num_elements == 4:
            # 变长模式 + 静态特征: (data, lengths, static_data, labels)
            data, lengths, static_data, labels = batch
            data = data.to(device)
            lengths = lengths.to(device)
            static_data = static_data.to(device)
            labels = labels.to(device)
        elif use_variable_length and num_elements == 4:
            # 变长模式 + t6: (data, lengths, t6_context, labels)
            data, lengths, t6_context, labels = batch
            data = data.to(device)
            lengths = lengths.to(device)
            t6_context = t6_context.to(device)  # [新增]
            labels = labels.to(device)
            static_data = None
        elif use_variable_length and num_elements == 3:
            # 变长模式: (data, lengths, labels)
            data, lengths, labels = batch
            data = data.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            static_data = None
        elif num_elements == 4 and use_static:
            # 固定长度 + 静态 + t6: (data, static_data, t6_context, labels)
            data, static_data, t6_context, labels = batch
            data = data.to(device)
            static_data = static_data.to(device)
            t6_context = t6_context.to(device)  # [新增]
            labels = labels.to(device)
        elif num_elements == 3 and use_static:
            # 固定长度 + 静态: (data, static_data, labels)
            data, static_data, labels = batch
            static_data = static_data.to(device)
            data = data.to(device)
            labels = labels.to(device)
        elif num_elements == 3:
            # 固定长度 + t6: (data, t6_context, labels)
            data, t6_context, labels = batch
            data = data.to(device)
            t6_context = t6_context.to(device)  # [新增]
            labels = labels.to(device)
            static_data = None
        else:
            # 固定长度模式: (data, labels)
            data, labels = batch
            static_data = None
            data = data.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()

        # 前向传播
        if use_variable_length:
            outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data, t6_context=t6_context)
        elif adj is not None and not is_hdstgcn:
            # STFinalNet 固定长度模式: 使用邻接矩阵作为位置参数
            adj_temp = adj.to(device)
            outputs = model(data, adj_temp, static_x=static_data, t6_context=t6_context)
        else:
            # HDSTGCN 固定长度模式: 不需要传递 adj 和 lengths
            outputs = model(data, static_x=static_data, t6_context=t6_context)

        # 计算损失
        if use_lcr_loss:
            loss, loss_dict = criterion(outputs, labels)
            for k, v in loss_dict.items():
                loss_dict_sum[k] += v
        else:
            # BCEWithLogitsLoss
            loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        running_loss += loss.item()

    mean_loss = running_loss / len(train_loader)
    loss_dict_avg = {k: v / len(train_loader) for k, v in loss_dict_sum.items()} if use_lcr_loss else None

    return mean_loss, loss_dict_avg


def validate_epoch_multilabel(model, val_loader, criterion, device, adj=None, config=None):
    """
    多标签验证 epoch

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        adj: 邻接矩阵
        config: Config 配置对象

    Returns:
        mean_loss: 平均损失
        metrics: 多标签评估指标
        all_probs: 所有预测概率 [N, C]
        all_labels: 所有标签 [N, C]
    """
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    use_variable_length = _get_config_value(config, 'model.use_variable_length', False)
    use_static = _get_config_value(config, 'model.static_features.enabled', False)
    use_lcr_loss = isinstance(criterion, LCRLoss)

    # 获取模型名称，用于区分 HDSTGCN 和 STFinalNet
    model_name = _get_config_value(config, 'model.name', 'HDSTGCN')
    is_hdstgcn = model_name == 'HDSTGCN'

    with torch.no_grad():
        for batch in val_loader:
            num_elements = len(batch)

            # [新增] Known-T6 Context 模式处理
            t6_context = None  # 默认为 None

            if use_variable_length and use_static and num_elements == 5:
                # 变长模式 + 静态特征 + t6: (data, lengths, static_data, t6_context, labels)
                data, lengths, static_data, t6_context, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                static_data = static_data.to(device)
                t6_context = t6_context.to(device)  # [新增]
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data, t6_context=t6_context)
            elif use_variable_length and use_static and num_elements == 4:
                # 变长模式 + 静态特征: (data, lengths, static_data, labels)
                data, lengths, static_data, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                static_data = static_data.to(device)
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, static_x=static_data)
            elif use_variable_length and num_elements == 4:
                # 变长模式 + t6: (data, lengths, t6_context, labels)
                data, lengths, t6_context, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                t6_context = t6_context.to(device)  # [新增]
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None, t6_context=t6_context)
            elif use_variable_length and num_elements == 3:
                # 变长模式: (data, lengths, labels)
                data, lengths, labels = batch
                data = data.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                outputs = model(data, lengths=lengths, prior_adj=None)
            elif num_elements == 4 and use_static:
                # 固定长度 + 静态 + t6: (data, static_data, t6_context, labels)
                data, static_data, t6_context, labels = batch
                data = data.to(device)
                static_data = static_data.to(device)
                t6_context = t6_context.to(device)  # [新增]
                labels = labels.to(device)
                if adj is not None and not is_hdstgcn:
                    adj_temp = adj.to(device)
                    outputs = model(data, adj_temp, static_x=static_data, t6_context=t6_context)
                else:
                    outputs = model(data, static_x=static_data, t6_context=t6_context)
            elif num_elements == 3 and use_static:
                # 固定长度 + 静态: (data, static_data, labels)
                data, static_data, labels = batch
                data = data.to(device)
                static_data = static_data.to(device)
                labels = labels.to(device)
                if adj is not None and not is_hdstgcn:
                    adj_temp = adj.to(device)
                    outputs = model(data, adj_temp, static_x=static_data)
                else:
                    outputs = model(data, static_x=static_data)
            elif num_elements == 3:
                # 固定长度 + t6: (data, t6_context, labels)
                data, t6_context, labels = batch
                data = data.to(device)
                t6_context = t6_context.to(device)  # [新增]
                labels = labels.to(device)
                outputs = model(data, t6_context=t6_context)
            else:
                # 固定长度模式: (data, labels)
                data, labels = batch
                data = data.to(device)
                labels = labels.to(device)
                static_data = None

                if adj is not None and not is_hdstgcn:
                    # STFinalNet 固定长度模式: 使用邻接矩阵作为位置参数
                    adj_temp = adj.to(device)
                    outputs = model(data, adj_temp, static_x=static_data)
                else:
                    # HDSTGCN 固定长度模式: 不需要传递 adj 和 lengths
                    outputs = model(data, static_x=static_data)

            # 计算损失
            if use_lcr_loss:
                loss, _ = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            running_loss += loss.item()

            # 收集预测和标签
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    mean_loss = running_loss / len(val_loader)

    # 合并所有批次
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # [修复] 使用自适应阈值计算指标
    optimal_thresholds = find_optimal_thresholds(all_probs, all_labels)
    metrics = compute_multilabel_metrics(all_probs, all_labels, adaptive_thresholds=optimal_thresholds)
    metrics['optimal_thresholds'] = optimal_thresholds.tolist()  # 记录阈值

    return mean_loss, metrics, all_probs, all_labels


# 检验逻辑
def check_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            if torch.isnan(param_norm):
                print("!!! CRITICAL: NaN Gradient detected !!!")
    total_norm = total_norm ** 0.5
    write_debug_log(f"Gradient Norm: {total_norm:.4f}")

def detect_minority_class_index(dataset):
    """
    动态检测少数类索引 (样本数最少的类别)

    标签映射按字母顺序排序，而非按频次排序，因此需要动态检测少数类。

    Args:
        dataset: 数据集对象 (需包含 labellist 和 label_mapping)

    Returns:
        minority_idx: 少数类索引 (样本数最少的类别)
        majority_idx: 多数类索引 (样本数最多的类别)
        samples_per_cls: 各类别样本数列表 [idx0_count, idx1_count]
    """
    from collections import Counter

    label_counts = Counter(dataset.labellist)
    sorted_labels = sorted(dataset.label_mapping.items(), key=lambda x: x[1])
    samples_per_cls = [label_counts.get(name, 0) for name, _ in sorted_labels]

    if len(samples_per_cls) != 2:
        return None, None, samples_per_cls

    # 找出样本数较少的索引作为少数类
    if samples_per_cls[0] < samples_per_cls[1]:
        minority_idx = 0
        majority_idx = 1
    else:
        minority_idx = 1
        majority_idx = 0

    print(f"[二分类检测] 索引0 ({sorted_labels[0][0]}): {samples_per_cls[0]}样本")
    print(f"[二分类检测] 索引1 ({sorted_labels[1][0]}): {samples_per_cls[1]}样本")
    print(f"[二分类检测] 少数类索引={minority_idx} ({sorted_labels[minority_idx][0]}), 多数类索引={majority_idx}")

    return minority_idx, majority_idx, samples_per_cls


def compute_binary_pos_weight(dataset, device):
    """
    计算二分类 pos_weight = n_negative / n_positive

    用于 BCEWithLogitsLoss，平衡少数类和多数类的损失权重。

    Args:
        dataset: 数据集对象
        device: 设备

    Returns:
        pos_weight: 正类权重张量 [1]
    """
    minority_idx, majority_idx, samples_per_cls = detect_minority_class_index(dataset)

    if minority_idx is None:
        print("[二分类权重] 非二分类场景，使用默认权重 1.0")
        return torch.tensor([1.0], dtype=torch.float).to(device)

    n_positive = samples_per_cls[minority_idx]  # 少数类
    n_negative = samples_per_cls[majority_idx]  # 多数类
    pos_weight_val = n_negative / max(n_positive, 1)

    print(f"[二分类权重] 少数类(索引{minority_idx})={n_positive}, 多数类(索引{majority_idx})={n_negative}")
    print(f"[二分类权重] pos_weight={pos_weight_val:.4f} (负类/正类)")

    return torch.tensor([pos_weight_val], dtype=torch.float).to(device)


def extract_weights_from_trainset(dataset):
    """
    [改进] 基于原始样本分布计算类别权重
    过采样后各类别样本数相同，权重全为1.0会失去平衡效果
    使用原始分布计算权重，保留类别不平衡信息
    """
    from collections import Counter
    import math

    # 优先使用原始分布（过采样前）
    if hasattr(dataset, 'original_counts') and dataset.original_counts:
        samples_per_cls = dataset.original_counts
        print("[权重计算] 使用原始样本分布（过采样前）")
    else:
        # 回退到当前分布
        label_counts = Counter(dataset.labellist)
        sorted_labels = sorted(dataset.label_mapping.items(), key=lambda item: item[1])
        samples_per_cls = [label_counts.get(label_name, 0) for label_name, _ in sorted_labels]
        print("[权重计算] 使用当前样本分布")

    print("各类别样本数:", samples_per_cls)

    # 平方根平滑 + 归一化
    max_s = max(samples_per_cls) if samples_per_cls else 1
    weights = [math.sqrt(max_s / max(s, 1)) for s in samples_per_cls]
    mean_w = sum(weights) / len(weights) if weights else 1
    weights = [w / mean_w for w in weights]

    print("平滑后权重:", [round(w, 2) for w in weights])
    return weights

def extract_weights_from_label(label_dict, label_mapping):
    # 1. 统计每个标签出现的次数
    # label_dict.values() 就是所有样本的标签列表 ['Normal', 'Disease', 'Normal', ...]
    all_labels_list = list(label_dict.values())
    counts = Counter(all_labels_list)

    # 2. 按照 label_mapping 的索引顺序生成列表
    # 关键点：必须确保列表里的数字顺序，和模型训练时的 label index (0, 1, 2...) 一一对应
    # label_mapping.items() 是 ('标签名', 索引) 的键值对
    sorted_labels = sorted(label_mapping.items(), key=lambda item: item[1])

    # 3.生成列表
    samples_per_cls = [counts[label_name] for label_name, index in sorted_labels]

    # 4.计算权重: 平方根平滑 + 归一化 (避免权重过于激进)
    max_s = max(samples_per_cls)
    import math
    weights = [math.sqrt(max_s / s) for s in samples_per_cls]
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    print("类别样本数:", samples_per_cls)
    print("平滑后权重:", [round(w, 2) for w in weights])
    return weights

if __name__ == "__main__":
    print("="*80)
    print("SwanLab训练模块测试")
    print("="*80)

    if SWANLAB_AVAILABLE:
        print("✓ SwanLab已安装，可以使用完整功能")
    else:
        print("✗ SwanLab未安装，使用简化模式")

    print("\n使用示例:")
    print("python train_with_swanlab.py")
