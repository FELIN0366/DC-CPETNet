#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPET 特征可视化模块 (多模型支持)
========================

支持模型: HDSTGCN, STFinalNet, LSTMNet, ResNet1D, MedNet

验证特征空间:
- HDSTGCN: Temporal [B, C, D_time] -> Spatial [B, 48] -> Fused [B, 64]
- STFinalNet: TFE [B, 48] + SFE [B, 48] -> Fused [B, 96]
- 其他模型: 直接提取最终分类前特征

使用 t-SNE/PCA/UMAP 可视化特征空间的类别聚类情况。

Usage:
    # 基本使用 (自动从 config.yaml 读取模型配置)
    python scripts/visualize_features.py

    # 完整分析
    python scripts/visualize_features.py --method all --layer all --channel_analysis

    # 指定模型路径
    python scripts/visualize_features.py --model_path models/best_HDSTGCN_CPET_New_full.pth
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 降维方法
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 可选: UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("UMAP 未安装, 将跳过 UMAP 可视化。安装: pip install umap-learn")

# 项目模块
from config import Config
from dataset_new import CPETDatasetNew, collate_fn_variable_length
from feature_mapping import create_adjacency_matrix, NEW_FEATURES

# 动态导入所有模型
from model import (
    HDSTGCN, STFinalNet, LSTMNet, ResNet1D, MedNet
)

# =============================================================================
# 全局配置
# =============================================================================

# 类别名称映射 (将在运行时从数据集中动态更新)
CLASS_NAMES = {}
CLASS_COLORS = []

# 默认颜色列表
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 通道分组名称 (26个特征: 22原始 + 4衍生)
CHANNEL_GROUPS = {
    'G0_代谢': [0, 1, 2, 3, 4, 22],           # MET, Load, RER, HR, HRR, OUES
    'G1_循环': [5, 6, 7, 8, 9, 10, 23, 25],   # dH/dO2, SVc, Psys, Pdia, SpO2, V'O2, PP, HR_diff
    'G2_呼吸': [11, 12, 13],                  # VO2/kg, dO2/dW, BF
    'G3_气体': [14, 15, 16, 17, 18, 19, 20, 21, 24]  # V'E, BR, EqO2, EqCO2, PETO2, PETCO2, VDc/VT, VTex, EqO2_COP
}


# =============================================================================
# 特征提取器 (多模型支持)
# =============================================================================

class FeatureExtractor:
    """
    Hook-based 特征提取器 (支持多种模型)

    使用 PyTorch 的 forward hook 机制提取模型中间层特征,
    无需修改模型代码。

    支持模型:
    - HDSTGCN: temporal_encoder, spatial_graph, static_encoder
    - STFinalNet: tfe_branch (st_gcn), sfe_branch (sfe_encoder)
    - 其他模型: 最终分类前特征

    Attributes:
        model: 模型实例
        model_name: 模型名称
        features: dict, 存储提取的特征
        hooks: list, 注册的 hook 句柄
    """

    def __init__(self, model, model_name='HDSTGCN'):
        """
        Args:
            model: 模型实例
            model_name: 模型名称 ('HDSTGCN', 'STFinalNet', etc.)
        """
        self.model = model
        self.model_name = model_name
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """根据模型类型注册不同的 forward hooks"""
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.features[name] = output[0].detach().cpu()
                else:
                    self.features[name] = output.detach().cpu()
            return hook

        if self.model_name == 'HDSTGCN':
            # HDSTGCN: temporal_encoder -> spatial_graph -> static_encoder (可选)
            self.hooks.append(
                self.model.temporal_encoder.register_forward_hook(get_hook('temporal'))
            )
            self.hooks.append(
                self.model.spatial_graph.register_forward_hook(get_hook('spatial'))
            )
            if hasattr(self.model, 'static_encoder'):
                self.hooks.append(
                    self.model.static_encoder.register_forward_hook(get_hook('static'))
                )

        elif self.model_name == 'STFinalNet':
            # STFinalNet: TFE分支 (st_gcn) + SFE分支 (sfe_encoder)
            self.hooks.append(
                self.model.st_gcn.register_forward_hook(get_hook('tfe'))
            )
            self.hooks.append(
                self.model.sfe_encoder.register_forward_hook(get_hook('sfe'))
            )
            if hasattr(self.model, 'dynamic_graph'):
                self.hooks.append(
                    self.model.dynamic_graph.register_forward_hook(get_hook('dynamic_graph'))
                )

        else:
            # 其他模型: 注册分类器前最后一层
            # 尝试找到 classifier 或 fc 层
            if hasattr(self.model, 'classifier'):
                # classifier 通常是 Sequential, 注册第一个 Linear 前的特征
                # 这里注册整个 classifier, 后续处理时取输入
                pass  # 对于简单模型, 使用 forward 方法直接提取
            elif hasattr(self.model, 'fc'):
                pass

    def extract(self, data, lengths=None, static_x=None, adj=None):
        """
        提取特征

        Args:
            data: [B, L, C] 输入序列
            lengths: [B] 序列长度 (可选, HDSTGCN 需要)
            static_x: [B, 5] 静态特征 (可选, HDSTGCN 需要)
            adj: 邻接矩阵 (可选, STFinalNet 需要)

        Returns:
            dict: 根据模型类型返回不同的特征字典
        """
        self.features = {}

        with torch.no_grad():
            if self.model_name == 'HDSTGCN':
                _ = self.model(data, lengths=lengths, static_x=static_x)
                return self._process_hdstgcn_features()

            elif self.model_name == 'STFinalNet':
                _ = self.model(data, adj=adj)
                return self._process_stfinalnet_features()

            elif self.model_name == 'LSTMNet':
                return self._extract_lstm_features(data)

            elif self.model_name == 'ResNet1D':
                return self._extract_resnet_features(data)

            elif self.model_name == 'MedNet':
                return self._extract_mednet_features(data)

            else:
                # 通用提取: 尝试调用 forward_with_features 或直接 forward
                if hasattr(self.model, 'forward_with_features'):
                    _, features = self.model.forward_with_features(data)
                    return {'features': features.cpu().numpy()}
                else:
                    _ = self.model(data)
                    return {'features': self.features.get('features', None)}

    def _process_hdstgcn_features(self):
        """处理 HDSTGCN 模型的特征"""
        result = {}

        # Temporal: [B, C, D_time]
        if 'temporal' in self.features:
            result['temporal'] = self.features['temporal'].numpy()

        # Spatial: [B, 48]
        if 'spatial' in self.features:
            result['spatial'] = self.features['spatial'].numpy()

        # Static: [B, 16]
        if 'static' in self.features:
            result['static'] = self.features['static'].numpy()
            # Fused = spatial + static
            result['fused'] = np.concatenate(
                [result['spatial'], result['static']], axis=1
            )
        else:
            result['fused'] = result.get('spatial')

        return result

    def _process_stfinalnet_features(self):
        """处理 STFinalNet 模型的特征"""
        result = {}

        # TFE: [B, 48] (st_gcn 输出是 tuple, 第一个元素是特征)
        if 'tfe' in self.features:
            tfe_feat = self.features['tfe']
            result['tfe'] = tfe_feat.numpy()

        # SFE: [B, C, output_len] -> 需要展平
        if 'sfe' in self.features:
            sfe_feat = self.features['sfe']  # [B, C, output_len]
            # 展平为 [B, C * output_len]
            result['sfe'] = sfe_feat.reshape(sfe_feat.shape[0], -1).numpy()

        # Fused: TFE + SFE
        if 'tfe' in result and 'sfe' in result:
            # SFE 可能需要投影到相同维度
            # 这里简单处理: 使用 TFE 作为主要特征
            result['fused'] = result['tfe']
        elif 'tfe' in result:
            result['fused'] = result['tfe']
        elif 'sfe' in result:
            result['fused'] = result['sfe']

        return result

    def _extract_lstm_features(self, data):
        """提取 LSTM 模型特征"""
        # LSTMNet: 输入 [B, T, C] -> LSTM -> [B, hidden]
        self.features = {}
        with torch.no_grad():
            # 直接获取 LSTM 输出
            x = data.permute(0, 2, 1)  # [B, C, T]
            if hasattr(self.model, 'lstm'):
                output, (h_n, c_n) = self.model.lstm(x)
                features = h_n[-1]  # 最后一层隐状态
                return {'features': features.cpu().numpy(), 'fused': features.cpu().numpy()}
        return {}

    def _extract_resnet_features(self, data):
        """提取 ResNet 模型特征"""
        with torch.no_grad():
            x = data.permute(0, 2, 1)  # [B, C, T]
            # 遍历 ResNet 层直到分类器前
            features = x
            for name, module in self.model.named_children():
                if name == 'fc' or name == 'classifier':
                    break
                features = module(features)
            # 全局平均池化
            features = features.mean(dim=2)  # [B, hidden]
            return {'features': features.cpu().numpy(), 'fused': features.cpu().numpy()}

    def _extract_mednet_features(self, data):
        """提取 MedNet 模型特征"""
        with torch.no_grad():
            # MedNet 可能有特定的特征提取方式
            if hasattr(self.model, 'forward_with_features'):
                _, features = self.model.forward_with_features(data)
                return {'features': features.cpu().numpy(), 'fused': features.cpu().numpy()}
            else:
                # 通用方法
                _ = self.model(data)
                if 'features' in self.features:
                    return {'features': self.features['features'].numpy(),
                            'fused': self.features['features'].numpy()}
        return {}

    def remove_hooks(self):
        """移除所有 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# 数据加载
# =============================================================================

def load_dataset_and_model(model_path=None, device='cuda'):
    """
    加载数据集和训练好的模型 (支持多种模型类型)

    根据 config.yaml 中的 models.default 自动选择模型类型。

    Args:
        model_path: 模型权重路径 (默认: 根据 model_name 自动推断)
        device: 计算设备

    Returns:
        test_loader: 测试集 DataLoader
        model: 加载权重的模型
        config: 配置对象
    """
    global CLASS_NAMES, CLASS_COLORS

    # 加载配置
    config = Config.load()
    model_name = config.model.name

    print(f"\n[配置] 使用模型: {model_name}")

    # 根据模型类型确定是否使用变长序列和静态特征
    use_var_length = getattr(config.model, 'use_variable_length', False)
    use_static = (hasattr(config.model, 'static_features') and
                  config.model.static_features is not None and
                  config.model.static_features.enabled)

    # STFinalNet 使用固定长度
    if model_name == 'STFinalNet':
        use_var_length = False
        use_static = False

    print(f"[数据集] 变长模式: {use_var_length}, 静态特征: {use_static}")

    # 创建数据集 (单实例模式)
    full_dataset = CPETDatasetNew(
        config,
        use_variable_length=use_var_length,
        use_static_features=use_static
    )

    # 从数据集中动态获取真实的类别名称
    label_mapping = full_dataset.label_mapping
    idx_to_label = {idx: name for name, idx in label_mapping.items()}
    CLASS_NAMES = idx_to_label
    CLASS_COLORS = DEFAULT_COLORS[:len(label_mapping)]

    print(f"[数据集] 类别数量: {len(label_mapping)}")
    print(f"[数据集] 类别名称: {list(label_mapping.keys())}")

    # 获取测试集
    test_dataset = full_dataset.get_split("test")
    print(f"[数据集] 测试集样本数: {len(test_dataset)}")

    # 创建 DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_variable_length if use_var_length else None
    )

    # 根据模型类型创建模型
    semantic_adj = create_adjacency_matrix(config.features.adapt_mode) if model_name in ['HDSTGCN', 'STFinalNet'] else None

    model = _create_model(model_name, config, full_dataset.n_classes, semantic_adj, device)

    # 推断模型权重路径
    if model_path is None:
        model_path = _get_default_model_path(model_name, use_static)

    if not os.path.exists(model_path):
        # 尝试其他可能的路径
        alt_paths = [
            f"models/best_{model_name}_CPET_New_full.pth",
            f"models/best_{model_name}_CPET_New.pth",
            f"models/best_{model_name}.pth",
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"模型文件不存在，尝试路径: {model_path}")

    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[模型] 已加载权重: {model_path}")

    return test_loader, model, config, semantic_adj


def _create_model(model_name, config, n_classes, semantic_adj, device):
    """根据模型名称创建模型实例"""
    hidden_dim = config.model.hidden_dim or 16
    dropout = config.model.dropout

    if model_name == 'HDSTGCN':
        # 获取静态特征配置
        static_dim = 16
        use_static = False
        if config.model.static_features:
            static_dim = config.model.static_features.static_dim
            use_static = config.model.static_features.enabled

        model = HDSTGCN(
            input_dim=config.data.max_length,
            hidden_dim=hidden_dim,
            output_dim=n_classes,
            channel_groups=config.features.channel_groups,
            num_channel=config.features.num_channels,
            D_time=config.model.D_time or 16,
            dropout=dropout,
            semantic_adj=semantic_adj,
            use_static_features=use_static,
            static_dim=static_dim,
            num_static_features=5 if use_static else 0,
            graph_ablation=config.model.graph_ablation or 'global_only',
            temporal_encoder_type=getattr(config.model.temporal_encoder, 'type', 'gru') if config.model.temporal_encoder else 'gru',
            T_mid=getattr(config.model.temporal_encoder, 'T_mid', 24) if config.model.temporal_encoder else 24,
            gamma_init=getattr(config.model.prior_gate, 'gamma_init', 1.0) if config.model.prior_gate else 1.0,
            gamma_min=getattr(config.model.prior_gate, 'gamma_min', 0.1) if config.model.prior_gate else 0.1,
        )

    elif model_name == 'STFinalNet':
        model = STFinalNet(
            input_dim=config.data.L_win,
            hidden_dim=hidden_dim,
            channel_groups=config.features.channel_groups,
            output_dim=n_classes,
            num_channel=config.features.num_channels,
            num_patches=config.data.L_win,
            ablation=config.model.ablation or 'both',
            use_var_embedding=config.model.use_var_embedding if config.model.use_var_embedding is not None else True,
            use_dynamic_graph=config.model.use_dynamic_graph if config.model.use_dynamic_graph is not None else True,
            var_embed_dim=config.model.var_embed_dim or 8,
            semantic_adj=semantic_adj
        )

    elif model_name == 'lstm':
        model = LSTMNet(
            input_dim=config.features.num_channels,
            hidden_dim=64,
            num_layers=2,
            output_dim=n_classes,
            dropout=dropout
        )

    elif model_name == 'resnet':
        model = ResNet1D(
            input_channels=config.features.num_channels,
            num_classes=n_classes,
            hidden_dim=64
        )

    elif model_name == 'mednet':
        model = MedNet(
            input_dim=config.features.num_channels,
            hidden_dim=None,
            output_dim=n_classes
        )

    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model.to(device)


def _get_default_model_path(model_name, use_static):
    """获取默认的模型权重路径"""
    static_suffix = "_full" if use_static or model_name == 'STFinalNet' else ""
    return f"models/best_{model_name}_CPET_New{static_suffix}.pth"


def collect_features(test_loader, model, config, semantic_adj=None, device='cuda'):
    """
    遍历测试集收集所有特征 (支持多种模型)

    Args:
        test_loader: 测试集 DataLoader
        model: 模型实例
        config: 配置对象
        semantic_adj: 邻接矩阵 (STFinalNet 需要)
        device: 计算设备

    Returns:
        features_dict: 根据模型类型返回不同的特征字典
        labels: np.array [N]
    """
    model_name = config.model.name
    extractor = FeatureExtractor(model, model_name=model_name)

    # 确定是否使用变长序列
    use_var_length = getattr(config.model, 'use_variable_length', False)
    if model_name == 'STFinalNet':
        use_var_length = False

    all_features = {}
    all_labels = []

    # 根据模型类型初始化特征收集列表
    if model_name == 'HDSTGCN':
        all_features['temporal'] = []
        all_features['spatial'] = []
        all_features['static'] = []
    elif model_name == 'STFinalNet':
        all_features['tfe'] = []
        all_features['sfe'] = []
    else:
        all_features['features'] = []

    print(f"[特征提取] 遍历测试集 (模型: {model_name})...")

    with torch.no_grad():
        for batch in test_loader:
            # 解析 batch (根据模型类型和数据集配置)
            if use_var_length:
                if len(batch) == 4:
                    data, lengths, static_x, labels = batch
                    data = data.to(device)
                    lengths = lengths.to(device)
                    static_x = static_x.to(device)
                elif len(batch) == 3:
                    data, lengths, labels = batch
                    data = data.to(device)
                    lengths = lengths.to(device)
                    static_x = None
                else:
                    data, labels = batch
                    data = data.to(device)
                    lengths = None
                    static_x = None
            else:
                # 固定长度模式 (STFinalNet, etc.)
                if len(batch) == 2:
                    data, labels = batch
                    data = data.to(device)
                    lengths = None
                    static_x = None
                elif len(batch) == 3:
                    data, extra, labels = batch
                    data = data.to(device)
                    lengths = None
                    static_x = None
                else:
                    data, lengths, static_x, labels = batch
                    data = data.to(device)
                    lengths = lengths.to(device) if lengths is not None else None
                    static_x = static_x.to(device) if static_x is not None else None

            # 提取特征
            features = extractor.extract(
                data, lengths=lengths, static_x=static_x, adj=semantic_adj
            )

            # 收集特征
            for key in all_features:
                if key in features and features[key] is not None:
                    all_features[key].append(features[key])

            all_labels.append(labels.numpy())

    # 合并特征
    features_dict = {}
    for key, feat_list in all_features.items():
        if feat_list:
            features_dict[key] = np.concatenate(feat_list, axis=0)

    features_dict['labels'] = np.concatenate(all_labels, axis=0)

    # 构建 fused 特征 (如果尚未存在)
    if 'fused' not in features_dict:
        features_dict = _build_fused_features(features_dict, model_name)

    extractor.remove_hooks()

    # 打印特征信息
    feat_info = ", ".join([f"{k}: {v.shape}" for k, v in features_dict.items() if k != 'labels'])
    print(f"[特征提取] 完成: {feat_info}")

    return features_dict


def _build_fused_features(features_dict, model_name):
    """根据模型类型构建融合特征"""
    if 'fused' in features_dict:
        return features_dict

    if model_name == 'HDSTGCN':
        if 'spatial' in features_dict and 'static' in features_dict:
            features_dict['fused'] = np.concatenate(
                [features_dict['spatial'], features_dict['static']], axis=1
            )
        elif 'spatial' in features_dict:
            features_dict['fused'] = features_dict['spatial']

    elif model_name == 'STFinalNet':
        if 'tfe' in features_dict:
            features_dict['fused'] = features_dict['tfe']
        elif 'sfe' in features_dict:
            features_dict['fused'] = features_dict['sfe']

    else:
        if 'features' in features_dict:
            features_dict['fused'] = features_dict['features']

    return features_dict


# =============================================================================
# 可视化函数
# =============================================================================

def visualize_features(features, labels, method='tsne', layer_name='temporal',
                       output_dir=None, perplexity=30, n_neighbors=15, n_components=3,
                       ax=None, show_plot=True, model_name='HDSTGCN'):
    """
    降维可视化 (支持嵌入到大图)

    Args:
        features: np.array [N, D] 特征矩阵
        labels: np.array [N] 标签
        method: 降维方法 ('tsne', 'pca', 'umap')
        layer_name: 特征层名称 (用于标题和文件名)
        output_dir: 输出目录
        perplexity: t-SNE perplexity 参数
        n_neighbors: UMAP n_neighbors 参数
        n_components: 降维维度 (2 或 3)
        ax: 外部传入的 axes 对象 (用于嵌入大图)
        show_plot: 是否显示图片
        model_name: 模型名称 (用于标题)

    Returns:
        embedding: np.array [N, n_components] 降维结果
    """
    # 展平 temporal 特征 [N, C, D] -> [N, C*D]
    if len(features.shape) == 3:
        features = features.reshape(features.shape[0], -1)

    # 降维
    print(f"[{method.upper()}] 降维 {layer_name} 特征: {features.shape} -> {n_components}D")
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42,
                       max_iter=1000, learning_rate='auto', init='pca')
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'umap':
        if not HAS_UMAP:
            print("[警告] UMAP 未安装, 跳过")
            return None
        reducer = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0.1,
                       random_state=42)
    else:
        raise ValueError(f"未知降维方法: {method}")

    embedding = reducer.fit_transform(features)

    # 绑图
    if ax is not None:
        # 嵌入外部大图
        if n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D
            # 3D 需要特殊处理，这里简化为 2D
            unique_labels = np.unique(labels)
            for cls_idx in unique_labels:
                mask = labels == cls_idx
                cls_name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
                color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=color, label=cls_name, alpha=0.6, s=30)
        else:
            unique_labels = np.unique(labels)
            for cls_idx in unique_labels:
                mask = labels == cls_idx
                cls_name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
                color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=color, label=cls_name, alpha=0.6, s=30, edgecolors='white', linewidths=0.5)

        ax.set_title(f'{layer_name.capitalize()} [{method.upper()}]', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=10)
        ax.grid(True, alpha=0.3)

        return embedding

    # 独立图片
    if n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        unique_labels = np.unique(labels)
        for cls_idx in unique_labels:
            mask = labels == cls_idx
            cls_name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
            color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
            ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                       c=color, label=cls_name, alpha=0.6, s=30)

        ax.set_title(f'{model_name} {layer_name.capitalize()} Features - {method.upper()} (3D)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=10)
        ax.set_zlabel(f'{method.upper()} 3', fontsize=10)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.view_init(elev=20, azim=45)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

        unique_labels = np.unique(labels)
        for cls_idx in unique_labels:
            mask = labels == cls_idx
            cls_name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
            color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, label=cls_name, alpha=0.6, s=30, edgecolors='white', linewidths=0.5)

        ax.set_title(f'{model_name} {layer_name.capitalize()} Features - {method.upper()}',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        dim_suffix = '3d' if n_components == 3 else '2d'
        save_path = os.path.join(output_dir, f'{layer_name}_{method}_{dim_suffix}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[保存] {save_path}")

    if show_plot:
        plt.show()

    return embedding


def visualize_channels(temporal_features, labels, method='tsne',
                       output_dir=None, perplexity=30, n_neighbors=15, model_name='HDSTGCN'):
    """
    通道单独分析 (支持 26 个通道)

    将各通道的特征分别降维可视化, 分析各通道的聚类效果。
    合并为一张大图输出。

    Args:
        temporal_features: np.array [N, C, D] Temporal 特征
        labels: np.array [N] 标签
        method: 降维方法
        output_dir: 输出目录
        perplexity: t-SNE perplexity
        n_neighbors: UMAP n_neighbors
        model_name: 模型名称
    """
    N, C, D = temporal_features.shape

    # 动态计算网格大小 (每行6个，向上取整)
    n_cols = 6
    n_rows = (C + n_cols - 1) // n_cols  # 向上取整

    # 创建子图网格
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3.5))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25)

    print(f"[通道分析] 分析 {C} 个通道 (网格: {n_rows}x{n_cols})...")

    for ch in range(C):
        row, col = ch // n_cols, ch % n_cols
        ax = fig.add_subplot(gs[row, col])

        # 该通道特征 [N, D]
        ch_feat = temporal_features[:, ch, :]

        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=min(perplexity, N // 4),
                           random_state=42, max_iter=500, learning_rate='auto', init='pca')
        elif method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'umap':
            if not HAS_UMAP:
                continue
            reducer = UMAP(n_components=2, n_neighbors=min(n_neighbors, N // 4),
                           min_dist=0.1, random_state=42)

        embedding = reducer.fit_transform(ch_feat)

        # 绑图
        unique_labels = np.unique(labels)
        for cls_idx in unique_labels:
            mask = labels == cls_idx
            color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, alpha=0.5, s=10)

        # 通道名称
        ch_name = NEW_FEATURES[ch] if ch < len(NEW_FEATURES) else f"Ch{ch}"

        # 标注分组
        group_name = ""
        for gname, indices in CHANNEL_GROUPS.items():
            if ch in indices:
                group_name = gname.split('_')[0]
                break

        ax.set_title(f"{ch}: {ch_name}\n({group_name})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏空白子图
    for idx in range(C, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')

    # 添加图例
    handles = []
    for cls_idx in np.unique(labels):
        cls_name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
        color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=color, markersize=8, label=cls_name))

    fig.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)

    fig.suptitle(f'{model_name} Temporal Features per Channel ({C} channels) - {method.upper()}',
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'temporal_channels_{method}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[保存] {save_path}")

    plt.show()


def visualize_channel_groups(temporal_features, labels, method='tsne',
                             output_dir=None, perplexity=30, model_name='HDSTGCN'):
    """
    按通道组可视化 (合并为一张图)

    将每个子系统的通道特征聚合后可视化。

    Args:
        temporal_features: np.array [N, C, D]
        labels: np.array [N]
        method: 降维方法
        output_dir: 输出目录
        perplexity: t-SNE perplexity
        model_name: 模型名称
    """
    n_groups = len(CHANNEL_GROUPS)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    print(f"[分组分析] 分析 {n_groups} 个子系统...")

    for idx, (group_name, channel_indices) in enumerate(CHANNEL_GROUPS.items()):
        ax = axes[idx]

        # 过滤超出范围的索引
        valid_indices = [i for i in channel_indices if i < temporal_features.shape[1]]
        if not valid_indices:
            ax.axis('off')
            ax.set_title(f'{group_name}\n(无有效通道)', fontsize=12)
            continue

        # 聚合该组通道特征 [N, num_channels*D]
        group_feat = temporal_features[:, valid_indices, :].reshape(len(labels), -1)

        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                           max_iter=500, learning_rate='auto', init='pca')
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

        embedding = reducer.fit_transform(group_feat)

        # 绑图
        unique_labels = np.unique(labels)
        for cls_idx in unique_labels:
            mask = labels == cls_idx
            color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, label=CLASS_NAMES.get(cls_idx, f"Class {cls_idx}"),
                       alpha=0.6, s=30, edgecolors='white', linewidths=0.5)

        ax.set_title(f'{group_name}\n(Channels: {len(valid_indices)})', fontsize=12)
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.grid(True, alpha=0.3)

    # 统一图例
    handles = []
    for cls_idx in np.unique(labels):
        color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=color, markersize=8,
                                   label=CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")))

    fig.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)

    fig.suptitle(f'{model_name} Temporal Features by Channel Groups - {method.upper()}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'temporal_groups_{method}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[保存] {save_path}")

    plt.show()


def compare_layers(features_dict, method='tsne', output_dir=None, perplexity=30, n_components=3, model_name='HDSTGCN'):
    """
    层次对比可视化 (合并为一张图)

    对比不同特征层的聚类质量变化。
    - HDSTGCN: Temporal -> Spatial -> Fused
    - STFinalNet: TFE -> SFE -> Fused
    - 其他: Features -> Fused

    Args:
        features_dict: 特征字典
        method: 降维方法
        output_dir: 输出目录
        perplexity: t-SNE perplexity
        n_components: 降维维度 (2 或 3)
        model_name: 模型名称
    """
    # 根据模型类型确定层名称
    if model_name == 'HDSTGCN':
        layers = ['temporal', 'spatial', 'fused']
        titles = ['Temporal', 'Spatial', 'Fused']
    elif model_name == 'STFinalNet':
        layers = ['tfe', 'sfe', 'fused']
        titles = ['TFE', 'SFE', 'Fused']
    else:
        layers = ['features', 'fused']
        titles = ['Features', 'Fused']

    labels = features_dict['labels']

    # 过滤存在的层
    available_layers = [l for l in layers if l in features_dict]

    if len(available_layers) < 2:
        print(f"[警告] 可用层数不足 ({len(available_layers)}), 跳过层次对比")
        return

    # 创建大图
    n_layers = len(available_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    print(f"[层次对比] 对比 {n_layers} 个特征层...")

    for idx, layer in enumerate(available_layers):
        ax = axes[idx]
        feat = features_dict[layer]
        if len(feat.shape) == 3:
            feat = feat.reshape(feat.shape[0], -1)

        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                           max_iter=500, learning_rate='auto', init='pca')
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

        embedding = reducer.fit_transform(feat)

        # 绑图
        unique_labels = np.unique(labels)
        for cls_idx in unique_labels:
            mask = labels == cls_idx
            color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, label=CLASS_NAMES.get(cls_idx, f"Class {cls_idx}"),
                       alpha=0.6, s=30)

        # 获取特征维度
        feat_dim = features_dict[layer].shape[-1] if len(features_dict[layer].shape) == 2 else features_dict[layer].shape[1] * features_dict[layer].shape[2]
        layer_idx = layers.index(layer) if layer in layers else 0
        ax.set_title(f'{titles[layer_idx]}\n[dim={feat_dim}]', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.grid(True, alpha=0.3)

    # 图例
    handles = []
    for cls_idx in np.unique(labels):
        color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=color, markersize=8,
                                   label=CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")))

    fig.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)

    fig.suptitle(f'{model_name} Layer-wise Feature Comparison - {method.upper()}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'layer_comparison_{method}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[保存] {save_path}")

    plt.show()


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CPET 特征可视化 (多模型支持)")
    parser.add_argument('--method', default='tsne', choices=['tsne', 'pca', 'umap', 'all'],
                        help='降维方法')
    parser.add_argument('--layer', default='all', choices=['temporal', 'spatial', 'fused', 'tfe', 'sfe', 'features', 'all'],
                        help='特征层')
    parser.add_argument('--dim', type=int, default=2, choices=[2, 3],
                        help='降维维度 (2 或 3)')
    parser.add_argument('--channel_analysis', action='store_true',
                        help='通道单独分析 (仅 HDSTGCN)')
    parser.add_argument('--group_analysis', action='store_true',
                        help='通道分组分析 (仅 HDSTGCN)')
    parser.add_argument('--compare_layers', action='store_true',
                        help='层次对比可视化')
    parser.add_argument('--all_in_one', action='store_true',
                        help='所有可视化合并为一张大图')
    parser.add_argument('--output_dir', default='results/feature_vis',
                        help='输出目录')
    parser.add_argument('--model_path', default=None,
                        help='模型权重路径 (默认自动推断)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='计算设备')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity 参数')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors 参数')

    args = parser.parse_args()

    # 设置 matplotlib 中文支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 加载数据和模型 (自动从 config.yaml 读取模型配置)
    test_loader, model, config, semantic_adj = load_dataset_and_model(
        model_path=args.model_path,
        device=args.device
    )

    model_name = config.model.name

    # 收集特征
    features_dict = collect_features(
        test_loader, model, config, semantic_adj=semantic_adj, device=args.device
    )

    # 确定降维方法列表
    methods = ['tsne', 'pca', 'umap'] if args.method == 'all' else [args.method]
    methods = [m for m in methods if m != 'umap' or HAS_UMAP]

    print(f"\n{'=' * 60}")
    print(f"开始可视化: 模型={model_name}, 方法={methods}, 层={args.layer}, 维度={args.dim}")
    print(f"{'=' * 60}\n")

    # 一张图输出所有可视化
    if args.all_in_one:
        create_all_in_one_figure(features_dict, methods, args, model_name)
    else:
        # 分开输出
        # 层次对比
        if args.compare_layers:
            for method in methods:
                compare_layers(
                    features_dict, method=method, output_dir=args.output_dir,
                    perplexity=args.perplexity, n_components=args.dim,
                    model_name=model_name
                )

        # 单层可视化
        # 根据模型类型确定可用层
        if model_name == 'HDSTGCN':
            available_layers = ['temporal', 'spatial', 'fused']
        elif model_name == 'STFinalNet':
            available_layers = ['tfe', 'sfe', 'fused']
        else:
            available_layers = ['features', 'fused']

        layers = available_layers if args.layer == 'all' else [args.layer]
        layers = [l for l in layers if l in features_dict]

        for method in methods:
            for layer in layers:
                if layer in features_dict:
                    visualize_features(
                        features_dict[layer], features_dict['labels'],
                        method=method, layer_name=layer, output_dir=args.output_dir,
                        perplexity=args.perplexity, n_neighbors=args.n_neighbors,
                        n_components=args.dim, model_name=model_name
                    )

        # 通道分析 (仅 HDSTGCN 支持)
        if args.channel_analysis and model_name == 'HDSTGCN' and 'temporal' in features_dict:
            for method in methods:
                visualize_channels(
                    features_dict['temporal'], features_dict['labels'],
                    method=method, output_dir=args.output_dir,
                    perplexity=args.perplexity, n_neighbors=args.n_neighbors,
                    model_name=model_name
                )
        elif args.channel_analysis and model_name != 'HDSTGCN':
            print(f"[警告] 通道分析仅支持 HDSTGCN 模型, 当前模型: {model_name}")

        # 分组分析 (仅 HDSTGCN 支持)
        if args.group_analysis and model_name == 'HDSTGCN' and 'temporal' in features_dict:
            for method in methods:
                visualize_channel_groups(
                    features_dict['temporal'], features_dict['labels'],
                    method=method, output_dir=args.output_dir,
                    perplexity=args.perplexity, model_name=model_name
                )
        elif args.group_analysis and model_name != 'HDSTGCN':
            print(f"[警告] 分组分析仅支持 HDSTGCN 模型, 当前模型: {model_name}")

    print("\n" + "=" * 60)
    print("可视化完成!")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)


def create_all_in_one_figure(features_dict, methods, args, model_name='HDSTGCN'):
    """
    创建一张大图包含所有可视化内容 (支持多模型)

    布局:
    - 第一行: 层次对比 (temporal/spatial/fused 或 tfe/sfe/fused)
    - 第二行: 通道分组分析 (仅 HDSTGCN, 4个子系统)
    - 第三行开始: 通道单独分析 (仅 HDSTGCN, 26个通道)
    """
    labels = features_dict['labels']
    method = methods[0]  # 使用第一个方法

    print(f"[大图模式] 创建包含所有可视化的单张图片 (模型: {model_name})...")

    # 根据模型类型确定布局
    if model_name == 'HDSTGCN' and 'temporal' in features_dict:
        n_channels = features_dict['temporal'].shape[1]
        n_channel_rows = (n_channels + 5) // 6
        total_rows = 1 + 2 + n_channel_rows  # 层次对比 + 分组 + 通道
        do_channel_analysis = True
    elif model_name == 'STFinalNet':
        n_channels = 0
        n_channel_rows = 0
        total_rows = 1  # 仅层次对比
        do_channel_analysis = False
    else:
        n_channels = 0
        n_channel_rows = 0
        total_rows = 1
        do_channel_analysis = False

    # 创建大图
    fig = plt.figure(figsize=(24, 4 * total_rows))
    gs = GridSpec(total_rows, 6, figure=fig, hspace=0.3, wspace=0.25)

    # ========== 第一行: 层次对比 ==========
    if model_name == 'HDSTGCN':
        layers = ['temporal', 'spatial', 'fused']
    elif model_name == 'STFinalNet':
        layers = ['tfe', 'sfe', 'fused']
    else:
        layers = ['features', 'fused']

    # 过滤存在的层
    layers = [l for l in layers if l in features_dict]

    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[0, idx * 2:(idx + 1) * 2])

        feat = features_dict[layer]
        if len(feat.shape) == 3:
            feat = feat.reshape(feat.shape[0], -1)

        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=args.perplexity, random_state=42,
                           max_iter=500, learning_rate='auto', init='pca')
        elif method == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=0.1, random_state=42)

        embedding = reducer.fit_transform(feat)

        unique_labels = np.unique(labels)
        for cls_idx in unique_labels:
            mask = labels == cls_idx
            color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, label=CLASS_NAMES.get(cls_idx, f"Class {cls_idx}"),
                       alpha=0.6, s=30)

        ax.set_title(f'{layer.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        ax.grid(True, alpha=0.3)

    # ========== 第二行: 通道分组分析 (仅 HDSTGCN) ==========
    if do_channel_analysis and model_name == 'HDSTGCN':
        for idx, (group_name, channel_indices) in enumerate(CHANNEL_GROUPS.items()):
            ax = fig.add_subplot(gs[1, idx * 2:(idx + 1) * 2])

            # 过滤有效索引
            valid_indices = [i for i in channel_indices if i < n_channels]
            if not valid_indices:
                ax.axis('off')
                continue

            group_feat = features_dict['temporal'][:, valid_indices, :].reshape(len(labels), -1)

            if method == 'tsne':
                reducer = TSNE(n_components=2, perplexity=args.perplexity, random_state=42,
                               max_iter=500, learning_rate='auto', init='pca')
            elif method == 'pca':
                reducer = PCA(n_components=2)
            else:
                reducer = UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=0.1, random_state=42)

            embedding = reducer.fit_transform(group_feat)

            unique_labels = np.unique(labels)
            for cls_idx in unique_labels:
                mask = labels == cls_idx
                color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=color, alpha=0.5, s=20)

            ax.set_title(f'{group_name} ({len(valid_indices)} ch)', fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    # ========== 第三行开始: 通道单独分析 (仅 HDSTGCN) ==========
    if do_channel_analysis and model_name == 'HDSTGCN':
        for ch in range(n_channels):
            row = 2 + ch // 6
            col = ch % 6
            ax = fig.add_subplot(gs[row, col])

            ch_feat = features_dict['temporal'][:, ch, :]

            if method == 'tsne':
                reducer = TSNE(n_components=2, perplexity=min(args.perplexity, len(labels) // 4),
                               random_state=42, max_iter=500, learning_rate='auto', init='pca')
            elif method == 'pca':
                reducer = PCA(n_components=2)
            else:
                reducer = UMAP(n_components=2, n_neighbors=min(args.n_neighbors, len(labels) // 4),
                               min_dist=0.1, random_state=42)

            embedding = reducer.fit_transform(ch_feat)

            unique_labels = np.unique(labels)
            for cls_idx in unique_labels:
                mask = labels == cls_idx
                color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=color, alpha=0.4, s=8)

            ch_name = NEW_FEATURES[ch] if ch < len(NEW_FEATURES) else f"Ch{ch}"

            # 标注分组
            group_name = ""
            for gname, indices in CHANNEL_GROUPS.items():
                if ch in indices:
                    group_name = gname.split('_')[0]
                    break

            ax.set_title(f"{ch}: {ch_name}\n({group_name})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    # 添加图例
    handles = []
    for cls_idx in np.unique(labels):
        color = CLASS_COLORS[cls_idx] if cls_idx < len(CLASS_COLORS) else '#333333'
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=color, markersize=8,
                                   label=CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")))

    fig.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)

    # 标题
    if model_name == 'HDSTGCN':
        title_suffix = f'\nLayer Comparison | Channel Groups | Individual Channels ({n_channels})'
    else:
        title_suffix = f'\nLayer Comparison'

    fig.suptitle(f'{model_name} Feature Visualization - {method.upper()}{title_suffix}',
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f'all_in_one_{model_name.lower()}_{method}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[保存] {save_path}")

    plt.show()


if __name__ == "__main__":
    main()