"""
新数据集类 - 适配新的CPET数据格式
集成data_preprocess_new和label_extractor模块
** 已集成缓存机制，极大提高二次加载速度 **
** 已支持变长序列模式 (Variable Length Sequence) **
** 已适配统一配置系统 (Config) **
"""

import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import sys
from collections import Counter

# 导入自定义模块
from data_preprocess_new import get_data_new, get_data_variable_length
from label_extractor import (
    load_labels, get_label_for_file, create_label_encoder, load_static_features,
    load_pft_features, _build_filename_to_final_id_mapping, _merge_static_features,
    # [新增] 多标签支持
    load_labels_unified, create_multilabel_encoder
)
from feature_mapping import normalize_features, get_feature_statistics


def collate_fn_variable_length(batch):
    """
    处理变长序列的 batch
    在 DataLoader 中使用: DataLoader(dataset, collate_fn=collate_fn_variable_length, ...)

    Args:
        batch: list of tuples
               - 变长模式 + 静态特征 + t6: (data, length, static, t6_onehot, label)
               - 变长模式 + 静态特征: (data, length, static, label)
               - 变长模式 + t6: (data, length, t6_onehot, label)
               - 变长模式: (data, length, label)
               - 固定长度 + 静态 + t6: (data, static, t6_onehot, label)
               - 固定长度 + 静态: (data, static, label)
               - 固定长度 + t6: (data, t6_onehot, label)
               - 固定长度: (data, label)

    Returns:
        相应的 batch 元素堆叠
    """
    # 检测 batch 元素数量判断模式
    num_elements = len(batch[0])

    if num_elements == 5:
        # 变长模式 + 静态特征 + t6: (data, length, static, t6_onehot, label)
        data_list, length_list, static_list, t6_list, label_list = zip(*batch)

        data_batch = torch.stack(data_list, dim=0)       # [B, L_max, C]
        lengths = torch.stack(length_list, dim=0)        # [B]
        static_batch = torch.stack(static_list, dim=0)   # [B, static_dim]
        t6_batch = torch.stack(t6_list, dim=0)           # [B, t6_n_classes]
        labels = torch.stack(label_list, dim=0)          # [B]

        return data_batch, lengths, static_batch, t6_batch, labels

    elif num_elements == 4:
        # 可能是:
        #   - 变长模式 + 静态特征: (data, length, static, label)
        #   - 变长模式 + t6: (data, length, t6_onehot, label)
        #   - 固定长度 + 静态 + t6: (data, static, t6_onehot, label)
        # 通过检查第二个元素类型来判断
        second_element = batch[0][1]

        if isinstance(second_element, torch.Tensor) and second_element.dim() == 0:
            # second_element 是 scalar tensor (length) -> 变长模式 + 静态特征
            data_list, length_list, static_list, label_list = zip(*batch)
            data_batch = torch.stack(data_list, dim=0)
            lengths = torch.stack(length_list, dim=0)
            static_batch = torch.stack(static_list, dim=0)
            labels = torch.stack(label_list, dim=0)
            return data_batch, lengths, static_batch, labels
        elif isinstance(second_element, torch.Tensor) and second_element.dim() == 1 and second_element.shape[0] > 1:
            # second_element 是向量 -> 可能是 t6_onehot 或 static
            # 检查第三个元素判断是 static+t6 还是 t6 only
            third_element = batch[0][2]
            if isinstance(third_element, torch.Tensor) and third_element.dim() == 0:
                # 变长模式 + t6: (data, length, t6_onehot, label)
                data_list, length_list, t6_list, label_list = zip(*batch)
                data_batch = torch.stack(data_list, dim=0)
                lengths = torch.stack(length_list, dim=0)
                t6_batch = torch.stack(t6_list, dim=0)
                labels = torch.stack(label_list, dim=0)
                return data_batch, lengths, t6_batch, labels
            else:
                # 固定长度 + 静态 + t6: (data, static, t6_onehot, label)
                data_list, static_list, t6_list, label_list = zip(*batch)
                data_batch = torch.stack(data_list, dim=0)
                static_batch = torch.stack(static_list, dim=0)
                t6_batch = torch.stack(t6_list, dim=0)
                labels = torch.stack(label_list, dim=0)
                return data_batch, static_batch, t6_batch, labels
        else:
            # 固定长度 + 静态: (data, static, label) - 兼容旧模式
            data_list, static_list, label_list = zip(*batch)
            data_batch = torch.stack(data_list, dim=0)
            static_batch = torch.stack(static_list, dim=0)
            labels = torch.stack(label_list, dim=0)
            return data_batch, static_batch, labels

    elif num_elements == 3:
        # 可能是:
        #   - 变长模式: (data, length, label)
        #   - 固定长度 + t6: (data, t6_onehot, label)
        second_element = batch[0][1]

        if isinstance(second_element, torch.Tensor) and second_element.dim() == 0:
            # second_element 是 scalar (length) -> 变长模式
            data_list, length_list, label_list = zip(*batch)
            data_batch = torch.stack(data_list, dim=0)
            lengths = torch.stack(length_list, dim=0)
            labels = torch.stack(label_list, dim=0)
            return data_batch, lengths, labels
        else:
            # 固定长度 + t6: (data, t6_onehot, label)
            data_list, t6_list, label_list = zip(*batch)
            data_batch = torch.stack(data_list, dim=0)
            t6_batch = torch.stack(t6_list, dim=0)
            labels = torch.stack(label_list, dim=0)
            return data_batch, t6_batch, labels
    else:
        # 固定长度模式: (data, label)
        data_list, label_list = zip(*batch)
        data_batch = torch.stack(data_list, dim=0)
        labels = torch.stack(label_list, dim=0)
        return data_batch, labels


class CPETDatasetSubset(Dataset):
    """
    数据集子集视图 - 不复制数据，共享父集的统计量

    用于返回训练/测试子集，避免数据泄露问题
    """

    def __init__(self, parent_dataset, indices, phase):
        """
        Args:
            parent_dataset: CPETDatasetNew 父实例
            indices: 子集索引列表
            phase: "train" 或 "test"
        """
        self.parent = parent_dataset
        self.indices = indices
        self.phase = phase
        self.enable_aug = (phase == "train")

    def __len__(self):
        return len(self.indices)

    @property
    def filenames_list(self):
        """返回当前子集的文件名列表"""
        return [self.parent.raw_filenames[i] for i in self.indices]

    @property
    def label_mapping(self):
        """返回标签映射"""
        return self.parent.label_mapping

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # 获取动态数据 (未标准化的原始数据)
        data = self.parent.raw_datalist[real_idx].copy()

        # 训练阶段应用动态增强
        if self.enable_aug:
            data = self.parent.augment_data(data)

        # 标准化动态数据 (使用父集的训练集统计量)
        data, _ = normalize_features(data, method='robust', feature_stats=self.parent.stats)

        # ========== 获取标签 (支持多标签模式) ==========
        if self.parent.is_multilabel:
            # 多标签模式: 返回 multi-hot float tensor
            label_names = self.parent.raw_labellist[real_idx]  # List[str]
            encoder = self.parent.get_multilabel_encoder()
            label = encoder(label_names)  # [n_classes] multi-hot
            label_tensor = torch.as_tensor(label.astype(np.float32))
        else:
            # 单标签模式: 返回索引 long tensor
            label_name = self.parent.raw_labellist[real_idx]  # str
            label_idx = self.parent.label_mapping[label_name]
            label_tensor = torch.tensor(label_idx, dtype=torch.long)
        # ==================================================

        # ========== [新增] Known-T6 Context: 获取 t6 one-hot ==========
        t6_onehot = None
        if hasattr(self.parent, 'use_known_t6_context') and self.parent.use_known_t6_context:
            filename = self.parent.raw_filenames[real_idx]
            # 处理文件名格式 (移除 .xlsx 后缀)
            lookup_key = filename[:-5] if filename.endswith('.xlsx') else filename
            t6_label_str = self.parent.t6_label_dict.get(lookup_key)
            if t6_label_str is not None and t6_label_str in self.parent.t6_label_mapping:
                t6_idx = self.parent.t6_label_mapping[t6_label_str]
                t6_onehot = torch.zeros(self.parent.t6_n_classes, dtype=torch.float32)
                t6_onehot[t6_idx] = 1.0
            else:
                # Fallback: 生成零向量
                t6_onehot = torch.zeros(self.parent.t6_n_classes, dtype=torch.float32)
        # ================================================================

        # 变长序列处理
        if self.parent.use_variable_length:
            padded_data, length = self.parent._pad_sequence(data)

            # [修改] 根据 t6 context 状态返回不同元素数量
            if self.parent.use_static_features:
                static = np.array(self.parent.raw_staticlist[real_idx], dtype=np.float32)
                static_norm = (static - self.parent.static_stats['mean']) / (self.parent.static_stats['std'] + 1e-8)
                if t6_onehot is not None:
                    # 5 元素: (data, length, static, t6_onehot, label)
                    return (
                        torch.as_tensor(padded_data.astype(np.float32)),
                        torch.tensor(length, dtype=torch.long),
                        torch.as_tensor(static_norm.astype(np.float32)),
                        t6_onehot,
                        label_tensor
                    )
                else:
                    # 4 元素: (data, length, static, label)
                    return (
                        torch.as_tensor(padded_data.astype(np.float32)),
                        torch.tensor(length, dtype=torch.long),
                        torch.as_tensor(static_norm.astype(np.float32)),
                        label_tensor
                    )
            else:
                if t6_onehot is not None:
                    # 4 元素: (data, length, t6_onehot, label)
                    return (
                        torch.as_tensor(padded_data.astype(np.float32)),
                        torch.tensor(length, dtype=torch.long),
                        t6_onehot,
                        label_tensor
                    )
                else:
                    # 3 元素: (data, length, label)
                    return (
                        torch.as_tensor(padded_data.astype(np.float32)),
                        torch.tensor(length, dtype=torch.long),
                        label_tensor
                    )
        else:
            # 固定长度模式
            if self.parent.use_static_features:
                static = np.array(self.parent.raw_staticlist[real_idx], dtype=np.float32)
                static_norm = (static - self.parent.static_stats['mean']) / (self.parent.static_stats['std'] + 1e-8)
                if t6_onehot is not None:
                    # 4 元素: (data, static, t6_onehot, label)
                    return (
                        torch.as_tensor(data.astype(np.float32)),
                        torch.as_tensor(static_norm.astype(np.float32)),
                        t6_onehot,
                        label_tensor
                    )
                else:
                    # 3 元素: (data, static, label)
                    return (
                        torch.as_tensor(data.astype(np.float32)),
                        torch.as_tensor(static_norm.astype(np.float32)),
                        label_tensor
                    )
            else:
                if t6_onehot is not None:
                    # 3 元素: (data, t6_onehot, label)
                    return (
                        torch.as_tensor(data.astype(np.float32)),
                        t6_onehot,
                        label_tensor
                    )
                else:
                    # 2 元素: (data, label)
                    return torch.as_tensor(data.astype(np.float32)), label_tensor

    @property
    def n_classes(self):
        return self.parent.n_classes

    @property
    def label_mapping(self):
        return self.parent.label_mapping

    @property
    def L_max(self):
        return self.parent.L_max

    @property
    def labellist(self):
        """返回子集的标签列表 (用于权重计算)"""
        return [self.parent.raw_labellist[i] for i in self.indices]


class CPETDatasetNew(Dataset):
    """
    新CPET数据集类 - 集成类别平衡与生理流形增强
    针对长尾分布优化 (ICLR/NeurIPS 策略)

    ** 单实例模式: 只实例化一次，通过 get_split() 获取训练/测试子集 **
    ** 解决静态特征标准化的数据泄露问题 **

    支持两种模式:
    1. 固定长度模式 (use_variable_length=False):
       所有序列插值到 L_win 长度
    2. 变长序列模式 (use_variable_length=True):
       保留原始长度，通过 zero-padding 对齐到 L_max

    配置来源:
    - 支持 Config 对象 (统一配置系统)
    - 兼容旧版 args 对象
    """
    def __init__(self, config, test_ratio=0.2, feature_indices=None,
                 use_variable_length=False, max_length=330, use_static_features=False):
        """
        单实例模式 - 加载所有数据，内部完成划分

        Args:
            config: Config 配置对象 (或旧版 args 对象)
            test_ratio: 测试集比例
            feature_indices: 要使用的特征索引列表
            use_variable_length: 是否使用变长序列模式 (默认 False)
            max_length: 变长模式下的最大长度限制 (默认 330)
            use_static_features: 是否使用静态特征融合 (默认 False)
        """
        # 兼容 Config 对象和旧版 args 对象
        self._init_from_config(config)

        self.feature_indices = feature_indices
        self.test_ratio = test_ratio

        # ========== 变长序列参数 ==========
        self.use_variable_length = use_variable_length
        self.max_length = max_length
        self.L_max = None  # 将在数据加载后计算
        # =================================

        # ========== 静态特征参数 ==========
        self.use_static_features = use_static_features
        self.static_dict = None
        self.static_stats = None  # 标准化统计量 (只用训练集计算)
        # PFT 参数 (已在 _init_from_config 中初始化)
        # self.pft_enabled, self.skip_missing_pft, self._pft_file 已设置
        self.num_static_features = 5  # 默认 5 个 EHR 特征 (加载后更新)
        # =================================

        # ========== 多标签参数 (新增) ==========
        self.is_multilabel = False
        self.co_occurrence_matrix = None
        self._multilabel_encoder = None
        # =======================================

        # ========== Known-T6 Context 参数 (新增) ==========
        self.t6_label_dict = None        # {filename: t6_label_str}
        self.t6_label_mapping = None     # {t6_label_str: index}
        self.t6_n_classes = 0            # t6 类别数
        # =================================================

        # ========== 原始数据存储 ==========
        self.raw_datalist = []      # 所有原始动态数据
        self.raw_labellist = []     # 所有原始标签
        self.raw_staticlist = []    # 所有原始静态特征
        self.raw_filenames = []     # 所有文件名 (用于错误分析)
        self.train_idx = []         # 训练集索引
        self.test_idx = []          # 测试集索引
        # =================================

        # 设置缓存目录
        self.cache_dir = os.path.join(os.path.dirname(self.data_root), "npy_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        if feature_indices is not None:
            self.num_channels = len(feature_indices)
        else:
            self.num_channels = self._num_channels

        # 加载所有数据 (单实例模式)
        self._load_all_data()

    def _init_from_config(self, config):
        """从配置对象初始化属性"""
        # 检测是否为 Config 对象 (有 data 属性)
        if hasattr(config, 'data'):
            # 新版 Config 对象
            self.config = config
            self.data_root = config.data.data_root
            self.label_file = config.data.label_file
            self.L_win = config.data.L_win
            self._target_col_name = config.features.target_col_name
            self._num_channels = config.features.num_channels
            self._adapt_mode = config.features.adapt_mode
            self._augmentation = config.augmentation

            # [新增] 可插拔特征配置 - 氧脉搏导数特征
            self._o2pulse_enabled = getattr(config.features, 'o2pulse_enabled', False)
            # [新增] V'CO2 特征配置 (九图模式)
            self._vco2_enabled = getattr(config.features, 'vco2_enabled', False)
            # [新增] 基础衍生特征配置
            self._base_enabled = getattr(config.features, 'base_enabled', True)

            # [新增] 特征选择消融配置
            if hasattr(config.model, 'feature_ablation') and config.model.feature_ablation is not None:
                self._feature_ablation = config.model.feature_ablation
            else:
                self._feature_ablation = None

            # [新增] PFT 配置
            if hasattr(config.model, 'static_features') and config.model.static_features is not None:
                sf_cfg = config.model.static_features
                self.pft_enabled = getattr(sf_cfg, 'pft_enabled', False)
                self.skip_missing_pft = getattr(sf_cfg, 'skip_missing_pft', True)
                self._pft_file = getattr(sf_cfg, 'pft_file', "xx_path")
            else:
                self.pft_enabled = False
                self.skip_missing_pft = True
                self._pft_file = "xx_path"

            # [新增] Known-T6 Context 配置
            if hasattr(config, 'known_t6_context') and config.known_t6_context is not None:
                self.use_known_t6_context = config.known_t6_context.enabled
                self._t6_source_column = config.known_t6_context.source_column
            else:
                self.use_known_t6_context = False
                self._t6_source_column = "匹配的第一大类"
        else:
            # 旧版 args 对象 (兼容)
            self.config = config
            self.data_root = config.data_root
            self.label_file = config.label_file
            self.L_win = config.L_win
            self._target_col_name = getattr(config, 'target_col_name', '匹配的第一大类')
            self._num_channels = getattr(config, 'num_channels', 26)
            self._adapt_mode = getattr(config, 'adapt_mode', 'full')
            self._augmentation = None
            self._feature_ablation = None
            self._base_enabled = True  # [新增] 默认启用
            self._o2pulse_enabled = False
            self._vco2_enabled = False
            # [新增] Known-T6 Context (旧版 args 不支持)
            self.use_known_t6_context = False
            self._t6_source_column = "匹配的第一大类"

    def _get_feature_indices_for_ablation(self, original_indices):
        """
        根据特征消融配置获取特征索引

        Args:
            original_indices: 原始特征索引列表

        Returns:
            selected_indices: 选择后的特征索引列表
        """
        if self._feature_ablation is None or not self._feature_ablation.enabled:
            return original_indices

        mode = self._feature_ablation.mode
        print(f"[特征消融] 模式: {mode}")

        if mode == "full":
            # 使用全部特征
            return original_indices

        elif mode == "remove_weak":
            # 移除弱特征
            weak_channels = self._feature_ablation.weak_channels
            selected = [i for i in original_indices if i not in weak_channels]
            print(f"[特征消融] 移除弱特征 {weak_channels}, 保留 {len(selected)} 个特征")
            return selected

        elif mode == "strong_only":
            # 仅保留强特征
            strong_channels = self._feature_ablation.strong_channels
            selected = [i for i in original_indices if i in strong_channels]
            print(f"[特征消融] 仅保留强特征, 共 {len(selected)} 个特征")
            return selected

        elif mode == "remove_group":
            # 移除特定组
            remove_group = self._feature_ablation.remove_group
            if not remove_group:
                return original_indices

            # 定义特征组 (基于 NEW_FEATURES 索引)
            GROUPS = {
                'G0': [0, 1, 2, 3, 4, 22],           # 运动负荷与能量代谢
                'G1': [5, 6, 7, 8, 9, 10, 23, 25],   # 循环系统
                'G2': [11, 12, 13],                  # 呼吸动力学
                'G3': [14, 15, 16, 17, 18, 19, 20, 21, 24]  # 气体交换
            }

            if remove_group in GROUPS:
                remove_indices = GROUPS[remove_group]
                selected = [i for i in original_indices if i not in remove_indices]
                print(f"[特征消融] 移除组 {remove_group} ({len(remove_indices)} 个特征), 保留 {len(selected)} 个特征")
                return selected
            else:
                print(f"[特征消融] 警告: 未知组名 {remove_group}")
                return original_indices

        else:
            print(f"[特征消融] 警告: 未知模式 {mode}")
            return original_indices

    def _load_all_data(self):
        """
        单实例模式 - 加载所有数据，完成划分，计算统计量

        关键：统计量只用训练集计算，避免数据泄露
        """
        # ========== [新增] 确定任务模式 ==========
        task_cfg = getattr(self.config, 'task', None)
        if task_cfg is not None:
            task_mode = task_cfg.mode
            label_separator = task_cfg.label_separator
            min_label_freq = task_cfg.min_label_freq
        else:
            task_mode = "single_label"
            label_separator = ";"
            min_label_freq = 50

        # 1. 加载标签 (使用统一接口)
        self.label_dict, self.label_mapping, co_occurrence = load_labels_unified(
            label_file=self.label_file,
            task_mode=task_mode,
            target_col_name=self._target_col_name,
            label_separator=label_separator,
            min_label_freq=min_label_freq
        )

        # 设置多标签标志
        self.is_multilabel = (task_mode == "multi_label")
        self._multilabel_encoder = None  # 延迟初始化

        if self.is_multilabel and co_occurrence is not None:
            self.co_occurrence_matrix = co_occurrence
            # 同步到 config
            self.config.co_occurrence_matrix = co_occurrence
            self.config.is_multilabel = True
            print(f"[数据集] 多标签模式: {len(self.label_mapping)} 个标签")
        else:
            self.co_occurrence_matrix = None
            self.config.is_multilabel = False

        self.n_classes = len(self.label_mapping)

        # ========== [新增] Known-T6 Context: 独立加载 t6 标签 ==========
        if self.use_known_t6_context:
            # 使用 load_labels 加载 t6 标签 (单标签模式，返回2个值)
            self.t6_label_dict, self.t6_label_mapping = load_labels(
                label_file=self.label_file,
                target_col_name=self._t6_source_column
            )
            self.t6_n_classes = len(self.t6_label_mapping)
            print(f"[Known-T6 Context] 已启用: source_column={self._t6_source_column}, t6_n_classes={self.t6_n_classes}")
        # ================================================================

        # 1.5 加载静态特征 (如果启用)
        if self.use_static_features:
            # 加载 EHR 特征
            ehr_dict, ehr_feature_names = load_static_features(self.label_file)

            if self.pft_enabled:
                # 加载 PFT 特征
                pft_dict, pft_feature_names = load_pft_features(self._pft_file)

                if pft_dict:
                    # 建立 filename -> final_编号 映射
                    filename_to_final_id = _build_filename_to_final_id_mapping(self.label_file)

                    # 合并特征
                    self.static_dict, num_ehr, num_pft = _merge_static_features(
                        ehr_dict, pft_dict, filename_to_final_id,
                        skip_missing_pft=self.skip_missing_pft
                    )
                    self.static_feature_names = ehr_feature_names + pft_feature_names
                    self.num_static_features = num_ehr + num_pft
                    print(f"[数据集] 静态特征: EHR({num_ehr}) + PFT({num_pft}) = {self.num_static_features}")
                else:
                    # PFT 加载失败，仅使用 EHR
                    print(f"[数据集] PFT 加载失败，仅使用 EHR 特征")
                    self.static_dict = {k: [v['age'], v['gender'], v['weight'], v['height'], v['bmi']]
                                        for k, v in ehr_dict.items()}
                    self.static_feature_names = ehr_feature_names
                    self.num_static_features = 5
            else:
                # 仅使用 EHR 特征
                self.static_dict = {k: [v['age'], v['gender'], v['weight'], v['height'], v['bmi']]
                                    for k, v in ehr_dict.items()}
                self.static_feature_names = ehr_feature_names
                self.num_static_features = 5

        # 2. 扫描文件
        data_files = sorted([f for f in os.listdir(self.data_root) if f.endswith('.xlsx')])

        print(f"[数据集] 正在加载数据到内存...")
        print(f"[数据集] 模式: {'变长序列' if self.use_variable_length else '固定长度'}")
        if self.use_static_features:
            print(f"[数据集] 静态特征: 已启用")

        # [新增] 确定特征选择索引
        all_feature_indices = list(range(self._num_channels))
        if self._feature_ablation is not None and self._feature_ablation.enabled:
            self.selected_feature_indices = self._get_feature_indices_for_ablation(all_feature_indices)
        else:
            self.selected_feature_indices = all_feature_indices

        for filename in data_files:
            label_name = get_label_for_file(filename, self.label_dict)
            if label_name is None: continue

            try:
                # ========== 根据模式选择加载方式 ==========
                if self.use_variable_length:
                    # 变长模式: 加载原始长度数据，不进行插值
                    data = get_data_variable_length(
                        os.path.join(self.data_root, filename),
                        cache_dir=self.cache_dir,
                        adapt_mode=self._adapt_mode,
                        base_enabled=self._base_enabled,
                        o2pulse_enabled=self._o2pulse_enabled,
                        vco2_enabled=self._vco2_enabled
                    )
                else:
                    # 固定长度模式: 插值到 L_win
                    data = get_data_new(
                        os.path.join(self.data_root, filename),
                        target_length=self.L_win,
                        cache_dir=self.cache_dir,
                        adapt_mode=self._adapt_mode,
                        base_enabled=self._base_enabled,
                        o2pulse_enabled=self._o2pulse_enabled,
                        vco2_enabled=self._vco2_enabled
                    )
                # ==========================================

                if np.all(data == 0): continue

                # [修改] 应用特征选择 (优先使用消融配置，否则使用传入的 feature_indices)
                if self._feature_ablation is not None and self._feature_ablation.enabled:
                    data = data[:, self.selected_feature_indices]
                elif self.feature_indices is not None:
                    data = data[:, self.feature_indices]

                # 加载静态特征 (在添加数据之前检查)
                if self.use_static_features:
                    fname_key = filename[:-5] if filename.endswith('.xlsx') else filename
                    static_feat = self.static_dict.get(fname_key)
                    if static_feat is None:
                        # 如果找不到静态特征
                        if self.pft_enabled and self.skip_missing_pft:
                            # 跳过缺失 PFT 数据的患者
                            continue
                        else:
                            # 使用零向量
                            static_feat = [0.0] * self.num_static_features

                    # 先添加静态特征，再添加动态数据
                    self.raw_staticlist.append(static_feat)

                # 添加动态数据和标签
                self.raw_datalist.append(data)
                self.raw_labellist.append(label_name)
                self.raw_filenames.append(filename)  # 存储文件名

            except:
                print(f"[数据集] 警告: 读取文件失败 {filename}")
                continue

        print(f"[数据集] 原始样本总数: {len(self.raw_datalist)}")

        # [新增] 更新 num_channels 以反映特征选择后的实际数量
        if self._feature_ablation is not None and self._feature_ablation.enabled:
            self.num_channels = len(self.selected_feature_indices)
            print(f"[数据集] 特征选择后: {self.num_channels} 个通道")

        # 3. 划分数据集 - 支持多标签模式
        from sklearn.model_selection import train_test_split

        # 获取随机种子
        random_seed = 3407
        if hasattr(self.config, 'training'):
            random_seed = self.config.training.random_seed
        elif hasattr(self.config, 'random_seed'):
            random_seed = self.config.random_seed

        if self.is_multilabel:
            # 多标签模式: 使用随机划分 (无法使用 stratify)
            # 可选: 使用 iterstrat 包实现多标签分层划分
            print("[数据集] 多标签模式: 使用随机划分")
            self.train_idx, self.test_idx = train_test_split(
                list(range(len(self.raw_datalist))),
                test_size=self.test_ratio,
                random_state=random_seed
            )
        else:
            # 单标签模式: 使用分层划分
            label_indices = [self.label_mapping[l] for l in self.raw_labellist]
            self.train_idx, self.test_idx = train_test_split(
                list(range(len(self.raw_datalist))),
                test_size=self.test_ratio,
                stratify=label_indices,
                random_state=random_seed
            )

        # 4. 计算标准化统计量 (关键: 只用训练集)
        # 4.1 动态特征统计量
        train_data = [self.raw_datalist[i] for i in self.train_idx]
        self.stats = get_feature_statistics(train_data)

        # 4.2 静态特征统计量
        if self.use_static_features:
            train_static = np.array([self.raw_staticlist[i] for i in self.train_idx])
            self.static_stats = {
                'mean': np.mean(train_static, axis=0),
                'std': np.std(train_static, axis=0)
            }
            # 性别列(索引1)不标准化: mean=0, std=1 保持原值
            self.static_stats['mean'][1] = 0
            self.static_stats['std'][1] = 1
            print(f"[数据集] 静态特征统计量 ({self.num_static_features} 维): mean shape={self.static_stats['mean'].shape}")

        # 5. 处理变长序列
        if self.use_variable_length:
            self.L_max = self._compute_max_length_from_raw()
            print(f"[数据集] 变长模式: 最大序列长度 L_max = {self.L_max}")

        print(f"[数据集] 划分完成 - 训练集: {len(self.train_idx)}, 测试集: {len(self.test_idx)}")

    def get_split(self, phase: str):
        """
        返回训练/测试子集的视图 (不复制数据)

        Args:
            phase: "train" 或 "test"

        Returns:
            CPETDatasetSubset: 轻量级子集视图
        """
        if phase == "train":
            indices = self.train_idx
        elif phase == "test":
            indices = self.test_idx
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 'train' or 'test'.")

        return CPETDatasetSubset(self, indices, phase)

    def _compute_max_length_from_raw(self):
        """
        [变长序列] 遍历所有原始数据，找到最大序列长度

        Returns:
            max_len: 数据集中的最大序列长度 (限制不超过 self.max_length)
        """
        max_len = 0
        for data in self.raw_datalist:
            max_len = max(max_len, data.shape[0])

        # 限制最大长度，避免显存溢出
        max_len = min(max_len, self.max_length)

        return max_len

    def _pad_sequence(self, data):
        """
        [变长序列] Zero-padding 到 L_max

        Args:
            data: numpy array [T, C], 已标准化的序列

        Returns:
            padded_data: numpy array [L_max, C], padding后的序列
            length: int, 原始序列长度 (用于 attention mask)
        """
        T, C = data.shape

        if T >= self.L_max:
            # 超过最大长度，截断
            return data[:self.L_max], self.L_max
        else:
            # 不足最大长度，尾部补 0
            padded = np.zeros((self.L_max, C), dtype=data.dtype)
            padded[:T] = data
            return padded, T

    def get_multilabel_encoder(self):
        """
        获取多标签编码器 (延迟初始化)

        Returns:
            encoder: 函数，将标签列表转换为 multi-hot 向量
        """
        if self._multilabel_encoder is None:
            self._multilabel_encoder = create_multilabel_encoder(
                self.label_mapping, self.n_classes
            )
        return self._multilabel_encoder

    def get_co_occurrence_matrix(self):
        """
        获取共现矩阵

        Returns:
            co_matrix: [n_labels, n_labels] numpy数组 或 None
        """
        return self.co_occurrence_matrix

    @staticmethod
    def augment_data(data):
        """
        [Physiological Perturbation]
        模拟生理信号的自然波动和传感器噪声
        """
        # 1. Intensity Scaling (模拟代谢率差异)
        # 将整体信号强度在 85% - 115% 之间缩放
        if np.random.random() < 0.8:
            scale = np.random.uniform(0.85, 1.15)
            data = data * scale

        # 2. Time Shift / Channel Shift (模拟各指标响应延迟的微小差异)
        # 简单实现：整体平移 1-2 个时间点
        if np.random.random() < 0.3:
            shift = np.random.randint(-2, 3)
            data = np.roll(data, shift, axis=0)

        # 3. Gaussian Noise (模拟设备噪声)
        if np.random.random() < 0.5:
            noise = np.random.normal(0, 0.05, data.shape)
            data = data + noise

        return data


def preload_all_data_for_kfold(config, use_variable_length=False, max_length=330,
                                use_static_features=False, feature_indices=None,
                                strict_no_filter=False):
    """
    预加载所有数据用于 K-Fold 交叉验证 (避免每个 fold 重复加载)

    Args:
        config: Config 配置对象
        use_variable_length: 是否使用变长序列模式
        max_length: 变长模式下的最大长度限制
        use_static_features: 是否使用静态特征融合
        feature_indices: 要使用的特征索引列表
        strict_no_filter: True 时任何已标注样本被跳过都直接报错，并输出被跳过样本信息；
                          data_root 中无标签的池子文件会被忽略

    Returns:
        data_cache: 包含所有预加载数据的字典
    """
    print("\n" + "="*60)
    print("[预加载] 开始加载所有数据...")
    print("="*60)

    # 从配置获取参数
    if hasattr(config, 'data'):
        data_root = config.data.data_root
        label_file = config.data.label_file
        L_win = config.data.L_win
        adapt_mode = config.features.adapt_mode
        target_col_name = config.features.target_col_name
        num_channels = config.features.num_channels
        o2pulse_enabled = getattr(config.features, 'o2pulse_enabled', False)
        vco2_enabled = getattr(config.features, 'vco2_enabled', False)

        # PFT 配置
        if hasattr(config.model, 'static_features') and config.model.static_features is not None:
            sf_cfg = config.model.static_features
            pft_enabled = getattr(sf_cfg, 'pft_enabled', False)
            skip_missing_pft = getattr(sf_cfg, 'skip_missing_pft', True)
            pft_file = getattr(sf_cfg, 'pft_file', "xx_path")
        else:
            pft_enabled = False
            skip_missing_pft = True
            pft_file = "xx_path"

        # 任务配置
        task_cfg = getattr(config, 'task', None)
        if task_cfg is not None:
            task_mode = task_cfg.mode
            label_separator = task_cfg.label_separator
            min_label_freq = task_cfg.min_label_freq
        else:
            task_mode = "single_label"
            label_separator = ";"
            min_label_freq = 50
    else:
        raise ValueError("配置对象缺少必要的属性")

    # 设置缓存目录
    cache_dir = os.path.join(os.path.dirname(data_root), "npy_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. 加载标签
    label_dict, label_mapping, co_occurrence = load_labels_unified(
        label_file=label_file,
        task_mode=task_mode,
        target_col_name=target_col_name,
        label_separator=label_separator,
        min_label_freq=min_label_freq
    )

    is_multilabel = (task_mode == "multi_label")
    n_classes = len(label_mapping)

    # ========== [新增] Known-T6 Context: 加载 t6 标签 ==========
    t6_label_dict = None
    t6_label_mapping = None
    t6_n_classes = 0
    if hasattr(config, 'known_t6_context') and config.known_t6_context.enabled:
        t6_source_column = config.known_t6_context.source_column
        t6_label_dict, t6_label_mapping = load_labels(
            label_file=label_file,
            target_col_name=t6_source_column
        )
        t6_n_classes = len(t6_label_mapping)
        print(f"[预加载] Known-T6 Context: 已启用, t6_n_classes={t6_n_classes}")
    # ==========================================================

    if is_multilabel:
        print(f"[预加载] 多标签模式: {n_classes} 个标签")
    else:
        print(f"[预加载] 单标签模式: {n_classes} 个类别")

    # 2. 加载静态特征 (如果启用)
    static_dict = None
    num_static_features = 5
    if use_static_features:
        ehr_dict, ehr_feature_names = load_static_features(label_file)

        if pft_enabled:
            pft_dict, pft_feature_names = load_pft_features(pft_file)

            if pft_dict:
                filename_to_final_id = _build_filename_to_final_id_mapping(label_file)
                static_dict, num_ehr, num_pft = _merge_static_features(
                    ehr_dict, pft_dict, filename_to_final_id,
                    skip_missing_pft=skip_missing_pft
                )
                num_static_features = num_ehr + num_pft
                print(f"[预加载] 静态特征: EHR({num_ehr}) + PFT({num_pft}) = {num_static_features}")
            else:
                static_dict = {k: [v['age'], v['gender'], v['weight'], v['height'], v['bmi']]
                               for k, v in ehr_dict.items()}
                num_static_features = 5
        else:
            static_dict = {k: [v['age'], v['gender'], v['weight'], v['height'], v['bmi']]
                           for k, v in ehr_dict.items()}
            num_static_features = 5

    # 3. 扫描文件并加载数据
    data_files = sorted([f for f in os.listdir(data_root) if f.endswith('.xlsx')])

    raw_datalist = []
    raw_labellist = []
    raw_staticlist = []
    filenames = []
    skipped_records = []

    def record_skip(filename, reason, detail=None):
        skipped_records.append({
            'filename': filename,
            'reason': reason,
            'detail': str(detail) if detail is not None else None
        })

    print(f"[预加载] 正在加载 {len(data_files)} 个文件...")
    for i, filename in enumerate(data_files):
        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{len(data_files)} 个文件")

        filepath = os.path.join(data_root, filename)
        label_name = get_label_for_file(filename, label_dict)

        if label_name is None:
            # data_root is a superset pool; files absent from label_file are
            # intentionally ignored and are not strict-mode failures.
            continue

        try:
            # 根据模式选择加载方式
            if use_variable_length:
                data = get_data_variable_length(
                    filepath,
                    cache_dir=cache_dir,
                    adapt_mode=adapt_mode,
                    o2pulse_enabled=o2pulse_enabled,
                    vco2_enabled=vco2_enabled
                )
            else:
                data = get_data_new(
                    filepath,
                    target_length=L_win,
                    cache_dir=cache_dir,
                    adapt_mode=adapt_mode,
                    o2pulse_enabled=o2pulse_enabled,
                    vco2_enabled=vco2_enabled
                )

            if np.all(data == 0) or np.all(np.isnan(data)):
                record_skip(filename, 'empty_dynamic_data',
                            'all_zero_or_all_nan')
                continue

            if feature_indices is not None:
                data = data[:, feature_indices]

            # 加载静态特征
            static_feat = None
            if use_static_features:
                fname_key = filename[:-5] if filename.endswith('.xlsx') else filename
                static_feat = static_dict.get(fname_key)
                if static_feat is None:
                    if pft_enabled and skip_missing_pft:
                        record_skip(filename, 'missing_static_or_pft_features',
                                    f"lookup_key={fname_key}, pft_enabled={pft_enabled}, skip_missing_pft={skip_missing_pft}")
                        continue
                    else:
                        static_feat = [0.0] * num_static_features

            raw_datalist.append(data)
            raw_labellist.append(label_name)
            raw_staticlist.append(static_feat)
            filenames.append(filename)

        except Exception as e:
            record_skip(filename, 'data_load_exception', repr(e))
            continue

    if strict_no_filter and skipped_records:
        reason_counts = Counter(record['reason'] for record in skipped_records)
        preview = skipped_records[:50]
        preview_lines = [
            f"    - filename={record['filename']}, reason={record['reason']}, detail={record['detail']}"
            for record in preview
        ]
        raise ValueError(
            "[预加载严格模式] 检测到已标注样本在数据加载过程中被筛选/跳过，已停止运行。\n"
            f"  - data_root: {data_root}\n"
            f"  - label_file: {label_file}\n"
            f"  - total_files: {len(data_files)}\n"
            f"  - loaded_samples: {len(raw_datalist)}\n"
            f"  - skipped_samples: {len(skipped_records)}\n"
            f"  - reason_counts: {dict(reason_counts)}\n"
            f"  - skipped_examples(first {len(preview)}):\n"
            + "\n".join(preview_lines)
        )

    print(f"[预加载] 成功加载 {len(raw_datalist)} 个样本")
    print("="*60 + "\n")

    # 构建缓存字典
    data_cache = {
        'raw_datalist': raw_datalist,
        'raw_labellist': raw_labellist,
        'raw_staticlist': raw_staticlist,
        'filenames': filenames,
        'label_mapping': label_mapping,
        'label_dict': label_dict,
        'n_classes': n_classes,
        'is_multilabel': is_multilabel,
        'co_occurrence_matrix': co_occurrence,
        'static_dict': static_dict,
        'num_static_features': num_static_features,
        # [新增] Known-T6 Context 数据
        't6_label_dict': t6_label_dict,
        't6_label_mapping': t6_label_mapping,
        't6_n_classes': t6_n_classes
    }

    return data_cache


class CPETDatasetNewKFold(Dataset):
    """
    支持K折交叉验证的新数据集类

    ** 已适配统一配置系统 (Config) **
    ** 支持: 变长序列、静态特征、多标签、医学先验衍生特征 **

    支持两种模式:
    1. 固定长度模式 (use_variable_length=False):
       所有序列插值到 L_win 长度
    2. 变长序列模式 (use_variable_length=True):
       保留原始长度，通过 zero-padding 对齐到 L_max
    """
    def __init__(self, config, fold_idx=0, n_folds=5, phase="train",
                 random_seed=42, feature_indices=None,
                 use_variable_length=False, max_length=330, use_static_features=False,
                 dev_indices=None, test_indices=None, all_data_cache=None,
                 use_holdout_test=False, train_stats=None, train_static_stats=None,
                 mtl_mode=False):
        """
        Args:
            config: Config 配置对象 (或旧版 args 对象)
            fold_idx: 当前fold索引 (0 to n_folds-1)
            n_folds: 总fold数
            phase: "train" 或 "test"
            random_seed: 随机种子
            feature_indices: 要使用的特征索引列表
            use_variable_length: 是否使用变长序列模式 (默认 False)
            max_length: 变长模式下的最大长度限制 (默认 330)
            use_static_features: 是否使用静态特征融合 (默认 False)
            dev_indices: [新增] 预划分的 Dev_Set 索引列表 (用于 K-Fold)
            test_indices: [新增] 预划分的 Test_Set 索引列表 (独立测试集)
            all_data_cache: [新增] 预加载的全局数据缓存 (避免重复加载)
            use_holdout_test: [新增] 是否直接使用独立测试集 (跳过 K-Fold)
            train_stats: [新增] 预计算的训练集动态特征统计量 (用于测试集归一化)
            train_static_stats: [新增] 预计算的训练集静态特征统计量 (用于测试集归一化)
            mtl_mode: [新增] 是否为 MTL 模式 (默认 False，单任务模式保持原有行为)
        """
        # 兼容 Config 对象和旧版 args 对象
        self._init_from_config(config)

        self.config = config
        self.fold_idx = fold_idx
        self.n_folds = n_folds
        self.phase = phase
        self.random_seed = random_seed
        self.feature_indices = feature_indices

        # ========== [新增] Holdout 测试集参数 ==========
        self.dev_indices = dev_indices        # Dev_Set 索引 (用于 K-Fold)
        self.test_indices = test_indices      # Test_Set 索引 (独立测试集)
        self.all_data_cache = all_data_cache  # 预加载的数据缓存
        self.use_holdout_test = use_holdout_test  # 是否直接使用测试集
        # [新增] 预计算的训练集统计量 (用于测试集归一化)
        self._precomputed_train_stats = train_stats
        self._precomputed_static_stats = train_static_stats
        # [新增] MTL 模式标识 (用于边界检查错误处理)
        self._mtl_mode = mtl_mode
        # =============================================

        # ========== 变长序列参数 ==========
        self.use_variable_length = use_variable_length
        self.max_length = max_length
        self.L_max = None  # 将在数据加载后计算
        # =================================

        # ========== 静态特征参数 ==========
        self.use_static_features = use_static_features
        self.static_dict = None
        self.static_stats = None
        self.num_static_features = 5
        # =================================

        # ========== 多标签参数 ==========
        self.is_multilabel = False
        self.co_occurrence_matrix = None
        self._multilabel_encoder = None
        # =================================

        # ========== Known-T6 Context 参数 (新增) ==========
        self.t6_label_dict = None        # {filename: t6_label_str}
        self.t6_label_mapping = None     # {t6_label_str: index}
        self.t6_n_classes = 0            # t6 类别数
        # =================================================

        # 设置缓存目录
        self.cache_dir = os.path.join(os.path.dirname(self.data_root), "npy_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        if feature_indices is not None:
            self.num_channels = len(feature_indices)
        else:
            self.num_channels = self._num_channels

        print(f"初始化K折数据集 - Fold {fold_idx+1}/{n_folds}, "
              f"阶段: {phase}, 特征数: {self.num_channels}")
        print(f"模式: {'变长序列' if self.use_variable_length else '固定长度'}")
        if use_static_features:
            print(f"静态特征: 已启用")

        self._load_all_data_kfold()

    def _init_from_config(self, config):
        """从配置对象初始化属性"""
        # 检测是否为 Config 对象 (有 data 属性)
        if hasattr(config, 'data'):
            # 新版 Config 对象
            self.data_root = config.data.data_root
            self.label_file = config.data.label_file
            self.L_win = config.data.L_win
            self._target_col_name = config.features.target_col_name
            self._num_channels = config.features.num_channels
            self._adapt_mode = config.features.adapt_mode
            self._o2pulse_enabled = getattr(config.features, 'o2pulse_enabled', False)
            self._vco2_enabled = getattr(config.features, 'vco2_enabled', False)

            # [新增] PFT 配置
            if hasattr(config.model, 'static_features') and config.model.static_features is not None:
                sf_cfg = config.model.static_features
                self.pft_enabled = getattr(sf_cfg, 'pft_enabled', False)
                self.skip_missing_pft = getattr(sf_cfg, 'skip_missing_pft', True)
                self._pft_file = getattr(sf_cfg, 'pft_file', "xx_path")
            else:
                self.pft_enabled = False
                self.skip_missing_pft = True
                self._pft_file = "xx_path"

            # [新增] Known-T6 Context 配置
            if hasattr(config, 'known_t6_context') and config.known_t6_context is not None:
                self.use_known_t6_context = config.known_t6_context.enabled
                self._t6_source_column = config.known_t6_context.source_column
            else:
                self.use_known_t6_context = False
                self._t6_source_column = "匹配的第一大类"
        else:
            # 旧版 args 对象 (兼容)
            self.data_root = config.data_root
            self.label_file = config.label_file
            self.L_win = config.L_win
            self._target_col_name = getattr(config, 'target_col_name', '匹配的第一大类')
            self._num_channels = getattr(config, 'num_channels', 26)
            self._adapt_mode = getattr(config, 'adapt_mode', 'full')
            self._o2pulse_enabled = getattr(config, 'o2pulse_enabled', False)
            self._vco2_enabled = getattr(config, 'vco2_enabled', False)
            self.pft_enabled = False
            self.skip_missing_pft = True
            self._pft_file = "xx_path"
            # [新增] Known-T6 Context (旧版 args 不支持)
            self.use_known_t6_context = False
            self._t6_source_column = "匹配的第一大类"

    def _load_all_data_kfold(self):
        """
        加载所有数据并进行K折划分

        支持两种模式:
        1. 传统模式: 直接对全部数据进行 K-Fold 划分
        2. Holdout 模式: 使用预划分的 dev_indices 在 Dev_Set 上进行 K-Fold
        """
        # ========== [新增] 检查是否使用预加载的数据缓存 ==========
        if self.all_data_cache is not None:
            # 使用预加载的数据缓存 (避免重复加载)
            raw_datalist = self.all_data_cache['raw_datalist']
            raw_labellist = self.all_data_cache['raw_labellist']
            raw_staticlist = self.all_data_cache['raw_staticlist']
            filenames = self.all_data_cache['filenames']
            self.label_mapping = self.all_data_cache['label_mapping']
            self.n_classes = self.all_data_cache['n_classes']
            self.is_multilabel = self.all_data_cache['is_multilabel']
            self.co_occurrence_matrix = self.all_data_cache.get('co_occurrence_matrix')
            self.static_dict = self.all_data_cache.get('static_dict')
            self.num_static_features = self.all_data_cache.get('num_static_features', 5)
            # [新增] 从缓存加载 t6 标签
            self.t6_label_dict = self.all_data_cache.get('t6_label_dict')
            self.t6_label_mapping = self.all_data_cache.get('t6_label_mapping')
            self.t6_n_classes = self.all_data_cache.get('t6_n_classes', 0)
            print(f"[K-Fold] 使用预加载数据缓存: {len(raw_datalist)} 个样本")
        else:
            # ========== 传统模式: 加载所有数据 ==========
            # [新增] 确定任务模式
            task_cfg = getattr(self.config, 'task', None)
            if task_cfg is not None:
                task_mode = task_cfg.mode
                label_separator = task_cfg.label_separator
                min_label_freq = task_cfg.min_label_freq
            else:
                task_mode = "single_label"
                label_separator = ";"
                min_label_freq = 50

            # 1. 加载标签 (使用统一接口)
            self.label_dict, self.label_mapping, co_occurrence = load_labels_unified(
                label_file=self.label_file,
                task_mode=task_mode,
                target_col_name=self._target_col_name,
                label_separator=label_separator,
                min_label_freq=min_label_freq
            )

            # 设置多标签标志
            self.is_multilabel = (task_mode == "multi_label")

            if self.is_multilabel and co_occurrence is not None:
                self.co_occurrence_matrix = co_occurrence
                self.config.co_occurrence_matrix = co_occurrence
                self.config.is_multilabel = True
                print(f"[K-Fold] 多标签模式: {len(self.label_mapping)} 个标签")
            else:
                self.co_occurrence_matrix = None
                self.config.is_multilabel = False

            self.n_classes = len(self.label_mapping)

            # ========== [新增] Known-T6 Context: 独立加载 t6 标签 ==========
            if self.use_known_t6_context:
                # 使用 load_labels 加载 t6 标签 (单标签模式，返回2个值)
                self.t6_label_dict, self.t6_label_mapping = load_labels(
                    label_file=self.label_file,
                    target_col_name=self._t6_source_column
                )
                self.t6_n_classes = len(self.t6_label_mapping)
                print(f"[Known-T6 Context] 已启用: source_column={self._t6_source_column}, t6_n_classes={self.t6_n_classes}")
            # ================================================================

            # 1.5 加载静态特征 (如果启用)
            if self.use_static_features:
                # 加载 EHR 特征
                ehr_dict, ehr_feature_names = load_static_features(self.label_file)

                if self.pft_enabled:
                    # 加载 PFT 特征
                    pft_dict, pft_feature_names = load_pft_features(self._pft_file)

                    if pft_dict:
                        # 建立 filename -> final_编号 映射
                        filename_to_final_id = _build_filename_to_final_id_mapping(self.label_file)

                        # 合并特征
                        self.static_dict, num_ehr, num_pft = _merge_static_features(
                            ehr_dict, pft_dict, filename_to_final_id,
                            skip_missing_pft=self.skip_missing_pft
                        )
                        self.static_feature_names = ehr_feature_names + pft_feature_names
                        self.num_static_features = num_ehr + num_pft
                        print(f"[K-Fold] 静态特征: EHR({num_ehr}) + PFT({num_pft}) = {self.num_static_features}")
                    else:
                        # PFT 加载失败，仅使用 EHR
                        print(f"[K-Fold] PFT 加载失败，仅使用 EHR 特征")
                        self.static_dict = {k: [v['age'], v['gender'], v['weight'], v['height'], v['bmi']]
                                            for k, v in ehr_dict.items()}
                        self.static_feature_names = ehr_feature_names
                        self.num_static_features = 5
                else:
                    # 仅使用 EHR 特征
                    self.static_dict = {k: [v['age'], v['gender'], v['weight'], v['height'], v['bmi']]
                                        for k, v in ehr_dict.items()}
                    self.static_feature_names = ehr_feature_names
                    self.num_static_features = 5

            # 2. 扫描文件
            data_files = sorted([f for f in os.listdir(self.data_root) if f.endswith('.xlsx')])

            raw_datalist = []
            raw_labellist = []
            raw_staticlist = []
            filenames = []

            print(f"加载数据 (Fold {self.fold_idx+1})...")
            for i, filename in enumerate(data_files):
                if (i + 1) % 500 == 0:
                    print(f"  已处理 {i+1}/{len(data_files)} 个文件")

                filepath = os.path.join(self.data_root, filename)
                label_name = get_label_for_file(filename, self.label_dict)

                if label_name is None:
                    continue

                try:
                    # ========== 根据模式选择加载方式 ==========
                    if self.use_variable_length:
                        # 变长模式: 加载原始长度数据
                        data = get_data_variable_length(
                            filepath,
                            cache_dir=self.cache_dir,
                            adapt_mode=self._adapt_mode,
                            o2pulse_enabled=self._o2pulse_enabled,
                            vco2_enabled=self._vco2_enabled
                        )
                    else:
                        # 固定长度模式: 插值到 L_win
                        data = get_data_new(
                            filepath,
                            target_length=self.L_win,
                            cache_dir=self.cache_dir,
                            adapt_mode=self._adapt_mode,
                            o2pulse_enabled=self._o2pulse_enabled,
                            vco2_enabled=self._vco2_enabled
                        )
                    # ==========================================

                    if np.all(data == 0) or np.all(np.isnan(data)):
                        continue

                    if self.feature_indices is not None:
                        data = data[:, self.feature_indices]

                    # ========== 加载静态特征 (如果启用) ==========
                    static_feat = None
                    if self.use_static_features:
                        fname_key = filename[:-5] if filename.endswith('.xlsx') else filename
                        static_feat = self.static_dict.get(fname_key)
                        if static_feat is None:
                            if self.pft_enabled and self.skip_missing_pft:
                                continue  # 跳过缺失 PFT 数据的患者
                            else:
                                static_feat = [0.0] * self.num_static_features
                    # ============================================

                    raw_datalist.append(data)
                    raw_labellist.append(label_name)
                    raw_staticlist.append(static_feat)
                    filenames.append(filename)

                except Exception as e:
                    continue

            print(f"成功加载 {len(raw_datalist)} 个样本")

        # ========== [新增] Holdout 模式: 使用预划分的 Dev_Set ==========
        if self.dev_indices is not None:
            # ========== [新增] 边界检查 ==========
            n_loaded = len(raw_datalist)
            max_dev_idx = max(self.dev_indices) if self.dev_indices else 0
            max_test_idx = max(self.test_indices) if self.test_indices else 0

            if max_dev_idx >= n_loaded or max_test_idx >= n_loaded:
                # [关键] MTL 模式报错，单任务模式回退
                if self._mtl_mode:
                    # MTL 模式：必须报错，强制用户生成正确的划分文件
                    raise IndexError(
                        f"\n[Holdout Error] MTL 模式划分文件索引超出范围!\n"
                        f"  - 划分文件期望: {len(self.dev_indices) + len(self.test_indices)} 样本\n"
                        f"  - 实际加载: {n_loaded} 样本\n"
                        f"  - 最大 dev_idx: {max_dev_idx}, 最大 test_idx: {max_test_idx}\n"
                        f"  - 解决方案: 运行 'python scripts/create_holdout_split_mtl.py' 生成 MTL 专用划分文件\n"
                        f"  - 或者设置 config: mtl.holdout.enabled=false 禁用 Holdout 模式"
                    )
                else:
                    # 单任务模式：保持原有行为，回退到传统 K-Fold
                    print(f"[Holdout Warning] 划分文件索引超出当前数据范围!")
                    print(f"  - 划分文件期望: {len(self.dev_indices) + len(self.test_indices)} 样本")
                    print(f"  - 实际加载: {n_loaded} 样本")
                    print(f"  - 最大 dev_idx: {max_dev_idx}, 最大 test_idx: {max_test_idx}")
                    print(f"[Holdout Warning] 回退到传统 K-Fold 模式 (不使用预划分)")

                    # 回退到传统模式
                    self.dev_indices = None
                    self.test_indices = None
                    dev_datalist = raw_datalist
                    dev_labellist = raw_labellist
                    dev_filenames = filenames
                    dev_staticlist = raw_staticlist if self.use_static_features else None

            if self.dev_indices is None:  # 已经回退，跳过后续处理
                pass  # 继续到后面的 K-Fold 逻辑
            elif self.use_holdout_test and self.test_indices is not None:
                print(f"[Holdout] 直接使用独立测试集: {len(self.test_indices)} 个样本")
                # 直接使用 test_indices 中的数据
                train_raw = [raw_datalist[i] for i in self.test_indices]
                train_labellist = [raw_labellist[i] for i in self.test_indices]
                train_filenames = [filenames[i] for i in self.test_indices]
                if self.use_static_features and raw_staticlist:
                    train_static = [raw_staticlist[i] for i in self.test_indices]
                else:
                    train_static = None

                # 设置数据列表 (测试集作为 "train"，因为没有验证集)
                self.train_datalist = []

                # [关键修复] 使用训练集的统计量而非测试集自身的统计量
                if self._precomputed_train_stats is not None:
                    stats = self._precomputed_train_stats
                    print(f"[Holdout] 使用预计算的训练集统计量进行归一化")
                    # [调试] 打印统计量信息
                    print(f"  - 统计量键: {list(stats.keys())}")
                else:
                    stats = get_feature_statistics(train_raw)  # 兼容旧逻辑
                    print(f"[警告] 未提供训练集统计量，使用测试集自身统计量 (不推荐)")

                for d in train_raw:
                    norm_d, _ = normalize_features(d, method='robust', feature_stats=stats)
                    self.train_datalist.append(norm_d)

                self.test_datalist = []  # 空的验证集
                self.train_labellist = train_labellist
                self.test_labellist = []
                self.train_filenames = train_filenames
                self.test_filenames = []
                self.train_idx = list(range(len(train_raw)))
                self.test_idx = []
                self.stats = stats

                # 标准化静态特征
                if self.use_static_features and train_static is not None:
                    # [关键修复] 使用训练集的静态特征统计量
                    if self._precomputed_static_stats is not None:
                        self.static_stats = self._precomputed_static_stats
                        print(f"[Holdout] 使用预计算的静态特征统计量")
                    else:
                        train_static_arr = np.array(train_static, dtype=np.float32)
                        self.static_stats = {
                            'mean': np.mean(train_static_arr, axis=0),
                            'std': np.std(train_static_arr, axis=0) + 1e-8
                        }
                        if self.static_stats['mean'].shape[0] > 1:
                            self.static_stats['mean'][1] = 0
                            self.static_stats['std'][1] = 1
                        print(f"[警告] 未提供静态特征统计量，使用测试集自身统计量 (不推荐)")
                    self.train_staticlist = train_static
                    self.test_staticlist = []

                # 变长序列
                if self.use_variable_length:
                    self.L_max = self._compute_max_length()
                    print(f"Holdout 测试集 - 变长模式: L_max = {self.L_max}")

                print(f"Holdout 测试集 - 样本数: {len(self.train_datalist)}")
                return  # 直接返回，跳过 K-Fold 逻辑
            # ===========================================

            # 使用预划分的 Dev_Set 索引 (边界检查已通过)
            print(f"[Holdout] 使用预划分的 Dev_Set: {len(self.dev_indices)} 个样本")
            dev_datalist = [raw_datalist[i] for i in self.dev_indices]
            dev_labellist = [raw_labellist[i] for i in self.dev_indices]
            dev_filenames = [filenames[i] for i in self.dev_indices]
            if self.use_static_features and raw_staticlist:
                dev_staticlist = [raw_staticlist[i] for i in self.dev_indices]
            else:
                dev_staticlist = None

            # 保存原始数据引用 (用于后续 Test_Set 评估)
            self._raw_datalist = raw_datalist
            self._raw_labellist = raw_labellist
            self._raw_staticlist = raw_staticlist
            self._filenames = filenames
            self._test_indices = self.test_indices  # 独立测试集索引
        else:
            # 回退: 使用全部数据 (传统 K-Fold)
            dev_datalist = raw_datalist
            dev_labellist = raw_labellist
            dev_filenames = filenames
            dev_staticlist = raw_staticlist if self.use_static_features else None

        # ========== K折划分 (在 Dev_Set 上) ==========
        if self.is_multilabel:
            # 多标签模式: 使用普通 KFold
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
            splits = list(kf.split(dev_datalist))
        else:
            # 单标签模式: 使用分层 KFold
            label_indices = [self.label_mapping[label] for label in dev_labellist]
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                  random_state=self.random_seed)
            splits = list(skf.split(dev_datalist, label_indices))

        # 获取当前fold的训练集和测试集索引
        train_idx, val_idx = splits[self.fold_idx]
        self.train_idx = train_idx
        self.test_idx = val_idx

        # 4. 根据phase选择数据
        train_raw = [dev_datalist[i] for i in train_idx]
        test_raw = [dev_datalist[i] for i in val_idx]

        self.train_labellist = [dev_labellist[i] for i in train_idx]
        self.test_labellist = [dev_labellist[i] for i in val_idx]

        # 保存文件名列表 (用于错误分析)
        self.train_filenames = [dev_filenames[i] for i in train_idx]
        self.test_filenames = [dev_filenames[i] for i in val_idx]

        if self.use_static_features and dev_staticlist is not None:
            train_static = [dev_staticlist[i] for i in train_idx]
            test_static = [dev_staticlist[i] for i in val_idx]

            # 保存静态特征列表
            self.train_staticlist = train_static
            self.test_staticlist = test_static

        # ========== 标准化逻辑 ==========
        # 仅在当前 Fold 的训练集上计算统计量
        stats = get_feature_statistics(train_raw)

        # 标准化动态特征
        self.train_datalist = []
        for d in train_raw:
            norm_d, _ = normalize_features(d, method='robust', feature_stats=stats)
            self.train_datalist.append(norm_d)

        self.test_datalist = []
        for d in test_raw:
            norm_d, _ = normalize_features(d, method='robust', feature_stats=stats)
            self.test_datalist.append(norm_d)

        self.stats = stats

        # 标准化静态特征 (使用训练集统计量)
        if self.use_static_features:
            train_static_arr = np.array(train_static, dtype=np.float32)
            self.static_stats = {
                'mean': np.mean(train_static_arr, axis=0),
                'std': np.std(train_static_arr, axis=0) + 1e-8
            }
            # 性别不标准化 (第1列)
            if self.static_stats['mean'].shape[0] > 1:
                self.static_stats['mean'][1] = 0
                self.static_stats['std'][1] = 1
        # =================================

        # ========== 变长序列: 计算最大长度 ==========
        if self.use_variable_length:
            self.L_max = self._compute_max_length()
            print(f"Fold {self.fold_idx+1} - 变长模式: L_max = {self.L_max}")
        # ==========================================

        print(f"Fold {self.fold_idx+1} - 训练集: {len(self.train_datalist)}, "
              f"测试集: {len(self.test_datalist)}")

    def _compute_max_length(self):
        """遍历训练集数据，找到最大序列长度"""
        max_len = 0
        for data in self.train_datalist:
            max_len = max(max_len, data.shape[0])
        max_len = min(max_len, self.max_length)
        return max_len

    def _pad_sequence(self, data):
        """Zero-padding 到 L_max"""
        T, C = data.shape

        if T >= self.L_max:
            return data[:self.L_max], self.L_max
        else:
            padded = np.zeros((self.L_max, C), dtype=data.dtype)
            padded[:T] = data
            return padded, T

    def get_multilabel_encoder(self):
        """获取多标签编码器 (延迟初始化)"""
        if self._multilabel_encoder is None and self.is_multilabel:
            self._multilabel_encoder = create_multilabel_encoder(self.label_mapping)
        return self._multilabel_encoder

    def __len__(self):
        if self.phase == "train":
            return len(self.train_datalist)
        else:
            return len(self.test_datalist)

    @property
    def labellist(self):
        """返回当前阶段的标签列表 (用于权重计算)"""
        if self.phase == "train":
            return self.train_labellist
        else:
            return self.test_labellist

    @property
    def filenames_list(self):
        """返回当前阶段的文件名列表"""
        if self.phase == "train":
            return self.train_filenames
        else:
            return self.test_filenames

    def __getitem__(self, idx):
        if self.phase == 'train':
            datalist = self.train_datalist
            labellist = self.train_labellist
            staticlist = getattr(self, 'train_staticlist', None)
            filenames = getattr(self, 'train_filenames', None)
        else:
            datalist = self.test_datalist
            labellist = self.test_labellist
            staticlist = getattr(self, 'test_staticlist', None)
            filenames = getattr(self, 'test_filenames', None)

        w_data = datalist[idx]

        # ========== 获取标签 (支持多标签模式) ==========
        if self.is_multilabel:
            # 多标签模式: 返回 multi-hot float tensor
            label_names = labellist[idx]  # List[str] 或 str
            if isinstance(label_names, str):
                label_names = [label_names]
            encoder = self.get_multilabel_encoder()
            label = encoder(label_names)  # [n_classes] multi-hot
            label_tensor = torch.as_tensor(label.astype(np.float32))
        else:
            # 单标签模式: 返回索引 long tensor
            label_name = labellist[idx]  # str
            label_idx = self.label_mapping[label_name]
            label_tensor = torch.tensor(label_idx, dtype=torch.long)
        # ==================================================

        # ========== [新增] Known-T6 Context: 获取 t6 one-hot ==========
        t6_onehot = None
        if self.use_known_t6_context and filenames is not None:
            filename = filenames[idx]
            # 处理文件名格式 (移除 .xlsx 后缀)
            lookup_key = filename[:-5] if filename.endswith('.xlsx') else filename
            t6_label_str = self.t6_label_dict.get(lookup_key)
            if t6_label_str is not None and t6_label_str in self.t6_label_mapping:
                t6_idx = self.t6_label_mapping[t6_label_str]
                t6_onehot = torch.zeros(self.t6_n_classes, dtype=torch.float32)
                t6_onehot[t6_idx] = 1.0
            else:
                # Fallback: 生成零向量
                t6_onehot = torch.zeros(self.t6_n_classes, dtype=torch.float32)
        # ================================================================

        # ========== 变长序列处理 ==========
        if self.use_variable_length:
            padded_data, length = self._pad_sequence(w_data)

            # [修改] 根据 t6 context 状态返回不同元素数量
            if self.use_static_features and staticlist is not None:
                # 获取静态特征并标准化
                static = np.array(staticlist[idx], dtype=np.float32)
                static_norm = (static - self.static_stats['mean']) / self.static_stats['std']
                if t6_onehot is not None:
                    # 5 元素: (data, length, static, t6_onehot, label)
                    return (
                        torch.as_tensor(padded_data.astype(np.float32)),
                        torch.tensor(length, dtype=torch.long),
                        torch.as_tensor(static_norm.astype(np.float32)),
                        t6_onehot,
                        label_tensor
                    )
                else:
                    return (
                        torch.as_tensor(padded_data.astype(np.float32)),
                        torch.tensor(length, dtype=torch.long),
                        torch.as_tensor(static_norm.astype(np.float32)),
                        label_tensor
                    )
            else:
                if t6_onehot is not None:
                    # 4 元素: (data, length, t6_onehot, label)
                    return (
                        torch.as_tensor(padded_data.astype(np.float32)),
                        torch.tensor(length, dtype=torch.long),
                        t6_onehot,
                        label_tensor
                    )
                else:
                    return (
                        torch.as_tensor(padded_data.astype(np.float32)),
                        torch.tensor(length, dtype=torch.long),
                        label_tensor
                    )
        else:
            # 固定长度模式
            if self.use_static_features and staticlist is not None:
                # 获取静态特征并标准化
                static = np.array(staticlist[idx], dtype=np.float32)
                static_norm = (static - self.static_stats['mean']) / self.static_stats['std']
                if t6_onehot is not None:
                    # 4 元素: (data, static, t6_onehot, label)
                    return (
                        torch.as_tensor(w_data.astype(np.float32)),
                        torch.as_tensor(static_norm.astype(np.float32)),
                        t6_onehot,
                        label_tensor
                    )
                else:
                    return (
                        torch.as_tensor(w_data.astype(np.float32)),
                        torch.as_tensor(static_norm.astype(np.float32)),
                        label_tensor
                    )
            else:
                if t6_onehot is not None:
                    # 3 元素: (data, t6_onehot, label)
                    return (
                        torch.as_tensor(w_data.astype(np.float32)),
                        t6_onehot,
                        label_tensor
                    )
                else:
                    return torch.as_tensor(w_data.astype(np.float32)), label_tensor
        # ==================================


if __name__ == "__main__":
    # 测试数据集类
    print("="*80)
    print("测试数据集类")
    print("="*80)

    # 软编码：从配置文件加载
    from config import Config
    config = Config.load()

    args = config

    try:
        # 测试基本数据集
        print("\n测试基本数据集...")
        dataset = CPETDatasetNew(args, test_ratio=0.2, phase="train")

        print(f"\n数据集大小: {len(dataset)}")
        print(f"类别数: {dataset.n_classes}")

        # 测试获取一个样本
        if len(dataset) > 0:
            data, label = dataset[0]
            print(f"\n样本数据:")
            print(f"  数据维度: {data.shape}")
            print(f"  标签维度: {label.shape}")
            print(f"  标签索引: {torch.argmax(label).item()}")

    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 别名函数 (供 MTL 使用)
# =============================================================================
preload_all_data = preload_all_data_for_kfold
