"""
MTL 数据集类 - 多任务学习标签适配
==================================

扩展 CPETDatasetNewKFold，支持:
- 多任务标签返回格式: {"t1": int, ..., "t6": int}
- 任务掩码: {"t1": 1, ..., "t6": 1}
- MTL 专用 collate_fn

不破坏现有单任务数据集。

创建日期: 2026-04-14
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import sys

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入现有模块
from dataset_new import CPETDatasetNewKFold, collate_fn_variable_length
from label_extractor import AVAILABLE_LABEL_COLUMNS, load_labels, create_label_encoder
from feature_mapping import normalize_features


# =============================================================================
# MTL 标签列定义 (对应 6 个任务)
# =============================================================================

MTL_LABEL_COLUMNS = {
    "t1": "运动心功能分级",           # Alpha 分支, 3 分类, CE
    "t2": "运动耐量",                 # Beta 分支, 3 分类, LDAM
    "t3": "标准心电运动负荷试验",     # Beta 分支, 2 分类, BCE
    "t4": "运动中换气肺功能",         # Beta 分支, 2 分类, LDAM
    "t5": "心率储备",                 # Beta 分支, 2 分类, LDAM
    "t6": "匹配的第一大类"            # Alpha 分支, 多分类, CE + KD
}


# =============================================================================
# MTL 专用 collate_fn
# =============================================================================

def collate_fn_mtl(batch):
    """
    处理 MTL 模式的 batch

    Args:
        batch: list of dicts
            {
                "x_dyn": Tensor[L, C],
                "x_static": Tensor[5],
                "lengths": Tensor (变长模式),
                "labels": {"t1": int, ..., "t6": int},
                "label_mask": {"t1": 1, ..., "t6": 1}
            }

    Returns:
        dict:
            {
                "x_dyn": Tensor[B, L_max, C],
                "x_static": Tensor[B, 5],
                "lengths": Tensor[B] (变长模式),
                "labels": {"t1": Tensor[B], ..., "t6": Tensor[B]},
                "label_mask": {"t1": Tensor[B], ..., "t6": Tensor[B]}
            }
    """
    # 检测是否为变长模式
    has_lengths = "lengths" in batch[0]

    # 堆叠动态数据
    x_dyn_list = [item["x_dyn"] for item in batch]
    x_dyn_batch = torch.stack(x_dyn_list, dim=0)  # [B, L, C]

    # 堆叠静态数据
    if "x_static" in batch[0]:
        x_static_list = [item["x_static"] for item in batch]
        x_static_batch = torch.stack(x_static_list, dim=0)  # [B, 5]
    else:
        x_static_batch = None

    # 长度 (变长模式)
    if has_lengths:
        lengths_list = [item["lengths"] for item in batch]
        lengths_batch = torch.stack(lengths_list, dim=0)  # [B]
    else:
        lengths_batch = None

    # 多任务标签
    task_keys = list(MTL_LABEL_COLUMNS.keys())
    labels_batch = {}
    label_mask_batch = {}

    for task_key in task_keys:
        label_list = [item["labels"][task_key] for item in batch]
        mask_list = [item["label_mask"][task_key] for item in batch]

        labels_batch[task_key] = torch.stack(label_list, dim=0)  # [B]
        label_mask_batch[task_key] = torch.stack(mask_list, dim=0)  # [B]

    result = {
        "x_dyn": x_dyn_batch,
        "labels": labels_batch,
        "label_mask": label_mask_batch
    }

    if x_static_batch is not None:
        result["x_static"] = x_static_batch

    if lengths_batch is not None:
        result["lengths"] = lengths_batch

    # Optional per-sample metadata for downstream holdout prediction exports.
    metadata_keys = ["filename", "patient_id", "age", "sex"]
    for key in metadata_keys:
        if key in batch[0]:
            result[key] = [item.get(key) for item in batch]

    return result


# =============================================================================
# MTL 标签加载
# =============================================================================

def load_mtl_labels(
    label_file: str,
    header_row: int = 1,
    min_label_freq: int = 50
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, List[int]]], Dict[str, Counter]]:
    """
    加载所有任务的标签

    Args:
        label_file: 标签文件路径
        header_row: 表头行号
        min_label_freq: 最小标签频次阈值

    Returns:
        filename_to_labels: {"filename": {"t1": label_str, ..., "t6": label_str}}
        task_label_mappings: {"t1": {"label_str": idx, ...}, ...}
        task_class_counts: {"t1": Counter({0: N0, 1: N1, ...}), ...}
    """
    # 读取 Excel
    df = pd.read_excel(label_file, engine='openpyxl', header=header_row)

    # 动态查找文件名列 (参考 label_extractor.py 的方法)
    filename_col = None
    for col in df.columns:
        col_str = str(col)
        if '匹配' in col_str and 'Excel' in col_str and '文件' in col_str:
            filename_col = col
            break

    if filename_col is None:
        print("[Warning] 未找到文件名列 (包含 '匹配'+'Excel'+'文件' 的列)")
    else:
        print(f"[MTL] 使用文件名列: {filename_col}")

    filename_to_labels = {}
    task_label_mappings = {}
    task_class_counts = {}

    # 遍历任务
    for task_key, label_col in MTL_LABEL_COLUMNS.items():
        if label_col not in df.columns:
            print(f"[Warning] 任务 {task_key} 标签列 '{label_col}' 不存在，跳过")
            continue

        # 获取有效标签
        valid_labels = df[label_col].dropna().astype(str).str.strip()
        valid_labels = valid_labels[valid_labels != '']

        # 统计频次
        label_counter = Counter(valid_labels)

        # 过滤低频标签
        filtered_labels = {k: v for k, v in label_counter.items() if v >= min_label_freq}

        if len(filtered_labels) == 0:
            print(f"[Warning] 任务 {task_key} 无有效标签 (min_freq={min_label_freq})")
            continue

        # 创建映射
        sorted_labels = sorted(filtered_labels.keys())
        label_mapping = {label: idx for idx, label in enumerate(sorted_labels)}

        task_label_mappings[task_key] = label_mapping
        task_class_counts[task_key] = Counter({label_mapping[k]: v for k, v in filtered_labels.items()})

    # 按文件名构建标签字典
    if filename_col is None:
        print("[Warning] 无法构建 filename_to_labels (缺少文件名列)")
    else:
        for idx, row in df.iterrows():
            filename = row.get(filename_col, '')
            if not filename or pd.isna(filename):
                continue

            # 清理文件名
            filename = str(filename).strip()

            # 获取各任务标签
            task_labels = {}
            for task_key, label_col in MTL_LABEL_COLUMNS.items():
                if label_col in df.columns:
                    label_val = row.get(label_col, '')
                    if pd.notna(label_val) and str(label_val).strip():
                        task_labels[task_key] = str(label_val).strip()

            if task_labels:
                filename_to_labels[filename] = task_labels

    print(f"[MTL] 加载 {len(filename_to_labels)} 个样本的多任务标签")
    for task_key in task_label_mappings.keys():
        print(f"  - {task_key}: {len(task_label_mappings[task_key])} 类, 分布: {task_class_counts[task_key]}")

    return filename_to_labels, task_label_mappings, task_class_counts


# =============================================================================
# MTL 数据集类
# =============================================================================

class CPETDatasetMTL(CPETDatasetNewKFold):
    """
    MTL 数据集类 - 多任务学习标签适配

    继承 CPETDatasetNewKFold，扩展:
    - mtl_mode 参数切换
    - 多任务标签返回格式
    - 任务掩码支持

    Args:
        config: Config 配置对象
        fold_idx: Fold 编号
        n_folds: 总 Fold 数
        phase: "train" 或 "test"
        mtl_mode: 是否启用 MTL 模式 (默认 False)
        task_keys: 要使用的任务列表 (默认 ["t1", ..., "t6"])
        其他参数同 CPETDatasetNewKFold
    """

    def __init__(
        self,
        config,
        fold_idx=0,
        n_folds=5,
        phase="train",
        random_seed=42,
        feature_indices=None,
        use_variable_length=False,
        max_length=330,
        use_static_features=True,
        dev_indices=None,
        test_indices=None,
        all_data_cache=None,
        use_holdout_test=False,
        train_stats=None,
        train_static_stats=None,
        mtl_mode=True,
        task_keys=None
    ):
        """初始化 MTL 数据集"""
        # 调用父类初始化
        super().__init__(
            config=config,
            fold_idx=fold_idx,
            n_folds=n_folds,
            phase=phase,
            random_seed=random_seed,
            feature_indices=feature_indices,
            use_variable_length=use_variable_length,
            max_length=max_length,
            use_static_features=use_static_features,
            dev_indices=dev_indices,
            test_indices=test_indices,
            all_data_cache=all_data_cache,
            use_holdout_test=use_holdout_test,
            train_stats=train_stats,
            train_static_stats=train_static_stats,
            mtl_mode=True  # [新增] MTL 模式标识，用于边界检查报错
        )

        # MTL 参数
        self.mtl_mode = mtl_mode
        self.task_keys = task_keys if task_keys else list(MTL_LABEL_COLUMNS.keys())

        # 加载 MTL 标签 (如果 mtl_mode=True)
        if self.mtl_mode:
            self._load_mtl_labels()

    def _load_mtl_labels(self):
        """加载多任务标签"""
        # 加载标签
        filename_to_labels, task_label_mappings, task_class_counts = load_mtl_labels(
            self.label_file,
            min_label_freq=50
        )

        self._mtl_filename_to_labels = filename_to_labels
        self._mtl_label_mappings = task_label_mappings
        self._mtl_class_counts = task_class_counts

        # 构建 task_specs 需要的统计信息
        self.task_stats = {}
        for task_key in self.task_keys:
            if task_key in task_class_counts:
                counter = task_class_counts[task_key]
                self.task_stats[task_key] = {
                    "num_classes": len(counter),
                    "class_counts": [counter[i] for i in range(len(counter))],
                    "label_mapping": task_label_mappings[task_key]
                }

        print(f"[MTL Dataset] 加载 {len(self.task_keys)} 个任务的标签统计")

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            MTL 模式:
                {
                    "x_dyn": Tensor[L, C],
                    "x_static": Tensor[5],
                    "lengths": Tensor (变长模式),
                    "labels": {"t1": LongTensor, ..., "t6": LongTensor},
                    "label_mask": {"t1": Tensor(1), ..., "t6": Tensor(1)}
                }
            单任务模式:
                同父类返回格式
        """
        if not self.mtl_mode:
            return super().__getitem__(idx)

        # ========== 获取动态数据 (父类已标准化) ==========
        if self.phase == 'train':
            datalist = self.train_datalist
            staticlist = getattr(self, 'train_staticlist', None)
            filenames_list = getattr(self, 'train_filenames', [])
        else:
            datalist = self.test_datalist
            staticlist = getattr(self, 'test_staticlist', None)
            filenames_list = getattr(self, 'test_filenames', [])

        # 父类 train_datalist 已标准化，直接使用
        w_data = datalist[idx]

        # ========== 变长序列处理 ==========
        if self.use_variable_length:
            # 父类已处理 padding
            padded_data, length = self._pad_sequence(w_data)
            x_dyn = torch.as_tensor(padded_data.astype(np.float32))
            lengths = torch.tensor(length, dtype=torch.long)
        else:
            x_dyn = torch.as_tensor(w_data.astype(np.float32))
            lengths = None

        # ========== 静态特征 (父类已标准化) ==========
        if self.use_static_features and staticlist is not None:
            static = np.array(staticlist[idx], dtype=np.float32)
            if self.static_stats is not None:
                static_norm = (static - self.static_stats['mean']) / (self.static_stats['std'] + 1e-8)
            else:
                static_norm = static
            x_static = torch.as_tensor(static_norm.astype(np.float32))
        else:
            x_static = None

        # ========== 多任务标签 ==========
        # 获取文件名 (用于查找标签)
        filename = filenames_list[idx] if idx < len(filenames_list) else None

        labels_dict = {}
        label_mask_dict = {}

        if filename and filename in self._mtl_filename_to_labels:
            task_labels = self._mtl_filename_to_labels[filename]

            for task_key in self.task_keys:
                if task_key in task_labels and task_key in self._mtl_label_mappings:
                    label_str = task_labels[task_key]
                    label_idx = self._mtl_label_mappings[task_key].get(label_str, -1)

                    if label_idx >= 0:
                        labels_dict[task_key] = torch.tensor(label_idx, dtype=torch.long)
                        label_mask_dict[task_key] = torch.tensor(1, dtype=torch.float32)
                    else:
                        # 标签不在映射中 (低频标签)
                        labels_dict[task_key] = torch.tensor(0, dtype=torch.long)  # 默认值
                        label_mask_dict[task_key] = torch.tensor(0, dtype=torch.float32)  # 屏蔽
                else:
                    # 缺失标签
                    labels_dict[task_key] = torch.tensor(0, dtype=torch.long)
                    label_mask_dict[task_key] = torch.tensor(0, dtype=torch.float32)
        else:
            # 无法匹配文件名，所有任务标签缺失
            for task_key in self.task_keys:
                labels_dict[task_key] = torch.tensor(0, dtype=torch.long)
                label_mask_dict[task_key] = torch.tensor(0, dtype=torch.float32)

        # ========== 构建返回字典 ==========
        result = {
            "x_dyn": x_dyn,
            "labels": labels_dict,
            "label_mask": label_mask_dict
        }

        sample_id = None
        if filename:
            sample_id = f"sample_{idx:06d}"

        raw_static = None
        if self.use_static_features and staticlist is not None and idx < len(staticlist):
            raw_static = staticlist[idx]

        result["filename"] = f"{sample_id}.xlsx" if sample_id else None
        result["patient_id"] = sample_id
        result["age"] = float(raw_static[0]) if raw_static is not None and len(raw_static) > 0 else None
        result["sex"] = int(raw_static[1]) if raw_static is not None and len(raw_static) > 1 else None

        if x_static is not None:
            result["x_static"] = x_static

        if lengths is not None:
            result["lengths"] = lengths

        return result

    def get_task_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取任务统计信息 (用于 build_task_specs)"""
        return self.task_stats

    def get_num_classes(self, task_key: str) -> int:
        """获取指定任务的类别数"""
        if task_key in self.task_stats:
            return self.task_stats[task_key]["num_classes"]
        return 2  # 默认二分类


# =============================================================================
# 创建 MTL DataLoader 的辅助函数
# =============================================================================

def create_mtl_dataloaders(
    config,
    fold_idx: int = 0,
    n_folds: int = 5,
    batch_size: int = 16,
    use_variable_length: bool = False,
    task_keys: List[str] = None,
    num_workers: int = 4,
    # [新增] Holdout 参数
    dev_indices: List[int] = None,
    test_indices: List[int] = None,
    use_holdout_test: bool = False,
    train_stats: dict = None,
    train_static_stats: dict = None,
    strict_no_filter: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, CPETDatasetMTL]:
    """
    创建 MTL DataLoader (支持 Holdout 模式)

    Args:
        config: Config 配置对象
        fold_idx: Fold 编号
        n_folds: 总 Fold 数
        batch_size: 批次大小
        use_variable_length: 是否使用变长模式
        task_keys: 任务列表
        num_workers: DataLoader 工作线程数
        dev_indices: [新增] 预划分的 Dev_Set 索引列表
        test_indices: [新增] 预划分的 Test_Set 索引列表
        use_holdout_test: [新增] 是否直接使用独立测试集 (仅用于创建 holdout test loader)
        train_stats: [新增] 预计算的训练集动态特征统计量
        train_static_stats: [新增] 预计算的训练集静态特征统计量

    Returns:
        train_loader: 训练集 DataLoader
        test_loader: 测试集 DataLoader
        dataset: 数据集对象 (用于获取统计信息)
    """
    from torch.utils.data import DataLoader
    from dataset_new import preload_all_data

    # 预加载数据缓存 (避免重复加载)
    data_cache = preload_all_data(
        config=config,
        use_variable_length=use_variable_length,
        use_static_features=True,
        strict_no_filter=strict_no_filter
    )

    # 创建训练集 (传入 dev_indices)
    train_dataset = CPETDatasetMTL(
        config=config,
        fold_idx=fold_idx,
        n_folds=n_folds,
        phase="train",
        use_variable_length=use_variable_length,
        use_static_features=True,
        mtl_mode=True,
        task_keys=task_keys,
        all_data_cache=data_cache,
        dev_indices=dev_indices,            # [新增]
        test_indices=test_indices,          # [新增]
        use_holdout_test=False,             # 训练模式使用 dev set
        train_stats=train_stats,
        train_static_stats=train_static_stats
    )

    # 创建验证集 (共享训练集统计信息)
    test_dataset = CPETDatasetMTL(
        config=config,
        fold_idx=fold_idx,
        n_folds=n_folds,
        phase="test",
        use_variable_length=use_variable_length,
        use_static_features=True,
        mtl_mode=True,
        task_keys=task_keys,
        train_stats=train_dataset.stats,
        train_static_stats=train_dataset.static_stats,
        all_data_cache=data_cache,
        dev_indices=dev_indices,
        test_indices=test_indices,
        use_holdout_test=False
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_mtl,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_mtl,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset


def create_mtl_holdout_test_loader(
    config,
    test_indices: List[int],
    dev_indices: List[int],
    fold_stats: dict,
    fold_static_stats: dict,
    batch_size: int = 16,
    use_variable_length: bool = False,
    task_keys: List[str] = None,
    num_workers: int = 4,
    all_data_cache: dict = None,
    strict_no_filter: bool = False
) -> torch.utils.data.DataLoader:
    """
    创建 Holdout 独立测试集 DataLoader

    Args:
        config: Config 配置对象
        test_indices: 独立测试集索引列表
        dev_indices: Dev_Set 索引列表 (用于数据缓存)
        fold_stats: 当前 Fold 训练集动态特征统计量
        fold_static_stats: 当前 Fold 静态特征统计量
        batch_size: 批次大小
        use_variable_length: 是否使用变长模式
        task_keys: 任务列表
        num_workers: DataLoader 工作线程数
        all_data_cache: 预加载的数据缓存

    Returns:
        holdout_loader: Holdout 测试集 DataLoader
    """
    from torch.utils.data import DataLoader
    from dataset_new import preload_all_data

    if all_data_cache is None:
        all_data_cache = preload_all_data(
            config=config,
            use_variable_length=use_variable_length,
            use_static_features=True,
            strict_no_filter=strict_no_filter
        )

    # 直接使用独立测试集 (use_holdout_test=True)
    holdout_dataset = CPETDatasetMTL(
        config=config,
        fold_idx=0,                # Holdout test 不需要 fold
        n_folds=1,                 # 单折
        phase="train",             # phase="train" 但实际是测试集
        use_variable_length=use_variable_length,
        use_static_features=True,
        mtl_mode=True,
        task_keys=task_keys,
        all_data_cache=all_data_cache,
        dev_indices=dev_indices,
        test_indices=test_indices,
        use_holdout_test=True,     # [关键] 直接使用测试集
        train_stats=fold_stats,    # 使用当前 Fold 统计量
        train_static_stats=fold_static_stats
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=batch_size,
        shuffle=False,             # 测试集不打乱
        collate_fn=collate_fn_mtl,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"[Holdout Test Loader] 创建独立测试集: {len(holdout_dataset)} 样本")

    return holdout_loader


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MTL 数据集测试")
    print("=" * 80)

    # 软编码：从配置文件加载
    from config import Config
    config = Config.load()
    label_file = config.data.label_file

    filename_to_labels, task_label_mappings, task_class_counts = load_mtl_labels(label_file)

    print("\n任务标签映射:")
    for task_key, mapping in task_label_mappings.items():
        print(f"  {task_key}: {mapping}")

    print("\n类别分布:")
    for task_key, counter in task_class_counts.items():
        print(f"  {task_key}: {counter}")

    # 测试 collate_fn
    print("\n测试 collate_fn_mtl:")

    # 模拟 batch
    mock_batch = [
        {
            "x_dyn": torch.randn(200, 30),
            "x_static": torch.randn(5),
            "labels": {"t1": torch.tensor(0), "t2": torch.tensor(1), "t6": torch.tensor(2)},
            "label_mask": {"t1": torch.tensor(1.0), "t2": torch.tensor(1.0), "t6": torch.tensor(1.0)}
        },
        {
            "x_dyn": torch.randn(200, 30),
            "x_static": torch.randn(5),
            "labels": {"t1": torch.tensor(1), "t2": torch.tensor(0), "t6": torch.tensor(3)},
            "label_mask": {"t1": torch.tensor(1.0), "t2": torch.tensor(1.0), "t6": torch.tensor(1.0)}
        }
    ]

    batch_result = collate_fn_mtl(mock_batch)

    print(f"  x_dyn: {batch_result['x_dyn'].shape}")
    print(f"  x_static: {batch_result['x_static'].shape}")
    print(f"  labels['t1']: {batch_result['labels']['t1']}")
    print(f"  label_mask['t1']: {batch_result['label_mask']['t1']}")

    print("\n✓ 所有测试通过")
