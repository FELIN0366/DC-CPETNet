"""
任务规格数据结构 (TaskSpec)
==========================

定义多任务学习 (MTL) 中每个任务的元信息。

使用方法:
    from task_specs import TaskSpec, build_task_specs_from_config
    task_specs = build_task_specs_from_config(config, dataset_stats)
    print(task_specs["t1"].num_classes)  # 访问任务配置
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal


@dataclass
class TaskSpec:
    """
    任务规格数据结构

    包含每个任务的完整元信息，用于:
    1. 模型构建时确定分支类型和输出维度
    2. 损失函数构建时确定损失类型和权重
    3. 训练流程中确定冻结/解冻状态

    Attributes:
        name: 任务内部键名 ("t1" ~ "t6")
        display_name: 任务显示名称 (中文)
        num_classes: 类别数 (多分类: >=3, 二分类: 2)
        branch: 分支类型 ("alpha" 或 "beta")
        loss_name: 主损失函数名称 ("ce", "ldam", "bce")
        is_binary: 是否为二分类任务
        dropout: 任务头 dropout 率
        label_column: 标签列名称
        class_counts: 各类别样本数 [C1, C2, ...]
        class_weights: 类别权重 Tensor (多分类) 或 None
        minority_idx: 少数类索引 (二分类) 或 None
        majority_idx: 多数类索引 (二分类) 或 None
        pos_weight: 正类权重 Tensor (二分类 BCE) 或 None
        kd_teacher: 教师模型路径 (仅 t6) 或 None
    """
    name: str
    display_name: str
    num_classes: int
    branch: Literal["alpha", "beta"]
    loss_name: Literal["ce", "ldam", "bce"]
    is_binary: bool
    dropout: float
    label_column: str

    # 动态计算的属性 (从数据集统计)
    class_counts: List[int] = field(default_factory=list)
    class_weights: Optional[torch.Tensor] = None
    minority_idx: Optional[int] = None
    majority_idx: Optional[int] = None
    pos_weight: Optional[torch.Tensor] = None

    # KD 教师模型 (仅 t6)
    kd_teacher: Optional[str] = None

    def __post_init__(self):
        """
        后处理：验证基础约束

        [已移除] Alpha/Beta 分支任务约束
        原因: 消融实验 (E2) 需要将 t2-t5 临时分配给 Alpha 分支
        分支完整性检查已移至 validate_task_specs，支持消融模式检测
        """
        # 约束 1: 二分类任务必须 is_binary=True
        if self.num_classes == 2:
            assert self.is_binary, f"{self.name}: 二分类任务必须 is_binary=True"

        # 约束 4: t3 使用 BCE
        if self.name == "t3":
            assert self.loss_name == "bce", f"{self.name}: t3 必须使用 BCE 损失"

        # 约束 5: t4, t5 使用 LDAM
        if self.name in ["t4", "t5"]:
            assert self.loss_name == "ldam", f"{self.name}: t4, t5 必须使用 LDAM 损失"

    def update_stats(self, class_counts, device: str = "cpu"):
        """
        根据数据集统计更新任务属性

        Args:
            class_counts: 各类别样本数 (支持 List[int] 或 Counter)
            device: Tensor 设备
        """
        # 支持 Counter 类型
        if isinstance(class_counts, dict):
            # 从 Counter 转换为 List (按索引排序)
            if len(class_counts) == 0:
                return
            max_idx = max(class_counts.keys())
            counts_list = [class_counts.get(i, 0) for i in range(max_idx + 1)]
            class_counts = counts_list

        self.class_counts = [int(c) for c in class_counts]  # 确保 class_counts 是 int 类型

        if self.is_binary:
            # 二分类: 计算 pos_weight 和 minority_idx
            assert len(class_counts) == 2, f"{self.name}: 二分类应有 2 个类别"

            # 少数类索引 (阳性类)
            self.minority_idx = 1 if class_counts[1] < class_counts[0] else 0
            self.majority_idx = 1 - self.minority_idx

            # pos_weight = N_negative / N_positive
            n_positive = class_counts[self.minority_idx]
            n_negative = class_counts[self.majority_idx]
            pos_weight_value = n_negative / (n_positive + 1e-6)
            self.pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

            # 二分类不使用 class_weights (LDAM 内部处理)
            if self.loss_name == "bce":
                self.class_weights = None
            else:  # LDAM 二分类
                self.class_weights = self.pos_weight  # LDAM 使用 pos_weight

        else:
            # 多分类: 计算 class_weights
            total = sum(class_counts)
            n_classes = len(class_counts)

            # 频率倒数加权: w_c = 1 / sqrt(N_c / N_total)
            class_weights_list = []
            for count in class_counts:
                freq = count / total
                weight = 1.0 / np.sqrt(freq + 1e-6)
                class_weights_list.append(weight)

            # 归一化
            weights_sum = sum(class_weights_list)
            class_weights_list = [w / weights_sum * n_classes for w in class_weights_list]

            self.class_weights = torch.tensor(class_weights_list, dtype=torch.float32, device=device)
            self.minority_idx = None
            self.majority_idx = None
            self.pos_weight = None

    def get_output_dim(self) -> int:
        """
        获取输出维度

        Returns:
            多分类: num_classes
            二分类: 1 (单节点 logits)
        """
        return 1 if self.is_binary else self.num_classes

    def __repr__(self) -> str:
        """简洁打印"""
        return (
            f"TaskSpec({self.name}: {self.display_name}, "
            f"branch={self.branch}, loss={self.loss_name}, "
            f"classes={self.num_classes}, binary={self.is_binary})"
        )


def build_task_specs_from_config(
    config: Dict[str, Any],
    dataset_stats: Optional[Dict[str, List[int]]] = None,
    device: str = "cpu"
) -> Dict[str, TaskSpec]:
    """
    从配置字典构建 TaskSpec 字典

    Args:
        config: MTL 配置字典 (来自 config_mtl.yaml)
        dataset_stats: 各任务类别统计 {"t1": [C1, C2, ...], ...}
        device: Tensor 设备

    Returns:
        TaskSpec 字典 {"t1": TaskSpec(...), ..., "t6": TaskSpec(...)}
    """
    mtl_config = config.get("mtl", {})
    tasks_config = mtl_config.get("tasks", {})

    task_specs = {}

    for task_key, task_cfg in tasks_config.items():
        # 动态计算 num_classes (如果为 null 或未指定)
        num_classes = task_cfg.get("num_classes", None)
        if num_classes is None and dataset_stats and task_key in dataset_stats:
            num_classes = len(dataset_stats[task_key])
        elif num_classes is None:
            num_classes = 2  # 默认值

        # 构建 TaskSpec
        spec = TaskSpec(
            name=task_key,
            display_name=task_cfg.get("name", task_key),
            num_classes=num_classes,
            branch=task_cfg.get("branch", "beta"),
            loss_name=task_cfg.get("loss_name", "ce"),
            is_binary=task_cfg.get("is_binary", False),
            dropout=task_cfg.get("dropout", 0.3),
            label_column=task_cfg.get("label_column", task_key),
            kd_teacher=task_cfg.get("kd_teacher", None)
        )

        # 如果提供了数据集统计，更新属性
        if dataset_stats and task_key in dataset_stats:
            spec.update_stats(dataset_stats[task_key], device)

        task_specs[task_key] = spec

    return task_specs


def validate_task_specs(
    task_specs: Dict[str, TaskSpec],
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    验证 TaskSpec 字典完整性

    Args:
        task_specs: TaskSpec 字典
        config: MTL 配置字典 (用于消融模式检测)

    Returns:
        是否通过验证
    """
    # 检查必需任务
    required_tasks = ["t1", "t2", "t3", "t4", "t5", "t6"]
    for task_key in required_tasks:
        if task_key not in task_specs:
            print(f"[TaskSpec] 错误: 缺少任务 {task_key}")
            return False

    # 检测消融模式
    is_e2_ablation = False
    if config is not None:
        ablation_cfg = config.get("mtl", {}).get("hcgc_v4", {}).get("ablation", {})
        is_e2_ablation = ablation_cfg.get("single_shared_alpha", False)

    # 检查 Alpha 分支
    alpha_tasks = [k for k, v in task_specs.items() if v.branch == "alpha"]
    if is_e2_ablation:
        # E2 消融: 所有任务应走 Alpha
        if set(alpha_tasks) != {"t1", "t2", "t3", "t4", "t5", "t6"}:
            print(f"[TaskSpec] 错误 (E2消融): Alpha 分支应为 t1~t6，实际为 {alpha_tasks}")
            return False
        print(f"[TaskSpec] E2消融验证通过: Alpha 分支 = {alpha_tasks}")
    else:
        # 正常模式: Alpha={t1, t6}
        if set(alpha_tasks) != {"t1", "t6"}:
            print(f"[TaskSpec] 错误: Alpha 分支应为 t1, t6，实际为 {alpha_tasks}")
            return False

    # 检查 Beta 分支
    beta_tasks = [k for k, v in task_specs.items() if v.branch == "beta"]
    if is_e2_ablation:
        # E2 消融: Beta 分支应为空
        if len(beta_tasks) > 0:
            print(f"[TaskSpec] 错误 (E2消融): Beta 分支应为空，实际为 {beta_tasks}")
            return False
        print(f"[TaskSpec] E2消融验证通过: Beta 分支为空")
    else:
        # 正常模式: Beta={t2~t5}
        if set(beta_tasks) != {"t2", "t3", "t4", "t5"}:
            print(f"[TaskSpec] 错误: Beta 分支应为 t2~t5，实际为 {beta_tasks}")
            return False

    # 检查 t6 KD 教师路径
    if task_specs["t6"].kd_teacher is None:
        print("[TaskSpec] 警告: t6 未设置 KD 教师路径")

    # 检查类别统计已更新
    for task_key, spec in task_specs.items():
        if not spec.class_counts:
            print(f"[TaskSpec] 警告: {task_key} 类别统计未更新")

    return True


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TaskSpec 测试")
    print("=" * 80)

    # 模拟配置
    test_config = {
        "mtl": {
            "tasks": {
                "t1": {
                    "name": "运动心功能分级",
                    "num_classes": 3,
                    "branch": "alpha",
                    "loss_name": "ce",
                    "is_binary": False,
                    "dropout": 0.3,
                    "label_column": "运动心功能分级"
                },
                "t2": {
                    "name": "运动耐量",
                    "num_classes": 3,
                    "branch": "beta",
                    "loss_name": "ldam",
                    "is_binary": False,
                    "dropout": 0.3,
                    "label_column": "运动耐量"
                },
                "t3": {
                    "name": "标准心电运动负荷试验",
                    "num_classes": 2,
                    "branch": "beta",
                    "loss_name": "bce",
                    "is_binary": True,
                    "dropout": 0.3,
                    "label_column": "标准心电运动负荷试验"
                },
                "t4": {
                    "name": "运动中换气肺功能",
                    "num_classes": 2,
                    "branch": "beta",
                    "loss_name": "ldam",
                    "is_binary": True,
                    "dropout": 0.3,
                    "label_column": "运动中换气肺功能"
                },
                "t5": {
                    "name": "心率储备",
                    "num_classes": 2,
                    "branch": "beta",
                    "loss_name": "ldam",
                    "is_binary": True,
                    "dropout": 0.3,
                    "label_column": "心率储备"
                },
                "t6": {
                    "name": "匹配的第一大类",
                    "num_classes": 6,
                    "branch": "alpha",
                    "loss_name": "ce",
                    "is_binary": False,
                    "dropout": 0.3,
                    "label_column": "匹配的第一大类",
                    "kd_teacher": "models/best_HDSTGCN_CPET_New_nine_graph_fold1.pth"
                }
            }
        }
    }

    # 模拟数据集统计
    test_stats = {
        "t1": [100, 200, 150],           # 多分类
        "t2": [80, 220, 150],
        "t3": [450, 50],                  # 二分类 (阳性率 10%)
        "t4": [400, 100],                 # 二分类 (阳性率 20%)
        "t5": [400, 100],
        "t6": [150, 100, 80, 60, 40, 20]  # 多分类 (长尾)
    }

    # 构建 TaskSpec
    task_specs = build_task_specs_from_config(test_config, test_stats)

    # 打印结果
    for task_key, spec in task_specs.items():
        print(f"\n{spec}")
        print(f"  输出维度: {spec.get_output_dim()}")
        print(f"  类别统计: {spec.class_counts}")
        if spec.is_binary:
            print(f"  少数类索引: {spec.minority_idx}")
            print(f"  pos_weight: {spec.pos_weight}")
        else:
            print(f"  类别权重: {spec.class_weights}")

    # 验证
    print(f"\n验证结果: {validate_task_specs(task_specs, test_config)}")