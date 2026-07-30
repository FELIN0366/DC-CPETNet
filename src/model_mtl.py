"""
多任务学习模型 (HDSTGCNMTL)
============================

双引擎架构:
- Alpha 引擎 (t1, t6): 多尺度 CNN + PMGT (PriorMaskedGlobalTransformer)
- Beta 引擎 (t2~t5): 单尺度 CNN + Flatten MLP

关键设计:
- 任务簇内共享编码器，簇间物理隔离
- 任务专属交互头 (gamma 门控独立)
- 同方差不确定性加权 (log_vars 可学习)
- 任务六 KD 保护 (教师模型蒸馏)

参考文献:
- MMoE (Multi-gate Mixture-of-Experts): 任务关系异质性显式建模
- Kendall et al.: 同方差不确定性加权
- LwF (Learning without Forgetting): 输出约束保护旧能力
- LDAM: 标签分布感知间隔损失

创建日期: 2026-04-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass

# 导入现有模块 (直接复用)
from model import (
    CNNTemporalEncodingBranch,
    PriorMaskedGlobalTransformer
)

from config import TemporalEncoderConfig, StaticFeatureConfig
from task_specs import TaskSpec


# =============================================================================
# Alpha/Beta 双引擎编码器
# =============================================================================

class AlphaDynamicEncoder(nn.Module):
    """
    Alpha 动态编码器 (服务 t1, t6)

    特点:
    - 多尺度卷积 (k=3,5,7)
    - 残差连接
    - 输出: [B, C, D_time] 纯净节点特征

    Args:
        num_channels: 特征通道数 (默认 30, nine_graph 模式)
        D_time: 时序编码维度 (默认 16)
        T_mid: CNN 中间时序维度 (默认 24)
        dropout: Dropout 率
        config: TemporalEncoderConfig (可选，用于详细配置)
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        dropout: float = 0.3,
        config: Optional[TemporalEncoderConfig] = None
    ):
        super().__init__()
        self.num_channels = num_channels
        self.D_time = D_time
        self.name = "alpha"

        # 构建 Alpha 专用配置 (多尺度 + 残差)
        if config is None:
            config = TemporalEncoderConfig(
                type="cnn",
                T_mid=T_mid,
                use_multiscale=True,      # Alpha 核心: 多尺度
                use_residual=True,        # Alpha 核心: 残差
                multiscale_kernels=[3, 5, 7],
                block1_kernel=7,
                block2_kernel=5,
                use_masked_conv=False
            )

        # 直接复用 CNNTemporalEncodingBranch
        self.encoder = CNNTemporalEncodingBranch(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=dropout,
            config=config
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L_max, C] 动态序列
            lengths: [B] 真实长度 (可选)

        Returns:
            H_nodes: [B, C, D_time] 纯净节点特征
        """
        return self.encoder(x, lengths)


class BetaDynamicEncoder(nn.Module):
    """
    Beta 动态编码器 (服务 t2~t5)

    特点:
    - 单尺度卷积 (无多尺度)
    - 无残差连接
    - 轻量化设计
    - 输出: [B, C, D_time] 纯净节点特征

    Args:
        num_channels: 特征通道数 (默认 30)
        D_time: 时序编码维度 (默认 16)
        T_mid: CNN 中间时序维度 (默认 24)
        dropout: Dropout 率
        config: TemporalEncoderConfig (可选)
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        dropout: float = 0.3,
        config: Optional[TemporalEncoderConfig] = None
    ):
        super().__init__()
        self.num_channels = num_channels
        self.D_time = D_time
        self.name = "beta"

        # 构建 Beta 专用配置 (单尺度，无残差)
        if config is None:
            config = TemporalEncoderConfig(
                type="cnn",
                T_mid=T_mid,
                use_multiscale=False,     # Beta 核心: 单尺度
                use_residual=False,       # Beta 核心: 无残差
                block1_kernel=7,
                block2_kernel=5,
                use_masked_conv=False
            )

        # 直接复用 CNNTemporalEncodingBranch
        self.encoder = CNNTemporalEncodingBranch(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=dropout,
            config=config
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L_max, C] 动态序列
            lengths: [B] 真实长度 (可选)

        Returns:
            H_nodes: [B, C, D_time] 纯净节点特征
        """
        return self.encoder(x, lengths)


# =============================================================================
# 任务交互头 (Alpha: PMGT, Beta: Flatten)
# =============================================================================

class PriorMaskedTaskHead(nn.Module):
    """
    先验掩码任务头 (Alpha 分支专用，t1, t6)

    封装 PriorMaskedGlobalTransformer + 投影层

    关键:
    - gamma 门控参数任务专属 (t1 和 t6 不共享)
    - 共享医学先验邻接矩阵

    Args:
        num_nodes: 节点数 (默认 30)
        hidden_dim: 时序编码维度 (默认 16)
        out_dim: 输出维度 (默认 48)
        semantic_adj: 医学先验邻接矩阵
        dropout: Dropout 率
        gamma_init: gamma 初始值
        gamma_min: gamma 下限
    """

    def __init__(
        self,
        num_nodes: int = 30,
        hidden_dim: int = 16,
        out_dim: int = 48,
        semantic_adj: Optional[torch.Tensor] = None,
        dropout: float = 0.5,
        gamma_init: float = 1.0,
        gamma_min: float = 0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # PMGT 核心模块 (gamma 任务专属)
        self.pmgt = PriorMaskedGlobalTransformer(
            hidden_dim=hidden_dim,
            num_heads=2,
            num_nodes=num_nodes,
            semantic_adj=semantic_adj,
            dropout=dropout,
            gamma_init=gamma_init,
            gamma_min=gamma_min
        )

        # 投影层: [B, C*D] -> [B, out_dim]
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_nodes * hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, H_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_nodes: [B, C, D_time] 来自 Alpha 编码器

        Returns:
            dynamic_feat: [B, out_dim] 动态特征
        """
        # PMGT: [B, C, D] -> [B, C, D] (跨变量交互)
        H_interacted = self.pmgt(H_nodes)

        # 投影: [B, C*D] -> [B, out_dim]
        return self.proj(H_interacted)


class FlattenProjector(nn.Module):
    """
    展平投影头 (Beta 分支专用，t2~t5)

    直接展平 + MLP 降维，不使用 PMGT

    Args:
        num_nodes: 节点数 (默认 30)
        hidden_dim: 时序编码维度 (默认 16)
        out_dim: 输出维度 (默认 48)
        dropout: Dropout 率
        use_two_layer: 是否使用两层 MLP
    """

    def __init__(
        self,
        num_nodes: int = 30,
        hidden_dim: int = 16,
        out_dim: int = 48,
        dropout: float = 0.3,
        use_two_layer: bool = True
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        in_dim = num_nodes * hidden_dim  # 480

        if use_two_layer:
            # 两层 MLP: 480 -> 128 -> 48
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # 单层 MLP: 480 -> 48
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(self, H_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_nodes: [B, C, D_time] 来自 Beta 编码器

        Returns:
            dynamic_feat: [B, out_dim] 动态特征
        """
        return self.net(H_nodes)


# =============================================================================
# T3 消融实验专用模块
# =============================================================================

class T3FeatureAdapter(nn.Module):
    """
    T3 残差式适配器 (Variant 2)

    在 beta_nodes → t3 head 之间插入轻量适配器

    输入/输出: [B, num_nodes, D_time] (维度不变)

    两种模式:
    - residual: x_out = x + MLP(x)    # 参数量 ~231k (480→480)
    - bottleneck: x_out = MLP_down(x) + MLP_up(x)  # 参数量 ~15k (480→hidden→480)

    不影响 t2/t4/t5 路径 (独立实例)

    Args:
        num_nodes: 节点数 (默认 30)
        D_time: 时序编码维度 (默认 16)
        type: 适配器类型 ("residual" 或 "bottleneck")
        hidden_dim: bottleneck 模式瓶颈维度 (默认 64)
        dropout: Dropout 率
    """

    def __init__(
        self,
        num_nodes: int = 30,
        D_time: int = 16,
        type: str = "residual",
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.D_time = D_time
        self.adapter_type = type

        in_dim = num_nodes * D_time  # 480

        if type == "residual":
            # 残差模式: x_out = x + MLP(x)
            # 参数量: 480*480 + 480 = 230,880 (约 231k)
            self.adapter = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, in_dim),
                nn.LayerNorm(in_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        elif type == "bottleneck":
            # 瓶颈模式: 先降维再升维
            # 参数量: 480*64 + 64*480 + 64 + 480 = 61,504 (约 15k×2)
            self.adapter = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown adapter type: {type}")

    def forward(self, beta_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            beta_nodes: [B, num_nodes, D_time] 来自 Beta 编码器

        Returns:
            adapted_nodes: [B, num_nodes, D_time] 适配后的节点特征
        """
        B = beta_nodes.size(0)

        # 展平并适配
        x_flat = beta_nodes.view(B, -1)  # [B, 480]
        adapted = self.adapter(x_flat)   # [B, 480]

        # 残差连接 (两种模式都适用)
        if self.adapter_type == "residual":
            adapted = x_flat + adapted

        # 恢复形状
        return adapted.view(B, self.num_nodes, self.D_time)


class T3TopRefiner(nn.Module):
    """
    T3 上层强化投影头 (Variant 3)

    替代原 FlattenProjector，提供更强表征能力

    输入: [B, num_nodes, D_time] beta_nodes
    输出: [B, out_dim] (替代原 FlattenProjector)

    容量 2x:
    - Baseline FlattenProjector: 480→128→48 (~62k)
    - T3TopRefiner: 480→256→128→48 (~124k)

    Args:
        num_nodes: 节点数 (默认 30)
        D_time: 时序编码维度 (默认 16)
        out_dim: 输出维度 (默认 48)
        hidden_dim: 第一隐藏层维度 (默认 256, 比 baseline 128 更大)
        dropout: Dropout 率
    """

    def __init__(
        self,
        num_nodes: int = 30,
        D_time: int = 16,
        out_dim: int = 48,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.D_time = D_time
        self.out_dim = out_dim

        in_dim = num_nodes * D_time  # 480

        # 三层 MLP: 480 → 256 → 128 → 48
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),    # 480 → 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),       # 256 → 128
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),          # 128 → 48
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, beta_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            beta_nodes: [B, num_nodes, D_time] 来自 Beta 编码器

        Returns:
            dynamic_feat: [B, out_dim] 强化后的动态特征
        """
        return self.net(beta_nodes)


# =============================================================================
# 静态编码器 (任务专属)
# =============================================================================

class TaskStaticEncoder(nn.Module):
    """
    任务专属静态编码器

    每个任务独立编码静态特征 (EHR: age, gender, weight, height, bmi)

    Args:
        in_dim: 输入维度 (默认 5)
        out_dim: 输出维度 (默认 16)
        dropout: Dropout 率
    """

    def __init__(
        self,
        in_dim: int = 5,
        out_dim: int = 16,
        dropout: float = 0.2
    ):
        super().__init__()

        # 两层 MLP
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim),
            nn.ReLU()
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_static: [B, in_dim] 静态特征

        Returns:
            static_feat: [B, out_dim] 静态编码
        """
        return self.encoder(x_static)


# =============================================================================
# 分类头
# =============================================================================

class TaskClassifier(nn.Module):
    """
    任务分类头

    融合动态特征 + 静态特征，输出分类 logits

    Args:
        dyn_dim: 动态特征维度 (默认 48)
        static_dim: 静态特征维度 (默认 16)
        hidden_dim: 中间层维度 (默认 32)
        num_classes: 类别数 (多分类: >=3, 二分类: 2)
        is_binary: 是否为二分类
        dropout: Dropout 率
    """

    def __init__(
        self,
        dyn_dim: int = 48,
        static_dim: int = 16,
        hidden_dim: int = 32,
        num_classes: int = 3,
        is_binary: bool = False,
        dropout: float = 0.3
    ):
        super().__init__()
        self.is_binary = is_binary
        self.num_classes = num_classes

        fused_dim = dyn_dim + static_dim  # 64

        # 两层 MLP
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 输出层
        if is_binary:
            # 二分类: 单节点 logits [B, 1]
            self.output = nn.Linear(hidden_dim, 1)
        else:
            # 多分类: [B, num_classes]
            self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, dyn_feat: torch.Tensor, static_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dyn_feat: [B, dyn_dim] 动态特征
            static_feat: [B, static_dim] 静态特征

        Returns:
            logits: [B, num_classes] 或 [B, 1]
        """
        # 融合
        fused = torch.cat([dyn_feat, static_feat], dim=1)

        # 分类
        hidden = self.classifier(fused)
        logits = self.output(hidden)

        return logits


# =============================================================================
# HDSTGCNMTL 总模型
# =============================================================================

class HDSTGCNMTL(nn.Module):
    """
    多任务学习模型总架构

    双引擎 + 任务簇隔离 + KD 保护

    Args:
        task_specs: Dict[str, TaskSpec] 任务规格字典
        num_channels: 特征通道数 (默认 30)
        D_time: 时序编码维度 (默认 16)
        semantic_adj: 医学先验邻接矩阵
        config: MTL 配置对象 (含 ablation 配置)
    """

    def __init__(
        self,
        task_specs: Dict[str, TaskSpec],
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        semantic_adj: Optional[torch.Tensor] = None,
        config: Optional[Any] = None
    ):
        super().__init__()
        self.task_specs = task_specs
        self.num_channels = num_channels
        self.D_time = D_time
        self.T_mid = T_mid

        # === 解析消融配置 ===
        self.variant = "baseline"  # 默认值
        if config is not None:
            mtl_cfg = config.get('mtl', config) if isinstance(config, dict) else {}
            ablation_cfg = mtl_cfg.get('ablation', {})
            if ablation_cfg.get('enabled', False):
                self.variant = ablation_cfg.get('variant', 'baseline')
                print(f"\n[Ablation] 启用 T3 消融变体: {self.variant}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if semantic_adj is not None:
            self.semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32).to(device)
        else:
            self.semantic_adj = None

        # === Alpha 引擎 (t1, t6) ===
        self.alpha_encoder = AlphaDynamicEncoder(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=0.3
        )

        # Alpha 任务交互头 (gamma 任务专属)
        self.alpha_interactors = nn.ModuleDict({
            "t1": PriorMaskedTaskHead(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=48,
                semantic_adj=self.semantic_adj,
                dropout=0.5,
                gamma_init=1.0  # t1 专属 gamma
            ),
            "t6": PriorMaskedTaskHead(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=48,
                semantic_adj=self.semantic_adj,
                dropout=0.5,
                gamma_init=1.0  # t6 专属 gamma
            )
        })

        # === Beta 引擎 (t2~t5) ===
        self.beta_encoder = BetaDynamicEncoder(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=0.3
        )

        # === Variant 4: T3 独立编码器 ===
        if self.variant == "t3_private_encoder":
            print("[Ablation] 创建 T3 独立 Beta 编码器 (参数不共享)")
            self.beta_encoder_t3 = BetaDynamicEncoder(
                num_channels=num_channels,
                D_time=D_time,
                T_mid=T_mid,
                dropout=0.3
            )

        # Beta 任务投影头
        self.beta_projectors = nn.ModuleDict({
            "t2": FlattenProjector(num_nodes=num_channels, hidden_dim=D_time, out_dim=48),
            "t3": FlattenProjector(num_nodes=num_channels, hidden_dim=D_time, out_dim=48),
            "t4": FlattenProjector(num_nodes=num_channels, hidden_dim=D_time, out_dim=48),
            "t5": FlattenProjector(num_nodes=num_channels, hidden_dim=D_time, out_dim=48)
        })

        # === Variant 2: T3 适配器 ===
        if self.variant == "t3_adapter":
            ablation_cfg = config.get('mtl', config) if isinstance(config, dict) else {}
            ablation_cfg = ablation_cfg.get('ablation', {})
            t3_adapter_cfg = ablation_cfg.get('t3_adapter', {})
            print(f"[Ablation] 创建 T3 适配器: type={t3_adapter_cfg.get('type', 'residual')}")
            self.t3_adapter = T3FeatureAdapter(
                num_nodes=num_channels,
                D_time=D_time,
                type=t3_adapter_cfg.get('type', 'residual'),
                hidden_dim=t3_adapter_cfg.get('hidden_dim', 64),
                dropout=t3_adapter_cfg.get('dropout', 0.1)
            )

        # === Variant 3: T3 强化投影头 (替代 beta_projectors["t3"]) ===
        if self.variant == "t3_private_top":
            ablation_cfg = config.get('mtl', config) if isinstance(config, dict) else {}
            ablation_cfg = ablation_cfg.get('ablation', {})
            t3_top_cfg = ablation_cfg.get('t3_private_top', {})
            print(f"[Ablation] 创建 T3 强化投影头: hidden_dim={t3_top_cfg.get('hidden_dim', 256)}")
            self.t3_top_refiner = T3TopRefiner(
                num_nodes=num_channels,
                D_time=D_time,
                hidden_dim=t3_top_cfg.get('hidden_dim', 256),
                dropout=t3_top_cfg.get('dropout', 0.2)
            )

        # === 静态编码器 (每个任务独立) ===
        self.static_encoders = nn.ModuleDict({
            task_key: TaskStaticEncoder(in_dim=5, out_dim=16, dropout=0.2)
            for task_key in task_specs.keys()
        })

        # === 分类头 (每个任务独立) ===
        self.classifiers = nn.ModuleDict({
            task_key: TaskClassifier(
                dyn_dim=48,
                static_dim=16,
                hidden_dim=32,
                num_classes=spec.num_classes,
                is_binary=spec.is_binary,
                dropout=spec.dropout
            )
            for task_key, spec in task_specs.items()
        })

        # === 同方差不确定性权重 (log_vars) ===
        # s_t = log(sigma_t^2), 初始化为 0
        self.log_vars = nn.ParameterDict({
            task_key: nn.Parameter(torch.zeros(1, device=device))
            for task_key in task_specs.keys()
        })

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_static: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        task_mask: Optional[Dict[str, torch.Tensor]] = None,
        return_aux: bool = False
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        前向传播 (一次输出所有任务)

        Args:
            x_dyn: [B, L_max, C] 动态序列
            x_static: [B, 5] 静态特征
            lengths: [B] 真实长度 (可选)
            task_mask: {"t1": [B], ...} 任务掩码 (可选)
            return_aux: 是否返回辅助信息 (alpha_nodes, beta_nodes)

        Returns:
            outputs: {"t1": {"logits": ..., "dyn_feat": ..., ...}, ..., "t6": {...}}
            aux: {"alpha_nodes": [B, C, D], "beta_nodes": [B, C, D]} (可选)
        """
        B = x_dyn.size(0)
        device = x_dyn.device

        # 默认 task_mask 为全 1
        if task_mask is None:
            task_mask = {task_key: torch.ones(B, device=device) for task_key in self.task_specs.keys()}

        # === Alpha 引擎编码 ===
        alpha_nodes = self.alpha_encoder(x_dyn, lengths)  # [B, C, D]

        # === Beta 引擎编码 ===
        beta_nodes = self.beta_encoder(x_dyn, lengths)    # [B, C, D]

        # === 各任务分支处理 ===
        outputs = {}

        # Alpha 任务 (t1, t6)
        for task_key in ["t1", "t6"]:
            spec = self.task_specs[task_key]

            # 动态特征: PMGT 交互 + 投影
            dyn_feat = self.alpha_interactors[task_key](alpha_nodes)  # [B, 48]

            # 静态特征
            static_feat = self.static_encoders[task_key](x_static)    # [B, 16]

            # 分类
            logits = self.classifiers[task_key](dyn_feat, static_feat)

            outputs[task_key] = {
                "logits": logits,
                "dyn_feat": dyn_feat,
                "static_feat": static_feat,
                "fused_feat": torch.cat([dyn_feat, static_feat], dim=1)
            }

        # Beta 任务 (t2~t5)
        for task_key in ["t2", "t3", "t4", "t5"]:
            spec = self.task_specs[task_key]

            # === T3 变体路径 ===
            if task_key == "t3" and self.variant != "baseline":
                if self.variant == "t3_private_encoder":
                    # Variant 4: T3 使用独立编码器
                    nodes_t3 = self.beta_encoder_t3(x_dyn, lengths)  # [B, C, D]
                    dyn_feat = self.beta_projectors["t3"](nodes_t3)  # [B, 48]
                elif self.variant == "t3_adapter":
                    # Variant 2: T3 使用适配器
                    nodes_t3 = self.t3_adapter(beta_nodes)  # [B, C, D]
                    dyn_feat = self.beta_projectors["t3"](nodes_t3)  # [B, 48]
                elif self.variant == "t3_private_top":
                    # Variant 3: T3 使用强化投影头
                    dyn_feat = self.t3_top_refiner(beta_nodes)  # [B, 48]
                else:
                    # 未知的 variant，回退到 baseline
                    dyn_feat = self.beta_projectors[task_key](beta_nodes)
            else:
                # t2/t4/t5 或 baseline: 原路径
                dyn_feat = self.beta_projectors[task_key](beta_nodes)    # [B, 48]

            # 静态特征
            static_feat = self.static_encoders[task_key](x_static)   # [B, 16]

            # 分类
            logits = self.classifiers[task_key](dyn_feat, static_feat)

            outputs[task_key] = {
                "logits": logits,
                "dyn_feat": dyn_feat,
                "static_feat": static_feat,
                "fused_feat": torch.cat([dyn_feat, static_feat], dim=1)
            }

        # 辅助信息
        if return_aux:
            outputs["aux"] = {
                "alpha_nodes": alpha_nodes,
                "beta_nodes": beta_nodes
            }

        return outputs

    def freeze_alpha_branch(self):
        """冻结 Alpha 分支 (阶段二使用)"""
        for param in self.alpha_encoder.parameters():
            param.requires_grad = False
        for param in self.alpha_interactors["t1"].parameters():
            param.requires_grad = False
        for param in self.alpha_interactors["t6"].parameters():
            param.requires_grad = False
        for param in self.static_encoders["t1"].parameters():
            param.requires_grad = False
        for param in self.static_encoders["t6"].parameters():
            param.requires_grad = False
        for param in self.classifiers["t1"].parameters():
            param.requires_grad = False
        for param in self.classifiers["t6"].parameters():
            param.requires_grad = False

    def freeze_beta_branch(self):
        """冻结 Beta 分支 (阶段一使用)

        注意: T3 消融变体的专属组件保持可训练
        - Variant 2 (t3_adapter): t3_adapter 不冻结
        - Variant 3 (t3_private_top): t3_top_refiner 不冻结
        - Variant 4 (t3_private_encoder): beta_encoder_t3 不冻结
        """
        # 共享 Beta encoder (冻结)
        for param in self.beta_encoder.parameters():
            param.requires_grad = False

        # Variant 4: T3 独立 encoder 保持可训练
        if hasattr(self, 'beta_encoder_t3'):
            for param in self.beta_encoder_t3.parameters():
                param.requires_grad = True  # 保持可训练
            print("[freeze_beta_branch] T3 独立编码器保持可训练")

        for task_key in ["t2", "t3", "t4", "t5"]:
            # Beta projectors (冻结，但 t3 变体有例外)
            if hasattr(self, 'beta_projectors') and task_key in self.beta_projectors:
                for param in self.beta_projectors[task_key].parameters():
                    param.requires_grad = False

            # Variant 3: T3 强化投影头保持可训练
            if task_key == "t3" and hasattr(self, 't3_top_refiner'):
                for param in self.t3_top_refiner.parameters():
                    param.requires_grad = True  # 保持可训练
                print("[freeze_beta_branch] T3 强化投影头保持可训练")

            # Variant 2: T3 适配器保持可训练
            if task_key == "t3" and hasattr(self, 't3_adapter'):
                for param in self.t3_adapter.parameters():
                    param.requires_grad = True  # 保持可训练
                print("[freeze_beta_branch] T3 适配器保持可训练")

            # 静态编码器和分类头 (冻结)
            for param in self.static_encoders[task_key].parameters():
                param.requires_grad = False
            for param in self.classifiers[task_key].parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        """解冻所有分支 (阶段三使用)"""
        for param in self.parameters():
            param.requires_grad = True

    def get_num_parameters(self) -> Dict[str, int]:
        """获取各模块参数量 (包含消融变体组件)"""
        counts = {}

        counts["alpha_encoder"] = sum(p.numel() for p in self.alpha_encoder.parameters())
        counts["beta_encoder"] = sum(p.numel() for p in self.beta_encoder.parameters())

        counts["alpha_interactors"] = sum(
            sum(p.numel() for p in self.alpha_interactors[k].parameters())
            for k in self.alpha_interactors.keys()
        )

        counts["beta_projectors"] = sum(
            sum(p.numel() for p in self.beta_projectors[k].parameters())
            for k in self.beta_projectors.keys()
        )

        counts["static_encoders"] = sum(
            sum(p.numel() for p in self.static_encoders[k].parameters())
            for k in self.static_encoders.keys()
        )

        counts["classifiers"] = sum(
            sum(p.numel() for p in self.classifiers[k].parameters())
            for k in self.classifiers.keys()
        )

        counts["log_vars"] = sum(p.numel() for p in self.log_vars.parameters())

        # === 消融变体组件参数量 ===
        if hasattr(self, 'beta_encoder_t3'):
            counts["beta_encoder_t3"] = sum(p.numel() for p in self.beta_encoder_t3.parameters())
        if hasattr(self, 't3_adapter'):
            counts["t3_adapter"] = sum(p.numel() for p in self.t3_adapter.parameters())
        if hasattr(self, 't3_top_refiner'):
            counts["t3_top_refiner"] = sum(p.numel() for p in self.t3_top_refiner.parameters())

        counts["total"] = sum(counts.values())

        return counts


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HDSTGCNMTL 测试")
    print("=" * 80)

    # 构建 TaskSpec
    from task_specs import TaskSpec

    task_specs = {
        "t1": TaskSpec("t1", "运动心功能分级", 3, "alpha", "ce", False, 0.3, "运动心功能分级"),
        "t2": TaskSpec("t2", "运动耐量", 3, "beta", "ldam", False, 0.3, "运动耐量"),
        "t3": TaskSpec("t3", "标准心电运动负荷试验", 2, "beta", "bce", True, 0.3, "标准心电运动负荷试验"),
        "t4": TaskSpec("t4", "运动中换气肺功能", 2, "beta", "ldam", True, 0.3, "运动中换气肺功能"),
        "t5": TaskSpec("t5", "心率储备", 2, "beta", "ldam", True, 0.3, "心率储备"),
        "t6": TaskSpec("t6", "匹配的第一大类", 6, "alpha", "ce", False, 0.3, "匹配的第一大类", kd_teacher="dummy.pth")
    }

    # 创建模型
    model = HDSTGCNMTL(task_specs, num_channels=30, D_time=16)

    # 打印参数量
    counts = model.get_num_parameters()
    for key, count in counts.items():
        print(f"{key}: {count} params")

    # 测试前向传播
    B, L, C = 4, 200, 30
    x_dyn = torch.randn(B, L, C)
    x_static = torch.randn(B, 5)

    outputs = model(x_dyn, x_static, return_aux=True)

    print("\n前向输出:")
    for task_key, task_output in outputs.items():
        if task_key == "aux":
            print(f"aux: alpha_nodes={task_output['alpha_nodes'].shape}, beta_nodes={task_output['beta_nodes'].shape}")
        else:
            print(f"{task_key}: logits={task_output['logits'].shape}")

    print("\n测试通过！")


# =============================================================================
# Protected Dual-Engine Hierarchical CGC/PLE Architecture
# =============================================================================

# Import expert and gate modules
try:
    from modules import (
        ExpertTemporalEncoder,
        EXPERT_CAPACITY_CONFIG,
        ResidualExpert,  # v2: 新增残差专家模块
        AlphaGateContextEncoder,
        BetaGateContextEncoder,
        SharedGateStaticMLP,
        TaskSpecificGate,
    )
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from modules.experts import ExpertTemporalEncoder, EXPERT_CAPACITY_CONFIG, ResidualExpert
    from modules.gates import (
        AlphaGateContextEncoder,
        BetaGateContextEncoder,
        SharedGateStaticMLP,
        TaskSpecificGate,
    )


class ProtectedDualEngineMTL(nn.Module):
    """
    Protected Dual-Engine Hierarchical CGC/PLE v2 多任务架构

    ================================================================
    v2 架构核心改变 (相对于 v1):
    ================================================================
    - v1: trunk output (A_base / B_base) 也是 gate 候选
    - v2: trunk output **不再**是 gate 候选

    v2 设计理念:
    - trunk 提取分支级基础表征 (H_alpha / H_beta)
    - 专家提供不同的残差特化方向
    - gate 只在专家之间路由，不包括 trunk
    - 避免 gate 走捷径一直选 base，削弱专家专业化

    ================================================================
    v2 专家输出 (残差变换):
    ================================================================
    Alpha 侧 (基于 H_alpha):
        A_shared = H_alpha + E_alpha_shared(H_alpha)
        A_t1     = H_alpha + E_t1(H_alpha)
        A_t6     = H_alpha + E_t6(H_alpha)

    Beta 侧 (基于 H_beta):
        B_shared = H_beta + E_beta_shared(H_beta)
        B_245    = H_beta + E_245(H_beta)
        B_t2     = H_beta + E_t2(H_beta)
        B_t3     = H_beta + E_t3(H_beta)
        B_t4     = H_beta + E_t4(H_beta)
        B_t5     = H_beta + E_t5(H_beta)

    ================================================================
    v2 Gate 路由集合 (移除 base):
    ================================================================
    Alpha:
        - t1 routes over [A_shared, A_t1]          (2 experts)
        - t6 routes over [A_shared, A_t6]          (2 experts)

    Beta:
        - t2 routes over [B_shared, B_245, B_t2]   (3 experts)
        - t3 routes over [B_shared, B_t3]          (2 experts)
        - t4 routes over [B_shared, B_245, B_t4]   (3 experts)
        - t5 routes over [B_shared, B_245, B_t5]   (3 experts)

    ================================================================
    保留的 v1 设计:
    ================================================================
    - Alpha/Beta 双引擎分离
    - PMGT path for t1/t6
    - FlattenProjector path for t2/t3/t4/t5
    - StaticEncoder / Classifier per task
    - CE / LDAM / BCE + pos_weight losses
    - KD protection on t6
    - uncertainty weighting (log_vars)
    - 3-stage training skeleton
    - dev-stage threshold protocol: t3=0.5, t4=0.5, t5=0.6

    Args:
        task_specs: Dict[str, TaskSpec] 任务规格字典
        num_channels: 特征通道数 (默认 30)
        D_time: 时序编码基础维度 (默认 16)
        T_mid: CNN 中间时序维度 (默认 24)
        semantic_adj: 医学先验邻接矩阵
        config: MTL 配置对象
    """

    def __init__(
        self,
        task_specs: Dict[str, TaskSpec],
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        semantic_adj: Optional[torch.Tensor] = None,
        config: Optional[Any] = None
    ):
        super().__init__()
        self.task_specs = task_specs
        self.num_channels = num_channels
        self.D_time = D_time
        self.T_mid = T_mid
        self.architecture_version = "v2"  # 标记架构版本

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if semantic_adj is not None:
            self.semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32).to(device)
        else:
            self.semantic_adj = None

        print("\n" + "="*60)
        print("Protected Dual-Engine HCGC/PLE v2 Architecture")
        print("="*60)
        print("[v2] trunk output 不再作为 gate 候选")
        print("[v2] 专家改为残差变换 (H_trunk + delta)")
        print("[v2] gate 只在专家之间路由")
        print("="*60 + "\n")

        # =====================================================
        # Alpha/Beta Trunk 编码器 (v2 核心)
        # =====================================================
        # Alpha trunk: 提取 Alpha 分支基础表征
        self.alpha_trunk = AlphaDynamicEncoder(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=0.3
        )

        # Beta trunk: 提取 Beta 分支基础表征
        self.beta_trunk = BetaDynamicEncoder(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=0.3
        )

        print("[v2] Alpha trunk: AlphaDynamicEncoder (参数量: {:,})".format(
            sum(p.numel() for p in self.alpha_trunk.parameters())
        ))
        print("[v2] Beta trunk: BetaDynamicEncoder (参数量: {:,})".format(
            sum(p.numel() for p in self.beta_trunk.parameters())
        ))

        # =====================================================
        # Alpha 分支残差专家 (v2: 3个: shared, t1_private, t6_private)
        # =====================================================
        # 注意: 移除了 base，只保留 shared 和 private
        self.alpha_residual_experts = nn.ModuleDict({
            "shared": ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity="medium",
                expert_name="E_alpha_shared",
                dropout=0.1
            ),
            "t1_private": ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity="light",
                expert_name="E_t1",
                dropout=0.1
            ),
            "t6_private": ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity="light",
                expert_name="E_t6",
                dropout=0.1
            ),
        })

        print("[v2] Alpha 残差专家 (3个): shared(medium), t1_private(light), t6_private(light)")

        # =====================================================
        # Beta 分支残差专家 (v2: 6个: shared, group_245, t2~t5_private)
        # =====================================================
        # 注意: 移除了 base，只保留 shared, group_245 和 private
        self.beta_residual_experts = nn.ModuleDict({
            "shared": ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity="medium",
                expert_name="E_beta_shared",
                dropout=0.1
            ),
            "group_245": ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity="medium",
                expert_name="E_245",
                dropout=0.1
            ),
            "t2_private": ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity="light",
                expert_name="E_t2",
                dropout=0.1
            ),
            "t3_private": ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity="strong",  # t3 关键任务使用强容量
                expert_name="E_t3",
                dropout=0.1
            ),
            "t4_private": ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity="light",
                expert_name="E_t4",
                dropout=0.1
            ),
            "t5_private": ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity="light",
                expert_name="E_t5",
                dropout=0.1
            ),
        })

        print("[v2] Beta 残差专家 (6个): shared(medium), group_245(medium), t2~t5_private")

        # =====================================================
        # Gate Context 编码器 (保持 v1 设计)
        # =====================================================
        # Alpha context: H_alpha [B, C, D] -> [B, 32]
        self.alpha_gate_context = AlphaGateContextEncoder(
            input_channels=num_channels,
            input_dim=D_time,
            output_dim=32
        )

        # Beta context: H_beta [B, C, D] -> [B, 32]
        self.beta_gate_context = BetaGateContextEncoder(
            input_channels=num_channels,
            input_dim=D_time,
            output_dim=32
        )

        # 静态特征 MLP (共享): x_static [B, 5] -> [B, 8]
        self.shared_static_mlp = SharedGateStaticMLP(
            input_dim=5,
            hidden_dim=16,
            output_dim=8
        )

        # =====================================================
        # v2 任务特定门控网络 (维度更新: 移除 base)
        # =====================================================
        # v2 Gate 输出维度:
        #   - t1: 2 (shared, private)
        #   - t6: 2 (shared, private)
        #   - t2: 3 (shared, group_245, private)
        #   - t3: 2 (shared, private)
        #   - t4: 3 (shared, group_245, private)
        #   - t5: 3 (shared, group_245, private)

        self.alpha_gates = nn.ModuleDict({
            "t1": TaskSpecificGate(
                num_experts=2,  # v2: [shared, t1_private]
                context_dim=40,  # 32 (engine context) + 8 (static)
                tau_init=2.0,
                task_name="t1"
            ),
            "t6": TaskSpecificGate(
                num_experts=2,  # v2: [shared, t6_private]
                context_dim=40,
                tau_init=2.0,
                task_name="t6"
            ),
        })

        self.beta_gates = nn.ModuleDict({
            "t2": TaskSpecificGate(
                num_experts=3,  # v2: [shared, group_245, t2_private]
                context_dim=40,
                tau_init=2.0,
                task_name="t2"
            ),
            "t3": TaskSpecificGate(
                num_experts=2,  # v2: [shared, t3_private] (无组专家)
                context_dim=40,
                tau_init=2.0,
                task_name="t3"
            ),
            "t4": TaskSpecificGate(
                num_experts=3,  # v2: [shared, group_245, t4_private]
                context_dim=40,
                tau_init=2.0,
                task_name="t4"
            ),
            "t5": TaskSpecificGate(
                num_experts=3,  # v2: [shared, group_245, t5_private]
                context_dim=40,
                tau_init=2.0,
                task_name="t5"
            ),
        })

        print("[v2] Gate 维度: t1/t6/t3=2, t2/t4/t5=3")

        # =====================================================
        # v2 专家名称映射 (移除 base)
        # =====================================================
        self.alpha_expert_names = {
            "t1": ["shared", "t1_private"],      # v2: 移除 base
            "t6": ["shared", "t6_private"],      # v2: 移除 base
        }

        self.beta_expert_names = {
            "t2": ["shared", "group_245", "t2_private"],  # v2: 移除 base
            "t3": ["shared", "t3_private"],               # v2: 移除 base, 无组专家
            "t4": ["shared", "group_245", "t4_private"],  # v2: 移除 base
            "t5": ["shared", "group_245", "t5_private"],  # v2: 移除 base
        }
        # Alpha 任务: PMGT + 投影
        self.alpha_interactors = nn.ModuleDict({
            "t1": PriorMaskedTaskHead(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=48,
                semantic_adj=self.semantic_adj,
                dropout=0.5,
                gamma_init=1.0
            ),
            "t6": PriorMaskedTaskHead(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=48,
                semantic_adj=self.semantic_adj,
                dropout=0.5,
                gamma_init=1.0
            ),
        })

        # Beta 任务: FlattenProjector
        self.beta_projectors = nn.ModuleDict({
            "t2": FlattenProjector(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=48,
                use_two_layer=True
            ),
            "t3": FlattenProjector(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=48,
                use_two_layer=True
            ),
            "t4": FlattenProjector(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=48,
                use_two_layer=True
            ),
            "t5": FlattenProjector(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=48,
                use_two_layer=True
            ),
        })

        # =====================================================
        # 静态编码器和分类头 (每个任务独立)
        # =====================================================
        self.static_encoders = nn.ModuleDict({
            task_key: TaskStaticEncoder(in_dim=5, out_dim=16, dropout=0.2)
            for task_key in task_specs.keys()
        })

        self.classifiers = nn.ModuleDict({
            task_key: TaskClassifier(
                dyn_dim=48,
                static_dim=16,
                hidden_dim=32,
                num_classes=spec.num_classes,
                is_binary=spec.is_binary,
                dropout=spec.dropout
            )
            for task_key, spec in task_specs.items()
        })

        # =====================================================
        # 同方差不确定性权重 (log_vars)
        # =====================================================
        self.log_vars = nn.ParameterDict({
            task_key: nn.Parameter(torch.zeros(1, device=device))
            for task_key in task_specs.keys()
        })

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_static: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        task_mask: Optional[Dict[str, torch.Tensor]] = None,
        tau_override: Optional[float] = None,
        return_aux: bool = False,
        return_gate_weights: bool = False
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        前向传播 (v2 架构)

        v2 核心流程:
        1. Trunk 编码: H_alpha, H_beta
        2. 残差专家: A_shared = H_alpha + delta_shared
        3. Gate Context: 使用 trunk 输出 (不再是 base)
        4. Gate 路由: 只在专家之间路由，不包括 trunk
        5. 专家聚合: 加权求和残差专家输出
        6. 任务路径: PMGT / FlattenProjector

        Args:
            x_dyn: [B, L_max, C] 动态序列
            x_static: [B, 5] 静态特征
            lengths: [B] 真实长度 (可选)
            task_mask: {"t1": [B], ...} 任务掩码 (可选)
            tau_override: 温度覆盖值 (由温度调度器提供)
            return_aux: 是否返回辅助信息 (专家输出)
            return_gate_weights: 是否返回门控权重

        Returns:
            outputs: {"t1": {"logits": ..., ...}, ..., "t6": {...}}
            aux: {"alpha_expert_outputs": {...}, "beta_expert_outputs": {...}} (可选)
            gate_weights: {"t1": [...], ..., "t6": [...]} (可选)
        """
        B = x_dyn.size(0)
        device = x_dyn.device

        if task_mask is None:
            task_mask = {task_key: torch.ones(B, device=device) for task_key in self.task_specs.keys()}

        # =============================================
        # Step 1: Trunk 编码 (v2 核心)
        # =============================================
        # Alpha trunk: 提取 Alpha 分支基础表征
        H_alpha = self.alpha_trunk(x_dyn, lengths)  # [B, C, D_time]

        # Beta trunk: 提取 Beta 分支基础表征
        H_beta = self.beta_trunk(x_dyn, lengths)    # [B, C, D_time]

        # =============================================
        # Step 2: 计算残差专家输出 (v2 核心)
        # =============================================
        # Alpha 残差专家: A_xxx = H_alpha + delta
        alpha_expert_outputs = {}
        for expert_name, expert in self.alpha_residual_experts.items():
            delta = expert(H_alpha)  # 残差变换
            alpha_expert_outputs[expert_name] = H_alpha + delta  # 残差连接

        # Beta 残差专家: B_xxx = H_beta + delta
        beta_expert_outputs = {}
        for expert_name, expert in self.beta_residual_experts.items():
            delta = expert(H_beta)  # 残差变换
            beta_expert_outputs[expert_name] = H_beta + delta  # 残差连接

        # =============================================
        # Step 3: 计算 Gate Context (v2: 使用 trunk 输出)
        # =============================================
        # v2 关键改变: Gate context 使用 trunk 输出，不再是 base 专家输出
        c_alpha = self.alpha_gate_context(H_alpha)  # [B, 32]
        c_beta = self.beta_gate_context(H_beta)     # [B, 32]

        # 静态特征编码 (共享)
        c_static = self.shared_static_mlp(x_static)  # [B, 8]

        # 组合 context
        c_alpha_full = torch.cat([c_alpha, c_static], dim=1)  # [B, 40]
        c_beta_full = torch.cat([c_beta, c_static], dim=1)   # [B, 40]

        # =============================================
        # Step 4: 计算任务特定门控权重
        # =============================================
        alpha_gate_weights = {}
        for task_key in ["t1", "t6"]:
            alpha_gate_weights[task_key] = self.alpha_gates[task_key](
                c_alpha_full, tau_override=tau_override
            )  # [B, 2] (v2: 只有2个专家)

        beta_gate_weights = {}
        for task_key in ["t2", "t3", "t4", "t5"]:
            beta_gate_weights[task_key] = self.beta_gates[task_key](
                c_beta_full, tau_override=tau_override
            )  # [B, 2] or [B, 3] (v2: 根据任务不同)

        # =============================================
        # Step 5: 专家聚合 (加权求和) - v2: 维度统一
        # =============================================
        # Alpha 任务聚合
        alpha_gated_outputs = {}
        for task_key in ["t1", "t6"]:
            expert_names = self.alpha_expert_names[task_key]  # v2: ["shared", "t1_private"] 等
            weights = alpha_gate_weights[task_key]  # [B, 2]

            # v2: 维度统一，因为都是基于同一个 trunk
            H_gated = torch.zeros_like(H_alpha)
            for i, expert_name in enumerate(expert_names):
                H_gated = H_gated + weights[:, i:i+1, None] * alpha_expert_outputs[expert_name]

            alpha_gated_outputs[task_key] = H_gated

        # Beta 任务聚合
        beta_gated_outputs = {}
        for task_key in ["t2", "t3", "t4", "t5"]:
            expert_names = self.beta_expert_names[task_key]
            weights = beta_gate_weights[task_key]  # [B, 2] or [B, 3]

            # v2: 维度统一，因为都是基于同一个 trunk
            H_gated = torch.zeros_like(H_beta)
            for i, expert_name in enumerate(expert_names):
                H_gated = H_gated + weights[:, i:i+1, None] * beta_expert_outputs[expert_name]

            beta_gated_outputs[task_key] = H_gated

        # =============================================
        # Step 6: 任务路径处理 (PMGT / FlattenProjector)
        # =============================================
        outputs = {}

        # Alpha 任务 (t1, t6): PMGT 路线
        for task_key in ["t1", "t6"]:
            H_gated = alpha_gated_outputs[task_key]

            # PMGT 交互
            dyn_feat = self.alpha_interactors[task_key](H_gated)  # [B, 48]

            # 静态特征
            static_feat = self.static_encoders[task_key](x_static)  # [B, 16]

            # 分类
            logits = self.classifiers[task_key](dyn_feat, static_feat)

            outputs[task_key] = {
                "logits": logits,
                "dyn_feat": dyn_feat,
                "static_feat": static_feat,
                "fused_feat": torch.cat([dyn_feat, static_feat], dim=1)
            }

        # Beta 任务 (t2~t5): FlattenProjector 路线
        for task_key in ["t2", "t3", "t4", "t5"]:
            H_gated = beta_gated_outputs[task_key]

            # Flatten + MLP
            dyn_feat = self.beta_projectors[task_key](H_gated)  # [B, 48]

            # 静态特征
            static_feat = self.static_encoders[task_key](x_static)  # [B, 16]

            # 分类
            logits = self.classifiers[task_key](dyn_feat, static_feat)

            outputs[task_key] = {
                "logits": logits,
                "dyn_feat": dyn_feat,
                "static_feat": static_feat,
                "fused_feat": torch.cat([dyn_feat, static_feat], dim=1)
            }

        # 辅助信息
        if return_aux:
            outputs["aux"] = {
                "H_alpha": H_alpha,  # v2: trunk 输出
                "H_beta": H_beta,    # v2: trunk 输出
                "alpha_expert_outputs": alpha_expert_outputs,  # 残差专家输出
                "beta_expert_outputs": beta_expert_outputs,
                "alpha_gated_outputs": alpha_gated_outputs,
                "beta_gated_outputs": beta_gated_outputs,
            }

        # 门控权重 (用于日志)
        if return_gate_weights:
            outputs["gate_weights"] = {
                **alpha_gate_weights,
                **beta_gate_weights
            }

        return outputs

    def freeze_alpha_modules(self):
        """冻结 Alpha 分支所有模块 (v2)"""
        # Trunk
        for param in self.alpha_trunk.parameters():
            param.requires_grad = False

        # 残差专家
        for expert in self.alpha_residual_experts.values():
            for param in expert.parameters():
                param.requires_grad = False

        # 门控
        for gate in self.alpha_gates.values():
            for param in gate.parameters():
                param.requires_grad = False

        # 交互头
        for interactor in self.alpha_interactors.values():
            for param in interactor.parameters():
                param.requires_grad = False

        # 静态编码器和分类头
        for task_key in ["t1", "t6"]:
            for param in self.static_encoders[task_key].parameters():
                param.requires_grad = False
            for param in self.classifiers[task_key].parameters():
                param.requires_grad = False

        # log_vars
        for task_key in ["t1", "t6"]:
            self.log_vars[task_key].requires_grad = False

    def freeze_beta_modules(self):
        """冻结 Beta 分支所有模块 (v2 + Ablation E2 支持)"""
        # Trunk (Ablation E2: None)
        if self.beta_trunk is not None:
            for param in self.beta_trunk.parameters():
                param.requires_grad = False

        # 残差专家 (Ablation E2: None)
        if self.beta_residual_experts is not None and len(self.beta_residual_experts) > 0:
            for expert in self.beta_residual_experts.values():
                for param in expert.parameters():
                    param.requires_grad = False

        # 门控 (Ablation E2: None)
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            for gate in self.beta_gates.values():
                for param in gate.parameters():
                    param.requires_grad = False

        # 投影头 (Ablation E2: None)
        if self.beta_projectors is not None and len(self.beta_projectors) > 0:
            for projector in self.beta_projectors.values():
                for param in projector.parameters():
                    param.requires_grad = False

        # 静态编码器和分类头
        for task_key in ["t2", "t3", "t4", "t5"]:
            for param in self.static_encoders[task_key].parameters():
                param.requires_grad = False
            for param in self.classifiers[task_key].parameters():
                param.requires_grad = False

        # log_vars
        for task_key in ["t2", "t3", "t4", "t5"]:
            self.log_vars[task_key].requires_grad = False

    def unfreeze_all(self):
        """解冻所有模块"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_alpha_branch(self):
        """冻结 Alpha 分支 (别名方法，兼容 MTLTrainer)"""
        self.freeze_alpha_modules()

    def freeze_beta_branch(self):
        """冻结 Beta 分支 (别名方法，兼容 MTLTrainer)"""
        self.freeze_beta_modules()

    def get_num_parameters(self) -> Dict[str, int]:
        """获取各模块参数量 (v2)"""
        counts = {}

        # Alpha trunk
        counts["alpha_trunk"] = sum(p.numel() for p in self.alpha_trunk.parameters())

        # Alpha 残差专家
        counts["alpha_residual_experts"] = sum(
            sum(p.numel() for p in expert.parameters())
            for expert in self.alpha_residual_experts.values()
        )

        # Beta trunk
        counts["beta_trunk"] = sum(p.numel() for p in self.beta_trunk.parameters())

        # Beta 残差专家
        counts["beta_residual_experts"] = sum(
            sum(p.numel() for p in expert.parameters())
            for expert in self.beta_residual_experts.values()
        )

        # 门控
        counts["alpha_gates"] = sum(
            sum(p.numel() for p in gate.parameters())
            for gate in self.alpha_gates.values()
        )
        # Beta gates (Ablation E2: None - v2 不支持但保留检查)
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            counts["beta_gates"] = sum(
                sum(p.numel() for p in gate.parameters())
                for gate in self.beta_gates.values()
            )
        else:
            counts["beta_gates"] = 0

        # Gate context (Ablation E2: None - v2 不支持但保留检查)
        if self.beta_gate_context is not None:
            counts["beta_gate_context"] = sum(p.numel() for p in self.beta_gate_context.parameters())
        else:
            counts["beta_gate_context"] = 0
        counts["shared_static_mlp"] = sum(p.numel() for p in self.shared_static_mlp.parameters())

        # 交互头
        counts["alpha_interactors"] = sum(
            sum(p.numel() for p in interactor.parameters())
            for interactor in self.alpha_interactors.values()
        )
        counts["beta_projectors"] = sum(
            sum(p.numel() for p in projector.parameters())
            for projector in self.beta_projectors.values()
        )

        # 静态编码器和分类头
        counts["static_encoders"] = sum(
            sum(p.numel() for p in encoder.parameters())
            for encoder in self.static_encoders.values()
        )
        counts["classifiers"] = sum(
            sum(p.numel() for p in classifier.parameters())
            for classifier in self.classifiers.values()
        )

        counts["log_vars"] = sum(p.numel() for p in self.log_vars.parameters())
        counts["total"] = sum(counts.values())

        return counts

    def __repr__(self) -> str:
        counts = self.get_num_parameters()
        return (
            f"ProtectedDualEngineMTL (v2 Architecture)\n"
            f"  num_channels={self.num_channels}\n"
            f"  D_time={self.D_time}\n"
            f"  alpha_trunk + {len(self.alpha_residual_experts)} residual experts\n"
            f"  beta_trunk + {len(self.beta_residual_experts)} residual experts\n"
            f"  total_params={counts['total']:,}\n"
            f")"
        )


# =============================================================================
# ProtectedDualEngineMTL_v3 - 简化架构 (删除Alpha CGC, 收缩Beta专家)
# =============================================================================

class ProtectedDualEngineMTL_v3(nn.Module):
    """
    Protected Dual-Engine MTL v3 Architecture

    v3 核心改变:
    - Alpha删除CGC: trunk输出直接传递到PMGT，无专家聚合
    - Beta收缩为三专家: shared, group_245, t3_private
    - Beta gates改为2维: t2/t4/t5只有[shared, group_245]

    设计理念:
    - v2实验结果显示Alpha CGC未带来明确增益
    - Beta侧私有专家(t2/t4/t5)过度专家化，未能获益
    - t3是唯一需要强私有专家的任务

    创建日期: 2026-04-17
    """

    def __init__(
        self,
        task_specs: Dict[str, TaskSpec],
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        device: str = "cpu",
        semantic_adj: Optional[List[List[float]]] = None,
        config: Optional[Dict] = None  # [v3 Config-Driven] 新增配置参数
    ):
        super().__init__()
        self.task_specs = task_specs
        self.num_channels = num_channels
        self.D_time = D_time
        self.T_mid = T_mid
        self.device = device

        # =====================================================
        # [v3 Config-Driven] 配置解析
        # =====================================================
        if config is None:
            config = {}

        # 提取各级配置块
        hcgc_config = config.get('hcgc_v4', {})  # v3 继承 v4 配置结构
        alpha_config = config.get('alpha', {})
        beta_config = config.get('beta', {})
        gate_config = hcgc_config.get('gate', {})
        gate_routing_config = hcgc_config.get('gate_routing', {})
        beta_experts_config = hcgc_config.get('beta', {}).get('residual_experts', {})

        # 提取配置值
        self.tau_start = gate_config.get('tau_start', 2.0)
        alpha_dropout = alpha_config.get('dropout', 0.3)
        alpha_hidden_dim = alpha_config.get('hidden_dim', 48)
        pmgt_dropout = alpha_config.get('pmgt_dropout', 0.5)
        gamma_init = alpha_config.get('gamma_init', 1.0)
        beta_dropout = beta_config.get('dropout', 0.3)
        beta_hidden_dim = beta_config.get('hidden_dim', 48)
        expert_dropout = hcgc_config.get('beta', {}).get('dropout', 0.1)

        # =====================================================
        # [v3 Config-Driven] Alpha/Beta 任务分组
        # =====================================================
        self.alpha_tasks = [k for k, spec in task_specs.items() if spec.branch == "alpha"]
        self.beta_tasks = [k for k, spec in task_specs.items() if spec.branch == "beta"]

        # Semantic adjacency for PMGT
        if semantic_adj is not None:
            self.semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32).to(device)
        else:
            self.semantic_adj = None

        print("\n" + "="*60)
        print("Protected Dual-Engine HCGC/PLE v3 Architecture")
        print("="*60)
        print("[v3] Alpha删除CGC: trunk输出直接传递，无专家聚合")
        print("[v3] Beta收缩为三专家: shared, group_245, t3_private")
        print("[v3] Beta gates改为2维: t2/t4/t5只有[shared, group_245]")
        print(f"[v3] Alpha任务: {self.alpha_tasks}")
        print(f"[v3] Beta任务: {self.beta_tasks}")
        print(f"[v3] tau_start: {self.tau_start}")
        print("="*60 + "\n")

        # =====================================================
        # Alpha/Beta Trunk 编码器 (Config-Driven)
        # =====================================================
        self.alpha_trunk = AlphaDynamicEncoder(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=alpha_dropout
        )

        self.beta_trunk = BetaDynamicEncoder(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=beta_dropout
        )

        print("[v3] Alpha trunk: AlphaDynamicEncoder (参数量: {:,})".format(
            sum(p.numel() for p in self.alpha_trunk.parameters())
        ))
        print("[v3] Beta trunk: BetaDynamicEncoder (参数量: {:,})".format(
            sum(p.numel() for p in self.beta_trunk.parameters())
        ))

        # =====================================================
        # v3: Alpha无残差专家 (删除CGC)
        # =====================================================
        self.alpha_residual_experts = None
        self.alpha_gate_context = None
        self.alpha_gates = None
        self.alpha_expert_names = None

        print("[v3] Alpha无残差专家 (CGC已删除)")

        # =====================================================
        # [v3 Config-Driven] Beta分支残差专家
        # =====================================================
        self.beta_expert_names_from_config = list(beta_experts_config.keys()) if beta_experts_config else ["shared", "group_245", "t3_private"]

        self.beta_residual_experts = nn.ModuleDict({
            expert_name: ResidualExpert(
                num_channels=num_channels,
                D_time=D_time,
                capacity=beta_experts_config.get(expert_name, {}).get('capacity', 'medium'),
                expert_name=f"E_{expert_name}",
                dropout=expert_dropout
            )
            for expert_name in self.beta_expert_names_from_config
        })

        print(f"[v3] Beta 残差专家 ({len(self.beta_expert_names_from_config)}个): {self.beta_expert_names_from_config}")

        # =====================================================
        # v3: Gate Context 编码器 (仅Beta)
        # =====================================================
        self.beta_gate_context = BetaGateContextEncoder(
            input_channels=num_channels,
            input_dim=D_time,
            output_dim=32
        )

        # 静态特征 MLP (共享): x_static [B, 5] -> [B, 8]
        self.shared_static_mlp = SharedGateStaticMLP(
            input_dim=5,
            hidden_dim=16,
            output_dim=8
        )

        # =====================================================
        # [v3 Config-Driven] Beta Gates
        # =====================================================
        gate_routing_beta = gate_routing_config.get('beta', {})

        # 构建 beta_expert_names（配置优先）
        self.beta_expert_names = {}
        for task_key in self.beta_tasks:
            if task_key in gate_routing_beta:
                self.beta_expert_names[task_key] = gate_routing_beta[task_key]
            else:
                # 默认路由
                if task_key == "t3":
                    self.beta_expert_names[task_key] = ["shared", "t3_private"]
                else:
                    self.beta_expert_names[task_key] = ["shared", "group_245"]

        self.beta_gates = nn.ModuleDict({
            task_key: TaskSpecificGate(
                num_experts=len(self.beta_expert_names[task_key]),
                context_dim=40,
                tau_init=self.tau_start,
                task_name=task_key
            )
            for task_key in self.beta_tasks
        })

        print(f"[v3] Beta Gate tau_start={self.tau_start}, 专家路由: {self.beta_expert_names}")

        # =====================================================
        # [v3 Config-Driven] Alpha 任务 PMGT
        # =====================================================
        self.alpha_interactors = nn.ModuleDict({
            task_key: PriorMaskedTaskHead(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=alpha_hidden_dim,
                semantic_adj=self.semantic_adj,
                dropout=pmgt_dropout,
                gamma_init=gamma_init
            )
            for task_key in self.alpha_tasks
        })

        # =====================================================
        # [v3 Config-Driven] Beta 任务 FlattenProjector
        # =====================================================
        self.beta_projectors = nn.ModuleDict({
            task_key: FlattenProjector(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=beta_hidden_dim,
                use_two_layer=True
            )
            for task_key in self.beta_tasks
        })

        # =====================================================
        # 静态编码器和分类头 (每个任务独立)
        # =====================================================
        self.static_encoders = nn.ModuleDict({
            task_key: TaskStaticEncoder(in_dim=5, out_dim=16, dropout=0.2)
            for task_key in task_specs.keys()
        })

        self.classifiers = nn.ModuleDict({
            task_key: TaskClassifier(
                dyn_dim=48,
                static_dim=16,
                hidden_dim=32,
                num_classes=spec.num_classes,
                is_binary=spec.is_binary,
                dropout=spec.dropout
            )
            for task_key, spec in task_specs.items()
        })

        # =====================================================
        # 同方差不确定性权重 (log_vars)
        # =====================================================
        self.log_vars = nn.ParameterDict({
            task_key: nn.Parameter(torch.zeros(1, device=device))
            for task_key in task_specs.keys()
        })

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_static: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        task_mask: Optional[Dict[str, torch.Tensor]] = None,
        tau_override: Optional[float] = None,
        return_aux: bool = False,
        return_gate_weights: bool = False,
        skip_beta_branch: bool = False  # [新增 - 2026-04-29] 跳过 Beta 分支计算
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        前向传播 (v3 架构)

        v3 核心流程:
        1. Trunk 编码: H_alpha, H_beta
        2. Alpha直接传递: 无专家聚合，H_alpha直接送入PMGT
        3. Beta残差专家: B_xxx = H_beta + delta (只有3个专家)
        4. Beta Gate Context: 使用 trunk 输出
        5. Beta Gate 路由: 2专家聚合
        6. 任务路径: PMGT / FlattenProjector

        Args:
            x_dyn: [B, L_max, C] 动态序列
            x_static: [B, 5] 静态特征
            lengths: [B] 真实长度 (可选)
            task_mask: {"t1": [B], ...} 任务掩码 (可选)
            tau_override: 温度覆盖值
            return_aux: 是否返回辅助信息
            return_gate_weights: 是否返回门控权重
            skip_beta_branch: 是否跳过 Beta 分支计算 (Stage1/Stage2冻结时)

        Returns:
            outputs: {"t1": {"logits": ..., ...}, ..., "t6": {...}}
            aux: {"H_alpha": ..., "H_beta": ..., "beta_expert_outputs": {...}} (可选)
            gate_weights: {"t2": [...], ..., "t5": [...]} (可选，Alpha无gate)
        """
        B = x_dyn.size(0)
        device = x_dyn.device

        if task_mask is None:
            task_mask = {task_key: torch.ones(B, device=device) for task_key in self.task_specs.keys()}

        # =============================================
        # Step 1: Trunk 编码
        # =============================================
        H_alpha = self.alpha_trunk(x_dyn, lengths)  # [B, C, D_time]

        # [优化 - 2026-04-29] skip_beta_branch 时跳过 beta_trunk
        if skip_beta_branch:
            H_beta = None
        else:
            H_beta = self.beta_trunk(x_dyn, lengths)    # [B, C, D_time]

        # =============================================
        # Step 2: Alpha直接传递 (Config-Driven)
        # =============================================
        alpha_gated_outputs = {
            task_key: H_alpha for task_key in self.alpha_tasks
        }

        # =============================================
        # Step 3: Beta残差专家 (Config-Driven)
        # [优化 - 2026-04-29] skip_beta_branch 时跳过
        # =============================================
        beta_expert_outputs = {}
        if not skip_beta_branch:
            for expert_name in self.beta_expert_names_from_config:
                expert = self.beta_residual_experts[expert_name]
                delta = expert(H_beta)
                beta_expert_outputs[expert_name] = H_beta + delta

        # =============================================
        # Step 4: Beta Gate Context
        # [优化 - 2026-04-29] skip_beta_branch 时跳过
        # =============================================
        c_beta_full = None
        if not skip_beta_branch:
            c_beta = self.beta_gate_context(H_beta)  # [B, 32]
            c_static = self.shared_static_mlp(x_static)  # [B, 8]
            c_beta_full = torch.cat([c_beta, c_static], dim=1)  # [B, 40]

        # =============================================
        # Step 5: Beta Gate 权重 (Config-Driven)
        # [优化 - 2026-04-29] skip_beta_branch 时跳过
        # =============================================
        beta_gate_weights = {}
        if not skip_beta_branch:
            for task_key in self.beta_tasks:
                beta_gate_weights[task_key] = self.beta_gates[task_key](
                    c_beta_full, tau_override=tau_override
                )

        # =============================================
        # Step 6: Beta专家聚合 (Config-Driven)
        # [优化 - 2026-04-29] skip_beta_branch 时跳过
        # =============================================
        beta_gated_outputs = {}
        if not skip_beta_branch:
            for task_key in self.beta_tasks:
                expert_names = self.beta_expert_names[task_key]
                weights = beta_gate_weights[task_key]

                H_gated = torch.zeros_like(H_beta)
                for i, expert_name in enumerate(expert_names):
                    H_gated = H_gated + weights[:, i:i+1, None] * beta_expert_outputs[expert_name]

                beta_gated_outputs[task_key] = H_gated

        # =============================================
        # Step 7: 任务路径处理 (Config-Driven)
        # =============================================
        outputs = {}

        # Alpha任务: PMGT路线
        for task_key in self.alpha_tasks:
            H_gated = alpha_gated_outputs[task_key]

            dyn_feat = self.alpha_interactors[task_key](H_gated)
            static_feat = self.static_encoders[task_key](x_static)
            logits = self.classifiers[task_key](dyn_feat, static_feat)

            outputs[task_key] = {
                "logits": logits,
                "dyn_feat": dyn_feat,
                "static_feat": static_feat,
                "fused_feat": torch.cat([dyn_feat, static_feat], dim=1)
            }

        # Beta任务: FlattenProjector路线
        # [优化 - 2026-04-29] skip_beta_branch 时跳过
        if not skip_beta_branch:
            for task_key in self.beta_tasks:
                H_gated = beta_gated_outputs[task_key]

                dyn_feat = self.beta_projectors[task_key](H_gated)
                static_feat = self.static_encoders[task_key](x_static)
                logits = self.classifiers[task_key](dyn_feat, static_feat)

                outputs[task_key] = {
                    "logits": logits,
                    "dyn_feat": dyn_feat,
                    "static_feat": static_feat,
                    "fused_feat": torch.cat([dyn_feat, static_feat], dim=1)
                }

        # 辅助信息
        if return_aux:
            outputs["aux"] = {
                "H_alpha": H_alpha,
                "H_beta": H_beta,
                "beta_expert_outputs": beta_expert_outputs,
                "alpha_gated_outputs": alpha_gated_outputs,
                "beta_gated_outputs": beta_gated_outputs,
            }

        # 门控权重 (v3: 只有Beta)
        if return_gate_weights:
            outputs["gate_weights"] = beta_gate_weights  # v3: 只有t2~t5

        return outputs

    def freeze_alpha_modules(self):
        """冻结Alpha分支所有模块 (v3简化版)"""
        # Trunk
        for param in self.alpha_trunk.parameters():
            param.requires_grad = False

        # v3: Alpha无专家，无gate，无需冻结

        # 交互头 (PMGT)
        for interactor in self.alpha_interactors.values():
            for param in interactor.parameters():
                param.requires_grad = False

        # 静态编码器和分类头 (Config-Driven)
        for task_key in self.alpha_tasks:
            for param in self.static_encoders[task_key].parameters():
                param.requires_grad = False
            for param in self.classifiers[task_key].parameters():
                param.requires_grad = False
            self.log_vars[task_key].requires_grad = False

    def freeze_beta_modules(self):
        """冻结Beta分支所有模块 (Config-Driven + Ablation 支持)"""
        # Trunk (Ablation E2: None)
        if self.beta_trunk is not None:
            for param in self.beta_trunk.parameters():
                param.requires_grad = False

        # 残差专家 (Config-Driven + Ablation E2: None)
        if self.beta_residual_experts is not None and len(self.beta_residual_experts) > 0:
            for expert_name in self.beta_expert_names_from_config:
                if expert_name in self.beta_residual_experts:
                    for param in self.beta_residual_experts[expert_name].parameters():
                        param.requires_grad = False

        # Gate (Ablation E2/S3: None)
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            for gate in self.beta_gates.values():
                for param in gate.parameters():
                    param.requires_grad = False

        # Gate context (Ablation E2: None)
        if self.beta_gate_context is not None:
            for param in self.beta_gate_context.parameters():
                param.requires_grad = False

        # 投影头 (Ablation E2: None)
        if self.beta_projectors is not None and len(self.beta_projectors) > 0:
            for projector in self.beta_projectors.values():
                for param in projector.parameters():
                    param.requires_grad = False

        # 静态编码器和分类头 (Config-Driven)
        for task_key in self.beta_tasks:
            for param in self.static_encoders[task_key].parameters():
                param.requires_grad = False
            for param in self.classifiers[task_key].parameters():
                param.requires_grad = False
            if task_key in self.log_vars:
                self.log_vars[task_key].requires_grad = False

    def unfreeze_all(self):
        """解冻所有模块"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_alpha_branch(self):
        """冻结Alpha分支 (别名方法)"""
        self.freeze_alpha_modules()

    def freeze_beta_branch(self):
        """冻结Beta分支 (别名方法)"""
        self.freeze_beta_modules()

    def get_num_parameters(self) -> Dict[str, int]:
        """获取各模块参数量 (v3)"""
        counts = {}

        # Alpha trunk
        counts["alpha_trunk"] = sum(p.numel() for p in self.alpha_trunk.parameters())

        # v3: Alpha无残差专家
        counts["alpha_residual_experts"] = 0

        # Beta trunk
        counts["beta_trunk"] = sum(p.numel() for p in self.beta_trunk.parameters())

        # Beta 残差专家 (v3: 只有3个)
        counts["beta_residual_experts"] = sum(
            sum(p.numel() for p in expert.parameters())
            for expert in self.beta_residual_experts.values()
        )

        # v3: Alpha无gate
        counts["alpha_gates"] = 0
        counts["alpha_gate_context"] = 0

        # Beta gates (Ablation E2: None)
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            counts["beta_gates"] = sum(
                sum(p.numel() for p in gate.parameters())
                for gate in self.beta_gates.values()
            )
        else:
            counts["beta_gates"] = 0

        # Gate context (Ablation E2: None)
        if self.beta_gate_context is not None:
            counts["beta_gate_context"] = sum(p.numel() for p in self.beta_gate_context.parameters())
        else:
            counts["beta_gate_context"] = 0
        counts["shared_static_mlp"] = sum(p.numel() for p in self.shared_static_mlp.parameters())

        # 交互头
        counts["alpha_interactors"] = sum(
            sum(p.numel() for p in interactor.parameters())
            for interactor in self.alpha_interactors.values()
        )
        counts["beta_projectors"] = sum(
            sum(p.numel() for p in projector.parameters())
            for projector in self.beta_projectors.values()
        )

        # 静态编码器和分类头
        counts["static_encoders"] = sum(
            sum(p.numel() for p in encoder.parameters())
            for encoder in self.static_encoders.values()
        )
        counts["classifiers"] = sum(
            sum(p.numel() for p in classifier.parameters())
            for classifier in self.classifiers.values()
        )

        counts["log_vars"] = sum(p.numel() for p in self.log_vars.parameters())
        counts["total"] = sum(counts.values())

        return counts

    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        返回按模块分组的参数字典，用于精细化学习率缩放和冻结控制。

        Returns:
            {
                "alpha_trunk": [...],          # alpha_trunk encoder
                "alpha_interactors": [...],    # alpha_interactors (PMGT heads)
                "beta_trunk": [...],           # beta_trunk encoder
                "beta_residual_experts": [...], # beta_residual_experts (shared, group_245, t3_private)
                "beta_gates": [...],           # beta_gates (t2, t3, t4, t5)
                "t2_gate": [...],              # [新增] 任务专属 Gate
                "t3_gate": [...],
                "t4_gate": [...],
                "t5_gate": [...],
                "beta_gate_context": [...],    # beta_gate_context encoder
                "beta_projectors": [...],      # beta_projectors (t2, t3, t4, t5)
                "t2_projector": [...],         # [新增] 任务专属 Projector
                "t3_projector": [...],
                "t4_projector": [...],
                "t5_projector": [...],
                "beta_shared_expert": [...],   # [新增] 任务专属 Expert
                "beta_group_245_expert": [...],
                "beta_t3_private_expert": [...],
                "static_encoders": [...],      # static_encoders
                "classifiers": [...],          # classifiers (task heads)
                "log_vars": [...],             # uncertainty weighting log_vars
                "shared_static_mlp": [...],    # shared_static_mlp
            }
        """
        groups = {}

        # Alpha trunk
        groups["alpha_trunk"] = list(self.alpha_trunk.parameters())

        # Alpha interactors (PMGT for t1, t6)
        groups["alpha_interactors"] = []
        for interactor in self.alpha_interactors.values():
            groups["alpha_interactors"].extend(list(interactor.parameters()))

        # Beta trunk
        groups["beta_trunk"] = list(self.beta_trunk.parameters())

        # Beta residual experts (v3: only 3)
        groups["beta_residual_experts"] = []
        for expert in self.beta_residual_experts.values():
            groups["beta_residual_experts"].extend(list(expert.parameters()))

        # [新增 - 2026-04-22] 任务专属 Expert (用于精细冻结控制)
        for expert_name, expert in self.beta_residual_experts.items():
            groups[f"beta_{expert_name}_expert"] = list(expert.parameters())

        # Beta gates (t2, t3, t4, t5) (Ablation E2: None)
        groups["beta_gates"] = []
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            for gate in self.beta_gates.values():
                groups["beta_gates"].extend(list(gate.parameters()))

            # [新增 - 2026-04-22] 任务专属 Gate (用于精细冻结控制)
            for task_name, gate in self.beta_gates.items():
                groups[f"{task_name}_gate"] = list(gate.parameters())

        # Beta gate context (Ablation E2: None)
        if self.beta_gate_context is not None:
            groups["beta_gate_context"] = list(self.beta_gate_context.parameters())
        else:
            groups["beta_gate_context"] = []

        # Beta projectors (t2, t3, t4, t5)
        groups["beta_projectors"] = []
        for projector in self.beta_projectors.values():
            groups["beta_projectors"].extend(list(projector.parameters()))

        # [新增 - 2026-04-22] 任务专属 Projector (用于精细冻结控制)
        for task_name, projector in self.beta_projectors.items():
            groups[f"{task_name}_projector"] = list(projector.parameters())

        # Static encoders
        groups["static_encoders"] = []
        for encoder in self.static_encoders.values():
            groups["static_encoders"].extend(list(encoder.parameters()))

        # Classifiers (task heads)
        groups["classifiers"] = []
        for classifier in self.classifiers.values():
            groups["classifiers"].extend(list(classifier.parameters()))

        # Log vars (uncertainty weights)
        groups["log_vars"] = list(self.log_vars.parameters())

        # Shared static MLP
        groups["shared_static_mlp"] = list(self.shared_static_mlp.parameters())

        return groups

    def freeze_specific_modules(self, module_names: List[str]):
        """
        精细化冻结指定模块。

        Args:
            module_names: 要冻结的模块名称列表，如 ["beta_gates", "beta_gate_context", "alpha_trunk"]
        """
        param_groups = self.get_parameter_groups()

        frozen_count = 0
        for module_name in module_names:
            if module_name in param_groups:
                params = param_groups[module_name]
                for param in params:
                    param.requires_grad = False
                    frozen_count += 1
                print(f"[freeze_specific_modules] 冻结 {module_name}: {len(params)} 个参数")
            else:
                print(f"[freeze_specific_modules] 警告: 未找到模块 {module_name}")

        print(f"[freeze_specific_modules] 共冻结 {frozen_count} 个参数")

    def freeze_task_specific_modules(self, task_modules: Dict[str, List[str]]):
        """
        [新增 - 2026-04-22] 任务专属模块冻结，支持精细冻结特定任务的 Gate/Expert。

        Args:
            task_modules: 任务模块配置字典，如:
                {
                    "t3": ["t3_gate", "beta_t3_private_expert"],  # 冻结 t3 的 gate 和 private expert
                    "t2": ["t2_gate"],  # 仅冻结 t2 的 gate
                }

        Example:
            # v3.1: 仅冻结 t3 专属模块，保护 t3 单任务性能
            model.freeze_task_specific_modules({
                "t3": ["t3_gate", "beta_t3_private_expert"]
            })
        """
        param_groups = self.get_parameter_groups()
        frozen_count = 0

        for task_name, module_list in task_modules.items():
            for module_name in module_list:
                # 支持两种命名方式:
                # 1. "{task}_gate" -> "t3_gate"
                # 2. "beta_{expert}_expert" -> "beta_t3_private_expert"
                if module_name in param_groups:
                    params = param_groups[module_name]
                    for param in params:
                        param.requires_grad = False
                        frozen_count += 1
                    print(f"[freeze_task_specific_modules] 冻结 {task_name}/{module_name}: {len(params)} 个参数")
                else:
                    print(f"[freeze_task_specific_modules] 警告: 未找到模块 {module_name} (任务 {task_name})")

        print(f"[freeze_task_specific_modules] 共冻结 {frozen_count} 个参数")

    def get_freeze_status(self) -> Dict[str, Dict[str, Any]]:
        """
        返回各模块冻结状态统计，用于日志记录。

        Returns:
            {
                "alpha_trunk": {"total": N, "frozen": M, "ratio": M/N},
                ...
            }
        """
        param_groups = self.get_parameter_groups()
        status = {}

        for module_name, params in param_groups.items():
            total = len(params)
            frozen = sum(1 for p in params if not p.requires_grad)
            ratio = frozen / total if total > 0 else 0.0
            status[module_name] = {
                "total": total,
                "frozen": frozen,
                "ratio": ratio
            }

        return status

    def set_gate_temperature(self, tau: float):
        """设置所有 Beta Gate 的温度参数（用于 Phase2 温度锁定）。

        Args:
            tau: 目标温度值 (如 0.8)

        Note:
            v3架构只有 beta_gates，无 alpha_gates。
            Ablation E2 模式下 beta_gates 为 None，跳过。
        """
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            for gate_name, gate in self.beta_gates.items():
                if hasattr(gate, 'set_tau'):
                    gate.set_tau(tau)
                print(f"[set_gate_temperature] {gate_name}.tau = {tau}")
            else:
                print(f"[set_gate_temperature] Warning: {gate_name} 没有 set_tau 方法")

        print(f"[set_gate_temperature] 所有 Beta Gate 温度已设置为 {tau}")

    def __repr__(self) -> str:
        counts = self.get_num_parameters()
        return (
            f"ProtectedDualEngineMTL_v3 (简化架构)\n"
            f"  num_channels={self.num_channels}\n"
            f"  D_time={self.D_time}\n"
            f"  alpha_trunk -> PMGT (无CGC)\n"
            f"  beta_trunk + 3 residual experts\n"
            f"  beta_gates: 全部2维\n"
            f"  total_params={counts['total']:,}\n"
            f")"
        )


# =============================================================================
# ProtectedDualEngineMTL_v4: T6-guided Context Injection Architecture
# =============================================================================

class ProtectedDualEngineMTL_v4(nn.Module):
    """
    Protected Dual-Engine MTL v4 Architecture - T6-guided Context Injection.

    v4 核心改变:
    - t6 角色: 从"锚定保护任务"转为"疾病上下文提供者"
    - dyn_feat_t6: 压缩为 c6_deep，注入 t1-t5 预测路径
    - Beta Gate context_dim: 从 40 扩展到 56 (+ c6_deep[16])
    - t1 预测路径: 上下文增强 dyn_feat_t1 + alpha * delta
    - KD 机制: 废弃 (use_kd_t6=False)
    - checkpoint 指标: weighted_macro_f1_t1_to_t5

    设计理念:
    - v3/v3.1 中 KD 效果有限，且 t6 分支深层特征包含丰富疾病模式
    - v4 在 MTL 框架内部，让 t6 产生 deep feature context
    - 将该 context 注入 t1-t5 功能评估任务

    创建日期: 2026-04-27
    """

    def __init__(
        self,
        task_specs: Dict[str, TaskSpec],
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        device: str = "cpu",
        semantic_adj: Optional[List[List[float]]] = None,
        config: Optional[Dict] = None  # 接收完整配置（替换 t6_deep_context_config）
    ):
        super().__init__()
        self.task_specs = task_specs
        self.num_channels = num_channels
        self.D_time = D_time
        self.T_mid = T_mid
        self.device = device

        # =====================================================
        # [v4 Config-Driven] 配置解析
        # =====================================================
        if config is None:
            config = {}

        # 提取各级配置块
        hcgc_config = config.get('hcgc_v4', {})
        alpha_config = config.get('alpha', {})
        beta_config = config.get('beta', {})
        gate_config = hcgc_config.get('gate', {})
        gate_routing_config = hcgc_config.get('gate_routing', {})
        beta_experts_config = hcgc_config.get('beta', {}).get('residual_experts', {})
        t6_deep_context_config = hcgc_config.get('t6_deep_context', {})

        # =====================================================
        # [Ablation] 消融配置解析 (v4 Framework-level Ablation)
        # =====================================================
        ablation_config = hcgc_config.get('ablation', {})
        self.ablation_single_shared_alpha = ablation_config.get('single_shared_alpha', False)
        self.ablation_beta_shared_only = ablation_config.get('beta_shared_only', False)
        self.ablation_alpha_no_prior = ablation_config.get('alpha_no_prior', False)
        self.ablation_no_t3_private = ablation_config.get('no_t3_private', False)
        self.ablation_no_group245 = ablation_config.get('no_group245', False)
        self.ablation_no_beta_gates = ablation_config.get('no_beta_gates', False)
        self.ablation_alpha_no_pmgt = ablation_config.get('alpha_no_pmgt', False)
        self.ablation_alpha_no_multiscale_residual = ablation_config.get('alpha_no_multiscale_residual', False)
        self.ablation_teacher_no_prior = ablation_config.get('teacher_no_prior', False)

        # 消融综合日志
        ablation_active_flags = [
            self.ablation_single_shared_alpha,
            self.ablation_beta_shared_only,
            self.ablation_alpha_no_prior,
            self.ablation_no_t3_private,
            self.ablation_no_group245,
            self.ablation_no_beta_gates,
            self.ablation_alpha_no_pmgt,
            self.ablation_alpha_no_multiscale_residual,
            self.ablation_teacher_no_prior
        ]
        has_active_ablation = any(ablation_active_flags)

        if has_active_ablation:
            print("\n" + "="*60)
            print("v4 Framework-Level Ablation Active")
            print("="*60)
            if self.ablation_single_shared_alpha:
                print("[Ablation E2] single_shared_alpha: Beta disabled, t1-t5 all use Alpha trunk")
            if self.ablation_beta_shared_only:
                print("[Ablation E3] beta_shared_only: Beta only uses shared expert")
            if self.ablation_alpha_no_prior:
                print("[Ablation E4] alpha_no_prior: Alpha PMGT prior mask disabled")
            if self.ablation_no_t3_private:
                print("[Ablation S1] no_t3_private: t3_private expert removed")
            if self.ablation_no_group245:
                print("[Ablation S2] no_group245: group_245 expert removed")
            if self.ablation_no_beta_gates:
                print("[Ablation S3] no_beta_gates: Beta Gates disabled, fixed mixing weights")
            if self.ablation_alpha_no_pmgt:
                print("[Ablation S4] alpha_no_pmgt: PMGT replaced with FlattenMLP")
            if self.ablation_alpha_no_multiscale_residual:
                print("[Ablation S5] alpha_no_multiscale_residual: Alpha single-scale, no residual")
            if self.ablation_teacher_no_prior:
                print("[Ablation S6] teacher_no_prior: Teacher also has no prior mask")
            print("="*60 + "\n")

        # [修复 - 2026-05-07] 综合判断 t6_auxiliary_mode.enabled 和 t6_deep_context.enabled
        # t6_auxiliary_mode 是全局开关，t6_deep_context 是模块开关
        # 只有两者都为 true 时，才真正启用 t6_deep_context 功能（Gate创建56维）
        t6_auxiliary_mode_config = config.get('t6_auxiliary_mode', {})
        t6_auxiliary_mode_enabled = t6_auxiliary_mode_config.get('enabled', False)

        # [兼容性检查] 如果顶层配置为空，从 hcgc_v4 读取
        if not t6_auxiliary_mode_config:
            t6_auxiliary_mode_config = hcgc_config.get('t6_auxiliary_mode', {})
            t6_auxiliary_mode_enabled = t6_auxiliary_mode_config.get('enabled', False)
            if t6_auxiliary_mode_config:
                print(f"[Model] t6_auxiliary_mode config read from hcgc_v4.t6_auxiliary_mode (compatibility mode)")

        # 提取 Gate 配置值
        self.tau_start = gate_config.get('tau_start', 2.0)
        self.tau_mid = gate_config.get('tau_mid', 1.0)
        self.tau_end = gate_config.get('tau_end', 0.7)
        self.entropy_reg = gate_config.get('entropy_reg', 0.002)
        self.entropy_reg_epochs_ratio = gate_config.get('entropy_reg_epochs_ratio', 0.5)

        # 提取 Alpha/Beta 配置值
        alpha_dropout = alpha_config.get('dropout', 0.3)
        alpha_hidden_dim = alpha_config.get('hidden_dim', 48)
        pmgt_dropout = alpha_config.get('pmgt_dropout', 0.5)
        gamma_init = alpha_config.get('gamma_init', 1.0)

        beta_dropout = beta_config.get('dropout', 0.3)
        beta_hidden_dim = beta_config.get('hidden_dim', 48)
        expert_dropout = hcgc_config.get('beta', {}).get('dropout', 0.1)

        # =====================================================
        # [v4 Config-Driven] Alpha/Beta 任务分组
        # =====================================================
        self.alpha_tasks = [k for k, spec in task_specs.items() if spec.branch == "alpha"]
        self.beta_tasks = [k for k, spec in task_specs.items() if spec.branch == "beta"]

        # Semantic adjacency for PMGT
        if semantic_adj is not None:
            self.semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32).to(device)
        else:
            self.semantic_adj = None

        # T6 deep context configuration（从 hcgc_config 读取）
        if not t6_deep_context_config:
            t6_deep_context_config = {
                "enabled": True,
                "encoder": {"input_dim": 48, "hidden_dim": 32, "output_dim": 16, "dropout": 0.2},
                "bridge": {"c6_deep_dim": 16, "dyn_feat_dim": 48, "alpha_init": -2.2}
            }

        # [修复 - 2026-05-06] 综合判断: t6_auxiliary_mode 是全局开关，t6_deep_context 是模块开关
        # 只有两者都为 true 时，才真正启用 t6_deep_context 功能（Gate创建56维）
        # Baseline模式 (t6_auxiliary_mode.enabled=false): 强制 t6_deep_context_enabled=false (Gate创建40维)
        t6_deep_context_module_enabled = t6_deep_context_config.get("enabled", True)
        self.t6_deep_context_enabled = t6_auxiliary_mode_enabled and t6_deep_context_module_enabled
        self.apply_to_beta_gate = t6_deep_context_config.get("apply_to_beta_gate", True)
        self.apply_to_t1_head = t6_deep_context_config.get("apply_to_t1_head", True)

        print("\n" + "="*60)
        print("Protected Dual-Engine HCGC/PLE v4 Architecture")
        print("="*60)
        print(f"[v4] t6_auxiliary_mode.enabled: {t6_auxiliary_mode_enabled}")
        print(f"[v4] t6_deep_context.module_enabled: {t6_deep_context_module_enabled}")
        print("[v4] T6 Deep Feature Context Injection")
        print(f"[v4] t6_deep_context_enabled (综合): {self.t6_deep_context_enabled}")
        print(f"[v4] apply_to_beta_gate: {self.apply_to_beta_gate}")
        print(f"[v4] apply_to_t1_head: {self.apply_to_t1_head}")
        if self.t6_deep_context_enabled and self.apply_to_beta_gate:
            print("[v4] Beta Gate context_dim: 40 -> 56 (+ c6_deep[16])")
        else:
            print("[v4] Beta Gate context_dim: 40 (Baseline模式)")
        print("[v4] checkpoint 指标: weighted_macro_f1_t1_to_t5")
        print("="*60 + "\n")

        # =====================================================
        # Alpha/Beta Trunk 编码器 (Config-Driven + Ablation)
        # =====================================================
        # S5 ablation: Alpha trunk 使用单尺度非残差配置
        if self.ablation_alpha_no_multiscale_residual:
            alpha_encoder_config = TemporalEncoderConfig(
                type="cnn",
                T_mid=T_mid,
                use_multiscale=False,     # S5: 禁用多尺度
                use_residual=False,       # S5: 禁用残差
                block1_kernel=7,
                block2_kernel=5,
                use_masked_conv=False
            )
            self.alpha_trunk = AlphaDynamicEncoder(
                num_channels=num_channels,
                D_time=D_time,
                T_mid=T_mid,
                dropout=alpha_dropout,
                config=alpha_encoder_config
            )
            print("[Ablation S5] Alpha trunk: single-scale, no residual")
        else:
            self.alpha_trunk = AlphaDynamicEncoder(
                num_channels=num_channels,
                D_time=D_time,
                T_mid=T_mid,
                dropout=alpha_dropout
            )

        # E2 ablation: Beta trunk 完全禁用
        if self.ablation_single_shared_alpha:
            self.beta_trunk = None
            print("[Ablation E2] Beta trunk disabled (single_shared_alpha mode)")
        else:
            self.beta_trunk = BetaDynamicEncoder(
                num_channels=num_channels,
                D_time=D_time,
                T_mid=T_mid,
                dropout=beta_dropout
            )

        print("[v4] Alpha trunk: AlphaDynamicEncoder (参数量: {:,})".format(
            sum(p.numel() for p in self.alpha_trunk.parameters()) if self.alpha_trunk is not None else 0
        ))
        if self.beta_trunk is not None:
            print("[v4] Beta trunk: BetaDynamicEncoder (参数量: {:,})".format(
                sum(p.numel() for p in self.beta_trunk.parameters())
            ))

        # =====================================================
        # v4: Alpha无残差专家 (继承 v3 设计)
        # =====================================================
        self.alpha_residual_experts = None
        self.alpha_gate_context = None
        self.alpha_gates = None
        self.alpha_expert_names = None

        print("[v4] Alpha无残差专家 (CGC已删除, 继承 v3)")
        print(f"[v4] Alpha任务: {self.alpha_tasks}")

        # =====================================================
        # [v4 Config-Driven + Ablation] Beta分支残差专家
        # =====================================================
        # E2 ablation: Beta 完全禁用
        if self.ablation_single_shared_alpha:
            self.beta_residual_experts = None
            self.beta_expert_names_from_config = []
            print("[Ablation E2] Beta residual experts disabled")
        # E3 ablation: 仅 shared expert
        elif self.ablation_beta_shared_only:
            self.beta_expert_names_from_config = ["shared"]
            self.beta_residual_experts = nn.ModuleDict({
                "shared": ResidualExpert(
                    num_channels=num_channels,
                    D_time=D_time,
                    capacity="medium",
                    expert_name="E_shared",
                    dropout=expert_dropout
                )
            })
            print("[Ablation E3] Beta only shared expert (group_245 and t3_private removed)")
        # S1/S2 ablation: 移除特定专家
        elif self.ablation_no_t3_private or self.ablation_no_group245:
            # 从配置读取基础专家列表
            base_experts = list(beta_experts_config.keys()) if beta_experts_config else ["shared", "group_245", "t3_private"]
            # 应用消融移除
            if self.ablation_no_t3_private and "t3_private" in base_experts:
                base_experts.remove("t3_private")
                print("[Ablation S1] t3_private expert removed")
            if self.ablation_no_group245 and "group_245" in base_experts:
                base_experts.remove("group_245")
                print("[Ablation S2] group_245 expert removed")
            self.beta_expert_names_from_config = base_experts
            self.beta_residual_experts = nn.ModuleDict({
                expert_name: ResidualExpert(
                    num_channels=num_channels,
                    D_time=D_time,
                    capacity=beta_experts_config.get(expert_name, {}).get('capacity', 'medium'),
                    expert_name=f"E_{expert_name}",
                    dropout=expert_dropout
                )
                for expert_name in self.beta_expert_names_from_config
            })
            # [修复 - 2026-05-20] S1/S2 消融时初始化 beta_expert_names 路由映射
            # 必须从 gate_routing_config.beta 读取，而非使用 beta_expert_names_from_config
            # 因为 S1 移除 t3_private 后，t3 的路由应变为 ["shared"]，而非 ["shared", "group_245"]
            self.beta_expert_names = {}
            gate_routing_beta = gate_routing_config.get('beta', {})
            for task_key in self.beta_tasks:
                if task_key in gate_routing_beta:
                    self.beta_expert_names[task_key] = gate_routing_beta[task_key]
                else:
                    # 默认路由（向后兼容）
                    if task_key == "t3":
                        self.beta_expert_names[task_key] = ["shared", "t3_private"]
                    else:
                        self.beta_expert_names[task_key] = ["shared", "group_245"]
            print(f"[Ablation S1/S2] Beta expert routing from config: {self.beta_expert_names}")
        else:
            # 默认: 从配置读取专家名称和容量
            self.beta_expert_names_from_config = list(beta_experts_config.keys()) if beta_experts_config else ["shared", "group_245", "t3_private"]
            self.beta_residual_experts = nn.ModuleDict({
                expert_name: ResidualExpert(
                    num_channels=num_channels,
                    D_time=D_time,
                    capacity=beta_experts_config.get(expert_name, {}).get('capacity', 'medium'),
                    expert_name=f"E_{expert_name}",
                    dropout=expert_dropout
                )
                for expert_name in self.beta_expert_names_from_config
            })

        if self.beta_residual_experts is not None:
            print(f"[v4] Beta 残差专家 ({len(self.beta_expert_names_from_config)}个): {self.beta_expert_names_from_config}")

        # =====================================================
        # v4: Gate Context 编码器 (+ Ablation E2)
        # =====================================================
        if self.ablation_single_shared_alpha:
            self.beta_gate_context = None
            print("[Ablation E2] Beta gate context disabled")
        else:
            self.beta_gate_context = BetaGateContextEncoder(
                input_channels=num_channels,
                input_dim=D_time,
                output_dim=32
            )

        # 静态特征 MLP
        self.shared_static_mlp = SharedGateStaticMLP(
            input_dim=5,
            hidden_dim=16,
            output_dim=8
        )

        # =====================================================
        # v4: T6 Deep Feature Context Module (新增)
        # =====================================================
        if self.t6_deep_context_enabled:
            from modules.t6_context import T6DeepFeatureContextModule

            self.t6_deep_context_module = T6DeepFeatureContextModule(
                encoder_config=t6_deep_context_config.get("encoder", {
                    "input_dim": 48, "hidden_dim": 32, "output_dim": 16, "dropout": 0.2
                }),
                bridge_config=t6_deep_context_config.get("bridge", {
                    "c6_deep_dim": 16, "dyn_feat_dim": 48, "alpha_init": -2.2
                })
            )

            params = self.t6_deep_context_module.get_num_parameters()
            print("[v4] T6DeepFeatureContextModule (参数量: {:,})".format(params["total"]))
            print(f"[v4]   - encoder: {params['encoder']:,}")
            print(f"[v4]   - bridge: {params['bridge']:,}")
        else:
            self.t6_deep_context_module = None

        # =====================================================
        # [v4 Config-Driven + Ablation] Beta Gates
        # =====================================================
        beta_gate_context_dim = 56 if (self.t6_deep_context_enabled and self.apply_to_beta_gate) else 40

        # 从 gate_routing 配置读取路由映射
        gate_routing_beta = gate_routing_config.get('beta', {})

        # S3 ablation: 固定混合权重替代 Gate
        if self.ablation_no_beta_gates:
            self.beta_gates = None
            # [修复 - 2026-05-20] S3 消融时初始化 beta_expert_names (用于 forward Step 8)
            self.beta_expert_names = {}
            gate_routing_beta = gate_routing_config.get('beta', {})
            for task_key in self.beta_tasks:
                if task_key in gate_routing_beta:
                    self.beta_expert_names[task_key] = gate_routing_beta[task_key]
                else:
                    # 默认路由（向后兼容）
                    if task_key == "t3":
                        self.beta_expert_names[task_key] = ["shared", "t3_private"]
                    else:
                        self.beta_expert_names[task_key] = ["shared", "group_245"]
            # 从配置读取固定权重
            fixed_weights_config = ablation_config.get('fixed_mixing_weights', {})
            self.fixed_mixing_weights = {}
            for task_key in self.beta_tasks:
                # 使用 beta_expert_names[task_key] 作为专家列表（而非 beta_expert_names_from_config）
                expert_list = self.beta_expert_names.get(task_key, self.beta_expert_names_from_config)
                if task_key in fixed_weights_config:
                    # 转换为 tensor
                    weights_dict = fixed_weights_config[task_key]
                    weights_list = [weights_dict.get(exp_name, 0.5) for exp_name in expert_list]
                    total = sum(weights_list)
                    weights_normalized = [w / total for w in weights_list]  # 归一化
                    self.fixed_mixing_weights[task_key] = torch.tensor(weights_normalized, dtype=torch.float32)
                else:
                    # 默认均匀权重
                    n_experts = len(expert_list)
                    self.fixed_mixing_weights[task_key] = torch.tensor([1.0/n_experts] * n_experts, dtype=torch.float32)
            print("[Ablation S3] Beta Gates disabled, using fixed mixing weights")
            print(f"[Ablation S3] Fixed weights: {self.fixed_mixing_weights}")
            print(f"[Ablation S3] Beta expert routing: {self.beta_expert_names}")
        # E2 ablation: Beta Gates 完全禁用
        elif self.ablation_single_shared_alpha:
            self.beta_gates = None
            self.fixed_mixing_weights = None
            print("[Ablation E2] Beta Gates disabled (single_shared_alpha mode)")
        else:
            # 构建 beta_expert_names（配置优先，fallback 为默认值）
            self.beta_expert_names = {}
            for task_key in self.beta_tasks:
                if task_key in gate_routing_beta:
                    self.beta_expert_names[task_key] = gate_routing_beta[task_key]
                else:
                    # 默认路由（向后兼容）
                    if task_key == "t3":
                        self.beta_expert_names[task_key] = ["shared", "t3_private"]
                    else:
                        self.beta_expert_names[task_key] = ["shared", "group_245"]

            # 动态构建 Beta Gates（num_experts 从路由长度推断）
            self.beta_gates = nn.ModuleDict({
                task_key: TaskSpecificGate(
                    num_experts=len(self.beta_expert_names[task_key]),
                    context_dim=beta_gate_context_dim,
                    tau_init=self.tau_start,  # 从配置读取
                    task_name=task_key
                )
                for task_key in self.beta_tasks
            })
            print(f"[v4] Beta Gate context_dim={beta_gate_context_dim}, tau_start={self.tau_start}")
            print(f"[v4] Beta任务: {self.beta_tasks}")
            print(f"[v4] Beta专家路由: {self.beta_expert_names}")

        # =====================================================
        # [v4 Config-Driven + Ablation] Alpha 任务 PMGT
        # =====================================================
        # E2 ablation: 为 t2-t5 新增 PMGT heads (所有任务走 Alpha)
        if self.ablation_single_shared_alpha:
            # E2: alpha_tasks 包含 t1-t5 + t6
            self.alpha_interactors = nn.ModuleDict({
                task_key: PriorMaskedTaskHead(
                    num_nodes=num_channels,
                    hidden_dim=D_time,
                    out_dim=alpha_hidden_dim,
                    semantic_adj=self.semantic_adj,
                    dropout=pmgt_dropout,
                    gamma_init=gamma_init
                )
                for task_key in self.alpha_tasks  # 已包含 t1-t5 + t6
            })
            print(f"[Ablation E2] Alpha PMGT heads for all tasks: {self.alpha_tasks}")
        # S4 ablation: 使用 FlattenMLP 替代 PMGT
        elif self.ablation_alpha_no_pmgt:
            self.alpha_interactors = nn.ModuleDict({
                task_key: FlattenProjector(
                    num_nodes=num_channels,
                    hidden_dim=D_time,
                    out_dim=alpha_hidden_dim,
                    use_two_layer=True
                )
                for task_key in self.alpha_tasks
            })
            print("[Ablation S4] Alpha interactors: FlattenMLP (PMGT disabled)")
        # E4/S6 ablation: 禁用 prior mask (semantic_adj=None)
        elif self.ablation_alpha_no_prior or self.ablation_teacher_no_prior:
            semantic_adj_disabled = None  # 禁用 prior mask
            self.alpha_interactors = nn.ModuleDict({
                task_key: PriorMaskedTaskHead(
                    num_nodes=num_channels,
                    hidden_dim=D_time,
                    out_dim=alpha_hidden_dim,
                    semantic_adj=semantic_adj_disabled,  # None: 禁用 prior
                    dropout=pmgt_dropout,
                    gamma_init=gamma_init
                )
                for task_key in self.alpha_tasks
            })
            print("[Ablation E4/S6] Alpha PMGT prior mask disabled (semantic_adj=None)")
        else:
            # 默认: PMGT with prior mask
            self.alpha_interactors = nn.ModuleDict({
                task_key: PriorMaskedTaskHead(
                    num_nodes=num_channels,
                    hidden_dim=D_time,
                    out_dim=alpha_hidden_dim,
                    semantic_adj=self.semantic_adj,
                    dropout=pmgt_dropout,
                    gamma_init=gamma_init
                )
                for task_key in self.alpha_tasks
            })

        # =====================================================
        # [v4 Config-Driven + Ablation] Beta 任务 FlattenProjector
        # =====================================================
        # E2 ablation: Beta projectors 禁用
        if self.ablation_single_shared_alpha:
            self.beta_projectors = None
            print("[Ablation E2] Beta projectors disabled")
        else:
            self.beta_projectors = nn.ModuleDict({
                task_key: FlattenProjector(
                    num_nodes=num_channels,
                    hidden_dim=D_time,
                    out_dim=beta_hidden_dim,
                    use_two_layer=True
                )
                for task_key in self.beta_tasks
            })

        # =====================================================
        # 静态编码器和分类头 (继承 v3)
        # =====================================================
        self.static_encoders = nn.ModuleDict({
            task_key: TaskStaticEncoder(in_dim=5, out_dim=16, dropout=0.2)
            for task_key in task_specs.keys()
        })

        self.classifiers = nn.ModuleDict({
            task_key: TaskClassifier(
                dyn_dim=48,
                static_dim=16,
                hidden_dim=32,
                num_classes=spec.num_classes,
                is_binary=spec.is_binary,
                dropout=spec.dropout
            )
            for task_key, spec in task_specs.items()
        })

        # =====================================================
        # 同方差不确定性权重 (log_vars)
        # =====================================================
        self.log_vars = nn.ParameterDict({
            task_key: nn.Parameter(torch.zeros(1, device=device))
            for task_key in task_specs.keys()
        })

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_static: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        task_mask: Optional[Dict[str, torch.Tensor]] = None,
        tau_override: Optional[float] = None,
        return_aux: bool = False,
        return_gate_weights: bool = False,
        detach_t6_context: bool = False,
        skip_t6_injection: bool = False,
        skip_beta_branch: bool = False,  # [新增 - 2026-04-29] 跳过 Beta 分支计算
        return_intermediates: bool = False,  # [新增 - Interpretation] 返回中间变量
        context_mode: str = "normal"  # [新增 - Interpretation] context counterfactual mode: normal/zero/shuffle
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        前向传播 (v4 架构 + Ablation 支持)

        v4 核心流程:
        1. Trunk 编码: H_alpha, H_beta
        2. Alpha直接传递: 无专家聚合 (继承 v3)
        3. Beta残差专家: 3个专家 (继承 v3)
        4. T6 Deep Feature Context: dyn_feat_t6 -> c6_deep (新增)
        5. Beta Gate Context: 40 -> 56 (注入 c6_deep)
        6. Beta Gate 路由: 2专家聚合
        7. t1 上下文增强: dyn_feat_t1 + alpha * delta (新增)
        8. 任务路径: PMGT / FlattenProjector

        Args:
            x_dyn: [B, L_max, C] 动态序列
            x_static: [B, 5] 静态特征
            lengths: [B] 真实长度 (可选)
            task_mask: {"t1": [B], ...} 任务掩码 (可选)
            tau_override: 温度覆盖值
            return_aux: 是否返回辅助信息
            return_gate_weights: 是否返回门控权重
            detach_t6_context: 是否隔离 c6_deep 梯度 (Stage2/Phase1)
            skip_t6_injection: 是否跳过 t6 context 注入 (Stage1)
            skip_beta_branch: 是否跳过 Beta 分支计算 (Stage1/Stage2冻结时)

        Returns:
            outputs: {"t1": {"logits": ..., ...}, ..., "t6": {...}}
            aux: 辅助信息 (可选)
            gate_weights: 门控权重 (可选)
        """
        B = x_dyn.size(0)
        device = x_dyn.device

        if task_mask is None:
            task_mask = {task_key: torch.ones(B, device=device) for task_key in self.task_specs.keys()}

        # =============================================
        # Step 1: Trunk 编码 (+ Ablation E2)
        # =============================================
        H_alpha = self.alpha_trunk(x_dyn, lengths)  # [B, C, D_time]

        # E2 ablation: Beta trunk 禁用, 或 skip_beta_branch 标志
        if self.ablation_single_shared_alpha or skip_beta_branch:
            H_beta = None
        elif self.beta_trunk is not None:
            H_beta = self.beta_trunk(x_dyn, lengths)    # [B, C, D_time]
        else:
            H_beta = None

        # =============================================
        # Step 2: Alpha直接传递 (Config-Driven)
        # =============================================
        alpha_gated_outputs = {
            task_key: H_alpha for task_key in self.alpha_tasks
        }

        # =============================================
        # Step 3: Beta残差专家 (Config-Driven + Ablation)
        # =============================================
        beta_expert_outputs = {}
        if not (self.ablation_single_shared_alpha or skip_beta_branch) and H_beta is not None and self.beta_residual_experts is not None:
            for expert_name in self.beta_expert_names_from_config:
                expert = self.beta_residual_experts[expert_name]
                delta = expert(H_beta)
                beta_expert_outputs[expert_name] = H_beta + delta

        # =============================================
        # Step 4: Alpha 任务路径 (先计算 dyn_feat)
        # =============================================
        # 先计算 Alpha 任务的 dyn_feat，因为 t6_deep_context 需要用 dyn_feat_t6
        dyn_feat_alpha = {}
        for task_key in self.alpha_tasks:
            H_gated = alpha_gated_outputs[task_key]
            dyn_feat = self.alpha_interactors[task_key](H_gated)
            dyn_feat_alpha[task_key] = dyn_feat

        # =============================================
        # Step 5: T6 Deep Feature Context (v4 新增)
        # =============================================
        c6_deep = None
        if self.t6_deep_context_enabled and self.t6_deep_context_module is not None and not skip_t6_injection:
            dyn_feat_t6 = dyn_feat_alpha["t6"]
            c6_deep = self.t6_deep_context_module.encode(dyn_feat_t6)  # [B, 16]

            # [Interpretation Patch] Context counterfactual: normal / zero / shuffle
            if context_mode == "zero":
                c6_deep = torch.zeros_like(c6_deep)
            elif context_mode == "shuffle":
                idx = torch.randperm(c6_deep.size(0), device=c6_deep.device)
                c6_deep = c6_deep[idx]

        # =============================================
        # Step 6: Beta Gate Context (v4: 扩展到 56) (+ Ablation)
        # =============================================
        c_beta_full = None
        if not (self.ablation_single_shared_alpha or skip_beta_branch) and H_beta is not None and self.beta_gate_context is not None:
            c_beta = self.beta_gate_context(H_beta)  # [B, 32]
            c_static = self.shared_static_mlp(x_static)  # [B, 8]
            c_beta_full = torch.cat([c_beta, c_static], dim=1)  # [B, 40]

            # Injection A: Beta Gate context expansion
            if self.apply_to_beta_gate and c6_deep is not None and self.t6_deep_context_module is not None:
                c_beta_full = self.t6_deep_context_module.inject_to_beta_gate(
                    c_beta_full, c6_deep, detach_t6_context=detach_t6_context
                )  # [B, 56]

        # =============================================
        # Step 7: Beta Gate 权重 (+ Ablation S3)
        # =============================================
        beta_gate_weights = {}
        if not (self.ablation_single_shared_alpha or skip_beta_branch) and H_beta is not None:
            # S3 ablation: 使用固定混合权重
            if self.ablation_no_beta_gates and self.fixed_mixing_weights is not None:
                for task_key in self.beta_tasks:
                    weights = self.fixed_mixing_weights[task_key].to(device)
                    # 扩展为 batch 维度: [n_experts] -> [B, n_experts]
                    beta_gate_weights[task_key] = weights.unsqueeze(0).expand(B, -1)
            elif self.beta_gates is not None:
                for task_key in self.beta_tasks:
                    beta_gate_weights[task_key] = self.beta_gates[task_key](
                        c_beta_full, tau_override=tau_override
                    )

        # =============================================
        # Step 8: Beta专家聚合 (+ Ablation)
        # =============================================
        beta_gated_outputs = {}
        if not (self.ablation_single_shared_alpha or skip_beta_branch) and H_beta is not None and beta_expert_outputs:
            for task_key in self.beta_tasks:
                expert_names = self.beta_expert_names.get(task_key, self.beta_expert_names_from_config)
                weights = beta_gate_weights.get(task_key)

                if weights is not None:
                    H_gated = torch.zeros_like(H_beta)
                    for i, expert_name in enumerate(expert_names):
                        if i < weights.size(1) and expert_name in beta_expert_outputs:
                            H_gated = H_gated + weights[:, i:i+1, None] * beta_expert_outputs[expert_name]
                    beta_gated_outputs[task_key] = H_gated

        # =============================================
        # Step 9: 任务路径处理 (+ Ablation E2)
        # =============================================
        outputs = {}

        # [新增 2026-06-10] 保存 t1 中间表征（用于 interpretation）
        t1_base_feat = None  # 注入前的 dyn_feat_t1
        t1_guided_feat = None  # 注入后的 dyn_feat_t1
        t1_context_delta = None  # t6-to-t1 bridge 注入的增量

        # Alpha任务处理 (动态迭代)
        # E2 ablation: alpha_tasks 包含 t1-t5 + t6, 所有任务走 Alpha
        for task_key in self.alpha_tasks:
            dyn_feat = dyn_feat_alpha[task_key]

            # t1 上下文增强（仅对 t1 应用）
            if task_key == "t1" and self.apply_to_t1_head and c6_deep is not None and self.t6_deep_context_module is not None:
                # [新增] 保存 base feature（注入前）
                t1_base_feat = dyn_feat.clone() if return_intermediates else None

                # 调用 t6 injection
                dyn_feat = self.t6_deep_context_module.inject_to_t1_head(
                    dyn_feat, c6_deep, detach_t6_context=detach_t6_context
                )

                # [新增] 保存 guided feature（注入后）和 delta
                if return_intermediates:
                    t1_guided_feat = dyn_feat.clone()
                    # 计算 delta: delta_t1 = adapter(c6_deep), alpha = sigmoid(alpha_param)
                    # dyn_feat_guided = dyn_feat_base + alpha * delta
                    # 因此 delta_total = dyn_feat_guided - dyn_feat_base
                    t1_context_delta = t1_guided_feat - t1_base_feat  # [B, 48]

            static_feat = self.static_encoders[task_key](x_static)
            logits = self.classifiers[task_key](dyn_feat, static_feat)

            output_entry = {
                "logits": logits,
                "dyn_feat": dyn_feat,
                "static_feat": static_feat,
                "fused_feat": torch.cat([dyn_feat, static_feat], dim=1)
            }

            # t6 输出包含 c6_deep
            if task_key == "t6" and c6_deep is not None:
                output_entry["c6_deep"] = c6_deep

            outputs[task_key] = output_entry

        # Beta任务处理 (动态迭代)
        # E2 ablation: Beta tasks 已加入 alpha_tasks, skip
        if not self.ablation_single_shared_alpha and not skip_beta_branch and beta_gated_outputs and self.beta_projectors is not None:
            for task_key in self.beta_tasks:
                if task_key not in outputs:  # E2 已在 Alpha 处理
                    H_gated = beta_gated_outputs[task_key]

                    dyn_feat = self.beta_projectors[task_key](H_gated)
                    static_feat = self.static_encoders[task_key](x_static)
                    logits = self.classifiers[task_key](dyn_feat, static_feat)

                    outputs[task_key] = {
                        "logits": logits,
                        "dyn_feat": dyn_feat,
                        "static_feat": static_feat,
                        "fused_feat": torch.cat([dyn_feat, static_feat], dim=1)
                    }

        # 辅助信息
        if return_aux:
            outputs["aux"] = {
                "H_alpha": H_alpha,
                "H_beta": H_beta,
                "beta_expert_outputs": beta_expert_outputs,
                "alpha_gated_outputs": alpha_gated_outputs,
                "beta_gated_outputs": beta_gated_outputs,
                "c6_deep": c6_deep,
                "c_beta_full_dim": c_beta_full.size(1) if c_beta_full is not None else 0,
            }

        # 门控权重
        if return_gate_weights:
            outputs["gate_weights"] = beta_gate_weights

        # [Interpretation Patch] 返回中间变量
        if return_intermediates:
            inter = {}

            # t1/t6 PMGT attention scores and weights
            try:
                inter["t1_pmgt_attn"] = self.alpha_interactors["t1"].pmgt.last_attn_weights  # [B, H, N, N]
                inter["t1_pmgt_attn_scores"] = self.alpha_interactors["t1"].pmgt.last_attn_scores  # [B, H, N, N]
            except Exception:
                inter["t1_pmgt_attn"] = None
                inter["t1_pmgt_attn_scores"] = None

            try:
                inter["t6_pmgt_attn"] = self.alpha_interactors["t6"].pmgt.last_attn_weights
                inter["t6_pmgt_attn_scores"] = self.alpha_interactors["t6"].pmgt.last_attn_scores
            except Exception:
                inter["t6_pmgt_attn"] = None
                inter["t6_pmgt_attn_scores"] = None

            # c6_deep and dyn_feat_t6
            inter["c6_deep"] = c6_deep  # [B, 16]
            try:
                inter["dyn_feat_t6"] = outputs["t6"].get("dyn_feat", None)  # [B, 48]
            except Exception:
                inter["dyn_feat_t6"] = None

            # [新增 2026-06-10] t1 中间表征（用于方案A散点图可视化）
            # 注意：t1 不属于 Beta 分支，因此没有 FlattenProjector
            # t1_projector_feat 实际上是 t6-guided 后的 dyn_feat，命名是为了与 t2-t5 绘图脚本兼容
            inter["t1_base_feat"] = t1_base_feat  # [B, 48] 注入前的 t1 PMGT 输出
            inter["t1_projector_feat"] = t1_guided_feat  # [B, 48] 注入后的 t1 guided feature（推荐用于散点图）
            inter["t1_guided_feat"] = t1_guided_feat  # [B, 48] 别名，语义更明确
            inter["t1_context_delta"] = t1_context_delta  # [B, 48] t6-to-t1 bridge 注入的增量

            # t1 fused feature
            try:
                inter["t1_fused_feat"] = outputs["t1"].get("fused_feat", None)  # [B, 64]
            except Exception:
                inter["t1_fused_feat"] = None

            # beta_gate_weights (t2-t5)
            inter["beta_gate_weights"] = beta_gate_weights if beta_gate_weights else {}

            outputs["intermediates"] = inter

        return outputs

    def freeze_alpha_modules(self):
        """冻结Alpha分支所有模块 (Config-Driven)"""
        for param in self.alpha_trunk.parameters():
            param.requires_grad = False

        for interactor in self.alpha_interactors.values():
            for param in interactor.parameters():
                param.requires_grad = False

        for task_key in self.alpha_tasks:
            for param in self.static_encoders[task_key].parameters():
                param.requires_grad = False
            for param in self.classifiers[task_key].parameters():
                param.requires_grad = False
            self.log_vars[task_key].requires_grad = False

    def freeze_beta_modules(self):
        """冻结Beta分支所有模块 (Config-Driven + Ablation E2 支持)"""
        # Beta trunk (Ablation E2: None)
        if self.beta_trunk is not None:
            for param in self.beta_trunk.parameters():
                param.requires_grad = False

        # Beta residual experts (Ablation E2: None)
        if self.beta_residual_experts is not None and len(self.beta_residual_experts) > 0:
            for expert_name in self.beta_expert_names_from_config:
                if expert_name in self.beta_residual_experts:
                    for param in self.beta_residual_experts[expert_name].parameters():
                        param.requires_grad = False

        # Beta gates (Ablation E2/S3: None)
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            for gate in self.beta_gates.values():
                for param in gate.parameters():
                    param.requires_grad = False

        # Beta gate context (Ablation E2: None)
        if self.beta_gate_context is not None:
            for param in self.beta_gate_context.parameters():
                param.requires_grad = False

        # Beta projectors (Ablation E2: None)
        if self.beta_projectors is not None and len(self.beta_projectors) > 0:
            for projector in self.beta_projectors.values():
                for param in projector.parameters():
                    param.requires_grad = False

        # Beta task static encoders and classifiers (即使 E2 也保留，因为 alpha 使用)
        for task_key in self.beta_tasks:
            for param in self.static_encoders[task_key].parameters():
                param.requires_grad = False
            for param in self.classifiers[task_key].parameters():
                param.requires_grad = False
            if task_key in self.log_vars:
                self.log_vars[task_key].requires_grad = False

    def freeze_t6_context_modules(self):
        """冻结 T6 Deep Feature Context 模块 (v4 新增)"""
        if self.t6_deep_context_module is not None:
            for param in self.t6_deep_context_module.parameters():
                param.requires_grad = False
            print("[v4] T6DeepFeatureContextModule frozen")

    def unfreeze_all(self):
        """解冻所有模块"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_alpha_branch(self):
        """冻结Alpha分支 (别名方法)"""
        self.freeze_alpha_modules()

    def freeze_beta_branch(self):
        """冻结Beta分支 (别名方法)"""
        self.freeze_beta_modules()

    def freeze_alpha_trunk(self):
        """仅冻结 Alpha Trunk (Stage3 Phase2)"""
        for param in self.alpha_trunk.parameters():
            param.requires_grad = False

    def freeze_alpha_interactors(self):
        """冻结 Alpha Interactors (PMGT heads for t1, t6)"""
        for interactor in self.alpha_interactors.values():
            for param in interactor.parameters():
                param.requires_grad = False

    def freeze_log_vars_t1_t6(self):
        """冻结 t1, t6 的 log_vars (Stage2)"""
        self.log_vars["t1"].requires_grad = False
        self.log_vars["t6"].requires_grad = False

    def freeze_log_vars_beta(self):
        """冻结 t2-t5 的 log_vars"""
        for task_key in ["t2", "t3", "t4", "t5"]:
            self.log_vars[task_key].requires_grad = False

    def freeze_log_vars_t6(self):
        """冻结 t6 的 log_var (v4: freeze_t6_log_var=True)"""
        self.log_vars["t6"].requires_grad = False

    def get_num_parameters(self) -> Dict[str, int]:
        """获取各模块参数量 (v4 + Ablation E2 支持)"""
        counts = {}

        # Alpha trunk
        counts["alpha_trunk"] = sum(p.numel() for p in self.alpha_trunk.parameters())

        # v4: Alpha无残差专家
        counts["alpha_residual_experts"] = 0

        # Beta trunk (Ablation E2: None)
        if self.beta_trunk is not None:
            counts["beta_trunk"] = sum(p.numel() for p in self.beta_trunk.parameters())
        else:
            counts["beta_trunk"] = 0

        # Beta 残差专家 (Ablation E2: None)
        if self.beta_residual_experts is not None and len(self.beta_residual_experts) > 0:
            counts["beta_residual_experts"] = sum(
                sum(p.numel() for p in expert.parameters())
                for expert in self.beta_residual_experts.values()
            )
        else:
            counts["beta_residual_experts"] = 0

        # v4: Alpha无gate
        counts["alpha_gates"] = 0
        counts["alpha_gate_context"] = 0

        # Beta gates (Ablation E2/S3: None)
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            counts["beta_gates"] = sum(
                sum(p.numel() for p in gate.parameters())
                for gate in self.beta_gates.values()
            )
        else:
            counts["beta_gates"] = 0

        # Gate context (Ablation E2: None)
        if self.beta_gate_context is not None:
            counts["beta_gate_context"] = sum(p.numel() for p in self.beta_gate_context.parameters())
        else:
            counts["beta_gate_context"] = 0
        counts["shared_static_mlp"] = sum(p.numel() for p in self.shared_static_mlp.parameters())

        # 交互头
        counts["alpha_interactors"] = sum(
            sum(p.numel() for p in interactor.parameters())
            for interactor in self.alpha_interactors.values()
        )
        # Beta projectors (Ablation E2: None)
        if self.beta_projectors is not None and len(self.beta_projectors) > 0:
            counts["beta_projectors"] = sum(
                sum(p.numel() for p in projector.parameters())
                for projector in self.beta_projectors.values()
            )
        else:
            counts["beta_projectors"] = 0

        # v4: T6 Deep Feature Context Module
        if self.t6_deep_context_module is not None:
            params = self.t6_deep_context_module.get_num_parameters()
            counts["t6_deep_context_encoder"] = params["encoder"]
            counts["t6_deep_feature_bridge"] = params["bridge"]
            counts["t6_context_total"] = params["total"]
        else:
            counts["t6_deep_context_encoder"] = 0
            counts["t6_deep_feature_bridge"] = 0
            counts["t6_context_total"] = 0

        # 静态编码器和分类头
        counts["static_encoders"] = sum(
            sum(p.numel() for p in encoder.parameters())
            for encoder in self.static_encoders.values()
        )
        counts["classifiers"] = sum(
            sum(p.numel() for p in classifier.parameters())
            for classifier in self.classifiers.values()
        )

        counts["log_vars"] = sum(p.numel() for p in self.log_vars.parameters())
        counts["total"] = sum(counts.values())

        return counts

    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        返回按模块分组的参数字典 (v4 扩展版).

        Returns:
            {
                "alpha_trunk": [...],
                "alpha_interactors": [...],
                "beta_trunk": [...],
                "beta_residual_experts": [...],
                "beta_gates": [...],
                "beta_gate_context": [...],
                "beta_projectors": [...],
                "t6_deep_context_encoder": [...],    # v4 新增
                "t6_deep_feature_bridge": [...],     # v4 新增
                "t6_context_module": [...],     # v4 新增 (整体)
                "static_encoders": [...],
                "classifiers": [...],
                "log_vars": [...],
                "shared_static_mlp": [...],
                # 任务专属分组...
            }
        """
        groups = {}

        # Alpha trunk
        groups["alpha_trunk"] = list(self.alpha_trunk.parameters())

        # Alpha interactors
        groups["alpha_interactors"] = []
        for interactor in self.alpha_interactors.values():
            groups["alpha_interactors"].extend(list(interactor.parameters()))

        # Beta trunk (Ablation E2: None)
        if self.beta_trunk is not None:
            groups["beta_trunk"] = list(self.beta_trunk.parameters())
        else:
            groups["beta_trunk"] = []

        # Beta residual experts (Ablation E2: None)
        groups["beta_residual_experts"] = []
        if self.beta_residual_experts is not None and len(self.beta_residual_experts) > 0:
            for expert in self.beta_residual_experts.values():
                groups["beta_residual_experts"].extend(list(expert.parameters()))
            # 任务专属 Expert
            for expert_name, expert in self.beta_residual_experts.items():
                groups[f"beta_{expert_name}_expert"] = list(expert.parameters())

        # Beta gates (Ablation E2/S3: None)
        groups["beta_gates"] = []
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            for gate in self.beta_gates.values():
                groups["beta_gates"].extend(list(gate.parameters()))
            # 任务专属 Gate
            for task_name, gate in self.beta_gates.items():
                groups[f"{task_name}_gate"] = list(gate.parameters())

        # Beta gate context (Ablation E2: None)
        if self.beta_gate_context is not None:
            groups["beta_gate_context"] = list(self.beta_gate_context.parameters())
        else:
            groups["beta_gate_context"] = []

        # Beta projectors (Ablation E2: None)
        groups["beta_projectors"] = []
        if self.beta_projectors is not None and len(self.beta_projectors) > 0:
            for projector in self.beta_projectors.values():
                groups["beta_projectors"].extend(list(projector.parameters()))
            # 任务专属 Projector
            for task_name, projector in self.beta_projectors.items():
                groups[f"{task_name}_projector"] = list(projector.parameters())

        # v4: T6 Deep Feature Context Module
        if self.t6_deep_context_module is not None:
            groups["t6_deep_context_encoder"] = list(self.t6_deep_context_module.encoder.parameters())
            groups["t6_deep_feature_bridge"] = list(self.t6_deep_context_module.bridge.parameters())
            groups["t6_context_module"] = list(self.t6_deep_context_module.parameters())

        # Static encoders
        groups["static_encoders"] = []
        for encoder in self.static_encoders.values():
            groups["static_encoders"].extend(list(encoder.parameters()))

        # Classifiers
        groups["classifiers"] = []
        for classifier in self.classifiers.values():
            groups["classifiers"].extend(list(classifier.parameters()))

        # Log vars
        groups["log_vars"] = list(self.log_vars.parameters())

        # Shared static MLP
        groups["shared_static_mlp"] = list(self.shared_static_mlp.parameters())

        return groups

    def freeze_specific_modules(self, module_names: List[str]):
        """精细化冻结指定模块 (继承 v3)"""
        param_groups = self.get_parameter_groups()

        frozen_count = 0
        for module_name in module_names:
            if module_name in param_groups:
                params = param_groups[module_name]
                for param in params:
                    param.requires_grad = False
                    frozen_count += 1
                print(f"[freeze_specific_modules] 冻结 {module_name}: {len(params)} 个参数")
            else:
                print(f"[freeze_specific_modules] 警告: 未找到模块 {module_name}")

        print(f"[freeze_specific_modules] 共冻结 {frozen_count} 个参数")

    def get_freeze_status(self) -> Dict[str, Dict[str, Any]]:
        """
        返回各模块冻结状态统计，用于日志记录。

        Returns:
            {
                "alpha_trunk": {"total": N, "frozen": M, "ratio": M/N},
                ...
            }
        """
        param_groups = self.get_parameter_groups()
        status = {}

        for module_name, params in param_groups.items():
            total = len(params)
            frozen = sum(1 for p in params if not p.requires_grad)
            status[module_name] = {
                "total": total,
                "frozen": frozen,
                "ratio": frozen / total if total > 0 else 0.0
            }

        return status

    def get_alpha_t1_gate_value(self) -> float:
        """获取当前 t1 Alpha Gate 值 (v4 新增)"""
        if self.t6_deep_context_module is not None:
            return self.t6_deep_context_module.get_alpha_t1_value()
        return 0.0

    def set_gate_temperature(self, tau: float):
        """设置所有 Beta Gate 的温度 (继承 v3 + Ablation E2 支持)"""
        if self.beta_gates is not None and len(self.beta_gates) > 0:
            for gate in self.beta_gates.values():
                gate.set_tau(tau)
            print(f"[set_gate_temperature] 所有 Beta Gate 温度已设置为 {tau}")
        else:
            print(f"[set_gate_temperature] beta_gates 为空，跳过温度设置 (Ablation E2)")

    def __repr__(self) -> str:
        counts = self.get_num_parameters()
        beta_gate_context_dim = 56 if self.t6_deep_context_enabled else 40
        return (
            f"ProtectedDualEngineMTL_v4 (T6-guided Context Injection)\n"
            f"  num_channels={self.num_channels}\n"
            f"  D_time={self.D_time}\n"
            f"  t6_deep_context.enabled={self.t6_deep_context_enabled}\n"
            f"  beta_gate_context_dim={beta_gate_context_dim}\n"
            f"  alpha_trunk -> PMGT (无CGC)\n"
            f"  beta_trunk + 3 residual experts\n"
            f"  beta_gates: 全部2维\n"
            f"  t6_context: {counts['t6_context_total']:,} params\n"
            f"  total_params={counts['total']:,}\n"
            f")"
        )


# =============================================================================
# Initialization Functions for Protected Dual-Engine MTL v2
# =============================================================================

def init_trunks_from_baseline(
    model: ProtectedDualEngineMTL,
    baseline_checkpoint_path: str,
    device: str = "cpu"
):
    """
    Initialize trunk encoders from baseline checkpoint (v2).

    v2 Strategy:
    - alpha_trunk: copy baseline encoder (Alpha branch features)
    - beta_trunk: copy baseline encoder (Beta branch features)
    - 残差专家保持随机初始化（开始时接近identity）

    Args:
        model: ProtectedDualEngineMTL instance
        baseline_checkpoint_path: Path to baseline HDSTGCN checkpoint
        device: Device to load checkpoint on

    Returns:
        None (modifies model in place)

    Note:
        Baseline checkpoint structure:
        - temporal_encoder.* (CNN temporal encoder, D_time=16)
    """
    import os

    if not os.path.exists(baseline_checkpoint_path):
        print(f"[Warning] Baseline checkpoint not found: {baseline_checkpoint_path}")
        print("[Warning] Trunks will use random initialization.")
        return

    # Load baseline state_dict (添加 weights_only=False 解决 PyTorch 2.6 安全限制)
    state_dict = torch.load(baseline_checkpoint_path, map_location=device, weights_only=False)

    # Extract temporal encoder weights
    temporal_keys = [k for k in state_dict.keys() if k.startswith("temporal_encoder.")]
    temporal_state = {k.replace("temporal_encoder.", "encoder."): state_dict[k] for k in temporal_keys}

    print(f"\n[Init v2] Loading baseline temporal encoder from: {baseline_checkpoint_path}")
    print(f"[Init v2] Found {len(temporal_keys)} temporal encoder keys")

    # Initialize Alpha trunk
    print("[Init v2] Initializing alpha_trunk...")
    try:
        model.alpha_trunk.load_state_dict(temporal_state, strict=False)
        print("[Init v2] alpha_trunk: loaded from baseline")
    except Exception as e:
        print(f"[Init v2] alpha_trunk: partial load ({e})")

    # Initialize Beta trunk
    print("[Init v2] Initializing beta_trunk...")
    try:
        model.beta_trunk.load_state_dict(temporal_state, strict=False)
        print("[Init v2] beta_trunk: loaded from baseline")
    except Exception as e:
        print(f"[Init v2] beta_trunk: partial load ({e})")

    print("[Init v2] Residual experts will use random initialization (start near identity)")


def init_residual_experts_near_identity(
    model: ProtectedDualEngineMTL,
    identity_strength: float = 0.1,
    device: str = "cpu"
):
    """
    Initialize residual experts to produce small deltas initially (v2).

    This makes the experts start near identity mapping:
    H_out = H_trunk + delta, where delta ≈ 0 initially.

    Strategy:
    - Initialize linear layers with small weights (std=identity_strength)
    - Initialize biases to zero

    Args:
        model: ProtectedDualEngineMTL instance
        identity_strength: Weight initialization standard deviation
        device: Device

    Returns:
        None
    """
    import torch.nn as nn

    print(f"\n[Init v2] Initializing residual experts near identity (std={identity_strength})")

    def init_small_weights(m):
        if isinstance(m, nn.Linear):
            # Small random weights, zero bias
            nn.init.normal_(m.weight, mean=0.0, std=identity_strength)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            # Keep LayerNorm at default (identity-like)
            pass

    # Initialize Alpha residual experts
    for expert_name, expert in model.alpha_residual_experts.items():
        expert.apply(init_small_weights)
        print(f"[Init v2] alpha_residual_experts['{expert_name}']: small weights init")

    # Initialize Beta residual experts
    for expert_name, expert in model.beta_residual_experts.items():
        expert.apply(init_small_weights)
        print(f"[Init v2] beta_residual_experts['{expert_name}']: small weights init")




def init_protected_dual_engine_mtl_v4(
    model: ProtectedDualEngineMTL_v4,
    config: Dict[str, Any],
    task_specs: Dict[str, TaskSpec],
    device: str = "cpu"
):
    """
    Complete initialization pipeline for ProtectedDualEngineMTL_v4.

    v4 支持两种初始化策略:
    1. **from_scratch_structured**: 完全随机初始化（不继承checkpoint）
    2. **checkpoint_inheritance**: 从单任务 checkpoint 继承 trunk 权重

    v4 Initialization sequence (checkpoint_inheritance):
       - alpha_trunk: 从 alpha_trunk.source 指定的 checkpoint 继承
       - beta_trunk: 从 beta_trunk.source 指定的 checkpoint 继承（支持不同于alpha的来源）
       - beta_residual_experts: near-identity, identity_strength=0.1
       - beta_gates: uniform 初始化
       - alpha_interactors: Kaiming 初始化
       - t6_deep_context_encoder: Xavier

    Args:
        model: ProtectedDualEngineMTL_v4 instance
        config: Configuration dictionary with initialization paths
        task_specs: Task specification dictionary
        device: Device to load checkpoints on

    Returns:
        None (modifies model in place)

    Config structure expected:
        hcgc_v4:
          initialization:
            policy: "checkpoint_inheritance" | "from_scratch_structured"
            identity_strength: 0.1
            alpha_trunk:
              source: "t1_checkpoint"
            beta_trunk:
              source: "t1_checkpoint"    # 可选 "t3_checkpoint"
        checkpoints:
          t1_checkpoint: "path/to/t1_best.pth"
          t3_checkpoint: "path/to/t3_best.pth"  # 可选
    """
    import os

    init_config = config.get("initialization", {})
    hcgc_config = config.get("hcgc_v4", {})
    init_config = hcgc_config.get("initialization", init_config)

    # v4 核心配置
    policy = init_config.get("policy", "from_scratch_structured")
    identity_strength = init_config.get("identity_strength", 0.1)

    # Alpha/Beta trunk 来源配置
    alpha_trunk_cfg = init_config.get("alpha_trunk", {})
    beta_trunk_cfg = init_config.get("beta_trunk", {})
    alpha_source = alpha_trunk_cfg.get("source", "t1_checkpoint")
    beta_source = beta_trunk_cfg.get("source", "t1_checkpoint")

    # 从 checkpoints 块收集可用路径
    checkpoints_config = config.get("checkpoints", {})
    checkpoint_paths = {}
    for key in ["t6_checkpoint", "t1_checkpoint", "t3_checkpoint"]:
        path = checkpoints_config.get(key, None)
        if path:
            checkpoint_paths[key] = path

    print("\n" + "=" * 60)
    print("Protected Dual-Engine MTL v4 Initialization Pipeline")
    print("=" * 60)
    print(f"[Init v4] Policy: {policy}")
    print(f"[Init v4] alpha_trunk source: {alpha_source}")
    print(f"[Init v4] beta_trunk  source: {beta_source}")
    print(f"[Init v4] Available checkpoints: {list(checkpoint_paths.keys())}")

    # ========== Step 1: Trunk 初始化（支持 Alpha/Beta 分别指定来源） ==========
    if policy == "checkpoint_inheritance":
        # --- Alpha trunk 初始化 ---
        alpha_ckpt_path = checkpoint_paths.get(alpha_source, None) if alpha_source else None
        if alpha_ckpt_path and os.path.exists(alpha_ckpt_path):
            print(f"\n[Init v4] Loading alpha_trunk from {alpha_source}: {alpha_ckpt_path}")
            _init_single_trunk_from_checkpoint(
                model.alpha_trunk, alpha_ckpt_path, trunk_name="alpha_trunk", device=device
            )
        else:
            if alpha_source:
                print(f"[Init v4 Warning] {alpha_source} not found: {alpha_ckpt_path}")
            print("[Init v4] alpha_trunk: using default Kaiming initialization")
            _init_trunk_kaiming(model.alpha_trunk)

        # --- Beta trunk 初始化 ---
        # [Ablation E2] Beta trunk 可能为 None
        if model.beta_trunk is not None:
            beta_ckpt_path = checkpoint_paths.get(beta_source, None) if beta_source else None
            if beta_ckpt_path and os.path.exists(beta_ckpt_path):
                print(f"\n[Init v4] Loading beta_trunk from {beta_source}: {beta_ckpt_path}")
                _init_single_trunk_from_checkpoint(
                    model.beta_trunk, beta_ckpt_path, trunk_name="beta_trunk", device=device
                )
            else:
                if beta_source:
                    print(f"[Init v4 Warning] {beta_source} not found: {beta_ckpt_path}")
                print("[Init v4] beta_trunk: using default Kaiming initialization")
                _init_trunk_kaiming(model.beta_trunk)
        else:
            print("[Init v4] beta_trunk: None (ablation E2 mode, skipping initialization)")

        # --- 头部模块初始化（跟随 alpha 来源） ---
        heads_ckpt_path = alpha_ckpt_path
        if heads_ckpt_path and os.path.exists(heads_ckpt_path):
            print(f"\n[Init v4] Loading heads from alpha source: {heads_ckpt_path}")
            init_heads_from_baseline(model, heads_ckpt_path, task_specs, device=device)
        else:
            print("[Init v4] Heads using default initialization")
            _init_alpha_interactors_kaiming(model)
    else:
        print("\n[Init v4] Trunks using default Kaiming initialization (from_scratch)")
        _init_alpha_interactors_kaiming(model)

    # ========== Step 2: Beta residual experts near-identity 初始化 ==========
    # [Ablation] 使用动态专家列表，支持 S1/S2/E2/E3 消融
    if model.beta_residual_experts is not None and len(model.beta_residual_experts) > 0:
        print(f"\n[Init v4] Initializing Beta residual experts near identity (std={identity_strength})")
        # 使用 beta_expert_names_from_config 获取动态专家列表
        expert_names = model.beta_expert_names_from_config if hasattr(model, 'beta_expert_names_from_config') else list(model.beta_residual_experts.keys())
        for expert_name in expert_names:
            expert = model.beta_residual_experts[expert_name]
            for module in expert.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=identity_strength)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            print(f"[Init v4] beta_residual_experts['{expert_name}']: near-identity init")
    else:
        print("\n[Init v4] Beta residual experts: None (ablation mode)")

    # ========== Step 3: Beta gates uniform 初始化 ==========
    # [Ablation] 检查 gates 是否存在 (S3/E2 禁用)
    if model.beta_gates is not None and len(model.beta_gates) > 0:
        print("\n[Init v4] Initializing Beta gates with uniform initialization")
        for gate_name, gate in model.beta_gates.items():
            for module in gate.modules():
                if isinstance(module, nn.Linear):
                    nn.init.uniform_(module.weight, a=-0.1, b=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            print(f"[Init v4] beta_gates['{gate_name}']: uniform init")
    else:
        print("\n[Init v4] Beta gates: None (ablation mode)")

    # ========== Step 4: T6 context module (Xavier) ==========
    if model.t6_deep_context_module is not None:
        print("\n[Init v4] T6DeepFeatureContextModule ready (Xavier init in constructor)")
        alpha_t1_value = model.get_alpha_t1_gate_value()
        print(f"[Init v4] t6_to_t1_alpha gate: sigmoid(alpha)={alpha_t1_value:.3f} (weak injection)")

    print("\n" + "=" * 60)
    print(f"v4 Initialization Complete (policy={policy})")
    print("=" * 60)


def _init_single_trunk_from_checkpoint(
    trunk_module,
    checkpoint_path: str,
    trunk_name: str = "trunk",
    device: str = "cpu"
):
    """从单任务 checkpoint 提取 temporal_encoder 权重，加载到指定的 trunk 模块。"""
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    temporal_keys = [k for k in state_dict.keys() if k.startswith("temporal_encoder.")]
    temporal_state = {k.replace("temporal_encoder.", "encoder."): state_dict[k] for k in temporal_keys}

    print(f"[Init v4] Found {len(temporal_keys)} temporal encoder keys in checkpoint")

    try:
        missing, unexpected = trunk_module.load_state_dict(temporal_state, strict=False)
        print(f"[Init v4] {trunk_name}: loaded from checkpoint")
        if missing:
            print(f"[Init v4]   missing keys: {len(missing)}")
        if unexpected:
            print(f"[Init v4]   unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"[Init v4] {trunk_name}: partial load ({e})")


def _init_trunk_kaiming(trunk_module):
    """对 trunk 模块使用 Kaiming 初始化。"""
    for module in trunk_module.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _init_alpha_interactors_kaiming(model):
    """对 alpha_interactors 使用 Kaiming 初始化。"""
    print("\n[Init v4] Initializing Alpha interactors (PMGT heads)")
    for task_name, interactor in model.alpha_interactors.items():
        for module in interactor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        print(f"[Init v4] alpha_interactors['{task_name}']: Kaiming init")


def _init_trunks_from_t1_checkpoint(
    model: ProtectedDualEngineMTL_v4,
    t1_checkpoint_path: str,
    device: str = "cpu"
):
    """
    从 t1 单任务 checkpoint 初始化 alpha/beta trunk 权重。

    Args:
        model: ProtectedDualEngineMTL_v4 instance
        t1_checkpoint_path: t1 checkpoint 路径
        device: 设备

    Returns:
        None (modifies model in place)
    """
    state_dict = torch.load(t1_checkpoint_path, map_location=device, weights_only=False)

    # 提取 temporal_encoder 权重
    temporal_keys = [k for k in state_dict.keys() if k.startswith("temporal_encoder.")]
    temporal_state = {k.replace("temporal_encoder.", "encoder."): state_dict[k] for k in temporal_keys}

    print(f"[Init v4] Found {len(temporal_keys)} temporal encoder keys in t1 checkpoint")

    # 初始化 alpha_trunk
    try:
        missing, unexpected = model.alpha_trunk.load_state_dict(temporal_state, strict=False)
        print(f"[Init v4] alpha_trunk: loaded from t1 checkpoint")
        if missing:
            print(f"[Init v4]   missing keys: {len(missing)}")
        if unexpected:
            print(f"[Init v4]   unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"[Init v4] alpha_trunk: partial load ({e})")

    # 初始化 beta_trunk
    try:
        missing, unexpected = model.beta_trunk.load_state_dict(temporal_state, strict=False)
        print(f"[Init v4] beta_trunk: loaded from t1 checkpoint")
        if missing:
            print(f"[Init v4]   missing keys: {len(missing)}")
        if unexpected:
            print(f"[Init v4]   unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"[Init v4] beta_trunk: partial load ({e})")


def init_B_t3_from_t3_best(
    model: ProtectedDualEngineMTL,
    t3_checkpoint_path: str,
    device: str = "cpu"
):
    """
    Initialize B_t3 residual expert from t3_best checkpoint (v2).

    v2 Strategy:
    - beta_trunk is already initialized from baseline
    - E_t3 (t3_private residual expert) can be enhanced if we have t3_best
    - But since E_t3 is a residual MLP, we can't directly copy temporal encoder
    - Instead, we can boost its initialization strength

    Args:
        model: ProtectedDualEngineMTL instance
        t3_checkpoint_path: Path to best t3 checkpoint
        device: Device to load checkpoint on

    Returns:
        None (modifies model in place)

    Note:
        In v2, E_t3 is a residual expert (MLP), not a temporal encoder.
        We can't directly copy temporal encoder weights.
        This function serves as a placeholder for potential future enhancement.
    """
    import os

    if not os.path.exists(t3_checkpoint_path):
        print(f"[Warning v2] T3 checkpoint not found: {t3_checkpoint_path}")
        print("[Warning v2] E_t3 will use standard small-weight initialization.")
        return

    print(f"\n[Init v2] T3 checkpoint found: {t3_checkpoint_path}")
    print("[Init v2] Note: E_t3 is a residual MLP in v2, not a temporal encoder")
    print("[Init v2] Using enhanced initialization for E_t3")

    # Enhanced initialization: slightly larger weights for better expressiveness
    # This compensates for not being able to directly copy temporal encoder weights
    import torch.nn as nn

    def init_enhanced_weights(m):
        if isinstance(m, nn.Linear):
            # Larger weights for better capacity
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            pass

    # [Ablation] 检查 t3_private 是否存在 (S1 消融可能移除)
    if "t3_private" not in model.beta_residual_experts:
        print("[Init v2] Warning: t3_private expert not found (ablation mode), skipping enhanced init")
        return

    model.beta_residual_experts["t3_private"].apply(init_enhanced_weights)
    print("[Init v2] beta_residual_experts['t3_private']: Xavier init (enhanced)")


def init_heads_from_baseline(
    model: ProtectedDualEngineMTL,
    baseline_checkpoint_path: str,
    task_specs: Dict[str, TaskSpec],
    device: str = "cpu"
):
    """
    Initialize PMGT, FlattenProjector, StaticEncoder, Classifier from baseline.

    This function initializes the task-specific heads and shared components
    from a baseline HDSTGCN checkpoint.

    Args:
        model: ProtectedDualEngineMTL instance
        baseline_checkpoint_path: Path to baseline HDSTGCN checkpoint
        task_specs: Task specification dictionary
        device: Device to load checkpoint on

    Returns:
        None (modifies model in place)

    Note:
        Baseline checkpoint structure:
        - spatial_graph.prior_masked_transformer.* (PMGT components)
        - static_encoder.* (Static encoder)
        - classifier.* (Classifier)

        For ProtectedDualEngineMTL:
        - alpha_interactors[task_key] contains PriorMaskedTaskHead
        - static_encoders[task_key] contains TaskStaticEncoder
        - classifiers[task_key] contains TaskClassifier
    """
    import os

    if not os.path.exists(baseline_checkpoint_path):
        print(f"[Warning] Baseline checkpoint not found: {baseline_checkpoint_path}")
        print("[Warning] Heads will use random initialization.")
        return

    state_dict = torch.load(baseline_checkpoint_path, map_location=device, weights_only=False)

    print(f"\n[Init] Loading baseline heads from: {baseline_checkpoint_path}")

    # =========================================
    # Initialize alpha_interactors (PMGT-like)
    # =========================================
    # Extract PMGT weights from baseline
    pmgt_keys = [k for k in state_dict.keys() if k.startswith("spatial_graph.prior_masked_transformer.")]

    if pmgt_keys:
        pmgt_state = {k.replace("spatial_graph.prior_masked_transformer.", ""): state_dict[k]
                       for k in pmgt_keys}

        for task_key in ["t1", "t6"]:
            # PriorMaskedTaskHead contains self.pmgt (PriorMaskedGlobalTransformer)
            # S4 ablation: FlattenProjector has no pmgt attribute, skip PMGT init
            interactor = model.alpha_interactors[task_key]
            if hasattr(interactor, 'pmgt'):
                transformer = interactor.pmgt  # Correct attribute name
                transformer.load_state_dict(pmgt_state, strict=False)
                print(f"[Init] alpha_interactors['{task_key}'] PMGT initialized from baseline")
            else:
                print(f"[Init] alpha_interactors['{task_key}'] uses FlattenProjector (S4 ablation), PMGT init skipped")

    # =========================================
    # Initialize static_encoders
    # =========================================
    static_keys = [k for k in state_dict.keys() if k.startswith("static_encoder.")]

    if static_keys:
        # Map baseline keys to TaskStaticEncoder's internal structure
        # Baseline: static_encoder.0.weight -> TaskStaticEncoder: encoder.0.weight
        static_state = {k.replace("static_encoder.", "encoder."): state_dict[k] for k in static_keys}

        for task_key in task_specs.keys():
            # TaskStaticEncoder has self.encoder attribute
            encoder = model.static_encoders[task_key]
            encoder.load_state_dict(static_state, strict=False)

        print(f"[Init] static_encoders initialized from baseline for all {len(task_specs)} tasks")

    # =========================================
    # Initialize classifiers (partial)
    # =========================================
    # Note: Classifier structure differs between baseline and MTL
    # - Baseline: classifier.{0,1,4} (3-layer MLP for single task)
    # - MTL TaskClassifier: classifier.{0,1,2} (self.classifier Sequential)
    # We can copy the first two layers (feature extraction) but
    # the final layer (output dimension) differs by task.

    classifier_keys = [k for k in state_dict.keys() if k.startswith("classifier.")]
    if classifier_keys:
        # Map baseline keys to TaskClassifier's internal structure
        # Baseline: classifier.0.weight -> TaskClassifier: classifier.0.weight (same!)
        # Copy first two layers only (layer 0: Linear, layer 1: LayerNorm)
        classifier_state = {}
        for k in classifier_keys:
            if k.startswith("classifier.0.") or k.startswith("classifier.1."):
                # Map directly (same key structure)
                new_key = k  # Keep the same
                classifier_state[new_key] = state_dict[k]

        for task_key, spec in task_specs.items():
            classifier = model.classifiers[task_key]
            # TaskClassifier uses self.classifier attribute
            classifier.classifier.load_state_dict(classifier_state, strict=False)

        print(f"[Init] classifiers first two layers initialized from baseline")
        print(f"[Init] Note: Final layer (output) uses random init (task-specific dimension)")


def init_protected_dual_engine_mtl(
    model: ProtectedDualEngineMTL,
    config: Dict[str, Any],
    task_specs: Dict[str, TaskSpec],
    device: str = "cpu"
):
    """
    Complete initialization pipeline for ProtectedDualEngineMTL (v2).

    v2 Initialization sequence:
    1. init_trunks_from_baseline - Initialize trunk encoders from baseline
    2. init_residual_experts_near_identity - Initialize residual experts near identity
    3. init_B_t3_from_t3_best - Enhanced initialization for E_t3 (if available)
    4. init_heads_from_baseline - Initialize shared components

    Args:
        model: ProtectedDualEngineMTL instance
        config: Configuration dictionary with initialization paths
        task_specs: Task specification dictionary
        device: Device to load checkpoints on

    Returns:
        None (modifies model in place)

    Config structure expected:
        initialization:
          baseline_checkpoint: "path/to/baseline.pth"
          t3_best_checkpoint: "path/to/t3_best.pth" (optional)
    """
    init_config = config.get("initialization", {})
    hcgc_config = config.get("hcgc_v2", {})  # [重构] 使用 hcgc_v2 配置路径
    init_config = hcgc_config.get("initialization", init_config)

    baseline_path = init_config.get("baseline_checkpoint", None)
    t3_path = init_config.get("t3_best_checkpoint", None)

    print("\n" + "=" * 60)
    print("Protected Dual-Engine MTL v2 Initialization Pipeline")
    print("=" * 60)

    # Step 1: Initialize trunk encoders from baseline (v2)
    if baseline_path:
        init_trunks_from_baseline(model, baseline_path, device=device)
    else:
        print("[Warning v2] No baseline_checkpoint specified in config")
        print("[Warning v2] Trunks will use random initialization")

    # Step 2: Initialize residual experts near identity (v2)
    init_residual_experts_near_identity(model, identity_strength=0.1, device=device)

    # Step 3: Enhanced initialization for E_t3 (if T3 checkpoint available)
    if t3_path:
        init_B_t3_from_t3_best(model, t3_path, device=device)
    else:
        print("[Warning v2] No t3_best_checkpoint specified in config")
        print("[Warning v2] E_t3 will use standard small-weight initialization")

    # Step 4: Initialize shared heads
    if baseline_path:
        init_heads_from_baseline(model, baseline_path, task_specs, device=device)

    print("\n" + "=" * 60)
    print("v2 Initialization Complete")
    print("=" * 60)


# def init_protected_dual_engine_mtl_v3(
#     model: ProtectedDualEngineMTL_v3,
#     config: Dict[str, Any],
#     task_specs: Dict[str, TaskSpec],
#     device: str = "cpu"
# ):
#     """
#     Complete initialization pipeline for ProtectedDualEngineMTL_v3.

#     v3 Initialization sequence:
#     1. init_trunks_from_baseline - Initialize trunk encoders from baseline
#     2. init_residual_experts_near_identity_v3 - Initialize Beta residual experts (only 3)
#     3. init_B_t3_from_t3_best_v3 - Enhanced initialization for E_t3 (if available)
#     4. init_heads_from_baseline - Initialize shared components

#     Args:
#         model: ProtectedDualEngineMTL_v3 instance
#         config: Configuration dictionary with initialization paths
#         task_specs: Task specification dictionary
#         device: Device to load checkpoints on

#     Returns:
#         None (modifies model in place)

#     Config structure expected:
#         initialization:
#           baseline_checkpoint: "path/to/baseline.pth"
#           t3_best_checkpoint: "path/to/t3_best.pth" (optional)
#     """
#     init_config = config.get("initialization", {})
#     hcgc_config = config.get("hcgc_v3", {})
#     init_config = hcgc_config.get("initialization", init_config)

#     baseline_path = init_config.get("baseline_checkpoint", None)
#     t3_path = init_config.get("t3_best_checkpoint", None)

#     print("\n" + "=" * 60)
#     print("Protected Dual-Engine MTL v3 Initialization Pipeline")
#     print("=" * 60)

#     # Step 1: Initialize trunk encoders from baseline
#     if baseline_path:
#         init_trunks_from_baseline(model, baseline_path, device=device)
#     else:
#         print("[Warning v3] No baseline_checkpoint specified in config")
#         print("[Warning v3] Trunks will use random initialization")

#     # Step 2: Initialize Beta residual experts near identity (v3: only 3 experts)
#     for expert_name in ["shared", "group_245", "t3_private"]:
#         expert = model.beta_residual_experts[expert_name]
#         for module in expert.modules():
#             # 只对Linear层进行near-identity初始化 (Xavier需要>=2维)
#             if isinstance(module, nn.Linear):
#                 nn.init.normal_(module.weight, mean=0.0, std=identity_strength)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#         print(f"[Init v3] beta_residual_experts['{expert_name}']: near-identity init")

#     # Step 3: Enhanced initialization for E_t3 (if T3 checkpoint available)
#     if t3_path:
#         print(f"[Init v3] T3 checkpoint found: {t3_path}")
#         print("[Init v3] Using enhanced Xavier initialization for E_t3")
#         expert = model.beta_residual_experts["t3_private"]
#         for module in expert.modules():
#             # Xavier初始化只适用于Linear层
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#     else:
#         print("[Warning v3] No t3_best_checkpoint specified in config")

#     # Step 4: Initialize shared heads
#     if baseline_path:
#         init_heads_from_baseline(model, baseline_path, task_specs, device=device)

#     print("\n" + "=" * 60)
#     print("v3 Initialization Complete")
#     print("=" * 60)

# =============================================================================
# ProtectedDualEngineMTL 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HDSTGCNMTL 测试")
    print("=" * 80)

    # 构建 TaskSpec
    from task_specs import TaskSpec

    task_specs = {
        "t1": TaskSpec("t1", "运动心功能分级", 3, "alpha", "ce", False, 0.3, "运动心功能分级"),
        "t2": TaskSpec("t2", "运动耐量", 3, "beta", "ldam", False, 0.3, "运动耐量"),
        "t3": TaskSpec("t3", "标准心电运动负荷试验", 2, "beta", "bce", True, 0.3, "标准心电运动负荷试验"),
        "t4": TaskSpec("t4", "运动中换气肺功能", 2, "beta", "ldam", True, 0.3, "运动中换气肺功能"),
        "t5": TaskSpec("t5", "心率储备", 2, "beta", "ldam", True, 0.3, "心率储备"),
        "t6": TaskSpec("t6", "匹配的第一大类", 6, "alpha", "ce", False, 0.3, "匹配的第一大类", kd_teacher="dummy.pth")
    }

    # 创建 HDSTGCNMTL 模型
    model = HDSTGCNMTL(task_specs, num_channels=30, D_time=16)

    # 打印参数量
    counts = model.get_num_parameters()
    for key, count in counts.items():
        print(f"{key}: {count} params")

    # 测试前向传播
    B, L, C = 4, 200, 30
    x_dyn = torch.randn(B, L, C)
    x_static = torch.randn(B, 5)

    outputs = model(x_dyn, x_static, return_aux=True)

    print("\nHDSTGCNMTL 前向输出:")
    for task_key, task_output in outputs.items():
        if task_key == "aux":
            print(f"aux: alpha_nodes={task_output['alpha_nodes'].shape}, beta_nodes={task_output['beta_nodes'].shape}")
        else:
            print(f"{task_key}: logits={task_output['logits'].shape}")

    print("\nHDSTGCNMTL 测试通过！")

    # ============================================================
    # ProtectedDualEngineMTL 测试
    # ============================================================
    print("\n" + "=" * 80)
    print("ProtectedDualEngineMTL 测试")
    print("=" * 80)

    # 创建 ProtectedDualEngineMTL 模型
    pdmtl_model = ProtectedDualEngineMTL(task_specs, num_channels=30, D_time=16)

    # 打印模型结构
    print(f"\n模型结构:\n{pdmtl_model}")

    # 打印参数量
    counts = pdmtl_model.get_num_parameters()
    print("\n参数量统计:")
    for key, count in counts.items():
        print(f"  {key}: {count:,} params")

    # 测试前向传播
    outputs = pdmtl_model(x_dyn, x_static, return_aux=True, return_gate_weights=True)

    print("\nProtectedDualEngineMTL v2 前向输出:")
    for task_key, task_output in outputs.items():
        if task_key == "aux":
            print(f"aux:")
            print(f"  H_alpha: {task_output['H_alpha'].shape}")  # v2: trunk 输出
            print(f"  H_beta: {task_output['H_beta'].shape}")    # v2: trunk 输出
            print(f"  alpha_expert_outputs: {list(task_output['alpha_expert_outputs'].keys())}")
            print(f"  beta_expert_outputs: {list(task_output['beta_expert_outputs'].keys())}")
        elif task_key == "gate_weights":
            print(f"gate_weights:")
            for gw_key, gw in task_output.items():
                print(f"  {gw_key}: shape={gw.shape}, sum={gw.sum(dim=1).mean().item():.3f}")
        else:
            print(f"{task_key}: logits={task_output['logits'].shape}")

    # 测试冻结功能 (v2)
    print("\n测试冻结功能 (v2):")
    pdmtl_model.freeze_alpha_modules()
    alpha_trunk_grad_count = sum(1 for p in pdmtl_model.alpha_trunk.parameters() if p.requires_grad)
    alpha_expert_grad_count = sum(1 for p in pdmtl_model.alpha_residual_experts["shared"].parameters() if p.requires_grad)
    beta_trunk_grad_count = sum(1 for p in pdmtl_model.beta_trunk.parameters() if p.requires_grad)
    print(f"  freeze_alpha_modules: alpha_trunk requires_grad count = {alpha_trunk_grad_count} (expected 0)")
    print(f"  freeze_alpha_modules: alpha_residual_experts requires_grad count = {alpha_expert_grad_count} (expected 0)")
    print(f"  freeze_alpha_modules: beta_trunk requires_grad count = {beta_trunk_grad_count} (expected >0)")

    pdmtl_model.freeze_beta_modules()
    beta_trunk_grad_count = sum(1 for p in pdmtl_model.beta_trunk.parameters() if p.requires_grad)
    beta_expert_grad_count = sum(1 for p in pdmtl_model.beta_residual_experts["shared"].parameters() if p.requires_grad)
    print(f"  freeze_beta_modules: beta_trunk requires_grad count = {beta_trunk_grad_count} (expected 0)")
    print(f"  freeze_beta_modules: beta_residual_experts requires_grad count = {beta_expert_grad_count} (expected 0)")

    pdmtl_model.unfreeze_all()
    alpha_trunk_grad_count = sum(1 for p in pdmtl_model.alpha_trunk.parameters() if p.requires_grad)
    beta_trunk_grad_count = sum(1 for p in pdmtl_model.beta_trunk.parameters() if p.requires_grad)
    print(f"  unfreeze_all: alpha_trunk requires_grad count = {alpha_trunk_grad_count} (expected >0)")
    print(f"  unfreeze_all: beta_trunk requires_grad count = {beta_trunk_grad_count} (expected >0)")

    print("\nProtectedDualEngineMTL v2 测试通过！")

    # ============================================================
    # ProtectedDualEngineMTL_v3 测试
    # ============================================================
    print("\n" + "=" * 80)
    print("ProtectedDualEngineMTL_v3 测试")
    print("=" * 80)

    # 创建 ProtectedDualEngineMTL_v3 模型
    pdmtl_v3_model = ProtectedDualEngineMTL_v3(task_specs, num_channels=30, D_time=16)

    # 打印模型结构
    print(f"\n模型结构:\n{pdmtl_v3_model}")

    # 打印参数量
    counts_v3 = pdmtl_v3_model.get_num_parameters()
    print("\n参数量统计 (v3 vs v2):")
    for key, count in counts_v3.items():
        v2_count = counts.get(key, 0)
        diff = count - v2_count
        print(f"  {key}: {count:,} params (v2={v2_count:,}, diff={diff:,})")

    # 测试前向传播
    outputs_v3 = pdmtl_v3_model(x_dyn, x_static, return_aux=True, return_gate_weights=True)

    print("\nProtectedDualEngineMTL_v3 前向输出:")
    for task_key, task_output in outputs_v3.items():
        if task_key == "aux":
            print(f"aux:")
            print(f"  H_alpha: {task_output['H_alpha'].shape}")
            print(f"  H_beta: {task_output['H_beta'].shape}")
            print(f"  beta_expert_outputs: {list(task_output['beta_expert_outputs'].keys())}")
            print(f"  alpha_gated_outputs: {list(task_output['alpha_gated_outputs'].keys())} (direct trunk pass)")
        elif task_key == "gate_weights":
            print(f"gate_weights (v3: only Beta tasks):")
            for gw_key, gw in task_output.items():
                print(f"  {gw_key}: shape={gw.shape}, sum={gw.sum(dim=1).mean().item():.3f}")
        else:
            print(f"{task_key}: logits={task_output['logits'].shape}")

    # 测试冻结功能 (v3)
    print("\n测试冻结功能 (v3):")
    pdmtl_v3_model.freeze_alpha_modules()
    alpha_trunk_grad_count = sum(1 for p in pdmtl_v3_model.alpha_trunk.parameters() if p.requires_grad)
    # v3: Alpha无残差专家
    alpha_expert_grad_count = 0  # v3: None
    beta_trunk_grad_count = sum(1 for p in pdmtl_v3_model.beta_trunk.parameters() if p.requires_grad)
    print(f"  freeze_alpha_modules: alpha_trunk requires_grad count = {alpha_trunk_grad_count} (expected 0)")
    print(f"  freeze_alpha_modules: alpha_residual_experts = None (v3: no CGC)")
    print(f"  freeze_alpha_modules: beta_trunk requires_grad count = {beta_trunk_grad_count} (expected >0)")

    pdmtl_v3_model.freeze_beta_modules()
    beta_trunk_grad_count = sum(1 for p in pdmtl_v3_model.beta_trunk.parameters() if p.requires_grad)
    beta_expert_grad_count = sum(1 for p in pdmtl_v3_model.beta_residual_experts["shared"].parameters() if p.requires_grad)
    print(f"  freeze_beta_modules: beta_trunk requires_grad count = {beta_trunk_grad_count} (expected 0)")
    print(f"  freeze_beta_modules: beta_residual_experts requires_grad count = {beta_expert_grad_count} (expected 0)")

    pdmtl_v3_model.unfreeze_all()
    alpha_trunk_grad_count = sum(1 for p in pdmtl_v3_model.alpha_trunk.parameters() if p.requires_grad)
    beta_trunk_grad_count = sum(1 for p in pdmtl_v3_model.beta_trunk.parameters() if p.requires_grad)
    print(f"  unfreeze_all: alpha_trunk requires_grad count = {alpha_trunk_grad_count} (expected >0)")
    print(f"  unfreeze_all: beta_trunk requires_grad count = {beta_trunk_grad_count} (expected >0)")

    print("\nProtectedDualEngineMTL_v3 测试通过！")

    # v3 vs v2 参数量对比
    print("\n" + "=" * 80)
    print("v3 vs v2 参数量对比")
    print("=" * 80)
    total_v3 = counts_v3["total"]
    total_v2 = counts["total"]
    reduction = total_v2 - total_v3
    reduction_pct = reduction / total_v2 * 100
    print(f"  v2 total params: {total_v2:,}")
    print(f"  v3 total params: {total_v3:,}")
    print(f"  reduction: {reduction:,} ({reduction_pct:.1f}%)")
    print("=" * 80)