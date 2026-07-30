"""
Expert Temporal Encoders for Protected Dual-Engine MTL Architecture

This module defines expert-specific temporal encoders that reuse the existing
CNNTemporalEncodingBranch architecture with configurable capacity levels.

Architecture Overview:
    Each expert encoder independently processes CPET temporal sequences:
    Input: [B, L_max, num_channels] -> Output: [B, num_channels, D_time]

    The encoders are initialized from baseline checkpoints (trained on Task T1)
    and remain protected (frozen) during MTL training to preserve foundational
    feature extraction capabilities.

Capacity Levels:
    - light: ~114k params, D_time=16, single-scale conv (baseline)
    - medium: ~225k params, D_time=24, multiscale conv
    - strong: ~360k params, D_time=32, multiscale + residual

Author: MTL Architecture Implementation
Date: 2026-04-16
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Import existing CNN temporal encoder from model.py

from model import CNNTemporalEncodingBranch


# =============================================================================
# Expert Capacity Configuration
# =============================================================================

EXPERT_CAPACITY_CONFIG: Dict[str, Dict[str, Any]] = {
    # Expert encoder capacity configuration dictionary.
    #
    # Defines three capacity levels for expert temporal encoders:
    # - light: Minimal parameters (~114k), suitable for resource-constrained scenarios
    # - medium: Balanced capacity (~225k), multiscale feature extraction
    # - strong: Maximum capacity (~360k), multiscale + residual connections
    #
    # Each configuration maps to CNNTemporalEncodingBranch parameters:
    # - D_time: Temporal encoding dimension (output feature dimension per channel)
    # - use_multiscale: Enable multi-scale convolution kernels
    # - use_residual: Enable residual connections across temporal blocks
    #
    # Usage:
    #     config = EXPERT_CAPACITY_CONFIG["medium"]
    #     encoder = ExpertTemporalEncoder(num_channels=30, capacity="medium")
    "light": {
        "D_time": 16,
        "use_multiscale": False,
        "use_residual": False,
        # Estimated params: ~114k (baseline configuration)
    },
    "medium": {
        "D_time": 24,
        "use_multiscale": True,
        "use_residual": False,
        # Estimated params: ~225k (multiscale enabled)
    },
    "strong": {
        "D_time": 32,
        "use_multiscale": True,
        "use_residual": True,
        # Estimated params: ~360k (multiscale + residual)
    },
}


# =============================================================================
# Expert Temporal Encoder
# =============================================================================

class ExpertTemporalEncoder(nn.Module):
    """
    Expert Temporal Encoder - Wraps CNNTemporalEncodingBranch with capacity config.

    This encoder processes CPET temporal sequences for specific disease classification
    tasks. Each expert is initialized from a baseline checkpoint and remains protected
    (frozen) during MTL training.

    Architecture:
        Reuses CNNTemporalEncodingBranch:
        - Stage1: DepthwiseConv(1->16, k=7) + MaxPool(L->L/2)
        - Stage2: DepthwiseConv(16->32, k=5) + MaxPool(L/2->L/4)
        - Stage3: AdaptiveAvgPool(->T_mid=24)
        - Attention: Masked temporal attention aggregation
        - Projection: Linear(32, D_time) + LayerNorm + Dropout

    Args:
        num_channels: Number of input feature channels (default: 30 for nine_graph mode)
        capacity: Capacity level, one of ["light", "medium", "strong"]
        expert_name: Name identifier for the expert (for logging/debugging)
        T_mid: Intermediate temporal dimension (default: 24)
        dropout: Dropout rate (default: 0.3)

    Input:
        x: [B, L_max, num_channels] - Variable-length padded sequences
        lengths: [B] - Actual sequence lengths (optional)

    Output:
        H_nodes: [B, num_channels, D_time] - Node features for graph construction

    Example:
        >>> encoder = ExpertTemporalEncoder(num_channels=30, capacity="light")
        >>> x = torch.randn(16, 200, 30)  # Batch of 16, max length 200
        >>> lengths = torch.tensor([180, 195, 200, ...])  # Actual lengths
        >>> features = encoder(x, lengths)  # [16, 30, 16]
    """

    VALID_CAPACITIES = ["light", "medium", "strong"]

    def __init__(
        self,
        num_channels: int = 30,
        capacity: str = "light",
        expert_name: str = "unknown",
        T_mid: int = 24,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Validate capacity level
        if capacity not in self.VALID_CAPACITIES:
            raise ValueError(
                f"Invalid capacity '{capacity}'. Must be one of {self.VALID_CAPACITIES}"
            )

        # Store expert metadata
        self.num_channels = num_channels
        self.capacity = capacity
        self.expert_name = expert_name
        self.config = EXPERT_CAPACITY_CONFIG[capacity]

        # Extract capacity-specific parameters
        D_time = self.config["D_time"]
        use_multiscale = self.config["use_multiscale"]
        use_residual = self.config["use_residual"]

        # Create TemporalEncoderConfig-like object for CNNTemporalEncodingBranch
        # Note: CNNTemporalEncodingBranch expects a config object with specific attributes
        class TemporalEncoderConfig:
            pass

        temporal_config = TemporalEncoderConfig()
        temporal_config.use_multiscale = use_multiscale
        temporal_config.use_residual = use_residual
        temporal_config.multiscale_kernels = [3, 5, 7]  # Default kernel sizes
        temporal_config.block1_kernel = 7
        temporal_config.block2_kernel = 5
        temporal_config.use_masked_conv = True  # Enable masked convolution for variable length

        # Initialize the underlying CNN temporal encoder
        self.encoder = CNNTemporalEncodingBranch(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=dropout,
            config=temporal_config,
        )

        # Store output dimension for downstream modules
        self.output_dim = D_time

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the expert temporal encoder.

        Args:
            x: [B, L_max, num_channels] - Input temporal sequences
            lengths: [B] - Actual sequence lengths (optional, for masking)

        Returns:
            H_nodes: [B, num_channels, D_time] - Extracted node features
        """
        return self.encoder(x, lengths)

    def get_num_parameters(self) -> int:
        """
        Calculate total number of trainable parameters.

        Returns:
            Total parameter count for this expert encoder
        """
        return sum(p.numel() for p in self.parameters())

    def freeze(self) -> None:
        """
        Freeze all parameters for protected MTL training.

        Called after loading baseline checkpoint to prevent gradient updates.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters (for fine-tuning scenarios).
        """
        for param in self.parameters():
            param.requires_grad = True

    def load_baseline_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load weights from a baseline checkpoint file.

        Args:
            checkpoint_path: Path to .pth checkpoint file

        Note:
            This method should be called during MTL model initialization.
            After loading, call freeze() to protect the encoder.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(state_dict, strict=True)

    def __repr__(self) -> str:
        return (
            f"ExpertTemporalEncoder("
            f"expert_name='{self.expert_name}', "
            f"capacity='{self.capacity}', "
            f"num_channels={self.num_channels}, "
            f"output_dim={self.output_dim}, "
            f"params={self.get_num_parameters():,})"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_expert_config_summary() -> str:
    """
    Generate a summary string of all expert capacity configurations.

    Returns:
        Formatted string showing capacity levels and parameter estimates.
    """
    summary = "Expert Capacity Configuration Summary:\n"
    summary += "=" * 50 + "\n"

    for capacity, config in EXPERT_CAPACITY_CONFIG.items():
        summary += f"\n{capacity.upper()}:\n"
        summary += f"  D_time: {config['D_time']}\n"
        summary += f"  use_multiscale: {config['use_multiscale']}\n"
        summary += f"  use_residual: {config['use_residual']}\n"

        # Rough parameter estimation for CNNTemporalEncodingBranch
        # Base params: ~114k for light, scaling factor for D_time and multiscale
        base_params = 114000
        if config['use_multiscale']:
            base_params *= 2.0
        if config['use_residual']:
            base_params *= 1.6
        # Scale by D_time ratio
        base_params *= config['D_time'] / 16

        summary += f"  estimated_params: ~{int(base_params):,}\n"

    return summary


# =============================================================================
# Residual Expert Module (v2 Architecture)
# =============================================================================

class ResidualExpert(nn.Module):
    """
    Residual Expert Module - v2 Architecture

    在 v2 架构中，专家不再是完整的独立编码器，而是对 trunk 输出的残差变换。

    设计理念:
    - trunk (Alpha/Beta DynamicEncoder) 提取分支级基础表征
    - 专家提供不同的残差特化方向
    - gate 只在专家之间路由，不包括 trunk

    计算公式:
        A_shared = H_alpha + E_alpha_shared(H_alpha)
        A_t1     = H_alpha + E_t1(H_alpha)
        ...
        B_shared = H_beta  + E_beta_shared(H_beta)
        B_t3     = H_beta  + E_t3(H_beta)
        ...

    三种容量级别:
    - light: 轻量残差变换 (逐通道线性变换)
    - medium: 中等残差变换 (跨通道交互 + 线性变换)
    - strong: 强残差变换 (多层 MLP + 跨通道交互)

    Args:
        num_channels: 输入节点数 (默认 30)
        D_time: 时序编码维度 (默认 16)
        capacity: 容量级别 ("light", "medium", "strong")
        expert_name: 专家名称标识
        dropout: Dropout 率
    """

    VALID_CAPACITIES = ["light", "medium", "strong"]

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        capacity: str = "light",
        expert_name: str = "unknown",
        dropout: float = 0.1
    ):
        super().__init__()

        if capacity not in self.VALID_CAPACITIES:
            raise ValueError(f"Invalid capacity '{capacity}'. Must be one of {self.VALID_CAPACITIES}")

        self.num_channels = num_channels
        self.D_time = D_time
        self.capacity = capacity
        self.expert_name = expert_name

        in_dim = num_channels * D_time  # 480 (for default config)

        if capacity == "light":
            # 轻量残差: 逐通道线性变换
            # 参数量: D_time * D_time * num_channels = 16 * 16 * 30 = 7680
            # 或全局: in_dim * in_dim = 480 * 480 = 230400
            # 使用轻量设计: in_dim -> hidden -> in_dim
            hidden_dim = D_time * 4  # 64
            self.residual_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim),
                nn.Dropout(dropout)
            )

        elif capacity == "medium":
            # 中等残差: 跨通道交互
            # 参数量: ~231k
            hidden_dim = in_dim // 2  # 240
            self.residual_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim),
                nn.Dropout(dropout)
            )

        elif capacity == "strong":
            # 强残差: 多层 MLP + 跨通道交互
            # 参数量: ~360k
            hidden_dim = in_dim  # 480 (保持维度)
            self.residual_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, in_dim),
                nn.Dropout(dropout)
            )

    def forward(self, H_trunk: torch.Tensor) -> torch.Tensor:
        """
        计算残差输出

        Args:
            H_trunk: [B, num_channels, D_time] trunk 输出

        Returns:
            H_residual: [B, num_channels, D_time] 残差变换后的输出
            (注意: 返回的是残差 delta，不是 trunk + delta)
        """
        B = H_trunk.size(0)

        # 计算残差 delta
        delta = self.residual_net(H_trunk)  # [B, in_dim]

        # 恢复形状
        delta = delta.view(B, self.num_channels, self.D_time)  # [B, C, D]

        return delta

    def get_num_parameters(self) -> int:
        """获取参数量"""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"ResidualExpert("
            f"expert_name='{self.expert_name}', "
            f"capacity='{self.capacity}', "
            f"num_channels={self.num_channels}, "
            f"D_time={self.D_time}, "
            f"params={self.get_num_parameters():,})"
        )


if __name__ == "__main__":
    # Test expert encoder creation
    print(get_expert_config_summary())

    # Create and test each capacity level
    for capacity in ["light", "medium", "strong"]:
        encoder = ExpertTemporalEncoder(
            num_channels=30,
            capacity=capacity,
            expert_name=f"test_{capacity}"
        )
        print(f"\n{encoder}")

        # Test forward pass
        x = torch.randn(4, 100, 30)  # Batch=4, Length=100, Channels=30
        lengths = torch.tensor([80, 95, 100, 75])
        output = encoder(x, lengths)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

    # Test Residual Expert
    print("\n" + "="*50)
    print("Residual Expert Tests (v2 Architecture)")
    print("="*50)

    for capacity in ["light", "medium", "strong"]:
        expert = ResidualExpert(
            num_channels=30,
            D_time=16,
            capacity=capacity,
            expert_name=f"test_residual_{capacity}"
        )
        print(f"\n{expert}")

        # Test forward pass
        H_trunk = torch.randn(4, 30, 16)  # [B, C, D]
        delta = expert(H_trunk)
        output = H_trunk + delta  # 残差连接
        print(f"  Input (H_trunk) shape: {H_trunk.shape}")
        print(f"  Delta shape: {delta.shape}")
        print(f"  Output (H_trunk + delta) shape: {output.shape}")