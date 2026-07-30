"""
T6 Deep Feature Context Modules for Protected Dual-Engine MTL v4 Architecture.

This module provides the context injection mechanism that uses t6's deep features
to guide the prediction of t1-t5 functional assessment tasks.

Components:
- T6DeepFeatureContextEncoder: Encodes dyn_feat_t6 into disease context vector c6_deep
- T6DeepFeatureBridge: Injects c6_deep into Beta Gate context and t1 Alpha head

创建日期: 2026-04-27
版本: v4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class T6DeepFeatureContextEncoder(nn.Module):
    """
    T6 Deep Feature Context Encoder.

    Encodes the t6 branch's deep feature (dyn_feat_t6) into a compact
    disease context vector c6_deep for injection into t1-t5 tasks.

    Architecture:
        Input: dyn_feat_t6 [B, 48] (from alpha_interactors["t6"])
            ↓
        Linear(48, 32)
        LayerNorm(32)
        ReLU
        Dropout(0.2)
        Linear(32, 16)
        ReLU
            ↓
        Output: c6_deep [B, 16] (disease context vector)

    Args:
        input_dim (int): Input dimension from dyn_feat_t6. Default: 48
        hidden_dim (int): Hidden layer dimension. Default: 32
        output_dim (int): Output context dimension. Default: 16
        dropout (float): Dropout rate. Default: 0.2

    Parameter Count: **2160 params (~2.16k)**
        - Linear(48, 32): 48×32 + 32 = 1,568
        - LayerNorm(32): 32×2 = 64
        - Linear(32, 16): 32×16 + 16 = 528
        - **Total**: 1,568 + 64 + 528 = **2,160**
    """

    def __init__(
        self,
        input_dim: int = 48,
        hidden_dim: int = 32,
        output_dim: int = 16,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Two-layer MLP with LayerNorm
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU(inplace=True)

        # Initialize weights with Xavier for stable gradient flow
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, dyn_feat_t6: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            dyn_feat_t6: [B, 48] tensor from alpha_interactors["t6"]

        Returns:
            c6_deep: [B, 16] disease context vector
        """
        # First layer with LayerNorm
        h = self.layer1(dyn_feat_t6)  # [B, hidden_dim]
        h = self.layer_norm(h)
        h = self.relu1(h)
        h = self.dropout(h)

        # Second layer
        c6_deep = self.layer2(h)  # [B, output_dim]
        c6_deep = self.relu2(c6_deep)

        return c6_deep


class T6DeepFeatureBridge(nn.Module):
    """
    T6 Deep Feature Bridge for Context Injection.

    Provides two injection points for the disease context vector c6_deep:

    Injection A (Beta Gate Context):
        v3: c_full = concat(c_beta, c_static)  [B, 40]
        v4: c_full = concat(c_beta, c_static, c6_deep)  [B, 56]

    Injection B (t1 Alpha Head):
        delta_t1 = t6_to_t1_adapter(c6_deep)     # [B, 48]
        alpha_t1 = sigmoid(t6_to_t1_alpha)       # scalar, init -2.2 -> sigmoid ≈ 0.1
        dyn_feat_t1_guided = dyn_feat_t1 + alpha_t1 * delta_t1

    Gradient Isolation Strategy:
        - Stage1: skip injection (t6 not trained yet)
        - Stage2/Phase1: detach_t6_context=True (c6_deep detached)
        - Stage3 Phase2: detach_t6_context=False (full gradient flow)

    Args:
        c6_deep_dim (int): Input dimension from c6_deep. Default: 16
        dyn_feat_dim (int): dyn_feat dimension for t1. Default: 48
        alpha_init (float): Initial value for alpha gate parameter. Default: -2.2
            sigmoid(-2.2) ≈ 0.1, providing weak initial injection

    Parameter Count: **913 params (~0.91k)**
        - t6_to_t1_adapter: Linear(16, 48) = 16×48 + 48 = 816
        - LayerNorm(48): 48×2 = 96
        - t6_to_t1_alpha: nn.Parameter (1)
        - **Total**: 816 + 96 + 1 = **913**
    """

    def __init__(
        self,
        c6_deep_dim: int = 16,
        dyn_feat_dim: int = 48,
        alpha_init: float = -2.2
    ):
        super().__init__()

        self.c6_deep_dim = c6_deep_dim
        self.dyn_feat_dim = dyn_feat_dim

        # Adapter for t1 head injection
        self.t6_to_t1_adapter = nn.Sequential(
            nn.Linear(c6_deep_dim, dyn_feat_dim),
            nn.LayerNorm(dyn_feat_dim),
            nn.ReLU(inplace=True)
        )

        # Learnable alpha gate parameter (initialized to weak injection)
        self.t6_to_t1_alpha = nn.Parameter(torch.tensor(alpha_init))

        # Initialize adapter weights
        for module in self.t6_to_t1_adapter:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def inject_to_beta_gate(
        self,
        c_beta_full: torch.Tensor,
        c6_deep: torch.Tensor,
        detach_t6_context: bool = False
    ) -> torch.Tensor:
        """
        Injection A: Extend Beta Gate context from 40 to 56 dimensions.

        Note: detach_t6_context should be handled at the forward level (on dyn_feat_t6),
        not here. This method just concatenates.

        Args:
            c_beta_full: [B, 40] concat(c_beta, c_static)
            c6_deep: [B, 16] disease context vector
            detach_t6_context: Deprecated, handled at forward level

        Returns:
            c_full_expanded: [B, 56] expanded context vector
        """
        # detach is handled at T6DeepFeatureContextModule.forward level on dyn_feat_t6
        c_full_expanded = torch.cat([c_beta_full, c6_deep], dim=1)  # [B, 56]
        return c_full_expanded

    def inject_to_t1_head(
        self,
        dyn_feat_t1: torch.Tensor,
        c6_deep: torch.Tensor,
        detach_t6_context: bool = False
    ) -> torch.Tensor:
        """
        Injection B: Context enhancement for t1 Alpha head.

        Note: detach_t6_context should be handled at the forward level (on dyn_feat_t6),
        not here on c6_deep.

        Formula:
            delta_t1 = adapter(c6_deep)
            alpha_t1 = sigmoid(self.t6_to_t1_alpha)
            dyn_feat_t1_guided = dyn_feat_t1 + alpha_t1 * delta_t1

        Args:
            dyn_feat_t1: [B, 48] original t1 dynamic feature
            c6_deep: [B, 16] disease context vector
            detach_t6_context: Deprecated, handled at forward level

        Returns:
            dyn_feat_t1_guided: [B, 48] context-enhanced t1 feature
        """
        # detach is handled at T6DeepFeatureContextModule.forward level on dyn_feat_t6

        # Compute delta from disease context
        delta_t1 = self.t6_to_t1_adapter(c6_deep)  # [B, 48]

        # Compute alpha gate (sigmoid ensures 0-1 range)
        alpha_t1 = torch.sigmoid(self.t6_to_t1_alpha)  # scalar

        # Context enhancement
        dyn_feat_t1_guided = dyn_feat_t1 + alpha_t1 * delta_t1

        return dyn_feat_t1_guided

    def get_alpha_t1_value(self) -> float:
        """Get current alpha gate value (after sigmoid)."""
        return torch.sigmoid(self.t6_to_t1_alpha).item()

    def get_alpha_t1_raw(self) -> float:
        """Get raw alpha parameter value (before sigmoid)."""
        return self.t6_to_t1_alpha.item()

    def extra_repr(self) -> str:
        """Extra representation for module printing."""
        alpha_sigmoid = torch.sigmoid(self.t6_to_t1_alpha).item()
        return f"c6_deep_dim={self.c6_deep_dim}, dyn_feat_dim={self.dyn_feat_dim}, " \
               f"alpha_init={self.t6_to_t1_alpha.item():.3f}, alpha_sigmoid={alpha_sigmoid:.3f}"


class T6DeepFeatureContextModule(nn.Module):
    """
    Combined T6 Deep Feature Context Module.

    Wraps T6DeepFeatureContextEncoder and T6DeepFeatureBridge into a single
    module for convenient integration into the v4 architecture.

    Args:
        encoder_config (dict): Configuration for T6DeepFeatureContextEncoder
            - input_dim: 48
            - hidden_dim: 32
            - output_dim: 16
            - dropout: 0.2
        bridge_config (dict): Configuration for T6DeepFeatureBridge
            - c6_deep_dim: 16 (must match encoder output_dim)
            - dyn_feat_dim: 48
            - alpha_init: -2.2

    Example:
        config = {
            "encoder": {"input_dim": 48, "hidden_dim": 32, "output_dim": 16, "dropout": 0.2},
            "bridge": {"c6_deep_dim": 16, "dyn_feat_dim": 48, "alpha_init": -2.2}
        }
        module = T6DeepFeatureContextModule(config)
        c6_deep = module.encode(dyn_feat_t6)
        dyn_feat_t1_guided = module.inject_to_t1(dyn_feat_t1, c6_deep)
    """

    def __init__(
        self,
        encoder_config: Optional[Dict] = None,
        bridge_config: Optional[Dict] = None
    ):
        super().__init__()

        # Default configurations
        if encoder_config is None:
            encoder_config = {
                "input_dim": 48,
                "hidden_dim": 32,
                "output_dim": 16,
                "dropout": 0.2
            }
        if bridge_config is None:
            bridge_config = {
                "c6_deep_dim": 16,
                "dyn_feat_dim": 48,
                "alpha_init": -2.2
            }

        # Ensure dimensions match
        if encoder_config.get("output_dim", 16) != bridge_config.get("c6_deep_dim", 16):
            raise ValueError(
                f"Encoder output_dim ({encoder_config.get('output_dim')}) must match "
                f"bridge c6_deep_dim ({bridge_config.get('c6_deep_dim')})"
            )

        # Create sub-modules
        self.encoder = T6DeepFeatureContextEncoder(**encoder_config)
        self.bridge = T6DeepFeatureBridge(**bridge_config)

    def encode(self, dyn_feat_t6: torch.Tensor) -> torch.Tensor:
        """Encode dyn_feat_t6 to c6_deep."""
        return self.encoder(dyn_feat_t6)

    def inject_to_beta_gate(
        self,
        c_beta_full: torch.Tensor,
        c6_deep: torch.Tensor,
        detach_t6_context: bool = False
    ) -> torch.Tensor:
        """Inject c6_deep into Beta Gate context."""
        return self.bridge.inject_to_beta_gate(c_beta_full, c6_deep, detach_t6_context)

    def inject_to_t1_head(
        self,
        dyn_feat_t1: torch.Tensor,
        c6_deep: torch.Tensor,
        detach_t6_context: bool = False
    ) -> torch.Tensor:
        """Inject c6_deep into t1 Alpha head."""
        return self.bridge.inject_to_t1_head(dyn_feat_t1, c6_deep, detach_t6_context)

    def forward(
        self,
        dyn_feat_t6: torch.Tensor,
        dyn_feat_t1: Optional[torch.Tensor] = None,
        c_beta_full: Optional[torch.Tensor] = None,
        detach_t6_context: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining encoding and optional injections.

        Key design: detach_t6_context operates on dyn_feat_t6, not c6_deep.
        This allows the encoder to still learn, while preventing gradient
        backpropagation to alpha_interactors["t6"].

        Args:
            dyn_feat_t6: [B, 48] t6 deep feature
            dyn_feat_t1: [B, 48] t1 dynamic feature (optional, for Injection B)
            c_beta_full: [B, 40] Beta Gate context (optional, for Injection A)
            detach_t6_context: Whether to detach dyn_feat_t6 (prevents gradient to Alpha/t6)

        Returns:
            {
                "c6_deep": [B, 16] disease context vector,
                "c_full_expanded": [B, 56] (if c_beta_full provided),
                "dyn_feat_t1_guided": [B, 48] (if dyn_feat_t1 provided),
            }
        """
        outputs = {}

        # detach operates on dyn_feat_t6, allowing encoder to still learn
        if detach_t6_context:
            dyn_feat_t6_for_context = dyn_feat_t6.detach()
        else:
            dyn_feat_t6_for_context = dyn_feat_t6

        # Encode t6 deep feature
        c6_deep = self.encode(dyn_feat_t6_for_context)
        outputs["c6_deep"] = c6_deep

        # Injection A (Beta Gate)
        if c_beta_full is not None:
            outputs["c_full_expanded"] = self.inject_to_beta_gate(
                c_beta_full, c6_deep, detach_t6_context=False  # Already handled above
            )

        # Injection B (t1 head)
        if dyn_feat_t1 is not None:
            outputs["dyn_feat_t1_guided"] = self.inject_to_t1_head(
                dyn_feat_t1, c6_deep, detach_t6_context=False  # Already handled above
            )

        return outputs

    def get_alpha_t1_value(self) -> float:
        """Get current alpha gate value."""
        return self.bridge.get_alpha_t1_value()

    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for encoder and bridge."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        bridge_params = sum(p.numel() for p in self.bridge.parameters())
        return {
            "encoder": encoder_params,
            "bridge": bridge_params,
            "total": encoder_params + bridge_params
        }