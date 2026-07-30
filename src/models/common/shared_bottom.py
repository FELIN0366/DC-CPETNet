"""
Shared-bottom Common Components

Shared dynamic representation encoder for Shared-bottom MTL baseline.
All t1-t5 tasks share the same encoder, no expert routing, no task-specific gates.

Components:
- SharedBottomDynamicEncoder: Single shared encoder for all tasks

Design Principles:
- One shared bottom encoder (no Alpha/Beta split)
- All tasks use the same H_shared representation
- No MMoE expert pool
- No task-specific gates
- No routing constraints

Author: Shared-bottom Clean Implementation
Date: 2026-05-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# =============================================================================
# Shared Bottom Dynamic Encoder
# =============================================================================

class SharedBottomDynamicEncoder(nn.Module):
    """
    Shared dynamic representation encoder for Shared-bottom MTL.

    All t1-t5 tasks share this single encoder. No task-specific routing,
    no expert pool, no gates. Just one unified encoder producing H_shared.

    Input:  [B, L, 30] CPET dynamic sequence
    Output: [B, 30, 16] shared representation

    Key Features:
    - Temporal CNN encoding (single-scale, no multiscale)
    - Masked aggregation for variable-length sequences
    - Dropout and normalization
    - Channel-wise representation

    Args:
        num_channels: Input channel count (default 30 for nine_graph)
        D_time: Output temporal dimension (default 16)
        T_mid: Intermediate temporal dimension (default 24)
        dropout: Dropout rate (default 0.3)
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_channels = num_channels
        self.D_time = D_time
        self.T_mid = T_mid

        print(f"\n[SharedBottom Dynamic Encoder] Initialized:")
        print(f"  num_channels = {num_channels}")
        print(f"  D_time = {D_time}")
        print(f"  T_mid = {T_mid}")
        print(f"  dropout = {dropout}")
        print(f"  All tasks share this encoder (no routing)")

        # Temporal encoder: Depthwise Separable Conv
        # [B, L, C] -> [B, C, T_mid] -> [B, C, D_time]
        self.temporal_conv = nn.Sequential(
            # Depthwise conv: channel-wise temporal feature extraction
            nn.Conv1d(num_channels, T_mid, kernel_size=7, padding=3),
            nn.BatchNorm1d(T_mid),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Pointwise conv: reduce to D_time dimension
            nn.Conv1d(T_mid, D_time, kernel_size=1),
            nn.BatchNorm1d(D_time),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Masked temporal aggregation layer
        # Aggregates temporal dimension while preserving channel structure
        self.temporal_aggregation = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global pooling over time: [B, D, L] -> [B, D, 1]
            nn.Flatten(),            # [B, D]
        )

        # Channel projection: expand to [B, C, D]
        # Each channel gets the same aggregated temporal representation
        # This maintains the node structure for downstream task heads (PMGT/Flatten)
        self.channel_expand = nn.Linear(D_time, num_channels * D_time)

    def forward(
        self,
        x_dyn: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through SharedBottom encoder.

        Args:
            x_dyn: [B, L, C] dynamic sequence
            lengths: [B] sequence lengths (optional, for masking)

        Returns:
            H_shared: [B, C, D_time] shared representation for all tasks
        """
        B, L, C = x_dyn.shape

        # Transpose for Conv1d: [B, C, L]
        x = x_dyn.transpose(1, 2)  # [B, C, L]

        # Temporal convolution: [B, C, L] -> [B, T_mid, L] -> [B, D_time, L]
        h = self.temporal_conv(x)  # [B, D_time, L]

        # Masked aggregation if lengths provided
        if lengths is not None:
            # Create mask: [B, L]
            mask = torch.arange(L, device=x_dyn.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.float().unsqueeze(1)  # [B, 1, L]

            # Apply mask and compute weighted mean
            h_masked = h * mask  # [B, D_time, L]
            h_sum = h_masked.sum(dim=2)  # [B, D_time]
            h_agg = h_sum / lengths.unsqueeze(1).float()  # [B, D_time]
        else:
            # Global mean pooling: no masking needed
            h_agg = h.mean(dim=2)  # [B, D_time]

        # Expand to channel structure: [B, C, D_time]
        # This preserves the node structure for PMGT (t1) and FlattenProjector (t2-t5)
        # Each channel gets a copy of the aggregated temporal representation
        h_expanded = self.channel_expand(h_agg)  # [B, C * D_time]
        H_shared = h_expanded.view(B, self.num_channels, self.D_time)  # [B, C, D_time]

        return H_shared


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SharedBottom Dynamic Encoder Test")
    print("=" * 60)

    # Create encoder
    encoder = SharedBottomDynamicEncoder(num_channels=30, D_time=16, T_mid=24)

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {num_params}")

    # Test forward without lengths
    B, L, C = 4, 200, 30
    x_dyn = torch.randn(B, L, C)
    H = encoder(x_dyn)
    print(f"\nForward (no lengths): H_shared shape = {H.shape}")
    assert H.shape == (B, 30, 16), f"Expected [B, 30, 16], got {H.shape}"

    # Test forward with lengths
    lengths = torch.tensor([100, 150, 200, 180])
    H = encoder(x_dyn, lengths)
    print(f"Forward (with lengths): H_shared shape = {H.shape}")
    assert H.shape == (B, 30, 16), f"Expected [B, 30, 16], got {H.shape}"

    # Test gradient flow
    loss = H.sum()
    loss.backward()
    print("\nBackward pass successful")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)