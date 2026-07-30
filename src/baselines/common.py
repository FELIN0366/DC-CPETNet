"""
Common Components for Baseline Models

This module provides shared infrastructure for all baseline models:
- StaticFeatureEncoder: Encodes EHR/PFT static features
"""

import torch
import torch.nn as nn


class StaticFeatureEncoder(nn.Module):
    """
    Static Feature Encoder for EHR + PFT fusion

    Encodes demographic and pulmonary function test features
    to be concatenated with dynamic CPET features.

    Architecture:
        Linear(num_features -> 32) -> LayerNorm -> ReLU -> Dropout
        Linear(32 -> static_dim) -> ReLU

    Args:
        num_features: Number of static features (default: 5 for EHR only)
        static_dim: Output embedding dimension (default: 16)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(self, num_features: int = 5, static_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.num_features = num_features
        self.static_dim = static_dim

        self.encoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, static_dim),
            nn.ReLU()
        )

    def forward(self, static_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            static_x: [B, num_features] static feature tensor

        Returns:
            [B, static_dim] encoded static features
        """
        return self.encoder(static_x)