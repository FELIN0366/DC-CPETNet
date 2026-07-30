"""
CNN-GAF Baseline Model

Implements the architecture from:
"Encoding Cardiopulmonary Exercise Testing Time Series as Images for Classification
using Convolutional Neural Network" (Sharma et al., EMBC 2022)

Key Components:
- Module A: PAA downsampling + GADF (Gramian Angular Difference Field) image encoding
- Module B: Independent channel CNN encoding
- Module C: Attention pooling for multi-variable aggregation
- Module D: Static feature fusion for fair comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GADFEncoder(nn.Module):
    """
    Gramian Angular Difference Field (GADF) Encoder

    Transforms 1D time series into 2D image representation using GADF:
    GADF_{i,j} = X̃_i * sqrt(1 - X̃_j²) - X̃_j * sqrt(1 - X̃_i²)

    where X̃ is the min-max normalized series mapped to [-1, 1].

    Args:
        image_size: Target image size after PAA downsampling (default: 64)
    """

    def __init__(self, image_size: int = 64):
        super().__init__()
        self.image_size = image_size

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Transform time series to GADF images.

        Args:
            x: [B, L, C] input time series
            lengths: [B] actual sequence lengths (for masking padding)

        Returns:
            [B, C, L_img, L_img] GADF images
        """
        B, L, C = x.shape
        L_img = self.image_size

        # ===========================================================
        # Step 1: PAA (Piecewise Aggregate Approximation) Downsampling
        # ===========================================================
        # Use adaptive_avg_pool1d to downsample sequence length
        # Input: [B, L, C] -> [B, C, L] for pool1d
        x_paa = x.permute(0, 2, 1)  # [B, C, L]

        # Apply adaptive average pooling
        # Note: This treats padding as valid data; for strict handling,
        # we could mask padding before pooling, but adaptive_avg_pool1d
        # handles variable lengths reasonably by averaging over the full input
        x_paa = F.adaptive_avg_pool1d(x_paa, L_img)  # [B, C, L_img]

        # ===========================================================
        # Step 2: Min-Max Normalization to [-1, 1]
        # ===========================================================
        # Per-sample, per-channel normalization
        # x_min: [B, C, 1], x_max: [B, C, 1]
        x_min = x_paa.min(dim=-1, keepdim=True).values
        x_max = x_paa.max(dim=-1, keepdim=True).values

        # Avoid division by zero
        x_range = x_max - x_min
        x_range = torch.where(x_range < 1e-8, torch.ones_like(x_range), x_range)

        # Normalize to [0, 1] then scale to [-1, 1]
        x_norm = (x_paa - x_min) / x_range  # [0, 1]
        x_scaled = 2.0 * x_norm - 1.0  # [-1, 1]

        # ===========================================================
        # Step 3: GADF Transformation (Vectorized)
        # ===========================================================
        # GADF_{i,j} = X̃_i * sqrt(1 - X̃_j²) - X̃_j * sqrt(1 - X̃_i²)
        # where X̃ = cos(φ), so sqrt(1 - X̃²) = |sin(φ)|

        # x_scaled is treated as cos(φ)
        x_cos = x_scaled  # [B, C, L_img]

        # Compute sin(φ) = sqrt(1 - cos²(φ))
        # Use clamp to prevent numerical instability
        x_sin_sq = 1.0 - x_cos.pow(2)
        x_sin_sq = torch.clamp(x_sin_sq, min=0.0, max=1.0)
        x_sin = torch.sqrt(x_sin_sq)  # [B, C, L_img]

        # Expand for outer product computation
        # x_cos_i: [B, C, L_img, 1], x_cos_j: [B, C, 1, L_img]
        # x_sin_i: [B, C, L_img, 1], x_sin_j: [B, C, 1, L_img]
        x_cos_i = x_cos.unsqueeze(-1)  # [B, C, L_img, 1]
        x_cos_j = x_cos.unsqueeze(-2)  # [B, C, 1, L_img]
        x_sin_i = x_sin.unsqueeze(-1)  # [B, C, L_img, 1]
        x_sin_j = x_sin.unsqueeze(-2)  # [B, C, 1, L_img]

        # GADF = cos_i * sin_j - cos_j * sin_i
        gadf = x_cos_i * x_sin_j - x_cos_j * x_sin_i  # [B, C, L_img, L_img]

        return gadf


class ChannelCNNEncoder(nn.Module):
    """
    Independent Channel CNN Encoder

    Processes each channel's GADF image independently through shared CNN.

    Architecture:
        Conv2d(1, 16, 5x5, padding=2) -> ReLU
        Conv2d(16, 32, 5x5, padding=2) -> ReLU
        AdaptiveAvgPool2d((1, 1)) -> Flatten

    Args:
        in_channels: Number of input channels (always 1 for grayscale GADF)
        hidden_channels: List of hidden channel dimensions [16, 32]
    """

    def __init__(self, in_channels: int = 1, hidden_channels: list = [16, 32]):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.output_dim = hidden_channels[-1]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*C, 1, L_img, L_img] batch of single-channel images

        Returns:
            [B*C, 32] flattened features
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)  # Flatten (use reshape for safety)
        return x


class AttentionPooling(nn.Module):
    """
    Attention-based Multi-variable Aggregation

    Learns attention weights for each channel and aggregates features.

    Architecture:
        Linear(32, 16) -> ReLU -> Linear(16, 1) -> Softmax over channels

    Args:
        feature_dim: Input feature dimension (default: 32)
        attention_dim: Hidden attention dimension (default: 16)
    """

    def __init__(self, feature_dim: int = 32, attention_dim: int = 16):
        super().__init__()

        self.attention_fc1 = nn.Linear(feature_dim, attention_dim)
        self.attention_fc2 = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, D] channel-wise features

        Returns:
            [B, D] aggregated features
        """
        # Compute attention scores
        attn = F.relu(self.attention_fc1(x))  # [B, C, attention_dim]
        attn = self.attention_fc2(attn)  # [B, C, 1]

        # Softmax over channel dimension
        attn_weights = F.softmax(attn, dim=1)  # [B, C, 1]

        # Weighted sum
        out = (x * attn_weights).sum(dim=1)  # [B, D]

        return out


class CNNGAF(nn.Module):
    """
    CNN-GAF Model for CPET Time Series Classification

    Implements the pipeline from Sharma et al. (EMBC 2022):
    1. PAA downsampling + GADF image encoding
    2. Independent channel CNN encoding
    3. Attention pooling for multi-variable aggregation
    4. Static feature fusion (for fair comparison with other baselines)

    Args:
        input_dim: Time steps (e.g., 330 for variable-length mode)
        output_dim: Number of output classes
        num_channel: Number of input channels/features (default: 30)
        image_size: GADF image size after PAA (default: 64)
        cnn_channels: CNN hidden channel dimensions (default: [16, 32])
        attention_dim: Attention hidden dimension (default: 16)
        use_static_features: Enable static feature fusion (default: False)
        static_dim: Static feature encoding dimension (default: 16)
        num_static_features: Number of static features (default: 5)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_channel: int = 30,
        image_size: int = 64,
        cnn_channels: list = None,
        attention_dim: int = 16,
        use_static_features: bool = False,
        static_dim: int = 16,
        num_static_features: int = 5,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_channel = num_channel
        self.image_size = image_size
        self.use_static_features = use_static_features
        self.static_dim = static_dim

        if cnn_channels is None:
            cnn_channels = [16, 32]

        # ===========================================================
        # Module A: GADF Encoder
        # ===========================================================
        self.gadf_encoder = GADFEncoder(image_size=image_size)

        # ===========================================================
        # Module B: Channel CNN Encoder
        # ===========================================================
        self.cnn_encoder = ChannelCNNEncoder(
            in_channels=1,
            hidden_channels=cnn_channels
        )
        self.cnn_feature_dim = cnn_channels[-1]  # 32

        # ===========================================================
        # Module C: Attention Pooling
        # ===========================================================
        self.attention_pooling = AttentionPooling(
            feature_dim=self.cnn_feature_dim,
            attention_dim=attention_dim
        )

        # ===========================================================
        # Module D: Static Feature Fusion
        # ===========================================================
        if use_static_features:
            from .common import StaticFeatureEncoder
            self.static_encoder = StaticFeatureEncoder(
                num_static_features, static_dim, dropout
            )
            fusion_dim = self.cnn_feature_dim + static_dim
        else:
            self.static_encoder = None
            fusion_dim = self.cnn_feature_dim

        # ===========================================================
        # Classifier
        # ===========================================================
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None,
        prior_adj: torch.Tensor = None,
        static_x: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [B, L, C] input time series
            lengths: [B] actual sequence lengths (unused but kept for interface)
            prior_adj: Not used, kept for interface compatibility
            static_x: [B, num_static_features] static features

        Returns:
            [B, output_dim] logits
        """
        B, L, C = x.shape

        # ===========================================================
        # Module A: GADF Encoding
        # ===========================================================
        # Input: [B, L, C] -> Output: [B, C, L_img, L_img]
        gadf_images = self.gadf_encoder(x, lengths)

        # ===========================================================
        # Module B: CNN Encoding (Independent per channel)
        # ===========================================================
        # Reshape: [B, C, L_img, L_img] -> [B*C, 1, L_img, L_img]
        # Use reshape() instead of view() because GADF output may be non-contiguous
        B, C, H, W = gadf_images.shape
        gadf_flat = gadf_images.reshape(B * C, 1, H, W)

        # CNN forward
        cnn_features = self.cnn_encoder(gadf_flat)  # [B*C, 32]

        # Reshape back: [B*C, 32] -> [B, C, 32]
        cnn_features = cnn_features.reshape(B, C, -1)

        # ===========================================================
        # Module C: Attention Pooling
        # ===========================================================
        dyn_features = self.attention_pooling(cnn_features)  # [B, 32]

        # ===========================================================
        # Module D: Static Feature Fusion
        # ===========================================================
        if self.use_static_features and static_x is not None and self.static_encoder is not None:
            stat_features = self.static_encoder(static_x)  # [B, static_dim]
            features = torch.cat([dyn_features, stat_features], dim=1)
        else:
            features = dyn_features

        # ===========================================================
        # Classification
        # ===========================================================
        logits = self.classifier(features)

        return logits


# Alias for backward compatibility
CNNGAFModel = CNNGAF