"""
ResNet1D Baseline Model

1D ResNet for CPET time series classification with support for:
- Variable-length sequences (via masking)
- Static feature fusion (EHR + PFT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block for 1D ResNet"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for CPET time series classification

    Args:
        input_dim: Time steps (e.g., 162 or max_length for variable-length)
        output_dim: Number of output classes
        num_channel: Number of input channels/features
        hidden_dim: Hidden dimension (default: 64)
        use_variable_length: Enable variable-length sequence support (default: False)
        use_static_features: Enable static feature fusion (default: False)
        static_dim: Static feature encoding dimension (default: 16)
        num_static_features: Number of static features (default: 5)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_channel: int,
        hidden_dim: int = 64,
        use_variable_length: bool = False,
        use_static_features: bool = False,
        static_dim: int = 16,
        num_static_features: int = 5,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_channel = num_channel
        self.use_variable_length = use_variable_length
        self.use_static_features = use_static_features
        self.static_dim = static_dim

        self.inplanes = 64

        # 1. Initial convolution (Stem)
        self.conv1 = nn.Conv1d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 2. Residual layers (lightweight ResNet-18 style)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)

        # 3. Output layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # 4. Static feature encoder
        if use_static_features:
            from .common import StaticFeatureEncoder
            self.static_encoder = StaticFeatureEncoder(num_static_features, static_dim, dropout)
            fc_input_dim = 128 + static_dim
        else:
            self.static_encoder = None
            fc_input_dim = 128

        # 5. Classifier
        self.fc = nn.Linear(fc_input_dim, output_dim)

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
            x: [B, T, C] input tensor (Time, Channels)
            lengths: [B] actual sequence lengths (for variable-length mode)
            prior_adj: Not used, kept for interface compatibility
            static_x: [B, num_static_features] static features

        Returns:
            [B, output_dim] logits
        """
        B, T, C = x.shape

        # Variable-length masking
        if self.use_variable_length and lengths is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * mask.unsqueeze(-1).float()

        # Transpose for Conv1d: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        # CNN forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        feat = torch.flatten(x, 1)  # [B, 128]
        feat = self.dropout(feat)

        # Static feature fusion
        if self.use_static_features and static_x is not None and self.static_encoder is not None:
            feat_stat = self.static_encoder(static_x)  # [B, static_dim]
            feat = torch.cat([feat, feat_stat], dim=1)

        return self.fc(feat)