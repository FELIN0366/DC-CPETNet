"""
MedNet Baseline Model

Medical-informed network with multi-stream architecture:
- Stream A: Shape (CNN) - captures waveform patterns like O2 Pulse plateau
- Stream B: Stat (Global) - captures extreme values and ratios
- Stream C: Rhythm (HR Focus) - captures arrhythmia patterns

Supports:
- Variable-length sequences (via masking)
- Static feature fusion (EHR + PFT)
- Dynamic HR index detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """1D Residual block for capturing waveform features"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class MedNet(nn.Module):
    """
    Medical-Informed Network for CPET Classification

    Three-stream architecture based on clinical interpretation patterns:
    - Stream A (Shape): CNN for waveform morphology (ischemia, heart failure)
    - Stream B (Stat): Statistical features (pulmonary vascular, ventilation limits)
    - Stream C (Rhythm): HR dynamics (arrhythmia detection)

    Args:
        input_dim: Time steps (e.g., 162)
        hidden_dim: Not used directly, kept for interface compatibility
        channel_groups: Channel group indices (for compatibility)
        output_dim: Number of output classes
        num_channel: Number of input channels/features
        hr_idx: Index of HR channel (default: auto-detect from feature names)
        use_variable_length: Enable variable-length support (default: False)
        use_static_features: Enable static feature fusion (default: False)
        static_dim: Static feature encoding dimension (default: 16)
        num_static_features: Number of static features (default: 5)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        channel_groups,
        output_dim: int,
        num_channel: int,
        hr_idx: int = None,
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

        # Dynamic HR index detection
        self.hr_idx = hr_idx if hr_idx is not None else self._find_hr_index()
        # Ensure HR index is within bounds
        self.hr_idx = min(self.hr_idx, num_channel - 1)

        # --- Stream A: Shape (CNN) ---
        # Responsible for: ischemia (plateau), heart failure (oscillatory breathing)
        self.shape_encoder = nn.Sequential(
            ResBlock1D(num_channel, 32, kernel_size=7, stride=2, dropout=dropout),  # [B, 32, T/2]
            ResBlock1D(32, 64, kernel_size=5, stride=2, dropout=dropout),            # [B, 64, T/4]
            ResBlock1D(64, 128, kernel_size=3, stride=2, dropout=dropout),           # [B, 128, T/8]
            nn.AdaptiveAvgPool1d(1)                                                   # [B, 128, 1]
        )

        # --- Stream B: Stat (Global Features) ---
        # Responsible for: pulmonary vascular (high slope), ventilation limits (low reserve)
        self.stat_dim = num_channel * 3  # Mean, Std, Max
        self.stat_encoder = nn.Sequential(
            nn.Linear(self.stat_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1)
        )

        # --- Stream C: Rhythm (Arrhythmia Specialist) ---
        # Responsible for: arrhythmia detection via HR dynamics
        self.rhythm_encoder = nn.Sequential(
            nn.Linear(4, 16),  # Input: [HR_std, HR_diff_std, HR_diff2_std, HR_range]
            nn.ReLU()
        )

        # --- Static feature encoder ---
        if use_static_features:
            from .common import StaticFeatureEncoder
            self.static_encoder = StaticFeatureEncoder(num_static_features, static_dim, dropout)
        else:
            self.static_encoder = None

        # --- Fusion ---
        self.fusion_dim = 128 + 64 + 16 + (static_dim if use_static_features else 0)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout + 0.2),
            nn.Linear(64, output_dim)
        )

        self._init_weights()

    def _find_hr_index(self) -> int:
        """
        Auto-detect HR channel index from feature mapping

        Returns:
            Index of HR channel (default: 3 if not found)
        """
        try:
            from ..feature_mapping import NEW_FEATURES
            for i, name in enumerate(NEW_FEATURES):
                if name == 'HR' or 'HR' in name:
                    return i
        except ImportError:
            pass
        # Default fallback: HR is typically at index 3
        return 3

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

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
            x: [B, T, C] input tensor (Batch, Time, Channels)
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

        # Permute for CNN: [B, T, C] -> [B, C, T]
        x_cnn = x.permute(0, 2, 1)

        # --- Stream A: Shape (CNN) ---
        feat_shape = self.shape_encoder(x_cnn).squeeze(-1)  # [B, 128]

        # --- Stream B: Stat ---
        f_mean = x.mean(dim=1)
        f_std = x.std(dim=1)
        f_max, _ = x.max(dim=1)
        x_stat = torch.cat([f_mean, f_std, f_max], dim=1)  # [B, C*3]
        feat_stat = self.stat_encoder(x_stat)  # [B, 64]

        # --- Stream C: Rhythm (HR Focus) ---
        hr_data = x[:, :, self.hr_idx]  # [B, T]

        # 1. Basic variation
        hr_std = hr_data.std(dim=1, keepdim=True)
        hr_range = (hr_data.max(dim=1)[0] - hr_data.min(dim=1)[0]).unsqueeze(1)

        # 2. First-order difference (instantaneous rate) -> captures premature beats
        hr_diff = hr_data[:, 1:] - hr_data[:, :-1]
        hr_diff_std = hr_diff.std(dim=1, keepdim=True)

        # 3. Second-order difference (rate of change) -> captures oscillations
        hr_diff2 = hr_diff[:, 1:] - hr_diff[:, :-1]
        hr_diff2_std = hr_diff2.std(dim=1, keepdim=True)

        feat_rhythm = torch.cat([hr_std, hr_diff_std, hr_diff2_std, hr_range], dim=1)  # [B, 4]
        feat_rhythm = self.rhythm_encoder(feat_rhythm)  # [B, 16]

        # --- Fusion ---
        features = [feat_shape, feat_stat, feat_rhythm]

        # Static feature fusion
        if self.use_static_features and static_x is not None and self.static_encoder is not None:
            feat_stat_enc = self.static_encoder(static_x)  # [B, static_dim]
            features.append(feat_stat_enc)

        combined = torch.cat(features, dim=1)
        logits = self.classifier(combined)

        return logits