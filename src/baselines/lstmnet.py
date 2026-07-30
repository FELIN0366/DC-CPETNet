"""
LSTMNet Baseline Model

Bidirectional LSTM for CPET time series classification with support for:
- Variable-length sequences (via pack_padded_sequence)
- Static feature fusion (EHR + PFT)
"""

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    """
    Bidirectional LSTM for CPET time series classification

    Args:
        input_dim: Time steps (for interface compatibility, not used directly)
        output_dim: Number of output classes
        num_channel: Number of input channels/features (LSTM input_size)
        hidden_dim: LSTM hidden dimension (default: 64)
        num_layers: Number of LSTM layers (default: 2)
        use_variable_length: Enable variable-length sequence support (default: False)
        use_static_features: Enable static feature fusion (default: False)
        static_dim: Static feature encoding dimension (default: 16)
        num_static_features: Number of static features (default: 5)
        dropout: Dropout rate (default: 0.5)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_channel: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        use_variable_length: bool = False,
        use_static_features: bool = False,
        static_dim: int = 16,
        num_static_features: int = 5,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_channel = num_channel
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_variable_length = use_variable_length
        self.use_static_features = use_static_features
        self.static_dim = static_dim

        # LSTM layer (bidirectional)
        self.lstm = nn.LSTM(
            input_size=num_channel,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Static feature encoder
        if use_static_features:
            from .common import StaticFeatureEncoder
            self.static_encoder = StaticFeatureEncoder(num_static_features, static_dim, dropout)
            fc_input_dim = hidden_dim * 2 + static_dim
        else:
            self.static_encoder = None
            fc_input_dim = hidden_dim * 2

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
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
            x: [B, T, C] input tensor (Batch, Time, Channels)
            lengths: [B] actual sequence lengths (for variable-length mode)
            prior_adj: Not used, kept for interface compatibility
            static_x: [B, num_static_features] static features

        Returns:
            [B, output_dim] logits
        """
        # Variable-length processing with pack_padded_sequence
        if self.use_variable_length and lengths is not None:
            # Sort by length (descending) for pack_padded_sequence
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            x_sorted = x[sorted_idx]

            # Pack sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )

            # LSTM forward
            out_packed, _ = self.lstm(packed)

            # Unpack sequence
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

            # Unsort to original order
            _, unsorted_idx = sorted_idx.sort()
            out = out[unsorted_idx]
        else:
            # Standard LSTM forward
            out, _ = self.lstm(x)  # [B, T, hidden*2]

        # Global average pooling over time
        out = out.mean(dim=1)  # [B, hidden*2]

        # Static feature fusion
        if self.use_static_features and static_x is not None and self.static_encoder is not None:
            feat_stat = self.static_encoder(static_x)  # [B, static_dim]
            out = torch.cat([out, feat_stat], dim=1)

        return self.fc(out)