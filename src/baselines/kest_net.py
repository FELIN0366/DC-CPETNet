"""
KEST-Net: Knowledge-Enhanced Spatio-Temporal Network

Reference:
    Qu et al., "KEST: Knowledge-Enhanced Spatio-Temporal Network for
    Cardiopulmonary Exercise Testing", BIBE 2024.

Architecture:
    - View 1: Spatio-Temporal Feature View (Spatial GCN + Temporal Transformer)
    - View 2: Knowledge-Enhanced System Feature View (CST Block)
    - Multi-view Fusion with Static Features

Adapted from the original 3-system design to support 4 functional subsystems
(S0-S3) based on the Wasserman Nine-Graph paradigm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, List, Dict, Any

from .common import StaticFeatureEncoder


# =============================================================================
# Helper Modules
# =============================================================================

class GraphConvolution(nn.Module):
    """Graph Convolution Layer (GCN)"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        activation=F.relu
    ):
        super().__init__()
        self.activation = activation
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, F_in] or [B, T, C, F_in] node features
            adj: [C, C] normalized adjacency matrix

        Returns:
            [B, C, F_out] or [B, T, C, F_out] transformed features
        """
        # Linear transform
        support = torch.matmul(x, self.weight)

        # Neighbor aggregation
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return self.activation(output)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional masking"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, L_q, D]
            key: [B, L_k, D]
            value: [B, L_k, D]
            key_padding_mask: [B, L_k] True for padded positions
            attn_mask: [L_q, L_k] additive mask

        Returns:
            [B, L_q, D] attention output
        """
        B, L_q, _ = query.shape
        L_k = key.shape[1]

        # Project and reshape
        Q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_k]

        # Apply masks
        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            # key_padding_mask: [B, L_k] -> [B, 1, 1, L_k]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = (attn @ V).transpose(1, 2).reshape(B, L_q, self.embed_dim)
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Layer"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            key_padding_mask: [B, L] True for padded positions

        Returns:
            [B, L, D]
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(x)

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


# =============================================================================
# View 1: Spatio-Temporal Feature View
# =============================================================================

class SpatioTemporalView(nn.Module):
    """
    Spatio-Temporal Feature View

    Architecture:
        1. Spatial GCN: Aggregate features across channels at each time step
        2. Temporal Transformer: Model temporal dependencies
        3. Mean Pooling: Aggregate over time (respecting padding)
    """

    def __init__(
        self,
        num_channels: int,
        hidden_dim: int,
        num_gcn_layers: int,
        num_transformer_layers: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        # Channel embedding
        self.channel_embed = nn.Linear(1, hidden_dim)

        # Spatial GCN layers
        self.gcn_layers = nn.ModuleList([
            GraphConvolution(hidden_dim, hidden_dim, activation=F.relu)
            for _ in range(num_gcn_layers)
        ])
        self.gcn_norm = nn.LayerNorm(hidden_dim)

        # Temporal Transformer layers
        self.temporal_transformer = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] input tensor
            adj: [C, C] normalized adjacency matrix
            lengths: [B] actual sequence lengths

        Returns:
            z_st: [B, hidden_dim] spatio-temporal features
        """
        B, T, C = x.shape

        # 1. Channel embedding: [B, T, C] -> [B, T, C, D]
        x = x.unsqueeze(-1)  # [B, T, C, 1]
        x = self.channel_embed(x)  # [B, T, C, D]

        # 2. Spatial GCN: process each time step
        # Reshape: [B, T, C, D] -> [B*T, C, D]
        x = x.reshape(B * T, C, self.hidden_dim)

        for gcn in self.gcn_layers:
            x = gcn(x, adj)
        x = self.gcn_norm(x)

        # Reshape back and aggregate channels: [B*T, C, D] -> [B, T, D]
        x = x.reshape(B, T, C, self.hidden_dim)
        # Mean pooling over channels to get one vector per time step
        x = x.mean(dim=2)  # [B, T, D]

        # 3. Temporal Transformer
        # Create key_padding_mask from lengths
        if lengths is not None:
            max_len = T
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            key_padding_mask = mask  # [B, T], True for padded positions
        else:
            key_padding_mask = None

        for layer in self.temporal_transformer:
            x = layer(x, key_padding_mask=key_padding_mask)

        # 4. Mean Pooling (respecting padding)
        if lengths is not None:
            # Create mask for valid positions
            valid_mask = ~key_padding_mask.unsqueeze(-1)  # [B, T, 1]
            x = x * valid_mask.float()
            z_st = x.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            z_st = x.mean(dim=1)

        return self.dropout(z_st)


# =============================================================================
# View 2: Knowledge-Enhanced System Feature View
# =============================================================================

class CSTBlock(nn.Module):
    """
    Cardiopulmonary System Transformer (CST) Block

    Implements alternating update mechanism:
        Step 1: Signal-to-System (Cross-Attention)
        Step 2: Within-System Collaboration (Self-Attention)

    Args:
        signal_dim: Signal token dimension
        system_dim: System token dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        signal_dim: int,
        system_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.signal_dim = signal_dim
        self.system_dim = system_dim

        # Cross-attention: System tokens attend to Signal tokens
        self.cross_attn = MultiHeadAttention(system_dim, num_heads, dropout)
        self.norm_cross = nn.LayerNorm(system_dim)

        # Self-attention: System tokens attend to each other
        self.self_attn = MultiHeadAttention(system_dim, num_heads, dropout)
        self.norm_self = nn.LayerNorm(system_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(system_dim, system_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(system_dim * 4, system_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(system_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        signal_tokens: torch.Tensor,
        system_tokens: torch.Tensor,
        system_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            signal_tokens: [B, num_signals, signal_dim]
            system_tokens: [B, num_systems, system_dim]
            system_mask: [num_signals, num_systems] binary mask
            key_padding_mask: [B, T] for signal tokens (optional)

        Returns:
            updated_system_tokens: [B, num_systems, system_dim]
        """
        B = signal_tokens.shape[0]
        num_systems = system_tokens.shape[1]

        # Step 1: Signal-to-System Cross-Attention
        # Each system token queries its corresponding signal tokens
        residual = system_tokens
        system_tokens = self.norm_cross(system_tokens)

        # For cross-attention, we need to route each system to its signals
        # Using system_mask to create attention masks
        updated_systems = []
        for sys_idx in range(num_systems):
            # Get signal indices for this system
            signal_indices = system_mask[:, sys_idx].nonzero(as_tuple=True)[0]

            if len(signal_indices) == 0:
                # No signals for this system
                updated_systems.append(system_tokens[:, sys_idx:sys_idx+1, :])
                continue

            # Extract signals for this system
            sys_signals = signal_tokens[:, signal_indices, :]  # [B, num_sys_signals, signal_dim]

            # Cross-attention: query from system, key/value from signals
            query = system_tokens[:, sys_idx:sys_idx+1, :]  # [B, 1, system_dim]
            attended = self.cross_attn(query, sys_signals, sys_signals)
            updated_systems.append(attended)

        system_tokens = torch.cat(updated_systems, dim=1)  # [B, num_systems, system_dim]
        system_tokens = residual + self.dropout(system_tokens)

        # Step 2: Within-System Self-Attention
        residual = system_tokens
        system_tokens = self.norm_self(system_tokens)
        system_tokens = self.self_attn(system_tokens, system_tokens, system_tokens)
        system_tokens = residual + self.dropout(system_tokens)

        # Step 3: FFN
        residual = system_tokens
        system_tokens = self.norm_ffn(system_tokens)
        system_tokens = residual + self.ffn(system_tokens)

        return system_tokens


class KnowledgeEnhancedSystemView(nn.Module):
    """
    Knowledge-Enhanced System Feature View

    Architecture:
        1. Signal-level feature extraction: 1D Conv
        2. System mask construction: from channel groups
        3. System token construction: Position-wise dot product
        4. CST Block: Iterative update
        5. Pooling: Aggregate system tokens
    """

    def __init__(
        self,
        num_channels: int,
        signal_dim: int,
        system_dim: int,
        num_systems: int,
        num_cst_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        system_channel_indices: Optional[List[List[int]]] = None
    ):
        super().__init__()

        self.num_channels = num_channels
        self.signal_dim = signal_dim
        self.system_dim = system_dim
        self.num_systems = num_systems
        self.system_channel_indices = system_channel_indices

        # Signal-level feature extraction (1D Conv)
        self.signal_conv = nn.Sequential(
            nn.Conv1d(1, signal_dim // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(signal_dim // 2),
            nn.GELU(),
            nn.Conv1d(signal_dim // 2, signal_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(signal_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # System token initialization
        self.system_embed = nn.Parameter(torch.randn(1, num_systems, system_dim))
        nn.init.xavier_uniform_(self.system_embed)

        # Learnable aggregation weights for each channel
        self.agg_weights = nn.Parameter(torch.randn(num_channels, num_systems))
        nn.init.xavier_uniform_(self.agg_weights)

        # CST Blocks
        self.cst_blocks = nn.ModuleList([
            CSTBlock(signal_dim, system_dim, num_heads, dropout)
            for _ in range(num_cst_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(num_systems * system_dim, system_dim)
        self.dropout = nn.Dropout(dropout)

        # Register system mask as buffer
        self._create_system_mask()

    def _create_system_mask(self):
        """Create binary mask M ∈ {0, 1}^{num_channels × num_systems}"""
        mask = torch.zeros(self.num_channels, self.num_systems)

        if self.system_channel_indices is not None:
            for sys_idx, channel_indices in enumerate(self.system_channel_indices):
                for ch_idx in channel_indices:
                    if ch_idx < self.num_channels:
                        mask[ch_idx, sys_idx] = 1.0

        self.register_buffer('system_mask', mask)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] input tensor
            lengths: [B] actual sequence lengths (unused here, kept for interface consistency)

        Returns:
            z_sys: [B, system_dim] system-level features
        """
        B, T, C = x.shape

        # 1. Signal-level feature extraction
        # Process each channel independently: [B, T, C] -> [B*C, T, 1]
        x_reshaped = x.transpose(1, 2).reshape(B * C, T, 1).transpose(1, 2)  # [B*C, 1, T]

        # Apply 1D Conv: [B*C, 1, T] -> [B*C, signal_dim, 1] -> [B*C, signal_dim]
        signal_features = self.signal_conv(x_reshaped).squeeze(-1)  # [B*C, signal_dim]

        # Reshape to [B, C, signal_dim]
        signal_tokens = signal_features.reshape(B, C, self.signal_dim)

        # 2. System token initialization via position-wise dot product
        # Aggregate signal tokens using learnable weights and system mask
        # [B, C, signal_dim] @ [C, num_systems] -> [B, num_systems, signal_dim]
        masked_weights = F.softmax(self.agg_weights * self.system_mask, dim=0)

        # Project signal_dim to system_dim if needed
        if self.signal_dim != self.system_dim:
            signal_tokens = F.linear(signal_tokens, torch.randn(self.system_dim, self.signal_dim))

        system_tokens = torch.einsum('bcd,cs->bsd', signal_tokens, masked_weights)

        # Add learnable system embedding
        system_tokens = system_tokens + self.system_embed.expand(B, -1, -1)

        # 3. CST Block iterations
        for cst_block in self.cst_blocks:
            system_tokens = cst_block(
                signal_tokens if self.signal_dim == self.system_dim else
                F.linear(signal_tokens, torch.randn(self.signal_dim, self.system_dim)),
                system_tokens,
                self.system_mask,
                None
            )

        # 4. Pooling: Flatten system tokens
        z_sys = system_tokens.reshape(B, -1)  # [B, num_systems * system_dim]
        z_sys = self.output_proj(z_sys)  # [B, system_dim]

        return self.dropout(z_sys)


# =============================================================================
# Main Model: KEST-Net
# =============================================================================

class KESTNet(nn.Module):
    """
    Knowledge-Enhanced Spatio-Temporal Network (KEST-Net)

    Reference: Qu et al., BIBE 2024

    Two-view architecture:
    - View 1: Spatio-Temporal Feature View (Spatial GCN + Temporal Transformer)
    - View 2: Knowledge-Enhanced System Feature View (CST Block)

    Adapted to support 4 functional subsystems (S0-S3) based on Wasserman Nine-Graph.

    Args:
        input_dim: Maximum sequence length (e.g., 330 for variable-length mode)
        num_channel: Number of input channels (default: 30 for nine_graph mode)
        output_dim: Number of output classes
        hidden_dim: Base hidden dimension (default: 64)
        num_gcn_layers: Number of GCN layers (default: 2)
        num_transformer_layers: Number of temporal transformer layers (default: 2)
        num_cst_layers: Number of CST block iterations (default: 2)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.3)
        semantic_adj: Semantic adjacency matrix [C, C]
        system_channel_indices: List of channel indices for each system
        use_variable_length: Enable variable-length support (default: True)
        use_static_features: Enable static feature fusion (default: True)
        static_dim: Static feature encoding dimension (default: 16)
        num_static_features: Number of static features (default: 5)
    """

    def __init__(
        self,
        input_dim: int,
        num_channel: int = 30,
        output_dim: int = 5,
        hidden_dim: int = 64,
        num_gcn_layers: int = 2,
        num_transformer_layers: int = 2,
        num_cst_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        semantic_adj: Optional[np.ndarray] = None,
        system_channel_indices: Optional[List[List[int]]] = None,
        use_variable_length: bool = True,
        use_static_features: bool = True,
        static_dim: int = 16,
        num_static_features: int = 5,
        **kwargs
    ):
        super().__init__()

        # Store config
        self.use_variable_length = use_variable_length
        self.use_static_features = use_static_features
        self.num_channel = num_channel
        self.hidden_dim = hidden_dim

        # View dimensions
        self.st_view_dim = hidden_dim  # Spatio-temporal view output
        self.sys_view_dim = hidden_dim  # System view output

        # Default system channel indices (4 subsystems: S0-S3)
        if system_channel_indices is None:
            system_channel_indices = [
                list(range(0, 11)),    # S0: oxygen_delivery (11 channels)
                list(range(11, 17)),   # S1: ventilation_drive (6 channels)
                list(range(17, 25)),   # S2: vq_matching (8 channels)
                list(range(25, 30)),   # S3: stability_reserve (5 channels)
            ]

        self.num_systems = len(system_channel_indices)

        # Register semantic adjacency matrix
        if semantic_adj is not None:
            self.register_buffer('semantic_adj', torch.from_numpy(semantic_adj).float())
        else:
            # Identity matrix as fallback
            self.register_buffer('semantic_adj', torch.eye(num_channel))

        # Normalize adjacency matrix (symmetric normalization)
        self._normalize_adj()

        # View 1: Spatio-Temporal Feature View
        self.spatio_temporal_view = SpatioTemporalView(
            num_channels=num_channel,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gcn_layers,
            num_transformer_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # View 2: Knowledge-Enhanced System Feature View
        self.system_view = KnowledgeEnhancedSystemView(
            num_channels=num_channel,
            signal_dim=hidden_dim,
            system_dim=hidden_dim,
            num_systems=self.num_systems,
            num_cst_layers=num_cst_layers,
            num_heads=num_heads,
            dropout=dropout,
            system_channel_indices=system_channel_indices
        )

        # Static Feature Encoder
        if use_static_features:
            self.static_encoder = StaticFeatureEncoder(
                num_features=num_static_features,
                static_dim=static_dim,
                dropout=dropout
            )
            fusion_dim = self.st_view_dim + self.sys_view_dim + static_dim
        else:
            self.static_encoder = None
            fusion_dim = self.st_view_dim + self.sys_view_dim

        # View Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _normalize_adj(self):
        """Symmetric normalization: D^{-1/2} A D^{-1/2}"""
        adj = self.semantic_adj
        # Add self-loops
        adj = adj + torch.eye(adj.shape[0], device=adj.device)
        # Degree matrix
        degree = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag(d_inv_sqrt)
        # Normalize
        self.normalized_adj = d_mat @ adj @ d_mat

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        prior_adj: Optional[torch.Tensor] = None,
        static_x: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [B, T, C] input tensor
            lengths: [B] actual sequence lengths (for variable-length mode)
            prior_adj: [C, C] semantic adjacency matrix (optional, uses self.semantic_adj if None)
            static_x: [B, num_static_features] static features

        Returns:
            [B, output_dim] logits
        """
        # Use provided prior_adj or internal semantic_adj
        if prior_adj is not None:
            adj = prior_adj.to(x.device)
            # Normalize
            adj = adj + torch.eye(adj.shape[0], device=x.device)
            degree = adj.sum(dim=1)
            d_inv_sqrt = torch.pow(degree, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            d_mat = torch.diag(d_inv_sqrt)
            adj = d_mat @ adj @ d_mat
        else:
            adj = self.normalized_adj

        # View 1: Spatio-Temporal Feature View
        z_st = self.spatio_temporal_view(x, adj, lengths)

        # View 2: Knowledge-Enhanced System Feature View
        z_sys = self.system_view(x, lengths)

        # Multi-view Fusion
        combined = torch.cat([z_st, z_sys], dim=1)

        # Static Feature Fusion
        if self.use_static_features and static_x is not None and self.static_encoder is not None:
            static_feat = self.static_encoder(static_x)
            combined = torch.cat([combined, static_feat], dim=1)

        # Classification
        logits = self.classifier(combined)

        return logits


# =============================================================================
# Model Factory
# =============================================================================

def create_kest_net(
    config,
    num_channel: int,
    output_dim: int,
    semantic_adj: Optional[np.ndarray] = None,
    system_channel_indices: Optional[List[List[int]]] = None
) -> KESTNet:
    """
    Factory function to create KESTNet from config

    Args:
        config: Config object with model parameters
        num_channel: Number of input channels
        output_dim: Number of output classes
        semantic_adj: Semantic adjacency matrix
        system_channel_indices: Channel indices for each system

    Returns:
        KESTNet model instance
    """
    model_config = getattr(config.models, 'KESTNet', None)

    return KESTNet(
        input_dim=config.data.max_length,
        num_channel=num_channel,
        output_dim=output_dim,
        hidden_dim=getattr(model_config, 'hidden_dim', 64) if model_config else 64,
        num_gcn_layers=getattr(model_config, 'num_gcn_layers', 2) if model_config else 2,
        num_transformer_layers=getattr(model_config, 'num_transformer_layers', 2) if model_config else 2,
        num_cst_layers=getattr(model_config, 'num_cst_layers', 2) if model_config else 2,
        num_heads=getattr(model_config, 'num_heads', 4) if model_config else 4,
        dropout=getattr(model_config, 'dropout', 0.3) if model_config else 0.3,
        semantic_adj=semantic_adj,
        system_channel_indices=system_channel_indices,
        use_variable_length=getattr(model_config, 'use_variable_length', True) if model_config else True,
        use_static_features=getattr(model_config, 'use_static_features', True) if model_config else True,
        static_dim=getattr(model_config, 'static_dim', 16) if model_config else 16,
        num_static_features=config.models.HDSTGCN.static_features.num_features if hasattr(config.models, 'HDSTGCN') else 5,
    )