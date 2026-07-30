"""
STFinalNet Baseline Model

Spatio-Temporal Fusion Network for CPET classification with:
- TFE Branch: ST-GCN (Graph Convolution + GRU + Attention)
- SFE Branch: Progressive Temporal Encoder + Star Transformer
- Variable-length sequences (via masking)
- Static feature fusion (EHR + PFT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# Helper Modules
# =============================================================================

class GraphConvolution(nn.Module):
    """Graph Convolution Layer"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.5,
        bias: bool = False,
        activation=F.relu
    ):
        super().__init__()
        self.dropout = dropout
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
            x: [B, C, F_in] node features
            adj: [C, C] adjacency matrix

        Returns:
            [B, C, F_out] transformed features
        """
        # 1. Node feature transform
        support = torch.matmul(x, self.weight)  # [B, C, F_out]

        # 2. Neighbor aggregation
        output = torch.matmul(adj, support)  # [B, C, F_out]

        if self.bias is not None:
            output = output + self.bias
        return self.activation(output)


class MultiAttention(nn.Module):
    """Multi-head Attention"""

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} not divisible by {num_attention_heads}")

        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, L, D = q.shape
        q = self.q_proj(q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class VariableEmbedding(nn.Module):
    """
    Variable Identity Embedding

    Assigns learnable semantic vectors to each feature channel.
    """

    def __init__(self, num_channels: int, embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_channels, embed_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        var_ids = torch.arange(self.num_channels, device=device)
        var_emb = self.embedding(var_ids)
        return self.dropout(var_emb.unsqueeze(0).expand(batch_size, -1, -1))


class DynamicGraphLayer(nn.Module):
    """
    Dynamic Graph Topology Layer

    Fuses semantic adjacency (medical prior) with data-driven attention.
    Formula: A_t = Softmax(QK^T + alpha * A_semantic)
    """

    def __init__(
        self,
        num_channels: int,
        embed_dim: int,
        semantic_adj: np.ndarray,
        num_heads: int = 2
    ):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.semantic_weight = nn.Parameter(torch.tensor(0.3))
        self.register_buffer('semantic_adj', torch.from_numpy(semantic_adj).float())

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, D] input features

        Returns:
            output: [B, C, D] transformed features
            A_dynamic: [B, C, C] dynamic adjacency
        """
        B, C, D = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (D ** 0.5)

        semantic_adj_batch = self.semantic_adj.unsqueeze(0).expand(B, -1, -1)
        combined = attn_scores + self.semantic_weight * semantic_adj_batch

        A_dynamic = F.softmax(combined, dim=-1)

        output = torch.bmm(A_dynamic, x)
        output = self.out_proj(output)
        output = self.layer_norm(output + x)

        return output, A_dynamic


class StarTransformerLayer(nn.Module):
    """Star Transformer Layer with group-based hub aggregation"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_prob: float,
        channel_groups
    ):
        super().__init__()
        self.channel_groups = channel_groups
        self.num_groups = len(channel_groups)

        all_indices = [idx for group in channel_groups for idx in group]
        self.num_channels = len(all_indices)

        self.register_buffer('group_mask', self._create_group_mask())

        self.agg_weights = nn.Parameter(torch.randn(self.num_channels, self.num_groups))
        nn.init.xavier_uniform_(self.agg_weights)

        self.multi_att = MultiAttention(hidden_size, num_heads, dropout_prob)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout_prob)
        )

    def _create_group_mask(self) -> torch.Tensor:
        mask = torch.full((self.num_channels, self.num_groups), float('-inf'))
        for gid, indices in enumerate(self.channel_groups):
            for cid in indices:
                if cid < self.num_channels:
                    mask[cid, gid] = 0.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D = x.shape

        masked_weights = self.agg_weights + self.group_mask
        masked_weights = F.softmax(masked_weights, dim=0)

        hubs = torch.matmul(x.transpose(1, 2), masked_weights.unsqueeze(0).expand(B, -1, -1)).transpose(1, 2)

        resid = hubs
        hubs = self.multi_att(hubs, hubs, hubs)
        hubs = self.ln1(hubs + resid)

        resid = hubs
        hubs = self.mlp(hubs)
        hubs = self.ln2(hubs + resid)

        return hubs


class SpatioTemporalGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network

    v2 improvements:
    - Reduced GRU output dimension to prevent over-smoothing
    - Added residual connections for stable training
    - Uses LayerNorm instead of BatchNorm
    """

    def __init__(self, num_channels: int, hidden_dim: int, num_nodes: int, dropout: float = 0.3):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        self.gcn = GraphConvolution(1, hidden_dim, dropout=dropout)

        self.temporal_rnn = nn.GRU(
            input_size=num_channels * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.residual_proj = nn.Linear(num_channels * hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x: [B, T, C] input
            adj: [C, C] adjacency matrix

        Returns:
            output: [B, hidden_dim]
            attn_weights: [B, T]
        """
        B, T, C = x.shape

        x_gcn_in = x.reshape(B * T, C, 1)
        x_gcn_out = self.gcn(x_gcn_in, adj)
        x_spatial = x_gcn_out.reshape(B, T, -1)

        x_temporal, _ = self.temporal_rnn(x_spatial)

        residual = self.residual_proj(x_spatial)
        x_temporal = x_temporal + residual
        x_temporal = self.layer_norm(x_temporal)
        x_temporal = self.dropout(x_temporal)

        attn_scores = self.attention(x_temporal)
        attn_weights = F.softmax(attn_scores, dim=1)
        x_out = (x_temporal * attn_weights).sum(dim=1)

        return x_out, attn_weights.squeeze(-1)


class TemporalEncoder(nn.Module):
    """
    Progressive Temporal Encoder

    Multi-stage strided convolution with variable embedding support.
    """

    def __init__(
        self,
        num_channels: int,
        input_len: int = 162,
        output_len: int = 24,
        var_embed_dim: int = 8,
        use_var_embedding: bool = True
    ):
        super().__init__()

        self.num_channels = num_channels
        self.output_len = output_len
        self.use_var_embedding = use_var_embedding
        self.var_embed_dim = var_embed_dim

        if use_var_embedding:
            self.var_embedding = VariableEmbedding(num_channels, var_embed_dim)
            self.var_fusion = nn.Linear(output_len + var_embed_dim, output_len)

        # Stage 1: input_len -> input_len/2
        self.stage1 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 5, padding=2, groups=num_channels),
            nn.Conv1d(num_channels, num_channels, 1),
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.AvgPool1d(2)
        )

        # Stage 2: input_len/2 -> input_len/4
        self.stage2 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 5, padding=2, groups=num_channels),
            nn.Conv1d(num_channels, num_channels, 1),
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.AvgPool1d(2)
        )

        # Stage 3: input_len/4 -> output_len
        self.stage3 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 3, padding=1, groups=num_channels),
            nn.Conv1d(num_channels, num_channels, 1),
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(output_len)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] input

        Returns:
            [B, C, output_len] encoded features
        """
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, T]

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)  # [B, C, output_len]

        if self.use_var_embedding:
            var_emb = self.var_embedding(B, x.device)
            x_with_var = torch.cat([x, var_emb], dim=-1)
            x = self.var_fusion(x_with_var)

        return x


# =============================================================================
# Main Model: STFinalNet
# =============================================================================

class STFinalNet(nn.Module):
    """
    Spatio-Temporal Fusion Network

    Two-branch architecture:
    - TFE (Temporal Feature Extraction): ST-GCN with GRU
    - SFE (Spatial Feature Extraction): Progressive encoder + Star Transformer

    Args:
        input_dim: Time steps (e.g., 162 for fixed, max_length for variable)
        hidden_dim: Base hidden dimension (default: 16)
        channel_groups: List of channel index groups
        output_dim: Number of output classes
        num_channel: Number of input channels
        num_patches: Input sequence length (default: 162, deprecated - use input_dim)
        ablation: Ablation mode - "both", "tfe_only", "sfe_only" (default: "both")
        use_var_embedding: Enable variable identity embedding (default: True)
        use_dynamic_graph: Enable dynamic graph topology (default: True)
        var_embed_dim: Variable embedding dimension (default: 8)
        semantic_adj: Semantic adjacency matrix (numpy array)
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
        num_patches: int = 162,
        ablation: str = "both",
        use_var_embedding: bool = True,
        use_dynamic_graph: bool = True,
        var_embed_dim: int = 8,
        semantic_adj=None,
        use_variable_length: bool = False,
        use_static_features: bool = False,
        static_dim: int = 16,
        num_static_features: int = 5,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()

        # --- Ablation config ---
        self.ablation = ablation

        # --- v3 new config ---
        self.use_var_embedding = use_var_embedding
        self.use_dynamic_graph = use_dynamic_graph
        self.var_embed_dim = var_embed_dim

        # --- Variable-length and static feature config ---
        self.use_variable_length = use_variable_length
        self.use_static_features = use_static_features
        self.static_dim = static_dim

        # Use input_dim as the actual sequence length
        self.input_len = input_dim if input_dim != num_patches else num_patches

        # --- Optimized config ---
        self.tfe_hidden = 16
        self.sfe_spatial_dim = 24
        self.fusion_dim = 48
        self.dropout = dropout

        self.channel_groups = channel_groups
        self.num_groups = len(channel_groups)
        self.num_channel = num_channel

        # TFE branch output dimension
        self.tfe_dim = self.tfe_hidden

        # --- Adjacency matrix ---
        self.adj_learner = nn.Parameter(torch.randn(num_channel, num_channel))
        self.adj_initialized = False

        # --- TFE Branch (ST-GCN) ---
        self.st_gcn = SpatioTemporalGCN(
            num_channels=num_channel,
            hidden_dim=self.tfe_hidden,
            num_nodes=num_channel,
            dropout=dropout
        )

        self.tfe_proj = nn.Sequential(
            nn.Linear(self.tfe_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- SFE Branch (Progressive Encoder v3) ---
        self.sfe_encoder = TemporalEncoder(
            num_channels=num_channel,
            input_len=self.input_len,
            output_len=self.sfe_spatial_dim,
            var_embed_dim=var_embed_dim,
            use_var_embedding=use_var_embedding
        )

        # --- Dynamic Graph Layer ---
        if use_dynamic_graph and semantic_adj is not None:
            self.dynamic_graph = DynamicGraphLayer(
                num_channels=num_channel,
                embed_dim=self.sfe_spatial_dim,
                semantic_adj=semantic_adj,
                num_heads=2
            )
        else:
            self.dynamic_graph = None

        # Star Transformer
        self.star_trans = StarTransformerLayer(
            self.sfe_spatial_dim,
            num_heads=2,
            dropout_prob=dropout,
            channel_groups=channel_groups
        )

        # SFE projection
        self.sfe_flat_dim = self.num_groups * self.sfe_spatial_dim
        self.sfe_proj = nn.Sequential(
            nn.Linear(self.sfe_flat_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Static Feature Encoder ---
        if use_static_features:
            from .common import StaticFeatureEncoder
            self.static_encoder = StaticFeatureEncoder(num_static_features, static_dim, dropout)
            fusion_input_dim = self.fusion_dim * 2 + static_dim
        else:
            self.static_encoder = None
            fusion_input_dim = self.fusion_dim * 2

        # --- Fusion Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_dim, output_dim)
        )

        # --- Ablation classifiers ---
        self.classifier_tfe = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )
        self.classifier_sfe = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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
            x: [B, T, C] input tensor
            lengths: [B] actual sequence lengths (for variable-length mode)
            prior_adj: [C, C] semantic adjacency matrix
            static_x: [B, num_static_features] static features

        Returns:
            [B, output_dim] logits
        """
        B, T, C = x.shape

        # Variable-length masking
        if self.use_variable_length and lengths is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            x = x * mask.unsqueeze(-1).float()

        # --- 1. Adjacency ---
        if not self.adj_initialized and prior_adj is not None:
            with torch.no_grad():
                self.adj_learner.data.copy_(prior_adj)
            self.adj_initialized = True

        if prior_adj is not None:
            curr_adj = prior_adj.to(x.device) + self.adj_learner
        else:
            curr_adj = self.adj_learner
        curr_adj = F.relu(curr_adj)

        degree = curr_adj.sum(1) + 1e-6
        d_inv = torch.pow(degree, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat = torch.diag(d_inv)
        norm_adj = d_mat @ curr_adj @ d_mat

        # --- 2. TFE Branch (ST-GCN) ---
        tfe_feat, _ = self.st_gcn(x, norm_adj)
        tfe_feat = self.tfe_proj(tfe_feat)

        # --- 3. SFE Branch ---
        x_sfe = self.sfe_encoder(x)

        if self.dynamic_graph is not None:
            x_sfe, _ = self.dynamic_graph(x_sfe)

        hubs = self.star_trans(x_sfe)

        sfe_flat = hubs.reshape(B, -1)
        sfe_feat = self.sfe_proj(sfe_flat)

        # --- 4. Fusion ---
        tfe_feat_norm = F.normalize(tfe_feat, p=2, dim=1)
        sfe_feat_norm = F.normalize(sfe_feat, p=2, dim=1)

        if self.ablation == "tfe_only":
            logits = self.classifier_tfe(tfe_feat_norm)
        elif self.ablation == "sfe_only":
            logits = self.classifier_sfe(sfe_feat_norm)
        else:  # "both"
            combined = torch.cat([tfe_feat_norm, sfe_feat_norm], dim=1)

            # Static feature fusion
            if self.use_static_features and static_x is not None and self.static_encoder is not None:
                feat_stat = self.static_encoder(static_x)
                combined = torch.cat([combined, feat_stat], dim=1)

            logits = self.classifier(combined)

        return logits