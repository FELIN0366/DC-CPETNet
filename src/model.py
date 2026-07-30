import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Transformer
import math
from einops import repeat
import os

# ==========================================
# 辅助模块 (保持精简)
# ==========================================

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, bias=False, activation=F.relu):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None
        
        # Xavier 初始化
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x: [B, C, F_in]
        # adj: [C, C]
        # [优化] 移除内部 dropout，防止破坏图结构信息传递
        # Dropout 应该放在 GCN 层外部，而不是节点特征变换前

        # 1. 节点特征变换: [B, C, F_in] * [F_in, F_out] -> [B, C, F_out]
        support = torch.matmul(x, self.weight)
        
        # 2. 邻居信息聚合: [C, C] * [B, C, F_out] -> [B, C, F_out]
        # 注意: 这里让 adj 广播到 Batch 维度
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        return self.activation(output)


class ChannelAttention(nn.Module):
    """
    通道注意力模块 (Channel Attention Module)

    为每个通道学习一个权重，突出有效特征，抑制噪声特征。
    使用 Softmax 归一化确保权重总和为 1。

    Args:
        num_channels: 通道数量 (26)
        init_value: 初始权重值 (默认 1.0，即所有通道等权)
    """
    def __init__(self, num_channels, init_value=1.0):
        super().__init__()
        self.num_channels = num_channels
        # 可学习的通道权重参数
        self.weights = nn.Parameter(torch.ones(num_channels) * init_value)

    def forward(self, x):
        """
        Args:
            x: [B, C, D] 通道特征

        Returns:
            weighted: [B, C, D] 加权后的特征
        """
        # Softmax 归一化权重 (确保总和为 1)
        weights = F.softmax(self.weights, dim=0)

        # 应用权重: [B, C, D] * [C, 1] -> [B, C, D]
        return x * weights.view(1, -1, 1)

    def get_weights(self):
        """
        获取当前通道权重 (用于 SwanLab 记录)

        Returns:
            weights: [C] 归一化后的权重
        """
        return F.softmax(self.weights, dim=0).detach()


class MultiAttention(nn.Module):
    # 标准多头注意力，保持不变但移除不必要的复杂性
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(MultiAttention, self).__init__()
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

    def forward(self, q, k, v):
        B, L, D = q.shape
        # [B, L, Heads, Dim]
        q = self.q_proj(q).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Combine
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

class VariableEmbedding(nn.Module):
    """
    变量身份嵌入 (Variable Identity Embedding)

    为每个特征通道分配可学习的语义向量，使模型能区分 HR 和 VO2 等不同变量
    这解决了 Transformer 无法区分不同特征通道的问题

    Args:
        num_channels: 特征通道数 (22)
        embed_dim: 嵌入维度 (默认 8)
        dropout: Dropout 率
    """
    def __init__(self, num_channels, embed_dim=8, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_channels, embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 小方差初始化，避免破坏预训练权重
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, batch_size, device):
        """
        生成变量嵌入

        Returns:
            var_emb: [B, C, embed_dim] 变量嵌入矩阵
        """
        var_ids = torch.arange(self.num_channels, device=device)
        var_emb = self.embedding(var_ids)  # [C, embed_dim]
        # 扩展 batch 维度并应用 dropout
        return self.dropout(var_emb.unsqueeze(0).expand(batch_size, -1, -1))


class DynamicGraphLayer(nn.Module):
    """
    动态图拓扑层 (Dynamic Graph Topology Layer)

    融合语义邻接矩阵 (医学先验) 和数据驱动的动态注意力图
    公式: A_t = Softmax(QK^T + alpha * A_semantic)

    Args:
        num_channels: 节点(特征)数量
        embed_dim: 特征维度
        semantic_adj: 语义邻接矩阵 [num_channels, num_channels]
        num_heads: 注意力头数 (默认 2)
    """
    def __init__(self, num_channels, embed_dim, semantic_adj, num_heads=2):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        # Q/K 投影用于动态图构建
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        # 语义邻接矩阵权重 (可学习)
        self.semantic_weight = nn.Parameter(torch.tensor(0.3))

        # 注册语义邻接矩阵为 buffer (不参与梯度更新)
        self.register_buffer('semantic_adj', torch.from_numpy(semantic_adj).float())

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: [B, C, D] 输入特征 (C: 通道数, D: 特征维度)

        Returns:
            output: [B, C, D] 图卷积后的特征
            A_dynamic: [B, C, C] 动态邻接矩阵
        """
        B, C, D = x.shape

        # 1. 计算 Q, K
        Q = self.q_proj(x)  # [B, C, D]
        K = self.k_proj(x)  # [B, C, D]

        # 2. 计算注意力分数: QK^T / sqrt(D)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (D ** 0.5)  # [B, C, C]

        # 3. 融合语义邻接矩阵
        # 扩展语义矩阵到 batch 维度
        semantic_adj_batch = self.semantic_adj.unsqueeze(0).expand(B, -1, -1)  # [B, C, C]
        combined = attn_scores + self.semantic_weight * semantic_adj_batch  # [B, C, C]

        # 4. Softmax 得到动态邻接矩阵 (行和为 1)
        A_dynamic = F.softmax(combined, dim=-1)  # [B, C, C]

        # 5. 图卷积: A @ X
        output = torch.bmm(A_dynamic, x)  # [B, C, D]
        output = self.out_proj(output)
        output = self.layer_norm(output + x)  # 残差连接

        return output, A_dynamic


class StarTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_prob, channel_groups):
        super().__init__()
        self.channel_groups = channel_groups
        self.num_groups = len(channel_groups)
        
        # 计算总通道数
        all_indices = [idx for group in channel_groups for idx in group]
        self.num_channels = len(all_indices)
        
        # 组掩码 (Buffer: 不更新)
        self.register_buffer('group_mask', self._create_group_mask())
        
        # 可学习的聚合权重 (初始化为均匀分布)
        self.agg_weights = nn.Parameter(torch.randn(self.num_channels, self.num_groups))
        nn.init.xavier_uniform_(self.agg_weights)

        # 注意力层
        self.multi_att = MultiAttention(hidden_size, num_heads, dropout_prob)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), # 缩小 FFN 比例 (4 -> 2)
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout_prob)
        )

    def _create_group_mask(self):
        """
        创建加法掩码 (Additive Mask for Softmax)

        使用 -inf 填充无效位置，确保 softmax 后这些位置概率为 0
        这解决了乘法掩码 (e^0 = 1) 导致的权重稀释问题
        """
        mask = torch.full((self.num_channels, self.num_groups), float('-inf'))
        for gid, indices in enumerate(self.channel_groups):
            for cid in indices:
                if cid < self.num_channels:
                    mask[cid, gid] = 0.0  # 有效位置设为 0，softmax 后正常计算
        return mask

    def forward(self, x):
        # x: [B, C, D]
        B, C, D = x.shape
        
        # 1. 聚合生成 Hubs (Satellite -> Hub)
        # 使用加法掩码: -inf 位置在 softmax 后概率为 0
        # 这解决了乘法掩码 (e^0 = 1) 导致有效权重被稀释的问题
        masked_weights = self.agg_weights + self.group_mask  # 加法掩码
        masked_weights = F.softmax(masked_weights, dim=0)    # -inf -> 0 概率 
        
        # [B, D, C] @ [B, C, G] -> [B, D, G] -> [B, G, D]
        hubs = torch.matmul(x.transpose(1, 2), masked_weights.unsqueeze(0).expand(B, -1, -1)).transpose(1, 2)
        
        # 2. Hub 内部交互 (Global Context)
        resid = hubs
        hubs = self.multi_att(hubs, hubs, hubs)
        hubs = self.ln1(hubs + resid)
        
        # 3. FFN
        resid = hubs
        hubs = self.mlp(hubs)
        hubs = self.ln2(hubs + resid)
        
        return hubs # [B, G, D]

class SpatioTemporalGCN(nn.Module):
    """
    时空图卷积：先空间聚合，再时序建模
    替代原有的统计池化，保留时序动态演变信息

    v2 修复：
    - 减少GRU输出维度，防止过度平滑
    - 添加残差连接，稳定训练
    - 使用LayerNorm替代BatchNorm
    """
    def __init__(self, num_channels, hidden_dim, num_nodes, dropout=0.3):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        # 1. 空间维度：图卷积
        self.gcn = GraphConvolution(1, hidden_dim, dropout=dropout)

        # 2. 时序维度：单层GRU，输出维度减小
        self.temporal_rnn = nn.GRU(
            input_size=num_channels * hidden_dim,
            hidden_size=hidden_dim,  # 减小输出维度
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 3. 注意力池化
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 4. 残差投影：将GRU输入投影到输出维度
        self.residual_proj = nn.Linear(num_channels * hidden_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: [B, T, C]
        B, T, C = x.shape

        # 1. 空间图卷积 (每个时间点独立)
        x_gcn_in = x.reshape(B * T, C, 1)
        x_gcn_out = self.gcn(x_gcn_in, adj)  # [B*T, C, hidden]
        x_spatial = x_gcn_out.reshape(B, T, -1)  # [B, T, C*hidden]

        # 2. 时序建模
        x_temporal, _ = self.temporal_rnn(x_spatial)  # [B, T, hidden]

        # 3. 残差连接
        residual = self.residual_proj(x_spatial)  # [B, T, hidden]
        x_temporal = x_temporal + residual  # 残差连接
        x_temporal = self.layer_norm(x_temporal)
        x_temporal = self.dropout(x_temporal)

        # 4. 注意力池化
        attn_scores = self.attention(x_temporal)  # [B, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        x_out = (x_temporal * attn_weights).sum(dim=1)  # [B, hidden]

        return x_out, attn_weights.squeeze(-1)


class TemporalEncoder(nn.Module):
    """
    渐进式时序编码：多层步长卷积
    替代单次 AdaptiveAvgPool，保留更多局部时序细节
    输出: [B, num_channels, output_len] 兼容 StarTransformerLayer

    v2 改进：添加跨通道交互 (Pointwise Conv)，捕捉 VO2-HR 等医学关联
    v3 改进：支持变量嵌入融合
    """
    def __init__(self, num_channels, input_len=162, output_len=24, var_embed_dim=8, use_var_embedding=True):
        super().__init__()

        self.num_channels = num_channels
        self.output_len = output_len
        self.use_var_embedding = use_var_embedding
        self.var_embed_dim = var_embed_dim

        # 变量嵌入模块
        if use_var_embedding:
            self.var_embedding = VariableEmbedding(num_channels, var_embed_dim)
            # 融合投影: [B, C, output_len + var_embed_dim] -> [B, C, output_len]
            self.var_fusion = nn.Linear(output_len + var_embed_dim, output_len)

        # Stage 1: 162 -> 81 (Depthwise Separable Conv)
        self.stage1 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 5, padding=2, groups=num_channels),  # Depthwise
            nn.Conv1d(num_channels, num_channels, 1),  # Pointwise: 跨通道交互
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.AvgPool1d(2)
        )

        # Stage 2: 81 -> 40 (Depthwise Separable Conv)
        self.stage2 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 5, padding=2, groups=num_channels),
            nn.Conv1d(num_channels, num_channels, 1),  # Pointwise
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.AvgPool1d(2)
        )

        # Stage 3: 40 -> output_len (Adaptive)
        self.stage3 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 3, padding=1, groups=num_channels),
            nn.Conv1d(num_channels, num_channels, 1),  # Pointwise
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

    def forward(self, x):
        # x: [B, T=162, C]
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, T]

        x = self.stage1(x)  # [B, C, 81]
        x = self.stage2(x)  # [B, C, 40]
        x = self.stage3(x)  # [B, C, output_len]

        # 变量嵌入融合
        if self.use_var_embedding:
            # 获取变量嵌入 [B, C, var_embed_dim]
            var_emb = self.var_embedding(B, x.device)
            # 拼接: [B, C, output_len] + [B, C, var_embed_dim] -> [B, C, output_len + var_embed_dim]
            x_with_var = torch.cat([x, var_emb], dim=-1)
            # 投影回原维度: [B, C, output_len]
            x = self.var_fusion(x_with_var)

        # 输出: [B, C, output_len] 直接返回，适配 StarTransformerLayer
        return x
# ==========================================
# 主模型 STFinalNet (轻量化版)
# ==========================================

# model.py (请替换整个 STFinalNet 类)

class STFinalNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, channel_groups, output_dim, num_channel,
                 num_patches=162, ablation="both",
                 use_var_embedding=True, use_dynamic_graph=True, var_embed_dim=8,
                 semantic_adj=None):
        super(STFinalNet, self).__init__()

        # --- [消融实验配置] ---
        self.ablation = ablation  # "both", "tfe_only", "sfe_only"

        # --- [v3 新增配置] ---
        self.use_var_embedding = use_var_embedding
        self.use_dynamic_graph = use_dynamic_graph
        self.var_embed_dim = var_embed_dim

        # --- [优化后配置] ---
        self.tfe_hidden = 16         # GCN hidden dim (增加维度)
        self.sfe_spatial_dim = 24    # SFE output length
        self.fusion_dim = 48         # Fusion dimension
        self.dropout = 0.3           # Dropout rate (降低)

        self.channel_groups = channel_groups
        self.num_groups = len(channel_groups)
        self.num_channel = num_channel

        # TFE 分支输出维度: ST-GCN v2 输出 hidden_dim
        self.tfe_dim = self.tfe_hidden  # 16

        # --- 邻接矩阵 (保持不变) ---
        self.adj_learner = nn.Parameter(torch.randn(num_channel, num_channel))
        self.adj_initialized = False

        # --- TFE 分支 (ST-GCN 版本) ---
        self.st_gcn = SpatioTemporalGCN(
            num_channels=num_channel,
            hidden_dim=self.tfe_hidden,
            num_nodes=num_channel,
            dropout=self.dropout
        )

        # TFE 投影层
        self.tfe_proj = nn.Sequential(
            nn.Linear(self.tfe_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # --- SFE 分支 (渐进降维版 v3) ---
        self.sfe_encoder = TemporalEncoder(
            num_channels=num_channel,
            input_len=num_patches,
            output_len=self.sfe_spatial_dim,
            var_embed_dim=var_embed_dim,
            use_var_embedding=use_var_embedding
        )

        # --- [v3 新增] 动态图拓扑层 ---
        if use_dynamic_graph and semantic_adj is not None:
            self.dynamic_graph = DynamicGraphLayer(
                num_channels=num_channel,
                embed_dim=self.sfe_spatial_dim,
                semantic_adj=semantic_adj,
                num_heads=2
            )
        else:
            self.dynamic_graph = None

        # StarTransformer 输入维度 = sfe_encoder.out_channels
        self.star_trans = StarTransformerLayer(
            self.sfe_spatial_dim,
            num_heads=2,
            dropout_prob=self.dropout,
            channel_groups=channel_groups
        )

        # SFE 投影：StarTransformer 输出 [B, G, sfe_spatial_dim]
        self.sfe_flat_dim = self.num_groups * self.sfe_spatial_dim
        self.sfe_proj = nn.Sequential(
            nn.Linear(self.sfe_flat_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # --- Fusion Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_dim, output_dim)
        )

        # --- 消融实验专用分类器 (加深版本) ---
        self.classifier_tfe = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, output_dim)
        )
        self.classifier_sfe = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
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

    def forward(self, x, prior_adj):
        B, T, C = x.shape

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
        # x: [B, T, C]
        tfe_feat, tfe_attn = self.st_gcn(x, norm_adj)  # [B, hidden*2]
        tfe_feat = self.tfe_proj(tfe_feat)  # [B, fusion_dim]

        # --- 3. SFE Branch (渐进降维 v3) ---
        x_sfe = self.sfe_encoder(x)  # [B, C, output_len] = [B, num_channel, 24]

        # [v3 新增] 动态图拓扑融合
        if self.dynamic_graph is not None:
            x_sfe, _ = self.dynamic_graph(x_sfe)  # [B, C, output_len]

        hubs = self.star_trans(x_sfe)  # [B, G, sfe_spatial_dim] = [B, num_groups, 24]

        sfe_flat = hubs.reshape(B, -1)
        sfe_feat = self.sfe_proj(sfe_flat)

        # --- 4. Fusion (消融实验支持 + L2归一化) ---
        if self.ablation == "tfe_only":
            tfe_feat_norm = F.normalize(tfe_feat, p=2, dim=1)
            logits = self.classifier_tfe(tfe_feat_norm)
        elif self.ablation == "sfe_only":
            sfe_feat_norm = F.normalize(sfe_feat, p=2, dim=1)
            logits = self.classifier_sfe(sfe_feat_norm)
        else:  # "both"
            tfe_feat_norm = F.normalize(tfe_feat, p=2, dim=1)
            sfe_feat_norm = F.normalize(sfe_feat, p=2, dim=1)
            combined = torch.cat([tfe_feat_norm, sfe_feat_norm], dim=1)
            logits = self.classifier(combined)

        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------------------------
# 辅助模块
# ----------------------------------------------------------------------
class ResBlock1D(nn.Module):
    """1D残差块，用于捕捉波形特征 (如 O2 Pulse 平台期)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.3):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# ----------------------------------------------------------------------
# 主模型：Medical-Informed Net
# ----------------------------------------------------------------------
class MedNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, channel_groups, output_dim, num_channel, num_patches=162, ablation=None, **kwargs):
        super(MedNet, self).__init__()
        
        self.num_channel = num_channel
        
        # --- 通路 A: Shape Stream (CNN) ---
        # 负责：缺血(平台期)、心衰(振荡呼吸)
        # 逻辑：ResNet 结构能很好地提取局部波形特征
        self.shape_encoder = nn.Sequential(
            ResBlock1D(num_channel, 32, kernel_size=7, stride=2), # [B, 32, 81]
            ResBlock1D(32, 64, kernel_size=5, stride=2),        # [B, 64, 40]
            ResBlock1D(64, 128, kernel_size=3, stride=2),       # [B, 128, 20]
            nn.AdaptiveAvgPool1d(1)                             # [B, 128, 1]
        )
        
        # --- 通路 B: Stat Stream (Global Features) ---
        # 负责：肺血管(高Slope)、通气受限(低Reserve)
        # 逻辑：这些病变主要体现在数值的极值和比例上，不需要复杂的时序
        self.stat_dim = num_channel * 3 # Mean, Std, Max
        self.stat_encoder = nn.Sequential(
            nn.Linear(self.stat_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # --- 通路 C: Rhythm Stream (Arrhythmia Specialist) ---
        # 负责：心律失常
        # 逻辑：专门针对 HR (假设 HR 是第 3 列，根据你的 feature_mapping)
        # 计算一阶差分 (Velocity) 和二阶差分 (Acceleration) 的方差
        self.hr_idx = 3 # 假设 HR 在第 3 列，请根据实际 feature_mapping 确认！
        self.rhythm_encoder = nn.Sequential(
            nn.Linear(4, 16), # 输入: [HR_std, HR_diff_std, HR_diff2_std, HR_range]
            nn.ReLU()
        )
        
        # --- Fusion ---
        self.fusion_dim = 128 + 64 + 16
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x, prior_adj=None):
        # x: [B, 162, 22]
        B, T, C = x.shape
        
        # Permute for CNN: [B, 22, 162]
        x_cnn = x.permute(0, 2, 1)
        
        # --- Stream A: Shape (CNN) ---
        feat_shape = self.shape_encoder(x_cnn).squeeze(-1) # [B, 128]
        
        # --- Stream B: Stat ---
        # Mean, Std, Max
        f_mean = x.mean(dim=1)
        f_std = x.std(dim=1)
        f_max, _ = x.max(dim=1)
        x_stat = torch.cat([f_mean, f_std, f_max], dim=1) # [B, 66]
        feat_stat = self.stat_encoder(x_stat) # [B, 64]
        
        # --- Stream C: Rhythm (HR Focus) ---
        # 提取 HR 通道 (假设 index 3)
        # 如果你的 feature_mapping 中 HR 不是 3，请修改 self.hr_idx
        hr_data = x[:, :, self.hr_idx] # [B, 162]

        # 1. 基础波动
        hr_std = hr_data.std(dim=1, keepdim=True)
        hr_range = (hr_data.max(dim=1)[0] - hr_data.min(dim=1)[0]).unsqueeze(1)
        
        # 2. 一阶差分 (瞬时变化率) -> 捕捉早搏
        hr_diff = hr_data[:, 1:] - hr_data[:, :-1]
        hr_diff_std = hr_diff.std(dim=1, keepdim=True)
        
        # 3. 二阶差分 (变化率的变化) -> 捕捉震荡
        hr_diff2 = hr_diff[:, 1:] - hr_diff[:, :-1]
        hr_diff2_std = hr_diff2.std(dim=1, keepdim=True)
        
        feat_rhythm = torch.cat([hr_std, hr_diff_std, hr_diff2_std, hr_range], dim=1) # [B, 4]
        feat_rhythm = self.rhythm_encoder(feat_rhythm) # [B, 16]
        
        # --- Fusion ---
        combined = torch.cat([feat_shape, feat_stat, feat_rhythm], dim=1) # [B, 208]
        logits = self.classifier(combined)
        
        return logits
class LSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_channel, hidden_dim=64, num_layers=2, ablation=None, **kwargs):
        """
        标准 LSTM 用于时间序列分类
        input_dim: 时间步长 (162) - LSTM 初始化其实不需要这个，但为了接口统一保留
        num_channel: 特征数 (22) - 对应 LSTM 的 input_size
        output_dim: 类别数 (6)
        """
        super(LSTMNet, self).__init__()
        
        # LSTM 层
        # batch_first=True 意味着输入格式为 [Batch, Time, Channel]
        self.lstm = nn.LSTM(
            input_size=num_channel, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # 全连接层
        # 因为是双向 LSTM，所以输出维度是 hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, adj=None):
        # x: [Batch, Time=162, Channel=22]
        # LSTM 天然支持这种格式，不需要 permute
        
        # out: [Batch, Time, Hidden*2]
        # _ : (h_n, c_n)
        out, _ = self.lstm(x)
        
        # 聚合策略：取最后一个时间步，或者全局平均池化
        # 这里使用全局平均池化 (Mean Pooling)，对噪声更鲁棒
        out = torch.mean(out, dim=1)  # -> [Batch, Hidden*2]
        
        # 分类
        logits = self.fc(out)
        return logits

# ResNet 的基础残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
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
    def __init__(self, input_dim, output_dim, num_channel, hidden_dim=64, ablation=None, **kwargs):
        """
        1D ResNet 适配 CPET 数据
        """
        super(ResNet1D, self).__init__()
        
        self.inplanes = 64
        
        # 1. 初始卷积层 (Stem)
        self.conv1 = nn.Conv1d(num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 2. 残差层堆叠 (这里构建一个轻量级的 ResNet-18 风格)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        # 如果数据量大，可以继续加 layer3 (256), layer4 (512)
        
        # 3. 输出层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim) # 128 是 layer2 的输出通道数

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        # 如果步长不为1或者通道数改变，需要下采样调整残差边的维度
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

    def forward(self, x, adj=None):
        # x: [Batch, Time, Channel] -> 需要转为 [Batch, Channel, Time]
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def check_tensor_health(name, tensor):
    """
    检查张量的生命体征：
    1. 是否有 NaN/Inf (猝死)
    2. 方差是否过低 (脑死亡/模式坍塌)
    3. 数值范围是否爆炸 (梯度爆炸前兆)
    """
    if tensor is None:
        return
        
    with torch.no_grad():
        # 基础统计
        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        has_nan = torch.isnan(tensor).any().item()
        
        # 格式化日志
        log_msg = f"[{name}] Shape:{list(tensor.shape)} | Mean:{mean:.4f} | Std:{std:.4f} | Range:[{min_val:.2f}, {max_val:.2f}]"
        
        # 警报逻辑
        alerts = []
        if has_nan:
            alerts.append("!!! NaN DETECTED !!!")
        if std < 1e-4:
            alerts.append("!!! DEAD NEURON / COLLAPSE (Std < 1e-4) !!!")
        if abs(mean) > 100 or abs(max_val) > 100:
            alerts.append("!!! VALUE EXPLOSION (>100) !!!")
            
        if alerts:
            log_msg += " " + " ".join(alerts)
            
        # 只有在有警报或每隔一定步数（外部控制，这里简化为总是）写入
        # 为了不撑爆硬盘，建议仅在 std 异常或 nan 时写入，或者手动控制频率
        if alerts: 
            write_debug_log(log_msg)
            print(f"警报: {log_msg}") # 控制台也打印重要警报

def write_debug_log(msg):
    with open("debug_model_checks.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

if __name__ == '__main__':
    # 测试代码
    device = torch.device("cpu")
    # 模拟 22 个通道，分为 4 组
    groups = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13], [14,15,16,17,18,19,20,21]]
    model = STFinalNet(input_dim=162, hidden_dim=None, channel_groups=groups, output_dim=6, num_channel=22)

    x = torch.randn(5, 162, 22)
    adj = torch.eye(22) # 模拟先验

    out = model(x, adj)
    print(f"Output shape: {out.shape}")
    print(f"Total Params: {count_parameters(model)}")
    # 预期参数量应在 50k - 100k 之间，远小于原来的 500k


# ==========================================================================
# HDSTGCN: Hierarchical Dynamic Spatio-Temporal Graph Convolutional Network
# ==========================================================================
# 设计理念:
#   阶段一 (TemporalEncodingBranch): 单变量独立时序编码
#     - 每个通道独立处理，避免跨通道混淆
#     - 共享GRU权重，参数高效
#     - 变长序列支持 (pack_padded_sequence)
#   阶段二 (HierarchicalSpatialGraph): 多层级图聚合
#     - 子图GCN: 按医学先验分组聚合
#     - 全局GCN: 捕捉子系统间交互
# ==========================================================================


class SubGraphLayer(nn.Module):
    """
    子图图卷积层 (Sub-Graph Layer) - 双轨拓扑融合版

    单个子系统(如代谢、循环)内部的图卷积操作
    包含残差连接防止过度平滑

    双轨机制:
    - 轨道1: 医学先验 (register_buffer, 固定不更新)
    - 轨道2: 数据驱动微调 (nn.Parameter, 可学习)
    - 门控融合: 可学习 alpha 决定融合权重

    Args:
        num_nodes: 子图中的节点数
        hidden_dim: 特征维度
        dropout: Dropout 率
        prior_adj: 医学先验邻接矩阵子图
    """
    def __init__(self, num_nodes, hidden_dim, dropout=0.3, prior_adj=None):
        super().__init__()
        self.num_nodes = num_nodes

        # 图卷积参数
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # 残差投影
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ================== 双轨拓扑机制 ==================
        if prior_adj is not None:
            # 轨道1: 医学先验 (固定知识，不参与梯度更新，使用 register_buffer)
            self.register_buffer('prior_adj', prior_adj.clone().detach())
        else:
            # 兜底：如果没有先验，退化为自连接
            self.register_buffer('prior_adj', torch.eye(num_nodes))

        # 轨道2: 数据驱动微调 (可学习矩阵，初始化为很小的值，避免开局破坏先验)
        self.learned_adj = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)

        # 融合门控: 让模型自己决定多大程度上信任医学先验
        # 初始化为 0，经过 sigmoid 后是 0.5，表示起步时对半开，平滑过渡
        self.alpha = nn.Parameter(torch.tensor(0.0))
        # ==================================================

        # 初始化
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        Args:
            x: [B, num_nodes, hidden_dim] 子图节点特征

        Returns:
            out: [B, num_nodes, hidden_dim] 更新后的节点特征
        """
        B, N, D = x.shape

        # 1. 动态融合邻接矩阵
        # sigmoid 保证权重在 0-1 之间
        w_prior = torch.sigmoid(self.alpha)

        # 核心逻辑：加权融合 先验图 + 学习图
        raw_adj = w_prior * self.prior_adj + (1 - w_prior) * self.learned_adj

        # 确保对角线自连接保底，并使用 ReLU 抹平负相关噪音
        raw_adj = F.relu(raw_adj + torch.eye(self.num_nodes, device=x.device))

        # 行归一化 (对称归一化的平替，更适合有向/非对称图学习)
        adj_soft = F.softmax(raw_adj, dim=-1)

        # 2. 图卷积: A @ X @ W
        support = torch.matmul(x, self.weight)  # [B, N, D]
        output = torch.matmul(adj_soft.unsqueeze(0).expand(B, -1, -1), support)

        # 3. 残差连接 + LayerNorm
        residual = self.residual_proj(x)
        output = self.layer_norm(output + residual)

        return F.relu(output)


class GlobalGraphLayer(nn.Module):
    """
    全局图卷积层 (Global Graph Layer)

    对宏观节点(hub)进行全局交互
    使用注意力机制捕捉子系统间的动态关系

    Args:
        hidden_dim: 特征维度
        num_heads: 注意力头数
    """
    def __init__(self, hidden_dim, num_heads=2, dropout=0.3):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: [B, num_hubs, hidden_dim] 全局节点特征

        Returns:
            out: [B, num_hubs, hidden_dim] 更新后的全局特征
        """
        # 自注意力
        residual = x
        attn_out, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm1(residual + attn_out)

        # FFN
        residual = x
        x = self.layer_norm2(residual + self.ffn(x))

        return x


class PriorMaskedGlobalTransformer(nn.Module):
    """
    基于先验掩码的全局 Transformer (Prior-Masked Global Transformer)

    核心公式:
        A_dynamic = Softmax(QK^T / sqrt(D) + gamma * M_prior + log(W_attn)) * V

    其中:
        - Q, K, V: 标准 Transformer 的 Query, Key, Value
        - M_prior: 医学先验偏置矩阵
          - 有连接的节点对: 正偏置 (允许注意力)
          - 无连接的节点对: 大负数 (近似屏蔽注意力)
        - gamma: 可学习的门控标量，控制先验的影响强度
        - W_attn: [新增] 初始注意力权重预设矩阵 (九图核心斜率)

    特点:
        - 22 个特征永远是 22 个独立特征，不做 Mean Pooling
        - 医学先验通过 M_prior 注入注意力矩阵
        - 数据驱动的注意力 + 医学知识的软约束
        - [新增] 核心斜率权重预设，强化关键生理关系

    PriorWarmStart 改进:
        - gamma_init: 初始值 (默认 1.0，热启动)
        - gamma_min: 下限约束 (默认 0.1，保护先验贡献)
        - 使用 softplus 保证 gamma > 0

    Args:
        hidden_dim: 特征维度 (D_time)
        num_heads: 注意力头数
        num_nodes: 节点数量 (22)
        semantic_adj: 语义邻接矩阵 [num_nodes, num_nodes], 1=有连接, 0=无连接
        dropout: Dropout 率
        prior_bias: 有连接时的偏置值 (默认 0.0)
        mask_value: 无连接时的屏蔽值 (默认 -1e9，不使用 -inf 避免 NaN 梯度)
        gamma_init: gamma 初始值 (默认 1.0，热启动)
        gamma_min: gamma 下限 (默认 0.1)
        attention_weights: [新增] 初始注意力权重预设矩阵 [N, N] (九图核心斜率)
    """
    def __init__(self, hidden_dim, num_heads=2, num_nodes=22, semantic_adj=None,
                 dropout=0.3, prior_bias=0.0, mask_value=-1e9,
                 gamma_init=1.0, gamma_min=0.1, attention_weights=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.prior_bias = prior_bias
        self.mask_value = mask_value
        self.gamma_min = gamma_min

        # 多头注意力 (不使用 PyTorch 内置的 mask，我们自己实现)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        # Q, K, V 投影
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # 可学习的门控参数 gamma
        # [PriorWarmStart] 使用正初始值 (热启动) + softplus 保证正数 + 下限约束
        # 初始化为 gamma_init 的 softplus 逆变换，使得 softplus(gamma_raw) ≈ gamma_init
        gamma_raw_init = math.log(math.exp(gamma_init) - 1) if gamma_init > 0 else 0.0
        self.gamma_raw = nn.Parameter(torch.tensor(gamma_raw_init))

        # 构建先验偏置矩阵 M_prior
        if semantic_adj is not None:
            # semantic_adj: [N, N], 值为 0 或 1
            if not isinstance(semantic_adj, torch.Tensor):
                semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32)

            # M_prior:
            #   - 有连接 (adj=1): prior_bias (允许注意力)
            #   - 无连接 (adj=0): mask_value (大负数，近似屏蔽)
            #   - 对角线 (self-loop): prior_bias (允许自注意力)
            # 注意: 使用大负数而非 -inf，避免梯度 NaN 问题
            M_prior = torch.where(
                semantic_adj > 0,
                torch.tensor(prior_bias),
                torch.tensor(mask_value)
            )
            # 确保对角线有正偏置 (自注意力始终允许)
            M_prior = M_prior.fill_diagonal_(prior_bias)

            self.register_buffer('M_prior', M_prior)
        else:
            # 无先验时，所有连接都允许
            self.M_prior = None

        # [新增] 初始注意力权重预设 (九图核心斜率)
        self.use_attention_prior = False
        if attention_weights is not None:
            if not isinstance(attention_weights, torch.Tensor):
                attention_weights = torch.tensor(attention_weights, dtype=torch.float32)
            # 存储 log(weight) 作为偏置，避免数值问题
            # 注意力权重 > 1 表示增强连接，< 1 表示削弱连接
            # 使用 log 变换：attn_bias = log(weight)
            attn_prior_bias = torch.log(attention_weights + 1e-8)  # 避免log(0)
            self.register_buffer('attention_prior_bias', attn_prior_bias)
            self.use_attention_prior = True

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def get_effective_gamma(self):
        """
        获取有效的 gamma 值

        使用 softplus 保证正数，并应用下限约束

        Returns:
            gamma_effective: [B,] 标量张量
        """
        gamma_positive = F.softplus(self.gamma_raw)
        return torch.clamp(gamma_positive, min=self.gamma_min)

    def forward(self, x):
        """
        Args:
            x: [B, num_nodes, hidden_dim] 节点特征 (22 个独立节点)

        Returns:
            out: [B, num_nodes, hidden_dim] 更新后的节点特征
        """
        B, N, D = x.shape

        # 1. 计算 Q, K, V
        Q = self.W_q(x)  # [B, N, D]
        K = self.W_k(x)  # [B, N, D]
        V = self.W_v(x)  # [B, N, D]

        # 2. 重塑为多头: [B, N, D] -> [B, num_heads, N, head_dim]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数: QK^T / sqrt(D)
        # [B, num_heads, N, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 4. 加入先验偏置: + gamma * M_prior
        if self.M_prior is not None:
            # M_prior: [N, N] -> 扩展为 [B, num_heads, N, N]
            M_prior = self.M_prior.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            # [PriorWarmStart] 使用有效的 gamma 值
            gamma_effective = self.get_effective_gamma()
            attn_scores = attn_scores + gamma_effective * M_prior

        # 4.1 [新增] 加入初始注意力权重预设偏置 (九图核心斜率)
        if self.use_attention_prior:
            # attention_prior_bias: [N, N] -> 扩展为 [B, num_heads, N, N]
            attn_prior = self.attention_prior_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
            attn_scores = attn_scores + attn_prior

        # 5. Softmax 得到注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, N, N]
        attn_weights = self.dropout(attn_weights)

        # [Interpretation Patch] 缓存注意力分数和权重用于临床解释
        self.last_attn_scores = attn_scores.detach()  # [B, num_heads, N, N]
        self.last_attn_weights = attn_weights.detach()  # [B, num_heads, N, N]

        # 6. 加权求和: Attention * V
        attn_out = torch.matmul(attn_weights, V)  # [B, num_heads, N, head_dim]

        # 7. 合并多头: [B, num_heads, N, head_dim] -> [B, N, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)

        # 8. 输出投影
        attn_out = self.W_o(attn_out)

        # 9. 残差 + LayerNorm
        residual = x
        x = self.layer_norm1(residual + self.dropout(attn_out))

        # 10. FFN + 残差
        residual = x
        x = self.layer_norm2(residual + self.ffn(x))

        return x


class TemporalEncodingBranch(nn.Module):
    """
    时序编码分支 (Temporal Encoding Branch)

    阶段一：单变量独立时序编码与注意力池化

    核心设计:
    1. 每个通道独立处理，避免跨通道混淆
    2. 共享GRU权重，参数高效且保持一致性
    3. 支持变长序列 (pack_padded_sequence)
    4. 注意力池化消除时间维度

    Args:
        num_channels: 特征通道数 (默认 26，启用 o2pulse_enabled 后为 29)
        D_time: 时序编码维度 (默认 16)
        dropout: Dropout 率
        config: TemporalEncoderConfig 对象 (新增，用于控制 mask 消融)
    """
    def __init__(self, num_channels=26, D_time=16, dropout=0.3, config=None):
        super().__init__()
        self.num_channels = num_channels
        self.D_time = D_time

        # [新增] 解析 mask 配置
        if config is not None:
            use_masked_conv = getattr(config, 'use_masked_conv', True)
        else:
            use_masked_conv = True  # 默认使用 pack_padded_sequence
        self.use_masked_conv = use_masked_conv

        # 共享 GRU: 每个时间点输入标量值
        self.shared_gru = nn.GRU(
            input_size=1,
            hidden_size=D_time,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 时间注意力池化
        self.time_attention = nn.Sequential(
            nn.Linear(D_time, D_time // 2),
            nn.Tanh(),
            nn.Linear(D_time // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(D_time)

    def forward(self, x, lengths=None):
        """
        Args:
            x: [B, L_max, C] 变长序列 (已 padding)
            lengths: [B] 每个样本的真实长度 (可选)

        Returns:
            H_nodes: [B, C, D_time] 纯净节点特征
        """
        B, L_max, C = x.shape
        device = x.device

        # 默认长度为最大长度
        if lengths is None:
            lengths = torch.full((B,), L_max, dtype=torch.long, device=device)

        # 1. 重塑为 [B*C, L_max, 1] 并行处理所有通道
        # x: [B, L, C] -> [B, C, L] -> [B*C, L, 1]
        x_flat = x.transpose(1, 2).reshape(B * C, L_max, 1)

        # 2. 为每个通道复制长度
        # lengths: [B] -> [B*C]
        lengths_flat = lengths.repeat_interleave(C).cpu()

        # === 根据 use_masked_conv 条件分支处理 ===
        if self.use_masked_conv:
            # === 原有 pack_padded_sequence 逻辑 ===
            # 3. 处理变长序列
            if lengths_flat.max() > 1:
                # 按长度排序 (pack_padded_sequence 要求)
                sorted_lengths, sorted_indices = torch.sort(lengths_flat, descending=True)
                _, unsorted_indices = torch.sort(sorted_indices)

                x_sorted = x_flat[sorted_indices]

                # Pack 序列
                packed = nn.utils.rnn.pack_padded_sequence(
                    x_sorted, sorted_lengths,
                    batch_first=True, enforce_sorted=True
                )

                # GRU 前向
                packed_out, _ = self.shared_gru(packed)

                # Unpack
                gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out, batch_first=True, total_length=L_max
                )

                # 恢复原始顺序
                gru_out = gru_out[unsorted_indices]  # [B*C, L_max, D_time]
            else:
                # 所有序列长度为 1，直接处理
                gru_out, _ = self.shared_gru(x_flat)  # [B*C, L_max, D_time]

            # 4. 注意力池化消除时间维度
            # 需要屏蔽 padding 位置
            # 创建 mask: [B*C, L_max]
            mask = torch.arange(L_max, device=device).unsqueeze(0) >= lengths_flat.unsqueeze(1).to(device)
            mask = mask.unsqueeze(-1)  # [B*C, L_max, 1]

            # 注意力分数
            attn_scores = self.time_attention(gru_out)  # [B*C, L_max, 1]

            # 屏蔽 padding 位置
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)  # [B*C, L_max, 1]

        else:
            # === [新增] 直接 GRU 处理 ===
            # 不使用 pack_padded_sequence，让 padding 的零值直接参与计算

            gru_out, _ = self.shared_gru(x_flat)  # [B*C, L_max, D_time]

            # 注意力池化 (无 mask 屏蔽)
            attn_scores = self.time_attention(gru_out)  # [B*C, L_max, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # 所有位置参与

        # 共用: 加权求和
        H_flat = (gru_out * attn_weights).sum(dim=1)  # [B*C, D_time]

        # 5. 恢复形状: [B*C, D_time] -> [B, C, D_time]
        H_nodes = H_flat.reshape(B, C, self.D_time)

        # LayerNorm + Dropout
        H_nodes = self.layer_norm(H_nodes)
        H_nodes = self.dropout(H_nodes)

        return H_nodes


# ==========================================
# 可插拔卷积模块 (Pluggable Convolution Modules)
# ==========================================

class PluggableConv1d(nn.Module):
    """
    可插拔卷积模块

    核心设计:
    - use_multiscale=False: 直接调用 nn.Conv1d(kernel_size=base_kernel)，参数量完全一致
    - use_multiscale=True: 并行多尺度 Conv1d 分支 + Concat + 1x1 Conv 压缩

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        base_kernel: 基础卷积核大小 (Baseline 模式使用)
        use_multiscale: 是否启用多尺度卷积
        multiscale_kernels: 多尺度卷积核列表 (默认 [3, 5, 7])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_kernel: int,
        use_multiscale: bool = False,
        multiscale_kernels: list = None,
    ):
        super().__init__()
        self.use_multiscale = use_multiscale

        if not use_multiscale:
            # Baseline Mode: 100% 等价于原版
            self.conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=base_kernel,
                padding=base_kernel // 2
            )
        else:
            # MultiScale Mode
            if multiscale_kernels is None:
                multiscale_kernels = [3, 5, 7]

            self.branch_convs = nn.ModuleList([
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2)
                for k in multiscale_kernels
            ])
            self.fusion_conv = nn.Conv1d(
                len(multiscale_kernels) * out_channels,
                out_channels,
                kernel_size=1
            )

    def forward(self, x):
        if not self.use_multiscale:
            return self.conv(x)
        else:
            branches = [conv(x) for conv in self.branch_convs]
            concat = torch.cat(branches, dim=1)
            return self.fusion_conv(concat)


class PluggableTemporalBlock(nn.Module):
    """
    可插拔时序块 (不包含 Pool)

    核心设计:
    - 不包含 MaxPool (保持维度对齐，Pool 由外部控制)
    - 不添加额外 LayerNorm (保证 Baseline 模式参数量完全匹配)
    - 残差连接仅在 use_residual=True 时启用

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        use_multiscale: 是否启用多尺度卷积
        use_residual: 是否启用残差连接
        multiscale_kernels: 多尺度卷积核列表
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_multiscale: bool = False,
        use_residual: bool = False,
        multiscale_kernels: list = None,
    ):
        super().__init__()
        self.use_residual = use_residual

        # 卷积层
        self.conv = PluggableConv1d(
            in_channels, out_channels, kernel_size,
            use_multiscale, multiscale_kernels
        )

        # BN + GELU (与原版一致)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

        # 残差投影 (仅当启用残差且通道数不同时)
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        # 残差连接
        if self.use_residual and self.residual_proj is not None:
            out = out + self.residual_proj(identity)
        elif self.use_residual:
            out = out + identity

        return out


# ==========================================
# CNN 时序编码分支
# ==========================================

class CNNTemporalEncodingBranch(nn.Module):
    """
    CNN 时序编码分支 (CNN Temporal Encoding Branch)

    核心改进：渐进降维保留时序信息 + 可插拔多尺度/残差模块

    架构:
        输入: [B, L_max, C] 变长序列
            ↓ Stage1: DepthwiseConv + Pool (L_max → L_max/2)
            ↓ Stage2: DepthwiseConv + Pool (L_max/2 → L_max/4)
            ↓ Stage3: AdaptivePool (→ T_mid=24)
            ↓ 时间注意力聚合 (T_mid → 1)
        输出: [B, C, D_time]

    关键设计:
    1. Depthwise Separable Conv: 参数高效，每通道独立处理
    2. Masked Convolution: 处理变长序列的 padding
    3. 中间保留 T_mid 个时间帧: 再通过注意力聚合，而非直接压缩
    4. 可插拔模块: 多尺度卷积 + 残差连接 (默认关闭，保证向后兼容)

    Args:
        num_channels: 特征通道数 (默认 26，启用 o2pulse_enabled 后为 29)
        D_time: 时序编码维度 (默认 16)
        T_mid: 中间时序维度 (默认 24)
        dropout: Dropout 率
        config: TemporalEncoderConfig 对象 (新增，用于控制可插拔模块)
    """
    def __init__(self, num_channels=26, D_time=16, T_mid=24, dropout=0.3, config=None):
        super().__init__()
        self.num_channels = num_channels
        self.D_time = D_time
        self.T_mid = T_mid

        # 解析配置 (支持可插拔模块)
        if config is not None:
            use_multiscale = getattr(config, 'use_multiscale', False)
            use_residual = getattr(config, 'use_residual', False)
            multiscale_kernels = getattr(config, 'multiscale_kernels', [3, 5, 7])
            block1_kernel = getattr(config, 'block1_kernel', 7)
            block2_kernel = getattr(config, 'block2_kernel', 5)
            use_masked_conv = getattr(config, 'use_masked_conv', True)  # [新增]
        else:
            # 默认配置 (向后兼容，100% 等价于原版)
            use_multiscale = False
            use_residual = False
            multiscale_kernels = [3, 5, 7]
            block1_kernel = 7
            block2_kernel = 5
            use_masked_conv = True  # [新增] 默认使用掩膜

        # 保存配置用于调试和 forward
        self.use_multiscale = use_multiscale
        self.use_residual = use_residual
        self.use_masked_conv = use_masked_conv  # [新增]

        # Stage 1: Block + Pool (替代原 Sequential)
        # 输入: [B*C, 1, L] -> 输出: [B*C, 16, L/2]
        self.block1 = PluggableTemporalBlock(
            in_channels=1,
            out_channels=16,
            kernel_size=block1_kernel,
            use_multiscale=use_multiscale,
            use_residual=use_residual,
            multiscale_kernels=multiscale_kernels
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # L → L/2

        # Stage 2: Block + Pool
        # [B*C, 16, L/2] -> [B*C, 32, L/4]
        self.block2 = PluggableTemporalBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=block2_kernel,
            use_multiscale=use_multiscale,
            use_residual=use_residual,
            multiscale_kernels=multiscale_kernels
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # L/2 → L/4

        # Stage 3: Adaptive Pool (→ T_mid)
        # [B*C, 32, L/4] -> [B*C, 32, T_mid]
        self.stage3_pool = nn.AdaptiveAvgPool1d(T_mid)

        # 时间注意力池化 (T_mid → 1)
        # [B*C, 32, T_mid] -> [B*C, 32]
        self.time_attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # 投影层: [B, C, 32] -> [B, C, D_time]
        self.proj = nn.Linear(32, D_time)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(D_time)

    def forward(self, x, lengths=None):
        """
        Args:
            x: [B, L_max, C] 变长序列 (已 padding)
            lengths: [B] 每个样本的真实长度 (可选)

        Returns:
            H_nodes: [B, C, D_time] 纯净节点特征
        """
        B, L_max, C = x.shape
        device = x.device

        # 默认长度为最大长度
        if lengths is None:
            lengths = torch.full((B,), L_max, dtype=torch.long, device=device)

        # 1. 重塑为 [B*C, 1, L_max] 并行处理所有通道
        # x: [B, L, C] -> [B, C, L] -> [B*C, L] -> [B*C, 1, L]
        x_flat = x.transpose(1, 2).reshape(B * C, L_max).unsqueeze(1)

        # === 根据 use_masked_conv 条件分支处理 ===
        if self.use_masked_conv:
            # === 原有 Masked Conv 逻辑 ===
            # 2. 创建 padding mask (用于屏蔽无效区域)
            # mask: [B, L_max] -> [B*C, L_max]
            mask = torch.arange(L_max, device=device).unsqueeze(0) >= lengths.unsqueeze(1).to(device)
            mask_flat = mask.repeat_interleave(C, dim=0)  # [B*C, L_max]

            # 3. Stage 1: Block + Pool
            # 应用 mask: 将 padding 位置置为 0
            x_masked = x_flat * (~mask_flat.unsqueeze(1)).float()
            x1 = self.block1(x_masked)  # [B*C, 16, L_max]
            x1 = x1 * (~mask_flat.unsqueeze(1)).float()  # mask 在 block 后应用
            x1 = self.pool1(x1)  # [B*C, 16, L_max/2]

            # 更新 mask (Pool 后)
            L1 = x1.size(-1)
            mask1 = F.interpolate(mask_flat.unsqueeze(1).float(), size=L1, mode='nearest')
            mask1 = mask1.squeeze(1) > 0.5

            # 4. Stage 2: Block + Pool
            x1_masked = x1 * (~mask1.unsqueeze(1)).float()
            x2 = self.block2(x1_masked)  # [B*C, 32, L_max/2]
            x2 = x2 * (~mask1.unsqueeze(1)).float()  # mask 在 block 后应用
            x2 = self.pool2(x2)  # [B*C, 32, L_max/4]

            # 更新 mask
            L2 = x2.size(-1)
            mask2 = F.interpolate(mask1.unsqueeze(1).float(), size=L2, mode='nearest')
            mask2 = mask2.squeeze(1) > 0.5

            # 5. Stage 3: Adaptive Pool → T_mid
            x3 = self.stage3_pool(x2)  # [B*C, 32, T_mid]

            # 更新 mask
            mask3 = F.interpolate(mask2.unsqueeze(1).float(), size=self.T_mid, mode='nearest')
            mask3 = mask3.squeeze(1) > 0.5

            # 6. 时间注意力聚合 (T_mid → 1)
            # 转置: [B*C, 32, T_mid] -> [B*C, T_mid, 32]
            x3_t = x3.transpose(1, 2)

            # 计算注意力分数
            attn_scores = self.time_attention(x3_t)  # [B*C, T_mid, 1]

            # 屏蔽 padding 位置
            attn_scores = attn_scores.masked_fill(mask3.unsqueeze(-1), float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=1)  # [B*C, T_mid, 1]

        else:
            # === [新增] Force Padding 逻辑 ===
            # 不创建 mask，让 padding 的零值直接参与卷积和注意力计算

            # Stage 1 (无 mask)
            x1 = self.block1(x_flat)  # [B*C, 16, L_max]
            x1 = self.pool1(x1)  # [B*C, 16, L_max/2]

            # Stage 2 (无 mask)
            x2 = self.block2(x1)  # [B*C, 32, L_max/2]
            x2 = self.pool2(x2)  # [B*C, 32, L_max/4]

            # Stage 3: Adaptive Pool → T_mid
            x3 = self.stage3_pool(x2)  # [B*C, 32, T_mid]

            # 时间注意力聚合 (无 mask 屏蔽)
            x3_t = x3.transpose(1, 2)  # [B*C, T_mid, 32]
            attn_scores = self.time_attention(x3_t)  # [B*C, T_mid, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # 所有位置参与

        # 共用: 加权求和
        H_flat = (x3_t * attn_weights).sum(dim=1)  # [B*C, 32]

        # 7. 投影到 D_time 维度
        H_flat = self.proj(H_flat)  # [B*C, D_time]

        # 8. 恢复形状: [B*C, D_time] -> [B, C, D_time]
        H_nodes = H_flat.reshape(B, C, self.D_time)

        # LayerNorm + Dropout
        H_nodes = self.layer_norm(H_nodes)
        H_nodes = self.dropout(H_nodes)

        return H_nodes


# ==========================================
# 合并 Conv + 全局池化时序编码分支 (消融实验)
# ==========================================

class MergedConvGlobalPoolTemporalBranch(nn.Module):
    """
    合并 Conv + 全局池化时序编码分支 (消融实验版本)

    架构:
        输入: [B, L_max, C]
            ↓ 张量重组: [B*C, 1, L_max]
            ↓ 合并 Conv (替代 Stage1+Stage2): → [B*C, 32, L_max]
            ↓ BN + GELU
            ↓ Masked Global Avg Pool: → [B*C, 32]
            ↓ Linear: → [B*C, D_time]
            ↓ 恢复维度: → [B, C, D_time]
        输出: [B, C, D_time]

    实验假设:
        渐进降维（多级池化保留时序信息）优于单次全局池化

    参数量对比 (单通道):
        Baseline: ~3873 (Stage1+Stage2+TimeAttn+Proj+LN)
        Ablation: ~980  (MergedConv+GlobalPool+Proj+LN)
        减少: ~75%
    """

    def __init__(self, num_channels=26, D_time=16, dropout=0.3,
                 merged_kernel=11, use_masked_conv=True):
        """
        Args:
            num_channels: 特征通道数 (默认 26，启用 o2pulse_enabled 后为 29)
            D_time: 时序编码维度 (默认 16)
            dropout: Dropout 率
            merged_kernel: 合并后的卷积核大小
                - 默认 11 (合并 Stage1.k=7 + Stage2.k=5 的感受野)
                - 或直接用 7 或 5 作为简化版本
            use_masked_conv: 是否使用掩膜处理 padding (默认 True)
        """
        super().__init__()
        self.num_channels = num_channels
        self.D_time = D_time
        self.use_masked_conv = use_masked_conv

        # 合并 Conv: 1 → 32 (替代 Stage1: 1→16 + Stage2: 16→32)
        self.conv = nn.Conv1d(1, 32, kernel_size=merged_kernel,
                              padding=merged_kernel // 2)
        self.bn = nn.BatchNorm1d(32)
        self.activation = nn.GELU()

        # Linear 映射: 32 → D_time
        self.proj = nn.Linear(32, D_time)
        self.layer_norm = nn.LayerNorm(D_time)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths=None):
        """
        Args:
            x: [B, L_max, C] 变长序列 (已 padding)
            lengths: [B] 每个样本的真实长度 (可选)

        Returns:
            H_nodes: [B, C, D_time] 纯净节点特征
        """
        B, L_max, C = x.shape
        device = x.device

        if lengths is None:
            lengths = torch.full((B,), L_max, dtype=torch.long, device=device)

        # 1. 张量重组: [B, L, C] → [B*C, 1, L]
        x_flat = x.transpose(1, 2).reshape(B * C, L_max).unsqueeze(1)

        # 2. 创建 padding mask
        mask = torch.arange(L_max, device=device).unsqueeze(0) >= lengths.unsqueeze(1).to(device)
        mask_flat = mask.repeat_interleave(C, dim=0)  # [B*C, L]

        # 3. Masked Conv
        if self.use_masked_conv:
            x_masked = x_flat * (~mask_flat.unsqueeze(1)).float()
        else:
            x_masked = x_flat

        # 4. 合并 Conv 局部特征提取
        x_conv = self.conv(x_masked)  # [B*C, 32, L]
        x_conv = self.bn(x_conv)
        x_conv = self.activation(x_conv)

        # 再次 mask (卷积后 padding 区域仍有残留)
        if self.use_masked_conv:
            x_conv = x_conv * (~mask_flat.unsqueeze(1)).float()

        # 5. Masked Global Avg Pool
        if self.use_masked_conv:
            x_conv_t = x_conv.transpose(1, 2)  # [B*C, L, 32]
            mask_expanded = mask_flat.unsqueeze(-1)  # [B*C, L, 1]

            x_masked_t = x_conv_t * (~mask_expanded).float()

            valid_lengths = (L_max - mask_flat.sum(dim=1)).clamp(min=1)  # [B*C]

            x_global = x_masked_t.sum(dim=1) / valid_lengths.unsqueeze(1)  # [B*C, 32]
        else:
            x_global = x_conv.mean(dim=2)  # [B*C, 32]

        # 6. Linear 映射
        H_flat = self.proj(x_global)

        # 7. 恢复节点维度
        H_nodes = H_flat.reshape(B, C, self.D_time)

        # 8. LayerNorm + Dropout
        H_nodes = self.layer_norm(H_nodes)
        H_nodes = self.dropout(H_nodes)

        return H_nodes


class PoolingProjectionBlock(nn.Module):
    """
    池化投影模块 (Pooling_only 模式)

    架构:
        输入: [B, C, D_time]
            ↓ AdaptiveAvgPool1d/AdaptiveMaxPool1d (D_time 维度)
            ↓ [B, C, 1] → squeeze → [B, C]
            ↓ MLP Projection
        输出: [B, 48]

    参数量对比:
        - 单层 (C=30→48): ~1.6k
        - 双层 (30→64→48): ~5.2k

    用于消融实验: 验证通道级聚合是否足够，与 flatten_only 对比
    """
    def __init__(self, num_channels=30, D_time=16, dropout=0.3,
                 pooling_type="avg", mlp_layers=1, hidden_dim=None,
                 use_layer_norm=True):
        super().__init__()
        self.pooling_type = pooling_type

        # 池化层: [B, C, D_time] -> [B, C, 1]
        if pooling_type == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")

        # MLP 投影层
        if mlp_layers == 1:
            self.mlp = nn.Sequential(
                nn.Linear(num_channels, 48),
                nn.LayerNorm(48) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif mlp_layers == 2:
            hidden = hidden_dim or 64
            self.mlp = nn.Sequential(
                nn.Linear(num_channels, hidden),
                nn.LayerNorm(hidden) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 48),
                nn.LayerNorm(48) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"mlp_layers must be 1 or 2, got {mlp_layers}")

    def forward(self, H_nodes):
        """
        Args:
            H_nodes: [B, C, D_time] 节点特征 (来自 CNN 时序编码器)

        Returns:
            feat: [B, 48] 全局特征
        """
        # Pool along D_time dimension: [B, C, D_time] -> [B, C, 1]
        # 注意: AdaptiveAvgPool1d/AdaptiveMaxPool1d 操作在最后一个维度上
        # 输入形状为 [B, C, D_time]，池化后为 [B, C, 1]
        pooled = self.pool(H_nodes)  # [B, C, 1]
        pooled = pooled.squeeze(-1)  # [B, C]
        feat = self.mlp(pooled)      # [B, 48]
        return feat


class HierarchicalSpatialGraph(nn.Module):
    """
    层级空间图 (Hierarchical Spatial Graph) - 双轨拓扑融合版

    阶段二：多层级子图-全局图聚合与残差连接

    架构:
    1. 子图GCN: 按医学先验分组 (代谢G0, 循环G1, 呼吸G2, 气体G3)
       - 每个子图接收对应的医学先验子矩阵
    2. 全局GCN: 4个宏观节点的交互
    3. 残差连接: 防止过度平滑

    Args:
        D_time: 时序特征维度
        channel_groups: 通道分组列表
        dropout: Dropout 率
        semantic_adj: 全局语义邻接矩阵 (22x22)
        ablation_mode: 消融模式
            - "hierarchical": 原有的层次化融合 (子图GCN + 全局GCN + Mean Pooling)
            - "global_only": 直接全局注意力 (无子图)
            - "prior_masked": 基于先验掩码的全局 Transformer (新增)
            - "flatten_only": 移除跨变量交互，直接展平 + MLP 降维 (新增)
            - "pooling_only": 用全局池化替代展平，通道级聚合 + MLP (新增)
        gamma_init: gamma 初始值 (PriorWarmStart, 默认 1.0)
        gamma_min: gamma 下限 (PriorWarmStart, 默认 0.1)
        attention_weights: [新增] 核心斜率注意力权重预设矩阵 [N, N] (仅 prior_masked 模式生效)
        flatten_mlp_config: [新增] FlattenMLPConfig 对象 (仅 flatten_only 模式生效)
        pooling_only_config: [新增] PoolingOnlyConfig 对象 (仅 pooling_only 模式生效)
    """
    def __init__(self, num_channels=26, D_time=16, channel_groups=None, dropout=0.3,
                 semantic_adj=None, ablation_mode="hierarchical",
                 gamma_init=1.0, gamma_min=0.1, attention_weights=None,
                 flatten_mlp_config=None, pooling_only_config=None):
        super().__init__()
        self.D_time = D_time
        self.ablation_mode = ablation_mode  # 保存消融模式
        self.num_channels = num_channels
        self.channel_groups = channel_groups if channel_groups else [
            # G0: 运动负荷与能量代谢 (6 features)
            [0, 1, 2, 3, 4, 22],  # MET, Load, RER, HR, HRR, OUES
            # G1: 循环系统/血流动力学 (8 features)
            [5, 6, 7, 8, 9, 10, 23, 25],  # dH/dO2, SVc, Psys, Pdia, SpO2, V'O2, PP, HR_diff
            # G2: 呼吸动力学 (3 features)
            [11, 12, 13],  # VO2/kg, dO2/dW, BF
            # G3: 气体交换 (9 features)
            [14, 15, 16, 17, 18, 19, 20, 21, 24]  # V'E, BR, EqO2, EqCO2, PETO2, PETCO2, VDc/VT, VTex, EqO2_COP
        ]
        self.num_groups = len(self.channel_groups)

        # 确保 semantic_adj 是 Tensor
        if semantic_adj is not None and not isinstance(semantic_adj, torch.Tensor):
            semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32)

        # ==================== 新增: Prior-Masked Global Transformer ====================
        if ablation_mode == "prior_masked":
            # 使用基于先验掩码的全局 Transformer
            # 29 个节点直接参与全局注意力，医学先验通过 M_prior 注入
            self.prior_masked_transformer = PriorMaskedGlobalTransformer(
                hidden_dim=D_time,
                num_heads=2,
                num_nodes=num_channels,
                semantic_adj=semantic_adj,
                dropout=dropout,
                prior_bias=0.0,      # 有连接的节点对允许注意力
                mask_value=-1e9,     # 大负数代替 -inf，避免梯度 NaN
                gamma_init=gamma_init,  # [PriorWarmStart] 热启动
                gamma_min=gamma_min,     # [PriorWarmStart] 下限约束
                attention_weights=attention_weights  # [新增] 核心斜率权重预设
            )
            # 投影层: num_channels * D_time -> 48
            proj_in_dim = num_channels * D_time

        # ==================== 新增: Flatten Only (移除跨变量交互) ====================
        elif ablation_mode == "flatten_only":
            # 消融实验: w/o Stage2 跨变量交互
            # 直接展平时序编码输出 [B, C, D_time] → [B, C*D_time] → [B, 48]
            # 无跨变量注意力交互，仅通过 MLP 降维
            in_dim = num_channels * D_time  # 480 (30 * 16)

            if flatten_mlp_config and flatten_mlp_config.use_two_layer:
                # 两层 MLP: 480 → hidden_dim → 48
                hidden = flatten_mlp_config.hidden_dim or 256
                mlp_dropout = flatten_mlp_config.dropout if flatten_mlp_config else dropout
                self.flatten_proj = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden) if flatten_mlp_config.use_layer_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(mlp_dropout),
                    nn.Linear(hidden, 48),
                    nn.LayerNorm(48) if flatten_mlp_config.use_layer_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            else:
                # 单层 MLP: 480 → 48 (参数量最接近基线)
                self.flatten_proj = nn.Sequential(
                    nn.Linear(in_dim, 48),
                    nn.LayerNorm(48) if (flatten_mlp_config and flatten_mlp_config.use_layer_norm) else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            # flatten_only 模式不需要投影层，直接输出 48 维
            proj_in_dim = 48  # 标记 flatten_proj 直接输出 48 维

        # ==================== 新增: Pooling_only (通道级聚合) ====================
        elif ablation_mode == "pooling_only":
            # 消融实验: 用池化替代 Flatten + 跨变量交互
            # 输入: [B, C, D_time] -> Pool -> [B, C] -> MLP -> [B, 48]
            self.pooling_proj = PoolingProjectionBlock(
                num_channels=num_channels,
                D_time=D_time,
                dropout=dropout,
                pooling_type=pooling_only_config.pooling_type if pooling_only_config else "avg",
                mlp_layers=pooling_only_config.mlp_layers if pooling_only_config else 1,
                hidden_dim=pooling_only_config.hidden_dim if pooling_only_config else None,
                use_layer_norm=pooling_only_config.use_layer_norm if pooling_only_config else True
            )
            # pooling_only 直接输出 48 维，不需要 proj 层
            proj_in_dim = 48

        else:
            # ==================== 原有代码 (保留注释方便回退) ====================
            # 创建 4 个子图 GCN，并为其注入专属的医学先验子图
            self.sub_gcns = nn.ModuleList()
            for g_idx in self.channel_groups:
                if semantic_adj is not None:
                    # 魔法切片：直接提取子图矩阵
                    # 例如 g_idx = [0,1,2,3,4]，提取出的就是 5x5 的代谢系统内生网络
                    sub_prior = semantic_adj[g_idx][:, g_idx]
                else:
                    sub_prior = None

                self.sub_gcns.append(SubGraphLayer(len(g_idx), D_time, dropout, prior_adj=sub_prior))

            # 全局图卷积 (4个宏观节点)
            self.global_gcn = GlobalGraphLayer(D_time, num_heads=2, dropout=dropout)

            # 动态计算投影层维度 (全局直接融合是 22 * D_time，层次化是 4 * D_time)
            proj_in_dim = (num_channels if ablation_mode == "global_only" else self.num_groups) * D_time
            # ==================== 原有代码结束 ====================

        # 输出投影 (flatten_only 和 pooling_only 模式不需要此层，因为已直接输出 48 维)
        if ablation_mode not in ["flatten_only", "pooling_only"]:
            self.proj = nn.Sequential(
                nn.Linear(proj_in_dim, 48),
                nn.LayerNorm(48),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.proj = None  # flatten_only/pooling_only 模式不使用此投影层

    def forward(self, H_nodes):
        """
        Args:
            H_nodes: [B, C, D_time] 节点特征

        Returns:
            feat: [B, 48] 全局特征
        """
        B, C, D = H_nodes.shape

        # ==================== 新增: Prior-Masked Global Transformer ====================
        if self.ablation_mode == "prior_masked":
            # num_channels 个节点直接参与全局注意力
            # 医学先验通过 M_prior 注入注意力矩阵
            # 公式: A = Softmax(QK^T / sqrt(D) + gamma * M_prior) * V
            out = self.prior_masked_transformer(H_nodes)  # [B, num_channels, D]

            # 展平所有节点特征
            flat = out.reshape(B, -1)  # [B, num_channels * D]

        # ==================== 新增: Flatten Only (移除跨变量交互) ====================
        elif self.ablation_mode == "flatten_only":
            # 消融实验: w/o Stage2 跨变量交互
            # 直接展平 [B, C, D_time] → [B, C*D_time]，通过 MLP 降维
            # 跳过所有跨变量注意力机制
            flat = H_nodes.reshape(B, -1)  # [B, num_channels * D_time]
            feat = self.flatten_proj(flat)  # [B, 48]
            return feat  # 直接返回，跳过后续 proj 层

        # ==================== 新增: Pooling_only (通道级聚合) ====================
        elif self.ablation_mode == "pooling_only":
            # 消融实验: 用池化替代 Flatten + 跨变量交互
            # [B, C, D_time] -> Pool(D_time) -> [B, C] -> MLP -> [B, 48]
            feat = self.pooling_proj(H_nodes)  # [B, 48]
            return feat  # 直接返回，跳过 proj 层

        # ==================== 原有代码 (保留注释方便回退) ====================
        elif self.ablation_mode == "global_only":
            # 直接全系统融合：跳过分组和子图，让 num_channels 个通道直接在 GlobalGraphLayer 做全局注意力交互
            out = self.global_gcn(H_nodes)  # [B, num_channels, D]
            flat = out.reshape(B, -1)  # [B, num_channels*D]

        else:  # "hierarchical"
            # 原有的层次化融合：先局部聚合，再全局协同
            # 1. 子图聚合
            hub_features = []
            for gid, (indices, sub_gcn) in enumerate(zip(self.channel_groups, self.sub_gcns)):
                # 提取子图节点: [B, num_nodes_in_group, D]
                sub_nodes = H_nodes[:, indices, :]
                # 子图卷积
                sub_out = sub_gcn(sub_nodes)  # [B, num_nodes, D]
                # 平均池化生成 hub
                hub = sub_out.mean(dim=1)  # [B, D]
                hub_features.append(hub)

            # 2. 堆叠为全局节点: [B, num_groups, D]
            hubs = torch.stack(hub_features, dim=1)  # [B, 4, D]

            # 3. 全局图卷积
            out = self.global_gcn(hubs)  # [B, 4, D]

            # 4. 展平
            flat = out.reshape(B, -1)  # [B, 4*D]
        # ==================== 原有代码结束 ====================

        # 投影到输出维度
        feat = self.proj(flat)  # [B, 48]

        return feat


class HDSTGCN(nn.Module):
    """
    HDSTGCN: Hierarchical Dynamic Spatio-Temporal Graph Convolutional Network
    (双轨拓扑融合版 - 医学先验 + 数据驱动)

    支持双模态融合: CPET 动态特征 + EHR 静态特征

    两阶段架构:
    ┌─────────────────────────────────────────────────────────────┐
    │ 阶段一: TemporalEncodingBranch                              │
    │   输入: [B, L, 22] 变长序列                                 │
    │   处理: 共享GRU独立编码每个通道 + 注意力池化                 │
    │   输出: [B, 22, D_time] 纯净节点特征                        │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 阶段二: HierarchicalSpatialGraph (双轨拓扑融合)              │
    │   子图聚合: 4个子系统GCN (代谢/循环/呼吸/气体)               │
    │     - 每个子图: 医学先验(固定) + 数据驱动(可学习) + 门控融合 │
    │   全局聚合: 宏观节点交互                                     │
    │   输出: [B, 48] 全局特征                                    │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 静态特征编码器 (可选)                                        │
    │   输入: [B, 5] (年龄, 性别, 体重, 身高, BMI)                 │
    │   输出: [B, static_dim]                                     │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ 融合: Concat(动态特征, 静态特征) -> [B, 48+static_dim]      │
    │ 分类器: Linear -> ReLU -> Linear(32, output_dim)            │
    └─────────────────────────────────────────────────────────────┘

    Args:
        input_dim: 时间步长 (用于兼容接口)
        hidden_dim: 隐藏维度 (用于兼容接口)
        channel_groups: 通道分组
        output_dim: 类别数
        num_channel: 特征通道数
        D_time: 时序编码维度
        dropout: Dropout 率
        semantic_adj: 全局语义邻接矩阵 (22x22)，将被切分为子图先验
        use_static_features: 是否启用静态特征融合
        static_dim: 静态编码器输出维度
        num_static_features: 静态特征数量 (默认 5)
        static_ablation: 消融模式 (full, static_only, cpet_only)
        graph_ablation: 图消融模式 (hierarchical, global_only, prior_masked, flatten_only, pooling_only)
        temporal_encoder_type: 时序编码器类型 (gru, cnn)
        T_mid: CNN 时序编码器中间帧数
        gamma_init: gamma 初始值 (PriorWarmStart)
        gamma_min: gamma 下限 (PriorWarmStart)
        attention_weights: [新增] 核心斜率注意力权重预设矩阵 (仅 prior_masked 模式生效)
        flatten_mlp_config: [新增] FlattenMLPConfig 对象 (仅 flatten_only 模式生效)
        pooling_only_config: [新增] PoolingOnlyConfig 对象 (仅 pooling_only 模式生效)
    """
    def __init__(self, input_dim, hidden_dim, channel_groups, output_dim, num_channel,
                 D_time=16, dropout=0.3, semantic_adj=None,
                 # 静态特征融合参数
                 use_static_features=False, static_dim=16, num_static_features=5,
                 static_ablation="full", graph_ablation="global_only",
                 # 时序编码器参数
                 temporal_encoder_type="gru", T_mid=24,
                 temporal_encoder_cfg=None,  # 新增: TemporalEncoderConfig 对象
                 # PriorWarmStart 参数
                 gamma_init=1.0, gamma_min=0.1,
                 # 通道注意力参数
                 use_channel_attention=False, channel_attention_init=1.0,
                 # [新增] 核心斜率注意力权重预设 (仅 prior_masked 模式生效)
                 attention_weights=None,
                 # [新增] Flatten MLP 配置 (仅 flatten_only 模式生效)
                 flatten_mlp_config=None,
                 # [新增] Pooling_only 配置 (仅 pooling_only 模式生效)
                 pooling_only_config=None,
                 # [新增] 二分类模式参数
                 is_binary=False,
                 # [新增] Known-T6 Context 参数
                 use_known_t6_context=False,
                 t6_n_classes=0,
                 **kwargs):
        super().__init__()
        self.num_channel = num_channel
        self.D_time = D_time
        self.use_static_features = use_static_features
        self.static_ablation = static_ablation
        self.graph_ablation = graph_ablation
        self.temporal_encoder_type = temporal_encoder_type
        self.use_known_t6_context = use_known_t6_context  # [新增]
        self.t6_n_classes = t6_n_classes                  # [新增]
        self.use_channel_attention = use_channel_attention
        self.is_binary = is_binary  # [新增] 保存二分类模式标记
        self.use_static_features = use_static_features
        self.static_ablation = static_ablation
        self.graph_ablation = graph_ablation
        self.temporal_encoder_type = temporal_encoder_type
        self.use_channel_attention = use_channel_attention

        # [新增] 时序编码器消融模式标记 (用于 SwanLab 记录)
        self.temporal_encoder_ablation = "full"  # 默认值

        # 阶段一: 时序编码
        if temporal_encoder_type == "cnn":
            # 解析消融模式
            ablation_mode = getattr(temporal_encoder_cfg, 'ablation', 'full') if temporal_encoder_cfg else 'full'

            if ablation_mode == "merged_conv_global_pool":
                # 消融: 合并 Conv + 全局池化
                self.temporal_encoder = MergedConvGlobalPoolTemporalBranch(
                    num_channels=num_channel,
                    D_time=D_time,
                    dropout=dropout,
                    merged_kernel=11,  # 合并 k=7+k=5 的感受野
                    use_masked_conv=getattr(temporal_encoder_cfg, 'use_masked_conv', True) if temporal_encoder_cfg else True
                )
                self.temporal_encoder_ablation = "merged_conv_global_pool"
            else:
                # Baseline: 渐进降维 + 时间注意力
                self.temporal_encoder = CNNTemporalEncodingBranch(
                    num_channels=num_channel,
                    D_time=D_time,
                    dropout=dropout,
                    T_mid=T_mid,
                    config=temporal_encoder_cfg  # 传递配置对象
                )
                self.temporal_encoder_ablation = "full"
        else:
            # 默认使用 GRU 编码器
            self.temporal_encoder = TemporalEncodingBranch(
                num_channels=num_channel,
                D_time=D_time,
                dropout=dropout
            )
            self.temporal_encoder_ablation = "gru"  # GRU 模式标记

        # [新增] 通道注意力模块 (在时序编码后、空间聚合前)
        if use_channel_attention:
            self.channel_attention = ChannelAttention(
                num_channels=num_channel,
                init_value=channel_attention_init
            )
        else:
            self.channel_attention = None

        # 阶段二: 空间图聚合 (传入 semantic_adj)
        self.spatial_graph = HierarchicalSpatialGraph(
            D_time=D_time,
            channel_groups=channel_groups,
            dropout=dropout,
            semantic_adj=semantic_adj,  # <--- 打通数据流
            num_channels=num_channel,
            ablation_mode=graph_ablation,
            gamma_init=gamma_init,      # [PriorWarmStart]
            gamma_min=gamma_min,         # [PriorWarmStart]
            attention_weights=attention_weights,  # [新增] 核心斜率权重预设
            flatten_mlp_config=flatten_mlp_config,  # [新增] Flatten MLP 配置
            pooling_only_config=pooling_only_config  # [新增] Pooling_only 配置
        )

        # 静态特征编码器 (可选)
        # [修复] fusion_dim 应根据 static_ablation 参数动态调整，而非仅根据 use_static_features
        #   - full:       48 + 16 = 64 (动态 + 静态融合)
        #   - cpet_only:  48        (仅动态特征)
        #   - static_only: 16       (仅静态特征，但该模式下 fusion_dim = static_dim)
        fusion_dim = 48  # 动态特征维度 (默认)
        if use_static_features:
            self.static_encoder = nn.Sequential(
                nn.Linear(num_static_features, 32),
                nn.LayerNorm(32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, static_dim),
                nn.ReLU()
            )
            # [关键修复] 仅当 static_ablation 包含静态特征时才增加融合维度
            if static_ablation in ["full", "static_only"]:
                fusion_dim = 48 + static_dim  # 融合维度: 64
            # cpet_only 模式: fusion_dim 保持 48，分类器仅接收动态特征

        # [新增] Known-T6 Context: 增加融合维度
        if use_known_t6_context and t6_n_classes > 0:
            fusion_dim += t6_n_classes  # fusion_dim + N_t6

        # [新增] 二分类模式: 输出维度为 1 (BCEWithLogitsLoss 需要)
        output_dim_final = 1 if is_binary else output_dim

        # 分类器 (输入维度根据 static_ablation 动态调整)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim_final)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, lengths=None, prior_adj=None, static_x=None, t6_context=None):
        """
        Args:
            x: [B, L, C] 变长序列
            lengths: [B] 真实长度 (可选)
            prior_adj: 邻接矩阵 (兼容接口，本模型不使用)
            static_x: [B, 5] 静态特征 (年龄, 性别, 体重, 身高, BMI)
            t6_context: [B, t6_n_classes] Known-T6 Context one-hot (可选)

        Returns:
            logits: [B, output_dim] 分类输出
        """
        batch_size = x.size(0)
        device = x.device

        # 消融模式处理
        use_cpet = self.static_ablation in ["full", "cpet_only"]
        use_static = self.static_ablation in ["full", "static_only"]

        # 1. 动态特征提取
        if use_cpet:
            # 阶段一: 时序编码 -> [B, C, D_time]
            H_nodes = self.temporal_encoder(x, lengths)

            # [新增] 通道注意力加权
            if self.channel_attention is not None:
                H_nodes = self.channel_attention(H_nodes)

            # 阶段二: 空间图聚合 -> [B, 48]
            feat_dyn = self.spatial_graph(H_nodes)
        else:
            feat_dyn = torch.zeros(batch_size, 48, device=device)

        # 2. 静态特征提取
        if use_static and self.use_static_features and static_x is not None:
            feat_stat = self.static_encoder(static_x)  # [B, static_dim]
        else:
            feat_stat = torch.zeros(batch_size, 16, device=device)

        # 3. 特征融合
        if self.use_static_features and use_static:
            feat_fused = torch.cat([feat_dyn, feat_stat], dim=1)  # [B, 48+16]
        else:
            feat_fused = feat_dyn

        # [新增] Known-T6 Context 拼接
        if self.use_known_t6_context and t6_context is not None:
            feat_fused = torch.cat([feat_fused, t6_context], dim=1)  # [B, fusion_dim+N_t6]

        # 4. 分类
        logits = self.classifier(feat_fused)

        return logits

    def forward_with_features(self, x, lengths=None, prior_adj=None, static_x=None, t6_context=None, return_feature_dict=False):
        """
        前向传播并返回特征向量 (用于对比学习损失 / Feature Distillation)

        Args:
            x: [B, L, C] 变长序列
            lengths: [B] 真实长度 (可选)
            prior_adj: 邻接矩阵 (兼容接口，本模型不使用)
            static_x: [B, 5] 静态特征 (年龄, 性别, 体重, 身高, BMI)
            t6_context: [B, t6_n_classes] Known-T6 Context one-hot (可选)
            return_feature_dict: False -> return (logits, feat_fused)
                                 True  -> return (logits, {"dyn_feat":..., "static_feat":..., "fused_feat":...})

        Returns:
            logits: [B, output_dim] 分类输出
            features: [B, D] 融合特征向量 (return_feature_dict=False)
                      或 dict with dyn_feat, static_feat, fused_feat (return_feature_dict=True)
        """
        batch_size = x.size(0)
        device = x.device

        # 消融模式处理
        use_cpet = self.static_ablation in ["full", "cpet_only"]
        use_static = self.static_ablation in ["full", "static_only"]

        # 1. 动态特征提取
        if use_cpet:
            # 阶段一: 时序编码 -> [B, C, D_time]
            H_nodes = self.temporal_encoder(x, lengths)

            # 通道注意力加权
            if self.channel_attention is not None:
                H_nodes = self.channel_attention(H_nodes)

            # 阶段二: 空间图聚合 -> [B, 48]
            feat_dyn = self.spatial_graph(H_nodes)
        else:
            feat_dyn = torch.zeros(batch_size, 48, device=device)

        # 2. 静态特征提取
        if use_static and self.use_static_features and static_x is not None:
            feat_stat = self.static_encoder(static_x)  # [B, static_dim]
        else:
            feat_stat = torch.zeros(batch_size, 16, device=device)

        # 3. 特征融合
        if self.use_static_features and use_static:
            feat_fused = torch.cat([feat_dyn, feat_stat], dim=1)  # [B, 48+16]
        else:
            feat_fused = feat_dyn

        # [新增] Known-T6 Context 拼接
        if self.use_known_t6_context and t6_context is not None:
            feat_fused = torch.cat([feat_fused, t6_context], dim=1)  # [B, fusion_dim+N_t6]

        # 4. 分类
        logits = self.classifier(feat_fused)

        if return_feature_dict:
            return logits, {
                "dyn_feat": feat_dyn,
                "static_feat": feat_stat,
                "fused_feat": feat_fused,
            }
        return logits, feat_fused

    def get_gamma(self):
        """
        获取当前的 gamma 值 (用于监控)

        Returns:
            gamma: 标量张量，如果使用 prior_masked 模式；否则返回 None
        """
        if self.graph_ablation == "prior_masked":
            return self.spatial_graph.prior_masked_transformer.get_effective_gamma()
        return None

    def get_channel_weights(self):
        """
        获取当前通道注意力权重 (用于 SwanLab 记录)

        Returns:
            weights: [C] 归一化后的通道权重，如果未启用通道注意力则返回 None
        """
        if self.channel_attention is not None:
            return self.channel_attention.get_weights()
        return None


# ==========================================================================
# 模型测试代码
# ==========================================================================
if __name__ == '__main__':
    device = torch.device("cpu")

    # 测试 HDSTGCN
    print("=" * 60)
    print("Testing HDSTGCN")
    print("=" * 60)

    groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13], [14, 15, 16, 17, 18, 19, 20, 21]]

    model = HDSTGCN(
        input_dim=162,
        hidden_dim=None,
        channel_groups=groups,
        output_dim=5,
        num_channel=22,
        D_time=16,
        dropout=0.3
    )

    # 测试变长序列
    x = torch.randn(8, 162, 22)
    lengths = torch.tensor([162, 150, 140, 130, 120, 110, 100, 90])

    out = model(x, lengths)
    print(f"Output shape: {out.shape}")
    print(f"Total Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 检查输出健康度
    print(f"Logits mean: {out.mean().item():.4f}")
    print(f"Logits std: {out.std().item():.4f}")
    print(f"Logits range: [{out.min().item():.2f}, {out.max().item():.2f}]")