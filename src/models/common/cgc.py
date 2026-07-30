"""
CGC Common Module - Customized Gate Control Components

CGC = Single-level Customized Gate Control (baseline for PLE)

Key Design (vs MMoE):
- MMoE: All tasks share the SAME expert pool, each task gate selects from ALL experts
- CGC: Shared experts + Task-specific private experts, each task gate selects from shared + own private

Author: CGC Clean Implementation
Date: 2026-05-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class CGCExpert(nn.Module):
    """
    CGC Expert - Lightweight temporal encoder for CPET sequence.

    Input:  x_dyn [B, L, C] - CPET dynamic sequence
    Output: H     [B, C, D] - Expert output representation

    Design: Single-scale CNN (same as MMoEExpert)
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        expert_name: str = "expert_0",
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_channels = num_channels
        self.D_time = D_time
        self.expert_name = expert_name

        # Stage 1: Depthwise Conv + Pointwise
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=7, padding=3, groups=num_channels),
            nn.Conv1d(num_channels * 2, num_channels * 2, kernel_size=1),
            nn.BatchNorm1d(num_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )

        # Stage 2: Depthwise Conv + Pointwise
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_channels * 2, num_channels * 2, kernel_size=5, padding=2, groups=num_channels * 2),
            nn.Conv1d(num_channels * 2, num_channels * 2, kernel_size=1),
            nn.BatchNorm1d(num_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(T_mid)
        self.proj = nn.Sequential(
            nn.Linear(T_mid, D_time),
            nn.LayerNorm(D_time),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.T_mid = T_mid

    def forward(self, x_dyn: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x_dyn.shape
        x = x_dyn.transpose(1, 2)  # [B, C, L]
        x = self.conv1(x)  # [B, 2C, L/2]
        x = self.conv2(x)  # [B, 2C, L/4]
        x = x.view(B, C, -1)  # [B, C, 2*L']
        x = self.adaptive_pool(x)  # [B, C, T_mid]
        H = self.proj(x)  # [B, C, D_time]
        return H

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CGCGateContextEncoder(nn.Module):
    """
    CGC Gate Context Encoder.

    Input:  x_dyn [B, L, C] + x_static [B, 5]
    Output: c_gate [B, context_dim]
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        static_dim: int = 5,
        context_dim: int = 40,
        dropout: float = 0.2
    ):
        super().__init__()
        self.context_dim = context_dim

        self.dyn_proj = nn.Sequential(
            nn.Linear(num_channels, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        static_dim_out = context_dim - 32
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, static_dim_out),
            nn.ReLU()
        )

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_dyn_pooled = x_dyn.mean(dim=1)  # [B, C]
        dyn_context = self.dyn_proj(x_dyn_pooled)  # [B, 32]
        static_context = self.static_proj(x_static)  # [B, static_dim_out]
        c_gate = torch.cat([dyn_context, static_context], dim=1)  # [B, context_dim]
        return c_gate


class CGCTaskGate(nn.Module):
    """
    CGC Task-Specific Gate.

    Key difference from MMoE: Each task gate only selects from:
    - Shared experts (task-common knowledge)
    - Own private expert (task-specific knowledge)

    Output: w [B, num_candidates] where num_candidates = num_shared_experts + 1
    """

    def __init__(
        self,
        context_dim: int = 40,
        num_shared_experts: int = 2,
        task_name: str = "unknown",
        tau_init: float = 1.0
    ):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.task_name = task_name
        self.num_candidates = num_shared_experts + 1

        self.gate_proj = nn.Linear(context_dim, self.num_candidates)
        self.tau = tau_init

        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, c_gate: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_proj(c_gate)  # [B, num_candidates]
        weights = F.softmax(gate_logits / self.tau, dim=-1)  # [B, num_candidates]
        return weights


class CGCLayer(nn.Module):
    """
    CGC Layer - Shared Experts + Private Experts + Task Gates.

    Key Difference from MMoE:
    - MMoE: All tasks share same expert pool, all tasks can select from ALL experts
    - CGC: Shared pool + private expert per task, each task gate only sees shared + own private
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        num_shared_experts: int = 2,
        task_names: List[str] = ["t1", "t2", "t3", "t4", "t5"],
        context_dim: int = 40,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.task_names = task_names

        # Shared Expert Pool
        self.shared_experts = nn.ModuleList([
            CGCExpert(num_channels, D_time, T_mid, f"shared_{i}", dropout)
            for i in range(num_shared_experts)
        ])

        # Task-specific Private Experts
        self.private_experts = nn.ModuleDict({
            task_name: CGCExpert(num_channels, D_time, T_mid, f"private_{task_name}", dropout)
            for task_name in task_names
        })

        # Gate Context Encoder
        self.gate_context_encoder = CGCGateContextEncoder(num_channels, D_time, 5, context_dim, dropout)

        # Task-specific Gates
        self.task_gates = nn.ModuleDict({
            task_name: CGCTaskGate(context_dim, num_shared_experts, task_name)
            for task_name in task_names
        })

        print(f"[CGC Layer] num_shared_experts={num_shared_experts}, tasks={task_names}")
        print(f"[CGC Layer] Each task: shared experts + own private expert (CGC routing)")

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_static: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> tuple:
        B = x_dyn.size(0)

        # Shared expert outputs
        shared_outputs = []
        for expert in self.shared_experts:
            H_shared = expert(x_dyn, lengths)  # [B, C, D]
            shared_outputs.append(H_shared)
        shared_outputs = torch.stack(shared_outputs, dim=1)  # [B, num_shared, C, D]

        # Private expert outputs
        private_outputs = {}
        for task_name in self.task_names:
            H_private = self.private_experts[task_name](x_dyn, lengths)  # [B, C, D]
            private_outputs[task_name] = H_private

        # Gate context
        c_gate = self.gate_context_encoder(x_dyn, x_static, lengths)  # [B, context_dim]

        # Per-task aggregation
        H_gated = {}
        gate_weights = {}

        for task_name in self.task_names:
            w_t = self.task_gates[task_name](c_gate)  # [B, num_candidates]
            gate_weights[task_name] = w_t

            # Build candidates: shared + own private
            private_t = private_outputs[task_name].unsqueeze(1)  # [B, 1, C, D]
            candidates_t = torch.cat([shared_outputs, private_t], dim=1)  # [B, num_shared+1, C, D]

            # Weighted sum
            w_t_expanded = w_t.unsqueeze(-1).unsqueeze(-1)  # [B, num_candidates, 1, 1]
            H_t = (candidates_t * w_t_expanded).sum(dim=1)  # [B, C, D]
            H_gated[task_name] = H_t

        return H_gated, gate_weights

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_parameter_breakdown(self) -> Dict[str, int]:
        breakdown = {}
        breakdown["shared_experts"] = sum(e.get_num_parameters() for e in self.shared_experts)
        breakdown["private_experts"] = sum(self.private_experts[t].get_num_parameters() for t in self.task_names)
        breakdown["task_gates"] = sum(sum(p.numel() for p in self.task_gates[t].parameters()) for t in self.task_names)
        breakdown["gate_context_encoder"] = sum(p.numel() for p in self.gate_context_encoder.parameters())
        breakdown["total"] = sum(breakdown.values())
        return breakdown


if __name__ == "__main__":
    print("=" * 60)
    print("CGC Common Module Test")
    print("=" * 60)

    cgc_layer = CGCLayer(num_channels=30, D_time=16, num_shared_experts=2, task_names=["t1", "t2", "t3", "t4", "t5"])
    print(f"\n{cgc_layer}")

    B, L = 8, 200
    x_dyn = torch.randn(B, L, 30)
    x_static = torch.randn(B, 5)

    H_gated, gate_weights = cgc_layer(x_dyn, x_static)

    print("\nForward pass results:")
    for task_name, H in H_gated.items():
        print(f"  {task_name}: H shape = {H.shape}")

    for task_name, w in gate_weights.items():
        w_sum = w.sum(dim=1)
        print(f"  {task_name}: w shape = {w.shape}, sum = {w_sum[0]:.4f}")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)