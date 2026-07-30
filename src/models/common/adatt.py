"""
AdaTT Common Module - Adaptive Task-to-Task Fusion Components

AdaTT-style Task-to-Task Fusion Baseline (simplified for CPET structured interpretation).

Key Design (vs MMoE/CGC):
- MMoE: Tasks select from shared expert pool
- CGC: Tasks select from shared + own private experts
- AdaTT: Tasks directly fuse with other task streams + optional shared stream

Components:
- AdaTTTemporalStem: Shared initial CPET temporal encoder
- AdaTTTaskAdapter: Build task-specific initial stream from shared H0
- AdaTTFusionUnit: Task-specific or shared fusion unit
- AdaTTGateContextEncoder: Sample-wise gate context for each task
- AdaTTTaskGate: Task-specific gate over task streams + shared stream
- AdaTTFusionLayer: One AdaTT task-to-task fusion layer

Author: AdaTT Clean Implementation
Date: 2026-05-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any


class AdaTTTemporalStem(nn.Module):
    """
    Shared initial CPET temporal encoder.

    All tasks start from this shared temporal representation H0.

    Input:  x_dyn [B, L, C] - CPET dynamic sequence
    Output: H0    [B, C, D] - Shared initial representation

    Design: Single-scale CNN with depthwise separable convolution.
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

        print(f"\n[AdaTT Temporal Stem] Initialized:")
        print(f"  num_channels = {num_channels}")
        print(f"  D_time = {D_time}")
        print(f"  T_mid = {T_mid}")
        print(f"  All tasks share this initial encoder")

    def forward(
        self,
        x_dyn: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_dyn: [B, L, C] dynamic sequence
            lengths: [B] sequence lengths (optional)

        Returns:
            H0: [B, C, D_time] shared initial representation
        """
        B, L, C = x_dyn.shape
        x = x_dyn.transpose(1, 2)  # [B, C, L]
        x = self.conv1(x)  # [B, 2C, L/2]
        x = self.conv2(x)  # [B, 2C, L/4]
        x = x.view(B, C, -1)  # [B, C, 2*L']
        x = self.adaptive_pool(x)  # [B, C, T_mid]
        H0 = self.proj(x)  # [B, C, D_time]
        return H0

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AdaTTTaskAdapter(nn.Module):
    """
    Build task-specific initial stream from shared H0.

    H_t^0 = Adapter_t(H0) + H0

    This creates task-specific representation streams that start from
    shared knowledge but immediately diverge with task-specific adapters.

    Input:  H0   [B, C, D] - Shared initial representation
    Output: H_t0 [B, C, D] - Task-specific initial stream
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        task_name: str = "unknown",
        adapter_hidden: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.task_name = task_name
        self.num_channels = num_channels
        self.D_time = D_time

        # Adapter: lightweight transform
        self.adapter = nn.Sequential(
            nn.Linear(D_time, adapter_hidden),
            nn.LayerNorm(adapter_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden, D_time),
            nn.LayerNorm(D_time)
        )

        # Initialize near identity for residual stability
        nn.init.xavier_uniform_(self.adapter[0].weight, gain=0.1)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.xavier_uniform_(self.adapter[4].weight, gain=0.1)
        nn.init.zeros_(self.adapter[4].bias)

        print(f"[AdaTT Task Adapter] {task_name}: adapter_hidden={adapter_hidden}")

    def forward(self, H0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H0: [B, C, D_time] shared initial representation

        Returns:
            H_t0: [B, C, D_time] task-specific initial stream (with residual)
        """
        # Adapter transform
        H_adapted = self.adapter(H0)  # [B, C, D_time]

        # Residual connection: preserves shared knowledge
        H_t0 = H0 + H_adapted

        return H_t0

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AdaTTFusionUnit(nn.Module):
    """
    One task-specific or shared fusion unit.

    Each stream (t1, t2, t3, t4, t5, shared) has its own fusion unit
    that transforms the stream representation before fusion.

    Input:  H [B, C, D] - Stream representation
    Output: U [B, C, D] - Fusion-ready representation
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        unit_name: str = "unknown",
        hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.unit_name = unit_name

        self.fusion_transform = nn.Sequential(
            nn.Linear(D_time, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, D_time),
            nn.LayerNorm(D_time),
            nn.ReLU()
        )

        nn.init.xavier_uniform_(self.fusion_transform[0].weight)
        nn.init.zeros_(self.fusion_transform[0].bias)
        nn.init.xavier_uniform_(self.fusion_transform[4].weight)
        nn.init.zeros_(self.fusion_transform[4].bias)

        print(f"[AdaTT Fusion Unit] {unit_name}: hidden_dim={hidden_dim}")

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: [B, C, D_time] stream representation

        Returns:
            U: [B, C, D_time] fusion-ready representation
        """
        return self.fusion_transform(H)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AdaTTGateContextEncoder(nn.Module):
    """
    Build sample-wise gate context for each target task.

    For each task, the gate context combines:
    - Task stream representation (H_t)
    - Static features (x_static)

    This enables sample-adaptive fusion weights.

    Input:  H_t [B, C, D] + x_static [B, 5]
    Output: c_t [B, context_dim]
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        static_dim: int = 5,
        context_dim: int = 40,
        task_name: str = "unknown",
        dropout: float = 0.2
    ):
        super().__init__()
        self.task_name = task_name
        self.context_dim = context_dim

        # Dynamic context from task stream
        self.dyn_proj = nn.Sequential(
            nn.Linear(num_channels * D_time, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, context_dim // 2)
        )

        # Static context
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, context_dim // 2)
        )

        print(f"[AdaTT Gate Context Encoder] {task_name}: context_dim={context_dim}")

    def forward(
        self,
        H_t: torch.Tensor,
        x_static: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            H_t: [B, C, D_time] task stream representation
            x_static: [B, static_dim] static features
            lengths: [B] sequence lengths (optional)

        Returns:
            c_t: [B, context_dim] gate context for this task
        """
        B, C, D = H_t.shape

        # Flatten task stream for context
        H_flat = H_t.view(B, C * D)  # [B, C*D]
        dyn_context = self.dyn_proj(H_flat)  # [B, context_dim//2]

        static_context = self.static_proj(x_static)  # [B, context_dim//2]

        c_t = torch.cat([dyn_context, static_context], dim=1)  # [B, context_dim]

        return c_t

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AdaTTTaskGate(nn.Module):
    """
    Task-specific gate over task streams + optional shared stream.

    Each task gate computes weights over 6 sources:
    - t1, t2, t3, t4, t5 (other task streams)
    - shared (optional shared stream)

    Output: weights [B, num_sources] where num_sources = 6
    """

    def __init__(
        self,
        context_dim: int = 40,
        num_sources: int = 6,
        task_name: str = "unknown",
        tau_init: float = 1.0
    ):
        super().__init__()
        self.num_sources = num_sources
        self.task_name = task_name

        self.gate_proj = nn.Linear(context_dim, num_sources)
        self.tau = tau_init

        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

        print(f"[AdaTT Task Gate] {task_name}: num_sources={num_sources}, tau={tau_init}")

    def forward(self, c_gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_gate: [B, context_dim] gate context

        Returns:
            weights: [B, num_sources] softmax weights over all sources
        """
        gate_logits = self.gate_proj(c_gate)  # [B, num_sources]
        weights = F.softmax(gate_logits / self.tau, dim=-1)  # [B, num_sources]
        return weights

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AdaTTFusionLayer(nn.Module):
    """
    One AdaTT task-to-task fusion layer.

    Structure:
    1. Each stream (t1-t5, shared) passes through its own fusion unit
    2. Each target task computes gate weights over all source streams
    3. Each target task fuses information from weighted sources
    4. Residual update preserves original task knowledge

    Input streams:
        {
            "t1": H_t1^(l-1),
            "t2": H_t2^(l-1),
            "t3": H_t3^(l-1),
            "t4": H_t4^(l-1),
            "t5": H_t5^(l-1),
            "shared": H_shared^(l-1),
        }

    Output:
        new_streams, gate_weights
    """

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        task_names: List[str] = ["t1", "t2", "t3", "t4", "t5"],
        use_shared_stream: bool = True,
        context_dim: int = 40,
        layer_idx: int = 0,
        residual_scale: float = 1.0,
        dropout: float = 0.2
    ):
        super().__init__()
        self.task_names = task_names
        self.use_shared_stream = use_shared_stream
        self.layer_idx = layer_idx
        self.residual_scale = residual_scale

        # Build source names: task streams + optional shared
        self.source_names = list(task_names)
        if use_shared_stream:
            self.source_names.append("shared")
        self.num_sources = len(self.source_names)

        # Fusion units for each source stream
        self.units = nn.ModuleDict()
        for source in self.source_names:
            self.units[source] = AdaTTFusionUnit(
                num_channels=num_channels,
                D_time=D_time,
                unit_name=f"{source}_layer{layer_idx}",
                dropout=dropout
            )

        # Gate context encoders for each target task (and optional shared)
        self.context_encoders = nn.ModuleDict()
        for target in self.source_names:
            self.context_encoders[target] = AdaTTGateContextEncoder(
                num_channels=num_channels,
                D_time=D_time,
                context_dim=context_dim,
                task_name=target,
                dropout=dropout
            )

        # Gates for each target task
        self.gates = nn.ModuleDict()
        for target in self.source_names:
            self.gates[target] = AdaTTTaskGate(
                context_dim=context_dim,
                num_sources=self.num_sources,
                task_name=target,
                tau_init=1.0
            )

        # LayerNorm for residual updates
        self.norms = nn.ModuleDict()
        for target in self.source_names:
            self.norms[target] = nn.LayerNorm([num_channels, D_time])

        print(f"\n[AdaTT Fusion Layer {layer_idx}] Initialized:")
        print(f"  task_names = {task_names}")
        print(f"  use_shared_stream = {use_shared_stream}")
        print(f"  source_names = {self.source_names}")
        print(f"  num_sources = {self.num_sources}")
        print(f"  residual_scale = {residual_scale}")

    def forward(
        self,
        streams: Dict[str, torch.Tensor],
        x_static: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            streams: Dict of stream representations {source: H_source}
            x_static: [B, static_dim] static features
            lengths: [B] sequence lengths (optional)

        Returns:
            new_streams: Dict of updated streams
            aux: Dict containing gate_weights
        """
        B = streams["t1"].size(0)
        C = streams["t1"].size(1)
        D = streams["t1"].size(2)

        # Step 1: Each source stream through its fusion unit
        candidates = {}
        for source in self.source_names:
            candidates[source] = self.units[source](streams[source])

        # Stack candidates: [B, num_sources, C, D]
        candidate_stack = torch.stack(
            [candidates[src] for src in self.source_names],
            dim=1
        )

        # Step 2: Each target task independent gate and fusion
        new_streams = {}
        gate_weights = {}

        for target in self.source_names:
            # Gate context for this target
            context = self.context_encoders[target](
                streams[target], x_static, lengths=lengths
            )

            # Gate weights over all sources
            weights = self.gates[target](context)  # [B, num_sources]
            gate_weights[target] = weights

            # Weighted fusion
            # weights: [B, num_sources] -> [B, num_sources, 1, 1]
            weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)
            fused = torch.sum(
                candidate_stack * weights_expanded,
                dim=1
            )  # [B, C, D]

            # Residual update with LayerNorm
            new_streams[target] = self.norms[target](
                streams[target] + self.residual_scale * fused
            )

        return new_streams, {"gate_weights": gate_weights}

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_parameter_breakdown(self) -> Dict[str, int]:
        breakdown = {}
        breakdown["fusion_units"] = sum(self.units[src].get_num_parameters() for src in self.source_names)
        breakdown["context_encoders"] = sum(self.context_encoders[t].get_num_parameters() for t in self.source_names)
        breakdown["gates"] = sum(self.gates[t].get_num_parameters() for t in self.source_names)
        breakdown["norms"] = sum(sum(p.numel() for p in self.norms[t].parameters()) for t in self.source_names)
        breakdown["total"] = sum(breakdown.values())
        return breakdown


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AdaTT Common Module Test")
    print("=" * 60)

    # Create components
    stem = AdaTTTemporalStem(num_channels=30, D_time=16)

    task_adapters = nn.ModuleDict({
        task: AdaTTTaskAdapter(30, 16, task)
        for task in ["t1", "t2", "t3", "t4", "t5"]
    })

    shared_adapter = AdaTTTaskAdapter(30, 16, "shared")

    fusion_layer = AdaTTFusionLayer(
        num_channels=30,
        D_time=16,
        task_names=["t1", "t2", "t3", "t4", "t5"],
        use_shared_stream=True,
        layer_idx=0
    )

    # Test forward
    B, L = 8, 200
    x_dyn = torch.randn(B, L, 30)
    x_static = torch.randn(B, 5)

    # Step 1: Temporal stem
    H0 = stem(x_dyn)
    print(f"\nTemporal Stem Output: H0 shape = {H0.shape}")

    # Step 2: Task adapters
    streams = {}
    for task in ["t1", "t2", "t3", "t4", "t5"]:
        streams[task] = task_adapters[task](H0)
    streams["shared"] = shared_adapter(H0)

    print(f"\nTask Adapter Outputs:")
    for task, H in streams.items():
        print(f"  {task}: shape = {H.shape}")

    # Step 3: Fusion layer
    new_streams, aux = fusion_layer(streams, x_static)

    print(f"\nFusion Layer Output:")
    for task, H in new_streams.items():
        print(f"  {task}: shape = {H.shape}")

    print(f"\nGate Weights:")
    for task, weights in aux["gate_weights"].items():
        w_sum = weights.sum(dim=1)
        print(f"  {task}: weights shape = {weights.shape}, sum[0] = {w_sum[0]:.4f}")

    # Test backward
    loss = sum(H.sum() for H in new_streams.values())
    loss.backward()
    print("\nBackward pass successful")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)