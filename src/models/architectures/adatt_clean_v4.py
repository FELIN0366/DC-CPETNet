"""
AdaTT Clean V4 Architecture - Adaptive Task-to-Task Fusion Baseline

AdaTT-style task-to-task fusion for CPET structured interpretation.

Key Features (Clean Architecture):
- Explicit architecture_type = "adatt_clean_v4"
- No YAML ablation flags
- No Alpha/Beta task-family split
- Each task has its own representation stream
- Task-to-task fusion via adaptive gates
- Optional shared stream for common knowledge
- Residual mechanism preserves task-specific knowledge
- No t6 context injection
- No t6 training

Architecture Overview:
    Input: [B, L, 30] + [B, 5]
        ↓
    Shared CPET Temporal Stem → H0 [B, 30, 16]
        ↓
    Task-specific Initial Streams (t1-t5 + shared)
        ↓
    AdaTT Fusion Layer 1 (task-to-task fusion + residual)
        ↓
    AdaTT Fusion Layer 2 (task-to-task fusion + residual)
        ↓
    Task-specific Towers/Heads (PMGT for t1, FlattenProjector for t2-t5)
        ↓
    Output: logits for t1-t5

Key Difference from MMoE/CGC:
- MMoE: Tasks select from shared expert pool (data-driven routing)
- CGC: Tasks select from shared + own private experts (semi-explicit routing)
- AdaTT: Tasks directly fuse with other task streams (explicit task-to-task fusion)
- Our Method: Medical prior drives Alpha/Beta split + t6 context

Author: AdaTT Clean Implementation
Date: 2026-05-27
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.common.adatt import (
    AdaTTTemporalStem,
    AdaTTTaskAdapter,
    AdaTTFusionLayer
)

from model_mtl import (
    PriorMaskedTaskHead,
    FlattenProjector,
    TaskStaticEncoder,
    TaskClassifier
)

from task_specs import TaskSpec


# =============================================================================
# Task Configuration for AdaTT
# =============================================================================

ADATT_TASK_CONFIG = {
    "t1": {"num_classes": 3, "is_binary": False, "use_pmgt": True, "display_name": "运动心功能分级", "branch": "alpha", "label_column": "运动心功能分级", "loss_name": "ce"},
    "t2": {"num_classes": 3, "is_binary": False, "use_pmgt": False, "display_name": "运动耐量", "branch": "beta", "label_column": "运动耐量", "loss_name": "ce"},
    "t3": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "标准心电运动负荷试验", "branch": "beta", "label_column": "标准心电运动负荷试验", "loss_name": "bce"},
    "t4": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "心率储备", "branch": "beta", "label_column": "心率储备", "loss_name": "ldam"},
    "t5": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "无氧阈", "branch": "beta", "label_column": "无氧阈", "loss_name": "ldam"},
}


# =============================================================================
# AdaTT Clean V4 Architecture
# =============================================================================

class AdaTTCleanV4Architecture(nn.Module):
    """
    AdaTT Clean V4 Architecture.

    Adaptive task-to-task fusion for CPET multi-task learning.

    Design Principles:
    1. Each task has its own representation stream (t1-t5 + shared)
    2. Task-to-task fusion via adaptive gates at each layer
    3. Residual mechanism preserves task-specific knowledge
    4. Reuse existing task heads: PMGT for t1, FlattenProjector for t2-t5
    5. No t6 context: t6 not trained, not used as context

    Input:
        x_dyn:    [B, L, 30] CPET dynamic sequence
        x_static: [B, 5]     static features
        lengths:  [B]        sequence lengths (optional)

    Output:
        Dict containing logits, dyn_feat, gate_weights for t1-t5

    Args:
        num_channels: Input channel count (default 30)
        D_time: Stream dimension (default 16)
        T_mid: Intermediate temporal dimension (default 24)
        num_fusion_layers: Number of AdaTT fusion layers (default 2)
        use_shared_stream: Whether to use shared stream (default True)
        semantic_adj: Prior adjacency matrix (optional, only for t1 PMGT)
        task_specs: Task specifications dict
        dropout: Dropout rate
        hidden_dim: Task head hidden dimension (default 48)
        static_dim: Static encoder output dimension (default 16)
        context_dim: Gate context dimension (default 40)
    """

    # Explicit architecture type (Clean Architecture)
    architecture_type: str = "adatt_clean_v4"

    # Explicit flags (no YAML branching)
    t6_deep_context_enabled: bool = False
    uses_alpha_beta_split: bool = False
    uses_expert_routing_constraints: bool = False

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        num_fusion_layers: int = 2,
        use_shared_stream: bool = True,
        semantic_adj: Optional[torch.Tensor] = None,
        task_specs: Optional[Dict[str, TaskSpec]] = None,
        dropout: float = 0.3,
        hidden_dim: int = 48,
        static_dim: int = 16,
        context_dim: int = 40,
        **kwargs  # Accept but ignore any extra kwargs (for config compatibility)
    ):
        super().__init__()

        self.num_channels = num_channels
        self.D_time = D_time
        self.num_fusion_layers = num_fusion_layers
        self.use_shared_stream = use_shared_stream
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim

        # AdaTT tasks (t1-t5, no t6)
        self.active_tasks = ["t1", "t2", "t3", "t4", "t5"]

        # Source names for gate weights
        self.source_names = list(self.active_tasks)
        if use_shared_stream:
            self.source_names.append("shared")

        # Semantic adjacency for t1 PMGT (optional)
        if semantic_adj is not None:
            self.semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32)
        else:
            self.semantic_adj = None

        # Build task specs if not provided
        if task_specs is None:
            task_specs = self._build_default_task_specs()

        self.task_specs = task_specs

        # === AdaTT Temporal Stem ===
        self.temporal_stem = AdaTTTemporalStem(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            dropout=dropout
        )

        # === Task-specific Initial Adapters ===
        self.task_adapters = nn.ModuleDict()
        for task in self.active_tasks:
            self.task_adapters[task] = AdaTTTaskAdapter(
                num_channels=num_channels,
                D_time=D_time,
                task_name=task,
                dropout=dropout * 0.7
            )

        # Shared adapter
        if use_shared_stream:
            self.shared_adapter = AdaTTTaskAdapter(
                num_channels=num_channels,
                D_time=D_time,
                task_name="shared",
                dropout=dropout * 0.7
            )

        # === AdaTT Fusion Layers ===
        self.fusion_layers = nn.ModuleList()
        for layer_idx in range(num_fusion_layers):
            fusion_layer = AdaTTFusionLayer(
                num_channels=num_channels,
                D_time=D_time,
                task_names=self.active_tasks,
                use_shared_stream=use_shared_stream,
                context_dim=context_dim,
                layer_idx=layer_idx,
                residual_scale=1.0,
                dropout=dropout * 0.7
            )
            self.fusion_layers.append(fusion_layer)

        # === Task-specific Towers/Heads ===
        # t1 uses PMGT (if semantic_adj provided), t2-t5 use FlattenProjector

        self.task_heads = nn.ModuleDict()

        # t1: PriorMaskedTaskHead (PMGT) or FlattenProjector
        if self.semantic_adj is not None:
            self.task_heads["t1"] = PriorMaskedTaskHead(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=hidden_dim,
                semantic_adj=self.semantic_adj,
                dropout=dropout * 1.5,
                gamma_init=1.0
            )
        else:
            self.task_heads["t1"] = FlattenProjector(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=hidden_dim,
                dropout=dropout,
                use_two_layer=True
            )

        # t2-t5: FlattenProjector
        for task_key in ["t2", "t3", "t4", "t5"]:
            self.task_heads[task_key] = FlattenProjector(
                num_nodes=num_channels,
                hidden_dim=D_time,
                out_dim=hidden_dim,
                dropout=dropout,
                use_two_layer=True
            )

        # === Static Encoders ===
        self.static_encoders = nn.ModuleDict()
        for task_key in self.active_tasks:
            self.static_encoders[task_key] = TaskStaticEncoder(
                in_dim=kwargs.get('num_static_features', 5),
                out_dim=static_dim,
                dropout=dropout * 0.7
            )

        # === Classifiers ===
        self.classifiers = nn.ModuleDict()
        for task_key in self.active_tasks:
            task_spec = task_specs.get(task_key)
            if task_spec:
                num_classes = task_spec.num_classes
                is_binary = task_spec.is_binary
            else:
                num_classes = ADATT_TASK_CONFIG[task_key]["num_classes"]
                is_binary = ADATT_TASK_CONFIG[task_key]["is_binary"]

            self.classifiers[task_key] = TaskClassifier(
                dyn_dim=hidden_dim,
                static_dim=static_dim,
                num_classes=num_classes,
                is_binary=is_binary,
                dropout=dropout
            )

        print(f"\n[AdaTT Clean V4] Architecture initialized:")
        print(f"  architecture_type = {self.architecture_type}")
        print(f"  num_fusion_layers = {self.num_fusion_layers}")
        print(f"  use_shared_stream = {self.use_shared_stream}")
        print(f"  source_names = {self.source_names}")
        print(f"  active_tasks = {self.active_tasks}")
        print(f"  t6_deep_context_enabled = {self.t6_deep_context_enabled}")
        print(f"  Task-to-task fusion: each task adaptively fuses from other task streams")

    def _build_default_task_specs(self) -> Dict[str, TaskSpec]:
        """Build default task specs for AdaTT tasks."""
        task_specs = {}
        for task_key, config in ADATT_TASK_CONFIG.items():
            task_specs[task_key] = TaskSpec(
                name=task_key,
                display_name=config["display_name"],
                num_classes=config["num_classes"],
                branch=config["branch"],
                loss_name=config["loss_name"],
                is_binary=config["is_binary"],
                dropout=0.3,
                label_column=config["label_column"]
            )
        return task_specs

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_static: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through AdaTT Clean V4.

        Args:
            x_dyn: [B, L, 30] dynamic sequence
            x_static: [B, 5] static features
            lengths: [B] sequence lengths (optional)

        Returns:
            Dict containing:
                - t1, t2, t3, t4, t5: Dict with logits, dyn_feat, gate_weights
                - _aux: Auxiliary info (architecture_type, flags, fusion layer gate weights)
        """
        B = x_dyn.size(0)

        # === Step 1: Shared Temporal Stem ===
        H0 = self.temporal_stem(x_dyn, lengths)  # [B, C, D]

        # === Step 2: Build Initial Streams ===
        streams = {}

        # Task-specific initial streams
        for task in self.active_tasks:
            streams[task] = self.task_adapters[task](H0)  # H_t^0

        # Shared stream (optional)
        if self.use_shared_stream:
            streams["shared"] = self.shared_adapter(H0)  # H_shared^0

        # === Step 3: AdaTT Fusion Layers ===
        all_gate_weights = {}

        for layer_idx, fusion_layer in enumerate(self.fusion_layers):
            streams, aux_l = fusion_layer(
                streams=streams,
                x_static=x_static,
                lengths=lengths
            )
            all_gate_weights[f"layer_{layer_idx + 1}"] = aux_l["gate_weights"]

        # === Step 4: Task-specific Heads ===
        outputs = {}

        # t1: PMGT head
        H_t1 = streams["t1"]  # [B, C, D]
        dyn_feat_t1 = self.task_heads["t1"](H_t1)  # [B, hidden_dim]
        static_feat_t1 = self.static_encoders["t1"](x_static)  # [B, static_dim]
        logits_t1 = self.classifiers["t1"](dyn_feat_t1, static_feat_t1)

        outputs["t1"] = {
            "logits": logits_t1,
            "dyn_feat": dyn_feat_t1,
            "static_feat": static_feat_t1,
            "fused_feat": torch.cat([dyn_feat_t1, static_feat_t1], dim=1),
            "gate_weights": all_gate_weights.get("layer_2", {}).get("t1")  # Last layer gate weights
        }

        # t2-t5: FlattenProjector heads
        for task in ["t2", "t3", "t4", "t5"]:
            H_t = streams[task]  # [B, C, D]
            dyn_feat = self.task_heads[task](H_t)  # [B, hidden_dim]
            static_feat = self.static_encoders[task](x_static)  # [B, static_dim]
            logits = self.classifiers[task](dyn_feat, static_feat)

            outputs[task] = {
                "logits": logits,
                "dyn_feat": dyn_feat,
                "static_feat": static_feat,
                "fused_feat": torch.cat([dyn_feat, static_feat], dim=1),
                "gate_weights": all_gate_weights.get("layer_2", {}).get(task)  # Last layer gate weights
            }

        # === Step 5: Auxiliary Output ===
        outputs["_aux"] = {
            "architecture_type": self.architecture_type,
            "t6_deep_context_enabled": self.t6_deep_context_enabled,
            "uses_alpha_beta_split": self.uses_alpha_beta_split,
            "uses_expert_routing_constraints": self.uses_expert_routing_constraints,
            "num_fusion_layers": self.num_fusion_layers,
            "use_shared_stream": self.use_shared_stream,
            "source_names": self.source_names,
            "active_tasks": self.active_tasks,
            "adatt_gate_weights": all_gate_weights,  # All fusion layer gate weights
        }

        return outputs

    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts by module."""
        counts = {}

        # Temporal stem
        counts["temporal_stem"] = self.temporal_stem.get_num_parameters()

        # Task adapters
        counts["task_adapters"] = sum(
            self.task_adapters[t].get_num_parameters() for t in self.active_tasks
        )

        # Shared adapter (if exists)
        if self.use_shared_stream:
            counts["shared_adapter"] = self.shared_adapter.get_num_parameters()

        # Fusion layers
        counts["fusion_layers"] = sum(
            layer.get_num_parameters() for layer in self.fusion_layers
        )

        # Task heads
        counts["task_heads"] = sum(
            sum(p.numel() for p in self.task_heads[k].parameters())
            for k in self.task_heads.keys()
        )

        # Static encoders
        counts["static_encoders"] = sum(
            sum(p.numel() for p in self.static_encoders[k].parameters())
            for k in self.static_encoders.keys()
        )

        # Classifiers
        counts["classifiers"] = sum(
            sum(p.numel() for p in self.classifiers[k].parameters())
            for k in self.classifiers.keys()
        )

        # Total
        counts["total"] = sum(counts.values())

        return counts

    def freeze_modules(self, module_names: List[str]) -> None:
        """
        Freeze specified modules (for staged training).

        Args:
            module_names: List of module names to freeze
        """
        for name in module_names:
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.requires_grad = False
                    print(f"[AdaTT Clean] Frozen: {name}")

    def unfreeze_modules(self, module_names: List[str]) -> None:
        """
        Unfreeze specified modules.

        Args:
            module_names: List of module names to unfreeze
        """
        for name in module_names:
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"[AdaTT Clean] Unfrozen: {name}")

    def get_active_tasks(self) -> List[str]:
        """Get list of active tasks."""
        return self.active_tasks

    def __repr__(self) -> str:
        counts = self.get_num_parameters()
        return (
            f"AdaTTCleanV4Architecture("
            f"architecture_type='{self.architecture_type}', "
            f"num_fusion_layers={self.num_fusion_layers}, "
            f"use_shared_stream={self.use_shared_stream}, "
            f"source_names={self.source_names}, "
            f"params={counts['total']:,})"
        )


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AdaTT Clean V4 Architecture Test")
    print("=" * 60)

    # Create model
    model = AdaTTCleanV4Architecture(
        num_channels=30,
        D_time=16,
        num_fusion_layers=2,
        use_shared_stream=True
    )

    print(f"\n{model}")

    # Count parameters
    counts = model.get_num_parameters()
    print(f"\nParameter counts:")
    for module_name, count in counts.items():
        print(f"  {module_name}: {count:,}")

    # Test forward
    B, L = 8, 200
    x_dyn = torch.randn(B, L, 30)
    x_static = torch.randn(B, 5)

    outputs = model(x_dyn, x_static)

    print("\nForward pass results:")
    for task_key in model.active_tasks:
        logits = outputs[task_key]["logits"]
        gate_weights = outputs[task_key]["gate_weights"]
        print(f"  {task_key}: logits={logits.shape}")
        if gate_weights is not None:
            w_sum = gate_weights.sum(dim=1)
            print(f"  {task_key}: gate_weights shape={gate_weights.shape}, sum[0]={w_sum[0]:.4f}")

    # Check auxiliary output
    aux = outputs["_aux"]
    print(f"\nAuxiliary output:")
    print(f"  architecture_type = {aux['architecture_type']}")
    print(f"  t6_deep_context_enabled = {aux['t6_deep_context_enabled']}")
    print(f"  num_fusion_layers = {aux['num_fusion_layers']}")
    print(f"  source_names = {aux['source_names']}")
    print(f"  adatt_gate_weights keys = {list(aux['adatt_gate_weights'].keys())}")

    # Check gate weights for each layer
    for layer_name, layer_gates in aux["adatt_gate_weights"].items():
        print(f"\n  {layer_name}:")
        for target, weights in layer_gates.items():
            w_sum = weights.sum(dim=1)
            print(f"    {target}: shape={weights.shape}, sum[0]={w_sum[0]:.4f}")

    # Test backward
    loss = sum(outputs[t]["logits"].sum() for t in model.active_tasks)
    loss.backward()
    print("\nBackward pass successful")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)