"""
MMoE Clean V4 Architecture - Multi-gate Mixture-of-Experts Baseline

This architecture implements the classic MMoE approach for task-sharing strategy
comparison in Result2.

Key Features (Clean Architecture):
- Explicit architecture_type = "mmoe_clean_v4"
- No YAML ablation flags
- No Alpha/Beta task-family split
- Shared expert pool for all tasks (t1-t5)
- One task-specific gate per task
- All tasks can access all experts
- No t6 context injection
- No t6 training

Architecture Overview:
    Input: [B, L, 30] + [B, 5]
        ↓
    MMoE Shared Expert Pool (4 experts)
        ↓
    Task-specific Gates (t1-t5 each has its own gate)
        ↓
    Task-specific Towers/Heads
        ↓
    Output: logits for t1-t5

Author: MMoE Clean Implementation
Date: 2026-05-25
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

# Import MMoE components
from ..common.mmoe import MMoELayer

# Import existing task heads (reuse from model_mtl.py)
from model_mtl import (
    PriorMaskedTaskHead,
    FlattenProjector,
    TaskStaticEncoder,
    TaskClassifier
)

from task_specs import TaskSpec


# =============================================================================
# Task Configuration for MMoE
# =============================================================================

MMOE_TASK_CONFIG = {
    "t1": {"num_classes": 3, "is_binary": False, "use_pmgt": True, "display_name": "运动心功能分级", "branch": "alpha", "label_column": "运动心功能分级", "loss_name": "ce"},
    "t2": {"num_classes": 3, "is_binary": False, "use_pmgt": False, "display_name": "运动耐量", "branch": "beta", "label_column": "运动耐量", "loss_name": "ce"},
    "t3": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "标准心电运动负荷试验", "branch": "beta", "label_column": "标准心电运动负荷试验", "loss_name": "bce"},
    "t4": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "心率储备", "branch": "beta", "label_column": "心率储备", "loss_name": "ldam"},
    "t5": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "无氧阈", "branch": "beta", "label_column": "无氧阈", "loss_name": "ldam"},
}


# =============================================================================
# MMoE Clean V4 Architecture
# =============================================================================

class MMoECleanV4Architecture(nn.Module):
    """
    MMoE Clean V4 Architecture.

    Data-driven task-sharing baseline using Multi-gate Mixture-of-Experts.

    Design Principles:
    1. Shared expert pool: All t1-t5 tasks share 4 experts
    2. Task-specific gates: Each task has its own gate
    3. All tasks can access all experts (no routing constraints)
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
        D_time: Expert output dimension (default 16)
        T_mid: Intermediate temporal dimension (default 24)
        num_experts: Number of shared experts (default 4)
        semantic_adj: Prior adjacency matrix (optional, only for t1 PMGT)
        task_specs: Task specifications dict
        dropout: Dropout rate
        hidden_dim: Task head hidden dimension (default 48)
    """

    # Explicit architecture type (Clean Architecture)
    architecture_type: str = "mmoe_clean_v4"

    # Explicit flags (no YAML branching)
    t6_deep_context_enabled: bool = False
    uses_alpha_beta_split: bool = False
    uses_expert_routing_constraints: bool = False

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        num_experts: int = 4,
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
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim

        # MMoE tasks (t1-t5, no t6)
        self.active_tasks = ["t1", "t2", "t3", "t4", "t5"]

        # Semantic adjacency for t1 PMGT (optional)
        if semantic_adj is not None:
            self.semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32)
        else:
            self.semantic_adj = None

        # Build task specs if not provided
        if task_specs is None:
            task_specs = self._build_default_task_specs()

        self.task_specs = task_specs

        # === MMoE Core Layer ===
        self.mmoe_layer = MMoELayer(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            num_experts=num_experts,
            task_names=self.active_tasks,
            context_dim=context_dim,
            dropout=dropout
        )

        # === Task-specific Towers/Heads ===
        # t1 uses PMGT, t2-t5 use FlattenProjector

        self.task_heads = nn.ModuleDict()

        # t1: PriorMaskedTaskHead (PMGT)
        self.task_heads["t1"] = PriorMaskedTaskHead(
            num_nodes=num_channels,
            hidden_dim=D_time,
            out_dim=hidden_dim,
            semantic_adj=self.semantic_adj,
            dropout=dropout
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
                dropout=dropout
            )

        # === Classifiers ===
        self.classifiers = nn.ModuleDict()
        for task_key in self.active_tasks:
            task_spec = task_specs.get(task_key)
            if task_spec:
                num_classes = task_spec.num_classes
                is_binary = task_spec.is_binary
            else:
                num_classes = MMOE_TASK_CONFIG[task_key]["num_classes"]
                is_binary = MMOE_TASK_CONFIG[task_key]["is_binary"]

            self.classifiers[task_key] = TaskClassifier(
                dyn_dim=hidden_dim,
                static_dim=static_dim,
                num_classes=num_classes,
                is_binary=is_binary,
                dropout=dropout
            )

        print(f"\n[MMoE Clean V4] Architecture initialized:")
        print(f"  architecture_type = {self.architecture_type}")
        print(f"  num_experts = {self.num_experts}")
        print(f"  active_tasks = {self.active_tasks}")
        print(f"  t6_deep_context_enabled = {self.t6_deep_context_enabled}")
        print(f"  All tasks can access ALL experts (no routing restrictions)")

    def _build_default_task_specs(self) -> Dict[str, TaskSpec]:
        """Build default task specs for MMoE tasks."""
        task_specs = {}
        for task_key, config in MMOE_TASK_CONFIG.items():
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
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass through MMoE Clean V4.

        Args:
            x_dyn: [B, L, 30] dynamic sequence
            x_static: [B, 5] static features
            lengths: [B] sequence lengths (optional)

        Returns:
            Dict containing:
                - t1, t2, t3, t4, t5: Dict with logits, dyn_feat, gate_weights
                - _aux: Auxiliary info (architecture_type, flags)
        """
        # === MMoE Layer ===
        mmoe_outputs = self.mmoe_layer(x_dyn, x_static, lengths)
        H_gated = mmoe_outputs["H_gated"]
        gate_weights = mmoe_outputs["gate_weights"]

        # === Task-specific Processing ===
        outputs = {}

        for task_key in self.active_tasks:
            # Get gated representation
            H_t = H_gated[task_key]  # [B, C, D_time]

            # Process through task head
            dyn_feat = self.task_heads[task_key](H_t)  # [B, hidden_dim]

            # Encode static features
            static_feat = self.static_encoders[task_key](x_static)  # [B, static_dim]

            # Classify
            logits = self.classifiers[task_key](dyn_feat, static_feat)

            outputs[task_key] = {
                "logits": logits,
                "dyn_feat": dyn_feat,
                "static_feat": static_feat,
                "gate_weights": gate_weights[task_key]
            }

        # === Auxiliary Output ===
        outputs["_aux"] = {
            "architecture_type": self.architecture_type,
            "t6_deep_context_enabled": self.t6_deep_context_enabled,
            "uses_alpha_beta_split": self.uses_alpha_beta_split
        }

        return outputs

    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts by module (matching other MTL architectures)."""
        counts = {}

        # MMoE Layer (experts + gates)
        counts["mmoe_layer"] = sum(p.numel() for p in self.mmoe_layer.parameters())

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
                    print(f"[MMoE Clean] Frozen: {name}")

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
                    print(f"[MMoE Clean] Unfrozen: {name}")

    def get_active_tasks(self) -> List[str]:
        """Get list of active tasks."""
        return self.active_tasks

    def get_num_experts(self) -> int:
        """Get number of experts."""
        return self.num_experts