"""
Shared-bottom Clean V4 Architecture - Hard Parameter Sharing Baseline

This architecture implements the classic Shared-bottom approach for task-sharing
strategy comparison in Result2.

Key Features (Clean Architecture):
- Explicit architecture_type = "shared_bottom_clean_v4"
- No YAML ablation flags
- No Alpha/Beta task-family split
- Single shared encoder for all tasks (t1-t5)
- Task-specific towers/heads only
- No expert pool, no gates
- No t6 context injection
- No t6 training

Architecture Overview:
    Input: [B, L, 30] + [B, 5]
        ↓
    Shared Bottom Dynamic Encoder (single encoder)
        ↓
    H_shared [B, 30, 16] (same representation for all tasks)
        ↓
    Task-specific Towers/Heads:
        - t1: PMGT + classifier
        - t2-t5: FlattenProjector + classifier
        ↓
    Output: logits for t1-t5

This is the simplest MTL baseline - all tasks forced to share the same
encoder representation. Used to test:
- Can t1-t5 all share the same dynamic encoder without conflicts?
- Does MMoE's learned routing provide value over hard sharing?
- Does our dual-engine medical task grouping provide additional benefits?

Author: Shared-bottom Clean Implementation
Date: 2026-05-26
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

# Import Shared-bottom components
from ..common.shared_bottom import SharedBottomDynamicEncoder

# Import existing task heads (reuse from model_mtl.py)
from model_mtl import (
    PriorMaskedTaskHead,
    FlattenProjector,
    TaskStaticEncoder,
    TaskClassifier
)

from task_specs import TaskSpec


# =============================================================================
# Task Configuration for Shared-bottom
# =============================================================================

SHARED_BOTTOM_TASK_CONFIG = {
    "t1": {"num_classes": 3, "is_binary": False, "use_pmgt": True, "display_name": "运动心功能分级", "branch": "alpha", "label_column": "运动心功能分级", "loss_name": "ce"},
    "t2": {"num_classes": 3, "is_binary": False, "use_pmgt": False, "display_name": "运动耐量", "branch": "beta", "label_column": "运动耐量", "loss_name": "ce"},
    "t3": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "标准心电运动负荷试验", "branch": "beta", "label_column": "标准心电运动负荷试验", "loss_name": "bce"},
    "t4": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "心率储备", "branch": "beta", "label_column": "心率储备", "loss_name": "ldam"},
    "t5": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "无氧阈", "branch": "beta", "label_column": "无氧阈", "loss_name": "ldam"},
}


# =============================================================================
# Shared-bottom Clean V4 Architecture
# =============================================================================

class SharedBottomCleanV4Architecture(nn.Module):
    """
    Shared-bottom Clean V4 Architecture.

    Hard parameter sharing MTL baseline - all tasks share the same encoder.

    Design Principles:
    1. One shared bottom encoder: All t1-t5 tasks use H_shared
    2. Task-specific towers: Only towers differ per task
    3. Reuse existing task heads: PMGT for t1, FlattenProjector for t2-t5
    4. No t6 context: t6 not trained, not used as context
    5. No routing/gates/experts: Pure hard sharing

    Input:
        x_dyn:    [B, L, 30] CPET dynamic sequence
        x_static: [B, 5]     static features
        lengths:  [B]        sequence lengths (optional)

    Output:
        Dict containing logits, dyn_feat, shared_feat for t1-t5

    Args:
        num_channels: Input channel count (default 30)
        D_time: Encoder output dimension (default 16)
        T_mid: Intermediate temporal dimension (default 24)
        semantic_adj: Prior adjacency matrix (optional, only for t1 PMGT)
        task_specs: Task specifications dict
        dropout: Dropout rate
        hidden_dim: Task head hidden dimension (default 48)
        static_dim: Static feature dimension (default 16)
    """

    # Explicit architecture type (Clean Architecture)
    architecture_type: str = "shared_bottom_clean_v4"

    # Explicit flags (no YAML branching)
    t6_deep_context_enabled: bool = False
    uses_alpha_beta_split: bool = False
    uses_expert_routing_constraints: bool = False
    uses_expert_pool: bool = False
    uses_task_gates: bool = False

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        semantic_adj: Optional[torch.Tensor] = None,
        task_specs: Optional[Dict[str, TaskSpec]] = None,
        dropout: float = 0.3,
        hidden_dim: int = 48,
        static_dim: int = 16,
        **kwargs  # Accept but ignore any extra kwargs (for config compatibility)
    ):
        super().__init__()

        self.num_channels = num_channels
        self.D_time = D_time
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim

        # Shared-bottom tasks (t1-t5, no t6)
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

        print(f"\n[Shared-bottom Clean V4] Architecture initialized:")
        print(f"  architecture_type = {self.architecture_type}")
        print(f"  num_channels = {num_channels}")
        print(f"  D_time = {D_time}")
        print(f"  active_tasks = {self.active_tasks}")
        print(f"  t6_deep_context_enabled = {self.t6_deep_context_enabled}")
        print(f"  All tasks share SINGLE encoder (no routing, no gates, no experts)")

        # === Shared Bottom Dynamic Encoder ===
        # All tasks share this single encoder
        self.shared_bottom_encoder = SharedBottomDynamicEncoder(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
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
        # Each task has its own static encoder (task-specific, not shared)
        self.static_encoders = nn.ModuleDict()
        for task_key in self.active_tasks:
            self.static_encoders[task_key] = TaskStaticEncoder(
                in_dim=kwargs.get('num_static_features', 5),
                out_dim=static_dim,
                dropout=dropout
            )

        # === Classifiers ===
        # Each task has its own classifier (task-specific, not shared)
        self.classifiers = nn.ModuleDict()
        for task_key in self.active_tasks:
            task_spec = task_specs.get(task_key)
            if task_spec:
                num_classes = task_spec.num_classes
                is_binary = task_spec.is_binary
            else:
                num_classes = SHARED_BOTTOM_TASK_CONFIG[task_key]["num_classes"]
                is_binary = SHARED_BOTTOM_TASK_CONFIG[task_key]["is_binary"]

            self.classifiers[task_key] = TaskClassifier(
                dyn_dim=hidden_dim,
                static_dim=static_dim,
                num_classes=num_classes,
                is_binary=is_binary,
                dropout=dropout
            )

    def _build_default_task_specs(self) -> Dict[str, TaskSpec]:
        """Build default task specs for Shared-bottom tasks."""
        task_specs = {}
        for task_key, config in SHARED_BOTTOM_TASK_CONFIG.items():
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
        Forward pass through Shared-bottom Clean V4.

        Args:
            x_dyn: [B, L, 30] dynamic sequence
            x_static: [B, 5] static features
            lengths: [B] sequence lengths (optional)

        Returns:
            Dict containing:
                - t1, t2, t3, t4, t5: Dict with logits, dyn_feat, shared_feat
                - _aux: Auxiliary info (architecture_type, flags)
        """
        B = x_dyn.size(0)

        # === Shared Bottom Encoding ===
        # All tasks share the same encoder output
        H_shared = self.shared_bottom_encoder(x_dyn, lengths)  # [B, 30, 16]

        # === Task-specific Processing ===
        outputs = {}

        for task_key in self.active_tasks:
            # All tasks use the same H_shared
            # Task-specific processing happens in the tower/head only

            # Process through task head (PMGT for t1, Flatten for t2-t5)
            dyn_feat = self.task_heads[task_key](H_shared)  # [B, hidden_dim]

            # Encode static features (task-specific encoder)
            static_feat = self.static_encoders[task_key](x_static)  # [B, static_dim]

            # Classify (task-specific classifier)
            logits = self.classifiers[task_key](dyn_feat, static_feat)

            outputs[task_key] = {
                "logits": logits,
                "dyn_feat": dyn_feat,
                "static_feat": static_feat,
                "shared_feat": H_shared,  # All tasks reference the same H_shared
            }

        # === Auxiliary Output ===
        outputs["_aux"] = {
            "architecture_type": self.architecture_type,
            "t6_deep_context_enabled": self.t6_deep_context_enabled,
            "uses_alpha_beta_split": self.uses_alpha_beta_split,
            "uses_expert_pool": self.uses_expert_pool,
            "uses_task_gates": self.uses_task_gates,
            "shared_bottom_feat_shape": H_shared.shape,
        }

        return outputs

    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts by module (matching other MTL architectures)."""
        counts = {}

        # Shared bottom encoder
        counts["shared_bottom_encoder"] = sum(
            p.numel() for p in self.shared_bottom_encoder.parameters()
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

        Note: Shared-bottom has no Alpha/Beta branches, so typical freeze_modules
        calls from other architectures will be handled gracefully.

        Args:
            module_names: List of module names to freeze
        """
        for name in module_names:
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.requires_grad = False
                    print(f"[Shared-bottom Clean] Frozen: {name}")
            else:
                # Handle legacy freeze_module names gracefully
                # Alpha/Beta don't exist in Shared-bottom
                if name in ["alpha_trunk", "beta_trunk", "alpha_branch", "beta_branch",
                           "alpha_residual_experts", "beta_residual_experts",
                           "alpha_gates", "beta_gates", "mmoe_layer", "mmoe_experts"]:
                    print(f"[Shared-bottom Clean] Module '{name}' not present (no Alpha/Beta/MMoE)")
                else:
                    print(f"[Shared-bottom Clean] Warning: Unknown module '{name}'")

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
                    print(f"[Shared-bottom Clean] Unfrozen: {name}")

    def get_active_tasks(self) -> List[str]:
        """Get list of active tasks."""
        return self.active_tasks

    def __repr__(self) -> str:
        counts = self.get_num_parameters()
        return (
            f"SharedBottomCleanV4Architecture(\n"
            f"  architecture_type='{self.architecture_type}'\n"
            f"  num_channels={self.num_channels}\n"
            f"  D_time={self.D_time}\n"
            f"  active_tasks={self.active_tasks}\n"
            f"  shared_bottom_encoder_params={counts['shared_bottom_encoder']}\n"
            f"  total_params={counts['total']}\n"
            f")"
        )


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Shared-bottom Clean V4 Architecture Test")
    print("=" * 60)

    # Create model
    model = SharedBottomCleanV4Architecture(
        num_channels=30,
        D_time=16,
        T_mid=24,
        dropout=0.3
    )

    # Print architecture info
    print(f"\n{model}")

    # Print parameter counts
    counts = model.get_num_parameters()
    print("\nParameter counts:")
    for key, count in counts.items():
        print(f"  {key}: {count}")

    # Test forward without lengths
    B, L, C = 4, 200, 30
    x_dyn = torch.randn(B, L, C)
    x_static = torch.randn(B, 5)

    outputs = model(x_dyn, x_static)

    print("\nForward output (no lengths):")
    for task_key in model.active_tasks:
        print(f"  {task_key}:")
        print(f"    logits: {outputs[task_key]['logits'].shape}")
        print(f"    dyn_feat: {outputs[task_key]['dyn_feat'].shape}")
        print(f"    shared_feat: {outputs[task_key]['shared_feat'].shape}")

    print("\nAuxiliary info:")
    print(f"  architecture_type: {outputs['_aux']['architecture_type']}")
    print(f"  t6_deep_context_enabled: {outputs['_aux']['t6_deep_context_enabled']}")
    print(f"  shared_bottom_feat_shape: {outputs['_aux']['shared_bottom_feat_shape']}")

    # Test forward with lengths
    lengths = torch.tensor([100, 150, 200, 180])
    outputs = model(x_dyn, x_static, lengths)

    print("\nForward output (with lengths):")
    for task_key in model.active_tasks:
        print(f"  {task_key}:")
        print(f"    logits: {outputs[task_key]['logits'].shape}")
        print(f"    shared_feat: {outputs[task_key]['shared_feat'].shape}")

    # Verify all tasks share the same shared_feat
    print("\nVerifying shared_feat consistency:")
    shared_feats = [outputs[t]['shared_feat'] for t in model.active_tasks]
    all_same = all(torch.equal(shared_feats[0], f) for f in shared_feats)
    print(f"  All tasks use same shared_feat: {all_same}")

    # Test backward pass
    print("\nTesting backward pass:")
    total_loss = sum(outputs[t]['logits'].sum() for t in model.active_tasks)
    total_loss.backward()
    print("  Backward pass successful")

    # Check gradient flow
    encoder_has_grad = any(p.grad is not None for p in model.shared_bottom_encoder.parameters())
    print(f"  Shared encoder has gradient: {encoder_has_grad}")

    for task_key in model.active_tasks:
        head_has_grad = any(p.grad is not None for p in model.task_heads[task_key].parameters())
        print(f"  {task_key} head has gradient: {head_has_grad}")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)