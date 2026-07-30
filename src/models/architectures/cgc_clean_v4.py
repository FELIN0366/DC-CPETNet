"""
CGC Clean V4 Architecture - Customized Gate Control Baseline

CGC = Single-level Customized Gate Control (baseline for PLE)
- Shared experts (task-common knowledge)
- Private expert per task (task-specific knowledge)
- Each task gate selects from shared + own private

Author: CGC Clean Implementation
Date: 2026-05-26
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.common.cgc import CGCLayer
from model_mtl import PriorMaskedTaskHead, FlattenProjector, TaskStaticEncoder, TaskClassifier
from task_specs import TaskSpec


CGC_TASK_CONFIG = {
    "t1": {"num_classes": 3, "is_binary": False, "use_pmgt": True, "display_name": "运动心功能分级", "branch": "alpha", "label_column": "运动心功能分级", "loss_name": "ce"},
    "t2": {"num_classes": 3, "is_binary": False, "use_pmgt": False, "display_name": "运动耐量", "branch": "beta", "label_column": "运动耐量", "loss_name": "ce"},
    "t3": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "标准心电运动负荷试验", "branch": "beta", "label_column": "标准心电运动负荷试验", "loss_name": "bce"},
    "t4": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "心率储备", "branch": "beta", "label_column": "心率储备", "loss_name": "ldam"},
    "t5": {"num_classes": 2, "is_binary": True, "use_pmgt": False, "display_name": "无氧阈", "branch": "beta", "label_column": "无氧阈", "loss_name": "ldam"},
}


class CGCCleanV4Architecture(nn.Module):
    """
    CGC Clean V4 Architecture.

    Explicit separation of task-common and task-specific experts.
    """

    architecture_type: str = "cgc_clean_v4"
    t6_deep_context_enabled: bool = False
    uses_alpha_beta_split: bool = False
    uses_expert_routing_constraints: bool = True

    def __init__(
        self,
        num_channels: int = 30,
        D_time: int = 16,
        T_mid: int = 24,
        num_shared_experts: int = 2,
        semantic_adj: Optional[torch.Tensor] = None,
        task_specs: Optional[Dict[str, TaskSpec]] = None,
        dropout: float = 0.3,
        hidden_dim: int = 48,
        static_dim: int = 16,
        context_dim: int = 40,
        **kwargs
    ):
        super().__init__()

        self.num_channels = num_channels
        self.D_time = D_time
        self.num_shared_experts = num_shared_experts
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim
        self.active_tasks = ["t1", "t2", "t3", "t4", "t5"]

        if semantic_adj is not None:
            self.semantic_adj = torch.tensor(semantic_adj, dtype=torch.float32)
        else:
            self.semantic_adj = None

        if task_specs is None:
            task_specs = self._build_default_task_specs()
        self.task_specs = task_specs

        # CGC Layer
        self.cgc_layer = CGCLayer(
            num_channels=num_channels,
            D_time=D_time,
            T_mid=T_mid,
            num_shared_experts=num_shared_experts,
            task_names=self.active_tasks,
            context_dim=context_dim,
            dropout=dropout
        )

        # Task Heads
        self.task_heads = nn.ModuleDict()
        if self.semantic_adj is not None:
            self.task_heads["t1"] = PriorMaskedTaskHead(
                num_nodes=num_channels, hidden_dim=D_time, out_dim=hidden_dim,
                semantic_adj=self.semantic_adj, dropout=dropout * 1.5, gamma_init=1.0
            )
        else:
            self.task_heads["t1"] = FlattenProjector(
                num_nodes=num_channels, hidden_dim=D_time, out_dim=hidden_dim, dropout=dropout
            )

        for task_key in ["t2", "t3", "t4", "t5"]:
            self.task_heads[task_key] = FlattenProjector(
                num_nodes=num_channels, hidden_dim=D_time, out_dim=hidden_dim, dropout=dropout
            )

        # Static Encoders
        self.static_encoders = nn.ModuleDict({
            task_key: TaskStaticEncoder(in_dim=5, out_dim=static_dim, dropout=dropout * 0.7)
            for task_key in self.active_tasks
        })

        # Classifiers
        self.classifiers = nn.ModuleDict({
            task_key: TaskClassifier(
                dyn_dim=hidden_dim, static_dim=static_dim, hidden_dim=32,
                num_classes=CGC_TASK_CONFIG[task_key]["num_classes"],
                is_binary=CGC_TASK_CONFIG[task_key]["is_binary"],
                dropout=dropout
            )
            for task_key in self.active_tasks
        })

        print(f"\n[CGC Clean V4] Architecture initialized:")
        print(f"  architecture_type = {self.architecture_type}")
        print(f"  num_shared_experts = {num_shared_experts}")
        print(f"  num_private_experts_per_task = 1")
        print(f"  active_tasks = {self.active_tasks}")
        print(f"  t6_deep_context_enabled = {self.t6_deep_context_enabled}")
        print(f"  CGC routing: each task uses shared experts + own private expert")

    def _build_default_task_specs(self) -> Dict[str, TaskSpec]:
        from task_specs import TaskSpec
        task_specs = {}
        for task_key, config in CGC_TASK_CONFIG.items():
            spec = TaskSpec(
                name=task_key,
                display_name=config["display_name"],
                num_classes=config["num_classes"],
                branch=config["branch"],
                loss_name=config["loss_name"],
                is_binary=config["is_binary"],
                dropout=0.3,
                label_column=config["label_column"]
            )
            task_specs[task_key] = spec
        return task_specs

    def forward(
        self,
        x_dyn: torch.Tensor,
        x_static: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        B = x_dyn.size(0)

        H_gated, gate_weights = self.cgc_layer(x_dyn, x_static, lengths)

        dyn_feat = {}
        logits = {}

        for task_key in self.active_tasks:
            H_t = H_gated[task_key]
            dyn_feat_t = self.task_heads[task_key](H_t)
            dyn_feat[task_key] = dyn_feat_t

            static_feat_t = self.static_encoders[task_key](x_static)
            logits_t = self.classifiers[task_key](dyn_feat_t, static_feat_t)
            logits[task_key] = logits_t

        outputs = {}
        for task_key in self.active_tasks:
            outputs[task_key] = {
                "logits": logits[task_key],
                "dyn_feat": dyn_feat[task_key],
                "gate_weights": gate_weights[task_key],
                "static_feat": self.static_encoders[task_key](x_static)
            }
            outputs[task_key]["fused_feat"] = torch.cat([dyn_feat[task_key], outputs[task_key]["static_feat"]], dim=1)

        outputs["_aux"] = {
            "architecture_type": self.architecture_type,
            "cgc_gate_weights": gate_weights,
            "num_shared_experts": self.num_shared_experts,
            "num_private_experts_per_task": 1,
            "active_tasks": self.active_tasks,
            "t6_deep_context_enabled": self.t6_deep_context_enabled,
            "uses_alpha_beta_split": self.uses_alpha_beta_split,
            "uses_expert_routing_constraints": self.uses_expert_routing_constraints,
        }

        return outputs

    def get_num_parameters(self) -> Dict[str, int]:
        counts = {}
        counts["cgc_layer"] = sum(p.numel() for p in self.cgc_layer.parameters())
        cgc_breakdown = self.cgc_layer.get_parameter_breakdown()
        counts["cgc_shared_experts"] = cgc_breakdown["shared_experts"]
        counts["cgc_private_experts"] = cgc_breakdown["private_experts"]
        counts["cgc_task_gates"] = cgc_breakdown["task_gates"]
        counts["task_heads"] = sum(sum(p.numel() for p in self.task_heads[k].parameters()) for k in self.task_heads.keys())
        counts["static_encoders"] = sum(sum(p.numel() for p in self.static_encoders[k].parameters()) for k in self.static_encoders.keys())
        counts["classifiers"] = sum(sum(p.numel() for p in self.classifiers[k].parameters()) for k in self.classifiers.keys())
        counts["total"] = sum(counts.values())
        return counts

    def freeze_modules(self, module_names: List[str]) -> None:
        for name in module_names:
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.requires_grad = False
                    print(f"[CGC Clean] Frozen: {name}")

    def unfreeze_modules(self, module_names: List[str]) -> None:
        for name in module_names:
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"[CGC Clean] Unfrozen: {name}")

    def __repr__(self) -> str:
        counts = self.get_num_parameters()
        return f"CGCCleanV4Architecture(architecture_type='{self.architecture_type}', shared={self.num_shared_experts}, tasks={self.active_tasks}, params={counts['total']:,})"


if __name__ == "__main__":
    print("=" * 60)
    print("CGC Clean V4 Architecture Test")
    print("=" * 60)

    model = CGCCleanV4Architecture(num_channels=30, D_time=16, num_shared_experts=2)
    print(f"\n{model}")

    B, L = 8, 200
    x_dyn = torch.randn(B, L, 30)
    x_static = torch.randn(B, 5)

    outputs = model(x_dyn, x_static)

    print("\nForward pass:")
    for task_key in model.active_tasks:
        w_sum = outputs[task_key]['gate_weights'].sum(dim=1)
        print(f"  {task_key}: logits={outputs[task_key]['logits'].shape}, gate_sum={w_sum[0]:.4f}")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)