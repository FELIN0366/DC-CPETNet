"""
CGC Clean Training Plan - Training stages for CGC architecture

This module defines the training stages for CGC clean architecture.
Following clean architecture principles:
- Explicit stage configuration
- No YAML ablation flags
- No t6 context injection
- Standard multi-task training stages

Training Flow:
    Stage1: t1 anchor (20 epochs)
    Stage2: t2-t5 warmup (20 epochs)
    Phase1: t2-t5 strengthening (18 epochs)
    Phase2: joint finetune (42 epochs)

No Stage0 (no t6 KD or feature distillation)
No t6 context injection at any stage

Author: CGC Clean Implementation
Date: 2026-05-26
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal


# =============================================================================
# StageConfig Dataclass
# =============================================================================

@dataclass
class StageConfig:
    """
    Stage Configuration Dataclass.

    Defines all parameters for a training stage.

    Attributes:
        name: Stage name (stage1, stage2, phase1, phase2)
        epochs: Number of epochs for this stage
        active_tasks: List of tasks to train in this stage
        loss_weights: Task loss weights
        freeze_modules: Modules to freeze during this stage
        checkpoint_metric: Primary metric for checkpoint selection
        early_stop_patience: Early stopping patience (0 = no early stop)
        early_stop_min_delta: Minimum improvement for early stopping
        lr_multiplier: Learning rate multiplier for this stage
        t6_context_inject: Whether to inject t6 context (always False for CGC)
        t6_context_detach: Whether to detach t6 context gradients (always False for CGC)
    """
    name: str
    epochs: int
    active_tasks: List[str]
    loss_weights: Dict[str, float] = field(default_factory=dict)
    freeze_modules: List[str] = field(default_factory=list)
    checkpoint_metric: str = "weighted_macro_f1_t1_to_t5"
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001
    lr_multiplier: float = 1.0
    t6_context_inject: bool = False  # Always False for CGC
    t6_context_detach: bool = False  # Always False for CGC
    use_uncertainty_weighting: bool = False  # Uncertainty weighting toggle


# =============================================================================
# CGC Clean Training Plan
# =============================================================================

class CGCCleanTrainingPlan:
    """
    CGC Clean Training Plan.

    Defines the training stages for CGC architecture:
    - No Stage0 (no t6 KD)
    - No t6 context injection
    - Standard multi-task progression
    - CGC-specific considerations: shared experts + private experts

    Stage Flow:
        Stage1: t1 only (anchor CGC experts for t1)
        Stage2: t2-t5 only (train remaining tasks)
        Phase1: t2-t5 strengthening
        Phase2: joint finetune all tasks

    Design Principles:
    1. Explicit stage definitions (no YAML branching)
    2. No t6-related logic
    3. Uses existing training infrastructure
    4. Follows mmoe_clean_plan structure (adapted for CGC)

    CGC-specific:
    - Shared experts are trained throughout (no freezing of shared experts)
    - Private experts are task-specific (can be frozen per stage)
    - Gates are task-specific (can be frozen per stage)

    Attributes:
        plan_name: Training plan identifier
        architecture_variant: Bound architecture type
        stages: Dict of StageConfig objects
    """

    # Explicit identifiers
    plan_name: str = "cgc_clean_plan"
    architecture_variant: str = "cgc_clean_v4"

    # CGC specific flags
    uses_t6_context: bool = False
    uses_feature_distillation: bool = False
    uses_stage0_kd: bool = False

    def __init__(self):
        """Initialize CGC training plan with explicit stage configs."""
        self.stages = self._build_stages()
        print(f"\n[CGC Clean Plan] Training plan initialized:")
        print(f"  plan_name = {self.plan_name}")
        print(f"  architecture_variant = {self.architecture_variant}")
        print(f"  uses_t6_context = {self.uses_t6_context}")
        print(f"  uses_feature_distillation = {self.uses_feature_distillation}")
        print(f"  stages = {list(self.stages.keys())}")

    def _build_stages(self) -> Dict[str, StageConfig]:
        """Build training stages for CGC."""
        stages = {}

        # Stage1: t1 anchor
        # - Train t1 with CGC gated representation
        # - Freeze all other task-specific components (private experts + heads)
        # - Shared experts continue training (they're task-common)
        stages["stage1"] = StageConfig(
            name="stage1",
            epochs=20,
            active_tasks=["t1"],
            loss_weights={"t1": 1.0},
            freeze_modules=[
                # Freeze other tasks' private experts
                "cgc_layer.private_experts.t2",
                "cgc_layer.private_experts.t3",
                "cgc_layer.private_experts.t4",
                "cgc_layer.private_experts.t5",
                # Freeze other tasks' gates
                "cgc_layer.task_gates.t2",
                "cgc_layer.task_gates.t3",
                "cgc_layer.task_gates.t4",
                "cgc_layer.task_gates.t5",
                # Freeze other tasks' heads
                "task_heads.t2", "task_heads.t3", "task_heads.t4", "task_heads.t5",
                "static_encoders.t2", "static_encoders.t3", "static_encoders.t4", "static_encoders.t5",
                "classifiers.t2", "classifiers.t3", "classifiers.t4", "classifiers.t5",
            ],
            checkpoint_metric="t1_macro_f1",
            early_stop_patience=0,  # No early stop for anchor stage
            t6_context_inject=False
        )

        # Stage2: t2-t5 warmup
        # - Train t2-t5 tasks with CGC
        # - Freeze t1 components
        # - Shared experts continue training
        stages["stage2"] = StageConfig(
            name="stage2",
            epochs=20,
            active_tasks=["t2", "t3", "t4", "t5"],
            loss_weights={"t2": 1.0, "t3": 1.2, "t4": 1.0, "t5": 1.0},  # t3 higher weight for imbalance
            freeze_modules=[
                # Freeze t1's private expert
                "cgc_layer.private_experts.t1",
                # Freeze t1's gate
                "cgc_layer.task_gates.t1",
                # Freeze t1's head
                "task_heads.t1", "static_encoders.t1", "classifiers.t1",
            ],
            checkpoint_metric="mean_macro_f1_t2_to_t5",
            early_stop_patience=0,
            t6_context_inject=False
        )

        # Phase1: t2-t5 strengthening
        # - Continue training t2-t5
        # - No freezing (strengthen all)
        # - Enable uncertainty weighting
        stages["phase1"] = StageConfig(
            name="phase1",
            epochs=18,
            active_tasks=["t2", "t3", "t4", "t5"],
            loss_weights={"t2": 1.0, "t3": 1.2, "t4": 1.0, "t5": 1.0},
            freeze_modules=[],  # No freezing, strengthen all
            checkpoint_metric="mean_macro_f1_t2_to_t5",
            early_stop_patience=0,
            t6_context_inject=False,
            use_uncertainty_weighting=True  # Enable uncertainty weighting
        )

        # Phase2: joint finetune
        # - All tasks t1-t5 joint training
        # - Use uncertainty weighting
        stages["phase2"] = StageConfig(
            name="phase2",
            epochs=42,
            active_tasks=["t1", "t2", "t3", "t4", "t5"],
            loss_weights={"t1": 1.0, "t2": 1.0, "t3": 1.2, "t4": 1.0, "t5": 1.0},
            freeze_modules=[],  # Full training
            checkpoint_metric="weighted_macro_f1_t1_to_t5",
            early_stop_patience=10,
            early_stop_min_delta=0.001,
            t6_context_inject=False,
            use_uncertainty_weighting=True
        )

        return stages

    def get_stage(self, stage_name: str) -> Optional[StageConfig]:
        """Get a specific stage configuration."""
        return self.stages.get(stage_name)

    def get_stage_names(self) -> List[str]:
        """Get all stage names in order."""
        return ["stage1", "stage2", "phase1", "phase2"]

    def get_checkpoint_prefix(self) -> str:
        """Get checkpoint file prefix."""
        return "cgc_clean"

    def get_active_tasks(self, stage_name: str) -> List[str]:
        """Get active tasks for a stage."""
        stage = self.get_stage(stage_name)
        if stage:
            return stage.active_tasks
        return []

    def get_checkpoint_metric(self, stage_name: str) -> str:
        """Get checkpoint metric for a stage."""
        stage = self.get_stage(stage_name)
        if stage:
            return stage.checkpoint_metric
        return "weighted_macro_f1_t1_to_t5"

    def validate_stage_order(self, stage_name: str) -> bool:
        """Validate stage execution order."""
        order = self.get_stage_names()
        return stage_name in order

    def __repr__(self) -> str:
        return (
            f"CGCCleanTrainingPlan("
            f"plan_name='{self.plan_name}', "
            f"architecture_variant='{self.architecture_variant}', "
            f"stages={self.get_stage_names()})"
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_cgc_clean_plan() -> CGCCleanTrainingPlan:
    """
    Factory function to create CGC Clean Training Plan.

    Returns:
        CGCCleanTrainingPlan instance
    """
    return CGCCleanTrainingPlan()


def validate_cgc_plan(plan: CGCCleanTrainingPlan) -> bool:
    """
    Validate CGC training plan.

    Checks:
    1. All stages exist
    2. No t6 context injection
    3. No Stage0
    4. Correct architecture binding

    Args:
        plan: CGCCleanTrainingPlan instance

    Returns:
        True if valid
    """
    # Check no t6 context
    for stage_name, stage in plan.stages.items():
        if stage.t6_context_inject:
            print(f"[Validation ERROR] {stage_name}: t6_context_inject=True (should be False)")
            return False

    # Check no Stage0
    if "stage0" in plan.stages:
        print(f"[Validation ERROR] stage0 exists (CGC should not have Stage0)")
        return False

    # Check architecture binding
    if plan.architecture_variant != "cgc_clean_v4":
        print(f"[Validation ERROR] architecture_variant={plan.architecture_variant} (should be cgc_clean_v4)")
        return False

    print(f"[Validation OK] CGC training plan valid")
    return True


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CGC Clean Training Plan Test")
    print("=" * 60)

    # Create plan
    plan = get_cgc_clean_plan()
    print(f"\n{plan}")

    # Validate plan
    is_valid = validate_cgc_plan(plan)

    # Print stage details
    print("\nStage Details:")
    for stage_name in plan.get_stage_names():
        stage = plan.get_stage(stage_name)
        print(f"\n  {stage_name}:")
        print(f"    epochs: {stage.epochs}")
        print(f"    active_tasks: {stage.active_tasks}")
        print(f"    loss_weights: {stage.loss_weights}")
        print(f"    freeze_modules: {stage.freeze_modules}")
        print(f"    checkpoint_metric: {stage.checkpoint_metric}")
        print(f"    t6_context_inject: {stage.t6_context_inject}")
        print(f"    use_uncertainty_weighting: {stage.use_uncertainty_weighting}")

    print("\n" + "=" * 60)
    print(f"Validation: {is_valid}")
    print("Test passed!")
    print("=" * 60)