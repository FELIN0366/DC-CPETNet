"""
Training Plan Registry

Clean Architecture registry for training plans.
Explicit registration, no YAML branching.

Usage:
    from src.training.plan_registry import get_plan

    plan = get_plan("mmoe_clean_plan")
    stage1 = plan.get_stage("stage1")
"""

from typing import Dict, Type, Optional


# =============================================================================
# Training Plan Registry
# =============================================================================

PLAN_REGISTRY: Dict[str, Type] = {}


def register_plan(name: str, plan_cls: Type) -> None:
    """
    Register a training plan.

    Args:
        name: Plan name (e.g., "mmoe_clean_plan")
        plan_cls: Plan class
    """
    if name in PLAN_REGISTRY:
        raise ValueError(f"Plan '{name}' already registered")

    PLAN_REGISTRY[name] = plan_cls
    print(f"[Plan Registry] Registered plan: {name} -> {plan_cls.__name__}")


def get_plan(plan_name: str) -> Optional[object]:
    """
    Get training plan instance by name.

    Args:
        plan_name: Plan name

    Returns:
        Plan instance

    Raises:
        ValueError: If plan not found
    """
    if plan_name not in PLAN_REGISTRY:
        available = list(PLAN_REGISTRY.keys())
        raise ValueError(
            f"Unknown plan: '{plan_name}'. "
            f"Available plans: {available}. "
            f"No legacy fallback in clean architecture."
        )

    # Return instance (plans are singleton-like)
    plan_cls = PLAN_REGISTRY[plan_name]
    return plan_cls()


def list_plans() -> list:
    """List all registered plans."""
    return list(PLAN_REGISTRY.keys())


def is_valid_plan(plan_name: str) -> bool:
    """Check if plan name is valid."""
    return plan_name in PLAN_REGISTRY


# =============================================================================
# Register Plans
# =============================================================================

def _register_all_plans() -> None:
    """Register all clean training plans."""
    from .plans.mmoe_clean_plan import MMoECleanTrainingPlan
    from .plans.cgc_clean_plan import CGCCleanTrainingPlan
    from .plans.shared_bottom_clean_plan import SharedBottomCleanTrainingPlan
    from .plans.adatt_clean_plan import AdaTTCleanTrainingPlan

    # Register MMoE Clean Plan
    register_plan("mmoe_clean_plan", MMoECleanTrainingPlan)

    # Register CGC Clean Plan
    register_plan("cgc_clean_plan", CGCCleanTrainingPlan)

    # Register Shared-bottom Clean Plan
    register_plan("shared_bottom_clean_plan", SharedBottomCleanTrainingPlan)

    # Register AdaTT Clean Plan
    register_plan("adatt_clean_plan", AdaTTCleanTrainingPlan)

    # Future plans can be registered here:
    # register_plan("baseline_clean_plan", BaselineCleanTrainingPlan)
    # register_plan("ablation_b_clean_plan", AblationBCleanTrainingPlan)


# Auto-register on import
_register_all_plans()


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Training Plan Registry Test")
    print("=" * 60)

    # List all plans
    print(f"\nRegistered plans: {list_plans()}")

    # Test get_plan
    plan = get_plan("mmoe_clean_plan")
    print(f"\nget_plan('mmoe_clean_plan') -> {plan}")
    print(f"  architecture_variant: {plan.architecture_variant}")
    print(f"  stages: {plan.get_stage_names()}")

    # Test unknown plan (should raise error)
    try:
        get_plan("unknown_plan")
    except ValueError as e:
        print(f"\n[Expected Error] {e}")

    print("\n" + "=" * 60)
    print("Plan registry test passed!")
    print("=" * 60)