"""
Clean Architecture Model Registry.

Maps architecture variant names to explicit architecture classes.
Each architecture variant has fixed characteristics (NO YAML-driven configuration).

Supported variants:
- "baseline_clean_v4": BaselineCleanV4Architecture (placeholder)
- "ablation_b_clean_v4": AblationBCleanV4Architecture (placeholder)
- "mmoe_clean_v4": MMoECleanV4Architecture
- "cgc_clean_v4": CGCCleanV4Architecture
- "shared_bottom_clean_v4": SharedBottomCleanV4Architecture
- "adatt_clean_v4": AdaTTCleanV4Architecture

Unknown variant -> ValueError (NO fallback to legacy).

Author: Clean Architecture Team
Date: 2026-05-26
"""

import torch
from typing import Dict, Type, Any, Optional

# Import architecture classes
from .architectures.mmoe_clean_v4 import MMoECleanV4Architecture
from .architectures.cgc_clean_v4 import CGCCleanV4Architecture
from .architectures.shared_bottom_clean_v4 import SharedBottomCleanV4Architecture
from .architectures.adatt_clean_v4 import AdaTTCleanV4Architecture

# Placeholder imports (not yet implemented)
# from .architectures.baseline_clean_v4 import BaselineCleanV4Architecture
# from .architectures.ablation_b_clean_v4 import AblationBCleanV4Architecture


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY: Dict[str, Type] = {
    # Placeholders (will raise error if accessed before implementation)
    # "baseline_clean_v4": BaselineCleanV4Architecture,
    # "ablation_b_clean_v4": AblationBCleanV4Architecture,

    # Implemented baselines
    "mmoe_clean_v4": MMoECleanV4Architecture,
    "cgc_clean_v4": CGCCleanV4Architecture,
    "shared_bottom_clean_v4": SharedBottomCleanV4Architecture,
    "adatt_clean_v4": AdaTTCleanV4Architecture,
}


def get_model_class(variant: str) -> Type:
    """
    Get model class by variant name.

    Args:
        variant: Architecture variant name

    Returns:
        Model class (e.g., MMoECleanV4Architecture)

    Raises:
        ValueError: Unknown variant (NO fallback to legacy)
    """
    if variant not in MODEL_REGISTRY:
        valid_variants = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture variant: '{variant}'. "
            f"Valid variants: {valid_variants}. "
            f"Legacy variants (HDSTGCNMTL, ProtectedDualEngineMTL) are NOT supported."
        )
    return MODEL_REGISTRY[variant]


def create_model(
    variant: str,
    task_specs: Optional[Dict[str, Any]] = None,
    num_channels: int = 30,
    D_time: int = 16,
    T_mid: int = 24,
    dropout: float = 0.3,
    device: str = "cuda",
    **kwargs
) -> torch.nn.Module:
    """
    Create model instance by variant name.

    Args:
        variant: Architecture variant name
        task_specs: Task specification dictionary
        num_channels: Number of input channels (default: 30 for nine_graph)
        D_time: Temporal encoding dimension (default: 16)
        T_mid: Intermediate temporal dimension (default: 24)
        dropout: Dropout rate (default: 0.3)
        device: Device to place model on (default: "cuda")
        **kwargs: Additional architecture-specific arguments

    Returns:
        Model instance

    Raises:
        ValueError: Unknown variant
    """
    model_class = get_model_class(variant)
    model = model_class(
        task_specs=task_specs,
        num_channels=num_channels,
        D_time=D_time,
        T_mid=T_mid,
        dropout=dropout,
        **kwargs
    )
    model = model.to(device)
    return model


def list_available_variants() -> list:
    """
    List all available architecture variants.

    Returns:
        List of variant names
    """
    return list(MODEL_REGISTRY.keys())


def is_valid_variant(variant: str) -> bool:
    """
    Check if variant is valid.

    Args:
        variant: Variant name to check

    Returns:
        True if variant exists in registry
    """
    return variant in MODEL_REGISTRY


def register_variant(name: str, model_class: Type) -> None:
    """
    Register a new architecture variant.

    Args:
        name: Variant name
        model_class: Model class

    Raises:
        ValueError: If variant already registered
    """
    if name in MODEL_REGISTRY:
        raise ValueError(f"Variant '{name}' already registered")
    MODEL_REGISTRY[name] = model_class
    print(f"[Model Registry] Registered variant: {name} -> {model_class.__name__}")


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Model Registry Test")
    print("=" * 60)

    # List variants
    print(f"\nAvailable variants: {list_available_variants()}")

    # Test get_model_class
    for variant in ["mmoe_clean_v4", "cgc_clean_v4", "shared_bottom_clean_v4"]:
        cls = get_model_class(variant)
        print(f"  get_model_class('{variant}') -> {cls.__name__}")

    # Test invalid variant
    try:
        get_model_class("unknown_variant")
    except ValueError as e:
        print(f"\n[Expected Error] {e}")

    print("\n" + "=" * 60)
    print("Registry test passed!")
    print("=" * 60)