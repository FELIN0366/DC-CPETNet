"""
MTL Architectures for Clean Architecture

Each architecture variant has explicit characteristics:
- architecture_type: str class attribute
- t6_deep_context_enabled: bool class attribute
- No YAML-driven configuration branching
- No experiment_suffix-based mode switching

Supported architectures:
- mmoe_clean_v4: MMoE baseline (shared expert pool + task gates)
- cgc_clean_v4: CGC baseline (shared + private experts)
- shared_bottom_clean_v4: Shared-bottom baseline (single encoder)
- adatt_clean_v4: AdaTT baseline (task-to-task fusion)

Author: Clean Architecture Team
Date: 2026-05-26
"""

from .mmoe_clean_v4 import MMoECleanV4Architecture
from .cgc_clean_v4 import CGCCleanV4Architecture
from .shared_bottom_clean_v4 import SharedBottomCleanV4Architecture
from .adatt_clean_v4 import AdaTTCleanV4Architecture

__all__ = [
    "MMoECleanV4Architecture",
    "CGCCleanV4Architecture",
    "SharedBottomCleanV4Architecture",
    "AdaTTCleanV4Architecture",
]