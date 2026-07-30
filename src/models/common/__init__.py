"""
Common Components for MTL Architectures

Each module provides reusable building blocks:
- mmoe.py: MMoE experts, gates, and layer
- cgc.py: CGC shared/private experts, gates, and layer
- shared_bottom.py: Shared bottom encoder (no routing)
- adatt.py: AdaTT task-to-task fusion components

Author: Clean Architecture Team
Date: 2026-05-26
"""

from .mmoe import (
    MMoEExpert,
    MMoEGateContextEncoder,
    MMoETaskGate,
    MMoELayer
)

from .cgc import (
    CGCExpert,
    CGCGateContextEncoder,
    CGCTaskGate,
    CGCLayer
)

from .shared_bottom import (
    SharedBottomDynamicEncoder
)

from .adatt import (
    AdaTTTemporalStem,
    AdaTTTaskAdapter,
    AdaTTFusionUnit,
    AdaTTGateContextEncoder,
    AdaTTTaskGate,
    AdaTTFusionLayer
)

__all__ = [
    # MMoE
    "MMoEExpert",
    "MMoEGateContextEncoder",
    "MMoETaskGate",
    "MMoELayer",
    # CGC
    "CGCExpert",
    "CGCGateContextEncoder",
    "CGCTaskGate",
    "CGCLayer",
    # Shared Bottom
    "SharedBottomDynamicEncoder",
    # AdaTT
    "AdaTTTemporalStem",
    "AdaTTTaskAdapter",
    "AdaTTFusionUnit",
    "AdaTTGateContextEncoder",
    "AdaTTTaskGate",
    "AdaTTFusionLayer",
]