"""
Modules package for Protected Dual-Engine Hierarchical CGC/PLE Architecture.

Components:
- experts.py: Expert temporal encoders with configurable capacity levels
             + ResidualExpert (v2: 残差专家模块)
- gates.py: Gate modules for task-specific context encoding and selection
- t6_context.py: T6 deep feature context modules for v4 architecture
               + T6DeepFeatureContextEncoder, T6DeepFeatureBridge, T6DeepFeatureContextModule
"""

from .experts import ExpertTemporalEncoder, EXPERT_CAPACITY_CONFIG, ResidualExpert
from .gates import (
    AlphaGateContextEncoder,
    BetaGateContextEncoder,
    SharedGateStaticMLP,
    TaskSpecificGate,
)
from .t6_context import (
    T6DeepFeatureContextEncoder,
    T6DeepFeatureBridge,
    T6DeepFeatureContextModule,
)

__all__ = [
    # Experts
    "ExpertTemporalEncoder",
    "EXPERT_CAPACITY_CONFIG",
    "ResidualExpert",  # v2: 新增残差专家模块
    # Gates
    "AlphaGateContextEncoder",
    "BetaGateContextEncoder",
    "SharedGateStaticMLP",
    "TaskSpecificGate",
    # T6 Context (v4)
    "T6DeepFeatureContextEncoder",
    "T6DeepFeatureBridge",
    "T6DeepFeatureContextModule",
]