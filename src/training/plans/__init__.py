"""
Training Plans for Clean Architecture

Each plan is bound to a specific architecture variant.
Explicit stage configuration, no YAML branching.
"""

from .mmoe_clean_plan import MMoECleanTrainingPlan, get_mmoe_clean_plan
from .cgc_clean_plan import CGCCleanTrainingPlan, get_cgc_clean_plan
from .shared_bottom_clean_plan import SharedBottomCleanTrainingPlan, get_shared_bottom_clean_plan
from .adatt_clean_plan import AdaTTCleanTrainingPlan, get_adatt_clean_plan

__all__ = [
    "MMoECleanTrainingPlan",
    "get_mmoe_clean_plan",
    "CGCCleanTrainingPlan",
    "get_cgc_clean_plan",
    "SharedBottomCleanTrainingPlan",
    "get_shared_bottom_clean_plan",
    "AdaTTCleanTrainingPlan",
    "get_adatt_clean_plan",
]