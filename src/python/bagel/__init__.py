"""
BAGEL integration for CrowdFace
"""

from .scene_understanding import BAGELSceneUnderstanding
from .ad_placement import BAGELAdPlacement
from .ad_optimization import BAGELAdOptimization

__all__ = [
    'BAGELSceneUnderstanding',
    'BAGELAdPlacement',
    'BAGELAdOptimization',
]
