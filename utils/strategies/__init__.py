from .base import BaseStrategy, TrainingResult
from .original_strategy import OriginalStrategy
from .diffusion_ambient import DiffusionStrategy

__all__ = [
    "BaseStrategy",
    "TrainingResult",
    "OriginalStrategy",
    "DiffusionStrategy",
]

STRATEGIES = {
    'OriginalStrategy': OriginalStrategy,
    'DiffusionStrategy': DiffusionStrategy,
}
