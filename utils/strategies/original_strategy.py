from .base import BaseStrategy
import torch

class OriginalStrategy(BaseStrategy):
    """原始模型训练策略"""
    
    def __init__(self, 
                 display_name: str, 
                 model: torch.nn.Module, 
                 lr: float,
                 weight_decay: float,
                 dataset,
                 ):
        super().__init__(display_name, model, lr, weight_decay, dataset)