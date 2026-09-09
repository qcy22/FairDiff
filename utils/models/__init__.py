from .BaseModel import BaseRecommendationModel
from .PMF import PMF,PMFConfig
from .NGCF import NGCF,NGCFConfig
from .LightGCN import LightGCN,LightGCNConfig

__all__ = ['BaseRecommendationModel', 'PMF', 'NGCF', 'LightGCN']

# 模型类映射
MODEL_CLASSES = {
    'PMF': PMF,
    'NGCF': NGCF,
    'LightGCN': LightGCN
}

MODEL_CONFIG_CLASSES = {
    'PMF': PMFConfig,
    'NGCF': NGCFConfig,
    'LightGCN': LightGCNConfig
}