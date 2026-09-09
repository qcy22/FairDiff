import os
import yaml
import time
import random
from typing import Dict,List
from dataclasses import dataclass, field
import torch
import numpy as np

# 本地模块导入
from .models import MODEL_CLASSES,MODEL_CONFIG_CLASSES
from .models.BaseModel import BaseModelConfig
from .strategies import STRATEGIES,BaseStrategy

# ==================== 全局常量 ====================
SPLIT = "=" * 60

# ==================== 核心配置类 ====================
@dataclass
class Config:
    """统一配置类 - 包含所有训练和模型配置"""
    # 训练相关配置
    epochs: int
    seed: int
    batch_size: int
    eval_batch_size: int
    recording_epoch: int
    freeze_embedding_path: str
    user_tiers: int
    
    # 模型相关配置
    model: str
    model_config: BaseModelConfig
    
    # 数据相关配置
    dataset: str
    
    # 优化器相关配置
    optimizer: str
    adam_lr: float
    l2_regularization: float
    
    # 策略相关配置
    strategies: Dict[str, Dict]
    active_strategies: List[str]
    
    @property
    def cuda(self) -> bool:
        """返回CUDA可用性"""
        return not self.no_cuda and torch.cuda.is_available()
    
    @property
    def device(self) -> torch.device:
        """返回训练设备"""
        if self.cuda:
            return torch.device(f'cuda:{self.cuda_index}')
        return torch.device('cpu')


# ==================== 训练相关配置类 ====================
@dataclass
class ModelInstance:
    """模型实例封装类"""
    strategy_name: str
    display_name: str
    strategy: BaseStrategy
    best_model_results = None
    best_epoch: int = -1
    best_strategy_state_dict = None
    

@dataclass 
class EvaluationConfig:
    """评估配置类 - 支持动态模型列表"""
    model_instances: List[ModelInstance]
    batch_size: int
    user_tiers: int
    recording_epoch: int


# ==================== 配置解析与加载 ====================

def parse_args_and_config(args) -> Config:

    config_path = args.config
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    if args.model is not None:
        yaml_config['model'] = args.model
    if args.dataset is not None:
        yaml_config['dataset'] = args.dataset
    if len(args.active_strategies) > 0:
        yaml_config['active_strategies'] = args.active_strategies
    
    model_name = yaml_config['model']
    model_config_class = MODEL_CONFIG_CLASSES[model_name]
    model_config = model_config_class(**(yaml_config['models'][model_name]))
    
    config = Config(
        # 训练配置
        epochs=yaml_config['training']['epochs'],
        seed=yaml_config['training']['seed'],
        batch_size=yaml_config['training']['batch_size'],
        eval_batch_size=yaml_config['training']['eval_batch_size'],
        recording_epoch=yaml_config['training']['recording_epoch'],
        freeze_embedding_path=yaml_config['training']['freeze_embedding_path'],
        user_tiers=yaml_config['training']['user_tiers'],

        # 模型配置
        model=model_name,
        model_config=model_config,
        
        # 数据配置
        dataset=yaml_config['dataset'],
        
        # 优化器配置
        optimizer=yaml_config['optimizer']['name'],
        adam_lr=float(yaml_config['optimizer']['adam_lr']),
        l2_regularization=float(yaml_config['optimizer']['l2_regularization']),
        
        # 策略配置
        strategies=yaml_config['strategies'],
        active_strategies=yaml_config['active_strategies'],
    )
    
    return config


# ==================== 模型实例创建与管理 ====================
def create_model_instances(config, device, dataset):
    """动态创建模型实例"""
    # 获取训练字典用于构建图结构
    train_dict = dataset.train_dict
    
    model_kwargs = {
        'device': device,
        'train_dict': train_dict
    }
    model_instances = []

    # 从配置中获取激活的策略
    for strategy_name in config.active_strategies:
        if strategy_name not in config.strategies:
            raise ValueError(f"未找到策略配置: {strategy_name}")
        
        strategy_config = config.strategies[strategy_name]
        
        # 创建模型
        model_class = MODEL_CLASSES[config.model]
        model_config_dict = config.model_config.to_dict()
        model = model_class(model_config_dict, dataset.user_num, dataset.item_num, **model_kwargs).to(device)
        
        strategy_class = STRATEGIES[strategy_config['strategy_class']]
        strategy_params = {
            'display_name': strategy_name,
            'model': model,
            'lr': config.adam_lr,
            'weight_decay': config.l2_regularization,
            'dataset': dataset,
            **strategy_config['strategy_params']
        }
        strategy = strategy_class(**strategy_params)
        
        model_instance = ModelInstance(
            strategy_name=strategy_name,
            display_name=strategy_config['display_name'],
            strategy=strategy
        )
        
        model_instances.append(model_instance)
    
    if len(model_instances) > 1:
        base_model = model_instances[0].strategy.model
        for i in range(1, len(model_instances)):
            model_instances[i].strategy.model.load_state_dict(base_model.state_dict())

    return model_instances
