import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from abc import ABC
import numpy as np
import scipy.sparse as sp


@dataclass
class BaseModelConfig(ABC):
    def to_dict(self) -> dict:
        return asdict(self)


class BaseRecommendationModel(nn.Module, ABC):
    
    def __init__(self, config, user_num, item_num, device):
        super(BaseRecommendationModel, self).__init__()
        if isinstance(config, BaseModelConfig):
            config = config.to_dict()
        self.config = config
        self.num_users = user_num
        self.num_items = item_num
        self.device = device
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        """辅助函数：将 scipy 稀疏矩阵转换为 torch 稀疏 tensor"""
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape)).to(self.device)

    def _create_norm_adj(self, train_dict):
        """
        构建对称归一化邻接矩阵
        """
        # 1. 快速构建交互列表 (Vectorized construction)
        users_list = []
        items_list = []
        for u, items in train_dict.items():
            users_list.extend([u] * len(items))
            items_list.extend(items)
        
        users_np = np.array(users_list)
        items_np = np.array(items_list)
        
        # 2. 构建二部图的大邻接矩阵 (Adjacency Matrix)
        # 矩阵结构:
        # [0, R]
        # [R.T, 0]
        # 直接拼接行列索引，避免生成中间的 R 矩阵，节省内存
        n_all = self.num_users + self.num_items
        
        # 上半部分 (User -> Item)
        row_idx = users_np
        col_idx = items_np + self.num_users
        
        # 下半部分 (Item -> User)
        row_idx_all = np.concatenate([row_idx, col_idx])
        col_idx_all = np.concatenate([col_idx, row_idx])
        data_all = np.ones_like(row_idx_all, dtype=np.float32)
        
        adj_mat = sp.coo_matrix((data_all, (row_idx_all, col_idx_all)), shape=(n_all, n_all))
        
        # 3. 计算归一化系数: D^{-1/2}
        rowsum = np.array(adj_mat.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # 4. 计算 D^{-1/2} * A * D^{-1/2}
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        
        return self._convert_sp_mat_to_sp_tensor(norm_adj)

    def _init_embedding(self, num_embeddings: int, emb_dim: int, device=None):
        """
        创建并初始化 Embedding (xavier_uniform)，自动放到指定 device
        """
        emb = nn.Embedding(num_embeddings, emb_dim)
        nn.init.orthogonal_(emb.weight)
        # nn.init.xavier_uniform_(emb.weight)
        # nn.init.normal_(emb.weight, std=0.03)
        if device is None:
            device = self.device
        return emb.to(device)

    def compute_scores(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores

    @abstractmethod
    def forward(self, user_indices, item_indices, return_emb=False):
        pass
    
    @abstractmethod
    def get_embeddings_for_fair_loss(self, samples: torch.LongTensor) -> tuple:
        pass
    
    
    @abstractmethod
    def get_embedding(self)->tuple:
        pass
    
    


