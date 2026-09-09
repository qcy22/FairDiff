import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .BaseModel import BaseRecommendationModel, BaseModelConfig

@dataclass
class PMFConfig(BaseModelConfig):
    """PMF模型配置"""
    emb_dim: int = 32  

class PMF(BaseRecommendationModel):
    def __init__(self, config, user_num, item_num, device, **kwargs):
        super(PMF, self).__init__(config, user_num, item_num, device)
        self.emb_dim = config['emb_dim']
        self.user_embedding = self._init_embedding(user_num, self.emb_dim,device)
        self.item_embedding = self._init_embedding(item_num, self.emb_dim,device)

    def forward(self, user_indices, item_indices, return_emb=False):
        """前向传播计算预测分数"""
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)

        if return_emb:
            return user_vec, item_vec  
        
        scores = self.compute_scores(user_vec, item_vec)
        return scores

    def get_embeddings_for_fair_loss(self, samples: torch.LongTensor) -> tuple:
        """获取目标用户和邻居用户的嵌入"""
        device = self.user_embedding.weight.device
        samples = samples.to(device)
        inactive_emb = self.user_embedding.weight[samples[:, 0]]
        neighbor_emb = self.user_embedding.weight[samples[:, 1]]
        return inactive_emb, neighbor_emb

    @torch.no_grad()
    def get_embedding(self)->tuple:
        """获取用户和物品的嵌入表示（根据配置可返回归一化后结果）"""
        return self.user_embedding.weight, self.item_embedding.weight
    
    @torch.no_grad()
    def load_embedding(self, embedding_path: str, device):
        """加载预训练的嵌入向量"""
        data = torch.load(embedding_path, map_location=device,weights_only=False)
        user_data = data['user_data']
        item_data = data['item_data']

        for user_id, user_data in user_data.items():
            self.user_embedding.weight.data[user_id] = user_data['cached_user_emb'].to(device)

        for item_id, item_data in item_data.items():
            self.item_embedding.weight.data[item_id] = item_data['cached_item_emb'].to(device)

    